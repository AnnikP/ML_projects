"""
F1 Race Top-3 Predictor
=======================
Uses OpenF1 API + XGBoost LambdaMART ranking to predict the podium.

Install dependencies:
    pip install xgboost requests pandas numpy scikit-learn pyarrow

Usage:
    python f1_predictor.py --list-sessions          # see available sessions
    python f1_predictor.py --train                  # fetch data & train model
    python f1_predictor.py                          # predict latest race
    python f1_predictor.py --session 9158           # predict specific session
    python f1_predictor.py --debug                  # verbose API output
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────── Configuration ───────────────────────────────────

BASE_URL = "https://api.openf1.org/v1"
MODEL_PATH = Path("f1_xgb_ranker.json")
FEATURES_PATH = Path("feature_cache.parquet")
META_PATH = Path("model_meta.json")

# OpenF1 only reliably covers 2023 onwards
DEFAULT_TRAIN_YEARS = [2023, 2024]

XGBOOST_PARAMS = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@3",
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "n_estimators": 400,
    "early_stopping_rounds": 30,
    "random_state": 42,
    "tree_method": "hist",
}

FORM_WINDOW = 5

DEBUG = False

# Minimum seconds between ANY two API calls (global throttle)
API_CALL_DELAY = 0.4
_last_call_time = 0.0

FEATURE_COLS = [
    "grid_position",
    "rolling_finish",
    "rolling_podiums",
    "team_name",
    "country_code",
    "circuit_key",
    "num_stints",
    "compounds_used",
    "mean_lap",
    "std_lap",
    "lap_count",
    "air_temperature",
    "track_temperature",
    "humidity",
    "wind_speed",
    "rainfall",
]

# ─────────────────────────── OpenF1 API helpers ──────────────────────────────

def _get(endpoint: str, params: dict = None, retries: int = 5) -> list:
    """
    GET wrapper with:
    - global per-call throttle (API_CALL_DELAY between every request)
    - 429 handling: respects Retry-After header, backs off exponentially
    - always returns a list (error dicts → [])
    - verbose output when DEBUG=True
    """
    global _last_call_time

    url = f"{BASE_URL}/{endpoint}"
    if DEBUG:
        print(f"  [API] GET {url}  params={params}")

    for attempt in range(retries):
        # ── Global rate throttle ──────────────────────────────────────────
        elapsed = time.time() - _last_call_time
        if elapsed < API_CALL_DELAY:
            time.sleep(API_CALL_DELAY - elapsed)

        try:
            r = requests.get(url, params=params, timeout=30)
            _last_call_time = time.time()

            # ── 429 Too Many Requests ─────────────────────────────────────
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10)) + 2
                print(f"\n  ⏳ Rate limited — waiting {wait}s before retry …")
                time.sleep(wait)
                continue   # retry without counting as a failure

            # ── Other non-OK responses ────────────────────────────────────
            if not r.ok:
                try:
                    body = r.json()
                except Exception:
                    body = r.text[:200]
                if DEBUG or r.status_code != 404:
                    print(f"  ⚠  HTTP {r.status_code} [{endpoint}] {body}")
                return []

            data = r.json()

            # OpenF1 sometimes returns {"detail": "..."} for bad params
            if not isinstance(data, list):
                if DEBUG:
                    print(f"  [API] non-list response: {data}")
                return []

            if DEBUG:
                print(f"  [API] ← {len(data)} records")
            return data

        except requests.exceptions.Timeout:
            wait = 5 * (attempt + 1)
            print(f"  ⚠  Timeout [{endpoint}], waiting {wait}s …")
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = 5 * (attempt + 1)
            print(f"  ⚠  Connection error [{endpoint}]: {e}, waiting {wait}s …")
            time.sleep(wait)
        except Exception as e:
            print(f"  ⚠  Unexpected error [{endpoint}]: {e}")
            if attempt < retries - 1:
                time.sleep(3)

    return []


def _safe_df(data: list, required_cols: list = None) -> pd.DataFrame:
    """Create DataFrame, returning empty df if required columns are absent."""
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            if DEBUG:
                print(f"  [DF] missing cols {missing}, got {list(df.columns)}")
            return pd.DataFrame()
    return df


# ─────────────────────────── Session helpers ─────────────────────────────────

def fetch_sessions(year: int = None) -> pd.DataFrame:
    """
    Fetch Race sessions. OpenF1 data starts from 2023.
    Tries session_name='Race' filter first; falls back to fetching all and
    filtering locally so column naming differences don't break things.
    """
    params = {}
    if year:
        params["year"] = year

    # Try both common field name conventions
    for name_field in ("session_name", "session_type"):
        p = {**params, name_field: "Race"}
        data = _get("sessions", p)
        if data:
            break
    else:
        data = _get("sessions", params)

    df = _safe_df(data)
    if df.empty:
        return df

    # Normalise: keep only race sessions
    for col in ("session_name", "session_type"):
        if col in df.columns:
            mask = df[col].str.lower() == "race"
            df = df[mask]
            break

    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
        df = df.sort_values("date_start")

    return df.reset_index(drop=True)


def find_qualifying_key(race_session_key: int) -> int | None:
    """Return session_key of the qualifying session for the same race weekend."""
    sess_data = _get("sessions", {"session_key": int(race_session_key)})
    if not sess_data:
        return None
    meeting_key = sess_data[0].get("meeting_key")
    if not meeting_key:
        return None

    all_sess = _get("sessions", {"meeting_key": int(meeting_key)})
    if not all_sess:
        return None

    df = pd.DataFrame(all_sess)
    name_col = next((c for c in ("session_name", "session_type") if c in df.columns), None)
    if not name_col:
        return None

    qual = df[df[name_col].str.lower().str.contains("qualif", na=False)]
    if qual.empty:
        qual = df[df[name_col].str.lower().str.contains("sprint", na=False)]
    if qual.empty:
        return None

    return int(qual.iloc[-1]["session_key"])


# ─────────────────────────── Per-session fetchers ────────────────────────────

def fetch_final_positions(session_key: int) -> pd.DataFrame:
    """Last recorded position per driver in a session."""
    data = _get("position", {"session_key": int(session_key)})
    df = _safe_df(data, required_cols=["driver_number", "position"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = (df.sort_values("date")
            .groupby("driver_number", as_index=False)
            .last()[["driver_number", "position"]])
    df["session_key"] = session_key
    return df


def fetch_drivers(session_key: int) -> pd.DataFrame:
    data = _get("drivers", {"session_key": int(session_key)})
    df = _safe_df(data, required_cols=["driver_number"])
    if df.empty:
        return df

    df = df.drop_duplicates("driver_number")

    # Handle different name field conventions
    if "broadcast_name" not in df.columns:
        df["broadcast_name"] = df.get("full_name", df["driver_number"].astype(str))
    if "team_name" not in df.columns:
        df["team_name"] = "Unknown"
    if "country_code" not in df.columns:
        df["country_code"] = "UNK"

    return df[["driver_number", "broadcast_name", "team_name", "country_code"]].copy()


def fetch_weather_avg(session_key: int) -> dict:
    data = _get("weather", {"session_key": int(session_key)})
    if not data:
        return {}
    df = pd.DataFrame(data)
    cols = ["air_temperature", "track_temperature",
            "humidity", "wind_speed", "rainfall"]
    return {c: float(df[c].mean()) for c in cols if c in df.columns}


def fetch_stints(session_key: int) -> pd.DataFrame:
    data = _get("stints", {"session_key": int(session_key)})
    df = _safe_df(data)
    if df.empty or "compound" not in df.columns or "driver_number" not in df.columns:
        return pd.DataFrame()
    stint_col = "stint_number" if "stint_number" in df.columns else df.columns[0]
    return (df.groupby("driver_number")
              .agg(num_stints=(stint_col, "max"),
                   compounds_used=("compound", "nunique"))
              .reset_index())


def fetch_laps_agg(session_key: int) -> pd.DataFrame:
    data = _get("laps", {"session_key": int(session_key)})
    df = _safe_df(data)
    if df.empty or "lap_duration" not in df.columns or "driver_number" not in df.columns:
        return pd.DataFrame()
    df["lap_duration"] = pd.to_numeric(df["lap_duration"], errors="coerce")
    df = df.dropna(subset=["lap_duration"])
    if df.empty:
        return pd.DataFrame()
    med = df["lap_duration"].median()
    df = df[df["lap_duration"].between(med * 0.7, med * 1.4)]
    return (df.groupby("driver_number")["lap_duration"]
              .agg(mean_lap="mean", std_lap="std", lap_count="count")
              .reset_index())


# ─────────────────────────── Feature Engineering ─────────────────────────────

def build_session_features(session_key: int, circuit_key: str, year: int):
    print(f"    Session {session_key} ({circuit_key}) … ", end="", flush=True)

    drivers = fetch_drivers(session_key)
    if drivers.empty:
        print("SKIP — no driver data")
        return None

    result = fetch_final_positions(session_key)
    if result.empty:
        print("SKIP — no position data")
        return None

    # Qualifying grid
    qual_key = find_qualifying_key(session_key)
    grid_df = pd.DataFrame()
    if qual_key:
        grid_raw = fetch_final_positions(qual_key)
        if not grid_raw.empty:
            grid_df = grid_raw[["driver_number", "position"]].rename(
                columns={"position": "grid_position"})

    stints = fetch_stints(session_key)
    laps = fetch_laps_agg(session_key)
    weather = fetch_weather_avg(session_key)

    df = drivers.merge(
        result[["driver_number", "position", "session_key"]],
        on="driver_number", how="left"
    )
    if not grid_df.empty:
        df = df.merge(grid_df, on="driver_number", how="left")
    else:
        df["grid_position"] = np.nan

    if not stints.empty:
        df = df.merge(stints, on="driver_number", how="left")
    if not laps.empty:
        df = df.merge(laps, on="driver_number", how="left")

    for col, val in weather.items():
        df[col] = val

    df["year"] = year
    df["circuit_key"] = circuit_key
    df["finish_position"] = pd.to_numeric(df["position"], errors="coerce")
    n = len(df)
    df["rank_label"] = (n + 1 - df["finish_position"]).clip(lower=0)
    df["podium"] = (df["finish_position"] <= 3).astype(int)

    print(f"{n} drivers OK")
    return df


def add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["driver_number", "session_key"])
    df["rolling_finish"] = (
        df.groupby("driver_number")["finish_position"]
          .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    )
    df["rolling_podiums"] = (
        df.groupby("driver_number")["podium"]
          .transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).sum())
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["team_name", "country_code", "circuit_key"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))
    return df


# ─────────────────────────── Data Collection ─────────────────────────────────

def collect_training_data(years: list) -> pd.DataFrame:
    all_frames = []

    for year in years:
        print(f"\n{'='*50}")
        print(f" {year} season")
        print(f"{'='*50}")
        sessions = fetch_sessions(year=year)
        if sessions.empty:
            print(f"  No race sessions found for {year}.")
            continue

        n_sess = len(sessions)
        print(f"  Found {n_sess} race sessions\n")
        for i, (_, sess) in enumerate(sessions.iterrows(), 1):
            sk = int(sess["session_key"])
            circuit = str(sess.get("circuit_key",
                          sess.get("circuit_short_name",
                          sess.get("location", str(sk)))))
            print(f"  [{i}/{n_sess}]", end=" ")
            frame = build_session_features(sk, circuit, year)
            if frame is not None and not frame.empty:
                all_frames.append(frame)
            # Extra pause between sessions (per-call throttle also applies)
            time.sleep(1.5)

    if not all_frames:
        raise RuntimeError(
            "\nNo training data collected.\n"
            "  - OpenF1 only covers 2023+ → use --years 2023 2024\n"
            "  - Check internet connection\n"
            "  - Run with --debug to see raw API responses\n"
            "  - Run with --list-sessions to verify sessions are reachable"
        )

    data = pd.concat(all_frames, ignore_index=True)
    data = add_rolling_form(data)
    data = encode_categoricals(data)
    print(f"\n✅ Dataset: {len(data)} rows across "
          f"{data['session_key'].nunique()} races, "
          f"{data['driver_number'].nunique()} unique drivers")
    return data


# ─────────────────────────── Model ───────────────────────────────────────────

def train_model(data: pd.DataFrame):
    data = data.dropna(subset=["finish_position", "rank_label"])
    available = [c for c in FEATURE_COLS if c in data.columns]
    print(f"  Training with {len(available)} features: {available}\n")

    X = data[available].fillna(-1)
    y = data["rank_label"].astype(int)
    groups = data.groupby("session_key").size().values

    model = xgb.XGBRanker(**XGBOOST_PARAMS)
    model.fit(X, y, group=groups,
              eval_set=[(X, y)], eval_group=[groups],
              verbose=50)
    return model, available


def save_model(model, feature_cols: list):
    model.save_model(str(MODEL_PATH))
    META_PATH.write_text(json.dumps({"feature_cols": feature_cols}))
    print(f"💾 Model → {MODEL_PATH}  |  Meta → {META_PATH}")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'.\n"
            f"Run first:  python f1_predictor.py --train"
        )
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing {META_PATH}. Re-run --train.")
    model = xgb.XGBRanker()
    model.load_model(str(MODEL_PATH))
    feature_cols = json.loads(META_PATH.read_text())["feature_cols"]
    return model, feature_cols


# ─────────────────────────── Prediction ──────────────────────────────────────

def predict_race(session_key: int, model, feature_cols: list,
                 historical_data: pd.DataFrame = None) -> pd.DataFrame:
    info = _get("sessions", {"session_key": int(session_key)})
    circuit, year = "unknown", 2024
    if info:
        s = info[0]
        circuit = str(s.get("circuit_key",
                      s.get("circuit_short_name",
                      s.get("location", "unknown"))))
        year = int(s.get("year", 2024))
        print(f"  {s.get('location', session_key)} — {year}")

    frame = build_session_features(session_key, circuit, year)
    if frame is None or frame.empty:
        raise ValueError(
            f"Could not build features for session {session_key}.\n"
            f"Run --debug to see API responses, or --list-sessions to find valid keys."
        )

    # Inject rolling form from cached history
    if historical_data is not None and not historical_data.empty:
        for dn in frame["driver_number"].unique():
            hist = (historical_data[
                (historical_data["driver_number"] == dn) &
                (historical_data["session_key"] < session_key)
            ].sort_values("session_key").tail(FORM_WINDOW))
            if not hist.empty:
                frame.loc[frame["driver_number"] == dn, "rolling_finish"] = \
                    hist["finish_position"].mean()
                frame.loc[frame["driver_number"] == dn, "rolling_podiums"] = \
                    hist["podium"].sum()

    frame = encode_categoricals(frame.copy())
    available = [c for c in feature_cols if c in frame.columns]
    X = frame[available].fillna(-1)
    frame["predicted_score"] = model.predict(X)
    frame = frame.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    frame["predicted_position"] = range(1, len(frame) + 1)
    return frame


def print_podium(frame: pd.DataFrame):
    medals = ["🥇", "🥈", "🥉"]
    print("\n" + "═" * 52)
    print("           🏆  PREDICTED PODIUM  🏆")
    print("═" * 52)
    for i in range(min(3, len(frame))):
        row = frame.iloc[i]
        name = str(row.get("broadcast_name", row["driver_number"]))
        team = str(row.get("team_name", ""))
        print(f"  {medals[i]}  {name:<26} {team}")
    print("═" * 52)

    display_cols = [c for c in ["predicted_position", "broadcast_name",
                                 "team_name", "grid_position", "predicted_score"]
                    if c in frame.columns]
    print("\nFull predicted grid:")
    print(frame[display_cols].to_string(index=False))


def print_feature_importance(model, feature_cols: list):
    scores = model.get_booster().get_fscore()
    if not scores:
        print("No feature importance available.")
        return
    imp = (pd.DataFrame.from_dict(scores, orient="index", columns=["importance"])
             .sort_values("importance", ascending=False))
    print("\n📊 Feature importances (top 10):")
    print(imp.head(10).to_string())


# ─────────────────────────── CLI ─────────────────────────────────────────────

def main():
    global DEBUG

    parser = argparse.ArgumentParser(
        description="F1 Top-3 Race Predictor — OpenF1 + XGBoost LambdaMART",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python f1_predictor.py --list-sessions          list available race sessions
  python f1_predictor.py --train                  train model (2023+2024 data)
  python f1_predictor.py --train --years 2024     train on 2024 only
  python f1_predictor.py                          predict latest race
  python f1_predictor.py --session 9158           predict specific session
  python f1_predictor.py --debug                  verbose API output
        """)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_TRAIN_YEARS,
                        help="Seasons to train on (2023+ only)")
    parser.add_argument("--session", type=int, help="session_key to predict")
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--importance", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DEBUG = args.debug

    # ── List sessions ───────────────────────────────────────────────────────
    if args.list_sessions:
        print("Fetching recent race sessions from OpenF1 …")
        sessions = fetch_sessions()
        if sessions.empty:
            print("No sessions found. Try --debug to diagnose.")
            return
        show_cols = [c for c in ["session_key", "year", "location",
                                  "circuit_short_name", "date_start"]
                     if c in sessions.columns]
        print(sessions[show_cols].tail(30).to_string(index=False))
        return

    # ── Train ───────────────────────────────────────────────────────────────
    if args.train:
        invalid = [y for y in args.years if y < 2023]
        if invalid:
            print(f"WARNING: OpenF1 has no data for {invalid}. "
                  f"They will be skipped.")
            args.years = [y for y in args.years if y >= 2023]
        if not args.years:
            print("ERROR: No valid training years. Use --years 2023 2024")
            return

        print(f"🏎  Training on: {args.years}")
        data = collect_training_data(years=args.years)
        data.to_parquet(FEATURES_PATH)
        print(f"📦 Features cached → {FEATURES_PATH}")

        print("\n🤖 Training XGBoost LambdaMART …")
        model, feature_cols = train_model(data)
        save_model(model, feature_cols)
        print("\nDone! Predict with:  python f1_predictor.py")
        return

    # ── Predict ─────────────────────────────────────────────────────────────
    model, feature_cols = load_model()
    history = pd.read_parquet(FEATURES_PATH) if FEATURES_PATH.exists() else None

    if args.session:
        session_key = args.session
    else:
        print("Auto-detecting latest race session …")
        sessions = fetch_sessions()
        if sessions.empty:
            print("No sessions found. Use --session <key> or --list-sessions")
            return
        latest = sessions.iloc[-1]
        session_key = int(latest["session_key"])
        print(f"→ {latest.get('location', '?')} {latest.get('year', '')} "
              f"(key={session_key})")

    print(f"\n🏁 Predicting session {session_key} …")
    frame = predict_race(session_key, model, feature_cols, history)
    print_podium(frame)

    if args.importance:
        print_feature_importance(model, feature_cols)


if __name__ == "__main__":
    main()