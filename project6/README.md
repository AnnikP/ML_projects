# 🏎 F1 Top-3 Race Predictor

Predicts the **top 3 finishers** of an F1 race using:
- **Data**: [OpenF1 API](https://openf1.org) (free, no key required)
- **Model**: XGBoost LambdaMART pairwise ranker (`rank:pairwise`, optimised for `ndcg@3`)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train (first time or to refresh)

Fetches 2022–2024 seasons from OpenF1, engineers features, trains the XGBoost ranker, and saves the model:

```bash
python f1_predictor.py --train
```

Custom seasons:

```bash
python f1_predictor.py --train --years 2023 2024
```

### 2. Predict

Auto-detects the latest race session:

```bash
python f1_predictor.py
```

Predict a specific session (use session_key from OpenF1):

```bash
python f1_predictor.py --session 9158
```

Show feature importance:

```bash
python f1_predictor.py --importance
```

---

## Features Used

| Feature | Description |
|---|---|
| `grid_position` | Qualifying / grid start position |
| `rolling_finish` | Rolling avg finish position (last 5 races) |
| `rolling_podiums` | Rolling podium count (last 5 races) |
| `team_name` | Constructor (encoded) |
| `country_code` | Driver nationality (encoded) |
| `circuit_key` | Circuit identifier (encoded) |
| `num_stints` | Number of pit stops |
| `compounds_used` | Number of distinct tyre compounds |
| `mean_lap` | Mean lap time (safety-car laps removed) |
| `std_lap` | Consistency of lap times |
| `lap_count` | Clean laps counted |
| `air_temperature` | Average air temp during session |
| `track_temperature` | Average track temp |
| `humidity` | Average humidity |
| `wind_speed` | Average wind speed |
| `rainfall` | Rainfall indicator |

---

## Model Details

- **Algorithm**: XGBoost `XGBRanker` with `rank:pairwise` objective (LambdaMART)
- **Eval metric**: `ndcg@3` — directly optimises ranking quality at position 3
- **Training**: Groups are individual race sessions so the model learns relative ordering *within* a race
- **Form features**: Lagged (shift-1) rolling windows prevent data leakage

---

## Output Example

```
══════════════════════════════════════════════════
          🏆  PREDICTED PODIUM  🏆
══════════════════════════════════════════════════
  🥇  M VERSTAPPEN              Red Bull Racing
  🥈  L NORRIS                  McLaren
  🥉  C LECLERC                 Ferrari
══════════════════════════════════════════════════
```

---

## Notes

- OpenF1 has data from **2023 onwards**; 2022 data may be limited
- `--train` can take **10–30 minutes** depending on API response times (rate-limited politely)
- The cached feature file (`feature_cache.parquet`) lets you re-train quickly without re-fetching
- For live race predictions, run just before lights-out once qualifying results are in