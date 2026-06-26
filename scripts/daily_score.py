"""
daily_score.py — Daily KZ-MCI scoring
======================================
Fetches latest data from KASE, applies current month's PCA loadings,
appends new daily score to the index JSON file.

Run:    python scripts/daily_score.py
Schedule: daily at 18:00 Almaty time (after KASE closes at 17:00)

Output:
    data/kz_mci_latest.json  — full index history + metadata
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import date, timedelta

# Allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    INSTRUMENTS, fetch_kase, load_base_rate, build_variables, score_with_loadings
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR       = "data"
OUTPUT_JSON    = f"{DATA_DIR}/kz_mci_latest.json"
LOADINGS_CSV   = f"{DATA_DIR}/kz_mci_loadings.csv"
HISTORY_CSV    = f"{DATA_DIR}/kz_mci_option1.csv"
BASE_RATE_CSV  = f"{DATA_DIR}/nbk_base_rate_daily.csv"

# How many days of history to fetch from KASE for rolling calculations
# Need 21 trading days minimum for usdkzt_ret_21d rolling window
LOOKBACK_DAYS  = 45

# ── 1. Load existing index history ────────────────────────────────────────────

print("Loading existing index history...")
history = pd.read_csv(HISTORY_CSV, index_col="date", parse_dates=True)
history = history["KZ_MCI"]
print(f"  History: {len(history)} rows  "
      f"({history.dropna().index[0].date()} → {history.dropna().index[-1].date()})")

last_date = history.dropna().index[-1].date()
today     = date.today()

if today <= last_date:
    print(f"  Index already up to date ({last_date}) — nothing to do")
else:
    print(f"  Will score from {last_date + timedelta(days=1)} to {today}")

# ── 2. Load PCA loadings ──────────────────────────────────────────────────────

print("\nLoading PCA loadings...")
loadings = pd.read_csv(LOADINGS_CSV, index_col="date", parse_dates=True)
latest_loadings = loadings.iloc[-1]
loadings_date   = loadings.index[-1].strftime("%Y-%m")
print(f"  Using loadings from: {loadings_date}")
print(f"  Variables: {[c for c in latest_loadings.index if c != 'variance_explained']}")

# ── 3. Load historical variables for rolling window ───────────────────────────

print("\nLoading historical dataset for rolling windows...")
dataset = pd.read_csv(f"{DATA_DIR}/dataset_final.csv",
                      index_col="date", parse_dates=True)

# Compute mean and std from historical data for standardization
# These should match what was used in the last PCA estimation
var_cols = [c for c in latest_loadings.index if c != "variance_explained"]
hist_monthly = dataset[var_cols].resample("ME").mean()
mean_vals = hist_monthly.mean()
std_vals  = hist_monthly.std()

print(f"  Standardization based on {len(hist_monthly)} monthly observations")

# ── 4. Fetch recent KASE data ─────────────────────────────────────────────────

fetch_start = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
fetch_end   = today.strftime("%Y-%m-%d")

print(f"\nFetching KASE data ({fetch_start} → {fetch_end})...")

raw_series = {}
for symbol, ep_type in INSTRUMENTS.items():
    df = fetch_kase(symbol, ep_type, fetch_start, fetch_end)
    if df is not None and not df.empty:
        raw_series[f"{symbol}_close"]  = df["close"]
        raw_series[f"{symbol}_volume"] = df["volume"]
        print(f"  ✓ {symbol}: {len(df)} rows")
    else:
        print(f"  ✗ {symbol}: no data")

raw = pd.DataFrame(raw_series)

# ── 5. Load NBK base rate ─────────────────────────────────────────────────────

base_rate = load_base_rate(BASE_RATE_CSV)

# ── 6. Build variables ────────────────────────────────────────────────────────

print("\nBuilding variables...")
variables = build_variables(raw, base_rate)
print(f"  Variables computed: {list(variables.columns)}")

# ── 7. Score new dates ────────────────────────────────────────────────────────

print("\nScoring new dates...")
new_scores = {}

# Get dates to score — trading days after last known date
dates_to_score = [
    d for d in variables.index
    if d.date() > last_date and not variables.loc[d].isna().all()
]

if not dates_to_score:
    print("  No new trading days to score")
else:
    for dt in dates_to_score:
        row   = variables.loc[dt]
        score = score_with_loadings(row, latest_loadings, mean_vals, std_vals)
        new_scores[dt] = score
        print(f"  {dt.date()}  raw score: {score:+.4f}")

# ── 8. Normalize new scores to index scale ────────────────────────────────────

if new_scores:
    # Use historical index mean and std for normalization
    # This keeps new scores on the same scale as the historical index
    hist_valid = history.dropna()
    hist_mean  = hist_valid.mean()
    hist_std   = hist_valid.std()

    new_scores_norm = {
        dt: (score - hist_mean) / hist_std
        for dt, score in new_scores.items()
    }

    print(f"\nNormalized scores:")
    for dt, val in new_scores_norm.items():
        print(f"  {dt.date()}  {val:+.3f}σ")

    # Append to history
    new_series = pd.Series(new_scores_norm, name="KZ_MCI")
    new_series.index = pd.DatetimeIndex(new_series.index)
    history = pd.concat([history, new_series]).sort_index()
    history = history[~history.index.duplicated(keep="last")]

    # Save updated CSV
    history.to_csv(HISTORY_CSV, header=True)
    print(f"\n✓ Updated history CSV ({len(history)} rows)")

# ── 9. Build JSON output for Framer ──────────────────────────────────────────

print("\nBuilding JSON output...")

# Monthly series for chart (forward-filled monthly values)
monthly = history.resample("ME").last().dropna()

# Build records list
records = []
for dt, val in monthly.items():
    records.append({
        "date":  dt.strftime("%Y-%m"),
        "value": round(float(val), 3),
        "regime": (
            "severe"   if val >= 2.0  else
            "elevated" if val >= 1.0  else
            "calm"     if val >= 0.0  else
            "easy"
        )
    })

# Latest reading
latest_val  = history.dropna().iloc[-1]
latest_date = history.dropna().index[-1]
prev_val    = history.dropna().iloc[-2]

output = {
    "metadata": {
        "index_name":        "Kazakhstan Market Conditions Index",
        "short_name":        "KZ-MCI",
        "version":           "1.0",
        "methodology":       "Expanding-window PCA, Option 1 (13 variables)",
        "loadings_as_of":    loadings_date,
        "last_updated":      today.strftime("%Y-%m-%d"),
        "data_source":       "KASE, NBRK",
    },
    "latest": {
        "date":   today.strftime("%Y-%m-%d"),   # use run date not forward-fill end date
        "value":  round(float(latest_val), 3),
        "change": round(float(latest_val - prev_val), 3),
        "regime": (
            "Severe stress"    if latest_val >= 2.0 else
            "Elevated stress"  if latest_val >= 1.0 else
            "Near average"     if latest_val >= 0.0 else
            "Easy conditions"
        ),
        "interpretation": (
            f"{'Above' if latest_val >= 0 else 'Below'} historical average "
            f"by {abs(latest_val):.2f} standard deviations"
        ),
    },
    "thresholds": {
        "severe":   2.0,
        "elevated": 1.0,
        "average":  0.0,
        "easy":    -1.0,
    },
    "history": records,
    "episodes": [
        {"start": "2016-06", "end": "2016-09",
         "label": "Oil crash + FX adjustment", "color": "#e74c3c"},
        {"start": "2018-08", "end": "2018-11",
         "label": "EM selloff",                "color": "#e67e22"},
        {"start": "2020-03", "end": "2020-06",
         "label": "Covid + oil shock",         "color": "#c0392b"},
        {"start": "2022-02", "end": "2022-09",
         "label": "Russia invasion",           "color": "#8e44ad"},
        {"start": "2025-01", "end": "2025-08",
         "label": "Rates shock",               "color": "#16a085"},
    ]
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"✓ JSON output → {OUTPUT_JSON}")
print(f"\n── Summary ───────────────────────────────────────────────────────────")
print(f"  Latest reading: {latest_val:+.3f}σ  ({latest_date.strftime('%b %Y')})")
print(f"  Regime:         {output['latest']['regime']}")
print(f"  Change:         {latest_val - prev_val:+.3f}σ vs previous")
print(f"  History:        {len(records)} monthly data points")
print(f"\nDone.")
