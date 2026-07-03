"""
update_index.py — Unified KZ-MCI Daily Update Script
=====================================================
Replaces both monthly_pca.py and daily_score.py with a single script.

Runs daily. On the 1st of each month also reruns full expanding PCA.

Produces two index series:
  1. Monthly: expanding-window PCA on monthly averages (confirmed history)
  2. Daily:   current month's loadings applied to single-day variables

Output:
    data/kz_mci_latest.json      — full JSON for Framer
    data/kz_mci_monthly.csv      — confirmed monthly index values
    data/kz_mci_daily.csv        — daily single-day scores (last 180 days)
    data/kz_mci_loadings.csv     — monthly PCA loadings
    data/kz_mci_norm_constants.json — frozen normalization constants

Usage:
    python scripts/update_index.py

Schedule:
    Daily at 13:00 UTC (19:00 Almaty) via GitHub Actions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import os
import sys
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import INSTRUMENTS, fetch_kase, load_base_rate, build_variables


def _coerce_datetime_index(obj, label):
    """
    Defensive guard: force a DataFrame/Series index to real Timestamps and
    drop any row whose index value can't be parsed as a date (e.g. leftover
    git-conflict marker lines like '<<<<<<< Updated upstream' that end up
    in a CSV's date column). Prevents 'Timestamp vs str' sort/concat crashes
    regardless of how the bad value got there.
    """
    parsed = pd.to_datetime(obj.index, errors="coerce")
    bad_mask = parsed.isna()
    n_bad = int(bad_mask.sum())
    if n_bad:
        bad_values = list(obj.index[bad_mask])[:5]
        print(f"  [WARN] {label}: dropped {n_bad} row(s) with unparseable "
              f"date index, e.g. {bad_values}")
        obj = obj[~bad_mask]
        parsed = parsed[~bad_mask]
    obj = obj.copy()
    obj.index = parsed
    obj.index.name = "date"
    return obj

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR              = "data"
INITIAL_WINDOW_MONTHS = 18
DAILY_HISTORY_DAYS    = 180   # days of daily scores to store and display
LOOKBACK_DAYS         = 90    # days to fetch from KASE on each run

VARS = [
    "usdkzt_ret", "usdkzt_ret_5d", "usdkzt_ret_21d",
    "usdkzt_vol_5d", "usdkzt_vol_21d", "usdkzt_vol_log",
    "tonia_spread", "twina_spread", "term_spread_14d",
    "swap1d_spread", "swap2d_spread", "tonia_vol_log", "swap1d_vol_log",
]

EPISODES = [
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

today          = date.today()
# Allow forcing PCA reestimation via environment variable
is_month_start = today.day == 1 or os.environ.get("FORCE_MONTH_START", "").lower() == "true"

print(f"KZ-MCI Update — {today}  "
      f"({'Monthly PCA reestimation' if is_month_start else 'Daily scoring'})")
print("=" * 60)

# ── 1. Fetch latest KASE data ─────────────────────────────────────────────────

print("\n1. Fetching latest KASE data...")
fetch_start = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
# Fetch up to yesterday — today's data is incomplete until market close
fetch_end   = (today - timedelta(days=1)).strftime("%Y-%m-%d")

raw_series = {}
for symbol, ep_type in INSTRUMENTS.items():
    df_new = fetch_kase(symbol, ep_type, fetch_start, fetch_end)
    if df_new is not None and not df_new.empty:
        raw_series[f"{symbol}_close"]  = df_new["close"]
        raw_series[f"{symbol}_volume"] = df_new["volume"]
        print(f"  ✓ {symbol}: {len(df_new)} rows "
              f"({df_new.index[-1].date()})")
    else:
        print(f"  ✗ {symbol}: no data")

raw       = pd.DataFrame(raw_series)
base_rate = load_base_rate(f"{DATA_DIR}/nbk_base_rate_daily.csv")

# ── 2. Build variables from new data ─────────────────────────────────────────

print("\n2. Building variables...")
new_vars = build_variables(raw, base_rate)
print(f"  Variables: {list(new_vars.columns)}")

# ── 3. Load and update historical dataset ────────────────────────────────────

print("\n3. Updating historical dataset...")
dataset_path = f"{DATA_DIR}/dataset_final.csv"

if os.path.exists(dataset_path):
    existing = pd.read_csv(dataset_path, index_col="date", parse_dates=False)
    existing = _coerce_datetime_index(existing, "dataset_final.csv")
    # Merge: existing + new, new takes precedence for overlapping dates
    combined = pd.concat([existing, new_vars[[v for v in VARS
                                               if v in new_vars.columns]]])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    print(f"  Dataset: {len(existing)} → {len(combined)} rows")
else:
    combined = new_vars
    print(f"  New dataset: {len(combined)} rows")

combined.to_csv(dataset_path)
vars_present = [v for v in VARS if v in combined.columns]

# ── 4. Monthly PCA reestimation (1st of month or first run) ──────────────────

loadings_path       = f"{DATA_DIR}/kz_mci_loadings.csv"
norm_constants_path = f"{DATA_DIR}/kz_mci_norm_constants.json"
monthly_csv_path    = f"{DATA_DIR}/kz_mci_monthly.csv"

if is_month_start or not os.path.exists(loadings_path):
    print(f"\n4. Running full expanding-window PCA "
          f"({'month start' if is_month_start else 'first run'})...")

    monthly = combined[vars_present].resample("ME").mean()
    mci_monthly = pd.Series(index=monthly.index, dtype=float)
    loadings_list = []

    for i in range(INITIAL_WINDOW_MONTHS, len(monthly)):
        window = monthly.iloc[:i+1].copy()
        cols   = window.columns[window.isna().mean() <= 0.4].tolist()
        window = window[cols]
        window = window[window.isna().mean(axis=1) <= 0.5]
        if len(window) < INITIAL_WINDOW_MONTHS or len(cols) < 3:
            continue

        scaler = StandardScaler()
        scaled = scaler.fit_transform(window.fillna(window.mean()))
        pca    = PCA(n_components=1)
        pca.fit(scaled)

        current = scaler.transform(
            monthly[cols].iloc[[i]].fillna(window[cols].mean())
        )
        mci_monthly.iloc[i] = pca.transform(current)[0, 0]

        loading_row = pd.Series(
            pca.components_[0], index=cols, name=monthly.index[i]
        )
        loading_row["variance_explained"] = pca.explained_variance_ratio_[0]
        loadings_list.append(loading_row)

    # Save loadings
    loadings_df = pd.DataFrame(loadings_list)
    loadings_df.index.name = "date"
    loadings_df.to_csv(loadings_path)
    print(f"  ✓ Loadings saved ({len(loadings_df)} months, "
          f"latest: {loadings_df.index[-1].strftime('%Y-%m')})")

    # Save normalization constants — used for daily scoring
    valid       = mci_monthly.dropna()
    norm_mean   = float(valid.mean())
    norm_std    = float(valid.std())
    norm_consts = {
        "mean":        norm_mean,
        "std":         norm_std,
        "computed_at": monthly.index[-1].strftime("%Y-%m"),
        "n_months":    len(valid),
    }
    with open(norm_constants_path, "w") as f:
        json.dump(norm_consts, f, indent=2)
    print(f"  ✓ Norm constants saved "
          f"(mean={norm_mean:.4f}, std={norm_std:.4f})")

    # Normalize and save monthly series
    mci_norm = (mci_monthly - norm_mean) / norm_std
    mci_norm.name = "KZ_MCI_monthly"
    mci_norm.index.name = "date"
    mci_norm.to_csv(monthly_csv_path, header=True)
    print(f"  ✓ Monthly series saved ({len(mci_norm.dropna())} months)")

else:
    print("\n4. Loading existing PCA loadings (not month start)...")
    loadings_df = pd.read_csv(loadings_path, index_col="date",
                               parse_dates=True)
    mci_norm = pd.read_csv(monthly_csv_path, index_col="date",
                            parse_dates=True).squeeze()
    with open(norm_constants_path) as f:
        norm_consts = json.load(f)
    norm_mean = norm_consts["mean"]
    norm_std  = norm_consts["std"]
    print(f"  ✓ Loadings from {loadings_df.index[-1].strftime('%Y-%m')} "
          f"({len(loadings_df)} months)")

# ── 5. Daily scoring — full PCA per day, matching local option_a_preview.py ──
#
# Methodology must match option_a_preview.py exactly: for each day, treat it
# as if it were the current month's single observation, run an expanding-
# window PCA (fit fresh each time), and take that day's raw score. This is
# what preserves single-day extremes (spikes) instead of smoothing them out
# against a monthly average, which is what the old fixed-loadings approach did.
#
# To keep this fast on every run, the un-normalized raw PCA score for each
# day is cached in kz_mci_daily_raw.csv. Once a day is outside the current
# month, its raw score can never change (the data behind it is settled), so
# it's only computed once. Only new days and the current (still-forming)
# month get recomputed each run. Renormalizing with the latest norm_mean/std
# is cheap and always applied to the full cached window.

print("\n5. Computing daily scores (full PCA per day, matching local pipeline)...")

daily_csv_path     = f"{DATA_DIR}/kz_mci_daily.csv"
daily_raw_path     = f"{DATA_DIR}/kz_mci_daily_raw.csv"
FORCE_DAILY_RECOMPUTE = os.environ.get("FORCE_DAILY_RECOMPUTE", "").lower() == "true"

if os.path.exists(daily_raw_path) and not FORCE_DAILY_RECOMPUTE:
    daily_raw_existing = pd.read_csv(daily_raw_path, index_col="date",
                                      parse_dates=False)["raw_score"]
    daily_raw_existing = _coerce_datetime_index(
        daily_raw_existing.to_frame(), "kz_mci_daily_raw.csv"
    )["raw_score"]
else:
    daily_raw_existing = pd.Series(dtype=float, name="raw_score")

cutoff             = pd.Timestamp(today - timedelta(days=DAILY_HISTORY_DAYS))
daily_days         = combined[combined.index >= cutoff].index
current_month_start = pd.Timestamp(today.replace(day=1))

days_to_compute = [
    d for d in daily_days
    if d not in daily_raw_existing.index or d >= current_month_start
]

new_raw_scores = {}
for day in days_to_compute:
    month_start = day.replace(day=1)
    month_end   = day + pd.offsets.MonthEnd(0)

    monthly_to_day = combined[combined.index < month_start][vars_present] \
        .resample("ME").mean()

    single_day = combined.loc[[day], vars_present]
    if single_day.isna().all(axis=1).all():
        continue
    monthly_to_day.loc[month_end] = single_day.iloc[0]
    monthly_to_day = monthly_to_day.sort_index()

    if len(monthly_to_day) < INITIAL_WINDOW_MONTHS + 2:
        continue

    window = monthly_to_day.copy()
    cols   = window.columns[window.isna().mean() <= 0.4].tolist()
    window = window[cols]
    window = window[window.isna().mean(axis=1) <= 0.5]

    if len(window) < INITIAL_WINDOW_MONTHS or len(cols) < 3:
        continue

    scaler = StandardScaler()
    scaled = scaler.fit_transform(window.fillna(window.mean()))
    pca    = PCA(n_components=1)
    pca.fit(scaled)

    current_row = scaler.transform(
        monthly_to_day[cols].iloc[[-1]].fillna(window[cols].mean())
    )
    new_raw_scores[day] = pca.transform(current_row)[0, 0]

daily_raw_new = pd.Series(new_raw_scores, name="raw_score")
daily_raw_new.index = pd.DatetimeIndex(daily_raw_new.index)
daily_raw_new.index.name = "date"

daily_raw_combined = pd.concat([daily_raw_existing, daily_raw_new])
daily_raw_combined = _coerce_datetime_index(
    daily_raw_combined.to_frame(), "daily_raw_combined (pre-sort)"
)["raw_score"]
daily_raw_combined = daily_raw_combined[
    ~daily_raw_combined.index.duplicated(keep="last")].sort_index()
daily_raw_combined = daily_raw_combined[daily_raw_combined.index >= cutoff]
daily_raw_combined.to_csv(daily_raw_path, header=True)

# Renormalize the full cached window with the current norm constants
daily_combined = (daily_raw_combined - norm_mean) / norm_std
daily_combined.name = "KZ_MCI_daily"
daily_combined.index.name = "date"
daily_combined.to_csv(daily_csv_path, header=True)

n_backfill = len([d for d in days_to_compute if d < current_month_start])
n_current  = len([d for d in days_to_compute if d >= current_month_start])
print(f"  Recomputed {len(days_to_compute)} day(s) this run "
      f"({n_backfill} backfill, {n_current} current month)")
print(f"  ✓ Daily scores: {len(daily_combined)} trading days "
      f"({daily_combined.index[0].date()} → {daily_combined.index[-1].date()})")
print(f"  Latest daily reading: {daily_combined.iloc[-1]:+.3f}σ "
      f"({daily_combined.index[-1].date()})")

# ── 6. Build JSON output ──────────────────────────────────────────────────────

print("\n6. Building JSON output...")

# Monthly records (confirmed history)
monthly_valid  = mci_norm.dropna()
monthly_records = []
for dt, val in monthly_valid.items():
    monthly_records.append({
        "date":   dt.strftime("%Y-%m"),
        "value":  round(float(val), 3),
        "regime": (
            "severe"   if val >= 2.0 else
            "elevated" if val >= 1.0 else
            "calm"     if val >= 0.0 else
            "easy"
        )
    })

# Daily records (last DAILY_HISTORY_DAYS)
daily_records = []
for dt, val in daily_combined.items():
    if pd.isna(val):
        continue
    daily_records.append({
        "date":   dt.strftime("%Y-%m-%d"),
        "value":  round(float(val), 3),
        "regime": (
            "severe"   if val >= 2.0 else
            "elevated" if val >= 1.0 else
            "calm"     if val >= 0.0 else
            "easy"
        )
    })

# Latest reading
latest_daily_val  = daily_combined.dropna().iloc[-1]
latest_daily_date = daily_combined.dropna().index[-1]
prev_daily_val    = daily_combined.dropna().iloc[-2] \
    if len(daily_combined.dropna()) > 1 else latest_daily_val

# Current partial month info
current_month_start = pd.Timestamp(today.replace(day=1))
days_this_month     = len(combined[combined.index >= current_month_start])

output = {
    "metadata": {
        "index_name":        "Kazakhstan Market Conditions Index",
        "short_name":        "KZ-MCI",
        "version":           "2.0",
        "methodology":       "Expanding-window PCA, Option 1 (13 variables)",
        "loadings_as_of":    loadings_df.index[-1].strftime("%Y-%m"),
        "last_updated":      today.strftime("%Y-%m-%d"),
        "data_source":       "KASE, NBRK",
        "norm_mean":         round(norm_mean, 6),
        "norm_std":          round(norm_std,  6),
    },
    "latest": {
        "date":   latest_daily_date.strftime("%Y-%m-%d"),
        "value":  round(float(latest_daily_val), 3),
        "change": round(float(latest_daily_val - prev_daily_val), 3),
        "regime": (
            "Severe stress"   if latest_daily_val >= 2.0 else
            "Elevated stress" if latest_daily_val >= 1.0 else
            "Near average"    if latest_daily_val >= 0.0 else
            "Easy conditions"
        ),
        "interpretation": (
            f"{'Above' if latest_daily_val >= 0 else 'Below'} historical "
            f"average by {abs(latest_daily_val):.2f} standard deviations"
        ),
        "days_in_current_month": days_this_month,
    },
    "thresholds": {
        "severe": 2.0, "elevated": 1.0, "average": 0.0, "easy": -1.0
    },
    "history_monthly": monthly_records,
    "history_daily":   daily_records,
    "episodes":        EPISODES,
}

json_path = f"{DATA_DIR}/kz_mci_latest.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"  ✓ JSON → {json_path}")

# ── 7. Summary ────────────────────────────────────────────────────────────────

print(f"\n── Summary ───────────────────────────────────────────────────────────")
print(f"  Monthly series:  {len(monthly_valid)} confirmed months  "
      f"latest: {monthly_valid.index[-1].strftime('%Y-%m')} "
      f"{monthly_valid.iloc[-1]:+.3f}σ")
print(f"  Daily series:    {len(daily_combined)} trading days  "
      f"latest: {latest_daily_date.date()} "
      f"{latest_daily_val:+.3f}σ")
print(f"  Current month:   {today.strftime('%Y-%m')} — "
      f"{days_this_month} trading days so far")
print(f"  Loadings from:   {loadings_df.index[-1].strftime('%Y-%m')} "
      f"({len(loadings_df)} months history)")
print(f"\nDone.")
