"""
monthly_pca.py — Monthly KZ-MCI PCA reestimation
==================================================
Runs full expanding-window PCA reestimation on the first of each month.
Updates loadings file and renormalizes the full index history.

Run:    python scripts/monthly_pca.py
Schedule: 1st of each month at 06:00 UTC
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    INSTRUMENTS, fetch_kase, load_base_rate, build_variables
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR              = "data"
DATASET_CSV           = f"{DATA_DIR}/dataset_final.csv"
LOADINGS_CSV          = f"{DATA_DIR}/kz_mci_loadings.csv"
HISTORY_CSV           = f"{DATA_DIR}/kz_mci_option1.csv"
BASE_RATE_CSV         = f"{DATA_DIR}/nbk_base_rate_daily.csv"
INITIAL_WINDOW_MONTHS = 18

VARS = [
    "usdkzt_ret", "usdkzt_ret_5d", "usdkzt_ret_21d",
    "usdkzt_vol_5d", "usdkzt_vol_21d", "usdkzt_vol_log",
    "tonia_spread", "twina_spread", "term_spread_14d",
    "swap1d_spread", "swap2d_spread", "tonia_vol_log", "swap1d_vol_log",
]

# ── 1. Fetch latest data from KASE ────────────────────────────────────────────

from datetime import date, timedelta

print("Fetching latest KASE data for dataset update...")
fetch_start = (date.today() - timedelta(days=60)).strftime("%Y-%m-%d")
fetch_end   = date.today().strftime("%Y-%m-%d")

raw_series = {}
for symbol, ep_type in INSTRUMENTS.items():
    df = fetch_kase(symbol, ep_type, fetch_start, fetch_end)
    if df is not None and not df.empty:
        raw_series[f"{symbol}_close"]  = df["close"]
        raw_series[f"{symbol}_volume"] = df["volume"]
        print(f"  ✓ {symbol}")

raw       = pd.DataFrame(raw_series)
base_rate = load_base_rate(BASE_RATE_CSV)
new_vars  = build_variables(raw, base_rate)

# ── 2. Load and update dataset ────────────────────────────────────────────────

print("\nUpdating dataset...")
existing = pd.read_csv(DATASET_CSV, index_col="date", parse_dates=True)

# Merge new observations
combined = pd.concat([existing, new_vars[VARS]])
combined = combined[~combined.index.duplicated(keep="last")].sort_index()
combined.to_csv(DATASET_CSV)
print(f"  Dataset: {len(existing)} → {len(combined)} rows")

# ── 3. Expanding window PCA ───────────────────────────────────────────────────

print("\nRunning expanding-window PCA...")
vars_present = [v for v in VARS if v in combined.columns]
monthly      = combined[vars_present].resample("ME").mean()

mci_monthly   = pd.Series(index=monthly.index, dtype=float, name="KZ_MCI")
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

# ── 4. Normalize and save ─────────────────────────────────────────────────────

print("Normalizing and saving...")
valid      = mci_monthly.dropna()
mci_norm   = (mci_monthly - valid.mean()) / valid.std()
mci_norm.name = "KZ_MCI"

# Daily forward-fill
daily_index = pd.date_range(
    mci_norm.dropna().index[0],
    mci_norm.dropna().index[-1], freq="D"
)
mci_daily = mci_norm.reindex(daily_index).ffill()
mci_daily.index.name = "date"
mci_daily.to_csv(HISTORY_CSV, header=True)
print(f"  ✓ Index history → {HISTORY_CSV}  ({len(mci_daily)} rows)")

# Save loadings
if loadings_list:
    loadings_df = pd.DataFrame(loadings_list)
    loadings_df.index.name = "date"
    loadings_df.to_csv(LOADINGS_CSV)
    print(f"  ✓ Loadings → {LOADINGS_CSV}  ({len(loadings_df)} months)")

# ── 5. Update JSON ────────────────────────────────────────────────────────────

print("\nUpdating JSON output...")
os.system("python scripts/daily_score.py")

print("\n── Monthly PCA summary ───────────────────────────────────────────────")
print(f"  Months estimated: {len(valid)}")
print(f"  Latest loading date: {loadings_list[-1].name.strftime('%Y-%m')}")
print(f"  Variance explained (latest): "
      f"{loadings_list[-1]['variance_explained']:.1%}")
print(f"  Latest index reading: {mci_norm.dropna().iloc[-1]:+.3f}σ")
print("\nDone.")
