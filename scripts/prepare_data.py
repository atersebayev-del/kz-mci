"""
KASE Data Preparation & NBK Base Rate Processing
=================================================
1. Converts NBK base rate announcements to a daily forward-filled series
2. Constructs derived spread variables
3. Merges all series into a single clean daily dataset ready for PCA

Usage:
    python prepare_data.py

Input:
    ./kase_data/  — folder with CSVs from kase_downloader.py

Output:
    ./kase_data/nbk_base_rate_daily.csv   — daily base rate series
    ./kase_data/dataset_raw.csv           — all raw series merged
    ./kase_data/dataset_spreads.csv       — derived spread variables
    ./kase_data/dataset_final.csv         — final clean dataset for PCA
"""

import pandas as pd
import numpy as np
import os

DATA_DIR   = "kase_data"
START_DATE = "2015-01-05"   # first trading day in your data
END_DATE   = pd.Timestamp.today().strftime("%Y-%m-%d")

# ── 1. NBK Base Rate ──────────────────────────────────────────────────────────

base_rate_announcements = """Announcement,Base Rate,Lower Bound,Upper Bound
02.09.2015,12.00,7.00,17.00
02.10.2015,16.00,15.00,17.00
02.02.2016,17.00,15.00,19.00
06.05.2016,15.00,14.00,16.00
12.07.2016,13.00,12.00,14.00
04.10.2016,12.50,11.50,13.50
15.11.2016,12.00,11.00,13.00
21.02.2017,11.00,10.00,12.00
06.06.2017,10.50,9.50,11.50
22.08.2017,10.25,9.25,11.25
16.01.2018,9.75,8.75,10.75
06.03.2018,9.50,8.50,10.50
17.04.2018,9.25,8.25,10.25
05.06.2018,9.00,8.00,10.00
16.10.2018,9.25,8.25,10.25
16.04.2019,9.00,8.00,10.00
10.09.2019,9.25,8.25,10.25
10.03.2020,12.00,10.50,13.50
06.04.2020,9.50,7.50,11.50
21.07.2020,9.00,7.50,10.50
15.12.2020,9.00,8.00,10.00
27.07.2021,9.25,8.25,10.25
14.09.2021,9.50,8.50,10.50
26.10.2021,9.75,8.75,10.75
25.01.2022,10.25,9.25,11.25
24.02.2022,13.50,12.50,14.50
26.04.2022,14.00,13.00,15.00
26.07.2022,14.50,13.50,15.50
27.10.2022,16.00,15.00,17.00
06.12.2022,16.75,15.75,17.75
28.08.2023,16.50,15.50,17.50
09.10.2023,16.00,15.00,17.00
27.11.2023,15.75,14.75,16.75
22.01.2024,15.25,14.25,16.25
26.02.2024,14.75,13.75,15.75
03.06.2024,14.50,13.50,15.50
15.07.2024,14.25,13.25,15.25
02.12.2024,15.25,14.25,16.25
11.03.2025,16.50,15.50,17.50
13.10.2025,18.00,17.00,19.00
08.06.2026,17.00,16.00,18.00"""

from io import StringIO
br = pd.read_csv(StringIO(base_rate_announcements), parse_dates=["Announcement"],
                 dayfirst=True)
br = br.set_index("Announcement").sort_index()
br.columns = ["base_rate", "lower_bound", "upper_bound"]

# Forward-fill across full daily calendar
full_index = pd.date_range(START_DATE, END_DATE, freq="D")
br_daily   = br.reindex(full_index).ffill()

# Before first announcement (pre Sep 2015) — NBK base rate was 5.5%
# (introduced as instrument in Aug 2015, prior rate was refinancing rate ~5.5%)
br_daily   = br_daily.fillna(5.5)
br_daily.index.name = "date"

out_path = os.path.join(DATA_DIR, "nbk_base_rate_daily.csv")
br_daily.to_csv(out_path)
print(f"✓ NBK base rate daily series → {out_path}  ({len(br_daily)} rows)")

# ── 2. Load KASE series ───────────────────────────────────────────────────────

def load(symbol, col="close"):
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"  ⚠️  Missing file: {path}")
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df[col].rename(f"{symbol}_{col}")

# Close prices
usdkzt_tod     = load("USDKZT_TOD")
usdkzt_tod_vol = load("USDKZT_TOD", "volume")
usdkzt   = load("USDKZT_TOM")
rubkzt   = load("RUBKZT_TOM")
tonia    = load("TONIA")
twina    = load("TWINA")
trion    = load("TRION")
swap1d   = load("SWAP_1D")
swap2d   = load("SWAP_2D")
# MM_INDEX dropped — composite of TONIA/TWINA/SWAP already included

# Volumes
usdkzt_vol = load("USDKZT_TOM", "volume")
tonia_vol  = load("TONIA",      "volume")
swap1d_vol = load("SWAP_1D",    "volume")
swap2d_vol = load("SWAP_2D",    "volume")

# ── 3. Merge raw series ───────────────────────────────────────────────────────

raw = pd.concat([
    usdkzt, usdkzt_tod, rubkzt,
    tonia, twina, trion,
    swap1d, swap2d,
    usdkzt_vol, usdkzt_tod_vol, tonia_vol, swap1d_vol, swap2d_vol,
    br_daily["base_rate"],
], axis=1)

# Keep only trading days (where USDKZT or TONIA has data)
trading_days = raw[["USDKZT_TOM_close", "TONIA_close"]].dropna(how="all").index
raw = raw.loc[trading_days]

raw.to_csv(os.path.join(DATA_DIR, "dataset_raw.csv"))
print(f"✓ Raw dataset → {DATA_DIR}/dataset_raw.csv  ({len(raw)} rows x {len(raw.columns)} cols)")

# ── 4. Derived spread variables ───────────────────────────────────────────────

spreads = pd.DataFrame(index=raw.index)

# ── Combine USDKZT TOD + TOM ──────────────────────────────────────────────────
# Volume-weighted average rate, summed volume across both settlement types
tom_vol = raw["USDKZT_TOM_volume"].fillna(0)
tod_vol = raw["USDKZT_TOD_volume"].fillna(0)
tom_rate = raw["USDKZT_TOM_close"]
tod_rate = raw["USDKZT_TOD_close"]
total_vol = tom_vol + tod_vol
tom_contrib = tom_rate.fillna(0) * tom_vol
tod_contrib = tod_rate.fillna(0) * tod_vol
usdkzt_combined = (tom_contrib + tod_contrib) / total_vol.replace(0, np.nan)
# Restore NaN where both rates were NaN (genuine no-trade day)
both_nan = tom_rate.isna() & tod_rate.isna()
usdkzt_combined[both_nan] = np.nan
usdkzt_combined_vol = total_vol.replace(0, np.nan)

# FX stress
log_ret                    = np.log(usdkzt_combined).diff()
spreads["usdkzt_ret"]      = log_ret.abs()                                  # absolute return — turbulence
spreads["usdkzt_ret_5d"]   = log_ret.rolling(5).mean()                      # signed 5d MA — directional pressure
spreads["usdkzt_ret_21d"]  = log_ret.rolling(21).mean()                     # signed 21d MA — persistent depreciation
spreads["usdkzt_vol_5d"]   = log_ret.rolling(5).std() * np.sqrt(252) # annualised 5d realised vol
spreads["usdkzt_vol_21d"]  = log_ret.rolling(21).std() * np.sqrt(252)# annualised 21d realised vol
spreads["usdkzt_volume"]   = usdkzt_combined_vol

# Triangular arbitrage (RUB/USD implied via KZT vs direct)
# Implied RUBUSD = RUBKZT / USDKZT
# Deviation = implied - actual (need actual RUBUSD — placeholder for now)
spreads["rubkzt_close"]    = raw["RUBKZT_TOM_close"]   # keep for triangular arb later

# Money market spreads vs base rate
spreads["tonia_spread"]    = raw["TONIA_close"]  - raw["base_rate"]   # TONIA vs base rate
spreads["twina_spread"]    = raw["TWINA_close"]  - raw["base_rate"]   # 14d repo vs base rate
spreads["trion_spread"]    = raw["TRION_close"]  - raw["base_rate"]   # 7d repo vs base rate
spreads["swap1d_spread"]   = raw["SWAP_1D_close"] - raw["TONIA_close"] # swap premium over overnight
spreads["swap2d_spread"]   = raw["SWAP_2D_close"] - raw["TONIA_close"] # swap premium over overnight

# Term structure spreads
spreads["term_spread_7d"]  = raw["TRION_close"]  - raw["TONIA_close"]  # 7d minus overnight
spreads["term_spread_14d"] = raw["TWINA_close"]  - raw["TONIA_close"]  # 14d minus overnight

# Volume signals (log to handle scale)
spreads["tonia_vol_log"]   = np.log(raw["TONIA_volume"].replace(0, np.nan))
spreads["swap1d_vol_log"]  = np.log(raw["SWAP_1D_volume"].replace(0, np.nan))
spreads["usdkzt_vol_log"]  = np.log(usdkzt_combined_vol)


spreads.to_csv(os.path.join(DATA_DIR, "dataset_spreads.csv"))
print(f"✓ Spreads dataset → {DATA_DIR}/dataset_spreads.csv  ({len(spreads.columns)} variables)")

# ── 5. Final dataset summary ──────────────────────────────────────────────────

print("\n── Variable coverage ─────────────────────────────────────────────────")
summary = pd.DataFrame({
    "first_obs": spreads.apply(lambda x: x.first_valid_index()),
    "last_obs":  spreads.apply(lambda x: x.last_valid_index()),
    "n_obs":     spreads.apply(lambda x: x.count()),
    "pct_missing": (spreads.isna().mean() * 100).round(1),
})
print(summary.to_string())

# Save final (drop cols with >50% missing)
final = spreads.loc[START_DATE:].copy()
missing_pct = final.isna().mean()
dropped = missing_pct[missing_pct > 0.5].index.tolist()
if dropped:
    print(f"\n⚠️  Dropping {len(dropped)} variables with >50% missing: {dropped}")
    final = final.drop(columns=dropped)

final.to_csv(os.path.join(DATA_DIR, "dataset_final.csv"))
print(f"\n✓ Final dataset → {DATA_DIR}/dataset_final.csv")
print(f"  {len(final)} rows  x  {len(final.columns)} variables")
print(f"  Period: {final.index[0].date()} → {final.index[-1].date()}")
print("\nDone.")
