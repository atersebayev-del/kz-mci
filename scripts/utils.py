"""
utils.py — Shared helper functions for KZ-MCI pipeline
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, date
import time


# ── KASE API ──────────────────────────────────────────────────────────────────

KASE_BASE = "https://kase.kz/tv-charts"

INSTRUMENTS = {
    "USDKZT_TOM": "currency",
    "USDKZT_TOD": "currency",
    "TONIA":      "indicators",
    "TWINA":      "indicators",
    "SWAP_1D":    "indicators",
    "SWAP_2D":    "indicators",
}

# KASE uses specific from/to ranges for recent indicator data
# These match what KASE's chart actually fetches for recent data
INDICATOR_RECENT_FROM = 1754524800   # 2025-08-07 — start of recent chunk
INDICATOR_RECENT_TO   = 1783036800   # 2026-07-03 — end of recent chunk

def to_unix(date_str):
    # Use UTC midnight to avoid local timezone shifting dates
    from datetime import timezone
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def _fetch_single(symbol, endpoint_type, from_ts, to_ts, countback, retries=3):
    """Single KASE API call — internal helper."""
    url    = f"{KASE_BASE}/{endpoint_type}/history"
    params = {
        "symbol":              symbol,
        "resolution":          "1D",
        "from":                from_ts,
        "to":                  to_ts,
        "countback":           countback,
        "chart_language_code": "ru",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("s") != "ok" or not data.get("t"):
                return None
            df = pd.DataFrame({
                "date":   pd.to_datetime(data["t"], unit="s").normalize(),
                "open":   data["o"],
                "high":   data["h"],
                "low":    data["l"],
                "close":  data["c"],
                "volume": data.get("v", [None] * len(data["t"])),
            }).set_index("date").sort_index()
            # Drop rows where close is None/NaN (unpublished future dates)
            df = df.dropna(subset=["close"])
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  ✗ Failed {symbol}: {e}")
                return None

def fetch_kase(symbol, endpoint_type, start, end, retries=3):
    """Fetch daily OHLCV data from KASE chart API.
    
    For indicators, fetches in two chunks to ensure recent data is captured:
    - Historical chunk: start → Aug 2025 (using standard date range)  
    - Recent chunk: Aug 2025 → today (using KASE's known recent chunk params)
    Then merges both.
    """
    if endpoint_type == "indicators":
        # Chunk 1: historical data
        df_hist = _fetch_single(
            symbol, endpoint_type,
            to_unix(start), INDICATOR_RECENT_FROM,
            countback=3000
        )
        # Chunk 2: recent data using KASE's known working parameters
        df_recent = _fetch_single(
            symbol, endpoint_type,
            INDICATOR_RECENT_FROM, INDICATOR_RECENT_TO,
            countback=330
        )
        # Merge both chunks
        frames = [f for f in [df_hist, df_recent] if f is not None and not f.empty]
        if not frames:
            return None
        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined
    else:
        # Currency instruments — single fetch works fine
        return _fetch_single(
            symbol, endpoint_type,
            to_unix(start), to_unix(end),
            countback=5000
        )
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("s") != "ok":
                return None
            df = pd.DataFrame({
                "date":   pd.to_datetime(data["t"], unit="s").normalize(),
                "open":   data["o"],
                "high":   data["h"],
                "low":    data["l"],
                "close":  data["c"],
                "volume": data.get("v", [None] * len(data["t"])),
            }).set_index("date").sort_index()
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  ✗ Failed to fetch {symbol}: {e}")
                return None


# ── NBK base rate ─────────────────────────────────────────────────────────────

def load_base_rate(path="data/nbk_base_rate_daily.csv"):
    """Load NBK base rate daily series."""
    br = pd.read_csv(path, index_col=0, parse_dates=True)
    br.index.name = "date"
    return br["base_rate"]


# ── Variable construction ─────────────────────────────────────────────────────

def build_variables(raw, base_rate):
    """
    Construct all 13 index variables from raw KASE data and NBK base rate.
    Returns a single-row or multi-row DataFrame of variables.
    """
    spreads = pd.DataFrame(index=raw.index)

    # ── USDKZT combined TOD + TOM ─────────────────────────────────────────────
    tom_vol  = raw.get("USDKZT_TOM_volume", pd.Series(0, index=raw.index)).fillna(0)
    tod_vol  = raw.get("USDKZT_TOD_volume", pd.Series(0, index=raw.index)).fillna(0)
    tom_rate = raw.get("USDKZT_TOM_close",  pd.Series(dtype=float))
    tod_rate = raw.get("USDKZT_TOD_close",  pd.Series(dtype=float))

    tom_contrib    = tom_rate.fillna(0) * tom_vol
    tod_contrib    = tod_rate.fillna(0) * tod_vol
    total_vol      = tom_vol + tod_vol
    usdkzt         = (tom_contrib + tod_contrib) / total_vol.replace(0, np.nan)
    both_nan       = tom_rate.isna() & tod_rate.isna()
    usdkzt[both_nan] = np.nan
    usdkzt_vol     = total_vol.replace(0, np.nan)

    # ── FX cluster ───────────────────────────────────────────────────────────
    log_ret                    = np.log(usdkzt).diff()
    spreads["usdkzt_ret"]      = log_ret.abs()
    spreads["usdkzt_ret_5d"]   = log_ret.rolling(5).mean()
    spreads["usdkzt_ret_21d"]  = log_ret.rolling(21).mean()
    spreads["usdkzt_vol_5d"]   = log_ret.rolling(5).std()  * np.sqrt(252)
    spreads["usdkzt_vol_21d"]  = log_ret.rolling(21).std() * np.sqrt(252)
    spreads["usdkzt_vol_log"]  = np.log(usdkzt_vol)

    # ── Money market cluster ──────────────────────────────────────────────────
    br = base_rate.reindex(raw.index).ffill()

    spreads["tonia_spread"]    = raw["TONIA_close"]   - br
    spreads["twina_spread"]    = raw["TWINA_close"]   - br
    spreads["term_spread_14d"] = raw["TWINA_close"]   - raw["TONIA_close"]
    spreads["swap1d_spread"]   = raw["SWAP_1D_close"] - raw["TONIA_close"]
    spreads["swap2d_spread"]   = raw["SWAP_2D_close"] - raw["TONIA_close"]
    spreads["tonia_vol_log"]   = np.log(
        raw["TONIA_volume"].replace(0, np.nan))
    spreads["swap1d_vol_log"]  = np.log(
        raw.get("SWAP_1D_volume", pd.Series(dtype=float)).replace(0, np.nan))

    return spreads


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_with_loadings(variables_row, loadings_row, mean_vals, std_vals):
    """
    Score a single observation using pre-computed PCA loadings.

    Args:
        variables_row: pd.Series of variable values for one date
        loadings_row:  pd.Series of PCA loadings (from loadings file)
        mean_vals:     pd.Series of variable means (for standardization)
        std_vals:      pd.Series of variable stds  (for standardization)

    Returns:
        float: raw PCA score (not yet normalized to index scale)
    """
    # Align variables to loading columns
    cols = [c for c in loadings_row.index
            if c != "variance_explained" and c in variables_row.index]

    vals     = variables_row[cols].fillna(mean_vals[cols])
    scaled   = (vals - mean_vals[cols]) / std_vals[cols].replace(0, np.nan)
    score    = (scaled * loadings_row[cols]).sum()
    return score

def load_and_merge(symbol, endpoint_type, existing_csv_path, 
                   lookback_days=90, col="close"):
    """
    Load existing CSV data, fetch only recent data from KASE,
    merge and return combined series. Much faster than full re-download.
    """
    import os
    from datetime import date, timedelta

    end_date   = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Fetch recent data
    df_new = fetch_kase(symbol, endpoint_type, start_date, end_date)

    # Load existing if available
    if os.path.exists(existing_csv_path):
        df_old = pd.read_csv(existing_csv_path, index_col="date", parse_dates=True)
        if df_new is not None and not df_new.empty:
            # Merge: old data + new data, new takes precedence for overlapping dates
            combined = pd.concat([df_old, df_new])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            return combined
        return df_old
    return df_new
