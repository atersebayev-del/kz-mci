# KZ-MCI — Production Pipeline (this repo)

Kazakhstan Market Conditions Index. This repo feeds **eigenaxis.io/insights**
via a Framer frontend. `data/kz_mci_latest.json` is the file the frontend
actually reads — everything else in `data/` is intermediate/cache state.

## Architecture

- `scripts/update_index.py` — the only entry point. Runs daily via GitHub
  Actions (19:00 Almaty / 13:00 UTC) and does a full monthly PCA reestimation
  on the 1st of each month (or when `FORCE_MONTH_START=true` is set).
- `scripts/utils.py` — shared KASE fetch + variable-construction helpers.

## Data flow (in `data/`)

| File | What it is |
|---|---|
| `dataset_final.csv` | Full historical daily variables, 2015→present. Source of truth for scoring. |
| `kz_mci_daily_raw.csv` | **Cache** of un-normalized PCA raw scores per day. Once a day is outside the current month its raw score is final and never recomputed — only current-month days get refit each run. Keeps runs fast (~1s incremental vs ~4s full backfill). |
| `kz_mci_daily.csv` | `kz_mci_daily_raw.csv` renormalized with current `norm_mean`/`norm_std`. Cheap, recomputed every run. |
| `kz_mci_monthly.csv`, `kz_mci_loadings.csv`, `kz_mci_norm_constants.json` | Written only on month-start PCA reestimation. |
| `kz_mci_latest.json` | Final output. Rebuilt every run from the above. This is what the frontend consumes. |

## Methodology (do not silently change)

Daily scores use **full-PCA-per-day**: each day is treated as if it were the
current month's sole observation, and a fresh expanding-window PCA is fit for
it. This must stay in sync with `local_analysis/option_a_preview.py` on the
same machine — that script is the local reference implementation for this
exact methodology. If the two ever diverge, trust `option_a_preview.py` and
port the fix here, not the other way around.

**Normalization stays rolling (recomputed every month-start), by explicit
decision.** `local_analysis/frozen_norm_backtest.py` explored switching to
frozen normalization to eliminate month-boundary jumps — that is a local-only
experiment. Do not port frozen normalization to this production script
without an explicit instruction to do so.

## Known failure modes — already fixed once, don't reintroduce

1. **Never let git conflict markers leak into a CSV's date column
   unresolved.** `_coerce_datetime_index()` in `update_index.py` guards
   against this (drops + logs unparseable index values) — keep it on every
   CSV read in the pipeline.
2. **Never merge freshly-fetched data into `dataset_final.csv` with a plain
   `concat` + `keep="last"`.** The fetch window is short (`LOOKBACK_DAYS`),
   so rolling 5d/21d stats are NaN for the first ~21 trading days of every
   fetch — a naive merge lets those NaNs silently overwrite good stored
   history on every single run. Use the `combine_first` pattern (new value
   wins only when non-null, existing value is kept otherwise) — see step 3
   of `update_index.py`.
3. **Never hardcode a timestamp/date constant.** `INDICATOR_RECENT_TO` was
   once frozen as a literal epoch value and silently stopped fetching new
   indicator data once that date passed. Always compute "today" dynamically
   at call time (`_today_utc()` in `utils.py`).
4. **Never score a day off partial data.** If any required variable is
   missing for a specific day, skip scoring it (log it, retry next run)
   rather than filling the gap with a column mean — that flattens real
   signal (this is how a real February volatility spike got silently
   smoothed away and took an entire debugging session to find).

## Missing-data handling — daily and monthly diverge, by design

Reviewed 2026-07-14. If daily and monthly readings ever disagree in a way
that isn't explained by the known monthly-staleness behavior above, check
whether a missing/sparse variable is being handled differently on the two
paths — this is the likely next place to look, and is plausibly what the
2026-07-03 "daily/monthly chart divergence" fix (commit `940565b`) ran into.

- **Fetch failures** (`utils.py: _fetch_single`): 3 retries, then `None` on
  failure, logged to stdout only (`✗ Failed {symbol}`) — no exception, no
  alert. A dead feed only surfaces if someone reads the Action log.
- **Variable construction** (`utils.py: build_variables`): no imputation,
  NaN propagates directly. Money-market spreads have no redundancy — a
  missing `TONIA_close` blanks `tonia_spread`, `term_spread_14d`,
  `swap1d_spread`, and `swap2d_spread` all at once. USD/KZT is the one
  exception (volume-blended TOM/TOD, only NaN if both legs are missing).
- **Daily scoring** (`update_index.py:286-299`) — **strict, skip entirely**:
  if any of the 13 variables is missing for a day, that day gets no score at
  all (absent from `kz_mci_daily*.csv`, retried next run). This replaced an
  earlier mean-fill after it silently smoothed away a real Feb volatility
  spike (see failure mode #4 above).
- **Monthly scoring** (`update_index.py:174-188`) — **lenient, drop-then-
  impute**: drops a variable if >40% missing across the window, drops a
  month if >50% of variables missing, then fills any remaining gaps with
  the column mean (`window.fillna(window.mean())`) — the exact behavior the
  daily path was rewritten to avoid.

Net effect: a missing day leaves a self-healing gap in the daily series with
no alert; a missing variable quietly stops updating part of the daily score
for as long as the feed is down; and the daily/monthly asymmetry (skip vs.
impute) means the same gap can be treated completely differently depending
on which series you're looking at.

## Two separate `dataset_final.csv` files exist — do not confuse them

- `kz-mci/data/dataset_final.csv` (this repo) — built incrementally by
  `update_index.py`'s own KASE fetch, `LOOKBACK_DAYS` window only.
- `local_analysis/kase_data/dataset_final.csv` — built from scratch every
  time by `local_analysis/prepare_data.py`, using `local_analysis/
  kase_downloader.py`'s full 2015→present history. This one has no
  rolling-window edge effects and is generally the more complete/reliable
  source when the two disagree.

## Auth / git workflow notes

- Push via GitHub Desktop — terminal `git push` currently prompts for a
  password GitHub no longer accepts; a Personal Access Token hasn't been
  set up yet. GitHub Desktop's saved credentials work fine.
- GitHub Actions commits to this repo independently on its own schedule.
  Expect "newer commits on remote" / merge-conflict prompts when pushing
  local changes — this is normal, not a sign something's broken. For
  conflicts on `data/*.csv` or `data/*.json` after a local pipeline run,
  keeping the local version (`git checkout --ours <file>`) is usually
  correct, since the Action's version was likely generated by whatever
  script was live *before* your fix.
