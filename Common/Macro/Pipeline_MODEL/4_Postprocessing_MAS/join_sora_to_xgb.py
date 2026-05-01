"""
join_sora_to_xgb.py
-------------------
Reads the cleaned MAS daily outputs produced by parse_mas_sora.py:
  - IO/SRC/CSV_MAS/Output/sora_daily.csv
  - IO/SRC/CSV_MAS/Output/sora_3m_daily.csv

Then reproduces the existing join logic:
  - filter xgb_ready down to SGX trading days only (drop weekends and
    exchange holidays using the REIT index file as the trading calendar proxy)
  - append REIT index OHLC values onto the filtered rows
  - shift MAS SORA to T-2 business-day point-in-time safe values
  - keep a separate realized SORA path for future-target construction
  - expand SORA to a full calendar-day index
  - forward-fill weekends / Singapore public holidays
  - derive sora_90d_change
  - derive sora_term_spread
  - derive reit_index_fwd_21d_return from the full REIT index history
  - derive future SORA targets from the underlying realized SORA path:
      * sora_fwd_21d_level
      * sora_fwd_21d_change
      * sora_fwd_21d_abs_change
  - join onto xgb_ready by snapshot_ts

Outputs:
  IO/SRC/MODEL/sora_joined_to_xgb.csv

Join strategy for weekend/holiday gaps:
  SORA is only published on Singapore business days. The xgb_ready file has
  calendar-day rows including weekends. We forward-fill SORA onto non-business
  days so that Monday carries the Friday value (same convention used throughout
  the conversation JSONs for this project).

Point-in-time convention:
  For model R*, MAS SORA is treated as available with a fixed T-2 business-day
  lag relative to the row date. The emitted SORA columns therefore already
  reflect that lagged, point-in-time-safe convention.

Output naming convention:
  The emitted SORA-derived columns are suffixed with `_t2` to make the
  lag convention explicit in the joined dataset.

Future-target convention:
  The future SORA target columns are NOT derived from the lagged `_t2`
  feature columns. They are computed from the underlying realized SORA path,
  then aligned to the SGX-trading-day row universe and shifted forward by
  21 SGX trading rows.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CONSOLIDATED_ROOT = SCRIPT_DIR.parents[1]
IO_SRC_DIR = CONSOLIDATED_ROOT / "IO" / "SRC"
FED_OUTPUT_DIR = IO_SRC_DIR / "CSV_FED" / "Output"
MAS_OUTPUT_DIR = IO_SRC_DIR / "CSV_MAS" / "Output"
MODEL_DIR = IO_SRC_DIR / "MODEL"

XGB_READY = FED_OUTPUT_DIR / "timeseries_2022-07-27_2026-03-18_xgb_ready.csv"
SGX_REIT_INDEX = MODEL_DIR / "SGX_DLY_REIT, 1D.csv"
SORA_DAILY = MAS_OUTPUT_DIR / "sora_daily.csv"
SORA_3M_DAILY = MAS_OUTPUT_DIR / "sora_3m_daily.csv"
OUT_JOINED = MODEL_DIR / "sora_joined_to_xgb.csv"

# ---------------------------------------------------------------------------
# Load cleaned SORA outputs
# ---------------------------------------------------------------------------
print("Loading cleaned SORA daily files ...")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
sora = pd.read_csv(SORA_DAILY, parse_dates=["value_date"])
sora_3m = pd.read_csv(SORA_3M_DAILY, parse_dates=["value_date"])

# Preserve the realized, non-lagged SORA series for future-target construction.
sora_realized = sora.copy()

# Apply fixed T-2 business-day lag before any calendar expansion.
# This makes the emitted SORA fields point-in-time safe relative to the
# Parquet snapshot convention used in the project.
sora["sora_level"] = sora["sora_level"].shift(2)
sora_3m["sora_3m"] = sora_3m["sora_3m"].shift(2)

# ---------------------------------------------------------------------------
# Load SGX trading calendar proxy from REIT index history
# ---------------------------------------------------------------------------
print("Loading SGX trading-day calendar proxy ...")
reit_index = pd.read_csv(SGX_REIT_INDEX)
reit_index["snapshot_ts"] = pd.to_datetime(
    reit_index["time"], unit="s", utc=True
).dt.tz_localize(None).dt.normalize()
reit_index = reit_index.sort_values("snapshot_ts").reset_index(drop=True)
reit_index["reit_index_fwd_21d_return"] = (
    reit_index["close"].shift(-21) - reit_index["close"]
) / reit_index["close"]
reit_index["reit_index_fwd_21d_return"] = reit_index["reit_index_fwd_21d_return"].round(6)
reit_index = reit_index.rename(columns={
    "open": "reit_index_open",
    "high": "reit_index_high",
    "low": "reit_index_low",
    "close": "reit_index_close",
})
sgx_trading_days = (
    reit_index["snapshot_ts"].drop_duplicates().sort_values().reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# Build realized SORA path on the same SGX trading-day row universe
# ---------------------------------------------------------------------------
realized_cal_start = sora_realized["value_date"].min()
realized_cal_end = sora_realized["value_date"].max()
realized_cal_index = pd.DataFrame(
    {"value_date": pd.date_range(realized_cal_start, realized_cal_end, freq="D")}
)

realized_sora_cal = realized_cal_index.merge(
    sora_realized, on="value_date", how="left"
)
realized_sora_cal["sora_level_realized"] = realized_sora_cal["sora_level"].ffill()
realized_sora_cal = realized_sora_cal[["value_date", "sora_level_realized"]].copy()
realized_sora_cal = realized_sora_cal.rename(columns={"value_date": "snapshot_ts"})
realized_sora_cal["snapshot_ts"] = realized_sora_cal["snapshot_ts"].dt.normalize()

# ---------------------------------------------------------------------------
# Expand to full calendar-day index and forward-fill across weekends/holidays
# ---------------------------------------------------------------------------
cal_start = sora["value_date"].min()
cal_end = sora["value_date"].max()
cal_index = pd.DataFrame(
    {"value_date": pd.date_range(cal_start, cal_end, freq="D")}
)

sora_cal = cal_index.merge(sora, on="value_date", how="left")
sora_cal = sora_cal.merge(sora_3m, on="value_date", how="left")

# Forward-fill: Saturday/Sunday (and public holidays) get the last business-day value
sora_cal["sora_level"] = sora_cal["sora_level"].ffill()
sora_cal["sora_3m"] = sora_cal["sora_3m"].ffill()

# Derive sora_90d_change in basis points
# 90 calendar days back = shift by 90 rows in the daily calendar
sora_cal = sora_cal.set_index("value_date")
sora_cal["sora_90d_change"] = (
    (sora_cal["sora_level"] - sora_cal["sora_level"].shift(90)) * 100
).round(4)

# Derive term spread: 3M compounded minus overnight (in percentage points)
sora_cal["sora_term_spread"] = (
    sora_cal["sora_3m"] - sora_cal["sora_level"]
).round(4)

# Rename derived / lagged SORA columns to make the T-2 convention explicit
sora_cal = sora_cal.rename(columns={
    "sora_level": "sora_level_t2",
    "sora_90d_change": "sora_90d_change_t2",
    "sora_3m": "sora_3m_t2",
    "sora_term_spread": "sora_term_spread_t2",
})

sora_cal = sora_cal.reset_index()
sora_cal = sora_cal.rename(columns={"value_date": "date"})

# ---------------------------------------------------------------------------
# Join to xgb_ready
# ---------------------------------------------------------------------------
print("\nLoading xgb_ready ...")
xgb = pd.read_csv(XGB_READY, parse_dates=["snapshot_ts"])
xgb["snapshot_ts"] = xgb["snapshot_ts"].dt.normalize()

print("Filtering xgb_ready to SGX trading days only ...")
raw_rows = len(xgb)
xgb = xgb[xgb["snapshot_ts"].isin(sgx_trading_days)].copy()
filtered_rows = len(xgb)
print(f"  Kept {filtered_rows:,} / {raw_rows:,} rows after dropping weekends and "
      "SGX non-trading days.")

print("Appending REIT index time-series values ...")
xgb = xgb.merge(
    reit_index[["snapshot_ts", "reit_index_open", "reit_index_high",
                "reit_index_low", "reit_index_close",
                "reit_index_fwd_21d_return"]],
    on="snapshot_ts",
    how="left",
)

print("Appending realized SORA path for future-target construction ...")
xgb = xgb.merge(
    realized_sora_cal[["snapshot_ts", "sora_level_realized"]],
    on="snapshot_ts",
    how="left",
)

print("Computing future SORA target columns from realized SORA path ...")
xgb = xgb.sort_values("snapshot_ts").reset_index(drop=True)
xgb["sora_fwd_21d_level"] = xgb["sora_level_realized"].shift(-21)
xgb["sora_fwd_21d_change"] = (
    xgb["sora_fwd_21d_level"] - xgb["sora_level_realized"]
).round(6)
xgb["sora_fwd_21d_abs_change"] = xgb["sora_fwd_21d_change"].abs().round(6)

sora_cal_for_join = sora_cal.rename(columns={"date": "snapshot_ts"})

joined = xgb.merge(
    sora_cal_for_join[["snapshot_ts", "sora_level_t2", "sora_90d_change_t2",
                       "sora_3m_t2", "sora_term_spread_t2"]],
    on="snapshot_ts",
    how="left",
)

n_null_sora = joined["sora_level_t2"].isna().sum()
if n_null_sora > 0:
    print(f"  WARNING: {n_null_sora} rows in xgb_ready have no SORA match "
          f"(xgb_ready dates outside SORA coverage)")
else:
    print("  All xgb_ready rows matched — no SORA gaps.")

joined.to_csv(OUT_JOINED, index=False)
print(f"Saved: {OUT_JOINED.name}")
print(f"  Final shape: {joined.shape[0]:,} rows × {joined.shape[1]} columns")
print(f"  Columns: {list(joined.columns)}")

# ---------------------------------------------------------------------------
# Quick sanity printout
# ---------------------------------------------------------------------------
print("\n--- Sample rows (mid-series) ---")
sample = joined[joined["snapshot_ts"].dt.year == 2023].head(5)
print(sample[["snapshot_ts", "days_to_next_fomc", "expected_bps",
              "sora_level_t2", "sora_90d_change_t2",
              "sora_3m_t2", "sora_term_spread_t2",
              "sora_fwd_21d_level", "sora_fwd_21d_change",
              "sora_fwd_21d_abs_change"]].to_string(index=False))



"""
# How the non-trading days are dropped (explainer)

This script extracts Singapore Exchange (SGX) trading days by using a historical price file as a **physical proxy** for the trading calendar. Instead of manually listing holidays or using a library, it assumes that if a trade occurred in the `SGX_REIT_INDEX` file, the exchange was open.

### 1. Data Normalization
The script first cleans the timestamp data in the REIT index file to ensure it matches the format of your target dataset.

* **Timestamp Conversion:** It converts the `time` column (Unix seconds) into a UTC datetime object.
* **Timezone Removal:** The `.dt.tz_localize(None)` call strips timezone awareness to prevent "offset-naive" vs "offset-aware" merge errors.
* **Time Stripping:** The `.dt.normalize()` function sets all times to midnight ($00:00:00$). This ensures a clean date-to-date comparison, as trading hours (e.g., $17:00:00$) would otherwise fail to match a date-only index.

### 2. Creating the Trading Day Proxy
The core logic defines "Trading Days" based on empirical activity:

```python
sgx_trading_days = (
    reit_index["snapshot_ts"].drop_duplicates().sort_values().reset_index(drop=True)
)
```

* **Empirical Filtering:** By dropping duplicates and sorting the `snapshot_ts` from the REIT file, the script creates a Series containing only the dates where the SGX was actually operational. 
* **Automatic Holiday Handling:** This method automatically excludes weekends and Singapore public holidays (like Chinese New Year or National Day) because no price data exists for those dates in the source file.

### 3. Filtering the Target Dataset
Finally, the script uses this proxy to prune your `xgb_ready` data:

```python
xgb = xgb[xgb["snapshot_ts"].isin(sgx_trading_days)].copy()
```

The `.isin()` operator acts as a logical gate. Any row in your main dataset whose date does not exist in the REIT index (meaning the exchange was closed) is dropped. This aligns your final features with the actual rhythm of SGX trading sessions.
"""
