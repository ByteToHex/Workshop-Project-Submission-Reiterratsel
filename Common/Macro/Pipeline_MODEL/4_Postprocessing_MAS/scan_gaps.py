"""
scan_gaps.py — coverage and intra-period gap scanner across all data sources.
Outputs:
  1. Per-ticker date range + intra-period gaps (>7 calendar days)
  2. SORA date range + gaps
  3. Parquet date range + gaps
  4. Intersection summary
"""
import os, csv, re
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
import pandas as pd

SGT = timezone(timedelta(hours=8))
SCRIPT_DIR = Path(__file__).resolve().parent
CONSOLIDATED_ROOT = SCRIPT_DIR.parents[1]
IO_SRC_DIR = CONSOLIDATED_ROOT / "IO" / "SRC"
TV_DIR = IO_SRC_DIR / "CSV_TICKER"
RATE14 = IO_SRC_DIR / "CSV_MAS" / "Input" / "DomesticInterestRates_Idx14_SORA.csv"
RATE17 = IO_SRC_DIR / "CSV_MAS" / "Input" / "DomesticInterestRates_idx17_SORA3MthCompounded.csv"
PM_PATH = IO_SRC_DIR / "CSV_FED" / "Output" / "timeseries_2022-07-27_2026-03-18_xgb_ready.csv"

MONTH_MAP = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

def gaps_over_n(dates_sorted, n_days=7):
    """Return list of (gap_start, gap_end, delta_days) for gaps > n_days."""
    result = []
    for i in range(1, len(dates_sorted)):
        delta = (dates_sorted[i] - dates_sorted[i-1]).days
        if delta > n_days:
            result.append((dates_sorted[i-1], dates_sorted[i], delta))
    return result

# ── 1. TradingView ──────────────────────────────────────────────────────────
print("=" * 70)
print("TRADINGVIEW FILES")
print("=" * 70)

tv_ranges = {}   # ticker -> (first, last, sorted_dates)
tv_files = sorted(f for f in os.listdir(TV_DIR) if f.endswith('.csv') and f.startswith('SGX_DLY_'))
for fname in tv_files:
    with open(TV_DIR / fname) as fh:
        rows = list(csv.reader(fh))
    data = [r for r in rows[1:] if r and r[0].strip()]
    dates = sorted(set(
        datetime.fromtimestamp(int(float(r[0])), tz=timezone.utc)
                 .astimezone(SGT).date() for r in data
    ))
    ticker = re.sub(r'SGX_DLY_|, 1D.*$', '', fname)
    big_gaps = gaps_over_n(dates, n_days=7)
    tv_ranges[ticker] = (dates[0], dates[-1], dates)
    status = f"{dates[0]} to {dates[-1]}  rows={len(dates)}"
    print(f"\n{ticker}: {status}")
    if big_gaps:
        for g in big_gaps:
            print(f"  GAP >7d: {g[0]} -> {g[1]}  ({g[2]} days)")
    else:
        print("  No gaps >7 days (only normal weekends/holidays)")

# ── 2. SORA (MAS) ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MAS SORA FILES")
print("=" * 70)

def parse_mas_dates(path):
    dates = []
    cur_year = cur_month = None
    with open(path, encoding='utf-8') as fh:
        for lineno, raw in enumerate(fh, 1):
            if lineno <= 7: continue
            line = raw.rstrip('\n')
            if line.startswith('"'): break
            if not line.strip() or line.startswith('SORA Value Date'): continue
            parts = line.split(',')
            if len(parts) < 5: continue
            y, m, d = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if y: cur_year = int(y)
            if m: cur_month = MONTH_MAP[m]
            if not d or cur_year is None or cur_month is None: continue
            try:
                dates.append(date(cur_year, cur_month, int(d)))
            except ValueError:
                continue
    return sorted(set(dates))

for label, path in [('SORA overnight (idx14)', RATE14),
                    ('SORA 3M compounded (idx17)', RATE17)]:
    dates = parse_mas_dates(path)
    big_gaps = gaps_over_n(dates, n_days=5)  # MAS: flag gaps >5 days (long weekends etc.)
    print(f"\n{label}: {dates[0]} to {dates[-1]}  rows={len(dates)}")
    if big_gaps:
        for g in big_gaps:
            print(f"  GAP >5d: {g[0]} -> {g[1]}  ({g[2]} days)")
    else:
        print("  No gaps >5 days")

# ── 3. Parquet ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PARQUET (xgb_ready)")
print("=" * 70)

pm = pd.read_csv(PM_PATH, parse_dates=['snapshot_ts'])
pm_dates = sorted(pm['snapshot_ts'].dt.date.unique())
pm_gaps = gaps_over_n(pm_dates, n_days=3)
print(f"\nParquet: {pm_dates[0]} to {pm_dates[-1]}  rows={len(pm_dates)}")
if pm_gaps:
    for g in pm_gaps:
        print(f"  GAP >3d: {g[0]} -> {g[1]}  ({g[2]} days)")
else:
    print("  No gaps >3 days")

# ── 4. Intersection ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("INTERSECTION SUMMARY")
print("=" * 70)

sora_dates = parse_mas_dates(RATE14)
sora_start, sora_end = sora_dates[0], sora_dates[-1]
pm_start,   pm_end   = pm_dates[0],   pm_dates[-1]

print(f"\n{'Source':<35} {'Start':>12} {'End':>12}")
print("-" * 62)
for ticker, (s, e, _) in sorted(tv_ranges.items()):
    print(f"  {ticker:<33} {str(s):>12} {str(e):>12}")
print(f"  {'SORA overnight':<33} {str(sora_start):>12} {str(sora_end):>12}")
print(f"  {'Parquet xgb_ready':<33} {str(pm_start):>12} {str(pm_end):>12}")

# All tickers excluding REITN (insufficient coverage)
ticker_excl = ['REITN']
all_starts = [v[0] for k, v in tv_ranges.items() if k not in ticker_excl]
all_ends   = [v[1] for k, v in tv_ranges.items() if k not in ticker_excl]
overall_start = max(all_starts + [sora_start, pm_start])
overall_end   = min(all_ends   + [sora_end,   pm_end])
print(f"\nIntersection (excl. REITN, excl. BWCU post-delist):")
print(f"  Latest start across all = {max(all_starts + [sora_start, pm_start])}")
print(f"  Earliest end across all = {min(all_ends + [sora_end, pm_end])}")
print(f"  => Common window: {overall_start} to {overall_end}")
print(f"  (BWCU ends 2023-08-28 — if included it would restrict end to 2023-08-28)")
