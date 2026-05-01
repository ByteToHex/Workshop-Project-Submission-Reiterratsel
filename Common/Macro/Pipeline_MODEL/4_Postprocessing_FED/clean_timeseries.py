"""
SOURCE: ...\scripts\export\SQL\out
clean_timeseries.py
────────────────────────────────────────────────────────────────────────────────
Reads a raw timeseries_*.csv produced by build_timeseries.py and drops all
pre-trade rows (snapshots where the market hadn't opened yet and no price
signal exists).

Criterion: a row is "pre-trade" when `modal_prob` is NaN, meaning no trades
had occurred up to that snapshot's timestamp.  All pricing/probability columns
(expected_bps, p_no_change, modal_*, yes_prob_sum_pct) are also NaN on these
rows, so they carry zero signal for the XGBoost model.

Output: <original_stem>_clean.csv in the same folder.  Original is never
overwritten.

Usage (run directly or via VSCode play button):
    python clean_timeseries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────

# Absolute path to the raw file, or leave as None to auto-detect the single
# timeseries_*.csv in the shared Consolidated IO/SRC directory.
INPUT_FILE: Path | None = None

# ── Resolve input ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CONSOLIDATED_ROOT = SCRIPT_DIR.parents[1]
IO_SRC_DIR = CONSOLIDATED_ROOT / "IO" / "SRC"
FED_INPUT_DIR = IO_SRC_DIR / "CSV_FED" / "Input"
FED_OUTPUT_DIR = IO_SRC_DIR / "CSV_FED" / "Output"

if INPUT_FILE is None:
    candidates = sorted(FED_INPUT_DIR.glob("timeseries_*.csv"))
    # Exclude already-cleaned files
    candidates = [
        p for p in candidates
        if not p.stem.endswith("_clean") and not p.stem.endswith("_xgb_ready")
    ]
    if not candidates:
        print("[ERROR] No timeseries_*.csv found in", FED_INPUT_DIR)
        sys.exit(1)
    if len(candidates) > 1:
        print("[INFO] Multiple raw files found; using most recently modified:")
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for c in candidates:
            print(f"        {c.name}")
    src = candidates[0]
else:
    src = Path(INPUT_FILE)

if not src.exists():
    print(f"[ERROR] File not found: {src}")
    sys.exit(1)

FED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
dst = FED_OUTPUT_DIR / f"{src.stem}_clean{src.suffix}"
if dst == src:
    print("[ERROR] Output path equals input path — aborting to avoid overwrite.")
    sys.exit(1)

# ── Load ──────────────────────────────────────────────────────────────────────

print(f"Reading  : {src.name}  …", end="", flush=True)
df = pd.read_csv(src, low_memory=False)
print(f"  {len(df):,} rows")

# ── Drop pre-trade rows ───────────────────────────────────────────────────────
# modal_prob is None/NaN on every snapshot before the first trade for an event.

before = len(df)
df_clean = df[df["modal_prob"].notna()].copy()
dropped  = before - len(df_clean)

# ── Report ────────────────────────────────────────────────────────────────────

print(f"Dropped  : {dropped:,} pre-trade rows  ({dropped / before * 100:.1f}%)")
print(f"Retained : {len(df_clean):,} rows")
print()

# Per-event breakdown
event_summary = (
    df.groupby("parent_event_id")
    .agg(
        total_rows     = ("modal_prob", "count"),   # NaN not counted by count
        pre_trade_rows = ("modal_prob", lambda x: x.isna().sum()),
        fomc_date      = ("fomc_decision_date", "first"),
        market_type    = ("market_type", "first"),
    )
    .reset_index()
    .sort_values("parent_event_id")
)
event_summary["pct_dropped"] = (
    event_summary["pre_trade_rows"] / (event_summary["total_rows"] + event_summary["pre_trade_rows"]) * 100
).round(1)

print(f"{'PID':>8}  {'FOMC':>12}  {'Type':>4}  {'TotalOrig':>9}  "
      f"{'PreTrade':>8}  {'Kept':>6}  {'%Drop':>6}")
print("─" * 68)
for _, r in event_summary.iterrows():
    kept = r["total_rows"]
    orig = kept + r["pre_trade_rows"]
    print(f"{int(r['parent_event_id']):>8}  {r['fomc_date']:>12}  "
          f"{r['market_type']:>4}  {int(orig):>9}  "
          f"{int(r['pre_trade_rows']):>8}  {int(kept):>6}  {r['pct_dropped']:>5.1f}%")

# ── Write ─────────────────────────────────────────────────────────────────────

df_clean = df_clean.sort_values(["snapshot_ts", "parent_event_id"])
df_clean.to_csv(dst, index=False)
print(f"\nWritten  : {dst.name}  ({len(df_clean):,} rows)")
