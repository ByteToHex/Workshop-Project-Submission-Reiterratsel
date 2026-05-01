"""
SOURCE: ...\scripts\export\SQL\out
prep_xgb.py
────────────────────────────────────────────────────────────────────────────
Reads the *_clean.csv produced by clean_timeseries.py and applies model-
preparation steps required before XGBoost training:

  Step 1 — Sparse-trading flag  (CTF only)
            Instead of dropping sparse CTF rows (which causes multi-week
            timeline gaps), add a boolean flag `is_sparse_ctf`.
            Rows where yes_prob_sum_pct is outside [50, 150] for CTF markets
            are flagged True.  FPMM rows are always False — AMM prices
            legitimately sum > 100%.
            Use this flag at training time to filter or weight rows.

  Step 1.5 — FPMM calibration
            FPMM AMM prices sum to ~110–130% (not 100%), so raw
            expected_bps / p_no_change / margin_over_second are inflated
            relative to CTF values.  Divide FPMM rows by
            (yes_prob_sum_pct / 100) to put both market types on the same
            probability-normalised scale.
            CTF rows are unchanged (dividing by 1.0).
            This also removes the FPMM→CTF transition discontinuity.

  Step 2 — Nearest-event selection per calendar date
            Multiple FOMC markets are active simultaneously.
            Keep only the row with the smallest days_to_next_fomc per date
            → "always looking at the nearest upcoming FOMC".

  Step 3 — p_no_change imputation for no-hold-leg events
            Sep / Nov / Dec 2022 FPMM markets had no 0-bps (hold) bracket
            so p_no_change is structurally absent.  Impute with 0.0:
            the hold probability is effectively zero (not unknown).
            PIDs: 901490 (Sep 2022), 901491 (Nov 2022), 901492 (Dec 2022).

  Step 4 — Drop March 2026 FOMC event
            pid=67284 (fomc_decision_date=2026-03-18) has a 40-day frozen
            tail of identical feature values and no computable forward
            return label.  Dropped entirely.

  Step 5 — Column selection
            Drop sanity-check / diagnostic columns not used as features.
            Retain the Model R* feature set plus join keys for the next
            pipeline stage (SGX calendar join, SORA join, label attach).

Output: <original_stem>_xgb_ready.csv in the same folder.
        Does NOT overwrite the _clean.csv.

Steps NOT performed here (require external data):
  - Join to SGX trading calendar (keep only SGX trading days; carry
    Saturday/Sunday Parquet values forward to the following Monday)
  - Join to SORA daily series (sora_level, sora_90d_change)
  - Attach target label (iEdge_fwd_21d_return for Model R*;
    abnormal_fwd_21d_return per ticker for Model A)

Usage:
    python prep_xgb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────

# Leave None to auto-detect the single *_clean.csv in the shared
# Consolidated IO/SRC folder.
INPUT_FILE: Path | None = None

# PIDs that structurally have no hold (0-bps) leg.
# p_no_change is NaN by design; impute 0.0 (not unknown — just confirmed zero).
NO_HOLD_PIDS: set[int] = {901490, 901491, 901492}

# Columns retained in the output (join keys + Model R* features + reference).
# Downstream SORA columns (sora_level, sora_90d_change) and the label
# (iEdge_fwd_21d_return) are NOT present yet — they are joined later.
KEEP_COLS = [
    # ── Join keys / reference ──────────────────────────────────────────────
    "snapshot_ts",          # Calendar date — primary join key to SGX series
    "parent_event_id",      # Which FOMC event this row represents (reference)
    "fomc_decision_date",   # FOMC date (reference)
    "market_type",          # CTF or FPMM (reference; not a Model R* feature)
    # ── Data-quality flag (from Step 1) ───────────────────────────────────
    "is_sparse_ctf",        # True when CTF yes_prob_sum_pct outside [50,150]
                            # Use at train time to filter or down-weight these rows
    # ── Model R* features (FPMM-calibrated in Step 1.5) ──────────────────
    "days_to_next_fomc",    # Feature 1: calendar countdown
    "expected_bps",         # Feature 2: prob-weighted expected rate change
    "p_no_change",          # Feature 3: implied prob of hold bracket
    "margin_over_second",   # Feature 4: modal-bracket conviction
    # ── Kept for Model A / context (not Model R* features) ────────────────
    "n_brackets",           # Market structure context
    "modal_prob",           # Useful for Model A / inspection
    "modal_outcome_bps",    # Useful for Model A / inspection
    "modal_direction",      # Useful for Model A / inspection
    "distance_from_uniform",# Useful for Model A / inspection
]

# fomc_decision_date values for events to drop entirely (no label computable).
DROP_FOMC_DATES: list[str] = [
    "2026-03-18",   # pid=67284 — frozen tail; no label computable
    "2026-04-29",   # pid=75478 — also frozen; no label computable
    "2026-06-17",   # pid=101772 — also frozen; no label computable
]

# Hard training cutoff: drop all rows with snapshot_ts AFTER this date.
# Reason: every event with fomc_decision_date > 2026-01-28 either has no
# label or has frozen prices cascading into the dataset tail.
# pid=45883 (Jan 28 2026 FOMC) is the last event with active trading data.
# Set to None to disable.
SERIES_END_TRAINING: str | None = "2026-01-28"

# Sparse-trading filter bounds (CTF only).
PROB_SUM_LOW  = 50.0
PROB_SUM_HIGH = 150.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def _report(label: str, before: int, after: int) -> None:
    dropped = before - after
    pct     = dropped / before * 100 if before else 0
    print(f"  {label:<42} {before:>6} → {after:>6}  (dropped {dropped:>5}, {pct:.1f}%)")


# ── Resolve input ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CONSOLIDATED_ROOT = SCRIPT_DIR.parents[1]
IO_SRC_DIR = CONSOLIDATED_ROOT / "IO" / "SRC"
FED_OUTPUT_DIR = IO_SRC_DIR / "CSV_FED" / "Output"

if INPUT_FILE is None:
    candidates = sorted(FED_OUTPUT_DIR.glob("*_clean.csv"))
    if not candidates:
        print("[ERROR] No *_clean.csv found in", FED_OUTPUT_DIR)
        sys.exit(1)
    if len(candidates) > 1:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"[INFO] Multiple *_clean.csv found; using most recent: {candidates[0].name}")
    src = candidates[0]
else:
    src = Path(INPUT_FILE)

if not src.exists():
    print(f"[ERROR] File not found: {src}")
    sys.exit(1)

dst = FED_OUTPUT_DIR / f"{src.stem.replace('_clean', '')}_xgb_ready{src.suffix}"
if dst == src:
    print("[ERROR] Output path equals input — aborting.")
    sys.exit(1)

# ── Load ──────────────────────────────────────────────────────────────────────

_banner("Loading")
df = pd.read_csv(src, low_memory=False)
df["snapshot_ts"]     = pd.to_datetime(df["snapshot_ts"])
df["fomc_decision_date"] = pd.to_datetime(df["fomc_decision_date"])
print(f"  Source : {src.name}")
print(f"  Rows   : {len(df):,}")
print(f"  Events : {df['parent_event_id'].nunique()} unique PIDs")
print(f"  Range  : {df['snapshot_ts'].min().date()} → {df['snapshot_ts'].max().date()}")

# ── Training cutoff ───────────────────────────────────────────────────────────
# Drop all rows after SERIES_END_TRAINING to avoid cascading frozen tails from
# future FOMC markets that displace resolved events at the end of the dataset.

if SERIES_END_TRAINING is not None:
    cutoff = pd.Timestamp(SERIES_END_TRAINING)
    before = len(df)
    df = df[df["snapshot_ts"] <= cutoff].copy()
    print(f"  Training cutoff  : snapshot_ts ≤ {cutoff.date()}")
    print(f"  Rows after cutoff: {len(df):,}  (dropped {before - len(df):,})")

# ── Step 1 — Sparse-trading flag (CTF only) ───────────────────────────────────
# Dropping sparse rows caused multi-week gaps (e.g. 26 days Dec 2023–Jan 2024)
# because after the nearest-event selection there was no fallback row for those
# dates.  We now flag instead of drop, preserving timeline continuity.
# Use `is_sparse_ctf` at model-training time to filter or weight rows.

_banner("Step 1 — Sparse-trading flag (CTF only, no rows dropped)")

is_ctf    = df["market_type"] == "CTF"
is_sparse = ~df["yes_prob_sum_pct"].between(PROB_SUM_LOW, PROB_SUM_HIGH)
df["is_sparse_ctf"] = (is_ctf & is_sparse)

n_flagged = int(df["is_sparse_ctf"].sum())
print(f"  Flagged  : {n_flagged} CTF rows with yes_prob_sum_pct outside "
      f"[{PROB_SUM_LOW}, {PROB_SUM_HIGH}]  →  is_sparse_ctf = True")
print(f"  Retained : all {len(df):,} rows  (no rows dropped at this step)")
print(f"  FPMM rows: always False — AMM prices legitimately sum outside range")
print(f"\n  TIP: to replicate the old strict filter at train time:")
print(f"       df = df[~df['is_sparse_ctf']]")

# ── Step 1.5 — FPMM calibration (normalise by probability sum) ────────────────
# FPMM AMM prices do not sum to 100% — typically 110–130%.  The raw
# expected_bps / p_no_change / margin_over_second are therefore on a larger
# scale than CTF equivalents, which creates:
#   (a) an unconditional level difference between FPMM and CTF rows, and
#   (b) a sharp discontinuity at the FPMM→CTF handoff (~Feb 2023).
# Fix: divide FPMM rows by (yes_prob_sum_pct / 100).  CTF rows are unchanged
# (we explicitly use 1.0 as the divisor, not their yes_prob_sum_pct, because
# CTF sparse rows can have a low sum due to missing prices — dividing by that
# would inflate CTF values rather than shrink FPMM ones).

_banner("Step 1.5 — FPMM calibration (probability-sum normalisation)")

fpmm_mask = df["market_type"] == "FPMM"
# Divisor: actual yes_prob_sum_pct/100 for FPMM, 1.0 for CTF
divisor = (df["yes_prob_sum_pct"] / 100.0).where(fpmm_mask, other=1.0)

cal_cols = ["expected_bps", "p_no_change", "margin_over_second"]
for col in cal_cols:
    if col in df.columns:
        df[col] = (df[col] / divisor).round(6)

print(f"  Calibrated columns : {cal_cols}")
print(f"  FPMM rows adjusted : {fpmm_mask.sum()}")
print(f"  CTF rows unchanged : {(~fpmm_mask).sum()}")
# Show before/after scale for the FPMM era
fpmm_ex = df.loc[fpmm_mask, ["yes_prob_sum_pct", "expected_bps"]].describe().round(3)
print(f"\n  Post-calibration FPMM expected_bps stats:\n{fpmm_ex.to_string()}")

# ── Step 2 — Drop unlabelable FOMC events ─────────────────────────────────────

_banner("Step 2 — Drop unlabelable FOMC events")
before = len(df)

drop_dates = pd.to_datetime(DROP_FOMC_DATES)
drop_mask  = df["fomc_decision_date"].isin(drop_dates)
dropped_pids = df.loc[drop_mask, "parent_event_id"].unique().tolist()
df = df[~drop_mask].copy()

_report("After dropping unlabelable events", before, len(df))
print(f"  Dropped PIDs : {dropped_pids}")
print(f"  Dropped dates: {[str(d.date()) for d in drop_dates]}")

# ── Step 3 — Nearest-event selection per calendar date ────────────────────────

_banner("Step 3 — Nearest-event selection per date")
before = len(df)

# Per calendar date, keep only the row with the smallest days_to_next_fomc.
# Tie-break on parent_event_id for determinism.
df = (
    df.sort_values(["snapshot_ts", "days_to_next_fomc", "parent_event_id"])
    .groupby("snapshot_ts", sort=False)
    .first()
    .reset_index()
)

_report("After nearest-event selection", before, len(df))
print(f"  → {df['snapshot_ts'].nunique():,} unique calendar dates retained")

# Show which events were selected and for how many days
event_counts = (
    df.groupby("parent_event_id")["snapshot_ts"].count()
    .reset_index()
    .rename(columns={"snapshot_ts": "days_as_nearest"})
    .sort_values("days_as_nearest")
)
fomc_map = df.groupby("parent_event_id")["fomc_decision_date"].first()
event_counts["fomc_date"] = event_counts["parent_event_id"].map(fomc_map).dt.date
print(f"\n  {'PID':>8}  {'FOMC':>12}  {'Days as nearest':>16}")
print(f"  {'─'*8}  {'─'*12}  {'─'*16}")
for _, r in event_counts.iterrows():
    print(f"  {int(r['parent_event_id']):>8}  {str(r['fomc_date']):>12}  {int(r['days_as_nearest']):>16}")

# ── Step 4 — p_no_change imputation ──────────────────────────────────────────

_banner("Step 4 — p_no_change imputation for no-hold-leg events")
mask_no_hold = df["parent_event_id"].isin(NO_HOLD_PIDS)
null_before  = df.loc[mask_no_hold, "p_no_change"].isna().sum()
df.loc[mask_no_hold, "p_no_change"] = df.loc[mask_no_hold, "p_no_change"].fillna(0.0)
null_after   = df.loc[mask_no_hold, "p_no_change"].isna().sum()
print(f"  PIDs treated : {sorted(NO_HOLD_PIDS)}")
print(f"  Nulls filled : {null_before} → {null_after}")
print(f"  Rows affected: {mask_no_hold.sum()}")

# ── Step 5 — Column selection ─────────────────────────────────────────────────

_banner("Step 5 — Column selection")
present     = [c for c in KEEP_COLS if c in df.columns]
dropped_cols= [c for c in df.columns if c not in KEEP_COLS]
df          = df[present].copy()
print(f"  Columns kept   ({len(present):>2}): {present}")
print(f"  Columns dropped({len(dropped_cols):>2}): {dropped_cols}")

# ── Final sort ────────────────────────────────────────────────────────────────

df = df.sort_values("snapshot_ts").reset_index(drop=True)

# ── Summary ───────────────────────────────────────────────────────────────────

_banner("Summary")
print(f"  Output rows  : {len(df):,}")
print(f"  Date range   : {df['snapshot_ts'].min().date()} → {df['snapshot_ts'].max().date()}")
print(f"  NaN counts (feature columns):")
for col in ["expected_bps", "p_no_change", "margin_over_second", "days_to_next_fomc"]:
    n = df[col].isna().sum()
    print(f"    {col:<30} {n:>4} NaN  ({n/len(df)*100:.1f}%)")

print(f"\n  ⚠  Remaining pipeline steps (external data required):")
print(f"     1. Join to SGX trading calendar → keep SGX trading days only,")
print(f"        carry weekend Parquet values forward to the following Monday")
print(f"     2. Join SORA daily series → sora_level, sora_90d_change")
print(f"     3. Attach target label    → iEdge_fwd_21d_return (Model R*)")
print(f"                                 abnormal_fwd_21d_return per ticker (Model A)")

# ── Write ─────────────────────────────────────────────────────────────────────

df.to_csv(dst, index=False)
print(f"\n  Written : {dst.name}  ({len(df):,} rows)")
