"""
SOURCE: ...\scripts\export\SQL\out
build_timeseries.py
═══════════════════════════════════════════════════════════════════════════════
Batch orchestrator for Trades_CTF.sql / Trades_FPMM.sql.

For each FOMC event in the configured date window, this script:
  1. Loads all raw trades from parquet (one DuckDB query per event)
  2. Walks daily snapshots in pandas — no repeated DuckDB calls per snapshot
  3. Computes the SQL-layer features (expected_bps, p_no_change, n_brackets)
     and the Python-layer features (modal_prob, margin_over_second, etc.)
  4. Writes a daily time-series CSV to out/

Run from VSCode (play button) or terminal:
    python scripts/export/SQL/build_timeseries.py
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this section only
# ══════════════════════════════════════════════════════════════════════════════

# Daily snapshot window (inclusive both ends)
SERIES_START      = date(2022, 7, 27)   # First CTF event (March 2023 FOMC)
SERIES_END        = date(2026, 3, 18)   # Last day to include


'''
Range A — "Nov 2022 – Jan 2026" (27 events, FOMC Nov 2022 through Jan 2026)
SERIES_START = date(2022, 9,  1)   # safe margin before first Nov 2022 FPMM trade (~Sep 12)
SERIES_END   = date(2026, 1, 28)   # Jan 2026 FOMC decision

Range B — "Sep 2022 – Mar 2026" (29 events, FOMC Sep 2022 through Mar 2026)
SERIES_START = date(2022, 7, 27)   # first day of Sep 2022 FPMM market (confirmed)
SERIES_END   = date(2026, 3, 18)   # Mar 2026 FOMC decision

Test Short Series
SERIES_START      = date(2023, 3, 22)   # First CTF event (March 2023 FOMC)
SERIES_END        = date(2026, 4, 23)   # Last day to include
'''

# UTC hour used for every daily snapshot (0 = midnight, 17 = noon ET)
SNAPSHOT_HOUR_UTC = 0

# Parent event IDs to exclude entirely (structurally incompatible markets)
EXCLUDE_PIDS: set[int] = {
    901489,   # July 2022 — cumulative level markets, not discrete move markets
}

# Parent event IDs to force onto the FPMM path even when CTF tokens exist.
# Only use this for events where CTF tokens exist but ALL actual volume was FPMM.
FORCE_FPMM_PIDS: set[int] = {
    901490,   # September 2022 — CTF tokens present but zero real CTF volume
}

# Hybrid events: fetch BOTH CTF and FPMM trades, then switch per-snapshot.
# For snapshots before the first CTF trade → use FPMM (fills the gap).
# For snapshots on/after the first CTF trade → use CTF (better quality).
#
#   901498 — March 2023 FOMC: FPMM traded from 2023-02-02 (ledger 38,835,628);
#             CTF launched 2023-03-05 (ledger 40,003,188).  Routing as pure CTF
#             silently drops the 31-day FPMM-only window → 32-day timeseries gap.
#             Hybrid fills Feb 2 – Mar 4 with FPMM, then uses CTF from Mar 5.
HYBRID_PIDS: set[int] = {
    901498,   # March 2023 — FPMM-first hybrid
}

# Write per-leg intermediary CSVs (one file per event, all snapshots combined).
# ON  — essential for debugging a small date slice or a single event.
# OFF — skip during full-dataset export to avoid large file I/O overhead.
DEBUG_INTERMEDIARY = True

# Number of events processed concurrently.
# Each worker opens its own DuckDB in-memory connection.
# 2–4 is safe; higher values may contend on parquet file handles.
MAX_WORKERS = 3

# Heartbeat interval (seconds) — printed while workers are running.
HEARTBEAT_SEC = 10

# Debug: restrict processing to specific parent_event_ids.
# Empty set → all events in window are processed (normal / full-export mode).
# Non-empty → only listed PIDs are fetched (fast single-event debug).
#   Example: LIMIT_PIDS = {11696}  # September 2024 only
LIMIT_PIDS: set[int] = set()

# ══════════════════════════════════════════════════════════════════════════════

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parents[2]
_REF        = _REPO_ROOT / "scripts/export/REF"
_OUT        = _SCRIPT_DIR / "out"
_OUT_INTER  = _OUT / "intermediary"

VALIDATE_CSV  = _REF / "2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv"
EXTRACT_CSV   = _REF / "1EXTRACT_fed_parquet_events.csv"
TOKEN_MAP_CSV = _REF / "Market_Token_Map.csv"
PROBE_CSV     = _REF / "3PROBE_CTF_market_trade_coverage.csv"

MARKETS_GLOB = (_REPO_ROOT / "data/parquet/markets/markets_*.parquet").as_posix()
LEDGERS_GLOB  = (_REPO_ROOT / "data/parquet/ledgers/ledgers_*.parquet").as_posix()
TRADES_GLOB  = (_REPO_ROOT / "data/parquet/trades/trades_*.parquet").as_posix()
LEGACY_GLOB  = (_REPO_ROOT / "data/parquet/legacy_trades/trades_*.parquet").as_posix()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Leg:
    market_slug: str
    leg_label: str
    signed_leg_move_bps: Optional[int]   # None = "Other" catch-all
    token_id_yes: Optional[str] = None   # CTF only
    token_id_no:  Optional[str] = None   # CTF only


@dataclass
class EventConfig:
    pid: int
    fomc_decision_date: date
    market_type: str                     # "CTF" or "FPMM"
    legs: list[Leg] = field(default_factory=list)
    ledger_min: Optional[int] = None      # CTF only — from 3PROBE
    ledger_max: Optional[int] = None      # CTF only — from 3PROBE


# ── CSV loaders ───────────────────────────────────────────────────────────────

def _q(p: Path) -> str:
    return p.as_posix()


def _load_token_map() -> dict[str, tuple[str, str]]:
    """Market_Token_Map → {market_slug: (token_id_yes, token_id_no)} in file order."""
    slug_tokens: dict[str, list[str]] = defaultdict(list)
    with TOKEN_MAP_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            slug_tokens[row["market_slug"]].append(row["token_id"].strip())
    return {s: (toks[0], toks[1]) for s, toks in slug_tokens.items() if len(toks) == 2}


def load_event_configs() -> list[EventConfig]:
    """
    Build EventConfig list from 2VALIDATE + 3PROBE + Market_Token_Map.

    Routing:
      • Event is FPMM if ALL its slugs have n_trade_rows = 0 in 3PROBE
        OR the pid is in FORCE_FPMM_PIDS.
      • Otherwise CTF — ledger_min/max are the aggregate range across all slugs.
    """
    # ── 1. 2VALIDATE: group legs by parent_event_id ───────────────────────────
    pid_legs:       dict[int, list[dict]]  = defaultdict(list)
    pid_fomc:       dict[int, str]         = {}

    with VALIDATE_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = int(row["parent_event_id"])
            if pid in EXCLUDE_PIDS:
                continue
            pid_fomc.setdefault(pid, row["fomc_decision_date"])
            bps_raw = row["signed_leg_move_bps"].strip()
            bps = int(bps_raw) if bps_raw not in ("", "None", "null") else None
            pid_legs[pid].append({
                "market_slug": row["market_slug"].strip(),
                "leg_label":   row["group_item_title"].strip(),
                "bps":         bps,
            })

    # ── 2. 3PROBE: ledger ranges and CTF trade counts per slug ─────────────────
    slug_probe: dict[str, dict] = {}
    with PROBE_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            slug = row["market_slug"].strip()
            n    = int(row["n_trade_rows"] or 0)
            bmin = int(row["min_ledger_number"]) if row["min_ledger_number"].strip() else None
            bmax = int(row["max_ledger_number"]) if row["max_ledger_number"].strip() else None
            slug_probe[slug] = {"n_trade_rows": n, "ledger_min": bmin, "ledger_max": bmax}

    # ── 3. Market_Token_Map: token_ids per slug ───────────────────────────────
    token_map = _load_token_map()

    # ── 4. Assemble EventConfigs ──────────────────────────────────────────────
    events: list[EventConfig] = []

    for pid, raw_legs in pid_legs.items():
        fomc_str = pid_fomc.get(pid, "")
        try:
            fomc_date = date.fromisoformat(fomc_str)
        except (ValueError, TypeError):
            print(f"  [WARN] pid={pid} — cannot parse fomc_date={fomc_str!r}, skipping")
            continue

        # Deduplicate legs (2VALIDATE has 2 rows per slug: YES + NO token)
        seen_slugs: set[str] = set()
        unique_legs: list[dict] = []
        for leg in raw_legs:
            if leg["market_slug"] not in seen_slugs:
                seen_slugs.add(leg["market_slug"])
                unique_legs.append(leg)

        # Determine routing
        any_ctf_trades = any(
            slug_probe.get(lg["market_slug"], {}).get("n_trade_rows", 0) > 0
            for lg in unique_legs
        )
        is_fpmm = (not any_ctf_trades) or (pid in FORCE_FPMM_PIDS)
        market_type = "FPMM" if is_fpmm else "CTF"

        # Aggregate ledger range (CTF only)
        ledger_min, ledger_max = None, None
        if not is_fpmm:
            mins = [slug_probe[lg["market_slug"]]["ledger_min"]
                    for lg in unique_legs
                    if slug_probe.get(lg["market_slug"], {}).get("ledger_min") is not None]
            maxs = [slug_probe[lg["market_slug"]]["ledger_max"]
                    for lg in unique_legs
                    if slug_probe.get(lg["market_slug"], {}).get("ledger_max") is not None]
            ledger_min = min(mins) if mins else None
            ledger_max = max(maxs) if maxs else None

        # Build Leg objects with token_ids (CTF) or bare (FPMM)
        legs: list[Leg] = []
        for lg in unique_legs:
            slug = lg["market_slug"]
            toks = token_map.get(slug)
            legs.append(Leg(
                market_slug         = slug,
                leg_label           = lg["leg_label"],
                signed_leg_move_bps = lg["bps"],
                token_id_yes        = toks[0] if toks else None,
                token_id_no         = toks[1] if toks else None,
            ))

        events.append(EventConfig(
            pid                = pid,
            fomc_decision_date = fomc_date,
            market_type        = market_type,
            legs               = legs,
            ledger_min          = ledger_min,
            ledger_max          = ledger_max,
        ))

    events.sort(key=lambda e: e.fomc_decision_date)
    return events


# ── DuckDB trade fetchers (one call per event, run in executor thread) ────────

def _new_con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


def fetch_ctf_trades(event: EventConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch all CTF trades for this event within its ledger range.
    Returns (trades_df, leg_meta_df).
    trades_df  : outcome_token_id, implied_price, usdc_notional, ledger_number, log_index, ledger_ts
    leg_meta_df: leg_key (=outcome_token_id), leg_label, signed_leg_move_bps, outcome_index
    """
    if event.ledger_min is None or event.ledger_max is None:
        return pd.DataFrame(), pd.DataFrame()

    # Build token list (YES + NO for every leg that has tokens)
    all_tokens = []
    for leg in event.legs:
        if leg.token_id_yes:
            all_tokens.append(leg.token_id_yes)
        if leg.token_id_no:
            all_tokens.append(leg.token_id_no)
    if not all_tokens:
        return pd.DataFrame(), pd.DataFrame()

    token_in = ", ".join(f"'{t}'" for t in all_tokens)

    sql = f"""
        SELECT
            CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                 ELSE t.maker_asset_id END                                   AS outcome_token_id,
            CASE WHEN t.maker_asset_id = '0'
                 THEN TRY_CAST(t.maker_amount AS DOUBLE) / NULLIF(TRY_CAST(t.taker_amount AS DOUBLE), 0)
                 ELSE TRY_CAST(t.taker_amount AS DOUBLE) / NULLIF(TRY_CAST(t.maker_amount AS DOUBLE), 0)
            END                                                              AS implied_price,
            CASE WHEN t.maker_asset_id = '0'
                 THEN TRY_CAST(t.maker_amount AS DOUBLE) / 1e6
                 ELSE TRY_CAST(t.taker_amount AS DOUBLE) / 1e6
            END                                                              AS usdc_notional,
            CAST(t.ledger_number AS BIGINT)                                   AS ledger_number,
            CAST(t.log_index    AS BIGINT)                                   AS log_index,
            TRY_CAST(b.timestamp AS TIMESTAMP)                               AS ledger_ts
        FROM read_parquet('{TRADES_GLOB}', union_by_name=true) t
        JOIN read_parquet('{LEDGERS_GLOB}', union_by_name=true) b
          ON CAST(t.ledger_number AS BIGINT) = CAST(b.ledger_number AS BIGINT)
        WHERE CAST(b.ledger_number AS BIGINT) BETWEEN {event.ledger_min} AND {event.ledger_max}
          AND (t.maker_asset_id IN ({token_in}) OR t.taker_asset_id IN ({token_in}))
        ORDER BY outcome_token_id, ledger_number, log_index
    """
    con = _new_con()
    try:
        trades_df = con.execute(sql).df()
    finally:
        con.close()

    # Build unified leg_meta (one row per token_id, i.e. per outcome)
    meta_rows = []
    for leg in event.legs:
        if leg.token_id_yes:
            meta_rows.append({
                "leg_key":             leg.token_id_yes,
                "leg_label":           leg.leg_label,
                "signed_leg_move_bps": leg.signed_leg_move_bps,
                "outcome_index":       0,
            })
        if leg.token_id_no:
            meta_rows.append({
                "leg_key":             leg.token_id_no,
                "leg_label":           leg.leg_label,
                "signed_leg_move_bps": leg.signed_leg_move_bps,
                "outcome_index":       1,
            })

    leg_meta_df = pd.DataFrame(meta_rows)
    # Normalise token column name
    trades_df = trades_df.rename(columns={"outcome_token_id": "leg_key"})
    return trades_df, leg_meta_df


def fetch_fpmm_trades(event: EventConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch all FPMM buy trades for this event.
    Resolves: slug → condition_id (1EXTRACT) → fpmm_address (pm_markets parquet).
    Returns (trades_df, leg_meta_df).
    trades_df  : leg_key (=fpmm_address), outcome_index, implied_price, usdc_notional, ledger_number, log_index, ledger_ts
    leg_meta_df: leg_key, leg_label, signed_leg_move_bps, outcome_index
    """
    slugs = [lg.market_slug for lg in event.legs]
    if not slugs:
        return pd.DataFrame(), pd.DataFrame()

    slug_in = ", ".join(f"'{s}'" for s in slugs)
    con = _new_con()
    try:
        # Step 1 — condition_ids from 1EXTRACT
        cids_df = con.execute(f"""
            SELECT CAST(slug AS VARCHAR) AS slug,
                   lower(trim(CAST(condition_id AS VARCHAR))) AS condition_id
            FROM read_csv('{_q(EXTRACT_CSV)}')
            WHERE CAST(slug AS VARCHAR) IN ({slug_in})
        """).df()

        if cids_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        cid_in = ", ".join(f"'{c}'" for c in cids_df["condition_id"].dropna().unique())

        # Step 2 — fpmm_address from pm_markets
        fpmm_df = con.execute(f"""
            SELECT lower(trim(CAST(condition_id          AS VARCHAR))) AS condition_id,
                   lower(trim(CAST(market_maker_address  AS VARCHAR))) AS fpmm_address
            FROM read_parquet('{MARKETS_GLOB}', union_by_name=true)
            WHERE lower(trim(CAST(condition_id AS VARCHAR))) IN ({cid_in})
        """).df()

        if fpmm_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        fpmm_in = ", ".join(f"'{a}'" for a in fpmm_df["fpmm_address"].unique())

        # Step 3 — all buy trades (no snapshot filter; pandas handles that)
        trades_df = con.execute(f"""
            SELECT lower(trim(CAST(lt.fpmm_address AS VARCHAR)))        AS leg_key,
                   CAST(lt.outcome_index AS INTEGER)                    AS outcome_index,
                   CAST(lt.amount AS DOUBLE) / NULLIF(CAST(lt.outcome_tokens AS DOUBLE), 0)
                                                                        AS implied_price,
                   CAST(lt.amount AS DOUBLE) / 1e6                      AS usdc_notional,
                   CAST(lt.ledger_number AS BIGINT)                      AS ledger_number,
                   CAST(lt.log_index    AS BIGINT)                      AS log_index,
                   TRY_CAST(b.timestamp AS TIMESTAMP)                   AS ledger_ts
            FROM read_parquet('{LEGACY_GLOB}', union_by_name=true) lt
            JOIN read_parquet('{LEDGERS_GLOB}', union_by_name=true) b
              ON CAST(lt.ledger_number AS BIGINT) = CAST(b.ledger_number AS BIGINT)
            WHERE lt.is_buy = TRUE
              AND lower(trim(CAST(lt.fpmm_address AS VARCHAR))) IN ({fpmm_in})
            ORDER BY leg_key, outcome_index, ledger_number, log_index
        """).df()
    finally:
        con.close()

    # Build slug→fpmm_address map
    slug_to_cid  = dict(zip(cids_df["slug"],         cids_df["condition_id"]))
    cid_to_fpmm  = dict(zip(fpmm_df["condition_id"], fpmm_df["fpmm_address"]))

    # Build leg_meta (expand each leg to YES + NO rows)
    meta_rows = []
    for leg in event.legs:
        fpmm = cid_to_fpmm.get(slug_to_cid.get(leg.market_slug, ""), "")
        if not fpmm:
            continue
        for oi in (0, 1):
            meta_rows.append({
                "leg_key":             fpmm,
                "leg_label":           leg.leg_label,
                "signed_leg_move_bps": leg.signed_leg_move_bps,
                "outcome_index":       oi,
            })

    leg_meta_df = pd.DataFrame(meta_rows)
    return trades_df, leg_meta_df


# ── Snapshot computation (pandas, no DuckDB) ──────────────────────────────────

def _compute_snapshot(
    all_trades:  pd.DataFrame,
    leg_meta:    pd.DataFrame,
    snapshot_ts: datetime,
    pid:         int,
) -> tuple[pd.DataFrame, dict]:
    """
    Given ALL trades for an event (pre-loaded), compute the leg-level and
    event-level metrics as of `snapshot_ts`.

    Mirrors the logic of Trades_CTF.sql / Trades_FPMM.sql exactly:
      - last_price  = last traded price per (leg_key, outcome_index) ≤ snapshot
      - trade_count / volume_usdc = cumulative totals ≤ snapshot
      - expected_bps, p_no_change, n_brackets, yes_prob_sum on YES legs
    """
    # ── Filter to ≤ snapshot ─────────────────────────────────────────────────
    filt = all_trades[all_trades["ledger_ts"] <= pd.Timestamp(snapshot_ts)].copy()

    # CTF: each token_id (leg_key) is unique per outcome → group on leg_key only.
    # FPMM: same fpmm_address appears for both YES and NO → also group on outcome_index.
    has_oi_in_trades = "outcome_index" in all_trades.columns
    group_cols  = ["leg_key", "outcome_index"] if has_oi_in_trades else ["leg_key"]
    sort_cols   = group_cols + ["ledger_number", "log_index"]
    # leg_meta always has outcome_index; for CTF it is used as the merge source.
    merge_cols  = group_cols   # for CTF = ["leg_key"], outcome_index comes from leg_meta

    # ── Last price per group ──────────────────────────────────────────────────
    if filt.empty:
        last_price = pd.DataFrame(columns=group_cols + ["implied_price", "ledger_number", "ledger_ts"])
        stats      = pd.DataFrame(columns=group_cols + ["trade_count", "volume_usdc"])
    else:
        last_price = (
            filt.sort_values(sort_cols)
                .groupby(group_cols, sort=False)
                .last()
                .reset_index()
                [group_cols + ["implied_price", "ledger_number", "ledger_ts"]]
        )
        stats = (
            filt.groupby(group_cols, sort=False)
                .agg(trade_count=("implied_price", "count"),
                     volume_usdc=("usdc_notional", "sum"))
                .reset_index()
        )

    # ── Merge into per-leg result ─────────────────────────────────────────────
    # leg_meta is the left spine (has outcome_index for both CTF and FPMM).
    result = leg_meta.merge(last_price, on=merge_cols, how="left")
    result = result.merge(stats,       on=merge_cols, how="left")
    result["snapshot_ts"]      = snapshot_ts
    result["parent_event_id"]  = pid
    result["outcome_name"]     = result["outcome_index"].map({0: "YES", 1: "NO"})
    result["implied_price"]    = pd.to_numeric(result["implied_price"], errors="coerce")
    result["implied_prob"]     = result["implied_price"].round(6)
    result["implied_prob_pct"] = (result["implied_price"] * 100).round(2)
    result = result.rename(columns={"ledger_number": "last_trade_ledger",
                                    "ledger_ts":      "last_trade_at"})

    # ── Event-level features (YES side only) ──────────────────────────────────
    yes = result[result["outcome_index"] == 0].copy()

    # yes_prob_sum (sanity check — all YES legs including Other).
    # min_count=1 so all-NaN (pre-trade) returns NaN instead of 0.
    yes_prob_sum = yes["implied_price"].sum(min_count=1)

    # Exclude Other legs (bps IS NULL) for feature computation
    yes_k = yes[yes["signed_leg_move_bps"].notna()].copy()

    n_brackets = len(yes_k)

    # expected_bps: NULL × price = NaN → nansum skips.
    # Guard on has_prices so pre-trade snapshots (all NaN) return None not 0.0.
    has_prices   = yes_k["implied_price"].notna().any()
    expected_bps = (yes_k["signed_leg_move_bps"].astype(float) * yes_k["implied_price"]).sum()
    expected_bps = round(float(expected_bps), 4) if (n_brackets > 0 and has_prices) else None

    # p_no_change
    hold = yes_k[yes_k["signed_leg_move_bps"] == 0]["implied_price"]
    p_no_change = round(float(hold.iloc[0]), 6) if len(hold) > 0 else None

    # modal_prob, modal_outcome_bps
    valid = yes_k.dropna(subset=["implied_price"])
    if valid.empty:
        modal_prob = modal_outcome_bps = margin_over_second = None
        modal_direction = None
    else:
        sorted_p = valid.nlargest(2, "implied_price")
        modal_prob        = round(float(sorted_p.iloc[0]["implied_price"]), 6)
        modal_outcome_bps = int(sorted_p.iloc[0]["signed_leg_move_bps"])
        margin_over_second = (
            round(float(sorted_p.iloc[0]["implied_price"] - sorted_p.iloc[1]["implied_price"]), 6)
            if len(sorted_p) >= 2 else None
        )
        modal_direction = ("hike" if modal_outcome_bps > 0
                           else "cut" if modal_outcome_bps < 0 else "hold")

    distance_from_uniform = (
        round(modal_prob - 1.0 / n_brackets, 6)
        if (modal_prob is not None and n_brackets > 0) else None
    )

    # Write SQL-layer event features back onto the legs DataFrame (for intermediary)
    result["yes_prob_sum_pct"]      = round(yes_prob_sum * 100, 2)
    result["yes_prob_deviation_pct"]= round((yes_prob_sum - 1.0) * 100, 2)
    result["expected_bps"]          = expected_bps
    result["p_no_change"]           = p_no_change
    result["n_brackets"]            = n_brackets

    event_row = {
        "snapshot_ts":            snapshot_ts.isoformat(),
        "parent_event_id":        pid,
        "yes_prob_sum_pct":       round(yes_prob_sum * 100, 2),
        "yes_prob_deviation_pct": round((yes_prob_sum - 1.0) * 100, 2),
        "expected_bps":           expected_bps,
        "p_no_change":            p_no_change,
        "n_brackets":             n_brackets,
        "modal_prob":             modal_prob,
        "modal_outcome_bps":      modal_outcome_bps,
        "modal_direction":        modal_direction,
        "margin_over_second":     margin_over_second,
        "distance_from_uniform":  distance_from_uniform,
    }

    return result, event_row


# ── Per-event processor (runs in thread via executor) ────────────────────────

def _process_event_sync(event: EventConfig) -> tuple[list[dict], pd.DataFrame]:
    """
    Fetches all trades for one event, then walks daily snapshots.
    Returns (event_rows, legs_df) where:
      - event_rows : one dict per snapshot → written to timeseries CSV
      - legs_df    : all legs × snapshots  → written to intermediary CSV (if enabled)
    """
    t0        = time.time()
    is_hybrid = event.pid in HYBRID_PIDS
    tag       = f"pid={event.pid} [{'HYBRID' if is_hybrid else event.market_type}] fomc={event.fomc_decision_date}"

    print(f"  → {tag}  fetching trades …")

    try:
        if is_hybrid:
            ctf_trades,  ctf_meta  = fetch_ctf_trades(event)
            fpmm_trades, fpmm_meta = fetch_fpmm_trades(event)
            if ctf_trades.empty and fpmm_trades.empty:
                print(f"  [SKIP]  {tag}  no trade data (CTF and FPMM both empty)")
                return [], pd.DataFrame()
        elif event.market_type == "CTF":
            trades_df, leg_meta = fetch_ctf_trades(event)
        else:
            trades_df, leg_meta = fetch_fpmm_trades(event)
    except Exception as exc:
        print(f"  [ERROR] {tag}  trade fetch failed: {exc}")
        return [], pd.DataFrame()

    if not is_hybrid and (trades_df.empty or leg_meta.empty):
        print(f"  [SKIP]  {tag}  no trade data or no leg metadata")
        return [], pd.DataFrame()

    # Normalise ledger_ts to tz-naive UTC datetimes for comparisons
    def _fix_ts(df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and "ledger_ts" in df.columns:
            df["ledger_ts"] = pd.to_datetime(df["ledger_ts"], utc=True).dt.tz_localize(None)
        return df

    if is_hybrid:
        ctf_trades  = _fix_ts(ctf_trades)
        fpmm_trades = _fix_ts(fpmm_trades)
        # Earliest CTF trade timestamp — switch from FPMM to CTF on/after this moment
        ctf_first_ts: Optional[pd.Timestamp] = (
            ctf_trades["ledger_ts"].min() if not ctf_trades.empty else None
        )
        print(f"  [HYBRID] FPMM trades: {len(fpmm_trades):,}  CTF trades: {len(ctf_trades):,}  "
              f"CTF starts: {ctf_first_ts}")
    else:
        trades_df["ledger_ts"] = pd.to_datetime(trades_df["ledger_ts"], utc=True).dt.tz_localize(None)

    # ── Generate daily snapshots within [SERIES_START, min(SERIES_END, fomc_date)] ──
    snap_end   = min(SERIES_END, event.fomc_decision_date)
    snap_start = SERIES_START
    if snap_start > snap_end:
        print(f"  [SKIP]  {tag}  fomc_date before SERIES_START")
        return [], pd.DataFrame()

    fomc_date = event.fomc_decision_date

    all_legs:   list[pd.DataFrame] = []
    event_rows: list[dict]         = []

    d = snap_start
    while d <= snap_end:
        snap_ts = datetime(d.year, d.month, d.day, SNAPSHOT_HOUR_UTC)

        if is_hybrid:
            # Use CTF when its first trade falls at or before this snapshot
            use_ctf = (
                ctf_first_ts is not None
                and pd.Timestamp(snap_ts) >= ctf_first_ts
                and not ctf_trades.empty
                and not ctf_meta.empty
            )
            if use_ctf:
                legs_df, ev_row = _compute_snapshot(ctf_trades, ctf_meta, snap_ts, event.pid)
                row_market_type = "CTF"
            else:
                legs_df, ev_row = _compute_snapshot(fpmm_trades, fpmm_meta, snap_ts, event.pid)
                row_market_type = "FPMM"
        else:
            legs_df, ev_row = _compute_snapshot(trades_df, leg_meta, snap_ts, event.pid)
            row_market_type = event.market_type

        days_to_fomc = (fomc_date - d).days
        ev_row["fomc_decision_date"] = fomc_date.isoformat()
        ev_row["days_to_next_fomc"]  = days_to_fomc
        ev_row["market_type"]        = row_market_type

        event_rows.append(ev_row)
        if DEBUG_INTERMEDIARY:
            legs_df["days_to_next_fomc"] = days_to_fomc
            all_legs.append(legs_df)

        d += timedelta(days=1)

    non_empty_legs = [df for df in all_legs if not df.empty and not df.isna().all(axis=None)]
    combined_legs  = pd.concat(non_empty_legs, ignore_index=True) if non_empty_legs else pd.DataFrame()

    elapsed = time.time() - t0
    print(f"  ✓ {tag}  {len(event_rows)} snapshots in {elapsed:.1f}s")

    return event_rows, combined_legs


# ── Output ────────────────────────────────────────────────────────────────────

_TIMESERIES_COLS = [
    "snapshot_ts", "parent_event_id", "fomc_decision_date", "market_type",
    "days_to_next_fomc",
    "expected_bps", "p_no_change", "n_brackets",
    "modal_prob", "modal_outcome_bps", "modal_direction",
    "margin_over_second", "distance_from_uniform",
    "yes_prob_sum_pct", "yes_prob_deviation_pct",
]

_INTERMEDIARY_COLS = [
    "snapshot_ts", "parent_event_id", "leg_label", "signed_leg_move_bps",
    "outcome_index", "outcome_name", "last_trade_ledger", "last_trade_at",
    "implied_prob", "implied_prob_pct", "trade_count", "volume_usdc",
    "yes_prob_sum_pct", "yes_prob_deviation_pct",
    "expected_bps", "p_no_change", "n_brackets", "days_to_next_fomc",
]


def write_outputs(all_event_rows: list[dict], legs_by_pid: dict[int, pd.DataFrame]) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    _OUT_INTER.mkdir(parents=True, exist_ok=True)

    # ── Timeseries CSV ────────────────────────────────────────────────────────
    ts_path = _OUT / f"timeseries_{SERIES_START}_{SERIES_END}.csv"
    ts_df   = pd.DataFrame(all_event_rows)
    ts_df   = ts_df.sort_values(["parent_event_id", "snapshot_ts"])
    # Reorder columns (extras appended at end)
    present = [c for c in _TIMESERIES_COLS if c in ts_df.columns]
    extras  = [c for c in ts_df.columns if c not in _TIMESERIES_COLS]
    ts_df   = ts_df[present + extras]
    ts_df.to_csv(ts_path, index=False)
    print(f"\n  [OUT] timeseries → {ts_path}  ({len(ts_df):,} rows)")

    # ── Intermediary CSVs ─────────────────────────────────────────────────────
    # Build pid → fomc_date lookup from the timeseries rows already collected
    pid_to_fomc: dict[int, str] = {}
    for row in all_event_rows:
        pid_to_fomc.setdefault(int(row["parent_event_id"]), row["fomc_decision_date"])

    if DEBUG_INTERMEDIARY:
        for pid, legs_df in legs_by_pid.items():
            if legs_df.empty:
                continue
            # YYMM prefix from fomc_decision_date (e.g. 2024-09-18 → "2409")
            fomc_str = pid_to_fomc.get(pid, "")
            try:
                from datetime import date as _date
                fd = _date.fromisoformat(fomc_str)
                yymm = f"{fd.year % 100:02d}{fd.month:02d}"
            except (ValueError, TypeError):
                yymm = "0000"
            p = _OUT_INTER / f"{yymm}_{pid}_legs.csv"
            present = [c for c in _INTERMEDIARY_COLS if c in legs_df.columns]
            extras  = [c for c in legs_df.columns  if c not in _INTERMEDIARY_COLS]
            legs_df[present + extras].sort_values(
                ["snapshot_ts", "signed_leg_move_bps", "outcome_index"],
                na_position="last",
            ).to_csv(p, index=False)
            print(f"  [OUT] intermediary → {p}  ({len(legs_df):,} rows)")


# ── Async runner ──────────────────────────────────────────────────────────────

async def heartbeat(stop_event: asyncio.Event) -> None:
    """Prints a pulse every HEARTBEAT_SEC while workers run."""
    t0 = time.time()
    while not stop_event.is_set():
        await asyncio.sleep(HEARTBEAT_SEC)
        if not stop_event.is_set():
            print(f"  … running  {time.time() - t0:.0f}s elapsed")


async def main() -> None:
    print("═" * 70)
    print("  build_timeseries.py")
    print(f"  Range        : {SERIES_START} → {SERIES_END}  (UTC {SNAPSHOT_HOUR_UTC:02d}:00)")
    print(f"  Intermediary : {'ON' if DEBUG_INTERMEDIARY else 'OFF'}")
    print(f"  Workers      : {MAX_WORKERS}")
    print("═" * 70)

    # ── Load event metadata ───────────────────────────────────────────────────
    print("\nLoading event configs …")
    events = load_event_configs()
    # Keep events that are still unresolved as of SERIES_START.
    # Snapshot truncation to SERIES_END is handled per-event inside _process_event_sync.
    events = [e for e in events if e.fomc_decision_date >= SERIES_START]
    if LIMIT_PIDS:
        events = [e for e in events if e.pid in LIMIT_PIDS]
        print(f"  [DEBUG] LIMIT_PIDS={LIMIT_PIDS} → {len(events)} event(s)")
    print(f"  {len(events)} events in window  "
          f"({sum(1 for e in events if e.market_type=='CTF')} CTF, "
          f"{sum(1 for e in events if e.market_type=='FPMM')} FPMM)")

    if not events:
        print("  Nothing to process. Adjust SERIES_START / SERIES_END.")
        return

    # ── Process events concurrently ───────────────────────────────────────────
    print("\nProcessing events …\n")
    loop        = asyncio.get_running_loop()
    stop_hb     = asyncio.Event()
    hb_task     = asyncio.create_task(heartbeat(stop_hb))

    all_event_rows: list[dict]               = []
    legs_by_pid:    dict[int, pd.DataFrame]  = {}

    async def _run_one(ev: EventConfig) -> tuple[EventConfig, list[dict], pd.DataFrame]:
        ev_rows, legs_df = await loop.run_in_executor(executor, _process_event_sync, ev)
        return ev, ev_rows, legs_df

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [asyncio.ensure_future(_run_one(ev)) for ev in events]
        for coro in asyncio.as_completed(tasks):
            ev, ev_rows, legs_df = await coro
            if ev_rows:
                all_event_rows.extend(ev_rows)
                legs_by_pid[ev.pid] = legs_df

    stop_hb.set()
    await hb_task

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\nWriting outputs …")
    write_outputs(all_event_rows, legs_by_pid)

    print(f"\n  Done — {len(all_event_rows):,} total event-snapshot rows written.")
    print("═" * 70)


if __name__ == "__main__":
    asyncio.run(main())
