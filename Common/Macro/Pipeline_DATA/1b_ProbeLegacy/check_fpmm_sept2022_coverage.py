"""
check_fpmm_sept2022_coverage.py
────────────────────────────────────────────────────────────────────────────────
Verifies that pm_legacy_trades (FPMM parquet) contains actual trade records
for all legs of the September 2022 FOMC event (parent_event_id = 901490).

Existing sanity_check_outcome_index.py already confirms:
  - CHECK 5 PASS: all 4 slugs present in 1EXTRACT
  - CHECK 6 PASS: all condition_ids non-empty
This script goes one layer deeper: does the parquet data actually have trades?

Checks:
  A — Slugs for pid=901490 loaded from 2VALIDATE
  B — condition_ids resolved from 1EXTRACT (+ volume / dates from metadata)
  C — fpmm_address resolved from pm_markets parquet for each condition_id
  D — Trade records present in pm_legacy_trades for each fpmm_address
  E — Both outcome_index 0 (YES) and 1 (NO) covered per leg
  F — Trade date range vs market lifespan

Run:
    python scripts/export/SQL/check_fpmm_sept2022_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

# Windows PowerShell may default to cp1252; force UTF-8 for box-drawing chars.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parents[2]   # SQL/ → export/ → scripts/ → repo root
_REF         = _REPO_ROOT / "scripts/export/REF"

VALIDATE_CSV  = _REF / "2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv"
EXTRACT_CSV   = _REF / "1EXTRACT_fed_parquet_events.csv"

MARKETS_GLOB       = (_REPO_ROOT / "data/parquet/markets/markets_*.parquet").as_posix()
LEGACY_TRADES_GLOB = (_REPO_ROOT / "data/parquet/legacy_trades/trades_*.parquet").as_posix()
LEDGERS_GLOB        = (_REPO_ROOT / "data/parquet/ledgers/ledgers_*.parquet").as_posix()

TARGET_PID  = 901490
TARGET_NAME = "September 2022"

_SEP = "─" * 60


def _hdr(t: str) -> None:
    print(f"\n{_SEP}\n{t}\n{_SEP}")


def _ok(m: str)   -> None: print(f"  [PASS] {m}")
def _fail(m: str) -> None: print(f"  [FAIL] {m}")
def _info(m: str) -> None: print(f"  [INFO] {m}")


def _q(p: Path) -> str:
    """Forward-slash path string safe for DuckDB SQL."""
    return p.as_posix()


def main() -> int:
    for p, label in [(VALIDATE_CSV, "2VALIDATE"), (EXTRACT_CSV, "1EXTRACT")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            return 1

    con = duckdb.connect(":memory:")
    failures = 0

    # ── A. Load slugs for September 2022 ──────────────────────────────────────
    _hdr(f"CHECK A — 2VALIDATE slugs for {TARGET_NAME}  (pid={TARGET_PID})")

    slug_rows = con.execute(f"""
        SELECT DISTINCT
            CAST(parent_event_id    AS INTEGER)  AS pid,
            market_slug,
            TRIM(group_item_title)               AS title,
            CAST(signed_leg_move_bps AS VARCHAR) AS bps,
            fomc_decision_date
        FROM read_csv('{_q(VALIDATE_CSV)}')
        WHERE CAST(parent_event_id AS INTEGER) = {TARGET_PID}
        ORDER BY TRY_CAST(signed_leg_move_bps AS INTEGER)
    """).fetchall()

    if not slug_rows:
        _fail(f"No slugs found for parent_event_id={TARGET_PID}")
        return 1

    _ok(f"Found {len(slug_rows)} distinct slugs in 2VALIDATE:")
    for _, slug, title, bps, fomc in slug_rows:
        print(f"    bps={bps:>5}  {title:<45}  {slug}")

    target_slugs = [r[1] for r in slug_rows]

    # ── B. Resolve condition_id from 1EXTRACT ─────────────────────────────────
    _hdr("CHECK B — condition_id from 1EXTRACT  [FPMM required]")

    slug_in = ", ".join(f"'{s}'" for s in target_slugs)
    extract_rows = con.execute(f"""
        SELECT
            CAST(slug         AS VARCHAR) AS slug,
            CAST(condition_id AS VARCHAR) AS condition_id,
            CAST(volume       AS VARCHAR) AS volume,
            CAST(created_at   AS VARCHAR) AS created_at,
            CAST(end_date     AS VARCHAR) AS end_date
        FROM read_csv('{_q(EXTRACT_CSV)}')
        WHERE CAST(slug AS VARCHAR) IN ({slug_in})
    """).fetchall()

    slug_to_cid:  dict[str, str]  = {}
    slug_to_meta: dict[str, dict] = {}

    for slug, cid, vol, created, end in extract_rows:
        cid_clean = (cid or "").strip()
        if not cid_clean or cid_clean in ("None", "null", ""):
            _fail(f"slug={slug}  → condition_id missing/empty")
            failures += 1
        else:
            slug_to_cid[slug]  = cid_clean.lower()
            slug_to_meta[slug] = {"volume": vol, "created_at": created, "end_date": end}

    for s in target_slugs:
        if s not in slug_to_cid and s not in [r[0] for r in extract_rows]:
            _fail(f"slug={s}  → not found in 1EXTRACT at all")
            failures += 1

    if not any(True for _, slug, *_ in slug_rows if slug not in slug_to_cid):
        _ok(f"All {len(slug_to_cid)} slugs have valid condition_id")
        for slug, cid in slug_to_cid.items():
            meta = slug_to_meta[slug]
            print(f"    {slug}")
            print(f"      condition_id : {cid}")
            print(f"      volume_usdc  : {meta['volume']}")
            print(f"      created_at   : {meta['created_at']}")
            print(f"      end_date     : {meta['end_date']}")

    # ── C. Resolve fpmm_address from pm_markets parquet ───────────────────────
    _hdr("CHECK C — fpmm_address from pm_markets parquet")

    if not slug_to_cid:
        _fail("No condition_ids to look up — skipping C")
        return failures

    # condition_ids in 1EXTRACT are '0x...' hex strings; pm_markets may store
    # them with or without the '0x' prefix — match both via lower(trim(...)).
    cid_in = ", ".join(f"'{c}'" for c in slug_to_cid.values())
    # Also try without '0x' prefix
    cid_stripped = [c.lstrip("0x") for c in slug_to_cid.values()]
    cid_no0x_in = ", ".join(f"'{c}'" for c in cid_stripped)

    try:
        market_rows = con.execute(f"""
            SELECT
                lower(trim(cast(condition_id         AS VARCHAR))) AS cid,
                lower(trim(cast(market_maker_address AS VARCHAR))) AS fpmm_address
            FROM read_parquet('{MARKETS_GLOB}', union_by_name=true)
            WHERE lower(trim(cast(condition_id AS VARCHAR))) IN ({cid_in})
               OR lower(trim(cast(condition_id AS VARCHAR))) IN ({cid_no0x_in})
        """).fetchall()
    except Exception as e:
        _fail(f"Cannot read pm_markets parquet: {e}")
        _info("Check that MARKETS_GLOB path is correct and files exist")
        return failures + 1

    # Build lookup: handle both '0x...' and bare hex in pm_markets
    cid_to_fpmm: dict[str, str] = {}
    for cid_raw, fpmm in market_rows:
        cid_to_fpmm[cid_raw] = fpmm
        # also index without 0x in case 1EXTRACT has prefix but parquet doesn't
        cid_to_fpmm[cid_raw.lstrip("0x")] = fpmm

    slug_to_fpmm: dict[str, str] = {}
    for slug, cid in slug_to_cid.items():
        fpmm = cid_to_fpmm.get(cid) or cid_to_fpmm.get(cid.lstrip("0x"))
        if not fpmm:
            _fail(f"slug={slug}  condition_id={cid}  → no fpmm_address in pm_markets")
            failures += 1
        else:
            slug_to_fpmm[slug] = fpmm

    if len(slug_to_fpmm) == len(slug_to_cid):
        _ok(f"All {len(slug_to_fpmm)} slugs resolved to fpmm_address")
        for slug, fpmm in slug_to_fpmm.items():
            print(f"    {slug}")
            print(f"      fpmm_address : {fpmm}")

    # ── D + E + F. Trade coverage in pm_legacy_trades ─────────────────────────
    _hdr("CHECK D/E/F — trade records in pm_legacy_trades per leg")

    if not slug_to_fpmm:
        _fail("No fpmm_addresses resolved — cannot check trades")
        return failures + 1

    fpmm_in = ", ".join(f"'{a}'" for a in slug_to_fpmm.values())

    try:
        # Step 1: get trades per (fpmm, outcome_index) — ledger numbers only,
        # no join yet to keep this fast on large parquet sets.
        trade_rows = con.execute(f"""
            SELECT
                lower(trim(cast(fpmm_address AS VARCHAR)))  AS fpmm,
                CAST(outcome_index AS INTEGER)              AS outcome_idx,
                COUNT(*)                                    AS trade_count,
                SUM(CAST(amount AS DOUBLE) / 1e6)           AS volume_usdc,
                MIN(CAST(ledger_number AS BIGINT))           AS min_ledger,
                MAX(CAST(ledger_number AS BIGINT))           AS max_ledger
            FROM read_parquet('{LEGACY_TRADES_GLOB}', union_by_name=true)
            WHERE is_buy = TRUE
              AND lower(trim(cast(fpmm_address AS VARCHAR))) IN ({fpmm_in})
            GROUP BY fpmm, outcome_idx
            ORDER BY fpmm, outcome_idx
        """).fetchall()
    except Exception as e:
        _fail(f"Cannot read pm_legacy_trades parquet: {e}")
        _info("Check that LEGACY_TRADES_GLOB path is correct and files exist")
        return failures + 1

    # Step 2: look up ledger timestamps for min/max ledgers
    if trade_rows:
        all_ledgers = set()
        for _, _, _, _, mn, mx in trade_rows:
            all_ledgers.add(mn)
            all_ledgers.add(mx)
        ledger_in = ", ".join(str(b) for b in all_ledgers)
        try:
            ledger_ts = {
                int(bn): str(ts)
                for bn, ts in con.execute(f"""
                    SELECT
                        CAST(ledger_number AS BIGINT)              AS bn,
                        TRY_CAST(timestamp AS TIMESTAMP)          AS ts
                    FROM read_parquet('{LEDGERS_GLOB}', union_by_name=true)
                    WHERE CAST(ledger_number AS BIGINT) IN ({ledger_in})
                """).fetchall()
            }
        except Exception as e:
            _info(f"Could not resolve ledger timestamps: {e} — showing ledger numbers only")
            ledger_ts = {}
    else:
        ledger_ts = {}

    # Organise by fpmm
    fpmm_coverage: dict[str, list] = {}
    for fpmm, idx, cnt, vol, mn, mx in trade_rows:
        fpmm_coverage.setdefault(fpmm, []).append(
            (idx, cnt, vol or 0.0, mn, mx,
             ledger_ts.get(mn, f"ledger {mn}"),
             ledger_ts.get(mx, f"ledger {mx}"))
        )

    for _, slug, title, bps, _ in slug_rows:
        fpmm = slug_to_fpmm.get(slug)
        meta = slug_to_meta.get(slug, {})
        print(f"\n  Leg bps={bps}  —  {title}")
        print(f"  slug   : {slug}")
        if not fpmm:
            _fail("  fpmm_address not resolved (see CHECK C)")
            continue
        print(f"  fpmm   : {fpmm}")
        print(f"  market : created={meta.get('created_at','?')}  end={meta.get('end_date','?')}  vol_1EXTRACT={meta.get('volume','?')}")

        coverage = fpmm_coverage.get(fpmm, [])
        if not coverage:
            _fail("  NO TRADES found in pm_legacy_trades for this fpmm_address")
            failures += 1
            continue

        covered_idx = set()
        for idx, cnt, vol, mn, mx, ts_first, ts_last in sorted(coverage):
            side = "YES" if idx == 0 else "NO "
            _ok(f"  outcome_index={idx} ({side})  trades={cnt:>6}  "
                f"vol_usdc={vol:>12.2f}  "
                f"first={ts_first}  last={ts_last}")
            covered_idx.add(idx)

        if 0 not in covered_idx:
            _fail("  outcome_index=0 (YES) has NO buy trades")
            failures += 1
        if 1 not in covered_idx:
            _fail("  outcome_index=1 (NO) has NO buy trades — time series will have gaps")
            failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    _hdr("SUMMARY")
    # fpmm_coverage tuple: (idx, cnt, vol, min_ledger, max_ledger, ts_first, ts_last)
    total_trades = sum(r[1] for rows in fpmm_coverage.values() for r in rows)
    total_vol    = sum(r[2] for rows in fpmm_coverage.values() for r in rows)

    if failures == 0:
        _ok(f"{TARGET_NAME} FPMM is fully covered — "
            f"{total_trades:,} buy trades, {total_vol:,.2f} USDC volume "
            f"across all {len(slug_to_fpmm)} legs")
        _ok("Safe to use Trades_FPMM.sql with target_parent_event_id=901490")
        print()
        print("  NOTE — July 2022 (pid=901489): See recommendations in script docstring.")
    else:
        _fail(f"{failures} issue(s) found — review FAIL lines above")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
