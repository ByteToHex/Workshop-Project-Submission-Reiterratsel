"""
Sanity-check all three REF CSVs used by Trades_CTF.sql and Trades_FPMM.sql.

CHECK 1 — Market_Token_Map vs parquet  (CTF)
  Verifies YES token is row-0 for every slug.  Wrong order → outcome_index
  0/1 swapped → YES/NO labels flipped in output.

CHECK 2 — 2VALIDATE metadata ambiguity  (CTF + FPMM)
  For every (parent_event_id, market_slug), confirms exactly one unique
  (group_item_title, signed_leg_move_bps) pair exists.  If > 1 survives
  DISTINCT, the join fan-outs: 2 tokens × 2 metadata rows = 4 rows per leg
  → trade counts and expected_bps are silently double-counted.

CHECK 3 — 2VALIDATE signed_leg_move_bps parseable  (CTF + FPMM)
  NULL or non-integer bps → CAST returns NULL → expected_bps silently wrong.

CHECK 4 — CTF join coverage: 2VALIDATE slugs ↔ Market_Token_Map  (CTF)
  Slugs in 2VALIDATE with no matching row in Market_Token_Map have no token
  → that leg is silently dropped from Trades_CTF.sql output.

CHECK 5 — FPMM join coverage: 2VALIDATE slugs ↔ 1EXTRACT  (FPMM)
  Slugs in 2VALIDATE with no matching slug in 1EXTRACT have no condition_id
  → that leg is silently dropped from Trades_FPMM.sql output.

CHECK 6 — 1EXTRACT condition_id present  (FPMM)
  A NULL/empty condition_id for a matched slug → pm_markets join fails
  → that FPMM leg silently vanishes.

Run from anywhere:
    python scripts/export/SQL/sanity_check_outcome_index.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

# ── Paths (resolved relative to this file) ───────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_REPO_ROOT    = _SCRIPT_DIR.parents[2]   # SQL/ → export/ → scripts/ → repo root
_REF          = _REPO_ROOT / "scripts/export/REF"

MARKET_TOKEN_MAP = _REF / "Market_Token_Map.csv"
VALIDATE_CSV     = _REF / "2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv"
EXTRACT_CSV      = _REF / "1EXTRACT_fed_parquet_events.csv"
MARKETS_GLOB     = (_REPO_ROOT / "data/parquet/markets/markets_*.parquet").as_posix()
# ─────────────────────────────────────────────────────────────────────────────

_SEP = "─" * 60


def _hdr(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


# ── CSV loaders ───────────────────────────────────────────────────────────────

def _load_map(path: Path) -> dict[str, list[str]]:
    """Market_Token_Map → {market_slug: [token_row0, token_row1]} in file order."""
    slug_tokens: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            slug_tokens[row["market_slug"]].append(row["token_id"].strip())
    return dict(slug_tokens)


def _parse_clob(raw: object) -> list[str]:
    """Parse clob_token_ids JSON string → ordered token_id list."""
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s in {"[]", "null", "None"}:
        return []
    try:
        data = json.loads(s)
    except Exception:
        return []
    tokens: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                tokens.append(item.strip())
            elif isinstance(item, dict):
                for k in ("token_id", "tokenId", "id"):
                    if k in item:
                        tokens.append(str(item[k]).strip())
                        break
    return [t for t in tokens if t]


def _csv_path(p: Path) -> str:
    """Forward-slash path string safe for DuckDB SQL."""
    return p.as_posix()


# ── Individual checks ─────────────────────────────────────────────────────────

def check1_yes_no_order(con: duckdb.DuckDBPyConnection, slug_map: dict[str, list[str]]) -> int:
    """CHECK 1: YES token is row-0 in Market_Token_Map for every slug."""
    _hdr("CHECK 1 — Market_Token_Map YES-token order vs parquet  [CTF]")

    slugs_2 = {s: v for s, v in slug_map.items() if len(v) == 2}

    try:
        rows = con.execute(f"""
            SELECT cast(slug AS VARCHAR) AS slug,
                   cast(clob_token_ids AS VARCHAR) AS clob
            FROM read_parquet('{MARKETS_GLOB}', union_by_name=true)
            WHERE clob_token_ids IS NOT NULL
              AND cast(clob_token_ids AS VARCHAR) NOT IN ('', '[]')
        """).fetchall()
    except Exception as e:
        print(f"  ERROR reading parquet: {e}")
        return 1

    parquet_yes: dict[str, str] = {}
    for slug, raw in rows:
        s = str(slug).strip()
        if s in slugs_2:
            toks = _parse_clob(raw)
            if toks:
                parquet_yes[s] = toks[0]

    print(f"  {len(parquet_yes)} of {len(slugs_2)} slugs matched in parquet")

    fail_count = 0
    for slug, toks in sorted(slugs_2.items()):
        if slug not in parquet_yes:
            continue
        if toks[0] != parquet_yes[slug]:
            fail_count += 1
            _fail(slug)
            print(f"         Map row-0 (outcome_index=0) : {toks[0]}")
            print(f"         Parquet clob[0]  (true YES)  : {parquet_yes[slug]}")

    if fail_count:
        print(f"\n  {fail_count} slug(s) have YES/NO swapped → outcome_index wrong in Trades_CTF.sql")
        return fail_count
    _ok(f"All {len(parquet_yes)} verified slugs have correct YES-first ordering")
    return 0


def check2_metadata_ambiguity(con: duckdb.DuckDBPyConnection) -> int:
    """CHECK 2: Each (parent_event_id, market_slug) has exactly 1 metadata combo in 2VALIDATE."""
    _hdr("CHECK 2 — 2VALIDATE metadata uniqueness per (parent_event_id, market_slug)  [CTF+FPMM]")
    print("  If > 1 unique (group_item_title, signed_leg_move_bps) survives DISTINCT,")
    print("  the join produces 4 rows per leg instead of 2 (silent double-counting).\n")

    rows = con.execute(f"""
        SELECT
            CAST(parent_event_id AS VARCHAR)  AS pid,
            market_slug,
            COUNT(DISTINCT group_item_title || '||' || CAST(signed_leg_move_bps AS VARCHAR)) AS n_combos,
            string_agg(DISTINCT group_item_title, ' | ')  AS titles,
            string_agg(DISTINCT CAST(signed_leg_move_bps AS VARCHAR), ' | ') AS bps_vals
        FROM read_csv('{_csv_path(VALIDATE_CSV)}')
        GROUP BY CAST(parent_event_id AS VARCHAR), market_slug
        HAVING COUNT(DISTINCT group_item_title || '||' || CAST(signed_leg_move_bps AS VARCHAR)) > 1
        ORDER BY pid, market_slug
    """).fetchall()

    if not rows:
        _ok("Every (parent_event_id, market_slug) has exactly 1 metadata combo — no fan-out risk")
        return 0

    for pid, slug, n, titles, bps_vals in rows:
        _fail(f"parent_event_id={pid}  slug={slug}")
        print(f"         {n} distinct combos — titles: {titles}  |  bps: {bps_vals}")
    print(f"\n  {len(rows)} slug(s) have ambiguous metadata → JOIN will produce duplicate rows")
    return len(rows)


def check3_bps_parseable(con: duckdb.DuckDBPyConnection) -> int:
    """CHECK 3: signed_leg_move_bps is non-NULL and castable to INTEGER in 2VALIDATE."""
    _hdr("CHECK 3 — 2VALIDATE signed_leg_move_bps parseable  [CTF+FPMM]")
    print("  NULL or non-integer bps → expected_bps calculation silently wrong.\n")

    rows = con.execute(f"""
        SELECT
            CAST(parent_event_id AS VARCHAR) AS pid,
            market_slug,
            CAST(signed_leg_move_bps AS VARCHAR) AS raw_bps
        FROM read_csv('{_csv_path(VALIDATE_CSV)}')
        WHERE signed_leg_move_bps IS NULL
           OR TRY_CAST(signed_leg_move_bps AS INTEGER) IS NULL
        ORDER BY pid, market_slug
    """).fetchall()

    if not rows:
        _ok("All signed_leg_move_bps values are non-NULL and integer-castable")
        return 0

    for pid, slug, raw in rows:
        _fail(f"parent_event_id={pid}  slug={slug}  raw_bps={raw!r}")
    print(f"\n  {len(rows)} row(s) have unparseable bps")
    return len(rows)


def check4_ctf_coverage(con: duckdb.DuckDBPyConnection, slug_map: dict[str, list[str]]) -> int:
    """CHECK 4: Every slug in 2VALIDATE exists in Market_Token_Map (CTF join coverage)."""
    _hdr("CHECK 4 — CTF join coverage: 2VALIDATE slugs ↔ Market_Token_Map  [CTF]")
    print("  Slugs in 2VALIDATE missing from Market_Token_Map have no token_id")
    print("  → that leg is silently dropped from Trades_CTF.sql output.\n")

    map_slugs = set(slug_map.keys())

    rows = con.execute(f"""
        SELECT DISTINCT
            CAST(parent_event_id AS VARCHAR) AS pid,
            market_slug
        FROM read_csv('{_csv_path(VALIDATE_CSV)}')
        ORDER BY pid, market_slug
    """).fetchall()

    missing = [(pid, slug) for pid, slug in rows if slug not in map_slugs]

    if not missing:
        _ok(f"All {len(rows)} 2VALIDATE slugs found in Market_Token_Map")
        return 0

    for pid, slug in missing:
        _fail(f"parent_event_id={pid}  slug={slug}  → no token_id → CTF leg dropped")
    print(f"\n  {len(missing)} slug(s) will be silently excluded from Trades_CTF.sql")
    return len(missing)


def check5_fpmm_coverage(con: duckdb.DuckDBPyConnection) -> int:
    """CHECK 5: Every slug in 2VALIDATE exists as slug in 1EXTRACT (FPMM join coverage)."""
    _hdr("CHECK 5 — FPMM join coverage: 2VALIDATE slugs ↔ 1EXTRACT  [FPMM]")
    print("  Slugs in 2VALIDATE missing from 1EXTRACT have no condition_id")
    print("  → that leg is silently dropped from Trades_FPMM.sql output.\n")

    rows = con.execute(f"""
        SELECT
            CAST(v.parent_event_id AS VARCHAR) AS pid,
            v.market_slug
        FROM (
            SELECT DISTINCT CAST(parent_event_id AS VARCHAR) AS parent_event_id, market_slug
            FROM read_csv('{_csv_path(VALIDATE_CSV)}')
        ) v
        LEFT JOIN (
            SELECT DISTINCT CAST(slug AS VARCHAR) AS slug
            FROM read_csv('{_csv_path(EXTRACT_CSV)}')
        ) ex ON ex.slug = v.market_slug
        WHERE ex.slug IS NULL
        ORDER BY pid, v.market_slug
    """).fetchall()

    if not rows:
        _ok("All 2VALIDATE slugs found in 1EXTRACT — no FPMM legs will be dropped")
        return 0

    for pid, slug in rows:
        _warn(f"parent_event_id={pid}  slug={slug}  → no condition_id → FPMM leg dropped")
    print(f"\n  NOTE: {len(rows)} slug(s) missing from 1EXTRACT.")
    print("  This is expected for CTF-only events (no FPMM market). Review manually.")
    return 0   # warn only — not a hard failure since CTF events won't be in 1EXTRACT


def check6_condition_id(con: duckdb.DuckDBPyConnection) -> int:
    """CHECK 6: condition_id non-NULL/empty in 1EXTRACT for slugs present in 2VALIDATE."""
    _hdr("CHECK 6 — 1EXTRACT condition_id present for matched slugs  [FPMM]")
    print("  NULL/empty condition_id → pm_markets join finds nothing")
    print("  → that FPMM leg silently vanishes from Trades_FPMM.sql output.\n")

    rows = con.execute(f"""
        SELECT
            CAST(ex.slug AS VARCHAR) AS slug,
            CAST(ex.condition_id AS VARCHAR) AS cid
        FROM read_csv('{_csv_path(EXTRACT_CSV)}') ex
        INNER JOIN (
            SELECT DISTINCT CAST(market_slug AS VARCHAR) AS market_slug
            FROM read_csv('{_csv_path(VALIDATE_CSV)}')
        ) v ON v.market_slug = CAST(ex.slug AS VARCHAR)
        WHERE ex.condition_id IS NULL
           OR TRIM(CAST(ex.condition_id AS VARCHAR)) = ''
        ORDER BY slug
    """).fetchall()

    if not rows:
        _ok("All matched 1EXTRACT slugs have a non-empty condition_id")
        return 0

    for slug, cid in rows:
        _fail(f"slug={slug}  condition_id={cid!r}")
    print(f"\n  {len(rows)} slug(s) have missing condition_id → FPMM legs silently dropped")
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    for p, label in [
        (MARKET_TOKEN_MAP, "Market_Token_Map.csv"),
        (VALIDATE_CSV,     "2VALIDATE CSV"),
        (EXTRACT_CSV,      "1EXTRACT CSV"),
    ]:
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 1

    print(f"REF folder : {_REF}")
    print(f"Parquet    : {MARKETS_GLOB}")

    slug_map = _load_map(MARKET_TOKEN_MAP)
    print(f"\nMarket_Token_Map: {len(slug_map)} slugs, "
          f"{sum(len(v) for v in slug_map.values())} token rows")

    con = duckdb.connect(":memory:")

    failures = 0
    failures += check1_yes_no_order(con, slug_map)
    failures += check2_metadata_ambiguity(con)
    failures += check3_bps_parseable(con)
    failures += check4_ctf_coverage(con, slug_map)
    check5_fpmm_coverage(con)        # warn-only, not counted in failures
    failures += check6_condition_id(con)

    _hdr("SUMMARY")
    if failures == 0:
        print("  All checks passed. REF CSVs are consistent with SQL assumptions.")
    else:
        print(f"  {failures} issue(s) found — review FAIL lines above before running SQLs.")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
