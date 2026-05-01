"""
Probe overlap between Parquet CTF trades (data/parquet/trades) and legacy
FPMM trades (data/parquet/legacy_trades) at the market level.

A market is considered overlapping if:
  - CTF: at least one trade references an outcome token ID present in that market's
    `clob_token_ids` (from markets parquet), and
  - Legacy: at least one trade references that market's `fpmm_address`
    (`market_maker_address` in markets parquet).

This is a sister script to check_trade_inference_readiness.py (same data-dir
resolution and optional Step-01 CSV market filter).

Usage (repo root):
  python scripts/util/Step_03_ProbeTrades/probe_ctf_legacy_market_overlap.py
  python scripts/util/Step_03_ProbeTrades/probe_ctf_legacy_market_overlap.py --scan-mode sample \\
      --ctf-trade-sample-files 50 --legacy-trade-sample-files 50
  python scripts/util/Step_03_ProbeTrades/probe_ctf_legacy_market_overlap.py \\
      --input-csv scripts/util/out_Markets/1EXTRACT_fed_parquet_events.csv

---
- It only:
  - maps markets by `clob_token_ids` and `fpmm_address`
  - joins by token IDs / addresses (as strings)
  - counts rows and distinct IDs
  - tracks min/max ledger numbers
- The only literal `6` in the file that matched is `return 6` when no market parquet files are found. That is an **exit/status code**, not decimal precision.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import duckdb

_UTIL_DIR = Path(__file__).resolve().parents[1]
_MARKETS_OUT_DIR = _UTIL_DIR / "out_Markets"
_OUT_DIR = _UTIL_DIR / "out_Trades" / "LEGACY"
_STEP01_PREFIX = "1EXTRACT"
_DEFAULT_INPUT = _MARKETS_OUT_DIR / f"{_STEP01_PREFIX}_fed_parquet_events.csv"
_DEFAULT_OUT_PREFIX = "3PROBE_LEGACY"


def _glob_sql(path: Path) -> str:
    return str(path).replace("\\", "/")


def _csv_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    candidate = _OUT_DIR / p
    if candidate.exists():
        return candidate
    return _MARKETS_OUT_DIR / p


def _resolve_data_dirs(root: Path) -> tuple[Path, Path, Path, Path]:
    """Returns (markets_dir, ctf_trades_dir, legacy_trades_dir, ledgers_dir)."""
    m_dir = root / "markets"
    ctf_dir = root / "trades"
    leg_dir = root / "legacy_trades"
    b_dir = root / "ledgers"

    if m_dir.is_dir() and ctf_dir.is_dir() and leg_dir.is_dir():
        return m_dir, ctf_dir, leg_dir, b_dir

    alt_m = root / "parquet" / "markets"
    alt_ctf = root / "parquet" / "trades"
    alt_leg = root / "parquet" / "legacy_trades"
    alt_b = root / "parquet" / "ledgers"
    if alt_m.is_dir() and alt_ctf.is_dir() and alt_leg.is_dir():
        return alt_m, alt_ctf, alt_leg, alt_b

    return m_dir, ctf_dir, leg_dir, b_dir


def _read_market_slugs(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    slugs: list[str] = []
    for row in rows:
        s = (row.get("slug") or "").strip()
        if s:
            slugs.append(s)
            continue
        raw = (row.get("outcome_market_slugs") or "").strip()
        if raw:
            for part in raw.split(";"):
                p2 = part.strip()
                if p2:
                    slugs.append(p2)

    seen: set[str] = set()
    uniq: list[str] = []
    for slug in slugs:
        if slug in seen:
            continue
        seen.add(slug)
        uniq.append(slug)
    return uniq


def _first_files(d: Path, pattern: str, n: int | None) -> list[Path]:
    files = sorted(d.glob(pattern))
    if n is None or n <= 0:
        return files
    return files[:n]


def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _sql_values_str(vals: list[str]) -> str:
    return ", ".join(f"({_sql_quote(v)})" for v in vals)


def _write_csv(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})


def _run_fetchall_with_heartbeat(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    stage: str,
    heartbeat_sec: float,
) -> list[tuple]:
    if heartbeat_sec <= 0:
        return con.execute(sql).fetchall()

    stop = threading.Event()
    started = time.monotonic()

    def _beat() -> None:
        while not stop.wait(heartbeat_sec):
            elapsed = int(time.monotonic() - started)
            print(f"\r[{stage}] running... {elapsed}s", end="", file=sys.stderr, flush=True)

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    try:
        rows = con.execute(sql).fetchall()
    finally:
        stop.set()
        t.join(timeout=0.1)
        elapsed = int(time.monotonic() - started)
        print(f"\r[{stage}] done in {elapsed}s.{' ' * 24}", file=sys.stderr, flush=True)
    return rows


def _run_execute_with_heartbeat(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    stage: str,
    heartbeat_sec: float,
) -> None:
    if heartbeat_sec <= 0:
        con.execute(sql)
        return

    stop = threading.Event()
    started = time.monotonic()

    def _beat() -> None:
        while not stop.wait(heartbeat_sec):
            elapsed = int(time.monotonic() - started)
            print(f"\r[{stage}] running... {elapsed}s", end="", file=sys.stderr, flush=True)

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    try:
        con.execute(sql)
    finally:
        stop.set()
        t.join(timeout=0.1)
        elapsed = int(time.monotonic() - started)
        print(f"\r[{stage}] done in {elapsed}s.{' ' * 24}", file=sys.stderr, flush=True)


def _pick_first_present(cols: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Step-01 CSV (slug / outcome_market_slugs). If omitted, use default when present, else all markets.",
    )
    p.add_argument("--no-input-csv", action="store_true", help="Do not use any input CSV; scan all markets in parquet.")
    p.add_argument("--data-dir", type=Path, default=Path("data/parquet"))
    p.add_argument(
        "--scan-mode",
        choices=["full", "sample"],
        default="full",
        help="Read all parquet shards or a prefix (default: full).",
    )
    p.add_argument(
        "--ctf-trade-sample-files",
        type=int,
        default=250,
        metavar="N",
        help="When --scan-mode=sample: first N trades_*.parquet under CTF trades dir.",
    )
    p.add_argument(
        "--legacy-trade-sample-files",
        type=int,
        default=250,
        metavar="N",
        help="When --scan-mode=sample: first N trades_*.parquet under legacy_trades dir.",
    )
    p.add_argument(
        "--market-sample-files",
        type=int,
        default=None,
        metavar="N",
        help="Optional limit for markets_*.parquet file count.",
    )
    p.add_argument("--output-prefix", type=str, default=_DEFAULT_OUT_PREFIX)
    p.add_argument("--heartbeat-sec", type=float, default=5.0)
    p.add_argument("--duckdb-threads", type=int, default=0)
    args = p.parse_args(argv)

    input_csv: Path | None = None
    if args.no_input_csv:
        input_csv = None
    elif args.input_csv is not None:
        input_csv = _csv_path(args.input_csv)
        if not input_csv.is_file():
            print(f"Input CSV not found: {input_csv}", file=sys.stderr)
            return 1
    else:
        candidate = _csv_path(_DEFAULT_INPUT)
        input_csv = candidate if candidate.is_file() else None

    market_slugs_filter: list[str] | None = None
    if input_csv and input_csv.is_file():
        market_slugs_filter = _read_market_slugs(input_csv)
        if not market_slugs_filter:
            print("No market slugs in input CSV.", file=sys.stderr)
            return 2

    m_dir, ctf_dir, leg_dir, _b_dir = _resolve_data_dirs(args.data_dir)
    if not m_dir.is_dir():
        print(f"No markets directory: {m_dir.resolve()}", file=sys.stderr)
        return 3
    if not ctf_dir.is_dir():
        print(f"No CTF trades directory: {ctf_dir.resolve()}", file=sys.stderr)
        return 4
    if not leg_dir.is_dir():
        print(f"No legacy_trades directory: {leg_dir.resolve()}", file=sys.stderr)
        return 5

    ctf_limit = args.ctf_trade_sample_files if args.scan_mode == "sample" else None
    leg_limit = args.legacy_trade_sample_files if args.scan_mode == "sample" else None

    market_files = _first_files(m_dir, "markets_*.parquet", args.market_sample_files)
    ctf_files = _first_files(ctf_dir, "trades_*.parquet", ctf_limit)
    leg_files = _first_files(leg_dir, "trades_*.parquet", leg_limit)

    if not market_files:
        print(f"No markets_*.parquet under {m_dir.resolve()}", file=sys.stderr)
        return 6
    if not ctf_files:
        print(f"No trades_*.parquet under {ctf_dir.resolve()}", file=sys.stderr)
        return 7
    if not leg_files:
        print(f"No trades_*.parquet under {leg_dir.resolve()}", file=sys.stderr)
        return 8

    market_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in market_files)
    ctf_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in ctf_files)
    leg_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in leg_files)
    markets_from = f"read_parquet([{market_file_sql}], union_by_name=true)"
    ctf_from = f"read_parquet([{ctf_file_sql}], union_by_name=true)"
    leg_from = f"read_parquet([{leg_file_sql}], union_by_name=true)"

    con = duckdb.connect(database=":memory:")
    if args.duckdb_threads and args.duckdb_threads > 0:
        con.execute(f"PRAGMA threads={int(args.duckdb_threads)}")

    print("stage: detect schemas", file=sys.stderr)
    m_cols = {str(r[0]) for r in con.execute(f"DESCRIBE SELECT * FROM {markets_from} LIMIT 0").fetchall()}
    ctf_cols = {str(r[0]) for r in con.execute(f"DESCRIBE SELECT * FROM {ctf_from} LIMIT 0").fetchall()}
    leg_cols = {str(r[0]) for r in con.execute(f"DESCRIBE SELECT * FROM {leg_from} LIMIT 0").fetchall()}

    if "slug" not in m_cols:
        print("Markets parquet missing `slug`.", file=sys.stderr)
        return 9
    if "clob_token_ids" not in m_cols:
        print("Markets parquet missing `clob_token_ids`.", file=sys.stderr)
        return 10

    fpmm_col = _pick_first_present(m_cols, ["market_maker_address", "marketMakerAddress", "fpmm_address"])
    if not fpmm_col:
        print(
            "Markets parquet missing FPMM column (expected market_maker_address). Cannot map legacy trades.",
            file=sys.stderr,
        )
        return 11

    needed_ctf = {"maker_asset_id", "taker_asset_id"}
    if not needed_ctf.issubset(ctf_cols):
        print(f"CTF trades missing columns {needed_ctf}.", file=sys.stderr)
        return 12
    if "fpmm_address" not in leg_cols:
        print("Legacy trades parquet missing `fpmm_address`.", file=sys.stderr)
        return 13

    cond_col = _pick_first_present(m_cols, ["condition_id", "conditionId"])

    slug_filter_sql = ""
    if market_slugs_filter:
        slug_filter_sql = f"AND m.slug IN (SELECT slug FROM wanted_slugs)"
        wanted = _sql_values_str(market_slugs_filter)
        _run_execute_with_heartbeat(
            con,
            f"""
            CREATE TEMP TABLE wanted_slugs AS
            SELECT * FROM (VALUES {wanted}) AS t(slug)
            """,
            stage="build-wanted-slugs",
            heartbeat_sec=args.heartbeat_sec,
        )

    ledger_ctf = "ledger_number" if "ledger_number" in ctf_cols else None
    ledger_leg = "ledger_number" if "ledger_number" in leg_cols else None

    # Token map: slug, condition_id, token_id
    cond_select = f"m.{cond_col}" if cond_col else "CAST(NULL AS VARCHAR)"
    print("stage: build token_id -> market map", file=sys.stderr)
    _run_execute_with_heartbeat(
        con,
        f"""
        CREATE TEMP TABLE market_token_map AS
        SELECT
            m.slug AS market_slug,
            {cond_select} AS condition_id,
            unnest(from_json(m.clob_token_ids, '["VARCHAR"]')) AS token_id
        FROM {markets_from} m
        WHERE coalesce(m.clob_token_ids, '') NOT IN ('', '[]')
          AND left(trim(m.clob_token_ids), 1) = '['
          {slug_filter_sql}
        """,
        stage="build-market-token-map",
        heartbeat_sec=args.heartbeat_sec,
    )

    # FPMM map: slug, condition_id, fpmm_lower
    print("stage: build fpmm -> market map", file=sys.stderr)
    _run_execute_with_heartbeat(
        con,
        f"""
        CREATE TEMP TABLE fpmm_market_map AS
        SELECT DISTINCT
            m.slug AS market_slug,
            {cond_select} AS condition_id,
            lower(trim(cast(m.{fpmm_col} AS VARCHAR))) AS fpmm_lower
        FROM {markets_from} m
        WHERE m.{fpmm_col} IS NOT NULL
          AND trim(cast(m.{fpmm_col} AS VARCHAR)) <> ''
          {slug_filter_sql}
        """,
        stage="build-fpmm-market-map",
        heartbeat_sec=args.heartbeat_sec,
    )

    print("stage: aggregate CTF trades by mapped market", file=sys.stderr)
    if ledger_ctf:
        ctf_agg_sql = f"""
        CREATE TEMP TABLE ctf_by_market AS
        WITH hit AS (
            SELECT
                m.market_slug,
                m.condition_id,
                m.token_id AS token_id,
                tr.ledger_number AS ledger_number
            FROM {ctf_from} tr
            INNER JOIN market_token_map m
              ON cast(tr.maker_asset_id AS VARCHAR) = m.token_id
              OR cast(tr.taker_asset_id AS VARCHAR) = m.token_id
        )
        SELECT
            market_slug,
            max(condition_id) AS condition_id,
            count(*)::BIGINT AS n_ctf_trade_rows,
            count(DISTINCT token_id)::BIGINT AS n_distinct_matched_tokens,
            min(ledger_number) AS min_ledger,
            max(ledger_number) AS max_ledger
        FROM hit
        GROUP BY market_slug
        """
    else:
        ctf_agg_sql = f"""
        CREATE TEMP TABLE ctf_by_market AS
        WITH hit AS (
            SELECT
                m.market_slug,
                m.condition_id,
                m.token_id AS token_id
            FROM {ctf_from} tr
            INNER JOIN market_token_map m
              ON cast(tr.maker_asset_id AS VARCHAR) = m.token_id
              OR cast(tr.taker_asset_id AS VARCHAR) = m.token_id
        )
        SELECT
            market_slug,
            max(condition_id) AS condition_id,
            count(*)::BIGINT AS n_ctf_trade_rows,
            count(DISTINCT token_id)::BIGINT AS n_distinct_matched_tokens,
            CAST(NULL AS BIGINT) AS min_ledger,
            CAST(NULL AS BIGINT) AS max_ledger
        FROM hit
        GROUP BY market_slug
        """

    _run_execute_with_heartbeat(con, ctf_agg_sql, stage="ctf-agg", heartbeat_sec=args.heartbeat_sec)

    print("stage: aggregate legacy trades by mapped market", file=sys.stderr)
    if ledger_leg:
        leg_agg_sql = f"""
        CREATE TEMP TABLE legacy_by_market AS
        SELECT
            f.market_slug,
            max(f.condition_id) AS condition_id,
            count(*)::BIGINT AS n_legacy_trade_rows,
            count(DISTINCT lower(trim(cast(tr.fpmm_address AS VARCHAR))))::BIGINT AS n_distinct_fpmm,
            min(tr.ledger_number) AS min_ledger,
            max(tr.ledger_number) AS max_ledger
        FROM {leg_from} tr
        INNER JOIN fpmm_market_map f
          ON lower(trim(cast(tr.fpmm_address AS VARCHAR))) = f.fpmm_lower
        GROUP BY f.market_slug
        """
    else:
        leg_agg_sql = f"""
        CREATE TEMP TABLE legacy_by_market AS
        SELECT
            f.market_slug,
            max(f.condition_id) AS condition_id,
            count(*)::BIGINT AS n_legacy_trade_rows,
            count(DISTINCT lower(trim(cast(tr.fpmm_address AS VARCHAR))))::BIGINT AS n_distinct_fpmm,
            CAST(NULL AS BIGINT) AS min_ledger,
            CAST(NULL AS BIGINT) AS max_ledger
        FROM {leg_from} tr
        INNER JOIN fpmm_market_map f
          ON lower(trim(cast(tr.fpmm_address AS VARCHAR))) = f.fpmm_lower
        GROUP BY f.market_slug
        """

    _run_execute_with_heartbeat(con, leg_agg_sql, stage="legacy-agg", heartbeat_sec=args.heartbeat_sec)

    print("stage: compute overlap sets", file=sys.stderr)
    overlap_rows = _run_fetchall_with_heartbeat(
        con,
        """
        SELECT
            coalesce(c.market_slug, l.market_slug) AS market_slug,
            coalesce(c.condition_id, l.condition_id) AS condition_id,
            coalesce(c.n_ctf_trade_rows, 0) AS n_ctf_trade_rows,
            coalesce(c.n_distinct_matched_tokens, 0) AS n_ctf_distinct_tokens,
            coalesce(c.min_ledger, NULL) AS min_ledger_ctf,
            coalesce(c.max_ledger, NULL) AS max_ledger_ctf,
            coalesce(l.n_legacy_trade_rows, 0) AS n_legacy_trade_rows,
            coalesce(l.n_distinct_fpmm, 0) AS n_legacy_distinct_fpmm,
            coalesce(l.min_ledger, NULL) AS min_ledger_legacy,
            coalesce(l.max_ledger, NULL) AS max_ledger_legacy
        FROM ctf_by_market c
        INNER JOIN legacy_by_market l ON c.market_slug = l.market_slug
        ORDER BY (coalesce(c.n_ctf_trade_rows, 0) + coalesce(l.n_legacy_trade_rows, 0)) DESC, c.market_slug
        """,
        stage="overlap",
        heartbeat_sec=args.heartbeat_sec,
    )

    ctf_only = _run_fetchall_with_heartbeat(
        con,
        """
        SELECT c.market_slug, c.condition_id, c.n_ctf_trade_rows, c.n_distinct_matched_tokens,
               c.min_ledger AS min_ledger_ctf, c.max_ledger AS max_ledger_ctf
        FROM ctf_by_market c
        LEFT JOIN legacy_by_market l ON c.market_slug = l.market_slug
        WHERE l.market_slug IS NULL
        ORDER BY c.n_ctf_trade_rows DESC, c.market_slug
        """,
        stage="ctf-only",
        heartbeat_sec=args.heartbeat_sec,
    )

    leg_only = _run_fetchall_with_heartbeat(
        con,
        """
        SELECT l.market_slug, l.condition_id, l.n_legacy_trade_rows, l.n_distinct_fpmm,
               l.min_ledger AS min_ledger_legacy, l.max_ledger AS max_ledger_legacy
        FROM legacy_by_market l
        LEFT JOIN ctf_by_market c ON c.market_slug = l.market_slug
        WHERE c.market_slug IS NULL
        ORDER BY l.n_legacy_trade_rows DESC, l.market_slug
        """,
        stage="legacy-only",
        heartbeat_sec=args.heartbeat_sec,
    )

    glob_ctf = _run_fetchall_with_heartbeat(
        con,
        "SELECT count(*)::BIGINT FROM ctf_by_market",
        stage="count-ctf",
        heartbeat_sec=0,
    )[0][0]
    glob_leg = _run_fetchall_with_heartbeat(
        con,
        "SELECT count(*)::BIGINT FROM legacy_by_market",
        stage="count-leg",
        heartbeat_sec=0,
    )[0][0]

    out_overlap = _OUT_DIR / f"{args.output_prefix}_ctf_legacy_overlap_markets.csv"
    out_ctf_only = _OUT_DIR / f"{args.output_prefix}_ctf_only_markets.csv"
    out_leg_only = _OUT_DIR / f"{args.output_prefix}_legacy_only_markets.csv"
    out_summary = _OUT_DIR / f"{args.output_prefix}_overlap_summary.json"

    overlap_dicts: list[dict[str, Any]] = []
    for row in overlap_rows:
        overlap_dicts.append(
            {
                "market_slug": str(row[0]),
                "condition_id": row[1] or "",
                "n_ctf_trade_rows": int(row[2] or 0),
                "n_ctf_distinct_tokens": int(row[3] or 0),
                "min_ledger_ctf": row[4] if row[4] is not None else "",
                "max_ledger_ctf": row[5] if row[5] is not None else "",
                "n_legacy_trade_rows": int(row[6] or 0),
                "n_legacy_distinct_fpmm": int(row[7] or 0),
                "min_ledger_legacy": row[8] if row[8] is not None else "",
                "max_ledger_legacy": row[9] if row[9] is not None else "",
            }
        )

    ctf_only_dicts = [
        {
            "market_slug": str(row[0]),
            "condition_id": row[1] or "",
            "n_ctf_trade_rows": int(row[2] or 0),
            "n_ctf_distinct_tokens": int(row[3] or 0),
            "min_ledger_ctf": row[4] if row[4] is not None else "",
            "max_ledger_ctf": row[5] if row[5] is not None else "",
        }
        for row in ctf_only
    ]
    leg_only_dicts = [
        {
            "market_slug": str(row[0]),
            "condition_id": row[1] or "",
            "n_legacy_trade_rows": int(row[2] or 0),
            "n_legacy_distinct_fpmm": int(row[3] or 0),
            "min_ledger_legacy": row[4] if row[4] is not None else "",
            "max_ledger_legacy": row[5] if row[5] is not None else "",
        }
        for row in leg_only
    ]

    _write_csv(
        out_overlap,
        overlap_dicts,
        [
            "market_slug",
            "condition_id",
            "n_ctf_trade_rows",
            "n_ctf_distinct_tokens",
            "min_ledger_ctf",
            "max_ledger_ctf",
            "n_legacy_trade_rows",
            "n_legacy_distinct_fpmm",
            "min_ledger_legacy",
            "max_ledger_legacy",
        ],
    )
    _write_csv(
        out_ctf_only,
        ctf_only_dicts,
        [
            "market_slug",
            "condition_id",
            "n_ctf_trade_rows",
            "n_ctf_distinct_tokens",
            "min_ledger_ctf",
            "max_ledger_ctf",
        ],
    )
    _write_csv(
        out_leg_only,
        leg_only_dicts,
        [
            "market_slug",
            "condition_id",
            "n_legacy_trade_rows",
            "n_legacy_distinct_fpmm",
            "min_ledger_legacy",
            "max_ledger_legacy",
        ],
    )

    summary: dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "scan_mode": args.scan_mode,
        "input_csv": str(input_csv) if input_csv else None,
        "market_slugs_filter_count": len(market_slugs_filter) if market_slugs_filter else None,
        "markets_files_scanned": len(market_files),
        "ctf_trade_files_scanned": len(ctf_files),
        "legacy_trade_files_scanned": len(leg_files),
        "fpmm_column_used": fpmm_col,
        "condition_column_used": cond_col,
        "n_markets_with_ctf_trades": int(glob_ctf or 0),
        "n_markets_with_legacy_trades": int(glob_leg or 0),
        "n_markets_overlap_ctf_and_legacy": len(overlap_dicts),
        "n_markets_ctf_only": len(ctf_only_dicts),
        "n_markets_legacy_only": len(leg_only_dicts),
        "output_overlap_csv": str(out_overlap),
        "output_ctf_only_csv": str(out_ctf_only),
        "output_legacy_only_csv": str(out_leg_only),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"scan_mode: {args.scan_mode}")
    print(f"n_markets_with_ctf_trades:     {summary['n_markets_with_ctf_trades']}")
    print(f"n_markets_with_legacy_trades:  {summary['n_markets_with_legacy_trades']}")
    print(f"n_markets_overlap:             {summary['n_markets_overlap_ctf_and_legacy']}")
    print(f"n_markets_ctf_only:            {summary['n_markets_ctf_only']}")
    print(f"n_markets_legacy_only:         {summary['n_markets_legacy_only']}")
    print(f"wrote: {out_overlap}")
    print(f"wrote: {out_ctf_only}")
    print(f"wrote: {out_leg_only}")
    print(f"wrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
