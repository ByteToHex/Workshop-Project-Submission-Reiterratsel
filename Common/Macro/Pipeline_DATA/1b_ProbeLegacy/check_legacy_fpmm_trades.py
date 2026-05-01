"""
Query and summarize Parquet legacy FPMM trade data from local parquet shards.

Default behavior is intentionally runnable from VS Code / Cursor without arguments:

  python scripts/util/Step_05_Legacy/check_legacy_fpmm_trades.py

That default run scans all legacy trade shards and writes a compact workspace-level
summary under `scripts/util/out_Trades/LEGACY/`.

For targeted investigation, pass one or more FPMM addresses:

  python scripts/util/Step_05_Legacy/check_legacy_fpmm_trades.py \
      --fpmm-address 0x8b9805a2f595b6705e74f7310829f2d299d21522

  python scripts/util/Step_05_Legacy/check_legacy_fpmm_trades.py \
      --fpmm-address 0x8b9805a2f595b6705e74f7310829f2d299d21522 \
      --ledger-min 30000000 --ledger-max 34000000

What it does:
  - reads `data/parquet/legacy_trades/trades_*.parquet`
  - reads `scripts/util/out_Markets/1EXTRACT_fed_parquet_events.csv` by default
    to build explicit market / bracket-leg legacy coverage
  - ignores AppleDouble `._*` files by using the explicit `trades_*.parquet` pattern
  - aggregates overall trade coverage
  - optionally filters to one or more exact `fpmm_address` values
  - writes summary JSON plus CSVs for:
      * selected FPMM summary
      * per-outcome summary
      * top traders
      * sample trade rows
      * Step-01 market-level legacy coverage
      * Step-01 event-level / bracket coverage

Notes:
  - `amount`, `fee_amount`, and `outcome_tokens` are stored as strings in parquet, so
    this script uses `try_cast(... AS DOUBLE)` for approximate numeric summaries.
  - Collateral metadata is read from `data/parquet/fpmm_collateral_lookup.json`
    when available.
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
_OUT_DIR = _UTIL_DIR / "out_Trades" / "LEGACY"
_MARKETS_OUT_DIR = _UTIL_DIR / "out_Markets"
_DEFAULT_PREFIX = "5CHECK_LEGACY"
_DEFAULT_INPUT_CSV = _MARKETS_OUT_DIR / "1EXTRACT_fed_parquet_events.csv"


def _glob_sql(path: Path) -> str:
    return str(path).replace("\\", "/")


def _resolve_data_dirs(root: Path) -> tuple[Path, Path]:
    legacy_dir = root / "legacy_trades"
    collateral_json = root / "fpmm_collateral_lookup.json"
    if legacy_dir.is_dir():
        return legacy_dir, collateral_json

    alt_legacy = root / "parquet" / "legacy_trades"
    alt_collateral = root / "parquet" / "fpmm_collateral_lookup.json"
    if alt_legacy.is_dir():
        return alt_legacy, alt_collateral

    return legacy_dir, collateral_json


def _first_files(d: Path, pattern: str, n: int | None) -> list[Path]:
    files = sorted(d.glob(pattern))
    if n is None or n <= 0:
        return files
    return files[:n]


def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _sql_values_str(vals: list[str]) -> str:
    return ", ".join(f"({_sql_quote(v)})" for v in vals)


def _normalize_addresses(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        addr = str(value or "").strip().lower()
        if not addr:
            continue
        if not addr.startswith("0x"):
            addr = "0x" + addr
        if addr in seen:
            continue
        seen.add(addr)
        out.append(addr)
    return out


def _csv_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    candidate = _OUT_DIR / p
    if candidate.exists():
        return candidate
    return _MARKETS_OUT_DIR / p


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/parquet"))
    p.add_argument(
        "--input-csv",
        type=Path,
        default=_DEFAULT_INPUT_CSV,
        help="Step-01 CSV (absolute or relative to scripts/util/out_Markets).",
    )
    p.add_argument(
        "--no-input-csv",
        action="store_true",
        help="Skip market-level coverage outputs based on the Step-01 export CSV.",
    )
    p.add_argument(
        "--scan-mode",
        choices=["full", "sample"],
        default="full",
        help="Read all legacy trade shards or only a prefix subset.",
    )
    p.add_argument(
        "--legacy-trade-sample-files",
        type=int,
        default=250,
        metavar="N",
        help="When --scan-mode=sample: first N trades_*.parquet files.",
    )
    p.add_argument(
        "--fpmm-address",
        action="append",
        default=[],
        help="Exact FPMM address to inspect. May be passed multiple times.",
    )
    p.add_argument(
        "--fpmm-csv",
        type=Path,
        default=None,
        help="Optional CSV containing `fpmm_address` values to inspect.",
    )
    p.add_argument("--ledger-min", type=int, default=None, help="Optional minimum ledger_number filter.")
    p.add_argument("--ledger-max", type=int, default=None, help="Optional maximum ledger_number filter.")
    p.add_argument("--trader", action="append", default=[], help="Optional trader address filter.")
    p.add_argument("--top-n", type=int, default=50, help="Number of top grouped rows to write for broad summaries.")
    p.add_argument("--sample-rows", type=int, default=100, help="Number of sample trade rows to write.")
    p.add_argument("--output-prefix", type=str, default=_DEFAULT_PREFIX)
    p.add_argument("--heartbeat-sec", type=float, default=5.0)
    p.add_argument("--duckdb-threads", type=int, default=0)
    args = p.parse_args(argv)

    legacy_dir, collateral_json = _resolve_data_dirs(args.data_dir)
    input_csv: Path | None = None
    if args.no_input_csv:
        input_csv = None
    else:
        input_csv = _csv_path(args.input_csv)
        if not input_csv.is_file():
            print(f"Input CSV not found: {input_csv}", file=sys.stderr)
            return 1

    if not legacy_dir.is_dir():
        print(f"No legacy_trades directory: {legacy_dir.resolve()}", file=sys.stderr)
        return 2

    leg_limit = args.legacy_trade_sample_files if args.scan_mode == "sample" else None
    legacy_files = _first_files(legacy_dir, "trades_*.parquet", leg_limit)
    if not legacy_files:
        print(f"No trades_*.parquet under {legacy_dir.resolve()}", file=sys.stderr)
        return 3

    fpmm_values = list(args.fpmm_address)
    if args.fpmm_csv is not None:
        fpmm_csv = args.fpmm_csv
        if not fpmm_csv.is_absolute():
            fpmm_csv = Path.cwd() / fpmm_csv
        if not fpmm_csv.is_file():
            print(f"FPMM CSV not found: {fpmm_csv}", file=sys.stderr)
            return 4
        with fpmm_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                val = str(row.get("fpmm_address") or "").strip()
                if val:
                    fpmm_values.append(val)

    fpmm_targets = _normalize_addresses(fpmm_values)
    trader_targets = _normalize_addresses(args.trader)

    if args.ledger_min is not None and args.ledger_max is not None and args.ledger_min > args.ledger_max:
        print("--ledger-min cannot be greater than --ledger-max", file=sys.stderr)
        return 5

    legacy_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in legacy_files)
    legacy_from = f"read_parquet([{legacy_file_sql}], union_by_name=true)"
    markets_dir = args.data_dir / "markets"
    if not markets_dir.is_dir():
        alt_markets = args.data_dir / "parquet" / "markets"
        if alt_markets.is_dir():
            markets_dir = alt_markets
    market_files = _first_files(markets_dir, "markets_*.parquet", None)
    markets_from = ""
    if market_files:
        market_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in market_files)
        markets_from = f"read_parquet([{market_file_sql}], union_by_name=true)"

    con = duckdb.connect(database=":memory:")
    if args.duckdb_threads and args.duckdb_threads > 0:
        con.execute(f"PRAGMA threads={int(args.duckdb_threads)}")

    cols = {str(r[0]) for r in con.execute(f"DESCRIBE SELECT * FROM {legacy_from} LIMIT 0").fetchall()}
    needed = {
        "ledger_number",
        "transaction_hash",
        "log_index",
        "fpmm_address",
        "trader",
        "amount",
        "fee_amount",
        "outcome_index",
        "outcome_tokens",
        "is_buy",
    }
    missing = sorted(needed - cols)
    if missing:
        print(f"Legacy trades parquet missing columns: {missing}", file=sys.stderr)
        return 6

    where_parts = ["1=1"]
    if args.ledger_min is not None:
        where_parts.append(f"ledger_number >= {int(args.ledger_min)}")
    if args.ledger_max is not None:
        where_parts.append(f"ledger_number <= {int(args.ledger_max)}")
    if fpmm_targets:
        vals = _sql_values_str(fpmm_targets)
        _run_execute_with_heartbeat(
            con,
            f"CREATE TEMP TABLE wanted_fpmm AS SELECT * FROM (VALUES {vals}) AS t(fpmm_lower)",
            stage="build-wanted-fpmm",
            heartbeat_sec=args.heartbeat_sec,
        )
        where_parts.append(
            "lower(trim(cast(fpmm_address AS VARCHAR))) IN (SELECT fpmm_lower FROM wanted_fpmm)"
        )
    if trader_targets:
        vals = _sql_values_str(trader_targets)
        _run_execute_with_heartbeat(
            con,
            f"CREATE TEMP TABLE wanted_trader AS SELECT * FROM (VALUES {vals}) AS t(trader_lower)",
            stage="build-wanted-trader",
            heartbeat_sec=args.heartbeat_sec,
        )
        where_parts.append(
            "lower(trim(cast(trader AS VARCHAR))) IN (SELECT trader_lower FROM wanted_trader)"
        )

    where_sql = " AND ".join(where_parts)

    collateral_map: dict[str, dict[str, str]] = {}
    if collateral_json.is_file():
        try:
            raw = json.loads(collateral_json.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, value in raw.items():
                    k = str(key or "").strip().lower()
                    if not k:
                        continue
                    if not k.startswith("0x"):
                        k = "0x" + k
                    if isinstance(value, dict):
                        collateral_map[k] = {
                            "collateral_address": str(value.get("collateral_address") or ""),
                            "collateral_symbol": str(value.get("collateral_symbol") or ""),
                        }
        except Exception as e:
            print(f"Warning: could not read collateral lookup {collateral_json}: {e}", file=sys.stderr)

    print(f"data_dir:                  {args.data_dir}")
    print(f"legacy_trades_dir:         {legacy_dir.resolve()}")
    print(f"scan_mode:                 {args.scan_mode}")
    print(f"legacy_trade_files_scanned:{len(legacy_files)}")
    print(f"input_csv:                 {input_csv if input_csv else '(disabled)'}")
    print(f"fpmm_targets:              {len(fpmm_targets)}")
    print(f"trader_targets:            {len(trader_targets)}")
    print(f"ledger_min:                 {args.ledger_min}")
    print(f"ledger_max:                 {args.ledger_max}")

    print("stage: filtered legacy trade summary", file=sys.stderr)
    filtered_summary = _run_fetchall_with_heartbeat(
        con,
        f"""
        SELECT
            count(*)::BIGINT AS n_trade_rows,
            count(DISTINCT lower(trim(cast(fpmm_address AS VARCHAR))))::BIGINT AS n_distinct_fpmm,
            count(DISTINCT lower(trim(cast(trader AS VARCHAR))))::BIGINT AS n_distinct_traders,
            count(DISTINCT transaction_hash)::BIGINT AS n_distinct_transactions,
            sum(CASE WHEN is_buy THEN 1 ELSE 0 END)::BIGINT AS n_buy_rows,
            sum(CASE WHEN NOT is_buy THEN 1 ELSE 0 END)::BIGINT AS n_sell_rows,
            min(ledger_number) AS min_ledger_number,
            max(ledger_number) AS max_ledger_number,
            min(timestamp) AS min_timestamp,
            max(timestamp) AS max_timestamp,
            sum(try_cast(amount AS DOUBLE)) AS sum_amount_raw,
            sum(try_cast(fee_amount AS DOUBLE)) AS sum_fee_amount_raw,
            sum(try_cast(outcome_tokens AS DOUBLE)) AS sum_outcome_tokens_raw
        FROM {legacy_from}
        WHERE {where_sql}
        """,
        stage="filtered-summary",
        heartbeat_sec=args.heartbeat_sec,
    )
    fs = filtered_summary[0]

    per_fpmm_rows = _run_fetchall_with_heartbeat(
        con,
        f"""
        SELECT
            lower(trim(cast(fpmm_address AS VARCHAR))) AS fpmm_address,
            count(*)::BIGINT AS n_trade_rows,
            count(DISTINCT transaction_hash)::BIGINT AS n_distinct_transactions,
            count(DISTINCT lower(trim(cast(trader AS VARCHAR))))::BIGINT AS n_distinct_traders,
            sum(CASE WHEN is_buy THEN 1 ELSE 0 END)::BIGINT AS n_buy_rows,
            sum(CASE WHEN NOT is_buy THEN 1 ELSE 0 END)::BIGINT AS n_sell_rows,
            min(ledger_number) AS min_ledger_number,
            max(ledger_number) AS max_ledger_number,
            min(timestamp) AS min_timestamp,
            max(timestamp) AS max_timestamp,
            sum(try_cast(amount AS DOUBLE)) AS sum_amount_raw,
            sum(try_cast(fee_amount AS DOUBLE)) AS sum_fee_amount_raw,
            sum(try_cast(outcome_tokens AS DOUBLE)) AS sum_outcome_tokens_raw
        FROM {legacy_from}
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY n_trade_rows DESC, fpmm_address
        LIMIT {int(args.top_n)}
        """,
        stage="per-fpmm",
        heartbeat_sec=args.heartbeat_sec,
    )

    per_outcome_rows = _run_fetchall_with_heartbeat(
        con,
        f"""
        SELECT
            lower(trim(cast(fpmm_address AS VARCHAR))) AS fpmm_address,
            outcome_index,
            count(*)::BIGINT AS n_trade_rows,
            sum(CASE WHEN is_buy THEN 1 ELSE 0 END)::BIGINT AS n_buy_rows,
            sum(CASE WHEN NOT is_buy THEN 1 ELSE 0 END)::BIGINT AS n_sell_rows,
            count(DISTINCT lower(trim(cast(trader AS VARCHAR))))::BIGINT AS n_distinct_traders,
            min(ledger_number) AS min_ledger_number,
            max(ledger_number) AS max_ledger_number,
            sum(try_cast(amount AS DOUBLE)) AS sum_amount_raw,
            sum(try_cast(outcome_tokens AS DOUBLE)) AS sum_outcome_tokens_raw
        FROM {legacy_from}
        WHERE {where_sql}
        GROUP BY 1, 2
        ORDER BY n_trade_rows DESC, fpmm_address, outcome_index
        LIMIT {int(args.top_n)}
        """,
        stage="per-outcome",
        heartbeat_sec=args.heartbeat_sec,
    )

    top_trader_rows = _run_fetchall_with_heartbeat(
        con,
        f"""
        SELECT
            lower(trim(cast(fpmm_address AS VARCHAR))) AS fpmm_address,
            lower(trim(cast(trader AS VARCHAR))) AS trader,
            count(*)::BIGINT AS n_trade_rows,
            sum(CASE WHEN is_buy THEN 1 ELSE 0 END)::BIGINT AS n_buy_rows,
            sum(CASE WHEN NOT is_buy THEN 1 ELSE 0 END)::BIGINT AS n_sell_rows,
            count(DISTINCT transaction_hash)::BIGINT AS n_distinct_transactions,
            min(ledger_number) AS min_ledger_number,
            max(ledger_number) AS max_ledger_number,
            sum(try_cast(amount AS DOUBLE)) AS sum_amount_raw,
            sum(try_cast(fee_amount AS DOUBLE)) AS sum_fee_amount_raw
        FROM {legacy_from}
        WHERE {where_sql}
        GROUP BY 1, 2
        ORDER BY n_trade_rows DESC, fpmm_address, trader
        LIMIT {int(args.top_n)}
        """,
        stage="top-traders",
        heartbeat_sec=args.heartbeat_sec,
    )

    sample_rows = _run_fetchall_with_heartbeat(
        con,
        f"""
        SELECT
            ledger_number,
            transaction_hash,
            log_index,
            lower(trim(cast(fpmm_address AS VARCHAR))) AS fpmm_address,
            lower(trim(cast(trader AS VARCHAR))) AS trader,
            amount,
            fee_amount,
            outcome_index,
            outcome_tokens,
            is_buy,
            timestamp,
            cast(_fetched_at AS VARCHAR) AS _fetched_at
        FROM {legacy_from}
        WHERE {where_sql}
        ORDER BY ledger_number, transaction_hash, log_index
        LIMIT {int(args.sample_rows)}
        """,
        stage="sample-rows",
        heartbeat_sec=args.heartbeat_sec,
    )

    out_summary = _OUT_DIR / f"{args.output_prefix}_summary.json"
    out_fpmm = _OUT_DIR / f"{args.output_prefix}_fpmm_summary.csv"
    out_outcome = _OUT_DIR / f"{args.output_prefix}_outcome_summary.csv"
    out_trader = _OUT_DIR / f"{args.output_prefix}_top_traders.csv"
    out_samples = _OUT_DIR / f"{args.output_prefix}_sample_rows.csv"
    out_market_coverage = _OUT_DIR / f"{args.output_prefix}_market_coverage.csv"
    out_event_coverage = _OUT_DIR / f"{args.output_prefix}_event_coverage.csv"

    fpmm_dicts: list[dict[str, Any]] = []
    for row in per_fpmm_rows:
        addr = str(row[0] or "")
        collateral = collateral_map.get(addr, {})
        fpmm_dicts.append(
            {
                "fpmm_address": addr,
                "collateral_symbol": collateral.get("collateral_symbol", ""),
                "collateral_address": collateral.get("collateral_address", ""),
                "n_trade_rows": int(row[1] or 0),
                "n_distinct_transactions": int(row[2] or 0),
                "n_distinct_traders": int(row[3] or 0),
                "n_buy_rows": int(row[4] or 0),
                "n_sell_rows": int(row[5] or 0),
                "min_ledger_number": row[6] if row[6] is not None else "",
                "max_ledger_number": row[7] if row[7] is not None else "",
                "min_timestamp": row[8] if row[8] is not None else "",
                "max_timestamp": row[9] if row[9] is not None else "",
                "sum_amount_raw": row[10] if row[10] is not None else "",
                "sum_fee_amount_raw": row[11] if row[11] is not None else "",
                "sum_outcome_tokens_raw": row[12] if row[12] is not None else "",
            }
        )

    outcome_dicts = [
        {
            "fpmm_address": str(row[0] or ""),
            "outcome_index": row[1] if row[1] is not None else "",
            "n_trade_rows": int(row[2] or 0),
            "n_buy_rows": int(row[3] or 0),
            "n_sell_rows": int(row[4] or 0),
            "n_distinct_traders": int(row[5] or 0),
            "min_ledger_number": row[6] if row[6] is not None else "",
            "max_ledger_number": row[7] if row[7] is not None else "",
            "sum_amount_raw": row[8] if row[8] is not None else "",
            "sum_outcome_tokens_raw": row[9] if row[9] is not None else "",
        }
        for row in per_outcome_rows
    ]

    trader_dicts = [
        {
            "fpmm_address": str(row[0] or ""),
            "trader": str(row[1] or ""),
            "n_trade_rows": int(row[2] or 0),
            "n_buy_rows": int(row[3] or 0),
            "n_sell_rows": int(row[4] or 0),
            "n_distinct_transactions": int(row[5] or 0),
            "min_ledger_number": row[6] if row[6] is not None else "",
            "max_ledger_number": row[7] if row[7] is not None else "",
            "sum_amount_raw": row[8] if row[8] is not None else "",
            "sum_fee_amount_raw": row[9] if row[9] is not None else "",
        }
        for row in top_trader_rows
    ]

    sample_dicts = [
        {
            "ledger_number": row[0] if row[0] is not None else "",
            "transaction_hash": str(row[1] or ""),
            "log_index": row[2] if row[2] is not None else "",
            "fpmm_address": str(row[3] or ""),
            "trader": str(row[4] or ""),
            "amount": str(row[5] or ""),
            "fee_amount": str(row[6] or ""),
            "outcome_index": row[7] if row[7] is not None else "",
            "outcome_tokens": str(row[8] or ""),
            "is_buy": "" if row[9] is None else str(row[9]),
            "timestamp": row[10] if row[10] is not None else "",
            "_fetched_at": str(row[11] or ""),
        }
        for row in sample_rows
    ]

    market_coverage_dicts: list[dict[str, Any]] = []
    event_coverage_dicts: list[dict[str, Any]] = []
    if input_csv is not None:
        if not markets_from:
            print("Warning: no markets_*.parquet found; skipped Step-01 coverage outputs.", file=sys.stderr)
        else:
            market_cols = {str(r[0]) for r in con.execute(f"DESCRIBE SELECT * FROM {markets_from} LIMIT 0").fetchall()}
            fpmm_col = None
            for candidate in ("market_maker_address", "marketMakerAddress", "fpmm_address"):
                if candidate in market_cols:
                    fpmm_col = candidate
                    break
            if fpmm_col is None:
                print("Warning: markets parquet missing FPMM column; skipped Step-01 coverage outputs.", file=sys.stderr)
            else:
                input_csv_sql = _sql_quote(_glob_sql(input_csv))
                market_coverage_rows = _run_fetchall_with_heartbeat(
                    con,
                    f"""
                    WITH wanted AS (
                        SELECT *
                        FROM read_csv_auto({input_csv_sql}, header=true)
                    ),
                    wanted_markets AS (
                        SELECT
                            row_number() OVER () AS input_row_num,
                            cast(coalesce(event_key, '') AS VARCHAR) AS event_key,
                            cast(coalesce(parent_event_slug, '') AS VARCHAR) AS parent_event_slug,
                            cast(coalesce(parent_event_title, '') AS VARCHAR) AS parent_event_title,
                            cast(coalesce(parent_series_slug, '') AS VARCHAR) AS parent_series_slug,
                            cast(coalesce(group_item_titles, '') AS VARCHAR) AS group_item_titles,
                            cast(coalesce(top_volume_slug, '') AS VARCHAR) AS top_volume_slug,
                            cast(coalesce(slug, '') AS VARCHAR) AS slug,
                            cast(coalesce(question, '') AS VARCHAR) AS question,
                            cast(coalesce(condition_id, '') AS VARCHAR) AS condition_id,
                            cast(coalesce(id, '') AS VARCHAR) AS market_id,
                            cast(min_end_date AS VARCHAR) AS min_end_date,
                            cast(max_end_date AS VARCHAR) AS max_end_date
                        FROM wanted
                    ),
                    market_map AS (
                        SELECT DISTINCT
                            cast(m.slug AS VARCHAR) AS slug,
                            cast(coalesce(m.id, '') AS VARCHAR) AS market_id,
                            cast(coalesce(m.condition_id, '') AS VARCHAR) AS condition_id,
                            lower(trim(cast(m.{fpmm_col} AS VARCHAR))) AS fpmm_address
                        FROM {markets_from} m
                        WHERE m.{fpmm_col} IS NOT NULL
                          AND trim(cast(m.{fpmm_col} AS VARCHAR)) <> ''
                    ),
                    legacy_by_fpmm AS (
                        SELECT
                            lower(trim(cast(fpmm_address AS VARCHAR))) AS fpmm_address,
                            count(*)::BIGINT AS n_legacy_trade_rows,
                            count(DISTINCT transaction_hash)::BIGINT AS n_legacy_distinct_transactions,
                            count(DISTINCT lower(trim(cast(trader AS VARCHAR))))::BIGINT AS n_legacy_distinct_traders,
                            count(DISTINCT outcome_index)::BIGINT AS n_legacy_distinct_outcomes,
                            sum(CASE WHEN is_buy THEN 1 ELSE 0 END)::BIGINT AS n_legacy_buy_rows,
                            sum(CASE WHEN NOT is_buy THEN 1 ELSE 0 END)::BIGINT AS n_legacy_sell_rows,
                            min(ledger_number) AS min_ledger_legacy,
                            max(ledger_number) AS max_ledger_legacy,
                            min(timestamp) AS min_timestamp_legacy,
                            max(timestamp) AS max_timestamp_legacy
                        FROM {legacy_from}
                        GROUP BY 1
                    )
                    SELECT
                        w.input_row_num,
                        w.event_key,
                        w.parent_event_slug,
                        w.parent_event_title,
                        w.parent_series_slug,
                        w.group_item_titles,
                        w.top_volume_slug,
                        w.slug,
                        w.question,
                        coalesce(mm.market_id, w.market_id) AS market_id,
                        coalesce(mm.condition_id, w.condition_id) AS condition_id,
                        mm.fpmm_address,
                        coalesce(l.n_legacy_trade_rows, 0) AS n_legacy_trade_rows,
                        coalesce(l.n_legacy_distinct_transactions, 0) AS n_legacy_distinct_transactions,
                        coalesce(l.n_legacy_distinct_traders, 0) AS n_legacy_distinct_traders,
                        coalesce(l.n_legacy_distinct_outcomes, 0) AS n_legacy_distinct_outcomes,
                        coalesce(l.n_legacy_buy_rows, 0) AS n_legacy_buy_rows,
                        coalesce(l.n_legacy_sell_rows, 0) AS n_legacy_sell_rows,
                        l.min_ledger_legacy,
                        l.max_ledger_legacy,
                        l.min_timestamp_legacy,
                        l.max_timestamp_legacy,
                        w.min_end_date,
                        w.max_end_date,
                        CASE
                            WHEN mm.fpmm_address IS NULL THEN 'missing_fpmm_in_markets_parquet'
                            WHEN coalesce(l.n_legacy_trade_rows, 0) > 0 THEN 'covered_by_legacy_trades'
                            ELSE 'mapped_fpmm_but_no_legacy_trades'
                        END AS legacy_coverage_status
                    FROM wanted_markets w
                    LEFT JOIN market_map mm ON w.slug = mm.slug
                    LEFT JOIN legacy_by_fpmm l ON mm.fpmm_address = l.fpmm_address
                    ORDER BY w.input_row_num
                    """,
                    stage="market-coverage",
                    heartbeat_sec=args.heartbeat_sec,
                )

                for row in market_coverage_rows:
                    market_coverage_dicts.append(
                        {
                            "input_row_num": int(row[0] or 0),
                            "event_key": str(row[1] or ""),
                            "parent_event_slug": str(row[2] or ""),
                            "parent_event_title": str(row[3] or ""),
                            "parent_series_slug": str(row[4] or ""),
                            "group_item_titles": str(row[5] or ""),
                            "top_volume_slug": str(row[6] or ""),
                            "market_slug": str(row[7] or ""),
                            "question": str(row[8] or ""),
                            "market_id": str(row[9] or ""),
                            "condition_id": str(row[10] or ""),
                            "fpmm_address": str(row[11] or ""),
                            "n_legacy_trade_rows": int(row[12] or 0),
                            "n_legacy_distinct_transactions": int(row[13] or 0),
                            "n_legacy_distinct_traders": int(row[14] or 0),
                            "n_legacy_distinct_outcomes": int(row[15] or 0),
                            "n_legacy_buy_rows": int(row[16] or 0),
                            "n_legacy_sell_rows": int(row[17] or 0),
                            "min_ledger_legacy": row[18] if row[18] is not None else "",
                            "max_ledger_legacy": row[19] if row[19] is not None else "",
                            "min_timestamp_legacy": row[20] if row[20] is not None else "",
                            "max_timestamp_legacy": row[21] if row[21] is not None else "",
                            "min_end_date": str(row[22] or ""),
                            "max_end_date": str(row[23] or ""),
                            "legacy_coverage_status": str(row[24] or ""),
                        }
                    )

                event_coverage_rows = _run_fetchall_with_heartbeat(
                    con,
                    f"""
                    WITH wanted AS (
                        SELECT *
                        FROM read_csv_auto({input_csv_sql}, header=true)
                    ),
                    wanted_markets AS (
                        SELECT
                            row_number() OVER () AS input_row_num,
                            cast(coalesce(event_key, '') AS VARCHAR) AS event_key,
                            cast(coalesce(parent_event_slug, '') AS VARCHAR) AS parent_event_slug,
                            cast(coalesce(parent_event_title, '') AS VARCHAR) AS parent_event_title,
                            cast(coalesce(parent_series_slug, '') AS VARCHAR) AS parent_series_slug,
                            cast(coalesce(group_item_titles, '') AS VARCHAR) AS group_item_titles,
                            cast(coalesce(slug, '') AS VARCHAR) AS slug,
                            cast(coalesce(question, '') AS VARCHAR) AS question,
                            cast(coalesce(id, '') AS VARCHAR) AS market_id,
                            cast(min_end_date AS VARCHAR) AS min_end_date,
                            cast(max_end_date AS VARCHAR) AS max_end_date
                        FROM wanted
                    ),
                    market_map AS (
                        SELECT DISTINCT
                            cast(m.slug AS VARCHAR) AS slug,
                            lower(trim(cast(m.{fpmm_col} AS VARCHAR))) AS fpmm_address
                        FROM {markets_from} m
                        WHERE m.{fpmm_col} IS NOT NULL
                          AND trim(cast(m.{fpmm_col} AS VARCHAR)) <> ''
                    ),
                    legacy_by_fpmm AS (
                        SELECT
                            lower(trim(cast(fpmm_address AS VARCHAR))) AS fpmm_address,
                            count(*)::BIGINT AS n_legacy_trade_rows,
                            count(DISTINCT transaction_hash)::BIGINT AS n_legacy_distinct_transactions,
                            count(DISTINCT lower(trim(cast(trader AS VARCHAR))))::BIGINT AS n_legacy_distinct_traders,
                            min(ledger_number) AS min_ledger_legacy,
                            max(ledger_number) AS max_ledger_legacy
                        FROM {legacy_from}
                        GROUP BY 1
                    ),
                    coverage AS (
                        SELECT
                            w.event_key,
                            w.parent_event_slug,
                            w.parent_event_title,
                            w.parent_series_slug,
                            w.min_end_date,
                            w.max_end_date,
                            w.slug,
                            mm.fpmm_address,
                            coalesce(l.n_legacy_trade_rows, 0) AS n_legacy_trade_rows,
                            coalesce(l.n_legacy_distinct_transactions, 0) AS n_legacy_distinct_transactions,
                            coalesce(l.n_legacy_distinct_traders, 0) AS n_legacy_distinct_traders,
                            l.min_ledger_legacy,
                            l.max_ledger_legacy
                        FROM wanted_markets w
                        LEFT JOIN market_map mm ON w.slug = mm.slug
                        LEFT JOIN legacy_by_fpmm l ON mm.fpmm_address = l.fpmm_address
                    )
                    SELECT
                        event_key,
                        parent_event_slug,
                        parent_event_title,
                        parent_series_slug,
                        min_end_date,
                        max_end_date,
                        count(*)::BIGINT AS n_markets_in_input,
                        sum(CASE WHEN fpmm_address IS NOT NULL THEN 1 ELSE 0 END)::BIGINT AS n_markets_with_fpmm_mapping,
                        sum(CASE WHEN n_legacy_trade_rows > 0 THEN 1 ELSE 0 END)::BIGINT AS n_markets_with_legacy_trades,
                        sum(n_legacy_trade_rows)::BIGINT AS n_legacy_trade_rows,
                        sum(n_legacy_distinct_transactions)::BIGINT AS n_legacy_distinct_transactions,
                        sum(n_legacy_distinct_traders)::BIGINT AS n_legacy_distinct_traders,
                        min(min_ledger_legacy) AS min_ledger_legacy,
                        max(max_ledger_legacy) AS max_ledger_legacy,
                        string_agg(slug, '; ' ORDER BY slug) FILTER (WHERE fpmm_address IS NULL) AS missing_fpmm_market_slugs,
                        string_agg(slug, '; ' ORDER BY slug) FILTER (WHERE fpmm_address IS NOT NULL AND n_legacy_trade_rows = 0) AS mapped_but_uncovered_market_slugs,
                        string_agg(slug, '; ' ORDER BY slug) FILTER (WHERE n_legacy_trade_rows > 0) AS covered_market_slugs
                    FROM coverage
                    GROUP BY 1, 2, 3, 4, 5, 6
                    ORDER BY min_end_date, event_key
                    """,
                    stage="event-coverage",
                    heartbeat_sec=args.heartbeat_sec,
                )

                for row in event_coverage_rows:
                    n_markets_in_input = int(row[6] or 0)
                    n_markets_with_legacy_trades = int(row[8] or 0)
                    event_coverage_dicts.append(
                        {
                            "event_key": str(row[0] or ""),
                            "parent_event_slug": str(row[1] or ""),
                            "parent_event_title": str(row[2] or ""),
                            "parent_series_slug": str(row[3] or ""),
                            "min_end_date": str(row[4] or ""),
                            "max_end_date": str(row[5] or ""),
                            "n_markets_in_input": n_markets_in_input,
                            "n_markets_with_fpmm_mapping": int(row[7] or 0),
                            "n_markets_with_legacy_trades": n_markets_with_legacy_trades,
                            "legacy_leg_coverage_ratio": (
                                f"{(n_markets_with_legacy_trades / n_markets_in_input):.6f}"
                                if n_markets_in_input > 0
                                else ""
                            ),
                            "n_legacy_trade_rows": int(row[9] or 0),
                            "n_legacy_distinct_transactions": int(row[10] or 0),
                            "n_legacy_distinct_traders": int(row[11] or 0),
                            "min_ledger_legacy": row[12] if row[12] is not None else "",
                            "max_ledger_legacy": row[13] if row[13] is not None else "",
                            "missing_fpmm_market_slugs": str(row[14] or ""),
                            "mapped_but_uncovered_market_slugs": str(row[15] or ""),
                            "covered_market_slugs": str(row[16] or ""),
                        }
                    )

    _write_csv(
        out_fpmm,
        fpmm_dicts,
        [
            "fpmm_address",
            "collateral_symbol",
            "collateral_address",
            "n_trade_rows",
            "n_distinct_transactions",
            "n_distinct_traders",
            "n_buy_rows",
            "n_sell_rows",
            "min_ledger_number",
            "max_ledger_number",
            "min_timestamp",
            "max_timestamp",
            "sum_amount_raw",
            "sum_fee_amount_raw",
            "sum_outcome_tokens_raw",
        ],
    )
    _write_csv(
        out_outcome,
        outcome_dicts,
        [
            "fpmm_address",
            "outcome_index",
            "n_trade_rows",
            "n_buy_rows",
            "n_sell_rows",
            "n_distinct_traders",
            "min_ledger_number",
            "max_ledger_number",
            "sum_amount_raw",
            "sum_outcome_tokens_raw",
        ],
    )
    _write_csv(
        out_trader,
        trader_dicts,
        [
            "fpmm_address",
            "trader",
            "n_trade_rows",
            "n_buy_rows",
            "n_sell_rows",
            "n_distinct_transactions",
            "min_ledger_number",
            "max_ledger_number",
            "sum_amount_raw",
            "sum_fee_amount_raw",
        ],
    )
    _write_csv(
        out_samples,
        sample_dicts,
        [
            "ledger_number",
            "transaction_hash",
            "log_index",
            "fpmm_address",
            "trader",
            "amount",
            "fee_amount",
            "outcome_index",
            "outcome_tokens",
            "is_buy",
            "timestamp",
            "_fetched_at",
        ],
    )
    if market_coverage_dicts:
        _write_csv(
            out_market_coverage,
            market_coverage_dicts,
            [
                "input_row_num",
                "event_key",
                "parent_event_slug",
                "parent_event_title",
                "parent_series_slug",
                "group_item_titles",
                "top_volume_slug",
                "market_slug",
                "question",
                "market_id",
                "condition_id",
                "fpmm_address",
                "n_legacy_trade_rows",
                "n_legacy_distinct_transactions",
                "n_legacy_distinct_traders",
                "n_legacy_distinct_outcomes",
                "n_legacy_buy_rows",
                "n_legacy_sell_rows",
                "min_ledger_legacy",
                "max_ledger_legacy",
                "min_timestamp_legacy",
                "max_timestamp_legacy",
                "min_end_date",
                "max_end_date",
                "legacy_coverage_status",
            ],
        )
    if event_coverage_dicts:
        _write_csv(
            out_event_coverage,
            event_coverage_dicts,
            [
                "event_key",
                "parent_event_slug",
                "parent_event_title",
                "parent_series_slug",
                "min_end_date",
                "max_end_date",
                "n_markets_in_input",
                "n_markets_with_fpmm_mapping",
                "n_markets_with_legacy_trades",
                "legacy_leg_coverage_ratio",
                "n_legacy_trade_rows",
                "n_legacy_distinct_transactions",
                "n_legacy_distinct_traders",
                "min_ledger_legacy",
                "max_ledger_legacy",
                "missing_fpmm_market_slugs",
                "mapped_but_uncovered_market_slugs",
                "covered_market_slugs",
            ],
        )

    summary: dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "legacy_trades_dir": str(legacy_dir),
        "input_csv": str(input_csv) if input_csv else None,
        "collateral_lookup_json": str(collateral_json) if collateral_json.is_file() else None,
        "scan_mode": args.scan_mode,
        "legacy_trade_files_scanned": len(legacy_files),
        "fpmm_targets": fpmm_targets,
        "trader_targets": trader_targets,
        "ledger_min": args.ledger_min,
        "ledger_max": args.ledger_max,
        "n_trade_rows": int(fs[0] or 0),
        "n_distinct_fpmm": int(fs[1] or 0),
        "n_distinct_traders": int(fs[2] or 0),
        "n_distinct_transactions": int(fs[3] or 0),
        "n_buy_rows": int(fs[4] or 0),
        "n_sell_rows": int(fs[5] or 0),
        "min_ledger_number": fs[6] if fs[6] is not None else None,
        "max_ledger_number": fs[7] if fs[7] is not None else None,
        "min_timestamp": fs[8] if fs[8] is not None else None,
        "max_timestamp": fs[9] if fs[9] is not None else None,
        "sum_amount_raw": fs[10] if fs[10] is not None else None,
        "sum_fee_amount_raw": fs[11] if fs[11] is not None else None,
        "sum_outcome_tokens_raw": fs[12] if fs[12] is not None else None,
        "n_step01_market_rows": len(market_coverage_dicts),
        "n_step01_markets_with_legacy_trades": sum(
            1 for row in market_coverage_dicts if int(row.get("n_legacy_trade_rows") or 0) > 0
        ),
        "n_step01_events": len(event_coverage_dicts),
        "output_fpmm_summary_csv": str(out_fpmm),
        "output_outcome_summary_csv": str(out_outcome),
        "output_top_traders_csv": str(out_trader),
        "output_sample_rows_csv": str(out_samples),
        "output_market_coverage_csv": str(out_market_coverage) if market_coverage_dicts else None,
        "output_event_coverage_csv": str(out_event_coverage) if event_coverage_dicts else None,
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"matched_trade_rows:         {summary['n_trade_rows']}")
    print(f"matched_distinct_fpmm:      {summary['n_distinct_fpmm']}")
    print(f"matched_distinct_traders:   {summary['n_distinct_traders']}")
    print(f"matched_transactions:       {summary['n_distinct_transactions']}")
    print(f"wrote: {out_fpmm}")
    print(f"wrote: {out_outcome}")
    print(f"wrote: {out_trader}")
    print(f"wrote: {out_samples}")
    if market_coverage_dicts:
        print(f"wrote: {out_market_coverage}")
    if event_coverage_dicts:
        print(f"wrote: {out_event_coverage}")
    print(f"wrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
