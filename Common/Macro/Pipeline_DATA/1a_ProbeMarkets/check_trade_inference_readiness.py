"""
Check trade-inference readiness for probability time series construction.

This sister probe complements timestamp availability checks by evaluating:
  1) asset-role / side inferability from market token mapping,
  2) outcome-label joinability from markets metadata,
  3) amount-field sanity and ratio computability,
  4) ledger-time recoverability via ledgers parquet.

Usage (repo root):
  python scripts/util/Step_03_ProbeTrades/check_trade_inference_readiness.py
  python scripts/util/Step_03_ProbeTrades/check_trade_inference_readiness.py --scan-mode sample --trade-sample-files 250
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

_UTIL_DIR = Path(__file__).resolve().parents[1]
_MARKETS_OUT_DIR = _UTIL_DIR / "out_Markets"
_OUT_DIR = _UTIL_DIR / "out_Trades" / "RDYCHK"
_STEP01_PREFIX = "1EXTRACT"
_DEFAULT_INPUT = _MARKETS_OUT_DIR / f"{_STEP01_PREFIX}_fed_parquet_events.csv"
_DEFAULT_OUT_PREFIX = "3PROBE_RDYCHK"


@dataclass(frozen=True)
class MarketTokenMeta:
    market_slug: str
    token_id: str
    outcome_label: str
    outcome_label_source: str


def _glob_sql(path: Path) -> str:
    return str(path).replace("\\", "/")


def _csv_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    candidate_trades = _OUT_DIR / p
    if candidate_trades.exists():
        return candidate_trades
    return _MARKETS_OUT_DIR / p


def _resolve_data_dirs(root: Path) -> tuple[Path, Path, Path]:
    m_dir = root / "markets"
    t_dir = root / "trades"
    b_dir = root / "ledgers"
    if m_dir.is_dir() and t_dir.is_dir():
        return m_dir, t_dir, b_dir

    alt_m = root / "parquet" / "markets"
    alt_t = root / "parquet" / "trades"
    alt_b = root / "parquet" / "ledgers"
    if alt_m.is_dir() and alt_t.is_dir():
        return alt_m, alt_t, alt_b

    return m_dir, t_dir, b_dir


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
    esc = [f"({_sql_quote(v)})" for v in vals]
    return ", ".join(esc)


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
    """
    Execute query + fetchall while printing a single-line heartbeat.
    """
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
    """
    Execute statement while printing a single-line heartbeat.
    """
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


def _to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    s = str(value).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            return []
    if ";" in s:
        return [p.strip() for p in s.split(";") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _pick_first_present(cols: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _build_market_token_meta(
    con: duckdb.DuckDBPyConnection,
    markets_from: str,
    market_slugs: list[str],
    market_cols: set[str],
) -> tuple[list[MarketTokenMeta], str | None]:
    slug_values = _sql_values_str(market_slugs)

    outcome_col = _pick_first_present(
        market_cols,
        [
            "outcomes",
            "outcome_names",
            "outcomeNames",
            "outcome_labels",
            "market_outcomes",
        ],
    )
    if "clob_token_ids" not in market_cols:
        return [], outcome_col

    selected_cols = ["slug", "clob_token_ids"]
    if outcome_col:
        selected_cols.append(outcome_col)
    select_sql = ", ".join(selected_cols)

    rows = con.execute(
        f"""
        WITH wanted(slug) AS (VALUES {slug_values})
        SELECT {select_sql}
        FROM {markets_from}
        WHERE slug IN (SELECT slug FROM wanted)
        """
    ).fetchall()

    out: list[MarketTokenMeta] = []
    for row in rows:
        slug = str(row[0])
        token_ids = _to_list(row[1])
        labels = _to_list(row[2]) if outcome_col and len(row) > 2 else []
        for i, token_id in enumerate(token_ids):
            label = labels[i] if i < len(labels) else ""
            out.append(
                MarketTokenMeta(
                    market_slug=slug,
                    token_id=token_id,
                    outcome_label=label,
                    outcome_label_source=outcome_col or "",
                )
            )
    return out, outcome_col


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", type=Path, default=_DEFAULT_INPUT)
    p.add_argument("--data-dir", type=Path, default=Path("data/parquet"))
    p.add_argument(
        "--scan-mode",
        choices=["full", "sample"],
        default="full",
        help="Use full parquet set or a sampled subset (default: full).",
    )
    p.add_argument(
        "--trade-sample-files",
        type=int,
        default=250,
        metavar="N",
        help="When --scan-mode=sample, read first N trades_*.parquet files (default: 250).",
    )
    p.add_argument(
        "--market-sample-files",
        type=int,
        default=None,
        metavar="N",
        help="Optional limit for markets_*.parquet file count.",
    )
    p.add_argument(
        "--ledger-sample-files",
        type=int,
        default=300,
        metavar="N",
        help="When --scan-mode=sample, read first N ledgers_*.parquet files (default: 300).",
    )
    p.add_argument("--output-prefix", type=str, default=_DEFAULT_OUT_PREFIX)
    p.add_argument(
        "--heartbeat-sec",
        type=float,
        default=5.0,
        help="Heartbeat cadence in seconds for long DuckDB stages (0 disables).",
    )
    p.add_argument(
        "--duckdb-threads",
        type=int,
        default=0,
        help="DuckDB worker threads (0 keeps DuckDB default/auto).",
    )
    args = p.parse_args(argv)

    input_csv = _csv_path(args.input_csv)
    if not input_csv.is_file():
        print(f"Input CSV not found: {input_csv}", file=sys.stderr)
        return 1

    market_slugs = _read_market_slugs(input_csv)
    if not market_slugs:
        print("No market slugs found in input CSV (need `slug` or `outcome_market_slugs`).", file=sys.stderr)
        return 2

    m_dir, t_dir, b_dir = _resolve_data_dirs(args.data_dir)
    if not m_dir.is_dir():
        print(f"No markets directory: {m_dir.resolve()}", file=sys.stderr)
        return 3
    if not t_dir.is_dir():
        print(f"No trades directory: {t_dir.resolve()}", file=sys.stderr)
        return 4

    trade_limit = args.trade_sample_files if args.scan_mode == "sample" else None
    ledger_limit = args.ledger_sample_files if args.scan_mode == "sample" else None
    market_files = _first_files(m_dir, "markets_*.parquet", args.market_sample_files)
    trade_files = _first_files(t_dir, "trades_*.parquet", trade_limit)
    ledger_files = _first_files(b_dir, "ledgers_*.parquet", ledger_limit) if b_dir.is_dir() else []
    if not market_files:
        print(f"No markets_*.parquet under {m_dir.resolve()}", file=sys.stderr)
        return 5
    if not trade_files:
        print(f"No trades_*.parquet under {t_dir.resolve()}", file=sys.stderr)
        return 6

    market_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in market_files)
    trade_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in trade_files)
    markets_from = f"read_parquet([{market_file_sql}], union_by_name=true)"
    trades_from = f"read_parquet([{trade_file_sql}], union_by_name=true)"

    con = duckdb.connect(database=":memory:")
    if args.duckdb_threads and args.duckdb_threads > 0:
        con.execute(f"PRAGMA threads={int(args.duckdb_threads)}")

    print("stage: detect trade schema")
    trade_schema = _run_fetchall_with_heartbeat(
        con,
        f"DESCRIBE SELECT * FROM {trades_from} LIMIT 0",
        stage="trade-schema",
        heartbeat_sec=args.heartbeat_sec,
    )
    trade_cols = {str(r[0]) for r in trade_schema}
    needed_trade_cols = {"maker_asset_id", "taker_asset_id", "maker_amount", "taker_amount", "ledger_number"}
    if not needed_trade_cols.issubset(trade_cols):
        print(f"Trades schema missing one of required columns: {sorted(needed_trade_cols)}", file=sys.stderr)
        return 7

    print("stage: detect market schema")
    market_schema = _run_fetchall_with_heartbeat(
        con,
        f"DESCRIBE SELECT * FROM {markets_from} LIMIT 0",
        stage="market-schema",
        heartbeat_sec=args.heartbeat_sec,
    )
    market_cols = {str(r[0]) for r in market_schema}
    if "slug" not in market_cols:
        print("Markets parquet missing `slug`.", file=sys.stderr)
        return 8

    print("stage: map markets to token IDs/labels")
    token_meta_rows, outcome_col_used = _build_market_token_meta(con, markets_from, market_slugs, market_cols)
    token_meta_rows = [r for r in token_meta_rows if r.token_id]
    if not token_meta_rows:
        print("No token IDs resolved from selected market slugs.", file=sys.stderr)
        return 9

    pair_values = ", ".join(
        f"({_sql_quote(r.market_slug)}, {_sql_quote(r.token_id)}, {_sql_quote(r.outcome_label)})"
        for r in token_meta_rows
    )
    _run_execute_with_heartbeat(
        con,
        f"""
        CREATE TEMP TABLE market_tokens AS
        SELECT * FROM (VALUES {pair_values}) AS t(market_slug, token_id, outcome_label)
        """,
        stage="build-market-token-table",
        heartbeat_sec=args.heartbeat_sec,
    )

    print("stage: build matched trade hits")
    # Build hits in a single grouped pass (avoids hits_raw + correlated EXISTS).
    _run_execute_with_heartbeat(
        con,
        f"""
        CREATE TEMP TABLE hits AS
        WITH joined AS (
            SELECT
                mt.market_slug,
                cast(tr.maker_asset_id AS VARCHAR) AS maker_asset_id,
                cast(tr.taker_asset_id AS VARCHAR) AS taker_asset_id,
                tr.maker_amount::DOUBLE AS maker_amount,
                tr.taker_amount::DOUBLE AS taker_amount,
                tr.ledger_number::BIGINT AS ledger_number,
                cast(tr.transaction_hash AS VARCHAR) AS transaction_hash,
                cast(tr.log_index AS VARCHAR) AS log_index,
                cast(tr.order_hash AS VARCHAR) AS order_hash,
                CASE WHEN mt.token_id = cast(tr.maker_asset_id AS VARCHAR) THEN 1 ELSE 0 END AS maker_match_flag,
                CASE WHEN mt.token_id = cast(tr.taker_asset_id AS VARCHAR) THEN 1 ELSE 0 END AS taker_match_flag
            FROM {trades_from} tr
            JOIN market_tokens mt
              ON cast(tr.maker_asset_id AS VARCHAR) = mt.token_id
              OR cast(tr.taker_asset_id AS VARCHAR) = mt.token_id
        )
        SELECT
            market_slug,
            maker_asset_id,
            taker_asset_id,
            maker_amount,
            taker_amount,
            ledger_number,
            transaction_hash,
            log_index,
            order_hash,
            max(maker_match_flag) AS maker_is_market_token,
            max(taker_match_flag) AS taker_is_market_token
        FROM joined
        GROUP BY
            market_slug,
            maker_asset_id,
            taker_asset_id,
            maker_amount,
            taker_amount,
            ledger_number,
            transaction_hash,
            log_index,
            order_hash
        """,
        stage="build-hits",
        heartbeat_sec=args.heartbeat_sec,
    )

    print("stage: aggregate side + amount metrics")
    side_amt_rows_raw = _run_fetchall_with_heartbeat(
        con,
        """
        SELECT
            market_slug,
            count(*)::BIGINT AS n_trade_rows,
            sum(CASE WHEN maker_is_market_token = 1 AND taker_is_market_token = 0 THEN 1 ELSE 0 END)::BIGINT AS n_maker_token_only,
            sum(CASE WHEN maker_is_market_token = 0 AND taker_is_market_token = 1 THEN 1 ELSE 0 END)::BIGINT AS n_taker_token_only,
            sum(CASE WHEN maker_is_market_token = 1 AND taker_is_market_token = 1 THEN 1 ELSE 0 END)::BIGINT AS n_both_token,
            sum(CASE WHEN maker_is_market_token = 0 AND taker_is_market_token = 0 THEN 1 ELSE 0 END)::BIGINT AS n_neither_token,
            sum(CASE WHEN maker_amount IS NOT NULL THEN 1 ELSE 0 END)::BIGINT AS n_maker_non_null,
            sum(CASE WHEN taker_amount IS NOT NULL THEN 1 ELSE 0 END)::BIGINT AS n_taker_non_null,
            sum(CASE WHEN maker_amount > 0 THEN 1 ELSE 0 END)::BIGINT AS n_maker_positive,
            sum(CASE WHEN taker_amount > 0 THEN 1 ELSE 0 END)::BIGINT AS n_taker_positive,
            sum(CASE WHEN maker_amount > 0 AND taker_amount > 0 THEN 1 ELSE 0 END)::BIGINT AS n_both_positive,
            min(maker_amount) AS min_maker_amount,
            max(maker_amount) AS max_maker_amount,
            min(taker_amount) AS min_taker_amount,
            max(taker_amount) AS max_taker_amount,
            approx_quantile(maker_amount, 0.5) AS p50_maker_amount,
            approx_quantile(maker_amount, 0.9) AS p90_maker_amount,
            approx_quantile(maker_amount, 0.99) AS p99_maker_amount,
            approx_quantile(taker_amount, 0.5) AS p50_taker_amount,
            approx_quantile(taker_amount, 0.9) AS p90_taker_amount,
            approx_quantile(taker_amount, 0.99) AS p99_taker_amount,
            approx_quantile(CASE WHEN maker_amount > 0 AND taker_amount > 0 THEN (taker_amount / maker_amount) ELSE NULL END, 0.5) AS p50_ratio_taker_over_maker,
            approx_quantile(CASE WHEN maker_amount > 0 AND taker_amount > 0 THEN (taker_amount / maker_amount) ELSE NULL END, 0.9) AS p90_ratio_taker_over_maker,
            avg(CASE WHEN maker_amount IS NOT NULL AND mod(cast(maker_amount AS HUGEINT), 1000000) = 0 THEN 1.0 ELSE 0.0 END) AS pct_maker_multiple_1e6,
            avg(CASE WHEN taker_amount IS NOT NULL AND mod(cast(taker_amount AS HUGEINT), 1000000) = 0 THEN 1.0 ELSE 0.0 END) AS pct_taker_multiple_1e6,
            avg(CASE WHEN maker_amount IS NOT NULL AND mod(cast(maker_amount AS HUGEINT), 1000000000000000000) = 0 THEN 1.0 ELSE 0.0 END) AS pct_maker_multiple_1e18,
            avg(CASE WHEN taker_amount IS NOT NULL AND mod(cast(taker_amount AS HUGEINT), 1000000000000000000) = 0 THEN 1.0 ELSE 0.0 END) AS pct_taker_multiple_1e18
        FROM hits
        GROUP BY
            market_slug,
        ORDER BY market_slug
        """,
        stage="side-amount-agg",
        heartbeat_sec=args.heartbeat_sec,
    )

    side_rows: list[dict[str, object]] = []
    amt_rows: list[dict[str, object]] = []
    for row in side_amt_rows_raw:
        n = int(row[1] or 0)
        classifiable = int(row[2] or 0) + int(row[3] or 0)
        side_rows.append(
            {
                "market_slug": str(row[0]),
                "n_trade_rows": n,
                "n_maker_token_only": int(row[2] or 0),
                "n_taker_token_only": int(row[3] or 0),
                "n_both_token": int(row[4] or 0),
                "n_neither_token": int(row[5] or 0),
                "n_side_classifiable_rows": classifiable,
                "pct_side_classifiable_rows": round((classifiable / n) * 100.0, 4) if n else 0.0,
            }
        )
        n_both_positive = int(row[10] or 0)
        amt_rows.append(
            {
                "market_slug": str(row[0]),
                "n_trade_rows": n,
                "n_maker_non_null": int(row[6] or 0),
                "n_taker_non_null": int(row[7] or 0),
                "n_maker_positive": int(row[8] or 0),
                "n_taker_positive": int(row[9] or 0),
                "n_both_positive": n_both_positive,
                "pct_both_positive": round((n_both_positive / n) * 100.0, 4) if n else 0.0,
                "min_maker_amount": row[11],
                "max_maker_amount": row[12],
                "min_taker_amount": row[13],
                "max_taker_amount": row[14],
                "p50_maker_amount": row[15],
                "p90_maker_amount": row[16],
                "p99_maker_amount": row[17],
                "p50_taker_amount": row[18],
                "p90_taker_amount": row[19],
                "p99_taker_amount": row[20],
                "p50_ratio_taker_over_maker": row[21],
                "p90_ratio_taker_over_maker": row[22],
                "pct_maker_multiple_1e6": round(float((row[23] or 0.0) * 100.0), 4),
                "pct_taker_multiple_1e6": round(float((row[24] or 0.0) * 100.0), 4),
                "pct_maker_multiple_1e18": round(float((row[25] or 0.0) * 100.0), 4),
                "pct_taker_multiple_1e18": round(float((row[26] or 0.0) * 100.0), 4),
            }
        )

    # (2) Outcome-label joinability.
    label_rows: list[dict[str, object]] = []
    labels_by_market: dict[str, dict[str, int]] = {}
    for r in token_meta_rows:
        stats = labels_by_market.setdefault(
            r.market_slug,
            {
                "n_tokens": 0,
                "n_tokens_with_outcome_label": 0,
                "n_tokens_without_outcome_label": 0,
            },
        )
        stats["n_tokens"] += 1
        if r.outcome_label.strip():
            stats["n_tokens_with_outcome_label"] += 1
        else:
            stats["n_tokens_without_outcome_label"] += 1

    for slug in market_slugs:
        stats = labels_by_market.get(
            slug,
            {"n_tokens": 0, "n_tokens_with_outcome_label": 0, "n_tokens_without_outcome_label": 0},
        )
        n_tokens = int(stats["n_tokens"])
        n_with = int(stats["n_tokens_with_outcome_label"])
        label_rows.append(
            {
                "market_slug": slug,
                "n_tokens": n_tokens,
                "n_tokens_with_outcome_label": n_with,
                "n_tokens_without_outcome_label": int(stats["n_tokens_without_outcome_label"]),
                "pct_tokens_with_outcome_label": round((n_with / n_tokens) * 100.0, 4) if n_tokens else 0.0,
                "outcome_label_source_column": outcome_col_used or "",
            }
        )

    # (4) Ledger-time recoverability.
    ledger_rows: list[dict[str, object]] = []
    ledger_summary: dict[str, Any] = {
        "ledger_files_scanned": len(ledger_files),
        "ledger_time_col_used": "",
        "global_distinct_ledgers": 0,
        "global_distinct_ledgers_with_time": 0,
        "global_pct_distinct_ledgers_with_time": 0.0,
    }
    if ledger_files:
        ledger_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in ledger_files)
        ledgers_from = f"read_parquet([{ledger_file_sql}], union_by_name=true)"
        print("stage: detect ledger schema")
        ledger_schema = _run_fetchall_with_heartbeat(
            con,
            f"DESCRIBE SELECT * FROM {ledgers_from} LIMIT 0",
            stage="ledger-schema",
            heartbeat_sec=args.heartbeat_sec,
        )
        ledger_cols = {str(r[0]) for r in ledger_schema}

        ledger_num_col = _pick_first_present(ledger_cols, ["ledger_number", "number", "height", "ledger"])
        ledger_time_col = _pick_first_present(ledger_cols, ["timestamp", "ledger_timestamp", "time", "datetime"])

        if ledger_num_col and ledger_time_col:
            print("stage: build ledger_times lookup")
            _run_execute_with_heartbeat(
                con,
                f"""
                CREATE TEMP TABLE ledger_times AS
                SELECT
                    cast({ledger_num_col} AS BIGINT) AS ledger_number,
                    cast({ledger_time_col} AS VARCHAR) AS ledger_time_raw
                FROM {ledgers_from}
                WHERE {ledger_num_col} IS NOT NULL
                """,
                stage="build-ledger-times",
                heartbeat_sec=args.heartbeat_sec,
            )
            print("stage: aggregate ledger recoverability by market")
            ledger_rows_raw = _run_fetchall_with_heartbeat(
                con,
                """
                WITH hb AS (
                    SELECT DISTINCT market_slug, ledger_number
                    FROM hits
                    WHERE ledger_number IS NOT NULL
                )
                SELECT
                    hb.market_slug,
                    count(*)::BIGINT AS n_distinct_ledgers,
                    sum(CASE WHEN bt.ledger_time_raw IS NOT NULL AND trim(bt.ledger_time_raw) <> '' THEN 1 ELSE 0 END)::BIGINT AS n_distinct_ledgers_with_time,
                    min(bt.ledger_time_raw) AS min_ledger_time_raw,
                    max(bt.ledger_time_raw) AS max_ledger_time_raw
                FROM hb
                LEFT JOIN ledger_times bt ON hb.ledger_number = bt.ledger_number
                GROUP BY hb.market_slug
                ORDER BY hb.market_slug
                """,
                stage="ledger-recoverability",
                heartbeat_sec=args.heartbeat_sec,
            )
            for row in ledger_rows_raw:
                n_ledgers = int(row[1] or 0)
                n_with = int(row[2] or 0)
                ledger_rows.append(
                    {
                        "market_slug": str(row[0]),
                        "n_distinct_ledgers": n_ledgers,
                        "n_distinct_ledgers_with_time": n_with,
                        "pct_distinct_ledgers_with_time": round((n_with / n_ledgers) * 100.0, 4) if n_ledgers else 0.0,
                        "min_ledger_time_raw": row[3] or "",
                        "max_ledger_time_raw": row[4] or "",
                    }
                )
            g = _run_fetchall_with_heartbeat(
                con,
                """
                WITH hb AS (
                    SELECT DISTINCT ledger_number
                    FROM hits
                    WHERE ledger_number IS NOT NULL
                )
                SELECT
                    count(*)::BIGINT AS n_ledgers,
                    sum(CASE WHEN bt.ledger_time_raw IS NOT NULL AND trim(bt.ledger_time_raw) <> '' THEN 1 ELSE 0 END)::BIGINT AS n_ledgers_with_time
                FROM hb
                LEFT JOIN ledger_times bt ON hb.ledger_number = bt.ledger_number
                """,
                stage="ledger-recoverability-global",
                heartbeat_sec=args.heartbeat_sec,
            )[0]
            g_ledgers = int(g[0] or 0)
            g_with = int(g[1] or 0)
            ledger_summary["ledger_time_col_used"] = ledger_time_col
            ledger_summary["global_distinct_ledgers"] = g_ledgers
            ledger_summary["global_distinct_ledgers_with_time"] = g_with
            ledger_summary["global_pct_distinct_ledgers_with_time"] = (
                round((g_with / g_ledgers) * 100.0, 4) if g_ledgers else 0.0
            )
        else:
            ledger_summary["missing_required_ledger_columns"] = {
                "ledger_number_candidate_found": bool(ledger_num_col),
                "ledger_time_candidate_found": bool(ledger_time_col),
            }
    else:
        ledger_summary["missing_ledgers_dir_or_files"] = True

    # Ensure all selected market slugs are represented in each output.
    by_slug_side = {r["market_slug"]: r for r in side_rows}
    by_slug_label = {r["market_slug"]: r for r in label_rows}
    by_slug_amt = {r["market_slug"]: r for r in amt_rows}
    by_slug_ledger = {r["market_slug"]: r for r in ledger_rows}

    side_out: list[dict[str, object]] = []
    label_out: list[dict[str, object]] = []
    amt_out: list[dict[str, object]] = []
    ledger_out: list[dict[str, object]] = []
    for slug in market_slugs:
        side_out.append(
            by_slug_side.get(
                slug,
                {
                    "market_slug": slug,
                    "n_trade_rows": 0,
                    "n_maker_token_only": 0,
                    "n_taker_token_only": 0,
                    "n_both_token": 0,
                    "n_neither_token": 0,
                    "n_side_classifiable_rows": 0,
                    "pct_side_classifiable_rows": 0.0,
                },
            )
        )
        label_out.append(
            by_slug_label.get(
                slug,
                {
                    "market_slug": slug,
                    "n_tokens": 0,
                    "n_tokens_with_outcome_label": 0,
                    "n_tokens_without_outcome_label": 0,
                    "pct_tokens_with_outcome_label": 0.0,
                    "outcome_label_source_column": outcome_col_used or "",
                },
            )
        )
        amt_out.append(
            by_slug_amt.get(
                slug,
                {
                    "market_slug": slug,
                    "n_trade_rows": 0,
                    "n_maker_non_null": 0,
                    "n_taker_non_null": 0,
                    "n_maker_positive": 0,
                    "n_taker_positive": 0,
                    "n_both_positive": 0,
                    "pct_both_positive": 0.0,
                    "min_maker_amount": "",
                    "max_maker_amount": "",
                    "min_taker_amount": "",
                    "max_taker_amount": "",
                    "p50_maker_amount": "",
                    "p90_maker_amount": "",
                    "p99_maker_amount": "",
                    "p50_taker_amount": "",
                    "p90_taker_amount": "",
                    "p99_taker_amount": "",
                    "p50_ratio_taker_over_maker": "",
                    "p90_ratio_taker_over_maker": "",
                    "pct_maker_multiple_1e6": 0.0,
                    "pct_taker_multiple_1e6": 0.0,
                    "pct_maker_multiple_1e18": 0.0,
                    "pct_taker_multiple_1e18": 0.0,
                },
            )
        )
        ledger_out.append(
            by_slug_ledger.get(
                slug,
                {
                    "market_slug": slug,
                    "n_distinct_ledgers": 0,
                    "n_distinct_ledgers_with_time": 0,
                    "pct_distinct_ledgers_with_time": 0.0,
                    "min_ledger_time_raw": "",
                    "max_ledger_time_raw": "",
                },
            )
        )

    out_side = _OUT_DIR / f"{args.output_prefix}_asset_role_side_inferability.csv"
    out_label = _OUT_DIR / f"{args.output_prefix}_outcome_label_joinability.csv"
    out_amt = _OUT_DIR / f"{args.output_prefix}_amount_decimal_sanity.csv"
    out_ledger = _OUT_DIR / f"{args.output_prefix}_ledger_time_recoverability.csv"
    out_token_detail = _OUT_DIR / f"{args.output_prefix}_market_token_outcome_map.csv"
    out_summary = _OUT_DIR / f"{args.output_prefix}_inference_readiness_summary.json"

    _write_csv(
        out_side,
        side_out,
        [
            "market_slug",
            "n_trade_rows",
            "n_maker_token_only",
            "n_taker_token_only",
            "n_both_token",
            "n_neither_token",
            "n_side_classifiable_rows",
            "pct_side_classifiable_rows",
        ],
    )
    _write_csv(
        out_label,
        label_out,
        [
            "market_slug",
            "n_tokens",
            "n_tokens_with_outcome_label",
            "n_tokens_without_outcome_label",
            "pct_tokens_with_outcome_label",
            "outcome_label_source_column",
        ],
    )
    _write_csv(
        out_amt,
        amt_out,
        [
            "market_slug",
            "n_trade_rows",
            "n_maker_non_null",
            "n_taker_non_null",
            "n_maker_positive",
            "n_taker_positive",
            "n_both_positive",
            "pct_both_positive",
            "min_maker_amount",
            "max_maker_amount",
            "min_taker_amount",
            "max_taker_amount",
            "p50_maker_amount",
            "p90_maker_amount",
            "p99_maker_amount",
            "p50_taker_amount",
            "p90_taker_amount",
            "p99_taker_amount",
            "p50_ratio_taker_over_maker",
            "p90_ratio_taker_over_maker",
            "pct_maker_multiple_1e6",
            "pct_taker_multiple_1e6",
            "pct_maker_multiple_1e18",
            "pct_taker_multiple_1e18",
        ],
    )
    _write_csv(
        out_ledger,
        ledger_out,
        [
            "market_slug",
            "n_distinct_ledgers",
            "n_distinct_ledgers_with_time",
            "pct_distinct_ledgers_with_time",
            "min_ledger_time_raw",
            "max_ledger_time_raw",
        ],
    )

    _write_csv(
        out_token_detail,
        [
            {
                "market_slug": r.market_slug,
                "token_id": r.token_id,
                "outcome_label": r.outcome_label,
                "outcome_label_source_column": r.outcome_label_source,
            }
            for r in token_meta_rows
        ],
        ["market_slug", "token_id", "outcome_label", "outcome_label_source_column"],
    )

    global_trade_rows = int(
        _run_fetchall_with_heartbeat(
            con,
            "SELECT count(*)::BIGINT FROM hits",
            stage="global-trade-row-count",
            heartbeat_sec=args.heartbeat_sec,
        )[0][0]
        or 0
    )
    global_side_classifiable_rows = int(
        _run_fetchall_with_heartbeat(
            con,
            """
            SELECT
                sum(CASE WHEN maker_is_market_token = 1 AND taker_is_market_token = 0 THEN 1 ELSE 0 END)
              + sum(CASE WHEN maker_is_market_token = 0 AND taker_is_market_token = 1 THEN 1 ELSE 0 END)
            FROM hits
            """
            ,
            stage="global-side-classifiable-count",
            heartbeat_sec=args.heartbeat_sec,
        )[0][0]
        or 0
    )
    summary = {
        "input_csv": str(input_csv),
        "data_dir": str(args.data_dir),
        "scan_mode": args.scan_mode,
        "markets_files_scanned": len(market_files),
        "trades_files_scanned": len(trade_files),
        "ledgers_files_scanned": len(ledger_files),
        "input_market_slugs": len(market_slugs),
        "resolved_market_tokens": len(token_meta_rows),
        "outcome_label_source_column": outcome_col_used or "",
        "global_trade_rows_matched": global_trade_rows,
        "global_side_classifiable_rows": global_side_classifiable_rows,
        "global_pct_side_classifiable_rows": (
            round((global_side_classifiable_rows / global_trade_rows) * 100.0, 4) if global_trade_rows else 0.0
        ),
        "global_tokens_with_outcome_label": sum(
            1 for r in token_meta_rows if r.outcome_label.strip()
        ),
        "global_tokens_without_outcome_label": sum(
            1 for r in token_meta_rows if not r.outcome_label.strip()
        ),
        "ledger_time_recoverability": ledger_summary,
        "output_asset_role_side_inferability_csv": str(out_side),
        "output_outcome_label_joinability_csv": str(out_label),
        "output_amount_decimal_sanity_csv": str(out_amt),
        "output_ledger_time_recoverability_csv": str(out_ledger),
        "output_market_token_outcome_map_csv": str(out_token_detail),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"scan_mode:                         {args.scan_mode}")
    print(f"markets_files_scanned:             {len(market_files)}")
    print(f"trades_files_scanned:              {len(trade_files)}")
    print(f"ledgers_files_scanned:              {len(ledger_files)}")
    print(f"input_market_slugs:                {len(market_slugs)}")
    print(f"resolved_market_tokens:            {len(token_meta_rows)}")
    print(f"global_trade_rows_matched:         {global_trade_rows}")
    print(f"global_side_classifiable_rows:     {global_side_classifiable_rows}")
    print(f"global_pct_side_classifiable_rows: {summary['global_pct_side_classifiable_rows']}")
    print(f"ledger_time_col_used:               {ledger_summary.get('ledger_time_col_used', '')}")
    print(f"wrote: {out_side}")
    print(f"wrote: {out_label}")
    print(f"wrote: {out_amt}")
    print(f"wrote: {out_ledger}")
    print(f"wrote: {out_token_detail}")
    print(f"wrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
