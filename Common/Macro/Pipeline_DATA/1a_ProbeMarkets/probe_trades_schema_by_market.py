"""
Probe Parquet trades parquet schema and per-market trade coverage.

Goal:
  - identify the union trade schema available in local trades_*.parquet,
  - map selected market slugs to outcome token IDs (from markets parquet),
  - report whether each market has matching trades, plus time/date ranges.

Input market selection:
  Accepts either:
    - Step 01 market-level CSV (expects `slug`), or
    - Step 01 event-level CSV (expects `outcome_market_slugs`, ';' separated).

Usage (repo root):
  python scripts/util/Step_03_ProbeTrades/probe_trades_schema_by_market.py
  python scripts/util/Step_03_ProbeTrades/probe_trades_schema_by_market.py --input-csv ../out_Markets/bak_v8/1EXTRACT_fed_parquet_events.csv
  python scripts/util/Step_03_ProbeTrades/probe_trades_schema_by_market.py --trade-sample-files 3

  ---
    Quick test command

    ```bash
    python scripts/util/Step_03_ProbeTrades/probe_trades_schema_by_market.py --data-dir data --input-csv bak_v8/1EXTRACT_fed_parquet_events.csv --heartbeat-sec 2
    ```

    For faster debugging runs:
    - add `--trade-sample-files 1 --market-sample-files 1` first, then scale up.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import duckdb

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.util.trade_probe_exports import export_market_token_map_and_trade_coverage

_UTIL_DIR = Path(__file__).resolve().parents[1]
_MARKETS_OUT_DIR = _UTIL_DIR / "out_Markets"
_OUT_DIR = _UTIL_DIR / "out_Trades" / "SCHEMA"
_STEP01_PREFIX = "1EXTRACT"
_STEP03_PREFIX = "3PROBE"
_DEFAULT_INPUT = _MARKETS_OUT_DIR / f"{_STEP01_PREFIX}_fed_parquet_events.csv"


def _glob_sql(path: Path) -> str:
    return str(path).replace("\\", "/")


def _csv_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    candidate_trades = _OUT_DIR / p
    if candidate_trades.exists():
        return candidate_trades
    return _MARKETS_OUT_DIR / p


def _resolve_data_dirs(root: Path) -> tuple[Path, Path]:
    """
    Support both layouts:
      - data/parquet/{markets,trades}
      - data/{parquet/{markets,trades}}
    """
    m_dir = root / "markets"
    t_dir = root / "trades"
    if m_dir.is_dir() and t_dir.is_dir():
        return m_dir, t_dir

    alt_m = root / "parquet" / "markets"
    alt_t = root / "parquet" / "trades"
    if alt_m.is_dir() and alt_t.is_dir():
        return alt_m, alt_t

    return m_dir, t_dir


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

    # stable unique
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


def _extract_token_ids(raw: object) -> list[str]:
    """
    Parse token ids from clob_token_ids variants seen in historical markets shards.

    Expected canonical shape is a JSON array of strings, but some rows can be
    malformed or use object-like wrappers. Return deduped ids in stable order.
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s in {"[]", "null", "None"}:
        return []
    try:
        data = json.loads(s)
    except Exception:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def _add(v: object) -> None:
        t = str(v).strip()
        if not t or t in seen:
            return
        seen.add(t)
        out.append(t)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                _add(item)
            elif isinstance(item, dict):
                # Some dumps may nest ids under common keys.
                for k in ("token_id", "tokenId", "id"):
                    if k in item:
                        _add(item[k])
    elif isinstance(data, dict):
        # Handle object form like {"yes": "...", "no": "..."}.
        for v in data.values():
            if isinstance(v, str):
                _add(v)
            elif isinstance(v, dict):
                for k in ("token_id", "tokenId", "id"):
                    if k in v:
                        _add(v[k])
    return out


def _run_fetchall_with_heartbeat(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    stage: str,
    heartbeat_sec: float,
) -> list[tuple]:
    """
    Execute query and print a single-line heartbeat while it runs.
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-csv",
        type=Path,
        default=_DEFAULT_INPUT,
        help=(
            "Input CSV. Default is scripts/util/out_Markets/1EXTRACT_fed_parquet_events.csv; "
            "relative paths are resolved under out_Trades first, then out_Markets."
        ),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/parquet"),
        help="Parquet root containing markets/ and trades/ (default: data/parquet).",
    )
    p.add_argument(
        "--trade-sample-files",
        type=int,
        default=None,
        metavar="N",
        help="Only read first N trades_*.parquet files (alphabetical).",
    )
    p.add_argument(
        "--market-sample-files",
        type=int,
        default=None,
        metavar="N",
        help="Only read first N markets_*.parquet files (alphabetical).",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default=_STEP03_PREFIX,
        help=f"Output filename prefix under scripts/util/out_Trades/SCHEMA (default: {_STEP03_PREFIX}).",
    )
    p.add_argument(
        "--heartbeat-sec",
        type=float,
        default=5.0,
        help="Heartbeat cadence in seconds while long DuckDB queries run (0 disables).",
    )
    args = p.parse_args(argv)

    try:
        summary = export_market_token_map_and_trade_coverage(
            input_csv=args.input_csv,
            data_dir=args.data_dir,
            output_prefix=args.output_prefix,
            trade_sample_files=args.trade_sample_files,
            market_sample_files=args.market_sample_files,
            heartbeat_sec=args.heartbeat_sec,
        )
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    print("stage:                     write outputs")
    print(f"trade_schema_columns:      {summary['trade_schema_columns']}")
    print(f"resolved_market_tokens:    {summary['resolved_market_tokens']}")
    print(f"markets_with_trade_rows:   {summary['markets_with_any_trade_rows']}")
    print(
        "time_columns_present:      "
        f"{summary['time_columns_present'] if summary['time_columns_present'] else '(none)'}"
    )
    print(f"ledger_number_present:      {summary['ledger_number_present']}")
    print(f"wrote: {summary['output_trade_schema_csv']}")
    print(f"wrote: {summary['output_market_token_map_csv']}")
    print(f"wrote: {summary['output_market_trade_coverage_csv']}")
    print(f"wrote: {summary['output_summary_json']}")
    return 0

    input_csv = _csv_path(args.input_csv)
    if not input_csv.is_file():
        print(f"Input CSV not found: {input_csv}", file=sys.stderr)
        return 1

    market_slugs = _read_market_slugs(input_csv)
    if not market_slugs:
        print("No market slugs found in input CSV (need `slug` or `outcome_market_slugs`).", file=sys.stderr)
        return 2

    m_dir, t_dir = _resolve_data_dirs(args.data_dir)
    if not m_dir.is_dir():
        print(f"No markets directory: {m_dir.resolve()}", file=sys.stderr)
        return 3
    if not t_dir.is_dir():
        print(f"No trades directory: {t_dir.resolve()}", file=sys.stderr)
        return 4

    market_files = _first_files(m_dir, "markets_*.parquet", args.market_sample_files)
    trade_files = _first_files(t_dir, "trades_*.parquet", args.trade_sample_files)
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
    print(f"input_csv:                 {input_csv}")
    print(f"market_slugs_in_input:     {len(market_slugs)}")
    print(f"markets_files_scanned:     {len(market_files)}")
    print(f"trades_files_scanned:      {len(trade_files)}")
    print("stage:                     detect trade schema")

    # 1) Trades schema (union_by_name across selected shard set)
    schema_rows = con.execute(f"DESCRIBE SELECT * FROM {trades_from} LIMIT 0").fetchall()
    trade_cols = {str(r[0]) for r in schema_rows}
    schema_out_rows = [
        {
            "column_name": str(r[0]),
            "column_type": str(r[1]),
            "nullability": str(r[2]),
            "key": str(r[3]),
            "default": str(r[4]),
            "extra": str(r[5]),
        }
        for r in schema_rows
    ]

    # 2) Map market slug -> outcome token IDs (from markets parquet clob_token_ids JSON)
    if "slug" not in con.execute(f"DESCRIBE SELECT * FROM {markets_from} LIMIT 0").fetchnumpy()["column_name"]:
        print("Markets parquet schema missing `slug`; cannot map to input slugs.", file=sys.stderr)
        return 7

    if "clob_token_ids" not in con.execute(f"DESCRIBE SELECT * FROM {markets_from} LIMIT 0").fetchnumpy()["column_name"]:
        print("Markets parquet schema missing `clob_token_ids`; cannot map markets to trades.", file=sys.stderr)
        return 8

    slug_values = _sql_values_str(market_slugs)
    print("stage:                     map markets to token IDs")
    token_rows = _run_fetchall_with_heartbeat(
        con,
        f"""
        WITH wanted(slug) AS (
            VALUES {slug_values}
        ),
        matched AS (
            SELECT
                m.slug,
                m.clob_token_ids
            FROM {markets_from} m
            JOIN wanted w ON m.slug = w.slug
            WHERE coalesce(m.clob_token_ids, '') NOT IN ('', '[]')
        )
        SELECT slug AS market_slug, clob_token_ids
        FROM matched
        """,
        stage="token-map",
        heartbeat_sec=args.heartbeat_sec,
    )

    token_map_rows: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for slug, clob_raw in token_rows:
        s = str(slug).strip()
        if not s:
            continue
        for tid in _extract_token_ids(clob_raw):
            key = (s, tid)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            token_map_rows.append({"market_slug": s, "token_id": tid})
    if not token_map_rows:
        print("No token IDs resolved from input market slugs (check input CSV/data scope).", file=sys.stderr)
        return 9

    # Build temporary table for joins.
    pair_values = ", ".join(
        f"({_sql_quote(r['market_slug'])}, {_sql_quote(r['token_id'])})"
        for r in token_map_rows
    )
    con.execute(
        f"""
        CREATE TEMP TABLE market_tokens AS
        SELECT * FROM (VALUES {pair_values}) AS t(market_slug, token_id)
        """
    )

    needed_trade_cols = {"maker_asset_id", "taker_asset_id"}
    if not needed_trade_cols.issubset(trade_cols):
        print(
            f"Trades schema missing required token columns {needed_trade_cols}; "
            "cannot compute per-market trade coverage.",
            file=sys.stderr,
        )
        return 10

    # Time candidates commonly seen in parquet trades parquet variants.
    candidate_time_cols = [
        "timestamp",
        "time",
        "created_at",
        "createdAt",
        "matched_at",
        "matchedAt",
        "trade_time",
        "ledger_timestamp",
    ]
    present_time_cols = [c for c in candidate_time_cols if c in trade_cols]
    if not present_time_cols:
        # fallback: report by ledger number only if no explicit timestamp-like column appears
        present_time_cols = []

    ledger_col = "ledger_number" if "ledger_number" in trade_cols else None

    time_min_exprs = ", ".join(f"min(h.{c}) AS min_{c}" for c in present_time_cols)
    time_max_exprs = ", ".join(f"max(h.{c}) AS max_{c}" for c in present_time_cols)
    ledger_exprs = (
        ", min(h.ledger_number) AS min_ledger_number, max(h.ledger_number) AS max_ledger_number" if ledger_col else ""
    )
    extra_exprs = ", ".join(x for x in [time_min_exprs, time_max_exprs] if x)
    if extra_exprs:
        extra_exprs = ", " + extra_exprs

    # Only project columns needed for aggregate; avoids carrying full trade rows (`tr.*`).
    time_hits_expr = ", ".join(f"tr.{c}" for c in present_time_cols)
    ledger_hits_expr = "tr.ledger_number" if ledger_col else ""
    hits_extra_cols = ", ".join(x for x in [time_hits_expr, ledger_hits_expr] if x)
    if hits_extra_cols:
        hits_extra_cols = ", " + hits_extra_cols

    print("stage:                     aggregate per-market trade coverage")
    per_market_rows_raw = _run_fetchall_with_heartbeat(
        con,
        f"""
        WITH hits AS (
            SELECT
                mt.market_slug
                {hits_extra_cols}
            FROM {trades_from} tr
            JOIN market_tokens mt
              ON cast(tr.maker_asset_id AS VARCHAR) = mt.token_id
              OR cast(tr.taker_asset_id AS VARCHAR) = mt.token_id
        )
        SELECT
            h.market_slug,
            count(*)::BIGINT AS n_trade_rows
            {extra_exprs}
            {ledger_exprs}
        FROM hits h
        GROUP BY h.market_slug
        ORDER BY n_trade_rows DESC, h.market_slug
        """,
        stage="coverage-agg",
        heartbeat_sec=args.heartbeat_sec,
    )

    # Convert to dicts and left-join against requested market slug list.
    all_trade_cols = ["market_slug", "n_trade_rows"]
    all_trade_cols.extend([f"min_{c}" for c in present_time_cols])
    all_trade_cols.extend([f"max_{c}" for c in present_time_cols])
    if ledger_col:
        all_trade_cols.extend(["min_ledger_number", "max_ledger_number"])

    hits_by_slug: dict[str, dict[str, object]] = {}
    for row in per_market_rows_raw:
        d = {all_trade_cols[i]: row[i] for i in range(len(all_trade_cols))}
        hits_by_slug[str(d["market_slug"])] = d

    token_count_by_slug: dict[str, int] = {}
    for r in token_map_rows:
        s = r["market_slug"]
        token_count_by_slug[s] = token_count_by_slug.get(s, 0) + 1

    per_market_rows: list[dict[str, object]] = []
    for slug in market_slugs:
        base = {"market_slug": slug, "n_tokens": token_count_by_slug.get(slug, 0), "n_trade_rows": 0}
        for c in present_time_cols:
            base[f"min_{c}"] = ""
            base[f"max_{c}"] = ""
        if ledger_col:
            base["min_ledger_number"] = ""
            base["max_ledger_number"] = ""

        hit = hits_by_slug.get(slug)
        if hit:
            base["n_trade_rows"] = hit.get("n_trade_rows", 0)
            for c in present_time_cols:
                base[f"min_{c}"] = hit.get(f"min_{c}", "")
                base[f"max_{c}"] = hit.get(f"max_{c}", "")
            if ledger_col:
                base["min_ledger_number"] = hit.get("min_ledger_number", "")
                base["max_ledger_number"] = hit.get("max_ledger_number", "")
        per_market_rows.append(base)

    out_schema = _OUT_DIR / f"{args.output_prefix}_trade_schema.csv"
    out_map = _OUT_DIR / f"{args.output_prefix}_market_token_map.csv"
    out_cov = _OUT_DIR / f"{args.output_prefix}_market_trade_coverage.csv"
    out_summary = _OUT_DIR / f"{args.output_prefix}_trade_probe_summary.json"

    _write_csv(
        out_schema,
        schema_out_rows,
        ["column_name", "column_type", "nullability", "key", "default", "extra"],
    )
    _write_csv(out_map, token_map_rows, ["market_slug", "token_id"])

    cov_headers = ["market_slug", "n_tokens", "n_trade_rows"]
    cov_headers.extend([f"min_{c}" for c in present_time_cols])
    cov_headers.extend([f"max_{c}" for c in present_time_cols])
    if ledger_col:
        cov_headers.extend(["min_ledger_number", "max_ledger_number"])
    _write_csv(out_cov, per_market_rows, cov_headers)

    summary = {
        "input_csv": str(input_csv),
        "data_dir": str(args.data_dir),
        "markets_files_scanned": len(market_files),
        "trades_files_scanned": len(trade_files),
        "input_market_slugs": len(market_slugs),
        "trade_schema_columns": len(schema_out_rows),
        "resolved_market_tokens": len(token_map_rows),
        "markets_with_any_trade_rows": sum(1 for r in per_market_rows if int(r.get("n_trade_rows") or 0) > 0),
        "time_columns_present": present_time_cols,
        "ledger_number_present": bool(ledger_col),
        "output_trade_schema_csv": str(out_schema),
        "output_market_token_map_csv": str(out_map),
        "output_market_trade_coverage_csv": str(out_cov),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("stage:                     write outputs")
    print(f"trade_schema_columns:      {len(schema_out_rows)}")
    print(f"resolved_market_tokens:    {len(token_map_rows)}")
    print(f"markets_with_trade_rows:   {summary['markets_with_any_trade_rows']}")
    print(f"time_columns_present:      {present_time_cols if present_time_cols else '(none)'}")
    print(f"ledger_number_present:      {bool(ledger_col)}")
    print(f"wrote: {out_schema}")
    print(f"wrote: {out_map}")
    print(f"wrote: {out_cov}")
    print(f"wrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

