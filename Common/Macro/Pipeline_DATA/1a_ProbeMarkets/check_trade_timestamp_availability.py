"""
Check timestamp availability in Parquet trades parquet files.

Goal:
  - verify whether timestamp-like fields are populated on matched trade rows,
  - report per-market and global non-null coverage,
  - report min/max observed values for each detected time column.

Input market selection:
  Accepts either:
    - Step 01 market-level CSV (expects `slug`), or
    - Step 01 event-level CSV (expects `outcome_market_slugs`, ';' separated).

Usage (repo root):
  python scripts/util/Step_03_ProbeTrades/check_trade_timestamp_availability.py
  python scripts/util/Step_03_ProbeTrades/check_trade_timestamp_availability.py --scan-mode sample --trade-sample-files 250
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import duckdb

_UTIL_DIR = Path(__file__).resolve().parents[1]
_MARKETS_OUT_DIR = _UTIL_DIR / "out_Markets"
_OUT_DIR = _UTIL_DIR / "out_Trades" / "TSCHK"
_STEP01_PREFIX = "1EXTRACT"
_DEFAULT_INPUT = _MARKETS_OUT_DIR / f"{_STEP01_PREFIX}_fed_parquet_events.csv"
_DEFAULT_OUT_PREFIX = "3PROBE_TSCHK"


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


def _normalize_ts(v: object) -> int | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    lowered = s.lower()
    if lowered in {"nan", "null", "none"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _collect_raw_timestamp_examples(
    con: duckdb.DuckDBPyConnection,
    present_time_cols: list[str],
    *,
    sample_limit: int = 10,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """
    Collect direct SQL counts and sample values for non-null timestamp columns.

    This intentionally bypasses `_normalize_ts` so we can distinguish:
      - true missingness in the parquet data, vs
      - parseability limits in Python normalization logic.
    """
    non_null_counts: dict[str, int] = {}
    sample_values: dict[str, list[str]] = {}
    for col in present_time_cols:
        count_sql = f"""
            SELECT count(*)::BIGINT
            FROM hits
            WHERE {col} IS NOT NULL
              AND trim(cast({col} AS VARCHAR)) <> ''
        """
        non_null_count = int(con.execute(count_sql).fetchone()[0])
        non_null_counts[col] = non_null_count

        values_sql = f"""
            SELECT DISTINCT cast({col} AS VARCHAR) AS raw_value
            FROM hits
            WHERE {col} IS NOT NULL
              AND trim(cast({col} AS VARCHAR)) <> ''
            LIMIT {int(sample_limit)}
        """
        rows = con.execute(values_sql).fetchall()
        sample_values[col] = [str(r[0]) for r in rows if r and r[0] is not None]
    return non_null_counts, sample_values


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
    p.add_argument("--output-prefix", type=str, default=_DEFAULT_OUT_PREFIX)
    p.add_argument(
        "--raw-ts-sample-limit",
        type=int,
        default=10,
        metavar="N",
        help="Max distinct raw non-null timestamp samples to collect per time column (default: 10).",
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

    m_dir, t_dir = _resolve_data_dirs(args.data_dir)
    if not m_dir.is_dir():
        print(f"No markets directory: {m_dir.resolve()}", file=sys.stderr)
        return 3
    if not t_dir.is_dir():
        print(f"No trades directory: {t_dir.resolve()}", file=sys.stderr)
        return 4

    trade_limit = args.trade_sample_files if args.scan_mode == "sample" else None
    market_files = _first_files(m_dir, "markets_*.parquet", args.market_sample_files)
    trade_files = _first_files(t_dir, "trades_*.parquet", trade_limit)
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
    trade_schema = con.execute(f"DESCRIBE SELECT * FROM {trades_from} LIMIT 0").fetchall()
    trade_cols = {str(r[0]) for r in trade_schema}

    needed_trade_cols = {"maker_asset_id", "taker_asset_id"}
    if not needed_trade_cols.issubset(trade_cols):
        print(
            f"Trades schema missing required token columns {needed_trade_cols}; cannot match trades to markets.",
            file=sys.stderr,
        )
        return 7

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
        print("No timestamp-like columns found in trades schema.", file=sys.stderr)
        return 8

    m_schema = con.execute(f"DESCRIBE SELECT * FROM {markets_from} LIMIT 0").fetchnumpy()["column_name"]
    if "slug" not in m_schema or "clob_token_ids" not in m_schema:
        print("Markets parquet missing `slug` or `clob_token_ids` required for slug/token mapping.", file=sys.stderr)
        return 9

    slug_values = _sql_values_str(market_slugs)
    token_rows = con.execute(
        f"""
        WITH wanted(slug) AS (
            VALUES {slug_values}
        ),
        matched AS (
            SELECT m.slug, m.clob_token_ids
            FROM {markets_from} m
            JOIN wanted w ON m.slug = w.slug
            WHERE coalesce(m.clob_token_ids, '') NOT IN ('', '[]')
              AND left(trim(m.clob_token_ids), 1) = '['
        )
        SELECT slug AS market_slug, unnest(from_json(clob_token_ids, '["VARCHAR"]')) AS token_id
        FROM matched
        """
    ).fetchall()
    token_map_rows = [{"market_slug": str(s), "token_id": str(t)} for s, t in token_rows if t]
    if not token_map_rows:
        print("No token IDs resolved from input market slugs.", file=sys.stderr)
        return 10

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

    time_select = ", ".join(f"tr.{c} AS {c}" for c in present_time_cols)
    con.execute(
        f"""
        CREATE TEMP TABLE hits AS
        SELECT
            mt.market_slug,
            {time_select}
        FROM {trades_from} tr
        JOIN market_tokens mt
          ON cast(tr.maker_asset_id AS VARCHAR) = mt.token_id
          OR cast(tr.taker_asset_id AS VARCHAR) = mt.token_id
        """
    )

    raw_non_null_counts, raw_non_null_samples = _collect_raw_timestamp_examples(
        con,
        present_time_cols,
        sample_limit=max(1, int(args.raw_ts_sample_limit)),
    )

    all_rows = con.execute("SELECT market_slug, * EXCLUDE(market_slug) FROM hits").fetchall()
    col_names = ["market_slug"] + present_time_cols

    stats_by_slug: dict[str, dict[str, object]] = {}
    global_total_rows = 0
    global_rows_with_any_ts = 0
    global_rows_with_all_ts_null = 0

    for row in all_rows:
        rec = {col_names[i]: row[i] for i in range(len(col_names))}
        slug = str(rec["market_slug"])
        if slug not in stats_by_slug:
            stats_by_slug[slug] = {
                "market_slug": slug,
                "n_trade_rows": 0,
                "n_rows_with_any_timestamp": 0,
                "n_rows_all_timestamp_fields_null": 0,
                **{f"min_{c}": None for c in present_time_cols},
                **{f"max_{c}": None for c in present_time_cols},
            }
        dst = stats_by_slug[slug]
        dst["n_trade_rows"] = int(dst["n_trade_rows"]) + 1
        global_total_rows += 1

        row_has_any_ts = False
        for c in present_time_cols:
            val = _normalize_ts(rec.get(c))
            if val is None:
                continue
            row_has_any_ts = True
            lo_k = f"min_{c}"
            hi_k = f"max_{c}"
            if dst[lo_k] is None or val < int(dst[lo_k]):
                dst[lo_k] = val
            if dst[hi_k] is None or val > int(dst[hi_k]):
                dst[hi_k] = val
        if row_has_any_ts:
            dst["n_rows_with_any_timestamp"] = int(dst["n_rows_with_any_timestamp"]) + 1
            global_rows_with_any_ts += 1
        else:
            dst["n_rows_all_timestamp_fields_null"] = int(dst["n_rows_all_timestamp_fields_null"]) + 1
            global_rows_with_all_ts_null += 1

    token_count_by_slug: dict[str, int] = {}
    for r in token_map_rows:
        s = r["market_slug"]
        token_count_by_slug[s] = token_count_by_slug.get(s, 0) + 1

    out_rows: list[dict[str, object]] = []
    for slug in market_slugs:
        hit = stats_by_slug.get(slug)
        row: dict[str, object] = {
            "market_slug": slug,
            "n_tokens": token_count_by_slug.get(slug, 0),
            "n_trade_rows": 0,
            "n_rows_with_any_timestamp": 0,
            "n_rows_all_timestamp_fields_null": 0,
            "pct_rows_with_any_timestamp": 0.0,
        }
        for c in present_time_cols:
            row[f"min_{c}"] = ""
            row[f"max_{c}"] = ""
        if hit:
            row["n_trade_rows"] = int(hit["n_trade_rows"])
            row["n_rows_with_any_timestamp"] = int(hit["n_rows_with_any_timestamp"])
            row["n_rows_all_timestamp_fields_null"] = int(hit["n_rows_all_timestamp_fields_null"])
            if int(hit["n_trade_rows"]) > 0:
                row["pct_rows_with_any_timestamp"] = round(
                    (int(hit["n_rows_with_any_timestamp"]) / int(hit["n_trade_rows"])) * 100.0, 4
                )
            for c in present_time_cols:
                lo = hit.get(f"min_{c}")
                hi = hit.get(f"max_{c}")
                row[f"min_{c}"] = "" if lo is None else lo
                row[f"max_{c}"] = "" if hi is None else hi
        out_rows.append(row)

    out_csv = _OUT_DIR / f"{args.output_prefix}_timestamp_availability.csv"
    out_summary = _OUT_DIR / f"{args.output_prefix}_timestamp_availability_summary.json"

    headers = [
        "market_slug",
        "n_tokens",
        "n_trade_rows",
        "n_rows_with_any_timestamp",
        "n_rows_all_timestamp_fields_null",
        "pct_rows_with_any_timestamp",
    ]
    headers.extend([f"min_{c}" for c in present_time_cols])
    headers.extend([f"max_{c}" for c in present_time_cols])
    _write_csv(out_csv, out_rows, headers)

    summary = {
        "input_csv": str(input_csv),
        "data_dir": str(args.data_dir),
        "scan_mode": args.scan_mode,
        "markets_files_scanned": len(market_files),
        "trades_files_scanned": len(trade_files),
        "input_market_slugs": len(market_slugs),
        "resolved_market_tokens": len(token_map_rows),
        "time_columns_present": present_time_cols,
        "global_trade_rows_matched": global_total_rows,
        "global_rows_with_any_timestamp": global_rows_with_any_ts,
        "global_rows_all_timestamp_fields_null": global_rows_with_all_ts_null,
        "global_pct_rows_with_any_timestamp": (
            round((global_rows_with_any_ts / global_total_rows) * 100.0, 4) if global_total_rows else 0.0
        ),
        "raw_non_null_counts_by_time_col": raw_non_null_counts,
        "raw_non_null_samples_by_time_col": raw_non_null_samples,
        "output_timestamp_availability_csv": str(out_csv),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"scan_mode:                     {args.scan_mode}")
    print(f"markets_files_scanned:         {len(market_files)}")
    print(f"trades_files_scanned:          {len(trade_files)}")
    print(f"time_columns_present:          {present_time_cols}")
    print(f"global_trade_rows_matched:     {global_total_rows}")
    print(f"global_rows_with_any_timestamp:{global_rows_with_any_ts}")
    print(f"global_rows_all_ts_null:       {global_rows_with_all_ts_null}")
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
