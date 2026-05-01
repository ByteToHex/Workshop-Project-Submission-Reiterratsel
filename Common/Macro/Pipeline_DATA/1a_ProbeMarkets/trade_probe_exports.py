"""
Shared trade-probe exports for market-token mapping and trade coverage.
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import duckdb

_UTIL_DIR = Path(__file__).resolve().parent
_MARKETS_OUT_DIR = _UTIL_DIR / "out_Markets"
_OUT_DIR = _UTIL_DIR / "out_Trades" / "SCHEMA"


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


def _extract_token_ids(raw: object) -> list[str]:
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
                for k in ("token_id", "tokenId", "id"):
                    if k in item:
                        _add(item[k])
    elif isinstance(data, dict):
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
) -> list[tuple[Any, ...]]:
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


def export_market_token_map_and_trade_coverage(
    *,
    input_csv: Path,
    data_dir: Path = Path("data/parquet"),
    output_prefix: str = "3PROBE",
    trade_sample_files: int | None = None,
    market_sample_files: int | None = None,
    heartbeat_sec: float = 5.0,
) -> dict[str, Any]:
    resolved_input_csv = _csv_path(input_csv)
    if not resolved_input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {resolved_input_csv}")

    market_slugs = _read_market_slugs(resolved_input_csv)
    if not market_slugs:
        raise ValueError("No market slugs found in input CSV (need `slug` or `outcome_market_slugs`).")

    m_dir, t_dir = _resolve_data_dirs(data_dir)
    if not m_dir.is_dir():
        raise FileNotFoundError(f"No markets directory: {m_dir.resolve()}")
    if not t_dir.is_dir():
        raise FileNotFoundError(f"No trades directory: {t_dir.resolve()}")

    market_files = _first_files(m_dir, "markets_*.parquet", market_sample_files)
    trade_files = _first_files(t_dir, "trades_*.parquet", trade_sample_files)
    if not market_files:
        raise FileNotFoundError(f"No markets_*.parquet under {m_dir.resolve()}")
    if not trade_files:
        raise FileNotFoundError(f"No trades_*.parquet under {t_dir.resolve()}")

    market_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in market_files)
    trade_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in trade_files)
    markets_from = f"read_parquet([{market_file_sql}], union_by_name=true)"
    trades_from = f"read_parquet([{trade_file_sql}], union_by_name=true)"

    con = duckdb.connect(database=":memory:")
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

    market_cols = set(con.execute(f"DESCRIBE SELECT * FROM {markets_from} LIMIT 0").fetchnumpy()["column_name"])
    if "slug" not in market_cols:
        raise RuntimeError("Markets parquet schema missing `slug`; cannot map to input slugs.")
    if "clob_token_ids" not in market_cols:
        raise RuntimeError("Markets parquet schema missing `clob_token_ids`; cannot map markets to trades.")

    slug_values = _sql_values_str(market_slugs)
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
        heartbeat_sec=heartbeat_sec,
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
        raise RuntimeError("No token IDs resolved from input market slugs (check input CSV/data scope).")

    needed_trade_cols = {"maker_asset_id", "taker_asset_id"}
    if not needed_trade_cols.issubset(trade_cols):
        raise RuntimeError(
            f"Trades schema missing required token columns {needed_trade_cols}; cannot compute coverage."
        )

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
    ledger_col = "ledger_number" if "ledger_number" in trade_cols else None

    time_min_exprs = ", ".join(f"min(h.{c}) AS min_{c}" for c in present_time_cols)
    time_max_exprs = ", ".join(f"max(h.{c}) AS max_{c}" for c in present_time_cols)
    ledger_exprs = (
        ", min(h.ledger_number) AS min_ledger_number, max(h.ledger_number) AS max_ledger_number"
        if ledger_col
        else ""
    )
    extra_exprs = ", ".join(x for x in [time_min_exprs, time_max_exprs] if x)
    if extra_exprs:
        extra_exprs = ", " + extra_exprs

    time_hits_expr = ", ".join(f"tr.{c}" for c in present_time_cols)
    ledger_hits_expr = "tr.ledger_number" if ledger_col else ""
    hits_extra_cols = ", ".join(x for x in [time_hits_expr, ledger_hits_expr] if x)
    if hits_extra_cols:
        hits_extra_cols = ", " + hits_extra_cols

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
        heartbeat_sec=heartbeat_sec,
    )

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
        base: dict[str, object] = {"market_slug": slug, "n_tokens": token_count_by_slug.get(slug, 0), "n_trade_rows": 0}
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

    out_schema = _OUT_DIR / f"{output_prefix}_trade_schema.csv"
    out_map = _OUT_DIR / f"{output_prefix}_market_token_map.csv"
    out_cov = _OUT_DIR / f"{output_prefix}_market_trade_coverage.csv"
    out_summary = _OUT_DIR / f"{output_prefix}_trade_probe_summary.json"

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
        "input_csv": str(resolved_input_csv),
        "data_dir": str(data_dir),
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
    summary["output_summary_json"] = str(out_summary)
    return summary
