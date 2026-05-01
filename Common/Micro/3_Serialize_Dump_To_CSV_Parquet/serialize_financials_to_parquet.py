from __future__ import annotations

"""
Annual fundamentals: HTML → normalized financials (DuckDB + one Parquet per ticker).

Reuses parse_html_tables(), match_rows_to_schema(), and expand_with_change_rows() from
serialize_financials_to_csv.py. Replaces CSV grid export with long-form rows written to
DuckDB (schema_rows + financials) and one Parquet per ticker.

DESIGN FRAME
------------
Every run is a full refresh. The DuckDB is dropped and recreated from scratch on every
execution. There is no incremental/append logic. To add new tickers, add their HTML folder
to INPUT_DIRS and rerun.

INPUT_DIRS accepts a single path or a list of paths. If the same ticker appears in more
than one folder the script aborts before writing anything.

Edit INPUT_DIRS, OUTPUT_DIR, TIMEFRAME_MODE, TARGET_TICKERS at the top of this file.
"""

"""
THIS VERSION IS WRITTEN BY CLAUDE AND NOT COMPOSER
// the 'fixes' for incremental add is implementing more bugs, ignore
// NEW FRAME: assume every run is always a new refresh. to compensate, allow multiple folder inputs (and just check if same ticker exists across 2 folders)
    - assume broken and always refresh -> No manual deletion needed, script deletes DB itself
    - search "just run the script. No manual DB deletion needed" in claude chat history (D:\WS\-GH-A-Ref\REF-Study\GC_ASMT\Project\REF_SELF\IRS\Working\Data\Xform_FeatureEngineer\ForCompany\260428_0417_ask_claude_plan_CURSORWASTETIMEONLY.txt)
---
WRITES:
    - bugged/wrote wrongly on 260428 0549am need to rerun this again.
    - bugged/wrote wrongly on 260428 1509pm need to rerun this again. See: IO\\out\\_annual_warehouse\\out_log.txt
"""

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

import serialize_financials_to_csv as _sfc
from annual_schema_structured_indented import ensure_annual_schema_structured_indented
from serialize_financials_to_csv import (
    ASYNC_CONCURRENCY,
    MISSING,
    _filter_earnings_periods,
    _norm_key,
    discover_html_paths,
    expand_with_change_rows,
    extract_ticker,
    infer_schema_key,
    infer_timeframe_bucket,
    match_rows_to_schema,
    parse_html_tables,
    should_serialize_path,
)

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these before running
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SERIALIZER_ROOT = SCRIPT_DIR.parent

# Single path or list of paths. All are scanned; same ticker in two folders = abort.
INPUT_DIRS: list[str | Path] | str | Path = [
    r"D:\ProgramData\.git\DATA_TradingView\Ver_00_EarningsGlitched\00_REIT_OPTIONS",
    r"D:\ProgramData\.git\DATA_TradingView\Ver_00_EarningsGlitched\01_REIT_DISTRESSED",
]
OUTPUT_DIR: str | Path = SERIALIZER_ROOT / "IO" / "out"
TIMEFRAME_MODE: str = "Annual"
TARGET_TICKERS: list[str] = []  # e.g. ["A17U", "BTOU"]; [] = all tickers found

ANNUAL_SCHEMA_SOURCE = SERIALIZER_ROOT / "SCHEMA_DIFFERENCES" / "Tradingview_Schema_Annual_FromSS.txt"
ANNUAL_SCHEMA_INDENTED_JSON = SERIALIZER_ROOT / "SCHEMA_DIFFERENCES" / "AnnualSchema_Structured_Indented.json"

# ---------------------------------------------------------------------------
# DERIVED PATHS — do not edit
# ---------------------------------------------------------------------------
_WAREHOUSE_DIR = Path(OUTPUT_DIR) / "_annual_warehouse"
_DUCKDB_PATH = _WAREHOUSE_DIR / "fundamentals.duckdb"
_PARQUET_DIR = _WAREHOUSE_DIR / "parquet"


# ---------------------------------------------------------------------------
# INPUT DISCOVERY
# ---------------------------------------------------------------------------

def _coerce_input_dirs(value: Any) -> list[Path]:
    if isinstance(value, (list, tuple)):
        paths = [Path(v).resolve() for v in value]
    else:
        paths = [Path(value).resolve()]
    if not paths:
        raise ValueError("INPUT_DIRS cannot be empty")
    return paths


def _discover_all_paths(input_dirs: list[Path]) -> list[Path]:
    """
    Collect all matching HTML paths across all input dirs.
    Raises ValueError if the same ticker appears in more than one dir —
    this would cause duplicate rows in financials.
    """
    paths_by_dir: dict[Path, list[Path]] = {}
    for d in input_dirs:
        paths_by_dir[d] = [p for p in discover_html_paths(d) if should_serialize_path(p)]

    # Check for same ticker across multiple dirs
    ticker_to_dirs: dict[str, list[Path]] = defaultdict(list)
    for d, paths in paths_by_dir.items():
        tickers_in_dir = {extract_ticker(p) for p in paths if extract_ticker(p)}
        for t in tickers_in_dir:
            ticker_to_dirs[t].append(d)

    duplicates = {t: dirs for t, dirs in ticker_to_dirs.items() if len(dirs) > 1}
    if duplicates:
        lines = ["Same ticker found in multiple INPUT_DIRS — resolve before running:"]
        for ticker, dirs in sorted(duplicates.items()):
            lines.append(f"  {ticker}: " + ", ".join(str(d) for d in dirs))
        raise ValueError("\n".join(lines))

    # Flatten, sorted for determinism
    all_paths: list[Path] = []
    for d in sorted(paths_by_dir.keys(), key=str):
        all_paths.extend(paths_by_dir[d])
    return all_paths


# ---------------------------------------------------------------------------
# SCHEMA HELPERS
# ---------------------------------------------------------------------------

def _load_annual_bundle() -> dict[str, Any]:
    from serialize_financials_to_csv import ANNUAL_SCHEMA_STRUCTURED, ensure_structured_schema
    return ensure_structured_schema(ANNUAL_SCHEMA_SOURCE, ANNUAL_SCHEMA_STRUCTURED, "Annual")


def _statement_row_ids(indented: dict[str, Any], section: str) -> list[int]:
    return [r["row_id"] for r in indented["schema_rows"] if r["section"] == section]


def _build_group_row_id_maps(
    indented: dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Map statistics/earnings output_label -> ordered row_ids."""
    stat: dict[str, list[int]] = defaultdict(list)
    ear: dict[str, list[int]] = defaultdict(list)
    for r in indented["schema_rows"]:
        gol = r.get("group_output_label")
        if not gol:
            continue
        if r["section"] == "statistics":
            stat[gol].append(r["row_id"])
        elif r["section"] == "earnings":
            ear[gol].append(r["row_id"])
    return dict(stat), dict(ear)


def _row_ids_for_bundle_group(
    group: dict[str, Any],
    ids_by_output_label: dict[str, list[int]],
    *,
    kind: str,
) -> list[int]:
    ol = group["output_label"]
    ids = ids_by_output_label.get(ol, [])
    if len(ids) != len(group["rows"]):
        raise RuntimeError(
            f"{kind} group {ol!r}: bundle has {len(group['rows'])} rows "
            f"but indented schema has {len(ids)} row_ids"
        )
    return ids


def _validate_group_maps(
    bundle: dict[str, Any],
    stat_map: dict[str, list[int]],
    ear_map: dict[str, list[int]],
) -> None:
    for g in bundle["statistics"]["groups"]:
        _row_ids_for_bundle_group(g, stat_map, kind="statistics")
    for g in bundle["earnings"]["groups"]:
        _row_ids_for_bundle_group(g, ear_map, kind="earnings")


# ---------------------------------------------------------------------------
# ROW BUILDERS
# ---------------------------------------------------------------------------

def _merged_to_financial_rows(
    ticker: str,
    currency: str,
    periods: list[str],
    merged: list[dict[str, Any]],
    row_ids: list[int],
    *,
    section: str = "",
) -> list[dict[str, Any]]:
    """
    Convert merged schema rows to flat financial rows.
    Raises if merged has more rows than row_ids — indicates a schema/HTML mismatch
    that must be fixed rather than silently producing NULL row_ids.
    """
    if len(merged) > len(row_ids):
        raise RuntimeError(
            f"[{ticker}] section={section!r}: match_rows_to_schema returned {len(merged)} rows "
            f"but schema only has {len(row_ids)} row_ids. "
            f"Extra labels: {[m['label'] for m in merged[len(row_ids):]]}"
        )
    out: list[dict[str, Any]] = []
    for i, m in enumerate(merged):
        rid = row_ids[i]
        for j, per in enumerate(periods):
            val = m["values"][j] if j < len(m["values"]) else MISSING
            out.append({
                "ticker": ticker,
                "period": per,
                "currency": currency,
                "row_id": rid,
                "value": val,
            })
    return out


def _revenue_rows_pending(
    ticker: str,
    currency: str,
    periods: list[str],
    merged: list[dict[str, Any]],
    group_output_label: str,
) -> list[dict[str, Any]]:
    """Revenue rows before row_id assignment — marked with _rev_gol/_rev_label."""
    out: list[dict[str, Any]] = []
    for m in merged:
        label = m.get("schema_label", m["label"])
        for j, per in enumerate(periods):
            val = m["values"][j] if j < len(m["values"]) else MISSING
            out.append({
                "ticker": ticker,
                "period": per,
                "currency": currency,
                "row_id": None,
                "value": val,
                "_rev_gol": group_output_label,
                "_rev_label": label,
            })
    return out


def _assign_revenue_row_ids(
    financial_rows: list[dict[str, Any]],
    indented: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Replace _rev_gol/_rev_label markers with real row_ids.
    Returns (clean_rows, new_schema_rows_for_db).
    Since every run is a full refresh, ids are assigned fresh from max_static_id + 1.
    """
    max_static = max(r["row_id"] for r in indented["schema_rows"])
    next_id = max_static + 1
    key_to_id: dict[tuple[str, str], int] = {}
    new_schema: list[dict[str, Any]] = []

    def id_for(gol: str, label: str) -> int:
        nonlocal next_id
        k = (gol, _norm_key(label))
        if k in key_to_id:
            return key_to_id[k]
        rid = next_id
        next_id += 1
        key_to_id[k] = rid
        new_schema.append({
            "row_id": rid,
            "section": "revenue",
            "label": label,
            "depth": 1,
            "parent_id": None,
            "group_output_label": gol,
        })
        return rid

    out: list[dict[str, Any]] = []
    for r in financial_rows:
        if "_rev_gol" in r:
            rid = id_for(r["_rev_gol"], r["_rev_label"])
            out.append({
                "ticker": r["ticker"],
                "period": r["period"],
                "currency": r["currency"],
                "row_id": rid,
                "value": r["value"],
            })
        else:
            out.append({
                "ticker": r["ticker"],
                "period": r["period"],
                "currency": r["currency"],
                "row_id": r["row_id"],
                "value": r["value"],
            })

    # Safety assertion — no NULLs must reach the DB
    unresolved = [r for r in out if r["row_id"] is None]
    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} rows still have row_id=None after assignment. "
            f"First: ticker={unresolved[0]['ticker']} period={unresolved[0]['period']}"
        )

    return out, new_schema


# ---------------------------------------------------------------------------
# HTML -> FINANCIAL ROWS
# ---------------------------------------------------------------------------

def collect_financial_rows_from_html(
    html_path: Path,
    bundle: dict[str, Any],
    indented: dict[str, Any],
    stat_map: dict[str, list[int]],
    ear_map: dict[str, list[int]],
) -> list[dict[str, Any]]:
    key = infer_schema_key(html_path)
    if (infer_timeframe_bucket(html_path) or "Annual") != "Annual":
        return []

    ticker = extract_ticker(html_path) or "UNKNOWN"
    html = html_path.read_text(encoding="utf-8", errors="replace")
    tables = parse_html_tables(html)
    if not tables or not key:
        return []

    rows_out: list[dict[str, Any]] = []

    if key in {"income", "balance", "cashflow"}:
        currency, periods, parsed = tables[0]
        parsed = expand_with_change_rows(parsed)
        merged = match_rows_to_schema(list(bundle["statements"][key]["rows"]), parsed)
        # Annual schema has no YoY rows — drop any that expand_with_change_rows added
        merged = [m for m in merged if "YoY growth" not in m["label"]]
        rids = _statement_row_ids(indented, key)
        rows_out.extend(_merged_to_financial_rows(ticker, currency, periods, merged, rids, section=key))

    elif key == "statistics":
        currency, periods, parsed = tables[0]
        for group in bundle["statistics"]["groups"]:
            merged = match_rows_to_schema(group["rows"], parsed, include_extras=False)
            rids = _row_ids_for_bundle_group(group, stat_map, kind="statistics")
            rows_out.extend(_merged_to_financial_rows(ticker, currency, periods, merged, rids, section="statistics"))

    elif key == "dividends":
        currency, periods, parsed = tables[0]
        merged = match_rows_to_schema(bundle["dividends"]["table"]["rows"], parsed)
        rids = _statement_row_ids(indented, "dividends")
        rows_out.extend(_merged_to_financial_rows(ticker, currency, periods, merged, rids, section="dividends"))

    elif key == "earnings":
        for idx, group in enumerate(bundle["earnings"]["groups"]):
            if idx >= len(tables):
                break
            currency, periods, parsed = tables[idx]
            periods, parsed = _filter_earnings_periods(periods, parsed)
            if not periods:
                continue
            merged = match_rows_to_schema(group["rows"], parsed)
            rids = _row_ids_for_bundle_group(group, ear_map, kind="earnings")
            rows_out.extend(_merged_to_financial_rows(ticker, currency, periods, merged, rids, section="earnings"))

    elif key == "revenue":
        for idx, group in enumerate(bundle["revenue"]["groups"]):
            if idx >= len(tables):
                break
            currency, periods, parsed = tables[idx]
            merged = [{**r, "schema_label": r["label"]} for r in parsed]
            rows_out.extend(_revenue_rows_pending(ticker, currency, periods, merged, group["output_label"]))

    return rows_out


# ---------------------------------------------------------------------------
# DUCKDB — always full rebuild
# ---------------------------------------------------------------------------

def _write_duckdb(
    duck_path: Path,
    indented: dict[str, Any],
    all_financial: list[dict[str, Any]],
    revenue_schema: list[dict[str, Any]],
) -> None:
    """Drop and recreate the entire DuckDB from scratch. No append, no merge."""
    duck_path.parent.mkdir(parents=True, exist_ok=True)

    # Delete the file outright so there is zero chance of stale data
    if duck_path.exists():
        duck_path.unlink()

    con = duckdb.connect(str(duck_path))
    try:
        con.execute("""
            CREATE TABLE schema_rows (
                row_id      INTEGER PRIMARY KEY,
                section     VARCHAR,
                label       VARCHAR,
                depth       INTEGER,
                parent_id   INTEGER REFERENCES schema_rows(row_id),
                group_output_label VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE financials (
                ticker   VARCHAR,
                period   VARCHAR,
                currency VARCHAR,
                row_id   INTEGER REFERENCES schema_rows(row_id),
                value    VARCHAR
            )
        """)

        # Insert static schema rows
        con.executemany(
            "INSERT INTO schema_rows VALUES (?, ?, ?, ?, ?, ?)",
            [
                (r["row_id"], r["section"], r["label"],
                 r["depth"], r["parent_id"], r.get("group_output_label"))
                for r in indented["schema_rows"]
            ],
        )

        # Insert dynamic revenue schema rows
        if revenue_schema:
            con.executemany(
                "INSERT INTO schema_rows VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (r["row_id"], r["section"], r["label"],
                     r["depth"], r["parent_id"], r.get("group_output_label"))
                    for r in revenue_schema
                ],
            )

        # Insert all financial data
        if all_financial:
            con.executemany(
                "INSERT INTO financials VALUES (?, ?, ?, ?, ?)",
                [
                    (r["ticker"], r["period"], r["currency"], r["row_id"], r["value"])
                    for r in all_financial
                ],
            )
    finally:
        con.close()


# ---------------------------------------------------------------------------
# PARQUET
# ---------------------------------------------------------------------------

def _write_parquet(all_rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not all_rows:
        return []
    df = pd.DataFrame(all_rows)
    df["row_id"] = df["row_id"].astype("Int64")
    written: list[Path] = []
    for ticker, g in df.groupby("ticker"):
        path = out_dir / f"{ticker}_financials.parquet"
        g.to_parquet(path, index=False)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# ASYNC ORCHESTRATION
# ---------------------------------------------------------------------------

async def _collect_one_html(
    html_path: Path,
    bundle: dict[str, Any],
    indented: dict[str, Any],
    stat_map: dict[str, list[int]],
    ear_map: dict[str, list[int]],
    sem: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    async with sem:
        return await asyncio.to_thread(
            collect_financial_rows_from_html,
            html_path, bundle, indented, stat_map, ear_map,
        )


async def run(
    input_dirs: Any = None,
    output_dir: Any = None,
) -> tuple[list[Path], Path]:
    """
    Main entry point. Always performs a full rebuild of DuckDB and all parquet files.

    input_dirs: override INPUT_DIRS (single path or list of paths)
    output_dir: override OUTPUT_DIR
    """
    resolved_dirs = _coerce_input_dirs(input_dirs or INPUT_DIRS)
    wh = Path(output_dir or OUTPUT_DIR) / "_annual_warehouse"
    duck_path = wh / "fundamentals.duckdb"
    pq_dir = wh / "parquet"

    # Push runtime config into the CSV module so discovery/filtering is consistent
    _sfc.INPUT_DIR = resolved_dirs[0]
    _sfc.OUTPUT_DIR = Path(output_dir or OUTPUT_DIR).resolve()
    _sfc.TIMEFRAME_MODE = TIMEFRAME_MODE
    _sfc.TARGET_TICKERS = TARGET_TICKERS

    # Load and validate schema
    indented = ensure_annual_schema_structured_indented(
        ANNUAL_SCHEMA_SOURCE, ANNUAL_SCHEMA_INDENTED_JSON, verify=True,
    )
    bundle = _load_annual_bundle()
    stat_map, ear_map = _build_group_row_id_maps(indented)
    _validate_group_maps(bundle, stat_map, ear_map)

    # Discover HTML paths — aborts if duplicate ticker across dirs
    paths = _discover_all_paths(resolved_dirs)
    if not paths:
        dirs_str = ", ".join(str(d) for d in resolved_dirs)
        print(f"No annual HTML matched under [{dirs_str}] for TARGET_TICKERS={TARGET_TICKERS}")
        return [], duck_path

    # Parse all HTML concurrently
    sem = asyncio.Semaphore(max(1, ASYNC_CONCURRENCY))
    parts = await asyncio.gather(
        *[_collect_one_html(p, bundle, indented, stat_map, ear_map, sem) for p in paths]
    )
    raw_rows = [row for part in parts for row in part]

    # Assign row_ids to revenue rows (always fresh — full rebuild)
    all_financial, revenue_schema = _assign_revenue_row_ids(raw_rows, indented)

    # Write DuckDB (file deleted and recreated from scratch)
    _write_duckdb(duck_path, indented, all_financial, revenue_schema)

    # Write per-ticker parquet
    pq_paths = _write_parquet(all_financial, pq_dir)

    return pq_paths, duck_path


async def main() -> None:
    if TIMEFRAME_MODE != "Annual":
        print("This script is annual-only. Set TIMEFRAME_MODE = 'Annual'.")
        return

    pq_paths, db_path = await run()
    print(f"DuckDB : {db_path}")
    for p in sorted(pq_paths):
        print(f"Parquet: {p}")


if __name__ == "__main__":
    asyncio.run(main())
