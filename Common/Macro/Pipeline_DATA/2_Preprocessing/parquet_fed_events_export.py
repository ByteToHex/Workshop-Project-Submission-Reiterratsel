"""
Assume default runner is ``python scripts/util/parquet_fed_events_export.py --mode events --overwrite``;
all requested logic MUST route through this.

Export Parquet **FOMC / Fed funds decision** events (meeting bracket / neg-risk
groups like ``scripts/util/sample/fed-decreases-interest-rates-by-25-bps-after-january-2026-meeting.json``)
from local markets parquets.

The upstream indexer stores one row per *market* (one binary outcome contract). On Gamma,
those rows belong to a parent event (e.g. slug ``fed-decision-in-january``, series
``fed-interest-rates``) with one row per basis-point bracket.

**Tiers (for the default FOMC filter — independent predicates; a row can match more than one):**

  **Tier A — Gamma JSON (strictest metadata).** Requires an ``events_json`` or ``events``
  column. Row matches if ``events[0].seriesSlug`` is ``fed-interest-rates`` **or**
  ``events[0].slug`` starts with ``fed-decision-in-``. If the parquet has no events
  column, Tier A matches no rows. Written to ``1EXTRACT_fed_parquet_events_tier_a.csv``.

  **Tier B — URL slug heuristics.** Matches on ``slug`` only: prefixes such as
  ``fed-decreases-interest-rates-%``, ``fed-increases-interest-rates-%``,
  ``fed-decision-in-%``, or a ``no-change-in-%`` … ``after-%meeting%`` pattern with
  Fed/funds/interest-rate tokens in the slug. Excludes slugs that contain
  ``bank-of-englands``, ``bank-of-japans``, or ``ecb`` (other central banks). Does
  **not** use Gamma JSON or question text. Written to ``1EXTRACT_fed_parquet_events_tier_b.csv``.

  **Tier C — question + description text.** Matches when concatenated lowercased
  ``question`` (and ``description`` if present) satisfy the same meeting/FOMC/Fed+interest
  + bps clause used previously (interest rate(s), bps/basis point, meeting/FOMC, Fed
  context). Does **not** require Tier A or B. Written to ``1EXTRACT_fed_parquet_events_tier_c.csv``.

**Combined export:** ``(Tier A) OR (Tier B) OR (Tier C)`` is also written as
``1EXTRACT_fed_parquet_events.csv`` (or the path given with ``-o``). Use the tier files to see
which mechanism matched; use the combined file as the full union.

This script:

  - **Default filter (non-legacy):** the union of tiers A, B, and C (see above). With
    ``--mode events`` and CSV format, all four outputs are produced when ``--overwrite``
    allows it.
  - Optional ``--legacy-broad-fed-keywords``: previous fed_funds_coverage-style predicate;
    single-file export only; ``--extended-predicate`` only applies in that mode.
  - For ``--mode events``: one row per join group (prefer ``neg_risk_market_id``, else
    parent event slug from JSON, else slug). Outputs parent_event_slug / title,
    ``parent_series_slug`` (``fed-interest-rates`` when present), neg_risk_market_id,
    group_item_titles, outcome slugs when columns exist. Old dumps without
    ``events_json`` / neg-risk ids group less cleanly — re-run the markets indexer.
  - For ``--mode markets``: every matching **combined-predicate** market row (all columns)
    is written to a single file (tiers A/B/C split is events-mode only).

  Default output is CSV (and optional Markdown for --mode events), not Parquet, so you
  can open and inspect FOMC-style events in an editor or spreadsheet.

Reads only (never written):
  {--data-dir}/markets/markets_*.parquet

  Default --data-dir is data/parquet (repo root). If you still use the legacy
  layout data/parquet/markets under a top-level data/ folder only, pass
  --data-dir data (fallback: also checks data/parquet under that root).

Writes only (hardcoded next to this script):
  scripts/util/out_Markets/   (see _OUT_DIR)

  Default FOMC ``--mode events`` CSV run (non-legacy) writes **four** files:
  ``1EXTRACT_fed_parquet_events_tier_a.csv``, ``1EXTRACT_fed_parquet_events_tier_b.csv``,
  ``1EXTRACT_fed_parquet_events_tier_c.csv``, and ``1EXTRACT_fed_parquet_events.csv`` (combined;
  override name with ``-o`` for the combined file only).

  Output paths are resolved strictly under that folder; the data tree must never be
  used as a write destination.

Usage (from repo root):

  VS Code / Cursor: open Run and Debug (Ctrl+Shift+D), choose “Parquet Fed events
  export”, then Start (F5) — same as ``--mode events --overwrite``. Or Command Palette
  “Tasks: Run Task” → “Parquet: fed events export …”.

  # Count only (fast check; uses data/parquet by default)
  python scripts/util/parquet_fed_events_export.py --dry-run

  # All shards (40GB+ can take a long time; start with a shard cap while iterating)
  python scripts/util/parquet_fed_events_export.py --mode events

  python scripts/util/parquet_fed_events_export.py --mode markets -o 1EXTRACT_fed_parquet_markets.csv

  # First 3 parquet files only (alphabetical)
  python scripts/util/parquet_fed_events_export.py --mode markets \\
      --market-sample-files 3 -o sample_markets.csv

  # Markdown table (events mode only; easy to skim)
  python scripts/util/parquet_fed_events_export.py --mode events --format md -o fed_events.md --overwrite

  # Columnar binary (optional, for pipelines — not for manual inspection)
  python scripts/util/parquet_fed_events_export.py --mode events --format parquet -o fed_events.parquet --overwrite

  # Custom parquet tree (markets/ must exist under this path)
  python scripts/util/parquet_fed_events_export.py --data-dir D:/path/to/parquet --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import duckdb

_UTIL_DIR = Path(__file__).resolve().parent
# All exports must land under scripts/util/out_Markets; never write under --data-dir or data/parquet.
_OUT_DIR = _UTIL_DIR.parent / "out_Markets"
_STEP01_PREFIX = "1EXTRACT"
_DEFAULT_EVENTS_CSV = f"{_STEP01_PREFIX}_fed_parquet_events.csv"
_DEFAULT_MARKETS_CSV = f"{_STEP01_PREFIX}_fed_parquet_markets.csv"
_TIER_A_CSV = f"{_STEP01_PREFIX}_fed_parquet_events_tier_a.csv"
_TIER_B_CSV = f"{_STEP01_PREFIX}_fed_parquet_events_tier_b.csv"
_TIER_C_CSV = f"{_STEP01_PREFIX}_fed_parquet_events_tier_c.csv"

if str(_UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(_UTIL_DIR))
_UTIL_ROOT = _UTIL_DIR.parent
if str(_UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(_UTIL_ROOT))


def _resolved_output_path(user: Path) -> Path:
    """Resolve to a path strictly inside _OUT_DIR. Rejects .. escapes and paths outside out/."""
    if user == Path() or str(user).strip() == "":
        raise ValueError("Output path is empty.")
    base = _OUT_DIR.resolve()
    candidate = (user if user.is_absolute() else (_OUT_DIR / user)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as e:
        raise ValueError(
            f"Output must be inside {_OUT_DIR} (got {candidate}). "
            "Use a relative name like fed_events.csv or subdir/out.md."
        ) from e
    return candidate


def _next_backup_dir(base: Path) -> Path:
    """
    Compute the next backup directory under ``base`` as ``bak_vN``.
    Existing backup folders are scanned and the next integer suffix is used.
    """
    pat = re.compile(r"^bak_v(\d+)$")
    max_n = -1
    for child in base.iterdir():
        if not child.is_dir():
            continue
        m = pat.match(child.name)
        if not m:
            continue
        max_n = max(max_n, int(m.group(1)))
    return base / f"bak_v{max_n + 1}"


def _archive_loose_root_files(base: Path) -> tuple[Path | None, int]:
    """
    Move loose files directly under ``base`` into a freshly numbered ``bak_vN`` folder.
    Returns (backup_dir, moved_count). If no loose files are present, returns (None, 0).
    """
    loose_files = sorted([p for p in base.iterdir() if p.is_file()], key=lambda p: p.name)
    if not loose_files:
        return None, 0

    bak_dir = _next_backup_dir(base)
    bak_dir.mkdir(parents=True, exist_ok=False)
    for src in loose_files:
        shutil.move(str(src), str(bak_dir / src.name))
    return bak_dir, len(loose_files)

from csv_sorting import sort_csv_by_end_date  # noqa: E402
from fed_filters import (  # noqa: E402
    _fed_predicate_extended,
    _glob_sql,
    _parquet_fed_predicate,
    _parquet_fomc_decision_predicate_sql,
    _tier_a_predicate_sql,
    _tier_b_predicate_sql,
    _tier_c_predicate_sql,
)


def _resolve_predicate_sql(cols: set[str], legacy_broad: bool, extended: bool) -> str:
    if legacy_broad:
        return _fed_predicate_extended() if extended else _parquet_fed_predicate()
    return _parquet_fomc_decision_predicate_sql(cols)


def _markets_from_clause(markets_glob: str, markets_dir: Path, sample_files: int | None) -> tuple[str, str]:
    """DuckDB FROM clause and human label."""
    if sample_files is None or sample_files <= 0:
        return f"read_parquet('{markets_glob}', union_by_name=true)", "all markets parquet shards"
    files = sorted(markets_dir.glob("markets_*.parquet"))[:sample_files]
    if not files:
        return f"read_parquet('{markets_glob}', union_by_name=true)", "all markets parquet shards"
    mg = ", ".join(f"'{_glob_sql(f)}'" for f in files)
    return f"read_parquet([{mg}], union_by_name=true)", f"first {len(files)} market shard file(s)"


def _column_names(con: duckdb.DuckDBPyConnection, from_clause: str) -> set[str]:
    cur = con.execute(f"SELECT * FROM {from_clause} LIMIT 1")
    return {d[0] for d in cur.description}


def _events_json_column(cols: set[str]) -> str | None:
    """Column holding Gamma `events` JSON array (see events_json in indexer, or `events` in raw dumps)."""
    for name in ("events_json", "events"):
        if name in cols:
            return name
    return None


def _parent_series_slug_sql(cols: set[str]) -> str:
    """Gamma ``events[0].seriesSlug`` (e.g. ``fed-interest-rates``)."""
    ev_c = _events_json_column(cols)
    if not ev_c:
        return "CAST(NULL AS VARCHAR)"
    return (
        "nullif(trim(try("
        f"json_extract_string(try_cast({ev_c} AS JSON), '$[0].seriesSlug')"
        ")), '')"
    )


def _join_diagnostics(cols: set[str]) -> str:
    """One-line summary: whether parquet has fields needed for Gamma-style parent-event joins."""
    bits = [
        f"neg_risk_market_id={'yes' if 'neg_risk_market_id' in cols else 'no'}",
        f"negRiskMarketID={'yes' if 'negRiskMarketID' in cols else 'no'}",
        f"events_json={'yes' if 'events_json' in cols else 'no'}",
        f"events={'yes' if 'events' in cols else 'no'}",
        f"group_item_title={'yes' if 'group_item_title' in cols else 'no'}",
        f"groupItemTitle={'yes' if 'groupItemTitle' in cols else 'no'}",
    ]
    return "; ".join(bits)


def _grouping_sql(cols: set[str]) -> tuple[str, str, str, str, str, str]:
    """SQL fragments: group_key, parent_event_slug, parent_event_title, neg_risk_id row, outcome_label, note."""
    ev_c = _events_json_column(cols)

    neg_parts: list[str] = []
    if "neg_risk_market_id" in cols:
        neg_parts.append("nullif(trim(cast(neg_risk_market_id as varchar)), '')")
    if "negRiskMarketID" in cols:
        neg_parts.append('nullif(trim(cast("negRiskMarketID" as varchar)), '')')

    if ev_c:
        pe_slug = (
            "nullif(trim(try("
            f"json_extract_string(try_cast({ev_c} AS JSON), '$[0].slug')"
            ")), '')"
        )
        pe_title = (
            "nullif(trim(try("
            f"json_extract_string(try_cast({ev_c} AS JSON), '$[0].title')"
            ")), '')"
        )
    else:
        pe_slug = "CAST(NULL AS VARCHAR)"
        pe_title = "CAST(NULL AS VARCHAR)"

    if len(neg_parts) == 0:
        neg_id_row = "CAST(NULL AS VARCHAR)"
    elif len(neg_parts) == 1:
        neg_id_row = neg_parts[0]
    else:
        neg_id_row = "coalesce(" + ", ".join(neg_parts) + ")"

    git = "CAST(NULL AS VARCHAR)"
    if "group_item_title" in cols:
        git = "cast(group_item_title as varchar)"
    elif "groupItemTitle" in cols:
        git = 'cast("groupItemTitle" as varchar)'

    # group_key: same coalesce order as before — parent slug from JSON groups outcomes like Gamma
    gk_parts: list[str] = list(neg_parts)
    if ev_c:
        gk_parts.append(
            "nullif(trim(try("
            f"json_extract_string(try_cast({ev_c} AS JSON), '$[0].slug')"
            ")), '')"
        )
    if "event_slug" in cols:
        gk_parts.append("nullif(trim(cast(event_slug as varchar)), '')")
    if "eventSlug" in cols:
        gk_parts.append('nullif(trim(cast("eventSlug" as varchar)), '')')

    fallback = (
        "coalesce(nullif(trim(cast(slug as varchar)), ''), "
        "nullif(trim(cast(id as varchar)), ''), cast(condition_id as varchar))"
    )
    if not gk_parts:
        group_key = fallback
        note = "group_key = market slug/id only (re-index markets to add events_json + neg_risk_market_id)"
    else:
        group_key = "coalesce(" + ", ".join(gk_parts) + ", " + fallback + ")"
        note = (
            "group_key = coalesce(neg_risk, parent_event_slug from JSON, event_slug cols, slug); "
            "re-run parquet markets backfill to persist Gamma join fields"
        )

    return group_key, pe_slug, pe_title, neg_id_row, git, note


def _md_cell(val: object) -> str:
    if val is None:
        return ""
    return str(val).replace("|", "\\|").replace("\n", " ")


def _events_aggregate_schema_columns_sql(cols: set[str]) -> str:
    """
    Extra SELECT expressions for docs/SCHEMAS.md Parquet Markets columns (aggregated per event_key).
    Uses arg_max(..., COALESCE(volume, 0)) so representative rows are stable when volume is null.
    """
    vol_key = "COALESCE(volume, 0.0)"

    def _arg_max_varchar(col: str) -> str:
        return f"arg_max(cast({col} AS VARCHAR), {vol_key})"

    lines: list[str] = []

    if "id" in cols:
        lines.append(f"{_arg_max_varchar('id')} AS id")
    else:
        lines.append("CAST(NULL AS VARCHAR) AS id")

    if "condition_id" in cols:
        lines.append(f"{_arg_max_varchar('condition_id')} AS condition_id")
    else:
        lines.append("CAST(NULL AS VARCHAR) AS condition_id")

    if "question" in cols:
        lines.append(f"{_arg_max_varchar('question')} AS question")
    else:
        lines.append("CAST(NULL AS VARCHAR) AS question")

    if "slug" in cols:
        lines.append(f"arg_max(cast(slug AS VARCHAR), {vol_key}) AS slug")
    else:
        lines.append("CAST(NULL AS VARCHAR) AS slug")

    if "outcomes" in cols:
        lines.append(f"{_arg_max_varchar('outcomes')} AS outcomes")
    else:
        lines.append("CAST(NULL AS VARCHAR) AS outcomes")

    if "outcome_prices" in cols:
        lines.append(f"{_arg_max_varchar('outcome_prices')} AS outcome_prices")
    else:
        lines.append("CAST(NULL AS VARCHAR) AS outcome_prices")

    if "volume" in cols:
        lines.append("sum(volume)::DOUBLE AS volume")
    else:
        lines.append("CAST(NULL AS DOUBLE) AS volume")

    if "liquidity" in cols:
        lines.append("sum(liquidity)::DOUBLE AS liquidity")
    else:
        lines.append("CAST(NULL AS DOUBLE) AS liquidity")

    if "active" in cols:
        lines.append("bool_or(active) AS active")
    else:
        lines.append("CAST(NULL AS BOOLEAN) AS active")

    if "closed" in cols:
        lines.append("bool_and(COALESCE(closed, FALSE)) AS closed")
    else:
        lines.append("CAST(NULL AS BOOLEAN) AS closed")

    if "end_date" in cols:
        lines.append("min(end_date) AS end_date")
    else:
        lines.append("CAST(NULL AS TIMESTAMP) AS end_date")

    if "created_at" in cols:
        lines.append("min(created_at) AS created_at")
    else:
        lines.append("CAST(NULL AS TIMESTAMP) AS created_at")

    fetched_col = "_fetched_at" if "_fetched_at" in cols else ("fetched_at" if "fetched_at" in cols else None)
    if fetched_col:
        lines.append(f"max({fetched_col}) AS _fetched_at")
    else:
        lines.append("CAST(NULL AS TIMESTAMP) AS _fetched_at")

    return ",\n        ".join(lines)


def _make_events_aggregate_sql(
    from_clause: str,
    pred: str,
    cols: set[str],
    *,
    group_key_sql: str,
    pe_slug_sql: str,
    pe_title_sql: str,
    neg_id_sql: str,
    series_slug_sql: str,
    git_sql: str,
) -> str:
    """DuckDB SQL: aggregated event rows for one WHERE predicate."""
    schema_cols = _events_aggregate_schema_columns_sql(cols)
    return f"""
    WITH fed AS (
        SELECT * FROM {from_clause} m WHERE {pred}
    ),
    keyed AS (
        SELECT
            fed.*,
            ({group_key_sql}) AS _join_group_key,
            ({pe_slug_sql}) AS _parent_event_slug,
            ({pe_title_sql}) AS _parent_event_title,
            ({neg_id_sql}) AS _join_neg_risk_id,
            ({series_slug_sql}) AS _parent_series_slug,
            ({git_sql}) AS _outcome_label
        FROM fed AS fed
    )
    SELECT
        _join_group_key AS event_key,
        any_value(_parent_event_slug) AS parent_event_slug,
        any_value(_parent_event_title) AS parent_event_title,
        any_value(_parent_series_slug) AS parent_series_slug,
        any_value(_join_neg_risk_id) AS neg_risk_market_id,
        string_agg(DISTINCT _outcome_label, ' | ' ORDER BY _outcome_label) AS group_item_titles,
        count(*)::BIGINT AS n_markets,
        count(DISTINCT condition_id)::BIGINT AS n_distinct_conditions,
        sum(volume)::DOUBLE AS sum_volume,
        sum(liquidity)::DOUBLE AS sum_liquidity,
        min(end_date) AS min_end_date,
        max(end_date) AS max_end_date,
        bool_or(active) AS any_active,
        bool_or(NOT closed) AS any_not_closed,
        string_agg(cast(slug as varchar), '; ' ORDER BY volume DESC NULLS LAST) AS outcome_market_slugs,
        arg_max(slug, volume) AS top_volume_slug,
        min(slug) AS slug_min,
        max(slug) AS slug_max,
        {schema_cols}
    FROM keyed
    GROUP BY _join_group_key
    ORDER BY sum_volume DESC NULLS LAST
    """


def _write_markdown_events_table(path: Path, headers: list[str], rows: list[tuple]) -> None:
    """Write a small Markdown table for manual inspection (UTF-8)."""
    hline = "| " + " | ".join(_md_cell(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [
        "# Parquet FOMC / Fed funds decision events (Gamma fed-interest-rates style)",
        "",
        hline,
        sep,
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(v) for v in row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _maybe_sort_primary_out_csv(*, mode: str, fmt: str, out_path: Path) -> None:
    """
    Sort only the default Step 01 combined events CSV after export.
    """
    primary = (_OUT_DIR / _DEFAULT_EVENTS_CSV).resolve()
    if mode != "events" or fmt != "csv" or out_path.resolve() != primary:
        return
    n_sorted, used = sort_csv_by_end_date(out_path)
    bak_name = f"{out_path.stem}-bak{out_path.suffix}"
    print(f"Sorted {out_path.name}: backup -> {bak_name}, sorted {n_sorted} row(s) by {used}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/parquet"),
        help="Parquet data root (default: data/parquet). Reads markets_*.parquet from <dir>/markets/.",
    )
    p.add_argument(
        "--mode",
        choices=("events", "markets"),
        default="events",
        help="events: one aggregated row per event_key; markets: every matching market row",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        help=(
            "Output filename or subpath under scripts/util/out_Markets/ "
            "(combined A|B|C for default FOMC events CSV; tier A/B/C files use fixed names). "
            "Absolute paths are only accepted if they already lie under that folder."
        ),
    )
    p.add_argument(
        "--format",
        choices=("csv", "md", "parquet"),
        default="csv",
        help=(
            "csv: spreadsheet-friendly (default). md: Markdown table (events mode only). "
            "parquet: binary columnar (optional). Suffix .csv/.md/.parquet sets format."
        ),
    )
    p.add_argument(
        "--legacy-broad-fed-keywords",
        action="store_true",
        help=(
            "Use the old broad Fed keyword filter from fed_funds_coverage (fed-rate, fed cut, FOMC text …). "
            "Default is the FOMC decision / fed-interest-rates filter aligned with the bracket sample JSON."
        ),
    )
    p.add_argument(
        "--extended-predicate",
        action="store_true",
        help="Only with --legacy-broad-fed-keywords: add extra OR clauses (powell, interest rate, …).",
    )
    p.add_argument(
        "--market-sample-files",
        type=int,
        default=None,
        metavar="N",
        help="Only read first N markets_*.parquet shards (alphabetical). Omit or use 0 for all.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and event_key strategy only; do not write a file.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    args = p.parse_args(argv)

    root = args.data_dir
    # Primary: <parquet>/markets/ (default data root is data/parquet).
    # Legacy: <repo-data>/parquet/markets/ when --data-dir points at repo data/ only.
    poly_m_dir = root / "markets"
    if not poly_m_dir.is_dir():
        alt = root / "parquet" / "markets"
        if alt.is_dir():
            poly_m_dir = alt
    markets_glob = _glob_sql(poly_m_dir / "markets_*.parquet")

    if not poly_m_dir.is_dir():
        print(
            f"No markets directory at {(root / 'markets').resolve()} "
            f"(or legacy {(root / 'parquet' / 'markets').resolve()})",
            file=sys.stderr,
        )
        return 1
    if not list(poly_m_dir.glob("markets_*.parquet")):
        print(f"No markets_*.parquet under {poly_m_dir.resolve()}", file=sys.stderr)
        return 1

    ms = args.market_sample_files
    if ms is not None and ms <= 0:
        ms = None

    from_clause, scope_label = _markets_from_clause(markets_glob, poly_m_dir, ms)

    con = duckdb.connect(database=":memory:")
    try:
        cols = _column_names(con, from_clause)
    except Exception as e:
        print(f"Could not read markets schema: {e}", file=sys.stderr)
        return 2

    pred = _resolve_predicate_sql(cols, args.legacy_broad_fed_keywords, args.extended_predicate)

    group_key_sql, pe_slug_sql, pe_title_sql, neg_id_sql, git_sql, group_note = _grouping_sql(cols)
    series_slug_sql = _parent_series_slug_sql(cols)

    if args.extended_predicate and not args.legacy_broad_fed_keywords:
        print(
            "Note: --extended-predicate applies only with --legacy-broad-fed-keywords; ignoring.",
            file=sys.stderr,
        )

    pred_label: str
    if args.legacy_broad_fed_keywords:
        pred_label = "legacy broad keywords"
        if args.extended_predicate:
            pred_label += " + extended ORs"
    else:
        pred_label = "fomc-decision (fed-interest-rates, fed-decision-in-*, bracket slugs, meeting text)"

    print(f"data_dir:      {root.resolve()}")
    print(f"scope:         {scope_label}")
    print(f"mode:          {args.mode}")
    print(f"predicate:     {pred_label}")
    print(f"join_columns:  {_join_diagnostics(cols)}")
    print(f"grouping:      {group_note}")
    print(f"columns seen:  {len(cols)} (union across sample; use LIMIT 1 scan)")

    # Row count matching predicate
    try:
        n_match = con.execute(
            f"SELECT count(*)::BIGINT FROM {from_clause} m WHERE {pred}"
        ).fetchone()[0]
    except Exception as e:
        print(f"Count query failed: {e}", file=sys.stderr)
        return 3

    print(f"matched_rows:  {n_match}")

    print(f"write_dir:     {_OUT_DIR}  (only valid write destination)")

    if args.dry_run:
        if args.mode == "events" and n_match:
            try:
                n_ev = con.execute(
                    f"""
                    WITH fed AS (
                        SELECT * FROM {from_clause} m WHERE {pred}
                    ),
                    k AS (
                        SELECT {group_key_sql} AS group_key FROM fed
                    )
                    SELECT count(DISTINCT group_key)::BIGINT FROM k
                    """
                ).fetchone()[0]
                print(f"distinct event_key rows (estimate, combined): {n_ev}")
            except Exception as e:
                print(f"(distinct events count skipped: {e})")
        if args.mode == "events" and not args.legacy_broad_fed_keywords:
            pa, pb, pc = _tier_a_predicate_sql(cols), _tier_b_predicate_sql(cols), _tier_c_predicate_sql(cols)
            for label, px in (("tier_A_gamma_json", pa), ("tier_B_slug", pb), ("tier_C_text", pc)):
                try:
                    nc = con.execute(f"SELECT count(*)::BIGINT FROM {from_clause} m WHERE {px}").fetchone()[0]
                    print(f"matched_rows ({label}): {nc}")
                except Exception as e:
                    print(f"(count {label} skipped: {e})")
        print("Dry run complete (no file written).")
        return 0

    default_name = _DEFAULT_MARKETS_CSV if args.mode == "markets" else _DEFAULT_EVENTS_CSV
    try:
        out = _resolved_output_path(args.output if args.output is not None else Path(default_name))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4

    fmt = args.format
    suf = out.suffix.lower()
    if suf == ".csv":
        fmt = "csv"
    elif suf == ".md":
        fmt = "md"
    elif suf == ".parquet":
        fmt = "parquet"

    tier_csv_bundle = (
        args.mode == "events"
        and fmt == "csv"
        and not args.legacy_broad_fed_keywords
    )
    if tier_csv_bundle:
        out_tier_a = _resolved_output_path(Path(_TIER_A_CSV))
        out_tier_b = _resolved_output_path(Path(_TIER_B_CSV))
        out_tier_c = _resolved_output_path(Path(_TIER_C_CSV))
        paths_to_write = [out_tier_a, out_tier_b, out_tier_c, out]
    else:
        paths_to_write = [out]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    bak_dir, moved_count = _archive_loose_root_files(_OUT_DIR)
    if moved_count:
        print(f"Archived {moved_count} loose file(s) from {_OUT_DIR} -> {bak_dir}")

    for op in paths_to_write:
        if op.exists() and not args.overwrite:
            print(f"Refusing to overwrite {op} (pass --overwrite).", file=sys.stderr)
            return 5

    out.parent.mkdir(parents=True, exist_ok=True)
    if tier_csv_bundle:
        out_tier_a.parent.mkdir(parents=True, exist_ok=True)
        out_tier_b.parent.mkdir(parents=True, exist_ok=True)
        out_tier_c.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "markets" and fmt == "md":
        print("Error: --format md is only supported with --mode events. Use csv or parquet.", file=sys.stderr)
        return 6

    if args.mode == "markets":
        out_sql = _glob_sql(out)
        inner = f"SELECT * FROM {from_clause} m WHERE {pred}"
        if fmt == "parquet":
            con.execute(f"COPY ({inner}) TO '{out_sql}' (FORMAT PARQUET)")
        else:
            con.execute(f"COPY ({inner}) TO '{out_sql}' (FORMAT CSV, HEADER true)")
        print(f"Wrote {n_match} market row(s) to {out.resolve()}")
        return 0

    # events aggregate
    agg_kw = dict(
        group_key_sql=group_key_sql,
        pe_slug_sql=pe_slug_sql,
        pe_title_sql=pe_title_sql,
        neg_id_sql=neg_id_sql,
        series_slug_sql=series_slug_sql,
        git_sql=git_sql,
    )

    if tier_csv_bundle:
        pa, pb, pc = _tier_a_predicate_sql(cols), _tier_b_predicate_sql(cols), _tier_c_predicate_sql(cols)
        exports: list[tuple[Path, str, str]] = [
            (out_tier_a, pa, "tier A (Gamma JSON)"),
            (out_tier_b, pb, "tier B (slug)"),
            (out_tier_c, pc, "tier C (text)"),
            (out, pred, "combined A|B|C"),
        ]
        for dest, p_sql, label in exports:
            agg_sql = _make_events_aggregate_sql(from_clause, p_sql, cols, **agg_kw)
            dest_sql = _glob_sql(dest)
            try:
                n_m = con.execute(f"SELECT count(*)::BIGINT FROM {from_clause} m WHERE {p_sql}").fetchone()[0]
                n_ev2 = con.execute(
                    f"""
                    WITH fed AS (
                        SELECT * FROM {from_clause} m WHERE {p_sql}
                    ),
                    keyed AS (
                        SELECT ({group_key_sql}) AS group_key FROM fed
                    )
                    SELECT count(DISTINCT group_key)::BIGINT FROM keyed
                    """
                ).fetchone()[0]
            except Exception as e:
                print(f"Warning: pre-export count failed ({label}): {e}", file=sys.stderr)
                n_m, n_ev2 = None, None
            con.execute(f"COPY ({agg_sql}) TO '{dest_sql}' (FORMAT CSV, HEADER true)")
            if n_ev2 is not None and n_m is not None:
                print(f"Wrote {n_ev2} event row(s) ({label}, from {n_m} matched market row(s)) -> {dest.resolve()}")
            else:
                print(f"Wrote {label} -> {dest.resolve()}")
        _maybe_sort_primary_out_csv(mode=args.mode, fmt=fmt, out_path=out)
        return 0

    agg_sql = _make_events_aggregate_sql(from_clause, pred, cols, **agg_kw)
    out_sql = _glob_sql(out)
    n_events: int | None = None
    if fmt != "md":
        try:
            n_events = con.execute(
                f"""
                WITH fed AS (
                    SELECT * FROM {from_clause} m WHERE {pred}
                ),
                keyed AS (
                    SELECT ({group_key_sql}) AS group_key FROM fed
                )
                SELECT count(DISTINCT group_key)::BIGINT FROM keyed
                """
            ).fetchone()[0]
        except Exception as e:
            print(f"Warning: could not count distinct events before export: {e}", file=sys.stderr)

    if fmt == "md":
        try:
            cur = con.execute(agg_sql)
            col_names = [d[0] for d in cur.description]
            rows = cur.fetchall()
        except Exception as e:
            print(f"Export query failed: {e}", file=sys.stderr)
            return 7
        _write_markdown_events_table(out, col_names, rows)
        n_events = len(rows)
    elif fmt == "parquet":
        con.execute(f"COPY ({agg_sql}) TO '{out_sql}' (FORMAT PARQUET)")
    else:
        con.execute(f"COPY ({agg_sql}) TO '{out_sql}' (FORMAT CSV, HEADER true)")

    if n_events is not None:
        print(f"Wrote {n_events} event row(s) (from {n_match} matched market row(s)) to {out.resolve()}")
    else:
        print(f"Wrote event aggregate to {out.resolve()} (from {n_match} matched market row(s))")
    _maybe_sort_primary_out_csv(mode=args.mode, fmt=fmt, out_path=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
