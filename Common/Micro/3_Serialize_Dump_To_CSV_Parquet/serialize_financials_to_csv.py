"""
Serialize TradingView dumped fundamentals HTML (annual) to CSV.

HTML layout (TradingView CSS-module class hashes change over time; we match on stable patterns):

- Root block: div.js-financials-block-init-ssr (or any class containing "js-financials").
- Optional wrapper: div[class*="tableWrap"] groups the sticky header + scroll body.
- Header row: div[class*="container-OWKkVLyj"] with:
    - div[class*="firstColumn"]: "Currency: XXX"
    - div[class*="values-"] > div[class*="container-OxVAcLqi"] per period:
        - div[class*="value-"]: fiscal year label
        - optional div[class*="subvalue-"]: period end (e.g. Mar 2008)
        - optional "TTM" in a column with class "additional" on the container
- Body: div[class*="container-Tv7LSjUz"] > div[class*="wrapper-"] > div[class*="container-vKM0WfUu"]
  holds one div[class*="container-C9MdAMrq"] per line item.
- Each data row: title in span[class*="title-"] inside titleColumn/titleWrap; values in
  div[class*="values-C9MdAMrq"] (or values-* matching row hash) as a sequence of
  div[class*="container-OxVAcLqi"], each optionally containing:
    - div[class*="value-"]: main figure
    - div[class*="change-"]: YoY % when the row has withChange in its classes
  Period columns that are paywalled render a lock button instead of value-*; those cells
  are emitted as the missing marker (—), same as em dash / empty placeholders.

Earnings tab may omit tableWrap; the same container-vKM0WfUu / C9MdAMrq structure applies,
sometimes with an extra table-* class on the row stack. Two snapshot tables (EPS and Revenue)
produce two CSV files when both are present.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

# Path anchors resolved from this script, not from the shell's current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
SERIALIZER_ROOT = SCRIPT_DIR.parent

# VS Code-friendly run configuration.
# Edit these directly when running the script from the editor.
# INPUT_DIR = r"D:\ProgramData\.git\DATA_TradingView\Ver_00_EarningsGlitched\00_REIT_OPTIONS" # NOTE: comment out when using default, do NOT delete
INPUT_DIR = SERIALIZER_ROOT / "IO" / "in" / "01_grouped_html" # Consolidated pipeline input
OUTPUT_DIR = SERIALIZER_ROOT / "IO" / "out"
ANNUAL_SCHEMA_SOURCE = SERIALIZER_ROOT / "SCHEMA_DIFFERENCES" / "Tradingview_Schema_Annual_FromSS.txt"
NONANNUAL_SCHEMA_SOURCE = SERIALIZER_ROOT / "SCHEMA_DIFFERENCES" / "Tradingview_Schema_NonAnnual.txt"
ANNUAL_SCHEMA_STRUCTURED = SERIALIZER_ROOT / "SCHEMA_DIFFERENCES" / "AnnualSchema_Structured.json"
NONANNUAL_SCHEMA_STRUCTURED = SERIALIZER_ROOT / "SCHEMA_DIFFERENCES" / "NonAnnualSchema_Structured.json"

TIMEFRAME_MODE = "Annual"
TARGET_TICKERS: list[str] = []

# When False, editor runs ignore CLI arguments and use the variables above.
# Set to True only if you intentionally want command-line overrides again.
USE_CLI_ARGS = False

# HTML parsing remains synchronous per file, but files are processed concurrently via threads.
ASYNC_CONCURRENCY = 8

# Canonical missing marker in output (matches reference Selectors CSV)
MISSING = "—"

# Labels that appear as section headings in SCHEMA_02 but are not numeric rows in the TV table
SKIP_SCHEMA_SUBSECTION_TITLES = frozenset(
    {
        "valuation ratios",
        "profitability ratios",
        "liquidity ratios",
        "solvency ratios",
        "per share metrics",
    }
)

# Common HTML vs schema spelling drift
LABEL_ALIASES: dict[str, str] = {
    "deprecation and amortization": "Depreciation and amortization",
    "commision & fee income": "Commision & fee income",
}

STATISTICS_OUTPUT_LABELS = {
    "key stats": "0-Key_stats",
    "valuation ratios": "1-Valuation_ratios",
    "profitability ratios": "2-Profitability_Ratios",
    "liquidity ratios": "3-Liquidity_Ratios",
    "solvency ratios": "4-Solvency_Ratios",
    "per share metrics": "5-Per_Share_Metrics",
}

DIVIDENDS_OUTPUT_LABEL = "0-Dividend_yield_TTM"
DIVIDEND_PAYOUT_HISTORY_OUTPUT_LABEL = "1-Dividend_payout_history"
EARNINGS_OUTPUT_LABELS = ["0-EPS", "1-Revenue"]
REVENUE_OUTPUT_LABELS = ["0-By_Source", "1-By_Country"]
_EARNINGS_ANNUAL_RE = re.compile(r"^\d{4}$")


def _norm_key(s: str) -> str:
    s = s.strip().lower()
    s = LABEL_ALIASES.get(s, s)
    s = re.sub(r"[\u202a\u202b\u202c\u202d\u202e\u200e\u200f]", "", s)
    s = re.sub(r"[^\w\s&'().%\-/+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_cell_text(s: str) -> str:
    if not s:
        return MISSING
    for ch in "\u202a\u202b\u202c\u202d\u202e\u200e\u200f":
        s = s.replace(ch, "")
    s = s.replace("\u202f", " ").replace("\u2009", " ").strip()
    if not s:
        return MISSING
    if s in ("—", "–", "--", "−", "N/A", "n/a"):
        return MISSING
    return s


def find_financials_root(soup: BeautifulSoup) -> Tag | None:
    return soup.find("div", class_=re.compile(r"js-financials-block-init-ssr")) or soup.find(
        class_=re.compile(r"js-financials")
    )


def parse_header_periods(unit: Tag) -> tuple[str, list[str]]:
    """Return (currency_code, list of column header strings) from one table unit."""
    currency = ""
    hdr = unit.find(class_=re.compile(r"container-OWKkVLyj"))
    if not hdr:
        fc_only = unit.find(class_=re.compile(r"firstColumn-"))
        if fc_only:
            m = re.search(r"Currency:\s*(\S+)", fc_only.get_text(" ", strip=True))
            currency = m.group(1) if m else ""
        return currency, []

    fc = hdr.find(class_=re.compile(r"firstColumn-"))
    if fc:
        m = re.search(r"Currency:\s*(\S+)", fc.get_text(" ", strip=True))
        currency = m.group(1) if m else ""

    vals = hdr.find(class_=re.compile(r"values-"))
    if not vals:
        return currency, []

    cols: list[str] = []
    for cell in vals.find_all(class_=re.compile(r"container-OxVAcLqi"), recursive=False):
        v = cell.find(class_=re.compile(r"^value-"))
        sv = cell.find(class_=re.compile(r"subvalue-"))
        vt = clean_cell_text(v.get_text(" ", strip=True)) if v else MISSING
        if sv:
            st = clean_cell_text(sv.get_text(" ", strip=True))
            if st != MISSING:
                cols.append(f"{vt} / {st}")
            else:
                cols.append(vt)
        else:
            cols.append(vt)
    return currency, cols


def find_rows_container(shell: Tag) -> Tag | None:
    best: Tag | None = None
    best_n = 0
    for c in shell.find_all(class_=re.compile(r"container-vKM0WfUu")):
        n = len(
            [
                x
                for x in c.find_all(class_=re.compile(r"container-C9MdAMrq"), recursive=False)
                if isinstance(x, Tag)
            ]
        )
        if n > best_n:
            best_n = n
            best = c
    return best


def iter_table_units(financials: Tag):
    """Yield DOM roots that each contain one header + one row stack (Income) or one snapshot (Earnings)."""
    tw = financials.find(class_=re.compile(r"tableWrap"))
    if tw:
        yield tw
        return
    for sw in financials.find_all(class_=re.compile(r"shadowWrap-vKM0WfUu")):
        yield sw


def extract_row_cells(row: Tag) -> tuple[str, bool, list[str], list[str]]:
    """
    One fundamentals row: label, has YoY changes, value strings per period, change strings per period.
    """
    title_el = row.find(class_=re.compile(r"\btitle-[A-Za-z0-9_]+\b"))
    label = clean_cell_text(title_el.get_text(" ", strip=True)) if title_el else MISSING
    if label != MISSING:
        label = re.sub(r"[\u202a\u202b\u202c\u202d\u202e\u200e\u200f]", "", label).strip()

    classes = " ".join(row.get("class") or [])
    has_change = "withChange" in classes or "withchange" in classes.lower()

    vals_div = row.find(class_=re.compile(r"values-C9MdAMrq")) or row.find(
        class_=re.compile(r"values-.*C9MdAMrq")
    )
    if not vals_div:
        vals_div = row.find(class_=re.compile(r"^values-"))
    if not vals_div:
        return label, has_change, [], []

    values_out: list[str] = []
    changes_out: list[str] = []

    for cell in vals_div.find_all(class_=re.compile(r"container-OxVAcLqi"), recursive=False):
        v = cell.find(class_=re.compile(r"^value-"))
        ch = cell.find(class_=re.compile(r"^change-"))
        values_out.append(clean_cell_text(v.get_text(" ", strip=True)) if v else MISSING)
        if has_change:
            changes_out.append(clean_cell_text(ch.get_text(" ", strip=True)) if ch else MISSING)
        else:
            changes_out.append("")

    return label, has_change, values_out, changes_out


def expand_with_change_rows(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split rows that carry value + YoY% into two logical rows (values / YoY)."""
    out: list[dict[str, Any]] = []
    for r in parsed:
        if r.get("has_change") and any((r.get("changes") or [])):
            out.append(
                {
                    **r,
                    "has_change": False,
                    "changes": [],
                }
            )
            out.append(
                {
                    "label": f"{r['label']} YoY growth",
                    "has_change": False,
                    "values": [(c or MISSING) for c in (r.get("changes") or [])],
                    "changes": [],
                }
            )
        else:
            out.append({**r, "has_change": False})
    return out


def parse_html_tables(html: str) -> list[tuple[str, list[str], list[dict[str, Any]]]]:
    """Return one tuple (currency, periods, rows) per logical table in the dump."""
    soup = BeautifulSoup(html, "html.parser")
    fin = find_financials_root(soup)
    if not fin:
        return []

    tables: list[tuple[str, list[str], list[dict[str, Any]]]] = []
    for unit in iter_table_units(fin):
        currency, periods = parse_header_periods(unit)
        rc = find_rows_container(unit)
        if not rc:
            continue
        rows_out: list[dict[str, Any]] = []
        for row in rc.find_all(class_=re.compile(r"container-C9MdAMrq"), recursive=False):
            if not isinstance(row, Tag):
                continue
            label, has_change, vals, chgs = extract_row_cells(row)
            rows_out.append(
                {
                    "label": label,
                    "has_change": has_change,
                    "values": vals,
                    "changes": chgs,
                }
            )
        if rows_out:
            tables.append((currency, periods, rows_out))
    return tables


def _filter_earnings_periods(
    periods: list[str],
    parsed: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Keep only historical bare-year earnings periods such as "2023".

    This exists because earnings serialization does not flow through the old
    fallback branch near the bottom of process_file(); the live path is the
    early-return `if key == "earnings"` block above. That lower branch still
    contains old commented logic around suppressing the second earnings table,
    but for annual earnings it is effectively dead code and never got a chance
    to filter H1/H2 or forecast columns.

    By filtering periods here and calling this helper from the active earnings
    branch in both the CSV and parquet serializers, the suppression lives in
    the code path that actually writes output.

    Drops half-year labels, forecast years, and any non-YYYY formats.
    """
    current_year = datetime.date.today().year
    keep = [i for i, period in enumerate(periods) if _EARNINGS_ANNUAL_RE.match(period) and int(period) <= current_year]
    filtered_periods = [periods[i] for i in keep]

    filtered_rows: list[dict[str, Any]] = []
    for row in parsed:
        values = row.get("values") or []
        changes = row.get("changes") or []
        filtered_rows.append(
            {
                **row,
                "values": [values[i] for i in keep if i < len(values)],
                "changes": [changes[i] for i in keep if i < len(changes)],
            }
        )
    return filtered_periods, filtered_rows


def clean_schema_line(line: str) -> str:
    raw = line.strip()
    while raw.startswith("*"):
        raw = raw[1:].lstrip()
    raw = raw.replace("**", "").strip()
    return raw


def append_schema_row(rows: list[str], raw: str, expand_yoy: bool) -> None:
    if not raw or raw.startswith("//") or raw.startswith("Note:"):
        return
    if "//" in raw:
        left, right = [x.strip() for x in raw.split("//", 1)]
        if expand_yoy and "yoy growth" in right.lower():
            rows.append(left)
            rows.append(f"{left} YoY growth")
        else:
            rows.append(left)
            rows.append(f"{left} {right}")
        return
    rows.append(raw)


def parse_schema_source(schema_path: Path, mode: str) -> dict[str, Any]:
    text = schema_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    bundle: dict[str, Any] = {
        "mode": mode,
        "source_path": str(schema_path),
        "statements": {
            "income": {"rows": []},
            "balance": {"rows": []},
            "cashflow": {"rows": []},
        },
        "statistics": {"groups": []},
        "dividends": {
            "table": {"output_label": DIVIDENDS_OUTPUT_LABEL, "rows": []},
            "payout_history": {
                "output_label": DIVIDEND_PAYOUT_HISTORY_OUTPUT_LABEL,
                "columns": [],
            },
        },
        "earnings": {"groups": []},
        "revenue": {
            "groups": [
                {"name": "By Source", "output_label": REVENUE_OUTPUT_LABELS[0], "rows": []},
                {"name": "By Country", "output_label": REVENUE_OUTPUT_LABELS[1], "rows": []},
            ]
        },
    }

    schema_block: str | None = None
    statement_section: str | None = None
    current_stats_group: dict[str, Any] | None = None
    current_earnings_group: dict[str, Any] | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        if stripped.startswith("## SCHEMA_01"):
            schema_block = "statements"
            statement_section = None
            current_stats_group = None
            current_earnings_group = None
            continue
        if stripped.startswith("## SCHEMA_02"):
            schema_block = "statistics"
            statement_section = None
            current_stats_group = None
            current_earnings_group = None
            continue
        if stripped.startswith("## SCHEMA_03"):
            schema_block = "dividends"
            statement_section = None
            current_stats_group = None
            current_earnings_group = None
            continue
        if stripped.startswith("## SCHEMA_04"):
            schema_block = "earnings"
            statement_section = None
            current_stats_group = None
            current_earnings_group = None
            continue
        if stripped.startswith("## SCHEMA_05"):
            schema_block = "revenue"
            statement_section = None
            current_stats_group = None
            current_earnings_group = None
            continue

        if stripped.startswith("### "):
            header = clean_schema_line(stripped[4:])
            header_norm = _norm_key(header)
            if schema_block == "statements":
                if header_norm == "income statement":
                    statement_section = "income"
                elif header_norm == "balance sheet":
                    statement_section = "balance"
                elif header_norm == "cash flow":
                    statement_section = "cashflow"
                else:
                    statement_section = None
                continue
            if schema_block == "earnings":
                output_label = EARNINGS_OUTPUT_LABELS[0] if header_norm == "eps" else EARNINGS_OUTPUT_LABELS[1]
                current_earnings_group = {"name": header, "output_label": output_label, "rows": []}
                bundle["earnings"]["groups"].append(current_earnings_group)
                continue
            continue

        raw = clean_schema_line(line)
        if not raw:
            continue

        if schema_block == "statements" and statement_section:
            append_schema_row(bundle["statements"][statement_section]["rows"], raw, expand_yoy=(mode == "NonAnnual"))
            continue
        if schema_block == "statistics":
            header_norm = _norm_key(raw)
            if header_norm in STATISTICS_OUTPUT_LABELS:
                current_stats_group = {
                    "name": raw,
                    "output_label": STATISTICS_OUTPUT_LABELS[header_norm],
                    "rows": [],
                }
                bundle["statistics"]["groups"].append(current_stats_group)
                continue
            if current_stats_group:
                append_schema_row(current_stats_group["rows"], raw, expand_yoy=False)
            continue
        if schema_block == "dividends":
            if raw.lower() == "dividend payout history":
                continue
            if raw.startswith("|") and "ex-dividend date" in raw.lower():
                cols = [clean_cell_text(col.strip()) for col in raw.strip("|").split("|")]
                bundle["dividends"]["payout_history"]["columns"] = cols
                continue
            append_schema_row(bundle["dividends"]["table"]["rows"], raw, expand_yoy=False)
            continue
        if schema_block == "earnings" and current_earnings_group:
            append_schema_row(current_earnings_group["rows"], raw, expand_yoy=False)
            continue

    return bundle


def ensure_structured_schema(source_path: Path, structured_path: Path, mode: str) -> dict[str, Any]:
    bundle = parse_schema_source(source_path, mode)
    structured_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle


def load_schema_bundles() -> dict[str, dict[str, Any]]:
    return {
        "Annual": ensure_structured_schema(ANNUAL_SCHEMA_SOURCE, ANNUAL_SCHEMA_STRUCTURED, "Annual"),
        "NonAnnual": ensure_structured_schema(NONANNUAL_SCHEMA_SOURCE, NONANNUAL_SCHEMA_STRUCTURED, "NonAnnual"),
    }


def infer_schema_key(path: Path) -> str | None:
    """Map dump filename to schema section. Supports legacy `[TICKER]_…` and dumper `(TICKER)N-…` names."""
    n = path.name.lower()
    if "income_statement" in n or "0-income" in n:
        return "income"
    if "balance_sheet" in n or "1-balance" in n:
        return "balance"
    if "cash_flow" in n or "2-cash" in n:
        return "cashflow"
    if "statistics" in n:
        return "statistics"
    if "dividend" in n:
        return "dividends"
    if "earnings" in n:
        return "earnings"
    if "revenue" in n or "5-" in n:
        # SCHEMA_05 is dynamic; serialize rows as parsed when strict schema matching is unavailable.
        return "revenue"
    return None


def extract_ticker(path: Path) -> str | None:
    match = re.search(r"\(([A-Z0-9]+)-([A-Z0-9]+)\)", path.name, re.IGNORECASE)
    if match:
        return match.group(2).upper()
    return None


def infer_timeframe_bucket(path: Path) -> str | None:
    name = path.name.lower()
    if "_annual.html" in name or "_unknowntf.html" in name:
        return "Annual"
    if "_semiannual.html" in name or "_quarterly.html" in name:
        return "NonAnnual"
    return None


def should_serialize_path(path: Path) -> bool:
    """
    Apply the editor-configured dump filters.

    TIMEFRAME_MODE:
    - "Annual" keeps only `_Annual` and `_unknownTF` files.
    - "NonAnnual" keeps only `_Semiannual` and `_Quarterly` files.

    TARGET_TICKERS:
    - `[]` means export every ticker found under INPUT_DIR.
    - `["C38U"]` means export only files for ticker `C38U`.

    Both filters apply simultaneously.
    """
    if TIMEFRAME_MODE not in {"Annual", "NonAnnual"}:
        raise ValueError("TIMEFRAME_MODE must be either 'Annual' or 'NonAnnual'")

    timeframe_bucket = infer_timeframe_bucket(path)
    if timeframe_bucket != TIMEFRAME_MODE:
        return False

    normalized_targets = {ticker.upper() for ticker in TARGET_TICKERS}
    if not normalized_targets:
        return True

    ticker = extract_ticker(path)
    return ticker in normalized_targets if ticker else False


def resolve_output_timeframe_name(path: Path) -> str:
    name = path.name.lower()
    if "_unknowntf.html" in name or "_annual.html" in name:
        return "Annual"
    if "_semiannual.html" in name:
        return "Semiannual"
    if "_quarterly.html" in name:
        return "Quarterly"
    return "Annual"


def build_output_path(html_path: Path, out_dir: Path, replacement_label: str | None = None) -> Path:
    stem = html_path.stem
    stem = re.sub(r"_(unknownTF|Annual|Semiannual|Quarterly)$", f"_{resolve_output_timeframe_name(html_path)}", stem)
    if replacement_label:
        stem = stem.replace("(NA)", f"({replacement_label})")
    return out_dir / f"{stem}.csv"


def discover_html_paths(root: Path) -> list[Path]:
    """
    Recursively discover HTML files under root, then apply timeframe + ticker filters.
    """
    paths: list[Path] = []
    for p in root.rglob("*.html"):
        if p.is_file() and should_serialize_path(p):
            paths.append(p)
    return sorted(paths)


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_key(a), _norm_key(b)).ratio()


def match_rows_to_schema(
    schema_order: list[str],
    parsed_rows: list[dict[str, Any]],
    *,
    include_extras: bool = True,
) -> list[dict[str, Any]]:
    """Return ordered rows: each schema line + optional extra HTML rows; values aligned to periods."""
    by_label: dict[str, dict[str, Any]] = {}
    for r in parsed_rows:
        k = _norm_key(r["label"])
        if k not in by_label:
            by_label[k] = r

    used_html: set[str] = set()
    out: list[dict[str, Any]] = []

    for schema_line in schema_order:
        sk = _norm_key(schema_line)
        best_key = None
        best_score = 0.0
        for hk, row in by_label.items():
            if hk in used_html:
                continue
            score = _similar(schema_line, row["label"])
            if score > best_score:
                best_score = score
                best_key = hk
        if best_key and best_score >= 0.72:
            r = by_label[best_key]
            used_html.add(best_key)
            out.append({**r, "schema_label": schema_line})
        else:
            n = len(parsed_rows[0]["values"]) if parsed_rows else 0
            out.append(
                {
                    "schema_label": schema_line,
                    "label": schema_line,
                    "has_change": False,
                    "values": [MISSING] * n,
                    "changes": [""] * n,
                    "synthetic": True,
                }
            )

    if include_extras:
        for r in parsed_rows:
            k = _norm_key(r["label"])
            if k not in used_html:
                out.append({**r, "schema_label": r["label"]})

    return out


def rows_to_csv_rows(
    currency: str,
    periods: list[str],
    merged: list[dict[str, Any]],
) -> list[list[str]]:
    """Build CSV rows: currency row, header row, then data rows (value row; optional YoY row)."""
    header = ["label"] + periods
    grid: list[list[str]] = [
        ["currency", currency] + [""] * (len(periods) - 1),
        header,
    ]

    for r in merged:
        label = r.get("schema_label", r["label"])
        vals = r["values"]
        chgs = r.get("changes") or []
        # Pad / trim to period count
        if len(vals) < len(periods):
            vals = vals + [MISSING] * (len(periods) - len(vals))
        elif len(vals) > len(periods):
            vals = vals[: len(periods)]

        grid.append([label] + vals)

    return grid


def parse_dividend_payout_history(html: str, expected_header: list[str] | None = None) -> list[list[str]]:
    """Return the lower Dividends payout-history table as a standalone CSV grid."""
    soup = BeautifulSoup(html, "html.parser")
    fin = find_financials_root(soup)
    if not fin:
        return []

    heading = fin.find(string=lambda s: s and "dividend payout history" in s.lower())
    if not heading or not isinstance(heading.parent, Tag):
        return []

    payout_section = heading.parent.find_next_sibling("div", class_=re.compile(r"container-Tv7LSjUz"))
    if not payout_section:
        return []

    header_wrap = payout_section.find(class_=re.compile(r"container-OWKkVLyj"))
    rows_wrap = payout_section.find(class_=re.compile(r"container-vKM0WfUu"))
    if not header_wrap or not rows_wrap:
        return []

    header = [clean_cell_text(cell.get_text(" ", strip=True)) for cell in header_wrap.find_all(recursive=False)]
    if not header and expected_header:
        header = list(expected_header)
    expected_columns = max(1, len(header))
    grid: list[list[str]] = [header]

    for row in rows_wrap.find_all(class_=re.compile(r"container-C9MdAMrq"), recursive=False):
        if not isinstance(row, Tag):
            continue

        first_col = row.find(class_=re.compile(r"titleWrap-C9MdAMrq"))
        cells = [clean_cell_text(first_col.get_text(" ", strip=True)) if first_col else MISSING]
        for cell in row.find_all(class_=re.compile(r"container-OxVAcLqi"), recursive=False):
            cells.append(clean_cell_text(cell.get_text(" ", strip=True)))

        if len(cells) < expected_columns:
            cells.extend([MISSING] * (expected_columns - len(cells)))
        elif len(cells) > expected_columns:
            cells = cells[:expected_columns]

        grid.append(cells)

    return grid if len(grid) > 1 else []


def write_csv(path: Path, grid: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for row in grid:
            w.writerow(row)


def process_file(
    html_path: Path,
    schema_bundles: dict[str, dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    key = infer_schema_key(html_path)
    timeframe_bucket = infer_timeframe_bucket(html_path) or TIMEFRAME_MODE
    schema_bundle = schema_bundles[timeframe_bucket]
    html = html_path.read_text(encoding="utf-8", errors="replace")
    tables = parse_html_tables(html)
    if not tables:
        out = build_output_path(html_path, out_dir)
        write_csv(out, [["error", "no table parsed"], [str(html_path)]])
        return [out]

    out_paths: list[Path] = []
    if key == "dividends":
        payout_history_grid = parse_dividend_payout_history(
            html,
            schema_bundle["dividends"]["payout_history"]["columns"],
        )
        if payout_history_grid:
            payout_out = build_output_path(html_path, out_dir, DIVIDEND_PAYOUT_HISTORY_OUTPUT_LABEL)
            write_csv(payout_out, payout_history_grid)
            out_paths.append(payout_out)

    if key in {"income", "balance", "cashflow"}:
        currency, periods, parsed = tables[0]
        parsed = expand_with_change_rows(parsed)
        schema_lines = list(schema_bundle["statements"][key]["rows"])
        merged = match_rows_to_schema(schema_lines, parsed)
        out = build_output_path(html_path, out_dir)
        write_csv(out, rows_to_csv_rows(currency, periods, merged))
        out_paths.append(out)
        return out_paths

    if key == "statistics":
        currency, periods, parsed = tables[0]
        for group in schema_bundle["statistics"]["groups"]:
            merged = match_rows_to_schema(group["rows"], parsed, include_extras=False)
            out = build_output_path(html_path, out_dir, group["output_label"])
            write_csv(out, rows_to_csv_rows(currency, periods, merged))
            out_paths.append(out)
        return out_paths

    if key == "dividends":
        currency, periods, parsed = tables[0]
        merged = match_rows_to_schema(schema_bundle["dividends"]["table"]["rows"], parsed)
        out = build_output_path(html_path, out_dir, schema_bundle["dividends"]["table"]["output_label"])
        write_csv(out, rows_to_csv_rows(currency, periods, merged))
        out_paths.append(out)
        return out_paths

    if key == "earnings":
        for idx, group in enumerate(schema_bundle["earnings"]["groups"]):
            if idx >= len(tables):
                break
            currency, periods, parsed = tables[idx]
            periods, parsed = _filter_earnings_periods(periods, parsed)
            if not periods:
                continue
            merged = match_rows_to_schema(group["rows"], parsed)
            out = build_output_path(html_path, out_dir, group["output_label"])
            write_csv(out, rows_to_csv_rows(currency, periods, merged))
            out_paths.append(out)
        return out_paths

    if key == "revenue":
        for idx, group in enumerate(schema_bundle["revenue"]["groups"]):
            if idx >= len(tables):
                break
            currency, periods, parsed = tables[idx]
            merged = [{**r, "schema_label": r["label"]} for r in parsed]
            out = build_output_path(html_path, out_dir, group["output_label"])
            write_csv(out, rows_to_csv_rows(currency, periods, merged))
            out_paths.append(out)
        return out_paths

    schema_sections: dict[str, list[str]] = {}
    n_tab = len(tables)
    for idx, (currency, periods, parsed) in enumerate(tables):
        if not periods or not parsed:
            continue
        parsed = expand_with_change_rows(parsed)
        if key == "earnings" and n_tab >= 2:
            # Original two-table handling kept here intentionally for reference:
            # prefix = "EPS — " if idx == 0 else "Revenue — "
            # for r in parsed:
            #     r["label"] = prefix + r["label"]
            if idx > 0:
                # Revenue rows from the lower earnings table are intentionally disabled.
                # Keep EPS serialization only.
                continue
            prefix = "EPS — "
            for r in parsed:
                r["label"] = prefix + r["label"]

        schema_lines = list(schema_sections.get(key or "", []))
        if key == "earnings" and n_tab >= 2:
            # Original schema split kept here intentionally for reference:
            # if idx == 0:
            #     schema_lines = [s for s in schema_lines if s.startswith("EPS —")]
            # else:
            #     schema_lines = [s for s in schema_lines if s.startswith("Revenue —")]
            schema_lines = [s for s in schema_lines if s.startswith("EPS —")]
        if schema_lines:
            merged = match_rows_to_schema(schema_lines, parsed)
        else:
            merged = [{**r, "schema_label": r["label"]} for r in parsed]

        grid = rows_to_csv_rows(currency, periods, merged)
        suffix = f"__{idx + 1:02d}" if n_tab > 1 else ""
        out = out_dir / f"{html_path.stem}{suffix}.csv"
        write_csv(out, grid)
        out_paths.append(out)
    return out_paths


def resolve_runtime_paths() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Serialize TradingView fundamentals HTML to CSV.")
    ap.add_argument(
        "--input",
        type=Path,
        default=INPUT_DIR,
        help="Root directory containing TradingView HTML dumps",
    )
    ap.add_argument(
        "--schema",
        type=Path,
        default=NONANNUAL_SCHEMA_SOURCE,
        help="Deprecated compatibility flag; the script now auto-loads annual + nonannual schema sources",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for CSV files",
    )
    ap.add_argument(
        "--glob",
        dest="glob_pat",
        default="*.html",
        help="Kept for compatibility; discovery is recursive and filtered by TIMEFRAME_MODE/TARGET_TICKERS",
    )
    args = ap.parse_args() if USE_CLI_ARGS else ap.parse_args([])
    args.input = Path(args.input).resolve()
    args.schema = Path(args.schema).resolve()
    args.output = Path(args.output).resolve()
    return args


async def process_file_async(
    html_path: Path,
    schema_bundles: dict[str, dict[str, Any]],
    out_dir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[Path, list[Path]]:
    async with semaphore:
        outs = await asyncio.to_thread(process_file, html_path, schema_bundles, out_dir)
        return html_path, outs


async def main() -> None:
    args = resolve_runtime_paths()

    schema_bundles = load_schema_bundles()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = discover_html_paths(args.input)
    if not paths:
        print(f"No HTML files matched under {args.input} for TIMEFRAME_MODE={TIMEFRAME_MODE} and TARGET_TICKERS={TARGET_TICKERS}")
        return

    semaphore = asyncio.Semaphore(max(1, ASYNC_CONCURRENCY))
    tasks = [
        process_file_async(p, schema_bundles, args.output, semaphore)
        for p in paths
        if p.is_file()
    ]

    for task in asyncio.as_completed(tasks):
        p, outs = await task
        print(f"{p.name} -> {', '.join(str(x) for x in outs)}")


if __name__ == "__main__":
    asyncio.run(main())
