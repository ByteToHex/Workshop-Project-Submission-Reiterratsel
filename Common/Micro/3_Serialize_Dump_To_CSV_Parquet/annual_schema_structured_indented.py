"""
Build AnnualSchema_Structured_Indented.json from Tradingview_Schema_Annual_FromSS.txt.

Mirrors parse_schema_source(..., mode="Annual") row emission so row order and labels
match AnnualSchema_Structured.json, while adding depth and parent_id from the source
line indentation (leading whitespace in 4-space steps; same rule for bulleted and bare lines).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serialize_financials_to_csv import (
    EARNINGS_OUTPUT_LABELS,
    STATISTICS_OUTPUT_LABELS,
    _norm_key,
    clean_schema_line,
    parse_schema_source,
)


def _line_schema_depth(line: str) -> int:
    """
    Depth from leading whitespace: each 4-space step is one nesting level (same as bulleted lines).

    Bulleted and non-bulleted lines use the same rule so indented bare text (if present) is not
    forced to depth 1.
    """
    m = re.match(r"^(\s*)", line.rstrip("\r\n"))
    leading = m.group(1) if m else ""
    n = len(leading.expandtabs(4))
    return 1 + n // 4


@dataclass
class _Frame:
    depth: int
    row_id: int | None


def _parent_row_id(stack: list[_Frame], depth: int) -> int | None:
    while stack and stack[-1].depth >= depth:
        stack.pop()
    for f in reversed(stack):
        if f.row_id is not None:
            return f.row_id
    return None


def _emit_labels(
    labels: list[str],
    depth: int,
    stack: list[_Frame],
    section: str,
    schema_rows: list[dict[str, Any]],
    next_id: list[int],
    *,
    group_output_label: str | None = None,
) -> None:
    for lab in labels:
        if not lab:
            continue
        rid = next_id[0]
        next_id[0] += 1
        parent_id = _parent_row_id(stack, depth)
        row: dict[str, Any] = {
            "row_id": rid,
            "section": section,
            "label": lab,
            "depth": depth,
            "parent_id": parent_id,
            "group_output_label": group_output_label,
        }
        schema_rows.append(row)
        stack.append(_Frame(depth, rid))


def _append_schema_row_labels(raw: str, expand_yoy: bool) -> list[str]:
    """Same string expansion as append_schema_row for Annual (expand_yoy=False)."""
    out: list[str] = []
    if not raw or raw.startswith("//") or raw.startswith("Note:"):
        return out
    if "//" in raw:
        left, right = [x.strip() for x in raw.split("//", 1)]
        if expand_yoy and "yoy growth" in right.lower():
            out.append(left)
            out.append(f"{left} YoY growth")
        else:
            out.append(left)
            out.append(f"{left} {right}")
        return out
    out.append(raw)
    return out


def parse_annual_schema_indented(schema_path: Path) -> dict[str, Any]:
    """
    Walk Tradingview_Schema_Annual_FromSS.txt and emit schema_rows in bundle row order
    (matching parse_schema_source Annual output).
    """
    text = schema_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    schema_rows: list[dict[str, Any]] = []
    next_id = [1]
    stack: list[_Frame] = []

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
            stack.clear()
            continue
        if stripped.startswith("## SCHEMA_03"):
            schema_block = "dividends"
            statement_section = None
            current_stats_group = None
            current_earnings_group = None
            stack.clear()
            continue
        if stripped.startswith("## SCHEMA_04"):
            schema_block = "earnings"
            statement_section = None
            current_stats_group = None
            current_earnings_group = None
            stack.clear()
            continue
        if stripped.startswith("## SCHEMA_05"):
            schema_block = "revenue"
            statement_section = None
            current_stats_group = None
            current_earnings_group = None
            stack.clear()
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
                stack.clear()
                continue
            if schema_block == "earnings":
                output_label = EARNINGS_OUTPUT_LABELS[0] if header_norm == "eps" else EARNINGS_OUTPUT_LABELS[1]
                current_earnings_group = {"name": header, "output_label": output_label, "rows": []}
                stack.clear()
                continue
            continue

        raw = clean_schema_line(line)
        if not raw:
            continue

        depth = _line_schema_depth(line)

        if schema_block == "statements" and statement_section:
            labels = _append_schema_row_labels(raw, expand_yoy=False)
            _emit_labels(labels, depth, stack, statement_section, schema_rows, next_id)
            continue

        if schema_block == "statistics":
            header_norm = _norm_key(raw)
            if header_norm in STATISTICS_OUTPUT_LABELS:
                d = _line_schema_depth(line)
                while stack and stack[-1].depth >= d:
                    stack.pop()
                stack.append(_Frame(d, None))
                current_stats_group = {
                    "name": raw,
                    "output_label": STATISTICS_OUTPUT_LABELS[header_norm],
                    "rows": [],
                }
                continue
            if current_stats_group:
                labels = _append_schema_row_labels(raw, expand_yoy=False)
                _emit_labels(
                    labels,
                    depth,
                    stack,
                    "statistics",
                    schema_rows,
                    next_id,
                    group_output_label=current_stats_group["output_label"],
                )
            continue

        if schema_block == "dividends":
            if raw.lower() == "dividend payout history":
                continue
            if raw.startswith("|") and "ex-dividend date" in raw.lower():
                continue
            labels = _append_schema_row_labels(raw, expand_yoy=False)
            _emit_labels(labels, depth, stack, "dividends", schema_rows, next_id)
            continue

        if schema_block == "earnings" and current_earnings_group:
            labels = _append_schema_row_labels(raw, expand_yoy=False)
            _emit_labels(
                labels,
                depth,
                stack,
                "earnings",
                schema_rows,
                next_id,
                group_output_label=current_earnings_group["output_label"],
            )
            continue

        if schema_block == "revenue":
            # Annual txt lists ### By Source / ### By Country with no row lines.
            continue

    return {
        "mode": "Annual",
        "source_path": str(schema_path.resolve()),
        "schema_rows": schema_rows,
    }


def verify_indented_matches_structured(
    structured_bundle: dict[str, Any], indented: dict[str, Any]
) -> list[str]:
    """Return list of mismatch error strings; empty means OK."""
    errors: list[str] = []

    def flat_statement_rows(key: str) -> list[str]:
        return list(structured_bundle["statements"][key]["rows"])

    def rows_for_section(section: str) -> list[str]:
        return [r["label"] for r in indented["schema_rows"] if r["section"] == section]

    for key in ("income", "balance", "cashflow"):
        a, b = flat_statement_rows(key), rows_for_section(key)
        if a != b:
            errors.append(f"statements.{key}: structured {len(a)} vs indented {len(b)}")

    st_groups = structured_bundle["statistics"]["groups"]
    for g in st_groups:
        want = g["rows"]
        got = [
            r["label"]
            for r in indented["schema_rows"]
            if r["section"] == "statistics" and r.get("group_output_label") == g["output_label"]
        ]
        if want != got:
            errors.append(f"statistics group {g['output_label']!r}: label mismatch")

    div_t = structured_bundle["dividends"]["table"]["rows"]
    if div_t != rows_for_section("dividends"):
        errors.append("dividends.table rows mismatch")

    for g in structured_bundle["earnings"]["groups"]:
        want = g["rows"]
        got = [
            r["label"]
            for r in indented["schema_rows"]
            if r["section"] == "earnings" and r.get("group_output_label") == g["output_label"]
        ]
        if want != got:
            errors.append(f"earnings group {g['output_label']!r}: label mismatch")

    return errors


def ensure_annual_schema_structured_indented(
    schema_path: Path,
    out_json_path: Path,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    indented = parse_annual_schema_indented(schema_path)
    if verify:
        structured = parse_schema_source(schema_path, "Annual")
        errs = verify_indented_matches_structured(structured, indented)
        if errs:
            raise RuntimeError("Indented schema drift: " + "; ".join(errs))
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(indented, indent=2, ensure_ascii=False), encoding="utf-8")
    return indented


if __name__ == "__main__":
    from serialize_financials_to_csv import ANNUAL_SCHEMA_SOURCE, SERIALIZER_ROOT

    out = SERIALIZER_ROOT / "SCHEMA_DIFFERENCES" / "AnnualSchema_Structured_Indented.json"
    bundle = ensure_annual_schema_structured_indented(ANNUAL_SCHEMA_SOURCE, out)
    print(f"Wrote {out} ({len(bundle['schema_rows'])} rows)")
