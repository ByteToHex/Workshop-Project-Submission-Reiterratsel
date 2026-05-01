"""
Enrich REF overlap CSVs with parent event columns from the CTF bracket validate export.

Reads three OVERLAP_*.csv files, left-joins parent fields on market_slug, sorts by
parent_end_date_intended, overwrites each overlap file in place.

Run from VS Code / repo root (paths resolve from this file).
"""

from __future__ import annotations

import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

# --- REF inputs (hardcoded) ---
_SCRIPTS = Path(__file__).resolve().parents[2]
_REF = _SCRIPTS / "REF"

OVERLAP_CTF_ONLY_CSV = _REF / "OVERLAP_CTF_only.csv"
OVERLAP_FPMM_CTF_OVERLAP_CSV = _REF / "OVERLAP_FPMM_CTF_overlap.csv"
OVERLAP_FPMM_ONLY_CSV = _REF / "OVERLAP_FPMM_only.csv"
VALIDATE_CTF_BRACKET_WITH_PARENT_DATES_CSV = (
    _REF / "2VALIDATE_CTF_bracket_market_tokens_with_parent_dates.csv"
)

PARENT_COLS = ("parent_event_id", "parent_event_slug", "parent_end_date_intended")
OVERLAP_PATHS = (
    OVERLAP_CTF_ONLY_CSV,
    OVERLAP_FPMM_CTF_OVERLAP_CSV,
    OVERLAP_FPMM_ONLY_CSV,
)


def _parse_dt_for_sort(value: str) -> datetime | None:
    v = (value or "").strip()
    if not v:
        return None
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def _sort_key(row: dict[str, str]) -> tuple:
    raw = (row.get("parent_end_date_intended") or "").strip()
    dt = _parse_dt_for_sort(raw)
    if dt is not None:
        return (0, dt, raw)
    if not raw:
        return (2, raw)
    return (1, raw)


def _read_csv_sync(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header row: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv_sync(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _build_parent_by_slug(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        slug = (r.get("market_slug") or "").strip()
        if not slug:
            continue
        if slug in out:
            continue
        out[slug] = {c: (r.get(c) or "").strip() for c in PARENT_COLS}
    return out


def _merge_and_sort_rows(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    parent_by_slug: dict[str, dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    base = [c for c in fieldnames if c not in PARENT_COLS]
    out_fields = list(base) + list(PARENT_COLS)

    merged: list[dict[str, str]] = []
    for row in rows:
        slug = (row.get("market_slug") or "").strip()
        parent = parent_by_slug.get(slug, {c: "" for c in PARENT_COLS})
        new_row: dict[str, str] = {}
        for c in base:
            new_row[c] = row.get(c, "") or ""
        for c in PARENT_COLS:
            new_row[c] = parent[c]
        merged.append(new_row)

    merged.sort(key=_sort_key)
    return out_fields, merged


async def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    return await asyncio.to_thread(_read_csv_sync, path)


async def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    await asyncio.to_thread(_write_csv_sync, path, fieldnames, rows)


async def main() -> int:
    for p in (*OVERLAP_PATHS, VALIDATE_CTF_BRACKET_WITH_PARENT_DATES_CSV):
        if not p.is_file():
            print(f"Missing: {p}", file=sys.stderr)
            return 1

    v_fields, v_rows = await _read_csv(VALIDATE_CTF_BRACKET_WITH_PARENT_DATES_CSV)
    for c in PARENT_COLS:
        if c not in v_fields:
            print(f"Validate CSV missing column {c!r}: {VALIDATE_CTF_BRACKET_WITH_PARENT_DATES_CSV}", file=sys.stderr)
            return 1

    parent_by_slug = await asyncio.to_thread(_build_parent_by_slug, v_rows)

    overlap_reads = await asyncio.gather(*(_read_csv(p) for p in OVERLAP_PATHS))

    for path, (fields, rows) in zip(OVERLAP_PATHS, overlap_reads, strict=True):
        if "market_slug" not in fields:
            print(f"Expected market_slug in {path}", file=sys.stderr)
            return 1
        out_fields, out_rows = await asyncio.to_thread(
            _merge_and_sort_rows, fields, rows, parent_by_slug
        )
        await _write_csv(path, out_fields, out_rows)
        print(f"Wrote {len(out_rows)} rows -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
