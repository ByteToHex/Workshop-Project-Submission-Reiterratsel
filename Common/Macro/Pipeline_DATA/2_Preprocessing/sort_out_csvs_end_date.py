"""
Sort CSV files in a single folder's root (non-recursive) by intended end date.

Default target is ``scripts/util/out`` (hardcoded below). For each ``*.csv``,
copies ``name.csv`` -> ``name-bak.csv`` then overwrites ``name.csv`` with the
same rows sorted by the first available end-date column:

1) ``end_date``
2) ``min_end_date``
3) ``max_end_date``

Rows with missing/invalid dates are pushed to the bottom. If none of those
columns exist, it falls back to ``event_key`` (or first column).

Usage (from repo root)::

    python scripts/util/sort_out_csvs_end_date.py
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path


# Only CSVs directly under this directory are processed (not subfolders).
INPUT_DIR: Path = Path(__file__).resolve().parent / "out"


def _pick_date_column(header: list[str]) -> str | None:
    for name in ("end_date", "min_end_date", "max_end_date"):
        if name in header:
            return name
    return None


def _parse_dt(value: str) -> datetime | None:
    v = (value or "").strip()
    if not v:
        return None
    try:
        # Handles values like "2026-01-28 08:00:00+08"
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def _sort_key(header: list[str], row: list[str], date_col: str | None) -> tuple:
    if len(row) != len(header):
        return (1, None, str(row))

    row_dict = dict(zip(header, row))

    if date_col:
        dt = _parse_dt(row_dict.get(date_col, ""))
        if dt is not None:
            return (0, dt, "")

    if "event_key" in row_dict:
        return (1, None, row_dict.get("event_key") or "")

    return (1, None, row[0] if row else "")


def main() -> int:
    root = INPUT_DIR
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not paths:
        print(f"No CSV files in {root}")
        return 0

    for src in paths:
        bak = src.with_name(f"{src.stem}-bak{src.suffix}")
        shutil.copy2(src, bak)

        with src.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            print(f"skip (empty or header only): {src.name}")
            continue

        header, body = rows[0], rows[1:]
        date_col = _pick_date_column(header)
        body.sort(key=lambda r: _sort_key(header, r, date_col))

        with src.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(body)

        used = date_col or ("event_key" if "event_key" in header else "first column")
        print(f"{src.name}: backup -> {bak.name}, sorted {len(body)} row(s) by {used}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
