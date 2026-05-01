"""
Sort CSV files in a single folder's root (non-recursive).

Default target is ``scripts/util/out`` (hardcoded below). For each ``*.csv``,
copies ``name.csv`` -> ``name-bak.csv`` then overwrites ``name.csv`` with the
same rows sorted by ``event_key`` ascending (string order). If there is no
``event_key`` column, sorts by the first column.

Usage (from repo root)::

    python scripts/util/sort_out_csvs.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


# Only CSVs directly under this directory are processed (not subfolders).
INPUT_DIR: Path = Path(__file__).resolve().parent / "out"


def _sort_key(header: list[str], row: list[str]) -> str:
    if len(row) != len(header):
        return str(row)
    row_dict = dict(zip(header, row))
    if "event_key" in row_dict:
        return row_dict.get("event_key") or ""
    return row[0] if row else ""


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
        body.sort(key=lambda r: _sort_key(header, r))

        with src.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(body)

        print(f"{src.name}: backup -> {bak.name}, sorted {len(body)} row(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
