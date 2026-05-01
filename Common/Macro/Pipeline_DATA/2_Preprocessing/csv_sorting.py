"""
Shared CSV sorting helpers for util scripts.
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path


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


def sort_csv_by_end_date(path: Path, backup_suffix: str = "-bak") -> tuple[int, str]:
    """
    Sort one CSV by end-date-like columns and create a sidecar backup.

    Returns:
        (sorted_rows_count, used_sort_column_label)
    """
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    bak = path.with_name(f"{path.stem}{backup_suffix}{path.suffix}")
    shutil.copy2(path, bak)

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        return 0, "none (empty or header only)"

    header, body = rows[0], rows[1:]
    date_col = _pick_date_column(header)
    body.sort(key=lambda r: _sort_key(header, r, date_col))

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)

    used = date_col or ("event_key" if "event_key" in header else "first column")
    return len(body), used
