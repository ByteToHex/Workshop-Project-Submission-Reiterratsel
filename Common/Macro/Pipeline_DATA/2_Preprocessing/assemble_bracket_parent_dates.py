"""
Merge 2VALIDATE_bracket_market_tokens with parent-level end dates from 2VALIDATE_parent_markets.

Run from repo root or any cwd (paths are resolved from this file).
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

# --- inputs / output (edit here) ---
_UTIL = Path(__file__).resolve().parents[1]
BRACKET_MARKET_TOKENS_CSV = _UTIL / "out_Markets" / "2VALIDATE_bracket_market_tokens.csv"
PARENT_MARKETS_CSV = _UTIL / "out_Markets" / "2VALIDATE_parent_markets.csv"
OUT_ALL_DIR = _UTIL / "out_All"
OUTPUT_CSV = OUT_ALL_DIR / "2VALIDATE_bracket_market_tokens_with_parent_dates.csv"


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
    raw = (row.get("parent_end_date_actual_resolution") or "").strip()
    dt = _parse_dt_for_sort(raw)
    if dt is not None:
        return (0, dt, raw)
    if not raw:
        return (2, raw)
    return (1, raw)


def main() -> int:
    if not BRACKET_MARKET_TOKENS_CSV.is_file():
        print(f"Missing: {BRACKET_MARKET_TOKENS_CSV}", file=sys.stderr)
        return 1
    if not PARENT_MARKETS_CSV.is_file():
        print(f"Missing: {PARENT_MARKETS_CSV}", file=sys.stderr)
        return 1

    parent_by_id: dict[str, tuple[str, str]] = {}
    with PARENT_MARKETS_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            pid = str(r.get("parent_market_id") or "").strip()
            if not pid:
                continue
            intended = str(r.get("market_end_date_intended") or "").strip()
            actual = str(r.get("market_end_date_actual_resolution") or "").strip()
            parent_by_id[pid] = (intended, actual)

    with BRACKET_MARKET_TOKENS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Bracket CSV has no header.", file=sys.stderr)
            return 1
        if "parent_sort_date" not in reader.fieldnames:
            print("Expected column parent_sort_date in bracket CSV.", file=sys.stderr)
            return 1

        base_fields = [c for c in reader.fieldnames if c != "parent_sort_date"]
        insert_at = base_fields.index("parent_series_slug") + 1
        out_fields = (
            base_fields[:insert_at]
            + ["parent_end_date_intended", "parent_end_date_actual_resolution"]
            + base_fields[insert_at:]
        )

        rows: list[dict[str, str]] = []
        for row in reader:
            row.pop("parent_sort_date", None)
            peid = str(row.get("parent_event_id") or "").strip()
            intended, actual = parent_by_id.get(peid, ("", ""))
            row["parent_end_date_intended"] = intended
            row["parent_end_date_actual_resolution"] = actual
            rows.append({k: row.get(k, "") for k in out_fields})

    rows.sort(key=_sort_key)

    OUT_ALL_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
