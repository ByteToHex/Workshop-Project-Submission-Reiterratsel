"""
Export ledger-to-timestamp mapping for the market time window inferred from Step 02 output.

Window source:
  - earliest non-empty `created_at`
  - latest non-empty `end_date`
from scripts/util/out_Markets/2VALIDATE_bracket_market_tokens.csv by default.

Then filters data/parquet/ledgers/ledgers_*.parquet to rows whose `timestamp` falls inside
that inclusive time window and writes a CSV mapping:
  ledger_number,timestamp

Outputs are written under scripts/util/out_Trades/LEDGERTS/ by default.

Usage (repo root):
  python scripts/util/Step_04_LedgerTimestamp/export_ledgers_for_market_time_window.py
  python scripts/util/Step_04_LedgerTimestamp/export_ledgers_for_market_time_window.py --scan-mode sample --ledgers-sample-files 50

It is okay to run from the top-right play button for default behavior:
  - markets csv: scripts/util/out_Markets/2VALIDATE_bracket_market_tokens.csv
  - data dir: data/parquet
  - outputs: scripts/util/out_Trades/LEDGERTS/
  - full scan mode
  - pad-hours: 72
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb

_UTIL_DIR = Path(__file__).resolve().parents[1]
_OUT_DIR = _UTIL_DIR / "out_Trades" / "LEDGERTS"
_MARKETS_OUT_DIR = _UTIL_DIR / "out_Markets"
_DEFAULT_MARKETS_CSV = _MARKETS_OUT_DIR / "2VALIDATE_bracket_market_tokens.csv"
_DEFAULT_PREFIX = "4LEDGERTS_WINDOW"


def _glob_sql(path: Path) -> str:
    return str(path).replace("\\", "/")


def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _first_files(d: Path, pattern: str, n: int | None) -> list[Path]:
    files = sorted(d.glob(pattern))
    if n is None or n <= 0:
        return files
    return files[:n]


def _csv_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    candidate = _OUT_DIR / p
    if candidate.exists():
        return candidate
    return _MARKETS_OUT_DIR / p


def _resolve_ledgers_dir(root: Path) -> Path:
    direct = root / "ledgers"
    if direct.is_dir():
        return direct
    alt = root / "parquet" / "ledgers"
    if alt.is_dir():
        return alt
    return direct


def _run_fetchall_with_heartbeat_async(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    stage: str,
    heartbeat_sec: float,
) -> list[tuple]:
    if heartbeat_sec <= 0:
        return con.execute(sql).fetchall()

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: con.execute(sql).fetchall())
        while True:
            try:
                rows = fut.result(timeout=heartbeat_sec)
                elapsed = int(time.monotonic() - started)
                print(f"\r[{stage}] done in {elapsed}s.{' ' * 24}", file=sys.stderr, flush=True)
                return rows
            except concurrent.futures.TimeoutError:
                elapsed = int(time.monotonic() - started)
                print(f"\r[{stage}] running... {elapsed}s", end="", file=sys.stderr, flush=True)


def _run_execute_with_heartbeat_async(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    stage: str,
    heartbeat_sec: float,
) -> None:
    if heartbeat_sec <= 0:
        con.execute(sql)
        return

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: con.execute(sql))
        while True:
            try:
                fut.result(timeout=heartbeat_sec)
                elapsed = int(time.monotonic() - started)
                print(f"\r[{stage}] done in {elapsed}s.{' ' * 24}", file=sys.stderr, flush=True)
                return
            except concurrent.futures.TimeoutError:
                elapsed = int(time.monotonic() - started)
                print(f"\r[{stage}] running... {elapsed}s", end="", file=sys.stderr, flush=True)


def _parse_iso_utc(value: str) -> datetime | None:
    s = (value or "").strip()
    if not s:
        return None
    # Normalize trailing Z for fromisoformat.
    s2 = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(s2)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _read_market_window(markets_csv: Path) -> tuple[datetime, datetime, int]:
    with markets_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Input markets CSV has no rows.")

    earliest_created_at: datetime | None = None
    latest_end_date: datetime | None = None

    for r in rows:
        created = _parse_iso_utc(str(r.get("created_at", "")))
        if created is not None and (earliest_created_at is None or created < earliest_created_at):
            earliest_created_at = created

        end_dt = _parse_iso_utc(str(r.get("end_date", "")))
        if end_dt is not None and (latest_end_date is None or end_dt > latest_end_date):
            latest_end_date = end_dt

    if earliest_created_at is None:
        raise RuntimeError("No parseable `created_at` values in markets CSV.")
    if latest_end_date is None:
        raise RuntimeError("No parseable `end_date` values in markets CSV.")

    if earliest_created_at > latest_end_date:
        raise RuntimeError(
            f"Invalid window: earliest created_at ({earliest_created_at.isoformat()}) "
            f"is after latest end_date ({latest_end_date.isoformat()})."
        )

    return earliest_created_at, latest_end_date, len(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--markets-csv", type=Path, default=_DEFAULT_MARKETS_CSV)
    p.add_argument("--data-dir", type=Path, default=Path("data/parquet"))
    p.add_argument(
        "--scan-mode",
        choices=["full", "sample"],
        default="full",
        help="Use full ledgers parquet set or sampled subset (default: full).",
    )
    p.add_argument(
        "--ledgers-sample-files",
        type=int,
        default=100,
        metavar="N",
        help="When --scan-mode=sample, read first N ledgers_*.parquet files (default: 100).",
    )
    p.add_argument("--output-prefix", type=str, default=_DEFAULT_PREFIX)
    p.add_argument(
        "--pad-hours",
        type=float,
        default=72.0,
        help="Padding hours applied on both sides of market window (default: 72).",
    )
    p.add_argument(
        "--heartbeat-sec",
        type=float,
        default=2.0,
        help="Heartbeat interval during long SQL stages (<=0 disables).",
    )
    args = p.parse_args(argv)

    markets_csv = _csv_path(args.markets_csv)
    if not markets_csv.is_file():
        print(f"Markets CSV not found: {markets_csv}", file=sys.stderr)
        return 2

    try:
        earliest_created_at, latest_end_date, n_market_rows = _read_market_window(markets_csv)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 3

    ledgers_dir = _resolve_ledgers_dir(args.data_dir)
    if not ledgers_dir.is_dir():
        print(f"No ledgers directory: {ledgers_dir.resolve()}", file=sys.stderr)
        return 4

    ledger_limit = args.ledgers_sample_files if args.scan_mode == "sample" else None
    ledger_files = _first_files(ledgers_dir, "ledgers_*.parquet", ledger_limit)
    if not ledger_files:
        print(f"No ledgers_*.parquet under {ledgers_dir.resolve()}", file=sys.stderr)
        return 5

    ledger_file_sql = ", ".join(_sql_quote(_glob_sql(f)) for f in ledger_files)
    ledgers_from = f"read_parquet([{ledger_file_sql}], union_by_name=true)"

    con = duckdb.connect(database=":memory:")
    ledger_cols = {str(r[0]) for r in con.execute(f"DESCRIBE SELECT * FROM {ledgers_from} LIMIT 0").fetchall()}
    if "ledger_number" not in ledger_cols:
        print("Ledgers parquet missing required `ledger_number` column.", file=sys.stderr)
        return 6
    if "timestamp" not in ledger_cols:
        print("Ledgers parquet missing expected `timestamp` column.", file=sys.stderr)
        return 7

    pad_delta = timedelta(hours=max(0.0, float(args.pad_hours)))
    window_start = earliest_created_at - pad_delta
    window_end = latest_end_date + pad_delta
    earliest_iso = earliest_created_at.isoformat().replace("+00:00", "Z")
    latest_iso = latest_end_date.isoformat().replace("+00:00", "Z")
    window_start_iso = window_start.isoformat().replace("+00:00", "Z")
    window_end_iso = window_end.isoformat().replace("+00:00", "Z")

    # Inclusive padded bounds.
    _run_execute_with_heartbeat_async(
        con,
        f"""
        CREATE TEMP TABLE window_ledgers AS
        SELECT
            cast(ledger_number AS BIGINT) AS ledger_number,
            cast(timestamp AS VARCHAR) AS timestamp
        FROM {ledgers_from}
        WHERE ledger_number IS NOT NULL
          AND timestamp IS NOT NULL
          AND try_cast(timestamp AS TIMESTAMPTZ) >= try_cast({_sql_quote(window_start_iso)} AS TIMESTAMPTZ)
          AND try_cast(timestamp AS TIMESTAMPTZ) <= try_cast({_sql_quote(window_end_iso)} AS TIMESTAMPTZ)
        ORDER BY ledger_number
        """,
        stage="filter-window-ledgers",
        heartbeat_sec=args.heartbeat_sec,
    )

    stats = _run_fetchall_with_heartbeat_async(
        con,
        """
        SELECT
            count(*)::BIGINT AS n_ledgers,
            min(ledger_number)::BIGINT AS min_ledger_number,
            max(ledger_number)::BIGINT AS max_ledger_number,
            min(timestamp) AS min_timestamp,
            max(timestamp) AS max_timestamp
        FROM window_ledgers
        """,
        stage="window-stats",
        heartbeat_sec=args.heartbeat_sec,
    )[0]

    n_ledgers = int(stats[0] or 0)
    min_ledger = int(stats[1] or 0)
    max_ledger = int(stats[2] or 0)
    min_ts = str(stats[3] or "")
    max_ts = str(stats[4] or "")

    rows = _run_fetchall_with_heartbeat_async(
        con,
        """
        SELECT ledger_number, timestamp
        FROM window_ledgers
        ORDER BY ledger_number
        """,
        stage="export-window-ledgers",
        heartbeat_sec=args.heartbeat_sec,
    )

    out_csv = _OUT_DIR / f"{args.output_prefix}_ledger_timestamp_map.csv"
    out_summary = _OUT_DIR / f"{args.output_prefix}_summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ledger_number", "timestamp"])
        w.writeheader()
        for ledger_number, timestamp in rows:
            w.writerow({"ledger_number": int(ledger_number), "timestamp": str(timestamp)})

    summary = {
        "markets_csv": str(markets_csv),
        "data_dir": str(args.data_dir),
        "scan_mode": args.scan_mode,
        "ledgers_files_scanned": len(ledger_files),
        "market_rows_read": n_market_rows,
        "window_utc": {
            "earliest_created_at": earliest_iso,
            "latest_end_date": latest_iso,
            "pad_hours": float(args.pad_hours),
            "padded_window_start": window_start_iso,
            "padded_window_end": window_end_iso,
        },
        "output_window_ledgers": {
            "n_ledgers": n_ledgers,
            "min_ledger_number": min_ledger,
            "max_ledger_number": max_ledger,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
        },
        "outputs": {
            "ledger_timestamp_map_csv": str(out_csv),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"scan_mode:                    {args.scan_mode}")
    print(f"ledgers_files_scanned:         {len(ledger_files)}")
    print(f"market_rows_read:             {n_market_rows}")
    print(f"window earliest_created_at:   {earliest_iso}")
    print(f"window latest_end_date:       {latest_iso}")
    print(f"window pad_hours:             {float(args.pad_hours)}")
    print(f"window padded_start:          {window_start_iso}")
    print(f"window padded_end:            {window_end_iso}")
    print(f"output ledgers in window:      {n_ledgers}")
    print(f"output ledger range:           [{min_ledger}, {max_ledger}]")
    print(f"output timestamp range:       [{min_ts}, {max_ts}]")
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
