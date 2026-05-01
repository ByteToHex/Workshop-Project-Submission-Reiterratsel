"""
Targeted sanity check for a single Parquet market slug and/or token ID
against local CTF markets/trades parquet shards.

Default target:
  will-the-fed-increase-interest-rates-by-50-bps-after-their-september-meeting

Writes outputs under:
  scripts/util/out_Markets/sanitycheck/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import duckdb

_DEFAULT_MARKET_SLUG = "will-the-fed-increase-interest-rates-by-50-bps-after-their-september-meeting"
_UTIL_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _UTIL_ROOT / "out_Markets" / "sanitycheck"


def _glob_sql(path: Path) -> str:
    return str(path).replace("\\", "/")


def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _resolve_data_dirs(root: Path) -> tuple[Path, Path]:
    markets_dir = root / "markets"
    trades_dir = root / "trades"
    if markets_dir.is_dir() and trades_dir.is_dir():
        return markets_dir, trades_dir

    alt_markets_dir = root / "parquet" / "markets"
    alt_trades_dir = root / "parquet" / "trades"
    if alt_markets_dir.is_dir() and alt_trades_dir.is_dir():
        return alt_markets_dir, alt_trades_dir

    return markets_dir, trades_dir


def _first_files(directory: Path, pattern: str, limit: int | None) -> list[Path]:
    files = sorted(directory.glob(pattern))
    if limit is None or limit <= 0:
        return files
    return files[:limit]


def _extract_token_ids(raw: object) -> list[str]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s in {"[]", "null", "None"}:
        return []
    try:
        data = json.loads(s)
    except Exception:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def _add(v: object) -> None:
        token_id = str(v).strip()
        if not token_id or token_id in seen:
            return
        seen.add(token_id)
        out.append(token_id)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                _add(item)
            elif isinstance(item, dict):
                for key in ("token_id", "tokenId", "id"):
                    if key in item:
                        _add(item[key])
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, str):
                _add(value)
            elif isinstance(value, dict):
                for key in ("token_id", "tokenId", "id"):
                    if key in value:
                        _add(value[key])
    return out


def _safe_json_loads(raw: object) -> tuple[str, str]:
    if raw is None:
        return "null", ""
    s = str(raw).strip()
    if not s:
        return "empty_string", ""
    try:
        parsed = json.loads(s)
    except Exception as e:
        return "json_parse_error", str(e)
    return type(parsed).__name__, ""


def _extract_digit_runs(raw: object) -> list[str]:
    if raw is None:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in re.findall(r"\d{20,}", str(raw)):
        if match in seen:
            continue
        seen.add(match)
        out.append(match)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-slug", default=_DEFAULT_MARKET_SLUG, help="Exact market slug to inspect.")
    parser.add_argument("--token-id", default="", help="Optional explicit token ID to inspect.")
    parser.add_argument("--market-id", default="", help="Optional exact market id to inspect.")
    parser.add_argument("--question-contains", default="", help="Optional case-insensitive question substring.")
    parser.add_argument(
        "--raw-field-contains",
        default="",
        help="Optional case-insensitive substring to search inside clob_token_ids raw text.",
    )
    parser.add_argument(
        "--print-found-field",
        action="store_true",
        help="Print raw clob_token_ids debug details for every matched market row.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/parquet"),
        help="Parquet root containing markets/ and trades/ or parquet/markets and parquet/trades.",
    )
    parser.add_argument(
        "--market-sample-files",
        type=int,
        default=None,
        help="Optional limit for markets_*.parquet files.",
    )
    parser.add_argument(
        "--trade-sample-files",
        type=int,
        default=None,
        help="Optional limit for trades_*.parquet files.",
    )
    args = parser.parse_args(argv)

    markets_dir, trades_dir = _resolve_data_dirs(args.data_dir)
    if not markets_dir.is_dir():
        print(f"No markets directory: {markets_dir.resolve()}")
        return 1
    if not trades_dir.is_dir():
        print(f"No trades directory: {trades_dir.resolve()}")
        return 2

    market_files = _first_files(markets_dir, "markets_*.parquet", args.market_sample_files)
    trade_files = _first_files(trades_dir, "trades_*.parquet", args.trade_sample_files)
    if not market_files:
        print(f"No markets_*.parquet under {markets_dir.resolve()}")
        return 3
    if not trade_files:
        print(f"No trades_*.parquet under {trades_dir.resolve()}")
        return 4

    market_file_sql = ", ".join(_sql_quote(_glob_sql(path)) for path in market_files)
    trade_file_sql = ", ".join(_sql_quote(_glob_sql(path)) for path in trade_files)
    markets_from = f"read_parquet([{market_file_sql}], union_by_name=true)"
    trades_from = f"read_parquet([{trade_file_sql}], union_by_name=true)"

    con = duckdb.connect(database=":memory:")
    market_schema = con.execute(f"DESCRIBE SELECT * FROM {markets_from} LIMIT 0").fetchall()
    market_cols = {str(row[0]) for row in market_schema}
    trade_schema = con.execute(f"DESCRIBE SELECT * FROM {trades_from} LIMIT 0").fetchall()
    trade_cols = {str(row[0]) for row in trade_schema}

    if "slug" not in market_cols:
        print("Markets parquet missing `slug`.")
        return 5
    if "clob_token_ids" not in market_cols:
        print("Markets parquet missing `clob_token_ids`.")
        return 6
    if not {"maker_asset_id", "taker_asset_id"}.issubset(trade_cols):
        print("Trades parquet missing maker/taker asset id columns.")
        return 7

    selected_market_cols = [
        "slug",
        "id" if "id" in market_cols else "NULL AS id",
        "question" if "question" in market_cols else "NULL AS question",
        "conditionId" if "conditionId" in market_cols else "NULL AS conditionId",
        "active" if "active" in market_cols else "NULL AS active",
        "closed" if "closed" in market_cols else "NULL AS closed",
        "archived" if "archived" in market_cols else "NULL AS archived",
        "endDate" if "endDate" in market_cols else "NULL AS endDate",
        "typeof(clob_token_ids) AS clob_token_ids_type",
        "clob_token_ids",
    ]
    market_predicates: list[str] = []
    if args.market_slug:
        market_predicates.append(f"slug = {_sql_quote(args.market_slug)}")
    if args.market_id and "id" in market_cols:
        market_predicates.append(f"cast(id AS VARCHAR) = {_sql_quote(args.market_id)}")
    if args.question_contains and "question" in market_cols:
        market_predicates.append(f"lower(cast(question AS VARCHAR)) LIKE {_sql_quote('%' + args.question_contains.lower() + '%')}")
    if args.raw_field_contains:
        market_predicates.append(
            f"lower(cast(clob_token_ids AS VARCHAR)) LIKE {_sql_quote('%' + args.raw_field_contains.lower() + '%')}"
        )
    if not market_predicates:
        market_predicates.append("1 = 1")
    market_rows_raw = con.execute(
        f"""
        SELECT {", ".join(selected_market_cols)}
        FROM {markets_from}
        WHERE {" OR ".join(market_predicates)}
        """
    ).fetchall()

    market_rows: list[dict[str, Any]] = []
    inferred_token_ids: list[str] = []
    for row in market_rows_raw:
        clob_type = str(row[8] or "")
        clob_raw = row[9]
        token_ids = _extract_token_ids(clob_raw)
        json_shape, json_error = _safe_json_loads(clob_raw)
        digit_runs = _extract_digit_runs(clob_raw)
        inferred_token_ids.extend(token_ids)
        market_rows.append(
            {
                "slug": str(row[0] or ""),
                "market_id": str(row[1] or ""),
                "question": str(row[2] or ""),
                "condition_id": str(row[3] or ""),
                "active": "" if row[4] is None else str(row[4]),
                "closed": "" if row[5] is None else str(row[5]),
                "archived": "" if row[6] is None else str(row[6]),
                "end_date": str(row[7] or ""),
                "clob_token_ids_type": clob_type,
                "clob_token_ids_raw": str(clob_raw or ""),
                "clob_token_ids_json_shape": json_shape,
                "clob_token_ids_json_error": json_error,
                "digit_runs_found_in_raw": ";".join(digit_runs),
                "parsed_token_ids": ";".join(token_ids),
            }
        )
        if args.print_found_field:
            print("---- matched market row ----")
            print(f"slug: {row[0] or ''}")
            print(f"market_id: {row[1] or ''}")
            print(f"question: {row[2] or ''}")
            print(f"clob_token_ids_type: {clob_type}")
            print(f"clob_token_ids_json_shape: {json_shape}")
            if json_error:
                print(f"clob_token_ids_json_error: {json_error}")
            print(f"clob_token_ids_raw: {str(clob_raw or '')}")
            print(f"digit_runs_found_in_raw: {';'.join(digit_runs)}")
            print(f"parsed_token_ids: {';'.join(token_ids)}")

    token_ids_to_check: list[str] = []
    seen_tokens: set[str] = set()
    for token_id in ([args.token_id] if args.token_id else []) + inferred_token_ids:
        tid = str(token_id).strip()
        if not tid or tid in seen_tokens:
            continue
        seen_tokens.add(tid)
        token_ids_to_check.append(tid)

    token_values_sql = ", ".join(f"({_sql_quote(token_id)})" for token_id in token_ids_to_check)
    trade_summary_rows: list[dict[str, Any]] = []
    trade_sample_rows: list[dict[str, Any]] = []
    if token_ids_to_check:
        ledger_expr = "min(ledger_number) AS min_ledger_number, max(ledger_number) AS max_ledger_number," if "ledger_number" in trade_cols else ""
        time_expr = (
            "min(cast(timestamp AS VARCHAR)) AS min_timestamp, max(cast(timestamp AS VARCHAR)) AS max_timestamp,"
            if "timestamp" in trade_cols
            else "NULL AS min_timestamp, NULL AS max_timestamp,"
        )
        sample_extra_cols = []
        if "ledger_number" in trade_cols:
            sample_extra_cols.append("cast(ledger_number AS VARCHAR) AS ledger_number")
        if "timestamp" in trade_cols:
            sample_extra_cols.append("cast(timestamp AS VARCHAR) AS timestamp")
        if "transaction_hash" in trade_cols:
            sample_extra_cols.append("cast(transaction_hash AS VARCHAR) AS transaction_hash")
        if "log_index" in trade_cols:
            sample_extra_cols.append("cast(log_index AS VARCHAR) AS log_index")
        sample_extra_sql = ", " + ", ".join(sample_extra_cols) if sample_extra_cols else ""

        trade_summary_raw = con.execute(
            f"""
            WITH wanted(token_id) AS (
                VALUES {token_values_sql}
            ),
            hits AS (
                SELECT
                    w.token_id,
                    CASE WHEN cast(tr.maker_asset_id AS VARCHAR) = w.token_id THEN 1 ELSE 0 END AS maker_hit,
                    CASE WHEN cast(tr.taker_asset_id AS VARCHAR) = w.token_id THEN 1 ELSE 0 END AS taker_hit
                    {", cast(tr.ledger_number AS BIGINT) AS ledger_number" if "ledger_number" in trade_cols else ""}
                    {", tr.timestamp" if "timestamp" in trade_cols else ""}
                FROM {trades_from} tr
                JOIN wanted w
                  ON cast(tr.maker_asset_id AS VARCHAR) = w.token_id
                  OR cast(tr.taker_asset_id AS VARCHAR) = w.token_id
            )
            SELECT
                token_id,
                count(*)::BIGINT AS n_trade_rows,
                sum(maker_hit)::BIGINT AS n_maker_hits,
                sum(taker_hit)::BIGINT AS n_taker_hits,
                {ledger_expr}
                {time_expr}
                1 AS _sentinel
            FROM hits
            GROUP BY token_id
            ORDER BY token_id
            """
        ).fetchall()
        for row in trade_summary_raw:
            trade_summary_rows.append(
                {
                    "token_id": str(row[0] or ""),
                    "n_trade_rows": int(row[1] or 0),
                    "n_maker_hits": int(row[2] or 0),
                    "n_taker_hits": int(row[3] or 0),
                    "min_ledger_number": "" if len(row) < 5 or row[4] is None else str(row[4]),
                    "max_ledger_number": "" if len(row) < 6 or row[5] is None else str(row[5]),
                    "min_timestamp": "" if len(row) < 7 or row[6] is None else str(row[6]),
                    "max_timestamp": "" if len(row) < 8 or row[7] is None else str(row[7]),
                }
            )

        trade_sample_raw = con.execute(
            f"""
            WITH wanted(token_id) AS (
                VALUES {token_values_sql}
            )
            SELECT
                w.token_id,
                cast(tr.maker_asset_id AS VARCHAR) AS maker_asset_id,
                cast(tr.taker_asset_id AS VARCHAR) AS taker_asset_id
                {sample_extra_sql}
            FROM {trades_from} tr
            JOIN wanted w
              ON cast(tr.maker_asset_id AS VARCHAR) = w.token_id
              OR cast(tr.taker_asset_id AS VARCHAR) = w.token_id
            LIMIT 50
            """
        ).fetchall()
        sample_headers = ["token_id", "maker_asset_id", "taker_asset_id"]
        sample_headers.extend([col.split(" AS ")[-1] for col in sample_extra_cols])
        for row in trade_sample_raw:
            trade_sample_rows.append({sample_headers[i]: "" if row[i] is None else str(row[i]) for i in range(len(sample_headers))})

    market_exists = len(market_rows) > 0
    any_trade_hits = any(int(row.get("n_trade_rows") or 0) > 0 for row in trade_summary_rows)
    summary = {
        "target_market_slug": args.market_slug,
        "explicit_token_id": args.token_id,
        "market_id_filter": args.market_id,
        "question_contains_filter": args.question_contains,
        "raw_field_contains_filter": args.raw_field_contains,
        "markets_dir": str(markets_dir),
        "trades_dir": str(trades_dir),
        "markets_files_scanned": len(market_files),
        "trades_files_scanned": len(trade_files),
        "market_exists_in_markets_parquet": market_exists,
        "n_market_rows_found": len(market_rows),
        "inferred_token_ids_from_market_slug": inferred_token_ids,
        "token_ids_checked": token_ids_to_check,
        "any_trade_hits_for_checked_tokens": any_trade_hits,
        "n_trade_summary_rows": len(trade_summary_rows),
    }

    slug_tag = args.market_slug.replace("/", "_").replace("\\", "_")
    base_name = slug_tag[:120] if slug_tag else "token_only"
    out_summary = _OUT_DIR / f"{base_name}_summary.json"
    out_market = _OUT_DIR / f"{base_name}_market_rows.csv"
    out_trade_summary = _OUT_DIR / f"{base_name}_trade_hits.csv"
    out_trade_samples = _OUT_DIR / f"{base_name}_trade_samples.csv"

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(
        out_market,
        market_rows,
        [
            "slug",
            "market_id",
            "question",
            "condition_id",
            "active",
            "closed",
            "archived",
            "end_date",
            "clob_token_ids_type",
            "clob_token_ids_raw",
            "clob_token_ids_json_shape",
            "clob_token_ids_json_error",
            "digit_runs_found_in_raw",
            "parsed_token_ids",
        ],
    )
    _write_csv(
        out_trade_summary,
        trade_summary_rows,
        [
            "token_id",
            "n_trade_rows",
            "n_maker_hits",
            "n_taker_hits",
            "min_ledger_number",
            "max_ledger_number",
            "min_timestamp",
            "max_timestamp",
        ],
    )
    _write_csv(
        out_trade_samples,
        trade_sample_rows,
        list(trade_sample_rows[0].keys()) if trade_sample_rows else ["token_id", "maker_asset_id", "taker_asset_id"],
    )

    print(f"market_exists_in_markets_parquet: {market_exists}")
    print(f"inferred_token_ids_from_market_slug: {inferred_token_ids}")
    print(f"token_ids_checked: {token_ids_to_check}")
    print(f"any_trade_hits_for_checked_tokens: {any_trade_hits}")
    print(f"wrote: {out_summary}")
    print(f"wrote: {out_market}")
    print(f"wrote: {out_trade_summary}")
    print(f"wrote: {out_trade_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
