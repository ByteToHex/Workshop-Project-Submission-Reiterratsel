from __future__ import annotations

from pathlib import Path

import duckdb


SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_ROOT = SCRIPT_DIR.parent
CONSOLIDATED_ROOT = METRICS_ROOT.parent
SOURCE_DUCKDB = CONSOLIDATED_ROOT / "IO" / "out" / "_annual_warehouse" / "fundamentals.duckdb"


def main() -> None:
    con = duckdb.connect(str(SOURCE_DUCKDB), read_only=True)
    try:
        coverage = con.execute(
            """
            WITH expected AS (
                SELECT
                    p.ticker,
                    COUNT(*) * (SELECT COUNT(*) FROM reit_metrics.dim_metric) AS expected_rows
                FROM reit_metrics.dim_period p
                GROUP BY p.ticker
            ),
            actual AS (
                SELECT ticker, COUNT(*) AS actual_rows
                FROM reit_metrics.fact_metric_value
                GROUP BY ticker
            )
            SELECT
                e.ticker,
                e.expected_rows,
                a.actual_rows,
                a.actual_rows = e.expected_rows AS is_complete
            FROM expected e
            JOIN actual a USING (ticker)
            ORDER BY e.ticker
            """
        ).fetchdf()

        blanks = con.execute(
            """
            SELECT
                metric_code,
                calc_status,
                COUNT(*) AS row_count,
                SUM(CASE WHEN metric_value IS NULL THEN 1 ELSE 0 END) AS blank_value_count
            FROM reit_metrics.fact_metric_value
            GROUP BY metric_code, calc_status
            ORDER BY metric_code, calc_status
            """
        ).fetchdf()

        by_ticker_blank = con.execute(
            """
            SELECT
                ticker,
                COUNT(*) AS total_metric_rows,
                SUM(CASE WHEN metric_value IS NULL THEN 1 ELSE 0 END) AS blank_metric_rows,
                ROUND(
                    100.0 * SUM(CASE WHEN metric_value IS NULL THEN 1 ELSE 0 END) / COUNT(*),
                    2
                ) AS blank_pct
            FROM reit_metrics.fact_metric_value
            GROUP BY ticker
            ORDER BY ticker
            """
        ).fetchdf()

        print("Coverage by ticker")
        print(coverage.to_string(index=False))
        print()

        print("Blank metrics by code and status")
        print(blanks.to_string(index=False))
        print()

        print("Blank metrics by ticker")
        print(by_ticker_blank.to_string(index=False))
    finally:
        con.close()


if __name__ == "__main__":
    main()
