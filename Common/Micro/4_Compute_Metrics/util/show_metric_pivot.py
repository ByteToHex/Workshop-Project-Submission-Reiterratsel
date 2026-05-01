from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_ROOT = SCRIPT_DIR.parent
CONSOLIDATED_ROOT = METRICS_ROOT.parent
SOURCE_DUCKDB = CONSOLIDATED_ROOT / "IO" / "out" / "_annual_warehouse" / "fundamentals.duckdb"
METRICS_PARQUET = CONSOLIDATED_ROOT / "IO" / "out" / "_annual_warehouse" / "parquet" / "metrics.parquet"

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "A17U"
DEFAULT_CSV_OUT = str(CONSOLIDATED_ROOT / "IO" / "out" / "metric_pivot.csv")


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _load_metrics_df() -> pd.DataFrame:
    if METRICS_PARQUET.exists():
        return pd.read_parquet(METRICS_PARQUET)
    raise FileNotFoundError(f"Metrics parquet not found: {METRICS_PARQUET}")


def build_pivot_sql(ticker: str) -> str:
    df = _load_metrics_df()
    sub = df.loc[df["ticker"] == ticker].copy()
    if sub.empty:
        raise ValueError(f"No annual periods found for ticker: {ticker}")
    parquet_ref = METRICS_PARQUET.as_posix()
    source_ref = f"read_parquet('{parquet_ref}')"
    sub = sub[["source_period_label", "sort_key"]].drop_duplicates().sort_values("sort_key")

    period_exprs = []
    for period_label in sub["source_period_label"].tolist():
        safe_literal = _quote_sql_string(str(period_label))
        safe_alias = str(period_label).replace('"', '""')
        period_exprs.append(
            f'MAX(CASE WHEN source_period_label = {safe_literal} THEN metric_value END) AS "{safe_alias}"'
        )

    period_sql = ",\n    ".join(period_exprs)
    return f"""
SELECT
    ticker,
    MAX(reit_name) AS reit_name,
    metric_code,
    MAX(metric_name) AS metric_name,
    MAX(unit_type) AS unit_type,
    {period_sql}
FROM {source_ref}
WHERE ticker = {_quote_sql_string(ticker)}
GROUP BY
    ticker,
    metric_code
ORDER BY
    metric_code
""".strip()


def _format_numeric(value: float, unit_type: str) -> str:
    if unit_type == "flag":
        return "TRUE" if value == 1.0 else "FALSE"
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _format_display_value(row: pd.Series) -> str:
    metric_value = row["metric_value"]
    value_text = row.get("value_text")
    unit_type = str(row.get("unit_type") or "")
    calc_status = str(row.get("calc_status") or "")

    if pd.notna(value_text) and str(value_text).strip():
        text = str(value_text).strip()
    elif pd.isna(metric_value):
        text = "NULL"
    else:
        text = _format_numeric(float(metric_value), unit_type)

    if calc_status and calc_status != "OK":
        text = f"{text} [{calc_status}]"
    return text


def _build_status_summary(sub: pd.DataFrame) -> pd.DataFrame:
    flagged = sub.loc[sub["calc_status"] != "OK", ["metric_code", "sort_key", "source_period_label", "calc_status"]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=["metric_code", "status_summary"])
    flagged = flagged.sort_values(["metric_code", "sort_key"])
    flagged["piece"] = flagged["source_period_label"] + "=" + flagged["calc_status"]
    summary = flagged.groupby("metric_code", as_index=False)["piece"].agg("; ".join)
    return summary.rename(columns={"piece": "status_summary"})


def fetch_pivot_df(ticker: str) -> pd.DataFrame:
    df = _load_metrics_df()
    sub = df.loc[df["ticker"] == ticker].copy()
    if sub.empty:
        raise ValueError(f"No rows found for ticker: {ticker}")

    sub = sub.sort_values(["metric_code", "sort_key"]).reset_index(drop=True)
    sub["display_value"] = sub.apply(_format_display_value, axis=1)

    status_summary = _build_status_summary(sub)

    pivot = sub.pivot_table(
        index=["ticker", "reit_name", "metric_code", "metric_name", "unit_type"],
        columns="source_period_label",
        values="display_value",
        aggfunc="first",
        sort=False,
    ).reset_index()

    period_order = (
        sub[["source_period_label", "sort_key"]]
        .drop_duplicates()
        .sort_values("sort_key")["source_period_label"]
        .tolist()
    )
    fixed_cols = ["ticker", "reit_name", "metric_code", "metric_name", "unit_type"]
    pivot = pivot[fixed_cols + period_order]
    pivot = pivot.merge(status_summary, on="metric_code", how="left")
    pivot["status_summary"] = pivot["status_summary"].fillna("")
    pivot = pivot[fixed_cols + ["status_summary"] + period_order]
    return pivot.sort_values("metric_code").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show metrics as a pivot table for one REIT ticker.")
    parser.add_argument("ticker", nargs="?", help="Ticker to display, for example A17U")
    parser.add_argument(
        "--sql-only",
        action="store_true",
        help="Print the generated SQL only, without executing it.",
    )
    parser.add_argument(
        "--csv-out",
        help="Optional path to export the pivot result as CSV.",
    )
    args = parser.parse_args()

    ticker = (args.ticker or DEFAULT_TICKER).strip().upper()
    sql = build_pivot_sql(ticker)

    if args.sql_only:
        print(sql)
        return

    df = fetch_pivot_df(ticker)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print(df.to_string(index=False))

    csv_out = args.csv_out or DEFAULT_CSV_OUT
    if csv_out:
        out_path = Path(csv_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print()
        print(f"CSV written: {out_path}")


if __name__ == "__main__":
    main()
