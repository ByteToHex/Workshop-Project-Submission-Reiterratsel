from __future__ import annotations

import calendar
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_ROOT = SCRIPT_DIR
CONSOLIDATED_ROOT = SCRIPT_DIR.parent
SOURCE_DUCKDB = CONSOLIDATED_ROOT / "IO" / "out" / "_annual_warehouse" / "fundamentals.duckdb"
SOURCE_METRIC_SPEC = SCRIPT_DIR / "Data_Dict_Reit_Metrics.md"
SORA_3M_CSV = SCRIPT_DIR / "src" / "sora_3m_daily.csv"
OUTPUT_PARQUET = CONSOLIDATED_ROOT / "IO" / "out" / "_annual_warehouse" / "parquet" / "metrics.parquet"


REIT_METADATA: dict[str, dict[str, str]] = {
    "C38U": {
        "reit_name": "CapitaLand Integrated Commercial Trust",
        "sector": "Retail + Office",
        "health_bucket": "Operational",
        "notes": "Only 1Q and 3Q in source notes.",
    },
    "T82U": {
        "reit_name": "Suntec REIT",
        "sector": "Retail + Office",
        "health_bucket": "Operational",
        "notes": "",
    },
    "N2IU": {
        "reit_name": "Mapletree Pan Asia Commercial Trust",
        "sector": "Retail + Office",
        "health_bucket": "Operational",
        "notes": "Ticker note source listed N21U, warehouse uses N2IU.",
    },
    "J69U": {
        "reit_name": "Frasers Centrepoint Trust",
        "sector": "Suburban",
        "health_bucket": "Operational",
        "notes": "",
    },
    "A17U": {
        "reit_name": "CapitaLand Ascendas REIT",
        "sector": "Industrial",
        "health_bucket": "Operational",
        "notes": "",
    },
    "AJBU": {
        "reit_name": "Keppel DC REIT",
        "sector": "Data Center",
        "health_bucket": "Operational",
        "notes": "",
    },
    "ME8U": {
        "reit_name": "Mapletree Industrial Trust",
        "sector": "Data Center",
        "health_bucket": "Operational",
        "notes": "",
    },
    "M44U": {
        "reit_name": "Mapletree Logistics Trust",
        "sector": "Logistic",
        "health_bucket": "Operational",
        "notes": "",
    },
    "BUOU": {
        "reit_name": "Frasers Logistics & Commercial Trust",
        "sector": "Logistic",
        "health_bucket": "Operational",
        "notes": "",
    },
    "D5IU": {
        "reit_name": "Lippo Malls Indonesia Retail Trust",
        "sector": "Retail",
        "health_bucket": "Collapsed",
        "notes": "Distressed cohort in source metric doc.",
    },
    "OXMU": {
        "reit_name": "Prime US REIT",
        "sector": "Office",
        "health_bucket": "Collapsed",
        "notes": "Distressed cohort in source metric doc.",
    },
    "BTOU": {
        "reit_name": "Manulife US REIT",
        "sector": "Office",
        "health_bucket": "Collapsed",
        "notes": "Distressed cohort in source metric doc.",
    },
    "BWCU": {
        "reit_name": "EC World",
        "sector": "Logistics",
        "health_bucket": "Collapsed",
        "notes": "Distressed cohort in source metric doc.",
    },
    "CMOU": {
        "reit_name": "KORE US REIT",
        "sector": "Office",
        "health_bucket": "Collapsed",
        "notes": "Distressed cohort in source metric doc.",
    },
    "M1GU": {
        "reit_name": "Sabana Industrial REIT",
        "sector": "Industrial",
        "health_bucket": "Control",
        "notes": "Use as control based on source metric doc note.",
    },
}


METRIC_DEFINITIONS: list[dict[str, Any]] = [
    {
        "metric_code": "ICR",
        "metric_name": "Interest Coverage Ratio",
        "formula_short": "EBITDA / abs(Total Interest Expense)",
        "unit_type": "ratio",
        "higher_is_better": True,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Uses EBITDA and total interest expense from annual fundamentals.",
    },
    {
        "metric_code": "GEARING",
        "metric_name": "Gearing Ratio",
        "formula_short": "Total Debt / Total Assets",
        "unit_type": "ratio",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_02",
        "requires_external": False,
        "description": "Uses raw debt and assets, with fallback to warehouse debt-to-assets ratio.",
    },
    {
        "metric_code": "NET_DEBT_EBITDA",
        "metric_name": "Net Debt / EBITDA",
        "formula_short": "Net Debt / EBITDA",
        "unit_type": "ratio",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Uses annual net debt and EBITDA.",
    },
    {
        "metric_code": "REFI_RISK",
        "metric_name": "Refinancing Risk Ratio",
        "formula_short": "Short Term Debt / Total Debt",
        "unit_type": "ratio",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Uses annual short term debt and total debt.",
    },
    {
        "metric_code": "FFO_YOY",
        "metric_name": "FFO YoY Growth",
        "formula_short": "(FFO_t - FFO_t-1) / FFO_t-1",
        "unit_type": "pct",
        "higher_is_better": True,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Year-over-year FFO growth against the previous annual period for the same ticker.",
    },
    {
        "metric_code": "IMPLICIT_COD",
        "metric_name": "Implicit Cost of Debt",
        "formula_short": "abs(Interest Paid) / Total Debt",
        "unit_type": "pct",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Uses annual cash interest paid and annual total debt.",
    },
    {
        "metric_code": "REV_CONC_TOPSEG",
        "metric_name": "Revenue Concentration (Top Segment Share)",
        "formula_short": "Largest By_Source segment / denominator",
        "unit_type": "ratio",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_05",
        "requires_external": False,
        "description": "Uses revenue section group 0-By_Source, comparing segment sum against annual total revenue.",
    },
    {
        "metric_code": "PAYOUT_RATIO",
        "metric_name": "Payout Ratio",
        "formula_short": "abs(Total Cash Dividends Paid) / FFO",
        "unit_type": "pct",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Uses annual dividends paid and FFO.",
    },
    {
        "metric_code": "FFO_COVERAGE",
        "metric_name": "FFO Coverage Margin",
        "formula_short": "(FFO - abs(Total Cash Dividends Paid)) / FFO",
        "unit_type": "pct",
        "higher_is_better": True,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Measures dividend headroom against FFO.",
    },
    {
        "metric_code": "PNAV_PROXY",
        "metric_name": "P/NAV Proxy (Price to Book)",
        "formula_short": "Price to Book Ratio",
        "unit_type": "multiple",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_02",
        "requires_external": False,
        "description": "Directly uses the annual price-to-book ratio from statistics.",
    },
    {
        "metric_code": "FFO_YIELD_EQ",
        "metric_name": "FFO Yield [Equity]",
        "formula_short": "FFO / (Net Income * PE Ratio)",
        "unit_type": "pct",
        "higher_is_better": True,
        "source_schema_hint": "SCHEMA_01, SCHEMA_02",
        "requires_external": False,
        "description": "Uses a market-cap proxy from net income times price-to-earnings ratio.",
    },
    {
        "metric_code": "FFO_YIELD_CAP",
        "metric_name": "FFO Yield [Capital]",
        "formula_short": "FFO / Enterprise Value",
        "unit_type": "pct",
        "higher_is_better": True,
        "source_schema_hint": "SCHEMA_02",
        "requires_external": False,
        "description": "Uses annual enterprise value from statistics.",
    },
    {
        "metric_code": "LEVERAGE_PREMIUM",
        "metric_name": "Leverage Premium",
        "formula_short": "FFO Yield [Equity] - FFO Yield [Capital]",
        "unit_type": "pct",
        "higher_is_better": True,
        "source_schema_hint": "",
        "requires_external": False,
        "description": "Difference between the two FFO yield variants.",
    },
    {
        "metric_code": "REAL_YIELD_SORA",
        "metric_name": "Real Yield (Opportunity Cost)",
        "formula_short": "Dividend Yield (FY) % - SORA Compounded 3M",
        "unit_type": "pct_point",
        "higher_is_better": True,
        "source_schema_hint": "SCHEMA_03",
        "requires_external": False,
        "description": "Uses annual dividend yield and the latest SORA 3M value on or before fiscal year end.",
    },
    {
        "metric_code": "UNIT_DILUTION",
        "metric_name": "Unit Dilution Rate",
        "formula_short": "(Diluted Shares_t - Diluted Shares_t-1) / Diluted Shares_t-1",
        "unit_type": "pct",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Year-over-year change in diluted shares outstanding.",
    },
    {
        "metric_code": "OPEX_INTENSITY",
        "metric_name": "OpEx Intensity",
        "formula_short": "abs(Total Operating Expenses) / Total Revenue",
        "unit_type": "pct",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Measures operating expense burden against revenue.",
    },
    {
        "metric_code": "CAPEX_DRAG",
        "metric_name": "Capex Drag Diagnostic",
        "formula_short": "(abs(Dividends) / FCF) > (abs(Dividends) / FFO)",
        "unit_type": "flag",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "True when dividend burden on FCF is worse than on FFO.",
    },
    {
        "metric_code": "NONRECUR_SHARE",
        "metric_name": "Non-Recurring Income Share",
        "formula_short": "Unusual Income/Expense / Net Income",
        "unit_type": "pct",
        "higher_is_better": False,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Keeps the source sign to show positive or negative unusual contribution.",
    },
    {
        "metric_code": "DSCR",
        "metric_name": "DSCR",
        "formula_short": "FFO / (abs(Interest Paid) + Short Term Debt)",
        "unit_type": "ratio",
        "higher_is_better": True,
        "source_schema_hint": "SCHEMA_01",
        "requires_external": False,
        "description": "Uses FFO, cash interest paid, and short term debt.",
    },
]


MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def _parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "--", "---", "N/A", "na", "None"}:
        return None
    text = text.replace(",", "")
    text = text.replace("%", "")
    text = text.replace("x", "")
    text = text.replace("(", "-").replace(")", "")
    text = text.replace("\u2212", "-")
    text = text.replace("\u2014", "")
    text = text.strip()
    if not text:
        return None

    multiplier = 1.0
    unit_match = re.search(r"\s([KMBT])$", text)
    if unit_match:
        unit = unit_match.group(1)
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[unit]
        text = text[: unit_match.start()].strip()

    if text in {"", "-", "."}:
        return None

    try:
        return float(text) * multiplier
    except ValueError:
        return None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0 or math.isclose(denominator, 0.0, abs_tol=1e-12):
        return None
    return numerator / denominator


def _require_positive(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return None
    return value


def _parse_detailed_period_label(period: str) -> dict[str, Any] | None:
    match = re.match(r"^(\d{4})\s*/\s*([A-Za-z]{3})\s+(\d{4})$", period)
    if not match:
        return None
    fiscal_year = int(match.group(1))
    month_abbrev = match.group(2).title()
    fiscal_year_end_year = int(match.group(3))
    fiscal_year_end_month = MONTH_MAP[month_abbrev]
    fiscal_year_end_day = calendar.monthrange(fiscal_year_end_year, fiscal_year_end_month)[1]
    fiscal_year_end_date = pd.Timestamp(
        year=fiscal_year_end_year,
        month=fiscal_year_end_month,
        day=fiscal_year_end_day,
    ).date()
    return {
        "fiscal_year": fiscal_year,
        "fiscal_year_end_year": fiscal_year_end_year,
        "fiscal_year_end_month": fiscal_year_end_month,
        "fiscal_year_end_date": fiscal_year_end_date,
        "display_year": fiscal_year_end_year,
        "period_kind": "FY",
        "is_annual": True,
        "is_ttm": False,
        "is_current": False,
    }


def _load_source_financials(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT
            f.ticker,
            f.period,
            f.currency,
            f.row_id,
            s.section,
            s.label,
            s.group_output_label,
            f.value
        FROM financials f
        JOIN schema_rows s USING (row_id)
        """
    ).fetchdf()
    df["value_num"] = df["value"].map(_parse_numeric)
    return df


def _build_dim_reit(source_df: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(source_df["ticker"].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        meta = REIT_METADATA.get(
            ticker,
            {
                "reit_name": ticker,
                "sector": "",
                "health_bucket": "Unknown",
                "notes": "",
            },
        )
        rows.append(
            {
                "ticker": ticker,
                "reit_name": meta["reit_name"],
                "sector": meta["sector"],
                "health_bucket": meta["health_bucket"],
                "notes": meta["notes"],
            }
        )
    return pd.DataFrame(rows)


def _build_dim_period(source_df: pd.DataFrame) -> pd.DataFrame:
    periods = sorted(source_df["period"].dropna().unique().tolist())
    detailed_period_rows: list[dict[str, Any]] = []
    for ticker in sorted(source_df["ticker"].dropna().unique().tolist()):
        ticker_periods = sorted(source_df.loc[source_df["ticker"] == ticker, "period"].dropna().unique().tolist())
        for period in ticker_periods:
            parsed = _parse_detailed_period_label(period)
            if not parsed:
                continue
            row = {
                "ticker": ticker,
                "source_period_label": period,
                **parsed,
            }
            detailed_period_rows.append(row)

    dim_period = pd.DataFrame(detailed_period_rows)
    if dim_period.empty:
        raise RuntimeError("No detailed annual period labels were found in the fundamentals warehouse.")

    dim_period = dim_period.sort_values(
        by=["ticker", "fiscal_year_end_date", "source_period_label"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    dim_period["sort_key"] = dim_period.groupby("ticker").cumcount() + 1
    dim_period["period_id"] = dim_period.index + 1
    return dim_period[
        [
            "period_id",
            "ticker",
            "source_period_label",
            "period_kind",
            "fiscal_year",
            "fiscal_year_end_month",
            "fiscal_year_end_year",
            "fiscal_year_end_date",
            "display_year",
            "sort_key",
            "is_annual",
            "is_ttm",
            "is_current",
        ]
    ]


def _build_dim_metric() -> pd.DataFrame:
    return pd.DataFrame(METRIC_DEFINITIONS)


def _build_lookups(source_df: pd.DataFrame) -> dict[str, Any]:
    annual_mask = source_df["period"].map(lambda p: _parse_detailed_period_label(str(p)) is not None)
    annual_df = source_df.loc[annual_mask].copy()

    yearly_mask = source_df["period"].astype(str).str.fullmatch(r"\d{4}")
    yearly_df = source_df.loc[yearly_mask].copy()
    yearly_df["fiscal_year"] = yearly_df["period"].astype(int)

    annual_lookup_num = annual_df.set_index(["ticker", "period", "label"])["value_num"].to_dict()
    annual_lookup_text = annual_df.set_index(["ticker", "period", "label"])["value"].to_dict()

    yearly_lookup_num = yearly_df.set_index(["ticker", "fiscal_year", "label"])["value_num"].to_dict()
    yearly_lookup_text = yearly_df.set_index(["ticker", "fiscal_year", "label"])["value"].to_dict()

    revenue_source_df = yearly_df.loc[
        (yearly_df["section"] == "revenue") & (yearly_df["group_output_label"] == "0-By_Source")
    ].copy()

    return {
        "annual_lookup_num": annual_lookup_num,
        "annual_lookup_text": annual_lookup_text,
        "yearly_lookup_num": yearly_lookup_num,
        "yearly_lookup_text": yearly_lookup_text,
        "revenue_source_df": revenue_source_df,
    }


def _load_sora_3m() -> pd.DataFrame:
    df = pd.read_csv(SORA_3M_CSV)
    df["value_date"] = pd.to_datetime(df["value_date"]).dt.date
    df["sora_3m"] = pd.to_numeric(df["sora_3m"], errors="coerce")
    df = df.dropna(subset=["value_date", "sora_3m"]).sort_values("value_date").reset_index(drop=True)
    return df


def _build_external_series(dim_period: pd.DataFrame, sora_3m_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float | None]]:
    left = dim_period[["period_id", "fiscal_year_end_date"]].sort_values("fiscal_year_end_date").copy()
    left["fiscal_year_end_date"] = pd.to_datetime(left["fiscal_year_end_date"])

    right = sora_3m_df.copy()
    right["value_date"] = pd.to_datetime(right["value_date"])

    merged = pd.merge_asof(
        left.sort_values("fiscal_year_end_date"),
        right.sort_values("value_date"),
        left_on="fiscal_year_end_date",
        right_on="value_date",
        direction="backward",
    )
    merged["source_name"] = "sora_3m_daily.csv"
    merged["series_code"] = "SORA_3M"
    merged["value_text"] = merged["sora_3m"].map(lambda v: None if pd.isna(v) else f"{v:.4f}")
    fact_external = merged.rename(columns={"sora_3m": "value"})[
        ["series_code", "period_id", "value", "value_text", "value_date", "source_name"]
    ]
    external_lookup = {int(row.period_id): (None if pd.isna(row.value) else float(row.value)) for row in fact_external.itertuples()}
    fact_external["value_date"] = pd.to_datetime(fact_external["value_date"]).dt.date
    return fact_external, external_lookup


def _annual_num(lookups: dict[str, Any], ticker: str, period: str, label: str) -> float | None:
    value = lookups["annual_lookup_num"].get((ticker, period, label))
    return None if pd.isna(value) else value


def _yearly_num(lookups: dict[str, Any], ticker: str, fiscal_year: int, label: str) -> float | None:
    value = lookups["yearly_lookup_num"].get((ticker, fiscal_year, label))
    return None if pd.isna(value) else value


def _append_component(
    rows: list[dict[str, Any]],
    *,
    ticker: str,
    period_id: int,
    metric_code: str,
    component_role: str,
    component_name: str,
    component_value: float | None,
    source_table: str,
    source_section: str,
    source_label: str,
    source_row_id: int | None = None,
    component_text: str | None = None,
) -> None:
    rows.append(
        {
            "ticker": ticker,
            "period_id": period_id,
            "metric_code": metric_code,
            "component_role": component_role,
            "component_name": component_name,
            "component_value": component_value,
            "component_text": component_text,
            "source_table": source_table,
            "source_section": source_section,
            "source_row_id": source_row_id,
            "source_label": source_label,
        }
    )


def _combine_notes(*parts: str | None) -> str | None:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return None
    return " | ".join(cleaned)


def _net_income_is_low_denominator(net_income: float | None, total_revenue: float | None) -> bool:
    if net_income is None:
        return False
    if abs(net_income) == 0:
        return True
    if total_revenue is None or total_revenue <= 0:
        return abs(net_income) < 1.0
    return abs(net_income) < (0.02 * total_revenue)


def _compute_revenue_concentration(
    lookups: dict[str, Any],
    ticker: str,
    fiscal_year: int,
    total_revenue: float | None,
) -> tuple[float | None, str, str, list[tuple[str, float | None, str | None]]]:
    revenue_source_df: pd.DataFrame = lookups["revenue_source_df"]
    subset = revenue_source_df.loc[
        (revenue_source_df["ticker"] == ticker) & (revenue_source_df["fiscal_year"] == fiscal_year)
    ].copy()
    if subset.empty:
        return None, "MISSING_INPUT", "No 0-By_Source revenue rows found.", []

    subset["clean_value"] = subset["value_num"].apply(lambda v: None if pd.isna(v) else float(v))
    positive_subset = subset.loc[subset["clean_value"].notna() & (subset["clean_value"] > 0)].copy()
    if positive_subset.empty:
        return None, "MISSING_INPUT", "0-By_Source revenue rows were present but all values were blank or non-positive.", []

    positive_subset = positive_subset.sort_values(["clean_value", "label"], ascending=[False, True])
    top_row = positive_subset.iloc[0]
    segment_sum = float(positive_subset["clean_value"].sum())
    top_value = float(top_row["clean_value"])
    top_label = str(top_row["label"])

    denominator = segment_sum
    status = "OK"
    note = f"Top segment label={top_label}; denominator=segment_sum"

    if total_revenue is not None and total_revenue > 0:
        rel_diff = abs(segment_sum - total_revenue) / total_revenue
        if rel_diff <= 0.05:
            denominator = total_revenue
            note = f"Top segment label={top_label}; denominator=income_total_revenue; rel_diff={rel_diff:.4f}"
        else:
            status = "PARTIAL"
            note = (
                f"Top segment label={top_label}; segment_sum diverged from income total revenue "
                f"(rel_diff={rel_diff:.4f}), so denominator=segment_sum"
            )

    value = _safe_div(top_value, denominator)
    if value is not None and value > 1.0:
        value = 1.0
        status = "CLIPPED_SOURCE_SHARE"
        note = _combine_notes(
            note,
            "Raw share exceeded 1.0 because top segment slightly exceeded denominator; clipped to 1.0.",
        )
    components = [
        ("top_segment_value", top_value, top_label),
        ("segment_sum", segment_sum, "0-By_Source positive rows"),
        ("income_total_revenue", total_revenue, "Total revenue"),
    ]
    return value, status, note, components


def _compute_metric_rows(
    dim_period: pd.DataFrame,
    lookups: dict[str, Any],
    external_lookup: dict[int, float | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_codes = [m["metric_code"] for m in METRIC_DEFINITIONS]
    fact_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    periods_by_ticker = {
        ticker: grp.sort_values("sort_key").reset_index(drop=True)
        for ticker, grp in dim_period.groupby("ticker")
    }

    for ticker, ticker_periods in periods_by_ticker.items():
        prior_inputs: dict[str, float | None] | None = None
        for row in ticker_periods.itertuples(index=False):
            period_id = int(row.period_id)
            period_label = str(row.source_period_label)
            fiscal_year = int(row.fiscal_year)

            inputs = {
                "ebitda": _annual_num(lookups, ticker, period_label, "EBITDA"),
                "interest_expense": _annual_num(lookups, ticker, period_label, "Total interest expense (banks)"),
                "total_assets": _annual_num(lookups, ticker, period_label, "Total assets"),
                "total_debt": _annual_num(lookups, ticker, period_label, "Total debt"),
                "short_term_debt": _annual_num(lookups, ticker, period_label, "Short term debt"),
                "net_debt": _annual_num(lookups, ticker, period_label, "Net debt"),
                "ffo": _annual_num(lookups, ticker, period_label, "Funds from operations"),
                "interest_paid": _annual_num(lookups, ticker, period_label, "Interest paid"),
                "cash_dividends_paid": _annual_num(lookups, ticker, period_label, "Total cash dividends paid"),
                "free_cash_flow": _annual_num(lookups, ticker, period_label, "Free cash flow"),
                "unusual_income_expense": _annual_num(lookups, ticker, period_label, "Unusual income/expense"),
                "net_income": _annual_num(lookups, ticker, period_label, "Net income"),
                "diluted_shares": _annual_num(lookups, ticker, period_label, "Diluted shares outstanding"),
                "total_operating_expenses": _annual_num(lookups, ticker, period_label, "Total operating expenses"),
                "total_revenue": _annual_num(lookups, ticker, period_label, "Total revenue"),
                "price_to_book_ratio": _annual_num(lookups, ticker, period_label, "Price to book ratio"),
                "price_to_earnings_ratio": _annual_num(lookups, ticker, period_label, "Price to earnings ratio"),
                "enterprise_value": _annual_num(lookups, ticker, period_label, "Enterprise value"),
                "debt_to_assets_ratio": _annual_num(lookups, ticker, period_label, "Debt to assets ratio"),
                "dividend_yield_fy_pct": _yearly_num(lookups, ticker, fiscal_year, "Dividend yield (FY) %"),
                "sora_3m": external_lookup.get(period_id),
            }

            computed: dict[str, tuple[float | None, str, str | None]] = {}

            for metric_code in metric_codes:
                value: float | None = None
                status = "MISSING_INPUT"
                notes: str | None = None

                if metric_code == "ICR":
                    numerator = inputs["ebitda"]
                    denominator = abs(inputs["interest_expense"]) if inputs["interest_expense"] is not None else None
                    value = _safe_div(numerator, denominator)
                    status = "OK" if value is not None else "MISSING_INPUT"
                    if value is not None and numerator is not None and numerator < 0:
                        status = "NEGATIVE_BASE"
                        notes = "EBITDA is negative, so this is a distress-style ratio rather than a normal coverage ratio."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="EBITDA", component_value=numerator, source_table="financials", source_section="income", source_label="EBITDA")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="Total Interest Expense", component_value=denominator, source_table="financials", source_section="income", source_label="Total interest expense (banks)")

                elif metric_code == "GEARING":
                    raw_ratio = _safe_div(inputs["total_debt"], inputs["total_assets"])
                    fallback_ratio = inputs["debt_to_assets_ratio"]
                    if raw_ratio is not None:
                        value = raw_ratio
                        status = "OK"
                        notes = "Used Total Debt / Total Assets."
                    elif fallback_ratio is not None:
                        value = fallback_ratio
                        status = "PARTIAL"
                        notes = "Used warehouse Debt to Assets Ratio fallback."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="Total Debt", component_value=inputs["total_debt"], source_table="financials", source_section="balance", source_label="Total debt")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="Total Assets", component_value=inputs["total_assets"], source_table="financials", source_section="balance", source_label="Total assets")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="fallback", component_name="Debt to Assets Ratio", component_value=fallback_ratio, source_table="financials", source_section="statistics", source_label="Debt to assets ratio")

                elif metric_code == "NET_DEBT_EBITDA":
                    value = _safe_div(inputs["net_debt"], inputs["ebitda"])
                    status = "OK" if value is not None else "MISSING_INPUT"
                    if value is not None and inputs["ebitda"] is not None and inputs["ebitda"] <= 0:
                        status = "NEGATIVE_BASE"
                        notes = "EBITDA is non-positive, so this is a distress-style leverage ratio rather than a normal multiple."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="Net Debt", component_value=inputs["net_debt"], source_table="financials", source_section="balance", source_label="Net debt")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="EBITDA", component_value=inputs["ebitda"], source_table="financials", source_section="income", source_label="EBITDA")

                elif metric_code == "REFI_RISK":
                    value = _safe_div(inputs["short_term_debt"], inputs["total_debt"])
                    status = "OK" if value is not None else "MISSING_INPUT"
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="Short Term Debt", component_value=inputs["short_term_debt"], source_table="financials", source_section="balance", source_label="Short term debt")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="Total Debt", component_value=inputs["total_debt"], source_table="financials", source_section="balance", source_label="Total debt")

                elif metric_code == "FFO_YOY":
                    previous_ffo = None if prior_inputs is None else prior_inputs.get("ffo")
                    if previous_ffo is not None and previous_ffo != 0 and inputs["ffo"] is not None:
                        value = (inputs["ffo"] - previous_ffo) / previous_ffo
                        status = "OK"
                    notes = "Requires previous annual period."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="current", component_name="FFO", component_value=inputs["ffo"], source_table="financials", source_section="cashflow", source_label="Funds from operations")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="prior_period_input", component_name="Prior FFO", component_value=previous_ffo, source_table="financials", source_section="cashflow", source_label="Funds from operations")

                elif metric_code == "IMPLICIT_COD":
                    interest_paid_abs = abs(inputs["interest_paid"]) if inputs["interest_paid"] is not None else None
                    value = _safe_div(interest_paid_abs, inputs["total_debt"])
                    status = "OK" if value is not None else "MISSING_INPUT"
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="Interest Paid", component_value=interest_paid_abs, source_table="financials", source_section="cashflow", source_label="Interest paid")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="Total Debt", component_value=inputs["total_debt"], source_table="financials", source_section="balance", source_label="Total debt")

                elif metric_code == "REV_CONC_TOPSEG":
                    value, status, notes, components = _compute_revenue_concentration(
                        lookups=lookups,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        total_revenue=inputs["total_revenue"],
                    )
                    for component_name, component_value, component_text in components:
                        _append_component(
                            component_rows,
                            ticker=ticker,
                            period_id=period_id,
                            metric_code=metric_code,
                            component_role="input",
                            component_name=component_name,
                            component_value=component_value,
                            component_text=component_text,
                            source_table="financials",
                            source_section="revenue",
                            source_label="0-By_Source",
                        )

                elif metric_code == "PAYOUT_RATIO":
                    dividends_abs = abs(inputs["cash_dividends_paid"]) if inputs["cash_dividends_paid"] is not None else None
                    value = _safe_div(dividends_abs, inputs["ffo"])
                    status = "OK" if value is not None else "MISSING_INPUT"
                    if value is not None and inputs["ffo"] is not None and inputs["ffo"] <= 0:
                        status = "DISTRESS_BASE"
                        notes = "FFO is non-positive, so payout ratio is not behaving like a normal payout percentage."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="Total Cash Dividends Paid", component_value=dividends_abs, source_table="financials", source_section="cashflow", source_label="Total cash dividends paid")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="FFO", component_value=inputs["ffo"], source_table="financials", source_section="cashflow", source_label="Funds from operations")

                elif metric_code == "FFO_COVERAGE":
                    dividends_abs = abs(inputs["cash_dividends_paid"]) if inputs["cash_dividends_paid"] is not None else None
                    if inputs["ffo"] is not None and dividends_abs is not None and inputs["ffo"] != 0:
                        value = (inputs["ffo"] - dividends_abs) / inputs["ffo"]
                        status = "OK"
                        if inputs["ffo"] <= 0:
                            status = "DISTRESS_BASE"
                            notes = "FFO is non-positive, so coverage margin is not behaving like a normal coverage buffer."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="input", component_name="FFO", component_value=inputs["ffo"], source_table="financials", source_section="cashflow", source_label="Funds from operations")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="input", component_name="Total Cash Dividends Paid", component_value=dividends_abs, source_table="financials", source_section="cashflow", source_label="Total cash dividends paid")

                elif metric_code == "PNAV_PROXY":
                    value = inputs["price_to_book_ratio"]
                    status = "OK" if value is not None else "MISSING_INPUT"
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="input", component_name="Price to Book Ratio", component_value=value, source_table="financials", source_section="statistics", source_label="Price to book ratio")

                elif metric_code == "FFO_YIELD_EQ":
                    equity_proxy = None
                    if inputs["net_income"] is not None and inputs["price_to_earnings_ratio"] is not None:
                        equity_proxy = inputs["net_income"] * inputs["price_to_earnings_ratio"]
                    value = _safe_div(inputs["ffo"], equity_proxy)
                    status = "OK" if value is not None else "MISSING_INPUT"
                    notes = "Uses market-cap proxy Net Income * PE Ratio."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="FFO", component_value=inputs["ffo"], source_table="financials", source_section="cashflow", source_label="Funds from operations")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="Net Income * PE Ratio", component_value=equity_proxy, source_table="derived", source_section="statistics", source_label="Net income * Price to earnings ratio")

                elif metric_code == "FFO_YIELD_CAP":
                    value = _safe_div(inputs["ffo"], inputs["enterprise_value"])
                    status = "OK" if value is not None else "MISSING_INPUT"
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="FFO", component_value=inputs["ffo"], source_table="financials", source_section="cashflow", source_label="Funds from operations")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="Enterprise Value", component_value=inputs["enterprise_value"], source_table="financials", source_section="statistics", source_label="Enterprise value")

                elif metric_code == "LEVERAGE_PREMIUM":
                    left = computed.get("FFO_YIELD_EQ", (None, "", None))[0]
                    right = computed.get("FFO_YIELD_CAP", (None, "", None))[0]
                    if left is not None and right is not None:
                        value = left - right
                        status = "OK"
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="input", component_name="FFO Yield [Equity]", component_value=left, source_table="reit_metrics", source_section="fact_metric_value", source_label="FFO_YIELD_EQ")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="input", component_name="FFO Yield [Capital]", component_value=right, source_table="reit_metrics", source_section="fact_metric_value", source_label="FFO_YIELD_CAP")

                elif metric_code == "REAL_YIELD_SORA":
                    if inputs["dividend_yield_fy_pct"] is not None and inputs["sora_3m"] is not None:
                        value = inputs["dividend_yield_fy_pct"] - inputs["sora_3m"]
                        status = "OK"
                    notes = "Uses latest SORA 3M on or before fiscal year end."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="input", component_name="Dividend Yield (FY) %", component_value=inputs["dividend_yield_fy_pct"], source_table="financials", source_section="dividends", source_label="Dividend yield (FY) %")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="input", component_name="SORA 3M", component_value=inputs["sora_3m"], source_table="external", source_section="sora", source_label="SORA_3M")

                elif metric_code == "UNIT_DILUTION":
                    previous_shares = None if prior_inputs is None else prior_inputs.get("diluted_shares")
                    if (
                        previous_shares is not None
                        and previous_shares != 0
                        and inputs["diluted_shares"] is not None
                    ):
                        value = (inputs["diluted_shares"] - previous_shares) / previous_shares
                        status = "OK"
                    notes = "Requires previous annual period."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="current", component_name="Diluted Shares Outstanding", component_value=inputs["diluted_shares"], source_table="financials", source_section="income", source_label="Diluted shares outstanding")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="prior_period_input", component_name="Prior Diluted Shares Outstanding", component_value=previous_shares, source_table="financials", source_section="income", source_label="Diluted shares outstanding")

                elif metric_code == "OPEX_INTENSITY":
                    opex_abs = abs(inputs["total_operating_expenses"]) if inputs["total_operating_expenses"] is not None else None
                    value = _safe_div(opex_abs, inputs["total_revenue"])
                    status = "OK" if value is not None else "MISSING_INPUT"
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="Total Operating Expenses", component_value=opex_abs, source_table="financials", source_section="income", source_label="Total operating expenses")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="Total Revenue", component_value=inputs["total_revenue"], source_table="financials", source_section="income", source_label="Total revenue")

                elif metric_code == "CAPEX_DRAG":
                    dividends_abs = abs(inputs["cash_dividends_paid"]) if inputs["cash_dividends_paid"] is not None else None
                    left = _safe_div(dividends_abs, inputs["free_cash_flow"])
                    right = _safe_div(dividends_abs, inputs["ffo"])
                    if left is not None and right is not None:
                        value = 1.0 if left > right else 0.0
                        status = "OK"
                        notes = f"left={left:.6f}; right={right:.6f}"
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="input", component_name="Dividend / FCF", component_value=left, source_table="derived", source_section="cashflow", source_label="abs(dividends) / free cash flow")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="input", component_name="Dividend / FFO", component_value=right, source_table="derived", source_section="cashflow", source_label="abs(dividends) / FFO")

                elif metric_code == "NONRECUR_SHARE":
                    value = _safe_div(inputs["unusual_income_expense"], inputs["net_income"])
                    status = "OK" if value is not None else "MISSING_INPUT"
                    if value is not None and _net_income_is_low_denominator(inputs["net_income"], inputs["total_revenue"]):
                        status = "LOW_DENOMINATOR"
                        notes = "Net income is very small relative to revenue, so the ratio is numerically unstable and can look extreme."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="Unusual Income/Expense", component_value=inputs["unusual_income_expense"], source_table="financials", source_section="income", source_label="Unusual income/expense")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="Net Income", component_value=inputs["net_income"], source_table="financials", source_section="income", source_label="Net income")

                elif metric_code == "DSCR":
                    interest_paid_abs = abs(inputs["interest_paid"]) if inputs["interest_paid"] is not None else None
                    denominator = None
                    if interest_paid_abs is not None and inputs["short_term_debt"] is not None:
                        denominator = interest_paid_abs + inputs["short_term_debt"]
                    value = _safe_div(inputs["ffo"], denominator)
                    status = "OK" if value is not None else "MISSING_INPUT"
                    if value is not None and inputs["ffo"] is not None and inputs["ffo"] < 0:
                        status = "NEGATIVE_BASE"
                        notes = "FFO is negative, so this is a distress-style debt service ratio rather than a normal coverage ratio."
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="numerator", component_name="FFO", component_value=inputs["ffo"], source_table="financials", source_section="cashflow", source_label="Funds from operations")
                    _append_component(component_rows, ticker=ticker, period_id=period_id, metric_code=metric_code, component_role="denominator", component_name="abs(Interest Paid) + Short Term Debt", component_value=denominator, source_table="derived", source_section="cashflow", source_label="abs(Interest paid) + Short term debt")

                else:
                    raise RuntimeError(f"Unhandled metric code: {metric_code}")

                computed[metric_code] = (value, status, notes)

                fact_rows.append(
                    {
                        "ticker": ticker,
                        "period_id": period_id,
                        "metric_code": metric_code,
                        "metric_value": value,
                        "value_text": None if metric_code != "CAPEX_DRAG" or value is None else ("TRUE" if value == 1.0 else "FALSE"),
                        "calc_status": status,
                        "calc_version": "v1",
                        "source_period_label": period_label,
                        "notes": notes,
                    }
                )

            prior_inputs = inputs

    fact_df = pd.DataFrame(fact_rows)
    component_df = pd.DataFrame(component_rows)
    return fact_df, component_df


def _write_duckdb_and_parquet(
    dim_reit: pd.DataFrame,
    dim_period: pd.DataFrame,
    dim_metric: pd.DataFrame,
    fact_metric_value: pd.DataFrame,
    fact_metric_component: pd.DataFrame,
    fact_external_series: pd.DataFrame,
) -> None:
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PARQUET.exists():
        OUTPUT_PARQUET.unlink()

    con = duckdb.connect(str(SOURCE_DUCKDB))
    try:
        con.execute("DROP SCHEMA IF EXISTS reit_metrics CASCADE")
        con.execute("CREATE SCHEMA reit_metrics")

        con.execute(
            """
            CREATE TABLE reit_metrics.dim_reit (
                ticker VARCHAR PRIMARY KEY,
                reit_name VARCHAR,
                sector VARCHAR,
                health_bucket VARCHAR,
                notes VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE reit_metrics.dim_period (
                period_id BIGINT PRIMARY KEY,
                ticker VARCHAR NOT NULL,
                source_period_label VARCHAR NOT NULL,
                period_kind VARCHAR NOT NULL,
                fiscal_year INTEGER,
                fiscal_year_end_month INTEGER,
                fiscal_year_end_year INTEGER,
                fiscal_year_end_date DATE,
                display_year INTEGER,
                sort_key INTEGER,
                is_annual BOOLEAN NOT NULL,
                is_ttm BOOLEAN NOT NULL,
                is_current BOOLEAN NOT NULL,
                UNIQUE (ticker, source_period_label)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE reit_metrics.dim_metric (
                metric_code VARCHAR PRIMARY KEY,
                metric_name VARCHAR NOT NULL,
                formula_short VARCHAR,
                unit_type VARCHAR NOT NULL,
                higher_is_better BOOLEAN,
                source_schema_hint VARCHAR,
                requires_external BOOLEAN NOT NULL,
                description VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE reit_metrics.fact_metric_value (
                ticker VARCHAR NOT NULL,
                period_id BIGINT NOT NULL,
                metric_code VARCHAR NOT NULL,
                metric_value DOUBLE,
                value_text VARCHAR,
                calc_status VARCHAR NOT NULL,
                calc_version VARCHAR NOT NULL,
                asof_ts TIMESTAMP NOT NULL DEFAULT current_timestamp,
                source_period_label VARCHAR,
                notes VARCHAR,
                PRIMARY KEY (ticker, period_id, metric_code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE reit_metrics.fact_metric_component (
                ticker VARCHAR NOT NULL,
                period_id BIGINT NOT NULL,
                metric_code VARCHAR NOT NULL,
                component_role VARCHAR NOT NULL,
                component_name VARCHAR NOT NULL,
                component_value DOUBLE,
                component_text VARCHAR,
                source_table VARCHAR,
                source_section VARCHAR,
                source_row_id INTEGER,
                source_label VARCHAR,
                PRIMARY KEY (ticker, period_id, metric_code, component_role, component_name)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE reit_metrics.fact_external_series (
                series_code VARCHAR NOT NULL,
                period_id BIGINT NOT NULL,
                value DOUBLE,
                value_text VARCHAR,
                value_date DATE,
                source_name VARCHAR,
                PRIMARY KEY (series_code, period_id)
            )
            """
        )

        con.register("dim_reit_df", dim_reit)
        con.register("dim_period_df", dim_period)
        con.register("dim_metric_df", dim_metric)
        con.register("fact_metric_value_df", fact_metric_value)
        con.register("fact_metric_component_df", fact_metric_component)
        con.register("fact_external_series_df", fact_external_series)

        con.execute("INSERT INTO reit_metrics.dim_reit SELECT * FROM dim_reit_df")
        con.execute("INSERT INTO reit_metrics.dim_period SELECT * FROM dim_period_df")
        con.execute("INSERT INTO reit_metrics.dim_metric SELECT * FROM dim_metric_df")
        con.execute(
            """
            INSERT INTO reit_metrics.fact_metric_value
                (ticker, period_id, metric_code, metric_value, value_text, calc_status, calc_version, source_period_label, notes)
            SELECT
                ticker, period_id, metric_code, metric_value, value_text, calc_status, calc_version, source_period_label, notes
            FROM fact_metric_value_df
            """
        )
        con.execute("INSERT INTO reit_metrics.fact_metric_component SELECT * FROM fact_metric_component_df")
        con.execute("INSERT INTO reit_metrics.fact_external_series SELECT * FROM fact_external_series_df")

        con.execute(
            f"""
            COPY (
                SELECT
                    v.ticker,
                    r.reit_name,
                    r.sector,
                    r.health_bucket,
                    p.period_id,
                    p.source_period_label,
                    p.fiscal_year,
                    p.fiscal_year_end_month,
                    p.fiscal_year_end_year,
                    p.fiscal_year_end_date,
                    p.sort_key,
                    m.metric_code,
                    m.metric_name,
                    m.unit_type,
                    v.metric_value,
                    v.value_text,
                    v.calc_status,
                    v.calc_version,
                    v.notes
                FROM reit_metrics.fact_metric_value v
                JOIN reit_metrics.dim_reit r USING (ticker)
                JOIN reit_metrics.dim_period p USING (period_id)
                JOIN reit_metrics.dim_metric m USING (metric_code)
                ORDER BY v.ticker, p.sort_key, m.metric_code
            ) TO '{OUTPUT_PARQUET.as_posix()}'
            (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def main() -> None:
    if not SOURCE_DUCKDB.exists():
        raise FileNotFoundError(f"Source DuckDB not found: {SOURCE_DUCKDB}")
    if not SORA_3M_CSV.exists():
        raise FileNotFoundError(f"SORA 3M CSV not found: {SORA_3M_CSV}")

    print(f"Source warehouse: {SOURCE_DUCKDB}")
    print(f"Metric spec reference: {SOURCE_METRIC_SPEC}")

    read_con = duckdb.connect(str(SOURCE_DUCKDB), read_only=True)
    try:
        source_df = _load_source_financials(read_con)
    finally:
        read_con.close()

    dim_reit = _build_dim_reit(source_df)
    dim_period = _build_dim_period(source_df)
    dim_metric = _build_dim_metric()
    lookups = _build_lookups(source_df)
    sora_3m_df = _load_sora_3m()
    fact_external_series, external_lookup = _build_external_series(dim_period, sora_3m_df)
    fact_metric_value, fact_metric_component = _compute_metric_rows(dim_period, lookups, external_lookup)

    expected_rows = len(dim_period) * len(dim_metric)
    actual_rows = len(fact_metric_value)
    if actual_rows != expected_rows:
        raise RuntimeError(f"Metric row count mismatch. expected={expected_rows}, actual={actual_rows}")

    _write_duckdb_and_parquet(
        dim_reit=dim_reit,
        dim_period=dim_period,
        dim_metric=dim_metric,
        fact_metric_value=fact_metric_value,
        fact_metric_component=fact_metric_component,
        fact_external_series=fact_external_series,
    )

    blank_count = int(fact_metric_value["metric_value"].isna().sum())
    print(f"Tickers: {len(dim_reit)}")
    print(f"Annual periods: {len(dim_period)}")
    print(f"Metrics: {len(dim_metric)}")
    print(f"Fact metric rows: {len(fact_metric_value)}")
    print(f"Blank metric values: {blank_count}")
    print(f"Wrote metrics shard: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
