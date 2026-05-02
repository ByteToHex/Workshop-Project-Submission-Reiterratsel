from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
KG_DIR = ROOT_DIR / "Common" / "Micro" / "5_Model_KG"
LOGO_ICON_PATH = ROOT_DIR / "Common" / "Frontend" / "DesignDoc" / "Ritteratsel_Logo.svg"
LOGO_WORDMARK_PATH = ROOT_DIR / "Common" / "Frontend" / "DesignDoc" / "Reiterratsel_Wordmark.svg"
if str(KG_DIR) not in sys.path:
    sys.path.insert(0, str(KG_DIR))
if not LOGO_ICON_PATH.exists():
    raise FileNotFoundError(f"Required frontend logo icon asset is missing: {LOGO_ICON_PATH}")
if not LOGO_WORDMARK_PATH.exists():
    raise FileNotFoundError(f"Required frontend wordmark asset is missing: {LOGO_WORDMARK_PATH}")

from reitteratsel_core import (  # noqa: E402
    DEFAULT_HORIZON_DAYS,
    DUCKDB_PATH,
    compute_sora_distress_score,
    load_macro_holdout_frame,
    load_macro_prediction_frame,
    load_macro_train_end,
)
from reitteratsel_view_logic import (  # noqa: E402
    build_macro_panel_context,
    build_ranking_view,
    get_label_row_for_period,
)


st.set_page_config(page_title="REITterratsel", layout="wide")
st.logo(str(LOGO_WORDMARK_PATH), icon_image=str(LOGO_ICON_PATH), size="large")

SIMULATION_DATE_HELP = (
    "Resolve each view using the latest eligible annual filing row, macro snapshot, and daily cumulative abnormal return (CAR) path row "
    "on or before this date."
)
MACRO_HEADER_KICKER = "XGBoost SORA forecast from overnight bank-rate inputs"
MACRO_CHANGE_HELP = (
    "Predicted 10-trading-day change in SORA. Computed from the run_21 macro model artifacts and converted "
    "to a 0-1 stress score by reitteratsel_core.compute_sora_distress_score()."
)
MACRO_LEVEL_HELP = (
    "Predicted SORA level 10 trading days ahead from the same macro model run."
)
MACRO_DISTRESS_HELP = (
    "0-1 macro rate-stress score derived from the predicted 10-day SORA change. This is the macro overlay "
    "used at runtime, not an annual accounting metric."
)
MACRO_SNAPSHOT_HELP = (
    "Macro model snapshot date selected on or before the chosen simulation date."
)
MACRO_SHOCK_HELP = (
    "Signed macro shock used inside reitteratsel_core.compute_final_distress_score(), equal to "
    "Macro Rate-Stress Overlay Score minus the neutral 0.50 baseline before any REFI_RISK scaling."
)
FINAL_SCORE_HELP = (
    "Final 0-1 runtime distress score shown in the app. It starts from the annual Mamdani base score in "
    "reit_fuzzy.fact_fuzzy_cache, then adds a macro rate overlay and a REIT-specific abnormal-return-path overlay "
    "via reitteratsel_core.compute_final_distress_score()."
)
ANNUAL_MAMDANI_HELP = (
    "Annual base distress score from the Mamdani fuzzy rule engine, persisted in reit_fuzzy.fact_fuzzy_cache. "
    "Built from annual fundamentals in reit_metrics.fact_metric_value through build_fuzzy_input_frame(), "
    "evaluate_fuzzy_row(), and build_fuzzy_cache_frame()."
)
ICR_HELP = (
    "Interest Coverage Ratio. Built from annual EBITDA divided by absolute bank interest expense "
    "per Data_Dict_Reit_Metrics.md and stored in reit_metrics.fact_metric_value."
)
GEARING_HELP = (
    "Gearing Ratio. Annual total debt divided by total assets, with a warehouse fallback to the existing "
    "debt-to-assets ratio when required."
)
DSCR_HELP = (
    "Debt Service Coverage Ratio. Annual Funds From Operation (FFO) divided by cash interest paid plus short-term debt."
)
REFI_HELP = (
    "Refinancing Risk Ratio. Annual short-term debt divided by total debt. At runtime this scales how strongly "
    "the macro rate-shock overlay affects the final distress score."
)
REFI_PROXY_HELP = (
    "0-1 refinancing stress proxy derived from the annual Refinancing Risk Ratio, following the same helper used in evaluation "
    "to compare a REFI-only stress view against Mamdani and final_distress."
)
MACRO_SENSITIVITY_HELP = (
    "Per-ticker sensitivity weight used in the final distress formula. It is derived from Refinancing Risk Ratio as "
    "max(0.25, min(1.0, REFI_RISK * 2.5))."
)
MACRO_ADJUSTMENT_HELP = (
    "The actual signed macro contribution added to this REIT's final distress score, computed as "
    "0.15 * macro sensitivity * (distress_sora - 0.50)."
)
TOP_GEO_HELP = (
    "Largest revenue-contributing geography for the selected annual filing row."
)
TOP_GEO_SHARE_HELP = (
    "Share of revenue attributed to the top revenue geography for the selected annual filing row."
)
PAYOUT_HELP = (
    "Payout Ratio. Annual cash dividends paid divided by FFO. If FFO is non-positive, the raw number is still "
    "stored but should be read as a distress-style signal."
)
NULL_COUNT_HELP = (
    "Count of annual input metrics for this ticker-period whose calculation status was MISSING_INPUT when the "
    "fuzzy input row was built."
)
RULE_TRACE_HELP = (
    "Top Mamdani rules that fired for the selected annual filing row, taken from reit_fuzzy.fact_fuzzy_cache.rule_trace_text."
)
NON_OK_COUNT_HELP = (
    "Number of annual input metrics for this fiscal-year snapshot whose calculation status was not OK "
    "(for example missing input, partial calculation, or error) when the Mamdani input row was built."
)
FIRED_RULE_COUNT_HELP = (
    "Number of Mamdani rules that fired with non-zero strength for the selected annual filing row."
)
FINAL_SCORE_BUILD_HELP = (
    "This card shows how the final runtime distress score is assembled: annual Mamdani base score from "
    "reit_fuzzy.fact_fuzzy_cache, plus the macro rate-stress overlay and the REIT-specific abnormal-return-path overlay."
)
CAR_PANEL_HELP = (
    "These fields come from reit_labels.fact_distress_label and reit_labels.fact_car_path_daily, which are built "
    "from REIT daily returns minus the SGX iEdge REIT index daily return."
)
ACCUM_CAR_HELP = (
    "Accumulated cumulative abnormal return from the annual filing anchor trade date up to the selected simulation date."
)
CAR_PATH_DISTRESS_HELP = (
    "0-1 score translated from the accumulated abnormal-return path. This is the REIT-specific market overlay used "
    "inside the final runtime distress score."
)
CAR_63_HELP = (
    "Forward 63-trading-day cumulative abnormal return after the annual filing anchor date. This is intended to display future data for proof-of-concept demonstration only."
)
CAR_126_HELP = (
    "Forward 126-trading-day cumulative abnormal return after the annual filing anchor date. This is the basis for "
    "the persisted distress label. This is intended to display future data for proof-of-concept demonstration only."
)
CAR_LABEL_HELP = (
    "Label derived from the 126-trading-day cumulative abnormal return in reit_labels.fact_distress_label."
)
MACRO_PANEL_HELP = (
    "These fields come from the run_21 XGBoost macro artifacts loaded at runtime rather than from DuckDB."
)
FOMC_HELP = (
    "FOMC decision date associated with the selected macro model snapshot."
)
ANNUAL_ANCHOR_HELP = (
    "Fiscal-year-end anchor date from reit_metrics.dim_period used to select the annual filing row for this view."
)
RANKING_TABLE_HELP = (
    "Ranking table combines the persisted annual Mamdani base score, the runtime macro overlay, and the runtime "
    "daily abnormal-return-path overlay for the selected simulation date."
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #181515 0%, #231f1f 100%);
        color: #f2f0ec;
    }
    [data-testid="stHeader"] {
        background: rgba(23, 20, 20, 0.92);
        border-bottom: 1px solid #3d3434;
        backdrop-filter: blur(10px);
    }
    [data-testid="stToolbar"] {
        top: 0.65rem;
        right: 0.9rem;
    }
    [data-testid="stSidebar"] {
        background: #171414;
        border-right: 1px solid #3d3434;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 0.35rem;
    }
    .block-container {
        padding-top: 5.75rem;
        padding-bottom: 2rem;
    }
    .reit-card {
        background: #1e1a1a;
        border: 1px solid #2d2828;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
    }
    .reit-kicker {
        color: #09b6ff;
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 700;
    }
    .reit-score {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
    }
    .reit-muted {
        color: #bfb7b2;
    }
    .reit-header-grid {
        display: grid;
        grid-template-columns: minmax(220px, 0.95fr) minmax(180px, 0.8fr) repeat(6, minmax(110px, 1fr));
        gap: 0.85rem;
        align-items: stretch;
    }
    .reit-header-score {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .reit-label {
        color: #7f7a76;
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .reit-value {
        font-size: 2.1rem;
        line-height: 1.05;
        font-weight: 800;
        color: #f6f2ee;
    }
    .reit-value-score {
        font-size: 4.4rem;
        color: #ff4a4a;
    }
    .reit-subvalue {
        margin-top: 0.35rem;
        color: #bfb7b2;
        font-size: 0.92rem;
    }
    .reit-subvalue-highlight {
        color: #f6d365;
        font-weight: 800;
    }
    .reit-thresholds {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        justify-content: center;
    }
    .reit-pill {
        border-radius: 8px;
        border: 1px solid #574c4c;
        padding: 0.4rem 0.65rem;
        font-size: 0.86rem;
        font-weight: 700;
        color: #efe9e4;
        background: rgba(255, 255, 255, 0.03);
    }
    .reit-pill-active {
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.15);
    }
    .reit-pill-stable.reit-pill-active {
        background: rgba(34, 197, 94, 0.18);
        border-color: #22c55e;
    }
    .reit-pill-watch.reit-pill-active {
        background: rgba(234, 179, 8, 0.18);
        border-color: #eab308;
    }
    .reit-pill-high.reit-pill-active {
        background: rgba(249, 115, 22, 0.20);
        border-color: #f97316;
    }
    .reit-pill-critical.reit-pill-active {
        background: rgba(239, 68, 68, 0.22);
        border-color: #ef4444;
    }
    .reit-metric-box {
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 118px;
    }
    .reit-section-banner {
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin: 1.1rem 0 0.8rem 0;
        border: 1px solid #3b3232;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
    }
    .reit-section-banner h3 {
        margin: 0;
        font-size: 1.05rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .reit-section-banner p {
        margin: 0.35rem 0 0 0;
        color: #d5ccc6;
        font-size: 0.92rem;
    }
    .reit-section-change {
        background: linear-gradient(135deg, rgba(9, 182, 255, 0.18), rgba(18, 26, 36, 0.92));
        border-color: #167aa8;
    }
    .reit-section-change h3 {
        color: #69d2ff;
    }
    .reit-section-level {
        background: linear-gradient(135deg, rgba(244, 137, 61, 0.18), rgba(37, 24, 18, 0.92));
        border-color: #a85b23;
    }
    .reit-section-level h3 {
        color: #ffb274;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_app_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        fuzzy_df = con.execute(
            """
            SELECT
                f.ticker,
                f.period_id,
                r.reit_name,
                r.sector,
                p.fiscal_year,
                p.fiscal_year_end_date,
                f.distress_score_mamdani,
                f.distress_level,
                f.null_count,
                f.non_ok_count,
                f.fired_rule_count,
                f.top_rule_ids,
                f.rule_trace_text,
                l.car_63wd,
                l.car_126wd,
                l.label_126wd
            FROM reit_fuzzy.fact_fuzzy_cache f
            JOIN reit_metrics.dim_period p ON p.period_id = f.period_id
            JOIN reit_metrics.dim_reit r ON r.ticker = f.ticker
            LEFT JOIN reit_labels.fact_distress_label l
              ON l.ticker = f.ticker
             AND l.period_id = f.period_id
            ORDER BY f.ticker, p.fiscal_year_end_date
            """
        ).fetchdf()
        metric_df = con.execute(
            """
            SELECT
                v.ticker,
                v.period_id,
                p.fiscal_year,
                p.fiscal_year_end_date,
                v.metric_code,
                v.metric_value,
                v.calc_status
            FROM reit_metrics.fact_metric_value v
            JOIN reit_metrics.dim_period p ON p.period_id = v.period_id
            ORDER BY v.ticker, p.fiscal_year_end_date, v.metric_code
            """
        ).fetchdf()
        label_df = con.execute(
            """
            SELECT
                l.ticker,
                l.period_id,
                p.fiscal_year,
                l.anchor_date,
                l.anchor_trade_date,
                l.car_63wd,
                l.car_126wd,
                l.label_126wd,
                l.null_count,
                l.non_ok_count,
                l.notes
            FROM reit_labels.fact_distress_label l
            JOIN reit_metrics.dim_period p ON p.period_id = l.period_id
            ORDER BY l.ticker, p.fiscal_year
            """
        ).fetchdf()
        car_path_df = con.execute(
            """
            SELECT
                c.ticker,
                c.period_id,
                p.fiscal_year,
                c.anchor_date,
                c.anchor_trade_date,
                c.trade_date,
                c.days_from_anchor,
                c.abnormal_return,
                c.accum_car_to_date,
                c.car_path_distress
            FROM reit_labels.fact_car_path_daily c
            JOIN reit_metrics.dim_period p ON p.period_id = c.period_id
            ORDER BY c.ticker, p.fiscal_year, c.trade_date
            """
        ).fetchdf()
        component_df = con.execute(
            """
            SELECT
                c.ticker,
                c.period_id,
                p.fiscal_year,
                p.fiscal_year_end_date,
                c.metric_code,
                c.component_role,
                c.component_name,
                c.component_value,
                c.component_text,
                c.source_label
            FROM reit_metrics.fact_metric_component c
            JOIN reit_metrics.dim_period p ON p.period_id = c.period_id
            WHERE (c.metric_code = 'REFI_RISK' AND c.component_name = 'Short Term Debt')
               OR (c.metric_code = 'REV_CONC_TOPGEO' AND c.component_name = 'top_geography_value')
            ORDER BY c.ticker, p.fiscal_year_end_date, c.metric_code, c.component_name
            """
        ).fetchdf()
        dividend_df = con.execute(
            """
            SELECT
                d.ticker,
                p.period_id,
                p.fiscal_year,
                p.fiscal_year_end_date,
                TRY_CAST(f.value AS DOUBLE) AS dps_fy
            FROM main.financials f
            JOIN main.schema_rows s USING (row_id)
            JOIN reit_metrics.dim_period p
              ON p.ticker = f.ticker
             AND p.fiscal_year = TRY_CAST(f.period AS INTEGER)
            JOIN reit_metrics.dim_reit d ON d.ticker = f.ticker
            WHERE s.section = 'dividends'
              AND s.label = 'Dividends per share (FY)'
            ORDER BY d.ticker, p.fiscal_year_end_date
            """
        ).fetchdf()
    finally:
        con.close()
    return fuzzy_df, metric_df, label_df, car_path_df, component_df, dividend_df


fuzzy_df, metric_df, label_df, car_path_df, component_df, dividend_df = load_app_frames()
macro_df = load_macro_prediction_frame(DEFAULT_HORIZON_DAYS)
macro_holdout_df = load_macro_holdout_frame(DEFAULT_HORIZON_DAYS)
macro_train_end = load_macro_train_end(DEFAULT_HORIZON_DAYS)
if macro_df["prediction_source"].nunique() != 1 or macro_df["prediction_source"].iloc[0] != "xgboost_final_model":
    raise RuntimeError(
        "Macro runtime must use direct XGBoost inference from the run_21 model artifacts."
    )
latest_macro = macro_df.iloc[-1]
MIN_MACRO_DATE = macro_df["snapshot_ts"].min().date()
MAX_MACRO_DATE = macro_df["snapshot_ts"].max().date()


def get_selected_simulation_date() -> object:
    if "simulation_date_value" not in st.session_state:
        st.session_state["simulation_date_value"] = MAX_MACRO_DATE

    selected_date = st.date_input(
        "Simulation Date",
        value=st.session_state["simulation_date_value"],
        min_value=MIN_MACRO_DATE,
        max_value=MAX_MACRO_DATE,
        key="selected_simulation_date_widget",
        help=SIMULATION_DATE_HELP,
    )
    st.session_state["simulation_date_value"] = selected_date
    return selected_date


def get_selected_ticker(ticker_options: list[str]) -> str:
    if not ticker_options:
        raise ValueError("Ticker selector requires at least one available ticker option.")

    durable_key = "selected_ticker_value"
    widget_key = "selected_ticker_widget"

    default_ticker = st.session_state.get(durable_key, ticker_options[0])
    if default_ticker not in ticker_options:
        default_ticker = ticker_options[0]
    st.session_state[durable_key] = default_ticker

    widget_value = st.session_state.get(widget_key)
    if widget_value not in ticker_options:
        st.session_state[widget_key] = default_ticker

    selected_ticker = st.selectbox(
        "Select REIT",
        ticker_options,
        key=widget_key,
    )
    st.session_state[durable_key] = selected_ticker
    return selected_ticker


def render_macro_header(*, macro_row: pd.Series, distress_sora: float) -> None:
    st.markdown(f'<div class="reit-kicker">{MACRO_HEADER_KICKER}</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted 10-Trading-Day SORA Change", f"{macro_row['y_pred']:+.3f}", help=MACRO_CHANGE_HELP)
    m2.metric("Predicted SORA Level In 10 Trading Days", f"{macro_row['predicted_level']:.3f}", help=MACRO_LEVEL_HELP)
    m3.metric("Macro Rate-Stress Overlay Score", f"{distress_sora:.2f}", help=MACRO_DISTRESS_HELP)
    m4.metric("Macro Shock Vs Neutral", f"{(distress_sora - 0.5):+.2f}", help=MACRO_SHOCK_HELP)


def resolve_simulation_context() -> tuple[object, pd.DataFrame, pd.Series, float]:
    selected_date = get_selected_simulation_date()
    ranking_view, macro_row, distress_sora = build_ranking_view(
        fuzzy_df,
        metric_df,
        component_df,
        macro_df,
        car_path_df,
        selected_date,
    )
    return selected_date, ranking_view, macro_row, distress_sora


def format_currency_compact(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000:
        return f"SGD {value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"SGD {value / 1_000_000:.0f}M"
    if abs_value >= 1_000:
        return f"SGD {value / 1_000:.0f}K"
    return f"SGD {value:,.0f}"


def derive_absolute_trend(values: pd.Series, *, threshold: float, up_label: str, down_label: str) -> str:
    """
    Classify a level-style annual series using the latest point versus a recent baseline.

    Current use:
    - ICR Trend

    Definition:
    - if 3+ annual points exist, baseline = mean of all but the latest point
    - if only 2 points exist, baseline = prior point
    - delta = latest - baseline
    - if abs(delta) < threshold -> "Flat"
    - if delta > 0 -> up_label
    - if delta < 0 -> down_label

    For ICR Trend, this means:
    - "Improving" when the latest ICR is materially above the recent annual baseline
    - "Deteriorating" when it is materially below
    """
    clean = pd.to_numeric(values, errors="coerce").dropna().reset_index(drop=True)
    if len(clean) < 2:
        return "N/A"
    baseline = clean.iloc[:-1].mean() if len(clean) >= 3 else clean.iloc[-2]
    delta = float(clean.iloc[-1] - baseline)
    if abs(delta) < threshold:
        return "Flat"
    return up_label if delta > 0 else down_label


def derive_relative_trend(values: pd.Series, *, threshold_ratio: float, up_label: str, down_label: str) -> str:
    """
    Classify a percentage-style annual series using relative change versus a recent baseline.

    Current use:
    - DPU Trend, where DPU is annual `Dividends per share (FY)` from the raw annual source layer

    Definition:
    - if 3+ annual points exist, baseline = mean of all but the latest point
    - if only 2 points exist, baseline = prior point
    - delta_ratio = (latest - baseline) / abs(baseline)
    - if abs(delta_ratio) < threshold_ratio -> "Flat"
    - if delta_ratio > 0 -> up_label
    - if delta_ratio < 0 -> down_label

    Fallback when baseline is zero:
    - use the latest absolute delta against the prior point
    """
    clean = pd.to_numeric(values, errors="coerce").dropna().reset_index(drop=True)
    if len(clean) < 2:
        return "N/A"
    baseline = float(clean.iloc[:-1].mean() if len(clean) >= 3 else clean.iloc[-2])
    if baseline == 0:
        latest_delta = float(clean.iloc[-1] - clean.iloc[-2])
        if abs(latest_delta) < threshold_ratio:
            return "Flat"
        return up_label if latest_delta > 0 else down_label
    delta_ratio = float((clean.iloc[-1] - baseline) / abs(baseline))
    if abs(delta_ratio) < threshold_ratio:
        return "Flat"
    return up_label if delta_ratio > 0 else down_label


def build_threshold_pills(final_level: str) -> str:
    level_key = final_level.lower()
    levels = [
        ("stable", "0.00-0.34 Stable"),
        ("watch", "0.35-0.54 Watch"),
        ("high", "0.55-0.74 High"),
        ("critical", "0.75-1.00 Critical"),
    ]
    pills: list[str] = []
    for css_key, label in levels:
        active_class = " reit-pill-active" if css_key == level_key else ""
        pills.append(f'<div class="reit-pill reit-pill-{css_key}{active_class}">{label}</div>')
    return "".join(pills)


def build_reit_history_frames(
    *,
    selected_metric_df: pd.DataFrame,
    selected_label_df: pd.DataFrame,
    selected_period_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_history_df = selected_metric_df.loc[
        selected_metric_df["fiscal_year_end_date"] <= selected_period_end
    ].copy()
    label_history_df = selected_label_df.loc[
        pd.to_datetime(selected_label_df["anchor_date"]) <= selected_period_end
    ].copy()
    return metric_history_df, label_history_df


def render_expanded_reit_header(
    *,
    final_distress: float,
    final_level: str,
    latest_metrics: pd.Series,
    icr_trend: str,
    top_geo_label: str,
    top_geo_share: float | None,
    refi_cliff_value: float | None,
    dpu_trend: str,
) -> None:
    top_geo_display = top_geo_label or "N/A"
    if top_geo_share is not None and not pd.isna(top_geo_share):
        top_geo_subvalue = f'<span class="reit-subvalue-highlight">{top_geo_share:.0%}</span> of revenue'
    else:
        top_geo_subvalue = "Share unavailable"
    header_html = f"""
    <div class="reit-card">
      <div class="reit-header-grid">
        <div class="reit-header-score">
          <div class="reit-label">Final Runtime Distress Score</div>
          <div class="reit-value reit-value-score">{final_distress:.2f}</div>
          <div class="reit-subvalue">Runtime level after annual, macro, and CAR-path overlays: {final_level}</div>
        </div>
        <div class="reit-thresholds">
          {build_threshold_pills(final_level)}
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">Interest Coverage Ratio (ICR)</div>
          <div class="reit-value">{latest_metrics.get('ICR', float('nan')):.2f}</div>
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">Gearing Ratio</div>
          <div class="reit-value">{latest_metrics.get('GEARING', float('nan')):.1%}</div>
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">Interest Coverage Trend</div>
          <div class="reit-value" style="font-size: 1.7rem;">{icr_trend}</div>
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">Top Revenue Geography</div>
          <div class="reit-value" style="font-size: 1.7rem;">{top_geo_display}</div>
          <div class="reit-subvalue">{top_geo_subvalue}</div>
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">Short-Term Debt Due ("Refi Cliff")</div>
          <div class="reit-value" style="font-size: 1.8rem;">{format_currency_compact(refi_cliff_value)}</div>
          <div class="reit-subvalue">Annual short-term debt component used inside REFI_RISK</div>
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">Dividend Per Unit Trend</div>
          <div class="reit-value" style="font-size: 1.7rem;">{dpu_trend}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def build_reit_header_context(
    *,
    selected_ticker: str,
    selected_period_id: int,
    selected_row: pd.Series,
    selected_metric_df: pd.DataFrame,
    selected_component_df: pd.DataFrame,
    selected_dividend_df: pd.DataFrame,
    final_distress: float,
    final_level: str,
) -> dict[str, Any]:
    """
    Build one canonical annual-anchor context object for the expanded REIT header.

    This keeps the header logic aligned to a single selected `period_id` so the page
    does not mix "latest" rows from different annual anchors.

    Definitions used here:
    - ICR Trend:
      derived from the annual ICR series with `derive_absolute_trend(...)`
    - DPU Trend:
      derived from annual `Dividends per share (FY)` with `derive_relative_trend(...)`
    - Refi Cliff:
      currently defined in the header as annual `Short Term Debt`
      read from `reit_metrics.fact_metric_component` for metric `REFI_RISK`
    - TOPGEO:
      latest annual top geography label/share from `REV_CONC_TOPGEO`
    """
    selected_period_end = pd.Timestamp(selected_row["fiscal_year_end_date"])
    metric_history_df = selected_metric_df.loc[
        selected_metric_df["fiscal_year_end_date"] <= selected_period_end
    ].copy()
    dividend_history_df = selected_dividend_df.loc[
        selected_dividend_df["fiscal_year_end_date"] <= selected_period_end
    ].copy()
    metric_pivot = (
        metric_history_df.sort_values(["fiscal_year_end_date", "metric_code"])
        .pivot_table(
            index=["period_id", "fiscal_year", "fiscal_year_end_date"],
            columns="metric_code",
            values="metric_value",
            aggfunc="first",
        )
        .reset_index()
    )
    latest_metrics = metric_pivot.loc[metric_pivot["period_id"] == selected_period_id].iloc[0]
    icr_trend = derive_absolute_trend(
        metric_pivot["ICR"],
        threshold=0.15,
        up_label="Improving",
        down_label="Deteriorating",
    )
    dpu_trend = derive_relative_trend(
        dividend_history_df.sort_values("fiscal_year_end_date")["dps_fy"],
        threshold_ratio=0.05,
        up_label="Rising",
        down_label="Falling",
    )
    top_geo_share = latest_metrics.get("REV_CONC_TOPGEO", float("nan"))
    top_geo_component = selected_component_df.loc[
        (selected_component_df["period_id"] == selected_period_id)
        & (selected_component_df["metric_code"] == "REV_CONC_TOPGEO")
        & (selected_component_df["component_name"] == "top_geography_value")
    ]
    top_geo_label = (
        str(top_geo_component.iloc[0]["component_text"])
        if not top_geo_component.empty and pd.notna(top_geo_component.iloc[0]["component_text"])
        else "N/A"
    )
    refi_cliff_component = selected_component_df.loc[
        (selected_component_df["period_id"] == selected_period_id)
        & (selected_component_df["metric_code"] == "REFI_RISK")
        & (selected_component_df["component_name"] == "Short Term Debt")
    ]
    refi_cliff_value = (
        float(refi_cliff_component.iloc[0]["component_value"])
        if not refi_cliff_component.empty and pd.notna(refi_cliff_component.iloc[0]["component_value"])
        else None
    )
    return {
        "ticker": selected_ticker,
        "period_id": selected_period_id,
        "fiscal_year": int(selected_row["fiscal_year"]),
        "reit_name": str(selected_row["reit_name"]),
        "sector": str(selected_row["sector"]),
        "distress_score_mamdani": float(selected_row["distress_score_mamdani"]),
        "final_distress": float(final_distress),
        "final_level": str(final_level),
        "latest_metrics": latest_metrics,
        "icr": latest_metrics.get("ICR", float("nan")),
        "gearing": latest_metrics.get("GEARING", float("nan")),
        "icr_trend": icr_trend,
        "topgeo_label": top_geo_label,
        "topgeo_share": None if pd.isna(top_geo_share) else float(top_geo_share),
        "dpu_trend": dpu_trend,
        "refi_cliff_value": refi_cliff_value,
        "metric_pivot": metric_pivot,
    }


def render_ranking_intro_card() -> None:
    st.markdown(
        """
        <div class="reit-card">
        <div class="reit-kicker">REITs Ranking</div>
        <div class="reit-muted">The ranking starts with each REIT's annual Mamdani fuzzy-rule score, then applies a macro rate-stress overlay and a REIT-specific abnormal-return-path overlay as of the selected simulation date.</div>
        <div class="reit-muted" style="font-size: 0.85rem;"><em>Refinancing Risk Ratio</em> (short-term debt divided by total debt) drives both the per-ticker macro sensitivity weight and the REFI-only stress proxy, while the abnormal-return path is built from REIT daily returns minus the SGX iEdge REIT index daily return.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(RANKING_TABLE_HELP)


def render_ranking_page() -> None:
    filter_col, _ = st.columns([1.1, 4.2])
    with filter_col:
        selected_date = get_selected_simulation_date()
    ranking_view, macro_row, distress_sora = build_ranking_view(
        fuzzy_df,
        metric_df,
        component_df,
        macro_df,
        car_path_df,
        selected_date,
    )
    render_macro_header(macro_row=macro_row, distress_sora=distress_sora)
    render_ranking_intro_card()
    ranking_view.index = ranking_view.index + 1
    st.dataframe(
        ranking_view[
            [
                "ticker",
                "reit_name",
                "sector",
                "fiscal_year_end_date",
                "icr",
                "gearing",
                "dscr",
                "top_revenue_geography",
                "top_revenue_geo_share",
                "refi_risk",
                "distress_score_mamdani",
                "distress_score_refi",
                "macro_sensitivity",
                "macro_overlay_adjustment",
                "accum_car_to_date",
                "final_distress",
                "level",
                "label_126wd",
            ]
        ].rename(
            columns={
                "ticker": "Ticker",
                "reit_name": "REIT Name",
                "sector": "Sector",
                "fiscal_year_end_date": "Annual Filing Anchor",
                "icr": "Interest Coverage Ratio (ICR)",
                "gearing": "Gearing Ratio",
                "dscr": "Debt Service Coverage Ratio (DSCR)",
                "top_revenue_geography": "Top Revenue Geography",
                "top_revenue_geo_share": "% Of Revenue",
                "refi_risk": "Refinancing Risk Ratio",
                "distress_score_mamdani": "Annual Mamdani Base Score",
                "distress_score_refi": "REFI-Only Stress Proxy",
                "macro_sensitivity": "Macro Sensitivity Weight",
                "macro_overlay_adjustment": "Macro Score Adjustment",
                "accum_car_to_date": "Accumulated Abnormal Return To Date",
                "final_distress": "Final Runtime Distress Score",
                "level": "Runtime Distress Level",
                "label_126wd": "126-Day CAR Label",
            }
        ),
        column_config={
            "Annual Filing Anchor": st.column_config.DateColumn(
                "Annual Filing Anchor",
                help="Fiscal-year-end anchor date from reit_metrics.dim_period used to select the annual filing row."
            ),
            "Interest Coverage Ratio (ICR)": st.column_config.NumberColumn(
                "Interest Coverage Ratio (ICR)",
                help=ICR_HELP,
                format="%.3f",
            ),
            "Gearing Ratio": st.column_config.NumberColumn(
                "Gearing Ratio",
                help=GEARING_HELP,
                format="%.3f%%",
            ),
            "Debt Service Coverage Ratio (DSCR)": st.column_config.NumberColumn(
                "Debt Service Coverage Ratio (DSCR)",
                help=DSCR_HELP,
                format="%.3f",
            ),
            "Top Revenue Geography": st.column_config.TextColumn(
                "Top Revenue Geography",
                help=TOP_GEO_HELP,
            ),
            "% Of Revenue": st.column_config.NumberColumn(
                "% Of Revenue",
                help=TOP_GEO_SHARE_HELP,
                format="%.3f%%",
            ),
            "Refinancing Risk Ratio": st.column_config.NumberColumn(
                "Refinancing Risk Ratio",
                help=REFI_HELP,
                format="%.2f",
            ),
            "Annual Mamdani Base Score": st.column_config.NumberColumn(
                "Annual Mamdani Base Score",
                help=ANNUAL_MAMDANI_HELP,
                format="%.2f",
            ),
            "REFI-Only Stress Proxy": st.column_config.NumberColumn(
                "REFI-Only Stress Proxy",
                help=REFI_PROXY_HELP,
                format="%.2f",
            ),
            "Macro Sensitivity Weight": st.column_config.NumberColumn(
                "Macro Sensitivity Weight",
                help=MACRO_SENSITIVITY_HELP,
                format="%.2f",
            ),
            "Macro Score Adjustment": st.column_config.NumberColumn(
                "Macro Score Adjustment",
                help=MACRO_ADJUSTMENT_HELP,
                format="%+.3f",
            ),
            "Accumulated Abnormal Return To Date": st.column_config.NumberColumn(
                "Accumulated Abnormal Return To Date",
                help=ACCUM_CAR_HELP,
                format="%.2f%%",
            ),
            "Final Runtime Distress Score": st.column_config.NumberColumn(
                "Final Runtime Distress Score",
                help=FINAL_SCORE_HELP,
                format="%.2f",
            ),
            "Runtime Distress Level": st.column_config.TextColumn(
                "Runtime Distress Level",
                help="Bucket derived from the final runtime distress score: Stable, Watch, High, or Critical."
            ),
            "126-Day CAR Label": st.column_config.TextColumn(
                "126-Day CAR Label",
                help=CAR_LABEL_HELP,
            ),
        },
        width="stretch",
    )


def render_reit_page() -> None:
    filter_date_col, filter_ticker_col, _ = st.columns([1.1, 1.25, 2.65])
    with filter_date_col:
        selected_date = get_selected_simulation_date()
    ranking_view, macro_row, distress_sora = build_ranking_view(
        fuzzy_df,
        metric_df,
        component_df,
        macro_df,
        car_path_df,
        selected_date,
    )
    ticker_options = ranking_view["ticker"].tolist()
    with filter_ticker_col:
        selected_ticker = get_selected_ticker(ticker_options)
    selected_row = ranking_view.loc[ranking_view["ticker"] == selected_ticker].iloc[0]
    selected_metric_df = metric_df.loc[metric_df["ticker"] == selected_ticker].copy()
    selected_label_df = label_df.loc[label_df["ticker"] == selected_ticker].copy()
    selected_component_df = component_df.loc[component_df["ticker"] == selected_ticker].copy()
    selected_dividend_df = dividend_df.loc[dividend_df["ticker"] == selected_ticker].copy()
    selected_period_id = int(selected_row["period_id"])
    selected_period_end = pd.Timestamp(selected_row["fiscal_year_end_date"])
    metric_history_df, label_history_df = build_reit_history_frames(
        selected_metric_df=selected_metric_df,
        selected_label_df=selected_label_df,
        selected_period_end=selected_period_end,
    )

    final_distress = float(selected_row["final_distress"])
    final_level = str(selected_row["level"])
    header_ctx = build_reit_header_context(
        selected_ticker=selected_ticker,
        selected_period_id=selected_period_id,
        selected_row=selected_row,
        selected_metric_df=selected_metric_df,
        selected_component_df=selected_component_df,
        selected_dividend_df=selected_dividend_df,
        final_distress=final_distress,
        final_level=final_level,
    )
    st.markdown(
        f"""
        <div class="reit-card">
        <div class="reit-kicker">{header_ctx['ticker']}</div>
        <div style="font-size: 3rem; font-weight: 800;">{header_ctx['reit_name']}</div>
        <div class="reit-muted">{header_ctx['sector']} | Fiscal year {header_ctx['fiscal_year']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_expanded_reit_header(
        final_distress=header_ctx["final_distress"],
        final_level=header_ctx["final_level"],
        latest_metrics=header_ctx["latest_metrics"],
        icr_trend=header_ctx["icr_trend"],
        top_geo_label=header_ctx["topgeo_label"],
        top_geo_share=header_ctx["topgeo_share"],
        refi_cliff_value=header_ctx["refi_cliff_value"],
        dpu_trend=header_ctx["dpu_trend"],
    )

    tab_score, tab_financials = st.tabs(["Score", "Financial Statements"])

    with tab_score:
        c1, c2 = st.columns([1.05, 1.4])
        with c1:
            with st.container(border=True):
                st.markdown("**Final Runtime Distress Score**", help=FINAL_SCORE_BUILD_HELP)
                st.metric("", f"{final_distress:.2f}", help=FINAL_SCORE_HELP, label_visibility="collapsed")
                score_cols = st.columns([1.25, 1, 1, 1])
                score_cols[0].metric("Runtime Level", final_level, help="Bucket derived from the final runtime distress score.")
                score_cols[1].metric("Annual Mamdani Base", f"{header_ctx['distress_score_mamdani']:.2f}", help=ANNUAL_MAMDANI_HELP)
                score_cols[2].metric("Macro Overlay", f"{distress_sora:.2f}", help=MACRO_DISTRESS_HELP)
                score_cols[3].metric(
                    "CAR-Path Overlay",
                    "N/A" if pd.isna(selected_row["car_path_distress"]) else f"{selected_row['car_path_distress']:.2f}",
                    help=CAR_PATH_DISTRESS_HELP,
                )
                st.metric(
                    "Annual Filing Anchor",
                    pd.Timestamp(selected_row["fiscal_year_end_date"]).strftime("%Y-%m-%d"),
                    help=ANNUAL_ANCHOR_HELP,
                )
        with c2:
            stats = st.columns(5)
            stats[0].metric("Debt Service Coverage Ratio (DSCR)", f"{header_ctx['latest_metrics'].get('DSCR', float('nan')):.2f}", help=DSCR_HELP)
            stats[1].metric("Refinancing Risk Ratio", f"{header_ctx['latest_metrics'].get('REFI_RISK', float('nan')):.2f}", help=REFI_HELP)
            stats[2].metric("Payout Ratio", f"{header_ctx['latest_metrics'].get('PAYOUT_RATIO', float('nan')):.2f}", help=PAYOUT_HELP)
            stats[3].metric("Annual Metric Issues", f"{int(selected_row['non_ok_count'])}", help=NON_OK_COUNT_HELP)
            stats[4].metric("Fired Fuzzy Rules", f"{int(selected_row['fired_rule_count'])}", help=FIRED_RULE_COUNT_HELP)
            with st.expander("Why did the fuzzy rule engine flag this REIT?", expanded=True):
                st.caption(RULE_TRACE_HELP)
                st.text(selected_row["rule_trace_text"] or "No rule trace available.")
        c3, c4 = st.columns(2)
        with c3:
            selected_label = get_label_row_for_period(
                selected_label_df,
                ticker=selected_ticker,
                period_id=selected_period_id,
            )
            st.markdown("**REIT Market Path Since Filing**", help=CAR_PANEL_HELP)
            st.metric(
                "Accumulated Abnormal Return To Date",
                "N/A" if pd.isna(selected_row["accum_car_to_date"]) else f"{selected_row['accum_car_to_date']:.2%}",
                help=ACCUM_CAR_HELP,
            )
            st.metric(
                "Abnormal-Return-Path Overlay Score",
                "N/A" if pd.isna(selected_row["car_path_distress"]) else f"{selected_row['car_path_distress']:.2f}",
                help=CAR_PATH_DISTRESS_HELP,
            )
            st.metric("Forward 63-Trading-Day Abnormal Return", "N/A" if pd.isna(selected_label["car_63wd"]) else f"{selected_label['car_63wd']:.2%}", help=CAR_63_HELP)
            st.metric("Forward 126-Trading-Day Abnormal Return", "N/A" if pd.isna(selected_label["car_126wd"]) else f"{selected_label['car_126wd']:.2%}", help=CAR_126_HELP)
            st.metric("126-Day Distress Label", selected_label["label_126wd"] or "PENDING", help=CAR_LABEL_HELP)
        with c4:
            macro_panel = build_macro_panel_context(macro_row, distress_sora)
            st.markdown("**Macro Rate Overlay**", help=MACRO_PANEL_HELP)
            st.metric("Predicted 10-Trading-Day SORA Change", f"{macro_panel['predicted_change']:+.3f}", help=MACRO_CHANGE_HELP)
            st.metric("Predicted SORA Level In 10 Trading Days", f"{macro_panel['predicted_level']:.3f}", help=MACRO_LEVEL_HELP)
            st.metric("FOMC Decision Date", macro_panel["fomc_decision_date"].strftime("%Y-%m-%d"), help=FOMC_HELP)

    with tab_financials:
        st.markdown("**Annual metric history**")
        financial_pivot = metric_history_df.pivot_table(
            index=["fiscal_year", "fiscal_year_end_date"],
            columns="metric_code",
            values="metric_value",
            aggfunc="first",
        ).reset_index()
        st.dataframe(financial_pivot, width="stretch")
        with st.expander("Label history"):
            st.dataframe(label_history_df, width="stretch")


def render_rates_page() -> None:
    latest_distress_sora = compute_sora_distress_score(float(latest_macro["y_pred"]))
    render_macro_header(macro_row=latest_macro, distress_sora=latest_distress_sora)
    st.markdown("**SORA macro walk-forward**")
    st.markdown(
        """
        <div class="reit-section-banner reit-section-change">
            <h3>Change Forecasts</h3>
            <p>Direct 10-day change predictions compared against actual 10-day changes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hover_change_oos = alt.selection_point(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty=False,
        clear="mouseout",
    )

    change_actual = macro_df[["snapshot_ts", "y_true"]].copy().rename(
        columns={"snapshot_ts": "date", "y_true": "actual_sora_fwd_10d_change"}
    )
    change_pred = macro_holdout_df[["snapshot_ts", "y_pred"]].copy().rename(
        columns={"snapshot_ts": "date", "y_pred": "predicted_change_10d"}
    )
    change_chart = change_actual.merge(change_pred, on="date", how="left", validate="one_to_one")
    st.markdown("**OOS holdout: Predicted 10D change vs actual 10D change**")
    change_long = change_chart.melt(
        id_vars="date",
        value_vars=["predicted_change_10d", "actual_sora_fwd_10d_change"],
        var_name="series",
        value_name="value",
    )
    train_rule_df = pd.DataFrame(
        {
            "date": [macro_train_end],
            "label": [f"Train end: {macro_train_end.strftime('%Y-%m-%d')}"],
        }
    )
    change_base = alt.Chart(change_long).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Change"),
        color=alt.Color("series:N", title="Series"),
    )
    change_hover_base = alt.Chart(change_chart).encode(
        x=alt.X("date:T", title="Date"),
    )
    change_plot = (
        change_base.mark_line()
        + alt.Chart(train_rule_df).mark_rule(color="#ff4a4a", strokeDash=[6, 4]).encode(x="date:T")
        + alt.Chart(train_rule_df)
        .mark_text(color="#ff4a4a", align="left", dx=6, dy=-120)
        .encode(x="date:T", text="label:N")
        + change_hover_base.mark_point(opacity=0, size=120).add_params(hover_change_oos).encode(
            y=alt.value(0),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("predicted_change_10d:Q", title="Predicted 10D Change", format=".3f"),
                alt.Tooltip("actual_sora_fwd_10d_change:Q", title="Actual 10D Change", format=".3f"),
            ],
        )
        + change_base.mark_circle(size=60).transform_filter(hover_change_oos)
        + change_hover_base.mark_rule(color="#8b949e")
        .transform_filter(hover_change_oos)
    )
    st.altair_chart(change_plot.properties(height=260), use_container_width=True)

    hover_change_full = alt.selection_point(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty=False,
        clear="mouseout",
    )
    change_full_chart = macro_df[["snapshot_ts", "y_pred", "y_true"]].copy().rename(
        columns={
            "snapshot_ts": "date",
            "y_pred": "predicted_change_10d",
            "y_true": "actual_sora_fwd_10d_change",
        }
    )
    change_full_long = change_full_chart.melt(
        id_vars="date",
        value_vars=["predicted_change_10d", "actual_sora_fwd_10d_change"],
        var_name="series",
        value_name="value",
    )
    change_full_base = alt.Chart(change_full_long).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Change"),
        color=alt.Color("series:N", title="Series"),
    )
    change_full_hover_base = alt.Chart(change_full_chart).encode(
        x=alt.X("date:T", title="Date"),
    )
    change_full_plot = (
        change_full_base.mark_line()
        + alt.Chart(train_rule_df).mark_rule(color="#ff4a4a", strokeDash=[6, 4]).encode(x="date:T")
        + alt.Chart(train_rule_df)
        .mark_text(color="#ff4a4a", align="left", dx=6, dy=-120)
        .encode(x="date:T", text="label:N")
        + change_full_hover_base.mark_point(opacity=0, size=120).add_params(hover_change_full).encode(
            y=alt.value(0),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("predicted_change_10d:Q", title="Predicted 10D Change", format=".3f"),
                alt.Tooltip("actual_sora_fwd_10d_change:Q", title="Actual 10D Change", format=".3f"),
            ],
        )
        + change_full_base.mark_circle(size=60).transform_filter(hover_change_full)
        + change_full_hover_base.mark_rule(color="#8b949e")
        .transform_filter(hover_change_full)
    )
    st.markdown("**Refit full-model: Predicted 10D change vs actual 10D change**")
    st.altair_chart(change_full_plot.properties(height=260), use_container_width=True)

    horizon = DEFAULT_HORIZON_DAYS
    st.markdown(
        """
        <div class="reit-section-banner reit-section-level">
            <h3>Level Forecasts</h3>
            <p>10-day-ahead level forecasts shifted onto their target date and compared against realized future levels.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    hover_level_oos = alt.selection_point(
        fields=["target_date"],
        nearest=True,
        on="mouseover",
        empty=False,
        clear="mouseout",
    )
    full_target_dates = macro_df[["snapshot_ts"]].copy()
    full_target_dates["target_date"] = full_target_dates["snapshot_ts"].shift(-horizon)
    actual_level_chart = macro_df[[f"sora_fwd_{horizon}d_level"]].copy()
    actual_level_chart["target_date"] = full_target_dates["target_date"]
    actual_level_chart = actual_level_chart.rename(
        columns={
            f"sora_fwd_{horizon}d_level": "actual_future_level_10d",
        }
    ).dropna(subset=["target_date"])
    predicted_level_chart = macro_holdout_df[["target_date", "predicted_level"]].copy().rename(
        columns={"predicted_level": "predicted_level_10d"}
    )
    future_level_chart = actual_level_chart.merge(
        predicted_level_chart, on="target_date", how="left", validate="one_to_one"
    )
    boundary_target_matches = full_target_dates.loc[full_target_dates["snapshot_ts"] == macro_train_end, "target_date"]
    if boundary_target_matches.empty or pd.isna(boundary_target_matches.iloc[0]):
        raise ValueError(
            f"Could not align train_end {macro_train_end.strftime('%Y-%m-%d')} "
            "to a target_date in the full future level chart."
        )
    boundary_target_date = pd.Timestamp(boundary_target_matches.iloc[0])
    st.markdown("**OOS holdout: Horizon-shifted predicted 10D level vs actual future level**")
    level_long = future_level_chart.melt(
        id_vars="target_date",
        value_vars=["predicted_level_10d", "actual_future_level_10d"],
        var_name="series",
        value_name="value",
    )
    future_train_rule_df = pd.DataFrame(
        {
            "target_date": [boundary_target_date],
            "label": [f"Forecasts trained through {macro_train_end.strftime('%Y-%m-%d')}"],
        }
    )
    level_base = alt.Chart(level_long).encode(
        x=alt.X("target_date:T", title="Forecast target date"),
        y=alt.Y("value:Q", title="Level"),
        color=alt.Color("series:N", title="Series"),
    )
    level_hover_base = alt.Chart(future_level_chart).encode(
        x=alt.X("target_date:T", title="Forecast target date"),
    )
    level_plot = (
        level_base.mark_line()
        + alt.Chart(future_train_rule_df).mark_rule(color="#ff4a4a", strokeDash=[6, 4]).encode(x="target_date:T")
        + alt.Chart(future_train_rule_df)
        .mark_text(color="#ff4a4a", align="left", dx=6, dy=-120)
        .encode(x="target_date:T", text="label:N")
        + level_hover_base.mark_point(opacity=0, size=120).add_params(hover_level_oos).encode(
            y=alt.value(0),
            tooltip=[
                alt.Tooltip("target_date:T", title="Forecast target date"),
                alt.Tooltip("predicted_level_10d:Q", title="Predicted 10D Level", format=".3f"),
                alt.Tooltip("actual_future_level_10d:Q", title="Actual Future Level", format=".3f"),
            ],
        )
        + level_base.mark_circle(size=60).transform_filter(hover_level_oos)
        + level_hover_base.mark_rule(color="#8b949e")
        .transform_filter(hover_level_oos)
    )
    st.altair_chart(level_plot.properties(height=260), use_container_width=True)

    hover_level_full = alt.selection_point(
        fields=["target_date"],
        nearest=True,
        on="mouseover",
        empty=False,
        clear="mouseout",
    )
    predicted_level_full_chart = full_target_dates.copy()
    predicted_level_full_chart["predicted_level_10d"] = macro_df["predicted_level"]
    predicted_level_full_chart = predicted_level_full_chart.dropna(subset=["target_date"])
    future_level_full_chart = actual_level_chart.merge(
        predicted_level_full_chart[["target_date", "predicted_level_10d"]],
        on="target_date",
        how="left",
        validate="one_to_one",
    )
    level_full_long = future_level_full_chart.melt(
        id_vars="target_date",
        value_vars=["predicted_level_10d", "actual_future_level_10d"],
        var_name="series",
        value_name="value",
    )
    level_full_base = alt.Chart(level_full_long).encode(
        x=alt.X("target_date:T", title="Forecast target date"),
        y=alt.Y("value:Q", title="Level"),
        color=alt.Color("series:N", title="Series"),
    )
    level_full_hover_base = alt.Chart(future_level_full_chart).encode(
        x=alt.X("target_date:T", title="Forecast target date"),
    )
    level_full_plot = (
        level_full_base.mark_line()
        + alt.Chart(future_train_rule_df).mark_rule(color="#ff4a4a", strokeDash=[6, 4]).encode(x="target_date:T")
        + alt.Chart(future_train_rule_df)
        .mark_text(color="#ff4a4a", align="left", dx=6, dy=-120)
        .encode(x="target_date:T", text="label:N")
        + level_full_hover_base.mark_point(opacity=0, size=120).add_params(hover_level_full).encode(
            y=alt.value(0),
            tooltip=[
                alt.Tooltip("target_date:T", title="Forecast target date"),
                alt.Tooltip("predicted_level_10d:Q", title="Predicted 10D Level", format=".3f"),
                alt.Tooltip("actual_future_level_10d:Q", title="Actual Future Level", format=".3f"),
            ],
        )
        + level_full_base.mark_circle(size=60).transform_filter(hover_level_full)
        + level_full_hover_base.mark_rule(color="#8b949e")
        .transform_filter(hover_level_full)
    )
    st.markdown("**Refit full-model: Horizon-shifted predicted 10D level vs actual future level**")
    st.altair_chart(level_full_plot.properties(height=260), use_container_width=True)
    st.caption(
        "Charts 1 and 3 use saved OOS holdout predictions from the winning XGBoost run. "
        "Charts 2 and 4 use the original refit full-model predictions, which are not OOS. "
        "The red dashed marker shows where the training period ended."
    )


navigation = st.navigation(
    [
        st.Page(render_ranking_page, title="Ranking", icon=":material/leaderboard:"),
        st.Page(render_reit_page, title="Individual REIT Navigator", icon=":material/apartment:"),
        st.Page(render_rates_page, title="Time Series (Rates)", icon=":material/show_chart:"),
    ],
    position="sidebar",
    expanded=True,
)
navigation.run()
