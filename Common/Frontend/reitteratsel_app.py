from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
KG_DIR = ROOT_DIR / "Common" / "Micro" / "5_Model_KG"
if str(KG_DIR) not in sys.path:
    sys.path.insert(0, str(KG_DIR))

from reitteratsel_core import (  # noqa: E402
    DEFAULT_HORIZON_DAYS,
    DUCKDB_PATH,
    compute_final_distress_score,
    compute_sora_distress_score,
    load_macro_prediction_frame,
    score_to_level,
)


st.set_page_config(page_title="REITterratsel", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #181515 0%, #231f1f 100%);
        color: #f2f0ec;
    }
    [data-testid="stSidebar"] {
        background: #171414;
        border-right: 1px solid #3d3434;
    }
    .block-container {
        padding-top: 1rem;
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
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_app_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        ranking_df = con.execute(
            """
            WITH latest_fuzzy AS (
                SELECT
                    f.*,
                    p.fiscal_year,
                    p.fiscal_year_end_date,
                    r.reit_name,
                    r.sector,
                    ROW_NUMBER() OVER (PARTITION BY f.ticker ORDER BY p.fiscal_year_end_date DESC) AS rn
                FROM reit_fuzzy.fact_fuzzy_cache f
                JOIN reit_metrics.dim_period p ON p.period_id = f.period_id
                JOIN reit_metrics.dim_reit r ON r.ticker = f.ticker
            ),
            latest_labels AS (
                SELECT
                    l.ticker,
                    l.period_id,
                    l.car_63wd,
                    l.car_126wd,
                    l.label_126wd,
                    ROW_NUMBER() OVER (PARTITION BY l.ticker ORDER BY p.fiscal_year_end_date DESC) AS rn
                FROM reit_labels.fact_distress_label l
                JOIN reit_metrics.dim_period p ON p.period_id = l.period_id
            )
            SELECT
                f.ticker,
                f.period_id,
                f.reit_name,
                f.sector,
                f.fiscal_year,
                f.fiscal_year_end_date,
                f.distress_score_mamdani,
                f.distress_level,
                f.null_count,
                f.non_ok_count,
                f.top_rule_ids,
                f.rule_trace_text,
                l.car_63wd,
                l.car_126wd,
                l.label_126wd
            FROM latest_fuzzy f
            LEFT JOIN latest_labels l
              ON l.ticker = f.ticker
             AND l.rn = 1
            WHERE f.rn = 1
            ORDER BY f.distress_score_mamdani DESC, f.ticker
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
    return ranking_df, metric_df, label_df, component_df, dividend_df


ranking_df, metric_df, label_df, component_df, dividend_df = load_app_frames()
macro_df = load_macro_prediction_frame(DEFAULT_HORIZON_DAYS)
if macro_df["prediction_source"].nunique() != 1 or macro_df["prediction_source"].iloc[0] != "xgboost_final_model":
    raise RuntimeError(
        "Macro runtime must use direct XGBoost inference from the run_21 model artifacts."
    )
latest_macro = macro_df.iloc[-1]
latest_refi_by_ticker = (
    metric_df.loc[metric_df["metric_code"] == "REFI_RISK"]
    .sort_values(["ticker", "fiscal_year"])
    .groupby("ticker", as_index=False)
    .tail(1)[["ticker", "metric_value"]]
    .rename(columns={"metric_value": "latest_refi_risk"})
)
ranking_df = ranking_df.merge(latest_refi_by_ticker, on="ticker", how="left")
distress_sora = compute_sora_distress_score(float(latest_macro["y_pred"]))


def render_macro_header() -> None:
    st.markdown('<div class="reit-kicker">Prediction from overnight bank rates</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("10D Predicted SORA Change", f"{latest_macro['y_pred']:+.3f}")
    m2.metric("Predicted SORA Level", f"{latest_macro['predicted_level']:.3f}")
    m3.metric("Macro Distress", f"{distress_sora:.2f}")
    m4.metric("Snapshot Date", latest_macro["snapshot_ts"].strftime("%Y-%m-%d"))


def build_ranking_view() -> pd.DataFrame:
    ranking_view = ranking_df.copy()
    ranking_view["final_distress"] = ranking_view.apply(
        lambda row: compute_final_distress_score(
            float(row["distress_score_mamdani"]),
            distress_sora,
            row["latest_refi_risk"],
        ),
        axis=1,
    )
    ranking_view["level"] = ranking_view["final_distress"].map(score_to_level)
    return ranking_view.sort_values("final_distress", ascending=False).reset_index(drop=True)


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
    clean = pd.to_numeric(values, errors="coerce").dropna().reset_index(drop=True)
    if len(clean) < 2:
        return "N/A"
    baseline = clean.iloc[:-1].mean() if len(clean) >= 3 else clean.iloc[-2]
    delta = float(clean.iloc[-1] - baseline)
    if abs(delta) < threshold:
        return "Flat"
    return up_label if delta > 0 else down_label


def derive_relative_trend(values: pd.Series, *, threshold_ratio: float, up_label: str, down_label: str) -> str:
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
        top_geo_subvalue = f"{top_geo_share:.0%} of revenue"
    else:
        top_geo_subvalue = "Share unavailable"
    header_html = f"""
    <div class="reit-card">
      <div class="reit-header-grid">
        <div class="reit-header-score">
          <div class="reit-label">Distress Score</div>
          <div class="reit-value reit-value-score">{final_distress:.2f}</div>
          <div class="reit-subvalue">Final level: {final_level}</div>
        </div>
        <div class="reit-thresholds">
          {build_threshold_pills(final_level)}
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">ICR</div>
          <div class="reit-value">{latest_metrics.get('ICR', float('nan')):.2f}</div>
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">Gearing</div>
          <div class="reit-value">{latest_metrics.get('GEARING', float('nan')):.1%}</div>
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">ICR Trend</div>
          <div class="reit-value" style="font-size: 1.7rem;">{icr_trend}</div>
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">TOPGEO</div>
          <div class="reit-value" style="font-size: 1.7rem;">{top_geo_display}</div>
          <div class="reit-subvalue">{top_geo_subvalue}</div>
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">Refi Cliff</div>
          <div class="reit-value" style="font-size: 1.8rem;">{format_currency_compact(refi_cliff_value)}</div>
          <div class="reit-subvalue">Short-term debt</div>
        </div>
        <div class="reit-metric-box">
          <div class="reit-label">DPU Trend</div>
          <div class="reit-value" style="font-size: 1.7rem;">{dpu_trend}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_ranking_page() -> None:
    render_macro_header()
    st.markdown(
        """
        <div class="reit-card">
        <div class="reit-kicker">REITs Ranking</div>
        <div class="reit-muted">Runtime ranking combines frozen Mamdani score with the current 10D macro layer.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ranking_view = build_ranking_view()
    ranking_view.index = ranking_view.index + 1
    st.dataframe(
        ranking_view[
            ["ticker", "reit_name", "sector", "distress_score_mamdani", "final_distress", "level", "label_126wd"]
        ].rename(
            columns={
                "ticker": "Ticker",
                "reit_name": "Name",
                "sector": "Sector",
                "distress_score_mamdani": "Mamdani",
                "final_distress": "Final Score",
                "level": "Level",
                "label_126wd": "CAR Label",
            }
        ),
        width="stretch",
    )


def render_reit_page() -> None:
    render_macro_header()
    ranking_view = build_ranking_view()
    ticker_options = ranking_view["ticker"].tolist()
    default_ticker = st.session_state.get("selected_ticker", ticker_options[0])
    if default_ticker not in ticker_options:
        default_ticker = ticker_options[0]
    selected_ticker = st.selectbox(
        "Select REIT",
        ticker_options,
        index=ticker_options.index(default_ticker),
        key="selected_ticker",
    )
    selected_row = ranking_df.loc[ranking_df["ticker"] == selected_ticker].iloc[0]
    selected_metric_df = metric_df.loc[metric_df["ticker"] == selected_ticker].copy()
    selected_label_df = label_df.loc[label_df["ticker"] == selected_ticker].copy()
    selected_component_df = component_df.loc[component_df["ticker"] == selected_ticker].copy()
    selected_dividend_df = dividend_df.loc[dividend_df["ticker"] == selected_ticker].copy()
    selected_period_id = int(selected_row["period_id"])

    selected_refi = selected_metric_df.loc[
        selected_metric_df["metric_code"] == "REFI_RISK"
    ].sort_values("fiscal_year").iloc[-1]["metric_value"]
    final_distress = compute_final_distress_score(
        float(selected_row["distress_score_mamdani"]),
        distress_sora,
        selected_refi,
    )
    final_level = score_to_level(final_distress)

    tab_score, tab_financials = st.tabs(["Score", "Financial Statements"])

    with tab_score:
        metric_pivot = (
            selected_metric_df.sort_values(["fiscal_year_end_date", "metric_code"])
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
            selected_dividend_df.sort_values("fiscal_year_end_date")["dps_fy"],
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
        st.markdown(
            f"""
            <div class="reit-card">
            <div class="reit-kicker">{selected_ticker}</div>
            <div style="font-size: 3rem; font-weight: 800;">{selected_row['reit_name']}</div>
            <div class="reit-muted">{selected_row['sector']} | Fiscal year {int(selected_row['fiscal_year'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_expanded_reit_header(
            final_distress=final_distress,
            final_level=final_level,
            latest_metrics=latest_metrics,
            icr_trend=icr_trend,
            top_geo_label=top_geo_label,
            top_geo_share=(None if pd.isna(top_geo_share) else float(top_geo_share)),
            refi_cliff_value=refi_cliff_value,
            dpu_trend=dpu_trend,
        )
        c1, c2 = st.columns([1.05, 1.4])
        with c1:
            st.markdown(
                f"""
                <div class="reit-card">
                <div class="reit-kicker">Distress Score</div>
                <div class="reit-score">{final_distress:.2f}</div>
                <div class="reit-muted">Level: {final_level}</div>
                <div class="reit-muted">Frozen Mamdani: {selected_row['distress_score_mamdani']:.2f}</div>
                <div class="reit-muted">Macro layer: {distress_sora:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            stats = st.columns(6)
            stats[0].metric("ICR", f"{latest_metrics.get('ICR', float('nan')):.2f}")
            stats[1].metric("GEARING", f"{latest_metrics.get('GEARING', float('nan')):.2f}")
            stats[2].metric("DSCR", f"{latest_metrics.get('DSCR', float('nan')):.2f}")
            stats[3].metric("REFI RISK", f"{latest_metrics.get('REFI_RISK', float('nan')):.2f}")
            stats[4].metric("PAYOUT", f"{latest_metrics.get('PAYOUT_RATIO', float('nan')):.2f}")
            stats[5].metric("NULL COUNT", f"{int(selected_row['null_count'])}")
            with st.expander("Why was this flagged?", expanded=True):
                st.text(selected_row["rule_trace_text"] or "No rule trace available.")
        c3, c4 = st.columns(2)
        with c3:
            latest_label = selected_label_df.iloc[-1]
            st.markdown("**Forward CAR Panel**")
            st.metric("CAR 63wd", "N/A" if pd.isna(latest_label["car_63wd"]) else f"{latest_label['car_63wd']:.2%}")
            st.metric("CAR 126wd", "N/A" if pd.isna(latest_label["car_126wd"]) else f"{latest_label['car_126wd']:.2%}")
            st.metric("CAR Label", latest_label["label_126wd"] or "PENDING")
        with c4:
            st.markdown("**Macro Layer**")
            st.metric("Predicted 10D Change", f"{latest_macro['y_pred']:+.3f}")
            st.metric("Predicted 10D Level", f"{latest_macro['predicted_level']:.3f}")
            st.metric("FOMC Decision", latest_macro["fomc_decision_date"].strftime("%Y-%m-%d"))

    with tab_financials:
        st.markdown("**Annual metric history**")
        financial_pivot = selected_metric_df.pivot_table(
            index=["fiscal_year", "fiscal_year_end_date"],
            columns="metric_code",
            values="metric_value",
            aggfunc="first",
        ).reset_index()
        st.dataframe(financial_pivot, width="stretch")
        with st.expander("Label history"):
            st.dataframe(selected_label_df, width="stretch")


def render_rates_page() -> None:
    render_macro_header()
    st.markdown("**SORA macro walk-forward**")
    macro_chart = macro_df[["snapshot_ts", "y_pred", "predicted_level", "sora_level_realized"]].copy()
    macro_chart = macro_chart.rename(
        columns={
            "snapshot_ts": "date",
            "y_pred": "predicted_change_10d",
            "predicted_level": "predicted_level_10d",
            "sora_level_realized": "realized_level",
        }
    ).set_index("date")
    st.line_chart(macro_chart, height=360)
    st.caption("Current implementation uses direct 10D XGBoost inference from the local run_21 model artifacts.")


navigation = st.navigation(
    [
        st.Page(render_ranking_page, title="Ranking"),
        st.Page(render_reit_page, title="Individual REIT Navigator"),
        st.Page(render_rates_page, title="Time Series (Rates)"),
    ],
    position="sidebar",
    expanded=True,
)
navigation.run()
