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
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_app_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    finally:
        con.close()
    return ranking_df, metric_df, label_df


ranking_df, metric_df, label_df = load_app_frames()
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
            selected_metric_df.sort_values("fiscal_year")
            .pivot_table(index="fiscal_year", columns="metric_code", values="metric_value", aggfunc="first")
            .reset_index()
        )
        latest_metrics = metric_pivot.iloc[-1]
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
