from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
KG_DIR = ROOT_DIR / "Common" / "Micro" / "5_Model_KG"
if str(KG_DIR) not in sys.path:
    sys.path.insert(0, str(KG_DIR))

from reitteratsel_core import (
    compute_final_distress_score,
    compute_refi_distress_score,
    compute_sora_distress_score,
    score_to_level,
)


def normalize_selected_date(selected_date: Any) -> pd.Timestamp:
    return pd.Timestamp(selected_date).normalize()


def resolve_macro_row(macro_df: pd.DataFrame, selected_date: Any) -> pd.Series:
    target_ts = normalize_selected_date(selected_date)
    eligible = macro_df.loc[macro_df["snapshot_ts"] <= target_ts].sort_values("snapshot_ts")
    if eligible.empty:
        raise ValueError(f"No macro snapshot exists on or before selected date {target_ts.date()}.")
    return eligible.iloc[-1]


def resolve_latest_period_rows(period_df: pd.DataFrame, selected_date: Any) -> pd.DataFrame:
    target_ts = normalize_selected_date(selected_date)
    eligible = period_df.loc[period_df["fiscal_year_end_date"] <= target_ts].copy()
    if eligible.empty:
        raise ValueError(f"No annual period rows exist on or before selected date {target_ts.date()}.")
    resolved = (
        eligible.sort_values(["ticker", "fiscal_year_end_date", "period_id"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return resolved


def get_metric_value_for_period(
    metric_df: pd.DataFrame,
    *,
    ticker: str,
    period_id: int,
    metric_code: str,
) -> float | None:
    matches = metric_df.loc[
        (metric_df["ticker"] == ticker)
        & (metric_df["period_id"] == period_id)
        & (metric_df["metric_code"] == metric_code)
    ]
    if matches.empty:
        return None
    value = matches.iloc[0]["metric_value"]
    if pd.isna(value):
        return None
    return float(value)


def get_label_row_for_period(label_df: pd.DataFrame, *, ticker: str, period_id: int) -> pd.Series:
    matches = label_df.loc[
        (label_df["ticker"] == ticker)
        & (label_df["period_id"] == period_id)
    ]
    if matches.empty:
        raise ValueError(f"Missing label row for ticker={ticker} period_id={period_id}.")
    return matches.iloc[0]


def get_latest_car_path_row_for_period(
    car_path_df: pd.DataFrame,
    *,
    ticker: str,
    period_id: int,
    selected_date: Any,
) -> pd.Series | None:
    target_ts = normalize_selected_date(selected_date)
    matches = car_path_df.loc[
        (car_path_df["ticker"] == ticker)
        & (car_path_df["period_id"] == period_id)
        & (pd.to_datetime(car_path_df["trade_date"]) <= target_ts)
    ].sort_values("trade_date")
    if matches.empty:
        return None
    return matches.iloc[-1]


def _extract_car_path_value(
    car_path_df: pd.DataFrame,
    *,
    ticker: str,
    period_id: int,
    selected_date: Any,
    field: str,
) -> Any:
    row = get_latest_car_path_row_for_period(
        car_path_df,
        ticker=ticker,
        period_id=period_id,
        selected_date=selected_date,
    )
    if row is None:
        return None
    return row[field]


def build_ranking_view(
    fuzzy_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    component_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    car_path_df: pd.DataFrame,
    selected_date: Any,
) -> tuple[pd.DataFrame, pd.Series, float]:
    macro_row = resolve_macro_row(macro_df, selected_date)
    distress_sora = compute_sora_distress_score(float(macro_row["y_pred"]))
    macro_shock = float(distress_sora) - 0.5
    resolved_rows = resolve_latest_period_rows(fuzzy_df, selected_date).copy()
    resolved_rows["refi_risk"] = resolved_rows.apply(
        lambda row: get_metric_value_for_period(
            metric_df,
            ticker=str(row["ticker"]),
            period_id=int(row["period_id"]),
            metric_code="REFI_RISK",
        ),
        axis=1,
    )
    resolved_rows["icr"] = resolved_rows.apply(
        lambda row: get_metric_value_for_period(
            metric_df,
            ticker=str(row["ticker"]),
            period_id=int(row["period_id"]),
            metric_code="ICR",
        ),
        axis=1,
    )
    resolved_rows["gearing"] = resolved_rows.apply(
        lambda row: get_metric_value_for_period(
            metric_df,
            ticker=str(row["ticker"]),
            period_id=int(row["period_id"]),
            metric_code="GEARING",
        ),
        axis=1,
    )
    resolved_rows["dscr"] = resolved_rows.apply(
        lambda row: get_metric_value_for_period(
            metric_df,
            ticker=str(row["ticker"]),
            period_id=int(row["period_id"]),
            metric_code="DSCR",
        ),
        axis=1,
    )
    resolved_rows["top_revenue_geo_share"] = resolved_rows.apply(
        lambda row: get_metric_value_for_period(
            metric_df,
            ticker=str(row["ticker"]),
            period_id=int(row["period_id"]),
            metric_code="REV_CONC_TOPGEO",
        ),
        axis=1,
    )
    resolved_rows["top_revenue_geography"] = resolved_rows.apply(
        lambda row: (
            str(
                component_df.loc[
                    (component_df["ticker"] == str(row["ticker"]))
                    & (component_df["period_id"] == int(row["period_id"]))
                    & (component_df["metric_code"] == "REV_CONC_TOPGEO")
                    & (component_df["component_name"] == "top_geography_value"),
                    "component_text",
                ].iloc[0]
            )
            if not component_df.loc[
                (component_df["ticker"] == str(row["ticker"]))
                & (component_df["period_id"] == int(row["period_id"]))
                & (component_df["metric_code"] == "REV_CONC_TOPGEO")
                & (component_df["component_name"] == "top_geography_value")
                & (component_df["component_text"].notna())
            ].empty
            else "N/A"
        ),
        axis=1,
    )
    resolved_rows["distress_score_refi"] = resolved_rows["refi_risk"].map(compute_refi_distress_score)
    resolved_rows["macro_sensitivity"] = resolved_rows["refi_risk"].map(
        lambda value: 0.50 if value is None or pd.isna(value) else max(0.25, min(1.0, float(value) * 2.5))
    )
    resolved_rows["macro_overlay_adjustment"] = resolved_rows["macro_sensitivity"].map(
        lambda sensitivity: 0.15 * float(sensitivity) * macro_shock
    )
    resolved_rows["car_path_trade_date"] = resolved_rows.apply(
        lambda row: _extract_car_path_value(
            car_path_df,
            ticker=str(row["ticker"]),
            period_id=int(row["period_id"]),
            selected_date=selected_date,
            field="trade_date",
        ),
        axis=1,
    )
    resolved_rows["accum_car_to_date"] = resolved_rows.apply(
        lambda row: _extract_car_path_value(
            car_path_df,
            ticker=str(row["ticker"]),
            period_id=int(row["period_id"]),
            selected_date=selected_date,
            field="accum_car_to_date",
        ),
        axis=1,
    )
    resolved_rows["car_path_distress"] = resolved_rows.apply(
        lambda row: _extract_car_path_value(
            car_path_df,
            ticker=str(row["ticker"]),
            period_id=int(row["period_id"]),
            selected_date=selected_date,
            field="car_path_distress",
        ),
        axis=1,
    )
    resolved_rows["final_distress"] = resolved_rows.apply(
        lambda row: compute_final_distress_score(
            float(row["distress_score_mamdani"]),
            distress_sora,
            row["refi_risk"],
            row["car_path_distress"],
        ),
        axis=1,
    )
    resolved_rows["level"] = resolved_rows["final_distress"].map(score_to_level)
    ranking_view = resolved_rows.sort_values("final_distress", ascending=False).reset_index(drop=True)
    return ranking_view, macro_row, distress_sora


def build_macro_panel_context(macro_row: pd.Series, distress_sora: float) -> dict[str, Any]:
    return {
        "predicted_change": float(macro_row["y_pred"]),
        "predicted_level": float(macro_row["predicted_level"]),
        "fomc_decision_date": pd.Timestamp(macro_row["fomc_decision_date"]),
        "snapshot_ts": pd.Timestamp(macro_row["snapshot_ts"]),
        "distress_sora": float(distress_sora),
    }
