from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "Common" / "Frontend"
KG_DIR = ROOT_DIR / "Common" / "Micro" / "5_Model_KG"
for path in (FRONTEND_DIR, KG_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reitteratsel_core import DUCKDB_PATH, car_to_distress_score, load_macro_prediction_frame, score_to_level
from reitteratsel_view_logic import build_ranking_view


OUTPUT_DIR = ROOT_DIR / "Common" / "Eval"
DETAIL_PATH = OUTPUT_DIR / "reitteratsel_eval_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "reitteratsel_eval_summary.csv"
DISAGREEMENT_PATH = OUTPUT_DIR / "reitteratsel_eval_disagreements.csv"


def compute_baseline_level(icr_value: float | None) -> str:
    if icr_value is None or pd.isna(icr_value):
        return "WATCH"
    if float(icr_value) < 1.5:
        return "DISTRESSED"
    if float(icr_value) > 3.0:
        return "HEALTHY"
    return "WATCH"


def build_eval_detail() -> pd.DataFrame:
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
        car_path_df = con.execute(
            """
            SELECT
                ticker,
                period_id,
                trade_date,
                accum_car_to_date,
                car_path_distress
            FROM reit_labels.fact_car_path_daily
            ORDER BY ticker, period_id, trade_date
            """
        ).fetchdf()
    finally:
        con.close()

    macro_df = load_macro_prediction_frame()
    rows: list[pd.DataFrame] = []
    for selected_date in macro_df["snapshot_ts"].dt.normalize().drop_duplicates().tolist():
        ranking_view, macro_row, distress_sora = build_ranking_view(
            fuzzy_df,
            metric_df,
            macro_df,
            car_path_df,
            selected_date,
        )
        icr_rows = metric_df.loc[metric_df["metric_code"] == "ICR", ["ticker", "period_id", "metric_value"]].rename(
            columns={"metric_value": "icr_value"}
        )
        frame = ranking_view.merge(icr_rows, on=["ticker", "period_id"], how="left", validate="one_to_one")
        frame["selected_date"] = pd.Timestamp(selected_date)
        frame["macro_snapshot_ts"] = pd.Timestamp(macro_row["snapshot_ts"])
        frame["distress_sora"] = float(distress_sora)
        frame["distress_baseline"] = frame["icr_value"].map(compute_baseline_level)
        frame["distress_mamdani_level"] = frame["distress_score_mamdani"].map(score_to_level)
        frame["car_target_normalized"] = frame["car_126wd"].map(car_to_distress_score)
        frame["baseline_correct"] = frame["distress_baseline"] == frame["label_126wd"]
        frame["mamdani_correct"] = frame["distress_mamdani_level"] == frame["label_126wd"]
        frame["final_correct"] = frame["level"] == frame["label_126wd"]
        frame["mamdani_gap_abs"] = (frame["distress_score_mamdani"] - frame["car_target_normalized"]).abs()
        frame["final_gap_abs"] = (frame["final_distress"] - frame["car_target_normalized"]).abs()
        rows.append(frame)

    if not rows:
        raise ValueError("No evaluation rows were generated from macro snapshots.")
    return pd.concat(rows, ignore_index=True)


def build_eval_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    records = [
        {
            "model_name": "distress_baseline",
            "label_accuracy": float(detail_df["baseline_correct"].mean()),
            "continuous_gap_abs_mean": pd.NA,
        },
        {
            "model_name": "distress_score_mamdani",
            "label_accuracy": float(detail_df["mamdani_correct"].mean()),
            "continuous_gap_abs_mean": float(detail_df["mamdani_gap_abs"].dropna().mean()),
        },
        {
            "model_name": "final_distress",
            "label_accuracy": float(detail_df["final_correct"].mean()),
            "continuous_gap_abs_mean": float(detail_df["final_gap_abs"].dropna().mean()),
        },
    ]
    return pd.DataFrame(records)


def build_disagreement_export(detail_df: pd.DataFrame) -> pd.DataFrame:
    return detail_df.loc[
        detail_df["final_correct"] == False,  # noqa: E712
        [
            "selected_date",
            "ticker",
            "reit_name",
            "sector",
            "period_id",
            "fiscal_year_end_date",
            "macro_snapshot_ts",
            "distress_baseline",
            "distress_score_mamdani",
            "distress_mamdani_level",
            "distress_sora",
            "refi_risk",
            "accum_car_to_date",
            "car_path_distress",
            "final_distress",
            "level",
            "car_63wd",
            "car_126wd",
            "label_126wd",
            "top_rule_ids",
            "rule_trace_text",
            "null_count",
            "non_ok_count",
            "mamdani_gap_abs",
            "final_gap_abs",
        ],
    ].sort_values(["selected_date", "final_gap_abs"], ascending=[True, False])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df = build_eval_detail()
    summary_df = build_eval_summary(detail_df)
    disagreement_df = build_disagreement_export(detail_df)
    detail_df.to_csv(DETAIL_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    disagreement_df.to_csv(DISAGREEMENT_PATH, index=False)
    print(f"Wrote detail evaluation: {DETAIL_PATH}")
    print(f"Wrote summary evaluation: {SUMMARY_PATH}")
    print(f"Wrote disagreement export: {DISAGREEMENT_PATH}")


if __name__ == "__main__":
    main()
