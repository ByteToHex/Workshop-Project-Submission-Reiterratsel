from __future__ import annotations

import asyncio
import math
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

from reitteratsel_core import (  # noqa: E402
    DUCKDB_PATH,
    car_to_distress_score,
    compute_refi_distress_score,
    load_macro_prediction_frame,
    score_to_level,
)
from reitteratsel_view_logic import build_ranking_view  # noqa: E402


OUTPUT_DIR = ROOT_DIR / "Common" / "Eval" / "IO"

CLASS_ORDER = ["DISTRESSED", "WATCH", "HEALTHY"]
MODEL_SCORE_COLS = {
    "distress_baseline": "distress_score_baseline",
    "distress_score_mamdani": "distress_score_mamdani",
    "distress_score_refi": "distress_score_refi",
    "final_distress": "final_distress",
}
MODEL_LEVEL_COLS = {
    "distress_baseline": "distress_baseline",
    "distress_score_mamdani": "distress_mamdani_level",
    "distress_score_refi": "distress_refi_level",
    "final_distress": "level",
}
TOP_K_VALUES = [1, 3, 5]
HEARTBEAT_SECONDS = 15.0
MAX_CONCURRENT_DATES = 8


def allocate_run_dir(output_root: Path = OUTPUT_DIR) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing_numbers: list[int] = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("run_"):
            continue
        suffix = name[4:]
        if suffix.isdigit():
            existing_numbers.append(int(suffix))
    next_number = max(existing_numbers, default=0) + 1
    run_dir = output_root / f"run_{next_number}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class EvalProgress:
    def __init__(self, total_dates: int) -> None:
        self.total_dates = total_dates
        self.completed_dates = 0


def build_eval_frame_for_date(
    selected_date: object,
    fuzzy_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    car_path_df: pd.DataFrame,
    icr_rows: pd.DataFrame,
) -> pd.DataFrame:
    ranking_view, macro_row, distress_sora = build_ranking_view(
        fuzzy_df,
        metric_df,
        macro_df,
        car_path_df,
        selected_date,
    )
    frame = ranking_view.merge(icr_rows, on=["ticker", "period_id"], how="left", validate="one_to_one")
    frame["selected_date"] = pd.Timestamp(selected_date)
    frame["macro_snapshot_ts"] = pd.Timestamp(macro_row["snapshot_ts"])
    frame["distress_sora"] = float(distress_sora)
    frame["distress_baseline"] = frame["icr_value"].map(compute_baseline_level)
    frame["distress_score_baseline"] = frame["distress_baseline"].map(label_to_score)
    frame["distress_mamdani_level"] = frame["distress_score_mamdani"].map(score_to_level)
    frame["distress_score_refi"] = frame["refi_risk"].map(compute_refi_distress_score)
    frame["distress_refi_level"] = frame["distress_score_refi"].map(
        lambda value: score_to_level(value) if value is not None and not pd.isna(value) else None
    )
    frame["car_target_normalized"] = frame["car_126wd"].map(car_to_distress_score)
    frame["is_distressed_truth"] = frame["label_126wd"] == "DISTRESSED"
    for model_name, level_col in MODEL_LEVEL_COLS.items():
        frame[f"{model_name}_correct"] = frame[level_col] == frame["label_126wd"]
    for model_name, score_col in MODEL_SCORE_COLS.items():
        frame[f"{model_name}_gap_abs"] = (frame[score_col] - frame["car_target_normalized"]).abs()
    return frame


async def heartbeat(progress: EvalProgress) -> None:
    while progress.completed_dates < progress.total_dates:
        print(
            f"[heartbeat] completed {progress.completed_dates}/{progress.total_dates} "
            f"simulation dates"
        )
        await asyncio.sleep(HEARTBEAT_SECONDS)


async def build_eval_detail_async(
    *,
    fuzzy_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    car_path_df: pd.DataFrame,
) -> pd.DataFrame:
    selected_dates = macro_df["snapshot_ts"].dt.normalize().drop_duplicates().tolist()
    if not selected_dates:
        raise ValueError("No simulation dates were found in the macro frame.")
    progress = EvalProgress(total_dates=len(selected_dates))
    heartbeat_task = asyncio.create_task(heartbeat(progress))
    icr_rows = metric_df.loc[metric_df["metric_code"] == "ICR", ["ticker", "period_id", "metric_value"]].rename(
        columns={"metric_value": "icr_value"}
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DATES)

    async def worker(selected_date: object) -> pd.DataFrame:
        async with semaphore:
            frame = await asyncio.to_thread(
                build_eval_frame_for_date,
                selected_date,
                fuzzy_df,
                metric_df,
                macro_df,
                car_path_df,
                icr_rows,
            )
            progress.completed_dates += 1
            if progress.completed_dates % 25 == 0 or progress.completed_dates == progress.total_dates:
                print(
                    f"[progress] finished {progress.completed_dates}/{progress.total_dates} "
                    f"dates; latest={pd.Timestamp(selected_date).date()}"
                )
            return frame

    try:
        tasks = [asyncio.create_task(worker(selected_date)) for selected_date in selected_dates]
        frames = await asyncio.gather(*tasks)
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    return pd.concat(frames, ignore_index=True)


def compute_baseline_level(icr_value: float | None) -> str:
    if icr_value is None or pd.isna(icr_value):
        return "WATCH"
    if float(icr_value) < 1.5:
        return "DISTRESSED"
    if float(icr_value) > 3.0:
        return "HEALTHY"
    return "WATCH"


def label_to_score(level: str | None) -> float | None:
    mapping = {
        "DISTRESSED": 1.0,
        "WATCH": 0.5,
        "HEALTHY": 0.0,
    }
    if level is None or pd.isna(level):
        return None
    return mapping.get(str(level).upper())


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
    return asyncio.run(
        build_eval_detail_async(
            fuzzy_df=fuzzy_df,
            metric_df=metric_df,
            macro_df=macro_df,
            car_path_df=car_path_df,
        )
    )


def build_confusion_frame(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, level_col in MODEL_LEVEL_COLS.items():
        confusion = pd.crosstab(
            detail_df["label_126wd"],
            detail_df[level_col],
            dropna=False,
        ).reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
        for true_label in CLASS_ORDER:
            for predicted_label in CLASS_ORDER:
                rows.append(
                    {
                        "model_name": model_name,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(confusion.loc[true_label, predicted_label]),
                    }
                )
    return pd.DataFrame(rows)


def _precision_recall_f1(confusion: pd.DataFrame, class_name: str) -> tuple[float, float, float, int]:
    tp = float(confusion.loc[class_name, class_name])
    fp = float(confusion[class_name].sum() - tp)
    fn = float(confusion.loc[class_name].sum() - tp)
    support = int(confusion.loc[class_name].sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, support


def multiclass_mcc(confusion: pd.DataFrame) -> float:
    c = float(sum(confusion.loc[label, label] for label in CLASS_ORDER))
    s = float(confusion.to_numpy().sum())
    if s == 0:
        return 0.0
    true_totals = confusion.sum(axis=1).astype(float)
    pred_totals = confusion.sum(axis=0).astype(float)
    numerator = c * s - float((true_totals * pred_totals).sum())
    denominator_left = s * s - float((pred_totals * pred_totals).sum())
    denominator_right = s * s - float((true_totals * true_totals).sum())
    denominator = math.sqrt(max(denominator_left, 0.0) * max(denominator_right, 0.0))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_per_class_and_summary(detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_class_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for model_name, level_col in MODEL_LEVEL_COLS.items():
        confusion = pd.crosstab(
            detail_df["label_126wd"],
            detail_df[level_col],
            dropna=False,
        ).reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
        class_f1_values: list[float] = []
        for class_name in CLASS_ORDER:
            precision, recall, f1, support = _precision_recall_f1(confusion, class_name)
            class_f1_values.append(f1)
            per_class_rows.append(
                {
                    "model_name": model_name,
                    "class_name": class_name,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                }
            )
        summary_rows.append(
            {
                "model_name": model_name,
                "label_accuracy": float((detail_df[level_col] == detail_df["label_126wd"]).mean()),
                "macro_f1": float(sum(class_f1_values) / len(class_f1_values)),
                "mcc": float(multiclass_mcc(confusion)),
                "continuous_mae": float(detail_df[f"{model_name}_gap_abs"].dropna().mean()),
                "continuous_rmse": float((detail_df[f"{model_name}_gap_abs"].dropna().pow(2).mean()) ** 0.5),
            }
        )
    return pd.DataFrame(per_class_rows), pd.DataFrame(summary_rows)


def precision_at_k(sorted_truth: list[bool], k: int) -> float:
    top_k = sorted_truth[:k]
    if not top_k:
        return 0.0
    return sum(1 for item in top_k if item) / len(top_k)


def average_precision_at_k(sorted_truth: list[bool], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    limit = min(k, len(sorted_truth))
    for idx in range(limit):
        if sorted_truth[idx]:
            hits += 1
            precision_sum += hits / float(idx + 1)
    relevant_in_top_k = sum(1 for item in sorted_truth[:limit] if item)
    if relevant_in_top_k == 0:
        return 0.0
    return precision_sum / float(relevant_in_top_k)


def build_ranking_metrics(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = detail_df.groupby("selected_date", sort=True)
    for model_name, score_col in MODEL_SCORE_COLS.items():
        for k in TOP_K_VALUES:
            per_date_p: list[float] = []
            per_date_ap: list[float] = []
            for _, frame in grouped:
                ranked = frame.sort_values(score_col, ascending=False)
                truth = ranked["is_distressed_truth"].astype(bool).tolist()
                per_date_p.append(precision_at_k(truth, k))
                per_date_ap.append(average_precision_at_k(truth, k))
            rows.append(
                {
                    "model_name": model_name,
                    "k": k,
                    "precision_at_k": float(sum(per_date_p) / len(per_date_p)),
                    "map_at_k": float(sum(per_date_ap) / len(per_date_ap)),
                }
            )
    return pd.DataFrame(rows)


def build_disagreement_export(detail_df: pd.DataFrame) -> pd.DataFrame:
    return detail_df.loc[
        detail_df["final_distress_correct"] == False,  # noqa: E712
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
            "distress_score_refi",
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
            "distress_baseline_gap_abs",
            "distress_score_mamdani_gap_abs",
            "distress_score_refi_gap_abs",
            "final_distress_gap_abs",
        ],
    ].sort_values(["selected_date", "final_distress_gap_abs"], ascending=[True, False])


def main() -> None:
    run_dir = allocate_run_dir()
    detail_path = run_dir / "reitteratsel_eval_detail.csv"
    summary_path = run_dir / "reitteratsel_eval_summary.csv"
    disagreement_path = run_dir / "reitteratsel_eval_disagreements.csv"
    confusion_path = run_dir / "reitteratsel_eval_confusion_matrices.csv"
    per_class_path = run_dir / "reitteratsel_eval_per_class_metrics.csv"
    ranking_path = run_dir / "reitteratsel_eval_ranking_metrics.csv"

    detail_df = build_eval_detail()
    confusion_df = build_confusion_frame(detail_df)
    per_class_df, summary_df = build_per_class_and_summary(detail_df)
    ranking_df = build_ranking_metrics(detail_df)
    disagreement_df = build_disagreement_export(detail_df)

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    disagreement_df.to_csv(disagreement_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    print(f"Allocated evaluation run directory: {run_dir}")
    print(f"Wrote detail evaluation: {detail_path}")
    print(f"Wrote summary evaluation: {summary_path}")
    print(f"Wrote disagreement export: {disagreement_path}")
    print(f"Wrote confusion matrices: {confusion_path}")
    print(f"Wrote per-class metrics: {per_class_path}")
    print(f"Wrote ranking metrics: {ranking_path}")


if __name__ == "__main__":
    main()
