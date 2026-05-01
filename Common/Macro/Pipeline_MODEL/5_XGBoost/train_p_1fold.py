"""
train_p_1fold.py
================

Single-holdout Model P script.

Cloned conceptually from:
    train_rstar_xgboost_walkforward_optuna_deap_1fold.py

Purpose:
    Predict future SORA outcomes instead of S-REIT index outcomes.

Targets trained in one run:
    Option 1: future SORA level
        - sora_fwd_21d_level

    Option 2: future SORA change
        - sora_fwd_21d_change

    Option 3: absolute magnitude of future SORA change
        - sora_fwd_21d_abs_change

    You can run any subset. Defaults are set by RUN_TARGET_OPTION* below; override on the
    command line with --all-targets or --option1 / --no-option1, etc.

Feature design (analogous to train_rstar_xgboost_walkforward_optuna_deap_1fold.py):
    - Base: T-2 SORA / curve + Parquet (expected_bps, p_no_change, margin, days_to_fomc,
      missingness flags)
    - Spreads: Fed-implied bps vs local 90d SORA drift; level vs 3m curve (steepness)
    - Engineered SORA path from sora_level_realized: 21d/63d/5d/10d *level diffs*, rolling std of
      daily level changes, distance below 63d peak, distance from 63d MA, 21d momentum acceleration.
    - For option1_level only: drop sora_level_t2 and sora_curve_steepness to avoid near-tautology
      with the forward level target (see train_lp_1fold_issues.txt).
    - For option2_change: drop sora_level_t2 and sora_3m_t2 (high train/test regime shift vs weak
      partial signal for a change target; see train_lp_1fold_01_improve.txt).

Output location:
    Consolidated/IO/Model_Train/train_p_1fold/run_<n>/  (n = 0, 1, 2, ... new folder each run)

This script also writes:
    run_contents_summary.txt

so the contents are self-describing even if the folder name is generic.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _require_dependency(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except Exception as exc:
        raise ImportError(
            f"Missing required dependency '{module_name}'. Install hint: {install_hint}"
        ) from exc


xgboost = _require_dependency("xgboost", "pip install xgboost")
optuna = _require_dependency("optuna", "pip install optuna")
shap = _require_dependency("shap", "pip install shap")
deap = _require_dependency("deap", "pip install deap")

from xgboost import XGBRegressor
from deap import base, creator, tools


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
GAP_ROWS = 63
TRAIN_FRAC = 0.70
TEST_FRAC = 0.20

SCRIPT_DIR = Path(__file__).resolve().parent
CONSOLIDATED_ROOT = SCRIPT_DIR.parents[1]
IO_SRC_DIR = CONSOLIDATED_ROOT / "IO" / "SRC"
MODEL_DIR = IO_SRC_DIR / "MODEL"
MODEL_TRAIN_DIR = CONSOLIDATED_ROOT / "IO" / "Model_Train"
DATA_PATH = MODEL_DIR / "sora_joined_to_xgb.csv"
RUNS_ROOT = MODEL_TRAIN_DIR / Path(__file__).stem
RUNS_ROOT.mkdir(exist_ok=True)
_run_n = 0
while (RUNS_ROOT / f"run_{_run_n}").exists():
    _run_n += 1
OUT_DIR = RUNS_ROOT / f"run_{_run_n}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "snapshot_ts"
TRACE_COLS = ["fomc_decision_date"]

# SORA "price" series for path features (one row = one SGX business date), aligned to R* index logic.
SORA_PATH_LEVEL_COL = "sora_level_realized"

BASE_FEATURE_COLS = [
    "sora_level_t2",
    "sora_3m_t2",
    "sora_term_spread_t2",
    "expected_bps",
    "p_no_change",
    "margin_over_second",
    "days_to_next_fomc",
    "p_no_change_missing",
    "margin_over_second_missing",
]

SPREAD_FEATURE_COLS = [
    "expected_bps_minus_sora_90d",
    "sora_curve_steepness",
]

# Level deltas in *rate units* (e.g. percentage points), not pct_change (inappropriate for rates).
ENGINEERED_SORA_PATH_COLS = [
    "sora_lag_21d_diff",
    "sora_lag_63d_diff",
    "sora_lag_10d_diff",
    "sora_lag_5d_diff",
    "sora_realized_vol_21d",
    "sora_realized_vol_63d",
    "sora_below_63d_peak",
    "sora_dist_from_63d_ma",
    "sora_accel_21d",
]

FEATURE_COLS = BASE_FEATURE_COLS + SPREAD_FEATURE_COLS + ENGINEERED_SORA_PATH_COLS

# sora_fwd_21d_level aligns with realized level at t+21; these features are ~the same information.
OPTION1_LEVEL_EXCLUDED_FEATURES = ("sora_level_t2", "sora_curve_steepness")

# Change target: level at T-2 has weak partial corr but large train/test distribution shift.
OPTION2_CHANGE_EXCLUDED_FEATURES = ("sora_level_t2", "sora_3m_t2")


def feature_cols_for_target(target_key: str) -> List[str]:
    if target_key == "option1_level":
        return [c for c in FEATURE_COLS if c not in OPTION1_LEVEL_EXCLUDED_FEATURES]
    if target_key == "option2_change":
        return [c for c in FEATURE_COLS if c not in OPTION2_CHANGE_EXCLUDED_FEATURES]
    return list(FEATURE_COLS)

TARGET_SPECS = {
    "option1_level": {
        "target_col": "sora_fwd_21d_level",
        "description": "Future SORA level 21 SGX trading rows ahead",
    },
    "option2_change": {
        "target_col": "sora_fwd_21d_change",
        "description": "Future SORA change over 21 SGX trading rows",
    },
    "option3_abs_change": {
        "target_col": "sora_fwd_21d_abs_change",
        "description": "Absolute magnitude of future SORA change over 21 SGX trading rows",
    },
}

# Which of the three SORA targets to fit in a run (overridable via CLI: --all-targets, --option1, …).
RUN_TARGET_OPTION1_LEVEL = False
RUN_TARGET_OPTION2_CHANGE = True
RUN_TARGET_OPTION3_ABS_CHANGE = False

OPTUNA_N_TRIALS = 80
DEAP_GENERATIONS = 8
DEAP_POPULATION_SIZE = 20
# 9 discrete hyperparameter genes; ~1 gene mutated per individual on average (was 0.0015 → frozen).
DEAP_MUTATION_PROB = 1.0 / 9.0
DEAP_CROSSOVER_PROB = 0.6
DEAP_TOURNAMENT_SIZE = 3

SHAP_MAX_ROWS = 250
# Aligned with train_rstar ... 1fold.py: nudge optimizers away from constant predictions.
OBJECTIVE_PRED_STD_FLOOR = 0.005
OBJECTIVE_SIGN_BALANCE_FLOOR = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def directional_metrics_if_signed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    enabled: bool,
) -> Dict[str, float]:
    if not enabled:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
        }

    y_true_dir = (y_true > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true_dir, y_pred_dir)),
        "precision": float(precision_score(y_true_dir, y_pred_dir, zero_division=0)),
        "recall": float(recall_score(y_true_dir, y_pred_dir, zero_division=0)),
        "f1": float(f1_score(y_true_dir, y_pred_dir, zero_division=0)),
    }
    if len(np.unique(y_true_dir)) < 2:
        metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float(roc_auc_score(y_true_dir, y_pred))
    return metrics


def combined_metrics(y_true: np.ndarray, y_pred: np.ndarray, signed_target: bool) -> Dict[str, float]:
    out = {}
    out.update(regression_metrics(y_true, y_pred))
    out.update(directional_metrics_if_signed(y_true, y_pred, enabled=signed_target))
    return out


def save_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Model P: train one or more SORA targets (option 1/2/3) with Optuna+DEAP."
    )
    p.add_argument(
        "--all-targets",
        action="store_true",
        help="Run all three targets: level, signed change, and absolute change.",
    )
    p.add_argument(
        "--option1",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run option 1 (sora_fwd_21d_level). "
        f"Omit to use RUN_TARGET_OPTION1_LEVEL ({RUN_TARGET_OPTION1_LEVEL}) in the script.",
    )
    p.add_argument(
        "--option2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run option 2 (sora_fwd_21d_change). "
        f"Omit to use RUN_TARGET_OPTION2_CHANGE ({RUN_TARGET_OPTION2_CHANGE}).",
    )
    p.add_argument(
        "--option3",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run option 3 (sora_fwd_21d_abs_change). "
        f"Omit to use RUN_TARGET_OPTION3_ABS_CHANGE ({RUN_TARGET_OPTION3_ABS_CHANGE}).",
    )
    return p.parse_args()


def enabled_target_keys_from_args(args: argparse.Namespace) -> List[str]:
    if args.all_targets:
        return list(TARGET_SPECS.keys())
    o1 = args.option1 if args.option1 is not None else RUN_TARGET_OPTION1_LEVEL
    o2 = args.option2 if args.option2 is not None else RUN_TARGET_OPTION2_CHANGE
    o3 = args.option3 if args.option3 is not None else RUN_TARGET_OPTION3_ABS_CHANGE
    out: List[str] = []
    if o1:
        out.append("option1_level")
    if o2:
        out.append("option2_change")
    if o3:
        out.append("option3_abs_change")
    return out


def write_data_manifests(base_df: pd.DataFrame, run_target_keys: List[str]) -> None:
    """Match R* 1fold script: record feature groupings and data window for the run folder."""
    manifest = {
        "data_path": str(DATA_PATH),
        "rows_base": int(len(base_df)),
        "date_min": str(base_df[DATE_COL].min()),
        "date_max": str(base_df[DATE_COL].max()),
        "feature_cols": FEATURE_COLS,
        "base_feature_cols": BASE_FEATURE_COLS,
        "spread_feature_cols": SPREAD_FEATURE_COLS,
        "engineered_sora_path_cols": ENGINEERED_SORA_PATH_COLS,
        "sora_path_level_col": SORA_PATH_LEVEL_COL,
        "option1_level_excluded_features": list(OPTION1_LEVEL_EXCLUDED_FEATURES),
        "option2_change_excluded_features": list(OPTION2_CHANGE_EXCLUDED_FEATURES),
        "split_mode": "custom_1fold",
        "train_frac": TRAIN_FRAC,
        "test_frac": TEST_FRAC,
        "gap_rows": GAP_ROWS,
        "run_target_keys": list(run_target_keys),
        "run_flags_default": {
            "RUN_TARGET_OPTION1_LEVEL": RUN_TARGET_OPTION1_LEVEL,
            "RUN_TARGET_OPTION2_CHANGE": RUN_TARGET_OPTION2_CHANGE,
            "RUN_TARGET_OPTION3_ABS_CHANGE": RUN_TARGET_OPTION3_ABS_CHANGE,
        },
    }
    save_json(OUT_DIR / "data_manifest.json", manifest)
    save_json(
        OUT_DIR / "feature_manifest.json",
        {
            "features": FEATURE_COLS,
            "base_features": BASE_FEATURE_COLS,
            "spread_features": SPREAD_FEATURE_COLS,
            "engineered_sora_path": ENGINEERED_SORA_PATH_COLS,
            "option1_level_excluded_features": list(OPTION1_LEVEL_EXCLUDED_FEATURES),
            "option2_change_excluded_features": list(OPTION2_CHANGE_EXCLUDED_FEATURES),
            "trace_cols": TRACE_COLS,
            "all_target_keys": list(TARGET_SPECS.keys()),
            "run_target_keys": list(run_target_keys),
        },
    )


def get_custom_single_split(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    test_size = int(round(n * TEST_FRAC))
    train_end = int(round(n * TRAIN_FRAC))
    test_start = train_end + GAP_ROWS

    if test_start + test_size > n:
        test_size = n - test_start

    train_idx = np.arange(0, train_end)
    test_idx = np.arange(test_start, test_start + test_size)

    if len(train_idx) <= 0 or len(test_idx) <= 0:
        raise ValueError("Custom 1-fold split produced empty train or test window.")
    return train_idx, test_idx


def load_base_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL, "fomc_decision_date"])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Missingness indicators for sparse Parquet-derived fields.
    df["p_no_change_missing"] = df["p_no_change"].isna().astype(int)
    df["margin_over_second_missing"] = df["margin_over_second"].isna().astype(int)

    for col in (SORA_PATH_LEVEL_COL, "sora_90d_change_t2", "sora_level_t2", "sora_3m_t2"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- SORA path features: use level *differences* (rates are already in %), not pct_change. ---
    level = df[SORA_PATH_LEVEL_COL]
    df["sora_lag_21d_diff"] = level.diff(21)
    df["sora_lag_63d_diff"] = level.diff(63)
    df["sora_lag_10d_diff"] = level.diff(10)
    df["sora_lag_5d_diff"] = level.diff(5)
    sora_daily_dlevel = level.diff(1)
    df["sora_realized_vol_21d"] = sora_daily_dlevel.rolling(21).std()
    df["sora_realized_vol_63d"] = sora_daily_dlevel.rolling(63).std()
    # Distance below trailing 63d max level (<= 0); interpretable for a rate, vs price return drawdown.
    roll_max = level.rolling(63).max()
    df["sora_below_63d_peak"] = level - roll_max
    df["sora_dist_from_63d_ma"] = level - level.rolling(63).mean()
    df["sora_accel_21d"] = level.diff(21) - level.diff(21).shift(21)

    # Spreads: Fed meeting bps vs recent local SORA drift (both in bps in source table).
    if "sora_90d_change_t2" in df.columns and "expected_bps" in df.columns:
        df["expected_bps"] = pd.to_numeric(df["expected_bps"], errors="coerce")
        df["expected_bps_minus_sora_90d"] = df["expected_bps"] - df["sora_90d_change_t2"]
    else:
        df["expected_bps_minus_sora_90d"] = np.nan

    df["sora_curve_steepness"] = df["sora_level_t2"] - df["sora_3m_t2"]

    target_cols = [
        "sora_fwd_21d_level",
        "sora_fwd_21d_change",
        "sora_fwd_21d_abs_change",
    ]
    numeric_cols = list(
        dict.fromkeys(
            FEATURE_COLS + target_cols + [SORA_PATH_LEVEL_COL, "sora_90d_change_t2", "expected_bps"]
        )
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_target_dataset(df: pd.DataFrame, target_key: str, target_col: str) -> pd.DataFrame:
    fcols = feature_cols_for_target(target_key)
    needed_cols = [DATE_COL, target_col] + fcols + TRACE_COLS
    out = df[needed_cols].copy()
    out = out.dropna(subset=[target_col] + fcols).copy()
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    return out


def build_base_model(params: Dict) -> XGBRegressor:
    model_params = {
        "objective": "reg:squarederror",
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "missing": np.nan,
        **params,
    }
    return XGBRegressor(**model_params)


@dataclass
class HoldoutResult:
    holdout_metrics: pd.DataFrame
    oos_predictions: pd.DataFrame
    summary: Dict[str, float]


def evaluate_single_holdout(
    df: pd.DataFrame,
    params: Dict,
    label: str,
    target_col: str,
    signed_target: bool,
    feature_cols: List[str],
) -> HoldoutResult:
    train_idx, test_idx = get_custom_single_split(df)
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    model = build_base_model(params)
    model.fit(train_df[feature_cols], train_df[target_col])

    preds = model.predict(test_df[feature_cols])
    y_true = test_df[target_col].to_numpy()

    holdout_metric = {
        "fold": 1,
        "train_start": str(train_df[DATE_COL].min().date()),
        "train_end": str(train_df[DATE_COL].max().date()),
        "test_start": str(test_df[DATE_COL].min().date()),
        "test_end": str(test_df[DATE_COL].max().date()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    holdout_metric.update(combined_metrics(y_true, preds, signed_target=signed_target))
    holdout_metric["pred_std"] = float(np.std(preds))
    holdout_metric["pred_positive_rate"] = float(np.mean(preds > 0)) if signed_target else float("nan")

    pred_export = test_df[[DATE_COL] + TRACE_COLS].copy()
    pred_export["fold"] = 1
    pred_export["target_col"] = target_col
    pred_export["y_true"] = y_true
    pred_export["y_pred"] = preds
    if signed_target:
        pred_export["y_true_dir"] = (y_true > 0).astype(int)
        pred_export["y_pred_dir"] = (preds > 0).astype(int)

    mean_baseline_pred = np.full(len(test_df), train_df[target_col].mean())
    zero_baseline_pred = np.zeros(len(test_df))
    baseline_rows = [
        {
            "fold": 1,
            "baseline": "train_mean",
            **combined_metrics(y_true, mean_baseline_pred, signed_target=signed_target),
        },
        {
            "fold": 1,
            "baseline": "zero_baseline",
            **combined_metrics(y_true, zero_baseline_pred, signed_target=signed_target),
        },
    ]

    baseline_pred_rows = []
    for baseline_name, baseline_pred in (
        ("train_mean", mean_baseline_pred),
        ("zero_baseline", zero_baseline_pred),
    ):
        base_export = test_df[[DATE_COL] + TRACE_COLS].copy()
        base_export["fold"] = 1
        base_export["baseline"] = baseline_name
        base_export["target_col"] = target_col
        base_export["y_true"] = y_true
        base_export["y_pred"] = baseline_pred
        baseline_pred_rows.append(base_export)

    summary = {
        "label": label,
        "target_col": target_col,
        "r2": float(holdout_metric["r2"]),
        "mse": float(holdout_metric["mse"]),
        "rmse": float(holdout_metric["rmse"]),
        "mae": float(holdout_metric["mae"]),
        "accuracy": float(holdout_metric["accuracy"]) if signed_target else float("nan"),
        "precision": float(holdout_metric["precision"]) if signed_target else float("nan"),
        "recall": float(holdout_metric["recall"]) if signed_target else float("nan"),
        "f1": float(holdout_metric["f1"]) if signed_target else float("nan"),
        "auc": float(holdout_metric["auc"]) if signed_target else float("nan"),
        "pred_std": float(holdout_metric["pred_std"]),
        "pred_positive_rate": float(holdout_metric["pred_positive_rate"]) if signed_target else float("nan"),
        "gamma": float(params.get("gamma", np.nan)),
    }

    result = HoldoutResult(
        holdout_metrics=pd.DataFrame([holdout_metric]),
        oos_predictions=pred_export.reset_index(drop=True),
        summary=summary,
    )
    result.baseline_metrics = pd.DataFrame(baseline_rows)
    result.baseline_oos_predictions = pd.concat(baseline_pred_rows, ignore_index=True)
    return result


def objective_with_penalty(summary: Dict[str, float], signed_target: bool) -> float:
    score = summary["rmse"]

    pred_std = summary["pred_std"]
    if pred_std < OBJECTIVE_PRED_STD_FLOOR:
        score += (OBJECTIVE_PRED_STD_FLOOR - pred_std) * 10.0

    if summary["r2"] < 0:
        score += abs(summary["r2"]) * 0.01

    if signed_target:
        pos_rate = summary["pred_positive_rate"]
        if pos_rate < OBJECTIVE_SIGN_BALANCE_FLOOR:
            score += (OBJECTIVE_SIGN_BALANCE_FLOOR - pos_rate) * 2.0
        if pos_rate > 1.0 - OBJECTIVE_SIGN_BALANCE_FLOOR:
            score += (pos_rate - (1.0 - OBJECTIVE_SIGN_BALANCE_FLOOR)) * 2.0

    return float(score)


def optuna_objective(
    trial,
    df: pd.DataFrame,
    target_col: str,
    signed_target: bool,
    feature_cols: List[str],
) -> float:
    params = {
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    result = evaluate_single_holdout(
        df,
        params,
        label="optuna_trial",
        target_col=target_col,
        signed_target=signed_target,
        feature_cols=feature_cols,
    )
    penalized_score = objective_with_penalty(result.summary, signed_target=signed_target)
    trial.set_user_attr("r2", result.summary["r2"])
    trial.set_user_attr("rmse", result.summary["rmse"])
    trial.set_user_attr("mae", result.summary["mae"])
    trial.set_user_attr("f1", result.summary["f1"])
    trial.set_user_attr("penalized_score", penalized_score)
    return penalized_score


def run_optuna_search(
    df: pd.DataFrame,
    target_key: str,
    target_col: str,
    signed_target: bool,
    feature_cols: List[str],
) -> Dict:
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: optuna_objective(
            trial, df, target_col=target_col, signed_target=signed_target, feature_cols=feature_cols
        ),
        n_trials=OPTUNA_N_TRIALS,
        show_progress_bar=False,
    )
    payload = {
        "target_key": target_key,
        "target_col": target_col,
        "best_value_penalized_objective": float(study.best_value),
        "best_params": study.best_params,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }
    save_json(OUT_DIR / f"{target_key}_optuna_best_params.json", payload)
    return study.best_params


def run_deap_search(
    df: pd.DataFrame,
    target_key: str,
    target_col: str,
    signed_target: bool,
    feature_cols: List[str],
) -> Dict:
    search_space = {
        "gamma": [float(x) for x in np.linspace(0.0, 5.0, 11)],
        "n_estimators": [100, 150, 200, 300, 400, 500, 700, 900],
        "max_depth": [2, 3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
        "min_child_weight": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0],
        "reg_lambda": [1e-3, 1e-2, 0.1, 1.0, 3.0, 5.0, 10.0],
    }
    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]

    fitness_name = f"FitnessMin_{target_key}"
    individual_name = f"Individual_{target_key}"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0,))
    if not hasattr(creator, individual_name):
        creator.create(individual_name, list, fitness=getattr(creator, fitness_name))

    toolbox = base.Toolbox()

    def make_gene(i: int):
        return random.randrange(len(param_values[i]))

    toolbox.register(
        "individual",
        tools.initIterate,
        getattr(creator, individual_name),
        lambda: [make_gene(i) for i in range(len(param_names))]
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    cache: Dict[Tuple[int, ...], float] = {}

    def decode(individual) -> Dict:
        return {
            name: param_values[i][gene_idx]
            for i, (name, gene_idx) in enumerate(zip(param_names, individual))
        }

    def evaluate_individual(individual):
        key = tuple(individual)
        if key in cache:
            return (cache[key],)
        params = decode(individual)
        result = evaluate_single_holdout(
            df,
            params,
            label="deap_trial",
            target_col=target_col,
            signed_target=signed_target,
            feature_cols=feature_cols,
        )
        score = objective_with_penalty(result.summary, signed_target=signed_target)
        cache[key] = score
        return (score,)

    def mate_discrete(ind1, ind2):
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2

    def mutate_discrete(individual):
        for i in range(len(individual)):
            if random.random() < DEAP_MUTATION_PROB:
                individual[i] = make_gene(i)
        return (individual,)

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", mate_discrete)
    toolbox.register("mutate", mutate_discrete)
    toolbox.register("select", tools.selTournament, tournsize=DEAP_TOURNAMENT_SIZE)

    population = toolbox.population(n=DEAP_POPULATION_SIZE)
    hall_of_fame = tools.HallOfFame(1)

    invalid = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit

    hall_of_fame.update(population)

    for _generation in range(DEAP_GENERATIONS):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < DEAP_CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < DEAP_MUTATION_PROB:
                toolbox.mutate(mutant)
                try:
                    del mutant.fitness.values
                except AttributeError:
                    pass

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        hall_of_fame.update(population)

    best_individual = hall_of_fame[0]
    best_params = decode(best_individual)
    best_penalized_score = float(best_individual.fitness.values[0])
    payload = {
        "target_key": target_key,
        "target_col": target_col,
        "best_score_penalized_objective": best_penalized_score,
        "best_params": best_params,
    }
    save_json(OUT_DIR / f"{target_key}_deap_best_params.json", payload)
    return best_params


def export_holdout_results(result: HoldoutResult, target_key: str, optimizer_name: str) -> None:
    prefix = f"{target_key}_{optimizer_name}"
    result.holdout_metrics.to_csv(OUT_DIR / f"{prefix}_holdout_metrics.csv", index=False)
    result.oos_predictions.to_csv(OUT_DIR / f"{prefix}_holdout_oos_predictions.csv", index=False)
    result.baseline_metrics.to_csv(OUT_DIR / f"{prefix}_baseline_holdout_metrics.csv", index=False)
    result.baseline_oos_predictions.to_csv(OUT_DIR / f"{prefix}_baseline_holdout_oos_predictions.csv", index=False)
    save_json(OUT_DIR / f"{prefix}_holdout_summary.json", result.summary)


def choose_winner(optuna_result: HoldoutResult, deap_result: HoldoutResult, signed_target: bool) -> str:
    if signed_target:
        if optuna_result.summary["f1"] > deap_result.summary["f1"]:
            return "optuna"
        if deap_result.summary["f1"] > optuna_result.summary["f1"]:
            return "deap"
        return "optuna" if optuna_result.summary["auc"] >= deap_result.summary["auc"] else "deap"
    if optuna_result.summary["rmse"] < deap_result.summary["rmse"]:
        return "optuna"
    if deap_result.summary["rmse"] < optuna_result.summary["rmse"]:
        return "deap"
    return "optuna" if optuna_result.summary["r2"] >= deap_result.summary["r2"] else "deap"


def fit_final_model(df: pd.DataFrame, params: Dict, target_col: str, feature_cols: List[str]) -> XGBRegressor:
    model = build_base_model(params)
    return model.fit(df[feature_cols], df[target_col])


def run_shap(
    final_model: XGBRegressor,
    df: pd.DataFrame,
    target_col: str,
    target_key: str,
    feature_cols: List[str],
) -> None:
    shap_df = df[[DATE_COL] + feature_cols + [target_col]].copy()
    shap_df = shap_df.tail(min(SHAP_MAX_ROWS, len(shap_df))).reset_index(drop=True)
    X_shap = shap_df[feature_cols]

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_shap)

    shap_values_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_values_df.insert(0, DATE_COL, shap_df[DATE_COL].astype(str))
    shap_values_df.to_csv(OUT_DIR / f"{target_key}_shap_summary_values.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{target_key}_shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{target_key}_shap_summary_bar.png", dpi=160, bbox_inches="tight")
    plt.close()


def write_run_contents_summary(run_target_keys: List[str]) -> None:
    lines = [
        "train_p_1fold.py output summary",
        "",
        f"Dataset: {DATA_PATH}",
        f"Output directory: {OUT_DIR}",
        "",
        "Default run flags in script: "
        f"opt1={RUN_TARGET_OPTION1_LEVEL} opt2={RUN_TARGET_OPTION2_CHANGE} opt3={RUN_TARGET_OPTION3_ABS_CHANGE}",
        "",
        "Targets in this run:",
    ]
    for key in run_target_keys:
        spec = TARGET_SPECS[key]
        lines.append(f"  - {key}: {spec['target_col']} :: {spec['description']}")
    lines += [
        "",
        "Base feature cols:",
    ] + [f"  - {col}" for col in BASE_FEATURE_COLS]
    lines += [
        "",
        "Spread feature cols:",
    ] + [f"  - {col}" for col in SPREAD_FEATURE_COLS]
    lines += [
        "",
        "Engineered SORA path cols (from sora_level_realized, diffs not pct_change):",
    ] + [f"  - {col}" for col in ENGINEERED_SORA_PATH_COLS]
    lines += [
        "",
        "option1_level: excludes (avoid tautology with forward level target):",
    ] + [f"  - {c}" for c in OPTION1_LEVEL_EXCLUDED_FEATURES]
    lines += [
        "",
        "option2_change: excludes (regime-shift-heavy T-2 levels vs change target):",
    ] + [f"  - {c}" for c in OPTION2_CHANGE_EXCLUDED_FEATURES]
    lines += [
        "",
        "Output naming pattern:",
        "  <target_key>_optuna_best_params.json",
        "  <target_key>_deap_best_params.json",
        "  <target_key>_optuna_holdout_metrics.csv",
        "  <target_key>_deap_holdout_metrics.csv",
        "  <target_key>_optuna_holdout_oos_predictions.csv",
        "  <target_key>_deap_holdout_oos_predictions.csv",
        "  <target_key>_optuna_baseline_holdout_metrics.csv",
        "  <target_key>_deap_baseline_holdout_metrics.csv",
        "  <target_key>_optuna_baseline_holdout_oos_predictions.csv",
        "  <target_key>_deap_baseline_holdout_oos_predictions.csv",
        "  <target_key>_optuna_holdout_summary.json",
        "  <target_key>_deap_holdout_summary.json",
        "  <target_key>_shap_summary_values.csv",
        "  <target_key>_shap_summary_beeswarm.png",
        "  <target_key>_shap_summary_bar.png",
        "",
        "Run-level manifest files:",
        "  data_manifest.json",
        "  feature_manifest.json",
        "Cross-target consolidated files:",
        "  all_targets_optimizer_comparison.json",
        "  all_targets_selected_params.json",
        "  run_contents_summary.txt",
    ]
    (OUT_DIR / "run_contents_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_global_seed(SEED)
    args = parse_args()
    run_target_keys = enabled_target_keys_from_args(args)
    if not run_target_keys:
        raise SystemExit(
            "No targets selected. Set RUN_TARGET_OPTION1_LEVEL / _OPTION2 / _OPTION3 in the script, "
            "or pass at least one of --all-targets, --option1, --option2, --option3 (use --no-option2 to turn off 2). "
            "If every flag is off, nothing runs."
        )

    print("Loading base dataset ...")
    base_df = load_base_dataset()
    write_data_manifests(base_df, run_target_keys=run_target_keys)
    print(f"Targets this run: {run_target_keys}")

    all_comparisons = {}
    all_selected_params = {}

    for target_key in run_target_keys:
        spec = TARGET_SPECS[target_key]
        target_col = spec["target_col"]
        signed_target = target_key == "option2_change"
        fcols = feature_cols_for_target(target_key)

        print(f"\n=== Running target: {target_key} ({target_col}) ===")
        print(f"Features (n={len(fcols)}): {fcols}")
        df = build_target_dataset(base_df, target_key=target_key, target_col=target_col)
        print(f"Usable rows: {len(df):,}")
        print(f"Window: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")

        train_idx, test_idx = get_custom_single_split(df)
        print(f"Custom split sizes: train={len(train_idx)} gap={GAP_ROWS} test={len(test_idx)}")

        print("Running Optuna search ...")
        optuna_params = run_optuna_search(
            df, target_key=target_key, target_col=target_col, signed_target=signed_target, feature_cols=fcols
        )
        optuna_eval = evaluate_single_holdout(
            df, optuna_params, label="optuna", target_col=target_col, signed_target=signed_target, feature_cols=fcols
        )
        export_holdout_results(optuna_eval, target_key=target_key, optimizer_name="optuna")

        print("Running DEAP search ...")
        deap_params = run_deap_search(
            df, target_key=target_key, target_col=target_col, signed_target=signed_target, feature_cols=fcols
        )
        deap_eval = evaluate_single_holdout(
            df, deap_params, label="deap", target_col=target_col, signed_target=signed_target, feature_cols=fcols
        )
        export_holdout_results(deap_eval, target_key=target_key, optimizer_name="deap")

        winner = choose_winner(optuna_eval, deap_eval, signed_target=signed_target)
        final_params = optuna_params if winner == "optuna" else deap_params

        all_comparisons[target_key] = {
            "target_col": target_col,
            "description": spec["description"],
            "signed_target": signed_target,
            "winner": winner,
            "optuna_summary": optuna_eval.summary,
            "deap_summary": deap_eval.summary,
            "winner_params": final_params,
        }
        all_selected_params[target_key] = final_params

        final_model = build_base_model(final_params)
        final_model.fit(df[fcols], df[target_col])
        final_model.save_model(str(OUT_DIR / f"{target_key}_final_model_xgb.json"))
        run_shap(final_model, df, target_col=target_col, target_key=target_key, feature_cols=fcols)

    save_json(OUT_DIR / "all_targets_optimizer_comparison.json", all_comparisons)
    save_json(OUT_DIR / "all_targets_selected_params.json", all_selected_params)
    write_run_contents_summary(run_target_keys=run_target_keys)
    print("\nDone. Outputs written to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
