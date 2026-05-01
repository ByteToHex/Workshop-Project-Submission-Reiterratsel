"""
train_rstar_directional_1fold.py
================================

Directional-label variant of the single-holdout Model R* pipeline.

This file is cloned from:
    train_rstar_xgboost_walkforward_optuna_deap_1fold.py

Key change:
    The target is binary direction of the existing 21-day forward REIT index
    return, derived from the current dataset instead of requiring new data.

Directional target definition:
    Option A (default threshold = 0.0):
        1 if reit_index_fwd_21d_return > 0
        0 otherwise

    Option B (dead-zone threshold > 0):
        1 if reit_index_fwd_21d_return > +threshold
        0 if reit_index_fwd_21d_return < -threshold
        rows in [-threshold, +threshold] are dropped

This lets you test a cleaner classification target without changing the
underlying dataset.

Outputs:
    Consolidated/IO/Model_Train/train_rstar_directional_1fold/run_<n>/  (n = 0, 1, 2, ... new folder each run)
"""

from __future__ import annotations

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
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
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

from xgboost import XGBClassifier
from deap import base, creator, tools


SEED = 42
GAP_ROWS = 63
TRAIN_FRAC = 0.70
TEST_FRAC = 0.20

# Set to 0.0 for plain sign label.
# Set to e.g. 0.005 for a +/-0.5% dead zone.
DIRECTION_THRESHOLD = 0.0

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
OUT_DIR.mkdir(exist_ok=True)

RAW_RETURN_COL = "reit_index_fwd_21d_return"
TARGET_COL = "reit_index_fwd_21d_direction"
DATE_COL = "snapshot_ts"
SORA_PATH_LEVEL_COL = "sora_level_realized"

# Raw SORA levels have strong train/test distribution shift; exclude from directional model (cf. option2 in train_p).
DIRECTIONAL_EXCLUDED_FEATURES = ("sora_level_t2", "sora_3m_t2")

BASE_FEATURE_COLS = [
    "sora_level_t2",
    "expected_bps",
    "days_to_next_fomc",
    "sora_3m_t2",
]

ENGINEERED_REIT_COLS = [
    "reit_index_lag_21d_return",
    "reit_index_lag_63d_return",
    "reit_index_vol_21d",
    "reit_index_vol_63d",
    "reit_index_drawdown_63d",
]

ENGINEERED_SORA_PATH_COLS = [
    "sora_lag_21d_diff",
    "sora_lag_63d_diff",
    "sora_realized_vol_21d",
    "sora_realized_vol_63d",
    "sora_below_63d_peak",
    "sora_dist_from_63d_ma",
    "sora_lag_10d_diff",
    "sora_lag_5d_diff",
    "sora_accel_21d",
]

FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_REIT_COLS + ENGINEERED_SORA_PATH_COLS


def get_feature_cols() -> List[str]:
    return [c for c in FEATURE_COLS if c not in DIRECTIONAL_EXCLUDED_FEATURES]

TRACE_COLS = [
    "reit_index_close",
    "fomc_decision_date",
    RAW_RETURN_COL,
]

OPTUNA_N_TRIALS = 80
DEAP_GENERATIONS = 8
DEAP_POPULATION_SIZE = 20
# Per-gene: ~1 locus updated per 9-gene individual in expectation (9 hyperparameters).
DEAP_MUTATION_PROB = 1.0 / 9.0
DEAP_CROSSOVER_PROB = 0.6
DEAP_TOURNAMENT_SIZE = 3

SHAP_MAX_ROWS = 250
OBJECTIVE_POS_RATE_FLOOR = 0.05


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) < 2:
        metrics["auc"] = float("nan")
        metrics["logloss"] = float("nan")
    else:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["logloss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    return metrics


def save_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def derive_direction_label(ret: pd.Series, threshold: float) -> pd.Series:
    if threshold <= 0:
        return (ret > 0).astype(float)

    label = pd.Series(np.nan, index=ret.index, dtype="float64")
    label[ret > threshold] = 1.0
    label[ret < -threshold] = 0.0
    return label


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL, "fomc_decision_date"])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    close = pd.to_numeric(df["reit_index_close"], errors="coerce")
    df["reit_index_lag_21d_return"] = close.pct_change(21)
    df["reit_index_lag_63d_return"] = close.pct_change(63)
    daily_ret = close.pct_change(1)
    df["reit_index_vol_21d"] = daily_ret.rolling(21).std()
    df["reit_index_vol_63d"] = daily_ret.rolling(63).std()
    df["reit_index_drawdown_63d"] = close / close.rolling(63).max() - 1.0

    df[RAW_RETURN_COL] = pd.to_numeric(df[RAW_RETURN_COL], errors="coerce")
    df[TARGET_COL] = derive_direction_label(df[RAW_RETURN_COL], DIRECTION_THRESHOLD)

    if SORA_PATH_LEVEL_COL not in df.columns:
        raise KeyError(
            f"Dataset is missing {SORA_PATH_LEVEL_COL!r} (required for SORA path features). "
            "Regenerate the Step4 XGB input CSV or point DATA_PATH to a file that has it."
        )
    for col in (SORA_PATH_LEVEL_COL, "sora_level_t2", "sora_3m_t2", "expected_bps"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # SORA path (rate *levels*; use diffs / vol in level units, not pct_change).
    level = df[SORA_PATH_LEVEL_COL]
    df["sora_lag_21d_diff"] = level.diff(21)
    df["sora_lag_63d_diff"] = level.diff(63)
    sora_daily_dlevel = level.diff(1)
    df["sora_realized_vol_21d"] = sora_daily_dlevel.rolling(21).std()
    df["sora_realized_vol_63d"] = sora_daily_dlevel.rolling(63).std()
    roll_max = level.rolling(63).max()
    df["sora_below_63d_peak"] = level - roll_max
    df["sora_dist_from_63d_ma"] = level - level.rolling(63).mean()
    df["sora_lag_10d_diff"] = level.diff(10)
    df["sora_lag_5d_diff"] = level.diff(5)
    df["sora_accel_21d"] = level.diff(21) - level.diff(21).shift(21)

    numeric_cols = list(
        dict.fromkeys(
            FEATURE_COLS + [RAW_RETURN_COL, SORA_PATH_LEVEL_COL] + TRACE_COLS[:1]
        )
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_cols = get_feature_cols()
    before = len(df)
    df = df.dropna(subset=[TARGET_COL, *feature_cols]).copy()
    after = len(df)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    manifest = {
        "data_path": str(DATA_PATH),
        "rows_before_target_drop": before,
        "rows_after_target_drop": after,
        "date_min": str(df[DATE_COL].min()),
        "date_max": str(df[DATE_COL].max()),
        "feature_cols": feature_cols,
        "all_feature_columns": FEATURE_COLS,
        "excluded_for_directional": list(DIRECTIONAL_EXCLUDED_FEATURES),
        "sora_path_level_col": SORA_PATH_LEVEL_COL,
        "base_feature_cols": BASE_FEATURE_COLS,
        "engineered_reit_cols": ENGINEERED_REIT_COLS,
        "engineered_sora_path_cols": ENGINEERED_SORA_PATH_COLS,
        "raw_return_col": RAW_RETURN_COL,
        "target_col": TARGET_COL,
        "target_type": "binary_direction",
        "direction_threshold": DIRECTION_THRESHOLD,
        "split_mode": "custom_1fold",
        "train_frac": TRAIN_FRAC,
        "test_frac": TEST_FRAC,
        "gap_rows": GAP_ROWS,
        "category_1_choice": "gamma",
        "category_2_metrics": [],
        "category_3_metrics": ["accuracy", "precision", "recall", "f1", "auc", "logloss"],
        "category_4_xai": "shap",
    }
    save_json(OUT_DIR / "data_manifest.json", manifest)
    save_json(OUT_DIR / "feature_manifest.json", {
        "features": feature_cols,
        "all_feature_columns": FEATURE_COLS,
        "excluded_for_directional": list(DIRECTIONAL_EXCLUDED_FEATURES),
        "base_features": BASE_FEATURE_COLS,
        "engineered_reit": ENGINEERED_REIT_COLS,
        "engineered_sora_path": ENGINEERED_SORA_PATH_COLS,
        "trace_cols": TRACE_COLS,
        "target": TARGET_COL,
        "direction_threshold": DIRECTION_THRESHOLD,
    })
    return df


def build_base_model(params: Dict) -> XGBClassifier:
    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "missing": np.nan,
        **params,
    }
    return XGBClassifier(**model_params)


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


@dataclass
class HoldoutResult:
    holdout_metrics: pd.DataFrame
    oos_predictions: pd.DataFrame
    summary: Dict[str, float]


def evaluate_single_holdout(df: pd.DataFrame, params: Dict, label: str) -> HoldoutResult:
    train_idx, test_idx = get_custom_single_split(df)
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    fcols = get_feature_cols()

    model = build_base_model(params)
    model.fit(train_df[fcols], train_df[TARGET_COL])

    prob_1 = model.predict_proba(test_df[fcols])[:, 1]
    pred = (prob_1 >= 0.5).astype(int)
    y_true = test_df[TARGET_COL].to_numpy()

    holdout_metric = {
        "fold": 1,
        "train_start": str(train_df[DATE_COL].min().date()),
        "train_end": str(train_df[DATE_COL].max().date()),
        "test_start": str(test_df[DATE_COL].min().date()),
        "test_end": str(test_df[DATE_COL].max().date()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    holdout_metric.update(classification_metrics(y_true, pred, prob_1))
    holdout_metric["pred_std"] = float(np.std(prob_1))
    holdout_metric["pred_positive_rate"] = float(np.mean(pred))

    pred_export = test_df[[DATE_COL] + TRACE_COLS].copy()
    pred_export["fold"] = 1
    pred_export["y_true"] = y_true
    pred_export["y_prob_1"] = prob_1
    pred_export["y_pred"] = pred

    majority_class = int(train_df[TARGET_COL].mode().iloc[0])
    majority_pred = np.full(len(test_df), majority_class)
    train_mean_prob = float(train_df[TARGET_COL].mean())
    train_mean_pred = np.full(len(test_df), int(train_mean_prob >= 0.5))
    train_mean_prob_vec = np.full(len(test_df), train_mean_prob)
    # Analogue of "zero change" in regression: always P(up)=0 (never predict a positive 21d move).
    _zpp = 1e-15
    zero_dir_pred = np.zeros(len(test_df), dtype=int)
    zero_dir_prob_1 = np.full(len(test_df), _zpp, dtype=float)

    baseline_rows = [
        {
            "fold": 1,
            "baseline": "train_majority",
            **classification_metrics(y_true, majority_pred, majority_pred.astype(float)),
        },
        {
            "fold": 1,
            "baseline": "train_mean_thresholded",
            **classification_metrics(y_true, train_mean_pred, train_mean_prob_vec),
        },
        {
            "fold": 1,
            "baseline": "zero_class_always_down",
            **classification_metrics(y_true, zero_dir_pred, zero_dir_prob_1),
        },
    ]

    baseline_pred_rows = []
    for baseline_name, baseline_pred, baseline_prob in (
        ("train_majority", majority_pred, majority_pred.astype(float)),
        ("train_mean_thresholded", train_mean_pred, train_mean_prob_vec),
        ("zero_class_always_down", zero_dir_pred, zero_dir_prob_1),
    ):
        base_export = test_df[[DATE_COL] + TRACE_COLS].copy()
        base_export["fold"] = 1
        base_export["baseline"] = baseline_name
        base_export["y_true"] = y_true
        base_export["y_pred"] = baseline_pred
        base_export["y_prob_1"] = baseline_prob
        baseline_pred_rows.append(base_export)

    summary = {
        "label": label,
        "accuracy": float(holdout_metric["accuracy"]),
        "precision": float(holdout_metric["precision"]),
        "recall": float(holdout_metric["recall"]),
        "f1": float(holdout_metric["f1"]),
        "auc": float(holdout_metric["auc"]),
        "logloss": float(holdout_metric["logloss"]),
        "pred_std": float(holdout_metric["pred_std"]),
        "pred_positive_rate": float(holdout_metric["pred_positive_rate"]),
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


def objective_with_penalty(summary: Dict[str, float]) -> float:
    score = 1.0 - summary["f1"]

    if not np.isnan(summary["logloss"]):
        score += summary["logloss"] * 0.05

    pos_rate = summary["pred_positive_rate"]
    if pos_rate < OBJECTIVE_POS_RATE_FLOOR:
        score += (OBJECTIVE_POS_RATE_FLOOR - pos_rate) * 2.0
    if pos_rate > 1.0 - OBJECTIVE_POS_RATE_FLOOR:
        score += (pos_rate - (1.0 - OBJECTIVE_POS_RATE_FLOOR)) * 2.0

    return float(score)


def optuna_objective(trial, df: pd.DataFrame) -> float:
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
    result = evaluate_single_holdout(df, params, label="optuna_trial")
    penalized_score = objective_with_penalty(result.summary)
    trial.set_user_attr("accuracy", result.summary["accuracy"])
    trial.set_user_attr("f1", result.summary["f1"])
    trial.set_user_attr("auc", result.summary["auc"])
    trial.set_user_attr("pred_positive_rate", result.summary["pred_positive_rate"])
    trial.set_user_attr("penalized_score", penalized_score)
    return penalized_score


def run_optuna_search(df: pd.DataFrame) -> Dict:
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: optuna_objective(trial, df), n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
    payload = {
        "best_value_penalized_objective": float(study.best_value),
        "best_params": study.best_params,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }
    save_json(OUT_DIR / "optuna_best_params.json", payload)
    return study.best_params


def run_deap_search(df: pd.DataFrame) -> Dict:
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

    if not hasattr(creator, "FitnessMinRStarDir1Fold"):
        creator.create("FitnessMinRStarDir1Fold", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualRStarDir1Fold"):
        creator.create("IndividualRStarDir1Fold", list, fitness=creator.FitnessMinRStarDir1Fold)

    toolbox = base.Toolbox()

    def make_gene(i: int):
        return random.randrange(len(param_values[i]))

    toolbox.register(
        "individual",
        tools.initIterate,
        creator.IndividualRStarDir1Fold,
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
        result = evaluate_single_holdout(df, params, label="deap_trial")
        score = objective_with_penalty(result.summary)
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
            # Always run discrete mutator; per-locus flips are controlled in mutate_discrete.
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
        "best_score_penalized_objective": best_penalized_score,
        "best_params": best_params,
    }
    save_json(OUT_DIR / "deap_best_params.json", payload)
    return best_params


def export_holdout_results(result: HoldoutResult, prefix: str) -> None:
    result.holdout_metrics.to_csv(OUT_DIR / f"{prefix}_holdout_metrics.csv", index=False)
    result.oos_predictions.to_csv(OUT_DIR / f"{prefix}_holdout_oos_predictions.csv", index=False)
    result.baseline_metrics.to_csv(OUT_DIR / f"{prefix}_baseline_holdout_metrics.csv", index=False)
    result.baseline_oos_predictions.to_csv(OUT_DIR / f"{prefix}_baseline_holdout_oos_predictions.csv", index=False)
    save_json(OUT_DIR / f"{prefix}_holdout_summary.json", result.summary)


def choose_winner(optuna_result: HoldoutResult, deap_result: HoldoutResult) -> str:
    if optuna_result.summary["f1"] > deap_result.summary["f1"]:
        return "optuna"
    if deap_result.summary["f1"] > optuna_result.summary["f1"]:
        return "deap"
    return "optuna" if optuna_result.summary["auc"] >= deap_result.summary["auc"] else "deap"


def fit_final_model(df: pd.DataFrame, params: Dict) -> XGBClassifier:
    model = build_base_model(params)
    model.fit(df[get_feature_cols()], df[TARGET_COL])
    model.save_model(str(OUT_DIR / "final_model_xgb.json"))
    return model


def run_shap(final_model: XGBClassifier, df: pd.DataFrame) -> None:
    fcols = get_feature_cols()
    shap_df = df[[DATE_COL] + fcols + [TARGET_COL]].copy()
    shap_df = shap_df.tail(min(SHAP_MAX_ROWS, len(shap_df))).reset_index(drop=True)
    X_shap = shap_df[fcols]

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values_df = pd.DataFrame(shap_values, columns=fcols)
    shap_values_df.insert(0, DATE_COL, shap_df[DATE_COL].astype(str))
    shap_values_df.to_csv(OUT_DIR / "shap_summary_values.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_bar.png", dpi=160, bbox_inches="tight")
    plt.close()


def write_run_contents_summary() -> None:
    lines = [
        "train_rstar_directional_1fold.py output summary",
        "",
        f"Dataset: {DATA_PATH}",
        f"Output directory: {OUT_DIR}",
        f"Raw return column: {RAW_RETURN_COL}",
        f"Target: {TARGET_COL} (binary direction from {RAW_RETURN_COL}, XGBoost classifier)",
        f"DIRECTION_THRESHOLD: {DIRECTION_THRESHOLD}",
        f"Split: train_frac={TRAIN_FRAC} gap_rows={GAP_ROWS} test_frac={TEST_FRAC}",
        f"DEAP_MUTATION_PROB (per-gene): {DEAP_MUTATION_PROB}",
        f"OPTUNA_N_TRIALS: {OPTUNA_N_TRIALS}",
        f"Excluded from model (regime-shift levels): {list(DIRECTIONAL_EXCLUDED_FEATURES)}",
        f"SORA path level column: {SORA_PATH_LEVEL_COL}",
        "",
        "Base feature cols (before exclusion):",
    ] + [f"  - {c}" for c in BASE_FEATURE_COLS]
    lines += [
        "",
        "Engineered REIT index cols:",
    ] + [f"  - {c}" for c in ENGINEERED_REIT_COLS]
    lines += [
        "",
        "Engineered SORA path cols:",
    ] + [f"  - {c}" for c in ENGINEERED_SORA_PATH_COLS]
    lines += [
        "",
        "Model feature cols (active, after exclusion):",
    ] + [f"  - {c}" for c in get_feature_cols()]
    lines += [
        "",
        "Baselines in *_baseline_holdout_metrics: train_majority, train_mean_thresholded, "
        "zero_class_always_down (P(up)=0, analogue of zero forecast for level/change).",
        "",
        "Output files:",
        "  data_manifest.json",
        "  feature_manifest.json",
        "  optuna_best_params.json",
        "  deap_best_params.json",
        "  <optuna|deap>_holdout_metrics.csv",
        "  <optuna|deap>_holdout_oos_predictions.csv",
        "  <optuna|deap>_baseline_holdout_metrics.csv",
        "  <optuna|deap>_baseline_holdout_oos_predictions.csv",
        "  <optuna|deap>_holdout_summary.json",
        "  optimizer_comparison.json",
        "  final_selected_params.json",
        "  final_model_xgb.json",
        "  shap_summary_values.csv",
        "  shap_summary_beeswarm.png",
        "  shap_summary_bar.png",
        "  run_contents_summary.txt",
    ]
    (OUT_DIR / "run_contents_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_global_seed(SEED)
    print("Loading dataset ...")
    df = load_dataset()
    print(f"Usable rows after target drop: {len(df):,}")
    print(f"Training window: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")

    train_idx, test_idx = get_custom_single_split(df)
    print(f"Custom split sizes: train={len(train_idx)} gap={GAP_ROWS} test={len(test_idx)}")
    print(f"Direction threshold: {DIRECTION_THRESHOLD}")
    fcols = get_feature_cols()
    print(f"Model feature count (excl. {list(DIRECTIONAL_EXCLUDED_FEATURES)}): {len(fcols)}")

    print("\nRunning Optuna search ...")
    optuna_params = run_optuna_search(df)
    optuna_eval = evaluate_single_holdout(df, optuna_params, label="optuna")
    export_holdout_results(optuna_eval, "optuna")

    print("\nRunning DEAP search ...")
    deap_params = run_deap_search(df)
    deap_eval = evaluate_single_holdout(df, deap_params, label="deap")
    export_holdout_results(deap_eval, "deap")

    winner = choose_winner(optuna_eval, deap_eval)
    final_params = optuna_params if winner == "optuna" else deap_params

    comparison = {
        "winner": winner,
        "optuna_summary": optuna_eval.summary,
        "deap_summary": deap_eval.summary,
        "winner_params": final_params,
    }
    save_json(OUT_DIR / "optimizer_comparison.json", comparison)
    save_json(OUT_DIR / "final_selected_params.json", final_params)

    print(f"\nSelected optimizer: {winner}")
    print(f"Selected gamma: {final_params.get('gamma')}")
    bdf = optuna_eval.baseline_metrics
    for name in (
        "train_mean_thresholded",
        "zero_class_always_down",
    ):
        row = bdf[bdf["baseline"] == name]
        if not row.empty:
            r0 = row.iloc[0]
            print(
                f"Baseline {name!r}: accuracy={r0['accuracy']:.4f} "
                f"f1={r0['f1']:.4f} auc={r0['auc']:.4f} logloss={r0['logloss']:.4f}"
            )
    win_sum = optuna_eval.summary if winner == "optuna" else deap_eval.summary
    print(
        f"Model (winner) holdout: accuracy={win_sum['accuracy']:.4f} f1={win_sum['f1']:.4f} "
        f"auc={win_sum['auc']:.4f} logloss={win_sum['logloss']:.4f} "
        f"pred_pos_rate={win_sum['pred_positive_rate']:.4f}"
    )

    print("\nFitting final model on all usable rows ...")
    final_model = fit_final_model(df, final_params)

    print("\nRunning SHAP ...")
    run_shap(final_model, df)
    write_run_contents_summary()
    print("\nDone. Outputs written to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
