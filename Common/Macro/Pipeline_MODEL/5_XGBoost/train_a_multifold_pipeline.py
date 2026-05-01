r"""
train_a_multifold_pipeline.py
=============================

Pooled multifold Model A pipeline.

Purpose:
    Build a one-row-per-ticker-per-SGX-day abnormal-returns panel, then train a
    single pooled XGBoost model with walkforward validation, Optuna, and DEAP.

Core Model A behavior:
    - Training rows are pooled across the REIT universe.
    - Index rows are never part of the training universe; the S-REIT index is
      only used as the benchmark for beta adjustment and sector-state features.
    - The effective ticker universe is controlled by hard-poison / caveat
      buckets plus manual overrides.
    - Row-level exclusions are supported for tickers whose early history is
      poisoned but later history is usable.

Output location:
    Consolidated/IO/Model_Train/train_a_multifold_pipeline/run_<n>/fwd_<horizon>_days/

This script writes:
    - model_a_panel_dataset.csv
    - data_manifest.json
    - feature_manifest.json
    - model_a_<optuna|deap>_walkforward_fold_metrics.csv
    - model_a_<optuna|deap>_walkforward_oos_predictions.csv
    - model_a_<optuna|deap>_walkforward_summary.json
    - model_a_final_model_xgb.json
    - model_a_shap_summary_values.csv
    - run_contents_summary.txt
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from sklearn.model_selection import TimeSeriesSplit


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
N_SPLITS = 3
FINAL_HOLDOUT_DATES = 126
DEFAULT_FORWARD_HORIZON_ROWS_LIST = (21,) # (10, 15)
PER_HORIZON_SEED_OFFSET = True
EXPORT_PANEL_CSV = True

BETA_WINDOW_DAYS = 252
CAR_LOOKBACK_DAYS = 63
CAR_SUBWINDOW_DAYS = 21

AUC_TOLERANCE = 0.005
RMSE_TOLERANCE = 0.002
R2_TOLERANCE = 0.02
PRED_POSITIVE_RATE_TOLERANCE = 0.08
WINNER_SELECTION_MODE = "directional"

SCRIPT_DIR = Path(__file__).resolve().parent
CONSOLIDATED_ROOT = SCRIPT_DIR.parents[1]
IO_SRC_DIR = CONSOLIDATED_ROOT / "IO" / "SRC"
FED_OUTPUT_DIR = IO_SRC_DIR / "CSV_FED" / "Output"
MAS_OUTPUT_DIR = IO_SRC_DIR / "CSV_MAS" / "Output"
MODEL_DIR = IO_SRC_DIR / "MODEL"
MODEL_TRAIN_DIR = CONSOLIDATED_ROOT / "IO" / "Model_Train"

XGB_READY_PATH = FED_OUTPUT_DIR / "timeseries_2022-07-27_2026-03-18_xgb_ready.csv"
SGX_REIT_INDEX_PATH = MODEL_DIR / "SGX_DLY_REIT, 1D.csv"
SORA_DAILY_PATH = MAS_OUTPUT_DIR / "sora_daily.csv"
SORA_3M_DAILY_PATH = MAS_OUTPUT_DIR / "sora_3m_daily.csv"
TICKER_UNIVERSE_REFERENCE_PATH = MODEL_DIR / "Tickers_SingleTable.md"

RUNS_ROOT = MODEL_TRAIN_DIR / Path(__file__).stem
RUNS_ROOT.mkdir(exist_ok=True)
MAX_HORIZON_WORKERS = 2

DATE_COL = "snapshot_ts"
TARGET_KEY = "model_a"
TARGET_SIGNED = True

TRACE_COLS = [
    "ticker",
    "fomc_decision_date",
    "ticker_open",
    "ticker_high",
    "ticker_low",
    "ticker_close",
    "reit_index_open",
    "reit_index_high",
    "reit_index_low",
    "reit_index_close",
    "beta_252d",
    "ticker_daily_return",
    "reit_index_daily_return",
    "ticker_fwd_return",
    "reit_index_fwd_return",
]

BASE_FEATURE_COLS = [
    "car_63d",
    "car_sign_consistency",
    "car_acceleration",
    "expected_bps",
    "sora_level_t2",
]

EXTENDED_CAR_FEATURE_COLS = [
    "car_window_recent",
    "car_window_mid",
    "car_window_old",
]

EXTENDED_SORA_FEATURE_COLS = [
    "sora_90d_change_t2",
    "sora_3m_t2",
    "sora_term_spread_t2",
]

ENGINEERED_REIT_STATE_COLS = [
    "reit_index_lag_21d_return",
    "reit_index_lag_63d_return",
    "reit_index_vol_21d",
    "reit_index_vol_63d",
    "reit_index_drawdown_63d",
]

INTERACTION_FEATURE_SPECS = [
    ("expected_bps_x_reit_index_vol_21d", ("expected_bps", "reit_index_vol_21d")),
    ("expected_bps_x_sora_level_t2", ("expected_bps", "sora_level_t2")),
    ("car_acceleration_x_reit_index_vol_21d", ("car_acceleration", "reit_index_vol_21d")),
    ("car_63d_x_reit_index_lag_63d_return", ("car_63d", "reit_index_lag_63d_return")),
    ("car_window_recent_x_reit_index_vol_21d", ("car_window_recent", "reit_index_vol_21d")),
    ("reit_index_vol_21d_x_drawdown_63d", ("reit_index_vol_21d", "reit_index_drawdown_63d")),
]


# ---------------------------------------------------------------------------
# Model A ticker-universe controls
# ---------------------------------------------------------------------------
INDEX_TICKERS = ("REIT", "REITN", "REITR")

ALL_REIT_TICKERS = (
    "A17U",
    "AJBU",
    "BTOU",
    "BUOU",
    "BWCU",
    "C38U",
    "CMOU",
    "D5IU",
    "J69U",
    "M1GU",
    "M44U",
    "ME8U",
    "N2IU",
    "OXMU",
    "T82U",
)

HARD_POISON_TICKERS = (
    "BWCU",
    "D5IU",
)

CAVEAT_TICKERS = (
    "BTOU",  # MUST
    "OXMU",  # Prime US REIT
    "CMOU",  # KORE
)

ROW_LEVEL_EXCLUSIONS = {
    # "D5IU": {
    #     "exclude_before": "2022-01-01",
    #     "reason": "LMIRT pre-2022 sponsor/credit-event regime is outside the intended macro feature schema.",
    # },  # NOTE: exclude D5IU for first training attempts for simplicity
}

EXCLUDE_HARD_POISON_TICKERS = True
EXCLUDE_CAVEAT_TICKERS = True

MANUAL_EXCLUDE_TICKERS = ()
MANUAL_INCLUDE_TICKERS = ()
ENABLE_TICKER_ONEHOT_FEATURES = False
ENABLE_EXTENDED_CAR_FEATURES = True
ENABLE_EXTENDED_SORA_FEATURES = True
ENABLE_INTERACTION_FEATURES = False
MANUAL_DISABLE_FEATURES = ()
MANUAL_ENABLE_FEATURES = ()

OPTUNA_N_TRIALS = 80
DEAP_GENERATIONS = 8
DEAP_POPULATION_SIZE = 20
DEAP_MUTATION_PROB = 1.0 / 9.0
DEAP_CROSSOVER_PROB = 0.6
DEAP_TOURNAMENT_SIZE = 3

SHAP_MAX_ROWS = 250
OBJECTIVE_NEGATIVE_R2_PENALTY = 0.10
OBJECTIVE_POSITIVE_R2_REWARD = 0.01
OBJECTIVE_PRED_STD_RATIO_FLOOR = 0.35
OBJECTIVE_PRED_STD_RATIO_PENALTY = 0.05
OBJECTIVE_SIGN_BALANCE_FLOOR = 0.25
OBJECTIVE_SIGN_BALANCE_CENTER = 0.50
OBJECTIVE_SIGN_BALANCE_TARGET_BAND = 0.10
OBJECTIVE_SIGN_BALANCE_PENALTY = 0.20
ENABLE_EARLY_STOPPING = False
EARLY_STOPPING_VALID_FRACTION = 0.20
EARLY_STOPPING_MIN_VALID_DATES = 21
EARLY_STOPPING_MIN_TRAIN_DATES = 63
EARLY_STOPPING_ROUNDS = 50


def interaction_feature_cols() -> List[str]:
    return [name for name, _ in INTERACTION_FEATURE_SPECS]


def effective_excluded_tickers() -> List[str]:
    excluded = set()
    if EXCLUDE_HARD_POISON_TICKERS:
        excluded.update(HARD_POISON_TICKERS)
    if EXCLUDE_CAVEAT_TICKERS:
        excluded.update(CAVEAT_TICKERS)
    excluded.update(MANUAL_EXCLUDE_TICKERS)
    excluded.difference_update(MANUAL_INCLUDE_TICKERS)
    return sorted(excluded)


def effective_reit_universe() -> List[str]:
    excluded = set(effective_excluded_tickers())
    return [ticker for ticker in ALL_REIT_TICKERS if ticker not in excluded]


def dynamic_ticker_dummy_cols(universe: Iterable[str]) -> List[str]:
    return [f"is_{ticker}" for ticker in universe]


def all_available_feature_cols(universe: Iterable[str]) -> List[str]:
    return (
        list(BASE_FEATURE_COLS)
        + list(EXTENDED_CAR_FEATURE_COLS)
        + list(EXTENDED_SORA_FEATURE_COLS)
        + list(ENGINEERED_REIT_STATE_COLS)
        + list(interaction_feature_cols())
        + dynamic_ticker_dummy_cols(universe)
    )


def feature_groups(
    universe: Iterable[str],
    include_ticker_identity: bool = ENABLE_TICKER_ONEHOT_FEATURES,
) -> Dict[str, List[str]]:
    ticker_cols = dynamic_ticker_dummy_cols(universe) if include_ticker_identity else []
    groups = {
        "abnormal_history": list(BASE_FEATURE_COLS[:3]),
        "macro_base": list(BASE_FEATURE_COLS[3:]),
        "abnormal_history_extended": list(EXTENDED_CAR_FEATURE_COLS) if ENABLE_EXTENDED_CAR_FEATURES else [],
        "macro_extended": list(EXTENDED_SORA_FEATURE_COLS) if ENABLE_EXTENDED_SORA_FEATURES else [],
        "reit_sector_state": list(ENGINEERED_REIT_STATE_COLS),
        "interaction_terms": list(interaction_feature_cols()) if ENABLE_INTERACTION_FEATURES else [],
        "ticker_identity": ticker_cols,
    }
    unknown_overrides = sorted(
        set(MANUAL_DISABLE_FEATURES).union(MANUAL_ENABLE_FEATURES) - set(all_available_feature_cols(universe))
    )
    if unknown_overrides:
        raise ValueError(f"Unknown manual feature overrides: {unknown_overrides}")

    filtered_groups: Dict[str, List[str]] = {}
    used = set()
    for group_name, cols in groups.items():
        filtered = [col for col in cols if col not in MANUAL_DISABLE_FEATURES]
        filtered_groups[group_name] = filtered
        used.update(filtered)

    manual_overrides = [col for col in MANUAL_ENABLE_FEATURES if col not in used]
    if manual_overrides:
        filtered_groups["manual_overrides"] = manual_overrides
    return filtered_groups


def feature_cols(
    universe: Iterable[str],
    include_ticker_identity: bool = ENABLE_TICKER_ONEHOT_FEATURES,
) -> List[str]:
    groups = feature_groups(
        universe,
        include_ticker_identity=include_ticker_identity,
    )
    return (
        groups["abnormal_history"]
        + groups["macro_base"]
        + groups.get("abnormal_history_extended", [])
        + groups.get("macro_extended", [])
        + groups["reit_sector_state"]
        + groups.get("interaction_terms", [])
        + groups["ticker_identity"]
        + groups.get("manual_overrides", [])
    )


def forward_suffix(rows: int) -> str:
    return f"fwd_{rows}d"


def model_a_target_col(rows: int) -> str:
    return f"abnormal_{forward_suffix(rows)}_return"


def create_run_root() -> Path:
    run_n = 0
    while (RUNS_ROOT / f"run_{run_n}").exists():
        run_n += 1
    run_root = RUNS_ROOT / f"run_{run_n}"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


@dataclass(frozen=True)
class HorizonConfig:
    horizon_rows: int
    run_root: Path
    worker_seed: int
    export_panel_csv: bool = EXPORT_PANEL_CSV

    @property
    def out_dir(self) -> Path:
        return self.run_root / f"fwd_{self.horizon_rows}_days"

    @property
    def panel_export_path(self) -> Path:
        return self.out_dir / "model_a_panel_dataset.csv"

    @property
    def data_path(self) -> Path:
        return self.panel_export_path

    @property
    def target_col(self) -> str:
        return model_a_target_col(self.horizon_rows)


def compute_worker_seed(
    horizon_rows: int,
    selected_horizons: List[int],
    base_seed: int = SEED,
    per_horizon_seed_offset: bool = PER_HORIZON_SEED_OFFSET,
) -> int:
    if len(selected_horizons) <= 1:
        return base_seed
    if per_horizon_seed_offset:
        return base_seed + horizon_rows
    return base_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Model A multifold pipeline: build pooled abnormal-returns panel and train with Optuna+DEAP."
    )
    p.add_argument(
        "--drop-calendar-by",
        choices=("sreit", "sora"),
        default="sreit",
        help="Row-universe switch for the macro feature calendar. Default uses the S-REIT trading calendar.",
    )
    p.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_FORWARD_HORIZON_ROWS_LIST),
        help="Forward SGX-row horizons to run.",
    )
    p.add_argument(
        "--max-horizon-workers",
        type=int,
        default=MAX_HORIZON_WORKERS,
        help="Max number of horizon jobs to run in parallel when multiple horizons are requested.",
    )
    return p.parse_args()


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


def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)
    valid = y_true_s.notna() & y_pred_s.notna()
    if int(valid.sum()) < 2:
        return {
            "spearman_corr": float("nan"),
            "top_decile_hit_rate": float("nan"),
            "bottom_decile_hit_rate": float("nan"),
        }

    y_true_valid = y_true_s.loc[valid]
    y_pred_valid = y_pred_s.loc[valid]
    y_true_rank = y_true_valid.rank(method="average")
    y_pred_rank = y_pred_valid.rank(method="average")
    if y_true_rank.nunique(dropna=True) < 2 or y_pred_rank.nunique(dropna=True) < 2:
        spearman_corr = float("nan")
    else:
        spearman_corr = y_true_rank.corr(y_pred_rank, method="pearson")

    decile_n = max(1, int(np.ceil(len(y_true_valid) * 0.10)))
    top_idx = y_pred_valid.nlargest(decile_n).index
    bottom_idx = y_pred_valid.nsmallest(decile_n).index
    top_hit = float((y_true_valid.loc[top_idx] > 0).mean()) if len(top_idx) > 0 else float("nan")
    bottom_hit = float((y_true_valid.loc[bottom_idx] < 0).mean()) if len(bottom_idx) > 0 else float("nan")
    return {
        "spearman_corr": float(spearman_corr) if pd.notna(spearman_corr) else float("nan"),
        "top_decile_hit_rate": top_hit,
        "bottom_decile_hit_rate": bottom_hit,
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
    out.update(ranking_metrics(y_true, y_pred))
    return out


def save_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def print_deap_generation_progress(
    target_key: str,
    generation: int,
    total_generations: int,
    population,
    hall_of_fame,
) -> None:
    fitness_values = [float(ind.fitness.values[0]) for ind in population if ind.fitness.valid]
    if not fitness_values:
        print(f"  [DEAP:{target_key}] gen {generation:02d}/{total_generations:02d} | no valid fitness values")
        return
    best_score = min(fitness_values)
    mean_score = float(np.mean(fitness_values))
    worst_score = max(fitness_values)
    hof_score = float(hall_of_fame[0].fitness.values[0]) if len(hall_of_fame) > 0 else float("nan")
    print(
        f"  [DEAP:{target_key}] gen {generation:02d}/{total_generations:02d} "
        f"| best={best_score:.6f} mean={mean_score:.6f} worst={worst_score:.6f} hof={hof_score:.6f}"
    )


def load_tradingview_price_csv(path: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["snapshot_ts"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("snapshot_ts").reset_index(drop=True)
    df["ticker"] = ticker
    return df


def build_reit_index_history(cfg: HorizonConfig) -> pd.DataFrame:
    reit_index = load_tradingview_price_csv(SGX_REIT_INDEX_PATH, ticker="REIT")
    reit_index["reit_index_open"] = pd.to_numeric(reit_index["open"], errors="coerce")
    reit_index["reit_index_high"] = pd.to_numeric(reit_index["high"], errors="coerce")
    reit_index["reit_index_low"] = pd.to_numeric(reit_index["low"], errors="coerce")
    reit_index["reit_index_close"] = pd.to_numeric(reit_index["close"], errors="coerce")
    reit_index["reit_index_daily_return"] = reit_index["reit_index_close"].pct_change(1)
    reit_index["reit_index_fwd_return"] = (
        reit_index["reit_index_close"].shift(-cfg.horizon_rows) - reit_index["reit_index_close"]
    ) / reit_index["reit_index_close"]
    reit_index["reit_index_fwd_return"] = reit_index["reit_index_fwd_return"].round(6)
    close = reit_index["reit_index_close"]
    reit_index["reit_index_lag_21d_return"] = close.pct_change(21)
    reit_index["reit_index_lag_63d_return"] = close.pct_change(63)
    reit_daily_ret = close.pct_change(1)
    reit_index["reit_index_vol_21d"] = reit_daily_ret.rolling(21).std()
    reit_index["reit_index_vol_63d"] = reit_daily_ret.rolling(63).std()
    reit_index["reit_index_drawdown_63d"] = close / close.rolling(63).max() - 1.0
    return reit_index[
        [
            "snapshot_ts",
            "reit_index_open",
            "reit_index_high",
            "reit_index_low",
            "reit_index_close",
            "reit_index_daily_return",
            "reit_index_fwd_return",
            "reit_index_lag_21d_return",
            "reit_index_lag_63d_return",
            "reit_index_vol_21d",
            "reit_index_vol_63d",
            "reit_index_drawdown_63d",
        ]
    ].copy()


def build_macro_feature_dataset(cfg: HorizonConfig, drop_calendar_by: str, reit_index: pd.DataFrame) -> pd.DataFrame:
    sora = pd.read_csv(SORA_DAILY_PATH, parse_dates=["value_date"])
    sora_3m = pd.read_csv(SORA_3M_DAILY_PATH, parse_dates=["value_date"])
    sora_realized = sora.copy()

    sora["sora_level"] = sora["sora_level"].shift(2)
    sora_3m["sora_3m"] = sora_3m["sora_3m"].shift(2)
    sgx_trading_days = reit_index["snapshot_ts"].drop_duplicates().sort_values().reset_index(drop=True)

    realized_sora_publication_days = (
        sora_realized["value_date"].drop_duplicates().sort_values().reset_index(drop=True)
    )

    realized_cal_start = sora_realized["value_date"].min()
    realized_cal_end = sora_realized["value_date"].max()
    realized_cal_index = pd.DataFrame(
        {"value_date": pd.date_range(realized_cal_start, realized_cal_end, freq="D")}
    )
    realized_sora_cal = realized_cal_index.merge(sora_realized, on="value_date", how="left")
    realized_sora_cal["sora_level_realized"] = realized_sora_cal["sora_level"].ffill()
    realized_sora_cal = realized_sora_cal[["value_date", "sora_level_realized"]].copy()
    realized_sora_cal = realized_sora_cal.rename(columns={"value_date": "snapshot_ts"})
    realized_sora_cal["snapshot_ts"] = realized_sora_cal["snapshot_ts"].dt.normalize()

    cal_start = sora["value_date"].min()
    cal_end = sora["value_date"].max()
    cal_index = pd.DataFrame({"value_date": pd.date_range(cal_start, cal_end, freq="D")})
    sora_cal = cal_index.merge(sora, on="value_date", how="left")
    sora_cal = sora_cal.merge(sora_3m, on="value_date", how="left")
    sora_cal["sora_level"] = sora_cal["sora_level"].ffill()
    sora_cal["sora_3m"] = sora_cal["sora_3m"].ffill()
    sora_cal = sora_cal.set_index("value_date")
    sora_cal["sora_90d_change"] = ((sora_cal["sora_level"] - sora_cal["sora_level"].shift(90)) * 100).round(4)
    sora_cal["sora_term_spread"] = (sora_cal["sora_3m"] - sora_cal["sora_level"]).round(4)
    sora_cal = sora_cal.rename(
        columns={
            "sora_level": "sora_level_t2",
            "sora_90d_change": "sora_90d_change_t2",
            "sora_3m": "sora_3m_t2",
            "sora_term_spread": "sora_term_spread_t2",
        }
    ).reset_index().rename(columns={"value_date": "snapshot_ts"})
    sora_cal["snapshot_ts"] = pd.to_datetime(sora_cal["snapshot_ts"]).dt.normalize()

    xgb = pd.read_csv(XGB_READY_PATH, parse_dates=["snapshot_ts"])
    xgb["snapshot_ts"] = xgb["snapshot_ts"].dt.normalize()

    if drop_calendar_by == "sora":
        allowed_days = realized_sora_publication_days
    else:
        allowed_days = sgx_trading_days

    xgb = xgb[xgb["snapshot_ts"].isin(set(allowed_days))].copy()
    xgb = xgb.sort_values("snapshot_ts").reset_index(drop=True)

    macro = xgb.merge(
        reit_index[
            [
                "snapshot_ts",
                "reit_index_open",
                "reit_index_high",
                "reit_index_low",
                "reit_index_close",
                "reit_index_daily_return",
                "reit_index_fwd_return",
            ]
        ],
        on="snapshot_ts",
        how="left",
    )
    macro = macro.merge(
        realized_sora_cal[["snapshot_ts", "sora_level_realized"]],
        on="snapshot_ts",
        how="left",
    )
    macro = macro.merge(
        sora_cal[["snapshot_ts", "sora_level_t2", "sora_90d_change_t2", "sora_3m_t2", "sora_term_spread_t2"]],
        on="snapshot_ts",
        how="left",
    )
    macro = macro.sort_values("snapshot_ts").reset_index(drop=True)
    return macro


def build_ticker_panel(cfg: HorizonConfig, reit_index: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ticker in effective_reit_universe():
        path = MODEL_DIR / f"SGX_DLY_{ticker}, 1D.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing ticker price file for {ticker}: {path}")
        df = load_tradingview_price_csv(path, ticker=ticker)
        df["ticker_open"] = pd.to_numeric(df["open"], errors="coerce")
        df["ticker_high"] = pd.to_numeric(df["high"], errors="coerce")
        df["ticker_low"] = pd.to_numeric(df["low"], errors="coerce")
        df["ticker_close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ticker_daily_return"] = df["ticker_close"].pct_change(1)
        df["ticker_fwd_return"] = (
            df["ticker_close"].shift(-cfg.horizon_rows) - df["ticker_close"]
        ) / df["ticker_close"]
        df = df.merge(reit_index, on="snapshot_ts", how="inner")
        frames.append(
            df[
                [
                    "snapshot_ts",
                    "ticker",
                    "ticker_open",
                    "ticker_high",
                    "ticker_low",
                    "ticker_close",
                    "ticker_daily_return",
                    "ticker_fwd_return",
                    "reit_index_open",
                    "reit_index_high",
                    "reit_index_low",
                    "reit_index_close",
                    "reit_index_daily_return",
                    "reit_index_fwd_return",
                    "reit_index_lag_21d_return",
                    "reit_index_lag_63d_return",
                    "reit_index_vol_21d",
                    "reit_index_vol_63d",
                    "reit_index_drawdown_63d",
                ]
            ].copy()
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "snapshot_ts"]).reset_index(drop=True)
    return panel


def apply_row_level_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for ticker, rule in ROW_LEVEL_EXCLUSIONS.items():
        if "exclude_before" in rule:
            cutoff = pd.Timestamp(rule["exclude_before"])
            out = out[
                ~((out["ticker"] == ticker) & (out["snapshot_ts"] < cutoff))
            ].copy()
        if "exclude_after" in rule:
            cutoff = pd.Timestamp(rule["exclude_after"])
            out = out[
                ~((out["ticker"] == ticker) & (out["snapshot_ts"] > cutoff))
            ].copy()
    return out


def rolling_beta(group: pd.DataFrame) -> pd.Series:
    y = group["ticker_daily_return"]
    x = group["reit_index_daily_return"]
    cov = y.rolling(BETA_WINDOW_DAYS, min_periods=BETA_WINDOW_DAYS).cov(x)
    var = x.rolling(BETA_WINDOW_DAYS, min_periods=BETA_WINDOW_DAYS).var()
    beta = cov / var.replace(0.0, np.nan)
    return beta


def add_model_a_features(
    panel: pd.DataFrame,
    universe: List[str],
) -> pd.DataFrame:
    out = panel.copy()
    out["beta_252d"] = np.nan
    out["car_63d"] = np.nan
    out["car_window_recent"] = np.nan
    out["car_window_mid"] = np.nan
    out["car_window_old"] = np.nan
    out["car_sign_consistency"] = np.nan
    out["car_acceleration"] = np.nan

    ticker_groups = out.groupby("ticker", sort=False).groups
    for _, raw_index in ticker_groups.items():
        ticker_index = pd.Index(raw_index)
        ordered_index = out.loc[ticker_index].sort_values(DATE_COL).index
        ordered_group = out.loc[ordered_index]

        beta = rolling_beta(ordered_group)
        out.loc[ordered_index, "beta_252d"] = beta.to_numpy()

    out["abnormal_daily_return"] = out["ticker_daily_return"] - out["beta_252d"] * out["reit_index_daily_return"]

    for _, raw_index in ticker_groups.items():
        ticker_index = pd.Index(raw_index)
        ordered_index = out.loc[ticker_index].sort_values(DATE_COL).index
        ordered_close = out.loc[ordered_index, "ticker_close"]
        ordered_bench_close = out.loc[ordered_index, "reit_index_close"]
        abnormal_daily = out.loc[ordered_index, "abnormal_daily_return"]

        ticker_ret_63d = ordered_close.div(ordered_close.shift(CAR_LOOKBACK_DAYS)) - 1.0
        bench_ret_63d = ordered_bench_close.div(ordered_bench_close.shift(CAR_LOOKBACK_DAYS)) - 1.0
        car_63d = ticker_ret_63d - out.loc[ordered_index, "beta_252d"] * bench_ret_63d

        ticker_recent = ordered_close.div(ordered_close.shift(CAR_SUBWINDOW_DAYS)) - 1.0
        bench_recent = ordered_bench_close.div(ordered_bench_close.shift(CAR_SUBWINDOW_DAYS)) - 1.0
        car_recent = ticker_recent - out.loc[ordered_index, "beta_252d"] * bench_recent

        ticker_mid = ordered_close.shift(CAR_SUBWINDOW_DAYS).div(ordered_close.shift(2 * CAR_SUBWINDOW_DAYS)) - 1.0
        bench_mid = ordered_bench_close.shift(CAR_SUBWINDOW_DAYS).div(
            ordered_bench_close.shift(2 * CAR_SUBWINDOW_DAYS)
        ) - 1.0
        car_mid = ticker_mid - out.loc[ordered_index, "beta_252d"] * bench_mid

        ticker_old = ordered_close.shift(2 * CAR_SUBWINDOW_DAYS).div(
            ordered_close.shift(3 * CAR_SUBWINDOW_DAYS)
        ) - 1.0
        bench_old = ordered_bench_close.shift(2 * CAR_SUBWINDOW_DAYS).div(
            ordered_bench_close.shift(3 * CAR_SUBWINDOW_DAYS)
        ) - 1.0
        car_old = ticker_old - out.loc[ordered_index, "beta_252d"] * bench_old

        valid_car_windows = car_recent.notna() & car_mid.notna() & car_old.notna()
        car_consistency = pd.Series(np.nan, index=ordered_index, dtype=float)
        car_consistency.loc[valid_car_windows] = (
            (car_old.loc[valid_car_windows] < 0).astype(float)
            + (car_mid.loc[valid_car_windows] < 0).astype(float)
            + (car_recent.loc[valid_car_windows] < 0).astype(float)
        )
        car_acceleration = car_recent - car_old

        out.loc[ordered_index, "car_63d"] = car_63d.to_numpy()
        out.loc[ordered_index, "car_window_recent"] = car_recent.to_numpy()
        out.loc[ordered_index, "car_window_mid"] = car_mid.to_numpy()
        out.loc[ordered_index, "car_window_old"] = car_old.to_numpy()
        out.loc[ordered_index, "car_sign_consistency"] = car_consistency.to_numpy()
        out.loc[ordered_index, "car_acceleration"] = car_acceleration.to_numpy()

    out["car_sign_consistency"] = out["car_sign_consistency"].astype(float)
    out["abnormal_target_component"] = out["beta_252d"] * out["reit_index_fwd_return"]
    out["abnormal_fwd_return"] = out["ticker_fwd_return"] - out["abnormal_target_component"]
    out["target_col_runtime"] = out["abnormal_fwd_return"]

    for ticker in universe:
        out[f"is_{ticker}"] = (out["ticker"] == ticker).astype(int)

    return out


def add_model_a_interaction_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for feature_name, (left_col, right_col) in INTERACTION_FEATURE_SPECS:
        out[feature_name] = pd.to_numeric(out[left_col], errors="coerce") * pd.to_numeric(
            out[right_col], errors="coerce"
        )
    return out


def build_model_a_panel_dataset(cfg: HorizonConfig, drop_calendar_by: str) -> Tuple[pd.DataFrame, List[str]]:
    universe = effective_reit_universe()
    reit_index = build_reit_index_history(cfg)
    macro = build_macro_feature_dataset(cfg=cfg, drop_calendar_by=drop_calendar_by, reit_index=reit_index)
    ticker_panel = build_ticker_panel(cfg=cfg, reit_index=reit_index)
    ticker_panel = apply_row_level_exclusions(ticker_panel)
    panel = add_model_a_features(
        panel=ticker_panel,
        universe=universe,
    )
    macro_cols = [
        col
        for col in macro.columns
        if col
        not in {
            "reit_index_open",
            "reit_index_high",
            "reit_index_low",
            "reit_index_close",
            "reit_index_daily_return",
            "reit_index_fwd_return",
            "reit_index_lag_21d_return",
            "reit_index_lag_63d_return",
            "reit_index_vol_21d",
            "reit_index_vol_63d",
            "reit_index_drawdown_63d",
        }
    ]
    panel = panel.merge(macro[macro_cols], on="snapshot_ts", how="inner")
    panel = add_model_a_interaction_features(panel)
    panel = panel.sort_values(["snapshot_ts", "ticker"]).reset_index(drop=True)
    panel[cfg.target_col] = panel["abnormal_fwd_return"]
    active_feature_cols = feature_cols(
        universe,
        include_ticker_identity=ENABLE_TICKER_ONEHOT_FEATURES,
    )
    missing_active_cols = [col for col in active_feature_cols if col not in panel.columns]
    if missing_active_cols:
        raise ValueError(f"Missing active feature columns in panel dataset: {missing_active_cols}")

    numeric_cols = list(
        dict.fromkeys(
            active_feature_cols
            + [
                cfg.target_col,
                "beta_252d",
                "ticker_daily_return",
                "reit_index_daily_return",
                "ticker_fwd_return",
                "reit_index_fwd_return",
            ]
        )
    )
    for col in numeric_cols:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = panel.sort_values([DATE_COL, "ticker"]).reset_index(drop=True)
    if cfg.export_panel_csv:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        panel.to_csv(cfg.panel_export_path, index=False)
        print(f"Saved Model A panel dataset: {cfg.panel_export_path}")
    return panel, active_feature_cols


def write_data_manifests(
    cfg: HorizonConfig,
    base_df: pd.DataFrame,
    feature_cols_used: List[str],
    drop_calendar_by: str,
) -> None:
    ticker_groups = feature_groups(
        effective_reit_universe(),
        include_ticker_identity=ENABLE_TICKER_ONEHOT_FEATURES,
    )
    manifest = {
        "data_path": str(cfg.data_path),
        "rows_base": int(len(base_df)),
        "unique_dates": int(base_df[DATE_COL].nunique()),
        "unique_tickers": int(base_df["ticker"].nunique()),
        "date_min": str(base_df[DATE_COL].min()),
        "date_max": str(base_df[DATE_COL].max()),
        "target_col": cfg.target_col,
        "ticker_universe_reference_path": str(TICKER_UNIVERSE_REFERENCE_PATH),
        "index_tickers": list(INDEX_TICKERS),
        "all_reit_tickers": list(ALL_REIT_TICKERS),
        "hard_poison_tickers": list(HARD_POISON_TICKERS),
        "caveat_tickers": list(CAVEAT_TICKERS),
        "row_level_exclusions": ROW_LEVEL_EXCLUSIONS,
        "exclude_hard_poison_tickers": EXCLUDE_HARD_POISON_TICKERS,
        "exclude_caveat_tickers": EXCLUDE_CAVEAT_TICKERS,
        "manual_exclude_tickers": list(MANUAL_EXCLUDE_TICKERS),
        "manual_include_tickers": list(MANUAL_INCLUDE_TICKERS),
        "effective_excluded_tickers": effective_excluded_tickers(),
        "effective_reit_universe": effective_reit_universe(),
        "feature_cols": feature_cols_used,
        "feature_groups": ticker_groups,
        "split_mode": "walkforward_by_date",
        "n_splits": N_SPLITS,
        "gap_rows": GAP_ROWS,
        "forward_horizon_rows": cfg.horizon_rows,
        "worker_seed": cfg.worker_seed,
        "beta_window_days": BETA_WINDOW_DAYS,
        "car_lookback_days": CAR_LOOKBACK_DAYS,
        "car_subwindow_days": CAR_SUBWINDOW_DAYS,
        "drop_calendar_by": drop_calendar_by,
        "export_panel_csv": cfg.export_panel_csv,
        "panel_export_path": str(cfg.panel_export_path),
        "run_flags_default": {
            "PER_HORIZON_SEED_OFFSET": PER_HORIZON_SEED_OFFSET,
            "EXPORT_PANEL_CSV": EXPORT_PANEL_CSV,
            "EXCLUDE_HARD_POISON_TICKERS": EXCLUDE_HARD_POISON_TICKERS,
            "EXCLUDE_CAVEAT_TICKERS": EXCLUDE_CAVEAT_TICKERS,
            "ENABLE_TICKER_ONEHOT_FEATURES": ENABLE_TICKER_ONEHOT_FEATURES,
            "ENABLE_EXTENDED_CAR_FEATURES": ENABLE_EXTENDED_CAR_FEATURES,
            "ENABLE_EXTENDED_SORA_FEATURES": ENABLE_EXTENDED_SORA_FEATURES,
            "ENABLE_INTERACTION_FEATURES": ENABLE_INTERACTION_FEATURES,
            "MANUAL_DISABLE_FEATURES": list(MANUAL_DISABLE_FEATURES),
            "MANUAL_ENABLE_FEATURES": list(MANUAL_ENABLE_FEATURES),
            "ENABLE_EARLY_STOPPING": ENABLE_EARLY_STOPPING,
            "EARLY_STOPPING_VALID_FRACTION": EARLY_STOPPING_VALID_FRACTION,
            "EARLY_STOPPING_MIN_VALID_DATES": EARLY_STOPPING_MIN_VALID_DATES,
            "EARLY_STOPPING_MIN_TRAIN_DATES": EARLY_STOPPING_MIN_TRAIN_DATES,
            "EARLY_STOPPING_ROUNDS": EARLY_STOPPING_ROUNDS,
            "WINNER_SELECTION_MODE": WINNER_SELECTION_MODE,
            "OBJECTIVE_SIGN_BALANCE_CENTER": OBJECTIVE_SIGN_BALANCE_CENTER,
            "OBJECTIVE_SIGN_BALANCE_TARGET_BAND": OBJECTIVE_SIGN_BALANCE_TARGET_BAND,
            "OBJECTIVE_SIGN_BALANCE_PENALTY": OBJECTIVE_SIGN_BALANCE_PENALTY,
        },
    }
    save_json(cfg.out_dir / "data_manifest.json", manifest)
    save_json(
        cfg.out_dir / "feature_manifest.json",
        {
            "target_col": cfg.target_col,
            "features": feature_cols_used,
            "feature_groups": ticker_groups,
            "trace_cols": TRACE_COLS,
            "effective_excluded_tickers": effective_excluded_tickers(),
            "effective_reit_universe": effective_reit_universe(),
            "manual_disable_features": list(MANUAL_DISABLE_FEATURES),
            "manual_enable_features": list(MANUAL_ENABLE_FEATURES),
            "winner_selection_mode": WINNER_SELECTION_MODE,
        },
    )


def get_walkforward_date_splits(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_dates = df[DATE_COL].drop_duplicates().sort_values().reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=resolve_walkforward_n_splits(unique_dates), gap=GAP_ROWS)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_date_idx, test_date_idx in splitter.split(unique_dates):
        train_dates = set(unique_dates.iloc[train_date_idx].tolist())
        test_dates = set(unique_dates.iloc[test_date_idx].tolist())
        train_idx = np.flatnonzero(df[DATE_COL].isin(train_dates).to_numpy())
        test_idx = np.flatnonzero(df[DATE_COL].isin(test_dates).to_numpy())
        splits.append((train_idx, test_idx))
    return splits


def build_training_dataset(df: pd.DataFrame, feature_cols_used: List[str], target_col: str) -> pd.DataFrame:
    trace_cols = [col for col in TRACE_COLS if col in df.columns and col not in {DATE_COL, "ticker"}]
    needed_cols = [DATE_COL, "ticker", target_col] + feature_cols_used + trace_cols
    out = df[needed_cols].copy()
    out = out.dropna(subset=[target_col] + feature_cols_used).copy()
    out = out.sort_values([DATE_COL, "ticker"]).reset_index(drop=True)
    return out


def summarize_model_a_missingness(df: pd.DataFrame, feature_cols_used: List[str], target_col: str) -> Dict[str, int]:
    cols = [col for col in [target_col] + feature_cols_used if col in df.columns]
    return {col: int(df[col].isna().sum()) for col in cols}


def resolve_walkforward_n_splits(unique_dates: pd.Series) -> int:
    n_dates = int(len(unique_dates))
    date_index = np.arange(n_dates)
    for candidate in range(N_SPLITS, 1, -1):
        try:
            list(TimeSeriesSplit(n_splits=candidate, gap=GAP_ROWS).split(date_index))
            return candidate
        except ValueError:
            continue
    raise ValueError(
        f"Insufficient unique dates ({n_dates}) for walkforward validation with gap={GAP_ROWS}."
    )


def split_train_holdout_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = df[DATE_COL].drop_duplicates().sort_values().reset_index(drop=True)
    holdout_dates = min(FINAL_HOLDOUT_DATES, max(1, len(unique_dates) // 5))
    if len(unique_dates) <= holdout_dates:
        raise ValueError("Not enough unique dates to create a final holdout window.")
    holdout_date_set = set(unique_dates.iloc[-holdout_dates:].tolist())
    train_df = df[~df[DATE_COL].isin(holdout_date_set)].copy()
    holdout_df = df[df[DATE_COL].isin(holdout_date_set)].copy()
    if train_df.empty or holdout_df.empty:
        raise ValueError("Failed to create non-empty train/holdout splits.")
    return (
        train_df.sort_values([DATE_COL, "ticker"]).reset_index(drop=True),
        holdout_df.sort_values([DATE_COL, "ticker"]).reset_index(drop=True),
    )


def build_base_model(params: Dict) -> XGBRegressor:
    model_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
        "missing": np.nan,
        **params,
    }
    return XGBRegressor(**model_params)


def split_train_for_early_stopping(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = train_df[DATE_COL].drop_duplicates().sort_values().reset_index(drop=True)
    if not ENABLE_EARLY_STOPPING or len(unique_dates) < (EARLY_STOPPING_MIN_TRAIN_DATES + 2):
        return train_df.copy(), pd.DataFrame(columns=train_df.columns)

    valid_dates = max(EARLY_STOPPING_MIN_VALID_DATES, int(np.ceil(len(unique_dates) * EARLY_STOPPING_VALID_FRACTION)))
    valid_dates = min(valid_dates, max(1, len(unique_dates) // 3))
    max_allowed_valid_dates = len(unique_dates) - EARLY_STOPPING_MIN_TRAIN_DATES
    if max_allowed_valid_dates < 1:
        return train_df.copy(), pd.DataFrame(columns=train_df.columns)

    valid_dates = min(valid_dates, max_allowed_valid_dates)
    if valid_dates < 1:
        return train_df.copy(), pd.DataFrame(columns=train_df.columns)

    valid_date_set = set(unique_dates.iloc[-valid_dates:].tolist())
    inner_train_df = train_df[~train_df[DATE_COL].isin(valid_date_set)].copy()
    valid_df = train_df[train_df[DATE_COL].isin(valid_date_set)].copy()
    if inner_train_df.empty or valid_df.empty:
        return train_df.copy(), pd.DataFrame(columns=train_df.columns)

    return (
        inner_train_df.sort_values([DATE_COL, "ticker"]).reset_index(drop=True),
        valid_df.sort_values([DATE_COL, "ticker"]).reset_index(drop=True),
    )


def best_iteration_to_n_estimators(model: XGBRegressor, default_n_estimators: int) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return int(default_n_estimators)
    return max(1, min(int(default_n_estimators), int(best_iteration) + 1))


def fit_model_with_optional_early_stopping(
    train_df: pd.DataFrame,
    params: Dict,
    target_col: str,
    feature_cols_used: List[str],
    refit_on_full_train: bool = True,
) -> Tuple[XGBRegressor, Dict[str, float]]:
    default_n_estimators = int(params.get("n_estimators", 100))
    base_meta = {
        "used_early_stopping": False,
        "best_n_estimators": float(default_n_estimators),
        "early_stopping_train_dates": float(train_df[DATE_COL].nunique()),
        "early_stopping_valid_dates": 0.0,
    }
    inner_train_df, valid_df = split_train_for_early_stopping(train_df)
    if valid_df.empty:
        model = build_base_model(params)
        model.fit(inner_train_df[feature_cols_used], inner_train_df[target_col])
        return model, base_meta

    early_params = dict(params)
    early_params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    early_model = build_base_model(early_params)
    early_model.fit(
        inner_train_df[feature_cols_used],
        inner_train_df[target_col],
        eval_set=[(valid_df[feature_cols_used], valid_df[target_col])],
        verbose=False,
    )
    best_n_estimators = best_iteration_to_n_estimators(early_model, default_n_estimators)
    meta = {
        "used_early_stopping": True,
        "best_n_estimators": float(best_n_estimators),
        "early_stopping_train_dates": float(inner_train_df[DATE_COL].nunique()),
        "early_stopping_valid_dates": float(valid_df[DATE_COL].nunique()),
    }
    if not refit_on_full_train:
        return early_model, meta

    refit_params = dict(params)
    refit_params["n_estimators"] = best_n_estimators
    final_model = build_base_model(refit_params)
    final_model.fit(train_df[feature_cols_used], train_df[target_col])
    return final_model, meta


@dataclass
class WalkForwardResult:
    fold_metrics: pd.DataFrame
    oos_predictions: pd.DataFrame
    summary: Dict[str, float]
    baseline_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_oos_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class HoldoutResult:
    summary: Dict[str, float]
    predictions: pd.DataFrame
    baseline_predictions: pd.DataFrame


def evaluate_walkforward(
    df: pd.DataFrame,
    params: Dict,
    label: str,
    target_col: str,
    signed_target: bool,
    feature_cols_used: List[str],
) -> WalkForwardResult:
    trace_cols = [col for col in TRACE_COLS if col in df.columns and col not in {DATE_COL, "ticker"}]
    date_splits = get_walkforward_date_splits(df)

    fold_rows: List[Dict] = []
    pred_rows: List[pd.DataFrame] = []
    baseline_rows: List[Dict] = []
    baseline_pred_rows: List[pd.DataFrame] = []

    for fold_no, (train_idx, test_idx) in enumerate(date_splits, start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        model, fit_meta = fit_model_with_optional_early_stopping(
            train_df,
            params,
            target_col=target_col,
            feature_cols_used=feature_cols_used,
            refit_on_full_train=True,
        )

        preds = model.predict(test_df[feature_cols_used])
        y_true = test_df[target_col].to_numpy()

        fold_metric = {
            "fold": fold_no,
            "train_start": str(train_df[DATE_COL].min().date()),
            "train_end": str(train_df[DATE_COL].max().date()),
            "test_start": str(test_df[DATE_COL].min().date()),
            "test_end": str(test_df[DATE_COL].max().date()),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_train_dates": int(train_df[DATE_COL].nunique()),
            "n_test_dates": int(test_df[DATE_COL].nunique()),
            "n_train_tickers": int(train_df["ticker"].nunique()),
            "n_test_tickers": int(test_df["ticker"].nunique()),
            "used_early_stopping": bool(fit_meta["used_early_stopping"]),
            "best_n_estimators": int(fit_meta["best_n_estimators"]),
            "early_stopping_train_dates": int(fit_meta["early_stopping_train_dates"]),
            "early_stopping_valid_dates": int(fit_meta["early_stopping_valid_dates"]),
        }
        fold_metric.update(combined_metrics(y_true, preds, signed_target=signed_target))
        fold_metric["target_std"] = float(np.std(y_true))
        fold_metric["pred_std"] = float(np.std(preds))
        fold_metric["pred_std_ratio"] = (
            float(fold_metric["pred_std"] / fold_metric["target_std"]) if fold_metric["target_std"] > 0 else float("nan")
        )
        fold_metric["pred_positive_rate"] = float(np.mean(preds > 0)) if signed_target else float("nan")
        fold_rows.append(fold_metric)

        pred_export = test_df[[DATE_COL, "ticker"] + trace_cols].copy()
        pred_export["fold"] = fold_no
        pred_export["target_col"] = target_col
        pred_export["y_true"] = y_true
        pred_export["y_pred"] = preds
        pred_export["prediction_error"] = preds - y_true
        pred_export["abs_prediction_error"] = np.abs(preds - y_true)
        if signed_target:
            pred_export["y_true_dir"] = (y_true > 0).astype(int)
            pred_export["y_pred_dir"] = (preds > 0).astype(int)
        for col in feature_cols_used:
            pred_export[f"feature__{col}"] = test_df[col].to_numpy()
        pred_rows.append(pred_export)

        mean_baseline_pred = np.full(len(test_df), train_df[target_col].mean())
        zero_baseline_pred = np.zeros(len(test_df))
        baseline_rows.append(
            {
                "fold": fold_no,
                "baseline": "train_mean",
                **combined_metrics(y_true, mean_baseline_pred, signed_target=signed_target),
            }
        )
        baseline_rows.append(
            {
                "fold": fold_no,
                "baseline": "zero_baseline",
                **combined_metrics(y_true, zero_baseline_pred, signed_target=signed_target),
            }
        )
        for baseline_name, baseline_pred in (
            ("train_mean", mean_baseline_pred),
            ("zero_baseline", zero_baseline_pred),
        ):
            base_export = test_df[[DATE_COL, "ticker"] + trace_cols].copy()
            base_export["fold"] = fold_no
            base_export["baseline"] = baseline_name
            base_export["target_col"] = target_col
            base_export["y_true"] = y_true
            base_export["y_pred"] = baseline_pred
            base_export["prediction_error"] = baseline_pred - y_true
            base_export["abs_prediction_error"] = np.abs(baseline_pred - y_true)
            baseline_pred_rows.append(base_export)

    fold_metrics_df = pd.DataFrame(fold_rows)
    summary = {
        "label": label,
        "target_col": target_col,
        "mean_r2": float(fold_metrics_df["r2"].mean()),
        "mean_mse": float(fold_metrics_df["mse"].mean()),
        "mean_rmse": float(fold_metrics_df["rmse"].mean()),
        "mean_mae": float(fold_metrics_df["mae"].mean()),
        "mean_accuracy": float(fold_metrics_df["accuracy"].mean()) if signed_target else float("nan"),
        "mean_precision": float(fold_metrics_df["precision"].mean()) if signed_target else float("nan"),
        "mean_recall": float(fold_metrics_df["recall"].mean()) if signed_target else float("nan"),
        "mean_f1": float(fold_metrics_df["f1"].mean()) if signed_target else float("nan"),
        "mean_auc": float(fold_metrics_df["auc"].mean(skipna=True)) if signed_target else float("nan"),
        "mean_spearman_corr": float(fold_metrics_df["spearman_corr"].mean(skipna=True)),
        "mean_top_decile_hit_rate": float(fold_metrics_df["top_decile_hit_rate"].mean(skipna=True)),
        "mean_bottom_decile_hit_rate": float(fold_metrics_df["bottom_decile_hit_rate"].mean(skipna=True)),
        "std_rmse": float(fold_metrics_df["rmse"].std(ddof=1)) if len(fold_metrics_df) > 1 else float("nan"),
        "std_r2": float(fold_metrics_df["r2"].std(ddof=1)) if len(fold_metrics_df) > 1 else float("nan"),
        "min_r2": float(fold_metrics_df["r2"].min()),
        "max_r2": float(fold_metrics_df["r2"].max()),
        "min_rmse": float(fold_metrics_df["rmse"].min()),
        "max_rmse": float(fold_metrics_df["rmse"].max()),
        "mean_target_std": float(fold_metrics_df["target_std"].mean()),
        "mean_pred_std": float(fold_metrics_df["pred_std"].mean()),
        "mean_pred_std_ratio": float(fold_metrics_df["pred_std_ratio"].mean(skipna=True)),
        "mean_pred_positive_rate": float(fold_metrics_df["pred_positive_rate"].mean()) if signed_target else float("nan"),
        "mean_best_n_estimators": float(fold_metrics_df["best_n_estimators"].mean()),
        "n_folds": int(len(fold_metrics_df)),
        "gamma": float(params.get("gamma", np.nan)),
    }

    result = WalkForwardResult(
        fold_metrics=fold_metrics_df,
        oos_predictions=pd.concat(pred_rows, ignore_index=True),
        summary=summary,
        baseline_metrics=pd.DataFrame(baseline_rows),
        baseline_oos_predictions=pd.concat(baseline_pred_rows, ignore_index=True),
    )
    return result


def evaluate_holdout(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    params: Dict,
    label: str,
    target_col: str,
    signed_target: bool,
    feature_cols_used: List[str],
) -> HoldoutResult:
    trace_cols = [col for col in TRACE_COLS if col in holdout_df.columns and col not in {DATE_COL, "ticker"}]
    model, fit_meta = fit_model_with_optional_early_stopping(
        train_df,
        params,
        target_col=target_col,
        feature_cols_used=feature_cols_used,
        refit_on_full_train=True,
    )

    preds = model.predict(holdout_df[feature_cols_used])
    y_true = holdout_df[target_col].to_numpy()
    metrics = combined_metrics(y_true, preds, signed_target=signed_target)
    target_std = float(np.std(y_true))
    pred_std = float(np.std(preds))
    summary = {
        "label": label,
        "target_col": target_col,
        "holdout_start": str(holdout_df[DATE_COL].min().date()),
        "holdout_end": str(holdout_df[DATE_COL].max().date()),
        "n_holdout_rows": int(len(holdout_df)),
        "n_holdout_dates": int(holdout_df[DATE_COL].nunique()),
        "n_holdout_tickers": int(holdout_df["ticker"].nunique()),
        "used_early_stopping": bool(fit_meta["used_early_stopping"]),
        "best_n_estimators": int(fit_meta["best_n_estimators"]),
        "early_stopping_train_dates": int(fit_meta["early_stopping_train_dates"]),
        "early_stopping_valid_dates": int(fit_meta["early_stopping_valid_dates"]),
        **metrics,
        "mean_r2": float(metrics["r2"]),
        "mean_mse": float(metrics["mse"]),
        "mean_rmse": float(metrics["rmse"]),
        "mean_mae": float(metrics["mae"]),
        "mean_accuracy": float(metrics["accuracy"]) if signed_target else float("nan"),
        "mean_precision": float(metrics["precision"]) if signed_target else float("nan"),
        "mean_recall": float(metrics["recall"]) if signed_target else float("nan"),
        "mean_f1": float(metrics["f1"]) if signed_target else float("nan"),
        "mean_auc": float(metrics["auc"]) if signed_target else float("nan"),
        "mean_spearman_corr": float(metrics["spearman_corr"]),
        "mean_top_decile_hit_rate": float(metrics["top_decile_hit_rate"]),
        "mean_bottom_decile_hit_rate": float(metrics["bottom_decile_hit_rate"]),
        "target_std": target_std,
        "mean_target_std": target_std,
        "pred_std": pred_std,
        "mean_pred_std": pred_std,
        "pred_std_ratio": float(pred_std / target_std) if target_std > 0 else float("nan"),
        "mean_pred_std_ratio": float(pred_std / target_std) if target_std > 0 else float("nan"),
        "pred_positive_rate": float(np.mean(preds > 0)) if signed_target else float("nan"),
        "mean_pred_positive_rate": float(np.mean(preds > 0)) if signed_target else float("nan"),
    }

    pred_export = holdout_df[[DATE_COL, "ticker"] + trace_cols].copy()
    pred_export["target_col"] = target_col
    pred_export["y_true"] = y_true
    pred_export["y_pred"] = preds
    pred_export["prediction_error"] = preds - y_true
    pred_export["abs_prediction_error"] = np.abs(preds - y_true)
    if signed_target:
        pred_export["y_true_dir"] = (y_true > 0).astype(int)
        pred_export["y_pred_dir"] = (preds > 0).astype(int)
    for col in feature_cols_used:
        pred_export[f"feature__{col}"] = holdout_df[col].to_numpy()

    baseline_exports = []
    for baseline_name, baseline_pred in (
        ("train_mean", np.full(len(holdout_df), train_df[target_col].mean())),
        ("zero_baseline", np.zeros(len(holdout_df))),
    ):
        base_export = holdout_df[[DATE_COL, "ticker"] + trace_cols].copy()
        base_export["baseline"] = baseline_name
        base_export["target_col"] = target_col
        base_export["y_true"] = y_true
        base_export["y_pred"] = baseline_pred
        base_export["prediction_error"] = baseline_pred - y_true
        base_export["abs_prediction_error"] = np.abs(baseline_pred - y_true)
        baseline_exports.append(base_export)

    return HoldoutResult(
        summary=summary,
        predictions=pred_export,
        baseline_predictions=pd.concat(baseline_exports, ignore_index=True),
    )


def objective_with_penalty(summary: Dict[str, float], signed_target: bool) -> float:
    score = summary["mean_rmse"]

    mean_r2 = summary["mean_r2"]
    if mean_r2 < 0:
        score += abs(mean_r2) * OBJECTIVE_NEGATIVE_R2_PENALTY
    else:
        score -= mean_r2 * OBJECTIVE_POSITIVE_R2_REWARD

    pred_std_ratio = summary.get("mean_pred_std_ratio", float("nan"))
    if np.isfinite(pred_std_ratio) and pred_std_ratio < OBJECTIVE_PRED_STD_RATIO_FLOOR:
        score += (OBJECTIVE_PRED_STD_RATIO_FLOOR - pred_std_ratio) * OBJECTIVE_PRED_STD_RATIO_PENALTY

    if signed_target:
        pos_rate = summary["mean_pred_positive_rate"]
        if pos_rate < OBJECTIVE_SIGN_BALANCE_FLOOR:
            score += (OBJECTIVE_SIGN_BALANCE_FLOOR - pos_rate) * 2.0
        if pos_rate > 1.0 - OBJECTIVE_SIGN_BALANCE_FLOOR:
            score += (pos_rate - (1.0 - OBJECTIVE_SIGN_BALANCE_FLOOR)) * 2.0
        sign_balance_deviation = abs(pos_rate - OBJECTIVE_SIGN_BALANCE_CENTER)
        excess_deviation = max(0.0, sign_balance_deviation - OBJECTIVE_SIGN_BALANCE_TARGET_BAND)
        score += excess_deviation * OBJECTIVE_SIGN_BALANCE_PENALTY
        if pos_rate > 0.65:
            score += (pos_rate - 0.65) * (OBJECTIVE_SIGN_BALANCE_PENALTY * 2.0)

        mean_auc = summary.get("mean_auc", float("nan"))
        if np.isfinite(mean_auc):
            score -= max(0.0, mean_auc - 0.50) * 0.02

        mean_spearman = summary.get("mean_spearman_corr", float("nan"))
        if np.isfinite(mean_spearman):
            score -= max(0.0, mean_spearman) * 0.01

    return float(score)


def optuna_objective(
    trial,
    df: pd.DataFrame,
    target_col: str,
    signed_target: bool,
    feature_cols_used: List[str],
) -> float:
    params = {
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "n_estimators": trial.suggest_int("n_estimators", 300, 2200, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }
    result = evaluate_walkforward(
        df,
        params,
        label="optuna_trial",
        target_col=target_col,
        signed_target=signed_target,
        feature_cols_used=feature_cols_used,
    )
    penalized_score = objective_with_penalty(result.summary, signed_target=signed_target)
    trial.set_user_attr("mean_r2", result.summary["mean_r2"])
    trial.set_user_attr("mean_rmse", result.summary["mean_rmse"])
    trial.set_user_attr("mean_mae", result.summary["mean_mae"])
    trial.set_user_attr("mean_f1", result.summary["mean_f1"])
    trial.set_user_attr("penalized_score", penalized_score)
    return penalized_score


def run_optuna_search(
    df: pd.DataFrame,
    target_key: str,
    target_col: str,
    signed_target: bool,
    feature_cols_used: List[str],
    out_dir: Path,
) -> Dict:
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: optuna_objective(
            trial, df, target_col=target_col, signed_target=signed_target, feature_cols_used=feature_cols_used
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
    save_json(out_dir / f"{target_key}_optuna_best_params.json", payload)
    return study.best_params


def run_deap_search(
    df: pd.DataFrame,
    target_key: str,
    target_col: str,
    signed_target: bool,
    feature_cols_used: List[str],
    out_dir: Path,
) -> Dict:
    search_space = {
        "gamma": [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0],
        "n_estimators": [300, 500, 800, 1200, 1600, 2000, 2200],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
        "min_child_weight": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0],
        "reg_lambda": [0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0],
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
        lambda: [make_gene(i) for i in range(len(param_names))],
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
        result = evaluate_walkforward(
            df,
            params,
            label="deap_trial",
            target_col=target_col,
            signed_target=signed_target,
            feature_cols_used=feature_cols_used,
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
    print_deap_generation_progress(
        target_key=target_key,
        generation=0,
        total_generations=DEAP_GENERATIONS,
        population=population,
        hall_of_fame=hall_of_fame,
    )

    for generation in range(DEAP_GENERATIONS):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < DEAP_CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
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
        print_deap_generation_progress(
            target_key=target_key,
            generation=generation + 1,
            total_generations=DEAP_GENERATIONS,
            population=population,
            hall_of_fame=hall_of_fame,
        )

    best_individual = hall_of_fame[0]
    best_params = decode(best_individual)
    best_penalized_score = float(best_individual.fitness.values[0])
    payload = {
        "target_key": target_key,
        "target_col": target_col,
        "best_score_penalized_objective": best_penalized_score,
        "best_params": best_params,
    }
    save_json(out_dir / f"{target_key}_deap_best_params.json", payload)
    return best_params


def export_walkforward_results(result: WalkForwardResult, target_key: str, optimizer_name: str, out_dir: Path) -> None:
    prefix = f"{target_key}_{optimizer_name}"
    result.fold_metrics.to_csv(out_dir / f"{prefix}_walkforward_fold_metrics.csv", index=False)
    result.oos_predictions.to_csv(out_dir / f"{prefix}_walkforward_oos_predictions.csv", index=False)
    result.baseline_metrics.to_csv(out_dir / f"{prefix}_baseline_walkforward_fold_metrics.csv", index=False)
    result.baseline_oos_predictions.to_csv(out_dir / f"{prefix}_baseline_walkforward_oos_predictions.csv", index=False)
    save_json(out_dir / f"{prefix}_walkforward_summary.json", result.summary)


def export_holdout_results(result: HoldoutResult, target_key: str, optimizer_name: str, out_dir: Path) -> None:
    prefix = f"{target_key}_{optimizer_name}"
    result.predictions.to_csv(out_dir / f"{prefix}_holdout_predictions.csv", index=False)
    result.baseline_predictions.to_csv(out_dir / f"{prefix}_holdout_baseline_predictions.csv", index=False)
    save_json(out_dir / f"{prefix}_holdout_summary.json", result.summary)


def choose_winner(optuna_result: WalkForwardResult, deap_result: WalkForwardResult, signed_target: bool) -> str:
    return choose_winner_from_summaries(optuna_result.summary, deap_result.summary, signed_target=signed_target)


def choose_winner_from_summaries(o: Dict[str, float], d: Dict[str, float], signed_target: bool) -> str:
    if signed_target and WINNER_SELECTION_MODE == "directional":
        auc_diff = o["mean_auc"] - d["mean_auc"]
        if auc_diff > AUC_TOLERANCE:
            return "optuna"
        if auc_diff < -AUC_TOLERANCE:
            return "deap"

        spearman_diff = o.get("mean_spearman_corr", float("nan")) - d.get("mean_spearman_corr", float("nan"))
        if np.isfinite(spearman_diff):
            if spearman_diff > 0.01:
                return "optuna"
            if spearman_diff < -0.01:
                return "deap"

        optuna_pos_dev = abs(o["mean_pred_positive_rate"] - 0.50)
        deap_pos_dev = abs(d["mean_pred_positive_rate"] - 0.50)
        pos_dev_diff = optuna_pos_dev - deap_pos_dev
        if pos_dev_diff > PRED_POSITIVE_RATE_TOLERANCE:
            return "deap"
        if pos_dev_diff < -PRED_POSITIVE_RATE_TOLERANCE:
            return "optuna"

        return "optuna" if o["mean_f1"] >= d["mean_f1"] else "deap"

    if signed_target:
        rmse_diff = o["mean_rmse"] - d["mean_rmse"]
        if rmse_diff > RMSE_TOLERANCE:
            return "deap"
        if rmse_diff < -RMSE_TOLERANCE:
            return "optuna"

        r2_diff = o["mean_r2"] - d["mean_r2"]
        if r2_diff > R2_TOLERANCE:
            return "optuna"
        if r2_diff < -R2_TOLERANCE:
            return "deap"

        auc_diff = o["mean_auc"] - d["mean_auc"]
        if auc_diff > AUC_TOLERANCE:
            return "optuna"
        if auc_diff < -AUC_TOLERANCE:
            return "deap"

        return "optuna" if o["mean_f1"] >= d["mean_f1"] else "deap"

    rmse_diff = o["mean_rmse"] - d["mean_rmse"]
    if rmse_diff < -RMSE_TOLERANCE:
        return "optuna"
    if rmse_diff > RMSE_TOLERANCE:
        return "deap"
    return "optuna" if o["mean_r2"] >= d["mean_r2"] else "deap"


def fit_final_model(df: pd.DataFrame, params: Dict, target_col: str, feature_cols_used: List[str]) -> XGBRegressor:
    model, _ = fit_model_with_optional_early_stopping(
        df,
        params,
        target_col=target_col,
        feature_cols_used=feature_cols_used,
        refit_on_full_train=True,
    )
    return model


def run_shap(
    final_model: XGBRegressor,
    oos_predictions: pd.DataFrame,
    target_col: str,
    target_key: str,
    feature_cols_used: List[str],
    out_dir: Path,
) -> None:
    feature_export_cols = [f"feature__{col}" for col in feature_cols_used]
    missing_feature_cols = [col for col in feature_export_cols if col not in oos_predictions.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing OOS feature columns for SHAP export: {missing_feature_cols}")

    shap_source = oos_predictions[[DATE_COL, "ticker", "y_true"] + feature_export_cols].copy()
    if len(shap_source) > SHAP_MAX_ROWS:
        shap_source = shap_source.sample(n=SHAP_MAX_ROWS, random_state=SEED)
    shap_source = shap_source.sort_values([DATE_COL, "ticker"]).reset_index(drop=True)
    X_shap = shap_source[feature_export_cols].rename(columns=dict(zip(feature_export_cols, feature_cols_used)))

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_shap)

    shap_values_df = pd.DataFrame(shap_values, columns=feature_cols_used)
    shap_values_df.insert(0, "ticker", shap_source["ticker"].astype(str))
    shap_values_df.insert(0, DATE_COL, shap_source[DATE_COL].astype(str))
    shap_values_df["y_true"] = shap_source["y_true"].to_numpy()
    shap_values_df.to_csv(out_dir / f"{target_key}_shap_summary_values.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{target_key}_shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{target_key}_shap_summary_bar.png", dpi=160, bbox_inches="tight")
    plt.close()


def write_run_contents_summary(
    cfg: HorizonConfig,
    feature_cols_used: List[str],
    drop_calendar_by: str,
) -> None:
    fg = feature_groups(
        effective_reit_universe(),
        include_ticker_identity=ENABLE_TICKER_ONEHOT_FEATURES,
    )
    panel_export_line = (
        f"Panel export: {cfg.panel_export_path}"
        if cfg.export_panel_csv
        else f"Panel export: disabled; would otherwise be {cfg.panel_export_path}"
    )
    lines = [
        "train_a_multifold_pipeline.py output summary",
        "",
        f"Dataset: {cfg.data_path}",
        panel_export_line,
        f"Forward horizon rows: {cfg.horizon_rows}",
        f"Worker seed: {cfg.worker_seed}",
        f"Drop calendar by: {drop_calendar_by}",
        f"Output directory: {cfg.out_dir}",
        f"Validation: walkforward by date, TimeSeriesSplit n_splits={N_SPLITS} gap_rows={GAP_ROWS}",
        f"Final holdout dates: {FINAL_HOLDOUT_DATES}",
        "",
        "Model A ticker-universe controls:",
        f"  all_reit_tickers={list(ALL_REIT_TICKERS)}",
        f"  index_tickers={list(INDEX_TICKERS)}",
        f"  hard_poison_tickers={list(HARD_POISON_TICKERS)}",
        f"  caveat_tickers={list(CAVEAT_TICKERS)}",
        f"  row_level_exclusions={ROW_LEVEL_EXCLUSIONS}",
        f"  exclude_hard_poison_tickers={EXCLUDE_HARD_POISON_TICKERS}",
        f"  exclude_caveat_tickers={EXCLUDE_CAVEAT_TICKERS}",
        f"  manual_exclude_tickers={list(MANUAL_EXCLUDE_TICKERS)}",
        f"  manual_include_tickers={list(MANUAL_INCLUDE_TICKERS)}",
        f"  enable_ticker_onehot_features={ENABLE_TICKER_ONEHOT_FEATURES}",
        f"  enable_extended_car_features={ENABLE_EXTENDED_CAR_FEATURES}",
        f"  enable_extended_sora_features={ENABLE_EXTENDED_SORA_FEATURES}",
        f"  enable_interaction_features={ENABLE_INTERACTION_FEATURES}",
        f"  manual_disable_features={list(MANUAL_DISABLE_FEATURES)}",
        f"  manual_enable_features={list(MANUAL_ENABLE_FEATURES)}",
        f"  effective_excluded_tickers={effective_excluded_tickers()}",
        f"  effective_reit_universe={effective_reit_universe()}",
        "",
        f"Target: {cfg.target_col}",
        f"Winner selection mode: {WINNER_SELECTION_MODE}",
        f"Early stopping enabled: {ENABLE_EARLY_STOPPING}",
        f"Early stopping valid fraction: {EARLY_STOPPING_VALID_FRACTION}",
        f"Early stopping min valid dates: {EARLY_STOPPING_MIN_VALID_DATES}",
        f"Early stopping min train dates: {EARLY_STOPPING_MIN_TRAIN_DATES}",
        f"Early stopping rounds: {EARLY_STOPPING_ROUNDS}",
        f"Objective sign balance center: {OBJECTIVE_SIGN_BALANCE_CENTER}",
        f"Objective sign balance target band: {OBJECTIVE_SIGN_BALANCE_TARGET_BAND}",
        f"Objective sign balance penalty: {OBJECTIVE_SIGN_BALANCE_PENALTY}",
        "",
        "Abnormal-history feature cols:",
    ] + [f"  - {col}" for col in fg["abnormal_history"]]
    lines += [
        "",
        "Macro base feature cols:",
    ] + [f"  - {col}" for col in fg["macro_base"]]
    lines += [
        "",
        "Extended abnormal-history feature cols:",
    ] + [f"  - {col}" for col in fg.get("abnormal_history_extended", [])]
    lines += [
        "",
        "Extended macro feature cols:",
    ] + [f"  - {col}" for col in fg.get("macro_extended", [])]
    lines += [
        "",
        "REIT sector-state feature cols:",
    ] + [f"  - {col}" for col in fg["reit_sector_state"]]
    lines += [
        "",
        "Interaction feature cols:",
    ] + [f"  - {col}" for col in fg.get("interaction_terms", [])]
    lines += [
        "",
        "Ticker identity feature cols:",
    ] + [f"  - {col}" for col in fg["ticker_identity"]]
    if "manual_overrides" in fg:
        lines += [
            "",
            "Manual override feature cols:",
        ] + [f"  - {col}" for col in fg["manual_overrides"]]
    lines += [
        "",
        f"Effective feature cols (n={len(feature_cols_used)}):",
    ] + [f"  - {col}" for col in feature_cols_used]
    lines += [
        "",
        "Output naming pattern:",
        "  model_a_optuna_best_params.json",
        "  model_a_deap_best_params.json",
        "  model_a_optuna_walkforward_fold_metrics.csv",
        "  model_a_deap_walkforward_fold_metrics.csv",
        "  model_a_optuna_walkforward_oos_predictions.csv",
        "  model_a_deap_walkforward_oos_predictions.csv",
        "  model_a_optuna_baseline_walkforward_fold_metrics.csv",
        "  model_a_deap_baseline_walkforward_fold_metrics.csv",
        "  model_a_optuna_baseline_walkforward_oos_predictions.csv",
        "  model_a_deap_baseline_walkforward_oos_predictions.csv",
        "  model_a_optuna_walkforward_summary.json",
        "  model_a_deap_walkforward_summary.json",
        "  model_a_final_model_xgb.json",
        "  model_a_shap_summary_values.csv",
        "  model_a_shap_summary_beeswarm.png",
        "  model_a_shap_summary_bar.png",
        "",
        "Run-level manifest files:",
        "  data_manifest.json",
        "  feature_manifest.json",
        "Cross-run consolidated files:",
        "  optimizer_comparison.json",
        "  final_selected_params.json",
        "  run_contents_summary.txt",
    ]
    (cfg.out_dir / "run_contents_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def normalize_horizons(horizons: List[int]) -> List[int]:
    normalized = sorted(set(int(h) for h in horizons))
    for horizon in normalized:
        if horizon < 1 or horizon > 63:
            raise ValueError(f"Invalid horizon {horizon}. Expected an integer between 1 and 63.")
    return normalized


def run_single_horizon_pipeline(
    run_root: Path,
    horizon_rows: int,
    drop_calendar_by: str,
    selected_horizons: List[int],
) -> Dict[str, str]:
    worker_seed = compute_worker_seed(horizon_rows=horizon_rows, selected_horizons=selected_horizons)
    set_global_seed(worker_seed)
    cfg = HorizonConfig(horizon_rows=horizon_rows, run_root=run_root, worker_seed=worker_seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Horizon {horizon_rows} rows -> {cfg.out_dir} ===")
    print("Building pooled Model A panel dataset ...")
    panel_df, feature_cols_used = build_model_a_panel_dataset(cfg=cfg, drop_calendar_by=drop_calendar_by)

    print("Building training dataset ...")
    df = build_training_dataset(panel_df, feature_cols_used=feature_cols_used, target_col=cfg.target_col)
    print(f"Effective universe: {effective_reit_universe()}")
    print(f"Usable rows: {len(df):,}")
    print(f"Usable dates: {df[DATE_COL].nunique():,}")
    print(f"Usable tickers: {df['ticker'].nunique():,}")
    if df.empty:
        missingness = summarize_model_a_missingness(panel_df, feature_cols_used=feature_cols_used, target_col=cfg.target_col)
        print("Training dataset is empty after dropping rows with incomplete features/target.")
        print("Panel missingness summary:")
        for col, null_count in missingness.items():
            print(f"  {col}: {null_count:,} nulls")
        write_data_manifests(cfg, panel_df, feature_cols_used=feature_cols_used, drop_calendar_by=drop_calendar_by)
        raise ValueError(
            "Model A training dataset is empty after feature/target filtering. "
            "Review the panel missingness summary above."
        )

    write_data_manifests(cfg, df, feature_cols_used=feature_cols_used, drop_calendar_by=drop_calendar_by)
    print(f"Window: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    train_df, holdout_df = split_train_holdout_by_date(df)
    resolved_splits = resolve_walkforward_n_splits(train_df[DATE_COL].drop_duplicates().sort_values().reset_index(drop=True))
    print(
        f"Walkforward validation: n_splits={resolved_splits} gap={GAP_ROWS} "
        f"(tuning window ends {train_df[DATE_COL].max().date()})"
    )
    print(
        f"Final holdout: {holdout_df[DATE_COL].min().date()} -> {holdout_df[DATE_COL].max().date()} "
        f"({holdout_df[DATE_COL].nunique()} dates)"
    )

    print("Running Optuna search ...")
    optuna_params = run_optuna_search(
        train_df,
        target_key=TARGET_KEY,
        target_col=cfg.target_col,
        signed_target=TARGET_SIGNED,
        feature_cols_used=feature_cols_used,
        out_dir=cfg.out_dir,
    )
    optuna_eval = evaluate_walkforward(
        train_df,
        optuna_params,
        label="optuna",
        target_col=cfg.target_col,
        signed_target=TARGET_SIGNED,
        feature_cols_used=feature_cols_used,
    )
    export_walkforward_results(optuna_eval, target_key=TARGET_KEY, optimizer_name="optuna", out_dir=cfg.out_dir)
    optuna_holdout = evaluate_holdout(
        train_df,
        holdout_df,
        optuna_params,
        label="optuna_holdout",
        target_col=cfg.target_col,
        signed_target=TARGET_SIGNED,
        feature_cols_used=feature_cols_used,
    )
    export_holdout_results(optuna_holdout, target_key=TARGET_KEY, optimizer_name="optuna", out_dir=cfg.out_dir)

    print("Running DEAP search ...")
    deap_params = run_deap_search(
        train_df,
        target_key=TARGET_KEY,
        target_col=cfg.target_col,
        signed_target=TARGET_SIGNED,
        feature_cols_used=feature_cols_used,
        out_dir=cfg.out_dir,
    )
    deap_eval = evaluate_walkforward(
        train_df,
        deap_params,
        label="deap",
        target_col=cfg.target_col,
        signed_target=TARGET_SIGNED,
        feature_cols_used=feature_cols_used,
    )
    export_walkforward_results(deap_eval, target_key=TARGET_KEY, optimizer_name="deap", out_dir=cfg.out_dir)
    deap_holdout = evaluate_holdout(
        train_df,
        holdout_df,
        deap_params,
        label="deap_holdout",
        target_col=cfg.target_col,
        signed_target=TARGET_SIGNED,
        feature_cols_used=feature_cols_used,
    )
    export_holdout_results(deap_holdout, target_key=TARGET_KEY, optimizer_name="deap", out_dir=cfg.out_dir)

    winner = choose_winner_from_summaries(optuna_holdout.summary, deap_holdout.summary, signed_target=TARGET_SIGNED)
    final_params = optuna_params if winner == "optuna" else deap_params
    winner_oos_predictions = optuna_eval.oos_predictions if winner == "optuna" else deap_eval.oos_predictions

    comparison = {
        "target_col": cfg.target_col,
        "winner": winner,
        "optuna_walkforward_summary": optuna_eval.summary,
        "deap_walkforward_summary": deap_eval.summary,
        "optuna_holdout_summary": optuna_holdout.summary,
        "deap_holdout_summary": deap_holdout.summary,
        "winner_params": final_params,
    }
    save_json(cfg.out_dir / "optimizer_comparison.json", comparison)
    save_json(cfg.out_dir / "final_selected_params.json", final_params)

    final_model = fit_final_model(
        df,
        final_params,
        target_col=cfg.target_col,
        feature_cols_used=feature_cols_used,
    )
    final_model.save_model(str(cfg.out_dir / "model_a_final_model_xgb.json"))
    run_shap(
        final_model,
        winner_oos_predictions,
        target_col=cfg.target_col,
        target_key=TARGET_KEY,
        feature_cols_used=feature_cols_used,
        out_dir=cfg.out_dir,
    )
    write_run_contents_summary(cfg, feature_cols_used=feature_cols_used, drop_calendar_by=drop_calendar_by)
    print("\nDone. Outputs written to:")
    print(cfg.out_dir)
    return {"horizon_rows": str(horizon_rows), "out_dir": str(cfg.out_dir)}


def main() -> None:
    set_global_seed(SEED)
    args = parse_args()
    horizons = normalize_horizons(args.horizons)
    run_root = create_run_root()
    print(f"Run root: {run_root}")
    print(f"Horizons: {horizons}")
    print(f"Effective universe: {effective_reit_universe()}")

    if len(horizons) == 1:
        run_single_horizon_pipeline(
            run_root=run_root,
            horizon_rows=horizons[0],
            drop_calendar_by=args.drop_calendar_by,
            selected_horizons=horizons,
        )
        return

    max_workers = max(1, min(int(args.max_horizon_workers), len(horizons)))
    print(f"Running horizons in parallel with max_workers={max_workers}")

    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for horizon_rows in horizons:
            future = executor.submit(
                run_single_horizon_pipeline,
                run_root,
                horizon_rows,
                args.drop_calendar_by,
                horizons,
            )
            futures[future] = horizon_rows

        for future in as_completed(futures):
            horizon_rows = futures[future]
            result = future.result()
            print(f"Completed horizon {horizon_rows}: {result['out_dir']}")


if __name__ == "__main__":
    main()
