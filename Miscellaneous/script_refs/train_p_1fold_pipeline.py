r"""
train_p_1fold_pipeline.py
=========================

Single-holdout Model P pipeline script.

Original script sources:
    - train_p_1fold.py
    - local SORA / SGX join logic now bundled inside this script

Purpose:
    Preserve the existing Model P training pipeline and parameters exactly as in
    train_p_1fold.py, but move the upstream SORA / S-REIT join and 21d target
    construction into this script so the joined CSV is produced in-process
    before training.

Forward-horizon behavior:
    - The script supports one or more forward SGX-row horizons.
    - Default behavior is a single horizon: 21 rows ahead.
    - Multi-horizon runs write each horizon to its own subfolder:
        out/run_<n>/fwd_<horizon>_days/
    - For each horizon, the joined CSV and all training artifacts are written
      into that horizon-specific folder.
    - Future target columns are horizon-specific, for example:
        * sora_fwd_1d_level
        * sora_fwd_5d_change
        * sora_fwd_21d_abs_change
    - The REIT forward-return column is also horizon-specific, for example:
        * reit_index_fwd_21d_return

Targets trained in one run:
    Option 1: future SORA level
        - sora_fwd_<horizon>d_level

    Option 2: future SORA change
        - sora_fwd_<horizon>d_change

    Option 3: absolute magnitude of future SORA change
        - sora_fwd_<horizon>d_abs_change

    You can run any subset. Defaults are set by RUN_TARGET_OPTION* below; override on the
    command line with --all-targets or --option1 / --no-option1, etc.

Default Option behavior:
    - Option 1 is OFF by default.
    - Option 2 is ON by default.
    - Option 3 is OFF by default.
    - This preserves the historical default behavior of train_p_1fold.py.

Preprocessing behavior copied from join_sora_to_xgb.py:
    - filter xgb_ready down to a chosen row-universe calendar
    - append REIT index OHLC values
    - shift MAS SORA inputs to fixed T-2 business-day point-in-time-safe values
    - forward-fill SORA across weekends / holidays after calendar expansion
    - derive sora_90d_change_t2 and sora_term_spread_t2
    - derive reit_index_fwd_21d_return
    - derive future SORA targets from the realized SORA path:
        * sora_fwd_21d_level
        * sora_fwd_21d_change
        * sora_fwd_21d_abs_change
    - export the joined CSV for visibility, then train using that in-memory dataset

Drop-row switch:
    - default: drop rows not in the S-REIT index trading calendar (base logic unchanged)
    - optional: drop rows not in the realized SORA publication calendar

Base-seed behavior:
    - The script uses a shared base seed constant: SEED.
    - In a single-horizon run, the worker seed remains exactly SEED.
    - In a multi-horizon run, the worker seed can be offset by horizon:
        worker_seed = SEED + horizon_rows
      when PER_HORIZON_SEED_OFFSET is enabled.
    - This keeps single-horizon runs aligned with the baseline seed path while
      allowing multi-horizon runs to use deterministic but distinct RNG streams.
    - The chosen worker seed for each horizon is recorded in that horizon's
      manifest and run summary outputs.

Output location:
    Consolidated/IO/Model_Train/train_p_1fold_pipeline/run_<n>/  (n = 0, 1, 2, ... new folder each run)

This script also writes:
    - fwd_<horizon>_days/sora_joined_to_xgb_pipeline.csv
    - fwd_<horizon>_days/run_contents_summary.txt
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
DEFAULT_FORWARD_HORIZON_ROWS_LIST = (10,15) # (3,7,10,15) (1,5,10,21)
TRAIN_FRAC = 0.70
TEST_FRAC = 0.20
PER_HORIZON_SEED_OFFSET = True
EXPORT_JOINED_CSV = True

# Winner selection tolerances.
# Differences smaller than these thresholds are treated as "effectively tied"
# to reduce noisy winner flips on a single holdout window.
AUC_TOLERANCE = 0.005
RMSE_TOLERANCE = 0.002

SCRIPT_DIR = Path(__file__).resolve().parent
CONSOLIDATED_ROOT = SCRIPT_DIR.parents[1]
IO_SRC_DIR = CONSOLIDATED_ROOT / "IO" / "SRC"
FED_OUTPUT_DIR = IO_SRC_DIR / "CSV_FED" / "Output"
MAS_OUTPUT_DIR = IO_SRC_DIR / "CSV_MAS" / "Output"
MODEL_DIR = IO_SRC_DIR / "MODEL"
CSV_TICKER_DIR = IO_SRC_DIR / "CSV_TICKER"
MODEL_TRAIN_DIR = CONSOLIDATED_ROOT / "IO" / "Model_Train"

XGB_READY_PATH = FED_OUTPUT_DIR / "timeseries_2022-07-27_2026-03-18_xgb_ready.csv"
SGX_REIT_INDEX_PATH = CSV_TICKER_DIR / "SGX_DLY_REIT, 1D.csv"
SORA_DAILY_PATH = MAS_OUTPUT_DIR / "sora_daily.csv"
SORA_3M_DAILY_PATH = MAS_OUTPUT_DIR / "sora_3m_daily.csv"

RUNS_ROOT = MODEL_TRAIN_DIR / Path(__file__).stem
RUNS_ROOT.mkdir(exist_ok=True)
MAX_HORIZON_WORKERS = 2

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

# sora_fwd_*_level aligns with realized level at t+forward_horizon; these features are ~the same information.
OPTION1_LEVEL_EXCLUDED_FEATURES = ("sora_level_t2", "sora_curve_steepness")

# Change target: level at T-2 has weak partial corr but large train/test distribution shift.
OPTION2_CHANGE_EXCLUDED_FEATURES = ("sora_level_t2", "sora_3m_t2")


def feature_cols_for_target(target_key: str) -> List[str]:
    if target_key == "option1_level":
        return [c for c in FEATURE_COLS if c not in OPTION1_LEVEL_EXCLUDED_FEATURES]
    if target_key == "option2_change":
        return [c for c in FEATURE_COLS if c not in OPTION2_CHANGE_EXCLUDED_FEATURES]
    return list(FEATURE_COLS)


def forward_suffix(rows: int) -> str:
    return f"fwd_{rows}d"


def reit_fwd_return_col(rows: int) -> str:
    return f"reit_index_{forward_suffix(rows)}_return"


def sora_fwd_level_col(rows: int) -> str:
    return f"sora_{forward_suffix(rows)}_level"


def sora_fwd_change_col(rows: int) -> str:
    return f"sora_{forward_suffix(rows)}_change"


def sora_fwd_abs_change_col(rows: int) -> str:
    return f"sora_{forward_suffix(rows)}_abs_change"


def build_target_specs(horizon_rows: int) -> Dict[str, Dict[str, str]]:
    return {
        "option1_level": {
            "target_col": sora_fwd_level_col(horizon_rows),
            "description": f"Future SORA level {horizon_rows} SGX trading rows ahead",
        },
        "option2_change": {
            "target_col": sora_fwd_change_col(horizon_rows),
            "description": f"Future SORA change over {horizon_rows} SGX trading rows",
        },
        "option3_abs_change": {
            "target_col": sora_fwd_abs_change_col(horizon_rows),
            "description": f"Absolute magnitude of future SORA change over {horizon_rows} SGX trading rows",
        },
    }


def horizon_folder_name(horizon_rows: int) -> str:
    return f"fwd_{horizon_rows}_days"


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
    export_joined_csv: bool = EXPORT_JOINED_CSV

    @property
    def out_dir(self) -> Path:
        return self.run_root / horizon_folder_name(self.horizon_rows)

    @property
    def joined_export_path(self) -> Path:
        return self.out_dir / "sora_joined_to_xgb_pipeline.csv"

    @property
    def data_path(self) -> Path:
        return self.joined_export_path

    @property
    def reit_fwd_return_col(self) -> str:
        return reit_fwd_return_col(self.horizon_rows)

    @property
    def sora_fwd_level_col(self) -> str:
        return sora_fwd_level_col(self.horizon_rows)

    @property
    def sora_fwd_change_col(self) -> str:
        return sora_fwd_change_col(self.horizon_rows)

    @property
    def sora_fwd_abs_change_col(self) -> str:
        return sora_fwd_abs_change_col(self.horizon_rows)

    @property
    def target_specs(self) -> Dict[str, Dict[str, str]]:
        return build_target_specs(self.horizon_rows)


def compute_worker_seed(
    horizon_rows: int,
    selected_horizons: List[int],
    base_seed: int = SEED,
    per_horizon_seed_offset: bool = PER_HORIZON_SEED_OFFSET,
) -> int:
    """
    Make seeding policy explicit.

    - Single-horizon default behavior keeps the baseline seed unchanged.
    - Multi-horizon runs can optionally offset by horizon so each worker gets a
      deterministic but distinct RNG stream.
    """
    if len(selected_horizons) <= 1:
        return base_seed
    if per_horizon_seed_offset:
        return base_seed + horizon_rows
    return base_seed

TARGET_KEYS_ALL = ("option1_level", "option2_change", "option3_abs_change")

# Which of the three SORA targets to fit in a run (overridable via CLI: --all-targets, --option1, ...).
RUN_TARGET_OPTION1_LEVEL = False
RUN_TARGET_OPTION2_CHANGE = True
RUN_TARGET_OPTION3_ABS_CHANGE = False

DEFAULT_DROP_CALENDAR = "sreit"

OPTUNA_N_TRIALS = 80
DEAP_GENERATIONS = 8
DEAP_POPULATION_SIZE = 20
# 9 discrete hyperparameter genes; ~1 gene mutated per individual on average (was 0.0015 -> frozen).
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


def print_deap_generation_progress(
    target_key: str,
    generation: int,
    total_generations: int,
    population,
    hall_of_fame,
) -> None:
    """Emit lightweight GA progress similar to a typical DEAP logbook printout."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Model P pipeline: build joined SORA dataset in-process, export it, then train one or more SORA targets with Optuna+DEAP."
    )
    default_target_specs = build_target_specs(DEFAULT_FORWARD_HORIZON_ROWS_LIST[0])
    p.add_argument(
        "--all-targets",
        action="store_true",
        help="Run all three targets: level, signed change, and absolute change.",
    )
    p.add_argument(
        "--option1",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Run option 1 ({default_target_specs['option1_level']['target_col']}). "
        f"Omit to use RUN_TARGET_OPTION1_LEVEL ({RUN_TARGET_OPTION1_LEVEL}) in the script.",
    )
    p.add_argument(
        "--option2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Run option 2 ({default_target_specs['option2_change']['target_col']}). "
        f"Omit to use RUN_TARGET_OPTION2_CHANGE ({RUN_TARGET_OPTION2_CHANGE}).",
    )
    p.add_argument(
        "--option3",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Run option 3 ({default_target_specs['option3_abs_change']['target_col']}). "
        f"Omit to use RUN_TARGET_OPTION3_ABS_CHANGE ({RUN_TARGET_OPTION3_ABS_CHANGE}).",
    )
    p.add_argument(
        "--drop-calendar-by",
        choices=("sreit", "sora"),
        default=DEFAULT_DROP_CALENDAR,
        help="Row-universe switch for dropping non-matching dates. Default preserves the original S-REIT trading-calendar logic.",
    )
    p.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_FORWARD_HORIZON_ROWS_LIST),
        help="Forward SGX-row horizons to run. Default keeps the current single-horizon behavior.",
    )
    p.add_argument(
        "--max-horizon-workers",
        type=int,
        default=MAX_HORIZON_WORKERS,
        help="Max number of horizon jobs to run in parallel when multiple horizons are requested.",
    )
    return p.parse_args()


def enabled_target_keys_from_args(args: argparse.Namespace) -> List[str]:
    if args.all_targets:
        return list(TARGET_KEYS_ALL)
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


def build_forward_value_lookup(
    calendar_dates: pd.Series,
    value_df: pd.DataFrame,
    value_col: str,
    out_col: str,
    horizon_rows: int,
) -> pd.DataFrame:
    """
    Build a forward lookup on a chosen row-universe calendar.

    This lets us compute future targets from the full future calendar coverage
    even when the model row universe (e.g. xgb_ready) ends earlier.
    """
    calendar_df = pd.DataFrame({
        "snapshot_ts": pd.Series(calendar_dates).drop_duplicates().sort_values().reset_index(drop=True)
    })
    lookup = calendar_df.merge(
        value_df[["snapshot_ts", value_col]],
        on="snapshot_ts",
        how="left",
    ).sort_values("snapshot_ts").reset_index(drop=True)
    lookup[out_col] = lookup[value_col].shift(-horizon_rows)
    return lookup[["snapshot_ts", out_col]]


def build_joined_pipeline_dataset(cfg: HorizonConfig, drop_calendar_by: str) -> pd.DataFrame:
    print("Loading cleaned SORA daily files ...")
    sora = pd.read_csv(SORA_DAILY_PATH, parse_dates=["value_date"])
    sora_3m = pd.read_csv(SORA_3M_DAILY_PATH, parse_dates=["value_date"])

    # Preserve the realized, non-lagged SORA series for future-target construction.
    sora_realized = sora.copy()

    # Apply fixed T-2 business-day lag before any calendar expansion.
    sora["sora_level"] = sora["sora_level"].shift(2)
    sora_3m["sora_3m"] = sora_3m["sora_3m"].shift(2)

    print("Loading SGX trading-day calendar proxy ...")
    reit_index = pd.read_csv(SGX_REIT_INDEX_PATH)
    reit_index["snapshot_ts"] = pd.to_datetime(
        reit_index["time"], unit="s", utc=True
    ).dt.tz_localize(None).dt.normalize()
    reit_index = reit_index.sort_values("snapshot_ts").reset_index(drop=True)
    reit_index[cfg.reit_fwd_return_col] = (
        reit_index["close"].shift(-cfg.horizon_rows) - reit_index["close"]
    ) / reit_index["close"]
    reit_index[cfg.reit_fwd_return_col] = reit_index[cfg.reit_fwd_return_col].round(6)
    reit_index = reit_index.rename(columns={
        "open": "reit_index_open",
        "high": "reit_index_high",
        "low": "reit_index_low",
        "close": "reit_index_close",
    })
    sgx_trading_days = (
        reit_index["snapshot_ts"].drop_duplicates().sort_values().reset_index(drop=True)
    )

    realized_sora_publication_days = (
        sora_realized["value_date"].drop_duplicates().sort_values().reset_index(drop=True)
    )

    # Build realized SORA path on a full calendar, then later align to the chosen row universe.
    realized_cal_start = sora_realized["value_date"].min()
    realized_cal_end = sora_realized["value_date"].max()
    realized_cal_index = pd.DataFrame(
        {"value_date": pd.date_range(realized_cal_start, realized_cal_end, freq="D")}
    )

    realized_sora_cal = realized_cal_index.merge(
        sora_realized, on="value_date", how="left"
    )
    realized_sora_cal["sora_level_realized"] = realized_sora_cal["sora_level"].ffill()
    realized_sora_cal = realized_sora_cal[["value_date", "sora_level_realized"]].copy()
    realized_sora_cal = realized_sora_cal.rename(columns={"value_date": "snapshot_ts"})
    realized_sora_cal["snapshot_ts"] = realized_sora_cal["snapshot_ts"].dt.normalize()

    # Expand to full calendar-day index and forward-fill across weekends/holidays.
    cal_start = sora["value_date"].min()
    cal_end = sora["value_date"].max()
    cal_index = pd.DataFrame(
        {"value_date": pd.date_range(cal_start, cal_end, freq="D")}
    )

    sora_cal = cal_index.merge(sora, on="value_date", how="left")
    sora_cal = sora_cal.merge(sora_3m, on="value_date", how="left")

    sora_cal["sora_level"] = sora_cal["sora_level"].ffill()
    sora_cal["sora_3m"] = sora_cal["sora_3m"].ffill()

    sora_cal = sora_cal.set_index("value_date")
    sora_cal["sora_90d_change"] = (
        (sora_cal["sora_level"] - sora_cal["sora_level"].shift(90)) * 100
    ).round(4)
    sora_cal["sora_term_spread"] = (
        sora_cal["sora_3m"] - sora_cal["sora_level"]
    ).round(4)

    sora_cal = sora_cal.rename(columns={
        "sora_level": "sora_level_t2",
        "sora_90d_change": "sora_90d_change_t2",
        "sora_3m": "sora_3m_t2",
        "sora_term_spread": "sora_term_spread_t2",
    })
    sora_cal = sora_cal.reset_index()
    sora_cal = sora_cal.rename(columns={"value_date": "date"})

    print("Loading xgb_ready ...")
    xgb = pd.read_csv(XGB_READY_PATH, parse_dates=["snapshot_ts"])
    xgb["snapshot_ts"] = xgb["snapshot_ts"].dt.normalize()

    if drop_calendar_by == "sora":
        allowed_days = realized_sora_publication_days
        calendar_label = "realized SORA publication calendar"
    else:
        allowed_days = sgx_trading_days
        calendar_label = "S-REIT trading calendar"

    print(f"Filtering xgb_ready to {calendar_label} ...")
    raw_rows = len(xgb)
    xgb = xgb[xgb["snapshot_ts"].isin(allowed_days)].copy()
    filtered_rows = len(xgb)
    print(
        f"  Kept {filtered_rows:,} / {raw_rows:,} rows after dropping dates not present in the {calendar_label}."
    )

    print("Appending REIT index time-series values ...")
    xgb = xgb.merge(
        reit_index[["snapshot_ts", "reit_index_open", "reit_index_high",
                    "reit_index_low", "reit_index_close",
                    cfg.reit_fwd_return_col]],
        on="snapshot_ts",
        how="left",
    )

    print("Appending realized SORA path for future-target construction ...")
    xgb = xgb.merge(
        realized_sora_cal[["snapshot_ts", "sora_level_realized"]],
        on="snapshot_ts",
        how="left",
    )

    xgb = xgb.sort_values("snapshot_ts").reset_index(drop=True)
    print(
        f"Computing future SORA target columns from the full SGX calendar with a +{cfg.horizon_rows}-row lookahead ..."
    )
    sora_forward_lookup = build_forward_value_lookup(
        calendar_dates=sgx_trading_days,
        value_df=realized_sora_cal,
        value_col="sora_level_realized",
        out_col=cfg.sora_fwd_level_col,
        horizon_rows=cfg.horizon_rows,
    )
    xgb = xgb.merge(
        sora_forward_lookup,
        on="snapshot_ts",
        how="left",
    )
    xgb[cfg.sora_fwd_change_col] = (
        xgb[cfg.sora_fwd_level_col] - xgb["sora_level_realized"]
    ).round(6)
    xgb[cfg.sora_fwd_abs_change_col] = xgb[cfg.sora_fwd_change_col].abs().round(6)

    sora_cal_for_join = sora_cal.rename(columns={"date": "snapshot_ts"})
    joined = xgb.merge(
        sora_cal_for_join[["snapshot_ts", "sora_level_t2", "sora_90d_change_t2",
                           "sora_3m_t2", "sora_term_spread_t2"]],
        on="snapshot_ts",
        how="left",
    )

    n_null_sora = joined["sora_level_t2"].isna().sum()
    if n_null_sora > 0:
        print(
            f"  WARNING: {n_null_sora} rows in xgb_ready have no SORA match (xgb_ready dates outside SORA coverage)"
        )
    else:
        print("  All xgb_ready rows matched - no SORA gaps.")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.export_joined_csv:
        joined.to_csv(cfg.joined_export_path, index=False)
        print(f"Saved joined pipeline dataset: {cfg.joined_export_path}")
    else:
        print(f"Skipped joined pipeline CSV export (--no-export-joined-csv): {cfg.joined_export_path}")
    print(f"  Final shape: {joined.shape[0]:,} rows x {joined.shape[1]} columns")

    return joined


def write_data_manifests(
    cfg: HorizonConfig,
    base_df: pd.DataFrame,
    run_target_keys: List[str],
    drop_calendar_by: str,
) -> None:
    """Match R* 1fold script: record feature groupings and data window for the run folder."""
    manifest = {
        "data_path": str(cfg.data_path),
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
        "forward_horizon_rows": cfg.horizon_rows,
        "worker_seed": cfg.worker_seed,
        "run_target_keys": list(run_target_keys),
        "drop_calendar_by": drop_calendar_by,
        "export_joined_csv": cfg.export_joined_csv,
        "joined_export_path": str(cfg.joined_export_path),
        "joined_source_paths": {
            "xgb_ready": str(XGB_READY_PATH),
            "sreit_index": str(SGX_REIT_INDEX_PATH),
            "sora_daily": str(SORA_DAILY_PATH),
            "sora_3m_daily": str(SORA_3M_DAILY_PATH),
        },
        "run_flags_default": {
            "RUN_TARGET_OPTION1_LEVEL": RUN_TARGET_OPTION1_LEVEL,
            "RUN_TARGET_OPTION2_CHANGE": RUN_TARGET_OPTION2_CHANGE,
            "RUN_TARGET_OPTION3_ABS_CHANGE": RUN_TARGET_OPTION3_ABS_CHANGE,
            "PER_HORIZON_SEED_OFFSET": PER_HORIZON_SEED_OFFSET,
            "EXPORT_JOINED_CSV": EXPORT_JOINED_CSV,
        },
    }
    save_json(cfg.out_dir / "data_manifest.json", manifest)
    save_json(
        cfg.out_dir / "feature_manifest.json",
        {
            "features": FEATURE_COLS,
            "base_features": BASE_FEATURE_COLS,
            "spread_features": SPREAD_FEATURE_COLS,
            "engineered_sora_path": ENGINEERED_SORA_PATH_COLS,
            "option1_level_excluded_features": list(OPTION1_LEVEL_EXCLUDED_FEATURES),
            "option2_change_excluded_features": list(OPTION2_CHANGE_EXCLUDED_FEATURES),
            "trace_cols": TRACE_COLS,
            "all_target_keys": list(TARGET_KEYS_ALL),
            "run_target_keys": list(run_target_keys),
            "drop_calendar_by": drop_calendar_by,
            "forward_horizon_rows": cfg.horizon_rows,
            "worker_seed": cfg.worker_seed,
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


def load_base_dataset(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    if "fomc_decision_date" in df.columns:
        df["fomc_decision_date"] = pd.to_datetime(df["fomc_decision_date"], errors="coerce")
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
    out_dir: Path,
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
    save_json(out_dir / f"{target_key}_optuna_best_params.json", payload)
    return study.best_params


def run_deap_search(
    df: pd.DataFrame,
    target_key: str,
    target_col: str,
    signed_target: bool,
    feature_cols: List[str],
    out_dir: Path,
) -> Dict:
    # Conservative DEAP search space for the current small-row regime.
    # If future datasets become materially larger, these caps may be loosened.
    search_space = {
        "gamma": [float(x) for x in np.linspace(0.0, 5.0, 11)],
        "n_estimators": [100, 150, 200, 300, 400, 500, 700, 900],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
        "min_child_weight": [2.0, 3.0, 5.0, 7.0, 10.0],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0],
        "reg_lambda": [0.1, 0.5, 1.0, 3.0, 5.0, 10.0],
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
    print_deap_generation_progress(
        target_key=target_key,
        generation=0,
        total_generations=DEAP_GENERATIONS,
        population=population,
        hall_of_fame=hall_of_fame,
    )

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
        print_deap_generation_progress(
            target_key=target_key,
            generation=_generation + 1,
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


def export_holdout_results(result: HoldoutResult, target_key: str, optimizer_name: str, out_dir: Path) -> None:
    prefix = f"{target_key}_{optimizer_name}"
    result.holdout_metrics.to_csv(out_dir / f"{prefix}_holdout_metrics.csv", index=False)
    result.oos_predictions.to_csv(out_dir / f"{prefix}_holdout_oos_predictions.csv", index=False)
    result.baseline_metrics.to_csv(out_dir / f"{prefix}_baseline_holdout_metrics.csv", index=False)
    result.baseline_oos_predictions.to_csv(out_dir / f"{prefix}_baseline_holdout_oos_predictions.csv", index=False)
    save_json(out_dir / f"{prefix}_holdout_summary.json", result.summary)


def choose_winner(optuna_result: HoldoutResult, deap_result: HoldoutResult, signed_target: bool) -> str:
    o = optuna_result.summary
    d = deap_result.summary

    if signed_target:
        # 1) Primary: AUC (threshold-independent directional ranking quality).
        auc_diff = o["auc"] - d["auc"]
        if auc_diff > AUC_TOLERANCE:
            return "optuna"
        if auc_diff < -AUC_TOLERANCE:
            return "deap"

        # 2) Secondary: RMSE (magnitude calibration quality).
        rmse_diff = o["rmse"] - d["rmse"]
        if rmse_diff > RMSE_TOLERANCE:
            return "deap"  # DEAP has lower RMSE.
        if rmse_diff < -RMSE_TOLERANCE:
            return "optuna"

        # 3) Tiebreaker: F1 at zero threshold.
        return "optuna" if o["f1"] >= d["f1"] else "deap"

    # Unsigned targets: RMSE primary with tolerance, R2 tiebreak.
    rmse_diff = o["rmse"] - d["rmse"]
    if rmse_diff < -RMSE_TOLERANCE:
        return "optuna"
    if rmse_diff > RMSE_TOLERANCE:
        return "deap"
    return "optuna" if o["r2"] >= d["r2"] else "deap"


def fit_final_model(df: pd.DataFrame, params: Dict, target_col: str, feature_cols: List[str]) -> XGBRegressor:
    model = build_base_model(params)
    return model.fit(df[feature_cols], df[target_col])


def run_shap(
    final_model: XGBRegressor,
    df: pd.DataFrame,
    target_col: str,
    target_key: str,
    feature_cols: List[str],
    out_dir: Path,
) -> None:
    shap_df = df[[DATE_COL] + feature_cols + [target_col]].copy()
    shap_df = shap_df.tail(min(SHAP_MAX_ROWS, len(shap_df))).reset_index(drop=True)
    X_shap = shap_df[feature_cols]

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_shap)

    shap_values_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_values_df.insert(0, DATE_COL, shap_df[DATE_COL].astype(str))
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
    run_target_keys: List[str],
    drop_calendar_by: str,
) -> None:
    target_specs = cfg.target_specs
    joined_export_line = (
        f"Joined export: {cfg.joined_export_path}"
        if cfg.export_joined_csv
        else f"Joined export: disabled (--no-export-joined-csv); would otherwise be {cfg.joined_export_path}"
    )
    lines = [
        "train_p_1fold_pipeline.py output summary",
        "",
        f"Dataset: {cfg.data_path}",
        joined_export_line,
        f"Forward horizon rows: {cfg.horizon_rows}",
        f"Worker seed: {cfg.worker_seed}",
        f"Drop calendar by: {drop_calendar_by}",
        f"Output directory: {cfg.out_dir}",
        "",
        "Default run flags in script: "
        f"opt1={RUN_TARGET_OPTION1_LEVEL} opt2={RUN_TARGET_OPTION2_CHANGE} "
        f"opt3={RUN_TARGET_OPTION3_ABS_CHANGE} per_horizon_seed_offset={PER_HORIZON_SEED_OFFSET}",
        "",
        "Targets in this run:",
    ]
    for key in run_target_keys:
        spec = target_specs[key]
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
    run_target_keys: List[str],
    selected_horizons: List[int],
) -> Dict[str, str]:
    worker_seed = compute_worker_seed(horizon_rows=horizon_rows, selected_horizons=selected_horizons)
    set_global_seed(worker_seed)
    cfg = HorizonConfig(
        horizon_rows=horizon_rows,
        run_root=run_root,
        worker_seed=worker_seed,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    target_specs = cfg.target_specs

    print(f"\n=== Horizon {horizon_rows} rows -> {cfg.out_dir} ===")
    print("Building joined pipeline dataset ...")
    joined_df = build_joined_pipeline_dataset(cfg=cfg, drop_calendar_by=drop_calendar_by)

    print("Loading base dataset ...")
    base_df = load_base_dataset(
        joined_df,
        target_cols=[
            cfg.sora_fwd_level_col,
            cfg.sora_fwd_change_col,
            cfg.sora_fwd_abs_change_col,
        ],
    )
    write_data_manifests(cfg, base_df, run_target_keys=run_target_keys, drop_calendar_by=drop_calendar_by)
    print(f"Targets this run: {run_target_keys}")

    all_comparisons = {}
    all_selected_params = {}

    for target_key in run_target_keys:
        spec = target_specs[target_key]
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
            df,
            target_key=target_key,
            target_col=target_col,
            signed_target=signed_target,
            feature_cols=fcols,
            out_dir=cfg.out_dir,
        )
        optuna_eval = evaluate_single_holdout(
            df, optuna_params, label="optuna", target_col=target_col, signed_target=signed_target, feature_cols=fcols
        )
        export_holdout_results(optuna_eval, target_key=target_key, optimizer_name="optuna", out_dir=cfg.out_dir)

        print("Running DEAP search ...")
        deap_params = run_deap_search(
            df,
            target_key=target_key,
            target_col=target_col,
            signed_target=signed_target,
            feature_cols=fcols,
            out_dir=cfg.out_dir,
        )
        deap_eval = evaluate_single_holdout(
            df, deap_params, label="deap", target_col=target_col, signed_target=signed_target, feature_cols=fcols
        )
        export_holdout_results(deap_eval, target_key=target_key, optimizer_name="deap", out_dir=cfg.out_dir)

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
        final_model.save_model(str(cfg.out_dir / f"{target_key}_final_model_xgb.json"))
        run_shap(
            final_model,
            df,
            target_col=target_col,
            target_key=target_key,
            feature_cols=fcols,
            out_dir=cfg.out_dir,
        )

    save_json(cfg.out_dir / "all_targets_optimizer_comparison.json", all_comparisons)
    save_json(cfg.out_dir / "all_targets_selected_params.json", all_selected_params)
    write_run_contents_summary(cfg, run_target_keys=run_target_keys, drop_calendar_by=drop_calendar_by)
    print("\nDone. Outputs written to:")
    print(cfg.out_dir)
    return {
        "horizon_rows": str(horizon_rows),
        "out_dir": str(cfg.out_dir),
    }


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

    horizons = normalize_horizons(args.horizons)
    run_root = create_run_root()
    print(f"Run root: {run_root}")
    print(f"Horizons: {horizons}")
    print(f"Targets: {run_target_keys}")

    if len(horizons) == 1:
        run_single_horizon_pipeline(
            run_root=run_root,
            horizon_rows=horizons[0],
            drop_calendar_by=args.drop_calendar_by,
            run_target_keys=run_target_keys,
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
                run_target_keys,
                horizons,
            )
            futures[future] = horizon_rows

        for future in as_completed(futures):
            horizon_rows = futures[future]
            result = future.result()
            print(f"Completed horizon {horizon_rows}: {result['out_dir']}")


if __name__ == "__main__":
    main()
