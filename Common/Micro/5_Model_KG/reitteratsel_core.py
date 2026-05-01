from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from dotenv import dotenv_values
from neo4j import GraphDatabase


ROOT_DIR = Path(__file__).resolve().parents[3]
DUCKDB_PATH = ROOT_DIR / "Common" / "Micro" / "IO" / "out" / "_annual_warehouse" / "fundamentals.duckdb"
PARQUET_DIR = DUCKDB_PATH.parent / "parquet"
DISTRESS_LABEL_SHARD = PARQUET_DIR / "distresslabels.parquet"
FUZZY_CACHE_SHARD = PARQUET_DIR / "fuzzycache.parquet"
CSV_TICKER_DIR = ROOT_DIR / "Common" / "Macro" / "IO" / "SRC" / "CSV_TICKER"
ENV_PATH = ROOT_DIR / ".env"
XGB_RUN_ROOT = ROOT_DIR / "Common" / "Macro" / "IO" / "Model_Train" / "Use" / "run_21"

DEFAULT_INDEX_TICKER = "REIT"
DEFAULT_HORIZON_DAYS = 10
LABEL_VERSION = "v1_car126_2026_05_01"
RULE_VERSION = "v1_seed_2026_05_01"
SCORE_VERSION = "v1_mamdani_2026_05_01"
LABEL_THRESHOLDS = {
    "DISTRESSED": -0.15,
    "HEALTHY": 0.05,
}
OUTPUT_BANDS = [
    (0.00, 0.34, "STABLE"),
    (0.35, 0.54, "WATCH"),
    (0.55, 0.74, "HIGH"),
    (0.75, 1.00, "CRITICAL"),
]
STATUS_CONFIDENCE = {
    "OK": 1.0,
    "PARTIAL": 0.85,
    "CLIPPED_SOURCE_SHARE": 0.85,
    "LOW_DENOMINATOR": 0.75,
    "MISSING_INPUT": 0.0,
}
STATUS_TERM_OVERRIDE = {
    "ICR": {"NEGATIVE_BASE": "distress"},
    "DSCR": {"NEGATIVE_BASE": "distress"},
    "NET_DEBT_EBITDA": {"NEGATIVE_BASE": "distress"},
    "PAYOUT_RATIO": {"DISTRESS_BASE": "over_distributing"},
    "FFO_COVERAGE": {"DISTRESS_BASE": "shortfall"},
}
MAMdANI_GRID = [x / 1000.0 for x in range(1001)]


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    database: str
    username: str
    password: str


INPUT_TERM_CONFIG: dict[str, list[dict[str, Any]]] = {
    "ICR": [
        {"term": "distress", "shape": "trapezoid", "params": [0.0, 0.0, 1.25, 1.60]},
        {"term": "watch", "shape": "triangle", "params": [1.40, 2.25, 3.40]},
        {"term": "healthy", "shape": "trapezoid", "params": [2.80, 3.60, 10.0, 10.0]},
    ],
    "GEARING": [
        {"term": "healthy", "shape": "trapezoid", "params": [0.0, 0.0, 0.38, 0.42]},
        {"term": "watch", "shape": "triangle", "params": [0.38, 0.45, 0.50]},
        {"term": "distress", "shape": "trapezoid", "params": [0.45, 0.50, 1.00, 1.00]},
    ],
    "DSCR": [
        {"term": "distress", "shape": "trapezoid", "params": [0.0, 0.0, 0.20, 0.35]},
        {"term": "watch", "shape": "triangle", "params": [0.25, 0.50, 0.80]},
        {"term": "healthy", "shape": "trapezoid", "params": [0.65, 0.90, 4.00, 4.00]},
    ],
    "REFI_RISK": [
        {"term": "healthy", "shape": "trapezoid", "params": [0.0, 0.0, 0.08, 0.15]},
        {"term": "watch", "shape": "triangle", "params": [0.10, 0.18, 0.30]},
        {"term": "distress", "shape": "triangle", "params": [0.20, 0.35, 0.55]},
        {"term": "critical", "shape": "trapezoid", "params": [0.55, 0.75, 1.00, 1.00]},
    ],
    "PAYOUT_RATIO": [
        {"term": "under_distributing", "shape": "trapezoid", "params": [0.0, 0.0, 0.40, 0.60]},
        {"term": "balanced", "shape": "triangle", "params": [0.40, 0.80, 1.00]},
        {"term": "over_distributing", "shape": "trapezoid", "params": [0.95, 1.10, 2.00, 2.00]},
    ],
    "FFO_COVERAGE": [
        {"term": "shortfall", "shape": "trapezoid", "params": [-1.00, -1.00, 0.00, 0.10]},
        {"term": "thin", "shape": "triangle", "params": [0.00, 0.10, 0.25]},
        {"term": "buffered", "shape": "trapezoid", "params": [0.10, 0.25, 1.00, 1.00]},
    ],
    "NET_DEBT_EBITDA": [
        {"term": "healthy", "shape": "trapezoid", "params": [0.0, 0.0, 6.5, 7.5]},
        {"term": "watch", "shape": "triangle", "params": [6.5, 8.5, 10.5]},
        {"term": "distress", "shape": "trapezoid", "params": [9.0, 11.0, 30.0, 30.0]},
    ],
    "NULL_COUNT": [
        {"term": "healthy", "shape": "trapezoid", "params": [0.0, 0.0, 1.0, 3.0]},
        {"term": "watch", "shape": "triangle", "params": [2.0, 5.0, 8.0]},
        {"term": "distress", "shape": "trapezoid", "params": [6.0, 9.0, 19.0, 19.0]},
    ],
}

OUTPUT_TERM_CONFIG = {
    "stable": {"shape": "trapezoid", "params": [0.00, 0.00, 0.20, 0.35]},
    "watch": {"shape": "triangle", "params": [0.25, 0.45, 0.60]},
    "high": {"shape": "triangle", "params": [0.55, 0.70, 0.85]},
    "critical": {"shape": "trapezoid", "params": [0.75, 0.90, 1.00, 1.00]},
}

RULE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "rule_id": "R1",
        "operator": "AND",
        "weight": 1.0,
        "output_term": "critical",
        "description": "ICR in distress zone is a direct solvency alarm.",
        "antecedents": [{"metric_code": "ICR", "term": "distress"}],
    },
    {
        "rule_id": "R2",
        "operator": "AND",
        "weight": 1.0,
        "output_term": "critical",
        "description": "DSCR in distress zone indicates debt service strain.",
        "antecedents": [{"metric_code": "DSCR", "term": "distress"}],
    },
    {
        "rule_id": "R3",
        "operator": "AND",
        "weight": 0.9,
        "output_term": "high",
        "description": "High gearing is a strong balance-sheet stress signal.",
        "antecedents": [{"metric_code": "GEARING", "term": "distress"}],
    },
    {
        "rule_id": "R4",
        "operator": "AND",
        "weight": 1.0,
        "output_term": "critical",
        "description": "Extreme refinancing concentration is a direct maturity wall alarm.",
        "antecedents": [{"metric_code": "REFI_RISK", "term": "critical"}],
    },
    {
        "rule_id": "R5",
        "operator": "AND",
        "weight": 0.9,
        "output_term": "high",
        "description": "Refinancing stress with weak leverage amplifies distress risk.",
        "antecedents": [
            {"metric_code": "REFI_RISK", "term": "distress"},
            {"metric_code": "GEARING", "term": "watch"},
        ],
    },
    {
        "rule_id": "R6",
        "operator": "AND",
        "weight": 0.95,
        "output_term": "critical",
        "description": "Weak coverage plus stretched leverage is a high-severity combination.",
        "antecedents": [
            {"metric_code": "ICR", "term": "watch"},
            {"metric_code": "GEARING", "term": "distress"},
        ],
    },
    {
        "rule_id": "R7",
        "operator": "AND",
        "weight": 0.85,
        "output_term": "high",
        "description": "ICR and DSCR both in the watch zone reinforce pressure.",
        "antecedents": [
            {"metric_code": "ICR", "term": "watch"},
            {"metric_code": "DSCR", "term": "watch"},
        ],
    },
    {
        "rule_id": "R8",
        "operator": "AND",
        "weight": 0.85,
        "output_term": "high",
        "description": "Over-distribution with FFO shortfall indicates payout strain.",
        "antecedents": [
            {"metric_code": "PAYOUT_RATIO", "term": "over_distributing"},
            {"metric_code": "FFO_COVERAGE", "term": "shortfall"},
        ],
    },
    {
        "rule_id": "R9",
        "operator": "AND",
        "weight": 0.75,
        "output_term": "high",
        "description": "High net debt with weak ICR corroborates leverage stress.",
        "antecedents": [
            {"metric_code": "NET_DEBT_EBITDA", "term": "distress"},
            {"metric_code": "ICR", "term": "watch"},
        ],
    },
    {
        "rule_id": "R10",
        "operator": "AND",
        "weight": 0.65,
        "output_term": "high",
        "description": "High missingness count is treated as a material confidence penalty.",
        "antecedents": [{"metric_code": "NULL_COUNT", "term": "distress"}],
    },
    {
        "rule_id": "R11",
        "operator": "AND",
        "weight": 0.65,
        "output_term": "high",
        "description": "Moderate missingness plus weak coverage should not be ignored.",
        "antecedents": [
            {"metric_code": "NULL_COUNT", "term": "watch"},
            {"metric_code": "ICR", "term": "watch"},
        ],
    },
    {
        "rule_id": "R12",
        "operator": "AND",
        "weight": 0.90,
        "output_term": "stable",
        "description": "Core balance-sheet and coverage metrics are all healthy.",
        "antecedents": [
            {"metric_code": "ICR", "term": "healthy"},
            {"metric_code": "GEARING", "term": "healthy"},
            {"metric_code": "DSCR", "term": "healthy"},
            {"metric_code": "REFI_RISK", "term": "healthy"},
        ],
    },
    {
        "rule_id": "R13",
        "operator": "AND",
        "weight": 0.60,
        "output_term": "watch",
        "description": "Leverage is elevated, but coverage remains resilient.",
        "antecedents": [
            {"metric_code": "GEARING", "term": "watch"},
            {"metric_code": "DSCR", "term": "healthy"},
            {"metric_code": "ICR", "term": "healthy"},
        ],
    },
    {
        "rule_id": "R14",
        "operator": "AND",
        "weight": 0.70,
        "output_term": "stable",
        "description": "Balanced payout and positive FFO buffer support a stable reading.",
        "antecedents": [
            {"metric_code": "PAYOUT_RATIO", "term": "balanced"},
            {"metric_code": "FFO_COVERAGE", "term": "buffered"},
            {"metric_code": "ICR", "term": "healthy"},
        ],
    },
]


def load_neo4j_config(env_path: Path = ENV_PATH) -> Neo4jConfig:
    values = dotenv_values(env_path)
    return Neo4jConfig(
        uri=values["NEO4J_URI"],
        database=values.get("NEO4J_DATABASE", "neo4j"),
        username=values["NEO4J_USERNAME"],
        password=values["NEO4J_PASSWORD"],
    )


def connect_neo4j(env_path: Path = ENV_PATH):
    cfg = load_neo4j_config(env_path)
    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.username, cfg.password))
    return driver, cfg


def read_annual_metric_frame(db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            """
            SELECT
                v.ticker,
                p.period_id,
                p.fiscal_year,
                p.fiscal_year_end_date,
                p.source_period_label,
                r.reit_name,
                r.sector,
                r.health_bucket,
                v.metric_code,
                v.metric_value,
                v.calc_status
            FROM reit_metrics.fact_metric_value v
            JOIN reit_metrics.dim_period p ON p.period_id = v.period_id
            JOIN reit_metrics.dim_reit r ON r.ticker = v.ticker
            ORDER BY v.ticker, p.fiscal_year_end_date, v.metric_code
            """
        ).fetchdf()
    finally:
        con.close()


def load_dim_period(db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            """
            SELECT
                period_id,
                ticker,
                source_period_label,
                fiscal_year,
                fiscal_year_end_date
            FROM reit_metrics.dim_period
            ORDER BY ticker, fiscal_year_end_date
            """
        ).fetchdf()
    finally:
        con.close()


def load_ticker_close_series(ticker: str) -> pd.DataFrame:
    csv_path = CSV_TICKER_DIR / f"SGX_DLY_{ticker}, 1D.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Ticker CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    close_col = "close" if "close" in df.columns else "Close"
    time_col = "time" if "time" in df.columns else "Time"
    out = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(df[time_col], unit="s").dt.date,
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    ).dropna()
    out["daily_return"] = out["close"].pct_change()
    return out.dropna(subset=["daily_return"]).reset_index(drop=True)


def build_abnormal_return_frame(
    tickers: list[str],
    index_ticker: str = DEFAULT_INDEX_TICKER,
) -> dict[str, pd.DataFrame]:
    index_df = load_ticker_close_series(index_ticker)[["trade_date", "daily_return"]].rename(
        columns={"daily_return": "index_return"}
    )
    abnormal_frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ticker_df = load_ticker_close_series(ticker)
        merged = ticker_df.merge(index_df, on="trade_date", how="inner")
        merged["abnormal_return"] = merged["daily_return"] - merged["index_return"]
        abnormal_frames[ticker] = merged
    return abnormal_frames


def _compound_return(series: pd.Series) -> float | None:
    if series.empty or series.isna().any():
        return None
    return float((1.0 + series).prod() - 1.0)


def label_from_car(car_126wd: float | None) -> str | None:
    if car_126wd is None or pd.isna(car_126wd):
        return None
    if car_126wd < LABEL_THRESHOLDS["DISTRESSED"]:
        return "DISTRESSED"
    if car_126wd > LABEL_THRESHOLDS["HEALTHY"]:
        return "HEALTHY"
    return "WATCH"


def build_distress_label_frame(
    db_path: Path = DUCKDB_PATH,
    label_version: str = LABEL_VERSION,
) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        counts_df = con.execute(
            """
            SELECT
                v.ticker,
                v.period_id,
                SUM(CASE WHEN v.calc_status = 'MISSING_INPUT' THEN 1 ELSE 0 END) AS null_count,
                SUM(CASE WHEN v.calc_status <> 'OK' THEN 1 ELSE 0 END) AS non_ok_count
            FROM reit_metrics.fact_metric_value v
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchdf()
        period_df = con.execute(
            """
            SELECT
                p.period_id,
                p.ticker,
                p.fiscal_year,
                p.fiscal_year_end_date
            FROM reit_metrics.dim_period p
            ORDER BY p.ticker, p.fiscal_year_end_date
            """
        ).fetchdf()
    finally:
        con.close()

    abnormal_map = build_abnormal_return_frame(sorted(period_df["ticker"].unique().tolist()))
    rows: list[dict[str, Any]] = []

    for period in period_df.itertuples():
        ticker_abnormal = abnormal_map[period.ticker]
        anchor_date = pd.Timestamp(period.fiscal_year_end_date).date()
        forward_slice = ticker_abnormal.loc[ticker_abnormal["trade_date"] >= anchor_date].copy()

        anchor_trade_date = None
        car_63wd = None
        car_126wd = None
        window_63_end_date = None
        window_126_end_date = None
        notes: list[str] = []

        if forward_slice.empty:
            notes.append("No trading data on or after anchor date.")
        else:
            anchor_trade_date = forward_slice.iloc[0]["trade_date"]
            future_returns = forward_slice.loc[forward_slice["trade_date"] > anchor_trade_date].reset_index(drop=True)
            if len(future_returns) >= 63:
                car_63wd = _compound_return(future_returns.loc[:62, "abnormal_return"])
                window_63_end_date = future_returns.iloc[62]["trade_date"]
            else:
                notes.append("Insufficient forward window for 63 trading days.")

            if len(future_returns) >= 126:
                car_126wd = _compound_return(future_returns.loc[:125, "abnormal_return"])
                window_126_end_date = future_returns.iloc[125]["trade_date"]
            else:
                notes.append("Insufficient forward window for 126 trading days.")

        count_row = counts_df.loc[
            (counts_df["ticker"] == period.ticker) & (counts_df["period_id"] == period.period_id)
        ].iloc[0]

        rows.append(
            {
                "ticker": period.ticker,
                "period_id": int(period.period_id),
                "anchor_date": anchor_date,
                "anchor_trade_date": anchor_trade_date,
                "window_63_end_date": window_63_end_date,
                "window_126_end_date": window_126_end_date,
                "car_63wd": car_63wd,
                "car_126wd": car_126wd,
                "null_count": int(count_row["null_count"]),
                "non_ok_count": int(count_row["non_ok_count"]),
                "label_scheme_version": label_version,
                "label_126wd": label_from_car(car_126wd),
                "source_index_code": DEFAULT_INDEX_TICKER,
                "notes": " ".join(notes) if notes else None,
            }
        )

    return pd.DataFrame(rows)


def triangle_membership(value: float, left: float, peak: float, right: float) -> float:
    if value <= left or value >= right:
        return 0.0
    if math.isclose(value, peak):
        return 1.0
    if value < peak:
        return (value - left) / (peak - left)
    return (right - value) / (right - peak)


def trapezoid_membership(value: float, a: float, b: float, c: float, d: float) -> float:
    if value <= a or value >= d:
        return 0.0
    if b <= value <= c:
        return 1.0
    if a < value < b:
        return (value - a) / (b - a)
    return (d - value) / (d - c)


def evaluate_term_shape(value: float, shape: str, params: list[float]) -> float:
    if shape == "triangle":
        return max(0.0, min(1.0, triangle_membership(value, *params)))
    if shape == "trapezoid":
        return max(0.0, min(1.0, trapezoid_membership(value, *params)))
    raise ValueError(f"Unknown membership shape: {shape}")


def compute_memberships(metric_code: str, value: float | None, calc_status: str | None) -> dict[str, float]:
    memberships = {item["term"]: 0.0 for item in INPUT_TERM_CONFIG[metric_code]}
    if calc_status in STATUS_TERM_OVERRIDE.get(metric_code, {}):
        memberships[STATUS_TERM_OVERRIDE[metric_code][calc_status]] = 1.0
        return memberships
    if value is None or pd.isna(value):
        return memberships

    confidence = STATUS_CONFIDENCE.get(calc_status or "OK", 1.0)
    for item in INPUT_TERM_CONFIG[metric_code]:
        memberships[item["term"]] = evaluate_term_shape(float(value), item["shape"], item["params"]) * confidence
    return memberships


def score_to_level(score: float) -> str:
    if score < 0.35:
        return "STABLE"
    if score < 0.55:
        return "WATCH"
    if score < 0.75:
        return "HIGH"
    return "CRITICAL"


def output_term_membership(term: str, x_value: float) -> float:
    cfg = OUTPUT_TERM_CONFIG[term]
    return evaluate_term_shape(x_value, cfg["shape"], cfg["params"])


def seed_rules_to_neo4j(driver, config: Neo4jConfig) -> None:
    with driver.session(database=config.database) as session:
        session.run(
            """
            MERGE (model:IRSModel {model_name: $model_name})
            SET model.rule_version = $rule_version,
                model.score_version = $score_version,
                model.updated_at = datetime()
            """,
            model_name="REITterratsel",
            rule_version=RULE_VERSION,
            score_version=SCORE_VERSION,
        )
        session.run(
            """
            MATCH (n)
            WHERE n.model_name = $model_name OR n.rule_version = $rule_version
            DETACH DELETE n
            """,
            model_name="REITterratsel",
            rule_version=RULE_VERSION,
        )
        session.run(
            """
            MERGE (model:IRSModel {model_name: $model_name})
            SET model.rule_version = $rule_version,
                model.score_version = $score_version,
                model.updated_at = datetime()
            """,
            model_name="REITterratsel",
            rule_version=RULE_VERSION,
            score_version=SCORE_VERSION,
        )
        for metric_code, term_defs in INPUT_TERM_CONFIG.items():
            session.run(
                """
                MATCH (model:IRSModel {model_name: $model_name})
                MERGE (metric:Metric {model_name: $model_name, metric_code: $metric_code})
                SET metric.rule_version = $rule_version
                MERGE (model)-[:HAS_METRIC]->(metric)
                """,
                model_name="REITterratsel",
                rule_version=RULE_VERSION,
                metric_code=metric_code,
            )
            for term_def in term_defs:
                session.run(
                    """
                    MATCH (metric:Metric {model_name: $model_name, metric_code: $metric_code})
                    MERGE (term:InputTerm {model_name: $model_name, metric_code: $metric_code, term: $term})
                    SET term.rule_version = $rule_version,
                        term.shape = $shape,
                        term.params = $params
                    MERGE (metric)-[:HAS_TERM]->(term)
                    """,
                    model_name="REITterratsel",
                    rule_version=RULE_VERSION,
                    metric_code=metric_code,
                    term=term_def["term"],
                    shape=term_def["shape"],
                    params=term_def["params"],
                )
        for output_term, cfg in OUTPUT_TERM_CONFIG.items():
            session.run(
                """
                MATCH (model:IRSModel {model_name: $model_name})
                MERGE (term:OutputTerm {model_name: $model_name, term: $term})
                SET term.rule_version = $rule_version,
                    term.shape = $shape,
                    term.params = $params
                MERGE (model)-[:HAS_OUTPUT_TERM]->(term)
                """,
                model_name="REITterratsel",
                rule_version=RULE_VERSION,
                term=output_term,
                shape=cfg["shape"],
                params=cfg["params"],
            )
        for rule in RULE_DEFINITIONS:
            session.run(
                """
                MATCH (model:IRSModel {model_name: $model_name})
                MATCH (output:OutputTerm {model_name: $model_name, term: $output_term})
                MERGE (rule:Rule {model_name: $model_name, rule_id: $rule_id})
                SET rule.rule_version = $rule_version,
                    rule.operator = $operator,
                    rule.weight = $weight,
                    rule.description = $description
                MERGE (model)-[:HAS_RULE]->(rule)
                MERGE (rule)-[:IMPLIES]->(output)
                """,
                model_name="REITterratsel",
                rule_version=RULE_VERSION,
                rule_id=rule["rule_id"],
                operator=rule["operator"],
                weight=rule["weight"],
                description=rule["description"],
                output_term=rule["output_term"],
            )
            for antecedent in rule["antecedents"]:
                session.run(
                    """
                    MATCH (rule:Rule {model_name: $model_name, rule_id: $rule_id})
                    MATCH (term:InputTerm {
                        model_name: $model_name,
                        metric_code: $metric_code,
                        term: $term
                    })
                    MERGE (rule)-[:USES_ANTECEDENT]->(term)
                    """,
                    model_name="REITterratsel",
                    rule_id=rule["rule_id"],
                    metric_code=antecedent["metric_code"],
                    term=antecedent["term"],
                )


def fetch_rule_bundle(driver, config: Neo4jConfig) -> dict[str, Any]:
    with driver.session(database=config.database) as session:
        rules = session.run(
            """
            MATCH (rule:Rule {model_name: $model_name})-[:USES_ANTECEDENT]->(term:InputTerm)
            MATCH (rule)-[:IMPLIES]->(output:OutputTerm)
            RETURN
                rule.rule_id AS rule_id,
                rule.operator AS operator,
                rule.weight AS weight,
                rule.description AS description,
                output.term AS output_term,
                collect({metric_code: term.metric_code, term: term.term}) AS antecedents
            ORDER BY rule.rule_id
            """,
            model_name="REITterratsel",
        ).data()
    return {"rules": rules}


def local_rule_bundle() -> dict[str, Any]:
    rules: list[dict[str, Any]] = []
    for rule in RULE_DEFINITIONS:
        rules.append(
            {
                "rule_id": rule["rule_id"],
                "operator": rule["operator"],
                "weight": rule["weight"],
                "description": rule["description"],
                "output_term": rule["output_term"],
                "antecedents": rule["antecedents"],
            }
        )
    return {"rules": rules}


def build_fuzzy_input_frame(db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    metric_df = read_annual_metric_frame(db_path)
    value_wide = metric_df.pivot_table(
        index=["ticker", "period_id", "fiscal_year", "fiscal_year_end_date", "source_period_label", "reit_name", "sector", "health_bucket"],
        columns="metric_code",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    status_wide = metric_df.pivot_table(
        index=["ticker", "period_id"],
        columns="metric_code",
        values="calc_status",
        aggfunc="first",
    ).reset_index()
    counts = (
        metric_df.assign(
            is_null=lambda df: (df["calc_status"] == "MISSING_INPUT").astype(int),
            is_non_ok=lambda df: (df["calc_status"] != "OK").astype(int),
        )
        .groupby(["ticker", "period_id"], as_index=False)[["is_null", "is_non_ok"]]
        .sum()
        .rename(columns={"is_null": "null_count", "is_non_ok": "non_ok_count"})
    )
    merged = value_wide.merge(status_wide, on=["ticker", "period_id"], suffixes=("", "_status"))
    merged = merged.merge(counts, on=["ticker", "period_id"], how="left")
    return merged


def evaluate_fuzzy_row(row: pd.Series, rule_bundle: dict[str, Any]) -> dict[str, Any]:
    membership_map: dict[tuple[str, str], float] = {}
    trace_memberships: dict[str, dict[str, float]] = {}

    for metric_code in ["ICR", "GEARING", "DSCR", "REFI_RISK", "PAYOUT_RATIO", "FFO_COVERAGE", "NET_DEBT_EBITDA"]:
        memberships = compute_memberships(metric_code, row.get(metric_code), row.get(f"{metric_code}_status"))
        trace_memberships[metric_code] = memberships
        for term, strength in memberships.items():
            membership_map[(metric_code, term)] = strength

    null_memberships = compute_memberships("NULL_COUNT", row.get("null_count"), "OK")
    trace_memberships["NULL_COUNT"] = null_memberships
    for term, strength in null_memberships.items():
        membership_map[("NULL_COUNT", term)] = strength

    fired_rules: list[dict[str, Any]] = []
    output_strengths = {term: 0.0 for term in OUTPUT_TERM_CONFIG}
    for rule in rule_bundle["rules"]:
        antecedent_strengths = [
            membership_map.get((item["metric_code"], item["term"]), 0.0)
            for item in rule["antecedents"]
        ]
        if not antecedent_strengths:
            continue
        if rule["operator"] == "OR":
            rule_strength = max(antecedent_strengths)
        else:
            rule_strength = min(antecedent_strengths)
        rule_strength *= float(rule["weight"])
        output_term = rule["output_term"]
        output_strengths[output_term] = max(output_strengths[output_term], rule_strength)
        if rule_strength > 0:
            fired_rules.append(
                {
                    "rule_id": rule["rule_id"],
                    "strength": round(rule_strength, 6),
                    "output_term": output_term,
                    "description": rule["description"],
                }
            )

    numerator = 0.0
    denominator = 0.0
    for x_value in MAMdANI_GRID:
        aggregate_membership = 0.0
        for term, strength in output_strengths.items():
            aggregate_membership = max(
                aggregate_membership,
                min(strength, output_term_membership(term, x_value)),
            )
        numerator += x_value * aggregate_membership
        denominator += aggregate_membership
    score = numerator / denominator if denominator else 0.5
    level = score_to_level(score)
    fired_rules.sort(key=lambda item: item["strength"], reverse=True)
    top_rules = fired_rules[:5]
    rule_trace_lines = [
        f"{item['rule_id']} ({item['output_term']}, {item['strength']:.3f}): {item['description']}"
        for item in top_rules
    ]
    return {
        "distress_score_mamdani": float(score),
        "distress_level": level,
        "fired_rule_count": len(fired_rules),
        "top_rule_ids": ",".join(item["rule_id"] for item in top_rules) if top_rules else None,
        "rule_trace_text": "\n".join(rule_trace_lines) if rule_trace_lines else None,
        "trace_memberships": trace_memberships,
    }


def build_fuzzy_cache_frame(
    db_path: Path = DUCKDB_PATH,
    env_path: Path = ENV_PATH,
    rule_bundle: dict[str, Any] | None = None,
    note: str | None = None,
) -> pd.DataFrame:
    inputs = build_fuzzy_input_frame(db_path)
    if rule_bundle is None:
        driver, config = connect_neo4j(env_path)
        try:
            rule_bundle = fetch_rule_bundle(driver, config)
        finally:
            driver.close()

    rows: list[dict[str, Any]] = []
    for row in inputs.to_dict(orient="records"):
        result = evaluate_fuzzy_row(pd.Series(row), rule_bundle)
        rows.append(
            {
                "ticker": row["ticker"],
                "period_id": int(row["period_id"]),
                "rule_version": RULE_VERSION,
                "score_version": SCORE_VERSION,
                "distress_score_mamdani": result["distress_score_mamdani"],
                "distress_level": result["distress_level"],
                "null_count": int(row["null_count"]),
                "non_ok_count": int(row["non_ok_count"]),
                "fired_rule_count": int(result["fired_rule_count"]),
                "top_rule_ids": result["top_rule_ids"],
                "rule_trace_text": result["rule_trace_text"],
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


def ensure_writable_duckdb_copy(source_path: Path = DUCKDB_PATH) -> Path:
    temp_path = source_path.with_name(f"{source_path.stem}.reitteratsel_tmp{source_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()
    temp_path.write_bytes(source_path.read_bytes())
    return temp_path


def persist_outputs_to_duckdb(
    label_df: pd.DataFrame,
    fuzzy_df: pd.DataFrame,
    db_path: Path = DUCKDB_PATH,
) -> None:
    work_path = ensure_writable_duckdb_copy(db_path)
    con = duckdb.connect(str(work_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS reit_labels")
        con.execute("CREATE SCHEMA IF NOT EXISTS reit_fuzzy")
        con.register("label_df", label_df)
        con.register("fuzzy_df", fuzzy_df)
        con.execute(
            """
            CREATE OR REPLACE TABLE reit_labels.fact_distress_label AS
            SELECT
                ticker::VARCHAR AS ticker,
                period_id::BIGINT AS period_id,
                anchor_date::DATE AS anchor_date,
                anchor_trade_date::DATE AS anchor_trade_date,
                window_63_end_date::DATE AS window_63_end_date,
                window_126_end_date::DATE AS window_126_end_date,
                car_63wd::DOUBLE AS car_63wd,
                car_126wd::DOUBLE AS car_126wd,
                null_count::INTEGER AS null_count,
                non_ok_count::INTEGER AS non_ok_count,
                label_scheme_version::VARCHAR AS label_scheme_version,
                label_126wd::VARCHAR AS label_126wd,
                source_index_code::VARCHAR AS source_index_code,
                current_timestamp AS asof_ts,
                notes::VARCHAR AS notes
            FROM label_df
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE reit_fuzzy.fact_fuzzy_cache AS
            SELECT
                ticker::VARCHAR AS ticker,
                period_id::BIGINT AS period_id,
                rule_version::VARCHAR AS rule_version,
                score_version::VARCHAR AS score_version,
                distress_score_mamdani::DOUBLE AS distress_score_mamdani,
                distress_level::VARCHAR AS distress_level,
                null_count::INTEGER AS null_count,
                non_ok_count::INTEGER AS non_ok_count,
                fired_rule_count::INTEGER AS fired_rule_count,
                top_rule_ids::VARCHAR AS top_rule_ids,
                rule_trace_text::VARCHAR AS rule_trace_text,
                current_timestamp AS asof_ts,
                notes::VARCHAR AS notes
            FROM fuzzy_df
            """
        )
        DISTRESS_LABEL_SHARD.parent.mkdir(parents=True, exist_ok=True)
        if DISTRESS_LABEL_SHARD.exists():
            DISTRESS_LABEL_SHARD.unlink()
        if FUZZY_CACHE_SHARD.exists():
            FUZZY_CACHE_SHARD.unlink()
        con.execute(
            f"""
            COPY (
                SELECT
                    l.ticker,
                    r.reit_name,
                    r.sector,
                    r.health_bucket,
                    p.fiscal_year,
                    p.fiscal_year_end_date,
                    l.period_id,
                    l.anchor_date,
                    l.anchor_trade_date,
                    l.window_63_end_date,
                    l.window_126_end_date,
                    l.car_63wd,
                    l.car_126wd,
                    l.null_count,
                    l.non_ok_count,
                    l.label_126wd,
                    l.label_scheme_version,
                    l.source_index_code,
                    l.notes
                FROM reit_labels.fact_distress_label l
                JOIN reit_metrics.dim_reit r ON r.ticker = l.ticker
                JOIN reit_metrics.dim_period p ON p.period_id = l.period_id
                ORDER BY l.ticker, p.fiscal_year_end_date
            ) TO '{DISTRESS_LABEL_SHARD.as_posix()}' (FORMAT PARQUET)
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT
                    f.ticker,
                    r.reit_name,
                    r.sector,
                    r.health_bucket,
                    p.fiscal_year,
                    p.fiscal_year_end_date,
                    f.period_id,
                    f.rule_version,
                    f.score_version,
                    f.distress_score_mamdani,
                    f.distress_level,
                    f.null_count,
                    f.non_ok_count,
                    f.fired_rule_count,
                    f.top_rule_ids,
                    f.rule_trace_text,
                    f.notes
                FROM reit_fuzzy.fact_fuzzy_cache f
                JOIN reit_metrics.dim_reit r ON r.ticker = f.ticker
                JOIN reit_metrics.dim_period p ON p.period_id = f.period_id
                ORDER BY f.ticker, p.fiscal_year_end_date
            ) TO '{FUZZY_CACHE_SHARD.as_posix()}' (FORMAT PARQUET)
            """
        )
        con.close()
        os.replace(work_path, db_path)
    finally:
        try:
            con.close()
        except Exception:
            pass
        if work_path.exists():
            work_path.unlink(missing_ok=True)


def load_macro_prediction_frame(horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    run_dir = XGB_RUN_ROOT / f"fwd_{horizon_days}_days"
    pred_path = run_dir / "option2_change_deap_holdout_oos_predictions.csv"
    joined_path = run_dir / "sora_joined_to_xgb_pipeline.csv"
    pred_df = pd.read_csv(pred_path, parse_dates=["snapshot_ts", "fomc_decision_date"])
    joined_df = pd.read_csv(joined_path, parse_dates=["snapshot_ts", "fomc_decision_date"])
    joined_cols = [
        "snapshot_ts",
        "sora_level_realized",
        f"sora_fwd_{horizon_days}d_change",
        f"sora_fwd_{horizon_days}d_level",
    ]
    macro_df = pred_df.merge(joined_df[joined_cols], on="snapshot_ts", how="left")
    macro_df["predicted_level"] = macro_df["sora_level_realized"] + macro_df["y_pred"]
    return macro_df.sort_values("snapshot_ts").reset_index(drop=True)


def compute_sora_distress_score(predicted_change: float) -> float:
    clipped = max(-0.50, min(0.50, float(predicted_change)))
    return max(0.0, min(1.0, 0.5 + clipped))


def compute_final_distress_score(
    distress_score_mamdani: float,
    distress_sora: float,
    refi_risk: float | None,
) -> float:
    sensitivity = 0.50
    if refi_risk is not None and not pd.isna(refi_risk):
        sensitivity = max(0.25, min(1.0, float(refi_risk) * 2.5))
    final_score = 0.80 * distress_score_mamdani + 0.20 * distress_sora * sensitivity
    return max(0.0, min(1.0, float(final_score)))
