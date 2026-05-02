from __future__ import annotations

import math
import os
import json
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
CAR_PATH_DAILY_SHARD = PARQUET_DIR / "carpathdaily.parquet"
CSV_TICKER_DIR = ROOT_DIR / "Common" / "Macro" / "IO" / "SRC" / "CSV_TICKER"
ENV_PATH = ROOT_DIR / ".env"
XGB_RUN_ROOT = ROOT_DIR / "Common" / "Macro" / "IO" / "Model_Train" / "Use" / "run_21"
RULE_SEED_PATH = ROOT_DIR / "Common" / "Micro" / "5_Model_KG" / "mamdani_rule_seed.json"

DEFAULT_INDEX_TICKER = "REIT"
DEFAULT_HORIZON_DAYS = 10
LABEL_VERSION = "v1_car126_2026_05_01"
RULE_VERSION = "v1_seed_2026_05_01"
SCORE_VERSION = "v1_mamdani_2026_05_01"
FINAL_DISTRESS_VERSION = "v2_macro_refi_carpath_2026_05_02"
LABEL_THRESHOLDS = {
    "DISTRESSED": -0.15,
    "HEALTHY": 0.05,
}
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


def load_rule_seed_bundle(seed_path: Path = RULE_SEED_PATH) -> dict[str, Any]:
    if not seed_path.exists():
        raise FileNotFoundError(f"Mamdani rule seed not found: {seed_path}")
    with seed_path.open("r", encoding="utf-8") as fh:
        bundle = json.load(fh)
    required_keys = {"model_name", "input_terms", "output_terms", "rules"}
    missing = required_keys.difference(bundle)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise KeyError(f"Mamdani rule seed is missing required keys: {missing_csv}")
    return bundle


def load_neo4j_config(env_path: Path = ENV_PATH) -> Neo4jConfig:
    if not env_path.exists():
        raise FileNotFoundError(f"Neo4j .env not found: {env_path}")
    values = dotenv_values(env_path)
    required_keys = [
        "NEO4J_URI",
        "NEO4J_DATABASE",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
    ]
    missing = [key for key in required_keys if not values.get(key)]
    if missing:
        missing_csv = ", ".join(missing)
        raise KeyError(f"Missing required Neo4j settings in {env_path}: {missing_csv}")
    return Neo4jConfig(
        uri=values["NEO4J_URI"],
        database=values["NEO4J_DATABASE"],
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


def load_distress_label_source_frames(db_path: Path = DUCKDB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the authoritative source inputs for `reit_labels.fact_distress_label`.

    Reads from DuckDB:
    - `reit_metrics.fact_metric_value` to derive `null_count` / `non_ok_count`
    - `reit_metrics.dim_period` to obtain annual ticker-period anchors

    Returns:
    - `counts_df`: per-ticker-period diagnostics derived from metric statuses
    - `period_df`: annual anchor rows keyed by ticker and period

    This function is read-only and does not write any data.
    """
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
    return counts_df, period_df


def derive_distress_label_row(
    period: Any,
    counts_df: pd.DataFrame,
    abnormal_map: dict[str, pd.DataFrame],
    label_version: str = LABEL_VERSION,
) -> dict[str, Any]:
    """
    Derive one row for the downstream `reit_labels.fact_distress_label` table.

    Reads from:
    - one annual period row from DuckDB `reit_metrics.dim_period`
    - derived status-count diagnostics from DuckDB `reit_metrics.fact_metric_value`
    - daily abnormal-return series built from REIT ticker CSVs and the SGX REIT index CSV

    Returns:
    - one label artifact row containing anchor dates, forward CAR windows,
      derived counts, and `label_126wd`

    This function does not write any data directly.
    """
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

    return {
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


def build_distress_label_frame(
    db_path: Path = DUCKDB_PATH,
    label_version: str = LABEL_VERSION,
) -> pd.DataFrame:
    """
    Build the full dataframe for downstream `reit_labels.fact_distress_label`.

    Reads from:
    - DuckDB `reit_metrics.dim_period`
    - DuckDB `reit_metrics.fact_metric_value`
    - REIT daily CSVs
    - SGX REIT index daily CSV

    Returns:
    - a dataframe ready to be written into DuckDB `reit_labels.fact_distress_label`
    - the same rows are later exported to `parquet/distresslabels.parquet`

    This function derives rows only. Persistence happens later in
    `persist_outputs_to_duckdb`.
    """
    counts_df, period_df = load_distress_label_source_frames(db_path)
    abnormal_map = build_abnormal_return_frame(sorted(period_df["ticker"].unique().tolist()))
    rows = [
        derive_distress_label_row(period, counts_df, abnormal_map, label_version=label_version)
        for period in period_df.itertuples()
    ]

    return pd.DataFrame(rows)


def car_to_distress_score(
    car_value: float | None,
    *,
    distressed_threshold: float = LABEL_THRESHOLDS["DISTRESSED"],
    healthy_threshold: float = LABEL_THRESHOLDS["HEALTHY"],
) -> float | None:
    if car_value is None or pd.isna(car_value):
        return None
    car_float = float(car_value)
    if car_float <= distressed_threshold:
        return 1.0
    if car_float >= healthy_threshold:
        return 0.0
    return (healthy_threshold - car_float) / (healthy_threshold - distressed_threshold)


def derive_car_path_daily_rows(
    period: Any,
    abnormal_map: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """
    Derive one ticker-period's daily accumulated CAR path rows.

    Rows are anchored to the first available trading day on or after
    `dim_period.fiscal_year_end_date`. The anchor trading day is stored with
    `accum_car_to_date = 0.0`, and subsequent rows compound abnormal returns
    forward from that anchor.
    """
    ticker_abnormal = abnormal_map[period.ticker]
    anchor_date = pd.Timestamp(period.fiscal_year_end_date).date()
    forward_slice = ticker_abnormal.loc[ticker_abnormal["trade_date"] >= anchor_date].copy()
    if forward_slice.empty:
        return []

    anchor_trade_date = forward_slice.iloc[0]["trade_date"]
    future_returns = forward_slice.loc[forward_slice["trade_date"] > anchor_trade_date].reset_index(drop=True)
    rows = [
        {
            "ticker": period.ticker,
            "period_id": int(period.period_id),
            "anchor_date": anchor_date,
            "anchor_trade_date": anchor_trade_date,
            "trade_date": anchor_trade_date,
            "days_from_anchor": 0,
            "abnormal_return": 0.0,
            "accum_car_to_date": 0.0,
            "car_path_distress": 0.5,
            "notes": None,
        }
    ]
    running_car = 0.0
    for idx, future_row in future_returns.iterrows():
        running_car = float((1.0 + running_car) * (1.0 + float(future_row["abnormal_return"])) - 1.0)
        rows.append(
            {
                "ticker": period.ticker,
                "period_id": int(period.period_id),
                "anchor_date": anchor_date,
                "anchor_trade_date": anchor_trade_date,
                "trade_date": future_row["trade_date"],
                "days_from_anchor": int(idx + 1),
                "abnormal_return": float(future_row["abnormal_return"]),
                "accum_car_to_date": running_car,
                "car_path_distress": float(car_to_distress_score(running_car) or 0.5),
                "notes": None,
            }
        )
    return rows


def build_car_path_daily_frame(db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    """
    Build the full dataframe for downstream `reit_labels.fact_car_path_daily`.

    Reads from:
    - DuckDB `reit_metrics.dim_period`
    - REIT daily CSVs
    - SGX REIT index daily CSV

    Returns:
    - a dataframe ready to be written into DuckDB `reit_labels.fact_car_path_daily`
    - the same rows are later exported to `parquet/carpathdaily.parquet`
    """
    _, period_df = load_distress_label_source_frames(db_path)
    abnormal_map = build_abnormal_return_frame(sorted(period_df["ticker"].unique().tolist()))
    rows: list[dict[str, Any]] = []
    for period in period_df.itertuples():
        rows.extend(derive_car_path_daily_rows(period, abnormal_map))
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


def compute_memberships(
    metric_code: str,
    value: float | None,
    calc_status: str | None,
    input_term_config: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    memberships = {item["term"]: 0.0 for item in input_term_config[metric_code]}
    if calc_status in STATUS_TERM_OVERRIDE.get(metric_code, {}):
        memberships[STATUS_TERM_OVERRIDE[metric_code][calc_status]] = 1.0
        return memberships
    if value is None or pd.isna(value):
        return memberships

    confidence = STATUS_CONFIDENCE.get(calc_status or "OK", 1.0)
    for item in input_term_config[metric_code]:
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


def output_term_membership(term: str, x_value: float, output_term_config: dict[str, dict[str, Any]]) -> float:
    cfg = output_term_config[term]
    return evaluate_term_shape(x_value, cfg["shape"], cfg["params"])


def seed_rules_to_neo4j(driver, config: Neo4jConfig) -> None:
    seed_bundle = load_rule_seed_bundle()
    model_name = seed_bundle["model_name"]
    with driver.session(database=config.database) as session:
        session.run(
            """
            MERGE (model:IRSModel {model_name: $model_name})
            SET model.rule_version = $rule_version,
                model.score_version = $score_version,
                model.updated_at = datetime()
            """,
            model_name=model_name,
            rule_version=RULE_VERSION,
            score_version=SCORE_VERSION,
        )
        session.run(
            """
            MATCH (n)
            WHERE n.model_name = $model_name OR n.rule_version = $rule_version
            DETACH DELETE n
            """,
            model_name=model_name,
            rule_version=RULE_VERSION,
        )
        session.run(
            """
            MERGE (model:IRSModel {model_name: $model_name})
            SET model.rule_version = $rule_version,
                model.score_version = $score_version,
                model.updated_at = datetime()
            """,
            model_name=model_name,
            rule_version=RULE_VERSION,
            score_version=SCORE_VERSION,
        )
        for metric_code, term_defs in seed_bundle["input_terms"].items():
            session.run(
                """
                MATCH (model:IRSModel {model_name: $model_name})
                MERGE (metric:Metric {model_name: $model_name, metric_code: $metric_code})
                SET metric.rule_version = $rule_version
                MERGE (model)-[:HAS_METRIC]->(metric)
                """,
                model_name=model_name,
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
                    model_name=model_name,
                    rule_version=RULE_VERSION,
                    metric_code=metric_code,
                    term=term_def["term"],
                    shape=term_def["shape"],
                    params=term_def["params"],
                )
        for output_term, cfg in seed_bundle["output_terms"].items():
            session.run(
                """
                MATCH (model:IRSModel {model_name: $model_name})
                MERGE (term:OutputTerm {model_name: $model_name, term: $term})
                SET term.rule_version = $rule_version,
                    term.shape = $shape,
                    term.params = $params
                MERGE (model)-[:HAS_OUTPUT_TERM]->(term)
                """,
                model_name=model_name,
                rule_version=RULE_VERSION,
                term=output_term,
                shape=cfg["shape"],
                params=cfg["params"],
            )
        for rule in seed_bundle["rules"]:
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
                model_name=model_name,
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
                    model_name=model_name,
                    rule_id=rule["rule_id"],
                    metric_code=antecedent["metric_code"],
                    term=antecedent["term"],
                )


def fetch_rule_bundle(driver, config: Neo4jConfig) -> dict[str, Any]:
    with driver.session(database=config.database) as session:
        model_names = session.run(
            """
            MATCH (model:IRSModel)
            RETURN model.model_name AS model_name
            ORDER BY model.updated_at DESC
            """
        ).value()
        if not model_names:
            raise RuntimeError(
                f"No IRSModel found in Neo4j database '{config.database}'. Seed the rule graph before scoring."
            )
        model_name = model_names[0]
        input_terms_raw = session.run(
            """
            MATCH (term:InputTerm {model_name: $model_name})
            RETURN
                term.metric_code AS metric_code,
                term.term AS term,
                term.shape AS shape,
                term.params AS params
            ORDER BY term.metric_code, term.term
            """,
            model_name=model_name,
        ).data()
        output_terms_raw = session.run(
            """
            MATCH (term:OutputTerm {model_name: $model_name})
            RETURN
                term.term AS term,
                term.shape AS shape,
                term.params AS params
            ORDER BY term.term
            """,
            model_name=model_name,
        ).data()
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
            model_name=model_name,
        ).data()
    input_terms: dict[str, list[dict[str, Any]]] = {}
    for row in input_terms_raw:
        input_terms.setdefault(row["metric_code"], []).append(
            {
                "term": row["term"],
                "shape": row["shape"],
                "params": row["params"],
            }
        )
    output_terms = {
        row["term"]: {
            "shape": row["shape"],
            "params": row["params"],
        }
        for row in output_terms_raw
    }
    return {
        "model_name": model_name,
        "input_terms": input_terms,
        "output_terms": output_terms,
        "rules": rules,
    }


def build_rule_trace_text(top_rules: list[dict[str, Any]]) -> str | None:
    """
    Build the persisted explanation text for `reit_fuzzy.fact_fuzzy_cache.rule_trace_text`.

    Reads from:
    - top fired-rule rows derived during Mamdani inference
    - rule descriptions and output terms sourced from the Neo4j-backed rule bundle

    Returns:
    - newline-joined explanation text for one fuzzy result row

    This function does not write any data directly.
    """
    if not top_rules:
        return None
    rule_trace_lines = [
        f"{item['rule_id']} ({item['output_term']}, {item['strength']:.3f}): {item['description']}"
        for item in top_rules
    ]
    return "\n".join(rule_trace_lines)


def build_fuzzy_input_frame(db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    """
    Build the authoritative annual source frame for `reit_fuzzy.fact_fuzzy_cache`.

    Reads from:
    - DuckDB `reit_metrics.fact_metric_value`
    - DuckDB `reit_metrics.dim_period`
    - DuckDB `reit_metrics.dim_reit`

    Returns:
    - one wide annual input frame per ticker-period including metric values,
      metric statuses, and derived `null_count` / `non_ok_count`

    This function is read-only and does not write any data.
    """
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
    """
    Derive one row-level Mamdani inference result for the fuzzy cache pipeline.

    Reads from:
    - one annual input row derived from DuckDB metrics
    - one rule bundle read from Neo4j after seeding from `mamdani_rule_seed.json`

    Returns:
    - one row-level fuzzy result including score, level, fired-rule metadata,
      and `rule_trace_text`

    This function derives row outputs only and does not write any data.
    """
    input_term_config = rule_bundle["input_terms"]
    output_term_config = rule_bundle["output_terms"]
    membership_map: dict[tuple[str, str], float] = {}
    trace_memberships: dict[str, dict[str, float]] = {}

    for metric_code in input_term_config:
        if metric_code == "NULL_COUNT":
            memberships = compute_memberships(metric_code, row.get("null_count"), "OK", input_term_config)
        else:
            memberships = compute_memberships(
                metric_code,
                row.get(metric_code),
                row.get(f"{metric_code}_status"),
                input_term_config,
            )
        trace_memberships[metric_code] = memberships
        for term, strength in memberships.items():
            membership_map[(metric_code, term)] = strength

    fired_rules: list[dict[str, Any]] = []
    output_strengths = {term: 0.0 for term in output_term_config}
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
                min(strength, output_term_membership(term, x_value, output_term_config)),
            )
        numerator += x_value * aggregate_membership
        denominator += aggregate_membership
    raw_score = numerator / denominator if denominator else 0.5
    activation_confidence = max(output_strengths.values()) if output_strengths else 0.0
    activation_confidence = max(0.0, min(1.0, activation_confidence))
    score = 0.5 * (1.0 - activation_confidence) + raw_score * activation_confidence
    level = score_to_level(score)
    fired_rules.sort(key=lambda item: item["strength"], reverse=True)
    top_rules = fired_rules[:5]
    return {
        "distress_score_mamdani": float(score),
        "distress_level": level,
        "fired_rule_count": len(fired_rules),
        "top_rule_ids": ",".join(item["rule_id"] for item in top_rules) if top_rules else None,
        "rule_trace_text": build_rule_trace_text(top_rules),
        "trace_memberships": trace_memberships,
    }


def derive_fuzzy_cache_row(row: dict[str, Any], rule_bundle: dict[str, Any]) -> dict[str, Any]:
    """
    Derive one persisted row for downstream `reit_fuzzy.fact_fuzzy_cache`.

    Reads from:
    - one wide annual Mamdani input row derived from DuckDB
    - one Neo4j-backed rule bundle

    Returns:
    - one dict-shaped fuzzy cache row ready for dataframe assembly and later persistence

    This function does not write any data directly.
    """
    result = evaluate_fuzzy_row(pd.Series(row), rule_bundle)
    return {
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
        "notes": None,
    }


def build_fuzzy_cache_frame(
    db_path: Path = DUCKDB_PATH,
    env_path: Path = ENV_PATH,
    rule_bundle: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build the full dataframe for downstream `reit_fuzzy.fact_fuzzy_cache`.

    Reads from:
    - DuckDB annual metric inputs via `build_fuzzy_input_frame`
    - Neo4j rule graph via `fetch_rule_bundle`

    Returns:
    - a dataframe ready to be written into DuckDB `reit_fuzzy.fact_fuzzy_cache`
    - the same rows are later exported to `parquet/fuzzycache.parquet`

    This function derives rows only. Persistence happens later in
    `persist_outputs_to_duckdb`.
    """
    inputs = build_fuzzy_input_frame(db_path)
    if rule_bundle is None:
        driver, config = connect_neo4j(env_path)
        try:
            rule_bundle = fetch_rule_bundle(driver, config)
        finally:
            driver.close()

    rows = [derive_fuzzy_cache_row(row, rule_bundle) for row in inputs.to_dict(orient="records")]
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
    car_path_df: pd.DataFrame,
    db_path: Path = DUCKDB_PATH,
) -> None:
    work_path = ensure_writable_duckdb_copy(db_path)
    con = duckdb.connect(str(work_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS reit_labels")
        con.execute("CREATE SCHEMA IF NOT EXISTS reit_fuzzy")
        con.register("label_df", label_df)
        con.register("fuzzy_df", fuzzy_df)
        con.register("car_path_df", car_path_df)
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
        con.execute(
            """
            CREATE OR REPLACE TABLE reit_labels.fact_car_path_daily AS
            SELECT
                ticker::VARCHAR AS ticker,
                period_id::BIGINT AS period_id,
                anchor_date::DATE AS anchor_date,
                anchor_trade_date::DATE AS anchor_trade_date,
                trade_date::DATE AS trade_date,
                days_from_anchor::INTEGER AS days_from_anchor,
                abnormal_return::DOUBLE AS abnormal_return,
                accum_car_to_date::DOUBLE AS accum_car_to_date,
                car_path_distress::DOUBLE AS car_path_distress,
                current_timestamp AS asof_ts,
                notes::VARCHAR AS notes
            FROM car_path_df
            """
        )
        DISTRESS_LABEL_SHARD.parent.mkdir(parents=True, exist_ok=True)
        if DISTRESS_LABEL_SHARD.exists():
            DISTRESS_LABEL_SHARD.unlink()
        if FUZZY_CACHE_SHARD.exists():
            FUZZY_CACHE_SHARD.unlink()
        if CAR_PATH_DAILY_SHARD.exists():
            CAR_PATH_DAILY_SHARD.unlink()
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
                    c.ticker,
                    r.reit_name,
                    r.sector,
                    r.health_bucket,
                    p.fiscal_year,
                    p.fiscal_year_end_date,
                    c.period_id,
                    c.anchor_date,
                    c.anchor_trade_date,
                    c.trade_date,
                    c.days_from_anchor,
                    c.abnormal_return,
                    c.accum_car_to_date,
                    c.car_path_distress,
                    c.notes
                FROM reit_labels.fact_car_path_daily c
                JOIN reit_metrics.dim_reit r ON r.ticker = c.ticker
                JOIN reit_metrics.dim_period p ON p.period_id = c.period_id
                ORDER BY c.ticker, p.fiscal_year_end_date, c.trade_date
            ) TO '{CAR_PATH_DAILY_SHARD.as_posix()}' (FORMAT PARQUET)
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
    joined_path = run_dir / "sora_joined_to_xgb_pipeline.csv"
    model_path = run_dir / "option2_change_final_model_xgb.json"
    joined_df = pd.read_csv(joined_path, parse_dates=["snapshot_ts", "fomc_decision_date"])
    engineered_df = _build_macro_feature_frame(joined_df)
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    feature_names = booster.feature_names or []
    if not feature_names:
        raise RuntimeError(
            f"XGBoost model at {model_path} does not expose feature names. "
            "Failing fast instead of guessing from a fallback manifest."
        )
    feature_frame = engineered_df.reindex(columns=feature_names)
    missing_features = [name for name in feature_names if name not in engineered_df.columns]
    if missing_features:
        missing_csv = ", ".join(missing_features)
        raise KeyError(
            f"Engineered macro feature frame is missing required XGBoost features: {missing_csv}"
        )
    predictions = booster.predict(xgb.DMatrix(feature_frame, feature_names=feature_names))
    macro_df = engineered_df.copy()
    macro_df["fold"] = pd.NA
    macro_df["target_col"] = f"sora_fwd_{horizon_days}d_change"
    macro_df["y_true"] = macro_df.get(f"sora_fwd_{horizon_days}d_change")
    macro_df["y_pred"] = predictions
    macro_df["y_true_dir"] = (pd.to_numeric(macro_df["y_true"], errors="coerce") > 0).astype("Int64")
    macro_df["y_pred_dir"] = (macro_df["y_pred"] > 0).astype(int)
    macro_df["predicted_level"] = macro_df["sora_level_realized"] + macro_df["y_pred"]
    macro_df["prediction_source"] = "xgboost_final_model"
    return macro_df.sort_values("snapshot_ts").reset_index(drop=True)


def load_macro_train_end(horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.Timestamp:
    run_dir = XGB_RUN_ROOT / f"fwd_{horizon_days}_days"
    metrics_path = run_dir / "option2_change_optuna_holdout_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Holdout metrics CSV not found: {metrics_path}")
    metrics_df = pd.read_csv(metrics_path, parse_dates=["train_end"])
    if metrics_df.empty:
        raise ValueError(f"Holdout metrics CSV is empty: {metrics_path}")
    train_end = metrics_df.loc[0, "train_end"]
    if pd.isna(train_end):
        raise ValueError(f"train_end is missing in holdout metrics CSV: {metrics_path}")
    return pd.Timestamp(train_end)


def load_macro_holdout_frame(horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    run_dir = XGB_RUN_ROOT / f"fwd_{horizon_days}_days"
    comparison_path = run_dir / "all_targets_optimizer_comparison.json"
    joined_path = run_dir / "sora_joined_to_xgb_pipeline.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Optimizer comparison JSON not found: {comparison_path}")
    if not joined_path.exists():
        raise FileNotFoundError(f"Joined macro pipeline CSV not found: {joined_path}")

    with comparison_path.open("r", encoding="utf-8") as fh:
        comparison = json.load(fh)
    option2 = comparison.get("option2_change")
    if option2 is None:
        raise KeyError(f"option2_change block missing in optimizer comparison JSON: {comparison_path}")
    winner = option2.get("winner")
    if winner not in {"optuna", "deap"}:
        raise ValueError(f"Unsupported or missing winner '{winner}' in {comparison_path}")

    oos_path = run_dir / f"option2_change_{winner}_holdout_oos_predictions.csv"
    if not oos_path.exists():
        raise FileNotFoundError(f"Holdout OOS predictions CSV not found: {oos_path}")

    oos_df = pd.read_csv(oos_path, parse_dates=["snapshot_ts", "fomc_decision_date"])
    joined_df = pd.read_csv(joined_path, parse_dates=["snapshot_ts", "fomc_decision_date"])
    joined_df = joined_df.sort_values("snapshot_ts").reset_index(drop=True)
    joined_df["target_date"] = joined_df["snapshot_ts"].shift(-horizon_days)
    merged = oos_df.merge(
        joined_df[
            [
                "snapshot_ts",
                "fomc_decision_date",
                "sora_level_realized",
                f"sora_fwd_{horizon_days}d_level",
                f"sora_fwd_{horizon_days}d_change",
                "target_date",
            ]
        ],
        on=["snapshot_ts", "fomc_decision_date"],
        how="left",
        validate="one_to_one",
    )
    required_cols = [f"sora_fwd_{horizon_days}d_level", f"sora_fwd_{horizon_days}d_change", "target_date"]
    if merged[required_cols].isna().any().any():
        raise ValueError(
            f"Holdout merge left missing future target columns in {oos_path}; "
            f"joined source was {joined_path}"
        )
    merged["predicted_level"] = merged["sora_level_realized"] + merged["y_pred"]
    merged["prediction_source"] = f"{winner}_holdout_oos_predictions"
    return merged.sort_values("snapshot_ts").reset_index(drop=True)


def _build_macro_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("snapshot_ts").reset_index(drop=True).copy()
    out["p_no_change_missing"] = out["p_no_change"].isna().astype(int)
    out["margin_over_second_missing"] = out["margin_over_second"].isna().astype(int)
    for col in ("sora_level_realized", "sora_90d_change_t2", "sora_level_t2", "sora_3m_t2", "expected_bps"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    level = out["sora_level_realized"]
    out["sora_lag_21d_diff"] = level.diff(21)
    out["sora_lag_63d_diff"] = level.diff(63)
    out["sora_lag_10d_diff"] = level.diff(10)
    out["sora_lag_5d_diff"] = level.diff(5)
    d_level = level.diff(1)
    out["sora_realized_vol_21d"] = d_level.rolling(21).std()
    out["sora_realized_vol_63d"] = d_level.rolling(63).std()
    roll_max = level.rolling(63).max()
    out["sora_below_63d_peak"] = level - roll_max
    out["sora_dist_from_63d_ma"] = level - level.rolling(63).mean()
    out["sora_accel_21d"] = level.diff(21) - level.diff(21).shift(21)
    out["expected_bps_minus_sora_90d"] = out["expected_bps"] - out["sora_90d_change_t2"]
    out["sora_curve_steepness"] = out["sora_level_t2"] - out["sora_3m_t2"]
    return out


def compute_sora_distress_score(predicted_change: float) -> float:
    clipped = max(-0.50, min(0.50, float(predicted_change)))
    return max(0.0, min(1.0, 0.5 + clipped))


def compute_refi_distress_score(refi_risk: float | None) -> float | None:
    if refi_risk is None or pd.isna(refi_risk):
        return None
    refi_float = float(refi_risk)
    healthy_cap = 0.15
    critical_floor = 0.55
    if refi_float <= healthy_cap:
        return 0.0
    if refi_float >= critical_floor:
        return 1.0
    return (refi_float - healthy_cap) / (critical_floor - healthy_cap)


def compute_final_distress_score(
    distress_score_mamdani: float,
    distress_sora: float,
    refi_risk: float | None,
    car_path_distress: float | None = None,
) -> float:
    sensitivity = 0.50
    if refi_risk is not None and not pd.isna(refi_risk):
        sensitivity = max(0.25, min(1.0, float(refi_risk) * 2.5))
    macro_shock = float(distress_sora) - 0.5
    car_path_shock = 0.0
    if car_path_distress is not None and not pd.isna(car_path_distress):
        car_path_shock = float(car_path_distress) - 0.5
    final_score = (
        float(distress_score_mamdani)
        + 0.25 * sensitivity * macro_shock
        + 0.35 * car_path_shock
    )
    return max(0.0, min(1.0, float(final_score)))
