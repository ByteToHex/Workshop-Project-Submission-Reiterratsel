# Current Implementation Schema Reference

This document captures the current live schema used by the REITterratsel implementation as observed from:

- `Common\Micro\IO\out\_annual_warehouse\fundamentals.duckdb`
- `Common\Micro\5_Model_KG\reitteratsel_core.py`
- `Common\Frontend\reitteratsel_app.py`

The shards written to in the annual warehouse for the mamdani pipeline are currently as follows:
- `distresslabels.parquet`
- `fuzzycache.parquet`

It is intended as a practical reference for the current implementation, not as a historical design note.

## Scope

This document focuses on the tables and fields actively used by the current build and app flow:

1. Annual metric anchor
2. Downstream label table
3. Downstream Mamdani cache table
4. App/runtime usage pattern

## Canonical Annual Anchor

The current annual anchor is:

- `reit_metrics.dim_period`
- `reit_metrics.fact_metric_value`

The app and Mamdani pipeline treat these as the frozen annual source data.

### `reit_metrics.dim_period`

Current columns:

- `period_id` `BIGINT`
- `ticker` `VARCHAR`
- `source_period_label` `VARCHAR`
- `period_kind` `VARCHAR`
- `fiscal_year` `INTEGER`
- `fiscal_year_end_month` `INTEGER`
- `fiscal_year_end_year` `INTEGER`
- `fiscal_year_end_date` `DATE`
- `display_year` `INTEGER`
- `sort_key` `INTEGER`
- `is_annual` `BOOLEAN`
- `is_ttm` `BOOLEAN`
- `is_current` `BOOLEAN`

Current role:

- one row per ticker-period
- provides the fiscal-year-end anchor date
- is used to order periods and join annual metrics to downstream outputs

### `reit_metrics.fact_metric_value`

Current columns:

- `ticker` `VARCHAR`
- `period_id` `BIGINT`
- `metric_code` `VARCHAR`
- `metric_value` `DOUBLE`
- `value_text` `VARCHAR`
- `calc_status` `VARCHAR`
- `calc_version` `VARCHAR`
- `asof_ts` `TIMESTAMP`
- `source_period_label` `VARCHAR`
- `notes` `VARCHAR`

Current role:

- long-form annual metric fact table
- one row per ticker-period-metric_code
- current live metric universe is `19` metric codes
- raw annual metrics remain separate from label and fuzzy outputs

Important current implementation notes:

- `NULL_COUNT` is not stored here as a metric row.
- `NON_OK_COUNT` is not stored here as a metric row.
- These are derived from `calc_status` and only persisted downstream in derived tables.

## Downstream Label Output

### `reit_labels.fact_distress_label`

Current columns:

- `ticker` `VARCHAR`
- `period_id` `BIGINT`
- `anchor_date` `DATE`
- `anchor_trade_date` `DATE`
- `window_63_end_date` `DATE`
- `window_126_end_date` `DATE`
- `car_63wd` `DOUBLE`
- `car_126wd` `DOUBLE`
- `null_count` `INTEGER`
- `non_ok_count` `INTEGER`
- `label_scheme_version` `VARCHAR`
- `label_126wd` `VARCHAR`
- `source_index_code` `VARCHAR`
- `asof_ts` `TIMESTAMP WITH TIME ZONE`
- `notes` `VARCHAR`

Current role:

- one downstream label row per ticker-period
- stores the annual anchor date plus forward return window outputs
- stores derived missingness diagnostics
- stores the final label derived from `car_126wd`

Current implementation behavior:

- `anchor_date` comes from `dim_period.fiscal_year_end_date`
- `anchor_trade_date` is the first available trading day on or after `anchor_date`
- `car_63wd` and `car_126wd` are computed from forward abnormal returns after the anchor trading day
- `label_126wd` is derived from `car_126wd`

Important note on incomplete labels:

- some rows currently have `label_126wd = NULL`
- this is not because the schema is missing
- it is because the forward 126-trading-day window is unavailable for some rows, or no trading data exists on/after the anchor date

Observed current causes:

- `Insufficient forward window for 126 trading days.`
- `No trading data on or after anchor date.`

## Downstream Mamdani Cache

### `reit_fuzzy.fact_fuzzy_cache`

Current columns:

- `ticker` `VARCHAR`
- `period_id` `BIGINT`
- `rule_version` `VARCHAR`
- `score_version` `VARCHAR`
- `distress_score_mamdani` `DOUBLE`
- `distress_level` `VARCHAR`
- `null_count` `INTEGER`
- `non_ok_count` `INTEGER`
- `fired_rule_count` `INTEGER`
- `top_rule_ids` `VARCHAR`
- `rule_trace_text` `VARCHAR`
- `asof_ts` `TIMESTAMP WITH TIME ZONE`
- `notes` `VARCHAR`

Current role:

- one downstream fuzzy score row per ticker-period
- stores persisted Mamdani output for auditability and app consumption
- carries derived diagnostics and compact rule-trace fields

## Current App Usage Pattern

The current app components are:

- `Common\Micro\5_Model_KG\build_reitteratsel_pipeline.py`
- `Common\Micro\5_Model_KG\reitteratsel_core.py`
- `Common\Frontend\reitteratsel_app.py`

Current behavior:

- annual metrics are loaded from DuckDB as frozen annual inputs
- labels are loaded from `reit_labels.fact_distress_label`
- Mamdani scores are loaded from `reit_fuzzy.fact_fuzzy_cache`
- macro predictions are loaded from `run_21` artifacts in Python, not from DuckDB
- the Streamlit app uses the latest available macro snapshot at runtime

Important implication:

- the annual layer is frozen by ticker-period
- the macro layer is currently applied using the single latest available snapshot in the app
- this is not yet a modular "query any daily date on the fly" design

## Current Schema Summary

The current implementation schema is already defined and operational.

The practical canonical structure is:

1. Raw annual metrics:
   `reit_metrics.dim_period` + `reit_metrics.fact_metric_value`
2. Derived label output:
   `reit_labels.fact_distress_label`
3. Derived Mamdani cache:
   `reit_fuzzy.fact_fuzzy_cache`

That means the main schema gap is not "missing schema definition" anymore.
The main remaining limitation is runtime modularity of the daily macro layer and the incomplete forward-window coverage for the latest annual anchors.
