## Raw Tradingview Export

Source: D:\WS\-GH-A-Ref\REF-Study\GC_ASMT\Project\REF_SELF\IRS\Working\Data\ForCompany\Check_Schema\Docs\Schema_RawTradingview.txt
```sql
CREATE TABLE schema_rows (
    row_id      INTEGER PRIMARY KEY,
    section     VARCHAR,
    label       VARCHAR,
    depth       INTEGER,
    parent_id   INTEGER REFERENCES schema_rows(row_id),
    group_output_label VARCHAR
);

CREATE TABLE financials (
    ticker   VARCHAR,
    period   VARCHAR,
    currency VARCHAR,
    row_id   INTEGER REFERENCES schema_rows(row_id),
    value    VARCHAR
);
```


## Computed Metrics

Source: D:\WS\-GH-A-Ref\REF-Study\GC_ASMT\Project\REF_SELF\IRS\Working\Data\ForCompany\Check_Schema\Docs\Schema_ComputedMetrics.txt

```sql
CREATE SCHEMA IF NOT EXISTS reit_metrics;

CREATE TABLE reit_metrics.dim_reit (
    ticker              VARCHAR PRIMARY KEY,
    reit_name           VARCHAR,
    sector              VARCHAR,
    health_bucket       VARCHAR,   -- Healthy / Distressed / Control / Unknown
    notes               VARCHAR
);

CREATE TABLE reit_metrics.dim_period (
    period_id                BIGINT PRIMARY KEY,
    ticker                   VARCHAR NOT NULL,
    source_period_label      VARCHAR NOT NULL,   -- e.g. '2024 / Mar 2025', 'TTM', 'Current'
    period_kind              VARCHAR NOT NULL,   -- FY, TTM, CURRENT, CAL_YEAR, OTHER
    fiscal_year              INTEGER,            -- logical reporting year, e.g. 2024
    fiscal_year_end_month    TINYINT,            -- 3, 9, 12, etc.
    fiscal_year_end_date     DATE,               -- if derivable; else NULL
    display_year             INTEGER,            -- optional convenience field
    sort_key                 INTEGER,            -- monotonic per ticker for ordering
    is_annual                BOOLEAN NOT NULL DEFAULT TRUE,
    is_ttm                   BOOLEAN NOT NULL DEFAULT FALSE,
    is_current               BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (ticker, source_period_label)
);

CREATE TABLE reit_metrics.dim_metric (
    metric_code          VARCHAR PRIMARY KEY,    -- e.g. ICR, GEARING, FFO_YOY
    metric_name          VARCHAR NOT NULL,
    formula_short        VARCHAR,
    numerator_desc       VARCHAR,
    denominator_desc     VARCHAR,
    unit_type            VARCHAR NOT NULL,       -- ratio, pct, multiple, currency, flag
    higher_is_better     BOOLEAN,
    source_schema_hint   VARCHAR,                -- SCHEMA_01 etc from your doc
    requires_external    BOOLEAN NOT NULL DEFAULT FALSE,
    description          VARCHAR
);

CREATE TABLE reit_metrics.fact_metric_value (
    ticker               VARCHAR NOT NULL,
    period_id            BIGINT NOT NULL,
    metric_code          VARCHAR NOT NULL,
    metric_value         DOUBLE,
    value_text           VARCHAR,                -- optional if diagnostic / non-numeric
    calc_status          VARCHAR NOT NULL,       -- OK, MISSING_INPUT, PARTIAL, ERROR
    calc_version         VARCHAR NOT NULL,       -- lets you recompute cleanly later
    asof_ts              TIMESTAMP NOT NULL DEFAULT current_timestamp,
    source_period_label  VARCHAR,                -- denormalized convenience copy
    notes                VARCHAR,
    PRIMARY KEY (ticker, period_id, metric_code),
    FOREIGN KEY (ticker) REFERENCES reit_metrics.dim_reit(ticker),
    FOREIGN KEY (period_id) REFERENCES reit_metrics.dim_period(period_id),
    FOREIGN KEY (metric_code) REFERENCES reit_metrics.dim_metric(metric_code)
);
```

For debugging formulas:

```sql
CREATE TABLE reit_metrics.fact_metric_component (
    ticker               VARCHAR NOT NULL,
    period_id            BIGINT NOT NULL,
    metric_code          VARCHAR NOT NULL,
    component_role       VARCHAR NOT NULL,   -- numerator, denominator, input, prior_period_input
    component_name       VARCHAR NOT NULL,   -- EBITDA, Total debt, Interest paid, etc.
    component_value      DOUBLE,
    component_text       VARCHAR,
    source_table         VARCHAR,            -- financials / schema_rows / external_macro / manual
    source_section       VARCHAR,            -- income / balance / cashflow / statistics / dividends
    source_row_id        INTEGER,
    source_label         VARCHAR,
    PRIMARY KEY (ticker, period_id, metric_code, component_role, component_name)
);
```
