-- Coverage: every ticker should have period_count * metric_count rows.
WITH expected AS (
    SELECT
        p.ticker,
        COUNT(*) * (SELECT COUNT(*) FROM reit_metrics.dim_metric) AS expected_rows
    FROM reit_metrics.dim_period p
    GROUP BY p.ticker
),
actual AS (
    SELECT ticker, COUNT(*) AS actual_rows
    FROM reit_metrics.fact_metric_value
    GROUP BY ticker
)
SELECT
    e.ticker,
    e.expected_rows,
    a.actual_rows,
    a.actual_rows = e.expected_rows AS is_complete
FROM expected e
JOIN actual a USING (ticker)
ORDER BY e.ticker;

-- Blank counts by metric and calc status.
SELECT
    metric_code,
    calc_status,
    COUNT(*) AS row_count,
    SUM(CASE WHEN metric_value IS NULL THEN 1 ELSE 0 END) AS blank_value_count
FROM reit_metrics.fact_metric_value
GROUP BY metric_code, calc_status
ORDER BY metric_code, calc_status;

-- Blank counts by ticker.
SELECT
    ticker,
    COUNT(*) AS total_metric_rows,
    SUM(CASE WHEN metric_value IS NULL THEN 1 ELSE 0 END) AS blank_metric_rows,
    ROUND(
        100.0 * SUM(CASE WHEN metric_value IS NULL THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS blank_pct
FROM reit_metrics.fact_metric_value
GROUP BY ticker
ORDER BY ticker;

-- Full metric set for one REIT across all annual periods.
-- Change the ticker in params before running.
WITH params AS (
    SELECT 'A17U' AS target_ticker
)
SELECT
    v.ticker,
    r.reit_name,
    p.sort_key,
    p.fiscal_year,
    p.fiscal_year_end_month,
    p.fiscal_year_end_year,
    p.fiscal_year_end_date,
    p.source_period_label,
    m.metric_code,
    m.metric_name,
    m.unit_type,
    v.metric_value,
    v.value_text,
    v.calc_status,
    v.notes
FROM reit_metrics.fact_metric_value v
JOIN reit_metrics.dim_period p
    ON v.period_id = p.period_id
JOIN reit_metrics.dim_metric m
    ON v.metric_code = m.metric_code
JOIN reit_metrics.dim_reit r
    ON v.ticker = r.ticker
JOIN params x
    ON v.ticker = x.target_ticker
ORDER BY
    p.sort_key,
    m.metric_code;