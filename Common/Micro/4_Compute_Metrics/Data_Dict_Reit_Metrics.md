# Data Dictionary: REIT Metrics

Source metric list:
[MetricsCalc_TickersChosen.txt](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/01_Serializer/003_SerializeToCSV/sample/MetricsCalc_TickersChosen.txt)

Current implementation:
[build_reit_metrics.py](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/01_Serializer/003b_Metrics/build_reit_metrics.py)

## Scope

This file explains how each requested metric is currently handled in `build_reit_metrics.py`.

## Read This First

Two metrics need special attention before reading the rest of the dictionary.

### REAL_YIELD_SORA is stored as percentage-point spread

- Metric code: `REAL_YIELD_SORA`
- Unit type: `pct_point`
- Meaning:
  - this is a spread between two percentage values
  - it is not stored as a decimal ratio
- Example:
  - dividend yield = `5.30`
  - SORA 3M = `1.19`
  - stored result = `4.11`
- Why this matters:
  - `4.11` here means `4.11 percentage points`
  - it does **not** mean `411%`
  - it also does **not** mean decimal ratio `0.0411`
- Reason for current handling:
  - storing it in percentage-point form preserves the business meaning of a spread
  - this keeps it conceptually separate from decimal ratios like `GEARING = 0.41`

### CAPEX_DRAG is a boolean-style flag stored in a shared numeric metric column

- Metric code: `CAPEX_DRAG`
- Warehouse `metric_value` stores:
  - `1.0` = `TRUE`
  - `0.0` = `FALSE`
- Warehouse `value_text` stores:
  - `TRUE`
  - `FALSE`
- Why this matters:
  - this is not a continuous ratio
  - it is a yes/no diagnostic
- Reason for current handling:
  - DuckDB can store native booleans
  - but this warehouse uses one shared `DOUBLE`-based metric value field for many metric types
  - so the raw warehouse keeps `1.0` / `0.0` for schema uniformity
  - human-facing exports should prefer `value_text`

General rules used by the script:

- Annual metric periods are based on detailed annual labels such as `2024 / Dec 2024` or `2024 / Mar 2025`.
- Most ratios are computed with `_safe_div(numerator, denominator)`.
- If either leg is missing, or denominator is zero, result becomes `NULL`.
- `NULL` outputs are usually stored with `calc_status = 'MISSING_INPUT'`.
- Some metrics can be stored with a non-`OK` status even when a numeric value exists, if the script judges the ratio as semantically unstable or lightly corrected.
- All metric outputs are stored in `reit_metrics.fact_metric_value`.
- Supporting legs and diagnostics are stored in `reit_metrics.fact_metric_component`.

## Status Meanings

- `OK`: calculation completed normally.
- `MISSING_INPUT`: one or more required inputs were missing or unusable.
- `PARTIAL`: a value was produced, but a fallback or imperfect source alignment was used.
- `NEGATIVE_BASE`: a numeric value was produced, but the numerator/base profitability leg is negative, so the ratio should be read as a distress-style signal rather than a normal ratio.
- `DISTRESS_BASE`: a numeric value was produced, but `FFO <= 0`, so payout-style coverage metrics are not behaving like normal payout/coverage ratios.
- `LOW_DENOMINATOR`: a numeric value was produced, but the denominator is very small relative to revenue, so the ratio is numerically unstable and can look extreme.
- `CLIPPED_SOURCE_SHARE`: a numeric value was produced for a share metric, but the raw share slightly exceeded `1.0` due to source mismatch, so it was clipped to `1.0`.

## Metric Dictionary

### ICR

- Full name: `Interest Coverage Ratio`
- Metric code: `ICR`
- Source intent from spec: `EBITDA / Total Interest Expense`
- Implemented formula: `EBITDA / abs(Total interest expense (banks))`
- Source labels:
  - `EBITDA`
  - `Total interest expense (banks)`
- Notes:
  - Interest expense is stored as negative in many cases, so the script uses absolute value.
  - If either leg is missing, result is `NULL`.
  - If `EBITDA < 0`, the numeric result is preserved but status becomes `NEGATIVE_BASE`.

### GEARING

- Full name: `Gearing Ratio`
- Metric code: `GEARING`
- Source intent from spec: `Debt to Assets Ratio`
- Implemented formula:
  - Primary: `Total debt / Total assets`
  - Fallback: use existing warehouse label `Debt to assets ratio`
- Source labels:
  - `Total debt`
  - `Total assets`
  - fallback: `Debt to assets ratio`
- Notes:
  - If primary works, status is `OK`.
  - If fallback is used, status is `PARTIAL`.
  - If neither works, result is `NULL`.

### NET_DEBT_EBITDA

- Full name: `Net Debt / EBITDA`
- Metric code: `NET_DEBT_EBITDA`
- Source intent from spec: `Net Debt / EBITDA`
- Implemented formula: `Net debt / EBITDA`
- Source labels:
  - `Net debt`
  - `EBITDA`
- Notes:
  - No fallback.
  - Missing legs give `NULL`.
  - If `EBITDA <= 0`, the numeric result is preserved but status becomes `NEGATIVE_BASE`.

### REFI_RISK

- Full name: `Refinancing Risk Ratio`
- Metric code: `REFI_RISK`
- Source intent from spec: `Short Term Debt / Total Debt`
- Implemented formula: `Short term debt / Total debt`
- Source labels:
  - `Short term debt`
  - `Total debt`
- Notes:
  - A true zero short-term debt can legitimately produce `0.0`.

### FFO_YOY

- Full name: `FFO YoY Growth`
- Metric code: `FFO_YOY`
- Source intent from spec: `(FFO_t - FFO_t-1) / FFO_t-1`
- Implemented formula: same as spec
- Source labels:
  - `Funds from operations`
- Notes:
  - Uses prior annual period for the same ticker.
  - First available year usually becomes `NULL` because there is no prior comparison year.
  - If prior FFO is zero or missing, result is `NULL`.

### IMPLICIT_COD

- Full name: `Implicit Cost of Debt`
- Metric code: `IMPLICIT_COD`
- Source intent from spec: `Interest Paid / Total Debt`
- Implemented formula: `abs(Interest paid) / Total debt`
- Source labels:
  - `Interest paid`
  - `Total debt`
- Notes:
  - Interest paid is often negative in source data, so the script uses absolute value.

### REV_CONC_TOPSEG

- Full name: `Revenue Concentration (Top Segment Share)`
- Metric code: `REV_CONC_TOPSEG`
- Source intent from spec: `Largest segment revenue / Total revenue`
- Implemented formula:
  - find largest positive segment inside revenue group `0-By_Source`
  - compare that largest segment to a denominator
- Source labels:
  - revenue section rows where `group_output_label = '0-By_Source'`
  - `Total revenue`
- Implemented denominator logic:
  - If segment sum is close to `Total revenue`, use `Total revenue`
  - If segment sum and `Total revenue` disagree materially, use segment sum instead
- Notes:
  - `OK`: segment breakdown and total revenue align well enough
  - `PARTIAL`: segment data exists, but segment sum does not align well with reported total revenue
  - `MISSING_INPUT`: no usable segment values
  - `CLIPPED_SOURCE_SHARE`: raw result slightly exceeded `1.0`, so it was clipped to `1.0`
- Layman meaning:
  - This measures how dependent the REIT is on its biggest revenue segment.

### PAYOUT_RATIO

- Full name: `Payout Ratio`
- Metric code: `PAYOUT_RATIO`
- Source intent from spec: `Total Cash Dividends Paid / FFO`
- Implemented formula: `abs(Total cash dividends paid) / Funds from operations`
- Source labels:
  - `Total cash dividends paid`
  - `Funds from operations`
- Notes:
  - Dividends paid are often negative in source data, so absolute value is used.
  - If `FFO <= 0`, the numeric result is preserved but status becomes `DISTRESS_BASE`.
  - This means the ratio no longer behaves like a normal payout percentage.

### FFO_COVERAGE

- Full name: `FFO Coverage Margin`
- Metric code: `FFO_COVERAGE`
- Source intent from spec: `(FFO - Total Cash Dividends Paid) / FFO`
- Implemented formula: `(FFO - abs(Total cash dividends paid)) / FFO`
- Source labels:
  - `Funds from operations`
  - `Total cash dividends paid`
- Notes:
  - Can be negative if dividends exceed FFO.
  - If `FFO <= 0`, the numeric result is preserved but status becomes `DISTRESS_BASE`.
  - This means the ratio no longer behaves like a normal coverage buffer.

### PNAV_PROXY

- Full name: `P/NAV Proxy (Price to Book)`
- Metric code: `PNAV_PROXY`
- Source intent from spec: `Price to Book Ratio`
- Implemented formula: direct pass-through
- Source labels:
  - `Price to book ratio`
- Notes:
  - No recalculation beyond copying the warehouse value.

### FFO_YIELD_EQ

- Full name: `FFO Yield [Equity]`
- Metric code: `FFO_YIELD_EQ`
- Source intent from spec: `FFO / (Net Income * PE Ratio)`
- Implemented formula: same as spec
- Source labels:
  - `Funds from operations`
  - `Net income`
  - `Price to earnings ratio`
- Notes:
  - This treats `Net income * PE ratio` as a market-cap proxy.
  - If net income is tiny, negative, or missing, this metric can become unstable or `NULL`.

### FFO_YIELD_CAP

- Full name: `FFO Yield [Capital]`
- Metric code: `FFO_YIELD_CAP`
- Source intent from spec: `FFO / Enterprise Value`
- Implemented formula: same as spec
- Source labels:
  - `Funds from operations`
  - `Enterprise value`
- Notes:
  - Missing enterprise value gives `NULL`.

### LEVERAGE_PREMIUM

- Full name: `Leverage Premium`
- Metric code: `LEVERAGE_PREMIUM`
- Source intent from spec: `FFO Yield [Equity] - FFO Yield [Capital]`
- Implemented formula: same as spec
- Source labels:
  - derived from `FFO_YIELD_EQ`
  - derived from `FFO_YIELD_CAP`
- Notes:
  - If either component metric is `NULL`, this metric becomes `NULL`.

### REAL_YIELD_SORA

- Full name: `Real Yield (Opportunity Cost)`
- Metric code: `REAL_YIELD_SORA`
- Source intent from spec: `Dividend Yield (FY) % - SORA Compounded 3M`
- Implemented formula: same as spec
- Source labels/files:
  - `Dividend yield (FY) %`
  - [sora_3m_daily.csv](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/01_Serializer/003b_Metrics/src/sora_3m_daily.csv)
- Notes:
  - For each annual period, the script takes the latest SORA 3M observation on or before fiscal year-end date.
  - Result is stored in percentage-point form, not decimal ratio form.
  - Unit type is explicitly stored as `pct_point`.

### REAL_YIELD_CPI

- Full name: `Real Yield (Inflation)`
- Source intent from spec: `Dividend Yield (FY) % - Expected CPI`
- Current handling: not implemented
- Notes:
  - This metric was explicitly excluded due to lack of availability for consumer CPI numbers. Refer to REAL_YIELD_SORA for real yield computations.
  - It is not present in the current `reit_metrics.dim_metric`.

### UNIT_DILUTION

- Full name: `Unit Dilution Rate`
- Metric code: `UNIT_DILUTION`
- Source intent from spec: `(Diluted Shares_t - Diluted Shares_t-1) / Diluted Shares_t-1`
- Implemented formula: same as spec
- Source labels:
  - `Diluted shares outstanding`
- Notes:
  - Uses prior annual period for the same ticker.
  - First available year usually becomes `NULL`.

### OPEX_INTENSITY

- Full name: `OpEx Intensity`
- Metric code: `OPEX_INTENSITY`
- Source intent from spec: `Total Operating Expenses / Total Revenue`
- Implemented formula: `abs(Total operating expenses) / Total revenue`
- Source labels:
  - `Total operating expenses`
  - `Total revenue`
- Notes:
  - Expense line is often negative in source data, so absolute value is used.

### CAPEX_DRAG

- Full name: `Capex Drag Diagnostic`
- Metric code: `CAPEX_DRAG`
- Source intent from spec: `(Total Cash Dividends Paid / FCF) > (Total Cash Dividends Paid / FFO)`
- Implemented formula:
  - `left = abs(dividends) / free cash flow`
  - `right = abs(dividends) / FFO`
  - output `1.0` if `left > right`, else `0.0`
- Source labels:
  - `Total cash dividends paid`
  - `Free cash flow`
  - `Funds from operations`
- Notes:
  - This is a flag, not a continuous ratio.
  - `1.0` means warning condition is true.
  - `0.0` means warning condition is false.
  - For warehouse uniformity, raw storage remains numeric in `metric_value`.
  - For human readability, `value_text` stores `TRUE` or `FALSE`.
  - Human-facing pivot export should prefer `value_text`.

### NONRECUR_SHARE

- Full name: `Non-Recurring Income Share`
- Metric code: `NONRECUR_SHARE`
- Source intent from spec: `Unusual Income/Expense / Net Income`
- Implemented formula: same as spec
- Source labels:
  - `Unusual income/expense`
  - `Net income`
- Notes:
  - Sign is preserved.
  - This means negative values can be valid and meaningful.
  - If `abs(Net income) < 2% of Total revenue`, the numeric result is preserved but status becomes `LOW_DENOMINATOR`.
  - This means the ratio can look extreme simply because net income is very small.

### DSCR

- Full name: `DSCR`
- Metric code: `DSCR`
- Source intent from spec: `FFO / (Interest Paid + Short Term Debt)`
- Implemented formula: `FFO / (abs(Interest paid) + Short term debt)`
- Source labels:
  - `Funds from operations`
  - `Interest paid`
  - `Short term debt`
- Notes:
  - Interest paid is converted to absolute value before being added to short-term debt.
  - If `FFO < 0`, the numeric result is preserved but status becomes `NEGATIVE_BASE`.

## Null Handling Summary

Most metrics follow the same null rule:

- if numerator is missing -> output `NULL`
- if denominator is missing -> output `NULL`
- if denominator is zero -> output `NULL`

This logic is implemented through `_safe_div(...)`.

Metrics that compare against prior year also become `NULL` when:

- there is no previous annual period
- previous value is missing
- previous value is zero

Some metrics can still produce a numeric value while carrying a warning-style status:

- negative profit/cash base -> `NEGATIVE_BASE`
- non-positive FFO for payout-style metrics -> `DISTRESS_BASE`
- very small net income denominator -> `LOW_DENOMINATOR`
- share metric clipped to 100% -> `CLIPPED_SOURCE_SHARE`

## Mamdani Fuzzy Induction Necessities

These items should be handled in the Mamdani fuzzy pipeline stage, even though the builder now tags edge cases in `calc_status`.

### 1. Use raw value plus status together

- Do not rely on `metric_value` alone for fuzzy input design.
- Use both:
  - raw metric value
  - `calc_status`
- Reason:
  - some values are mathematically valid but semantically unstable
  - examples: `NEGATIVE_BASE`, `DISTRESS_BASE`, `LOW_DENOMINATOR`

### 2. Treat negative-base ratios as distress states, not normal continuous ratios

Applies mainly to:

- `ICR`
- `DSCR`
- `NET_DEBT_EBITDA`

Recommendation:

- if status is `NEGATIVE_BASE`, do not feed the raw number into a normal low/medium/high ratio membership without adjustment
- instead map it into a distress-oriented branch such as:
  - `severe_distress`
  - `broken_profit_base`
  - `coverage_not_meaningful`

Reason:

- a negative coverage-style ratio is not just “slightly lower”
- it usually means the underlying profit/cash base is negative

### 3. Treat payout-style metrics with `FFO <= 0` as distress-base cases

Applies mainly to:

- `PAYOUT_RATIO`
- `FFO_COVERAGE`

Recommendation:

- if status is `DISTRESS_BASE`, do not interpret sign and magnitude as a normal payout or buffer signal
- instead map it into a dedicated stress state such as:
  - `unsustainable_payout_base`
  - `coverage_base_broken`

Reason:

- once `FFO <= 0`, these formulas can flip sign or explode in ways that are mathematically correct but not intuitively smooth

### 4. Clip or saturate extreme denominator-sensitive metrics before membership assignment

Applies especially to:

- `PAYOUT_RATIO`
- `FFO_COVERAGE`
- `NONRECUR_SHARE`
- `UNIT_DILUTION`
- `FFO_YOY`

Recommendation:

- build a rule-ready transformed value separate from the raw warehouse value
- use clipping, winsorizing, or saturation for very large magnitudes

Reason:

- otherwise a few extreme years can dominate the fuzzy memberships and make rules unstable

### 5. Downweight or special-case `LOW_DENOMINATOR`

Applies currently to:

- `NONRECUR_SHARE`

Recommendation:

- if status is `LOW_DENOMINATOR`, reduce confidence in the metric
- possible strategies:
  - downweight the rule contribution
  - assign a weaker membership
  - branch into an “unstable denominator” interpretation

Reason:

- the metric can look huge mainly because the denominator is tiny, not because the numerator is economically huge

### 6. Keep unit families separate in fuzzy preprocessing

Current metric families include:

- binary flag:
  - `CAPEX_DRAG`
- decimal ratios / multiples:
  - `GEARING`, `REFI_RISK`, `ICR`, `DSCR`, `PNAV_PROXY`
- percentage-point spread:
  - `REAL_YIELD_SORA`

Recommendation:

- do not treat all numeric columns as if they live on the same scale
- create separate fuzzy input definitions by metric family

Reason:

- `REAL_YIELD_SORA = 4.11` means `4.11 percentage points`
- `GEARING = 0.41` means `41% as a decimal ratio`
- these are numerically different scales with different business meaning

### 7. Use clipped share values carefully

Applies to:

- `REV_CONC_TOPSEG`

Recommendation:

- if status is `CLIPPED_SOURCE_SHARE` or `PARTIAL`, treat the metric as usable but slightly lower-confidence
- for fuzzy logic, you may still use the clipped `1.0`, but consider:
  - lowering confidence
  - or adding a side flag that source segment alignment was imperfect

Reason:

- the value is now human-safe, but source breakdown mismatch still matters analytically

### 8. Keep raw metrics in the warehouse, transform only in the fuzzy stage

Recommendation:

- do not overwrite the raw warehouse metrics with fuzzy-preprocessed values
- create separate rule-ready transformed inputs later

Reason:

- raw values are still useful for audit, traceability, and alternative rule designs
- fuzzy preprocessing may change over time

### 9. Suggested pattern for downstream design

Recommended downstream pipeline shape:

- raw metric from warehouse
- `calc_status`
- optional transformation / clipping / confidence adjustment
- fuzzy membership assignment
- Mamdani rule induction / inference

This preserves auditability while making the rule engine more stable.

## Output Tables

Main outputs written by `build_reit_metrics.py`:

- `reit_metrics.dim_reit`
- `reit_metrics.dim_period`
- `reit_metrics.dim_metric`
- `reit_metrics.fact_metric_value`
- `reit_metrics.fact_metric_component`
- `reit_metrics.fact_external_series`

Parquet shard written on each refresh:

- [metrics.parquet](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/01_Serializer/003_SerializeToCSV/IO/out/_annual_warehouse/parquet/metrics.parquet)
