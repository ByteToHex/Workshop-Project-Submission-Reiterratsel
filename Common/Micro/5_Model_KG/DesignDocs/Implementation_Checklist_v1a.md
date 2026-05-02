`Note: Please focus on user instruction FIRST as a priority.`

# Implementation Checklist v1a

## Status Key

- `[x]` Done
- `[~]` Partial / provisional
- `[ ]` Not done

## Pass 1

### A - Labelling

- `[x]` Read annual inputs from live DuckDB instead of trusting the draft doc.
- `[x]` Kept A1 aligned to current warehouse reality of `19` metrics.
- `[x]` Treated `null_count` as derived, not as a fake stored ratio.
- `[x]` Built abnormal return pipeline using ticker `close` returns minus `SGX_DLY_REIT`.
- `[x]` Ignored `REITN` and `REITR`.
- `[x]` Used return-based normalization instead of rescaling raw index levels.
- `[x]` Anchored forward windows to `dim_period.fiscal_year_end_date`.
- `[x]` Computed `CAR_63wd`.
- `[x]` Computed `CAR_126wd`.
- `[x]` Created DuckDB table `reit_labels.fact_distress_label`.
- `[x]` Created parquet shard `distresslabels.parquet`.
- `[x]` Added `label_126wd` using configurable threshold logic in code.
- `[~]` Thresholds are still first-pass placeholders and not fully tuned.

### B - Mamdani

- `[x]` Defined first-pass core inputs: `ICR`, `GEARING`, `DSCR`, `REFI_RISK`.
- `[x]` Defined supporting inputs: `PAYOUT_RATIO`, `FFO_COVERAGE`, `NET_DEBT_EBITDA`, `NULL_COUNT`.
- `[x]` Implemented status-aware preprocessing using `calc_status`.
- `[x]` Added special handling for `NEGATIVE_BASE`.
- `[x]` Added special handling for `DISTRESS_BASE`.
- `[x]` Added lower-confidence handling for `LOW_DENOMINATOR`, `PARTIAL`, `CLIPPED_SOURCE_SHARE`.
- `[x]` Seeded first-pass membership boundaries.
- `[x]` Seeded Neo4j rule graph.
- `[x]` Built Python Mamdani inference.
- `[x]` Created DuckDB table `reit_fuzzy.fact_fuzzy_cache`.
- `[x]` Created parquet shard `fuzzycache.parquet`.
- `[x]` Kept Mamdani persistence separate from `metrics.parquet` and financial shards.
- `[~]` First-pass fuzzy calibration was intentionally provisional.

### C - Dashboard

- `[x]` Created a first working Streamlit app entrypoint.
- `[x]` Wired the app to DuckDB outputs.
- `[x]` Wired the app to the Mamdani cache outputs.
- `[x]` Added ranking, score, financial, and time-series tabs.
- `[x]` Used the provided design direction instead of the basic sample only.
- `[~]` UI is functional but still first-pass and not a final fidelity implementation.

### D - Evaluation

- `[x]` Created label-ground-truth basis from `CAR_126wd`.
- `[~]` Did basic spot checks on score behavior.
- `[ ]` Did not build a formal evaluation report table yet.
- `[ ]` Did not build full disagreement-analysis outputs yet.

## Pass 2

### Environment / Runtime

- `[x]` Confirmed Neo4j connectivity from `.env`.
- `[x]` Confirmed Anaconda env `env` contains `xgboost`.
- `[x]` Updated repo guidance to use Anaconda env `env`.

### Macro / XGBoost

- `[x]` Replaced cached macro-only fallback with direct XGBoost inference when running in `env`.
- `[x]` Loaded `option2_change_final_model_xgb.json` directly.
- `[x]` Reproduced the engineered feature path required by the saved model.
- `[x]` Kept `10D` as first target.
- `[x]` Kept `15D` modular using the same inference path.
- `[~]` Macro layer still needs business-level calibration into `distress_sora` / `final_distress`.

### Mamdani Calibration

- `[x]` Reviewed label alignment against `fact_distress_label`.
- `[x]` Tightened some aggressive membership boundaries.
- `[x]` Softened some corroboration rule impact.
- `[x]` Added confidence-aware defuzzification blending toward neutral for weak activations.
- `[x]` Rebuilt fuzzy cache after second-pass changes.
- `[~]` Calibration improved but is not final.

### App / Runtime Check

- `[x]` Smoke-tested the Streamlit entrypoint under the intended runtime.
- `[x]` Confirmed macro prediction source is now `xgboost_final_model`.
- `[~]` App still needs full browser-level visual QA and interactive QA.

## Not Done Yet

### Data / Labels

- `[ ]` Tune `CAR_126wd` thresholds from placeholder values to final agreed values.
- `[ ]` Add a formal versioned label-threshold config layer if needed.
- `[ ]` Build optional clustering checks from the design doc.

### Mamdani / Neo4j

- `[ ]` Perform a deeper second-stage calibration using systematic evaluation instead of manual rule tightening only.
- `[ ]` Add richer audit fields for per-rule membership strengths if you want them persisted structurally instead of only text trace.
- `[ ]` Decide whether to persist separate normalized target comparisons for evaluation.

### Evaluation

- `[ ]` Build a formal evaluation output table comparing:
- `[ ]` `distress_baseline`
- `[ ]` `distress_score_mamdani`
- `[ ]` `final_distress`
- `[ ]` Add discrete label-accuracy metrics.
- `[ ]` Add continuous score-vs-CAR gap metrics.
- `[ ]` Add disagreement-case export for writeup.

### Dashboard / Frontend

- `[ ]` Do a full browser QA pass with actual `streamlit run`.
- `[ ]` Improve fidelity against the Figma/PDF layout.
- `[ ]` Add richer rule-firing visualization instead of text-only trace.
- `[ ]` Add clearer annual-vs-daily provenance labels in the UI.
- `[ ]` Add stronger mobile/responsive handling.

### Hybrid Final Distress

- `[ ]` Finalize how `distress_sora` maps into `final_distress`.
- `[ ]` Validate whether `REFI_RISK` is the right sensitivity bridge for macro stress.
- `[ ]` Tune final hybrid weighting using outcome comparisons instead of provisional heuristics.

## Current Bottom Line

- `[x]` Backend pipeline exists.
- `[x]` Neo4j rule store exists.
- `[x]` DuckDB label and fuzzy cache tables exist.
- `[x]` Separate parquet shards exist.
- `[x]` Direct XGBoost inference exists in the intended environment.
- `[~]` Scoring logic is usable but not final.
- `[ ]` Final calibration and formal evaluation are still pending.
