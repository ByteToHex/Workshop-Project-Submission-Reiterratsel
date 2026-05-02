`Note: Please focus on user instruction FIRST as a priority.`

# Implementation Checklist v1a

## Status Key

- `[x]` Done
- `[~]` Partial / provisional
- `[ ]` Not done

## Pass 1

### A - Labelling

- `[x]` Read annual inputs from live DuckDB instead of trusting the older draft doc.
- `[x]` Aligned metric-universe references to the implemented warehouse reality of `20` metrics with `REAL_YIELD_CPI` intentionally omitted.
- `[x]` Treated `null_count` as derived, not as a fake stored ratio.
- `[x]` Derived `non_ok_count` separately as a broader annual diagnostic.
- `[x]` Built abnormal return pipeline using ticker `close` returns minus `SGX_DLY_REIT`.
- `[x]` Ignored `REITN` and `REITR`.
- `[x]` Used return-based comparison instead of rescaling raw index levels.
- `[x]` Anchored forward windows to `dim_period.fiscal_year_end_date`.
- `[x]` Rolled anchor to the first available trading day on or after the annual anchor date.
- `[x]` Computed `CAR_63wd`.
- `[x]` Computed `CAR_126wd`.
- `[x]` Created DuckDB table `reit_labels.fact_distress_label`.
- `[x]` Created parquet shard `distresslabels.parquet`.
- `[x]` Added `label_126wd` using implemented strict threshold logic in code.
- `[x]` Persisted label versioning via `label_scheme_version`.
- `[~]` Thresholds are implemented and versioned, but not yet systematically tuned beyond the current `-15% / +5%` scheme.

### B - Mamdani

- `[x]` Defined implemented Mamdani inputs: `ICR`, `GEARING`, `DSCR`, `REFI_RISK`, `PAYOUT_RATIO`, `FFO_COVERAGE`, `NET_DEBT_EBITDA`, `NULL_COUNT`.
- `[x]` Implemented status-aware preprocessing using `calc_status`.
- `[x]` Added special handling for `NEGATIVE_BASE`.
- `[x]` Added special handling for `DISTRESS_BASE`.
- `[x]` Added lower-confidence handling for `LOW_DENOMINATOR`, `PARTIAL`, `CLIPPED_SOURCE_SHARE`.
- `[x]` Implemented `NULL_COUNT` as a fuzzy input derived from `MISSING_INPUT` annual statuses.
- `[x]` Seeded first-pass membership boundaries in `mamdani_rule_seed.json`.
- `[x]` Seeded Neo4j rule graph.
- `[x]` Built Python Mamdani inference.
- `[x]` Added confidence-aware defuzzification blending toward neutral for weak activations.
- `[x]` Created DuckDB table `reit_fuzzy.fact_fuzzy_cache`.
- `[x]` Created parquet shard `fuzzycache.parquet`.
- `[x]` Kept Mamdani persistence separate from `metrics.parquet` and financial shards.
- `[x]` Confirmed `non_ok_count` is persisted as a diagnostic only and is not currently a Mamdani rule input.
- `[~]` Fuzzy calibration is usable but still provisional rather than final.

### C - Daily CAR Path Overlay

- `[x]` Built `reit_labels.fact_car_path_daily`.
- `[x]` Persisted daily `accum_car_to_date`.
- `[x]` Persisted daily `car_path_distress`.
- `[x]` Kept daily CAR-path overlay separate from the annual Mamdani rule engine.
- `[x]` Used the CAR-path overlay later in final runtime distress scoring.

### D - Dashboard

- `[x]` Created a working Streamlit app entrypoint.
- `[x]` Wired the app to DuckDB annual metrics, label outputs, and Mamdani cache outputs.
- `[x]` Added sidebar navigation for `Ranking`, `Individual REIT Navigator`, and `Time Series (Rates)`.
- `[x]` Implemented simulation-date resolution against annual rows, macro snapshots, and daily CAR-path rows.
- `[x]` Kept REIT detail content under per-REIT score and financial tabs.
- `[x]` Added date persistence across pages.
- `[x]` Added durable ticker persistence across navigation using a non-widget session-state key.
- `[x]` Tightened the page-top control layout using compact column-constrained selectors instead of full-width controls.
- `[x]` Added clearer source/provenance labels and help text across annual, macro, and CAR-path fields.
- `[x]` Added ranking columns for `ICR`, `GEARING`, `DSCR`, `Top Revenue Geography`, `% Of Revenue`, and adjacent annual risk fields.
- `[x]` Kept app rule explanation runtime-local by reading persisted `rule_trace_text` from DuckDB rather than querying Neo4j live in the UI.
- `[~]` UI is functional and materially improved, but still not a final fidelity implementation.

### E - Evaluation

- `[x]` Created label-ground-truth basis from `CAR_126wd`.
- `[x]` Built formal evaluation script outputs under `Common\Eval\IO\run_n`.
- `[x]` Wrote summary metrics output.
- `[x]` Wrote per-class metrics output.
- `[x]` Wrote confusion matrices output.
- `[x]` Wrote ranking metrics output.
- `[x]` Wrote disagreement export output.
- `[x]` Wrote row-level detailed evaluation output.
- `[~]` Evaluation exists and is usable, but downstream rule/weight tuning is still not finalized.

## Pass 2

### Environment / Runtime

- `[x]` Confirmed Neo4j connectivity from `.env`.
- `[x]` Confirmed Anaconda env `env` contains the required Python dependencies including `xgboost`.
- `[x]` Updated repo guidance to use Anaconda env `env`.

### Macro / XGBoost

- `[x]` Replaced cached macro-only fallback with direct XGBoost inference when running in `env`.
- `[x]` Loaded the saved `run_21` model artifacts directly.
- `[x]` Reproduced the engineered feature path required by the saved model.
- `[x]` Kept `10D` as the first target.
- `[x]` Kept `15D` modular using the same inference path.
- `[x]` Integrated macro runtime fields into the app using simulation-date resolution.
- `[~]` Macro layer is implemented, but business-level calibration of the macro contribution is still provisional.

### Mamdani Calibration

- `[x]` Reviewed label alignment against `fact_distress_label`.
- `[x]` Tightened some aggressive membership boundaries.
- `[x]` Softened some corroboration rule impact.
- `[x]` Rebuilt fuzzy cache after second-pass changes.
- `[~]` Calibration improved but is not final.

### App / Runtime Check

- `[x]` Smoke-tested the Streamlit entrypoint under the intended runtime.
- `[x]` Confirmed macro prediction source is `xgboost_final_model`.
- `[x]` Confirmed the app now resolves macro and annual rows on or before the selected simulation date.
- `[~]` App still needs full browser-level visual QA and interactive QA across more user flows.

## Not Done Yet

### Data / Labels

- `[ ]` Tune `CAR_126wd` thresholds from the current placeholder-style `-15% / +5%` scheme to a final agreed calibration if needed.
- `[ ]` Add a formal externalized label-threshold config layer if needed.
- `[ ]` Build optional clustering checks from the older design note if still wanted.

### Mamdani / Neo4j

- `[ ]` Perform deeper systematic calibration using the formal evaluation outputs rather than manual rule tightening only.
- `[ ]` Decide whether to incorporate `non_ok_count` or a derived excess diagnostic into scoring, rather than leaving it display-only.
- `[ ]` Add richer audit fields for per-rule membership strengths if you want them persisted structurally instead of only text trace.

### Evaluation

- `[ ]` Compare alternative treatments for `non_ok_count` using the existing evaluation pipeline.
- `[ ]` Re-run evaluation after any future Mamdani or final-distress calibration changes.
- `[ ]` Add a cleaner writeup layer over the current raw evaluation exports if needed for reporting.

### Dashboard / Frontend

- `[ ]` Do a full browser QA pass with actual `streamlit run`.
- `[ ]` Improve fidelity against the Figma/PDF layout where still rough.
- `[ ]` Add richer rule-firing visualization instead of text-only trace.
- `[ ]` Strengthen responsive handling and narrow-width layouts.

### Hybrid Final Distress

- `[ ]` Finalize macro and CAR-path weighting using systematic outcome comparisons instead of provisional heuristics only.
- `[ ]` Validate whether `REFI_RISK` remains the right sensitivity bridge for macro stress.
- `[ ]` Revisit the CAR-path dead-zone and weighting after additional evaluation passes.

## Current Bottom Line

- `[x]` Backend pipeline exists.
- `[x]` Neo4j rule store exists.
- `[x]` DuckDB label, daily CAR-path, and fuzzy cache tables exist.
- `[x]` Separate parquet shards exist for label, CAR-path, and fuzzy-cache outputs.
- `[x]` Direct XGBoost inference exists in the intended environment.
- `[x]` Streamlit app exists with ranking, REIT navigator, and rates pages.
- `[x]` Formal evaluation outputs exist under `Common\Eval\IO\run_n`.
- `[~]` Scoring logic is usable and documented against implementation, but not final.
- `[ ]` Final calibration and full visual QA are still pending.
