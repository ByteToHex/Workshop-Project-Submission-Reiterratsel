NOTE: this outline is not final. Intended as a guide doc for the LLM for now. Update it as consolidation changes.

## Context

Refactor outcomes: 
- All stages should use a shared `IO` folder.
- Post-refactor, pipeline stages `1-3` (probing up til extraction) are not explicitly confirmed to work in the new folder structure, as the important information can already be found in the exported CSVs under SRC.
- Post-refactor, pipeline stages `4-5` (postprocessing up til XGBoost training) are explicitly confirmed to work and can be replicated using the feature engineered data found under SRC.
- For clarification on how the expected BPS was assembled before it reached the model training stage (eg. the individual probability legs/constituents or other specific information), please contact me directly (the author of this project).


Pipeline stages:
- `1a` probing: coverage between markets and trades
- `1b` probing: coverage between legacy and modern markets
- `2` preprocessing, including leg assembly
- `3` extraction
- `4` postprocessing
- `5` XGBoost training

Coverage:
- Steps 1-4 include extensive preprocessing of Federal Reserve Funds data (expected BPS and related metrics).
- Step 4 preprocesses Monetary Authority of Singapore (MAS) data.
- Exploratory work not covered here for brevity: CPI-specific scripts/filters, backups, caches, and notes (not used in the final model).

Intermediary CSV types used in the pipeline:
1. `independent`: not exported from the 40GB parquet dump but used. Examples: MAS data, Tradingview time series
2. `passover`: exported directly from the 40GB parquet dump  and then used downstream. Examples: Fed Funds Expected BPS data

## Scope

- `Pipeline_DATA` contains the consolidated copies for Stages 1-3.
- `Pipeline_MODEL` contains the consolidated copies for Stages 4-5.
- Stage 3 is the last stage that still reads the raw 40GB parquet dump.
- Stages 4-5 are downstream of intermediary CSVs only.

## Stage Map

### 1a) Probe markets vs trades

- [probe_trades_schema_by_market.py](./Pipeline_DATA/1a_ProbeMarkets/probe_trades_schema_by_market.py): main market-mapping and market-to-trade coverage probe. Writes `3PROBE_trade_schema.csv`, `3PROBE_market_token_map.csv`, `3PROBE_market_trade_coverage.csv`, `3PROBE_trade_probe_summary.json`.
- [trade_probe_exports.py](./Pipeline_DATA/1a_ProbeMarkets/trade_probe_exports.py): shared helper used by the trade probe exporters.
- [check_trade_timestamp_availability.py](./Pipeline_DATA/1a_ProbeMarkets/check_trade_timestamp_availability.py): timestamp availability check. Writes `3PROBE_TSCHK_timestamp_availability.csv`, `3PROBE_TSCHK_timestamp_availability_summary.json`.
- [check_trade_inference_readiness.py](./Pipeline_DATA/1a_ProbeMarkets/check_trade_inference_readiness.py): inference-readiness QA. Writes the `3PROBE_RDYCHK_*` outputs plus `3PROBE_RDYCHK_inference_readiness_summary.json`.
- [export_ledgers_for_market_time_window.py](./Pipeline_DATA/1a_ProbeMarkets/export_ledgers_for_market_time_window.py): ledger-to-timestamp export for the market window. Writes `4LEDGERTS_WINDOW_ledger_timestamp_map.csv`, `4LEDGERTS_WINDOW_summary.json`.
- Not included: `probe_trades_decimals_collateral_CTF.py`, `check_ledgers_vs_trades_coverage.py`.

### 1b) Probe legacy vs modern markets

- [probe_ctf_legacy_market_overlap.py](./Pipeline_DATA/1b_ProbeLegacy/probe_ctf_legacy_market_overlap.py): classifies modern-only, legacy-only, and overlap markets. Writes `3PROBE_LEGACY_*` outputs.
- [check_legacy_fpmm_trades.py](./Pipeline_DATA/1b_ProbeLegacy/check_legacy_fpmm_trades.py): deep legacy trade coverage and sanity checks. Writes `5CHECK_LEGACY_*` outputs.
- [check_fpmm_sept2022_coverage.py](./Pipeline_DATA/1b_ProbeLegacy/check_fpmm_sept2022_coverage.py): console-only guardrail for the early September 2022 legacy coverage period.

### 2) Preprocessing and leg assembly

- [parquet_fed_events_export.py](./Pipeline_DATA/2_Preprocessing/parquet_fed_events_export.py): exports the Fed/FOMC market universe. Writes `1EXTRACT_fed_parquet_events.csv` and tiered variants.
- [fed_filters.py](./Pipeline_DATA/2_Preprocessing/fed_filters.py): shared Fed/FOMC filter logic.
- [csv_sorting.py](./Pipeline_DATA/2_Preprocessing/csv_sorting.py): helper for end-date CSV ordering.
- [sort_out_csvs.py](./Pipeline_DATA/2_Preprocessing/sort_out_csvs.py) and [sort_out_csvs_end_date.py](./Pipeline_DATA/2_Preprocessing/sort_out_csvs_end_date.py): helper utilities.
- [check_bracket_completeness_gamma.py](./Pipeline_DATA/2_Preprocessing/check_bracket_completeness_gamma.py): validates the Step-01 export against Gamma ground truth. Writes `2VALIDATE_*` outputs and can also trigger some `3PROBE_*` outputs.
- [sanity_check_ctf_market_or_token.py](./Pipeline_DATA/2_Preprocessing/sanity_check_ctf_market_or_token.py): targeted CTF sanity check utility.
- [assemble_bracket_parent_dates.py](./Pipeline_DATA/2_Preprocessing/assemble_bracket_parent_dates.py): adds parent-date columns. Writes `2VALIDATE_bracket_market_tokens_with_parent_dates.csv`.
- [build_leg_map.py](./Pipeline_DATA/2_Preprocessing/build_leg_map.py): adds leg-move and FOMC-date fields. Writes `2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv`.
- [assemble_ref_overlap_parent_columns.py](./Pipeline_DATA/2_Preprocessing/assemble_ref_overlap_parent_columns.py): enriches the overlap CSVs with parent-event columns.

### 3) Extraction

- [build_timeseries.py](./Pipeline_DATA/3_Extraction/build_timeseries.py): core daily event-panel extraction. Writes `timeseries_2022-07-27_2026-03-18.csv` plus intermediary `*_legs.csv`.
- [Trades_CTF.sql](./Pipeline_DATA/3_Extraction/SQL/Trades_CTF.sql): CTF extraction SQL used by `build_timeseries.py`.
- [Trades_FPMM.sql](./Pipeline_DATA/3_Extraction/SQL/Trades_FPMM.sql): legacy FPMM extraction SQL used by `build_timeseries.py`.
- [sanity_check_outcome_index.py](./Pipeline_DATA/3_Extraction/sanity_check_outcome_index.py): console-only guardrail for extraction assumptions.
- [test_count_days.py](./Pipeline_DATA/3_Extraction/test_count_days.py): scratch helper, not a canonical pipeline stage.

### 4) Postprocessing

- [clean_timeseries.py](./Pipeline_MODEL/4_Postprocessing_FED/clean_timeseries.py): removes pre-trade rows. Writes `timeseries_2022-07-27_2026-03-18_clean.csv`.
- [prep_xgb.py](./Pipeline_MODEL/4_Postprocessing_FED/prep_xgb.py): converts the cleaned panel into the model-ready Fed dataset. Writes `timeseries_2022-07-27_2026-03-18_xgb_ready.csv`.
- [parse_mas_sora.py](./Pipeline_MODEL/4_Postprocessing_MAS/parse_mas_sora.py): parses raw MAS SORA downloads. Writes `sora_daily.csv`, `sora_3m_daily.csv`.
- [join_sora_to_xgb.py](./Pipeline_MODEL/4_Postprocessing_MAS/join_sora_to_xgb.py): joins the Fed panel to MAS SORA and REIT data. Writes `sora_joined_to_xgb.csv`.
- [scan_gaps.py](./Pipeline_MODEL/4_Postprocessing_MAS/scan_gaps.py): console-only coverage diagnostic.

### 5) XGBoost training

**Stable/Release Models**
- [train_p_1fold_pipeline.py](./Pipeline_MODEL/5_XGBoost/train_p_1fold_pipeline.py): in-process Model P pipeline that rebuilds the join instead of depending on `sora_joined_to_xgb.csv`. Notable predictive power in terms of SORA direction and some magnitude.
- [train_a_multifold_pipeline.py](./Pipeline_MODEL/5_XGBoost/train_a_multifold_pipeline.py): pooled multifold Model A pipeline. Currently not used as it has low predictive power on abnormal returns despite generalizing well.

**Intermediary/Development Models**
- [train_rstar_xgboost_walkforward_optuna_deap.py](./Pipeline_MODEL/5_XGBoost/train_rstar_xgboost_walkforward_optuna_deap.py): multi-fold Model R* regressor.
- [train_rstar_xgboost_walkforward_optuna_deap_1fold.py](./Pipeline_MODEL/5_XGBoost/train_rstar_xgboost_walkforward_optuna_deap_1fold.py): single-holdout Model R* regressor.
- [train_rstar_directional_1fold.py](./Pipeline_MODEL/5_XGBoost/train_rstar_directional_1fold.py): directional Model R* classifier.
- [train_p_1fold.py](./Pipeline_MODEL/5_XGBoost/train_p_1fold.py): single-holdout Model P trainer.

All Stage-5 scripts are downstream of CSV/table inputs in `Consolidated/IO/SRC` and do not touch `data/parquet/*`.

## upstream_refs

`Pipeline_MODEL` does not use files directly from `IO/SRC/upstream_refs`. It starts from downstream CSVs such as:

- `timeseries_2022-07-27_2026-03-18_xgb_ready.csv`
- `sora_daily.csv`
- `sora_3m_daily.csv`
- `sora_joined_to_xgb.csv`

The main consolidated scripts that do use CSVs from the upstream lineage are:

- [build_timeseries.py](./Pipeline_DATA/3_Extraction/build_timeseries.py): `2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv`, `1EXTRACT_fed_parquet_events.csv`, `Market_Token_Map.csv`, `3PROBE_CTF_market_trade_coverage.csv`
- [sanity_check_outcome_index.py](./Pipeline_DATA/3_Extraction/sanity_check_outcome_index.py): `Market_Token_Map.csv`, `2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv`, `1EXTRACT_fed_parquet_events.csv`
- [check_fpmm_sept2022_coverage.py](./Pipeline_DATA/1b_ProbeLegacy/check_fpmm_sept2022_coverage.py): `2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv`, `1EXTRACT_fed_parquet_events.csv`
- [assemble_ref_overlap_parent_columns.py](./Pipeline_DATA/2_Preprocessing/assemble_ref_overlap_parent_columns.py): `OVERLAP_CTF_only.csv`, `OVERLAP_FPMM_CTF_overlap.csv`, `OVERLAP_FPMM_only.csv`, `2VALIDATE_CTF_bracket_market_tokens_with_parent_dates.csv`

Nuance: before consolidation, similarly named CSVs were sometimes copied into other working folders such as `scripts/util/out_Markets` and `scripts/util/out_Trades`. Those duplicate working copies are not the source of truth for this consolidated outline.

Not found in the consolidated scripts above:

- `fomc_schedule.csv`
- `3PROBE_LEGACY_overlap_summary.json`
- `2VALIDATE_CTF_bracket_market_tokens_with_parent_dates.csv` except via `assemble_ref_overlap_parent_columns.py`

## CSV Lineage

### Independent intermediaries

- Definition used here: intermediary CSVs that are not exported from the 40GB parquet dump, but are still used in the pipeline.
- `sora_daily.csv`, `sora_3m_daily.csv`: from `parse_mas_sora.py`; these are MAS-derived support series used by `join_sora_to_xgb.py`, `train_p_1fold_pipeline.py`, and `train_a_multifold_pipeline.py`.
- TradingView / REIT time-series inputs: also belong to this `independent` bucket conceptually, even if they are not enumerated here as consolidated script outputs.

### Passover intermediaries

- Definition used here: intermediary CSVs exported directly from the 40GB parquet dump, then used downstream in later stages.
- `1EXTRACT_fed_parquet_events.csv`, `1EXTRACT_fed_parquet_events_tier_a.csv`, `1EXTRACT_fed_parquet_events_tier_b.csv`, `1EXTRACT_fed_parquet_events_tier_c.csv`: market-universe exports from raw parquet.
- `3PROBE_market_token_map.csv`, `3PROBE_market_trade_coverage.csv`: parquet-derived trade-coverage / token-map exports.
- In the extraction lineage, the passover files actually consumed are `1EXTRACT_fed_parquet_events.csv`, `Market_Token_Map.csv`, and `3PROBE_CTF_market_trade_coverage.csv`.
- The downstream Fed panel lineage remains part of this same passover side of the pipeline:
  - `2VALIDATE_bracket_market_tokens.csv`
  - `2VALIDATE_parent_markets.csv`
  - `2VALIDATE_bracket_market_tokens_with_parent_dates.csv`
  - `2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv`
  - `OVERLAP_CTF_only.csv`, `OVERLAP_FPMM_CTF_overlap.csv`, `OVERLAP_FPMM_only.csv`
  - `timeseries_2022-07-27_2026-03-18.csv`
  - `timeseries_2022-07-27_2026-03-18_clean.csv`
  - `timeseries_2022-07-27_2026-03-18_xgb_ready.csv`
  - `*_legs.csv`

### Mixed downstream outputs

- `sora_joined_to_xgb.csv`: mixed output. It joins passover-side Fed data with independent MAS / TradingView-style support data.
