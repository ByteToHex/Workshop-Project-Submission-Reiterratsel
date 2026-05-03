## Project Report

## REITterraetsel - Equity Risk Solver for S-REITs

Intelligent Reasoning Systems

Prepared by:

| Student Name | Student ID |
|---|---|
| [Name] | [ID] |
| [Name] | [ID] |
| [Name] | [ID] |
| [Name] | [ID] |

## Table of Contents

| Section | Description |
|---|---|
| 1 | Executive Summary |
| 2 | Business Case / Market Research |
| 3 | System Design / Model |
| 4 | System Development & Implementation |
| 5 | Findings and Discussion |
| 6 | Future Work |
| 7 | References |
| Appendix A | Project Proposal |
| Appendix B | Mapped System Functionalities against MR, RS, CGS Modules |
| Appendix C | Installation and User Guide |

## 1. Executive Summary

*[Complete this last after all other sections are filled.]*

[Write a concise abstract / executive summary here.]

## 2. Business Case / Market Research

## 2.1. Business Case

- Recent sector stress events like rate hike cycles.
  Include discussion of Covid, the US-Iran war, and the Hormuz Closure 2026 as relevant macro stress contexts.
- MAS rule change for gearing restrictions.
  Refer to `SAMPLES\Mine\MAS_Rule_Change_Risk_Implications.txt`.

[Explain why S-REIT distress monitoring matters, why the timing is commercially relevant, and why an early-warning style system is useful.]

## 2.2. Scoping and Competitive Positioning

- Refer to proposal:
  `SAMPLES\Mine\Proposal_Temp_04.md`
- Compare against at least three competitors.
- Preserve the core conclusion from the proposal:
  existing tools often display metrics without reasoning, while this project encodes machine-validated logic and exposes distress rankings programmatically.

[Summarize inclusions, exclusions, target users, and competitive differentiation.]

## 2.3. Literature Review

- Refer to:
  `D:\WS\-GH-A-Ref\REF-Study\GC_ASMT\Project\REF_SELF\IRS\Working\AcademicStudy\Combining\260503_1240_prompt_citations_response.txt`
- Compare against the four research papers and their methods.
- Position this section as detailed market / industry / academic research, with a short note on how it differs from a more product-oriented competitor analysis.

[Summarize the academic baseline, major methods compared, and where this project is similar or different.]

## 3. System Design / Model

## 3.1. Original Design

- Refer to proposal:
  `SAMPLES\Mine\Proposal_Temp_04.md`
- Include the original MermaidJS system diagram.
- Include a short write-up explaining the original intended architecture.

[Insert the original architecture narrative here.]

## 3.2. Data Definitions

## 3.2.1. Labels / Ground Truth

- For the Mamdani fuzzy model, the labels are cumulative abnormal returns:
  `CAR_63d`, `CAR_126d`.

## 3.2.2. Theoretical Benchmark

- The baseline evaluations used in XGBoost `run_21`.
- The baseline used for Mamdani fuzzy comparison.

## 3.2.3. Threshold-Based Label Engineering Rationale

- Why the labels were binarized at the selected threshold.
- Why `CAR_63d` and `CAR_126d` were chosen instead of other horizons.
- Why higher-timeframe windows were preferred due to lower volatility and better signal stability.

[Define what is being predicted, what "distress" means in this project, and why this label design was chosen.]

## 3.3. XGBoost / Macro Data Design

- Use the model terminology from the skeleton:
  `M` = Mamdani Fuzzy Pipeline
  `P` = Parquet-direct macro model
  `A` = Abnormal Returns model
  `R` = Regime model
- Refer to:
  `Common\Macro\Pipeline_MODEL\5_XGBoost\main_scripts.txt`
- Document:
  what macro features are used as inputs,
  what the prediction target is,
  why this feature set was chosen,
  and why the 1-fold constraint was acceptable given the data.
- Do not reproduce code in this section.
- Reserve script-by-script implementation detail for Section 4.

[Explain the macro model design in plain language.]

## 3.4. Mamdani Fuzzy Rules / Micro Data Design

- Refer to annual metrics dictionary:
  `Common\Micro\4_Compute_Metrics\Data_Dict_Reit_Metrics.md`
- Refer to implementation and rule seed:
  `Common\Micro\5_Model_KG\mamdani_rule_seed.json`
  `Common\Micro\5_Model_KG\reitteratsel_core.py`
- Refer to ratio-selection justification:
  `D:\WS\-GH-A-Ref\REF-Study\GC_ASMT\Project\REF_SELF\IRS\Working\Models\KG_Rules\DesignFeature\METRICS_FOR_MAMDANI_RULES.txt`
- Document:
  what financial ratios are encoded as fuzzy inputs,
  what the output membership functions represent,
  and why these ratios were selected over alternatives.
- Keep this section conceptual.
- Reserve script-by-script implementation detail for Section 4.

[Explain the micro / financial design and why these ratios were chosen.]

## 3.5. Full Pipeline / Hybrid Model

- Frame the system as:
  Annual Anchor + Daily Watchdog
- Annual Anchor:
  frozen annual report data and ratio-derived Mamdani rule application.
- Daily Watchdog:
  XGBoost Model `P` (`fwd_10d`) bridged into distress scoring.
- Refer to:
  `Common\Micro\5_Model_KG\DesignDocs\Design_v1a.txt`
  `Common\Micro\5_Model_KG\reitteratsel_core.py`
- Document how the Mamdani layer and macro model output are bridged into one distress score.
- Note any gap between intended design and implemented state.
- Do not reproduce code.

[Describe the hybrid architecture and how the two layers interact.]

## 3.6. UI Prototype

- Figma design:
  `https://www.figma.com/design/XBzg1EQej9Ghf0xkUFVO6p/Reitteratsel?node-id=259-570&t=7kTDq3ZrHpGxBcM9-0`
- Local design references:
  `Common\Frontend\DesignDoc`
- Prefer the PDF artifacts for screenshots or diagrams.

[Summarize the user interface concept, main pages, and design intent.]

## 4. System Development & Implementation

*[Use layman language. Describe the pipeline from data mining to engineering to visualization. For implementation gaps, follow `Design_v1a.txt` and the implementation progress references.]* 

## 4.1. Early Development / Dead Forks

## 4.1.1. Snorkel / Decision Tree / Orange

- Refer to:
  `D:\WS_NUS\REF_DATA\Test_Sample\_Models_Test\Snorkel`
  `IVT_Dividends\approach\DataSelect\Rules\PredictWithLib\Orange\Orange_Inaccurate.txt`
- Mention that these were part of the proposal direction but were not used in the final system.
- State that the small sample size prevented a useful conclusion.
- State that manual rule definition, evaluation against performance, and iterative adjustment became more realistic under the small-N constraint.

**Required statement:** As a result, Snorkel labelling was abandoned and threshold-based labelling was adopted instead. See Section 4.2.

## 4.2. Data Scraping and Data Engineering

- Time series such as MAS and tickers were exported directly via TradingView subscription.
- Self-built firm financial-statement and macro datasets were also created.
- Firm-level financial data came from the `Common\Micro\...` steps 1 to 3 pipeline.
- Macro data parquets came from `Common\Macro\Pipeline_DATA`.
- Note the custom plugin built for scraping firm-level financial data.
- Note the extensive probing, sanity checks, schema confirmation, and final extraction down to 1,227 rows before compression to trading days, with approximately 897 `expected_bps` values.

[Describe how raw data was sourced, validated, and standardized.]

## 4.2.1. Threshold-Based Annual Label Engineering

- Explain the conversion of time-series behavior into annual labels.
- Translate the threshold into plain English, for example:
  the market fell more than X% over Y days.
- Confirm that this is where Snorkel was replaced in practice.

[Explain the fallback from weak labelling to explicit threshold-based labels.]

## 4.3. XGBoost Development

## 4.3.1. Multi-Configuration Controlled Experiment Design

- Implemented early in `train_p_1fold_pipeline.py` and precursor versions.
- Added configurable labels.
  Example directions include SORA magnitude, direction, and related variants.
- Added configurable forward horizons.
- Different timelines tested:
  3, 5, 7, 10, 15, 21.
- Final emphasis settled around 10 to 15 trading days because the horizon was not too noisy yet still carried signal.

[Explain the experiment structure and why multiple configurations were necessary.]

## 4.3.2. Hyperparameter Search Strategy

- Use of Optuna for Bayesian optimization.
- Use of DEAP for evolutionary search.
- Cross-reference findings in Section 5 for why these mattered beyond raw score improvements.

[Explain how automated search was used to improve the model.]

## 4.4. Pipeline and Application Delivery

- Describe the end-to-end movement from data mining to engineered features, trained model artifacts, cached fuzzy outputs, and final dashboard presentation.
- Visually feature the main scripts used across the pipeline with short layman write-ups.

[Summarize the operational pipeline here.]

## 5. Findings and Discussion

## 5.1. Evaluation for XGBoost Best Version

- Main evaluation folder:
  `Common\Macro\IO\Model_Train\Use\run_21`
- Discuss why this is the best version.
- Reference the actual evaluation metrics stored there.

[Summarize best-version performance, strengths, and caveats.]

## 5.2. Evaluation for XGBoost Historical Versions

- Historical comparison folder:
  `Common\Macro\IO\Model_Train\Working`
- Only pull from:
  `shap_summary_bar.png`
  `shap_summary_beeswarm.png`
  statistics JSON files
- Do not pull `final_model_xgb.json`.

[Compare the final model against earlier runs and note what improved.]

## 5.3. Evaluation for Mamdani Layer and Full Pipeline

- Evaluation script:
  `Common\Eval\build_reitteratsel_eval.py`
- Evaluation outputs:
  `Common\Eval\IO\run_n`
- Reference the most recent / canonical evaluation run.
- Mention which metrics are present:
  F1, recall, confusion matrices, per-class metrics, ranking metrics, and related summaries.

[Summarize full-pipeline performance and what it means.]

## 5.4. Representation Tables and Graphs

- Include tables or figures based on:
  `Common\Macro\IO\Model_Train\Use\run_21`
  `Common\Eval\IO\run_n`
- Candidate visuals:
  F1 comparisons,
  recall comparisons,
  confusion matrices,
  model comparison tables.
- Future enhancement note:
  more realistic plots such as residuals and more specific confusion-matrix views can be added later.

[Insert charts, tables, and figure captions here.]

## 5.5. Optuna and DEAP as an Adversarial Error-Surfacing System

- Discuss how the optimization process surfaced design flaws instead of only improving scores.
- Points to cover:
  selection criteria had previously chosen a worse model,
  `n_optuna` trials were too low at 40 and later increased to 80,
  DEAP parameters were too aggressive and harmed generalization,
  mutation-rate settings caused severe instability before being corrected.

[Explain how the search process exposed weaknesses in the modeling pipeline.]

## 5.6. Developed Models and Final Interpretation

- Hybrid model:
  predicts roughly 70 to 80 percent of distressed cases, but is aggressive.
- Note the precision concern:
  around 44 percent precision among cases called distressed.
- Model `P`:
  has foundational signal and further potential.

[Provide the final interpretation of what worked, what did not, and what remains risky.]

## 6. Future Work

- Refer back to:
  `SAMPLES\Mine\Proposal_Temp_04.md`
- Copy forward proposal items that still apply, including the MLAI pipeline if still relevant.
- Add:
  this proof-of-concept repository works off frozen data with explicit date cutoffs and is intentionally designed that way,
  a real-time system would require major changes for automated scraping / API ingestion,
  and an automated DuckDB refresh mechanism is still needed.

[Describe realistic next steps for both research and productionization.]

## 7. References

[Insert references here.]

## Appendix A. Project Proposal

- Decide whether to paste the proposal directly, summarize it, or include selected excerpts.
- Keep the final presentation aligned with the cleaner examples from the sample reports.

[Insert proposal content or a formatted excerpt here.]

## Appendix B. Mapped System Functionalities against MR, RS, CGS Modules

The proposed project must develop, integrate, and demonstrate three or more aspects from the required technique groups.

## Appendix B.1. Decision Automation

- Business rules / knowledge-based reasoning techniques.
- Project mapping:
  Mamdani fuzzy pipeline and its integration into the hybrid model.
- Module links:
  `D:\WS\-GH-A-Ref\REF-Study\GC_ASMT\Project\REF_SELF\IRS\Working\Models\KG_Rules\MamdaniFuzzy\260501_1625_mamdani_fuzzy.txt`
- Suggested mapping:
  MR Day 1 to 3 and RS Day 2, with strongest emphasis on MR Day 2.

[Explain how the project satisfies this requirement.]

## Appendix B.2. Business Resource Optimization

- Informed search / evolutionary computing techniques.
- Project mapping:
  XGBoost evaluation and hyperparameter tuning using sklearn-DEAP and Optuna.
- Module links:
  [Add validated course-slide references here.]

[Explain how optimization techniques were applied.]

## Appendix B.3. Knowledge Discovery and Data Mining

- Project mapping:
  data scraping,
  firm-level financial statements,
  macro data parquets,
  feature engineering,
  feature selection,
  XGBoost models.
- Relevant paths:
  `Common\Micro\...`
  `Common\Macro\Pipeline_DATA`
- Module links:
  [Add validated references for XGBoost, SHAP, and related methods here.]

[Explain how the project satisfies the data-mining requirement.]

## Appendix B.4. Cognitive Techniques / Tools

- Project mapping:
  Neo4j integration with the Mamdani pipeline for real-time inference in the intended design.
- Deployment note:
  for stable development and submission, the project currently uses frozen data plus a DuckDB-facing Mamdani cache layer.
- Original design intent remained live Neo4j KG integration.
- Module links:
  `D:\WS\-GH-A-Ref\REF-Study\GC_ASMT\Project\REF_SELF\IRS\Working\Models\KG_Rules\DesignFeature\References.txt`

[Explain how the project satisfies the cognitive-systems requirement.]

## Appendix C. Installation and User Guide

- Refer to:
  `README.md`
- Align this appendix to the dockerized submission-ready workflow.
- Update only if paths or runtime instructions changed.

[Insert installation and user-guide material here.]
