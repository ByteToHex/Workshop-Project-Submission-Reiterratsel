# REITterratsel Evaluation

This folder contains structured evaluation outputs and scripts for the REITterratsel hybrid distress stack.

Current script:

- `build_reitteratsel_eval.py`

Current outputs:

- each evaluation run writes into a fresh numbered folder:
  - `run_1`
  - `run_2`
  - `run_3`
  - etc.
- each `run_n` folder contains:
  - `reitteratsel_eval_detail.csv`
  - `reitteratsel_eval_summary.csv`
  - `reitteratsel_eval_disagreements.csv`
  - `reitteratsel_eval_confusion_matrices.csv`
  - `reitteratsel_eval_per_class_metrics.csv`
  - `reitteratsel_eval_ranking_metrics.csv`

The evaluation follows `Design_v1a.txt` section `D - Evaluation`:

- compare `distress_baseline`
- compare `distress_score_mamdani`
- compare `distress_score_refi`
- compare `final_distress`
- evaluate discrete label agreement against `label_126wd`
- evaluate continuous score gap against normalized `CAR_126wd`
- export disagreement cases for structured review
- include confusion matrices, per-class precision/recall/F1, macro F1, MCC, MAE, RMSE, and ranking metrics such as `P@K` and `MAP@K`
