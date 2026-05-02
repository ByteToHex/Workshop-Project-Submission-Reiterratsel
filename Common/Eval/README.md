# REITterratsel Evaluation

This folder contains structured evaluation outputs and scripts for the REITterratsel hybrid distress stack.

Current script:

- `build_reitteratsel_eval.py`

Current outputs:

- `reitteratsel_eval_detail.csv`
- `reitteratsel_eval_summary.csv`
- `reitteratsel_eval_disagreements.csv`

The evaluation follows `Design_v1a.txt` section `D - Evaluation`:

- compare `distress_baseline`
- compare `distress_score_mamdani`
- compare `final_distress`
- evaluate discrete label agreement against `label_126wd`
- evaluate continuous score gap against normalized `CAR_126wd`
- export disagreement cases for structured review
