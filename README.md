### [ Practice Module ] Project Submission

---

## SECTION 1 : PROJECT TITLE
## REITterratsel: REIT Distress Reasoning System with Mamdani Rules, Macro Overlay, and XGBoost

<img src="Common/Frontend/DesignDoc/Reiterratsel_Wordmark.svg"
     style="float: left; margin-right: 0px;" />

---

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT

This project is a REIT distress reasoning system built around an integrated annual-fundamentals, macro-rates, and rule-based reasoning workflow. 

The system focuses on Singapore-listed REITs (SREITs) and aims to convert messy upstream financial statement, market, and rates data into a dashboard that supports interpretable distress assessment rather than only raw point prediction. The final application serves a ranked view of REIT distress, an individual REIT navigator, and time-series macro context through a Streamlit interface.

At the micro / company layer, the repository builds an authoritative DuckDB warehouse from TradingView-style annual statement data and derived REIT metrics. These metrics are then used to derive distress labels, a Mamdani fuzzy cache, and a daily abnormal-return CAR path layer. 

At the macro layer, the repository loads pre-trained XGBoost model artifacts from `Common/Macro/IO/Model_Train/Use/run_21` and performs runtime inference on SORA-related data so that the app can inject a rate-shock overlay into the final distress score.

The reasoning architecture combines several components rather than treating the problem as a single black-box model. A Neo4j-backed rule graph is used during rebuild mode to seed and fetch the Mamdani rule bundle from `Common/Micro/5_Model_KG/mamdani_rule_seed.json`. Those rules are evaluated into a persisted annual fuzzy cache. During app runtime, the system combines frozen Mamdani output with the latest eligible macro snapshot, `REFI_RISK`-driven sensitivity, and a daily CAR-path layer to produce the final distress ranking displayed in the user interface.

For project submission, the design intentionally ships the committed DuckDB warehouse as a stable snapshot so the application can run in Docker without forcing Neo4j-backed rebuild logic on every launch. For development, rebuild mode remains available so the shipped DuckDB and parquet cache artifacts can be refreshed in place when the underlying derived outputs need to be regenerated.

---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name | Student ID (MTech Applicable) | Work Items (Who Did What) | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
| Jason Tay | A0265092A | End-to-end solo delivery of the REIT distress project, including data pipeline design (for both micro and micro/company-layer), data sourcing, data engineering, DuckDB warehouse build, REIT metric derivation, Mamdani rule modelling, Neo4j rule-graph integration, macro XGBoost integration, Streamlit application development, Docker packaging, evaluation workflow, project documentation, and submission packaging. | tnw.jason@gmail.com |

---

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

`Refer to Github Folder: Video`

At the time of this README update, the `Video` folder exists in the repository root, but no embedded public video link or committed video file was found in this checkout.

---

## SECTION 5 : USER GUIDE

`Refer to appendix <Installation & User Guide> in project report at Github Folder: ProjectReport`

### [ 1 ] To run the system using Docker in submission / demo mode

From the repository root, meaning the folder that contains `README.md`, `Common/`, and the top-level `.git` folder:

```powershell
cd <path-to-this-repo>
```

Use the compose file in two modes:

- `app-only` for submission/demo
- `rebuild` when Neo4j-backed cache regeneration is needed

Important distinction:

- `app-only` serves the app against the committed DuckDB snapshot already shipped in the repo.
- `rebuild` starts Neo4j and reruns `build_reitteratsel_pipeline.py` so the DuckDB/parquet cache is refreshed in place.
- Rebuild is intentionally separate so the normal submission/demo path does not depend on Neo4j every time.

#### App-only mode

This serves the app against the committed DuckDB snapshot and does not need `Common/docker-compose.env`.

```powershell
docker compose -f Common/docker-compose.yml up --build
```

Then open:

```text
http://localhost:8501
```

To stop it:

```powershell
docker compose -f Common/docker-compose.yml down
```

#### Rebuild mode

This starts Neo4j and reruns `build_reitteratsel_pipeline.py`, which refreshes the DuckDB/parquet cache in place.

First create the runtime env file:

```powershell
Copy-Item Common\docker-compose.env.example Common\docker-compose.env
```

Then edit `Common/docker-compose.env` so it has the correct Neo4j container settings. At minimum it should stay aligned with the compose file:

```env
NEO4J_URI=neo4j://neo4j:7687
NEO4J_DATABASE=neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=mamdaniXGBoost
```

Run the rebuild:

```powershell
docker compose -f Common/docker-compose.yml --profile rebuild up --build reitteratsel-rebuild
```

If that succeeds, start the app:

```powershell
docker compose -f Common/docker-compose.yml up --build
```

#### Useful checks

View logs:

```powershell
docker compose -f Common/docker-compose.yml logs -f
```

View rebuild logs:

```powershell
docker compose -f Common/docker-compose.yml --profile rebuild logs -f reitteratsel-rebuild
```

Stop and remove containers:

```powershell
docker compose -f Common/docker-compose.yml down
```

Stop and also remove Neo4j named volumes:

```powershell
docker compose -f Common/docker-compose.yml --profile rebuild down -v
```

#### Recommended first run

1. `Copy-Item Common\docker-compose.env.example Common\docker-compose.env`
2. `docker compose -f Common/docker-compose.yml --profile rebuild up --build reitteratsel-rebuild`
3. `docker compose -f Common/docker-compose.yml up --build`
4. Open `http://localhost:8501`

### [ 2 ] To run the system in local development mode outside Docker

Project-standard Python runtime:

- `C:\ProgramData\anaconda3\envs\env\python.exe`
- `C:\ProgramData\anaconda3\envs\env\Scripts\streamlit.exe`

Local development orchestrator:

- `Common\Micro\5_Model_KG\run_reitteratsel.py`

Typical local development flow:

```powershell
C:\ProgramData\anaconda3\envs\env\python.exe Common\Micro\5_Model_KG\run_reitteratsel.py
```

Notes:

- Development mode assumes the app rebuilds the cache every time it is launched.
- Development mode depends on a valid root `.env` for Neo4j connectivity.
- Submission mode differs from development mode because submission defaults to serving the committed DuckDB snapshot directly.

---

## SECTION 6 : PROJECT REPORT / PAPER

`Refer to project report at Github Folder: ProjectReport`

In this repository checkout, the `ProjectReport` folder exists at the repository root but no populated report file was found during this pass.

**Recommended Sections for Project Report / Paper:**

- Executive Summary / Paper Abstract
- Business Problem Background
- Project Objectives & Success Measurements
- Project Solution: annual warehouse, fuzzy reasoning, macro overlay, and app design
- Project Implementation: TradingView-style raw-data handling, metric build, DuckDB persistence, Neo4j rebuild path, XGBoost inference, and Streamlit app workflow
- Project Performance & Validation: evaluation outputs under `Common/Eval/IO/run_n`
- Project Conclusions: Findings & Recommendation
- Appendix of report: Installation and User Guide
- Appendix of report: Evaluation exports and interpretation notes
- Appendix of report: References

---

## SECTION 7 : MISCELLANEOUS

`Refer to Github Folder: Miscellaneous`

`Refer to Common\PROJECT_REFERENCE_MAP.md` for the standing location map of implementation assets, warehouse files, runtime assets, and current source-of-truth order.

### Common\Frontend\DesignDoc\Reitteratsel.pdf

- Frontend / presentation design reference for the application
- Useful for understanding the intended dashboard direction and visual layout

### Common\Frontend\DesignDoc\figma.png

- Frontend design reference image
- Useful as a quick visual context asset alongside the PDF

### Common\Eval\IO\run_3

- Example populated evaluation output folder found in this repository checkout
- Contains:
  - `reitteratsel_eval_detail.csv`
  - `reitteratsel_eval_summary.csv`
  - `reitteratsel_eval_disagreements.csv`
  - `reitteratsel_eval_confusion_matrices.csv`
  - `reitteratsel_eval_per_class_metrics.csv`
  - `reitteratsel_eval_ranking_metrics.csv`

---

## APPENDIX : LOCAL GIT NOTE FOR SHIPPED DUCKDB CACHE ARTIFACTS

This repository intentionally ships the DuckDB warehouse and related derived parquet cache artifacts used by the app.

During local rebuilds, Git may repeatedly show changes in these tracked files even when the user does not want to review or commit every rebuild result immediately:

- `Common/Micro/IO/out/_annual_warehouse/fundamentals.duckdb`
- `Common/Micro/IO/out/_annual_warehouse/parquet/carpathdaily.parquet`
- `Common/Micro/IO/out/_annual_warehouse/parquet/distresslabels.parquet`
- `Common/Micro/IO/out/_annual_warehouse/parquet/fuzzycache.parquet`

To reduce local Git noise, use `git update-index --skip-worktree` on those four tracked files.

Important notes:

- This is a local Git setting only.
- It does not get committed or pushed upstream.
- It only tells your local Git checkout to stop bothering you about routine local changes to those tracked files.
- If you later want to intentionally commit updated versions of those files, remove the flag first with `--no-skip-worktree`.

### 1) Skip the local Git noise

Run this from the repository root:

```cmd
git update-index --skip-worktree Common/Micro/IO/out/_annual_warehouse/fundamentals.duckdb Common/Micro/IO/out/_annual_warehouse/parquet/carpathdaily.parquet Common/Micro/IO/out/_annual_warehouse/parquet/distresslabels.parquet Common/Micro/IO/out/_annual_warehouse/parquet/fuzzycache.parquet
```

### 2) Check whether the skip flag is active

Run this from the repository root:

```cmd
git ls-files -v | findstr "Common/Micro/IO/out/_annual_warehouse"
```

Expected behavior:

- Lines starting with `S` indicate that `skip-worktree` is active for those files.

### 3) Re-enable those files for normal Git tracking

If you later want Git to notice changes again so that you can add and commit updated artifacts, run:

```cmd
git update-index --no-skip-worktree Common/Micro/IO/out/_annual_warehouse/fundamentals.duckdb Common/Micro/IO/out/_annual_warehouse/parquet/carpathdaily.parquet Common/Micro/IO/out/_annual_warehouse/parquet/distresslabels.parquet Common/Micro/IO/out/_annual_warehouse/parquet/fuzzycache.parquet
```

After that, `git status` will once again report local changes to those four tracked files normally.
