## REITterratsel: REIT Distress Reasoning System with Mamdani Rules, Macro Overlay, and XGBoost

<img src="Common/Frontend/DesignDoc/Reiterratsel_Wordmark.svg"
     style="float: left; margin-right: 0px;" />

---

## Executive Summary

This project is a hybrid REIT distress-monitoring system for Singapore-listed REITs. It combines an annual Mamdani fuzzy reasoning layer, a short-horizon XGBoost macro overlay, and a cumulative abnormal return path overlay so the final score stays interpretable while still reacting to changing conditions.

At the micro layer, the repository builds an authoritative DuckDB warehouse from TradingView-style annual financial-statement data and derived REIT metrics. These outputs feed annual distress labels, a Mamdani fuzzy cache, and a daily abnormal-return CAR-path layer.

At the macro layer, the runnable submission bundle in `SystemCode` uses the pre-trained `run_21` XGBoost artifacts to forecast short-horizon SORA stress. The runtime app then combines the annual Mamdani base, macro-rate overlay, refinancing sensitivity, and CAR-path adjustment into a final distress ranking.

For submission, the project ships a committed DuckDB snapshot so the application can run directly in Docker from `SystemCode` without forcing a rebuild on every launch. For development, the rebuild path remains available when the shipped warehouse and cached outputs need to be refreshed.

---

## Credits / Project Contribution

| Official Full Name | Student ID (MTech Applicable) | Work Items (Who Did What) | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
| Jason Tay | A0265092A | End-to-end solo delivery of the REIT distress project, including data pipeline design (for both micro and micro/company-layer), data sourcing, data engineering, DuckDB warehouse build, REIT metric derivation, Mamdani rule modelling, Neo4j rule-graph integration, macro XGBoost integration, Streamlit application development, Docker packaging, evaluation workflow, project documentation, and submission packaging. | jason.tay.nw@u.nus.edu |

---

## Video of System Modelling & Use Case Demo

`Refer to Github Folder: Video`

The repository includes a `Video` folder for modelling and demo material.

---

## User Guide

`Refer also to Appendix C in ProjectReport\ProjectReport_Group16_JasonTay_REITterraetsel.md`

### 1. Run the system using Docker in submission / demo mode

From the repository root, meaning the folder that contains `README.md`, `Common/`, `SystemCode/`, and the top-level `.git` folder:

```powershell
cd <path-to-this-repo>
cd SystemCode
```

Use the compose setup in two modes:

- `app-only` for the normal submission / demo path
- `rebuild` when Neo4j-backed cache regeneration is needed

#### App-only mode

This serves the app against the committed DuckDB snapshot and does not need `docker-compose.env`.

```powershell
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

To stop it:

```powershell
docker compose down
```

#### Rebuild mode

This starts Neo4j and reruns `build_reitteratsel_pipeline.py`, which refreshes the DuckDB/parquet cache in place.

First create the runtime env file inside `SystemCode`:

```powershell
Copy-Item docker-compose.env.example docker-compose.env
```

Then edit `docker-compose.env` so it matches the Neo4j container settings:

```env
NEO4J_URI=neo4j://neo4j:7687
NEO4J_DATABASE=neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=mamdaniXGBoost
```

Run the rebuild:

```powershell
docker compose --profile rebuild up --build reitteratsel-rebuild
```

If that succeeds, start the app:

```powershell
docker compose up --build
```

#### Useful checks

View logs:

```powershell
docker compose logs -f
```

View rebuild logs:

```powershell
docker compose --profile rebuild logs -f reitteratsel-rebuild
```

Stop and remove containers:

```powershell
docker compose down
```

Stop and also remove Neo4j named volumes:

```powershell
docker compose --profile rebuild down -v
```

#### Recommended first run

1. `Copy-Item docker-compose.env.example docker-compose.env`
2. `docker compose --profile rebuild up --build reitteratsel-rebuild`
3. `docker compose up --build`
4. Open `http://localhost:8501`

### 2. Run the system in local development mode outside Docker

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

## Project Report / Paper

Main report file:

- `ProjectReport\ProjectReport_Group16_JasonTay_REITterraetsel.md`

Report coverage:

- business case and literature grounding
- system design and implementation
- evaluation and findings
- installation appendix
- references and data sources

---

## Miscellaneous

`Refer to Common\PROJECT_REFERENCE_MAP.md` for the standing location map of implementation assets, warehouse files, runtime assets, and current source-of-truth order.

### Common\Frontend\DesignDoc\Reitteratsel.pdf

- Frontend / presentation design reference for the application
- Useful for understanding the intended dashboard direction and visual layout

### Common\Frontend\DesignDoc\figma.png

- Frontend design reference image
- Useful as a quick visual context asset alongside the PDF

### Common\Eval\IO\run_3

- Example populated evaluation output folder
- Contains:
  - `reitteratsel_eval_detail.csv`
  - `reitteratsel_eval_summary.csv`
  - `reitteratsel_eval_disagreements.csv`
  - `reitteratsel_eval_confusion_matrices.csv`
  - `reitteratsel_eval_per_class_metrics.csv`
  - `reitteratsel_eval_ranking_metrics.csv`

---

## Appendix: Local Git Note for Shipped DuckDB Cache Artifacts

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
