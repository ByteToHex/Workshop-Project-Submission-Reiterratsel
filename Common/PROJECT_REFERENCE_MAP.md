# Project Reference Map

This file is the unified location map for this repository's REIT distress / Mamdani / XGBoost / dashboard work.

Use this first when orienting to the project so the same paths and precedence rules do not need to be retyped in future prompts.

## Source-of-Truth Order

When references conflict, use this order:

1. Live DuckDB warehouse contents
2. Warehouse schema / metric documentation
3. Current implementation scripts
4. Implementation checklist / progress tracker
5. Draft design doc
6. Course-style Neo4j workshop reference material
7. Course-style Mamdani workshop reference material

Critical override:

- Always defer to user instruction. Flag out significant differences (if any) between user instruction and DuckDB warehouse contents, if any.
- After implementation, always update these items (where applicable):
  - Warehouse schema and metric semantics -> Align it with any schema changes. Be specific; eg. changes to the mamdani fuzzy pipeline should update Current_Impl_Schema_Reference.md
  - Draft design flow (Design_v1a.txt) -> Align it with the current design
  - Progress checklist (Implementation_Checklist_v1a.md) -> Implementation progress/targets
  - This file (PROJECT_REFERENCE_MAP.md) -> Paths that were added or changed
  - Docker runtime assets -> Align docker-compose.yml, any container env override files, and container build files with the actual runtime design
  // Please take care pay attention that they do not bloat; keep points succinct
- Always override the draft design doc with the actual DuckDB state.
- Treat the warehouse as authoritative for what already exists.
- Do not assume a design-doc step is still pending just because it appears in the draft flow.

## Core Project Map

### 1) Overall flow and progress tracking

- Draft design flow:
  `Common\Micro\5_Model_KG\DesignDocs\Design_v1a.txt`
  Note: useful for the rough A -> B -> C -> D flow only. It is explicitly subordinate to DuckDB.

- Progress / implementation checklist:
  `Common\Micro\5_Model_KG\DesignDocs\Implementation_Checklist_v1a.md`
  Note: fastest place to check what is done, partial, and still pending.

### 2) Mamdani reference baseline

- Course / expectation reference for a basic Mamdani pipeline:
  `Common\Micro\5_Model_KG\DesignDocs\Mamdani_Pipeline_Ref.txt`
  Note: perspective-alignment reference only. Do not treat this folder as the active implementation source of truth.

### 3) Neo4j architecture references

- Reference folder:
  `Common\Micro\5_Model_KG\DesignRef\RS\Day_1\Workshop`

- Important usage rule for this folder:
  read the Jupytext `.py` files only, never the paired `.ipynb` notebooks.

- Likely key reference files in that folder:
  `Common\Micro\5_Model_KG\DesignRef\RS\Day_1\Workshop\enhancing_rag_with_graph(Gemini,MD).py`
  `Common\Micro\5_Model_KG\DesignRef\RS\Day_1\Workshop\rag_with_knowledge_graphs_neo4j.py`

- Purpose:
  architecture / structure reference only. Do not treat this folder as the active implementation source of truth.

### 4) Neo4j runtime connection source

- Root environment file:
  `.env`

- Use `.env` for the active Neo4j connection settings.
- Neo4j configuration is expected to be explicit in `.env`; do not rely on silent code defaults.
- Do not duplicate secrets into other docs unless necessary.
- Current `.env` includes:
  - `NEO4J_URI`
  - `NEO4J_INSTANCE_NAME`
  - `NEO4J_VERSION`
  - `NEO4J_DATABASE`
  - `NEO4J_USERNAME`
  - `NEO4J_PASSWORD`

- Docker compose runtime override for container-to-container Neo4j hostname:
  `Common\docker-compose.env`
  Note: this is mounted as the runtime `.env` inside the app container because `127.0.0.1` from the host `.env` is not valid from inside Docker.

### 5) Authoritative DuckDB warehouse

- Warehouse folder:
  `Common\Micro\IO\out\_annual_warehouse`

- Main DuckDB file:
  `Common\Micro\IO\out\_annual_warehouse\fundamentals.duckdb`

- Parquet shard folder:
  `Common\Micro\IO\out\_annual_warehouse\parquet`

- This warehouse is the authoritative implementation state for A1 and downstream outputs.

### 6) Warehouse schema and metric semantics

- Schema reference:
  `Common\Micro\4_Compute_Metrics\Schemas.md`

- Metric dictionary / computation quirks:
  `Common\Micro\4_Compute_Metrics\Data_Dict_Reit_Metrics.md`

- Current live implementation schema reference:
  `Common\Micro\5_Model_KG\Current_Impl_Schema_Reference.md`

- Important note from project context:
  `null_count` is currently expected to be derived at query / build time rather than stored as its own fake metric row.

### 7) Builder and upstream data scripts

- Metric builder:
  `Common\Micro\4_Compute_Metrics\build_reit_metrics.py`

- Serializer / upstream parquet builder:
  `Common\Micro\3_Serialize_Dump_To_CSV_Parquet\serialize_financials_to_parquet.py`

### 8) Daily abnormal return input for A1 / labels

- SGX iEdge REIT index CSV:
  `Common\Macro\IO\SRC\CSV_TICKER\SGX_DLY_REIT, 1D.csv`

- Use this for:
  daily abnormal return = REIT daily return - SGX iEdge REIT index daily return

### 9) XGBoost model artifacts

- Model root:
  `Common\Macro\IO\Model_Train\Use\run_21`

- Available subfolders:
  `Common\Macro\IO\Model_Train\Use\run_21\fwd_10_days`
  `Common\Macro\IO\Model_Train\Use\run_21\fwd_15_days`

- Selection hint:
  `Common\Macro\IO\Model_Train\Use\run_21\BEST_MODEL_P.txt`

- Project note:
  use `fwd_10_days` first, while keeping wiring modular for `fwd_15_days`.

### 10) Frontend / Figma reference assets

- Frontend design reference folder:
  `Common\Frontend\DesignDoc`

- Key assets currently present:
  `Common\Frontend\DesignDoc\figma.png`
  `Common\Frontend\DesignDoc\Reitteratsel.pdf`

### 11) Active app / pipeline components

- Pipeline build entrypoint:
  `Common\Micro\5_Model_KG\build_reitteratsel_pipeline.py`

- Build + app orchestrator:
  `Common\Micro\5_Model_KG\run_reitteratsel.py`

- Core implementation module:
  `Common\Micro\5_Model_KG\reitteratsel_core.py`

- Mamdani rule seed artifact:
  `Common\Micro\5_Model_KG\mamdani_rule_seed.json`

- Streamlit app entrypoint:
  `Common\Frontend\reitteratsel_app.py`

- Usage note:
  these three files are the fastest path for checking what the live app actually reads, computes, persists, and displays.

### 12) Docker runtime assets

- Docker compose entrypoint:
  `Common\docker-compose.yml`

- App image definition:
  `Common\Frontend\Dockerfile.reitteratsel`

- App container pip requirements:
  `Common\Frontend\requirements.reitteratsel.txt`

- Container runtime env override:
  `Common\docker-compose.env`

- Usage note:
  this compose setup starts Neo4j plus the Streamlit app container, runs the KG build first, then launches the app.

## Practical Working Rules

- If the design doc says something but DuckDB already shows a different reality, follow DuckDB.
- Use the implementation checklist to distinguish completed work from merely planned work.
- Use the schema and metric dictionary before inferring warehouse meanings from column names alone.
- Use Neo4j workshop materials only for architectural ideas, not as proof of current repo behavior.
- For notebook-paired reference materials under the workshop folder, inspect `.py` only.
- Use the root `.env` for Neo4j connection details rather than older per-folder `.env` files unless a task explicitly targets those references.

## Fast Start Pointers

For most implementation tasks, check these in order:

1. `Common\Micro\IO\out\_annual_warehouse\fundamentals.duckdb`
2. `Common\Micro\4_Compute_Metrics\Schemas.md`
3. `Common\Micro\4_Compute_Metrics\Data_Dict_Reit_Metrics.md`
4. `Common\Micro\5_Model_KG\DesignDocs\Implementation_Checklist_v1a.md`
5. `Common\Micro\5_Model_KG\DesignDocs\Design_v1a.txt`

## Intended Use In Future Prompts

You can simply refer to:

- `Common\PROJECT_REFERENCE_MAP.md`

and instruct the agent to use it as the standing project location map instead of re-listing all paths from scratch.
