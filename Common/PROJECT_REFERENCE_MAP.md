# Project Reference Map

This file is the unified location map for this repository's REIT distress / Mamdani / XGBoost / dashboard work.

Use this first when orienting to the project so the same paths and precedence rules do not need to be retyped in future prompts.

## Current Design

For project submission ONLY, current design assumes that the app is rebuilt (meaning the runner will rebuild the cache in reitteratsel_core.py) every time it is launched.

Current design caveats:

- Cache freshness risk is intentionally reduced because startup rebuilds `fact_distress_label`, `fact_fuzzy_cache`, and `rule_trace_text` on each launch.
- App startup now depends on build success. If the build step fails, the UI will not come up.
- Docker startup can still hit Neo4j readiness races even with `depends_on` and health checks.
- Current design reseeds Neo4j on each launch, which is workable for project submission/local use but can become awkward in future industry expansion eg. additions of shared or longer-lived graph state.
- Host `.env` and Docker runtime env intentionally differ on `NEO4J_URI`, so local success does not automatically imply Docker success.
- Docker image rebuilds can drift over time because app Python dependencies are not pinned yet.
- Cold-start time may grow as data volume or pipeline complexity grows, because rebuild happens before app serve.
- The compose app command currently couples build and serve into one startup path, which is simpler for submission but less clean for long-term operations/debugging.

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
  Note: keep this local and uncommitted.

- Docker compose env template:
  `Common\docker-compose.env.example`
  Note: use this as the committed template for the container runtime `.env` because `127.0.0.1` from the host `.env` is not valid from inside Docker.

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

### 6a) Original annual raw parquet schema for per-ticker shards

- Scope:
  this section describes the original TradingView-style raw annual data stored in the per-ticker parquet shards under `Common\Micro\IO\out\_annual_warehouse\parquet`.
  Exclude downstream derived shards such as `metrics.parquet`, `fuzzycache.parquet`, and `distresslabels.parquet`.

- Human-readable original schema reference:
  `Common\Micro\SCHEMA_DIFFERENCES\Tradingview_Schema_Annual_FromSS.txt`

- Generated structured schema references for script use:
  `Common\Micro\SCHEMA_DIFFERENCES\AnnualSchema_Structured.json`
  `Common\Micro\SCHEMA_DIFFERENCES\AnnualSchema_Structured_Indented.json`

- Raw TradingView export tables represented in those per-ticker shards:

```sql
CREATE TABLE schema_rows (
    row_id      INTEGER PRIMARY KEY,
    section     VARCHAR,
    label       VARCHAR,
    depth       INTEGER,
    parent_id   INTEGER REFERENCES schema_rows(row_id),
    group_output_label VARCHAR
);

CREATE TABLE financials (
    ticker   VARCHAR,
    period   VARCHAR,
    currency VARCHAR,
    row_id   INTEGER REFERENCES schema_rows(row_id),
    value    VARCHAR
);
```

- Practical meaning:
  `schema_rows` defines the annual TradingView row tree and grouping labels.
  `financials` stores ticker-period raw values keyed by `row_id`.

- Important distinction:
  treat this as the source schema for the original annual per-ticker raw data only.
  Do not confuse it with the downstream computed warehouse outputs in DuckDB or with derived parquet shards such as `metrics.parquet`, `fuzzycache.parquet`, and `distresslabels.parquet`.

### 7) Builder and upstream data scripts

- Metric builder:
  `Common\Micro\4_Compute_Metrics\build_reit_metrics.py`

- Serializer / upstream parquet builder:
  `Common\Micro\3_Serialize_Dump_To_CSV_Parquet\serialize_financials_to_parquet.py`

### 8) Daily abnormal return input for A1 / labels

- SGX iEdge REIT index CSV:
  `Common\Macro\IO\SRC\CSV_TICKER\SGX_DLY_REIT, 1D.csv`

- SGX Universe of 15 REITs CSV:
  `Common\Macro\IO\SRC\CSV_TICKER\*`
  All files excluding `SGX_DLY_REIT, 1D.csv`, `SGX_DLY_REITN, 1D(2024_2025_ONLY).csv`, `SGX_DLY_REITR, 1D.csv`

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

- Distress-label split pipeline methods in core:
  `load_distress_label_source_frames()`
  `derive_distress_label_row()`
  `build_distress_label_frame()`

- Fuzzy-cache split pipeline methods in core:
  `build_fuzzy_input_frame()`
  `evaluate_fuzzy_row()`
  `derive_fuzzy_cache_row()`
  `build_fuzzy_cache_frame()`

- Rule-trace text derivation method in core:
  `build_rule_trace_text()`

- Mamdani rule seed artifact:
  `Common\Micro\5_Model_KG\mamdani_rule_seed.json`

- Streamlit app entrypoint:
  `Common\Frontend\reitteratsel_app.py`

- Usage note:
  these files and split pipeline methods are the fastest path for checking what the live app reads, and how `fact_distress_label`, `fact_fuzzy_cache`, and `rule_trace_text` are derived and persisted.

### 12) Docker runtime assets

- Docker compose entrypoint:
  `Common\docker-compose.yml`

- App image definition:
  `Common\Frontend\Dockerfile.reitteratsel`

- App container pip requirements:
  `Common\Frontend\requirements.reitteratsel.txt`

- Container runtime env override:
  `Common\docker-compose.env` local secret file

- Container runtime env template:
  `Common\docker-compose.env.example`

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
