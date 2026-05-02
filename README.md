# Workshop Project Submission Working

## How To Run

Use the compose file in two modes:

- `app-only` for submission/demo
- `rebuild` when Neo4j-backed cache regeneration is needed

Important distinction:

- `app-only` serves the app against the committed DuckDB snapshot already shipped in the repo.
- `rebuild` starts Neo4j and reruns `build_reitteratsel_pipeline.py` so the DuckDB/parquet cache is refreshed in place.
- Rebuild is intentionally separate so the normal submission/demo path does not depend on Neo4j every time.

From the repository root, meaning the folder that contains `README.md`, `Common/`, and the top-level `.git` folder:

```powershell
cd <path-to-this-repo>
```

### 1) App-Only Mode

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

### 2) Rebuild Mode

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

### Useful Checks

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

### Recommended First Run

1. `Copy-Item Common\docker-compose.env.example Common\docker-compose.env`
2. `docker compose -f Common/docker-compose.yml --profile rebuild up --build reitteratsel-rebuild`
3. `docker compose -f Common/docker-compose.yml up --build`
4. Open `http://localhost:8501`

## Local Git Note For Shipped DuckDB Cache Artifacts

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
