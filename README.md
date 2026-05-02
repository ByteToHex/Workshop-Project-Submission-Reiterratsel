# Workshop Project Submission Working

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
