# TradingView Pipeline Consolidated

This folder is not a clean end-to-end pipeline. It is a consolidated assembly of scripts used at different times to build the TradingView-backed database that later fed the PE-Mamdani fuzzy pipeline.

The important distinction is:

- not every subfolder here is expected to work today
- some parts are preserved mainly as historical extraction/probing tooling
- the serializer and metric builder are the parts intended to remain functional

## Folder Layout

- [1_TradingView_Exploration](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/1_TradingView_Exploration)
- [2_HTML_Dumping](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/2_HTML_Dumping)
- [3_Serialize_Dump_To_CSV_Parquet](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/3_Serialize_Dump_To_CSV_Parquet)
- [4_Compute_Metrics](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/4_Compute_Metrics)
- [IO](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/IO)
- [SCHEMA_DIFFERENCES](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/SCHEMA_DIFFERENCES)

## 1. TradingView Exploration

Folder:

- [1_TradingView_Exploration](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/1_TradingView_Exploration)

What it is:

- an assortment of browser-console scripts used to probe TradingView's HTML structure before export
- selector experiments for tabs, subtabs, timeframe controls, and HTML dumping behavior
- historical exploration tooling, not a stable maintained stage

Expectation:

- current functionality is unknown
- scripts in this folder should be treated as unreliable unless re-validated manually against the live site

Examples:

- [DumpHtml.js](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/1_TradingView_Exploration/DumpHtml.js)
- [open_all_subtabs.js](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/1_TradingView_Exploration/open_all_subtabs.js)

## 2. HTML Dumping

Folder:

- [2_HTML_Dumping](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/2_HTML_Dumping)

What it is:

- an assortment of scripts intended to export TradingView HTML snapshots
- this includes the more integrated dumper variants and the download organizer

Important status note:

- shortly after the original export process was working, TradingView changed internal HTML details including a button ID
- that change broke the dumping flow
- fixing this is in scope and should not be especially hard, but this stage is not actively being worked on and should not currently be assumed to work

Relevant files:

- [run_tradingview_dumper_exhaustive.js](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/2_HTML_Dumping/run_tradingview_dumper_exhaustive.js)
- [run_tradingview_dumper_exhaustive_multipage.js](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/2_HTML_Dumping/run_tradingview_dumper_exhaustive_multipage.js)
- [organize_html_by_ticker.py](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/2_HTML_Dumping/organize_html_by_ticker.py)

An alternative reference can be found on the host computer at: `...\Scrape-Tradingview\00_Extractor\002b_URLs\00_try_extension`

## 3. Serialize Dump to CSV / Parquet

Folder:

- [3_Serialize_Dump_To_CSV_Parquet](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/3_Serialize_Dump_To_CSV_Parquet)

Intended status:

- this stage is intended to be functional

Main purpose:

- read grouped TradingView HTML dumps
- normalize them against the annual schema
- rebuild the annual warehouse in DuckDB and Parquet

Primary script:

- [serialize_financials_to_parquet.py](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/3_Serialize_Dump_To_CSV_Parquet/serialize_financials_to_parquet.py)

Important notes:

- the explicit user-computer paths in this stage are intentional
- those paths reflect how this serializer was actually used for dumping and warehouse rebuilds
- CSV existed as an intermediary inspection/debug step
- CSV is not the main downstream artifact anymore
- the more reliable centralized source is the Parquet plus annual warehouse output

Supporting files:

- [serialize_financials_to_csv.py](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/3_Serialize_Dump_To_CSV_Parquet/serialize_financials_to_csv.py)
- [annual_schema_structured_indented.py](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/3_Serialize_Dump_To_CSV_Parquet/annual_schema_structured_indented.py)
- [SCHEMA_DIFFERENCES](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/SCHEMA_DIFFERENCES)

Outputs:

- `IO/out/_annual_warehouse/fundamentals.duckdb`
- `IO/out/_annual_warehouse/parquet/*.parquet`

## 4. Compute Metrics

Folder:

- [4_Compute_Metrics](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/4_Compute_Metrics)

Intended status:

- this stage is intended to be functional

Primary script:

- [build_reit_metrics.py](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/4_Compute_Metrics/build_reit_metrics.py)

Key schema/reference file:

- [Data_Dict_Reit_Metrics.md](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/4_Compute_Metrics/Data_Dict_Reit_Metrics.md)

What it does:

- reads the annual warehouse from Step 3
- computes the REIT metric layer used downstream
- writes metric outputs back into the warehouse/parquet structure

Outputs:

- `IO/out/_annual_warehouse/parquet/metrics.parquet`
- `reit_metrics` schema inside `IO/out/_annual_warehouse/fundamentals.duckdb`

Utilities:

- utility and inspection files now live under [4_Compute_Metrics/util](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/4_Compute_Metrics/util)
- [check_reit_metrics.py](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/4_Compute_Metrics/util/check_reit_metrics.py)
- [show_metric_pivot.py](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/4_Compute_Metrics/util/show_metric_pivot.py)
- [metrics_quality_checks.sql](D:/WS/-GH-DEV-Ref/WS-Python/Tasks/Scrape-Tradingview/Consolidated/4_Compute_Metrics/util/metrics_quality_checks.sql)

## Practical Read of This Folder

If you are trying to understand what still matters operationally:

1. `1_TradingView_Exploration` is historical probing and should not be trusted without manual checking.
2. `2_HTML_Dumping` is historically important but currently assumed broken due to TradingView UI/internal HTML drift.
3. `3_Serialize_Dump_To_CSV_Parquet` is intended to work and is the warehouse-construction stage.
4. `4_Compute_Metrics` is intended to work and is the metric-construction stage for the PE-Mamdani fuzzy pipeline.
