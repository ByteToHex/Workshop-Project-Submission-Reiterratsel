from __future__ import annotations

from pathlib import Path

from reitteratsel_core import (
    DUCKDB_PATH,
    ENV_PATH,
    FUZZY_CACHE_SHARD,
    DISTRESS_LABEL_SHARD,
    build_distress_label_frame,
    build_fuzzy_cache_frame,
    connect_neo4j,
    fetch_rule_bundle,
    persist_outputs_to_duckdb,
    seed_rules_to_neo4j,
)


def main() -> None:
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"DuckDB not found: {DUCKDB_PATH}")
    if not ENV_PATH.exists():
        raise FileNotFoundError(f".env not found: {ENV_PATH}")

    driver, config = connect_neo4j()
    try:
        driver.verify_connectivity()
        seed_rules_to_neo4j(driver, config)
        rule_bundle = fetch_rule_bundle(driver, config)
    finally:
        driver.close()

    label_df = build_distress_label_frame(DUCKDB_PATH)
    fuzzy_df = build_fuzzy_cache_frame(DUCKDB_PATH, rule_bundle=rule_bundle)
    persist_outputs_to_duckdb(label_df, fuzzy_df, DUCKDB_PATH)

    print(f"Seeded Neo4j rule model: {config.database}")
    print(f"Built label rows: {len(label_df)}")
    print(f"Built fuzzy cache rows: {len(fuzzy_df)}")
    print(f"Wrote label shard: {DISTRESS_LABEL_SHARD}")
    print(f"Wrote fuzzy shard: {FUZZY_CACHE_SHARD}")


if __name__ == "__main__":
    main()
