"""
Shared SQL predicate helpers for Fed/FOMC parquet utilities.

These helpers are consumed by scripts under Step_00/Step_01/Step_02 so that
filter logic stays centralized and consistent.
"""

from __future__ import annotations

from pathlib import Path

# Kalshi Finance:Fed series prefixes (see src/analysis/kalshi/util/categories.py)
KALSHI_FED_PREFIXES = (
    "FEDDECISION",
    "FEDMENTION",
    "FEDCHAIRNOM",
    "FEDEMPLOYEES",
    "RATECUTCOUNT",
    "RATECUT",
    "TERMINALRATE",
    "LEAVEPOWELL",
    "POWELLMENTION",
)


def _glob_sql(path: Path) -> str:
    """DuckDB parquet glob; always use forward slashes."""
    return str(path.resolve()).replace("\\", "/")


def _kalshi_fed_predicate() -> str:
    parts = []
    for p in KALSHI_FED_PREFIXES:
        parts.append(f"event_ticker LIKE '{p}%'")
        parts.append(f"ticker LIKE '{p}%'")
    # Short "FED" series — match only as a series prefix (before first hyphen in ticker)
    parts.append("regexp_matches(ticker, '^FED-[^-]+')")
    parts.append("regexp_matches(event_ticker, '^FED-[^-]+')")
    return "(" + " OR ".join(parts) + ")"


def _parquet_fed_predicate() -> str:
    # Text match on question/slug; adjust if you need stricter scope
    return """(
        lower(coalesce(question, '')) LIKE '%fomc%'
        OR lower(coalesce(slug, '')) LIKE '%fomc%'
        OR lower(coalesce(question, '')) LIKE '%federal reserve%'
        OR lower(coalesce(slug, '')) LIKE '%federal-reserve%'
        OR lower(coalesce(question, '')) LIKE '%fed funds%'
        OR lower(coalesce(question, '')) LIKE '%federal funds%'
        OR lower(coalesce(slug, '')) LIKE '%fed-rate%'
        OR lower(coalesce(slug, '')) LIKE '%fed-decision%'
        OR lower(coalesce(question, '')) LIKE '%fed meeting%'
        OR lower(coalesce(question, '')) LIKE '%fed cut%'
        OR lower(coalesce(question, '')) LIKE '%fed hike%'
    )"""


def _fed_predicate_extended() -> str:
    """Extra OR clauses for discovery; keeps the same broad base predicate."""
    base = _parquet_fed_predicate().strip()
    assert base.startswith("(") and base.endswith(")")
    extra = """
        OR lower(coalesce(question, '')) LIKE '%interest rate%'
        OR lower(coalesce(question, '')) LIKE '%powell%'
        OR lower(coalesce(question, '')) LIKE '%federal funds rate%'
        OR lower(coalesce(question, '')) LIKE '%basis point%'
        OR lower(coalesce(question, '')) LIKE '%fomc meeting%'
        OR lower(coalesce(slug, '')) LIKE '%powell%'
        OR lower(coalesce(slug, '')) LIKE '%rate-cut%'
        OR lower(coalesce(slug, '')) LIKE '%rate-hike%'
    """
    return "(" + base[1:-1].strip() + extra + "\n    )"


def _events_json_column(cols: set[str]) -> str | None:
    """Preferred Gamma events array column name when available."""
    for name in ("events_json", "events"):
        if name in cols:
            return name
    return None


def _lower_text_blob_sql(cols: set[str]) -> str:
    """Expression: lower(question + optional description) for text matching."""
    parts: list[str] = ["coalesce(question, '')"]
    if "description" in cols:
        parts.append("coalesce(description, '')")
    if len(parts) == 1:
        return f"lower({parts[0]})"
    return "lower(" + " || ' ' || ".join(parts) + ")"


def _tier_a_predicate_sql(cols: set[str]) -> str:
    """
    Tier A — Gamma events JSON only: fed-interest-rates series or
    parent slug fed-decision-in-*.
    """
    ev_c = _events_json_column(cols)
    if not ev_c:
        return "FALSE"
    return f"""(
        (nullif(trim(try(json_extract_string(try_cast({ev_c} AS JSON), '$[0].seriesSlug'))), '') = 'fed-interest-rates')
        OR (lower(nullif(trim(try(json_extract_string(try_cast({ev_c} AS JSON), '$[0].slug'))), '')) LIKE 'fed-decision-in-%')
    )"""


def _tier_b_predicate_sql(cols: set[str]) -> str:
    """
    Tier B — slug-only heuristics.

    Includes legacy no-change patterns and explicit "will-there-be-no-change..."
    / "will-the-fed-change-rates-to-another-level..." variants observed in missing
    completeness rows.
    """
    del cols  # kept for a consistent function signature across tier helpers
    slug = "lower(trim(coalesce(cast(slug as varchar), '')))"
    return f"""
    (
        (
            {slug} LIKE 'fed-decreases-interest-rates-%'
            OR {slug} LIKE 'fed-increases-interest-rates-%'
            OR {slug} LIKE 'fed-decision-in-%'
            OR (
                {slug} LIKE 'no-change-in-%'
                AND {slug} LIKE '%after-%meeting%'
                AND (
                    {slug} LIKE '%fed%'
                    OR {slug} LIKE '%federal-fund%'
                    OR {slug} LIKE '%interest-rate%'
                )
            )
            OR (
                {slug} LIKE 'will-there-be-no-change-in-%'
                AND {slug} LIKE '%after-%meeting%'
                AND (
                    {slug} LIKE '%fed%'
                    OR {slug} LIKE '%federal-fund%'
                    OR {slug} LIKE '%interest-rate%'
                )
            )
            OR (
                {slug} LIKE 'will-the-fed-change-rates-to-another-level-after-%meeting%'
            )
        )
        AND {slug} NOT LIKE '%bank-of-englands%'
        AND {slug} NOT LIKE '%bank-of-japans%'
        AND {slug} NOT LIKE '%ecb%'
    )
    """.strip()


def _tier_c_predicate_sql(cols: set[str]) -> str:
    """Tier C — question + optional description text only."""
    blob = _lower_text_blob_sql(cols)
    return f"""
    (
        ({blob} LIKE '%interest rate%' OR {blob} LIKE '%interest rates%')
        AND ({blob} LIKE '%bps%' OR {blob} LIKE '%basis point%' OR {blob} LIKE '%basis points%')
        AND (
            {blob} LIKE '%meeting%'
            OR {blob} LIKE '%fomc%'
            OR {blob} LIKE '%federal open market%'
        )
        AND (
            {blob} LIKE '%fomc%'
            OR {blob} LIKE '%federal reserve%'
            OR {blob} LIKE '%federal funds%'
            OR {blob} LIKE '%federal open market committee%'
            OR (
                {blob} LIKE '%fed%'
                AND ({blob} LIKE '%interest rate%' OR {blob} LIKE '%interest rates%')
            )
        )
    )
    """.strip()


def _parquet_fomc_decision_predicate_sql(cols: set[str]) -> str:
    """Union of tiers A, B, and C (same row may satisfy multiple tiers)."""
    a = _tier_a_predicate_sql(cols)
    b = _tier_b_predicate_sql(cols)
    c = _tier_c_predicate_sql(cols)
    return f"(({a}) OR ({b}) OR ({c}))"
