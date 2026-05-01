-- ══════════════════════════════════════════════════════════════════════════
-- Parquet CTF Exchange – leg implied-probabilities at a point in time
-- ══════════════════════════════════════════════════════════════════════════
--
-- PARAMETERS (edit the `params` CTE only):
--   target_parent_event_id  →  integer from 2VALIDATE CSV
--   snapshot_ts             →  TIMESTAMP in UTC
--   ledger_min / ledger_max   →  from 3PROBE_CTF_market_trade_coverage.csv
--
-- TIMEZONE: All timestamps are UTC (Polygon chain standard). Use UTC for
-- every snapshot; never mix timezones within a time series.
--
-- GRANULARITY GUIDANCE:
--   Daily  → UTC midnight '…T00:00:00Z'.  Standard for most academic series.
--            Does not clip FOMC decision (~18:00 UTC), so a same-day daily
--            snapshot will capture pre-decision prices only.
--   Hourly → any UTC hour; use DATE_TRUNC('hour', ledger_ts) to floor.
--   Best practice for FOMC event studies: snapshot at 17:00 UTC (noon ET)
--   on the decision day and at 23:00 UTC same day (post-decision close).
--
-- OUTPUT SCHEMA (standardised):
--   snapshot_ts, parent_event_id, leg_label, signed_leg_move_bps,
--   outcome_index (0=YES / 1=NO), outcome_name,
--   last_trade_ledger, last_trade_at,
--   implied_prob (decimal 6dp), implied_prob_pct (pct 2dp),
--   trade_count, volume_usdc,
--   yes_prob_sum_pct, yes_prob_deviation_pct,   ← sanity check
--   expected_bps,                               ← feature: prob-weighted mean move
--   p_no_change,                                ← feature: P(hold / 0-bps outcome)
--   n_brackets                                  ← feature: # quantifiable legs (excl. Other)
--
-- CSV PATHS: absolute paths are required — DBeaver's JVM working directory
--   is the DBeaver installation folder, not the project root.
--   Update the D:/WS_NUS/... prefix if the repo moves.
--
-- ══════════════════════════════════════════════════════════════════════════

WITH

-- ── EDIT THIS LEDGER ONLY ─────────────────────────────────────────────────
params AS (
    SELECT
        11696                            AS target_parent_event_id,
        TIMESTAMP '2024-09-10 00:00:00'  AS snapshot_ts,
        59801665                         AS ledger_min,
        61989842                         AS ledger_max
),
-- ─────────────────────────────────────────────────────────────────────────

-- ── 0. Leg metadata ───────────────────────────────────────────────────────
-- Maps token_id → (leg_label, signed_leg_move_bps, outcome_index).
--
-- Market_Token_Map.csv : token_id as full-precision integer strings.
--   2VALIDATE CSV      : group_item_title, signed_leg_move_bps, parent_event_id.
--   Join key           : market_slug (present in both).
--
-- 2VALIDATE has 2 rows per market_slug (YES + NO token); the inner DISTINCT
-- collapses them to 1 metadata row per leg before joining Market_Token_Map.
--
-- outcome_index: row order within market_slug in Market_Token_Map (0=YES,
--   1=NO). Assumes CSV export order is stable — verify against a resolved
--   market where the YES token is known.
--
-- CAUTION: do NOT use token_id from 2VALIDATE — it is stored in scientific
--   notation (floating-point precision loss) and cannot safely match pm_trades.
--
leg_meta (token_id, leg_label, signed_leg_move_bps, outcome_index, parent_event_id) AS (
    SELECT
        tm.token_id,
        v.group_item_title                                                         AS leg_label,
        CAST(v.signed_leg_move_bps AS INTEGER)                                     AS signed_leg_move_bps,
        (ROW_NUMBER() OVER (PARTITION BY tm.market_slug ORDER BY tm._row_num) - 1) AS outcome_index,
        CAST(v.parent_event_id AS INTEGER)                                         AS parent_event_id
    FROM (
        -- rowid is unavailable on read_csv() virtual tables; capture file order
        -- with ROW_NUMBER() OVER () before any join reorders the rows.
        SELECT market_slug, token_id, ROW_NUMBER() OVER () AS _row_num
        FROM read_csv('D:/WS_NUS/REF_DATA/prediction-market-analysis/scripts/export/REF/Market_Token_Map.csv',
                      types={'token_id': 'VARCHAR'})
    )                                                                              tm
    JOIN (
        SELECT DISTINCT market_slug, group_item_title, signed_leg_move_bps, parent_event_id
        FROM read_csv('D:/WS_NUS/REF_DATA/prediction-market-analysis/scripts/export/REF/2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv')
        WHERE CAST(parent_event_id AS INTEGER) = (SELECT target_parent_event_id FROM params)
    ) v ON v.market_slug = tm.market_slug
),

-- ── 1. Narrow pm_ledgers to market lifespan ≤ snapshot ────────────────────
-- ledger_min / ledger_max: set per event in the params CTE above.
ledgers_in_scope AS (
    SELECT
        ledger_number,
        TRY_CAST(timestamp AS TIMESTAMP) AS ledger_ts
    FROM pm_ledgers, params
    WHERE ledger_number BETWEEN params.ledger_min AND params.ledger_max
      AND TRY_CAST(timestamp AS TIMESTAMP) <= params.snapshot_ts
),

-- ── 2. Trades for any leg token, with USDC notional ──────────────────────
-- Price = USDC_amount / token_amount.  maker_asset_id = '0' → maker sold
-- USDC to buy outcome tokens, so outcome token is on the taker side.
-- Both amounts are 6-decimal USDC integers; dividing gives a unitless ratio
-- directly interpretable as an implied probability.
trades_raw AS (
    SELECT
        t.ledger_number,
        t.log_index,
        b.ledger_ts,
        CASE
            WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
            ELSE                             t.maker_asset_id
        END AS outcome_token_id,
        CASE
            WHEN t.maker_asset_id = '0'
                THEN TRY_CAST(t.maker_amount AS DOUBLE)
                     / NULLIF(TRY_CAST(t.taker_amount AS DOUBLE), 0)
            ELSE     TRY_CAST(t.taker_amount AS DOUBLE)
                     / NULLIF(TRY_CAST(t.maker_amount AS DOUBLE), 0)
        END AS implied_price,
        CASE
            WHEN t.maker_asset_id = '0' THEN TRY_CAST(t.maker_amount AS DOUBLE) / 1e6
            ELSE                             TRY_CAST(t.taker_amount AS DOUBLE) / 1e6
        END AS usdc_notional
    FROM pm_trades t
    INNER JOIN ledgers_in_scope b ON t.ledger_number = b.ledger_number
    WHERE t.maker_asset_id IN (SELECT token_id FROM leg_meta)
       OR t.taker_asset_id IN (SELECT token_id FROM leg_meta)
),

-- ── 3a. Per-token aggregate stats ────────────────────────────────────────
token_stats AS (
    SELECT
        outcome_token_id,
        COUNT(*)           AS trade_count,
        SUM(usdc_notional) AS volume_usdc
    FROM trades_raw
    GROUP BY outcome_token_id
),

-- ── 3b. Last traded price per token ≤ snapshot ───────────────────────────
last_trade AS (
    SELECT
        outcome_token_id,
        implied_price    AS last_price,
        ledger_number     AS last_trade_ledger,
        ledger_ts         AS last_trade_at
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY outcome_token_id
                ORDER BY ledger_ts DESC, ledger_number DESC, log_index DESC
            ) AS rn
        FROM trades_raw
    )
    WHERE rn = 1
),

-- ── 4. Event-level YES-side probability sum  (sanity check) ───────────────
-- Theory: exactly one leg resolves YES, so SUM(YES implied_prob) = 1.0.
-- Deviation sources:
--   · Stale last-trade prices (different legs last traded at different times)
--   · Leg with NULL last_price (no trades observed before snapshot → LEFT JOIN)
-- Threshold guidance: flag |deviation| > 5 pct-pts for investigation.
yes_sum AS (
    SELECT
        COALESCE(ROUND(SUM(lt.last_price), 6), 0)          AS yes_prob_sum,
        COALESCE(ROUND(SUM(lt.last_price) * 100, 2), 0)    AS yes_prob_sum_pct
    FROM leg_meta lm
    LEFT JOIN last_trade lt ON lm.token_id = lt.outcome_token_id
    WHERE lm.outcome_index = 0
),

-- ── 5. Event-level scalar features ───────────────────────────────────────
-- All three features are computed in one pass over YES-side legs.
--
-- expected_bps  : prob-weighted mean rate move in bps.
--                 Other legs (signed_leg_move_bps IS NULL) are excluded via
--                 NULL arithmetic: NULL × price = NULL → SUM skips silently.
--                 Affected events: Nov 2024 (pid=11827), Dec 2024 (pid=11878).
--
-- p_no_change   : implied probability of the hold / 0-bps outcome.
--                 NULL when the event has no hold leg (e.g. Sept 2022 — hike-only).
--
-- n_brackets    : count of quantifiable legs (signed_leg_move_bps IS NOT NULL).
--                 Uses COUNT(CASE …) so legs with no trade data are still counted;
--                 only the "Other" catch-all leg is excluded.
--                 Python layer uses this for distance_from_uniform = modal_prob - 1/n.
event_stats AS (
    SELECT
        ROUND(SUM(CAST(lm.signed_leg_move_bps AS DOUBLE) * lt.last_price), 4)     AS expected_bps,
        ROUND(SUM(CASE WHEN lm.signed_leg_move_bps = 0 THEN lt.last_price END), 6) AS p_no_change,
        COUNT(CASE WHEN lm.signed_leg_move_bps IS NOT NULL THEN 1 END)              AS n_brackets
    FROM leg_meta lm
    LEFT JOIN last_trade lt ON lm.token_id = lt.outcome_token_id
    WHERE lm.outcome_index = 0
)

-- ── 6. Final result ───────────────────────────────────────────────────────
SELECT
    (SELECT snapshot_ts FROM params)                                        AS snapshot_ts,
    lm.parent_event_id,
    lm.leg_label,
    lm.signed_leg_move_bps,
    lm.outcome_index,
    CASE lm.outcome_index WHEN 0 THEN 'YES' ELSE 'NO' END                  AS outcome_name,
    lt.last_trade_ledger,
    lt.last_trade_at,
    ROUND(lt.last_price,        6)                                          AS implied_prob,
    ROUND(lt.last_price * 100,  2)                                          AS implied_prob_pct,
    ts.trade_count,
    ROUND(ts.volume_usdc,       2)                                          AS volume_usdc,
    ys.yes_prob_sum_pct,
    ROUND((ys.yes_prob_sum - 1.0) * 100, 2)                                AS yes_prob_deviation_pct,
    es.expected_bps,
    es.p_no_change,
    es.n_brackets

FROM leg_meta lm
LEFT JOIN last_trade  lt ON lm.token_id = lt.outcome_token_id
LEFT JOIN token_stats ts ON lm.token_id = ts.outcome_token_id
CROSS JOIN yes_sum    ys
CROSS JOIN event_stats es

ORDER BY lm.signed_leg_move_bps, lm.outcome_index
;
