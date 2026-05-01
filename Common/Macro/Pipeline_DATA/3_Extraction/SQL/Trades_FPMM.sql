-- ══════════════════════════════════════════════════════════════════════════
-- Parquet FPMM (legacy) – leg implied-probabilities at a point in time
-- ══════════════════════════════════════════════════════════════════════════
--
-- PARAMETERS (edit the `params` CTE only):
--   target_parent_event_id  →  integer from 2VALIDATE CSV
--   snapshot_ts             →  TIMESTAMP in UTC
--
-- TIMEZONE: All timestamps are UTC (Polygon chain standard).
-- GRANULARITY GUIDANCE: same as Trades_CTF.sql – see that file for detail.
--
-- NOTE ON DECIMAL PRECISION:
--   In Gnosis CTF FPMM markets, conditional outcome tokens inherit the
--   collateral token's decimal base.  For USDC-collateralised markets,
--   both `amount` and `outcome_tokens` are 6-decimal integers, so
--   amount_raw / tokens_raw is directly a unitless implied probability.
--   The schema note citing "18 decimals" for outcome_tokens refers to the
--   raw ERC-20 encoding before the CTF wrapper re-scales to match the
--   collateral; the on-chain FPMM events log the CTF-scaled value.
--   VERIFY: on a resolved market the last YES trade before resolution
--   should yield implied_prob ≈ 1.0 (winner) or ≈ 0.0 (loser).
--
-- NOTE ON PROB_SUM > 100% IN FPMM:
--   The AMM constant-product invariant enforces p_YES + p_NO = 1.0 only
--   at the instant of any given trade.  Because the last YES trade and the
--   last NO trade occur at DIFFERENT ledgers, the two stale snapshots need
--   not sum to 1.0.  This is NOT a data error; it reflects price staleness.
--   Typical magnitude: <3 pct-pts for liquid periods; larger gaps signal
--   inactivity or a structural price move between the two last trades.
--
-- OUTPUT SCHEMA (standardised, matches Trades_CTF.sql):
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
        901491                           AS target_parent_event_id,
        TIMESTAMP '2022-10-30 00:00:00'  AS snapshot_ts
),
-- ─────────────────────────────────────────────────────────────────────────

-- ── 0a. Leg reference: condition_id → label & signed_leg_move_bps ─────────
-- 1EXTRACT CSV   : condition_id (hex), slug (= market_slug).
-- 2VALIDATE CSV  : group_item_title, signed_leg_move_bps, parent_event_id.
-- Join key       : 1EXTRACT.slug = 2VALIDATE.market_slug.
--
-- 2VALIDATE has 2 rows per market_slug (YES + NO token); the inner DISTINCT
-- collapses them to 1 metadata row per leg before joining 1EXTRACT.
--
-- fpmm_address is NOT hardcoded here — it is resolved from pm_markets in
-- leg_meta below, so it remains correct if a market is redeployed.
--
-- EXPECTED_BPS: signed_leg_move_bps is signed (positive = rate hike,
--   negative = cut).  expected_bps = SUM(signed_leg_move_bps × YES_prob).
--
leg_ref (condition_id, leg_label, signed_leg_move_bps, parent_event_id) AS (
    SELECT
        ex.condition_id,
        v.group_item_title                  AS leg_label,
        CAST(v.signed_leg_move_bps AS INTEGER),
        CAST(v.parent_event_id AS INTEGER)  AS parent_event_id
    FROM read_csv('D:/WS_NUS/REF_DATA/prediction-market-analysis/scripts/export/REF/1EXTRACT_fed_parquet_events.csv') ex
    JOIN (
        SELECT DISTINCT market_slug, group_item_title, signed_leg_move_bps, parent_event_id
        FROM read_csv('D:/WS_NUS/REF_DATA/prediction-market-analysis/scripts/export/REF/2VALIDATE_CTF_REF_Bracket_ParentDate_LegMove_FomcDate.csv')
        WHERE CAST(parent_event_id AS INTEGER) = (SELECT target_parent_event_id FROM params)
    ) v ON v.market_slug = ex.slug
),

-- ── 0b. Resolve fpmm_address from pm_markets ─────────────────────────────
leg_meta AS (
    SELECT
        r.condition_id,
        lower(trim(cast(m.market_maker_address AS VARCHAR))) AS fpmm_address,
        r.leg_label,
        r.signed_leg_move_bps,
        r.parent_event_id
    FROM pm_markets m
    JOIN leg_ref r ON r.condition_id = m.condition_id
),

-- ── 1. Buy trades before snapshot ────────────────────────────────────────
-- Only buy-side trades are used: price = USDC_in / tokens_out.
-- Sell-side prices have the opposite direction and mix in exit liquidity.
trades_raw AS (
    SELECT
        lower(trim(cast(lt.fpmm_address AS VARCHAR)))  AS fpmm_address,
        lt.ledger_number,
        lt.log_index,
        lt.outcome_index,
        CAST(lt.amount         AS DOUBLE)              AS amount_raw,
        CAST(lt.outcome_tokens AS DOUBLE)              AS tokens_raw,
        b.timestamp                                    AS ledger_ts
    FROM pm_legacy_trades lt
    JOIN pm_ledgers  b  ON b.ledger_number = lt.ledger_number
    JOIN leg_meta   lm ON lm.fpmm_address = lower(trim(cast(lt.fpmm_address AS VARCHAR)))
    JOIN params     p  ON TRUE
    WHERE lt.is_buy = TRUE
      AND TRY_CAST(b.timestamp AS TIMESTAMP) <= p.snapshot_ts
),

-- ── 2a. Per (fpmm, outcome) aggregate stats ───────────────────────────────
token_stats AS (
    SELECT
        fpmm_address,
        outcome_index,
        COUNT(*)                                                   AS trade_count,
        -- volume_usdc: assuming 6-decimal USDC collateral.
        -- If collateral is DAI (18 dec), divide by 1e18 instead.
        SUM(amount_raw / 1e6)                                      AS volume_usdc
    FROM trades_raw
    GROUP BY fpmm_address, outcome_index
),

-- ── 2b. Last traded price per (fpmm, outcome) ≤ snapshot ─────────────────
last_trade AS (
    SELECT
        fpmm_address,
        outcome_index,
        ledger_number                                AS last_trade_ledger,
        ledger_ts                                    AS last_trade_at,
        amount_raw / NULLIF(tokens_raw, 0)          AS implied_price
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY fpmm_address, outcome_index
                ORDER BY ledger_number DESC, log_index DESC
            ) AS rn
        FROM trades_raw
    )
    WHERE rn = 1
),

-- ── 3. Event-level YES-side probability sum  (sanity check) ───────────────
-- Deviation from 100%: positive = collectively overpriced (sum > 1),
-- negative = price gap / missing leg (sum < 1).
-- For FPMM deviations >5 pct-pts: inspect last_trade_at timestamps —
-- large time gaps between last YES and last NO trades explain most cases.
yes_sum AS (
    SELECT
        COALESCE(ROUND(SUM(lt.implied_price), 6), 0)       AS yes_prob_sum,
        COALESCE(ROUND(SUM(lt.implied_price) * 100, 2), 0) AS yes_prob_sum_pct
    FROM leg_meta lm
    LEFT JOIN last_trade lt
           ON lm.fpmm_address = lt.fpmm_address
          AND lt.outcome_index = 0
),

-- ── 4. Event-level scalar features ───────────────────────────────────────
-- All three features are computed in one pass over YES-side legs.
--
-- expected_bps  : prob-weighted mean rate move in bps.
--                 Other legs (signed_leg_move_bps IS NULL) excluded via
--                 NULL arithmetic: NULL × price = NULL → SUM skips silently.
--
-- p_no_change   : implied probability of the hold / 0-bps outcome.
--                 For events with NO hold leg defined (2022 Jul/Sep/Nov/Dec),
--                 return epsilon=0.000001 instead of NULL to avoid divide-by-zero
--                 issues in downstream derived ratios.
--                NOTE: Events with no hold leg (bps=0 absent) are 4 only; by right should have the IDs for each but "FPMM only" is only a universe of 4 parent markets (already confirmed Jan/Feb 2023 has the no-change leg- so keep this)
--                (...\Xform_FeatureEngineer\ForMacro\ExportRatios\260423_2032_HandleMissingRatios.txt):
--                  - `901489` (2022-07-27)
--                  - `901490` (2022-09-21)
--                  - `901491` (2022-11-02)
--                  - `901492` (2022-12-14)
--
-- n_brackets    : count of quantifiable legs (signed_leg_move_bps IS NOT NULL).
--                 Legs with no trade data are still counted; only the "Other"
--                 catch-all leg is excluded.
event_stats AS (
    SELECT
        ROUND(SUM(CAST(lm.signed_leg_move_bps AS DOUBLE) * lt.implied_price), 4)     AS expected_bps,
        CASE
            WHEN COUNT(CASE WHEN lm.signed_leg_move_bps = 0 THEN 1 END) = 0 THEN 0.000001
            ELSE ROUND(SUM(CASE WHEN lm.signed_leg_move_bps = 0 THEN lt.implied_price END), 6)
        END                                                                            AS p_no_change,
        COUNT(CASE WHEN lm.signed_leg_move_bps IS NOT NULL THEN 1 END)                 AS n_brackets
    FROM leg_meta lm
    LEFT JOIN last_trade lt
           ON lm.fpmm_address = lt.fpmm_address
          AND lt.outcome_index = 0
)

-- ── 5. Final result ───────────────────────────────────────────────────────
SELECT
    (SELECT snapshot_ts FROM params)                                        AS snapshot_ts,
    lm.parent_event_id,
    lm.leg_label,
    lm.signed_leg_move_bps,
    lt.outcome_index,
    CASE lt.outcome_index WHEN 0 THEN 'YES' ELSE 'NO' END                  AS outcome_name,
    lt.last_trade_ledger,
    lt.last_trade_at,
    ROUND(lt.implied_price,        6)                                       AS implied_prob,
    ROUND(lt.implied_price * 100,  2)                                       AS implied_prob_pct,
    ts.trade_count,
    ROUND(ts.volume_usdc,          2)                                       AS volume_usdc,
    ys.yes_prob_sum_pct,
    ROUND((ys.yes_prob_sum - 1.0) * 100, 2)                                AS yes_prob_deviation_pct,
    es.expected_bps,
    es.p_no_change,
    es.n_brackets

FROM leg_meta lm
LEFT JOIN last_trade  lt ON lm.fpmm_address = lt.fpmm_address
LEFT JOIN token_stats ts ON lm.fpmm_address = ts.fpmm_address
                         AND lt.outcome_index = ts.outcome_index
CROSS JOIN yes_sum    ys
CROSS JOIN event_stats es

ORDER BY lm.signed_leg_move_bps, lt.outcome_index
;
