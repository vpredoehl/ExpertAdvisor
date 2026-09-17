-- AUDCAD 2023 Local/Dukascopy Data Integrity Certification
-- LSTM Trader / ExpertAdvisor
-- Saved 2026-09-16
--
-- Purpose:
--   Preserve the 2023 forensic/certification work as a permanent audit.
--
-- Frozen conclusions from the completed investigation:
--   * Production price side is ASK.
--   * Empirical 2023 local-clock mapping:
--       through 2023-03-24 trading: local = Dukascopy UTC - 5 hours
--       from 2023-03-26 through 2023-10-27 trading: UTC - 4 hours
--       from 2023-10-29 trading: UTC - 5 hours
--   * Price repair policy: LOCAL ASK OHLC -> DUKASCOPY ASK OHLC -> MISSING.
--   * No interpolation, forward fill, correlated-pair substitution, or OHLC synthesis.
--   * Existing candle vol semantics are SUM(raw vol), representing tick volume.
--   * For homogeneous reconstructed activity, Dukascopy tick_count is the canonical candidate.
--   * Dukascopy native askVolume is a separate candidate feature, not a replacement unit for tick volume.
--   * Do not overwrite audcadrmp; repaired data belongs in a provenance-bearing layer.
--
-- Prerequisite for sections 1-6:
--   tmp_duka_local_m1(dt, ask_open, ask_high, ask_low, ask_close, ask_volume, tick_count)
-- must exist in THIS PostgreSQL session and already be mapped to the certified local clock.
--
-- This script performs no persistent writes.

-- ============================================================
-- 1. LOCAL 15m activity: exact production volume semantics
-- ============================================================
DROP TABLE IF EXISTS tmp_local_activity_15m;
DROP TABLE IF EXISTS tmp_duka_activity_15m;
DROP TABLE IF EXISTS tmp_activity_15m_overlap;
DROP TABLE IF EXISTS tmp_activity_15m_features;

CREATE TEMP TABLE tmp_local_activity_15m AS
SELECT
    date_trunc('hour', time)
      + floor(extract(minute FROM time) / 15.0) * INTERVAL '15 minutes' AS dt,
    SUM(vol)::double precision AS local_vol,
    COUNT(*)::bigint AS local_row_count
FROM audcadrmp
WHERE time >= TIMESTAMP '2023-02-16 00:00:00'
  AND time <  TIMESTAMP '2023-08-01 00:00:00'
GROUP BY 1;

CREATE UNIQUE INDEX tmp_local_activity_15m_dt_idx ON tmp_local_activity_15m(dt);
ANALYZE tmp_local_activity_15m;

-- ============================================================
-- 2. Dukascopy 15m activity
-- ============================================================
CREATE TEMP TABLE tmp_duka_activity_15m AS
SELECT
    date_trunc('hour', dt)
      + floor(extract(minute FROM dt) / 15.0) * INTERVAL '15 minutes' AS dt,
    SUM(ask_volume)::double precision AS duka_volume,
    SUM(tick_count)::bigint AS duka_tick_count,
    COUNT(*)::integer AS duka_m1_count
FROM tmp_duka_local_m1
WHERE dt >= TIMESTAMP '2023-02-16 00:00:00'
  AND dt <  TIMESTAMP '2023-08-01 00:00:00'
GROUP BY 1;

CREATE UNIQUE INDEX tmp_duka_activity_15m_dt_idx ON tmp_duka_activity_15m(dt);
ANALYZE tmp_duka_activity_15m;

-- ============================================================
-- 3. Strict overlap: all 15 Dukascopy M1 bars present
-- ============================================================
CREATE TEMP TABLE tmp_activity_15m_overlap AS
SELECT
    l.dt, l.local_vol, l.local_row_count,
    d.duka_volume, d.duka_tick_count, d.duka_m1_count
FROM tmp_local_activity_15m l
JOIN tmp_duka_activity_15m d ON d.dt = l.dt
WHERE d.duka_m1_count = 15;

CREATE UNIQUE INDEX tmp_activity_15m_overlap_dt_idx ON tmp_activity_15m_overlap(dt);
ANALYZE tmp_activity_15m_overlap;

SELECT
    (SELECT COUNT(*) FROM tmp_local_activity_15m) AS local_15m_bars,
    (SELECT COUNT(*) FROM tmp_duka_activity_15m) AS duka_15m_bars,
    (SELECT COUNT(*) FROM tmp_duka_activity_15m WHERE duka_m1_count = 15)
        AS complete_duka_15m_bars,
    (SELECT COUNT(*) FROM tmp_activity_15m_overlap) AS strict_overlap_bars;

-- Tick-volume equivalence
SELECT
    COUNT(*) AS bars,
    ROUND(corr(local_vol, duka_volume)::numeric, 6) AS corr_localvol_dukavol,
    ROUND(corr(local_vol, duka_tick_count)::numeric, 6) AS corr_localvol_dukaticks,
    COUNT(*) FILTER (WHERE local_vol = duka_tick_count) AS exact_matches,
    ROUND(
        (100.0 * COUNT(*) FILTER (WHERE local_vol = duka_tick_count) / COUNT(*))::numeric,
        4
    ) AS exact_match_pct,
    COUNT(*) FILTER (WHERE local_vol <> duka_tick_count) AS different
FROM tmp_activity_15m_overlap;

-- Show every tick-volume discrepancy
SELECT
    dt, local_vol, duka_tick_count,
    local_vol - duka_tick_count AS difference,
    ROUND(
        (100.0 * (local_vol - duka_tick_count) / NULLIF(local_vol,0))::numeric,
        4
    ) AS pct_difference
FROM tmp_activity_15m_overlap
WHERE local_vol <> duka_tick_count
ORDER BY ABS(local_vol - duka_tick_count) DESC;

-- ============================================================
-- 4. Causal 96-bar activity normalization
-- ============================================================
CREATE TEMP TABLE tmp_activity_15m_features AS
WITH base AS (
    SELECT *,
        LN(1.0 + local_vol) AS log_local_vol,
        LN(1.0 + duka_volume) AS log_duka_volume,
        LN(1.0 + duka_tick_count) AS log_duka_ticks
    FROM tmp_activity_15m_overlap
),
rolling AS (
    SELECT *,
        COUNT(*) OVER w AS rolling_n,
        AVG(local_vol) OVER w AS local_vol_mean96,
        AVG(duka_volume) OVER w AS duka_vol_mean96,
        AVG(log_local_vol) OVER w AS log_local_mean96,
        STDDEV_SAMP(log_local_vol) OVER w AS log_local_sd96,
        AVG(log_duka_volume) OVER w AS log_duka_mean96,
        STDDEV_SAMP(log_duka_volume) OVER w AS log_duka_sd96,
        AVG(log_duka_ticks) OVER w AS log_duka_ticks_mean96,
        STDDEV_SAMP(log_duka_ticks) OVER w AS log_duka_ticks_sd96
    FROM base
    WINDOW w AS (
        ORDER BY dt ROWS BETWEEN 95 PRECEDING AND CURRENT ROW
    )
)
SELECT *,
    local_vol / NULLIF(local_vol_mean96,0) AS local_vol_relative96,
    duka_volume / NULLIF(duka_vol_mean96,0) AS duka_vol_relative96,
    (log_local_vol-log_local_mean96)/NULLIF(log_local_sd96,0) AS local_vol_z96,
    (log_duka_volume-log_duka_mean96)/NULLIF(log_duka_sd96,0) AS duka_vol_z96,
    (log_duka_ticks-log_duka_ticks_mean96)/NULLIF(log_duka_ticks_sd96,0) AS duka_ticks_z96
FROM rolling;

CREATE UNIQUE INDEX tmp_activity_15m_features_dt_idx ON tmp_activity_15m_features(dt);
ANALYZE tmp_activity_15m_features;

SELECT
    COUNT(*) AS bars,
    ROUND(corr(local_vol,duka_volume)::numeric,6) AS raw_volume_corr,
    ROUND(corr(log_local_vol,log_duka_volume)::numeric,6) AS log_volume_corr,
    ROUND(corr(local_vol_relative96,duka_vol_relative96)::numeric,6) AS relative96_corr,
    ROUND(corr(local_vol_z96,duka_vol_z96)::numeric,6) AS z96_volume_corr,
    ROUND(corr(local_vol_z96,duka_ticks_z96)::numeric,6) AS localvol_vs_dukaticks_z
FROM tmp_activity_15m_features
WHERE rolling_n = 96;

-- Monthly stability
SELECT
    date_trunc('month',dt)::date AS month,
    COUNT(*) AS bars,
    ROUND(corr(local_vol,duka_volume)::numeric,6) AS raw_vol_corr,
    ROUND(corr(log_local_vol,log_duka_volume)::numeric,6) AS log_vol_corr,
    ROUND(corr(local_vol_z96,duka_vol_z96)::numeric,6) AS z96_vol_corr,
    ROUND(corr(local_vol_z96,duka_ticks_z96)::numeric,6) AS localvol_vs_dukaticks_z
FROM tmp_activity_15m_features
WHERE rolling_n = 96
GROUP BY 1 ORDER BY 1;

-- ============================================================
-- 5. Five observed partial-outage candles: ASK OHLC comparison
-- ============================================================
WITH suspect(dt) AS (
    VALUES
        (TIMESTAMP '2023-03-09 09:45:00'),
        (TIMESTAMP '2023-07-27 12:45:00'),
        (TIMESTAMP '2023-04-03 01:45:00'),
        (TIMESTAMP '2023-04-03 02:45:00'),
        (TIMESTAMP '2023-07-06 04:45:00')
),
local_15m AS (
    SELECT
        date_trunc('hour', r.time)
          + floor(extract(minute FROM r.time)/15.0)*INTERVAL '15 minutes' AS dt,
        (array_agg(r.ask ORDER BY r.time))[1]::double precision AS local_open,
        MAX(r.ask)::double precision AS local_high,
        MIN(r.ask)::double precision AS local_low,
        (array_agg(r.ask ORDER BY r.time DESC))[1]::double precision AS local_close,
        COUNT(*)::bigint AS local_rows,
        SUM(r.vol)::double precision AS local_vol
    FROM audcadrmp r
    WHERE r.time >= TIMESTAMP '2023-03-09 09:45:00'
      AND r.time <  TIMESTAMP '2023-07-27 13:00:00'
    GROUP BY 1
),
duka_15m AS (
    SELECT
        date_trunc('hour', d.dt)
          + floor(extract(minute FROM d.dt)/15.0)*INTERVAL '15 minutes' AS dt,
        (array_agg(d.ask_open ORDER BY d.dt))[1]::double precision AS duka_open,
        MAX(d.ask_high)::double precision AS duka_high,
        MIN(d.ask_low)::double precision AS duka_low,
        (array_agg(d.ask_close ORDER BY d.dt DESC))[1]::double precision AS duka_close,
        SUM(d.tick_count)::bigint AS duka_ticks,
        COUNT(*)::integer AS duka_minutes
    FROM tmp_duka_local_m1 d
    WHERE d.dt >= TIMESTAMP '2023-03-09 09:45:00'
      AND d.dt <  TIMESTAMP '2023-07-27 13:00:00'
    GROUP BY 1
)
SELECT
    s.dt,
    l.local_rows, l.local_vol, d.duka_ticks,
    ROUND(l.local_open::numeric,6) AS local_open,
    ROUND(d.duka_open::numeric,6) AS duka_open,
    ROUND((ABS(l.local_open-d.duka_open)*10000)::numeric,3) AS open_diff_pips,
    ROUND(l.local_high::numeric,6) AS local_high,
    ROUND(d.duka_high::numeric,6) AS duka_high,
    ROUND((ABS(l.local_high-d.duka_high)*10000)::numeric,3) AS high_diff_pips,
    ROUND(l.local_low::numeric,6) AS local_low,
    ROUND(d.duka_low::numeric,6) AS duka_low,
    ROUND((ABS(l.local_low-d.duka_low)*10000)::numeric,3) AS low_diff_pips,
    ROUND(l.local_close::numeric,6) AS local_close,
    ROUND(d.duka_close::numeric,6) AS duka_close,
    ROUND((ABS(l.local_close-d.duka_close)*10000)::numeric,3) AS close_diff_pips,
    d.duka_minutes
FROM suspect s
LEFT JOIN local_15m l ON l.dt=s.dt
LEFT JOIN duka_15m d ON d.dt=s.dt
ORDER BY s.dt;

-- ============================================================
-- 6. Frozen interpretation / acceptance criteria
-- ============================================================
-- Expected certification values from the completed 2023 run:
--   strict overlap bars                    7650
--   local SUM(vol) == Duka tick_count      7645 / 7650 = 99.9346%
--   corr(local vol, Duka tick_count)       0.999941
--   corr(local vol, Duka ask volume)       0.925357
--   causal z96 local vol vs Duka ticks     0.999973
--   causal z96 local vol vs Duka volume    0.962359
--
-- Observed SUM(vol) discrepancies:
--   2023-03-09 09:45  local 2173, Duka 3012, delta -839
--   2023-07-27 12:45  local 1015, Duka 1233, delta -218
--   2023-04-03 01:45  local  790, Duka  887, delta  -97
--   2023-04-03 02:45  local  976, Duka 1024, delta  -48
--   2023-07-06 04:45  local 1659, Duka 1695, delta  -36
--
-- Direct raw-row counts on those five are lower still; raw COUNT(*) is NOT
-- the production local volume definition. Preserve SUM(raw vol).
--
-- Partial-outage ASK OHLC check:
--   maximum selected-candle OHLC discrepancy <= 1.0 pip.
-- Therefore no partial-outage replacement rule is currently justified.
--
-- Permanent policy:
--   PRICE: LOCAL ASK OHLC -> DUKASCOPY ASK OHLC -> MISSING.
--   ACTIVITY (reconstructed homogeneous dataset): Dukascopy tick_count.
--   DUKA askVolume: retain separately as candidate LSTM feature.
--   Never synthesize missing OHLC or activity.
