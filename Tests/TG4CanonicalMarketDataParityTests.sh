#!/usr/bin/env bash
# Read-only production-data proof for the shared absolute [start,end) SQL
# contract. It neither invokes LSTM_Release nor changes database state.
set -euo pipefail

symbol="${TG4_PARITY_SYMBOL:-eurusdrmp}"

assert_case() {
    local name="$1"
    local start="$2"
    local end="$3"
    local result
    result="$(PGOPTIONS='-c default_transaction_read_only=on' psql -X -v ON_ERROR_STOP=1 -At \
        -v symbol="$symbol" -v start="$start" -v end="$end" <<'SQL'
WITH normal_canonical AS (
  SELECT dt,open,close,high,low,vol
  FROM candlestick(:'symbol',15,'minute',
       (:'start'::timestamptz AT TIME ZONE 'America/New_York'),
       (:'end'::timestamptz AT TIME ZONE 'America/New_York'))
  WHERE (dt AT TIME ZONE 'America/New_York') >= :'start'::timestamptz
    AND (dt AT TIME ZONE 'America/New_York') <  :'end'::timestamptz
), tg_canonical AS (
  SELECT dt,open,close,high,low,vol
  FROM candlestick(:'symbol',15,'minute',
       (:'start'::timestamptz AT TIME ZONE 'America/New_York'),
       (:'end'::timestamptz AT TIME ZONE 'America/New_York'))
  WHERE (dt AT TIME ZONE 'America/New_York') >= :'start'::timestamptz
    AND (dt AT TIME ZONE 'America/New_York') <  :'end'::timestamptz
), paired AS (
  SELECT row_number() OVER (ORDER BY dt) AS ordinal,dt,open,close,high,low,vol
  FROM normal_canonical
), paired_tg AS (
  SELECT row_number() OVER (ORDER BY dt) AS ordinal,dt,open,close,high,low,vol
  FROM tg_canonical
), compared AS (
  SELECT n.ordinal,n.dt AS n_dt,t.dt AS t_dt,n.open AS n_open,t.open AS t_open,
         n.close AS n_close,t.close AS t_close,n.high AS n_high,t.high AS t_high,
         n.low AS n_low,t.low AS t_low,n.vol AS n_vol,t.vol AS t_vol
  FROM paired n FULL OUTER JOIN paired_tg t USING (ordinal)
)
SELECT (SELECT count(*) FROM normal_canonical) || '|' ||
       (SELECT count(*) FROM tg_canonical) || '|' ||
       count(*) FILTER (WHERE n_dt IS NULL OR t_dt IS NULL OR n_dt<>t_dt OR
                         n_open<>t_open OR n_close<>t_close OR n_high<>t_high OR
                         n_low<>t_low OR n_vol<>t_vol) || '|' ||
       (SELECT count(*) FROM normal_canonical
        WHERE (dt AT TIME ZONE 'America/New_York') = :'end'::timestamptz)
FROM compared;
SQL
)"
    IFS='|' read -r normal_count tg_count mismatches endpoint_count <<<"$result"
    printf 'TG4_CANONICAL_MARKET_PARITY case=%s,normal_rows=%s,tg_rows=%s,mismatches=%s,endpoint_rows=%s\n' \
        "$name" "$normal_count" "$tg_count" "$mismatches" "$endpoint_count"
    [[ "$normal_count" -gt 0 && "$normal_count" == "$tg_count" &&
       "$mismatches" -eq 0 && "$endpoint_count" -eq 0 ]]
}

assert_case ordinary '2025-03-05 05:00:00+00' '2025-03-06 05:00:00+00'
assert_case dst_spring '2025-03-07 05:00:00+00' '2025-03-11 04:00:00+00'
assert_case dst_fall '2025-10-31 04:00:00+00' '2025-11-04 05:00:00+00'
assert_case weekend_gap '2025-03-14 04:00:00+00' '2025-03-18 04:00:00+00'

echo "TG4CanonicalMarketDataParityTests passed"
