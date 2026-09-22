#!/usr/bin/env bash
# Read-only operational probe for the two current market-data queries.  It
# intentionally reports the endpoint difference rather than normalizing it.
set -euo pipefail

symbol="${TG4_PARITY_SYMBOL:-eurusdrmp}"
normal_start="${TG4_PARITY_NY_START:-2025-03-07 00:00:00}"
normal_end="${TG4_PARITY_NY_END:-2025-03-11 00:00:00}"
utc_start="${TG4_PARITY_UTC_START:-2025-03-07 05:00:00+00}"
utc_end="${TG4_PARITY_UTC_END:-2025-03-11 04:00:00+00}"

result="$(PGOPTIONS='-c default_transaction_read_only=on' psql -X -v ON_ERROR_STOP=1 -At \
    -v symbol="$symbol" -v normal_start="$normal_start" -v normal_end="$normal_end" \
    -v utc_start="$utc_start" -v utc_end="$utc_end" <<'SQL'
WITH normal AS (
  SELECT row_number() OVER (ORDER BY dt) AS row_number,dt,open,close,high,low,vol
  FROM candlestick(:'symbol',15,'minute',:'normal_start',:'normal_end')
), tg AS (
  SELECT row_number() OVER (ORDER BY dt) AS row_number,dt,open,close,high,low,vol
  FROM candlestick(:'symbol',15,'minute',
       (:'utc_start'::timestamptz AT TIME ZONE 'America/New_York'),
       (:'utc_end'::timestamptz AT TIME ZONE 'America/New_York'))
  WHERE (dt AT TIME ZONE 'America/New_York') >= :'utc_start'::timestamptz
    AND (dt AT TIME ZONE 'America/New_York') <  :'utc_end'::timestamptz
), joined AS (
  SELECT normal.row_number AS normal_row,tg.row_number AS tg_row,
         normal.dt AS normal_dt,tg.dt AS tg_dt,
         normal.open AS normal_open,tg.open AS tg_open,
         normal.close AS normal_close,tg.close AS tg_close,
         normal.high AS normal_high,tg.high AS tg_high,
         normal.low AS normal_low,tg.low AS tg_low,
         normal.vol AS normal_vol,tg.vol AS tg_vol
  FROM normal FULL OUTER JOIN tg USING (row_number)
)
SELECT count(*) FILTER (WHERE normal_row IS NOT NULL) || '|' ||
       count(*) FILTER (WHERE tg_row IS NOT NULL) || '|' ||
       count(*) FILTER (WHERE normal_row IS NULL OR tg_row IS NULL OR
                         normal_dt<>tg_dt OR normal_open<>tg_open OR
                         normal_close<>tg_close OR normal_high<>tg_high OR
                         normal_low<>tg_low OR normal_vol<>tg_vol) || '|' ||
       coalesce((max(normal_dt) FILTER (WHERE normal_row IS NOT NULL))::text,'') || '|' ||
       coalesce((max(tg_dt) FILTER (WHERE tg_row IS NOT NULL))::text,'')
FROM joined;
SQL
)"

IFS='|' read -r normal_count tg_count mismatch_count normal_last tg_last <<<"$result"
printf 'TG4_MARKET_PARITY normal_rows=%s,tg_rows=%s,mismatches=%s,normal_last=%s,tg_last=%s\n' \
    "$normal_count" "$tg_count" "$mismatch_count" "$normal_last" "$tg_last"

if [[ "$normal_count" -ne "$((tg_count + 1))" || "$mismatch_count" -ne 1 ||
      "$normal_last" != "$normal_end" || "$tg_last" == "$normal_end" ]]; then
    echo 'TG4 market-data parity probe: unexpected query relationship' >&2
    exit 1
fi
