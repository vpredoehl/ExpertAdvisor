#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/economic_event_consensus_persistence_tests"
TEST_BIN="$BUILD_DIR/EconomicEventConsensusPersistenceTests"
CLI_BIN="$BUILD_DIR/EconomicEventConsensusImportCliHarness"
CANDIDATES="$ROOT/EconomicCalendar/raw/oanda/oanda_consensus_enrichment_candidates.csv"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
PRODUCTION_DB="${LSTM_PRODUCTION_DB_NAME:-LSTM}"
RUN_ID="$(date -u +%Y%m%d%H%M%S)_$$"
FOCUSED_DB="ea_consensus_phase1_test_${RUN_ID}"
INTEGRATION_DB="ea_consensus_phase1_integration_${RUN_ID}"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ea-consensus-phase1.XXXXXX")"
SEED_FILE="$TEMP_DIR/economic_event_seed.csv"
BAD_HEADER_FILE="$TEMP_DIR/missing_required_column.csv"

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_USER" \
        "$FOCUSED_DB" >/dev/null
    dropdb --if-exists --host="$DB_HOST" --username="$DB_USER" \
        "$INTEGRATION_DB" >/dev/null
    rm -f "$SEED_FILE" "$BAD_HEADER_FILE"
    rmdir "$TEMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

test -f "$CANDIDATES"
test "$(wc -l < "$CANDIDATES" | tr -d ' ')" = "1417"
mkdir -p "$BUILD_DIR"

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a PQXX_CFLAGS <<< "$(pkg-config --cflags libpqxx)"
    read -r -a PQXX_LIBS <<< "$(pkg-config --libs libpqxx)"
else
    PQXX_PREFIX="$(brew --prefix libpqxx@7.10.1)"
    LIBPQ_PREFIX="$(brew --prefix libpq)"
    PQXX_CFLAGS=("-I${PQXX_PREFIX}/include" "-I${LIBPQ_PREFIX}/include")
    PQXX_LIBS=("-L${PQXX_PREFIX}/lib" "-L${LIBPQ_PREFIX}/lib" -lpqxx -lpq)
fi

COMMON_SOURCES=(
    "$ROOT/Sources/EconomicEventConsensusImport.cpp"
    "$ROOT/Sources/EconomicEventConsensusRepository.cpp"
)

clang++ -std=c++20 -Wall -Wextra -Werror \
    "${PQXX_CFLAGS[@]}" -I"$ROOT/Sources" \
    "${COMMON_SOURCES[@]}" \
    "$ROOT/Tests/EconomicEventConsensusPersistenceTests.cpp" \
    "${PQXX_LIBS[@]}" -o "$TEST_BIN"

clang++ -std=c++20 -Wall -Wextra -Werror \
    "${PQXX_CFLAGS[@]}" -I"$ROOT/Sources" \
    "${COMMON_SOURCES[@]}" \
    "$ROOT/Tests/EconomicEventConsensusImportCliHarness.cpp" \
    "${PQXX_LIBS[@]}" -o "$CLI_BIN"

database_must_not_exist() {
    local database="$1"
    if psql -X --host="$DB_HOST" --username="$DB_USER" --dbname=postgres \
        -tAc "SELECT 1 FROM pg_database WHERE datname = '$database';" |
        grep -q 1; then
        echo "Refusing to reuse disposable database: $database" >&2
        exit 1
    fi
}

create_schema() {
    local database="$1"
    database_must_not_exist "$database"
    createdb --host="$DB_HOST" --username="$DB_USER" \
        --template=template0 "$database"
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
        --dbname="$database" \
        -f "$ROOT/Database/migrations/072_economic_event.sql" >/dev/null
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
        --dbname="$database" \
        -f "$ROOT/Database/migrations/080_economic_event_distinct_same_time_identity.sql" \
        >/dev/null
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
        --dbname="$database" \
        -f "$ROOT/Database/migrations/081_economic_event_consensus.sql" \
        >/dev/null
}

create_schema "$FOCUSED_DB"
LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
LSTM_DB_NAME="$FOCUSED_DB" CONSENSUS_INPUT="$CANDIDATES" "$TEST_BIN"

test "$(psql -X --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$FOCUSED_DB" -tAc \
    "SELECT count(*) FROM pg_constraint WHERE conrelid = 'economic_event_consensus'::regclass AND conname IN ('economic_event_consensus_pkey','economic_event_consensus_economic_event_id_fkey','economic_event_consensus_source_event_uq','economic_event_consensus_forecast_semantics_ck','economic_event_consensus_previous_semantics_ck','economic_event_consensus_actual_semantics_ck');")" = "6"

sed '1s/forecast_parse_status/forecast_parse_state/' \
    "$CANDIDATES" > "$BAD_HEADER_FILE"
if LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_NAME="$FOCUSED_DB" "$CLI_BIN" \
    --import-economic-consensus --input="$BAD_HEADER_FILE" --dry-run; then
    echo "Importer unexpectedly accepted a missing required column" >&2
    exit 1
fi

dropdb --host="$DB_HOST" --username="$DB_USER" "$FOCUSED_DB"
database_must_not_exist "$FOCUSED_DB"
echo "FOCUSED_DISPOSABLE_DATABASE_DROPPED=true"

PRODUCTION_COUNT_BEFORE="$(PGOPTIONS='-c default_transaction_read_only=on' \
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -tAc \
    "SELECT current_setting('transaction_read_only') || '|' || count(*) FROM economic_event;")"
PRODUCTION_CONSENSUS_BEFORE="$(PGOPTIONS='-c default_transaction_read_only=on' \
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -tAc \
    "SELECT current_setting('transaction_read_only') || '|' || (to_regclass('public.economic_event_consensus') IS NULL)::text;")"
test "$PRODUCTION_CONSENSUS_BEFORE" = "on|true"

CANDIDATE_IDS="$(tail -n +2 "$CANDIDATES" | cut -d, -f1 | paste -sd, -)"
PGOPTIONS='-c default_transaction_read_only=on' \
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -c "\copy (
        SELECT economic_event_id, currency, event_family, event_timestamp_utc,
               source_agency, source_event_id, source_url, reference_period,
               event_importance, historical_time_confidence,
               source_release_date, source_release_time, source_timezone
        FROM economic_event
        WHERE economic_event_id = ANY(
            string_to_array('$CANDIDATE_IDS', ',')::bigint[])
        ORDER BY economic_event_id
    ) TO '$SEED_FILE' CSV HEADER" >/dev/null
test "$(wc -l < "$SEED_FILE" | tr -d ' ')" = "1417"

create_schema "$INTEGRATION_DB"
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$INTEGRATION_DB" -c "\copy economic_event (
        economic_event_id, currency, event_family, event_timestamp_utc,
        source_agency, source_event_id, source_url, reference_period,
        event_importance, historical_time_confidence,
        source_release_date, source_release_time, source_timezone
    ) FROM '$SEED_FILE' CSV HEADER" >/dev/null

DRY_RUN_OUTPUT="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_NAME="$INTEGRATION_DB" "$CLI_BIN" \
    --import-economic-consensus --input="$CANDIDATES" --dry-run)"
grep -Fq 'mode=dry-run,input=1416,inserted=1416,unchanged=0,rejected=0' \
    <<< "$DRY_RUN_OUTPUT"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$INTEGRATION_DB" -tAc \
    'SELECT count(*) FROM economic_event_consensus;')" = "0"

FIRST_APPLY_OUTPUT="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_NAME="$INTEGRATION_DB" "$CLI_BIN" \
    --import-economic-consensus --input="$CANDIDATES" --apply)"
grep -Fq 'mode=apply,input=1416,inserted=1416,unchanged=0,rejected=0' \
    <<< "$FIRST_APPLY_OUTPUT"
SECOND_APPLY_OUTPUT="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_NAME="$INTEGRATION_DB" "$CLI_BIN" \
    --import-economic-consensus --input="$CANDIDATES" --apply)"
grep -Fq 'mode=apply,input=1416,inserted=0,unchanged=1416,rejected=0' \
    <<< "$SECOND_APPLY_OUTPUT"

psql_scalar() {
    psql -X --host="$DB_HOST" --username="$DB_USER" \
        --dbname="$INTEGRATION_DB" -tAc "$1"
}

test "$(psql_scalar 'SELECT count(*) FROM economic_event_consensus;')" = "1416"
test "$(psql_scalar "SELECT count(*) FROM economic_event_consensus WHERE forecast_parse_status = 'missing' AND forecast_raw IS NULL AND forecast_value_low IS NULL;")" = "11"
test "$(psql_scalar "SELECT array_agg(economic_event_id ORDER BY economic_event_id)::text FROM economic_event_consensus WHERE forecast_parse_status = 'missing';")" = "{81,94,118,127,195,685,1659,2211,2212,2301,2569}"
test "$(psql_scalar "SELECT count(*) FROM economic_event_consensus c JOIN economic_event e USING (economic_event_id) WHERE (e.event_family = 'CPI' AND c.source_report_id NOT IN (699)) OR (e.event_family = 'DURABLE_GOODS' AND c.source_report_id <> 59) OR (e.event_family = 'EMPLOYMENT' AND c.source_report_id <> 707) OR (e.event_family = 'FOMC' AND c.source_report_id <> 82) OR (e.event_family = 'GDP' AND c.source_report_id <> 690) OR (e.event_family = 'JOLTS' AND c.source_report_id <> 1371) OR (e.event_family = 'PCE' AND c.source_report_id <> 694) OR (e.event_family = 'PPI' AND c.source_report_id <> 703) OR (e.event_family = 'RETAIL_SALES' AND c.source_report_id <> 696);")" = "1"
test "$(psql_scalar "SELECT count(*) FROM economic_event_consensus c JOIN economic_event e USING (economic_event_id) WHERE c.economic_event_id = 779 AND e.event_family = 'CPI' AND e.source_release_date = DATE '2025-12-18' AND c.source_report_id = 698 AND c.forecast_qualifier = 'y/y';")" = "1"
test "$(psql_scalar 'SELECT count(*) FROM (SELECT economic_event_id FROM economic_event_consensus GROUP BY 1 HAVING count(*) > 1) d;')" = "0"
test "$(psql_scalar 'SELECT count(*) FROM (SELECT consensus_source, source_event_id FROM economic_event_consensus GROUP BY 1,2 HAVING count(DISTINCT economic_event_id) > 1) d;')" = "0"

EXPECTED_FAMILY_COUNTS=$'CPI|181\nDURABLE_GOODS|181\nEMPLOYMENT|180\nFOMC|122\nGDP|179\nJOLTS|32\nPCE|178\nPPI|181\nRETAIL_SALES|182'
ACTUAL_FAMILY_COUNTS="$(psql -X --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$INTEGRATION_DB" -At -F '|' -c \
    'SELECT e.event_family, count(*) FROM economic_event_consensus c JOIN economic_event e USING (economic_event_id) GROUP BY e.event_family ORDER BY e.event_family;')"
test "$ACTUAL_FAMILY_COUNTS" = "$EXPECTED_FAMILY_COUNTS"

dropdb --host="$DB_HOST" --username="$DB_USER" "$INTEGRATION_DB"
database_must_not_exist "$INTEGRATION_DB"
echo "INTEGRATION_DISPOSABLE_DATABASE_DROPPED=true"

PRODUCTION_COUNT_AFTER="$(PGOPTIONS='-c default_transaction_read_only=on' \
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -tAc \
    "SELECT current_setting('transaction_read_only') || '|' || count(*) FROM economic_event;")"
PRODUCTION_CONSENSUS_AFTER="$(PGOPTIONS='-c default_transaction_read_only=on' \
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -tAc \
    "SELECT current_setting('transaction_read_only') || '|' || (to_regclass('public.economic_event_consensus') IS NULL)::text;")"
test "$PRODUCTION_COUNT_AFTER" = "$PRODUCTION_COUNT_BEFORE"
test "$PRODUCTION_CONSENSUS_AFTER" = "on|true"

echo "FULL_IMPORT_ROWS=1416"
echo "FULL_IMPORT_MISSING_FORECASTS=11"
echo "PRODUCTION_READ_ONLY_VERIFIED=true"
cleanup
trap - EXIT
