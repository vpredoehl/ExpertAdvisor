#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/economic_event_consensus_persistence_tests"
TEST_BIN="$BUILD_DIR/EconomicEventConsensusPersistenceTests"
CLI_BIN="$BUILD_DIR/EconomicEventConsensusImportCliHarness"
CANDIDATES="$ROOT/EconomicCalendar/raw/oanda/oanda_consensus_enrichment_candidates.csv"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
DB_ADMIN_USER="${LSTM_DB_ADMIN_USER:-${USER:-vjp}}"
PRODUCTION_DB="${LSTM_PRODUCTION_DB_NAME:-LSTM}"
RUN_ID="$(date -u +%Y%m%d%H%M%S)_$$"
FOCUSED_DB="ea_consensus_phase1_test_${RUN_ID}"
INTEGRATION_DB="ea_consensus_phase1_integration_${RUN_ID}"
WORKFLOW_DB="ea_consensus_authoritative_workflow_${RUN_ID}"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ea-consensus-phase1.XXXXXX")"
SEED_FILE="$TEMP_DIR/economic_event_seed.csv"
ALL_EVENT_SEED_FILE="$TEMP_DIR/all_economic_event_seed.csv"
CONSENSUS_SEED_FILE="$TEMP_DIR/economic_event_consensus_seed.csv"
BAD_HEADER_FILE="$TEMP_DIR/missing_required_column.csv"

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_ADMIN_USER" \
        "$FOCUSED_DB" >/dev/null
    dropdb --if-exists --host="$DB_HOST" --username="$DB_ADMIN_USER" \
        "$INTEGRATION_DB" >/dev/null
    dropdb --if-exists --host="$DB_HOST" --username="$DB_ADMIN_USER" \
        "$WORKFLOW_DB" >/dev/null
    rm -f "$SEED_FILE" "$ALL_EVENT_SEED_FILE" "$CONSENSUS_SEED_FILE" \
        "$BAD_HEADER_FILE"
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
    createdb --host="$DB_HOST" --username="$DB_ADMIN_USER" --owner="$DB_USER" \
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
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
        --dbname="$database" \
        -f "$ROOT/Database/migrations/082_economic_event_consensus_provider_provenance.sql" \
        >/dev/null
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
        --dbname="$database" \
        -f "$ROOT/Database/migrations/091_weekly_claims_historical_consensus.sql" \
        >/dev/null
}

create_schema "$FOCUSED_DB"
LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
LSTM_DB_NAME="$FOCUSED_DB" CONSENSUS_INPUT="$CANDIDATES" "$TEST_BIN"

test "$(psql -X --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$FOCUSED_DB" -tAc \
    "SELECT count(*) FROM pg_constraint WHERE conrelid = 'economic_event_consensus'::regclass AND conname IN ('economic_event_consensus_pkey','economic_event_consensus_economic_event_id_fkey','economic_event_consensus_provider_observation_uq','economic_event_consensus_forecast_semantics_ck','economic_event_consensus_previous_semantics_ck','economic_event_consensus_actual_semantics_ck');")" = "6"

sed '1s/forecast_parse_status/forecast_parse_state/' \
    "$CANDIDATES" > "$BAD_HEADER_FILE"
if LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_NAME="$FOCUSED_DB" "$CLI_BIN" \
    --import-economic-consensus --input="$BAD_HEADER_FILE" --dry-run; then
    echo "Importer unexpectedly accepted a missing required column" >&2
    exit 1
fi

dropdb --host="$DB_HOST" --username="$DB_ADMIN_USER" "$FOCUSED_DB"
database_must_not_exist "$FOCUSED_DB"
echo "FOCUSED_DISPOSABLE_DATABASE_DROPPED=true"

PRODUCTION_COUNT_BEFORE="$(PGOPTIONS='-c default_transaction_read_only=on' \
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -tAc \
    "SELECT current_setting('transaction_read_only') || '|' || count(*) FROM economic_event;")"
PRODUCTION_CONSENSUS_BEFORE="$(PGOPTIONS='-c default_transaction_read_only=on' \
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -tAc \
    "SELECT current_setting('transaction_read_only') || '|' || count(*) FROM economic_event_consensus;")"
test "$PRODUCTION_CONSENSUS_BEFORE" = "on|1532"

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

dropdb --host="$DB_HOST" --username="$DB_ADMIN_USER" "$INTEGRATION_DB"
database_must_not_exist "$INTEGRATION_DB"
echo "INTEGRATION_DISPOSABLE_DATABASE_DROPPED=true"

# Exercise the forward 081 -> 082 upgrade and the authoritative mixed-provider
# workflow against a disposable clone of production canonical/OANDA state.
PGOPTIONS='-c default_transaction_read_only=on' \
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -c "\copy (
        SELECT economic_event_id, currency, event_family, event_timestamp_utc,
               source_agency, source_event_id, source_url, reference_period,
               event_importance, historical_time_confidence,
               source_release_date, source_release_time, source_timezone,
               imported_at
        FROM economic_event ORDER BY economic_event_id
    ) TO '$ALL_EVENT_SEED_FILE' CSV HEADER" >/dev/null
PGOPTIONS='-c default_transaction_read_only=on' \
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -c "\copy (
        SELECT economic_event_id, consensus_source, source_report_id,
               source_event_id, source_event_name, source_period,
               source_priority, source_timestamp_epoch, source_date,
               source_artifact_path, match_rule, semantic_contract,
               forecast_raw, forecast_parse_status, forecast_value_kind,
               forecast_value_low, forecast_value_high,
               forecast_canonical_value_low, forecast_canonical_value_high,
               forecast_unit, forecast_scale, forecast_qualifier,
               previous_raw, previous_parse_status, previous_value_kind,
               previous_value_low, previous_value_high,
               previous_canonical_value_low, previous_canonical_value_high,
               previous_unit, previous_scale, previous_qualifier,
               actual_raw, actual_parse_status, actual_value_kind,
               actual_value_low, actual_value_high,
               actual_canonical_value_low, actual_canonical_value_high,
               actual_unit, actual_scale, actual_qualifier, imported_at
        FROM economic_event_consensus
        WHERE consensus_source = 'OANDA'
        ORDER BY economic_event_id
    ) TO '$CONSENSUS_SEED_FILE' CSV HEADER" >/dev/null

database_must_not_exist "$WORKFLOW_DB"
createdb --host="$DB_HOST" --username="$DB_ADMIN_USER" --owner="$DB_USER" --template=template0 \
    "$WORKFLOW_DB"
for migration in \
    072_economic_event.sql \
    080_economic_event_distinct_same_time_identity.sql \
    081_economic_event_consensus.sql; do
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
        --dbname="$WORKFLOW_DB" \
        -f "$ROOT/Database/migrations/$migration" >/dev/null
done
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$WORKFLOW_DB" \
    -c "\copy economic_event FROM '$ALL_EVENT_SEED_FILE' CSV HEADER" >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$WORKFLOW_DB" \
    -c "\copy economic_event_consensus FROM '$CONSENSUS_SEED_FILE' CSV HEADER" \
    >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$WORKFLOW_DB" \
    -f "$ROOT/Database/migrations/082_economic_event_consensus_provider_provenance.sql" \
    >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$WORKFLOW_DB" \
    -f "$ROOT/Database/migrations/091_weekly_claims_historical_consensus.sql" \
    >/dev/null

EVIDENCE_ROOT="$ROOT/EconomicCalendar/raw"

EXPECTED_MYFXBOOK_SHA256="1538055386838d8ec50ab9cce40153f69aa8bf841c2ae50f67751b2c465432cb"
MYFXBOOK_CAPTURE="$EVIDENCE_ROOT/myfxbook/myfxbook_consensus_history.json"

ACTUAL_MYFXBOOK_SHA256="$(shasum -a 256 "$MYFXBOOK_CAPTURE" | awk '{print $1}')"
test "$ACTUAL_MYFXBOOK_SHA256" = "$EXPECTED_MYFXBOOK_SHA256"

WORKFLOW_DRY_RUN="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_NAME="$WORKFLOW_DB" "$CLI_BIN" \
    --import-economic-consensus --evidence-root="$EVIDENCE_ROOT" --dry-run)"
grep -Fq 'canonical_events_examined=2601,oanda_candidates=1405,oanda_matched_blanks=11,myfxbook_jolts_gap_candidates=113,myfxbook_oanda_blank_candidates=3,matched_canonical_events=1521,missing_canonical_matches=0,ambiguous_canonical_matches=0,source_exclusions=69,inserted=116,unchanged=1405,rejected=0' \
    <<< "$WORKFLOW_DRY_RUN"

WORKFLOW_FIRST_APPLY="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_NAME="$WORKFLOW_DB" "$CLI_BIN" \
    --import-economic-consensus --evidence-root="$EVIDENCE_ROOT" --apply)"
grep -Fq 'inserted=116,unchanged=1405,rejected=0' \
    <<< "$WORKFLOW_FIRST_APPLY"
WORKFLOW_SECOND_APPLY="$(LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
    LSTM_DB_NAME="$WORKFLOW_DB" "$CLI_BIN" \
    --import-economic-consensus --evidence-root="$EVIDENCE_ROOT" --apply)"
grep -Fq 'inserted=0,unchanged=1521,rejected=0' \
    <<< "$WORKFLOW_SECOND_APPLY"

workflow_scalar() {
    psql -X --host="$DB_HOST" --username="$DB_USER" \
        --dbname="$WORKFLOW_DB" -tAc "$1"
}
test "$(workflow_scalar 'SELECT count(*) FROM economic_event_consensus;')" = "1532"
test "$(workflow_scalar "SELECT count(*) FROM pg_indexes WHERE tablename = 'economic_event_consensus' AND indexname = 'economic_event_consensus_one_populated_per_event_uq';")" = "1"
test "$(workflow_scalar 'SELECT count(*) FROM economic_event_selected_consensus;')" = "1521"
test "$(workflow_scalar 'SELECT count(*) FROM economic_event_consensus WHERE provider_observed_at IS NOT NULL OR forecast_available_at IS NOT NULL OR source_retrieved_at IS NOT NULL OR forecast_availability_proof IS NOT NULL;')" = "0"
test "$(workflow_scalar "SELECT count(*) FROM economic_event_selected_consensus WHERE consensus_source = 'OANDA';")" = "1405"
test "$(workflow_scalar "SELECT count(*) FROM economic_event_selected_consensus WHERE consensus_source = 'MYFXBOOK';")" = "116"
test "$(workflow_scalar "SELECT count(*) FROM economic_event_selected_consensus WHERE candidate_classification = 'myfxbook_jolts_gap_fill';")" = "113"
test "$(workflow_scalar "SELECT count(*) FROM economic_event_selected_consensus WHERE candidate_classification = 'myfxbook_oanda_blank_fill';")" = "3"
test "$(workflow_scalar "SELECT count(*) FROM economic_event_selected_consensus c JOIN economic_event e USING (economic_event_id) WHERE c.consensus_source = 'MYFXBOOK' AND e.source_agency IN ('BLS','CENSUS');")" = "116"
test "$(workflow_scalar "SELECT count(*) FROM economic_event_selected_consensus c JOIN economic_event e USING (economic_event_id) WHERE c.consensus_source = 'MYFXBOOK' AND c.candidate_classification = 'myfxbook_jolts_gap_fill' AND e.source_agency = 'BLS';")" = "113"
test "$(workflow_scalar "SELECT count(*) FROM economic_event_selected_consensus WHERE source_artifact_path = 'EconomicCalendar/raw/myfxbook/myfxbook_consensus_history.json' AND source_artifact_sha256 = '1538055386838d8ec50ab9cce40153f69aa8bf841c2ae50f67751b2c465432cb' AND provider_provenance ? 'source_observation_ordinal';")" = "116"
test "$(workflow_scalar "SELECT count(*) FROM economic_event_selected_consensus c JOIN economic_event e USING (economic_event_id) WHERE e.event_family = 'FOMC' AND e.source_release_date IN (DATE '2020-03-03', DATE '2020-03-15');")" = "0"
test "$(workflow_scalar "SELECT string_agg(e.event_family || ':' || e.source_release_date::text || ':' || c.consensus_value_low::text, ',' ORDER BY e.source_release_date) FROM economic_event_selected_consensus c JOIN economic_event e USING (economic_event_id) WHERE c.candidate_classification = 'myfxbook_oanda_blank_fill';")" = "PPI:2013-12-13:-0.1,RETAIL_SALES:2022-09-15:0.0,CPI:2023-12-12:0.0"
test "$(workflow_scalar "SELECT string_agg(e.source_release_date::text || ':' || c.consensus_value_low::text, ',' ORDER BY e.source_release_date) FROM economic_event_selected_consensus c JOIN economic_event e USING (economic_event_id) WHERE c.candidate_classification = 'myfxbook_jolts_gap_fill' AND e.source_release_date IN (DATE '2014-07-08', DATE '2023-11-01');")" = "2014-07-08:4530000,2023-11-01:9250000"

dropdb --host="$DB_HOST" --username="$DB_ADMIN_USER" "$WORKFLOW_DB"
database_must_not_exist "$WORKFLOW_DB"
echo "WORKFLOW_DISPOSABLE_DATABASE_DROPPED=true"

PRODUCTION_COUNT_AFTER="$(PGOPTIONS='-c default_transaction_read_only=on' \
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -tAc \
    "SELECT current_setting('transaction_read_only') || '|' || count(*) FROM economic_event;")"
PRODUCTION_CONSENSUS_AFTER="$(PGOPTIONS='-c default_transaction_read_only=on' \
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$PRODUCTION_DB" -tAc \
    "SELECT current_setting('transaction_read_only') || '|' || count(*) FROM economic_event_consensus;")"
test "$PRODUCTION_COUNT_AFTER" = "$PRODUCTION_COUNT_BEFORE"
test "$PRODUCTION_CONSENSUS_AFTER" = "$PRODUCTION_CONSENSUS_BEFORE"

echo "FULL_IMPORT_ROWS=1416"
echo "FULL_IMPORT_MISSING_FORECASTS=11"
echo "PRODUCTION_READ_ONLY_VERIFIED=true"
cleanup
trap - EXIT
