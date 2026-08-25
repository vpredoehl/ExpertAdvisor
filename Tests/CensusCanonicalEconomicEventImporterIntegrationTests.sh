#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/census_canonical_importer_integration"
DB_NAME="ea_census_canonical_$$"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
DB_PORT="${LSTM_DB_PORT:-}"
IMPORTER="$ROOT/EconomicCalendar/import_census_events.py"
INPUT="$ROOT/EconomicCalendar/raw/census/census_events_extracted.csv"

mkdir -p "$BUILD_DIR"

PSQL_TARGET=(--host="$DB_HOST" --username="$DB_USER")
if [[ -n "$DB_PORT" ]]; then
    PSQL_TARGET+=(--port="$DB_PORT")
fi

IMPORT_TARGET=(--host "$DB_HOST" --user "$DB_USER")
if [[ -n "$DB_PORT" ]]; then
    IMPORT_TARGET+=(--port "$DB_PORT")
fi

cleanup() {
    if [[ "$DB_NAME" != ea_census_canonical_* ]]; then
        echo "Refusing unsafe cleanup target: $DB_NAME" >&2
        exit 1
    fi
    dropdb --if-exists "${PSQL_TARGET[@]}" "$DB_NAME" >/dev/null
}
trap cleanup EXIT

if psql -X "${PSQL_TARGET[@]}" --dbname=postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';" | grep -q 1; then
    echo "Refusing to reuse existing disposable database: $DB_NAME" >&2
    exit 1
fi

createdb "${PSQL_TARGET[@]}" --template=template0 "$DB_NAME"
psql -X -v ON_ERROR_STOP=1 "${PSQL_TARGET[@]}" --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/072_economic_event.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 "${PSQL_TARGET[@]}" --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/080_economic_event_distinct_same_time_identity.sql" >/dev/null

ROWS_BEFORE="$(psql -X "${PSQL_TARGET[@]}" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event;')"
DRY_RUN_OUTPUT="$(python3 "$IMPORTER" --dry-run --input "$INPUT" \
    --prepared-report "$BUILD_DIR/dry_run_prepared.csv" \
    --db "$DB_NAME" "${IMPORT_TARGET[@]}")"
grep -Fq 'Would insert        : 399' <<< "$DRY_RUN_OUTPUT"
grep -Fq 'Would update        : 0' <<< "$DRY_RUN_OUTPUT"
grep -Fq 'Unchanged           : 0' <<< "$DRY_RUN_OUTPUT"
grep -Fq 'DATABASE WRITES: 0 (read-only transaction enforced)' <<< "$DRY_RUN_OUTPUT"
ROWS_AFTER="$(psql -X "${PSQL_TARGET[@]}" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event;')"
test "$ROWS_BEFORE" = "0"
test "$ROWS_AFTER" = "$ROWS_BEFORE"

FIRST_COMMIT_OUTPUT="$(python3 "$IMPORTER" --commit --input "$INPUT" \
    --prepared-report "$BUILD_DIR/first_commit_prepared.csv" \
    --db "$DB_NAME" "${IMPORT_TARGET[@]}")"
grep -Fq 'pre_matching_exact=0' <<< "$FIRST_COMMIT_OUTPUT"
grep -Fq 'inserted=399' <<< "$FIRST_COMMIT_OUTPUT"
grep -Fq 'post_matching_exact=399' <<< "$FIRST_COMMIT_OUTPUT"
grep -Fq 'RESULT: COMMITTED - all 399 Census rows verified atomically' \
    <<< "$FIRST_COMMIT_OUTPUT"

test "$(psql -X "${PSQL_TARGET[@]}" --dbname="$DB_NAME" -tAc \
    "SELECT count(*) FROM economic_event WHERE source_agency = 'CENSUS';")" = "399"
test "$(psql -X "${PSQL_TARGET[@]}" --dbname="$DB_NAME" -tAc \
    "SELECT count(*) FROM economic_event WHERE source_agency = 'CENSUS' AND event_family = 'RETAIL_SALES';")" = "200"
test "$(psql -X "${PSQL_TARGET[@]}" --dbname="$DB_NAME" -tAc \
    "SELECT count(*) FROM economic_event WHERE source_agency = 'CENSUS' AND event_family = 'DURABLE_GOODS';")" = "199"
test "$(psql -X "${PSQL_TARGET[@]}" --dbname="$DB_NAME" -tAc \
    "SELECT count(DISTINCT source_event_id) FROM economic_event WHERE source_agency = 'CENSUS';")" = "399"
test "$(psql -X "${PSQL_TARGET[@]}" --dbname="$DB_NAME" -tAc \
    "SELECT count(*) FROM (SELECT source_agency, event_family, event_timestamp_utc FROM economic_event WHERE source_agency = 'CENSUS' GROUP BY 1,2,3 HAVING count(*) > 1) duplicates;")" = "0"

IDENTITY_HASH_BEFORE="$(psql -X "${PSQL_TARGET[@]}" --dbname="$DB_NAME" -tAc \
    "SELECT md5(string_agg(source_event_id, ',' ORDER BY source_event_id)) FROM economic_event WHERE source_agency = 'CENSUS';")"

SECOND_COMMIT_OUTPUT="$(python3 "$IMPORTER" --commit --input "$INPUT" \
    --prepared-report "$BUILD_DIR/second_commit_prepared.csv" \
    --db "$DB_NAME" "${IMPORT_TARGET[@]}")"
grep -Fq 'pre_matching_exact=399' <<< "$SECOND_COMMIT_OUTPUT"
grep -Fq 'inserted=0' <<< "$SECOND_COMMIT_OUTPUT"
grep -Fq 'post_matching_exact=399' <<< "$SECOND_COMMIT_OUTPUT"
test "$(psql -X "${PSQL_TARGET[@]}" --dbname="$DB_NAME" -tAc \
    "SELECT count(*) FROM economic_event WHERE source_agency = 'CENSUS';")" = "399"

IDENTITY_HASH_AFTER="$(psql -X "${PSQL_TARGET[@]}" --dbname="$DB_NAME" -tAc \
    "SELECT md5(string_agg(source_event_id, ',' ORDER BY source_event_id)) FROM economic_event WHERE source_agency = 'CENSUS';")"
test "$IDENTITY_HASH_BEFORE" = "$IDENTITY_HASH_AFTER"

RERUN_DRY_OUTPUT="$(python3 "$IMPORTER" --dry-run --input "$INPUT" \
    --prepared-report "$BUILD_DIR/rerun_dry_prepared.csv" \
    --db "$DB_NAME" "${IMPORT_TARGET[@]}")"
grep -Fq 'Would insert        : 0' <<< "$RERUN_DRY_OUTPUT"
grep -Fq 'Unchanged           : 399' <<< "$RERUN_DRY_OUTPUT"
grep -Fq 'Rejected conflicts  : 0' <<< "$RERUN_DRY_OUTPUT"

echo "DISPOSABLE_DATABASE_USED=$DB_NAME"
cleanup
trap - EXIT
echo "DISPOSABLE_DATABASE_DROPPED=true"
