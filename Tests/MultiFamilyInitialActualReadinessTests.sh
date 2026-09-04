#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
ADMIN_USER="${LSTM_DB_ADMIN_USER:-${USER}}"
SOURCE_DB="${PHASE15_SOURCE_DB:-LSTM}"
DB_NAME="ea_multifamily_phase15_${$}"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ea-multifamily-phase15.XXXXXX")"
FIRST="$TEMP_DIR/first"
REPEAT="$TEMP_DIR/repeat"
AFTER="$TEMP_DIR/after"

case "$DB_NAME" in
    ea_multifamily_phase15_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$ADMIN_USER" \
        "$DB_NAME" >/dev/null 2>&1 || true
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

cd "$ROOT"
python3 -m unittest -v Tests/MultiFamilyInitialActualReadinessTests.py

if psql -X --host="$DB_HOST" --username="$ADMIN_USER" --dbname=postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';" | grep -q 1; then
    printf 'refusing to reuse existing database: %s\n' "$DB_NAME" >&2
    exit 1
fi

createdb --host="$DB_HOST" --username="$ADMIN_USER" \
    --template=template0 "$DB_NAME"
for migration in \
    072_economic_event.sql \
    081_economic_event_consensus.sql \
    082_economic_event_consensus_provider_provenance.sql \
    088_economic_event_release_actual_provenance.sql; do
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
        --username="$ADMIN_USER" --dbname="$DB_NAME" \
        -f "$ROOT/Database/migrations/$migration" >/dev/null
done

# Read production catalog evidence only; all writes target the disposable DB.
PGOPTIONS='-c default_transaction_read_only=on' \
pg_dump --host="$DB_HOST" --username="$ADMIN_USER" --dbname="$SOURCE_DB" \
    --data-only --no-owner --no-privileges \
    --table=public.economic_event \
    --table=public.economic_event_consensus \
    | psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
        --username="$ADMIN_USER" --dbname="$DB_NAME" >/dev/null

psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
    --username="$ADMIN_USER" --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/090_economic_event_actual_observation_provenance.sql" \
    >/dev/null

run_preparer() {
    local output="$1"
    PGHOST="$DB_HOST" PGUSER="$ADMIN_USER" \
        python3 "$ROOT/EconomicCalendar/prepare_multi_family_initial_actual_import.py" \
        --db "$DB_NAME" --output-directory "$output"
}

run_preparer "$FIRST" >/dev/null
run_preparer "$REPEAT" >/dev/null
diff -ru "$FIRST" "$REPEAT"

test "$(wc -l < "$FIRST/gdp-initial-actual-manifest.jsonl" | tr -d ' ')" = "67"
test "$(wc -l < "$FIRST/retail-sales-initial-actual-manifest.jsonl" | tr -d ' ')" = "194"
test "$(shasum -a 256 "$FIRST/gdp-initial-actual-manifest.jsonl" | awk '{print $1}')" = \
    "d0ae2e798fe24d01fa0b3a1c802fb86b7479b66a5d39507f985a99ecdcec8770"
test "$(shasum -a 256 "$FIRST/retail-sales-initial-actual-manifest.jsonl" | awk '{print $1}')" = \
    "fb971edf8483ed8975d22f9aecc1a401e1b429c0d1eabd8a57a5e7e0e1942d5e"

psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -f "$FIRST/gdp-initial-actual-import.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -f "$FIRST/retail-sales-initial-actual-import.sql" >/dev/null

test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc 'SELECT count(*) FROM economic_event_release_actual;')" = "261"
test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc 'SELECT count(*) FROM economic_event_feature_release_actual;')" = "261"
test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc \
    "SELECT count(*) FROM economic_event_actual_observation WHERE source_role='authoritative';")" = "261"
test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc \
    "SELECT count(*) FROM economic_event_release_actual WHERE publication_state <> 'initial' OR revision_sequence <> 0;")" = "0"

run_preparer "$AFTER" >/dev/null
cmp "$FIRST/gdp-initial-actual-manifest.jsonl" \
    "$AFTER/gdp-initial-actual-manifest.jsonl"
cmp "$FIRST/retail-sales-initial-actual-manifest.jsonl" \
    "$AFTER/retail-sales-initial-actual-manifest.jsonl"
test "$(cat "$AFTER/gdp-initial-actual-import.sql")" = $'BEGIN;\nCOMMIT;'
test "$(cat "$AFTER/retail-sales-initial-actual-import.sql")" = $'BEGIN;\nCOMMIT;'

cleanup
trap - EXIT
printf 'PHASE15_UNIT_TESTS=PASS\n'
printf 'PHASE15_BYTE_IDENTICAL_REGENERATION=PASS\n'
printf 'PHASE15_DISPOSABLE_IMPORT=PASS\n'
printf 'PHASE15_EXACT_DUPLICATE_IDEMPOTENCY=PASS\n'
printf 'PHASE15_FEATURE_VIEW_INITIAL_REVISION_ZERO_ONLY=PASS\n'
printf 'PHASE15_DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
