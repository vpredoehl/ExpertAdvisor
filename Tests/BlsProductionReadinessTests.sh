#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
ADMIN_USER="${LSTM_DB_ADMIN_USER:-${USER}}"
SOURCE_DB="${PHASE17_SOURCE_DB:-LSTM}"
DB_NAME="ea_bls_phase17_${$}"
TEMP_DIR="$(mktemp -d "/tmp/ea-bls-phase17.XXXXXX")"
FIRST="$TEMP_DIR/first"
REPEAT="$TEMP_DIR/repeat"
AFTER="$TEMP_DIR/after"

case "$DB_NAME" in
    ea_bls_phase17_[0-9]*) ;;
    *) exit 90 ;;
esac
case "$TEMP_DIR" in
    /tmp/ea-bls-phase17.*|/private/tmp/ea-bls-phase17.*) ;;
    *) exit 91 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$ADMIN_USER" \
        "$DB_NAME" >/dev/null 2>&1 || true
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

cd "$ROOT"
Tests/BlsEconomicEventReleaseActualTests.sh

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

# Production is a read-only source; every mutation below targets DB_NAME.
PGOPTIONS='-c default_transaction_read_only=on' \
pg_dump --host="$DB_HOST" --username="$ADMIN_USER" --dbname="$SOURCE_DB" \
    --data-only --disable-triggers --no-owner --no-privileges \
    --table=public.economic_event \
    --table=public.economic_event_consensus \
    --table=public.economic_event_release_actual \
    | psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
        --username="$ADMIN_USER" --dbname="$DB_NAME" >/dev/null

run_preparer() {
    local output="$1"
    PGHOST="$DB_HOST" PGUSER="$ADMIN_USER" \
        python3 "$ROOT/EconomicCalendar/prepare_bls_production_import.py" \
        --db "$DB_NAME" --output-directory "$output"
}

run_preparer "$FIRST" >/dev/null
run_preparer "$REPEAT" >/dev/null
diff -ru "$FIRST" "$REPEAT"

test "$(wc -l < "$FIRST/bls-initial-actual-manifest.jsonl" | tr -d ' ')" = "786"
test "$(shasum -a 256 "$FIRST/bls-initial-actual-manifest.jsonl" | awk '{print $1}')" = \
    "cd4a124979cde17c65d3550d8b28293650aada458ae10f4a2fa288f547583420"
CERTIFIED_SQL="$ROOT/AuditEvidence/AuthoritativeEconomicCalendar/Phase17/2026-08-31/bls-initial-actual-import.sql"
test "$(shasum -a 256 "$CERTIFIED_SQL" | awk '{print $1}')" = \
    "4622f334bb2fd196287e768e1863745a70ed5fb4e09274cf594fdce85a7cc1a9"

source_bls_count="$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc \
    "SELECT count(*) FROM economic_event_release_actual WHERE source_agency='BLS';")"
case "$source_bls_count" in
    0)
        source_state="pre_import"
        test "$(shasum -a 256 "$FIRST/bls-initial-actual-import.sql" | awk '{print $1}')" = \
            "4622f334bb2fd196287e768e1863745a70ed5fb4e09274cf594fdce85a7cc1a9"
        test "$(jq -r '.payload.expected_insert_count' "$FIRST/bls-coverage-audit.json")" = "786"
        test "$(jq -r '.payload.duplicate_identical_count' "$FIRST/bls-coverage-audit.json")" = "0"
        ;;
    786)
        source_state="post_import"
        test "$(cat "$FIRST/bls-initial-actual-import.sql")" = $'BEGIN;\nCOMMIT;'
        test "$(jq -r '.payload.expected_insert_count' "$FIRST/bls-coverage-audit.json")" = "0"
        test "$(jq -r '.payload.duplicate_identical_count' "$FIRST/bls-coverage-audit.json")" = "786"
        ;;
    *)
        printf 'unexpected production BLS row count: %s\n' "$source_bls_count" >&2
        exit 1
        ;;
esac
test "$(jq -r '.payload.conflict_count' "$FIRST/bls-coverage-audit.json")" = "0"
if grep -Eiq '\b(UPDATE|DELETE|UPSERT)\b|ON[[:space:]]+CONFLICT' \
    "$FIRST/bls-initial-actual-import.sql"; then
    echo "BLS import plan is not INSERT-only" >&2
    exit 1
fi

psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -f "$FIRST/bls-initial-actual-import.sql" >/dev/null

test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc 'SELECT count(*) FROM economic_event_release_actual;')" = "1230"
test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc "SELECT count(*) FROM economic_event_release_actual WHERE source_agency='BLS';")" = "786"
test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc "SELECT count(*) FROM economic_event_release_actual WHERE source_agency='BLS' AND (publication_state <> 'initial' OR revision_sequence <> 0);")" = "0"
test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc "SELECT count(*) FROM economic_event_feature_release_actual WHERE source_agency='BLS';")" = "786"
test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc "SELECT count(*) FROM economic_event_release_actual era JOIN economic_event ee USING(economic_event_id) WHERE era.source_agency='BLS' AND era.available_at < ee.event_timestamp_utc;")" = "0"
test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc "SELECT count(*) - count(DISTINCT source_observation_id) FROM economic_event_release_actual WHERE source_agency='BLS';")" = "0"

run_preparer "$AFTER" >/dev/null
cmp "$FIRST/bls-initial-actual-manifest.jsonl" \
    "$AFTER/bls-initial-actual-manifest.jsonl"
test "$(cat "$AFTER/bls-initial-actual-import.sql")" = $'BEGIN;\nCOMMIT;'
test "$(jq -r '.payload.duplicate_identical_count' "$AFTER/bls-coverage-audit.json")" = "786"
test "$(jq -r '.payload.conflict_count' "$AFTER/bls-coverage-audit.json")" = "0"

if psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -c \
    "UPDATE economic_event_release_actual SET actual_raw='changed' WHERE source_agency='BLS';" \
    >/dev/null 2>&1; then
    echo "immutable BLS actual unexpectedly accepted UPDATE" >&2
    exit 1
fi
if psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -c \
    "DELETE FROM economic_event_release_actual WHERE source_agency='BLS';" \
    >/dev/null 2>&1; then
    echo "immutable BLS actual unexpectedly accepted DELETE" >&2
    exit 1
fi

cleanup
trap - EXIT
printf 'PHASE17_BYTE_IDENTICAL_REGENERATION=PASS\n'
printf 'PHASE17_DISPOSABLE_IMPORT=PASS\n'
printf 'PHASE17_EXACT_DUPLICATE_IDEMPOTENCY=PASS\n'
printf 'PHASE17_FEATURE_VIEW_INITIAL_REVISION_ZERO_ONLY=PASS\n'
printf 'PHASE17_CAUSAL_AVAILABILITY=PASS\n'
printf 'PHASE17_APPEND_ONLY_GUARDS=PASS\n'
printf 'PHASE17_SOURCE_STATE=%s\n' "$source_state"
printf 'PHASE17_DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
