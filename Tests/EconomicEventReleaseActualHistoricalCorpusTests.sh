#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
DB_NAME="ea_release_actual_corpus_${$}"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ea-release-actual-corpus.XXXXXX")"
EVENT_SEED="$TEMP_DIR/economic_event.csv"
DRY_RUN="$TEMP_DIR/dry-run.jsonl"
COVERAGE="$TEMP_DIR/coverage.json"

case "$DB_NAME" in
    ea_release_actual_corpus_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_USER" \
        "$DB_NAME" >/dev/null 2>&1 || true
    rm -f "$EVENT_SEED" "$DRY_RUN" "$COVERAGE"
    rmdir "$TEMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

python3 - "$ROOT/EconomicCalendar/raw/census/census_import_prepared.csv" \
    "$EVENT_SEED" <<'PY'
import csv
import sys

source_path, output_path = sys.argv[1:]
columns = [
    "economic_event_id", "currency", "event_family", "event_timestamp_utc",
    "source_agency", "source_event_id", "source_url", "reference_period",
    "event_importance", "historical_time_confidence", "source_release_date",
    "source_release_time", "source_timezone",
]
with open(source_path, newline="", encoding="utf-8-sig") as source:
    rows = list(csv.DictReader(source))
with open(output_path, "w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for event_id, row in enumerate(rows, 1):
        result = {column: row[column] for column in columns if column != "economic_event_id"}
        result["economic_event_id"] = event_id
        writer.writerow(result)
PY

if psql -X --host="$DB_HOST" --username="$DB_USER" --dbname=postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';" | grep -q 1; then
    printf 'refusing to reuse existing database: %s\n' "$DB_NAME" >&2
    exit 1
fi

createdb --host="$DB_HOST" --username="$DB_USER" --template=template0 "$DB_NAME"
for migration in \
    072_economic_event.sql \
    081_economic_event_consensus.sql \
    082_economic_event_consensus_provider_provenance.sql \
    088_economic_event_release_actual_provenance.sql; do
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
        --dbname="$DB_NAME" -f "$ROOT/Database/migrations/$migration" >/dev/null
done

psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" -c "\copy economic_event (
        economic_event_id, currency, event_family, event_timestamp_utc,
        source_agency, source_event_id, source_url, reference_period,
        event_importance, historical_time_confidence, source_release_date,
        source_release_time, source_timezone
    ) FROM '$EVENT_SEED' CSV HEADER" >/dev/null

run_importer() {
    PGHOST="$DB_HOST" PGUSER="$DB_USER" \
        python3 "$ROOT/EconomicCalendar/import_economic_event_release_actual.py" \
        --db "$DB_NAME" --dry-run-output "$DRY_RUN" \
        --coverage-output "$COVERAGE" "$@"
}

FIRST_DRY_RUN="$(run_importer)"
grep -Fq 'Decisions: {"invalid_semantics": 14, "matched": 676, "unsupported": 1}' \
    <<< "$FIRST_DRY_RUN"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event_release_actual;')" = "0"

FIRST_IMPORT="$(run_importer --commit --allow-disposable-write)"
grep -Fq 'RESULT: COMMITTED TO EXPLICIT DISPOSABLE DATABASE' <<< "$FIRST_IMPORT"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event_release_actual;')" = "676"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event_feature_release_actual;')" = "384"

SECOND_IMPORT="$(run_importer --commit --allow-disposable-write)"
grep -Fq 'Decisions: {"duplicate_identical": 676, "invalid_semantics": 14, "unsupported": 1}' \
    <<< "$SECOND_IMPORT"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event_release_actual;')" = "676"

cleanup
trap - EXIT
printf 'PHASE9_HISTORICAL_CORPUS_DRY_RUN=PASS\n'
printf 'PHASE9_HISTORICAL_CORPUS_IMPORT=PASS\n'
printf 'PHASE9_HISTORICAL_CORPUS_IDEMPOTENCY=PASS\n'
printf 'PHASE9_HISTORICAL_CORPUS_DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
