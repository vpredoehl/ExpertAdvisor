#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
DB_NAME="ea_release_actual_phase9_${$}"

case "$DB_NAME" in
    ea_release_actual_phase9_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_USER" \
        "$DB_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python3 -m unittest -v Tests/EconomicEventReleaseActualImporterTests.py

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
    --dbname="$DB_NAME" <<'SQL' >/dev/null
INSERT INTO economic_event (
    economic_event_id, currency, event_family, event_timestamp_utc,
    source_agency, source_event_id, source_url, reference_period,
    event_importance, historical_time_confidence,
    source_release_date, source_release_time, source_timezone
) VALUES (
    1, 'USD', 'RETAIL_SALES', timestamptz '2010-02-12 13:30:00+00',
    'CENSUS', 'census:retail-sales-2010-01',
    'https://www.census.gov/economic-indicators/fixture.pdf', 'January 2010',
    3, 'exact', DATE '2010-02-12', TIME '08:30:00', 'America/New_York'
);
SQL

PGHOST="$DB_HOST" PGUSER="$DB_USER" PHASE9_DB="$DB_NAME" \
PYTHONPATH="$ROOT/EconomicCalendar" python3 - <<'PY'
import os
import pathlib

from import_economic_event_release_actual import (
    build_insert_sql,
    execute_disposable_import,
    load_database,
)
from release_actual_ingestion import (
    CensusArtifact,
    extract_census_candidates,
    match_candidates,
    sha256_file,
)

root = pathlib.Path.cwd()
path = root / "Tests/fixtures/economic_calendar/release_actual/census_retail_initial_revision.txt"
artifact = CensusArtifact(
    event_family="RETAIL_SALES",
    reference_period="January 2010",
    source_url="https://www.census.gov/economic-indicators/fixture.pdf",
    path=path,
    repository_path=path.relative_to(root).as_posix(),
    source_event_id="census:retail-sales-2010-01",
    available_at="2010-02-12T13:30:00Z",
    retrieved_at="2026-08-25T04:55:48Z",
    archive_commit="8a740a78a8c8ff10c8afd04c2d1c6eb6dfbbbb86",
    sha256=sha256_file(path),
)
candidates, rejected = extract_census_candidates(artifact, {
    ("RETAIL_SALES", "January 2010"): "census:retail-sales-2010-01",
})
assert not rejected
initial = [row for row in candidates if row.publication_state == "initial"]
database = os.environ["PHASE9_DB"]
events, _, existing = load_database(database)
first = match_candidates(initial, events, existing)
assert [row.decision for row in first] == ["matched"]
execute_disposable_import(database, build_insert_sql(first))

events, _, existing = load_database(database)
second = match_candidates(initial, events, existing)
assert [row.decision for row in second] == ["duplicate_identical"]
execute_disposable_import(database, build_insert_sql(second))
PY

test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event_release_actual;')" = "1"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event_feature_release_actual;')" = "1"

if psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" -c \
    "UPDATE economic_event_release_actual SET actual_raw='changed';" >/dev/null 2>&1; then
    echo "immutable actual unexpectedly accepted UPDATE" >&2
    exit 1
fi

cleanup
trap - EXIT
printf 'PHASE9_UNIT_TESTS=PASS\n'
printf 'PHASE9_DISPOSABLE_IMPORT=PASS\n'
printf 'PHASE9_DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
