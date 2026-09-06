#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
DB_ADMIN_USER="${LSTM_DB_ADMIN_USER:-${USER:-vjp}}"
DB_NAME="ea_dol_eta_weekly_claims_phase7_${$}"

case "$DB_NAME" in
    ea_dol_eta_weekly_claims_phase7_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_ADMIN_USER" \
        "$DB_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "$ROOT"
python3 -m unittest -v Tests.DolEtaWeeklyClaimsActualTests

if psql -X --host="$DB_HOST" --username="$DB_USER" --dbname=postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';" | grep -q 1; then
    printf 'refusing to reuse existing database: %s\n' "$DB_NAME" >&2
    exit 1
fi

createdb --host="$DB_HOST" --username="$DB_ADMIN_USER" --owner="$DB_USER" --template=template0 "$DB_NAME"
for migration in \
    072_economic_event.sql \
    081_economic_event_consensus.sql \
    082_economic_event_consensus_provider_provenance.sql \
    088_economic_event_release_actual_provenance.sql \
    090_economic_event_actual_observation_provenance.sql; do
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
) VALUES
(
    19, 'USD', 'WEEKLY_CLAIMS', timestamptz '2010-07-15 12:30:00+00',
    'DOL_ETA', 'dol_eta:usdl-10-950-nat',
    'https://oui.doleta.gov/press/2010/071510.asp',
    'week ending 2010-07-10', 3, 'exact', DATE '2010-07-15',
    TIME '08:30:00', 'America/New_York'
),
(
    20, 'USD', 'WEEKLY_CLAIMS', timestamptz '2010-07-22 12:30:00+00',
    'DOL_ETA', 'dol_eta:usdl-10-990-nat',
    'https://oui.doleta.gov/press/2010/072210.asp',
    'week ending 2010-07-17', 3, 'exact', DATE '2010-07-22',
    TIME '08:30:00', 'America/New_York'
);
SQL

PGHOST="$DB_HOST" PGUSER="$DB_USER" PHASE7_DB="$DB_NAME" \
PYTHONPATH="$ROOT/EconomicCalendar:$ROOT/Scripts" python3 - <<'PY'
import dataclasses
import hashlib
import os
import pathlib

from dol_eta_weekly_claims_actual import DolEtaArtifact, extract_dol_eta_candidates
from import_economic_event_release_actual import (
    build_insert_sql, execute_disposable_import, load_database,
)
from release_actual_ingestion import match_candidates

root = pathlib.Path.cwd()
path = root / "Tests/fixtures/economic_calendar/dol_eta/2010-07-22.html"
digest = hashlib.sha256(path.read_bytes()).hexdigest()
artifact = DolEtaArtifact(
    release_date="2010-07-22",
    source_url="https://oui.doleta.gov/press/2010/072210.asp",
    source_artifact_identity="dol_eta:press:2010:072210.asp",
    source_path=path,
    source_repository_path=path.relative_to(root).as_posix(),
    source_sha256=digest,
    parser_path=path,
    parser_repository_path=path.relative_to(root).as_posix(),
    parser_sha256=digest,
    extractor="none",
    retrieved_at="2026-09-05T12:00:00Z",
)
database = os.environ["PHASE7_DB"]
events, _, existing = load_database(database)
rows, rejected = extract_dol_eta_candidates(artifact, events)
assert not rejected
initial = next(row for row in rows if row.publication_state == "initial")

# Dry-run/preflight is read-only and does not alter either immutable table.
decision = match_candidates([initial], events, existing)
assert decision[0].decision == "matched"
assert len(load_database(database)[2]) == 0

# A distinct later official artifact is a separate immutable revision for the
# same logical event. It must not replace the first-release observation.
revision = dataclasses.replace(
    initial,
    source_observation_id="dol_eta:weekly_claims:2010-07-29:2010-07-17:revision",
    publication_state="revision",
    revision_sequence=1,
    available_at="2010-07-29T12:30:00Z",
    source_url="https://oui.doleta.gov/press/2010/072910.asp",
    source_artifact_sha256="1" * 64,
    source_provenance={**initial.source_provenance, "artifact_sha256": "1" * 64},
    actual_raw="previous week's revised figure of 465,000",
    actual_value_low="465000",
    actual_canonical_value_low="465000",
)
decisions = match_candidates([initial, revision], events, existing)
assert [row.decision for row in decisions] == ["matched", "matched"]
execute_disposable_import(database, build_insert_sql(decisions))

# An exact repeat emits no SQL writes.
events, _, existing = load_database(database)
repeat = match_candidates([initial, revision], events, existing)
assert [row.decision for row in repeat] == ["duplicate_identical", "duplicate_identical"]
execute_disposable_import(database, build_insert_sql(repeat))
PY

test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc 'SELECT count(*) FROM economic_event_actual_observation;')" = "2"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc "SELECT provenance_state||':'||selection_reason||':'||first_release_value_low::text FROM economic_event_first_release_actual WHERE economic_event_id=20;")" = \
    "proven_first_release:unique_authoritative_initial_at_earliest_source_publication:464000"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc "SELECT count(*) FROM economic_event_first_release_actual_at('2010-07-22 12:29:59.999999+00') WHERE economic_event_id=20;")" = "0"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc "SELECT first_release_actual_value_low::text FROM economic_event_first_release_actual_at('2010-07-22 12:30:00+00') WHERE economic_event_id=20;")" = "464000"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc "SELECT first_release_actual_value_low::text FROM economic_event_first_release_actual_at('2010-07-30 00:00:00+00') WHERE economic_event_id=20;")" = "464000"
test "$(psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
    -tAc "SELECT canonical_value_low::text||':'||canonical_differs_from_first_release::text FROM economic_event_first_release_actual WHERE economic_event_id=20;")" = "465000:true"

cleanup
trap - EXIT
printf 'DOL_ETA_WEEKLY_CLAIMS_ACTUAL_UNIT=PASS\n'
printf 'DOL_ETA_WEEKLY_CLAIMS_MIGRATION090=PASS\n'
printf 'DOL_ETA_WEEKLY_CLAIMS_DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
