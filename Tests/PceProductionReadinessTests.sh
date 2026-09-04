#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
ADMIN_USER="${LSTM_DB_ADMIN_USER:-${USER}}"
RUNTIME_USER="${PHASE11_RUNTIME_DB_USER:-pqxx}"
DB_NAME="ea_pce_phase11_${$}"

case "$DB_NAME" in
    ea_pce_phase11_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$ADMIN_USER" \
        "$DB_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python3 -m unittest -v Tests/PceProductionReadinessTests.py

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
    082_economic_event_consensus_provider_provenance.sql; do
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
        --username="$ADMIN_USER" --dbname="$DB_NAME" \
        -f "$ROOT/Database/migrations/$migration" >/dev/null
done

test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc \
    "SELECT to_regclass('economic_event_release_actual') IS NULL;")" = "t"

psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/088_economic_event_release_actual_provenance.sql" \
    >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/090_economic_event_actual_observation_provenance.sql" \
    >/dev/null

object_state="$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -Atc \
    "SELECT (to_regclass('economic_event_release_actual') IS NOT NULL)::int ||
            ':' || (to_regclass('economic_event_feature_release_actual') IS NOT NULL)::int ||
            ':' || (to_regprocedure('validate_economic_event_release_actual()') IS NOT NULL)::int ||
            ':' || (to_regprocedure('reject_economic_event_release_actual_mutation()') IS NOT NULL)::int;")"
test "$object_state" = "1:1:1:1"

privileges="$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -Atc \
    "SELECT has_table_privilege('$RUNTIME_USER','economic_event_release_actual','SELECT')::int ||
            ':' || has_table_privilege('$RUNTIME_USER','economic_event_release_actual','INSERT')::int ||
            ':' || has_table_privilege('$RUNTIME_USER','economic_event_release_actual','UPDATE')::int ||
            ':' || has_table_privilege('$RUNTIME_USER','economic_event_release_actual','DELETE')::int;")"
test "$privileges" = "1:0:0:0"
test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc \
    'SELECT count(*) FROM economic_event_release_actual;')" = "0"

psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" <<'SQL' >/dev/null
INSERT INTO economic_event (
    economic_event_id, currency, event_family, event_timestamp_utc,
    source_agency, source_event_id, source_url, reference_period,
    event_importance, historical_time_confidence,
    source_release_date, source_release_time, source_timezone
) VALUES (
    20, 'USD', 'PCE', timestamptz '2010-02-01 13:30:00+00',
    'BEA', 'bea:pce:december-2009', 'https://www.bea.gov/news/fixture',
    'December 2009', 3, 'exact', DATE '2010-02-01',
    TIME '08:30:00', 'America/New_York'
);
INSERT INTO economic_event_consensus (
    economic_event_id, consensus_source, source_report_id, source_event_id,
    source_observation_id, source_event_name, source_period, source_priority,
    source_timestamp_epoch, source_date, source_release_date,
    source_artifact_path, candidate_classification, match_rule,
    semantic_contract, provider_provenance, forecast_raw,
    forecast_parse_status, forecast_value_kind, forecast_value_low,
    forecast_canonical_value_low, forecast_unit, forecast_scale,
    forecast_qualifier, previous_parse_status, actual_parse_status
) VALUES (
    20, 'OANDA', 1, 1, 'oanda:event:1', 'Personal Consumption Expenditure',
    'Dec', 1, 1265031000, timestamp '2010-02-01 08:30:00',
    DATE '2010-02-01', 'fixture/oanda.csv', 'oanda_populated_initial',
    'fixture_match', 'oanda_economic_consensus_candidate_v1',
    '{"provider":"OANDA"}', '0.2%', 'parsed', 'scalar', 0.2, 0.2,
    'percent', 1, 'm/m', 'missing', 'missing'
);
SQL

PGHOST="$DB_HOST" PGUSER="$ADMIN_USER" PHASE11_DB="$DB_NAME" \
PYTHONPATH="$ROOT/EconomicCalendar" python3 - <<'PY'
import os

from import_economic_event_release_actual import (
    build_insert_sql,
    execute_disposable_import,
    load_database,
)
from release_actual_ingestion import ReleaseActualCandidate, match_candidates

candidate = ReleaseActualCandidate(
    source_family="BEA",
    candidate_event_family="PCE",
    candidate_source_event_id="bea:pce:december-2009",
    candidate_reference_period="December 2009",
    source_agency="BEA",
    source_observation_id="bea:pce:2009-12:initial:fixture",
    publication_state="initial",
    revision_sequence=0,
    available_at="2010-02-01T13:30:00Z",
    retrieved_at="2026-08-25T04:55:48Z",
    source_url="https://www.bea.gov/news/fixture",
    source_artifact_path="fixture/bea-pce.html",
    source_artifact_sha256="a" * 64,
    semantic_contract="bea_current_dollar_pce_mom_percent_v1",
    source_provenance={"provider": "BEA"},
    actual_raw="increased 0.2 percent",
    actual_value_kind="scalar",
    actual_value_low="0.2",
    actual_value_high=None,
    actual_canonical_value_low="0.2",
    actual_canonical_value_high=None,
    actual_unit="percent",
    actual_scale="1",
    actual_qualifier="m/m",
)
database = os.environ["PHASE11_DB"]
events, _, existing = load_database(database)
first = match_candidates([candidate], events, existing)
assert [row.decision for row in first] == ["matched"]
sql = build_insert_sql(first)
assert "UPDATE" not in sql and "DELETE" not in sql and "ON CONFLICT" not in sql
execute_disposable_import(database, sql)

events, _, existing = load_database(database)
second = match_candidates([candidate], events, existing)
assert [row.decision for row in second] == ["duplicate_identical"]
assert build_insert_sql(second) == "BEGIN;\nCOMMIT;\n"
execute_disposable_import(database, build_insert_sql(second))
PY

test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc \
    'SELECT count(*) FROM economic_event_release_actual;')" = "1"
test "$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc \
    'SELECT count(*) FROM economic_event_actual_observation;')" = "1"
test "$(psql -X --host="$DB_HOST" --username="$RUNTIME_USER" \
    --dbname="$DB_NAME" -tAc \
    'SELECT count(*) FROM economic_event_feature_release_actual;')" = "1"

if psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -c \
    "UPDATE economic_event_release_actual SET actual_raw='changed';" \
    >/dev/null 2>&1; then
    echo "immutable actual unexpectedly accepted UPDATE" >&2
    exit 1
fi
if psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$RUNTIME_USER" \
    --dbname="$DB_NAME" -c \
    "INSERT INTO economic_event_release_actual (
        economic_event_id,source_agency,source_observation_id,
        publication_state,revision_sequence,available_at,retrieved_at,
        source_url,source_artifact_path,source_artifact_sha256,
        semantic_contract,source_provenance,actual_raw,actual_value_kind,
        actual_value_low,actual_canonical_value_low,actual_unit,actual_scale
     ) VALUES (20,'BEA','forbidden','initial',0,now(),now(),
        'https://www.bea.gov/forbidden','forbidden',repeat('b',64),
        'forbidden','{"provider":"BEA"}','0','scalar',0,0,'percent',1);" \
    >/dev/null 2>&1; then
    echo "runtime role unexpectedly has INSERT privilege" >&2
    exit 1
fi

cleanup
trap - EXIT
printf 'PHASE11_UNIT_TESTS=PASS\n'
printf 'PHASE11_MIGRATION_READINESS=PASS\n'
printf 'PHASE11_FIRST_IMPORT=PASS\n'
printf 'PHASE11_IDEMPOTENT_REPEAT=PASS\n'
printf 'PHASE11_RUNTIME_WRITE_GUARD=PASS\n'
printf 'PHASE11_DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
