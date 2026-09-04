#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_NAME="ea_economic_event_actual_provenance_${$}"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
ADMIN_USER="${LSTM_DB_ADMIN_USER:-${USER:-vjp}}"

case "$DB_NAME" in
    ea_economic_event_actual_provenance_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$ADMIN_USER" \
        "$DB_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

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

# Seed legacy evidence before migration 090 so its conservative backfill is
# tested as part of the real forward migration.
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
    --username="$ADMIN_USER" --dbname="$DB_NAME" <<'SQL' >/dev/null
INSERT INTO economic_event (
    currency,event_family,event_timestamp_utc,source_agency,source_event_id,
    source_url,reference_period,event_importance,historical_time_confidence,
    source_release_date,source_release_time,source_timezone
) VALUES
('USD','PCE','2024-01-01 13:30:00+00','BEA','bea:test:revision',
 'https://www.bea.gov/test/revision','December 2023',3,'exact',
 '2024-01-01','08:30:00','America/New_York'),
('USD','CPI','2024-02-01 13:30:00+00','BLS','bls:test:alternate',
 'https://www.bls.gov/test/alternate','January 2024',3,'exact',
 '2024-02-01','08:30:00','America/New_York'),
('USD','CPI','2015-03-01 13:30:00+00','BLS','bls:test:late',
 'https://www.bls.gov/test/late','February 2015',3,'exact',
 '2015-03-01','08:30:00','America/New_York'),
('USD','CPI','2024-04-01 13:30:00+00','BLS','bls:test:ambiguous',
 'https://www.bls.gov/test/ambiguous','March 2024',3,'exact',
 '2024-04-01','08:30:00','America/New_York'),
('USD','CPI','2024-05-01 13:30:00+00','BLS','bls:test:boundary',
 'https://www.bls.gov/test/boundary','April 2024',3,'exact',
 '2024-05-01','08:30:00','America/New_York'),
('USD','FOMC','2024-06-01 18:00:00+00','FEDERAL_RESERVE','federal_reserve:test:none',
 'https://www.federalreserve.gov/test/none',NULL,3,'exact',
 '2024-06-01','14:00:00','America/New_York');

INSERT INTO economic_event_consensus (
    economic_event_id,consensus_source,source_report_id,source_event_id,
    source_observation_id,source_event_name,source_period,source_priority,
    source_timestamp_epoch,source_date,source_release_date,
    source_artifact_path,candidate_classification,match_rule,
    semantic_contract,provider_provenance,forecast_raw,
    forecast_parse_status,forecast_value_kind,forecast_value_low,
    forecast_canonical_value_low,forecast_unit,forecast_scale,
    previous_parse_status,actual_raw,actual_parse_status,actual_value_kind,
    actual_value_low,actual_canonical_value_low,actual_unit,actual_scale,
    imported_at
)
SELECT economic_event_id,'OANDA',1,9001,'oanda:event:9001',event_family,
       reference_period,3,1706794200,'2024-02-01 13:30:00',
       '2024-02-01','fixture/oanda.csv','oanda_populated_initial',
       'fixture_match','oanda_economic_consensus_candidate_v1',
       '{"provider":"OANDA"}','0.2%','parsed','scalar',0.2,0.2,
       'percent',1,'missing','0.4%','parsed','scalar',0.4,0.4,
       'percent',1,'2026-01-01 00:00:00+00'
FROM economic_event WHERE source_event_id='bls:test:alternate';

INSERT INTO economic_event_release_actual (
    economic_event_id,source_agency,source_observation_id,publication_state,
    revision_sequence,available_at,retrieved_at,source_url,
    source_artifact_path,source_artifact_sha256,semantic_contract,
    source_provenance,actual_raw,actual_value_kind,actual_value_low,
    actual_canonical_value_low,actual_unit,actual_scale,actual_qualifier,
    imported_at
)
SELECT economic_event_id,'BEA','bea:test:revision:initial','initial',0,
       event_timestamp_utc,'2026-01-01 00:00:00+00',source_url,
       'fixture/bea-initial.html',repeat('a',64),
       'bea_current_dollar_pce_mom_percent_v1','{"provider":"BEA"}',
       '0.5%','scalar',0.5,0.5,'percent',1,'m/m',
       '2026-01-02 00:00:00+00'
FROM economic_event WHERE source_event_id='bea:test:revision';

INSERT INTO economic_event_release_actual (
    economic_event_id,source_agency,source_observation_id,publication_state,
    revision_sequence,available_at,retrieved_at,source_url,
    source_artifact_path,source_artifact_sha256,semantic_contract,
    source_provenance,actual_raw,actual_value_kind,actual_value_low,
    actual_canonical_value_low,actual_unit,actual_scale,actual_qualifier,
    imported_at
)
SELECT economic_event_id,'BEA','bea:test:revision:revision:1','revision',1,
       event_timestamp_utc + interval '1 day','2026-01-01 00:00:00+00',
       source_url,'fixture/bea-revision.html',repeat('b',64),
       'bea_current_dollar_pce_mom_percent_v1','{"provider":"BEA"}',
       '0.7%','scalar',0.7,0.7,'percent',1,'m/m',
       '2026-01-02 00:00:00+00'
FROM economic_event WHERE source_event_id='bea:test:revision';
SQL

psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
    --username="$ADMIN_USER" --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/090_economic_event_actual_observation_provenance.sql" \
    >/dev/null

psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
    --username="$ADMIN_USER" --dbname="$DB_NAME" <<'SQL' >/dev/null
-- Authoritative initial plus an alternate source for the same event.
INSERT INTO economic_event_actual_observation (
    economic_event_id,source_name,source_role,source_native_event_id,
    source_observation_id,evidence_key,observation_kind,revision_sequence,
    source_publication_at,source_publication_time_status,observed_at,
    ingested_at,availability_proof,source_url,source_artifact_path,
    source_artifact_sha256,semantic_contract,source_provenance,actual_raw,
    actual_value_kind,actual_value_low,actual_canonical_value_low,actual_unit,
    actual_scale,actual_qualifier
)
SELECT economic_event_id,'BLS','authoritative',source_event_id,
       'bls:test:alternate:initial','fixture:alternate:official','initial',0,
       event_timestamp_utc,'exact',timestamptz '2026-01-01 00:00:00+00',
       timestamptz '2026-01-02 00:00:00+00','source_publication',source_url,
       'fixture/bls-official.html',repeat('c',64),'fixture_percent_v1',
       '{"provider":"BLS"}','0.3%','scalar',0.3,0.3,'percent',1,NULL
FROM economic_event WHERE source_event_id='bls:test:alternate';

-- A late 2026 historical backfill has a conservative observation boundary,
-- but no source publication or initial-release proof.
INSERT INTO economic_event_actual_observation (
    economic_event_id,source_name,source_role,source_native_event_id,
    source_observation_id,evidence_key,observation_kind,revision_sequence,
    source_publication_at,source_publication_time_status,observed_at,
    ingested_at,availability_proof,source_url,source_artifact_path,
    source_artifact_sha256,semantic_contract,source_provenance,actual_raw,
    actual_value_kind,actual_value_low,actual_canonical_value_low,actual_unit,
    actual_scale,actual_qualifier
)
SELECT economic_event_id,'BLS','authoritative',source_event_id,
       'bls:test:late:backfill','fixture:late:backfill','unclassified',NULL,
       NULL,'unavailable','2026-01-01 00:00:00+00',
       '2026-01-02 00:00:00+00','system_observation',source_url,
       'fixture/bls-late.html',repeat('d',64),'fixture_percent_v1',
       '{"provider":"BLS","publication":"unknown"}','0.8%',
       'scalar',0.8,0.8,'percent',1,NULL
FROM economic_event WHERE source_event_id='bls:test:late';

-- An exact initial plus a conflicting possible-first claim with no source
-- publication time has unprovable ordering and must fail closed.
INSERT INTO economic_event_actual_observation (
    economic_event_id,source_name,source_role,source_native_event_id,
    source_observation_id,evidence_key,observation_kind,revision_sequence,
    source_publication_at,source_publication_time_status,observed_at,
    ingested_at,availability_proof,source_url,source_artifact_path,
    source_artifact_sha256,semantic_contract,source_provenance,actual_raw,
    actual_value_kind,actual_value_low,actual_canonical_value_low,actual_unit,
    actual_scale,actual_qualifier
)
SELECT economic_event_id,'BLS','authoritative',source_event_id,
       'bls:test:ambiguous:a','fixture:ambiguous:a','initial',0,
       event_timestamp_utc,'exact','2026-01-01 00:00:00+00',
       '2026-01-02 00:00:00+00','source_publication',source_url,
       'fixture/bls-ambiguous-a.html',repeat('e',64),'fixture_percent_v1',
       '{"provider":"BLS","claim":"a"}'::jsonb,'0.1%',
       'scalar',0.1,0.1,'percent',1,NULL
FROM economic_event WHERE source_event_id='bls:test:ambiguous'
UNION ALL
SELECT economic_event_id,'BLS','authoritative',source_event_id,
       'bls:test:ambiguous:b','fixture:ambiguous:b','initial',0,
       event_timestamp_utc,'exact',timestamptz '2026-01-01 00:00:00+00',
       timestamptz '2026-01-02 00:00:00+00','source_publication',source_url,
       'fixture/bls-ambiguous-b.html',repeat('f',64),'fixture_percent_v1',
       '{"provider":"BLS","claim":"b"}'::jsonb,'0.2%',
       'scalar',0.2,0.2,'percent',1,NULL
FROM economic_event WHERE source_event_id='bls:test:ambiguous';

-- Availability is five minutes after the event, for exact boundary checks.
INSERT INTO economic_event_actual_observation (
    economic_event_id,source_name,source_role,source_native_event_id,
    source_observation_id,evidence_key,observation_kind,revision_sequence,
    source_publication_at,source_publication_time_status,observed_at,
    ingested_at,availability_proof,source_url,source_artifact_path,
    source_artifact_sha256,semantic_contract,source_provenance,actual_raw,
    actual_value_kind,actual_value_low,actual_canonical_value_low,actual_unit,
    actual_scale,actual_qualifier
)
SELECT economic_event_id,'BLS','authoritative',source_event_id,
       'bls:test:boundary:initial','fixture:boundary','initial',0,
       event_timestamp_utc + interval '5 minutes','exact',
       '2026-01-01 00:00:00+00','2026-01-02 00:00:00+00',
       'source_publication',source_url,'fixture/bls-boundary.html',
       repeat('1',64),'fixture_percent_v1','{"provider":"BLS"}',
       '0.6%','scalar',0.6,0.6,'percent',1,NULL
FROM economic_event WHERE source_event_id='bls:test:boundary';
SQL

psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
    --username="$ADMIN_USER" --dbname="$DB_NAME" <<'SQL' >/dev/null
DO $$
DECLARE
    revision_event bigint;
    alternate_event bigint;
    late_event bigint;
    ambiguous_event bigint;
    boundary_event bigint;
BEGIN
    SELECT economic_event_id INTO revision_event FROM economic_event
        WHERE source_event_id='bea:test:revision';
    SELECT economic_event_id INTO alternate_event FROM economic_event
        WHERE source_event_id='bls:test:alternate';
    SELECT economic_event_id INTO late_event FROM economic_event
        WHERE source_event_id='bls:test:late';
    SELECT economic_event_id INTO ambiguous_event FROM economic_event
        WHERE source_event_id='bls:test:ambiguous';
    SELECT economic_event_id INTO boundary_event FROM economic_event
        WHERE source_event_id='bls:test:boundary';

    IF (SELECT count(*) FROM economic_event_actual_observation
        WHERE economic_event_id=revision_event) <> 2 THEN
        RAISE EXCEPTION 'revision observations not retained';
    END IF;
    IF (SELECT first_release_value_low FROM economic_event_first_release_actual
        WHERE economic_event_id=revision_event) <> 0.5 THEN
        RAISE EXCEPTION 'first release changed after revision';
    END IF;
    IF NOT (SELECT canonical_differs_from_first_release
            FROM economic_event_first_release_actual
            WHERE economic_event_id=revision_event) THEN
        RAISE EXCEPTION 'canonical revision difference missing';
    END IF;
    IF (SELECT canonical_value_low FROM economic_event_first_release_actual
        WHERE economic_event_id=revision_event) <> 0.7 THEN
        RAISE EXCEPTION 'canonical revision not selected';
    END IF;
    IF (SELECT actual_canonical_value_low
        FROM economic_event_feature_release_actual
        WHERE economic_event_id=revision_event) <> 0.5 THEN
        RAISE EXCEPTION 'legacy feature view behavior changed';
    END IF;

    IF (SELECT count(*) FROM economic_event_actual_observation
        WHERE economic_event_id=alternate_event) <> 2 THEN
        RAISE EXCEPTION 'alternate source evidence collapsed';
    END IF;
    IF (SELECT first_release_source FROM economic_event_first_release_actual
        WHERE economic_event_id=alternate_event) <> 'BLS' THEN
        RAISE EXCEPTION 'authoritative source precedence failed';
    END IF;

    IF (SELECT provenance_state FROM economic_event_first_release_actual
        WHERE economic_event_id=late_event) <> 'provenance_unavailable' THEN
        RAISE EXCEPTION 'late backfill manufactured first release';
    END IF;
    IF EXISTS (SELECT 1 FROM economic_event_first_release_actual_at(
        '2026-01-02 00:00:00+00') WHERE economic_event_id=late_event) THEN
        RAISE EXCEPTION 'unproved late backfill entered PIT API';
    END IF;

    IF (SELECT provenance_state FROM economic_event_first_release_actual
        WHERE economic_event_id=ambiguous_event) <> 'ambiguous' THEN
        RAISE EXCEPTION 'conflicting earliest evidence did not fail closed';
    END IF;
    IF (SELECT first_release_value_low IS NOT NULL
        FROM economic_event_first_release_actual
        WHERE economic_event_id=ambiguous_event) THEN
        RAISE EXCEPTION 'ambiguous first release exposed a value';
    END IF;

    IF EXISTS (SELECT 1 FROM economic_event_first_release_actual_at(
        '2024-05-01 13:34:59.999999+00') WHERE economic_event_id=boundary_event) THEN
        RAISE EXCEPTION 'actual leaked before availability';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM economic_event_first_release_actual_at(
        '2024-05-01 13:35:00+00') WHERE economic_event_id=boundary_event) THEN
        RAISE EXCEPTION 'actual absent at inclusive availability boundary';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM economic_event_first_release_actual_at(
        '2024-05-01 13:35:00.000001+00') WHERE economic_event_id=boundary_event) THEN
        RAISE EXCEPTION 'actual absent after availability boundary';
    END IF;
END;
$$;
SQL

# A generated proven_available_at value is not available to a BEFORE trigger.
# The validation trigger must derive the system-observation fallback itself and
# reject evidence whose asserted availability predates the logical event.
if psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
    --username="$ADMIN_USER" --dbname="$DB_NAME" >/dev/null 2>&1 <<'SQL'
INSERT INTO economic_event_actual_observation (
    economic_event_id,source_name,source_role,source_native_event_id,
    source_observation_id,evidence_key,observation_kind,revision_sequence,
    source_publication_at,source_publication_time_status,observed_at,
    ingested_at,availability_proof,source_url,source_artifact_path,
    semantic_contract,source_provenance,actual_raw,actual_value_kind,
    actual_value_low,actual_canonical_value_low,actual_unit,actual_scale
)
SELECT economic_event_id,'FEDERAL_RESERVE','authoritative',source_event_id,
       'federal-reserve:test:predates-event','fixture:predates-event',
       'unclassified',NULL,NULL,'unavailable',
       event_timestamp_utc - interval '1 second',
       timestamptz '2026-01-02 00:00:00+00','system_observation',source_url,
       'fixture/predates-event.html','fixture_count_v1',
       '{"provider":"FEDERAL_RESERVE"}','1','scalar',1,1,'count',1
FROM economic_event
WHERE source_event_id='federal_reserve:test:none';
SQL
then
    printf 'pre-release observation unexpectedly succeeded\n' >&2
    exit 1
fi

# Persisted observations are immutable.
if psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
    --username="$ADMIN_USER" --dbname="$DB_NAME" \
    -c "UPDATE economic_event_actual_observation SET actual_raw='changed'" \
    >/dev/null 2>&1; then
    printf 'immutable observation update unexpectedly succeeded\n' >&2
    exit 1
fi

EVENT_ID="$(psql -X --host="$DB_HOST" --username="$ADMIN_USER" \
    --dbname="$DB_NAME" -tAc \
    "SELECT economic_event_id FROM economic_event WHERE source_event_id='bea:test:revision';")"
AUDIT="$(PGHOST="$DB_HOST" PGUSER="$ADMIN_USER" python3 \
    "$ROOT/EconomicCalendar/audit_economic_event_actual_provenance.py" \
    --db "$DB_NAME" --event-id "$EVENT_ID" \
    --cutoff 2024-01-02T13:30:00Z)"
python3 -c 'import json,sys; d=json.load(sys.stdin); assert len(d["observations"]) == 2; assert d["assessment"]["provenance_state"] == "proven_first_release"; assert len(d["pit_visible_first_release"]) == 1' <<<"$AUDIT"

SUMMARY="$(PGHOST="$DB_HOST" PGUSER="$ADMIN_USER" python3 \
    "$ROOT/EconomicCalendar/audit_economic_event_actual_provenance.py" \
    --db "$DB_NAME")"
python3 -c 'import json,sys; d=json.load(sys.stdin); s=d["summary"]; assert s["proven_first_release"] == "3"; assert s["ambiguous"] == "1"; assert s["provenance_unavailable"] == "1"; assert s["events_with_multiple_actual_observations"] == "3"; assert s["canonical_differs_from_first_release"] == "1"' <<<"$SUMMARY"

cleanup
trap - EXIT
printf 'ECONOMIC_EVENT_ACTUAL_PROVENANCE=PASS\n'
printf 'DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
