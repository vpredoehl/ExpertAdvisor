#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/weekly_claims_historical_consensus"
CLI_BIN="$BUILD_DIR/EconomicEventConsensusImportCli"
FEATURE_BIN="$BUILD_DIR/WeeklyClaimsHistoricalConsensusPersistenceTests"
DB_NAME="ea_weekly_claims_consensus_${$}"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
DB_ADMIN_USER="${LSTM_DB_ADMIN_USER:-${USER:-vjp}}"
TEMP_DIR="$(mktemp -d)"
SOURCE_CSV="$ROOT/AuditEvidence/AuthoritativeEconomicCalendar/Phase8/2026-09-05/myfxbook-weekly-claims-consensus-import.csv"

case "$DB_NAME" in
    ea_weekly_claims_consensus_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_ADMIN_USER" \
        "$DB_NAME" >/dev/null 2>&1 || true
    rm -r "$TEMP_DIR"
}
trap cleanup EXIT

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a PQXX_CFLAGS <<< "$(pkg-config --cflags libpqxx)"
    read -r -a PQXX_LIBS <<< "$(pkg-config --libs libpqxx)"
else
    PQXX_PREFIX="$(brew --prefix libpqxx)"
    LIBPQ_PREFIX="$(brew --prefix libpq)"
    PQXX_CFLAGS=("-I${PQXX_PREFIX}/include" "-I${LIBPQ_PREFIX}/include")
    PQXX_LIBS=("-L${PQXX_PREFIX}/lib" "-L${LIBPQ_PREFIX}/lib" -lpqxx -lpq)
fi
mkdir -p "$BUILD_DIR"

clang++ -std=c++20 -Wall -Wextra -Werror \
    "${PQXX_CFLAGS[@]}" -I"$ROOT/Sources" \
    "$ROOT/Sources/EconomicEventConsensusImport.cpp" \
    "$ROOT/Sources/EconomicEventConsensusRepository.cpp" \
    "$ROOT/Tests/EconomicEventConsensusImportCliHarness.cpp" \
    "${PQXX_LIBS[@]}" -o "$CLI_BIN"
clang++ -std=c++20 -Wall -Wextra -Werror \
    "${PQXX_CFLAGS[@]}" -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "$ROOT/Sources/EconomicEventRepository.cpp" \
    "$ROOT/Sources/EconomicEventFeatures.cpp" \
    "$ROOT/Tests/WeeklyClaimsHistoricalConsensusPersistenceTests.cpp" \
    "${PQXX_LIBS[@]}" -o "$FEATURE_BIN"

createdb --host="$DB_HOST" --username="$DB_ADMIN_USER" \
    --owner="$DB_USER" --template=template0 "$DB_NAME"
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/072_economic_event.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" <<'SQL' >/dev/null
INSERT INTO economic_event (
    economic_event_id,currency,event_family,event_timestamp_utc,source_agency,
    source_event_id,source_url,reference_period,event_importance,
    historical_time_confidence,source_release_date,source_release_time,
    source_timezone
) VALUES
(9001,'USD','CPI','2025-01-01 13:30:00+00','BLS','bls:cpi:legacy',
 'https://www.bls.gov/cpi','December 2024',3,'exact','2025-01-01','08:30','America/New_York'),
(9002,'USD','JOLTS','2025-02-01 15:00:00+00','BLS','bls:jolts:legacy',
 'https://www.bls.gov/jlt','December 2024',3,'exact','2025-02-01','10:00','America/New_York'),
(3467,'USD','WEEKLY_CLAIMS','2025-06-05 12:30:00+00','DOL_ETA',
 'dol_eta:usdl-25-942-nat','https://oui.doleta.gov/press/2025/060525.pdf',
 'week ending 2025-05-31',3,'exact','2025-06-05','08:30','America/New_York'),
(3468,'USD','WEEKLY_CLAIMS','2025-06-12 12:30:00+00','DOL_ETA',
 'dol_eta:usdl-25-missing-nat','https://oui.doleta.gov/press/2025/061225.pdf',
 'week ending 2025-06-07',3,'exact','2025-06-12','08:30','America/New_York');
SQL
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/081_economic_event_consensus.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" <<'SQL' >/dev/null
INSERT INTO economic_event_consensus (
    economic_event_id,consensus_source,source_report_id,source_event_id,
    source_event_name,source_period,source_priority,source_timestamp_epoch,
    source_date,source_artifact_path,match_rule,semantic_contract,
    forecast_raw,forecast_parse_status,forecast_value_kind,forecast_value_low,
    forecast_canonical_value_low,forecast_unit,forecast_scale,
    forecast_qualifier,previous_parse_status,actual_parse_status
) VALUES (
    9001,'OANDA',699,111,'Consumer Price Index','December',3,1735738200,
    '2025-01-01 13:30:00','fixture/oanda.json','fixture',
    'oanda_economic_consensus_candidate_v1','0.2% m/m','parsed','scalar',
    0.2,0.2,'percent',1,'m/m','missing','missing'
);
SQL
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/082_economic_event_consensus_provider_provenance.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" <<'SQL' >/dev/null
INSERT INTO economic_event_consensus (
    economic_event_id,consensus_source,source_event_id,source_observation_id,
    source_event_name,source_release_date,source_artifact_path,
    source_artifact_sha256,candidate_classification,match_rule,
    semantic_contract,provider_provenance,forecast_raw,
    forecast_parse_status,forecast_value_kind,forecast_value_low,
    forecast_canonical_value_low,forecast_unit,forecast_scale,
    previous_parse_status,actual_parse_status
) VALUES (
    9002,'MYFXBOOK',906938645,'myfxbook:legacy:jolts','JOLTS Job Openings',
    '2025-02-01','fixture/myfxbook.json',repeat('b',64),
    'myfxbook_jolts_gap_fill','fixture',
    'myfxbook_consensus_observation_v1','{"provider":"MYFXBOOK"}',
    '9.0','parsed','scalar',9,9000000,'count',1000000,
    'missing','missing'
);
SQL
for migration in \
    088_economic_event_release_actual_provenance.sql \
    090_economic_event_actual_observation_provenance.sql \
    091_weekly_claims_historical_consensus.sql; do
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
        --dbname="$DB_NAME" -f "$ROOT/Database/migrations/$migration" \
        >/dev/null
done

scalar() {
    psql -X --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" \
        -tAc "$1"
}

test "$(scalar "SELECT count(*) FROM economic_event_consensus WHERE consensus_source IN ('OANDA','MYFXBOOK');")" = "2"
test "$(scalar 'SELECT count(*) FROM economic_event_consensus WHERE provider_observed_at IS NOT NULL OR forecast_available_at IS NOT NULL OR source_retrieved_at IS NOT NULL OR forecast_availability_proof IS NOT NULL;')" = "0"

python3 - "$SOURCE_CSV" "$TEMP_DIR" <<'PY'
import csv, pathlib, sys
source, target = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
with source.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
row = next(item for item in rows if item["economic_event_id"] == "3467")
fields = list(row)
def write(name, value, fieldnames=fields):
    with (target / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerow({key: value[key] for key in fieldnames})
write("valid.csv", row)
variants = {
    "bad_event_id.csv": {"economic_event_id": "0"},
    "bad_provider.csv": {"consensus_source": "OTHER"},
    "bad_family.csv": {"event_family": "GDP"},
    "bad_agency.csv": {"source_agency": "BLS"},
    "bad_timestamp.csv": {"event_timestamp_utc": "2025-06-05T99:30:00.000000Z"},
    "missing_proof.csv": {"forecast_availability_proof": ""},
    "post_release.csv": {"forecast_available_at": row["event_timestamp_utc"]},
    "actual_present.csv": {"pre_release_actual_raw": "236K"},
    "missing_forecast.csv": {"forecast_raw": ""},
    "bad_unit.csv": {"forecast_unit": "percent"},
    "bad_observation.csv": {"source_observation_id": "myfxbook:wrong"},
    "bad_provenance.csv": {"provider_provenance": "{broken"},
    "bad_url.csv": {"provider_source_url": "https://example.com/calendar"},
}
for name, changes in variants.items():
    changed = dict(row); changed.update(changes); write(name, changed)
changed = dict(row)
changed.update(forecast_raw="236K", forecast_value_low="236000",
               forecast_canonical_value_low="236000")
write("conflict.csv", changed)
missing_fields = [field for field in fields if field != "forecast_availability_proof"]
write("missing_column.csv", row, missing_fields)
write("duplicate_column.csv", row, fields + [fields[0]])
(target / "bad_width.csv").write_text(
    (target / "valid.csv").read_text().splitlines()[0] + "\n3467\n",
    encoding="utf-8")
PY

run_cli() {
    LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
        "$CLI_BIN" --import-economic-consensus \
        --weekly-claims-input="$1" "$2"
}
must_fail() {
    if run_cli "$1" --dry-run >/dev/null 2>&1; then
        printf 'unexpectedly accepted invalid input: %s\n' "$1" >&2
        exit 1
    fi
}
for invalid in \
    bad_event_id.csv bad_provider.csv bad_family.csv bad_agency.csv \
    bad_timestamp.csv missing_proof.csv post_release.csv actual_present.csv \
    missing_forecast.csv bad_unit.csv bad_observation.csv bad_provenance.csv \
    bad_url.csv missing_column.csv duplicate_column.csv bad_width.csv; do
    must_fail "$TEMP_DIR/$invalid"
done

DRY_RUN="$(run_cli "$TEMP_DIR/valid.csv" --dry-run)"
grep -Fq 'mode=dry-run,input=1,inserted=1,unchanged=0,rejected=0' \
    <<< "$DRY_RUN"
test "$(scalar "SELECT count(*) FROM economic_event_consensus WHERE candidate_classification='myfxbook_weekly_claims_pre_release_snapshot';")" = "0"
FIRST="$(run_cli "$TEMP_DIR/valid.csv" --apply)"
grep -Fq 'mode=apply,input=1,inserted=1,unchanged=0,rejected=0' <<< "$FIRST"
REPLAY="$(run_cli "$TEMP_DIR/valid.csv" --apply)"
grep -Fq 'mode=apply,input=1,inserted=0,unchanged=1,rejected=0' <<< "$REPLAY"
if run_cli "$TEMP_DIR/conflict.csv" --apply >/dev/null 2>&1; then
    echo 'conflicting immutable evidence unexpectedly succeeded' >&2
    exit 1
fi

test "$(scalar "SELECT count(*) FROM economic_event_selected_consensus WHERE economic_event_id=3467 AND consensus_value_low=235000 AND consensus_unit='count' AND consensus_scale=1 AND source_artifact_sha256='e48ab1c6a6ad195d8c7d85f5b3c27693bd4c04ce85dc41426e0e689c3bac419b' AND provider_observed_at='2025-06-05 06:37:51.1+00' AND forecast_available_at='2025-06-05 06:37:51.1+00' AND source_retrieved_at='2026-09-05 17:00:00+00' AND forecast_availability_proof='internet_archive_pre_release_capture';")" = "1"
test "$(scalar 'SELECT count(*) FROM economic_event_selected_consensus;')" = "3"

if psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" -c "INSERT INTO economic_event_consensus (economic_event_id,consensus_source,source_event_id,source_observation_id,source_event_name,source_release_date,source_artifact_path,source_artifact_sha256,candidate_classification,match_rule,semantic_contract,provider_provenance,provider_observed_at,forecast_available_at,source_retrieved_at,forecast_availability_proof,forecast_raw,forecast_parse_status,forecast_value_kind,forecast_value_low,forecast_canonical_value_low,forecast_unit,forecast_scale,previous_parse_status,actual_parse_status) SELECT economic_event_id,consensus_source,source_event_id+10000,source_observation_id||':late',source_event_name,source_release_date,source_artifact_path,source_artifact_sha256,candidate_classification,match_rule,semantic_contract,provider_provenance,provider_observed_at,'2025-06-05 12:30:00+00',source_retrieved_at,forecast_availability_proof,forecast_raw,forecast_parse_status,forecast_value_kind,forecast_value_low,forecast_canonical_value_low,forecast_unit,forecast_scale,previous_parse_status,actual_parse_status FROM economic_event_consensus WHERE economic_event_id=3467" \
    >"$TEMP_DIR/post-release-db.out" 2>&1; then
    echo 'database accepted post-release historical evidence' >&2
    exit 1
fi
grep -Fq 'historical consensus was not archived before canonical release' \
    "$TEMP_DIR/post-release-db.out"

psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" <<'SQL' >/dev/null
INSERT INTO economic_event_actual_observation (
    economic_event_id,source_name,source_role,source_native_event_id,
    source_observation_id,evidence_key,observation_kind,revision_sequence,
    source_publication_at,source_publication_time_status,observed_at,
    availability_proof,source_url,source_artifact_path,source_artifact_sha256,
    semantic_contract,source_provenance,actual_raw,actual_value_kind,
    actual_value_low,actual_canonical_value_low,actual_unit,actual_scale
) VALUES
(3467,'DOL_ETA','authoritative','usdl-25-942-nat','dol:3467:initial',
 'dol:3467:initial','initial',0,'2025-06-05 12:35:00+00','exact',
 '2026-09-05 17:00:00+00','source_publication',
 'https://oui.doleta.gov/press/2025/060525.pdf','fixture/060525.pdf',
 repeat('c',64),'dol_eta_weekly_claims_v1','{"provider":"DOL_ETA"}',
 '235,000','scalar',235000,235000,'count',1),
(3468,'DOL_ETA','authoritative','usdl-25-missing-nat','dol:3468:initial',
 'dol:3468:initial','initial',0,'2025-06-12 12:35:00+00','exact',
 '2026-09-05 17:00:00+00','source_publication',
 'https://oui.doleta.gov/press/2025/061225.pdf','fixture/061225.pdf',
 repeat('d',64),'dol_eta_weekly_claims_v1','{"provider":"DOL_ETA"}',
 '250,000','scalar',250000,250000,'count',1);
SQL

test "$(scalar "SELECT count(*) FROM economic_event_first_release_actual_at('2025-06-05 12:34:59+00') WHERE economic_event_id=3467;")" = "0"
test "$(scalar "SELECT count(*) FROM economic_event_first_release_actual_at('2025-06-05 12:35:00+00') WHERE economic_event_id=3467 AND first_release_actual_value_low=235000;")" = "1"

LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_TEST_DB_NAME="$DB_NAME" \
    "$FEATURE_BIN"

cleanup
trap - EXIT
printf 'WEEKLY_CLAIMS_HISTORICAL_CONSENSUS_PERSISTENCE=PASS\n'
printf 'WEEKLY_CLAIMS_DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
