#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/economic_event_feature_range_repository_tests"
BIN="$BUILD_DIR/EconomicEventFeatureRangeRepositoryTests"
DB_NAME="ea_economic_event_feature_range_${$}"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
mkdir -p "$BUILD_DIR"

case "$DB_NAME" in
    ea_economic_event_feature_range_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_USER" \
        "$DB_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if psql -X --host="$DB_HOST" --username="$DB_USER" --dbname=postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME';" | grep -q 1; then
    printf 'refusing to reuse existing database: %s\n' "$DB_NAME" >&2
    exit 1
fi

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a PQXX_CFLAGS <<< "$(pkg-config --cflags libpqxx)"
    read -r -a PQXX_LIBS <<< "$(pkg-config --libs libpqxx)"
else
    PQXX_PREFIX="$(brew --prefix libpqxx)"
    LIBPQ_PREFIX="$(brew --prefix libpq)"
    PQXX_CFLAGS=("-I${PQXX_PREFIX}/include" "-I${LIBPQ_PREFIX}/include")
    PQXX_LIBS=("-L${PQXX_PREFIX}/lib" "-L${LIBPQ_PREFIX}/lib" -lpqxx -lpq)
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    "${PQXX_CFLAGS[@]}" -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "$ROOT/Sources/EconomicEventRepository.cpp" \
    "$ROOT/Sources/EconomicEventFeatures.cpp" \
    "$ROOT/Tests/EconomicEventFeatureRangeRepositoryTests.cpp" \
    "${PQXX_LIBS[@]}" -o "$BIN"

createdb --host="$DB_HOST" --username="$DB_USER" --template=template0 "$DB_NAME"
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/072_economic_event.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/081_economic_event_consensus.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/082_economic_event_consensus_provider_provenance.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" --username="$DB_USER" \
    --dbname="$DB_NAME" \
    -f "$ROOT/Database/migrations/088_economic_event_release_actual_provenance.sql" >/dev/null

if [[ -e "$ROOT/Database/migrations/083_economic_event_selected_consensus_release_semantics.sql" ]]; then
    printf 'unexpected migration 083 remains in deployment set\n' >&2
    exit 3
fi

expected_view_columns="economic_event_consensus_id,economic_event_id,consensus_value_low,consensus_value_high,consensus_value_kind,consensus_unit,consensus_scale,consensus_qualifier,consensus_source,source_report_id,source_event_id,source_observation_id,source_release_date,source_artifact_path,source_artifact_sha256,candidate_classification,match_rule,semantic_contract,provider_provenance,imported_at"
actual_view_columns="$(psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
    --username="$DB_USER" --dbname="$DB_NAME" -tAc \
    "SELECT string_agg(column_name, ',' ORDER BY ordinal_position) FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'economic_event_selected_consensus';")"
if [[ "$actual_view_columns" != "$expected_view_columns" ]]; then
    printf 'schema-082 selected-consensus projection mismatch: %s\n' \
        "$actual_view_columns" >&2
    exit 4
fi

actual_feature_view_columns="$(psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
    --username="$DB_USER" --dbname="$DB_NAME" -tAc \
    "SELECT string_agg(column_name, ',' ORDER BY ordinal_position) FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'economic_event_feature_release_actual';")"
expected_feature_view_columns="economic_event_release_actual_id,economic_event_id,available_at,actual_value_kind,actual_canonical_value_low,actual_canonical_value_high,actual_unit,actual_scale,actual_qualifier,source_agency,source_observation_id,source_artifact_path,source_artifact_sha256,semantic_contract,source_provenance"
if [[ "$actual_feature_view_columns" != "$expected_feature_view_columns" ]]; then
    printf 'schema-088 feature-actual projection mismatch: %s\n' \
        "$actual_feature_view_columns" >&2
    exit 5
fi

printf 'DISPOSABLE_SCHEMA_END=088\n'
printf 'MIGRATION_083_APPLIED=false\n'
printf 'SELECTED_CONSENSUS_VIEW_082_ONLY=true\n'
printf 'FEATURE_RELEASE_ACTUAL_VIEW_088=true\n'

LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$BIN"

cleanup
trap - EXIT
printf 'DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
