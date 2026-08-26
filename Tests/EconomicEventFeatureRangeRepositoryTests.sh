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
    -f "$ROOT/Database/migrations/083_economic_event_selected_consensus_release_semantics.sql" >/dev/null

LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$BIN"

cleanup
trap - EXIT
printf 'DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
