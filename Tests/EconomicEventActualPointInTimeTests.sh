#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/economic_event_actual_point_in_time_tests"
BIN="$BUILD_DIR/EconomicEventActualPointInTimeTests"
DB_NAME="ea_economic_event_actual_pit_${$}"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
DB_ADMIN_USER="${LSTM_DB_ADMIN_USER:-${USER:-vjp}}"
mkdir -p "$BUILD_DIR"

case "$DB_NAME" in
    ea_economic_event_actual_pit_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_ADMIN_USER" \
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
    "$ROOT/Tests/EconomicEventActualPointInTimeTests.cpp" \
    "${PQXX_LIBS[@]}" -o "$BIN"

createdb --host="$DB_HOST" --username="$DB_ADMIN_USER" --owner="$DB_USER" --template=template0 "$DB_NAME"
for migration in \
    072_economic_event.sql \
    081_economic_event_consensus.sql \
    082_economic_event_consensus_provider_provenance.sql \
    088_economic_event_release_actual_provenance.sql \
    090_economic_event_actual_observation_provenance.sql; do
    psql -X -v ON_ERROR_STOP=1 --host="$DB_HOST" \
        --username="$DB_USER" --dbname="$DB_NAME" \
        -f "$ROOT/Database/migrations/$migration" >/dev/null
done

LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" LSTM_DB_NAME="$DB_NAME" \
    "$BIN"

cleanup
trap - EXIT
printf 'PHASE12_POINT_IN_TIME_FEATURES=PASS\n'
printf 'PHASE12_DISPOSABLE_DATABASE_DROPPED=%s\n' "$DB_NAME"
