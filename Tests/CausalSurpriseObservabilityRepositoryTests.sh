#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/causal_surprise_observability_repository_tests"
BIN="$BUILD_DIR/CausalSurpriseObservabilityRepositoryTests"
DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
DB_USER="${LSTM_DB_USER:-pqxx}"
DB_NAME="ea_causal_surprise_observability_${$}"

case "$DB_NAME" in
    ea_causal_surprise_observability_*) ;;
    *) exit 2 ;;
esac

cleanup() {
    dropdb --if-exists --host="$DB_HOST" --username="$DB_USER" \
        "$DB_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if psql -X --host="$DB_HOST" --username="$DB_USER" --dbname=postgres \
    -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1; then
    printf 'refusing to reuse existing database: %s\n' "$DB_NAME" >&2
    exit 2
fi
createdb --host="$DB_HOST" --username="$DB_USER" --template=template0 \
    "$DB_NAME"

mkdir -p "$BUILD_DIR"
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a PQXX_CFLAGS <<< "$(pkg-config --cflags libpqxx)"
    read -r -a PQXX_LIBS <<< "$(pkg-config --libs libpqxx)"
else
    PQXX_PREFIX="$(brew --prefix libpqxx@7.10.1 2>/dev/null || brew --prefix libpqxx)"
    LIBPQ_PREFIX="$(brew --prefix libpq)"
    PQXX_CFLAGS=("-I${PQXX_PREFIX}/include" "-I${LIBPQ_PREFIX}/include")
    PQXX_LIBS=("-L${PQXX_PREFIX}/lib" "-L${LIBPQ_PREFIX}/lib" -lpqxx -lpq)
fi

xcrun --sdk macosx clang++ -std=c++20 -Wall -Wextra -Werror \
    "${PQXX_CFLAGS[@]}" \
    -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "$ROOT/Common/HistoricalFxTimestamp.cpp" \
    "$ROOT/Sources/EconomicEventFeatures.cpp" \
    "$ROOT/Sources/EconomicEventRepository.cpp" \
    "$ROOT/Sources/CausalSurpriseObservability.cpp" \
    "$ROOT/Sources/CausalSurpriseObservabilityRepository.cpp" \
    "$ROOT/Sources/CausalSurpriseObservabilityService.cpp" \
    "$ROOT/Tests/CausalSurpriseObservabilityRepositoryTests.cpp" \
    "${PQXX_LIBS[@]}" \
    -o "$BIN"

LSTM_DB_HOST="$DB_HOST" LSTM_DB_USER="$DB_USER" \
LSTM_TEST_DB_NAME="$DB_NAME" "$BIN"
