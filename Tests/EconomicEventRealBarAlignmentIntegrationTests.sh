#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

BUILD_DIR="$ROOT/Build/economic_event_real_bar_alignment_tests"
BIN="$BUILD_DIR/EconomicEventRealBarAlignmentIntegrationTests"

mkdir -p "$BUILD_DIR"

if command -v pkg-config >/dev/null 2>&1 \
   && pkg-config --exists libpqxx
then
    read -r -a PQXX_CFLAGS <<< "$(
        pkg-config --cflags libpqxx
    )"

    read -r -a PQXX_LIBS <<< "$(
        pkg-config --libs libpqxx
    )"
else
    PQXX_PREFIX="$(brew --prefix libpqxx)"
    LIBPQ_PREFIX="$(brew --prefix libpq)"

    PQXX_CFLAGS=(
        "-I${PQXX_PREFIX}/include"
        "-I${LIBPQ_PREFIX}/include"
    )

    PQXX_LIBS=(
        "-L${PQXX_PREFIX}/lib"
        "-L${LIBPQ_PREFIX}/lib"
        "-lpqxx"
        "-lpq"
    )
fi

clang++ \
    -std=c++20 \
    -Wall \
    -Wextra \
    -Werror \
    "${PQXX_CFLAGS[@]}" \
    -I"$ROOT/Headers" \
    -I"$ROOT/Sources" \
    "$ROOT/Common/HistoricalFxTimestamp.cpp" \
    "$ROOT/Sources/EconomicEventRepository.cpp" \
    "$ROOT/Sources/EconomicEventBarAlignment.cpp" \
    "$ROOT/Tests/EconomicEventRealBarAlignmentIntegrationTests.cpp" \
    "${PQXX_LIBS[@]}" \
    -o "$BIN"

INPUT15M_PATH="${INPUT15M_PATH:-$ROOT/input15m.txt}" \
LSTM_DB_HOST="${LSTM_DB_HOST:-localhost}" \
LSTM_DB_USER="${LSTM_DB_USER:-pqxx}" \
LSTM_DB_NAME="${LSTM_DB_NAME:-LSTM}" \
"$BIN"
