#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/causal_surprise_observability_tests"
BIN="$BUILD_DIR/CausalSurpriseObservabilityTests"

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
    "$ROOT/Sources/EconomicEventFeatures.cpp" \
    "$ROOT/Sources/CausalSurpriseObservability.cpp" \
    "$ROOT/Tests/CausalSurpriseObservabilityTests.cpp" \
    "${PQXX_LIBS[@]}" \
    -o "$BIN"

"$BIN"
