#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/census_economic_release_adapter_tests"
BIN="$BUILD_DIR/CensusEconomicReleaseAdapterTests"
mkdir -p "$BUILD_DIR"

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "$ROOT/Common/HistoricalFxTimestamp.cpp" \
    "$ROOT/Sources/CensusEconomicReleaseAdapter.cpp" \
    "$ROOT/Tests/CensusEconomicReleaseAdapterTests.cpp" \
    -o "$BIN"

"$BIN" "$ROOT/Tests/fixtures/economic_calendar/census"
