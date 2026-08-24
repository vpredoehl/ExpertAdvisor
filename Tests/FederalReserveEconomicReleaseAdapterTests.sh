#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Tests/federal_reserve_economic_release_adapter_tests"
BIN="$BUILD_DIR/FederalReserveEconomicReleaseAdapterTests"
mkdir -p "$BUILD_DIR"

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "$ROOT/Common/HistoricalFxTimestamp.cpp" \
    "$ROOT/Sources/EconomicEventImportValidation.cpp" \
    "$ROOT/Sources/FederalReserveEconomicReleaseAdapter.cpp" \
    "$ROOT/Tests/FederalReserveEconomicReleaseAdapterTests.cpp" \
    -o "$BIN"

"$BIN" "$ROOT/Tests/fixtures/economic_calendar/federal_reserve"
