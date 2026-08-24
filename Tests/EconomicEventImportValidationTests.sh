#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/Build/economic_event_import_validation_tests"
BIN="$BUILD_DIR/EconomicEventImportValidationTests"
mkdir -p "$BUILD_DIR"

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "$ROOT/Common/HistoricalFxTimestamp.cpp" \
    "$ROOT/Sources/EconomicEventImportValidation.cpp" \
    "$ROOT/Tests/EconomicEventImportValidationTests.cpp" \
    -o "$BIN"

"$BIN"
