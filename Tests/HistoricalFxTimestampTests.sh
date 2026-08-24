#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/Build/historical_fx_timestamp_tests"
BIN="$BUILD_DIR/HistoricalFxTimestampTests"
mkdir -p "$BUILD_DIR"

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"$ROOT/Headers" \
    "$ROOT/Common/HistoricalFxTimestamp.cpp" \
    "$ROOT/Tests/HistoricalFxTimestampTests.cpp" \
    -o "$BIN"

"$BIN"
