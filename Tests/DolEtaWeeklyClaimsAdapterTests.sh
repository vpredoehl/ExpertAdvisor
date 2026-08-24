#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/Build/dol_eta_weekly_claims_adapter_tests"
BIN="$BUILD_DIR/DolEtaWeeklyClaimsAdapterTests"
mkdir -p "$BUILD_DIR"

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "$ROOT/Common/HistoricalFxTimestamp.cpp" \
    "$ROOT/Sources/DolEtaWeeklyClaimsAdapter.cpp" \
    "$ROOT/Tests/DolEtaWeeklyClaimsAdapterTests.cpp" \
    -o "$BIN"

"$BIN" "$ROOT/Tests/fixtures/economic_calendar/dol_eta"
