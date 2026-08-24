#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/DerivedData/Development/Phase6A/tests/BlsScheduleReleaseAdapterTests"
mkdir -p "$(dirname "$OUT")"
clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "$ROOT/Common/HistoricalFxTimestamp.cpp" \
    "$ROOT/Sources/BlsScheduleReleaseAdapter.cpp" \
    "$ROOT/Tests/BlsScheduleReleaseAdapterTests.cpp" \
    -framework Security -o "$OUT"
"$OUT"
