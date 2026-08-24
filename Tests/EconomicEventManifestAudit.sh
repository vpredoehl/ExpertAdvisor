#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/DerivedData/Development/Phase6A/tools"
BIN="$BUILD_DIR/EconomicEventManifestAudit"
mkdir -p "$BUILD_DIR"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 AGENCY MANIFEST" >&2
    exit 64
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"$ROOT/Headers" -I"$ROOT/Sources" \
    "$ROOT/Common/HistoricalFxTimestamp.cpp" \
    "$ROOT/Sources/BeaEconomicReleaseAdapter.cpp" \
    "$ROOT/Sources/BlsScheduleReleaseAdapter.cpp" \
    "$ROOT/Sources/CensusEconomicReleaseAdapter.cpp" \
    "$ROOT/Sources/DolEtaWeeklyClaimsAdapter.cpp" \
    "$ROOT/Sources/FederalReserveEconomicReleaseAdapter.cpp" \
    "$ROOT/Sources/EconomicEventImportValidation.cpp" \
    "$ROOT/Tests/EconomicEventManifestAudit.cpp" \
    -o "$BIN"

exec "$BIN" "$1" "$2"
