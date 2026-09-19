#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
project="$root/ExpertAdvisor.xcodeproj/project.pbxproj"
header="$root/Sources/MarketDataCore/MarketDataCore.hpp"
implementation="$root/Sources/MarketDataCore/MarketDataCore.cpp"
main="$root/LSTM/main.cpp"

test -f "$header"
test -f "$implementation"
rg -q 'name = MarketDataCore;' "$project"
rg -q 'libMarketDataCore.a in Frameworks' "$project"
rg -q 'MarketDataCore' "$main"
rg -q 'LoadCandlesticks' "$implementation"
rg -q 'CheckProspectiveOutcomeCoverage' "$implementation"
! rg -q 'SchedulerCore' "$header" "$implementation"
! rg -q '#include "LSTM.hpp"' "$header" "$implementation"
