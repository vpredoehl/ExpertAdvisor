#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_tg4_preintegration.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/TG4PreintegrationCausalBoundaryClosureTests.cpp" \
    -o "${test_dir}/TG4PreintegrationCausalBoundaryClosureTests"

"${test_dir}/TG4PreintegrationCausalBoundaryClosureTests"

! rg -q 'Artifacts/tg4-confirmation-2025-v1|Scripts/tg4_analysis_config' \
    "${repo_root}/Headers/ProductionTG1TG3PulseConfiguration.hpp"
! rg -q 'Tensor|feature_size|FeatureLayout|kCurrentModelInputWidth' \
    "${repo_root}/Headers/ProductionTG1TG3PulseConfiguration.hpp" \
    "${repo_root}/Headers/CanonicalMarketDataRange.hpp"
rg -q 'CanonicalHalfOpenCandlestickCte' \
    "${repo_root}/Sources/TG4HistoricalMarketDataRepository.cpp"
rg -q 'LoadCanonicalHalfOpenCandlesticks' \
    "${repo_root}/Sources/MarketDataCore/MarketDataCore.cpp"

echo "TG4PreintegrationCausalBoundaryClosureTests source-boundary checks passed"
