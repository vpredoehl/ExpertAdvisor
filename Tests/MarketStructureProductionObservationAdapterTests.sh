#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_market_structure_production_observations.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/MarketStructureProductionObservationAdapterTests.cpp" \
    -o "${test_dir}/MarketStructureProductionObservationAdapterTests"
"${test_dir}/MarketStructureProductionObservationAdapterTests"

! rg -q 'confluence\.' "${repo_root}/Headers/FeatureLayout.hpp"
! rg -q '"confluence"' "${repo_root}/Headers/MarketStructureRegistry.hpp"
! rg -q 'case 11:' "${repo_root}/Headers/ModelInputExpansion.hpp"
echo "MarketStructureProductionObservationAdapterTests source-boundary checks passed"
