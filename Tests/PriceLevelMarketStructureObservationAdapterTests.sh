#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_price_level_market_structure.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/PriceLevelMarketStructureObservationAdapterTests.cpp" \
    -o "${test_dir}/PriceLevelMarketStructureObservationAdapterTests"
"${test_dir}/PriceLevelMarketStructureObservationAdapterTests"

adapter="${repo_root}/Headers/PriceLevelMarketStructureObservationAdapter.hpp"
! rg -q 'Tensor.hpp|FeatureLayout|FeatureAblation|ModelInputExpansion|LSTM' "$adapter"
! rg -q 'price_level' "${repo_root}/Headers/FeatureLayout.hpp" \
    "${repo_root}/Headers/FeatureAblation.hpp" \
    "${repo_root}/Headers/ModelInputExpansion.hpp"
echo "PriceLevelMarketStructureObservationAdapterTests source-boundary checks passed"
