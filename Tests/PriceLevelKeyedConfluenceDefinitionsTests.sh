#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_price_level_keyed_confluence.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/PriceLevelKeyedConfluenceDefinitionsTests.cpp" \
    -o "${test_dir}/PriceLevelKeyedConfluenceDefinitionsTests"
"${test_dir}/PriceLevelKeyedConfluenceDefinitionsTests"

header="${repo_root}/Headers/PriceLevelKeyedConfluenceDefinitions.hpp"
! rg -q 'Tensor.hpp|FeatureLayout|FeatureAblation|ModelInputExpansion|LSTM' "$header"
echo "PriceLevelKeyedConfluenceDefinitionsTests source-boundary checks passed"
