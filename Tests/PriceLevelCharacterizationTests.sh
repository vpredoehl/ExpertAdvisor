#!/usr/bin/env bash

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_price_level_characterization.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/PriceLevelCharacterizationTests.cpp" \
    -o "${test_dir}/PriceLevelCharacterizationTests"
"${test_dir}/PriceLevelCharacterizationTests"
