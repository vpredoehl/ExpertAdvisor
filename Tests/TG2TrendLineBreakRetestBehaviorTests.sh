#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_tg2_behavior.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -pedantic \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/TG2TrendLineBreakRetestBehaviorTests.cpp" \
    -o "${test_dir}/TG2TrendLineBreakRetestBehaviorTests"

"${test_dir}/TG2TrendLineBreakRetestBehaviorTests"
