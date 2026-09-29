#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_causal_pocket_features.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
  "${repo_root}/Tests/CausalPocketFeaturesTests.cpp" \
  -o "${test_dir}/CausalPocketFeaturesTests"
"${test_dir}/CausalPocketFeaturesTests"
printf '%s\n' 'CausalPocketFeaturesTests passed'
