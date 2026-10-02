#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_fixed_confluence_tensor_adapter.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -pedantic \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/FixedConfluenceTensorAdapterTests.cpp" \
    -o "${test_dir}/FixedConfluenceTensorAdapterTests"
"${test_dir}/FixedConfluenceTensorAdapterTests"
printf '%s\n' 'FixedConfluenceTensorAdapterTests passed'
