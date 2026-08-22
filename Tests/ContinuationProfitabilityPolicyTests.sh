#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_continuation_profitability.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    "${repo_root}/Sources/ContinuationPolicy.cpp" \
    "${repo_root}/Sources/ContinuationPolicyInheritance.cpp" \
    "${repo_root}/Tests/ContinuationProfitabilityPolicyTests.cpp" \
    -o "${test_dir}/ContinuationProfitabilityPolicyTests"

"${test_dir}/ContinuationProfitabilityPolicyTests"
