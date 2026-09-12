#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_phase19c_causal_path.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Sources" -I"${repo_root}/Headers" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/Phase19CCausalPathPredictability.cpp" \
    "${repo_root}/Tests/Phase19CCausalPathPredictabilityTests.cpp" \
    -o "${test_dir}/Phase19CCausalPathPredictabilityTests"

"${test_dir}/Phase19CCausalPathPredictabilityTests"
printf '%s\n' "Phase19CCausalPathPredictabilityTests passed"
