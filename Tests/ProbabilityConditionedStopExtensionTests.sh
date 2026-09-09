#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_phase18b_strategy.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Sources" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/StrategyEvaluation.cpp" \
    "${repo_root}/Tests/ProbabilityConditionedStopExtensionTests.cpp" \
    -o "${test_dir}/ProbabilityConditionedStopExtensionTests"

"${test_dir}/ProbabilityConditionedStopExtensionTests"
printf '%s\n' "ProbabilityConditionedStopExtensionTests passed"
