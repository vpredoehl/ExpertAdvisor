#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_phase17d_strategy.XXXXXX)"
trap 'rm -rf "${test_dir}"' EXIT

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Sources" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/StrategyEvaluation.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/ControlledFixedStopExperiment.cpp" \
    "${repo_root}/Tests/ControlledFixedStopExperimentTests.cpp" \
    -o "${test_dir}/ControlledFixedStopExperimentTests"

"${test_dir}/ControlledFixedStopExperimentTests"
printf '%s\n' "ControlledFixedStopExperimentTests passed"
