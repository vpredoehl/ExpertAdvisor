#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_phase19_state_analysis.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Sources" -I"${repo_root}/Headers" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/StrategyEvaluation.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/ControlledOneSidedStopExtensionExperiment.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/Phase19StateInteractionAnalysis.cpp" \
    "${repo_root}/Tests/Phase19StateInteractionAnalysisTests.cpp" \
    -o "${test_dir}/Phase19StateInteractionAnalysisTests"

"${test_dir}/Phase19StateInteractionAnalysisTests"
printf '%s\n' "Phase19StateInteractionAnalysisTests passed"
