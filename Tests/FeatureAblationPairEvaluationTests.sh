#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="${EA_CONSENSUS_EFFICACY_TEST_ROOT:-${repo_root}/DerivedData/Development/ConsensusEfficacyAnalysis/Tests}"
build_dir="${test_root}/FeatureAblationPairEvaluation"
mkdir -p "${build_dir}"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/FeatureAblationPairEvaluationTests.cpp" \
    "${repo_root}/Sources/FeatureAblationPairEvaluation.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    -o "${build_dir}/FeatureAblationPairEvaluationTests"

"${build_dir}/FeatureAblationPairEvaluationTests"
