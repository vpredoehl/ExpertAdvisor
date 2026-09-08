#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="${EA_PHASE16_TEST_ROOT:-${repo_root}/DerivedData/Validation/Phase16CorrectedCausalSurpriseReplicationMaterializationHardening/Tests}"
build_dir="${test_root}/CorrectedCausalSurpriseReplicationContinuation"
mkdir -p "${build_dir}"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/CorrectedCausalSurpriseReplicationContinuationTests.cpp" \
    "${repo_root}/Sources/CorrectedCausalSurpriseReplicationContinuation.cpp" \
    "${repo_root}/Sources/FeatureAblationReplicationEvaluation.cpp" \
    "${repo_root}/Sources/FeatureAblationPairEvaluation.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    -o "${build_dir}/CorrectedCausalSurpriseReplicationContinuationTests"

"${build_dir}/CorrectedCausalSurpriseReplicationContinuationTests"
