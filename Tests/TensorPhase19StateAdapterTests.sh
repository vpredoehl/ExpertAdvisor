#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_tensor_phase19_state.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

include_flags=()
while IFS= read -r include_dir; do
    include_flags+=("-I${include_dir}")
done < <(find "${repo_root}/MetaNN" -type d -print)

"${CXX:-clang++}" -std=c++20 -mmacosx-version-min=26.2 \
    -Wall -Wextra -Werror \
    -Wno-unused-parameter -Wno-ignored-qualifiers \
    -Wno-unused-but-set-variable \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${include_flags[@]}" \
    "${repo_root}/Tests/TensorPhase19StateAdapterTests.cpp" \
    "${repo_root}/Sources/StrategyEvaluationAdapters/TensorPhase19StateAdapter.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/Phase19StateInteractionAnalysis.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/ControlledOneSidedStopExtensionExperiment.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/StrategyEvaluation.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/LSTM/Tensor.cpp" \
    "${repo_root}/Common/PricePoint.cpp" \
    "${repo_root}/Sources/EconomicEventFeatures.cpp" \
    -L"${repo_root}/DerivedData/Development/Build/Products/Debug" \
    -lMetaNN -lMetalBuffer -framework Metal -framework Foundation \
    -o "${test_dir}/TensorPhase19StateAdapterTests"

"${test_dir}/TensorPhase19StateAdapterTests"
printf '%s\n' "TensorPhase19StateAdapterTests passed"
