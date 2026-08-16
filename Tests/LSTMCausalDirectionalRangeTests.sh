#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_lstm_causal_directional_range.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

include_flags=()
while IFS= read -r include_dir; do
    include_flags+=("-I${include_dir}")
done < <(find "${repo_root}/MetaNN" -type d -print)

"${CXX:-c++}" -std=c++20 -mmacosx-version-min=26.2 -Wall -Wextra -Werror \
    -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-unused-but-set-variable \
    -I"${repo_root}/Headers" "${include_flags[@]}" \
    "${repo_root}/Tests/LSTMCausalDirectionalRangeTests.cpp" \
    "${repo_root}/LSTM/Tensor.cpp" "${repo_root}/Common/PricePoint.cpp" \
    -L"${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release" \
    -lMetaNN -lMetalBuffer -framework Metal -framework Foundation \
    -o "${test_dir}/LSTMCausalDirectionalRangeTests"
"${test_dir}/LSTMCausalDirectionalRangeTests"
