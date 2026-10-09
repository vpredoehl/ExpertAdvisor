#!/usr/bin/env bash
set -euo pipefail
# Check active scheduler/training/inference processes before invoking GPU tests.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
products_dir="${LSTM_TEST_PRODUCTS_DIR:-${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release}"
test_dir="$(mktemp -d /tmp/ea_phase25b1_matrix.XXXXXX)"
trap 'printf "matrix evidence directory: %s\n" "${test_dir}"' EXIT
include_flags=("-I${repo_root}/Headers" "-I${repo_root}/MetaNN/MetaNN")
while IFS= read -r include_dir; do
    include_flags+=("-I${include_dir}")
done < <(find -L "${repo_root}/MetaNN/MetaNN/MetaNN" -type d -print)
xcrun --sdk macosx clang++ -std=c++20 -mmacosx-version-min=26.2 -O3 -DNDEBUG \
    -fobjc-arc -Wall -Wextra -Werror -Wno-unused-parameter -Wno-ignored-qualifiers \
    "${include_flags[@]}" "${repo_root}/Tests/MetalForwardAffineTests.mm" \
    "${repo_root}/LSTM/MetalForwardAffine.mm" \
    "${repo_root}/MetaNN/MetaNN/MetaNN/metal/metal_matmul.mm" \
    -L"${products_dir}" -lMetaNN \
    -framework Foundation -framework Metal -framework MetalPerformanceShaders \
    -o "${test_dir}/matrix-tests"
cp "${products_dir}/default.metallib" "${test_dir}/default.metallib"
(
    cd "${test_dir}"
    env -u EA_LSTM_FORWARD_AFFINE ./matrix-tests --selection
    EA_LSTM_FORWARD_AFFINE=metann ./matrix-tests --selection
    EA_LSTM_FORWARD_AFFINE=combined ./matrix-tests --selection
    if EA_LSTM_FORWARD_AFFINE=invalid ./matrix-tests --selection; then exit 1; fi
    EA_LSTM_FORWARD_AFFINE=metann ./matrix-tests
    EA_LSTM_FORWARD_AFFINE=combined ./matrix-tests
    if [[ "${1:-}" == --benchmark ]]; then ./matrix-tests --benchmark | tee benchmark.log; fi
)
