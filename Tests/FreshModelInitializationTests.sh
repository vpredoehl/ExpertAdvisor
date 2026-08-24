#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_fresh_model_initialization.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

products_dir="${LSTM_TEST_PRODUCTS_DIR:-${repo_root}/DerivedData/Phase4C/Build/Products/Debug}"
for required in libMetaNN.a libMetalBuffer.a default.metallib MetaNN_metal.metallib; do
    if [[ ! -f "${products_dir}/${required}" ]]; then
        printf 'missing test dependency: %s\n' "${products_dir}/${required}" >&2
        exit 2
    fi
done

include_flags=()
while IFS= read -r include_dir; do
    include_flags+=("-I${include_dir}")
done < <(find "${repo_root}/MetaNN" -type d -print)

xcrun --sdk macosx clang++ -std=c++20 -mmacosx-version-min=26.2 \
    -O1 -Wall -Wextra -Werror \
    -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function \
    -Wno-unused-but-set-variable -Wno-format -Wno-ignored-qualifiers \
    -Wno-reorder-ctor -Wno-sign-compare \
    -I"${repo_root}/Headers" "${include_flags[@]}" \
    "${repo_root}/Tests/FreshModelInitializationTests.cpp" \
    "${repo_root}/LSTM/LSTM.cpp" "${repo_root}/LSTM/Tensor.cpp" \
    "${repo_root}/Common/PricePoint.cpp" \
    -L"${products_dir}" -lMetaNN -lMetalBuffer \
    -framework Metal -framework MetalPerformanceShaders \
    -framework Foundation \
    -o "${test_dir}/FreshModelInitializationTests"

cp "${products_dir}/default.metallib" "${test_dir}/default.metallib"
cp "${products_dir}/MetaNN_metal.metallib" "${test_dir}/MetaNN.metallib"

(
    cd "${test_dir}"

    # Concurrent launches guarantee distinct fresh PIDs. Each executable image
    # has its own thread-local initialization RNG beginning at seed 42.
    ./FreshModelInitializationTests legacy legacy_a &
    legacy_pid=$!
    ./FreshModelInitializationTests auxiliary auxiliary_a &
    auxiliary_pid=$!
    [[ "${legacy_pid}" != "${auxiliary_pid}" ]]
    wait "${legacy_pid}"
    wait "${auxiliary_pid}"

    # A second pair, launched in reverse objective order, protects against
    # process launch/queue order influencing the initialization result.
    ./FreshModelInitializationTests auxiliary auxiliary_b &
    auxiliary_b_pid=$!
    ./FreshModelInitializationTests legacy legacy_b &
    legacy_b_pid=$!
    [[ "${auxiliary_b_pid}" != "${legacy_b_pid}" ]]
    wait "${auxiliary_b_pid}"
    wait "${legacy_b_pid}"

    cmp legacy_a.shared.bin auxiliary_a.shared.bin
    cmp legacy_a.shared.bin legacy_b.shared.bin
    cmp legacy_a.shared.bin auxiliary_b.shared.bin
    cmp auxiliary_a.auxiliary.bin auxiliary_b.auxiliary.bin
)

printf '%s\n' 'FreshModelInitializationTests passed'
