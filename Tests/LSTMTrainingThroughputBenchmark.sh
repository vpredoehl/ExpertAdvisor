#!/usr/bin/env bash
set -euo pipefail
# Builds isolated fixtures only; GPU runs are guarded by the Python runner.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
products_dir="${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release"
fixture_dir="${repo_root}/DerivedData/ExpertAdvisor/Phase25B2"
mkdir -p "${fixture_dir}"
for required in libMetaNN.a libMetalBuffer.a default.metallib MetaNN_metal.metallib; do
    test -f "${products_dir}/${required}"
done
include_flags=("-I${repo_root}/Headers" "-I${repo_root}/Tests" "-I${repo_root}/MetaNN/MetaNN")
while IFS= read -r include_dir; do include_flags+=("-I${include_dir}"); done \
    < <(find -L "${repo_root}/MetaNN/MetaNN/MetaNN" -type d -print)
read -r -a pqxx_compile_flags <<< "$(pkg-config --cflags libpqxx)"
read -r -a pqxx_link_flags <<< "$(pkg-config --libs libpqxx)"
for variant in timing evidence; do
    observer_flags=(-ULSTM_NUMERICAL_TEST_OBSERVERS)
    if [[ "${variant}" == evidence ]]; then observer_flags+=(-DLSTM_NUMERICAL_TEST_OBSERVERS); fi
    xcrun --sdk macosx clang++ -std=c++20 -mmacosx-version-min=26.2 \
        -O3 -DNDEBUG -fobjc-arc -Wall -Wextra -Werror \
        -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function \
        -Wno-unused-but-set-variable -Wno-format -Wno-ignored-qualifiers \
        -Wno-reorder-ctor -Wno-sign-compare \
        "${observer_flags[@]}" "${include_flags[@]}" "${pqxx_compile_flags[@]}" \
        "${repo_root}/Tests/LSTMTrainingThroughputBenchmark.mm" \
        "${repo_root}/LSTM/LSTM.cpp" "${repo_root}/LSTM/Tensor.cpp" \
        "${repo_root}/Sources/EconomicEventFeatures.cpp" "${repo_root}/Common/PricePoint.cpp" \
        "${repo_root}/LSTM/MetalForwardAffine.mm" \
        -L"${products_dir}" -lMetaNN -lMetalBuffer "${pqxx_link_flags[@]}" \
        -framework Metal -framework MetalPerformanceShaders -framework Foundation \
        -o "${fixture_dir}/${variant}"
done
cp "${products_dir}/default.metallib" "${fixture_dir}/default.metallib"
cp "${products_dir}/MetaNN_metal.metallib" "${fixture_dir}/MetaNN.metallib"
printf 'Built timing (observers absent) and evidence fixtures: %s\n' "${fixture_dir}"
