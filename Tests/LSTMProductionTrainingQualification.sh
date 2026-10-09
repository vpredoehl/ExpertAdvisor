#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
products="${root}/DerivedData/ExpertAdvisor/Build/Products/Release"
fixtures="${root}/DerivedData/ExpertAdvisor/Phase25B3"
mkdir -p "${fixtures}"
includes=("-I${root}/Headers" "-I${root}/Tests" "-I${root}/Sources" "-I${root}/Sources/ModelInputPreparation" "-I${root}/Sources/MarketDataCore" "-I${root}/MetaNN/MetaNN")
while IFS= read -r directory; do includes+=("-I${directory}"); done < <(find -L "${root}/MetaNN/MetaNN/MetaNN" -type d -print)
read -r -a pqxx_compile <<< "$(pkg-config --cflags libpqxx)"
read -r -a pqxx_link <<< "$(pkg-config --libs libpqxx)"
for variant in timing evidence; do
    observer=(-ULSTM_NUMERICAL_TEST_OBSERVERS)
    if [[ "${variant}" == evidence ]]; then observer+=(-DLSTM_NUMERICAL_TEST_OBSERVERS); fi
    xcrun --sdk macosx clang++ -std=c++20 -mmacosx-version-min=26.2 -O3 -DNDEBUG -fobjc-arc \
        -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function \
        -Wno-unused-but-set-variable -Wno-format -Wno-ignored-qualifiers -Wno-reorder-ctor -Wno-sign-compare \
        "${observer[@]}" "${includes[@]}" "${pqxx_compile[@]}" \
        "${root}/Tests/LSTMProductionTrainingQualification.mm" "${root}/LSTM/LSTM.cpp" \
        "${root}/LSTM/Tensor.cpp" "${root}/Sources/EconomicEventFeatures.cpp" \
        "${root}/Sources/EconomicEventRepository.cpp" "${root}/Common/HistoricalFxTimestamp.cpp" \
        "${root}/Common/PricePoint.cpp" "${root}/Common/db_cursor.cpp" \
        -L"${products}" -lModelInputPreparation -lMarketDataCore -lMetaNN -lMetalBuffer \
        "${root}/LSTM/MetalForwardAffine.mm" "${pqxx_link[@]}" \
        -framework Metal -framework MetalPerformanceShaders -framework Foundation -o "${fixtures}/${variant}"
done
cp "${products}/default.metallib" "${fixtures}/default.metallib"
cp "${products}/MetaNN_metal.metallib" "${fixtures}/MetaNN.metallib"
