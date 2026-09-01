#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_lstm_input_width_expansion_persistence.XXXXXX)"
test_db="ea_input_width_expansion_persistence_${$}"

case "${test_db}" in
    ea_input_width_expansion_persistence_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf -- "${test_dir}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

include_flags=()
while IFS= read -r include_dir; do
    include_flags+=("-I${include_dir}")
done < <(find "${repo_root}/MetaNN" -type d -print)

xcrun --sdk macosx clang++ -std=c++20 -mmacosx-version-min=26.2 \
    -O1 -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable \
    -Wno-unused-function -Wno-unused-but-set-variable -Wno-format \
    -Wno-ignored-qualifiers -Wno-reorder-ctor -Wno-sign-compare \
    -I"${repo_root}/Headers" "${include_flags[@]}" \
    -isystem /opt/homebrew/opt/libpqxx@7.10.1/include \
    -isystem /opt/homebrew/opt/libpq/include \
    "${repo_root}/Tests/LSTMInputWidthExpansionPersistenceTests.cpp" \
    "${repo_root}/LSTM/LSTM.cpp" "${repo_root}/LSTM/Tensor.cpp" "${repo_root}/Sources/EconomicEventFeatures.cpp" \
    "${repo_root}/Common/PricePoint.cpp" \
    -L"${repo_root}/DerivedData/Development/Build/Products/Debug" \
    -L/opt/homebrew/opt/libpqxx@7.10.1/lib \
    -L/opt/homebrew/opt/libpq/lib \
    -lMetaNN -lMetalBuffer -lpqxx -lpq \
    -framework Metal -framework MetalPerformanceShaders \
    -framework Foundation -o "${test_dir}/persistence_test"

# Standalone Metal tests need the same compiled libraries that Xcode places
# beside LSTM_Debug; the application target normally supplies this packaging.
cp "${repo_root}/DerivedData/Development/Build/Products/Debug/default.metallib" \
   "${test_dir}/default.metallib"
cp "${repo_root}/DerivedData/Development/Build/Products/Debug/MetaNN_metal.metallib" \
   "${test_dir}/MetaNN.metallib"

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/089_lstm_model_input_identity.sql"
LSTM_DB_NAME="${test_db}" "${test_dir}/persistence_test"

printf '%s\n' 'LSTMInputWidthExpansionPersistenceTests passed'
