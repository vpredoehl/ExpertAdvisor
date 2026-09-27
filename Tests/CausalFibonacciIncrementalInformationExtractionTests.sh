#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_fibonacci_extraction.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

# Tensor is the authoritative materializer under test.  These sources are the
# same production implementations used by LSTM_Release; no database is opened.
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-ignored-qualifiers -Wno-c++23-attribute-extensions \
  -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
  -I"${repo_root}/MetaNN" -I"${repo_root}/MetaNN/MetaNN" \
  -I"${repo_root}/MetaNN/MetaNN/data/facilities" \
  -I/opt/homebrew/opt/libpqxx@7.10.1/include -I/opt/homebrew/opt/libpq/include \
  "${repo_root}/Tests/CausalFibonacciIncrementalInformationExtractionTests.cpp" \
  "${repo_root}/LSTM/Tensor.cpp" "${repo_root}/Sources/EconomicEventFeatures.cpp" \
  -L"${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release" \
  -lMetaNN -lMetalBuffer -L/opt/homebrew/opt/libpqxx@7.10.1/lib -lpqxx \
  -L/opt/homebrew/opt/libpq/lib -lpq -framework Metal -framework Foundation \
  -o "${test_dir}/test"
set +e
output="$(cd "${repo_root}" && "${test_dir}/test" 2>&1)"
status=$?
set -e
if [[ $status -ne 0 ]]; then
  if [[ "$output" == *"Metal device not available"* ]]; then
    printf '%s\n' 'CausalFibonacciIncrementalInformationExtractionTests skipped: Metal device unavailable'
    exit 0
  fi
  printf '%s\n' "$output" >&2
  exit "$status"
fi
printf '%s\n' "$output"
