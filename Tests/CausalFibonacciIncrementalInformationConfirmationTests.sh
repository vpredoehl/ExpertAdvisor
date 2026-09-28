#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_fibonacci_confirmation.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
  "${repo_root}/Tests/CausalFibonacciIncrementalInformationConfirmationTests.cpp" \
  -o "${test_dir}/test"
(cd "${repo_root}" && "${test_dir}/test")
