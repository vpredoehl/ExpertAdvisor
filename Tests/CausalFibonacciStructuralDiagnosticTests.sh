#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_fibonacci_structural.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT
read -r -a cflags <<< "$(pkg-config --cflags libpqxx)"
read -r -a libs <<< "$(pkg-config --libs libpqxx)"
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -I"${repo_root}/Headers" -I"${repo_root}/Sources" "${cflags[@]}" \
  "${repo_root}/Tests/CausalFibonacciStructuralDiagnosticTests.cpp" \
  "${repo_root}/Sources/CausalFibonacciStructuralDiagnostic.cpp" \
  "${repo_root}/Sources/TG4HistoricalEmpiricalEvaluation.cpp" \
  "${repo_root}/Common/HistoricalFxTimestamp.cpp" "${libs[@]}" -o "${test_dir}/test"
cd "${repo_root}"
"${test_dir}/test"
