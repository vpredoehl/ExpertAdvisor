#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_fibonacci_trendline_diagnostic.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/CausalFibonacciTrendlineConfluenceDiagnosticTests.cpp" \
    -o "${test_dir}/CausalFibonacciTrendlineConfluenceDiagnosticTests"

"${test_dir}/CausalFibonacciTrendlineConfluenceDiagnosticTests"
