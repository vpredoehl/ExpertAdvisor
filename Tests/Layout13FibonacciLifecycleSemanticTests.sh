#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_layout13_fibonacci_lifecycle.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -I"${repo_root}/Headers" -I"${repo_root}/Sources" "${repo_root}/Tests/Layout13FibonacciLifecycleSemanticTests.cpp" -o "${test_dir}/Layout13FibonacciLifecycleSemanticTests"
"${test_dir}/Layout13FibonacciLifecycleSemanticTests"
printf '%s\n' 'Layout13FibonacciLifecycleSemanticTests passed'
