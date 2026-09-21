#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_tg3_fibonacci.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -pedantic \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/TG3FibonacciConfluenceIntegrationTests.cpp" \
    -o "${test_dir}/TG3FibonacciConfluenceIntegrationTests"

"${test_dir}/TG3FibonacciConfluenceIntegrationTests"
