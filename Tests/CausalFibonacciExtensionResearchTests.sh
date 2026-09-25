#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_fibonacci_extensions.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -DEA_TEST_TG4_CONFIG_PATH='"'"${repo_root}/Scripts/tg4_analysis_config.frozen_v1.conf"'"' \
    -I"${repo_root}/Headers" \
    "${repo_root}/Sources/TG4HistoricalEmpiricalEvaluation.cpp" \
    "${repo_root}/Sources/CausalFibonacciExtensionHistoricalEvaluation.cpp" \
    "${repo_root}/Tests/CausalFibonacciExtensionResearchTests.cpp" \
    -o "${test_dir}/CausalFibonacciExtensionResearchTests"

"${test_dir}/CausalFibonacciExtensionResearchTests"
