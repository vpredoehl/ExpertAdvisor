#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_tg4_historical.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -I"${repo_root}/Headers" \
    "${repo_root}/Sources/TG4HistoricalEmpiricalEvaluation.cpp" \
    "${repo_root}/Tests/TG4HistoricalEmpiricalEvaluationTests.cpp" \
    -o "${test_dir}/TG4HistoricalEmpiricalEvaluationTests"

"${test_dir}/TG4HistoricalEmpiricalEvaluationTests"
