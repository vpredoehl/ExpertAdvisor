#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/DerivedData/Development/ExperimentPairComparison/Tests"
mkdir -p "${build_dir}"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/ExperimentPairComparisonTests.cpp" \
    "${repo_root}/Sources/ExperimentPairComparison.cpp" \
    -o "${build_dir}/ExperimentPairComparisonTests"

"${build_dir}/ExperimentPairComparisonTests"
