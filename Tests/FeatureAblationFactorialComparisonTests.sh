#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/DerivedData/Development/FeatureAblationFactorialComparison/Tests"
mkdir -p "${build_dir}"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/FeatureAblationFactorialComparisonTests.cpp" \
    "${repo_root}/Sources/FeatureAblationFactorialComparison.cpp" \
    "${repo_root}/Sources/ExperimentPairComparison.cpp" \
    "${repo_root}/Sources/ExperimentPairComparisonService.cpp" \
    -o "${build_dir}/FeatureAblationFactorialComparisonTests"

"${build_dir}/FeatureAblationFactorialComparisonTests"

scheduler_source="${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"
service_source="${repo_root}/Sources/FeatureAblationFactorialComparisonService.cpp"
rg -q -- '--compare-feature-ablation-factorial=' "${scheduler_source}"
rg -q 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ' "${service_source}"
if rg -n '\b(pqxx::work|INSERT|UPDATE|DELETE|Persist)\b' "${service_source}"; then
    printf '%s\n' 'factorial comparison database adapter gained a write path' >&2
    exit 1
fi

printf '%s\n' 'Feature ablation factorial comparison CLI contract tests passed'
