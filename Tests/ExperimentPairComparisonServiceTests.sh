#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/DerivedData/Development/ExperimentPairComparisonService/Tests"
mkdir -p "${build_dir}"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/ExperimentPairComparisonServiceTests.cpp" \
    "${repo_root}/Sources/ExperimentPairComparisonService.cpp" \
    "${repo_root}/Sources/ExperimentPairComparison.cpp" \
    -o "${build_dir}/ExperimentPairComparisonServiceTests"

"${build_dir}/ExperimentPairComparisonServiceTests"

command_source="${repo_root}/Sources/ExperimentPairComparisonCommand.cpp"
scheduler_source="${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"

rg -q 'pqxx::read_transaction transaction' "${command_source}"
rg -q 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ' "${command_source}"
rg -q 'LoadAuthoritativeArmEvidence' "${command_source}"
if rg -n '\b(pqxx::work|INSERT|UPDATE|DELETE|Persist)\b' "${command_source}"; then
    printf '%s\n' 'generic experiment-pair database adapter gained a write path' >&2
    exit 1
fi
rg -q -- '--compare-experiment-pair=EXPERIMENT_A_ID:EXPERIMENT_B_ID' \
    "${scheduler_source}"
rg -q -- '\[--summary\]' "${scheduler_source}"
rg -q -- '--summary requires --compare-experiment-pair' "${scheduler_source}"
rg -q 'ExperimentPairComparison::ParseExperimentIdPair' "${scheduler_source}"
rg -q 'ExperimentPairComparison::RunComparisonCommand' "${scheduler_source}"

printf '%s\n' 'Experiment pair comparison CLI contract tests passed'
