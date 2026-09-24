#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/DerivedData/Development/ExperimentReplicationComparison/Tests"
mkdir -p "${build_dir}"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/ExperimentReplicationComparisonTests.cpp" \
    "${repo_root}/Sources/ExperimentReplicationComparisonService.cpp" \
    "${repo_root}/Sources/ExperimentReplicationComparison.cpp" \
    "${repo_root}/Sources/ExperimentPairComparisonService.cpp" \
    "${repo_root}/Sources/ExperimentPairComparison.cpp" \
    -o "${build_dir}/ExperimentReplicationComparisonTests"

"${build_dir}/ExperimentReplicationComparisonTests"

command_source="${repo_root}/Sources/ExperimentReplicationComparisonCommand.cpp"
scheduler_source="${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"

rg -q 'pqxx::read_transaction transaction' "${command_source}"
rg -q 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ' "${command_source}"
rg -q 'LoadAuthoritativeArmEvidence' "${command_source}"
if rg -n '\b(pqxx::work|INSERT|UPDATE|DELETE|Persist)\b' "${command_source}"; then
    printf '%s\n' 'experiment replication database adapter gained a write path' >&2
    exit 1
fi
rg -q -- '--compare-experiment-replications=A_ID:B_ID,C_ID:D_ID' \
    "${scheduler_source}"
rg -q 'ExperimentReplicationComparison::ParseExperimentIdPairs' \
    "${scheduler_source}"
rg -q 'ExperimentReplicationComparison::RunComparisonCommand' \
    "${scheduler_source}"

printf '%s\n' 'Experiment replication comparison CLI contract tests passed'
