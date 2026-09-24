#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/DerivedData/Development/ExperimentReplicationPlanning/Tests"
mkdir -p "${build_dir}"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/ExperimentReplicationPlanningTests.cpp" \
    "${repo_root}/Sources/ExperimentReplicationPlanningService.cpp" \
    "${repo_root}/Sources/ExperimentReplicationPlanning.cpp" \
    "${repo_root}/Sources/ExperimentReplicationComparison.cpp" \
    "${repo_root}/Sources/ExperimentPairComparisonService.cpp" \
    "${repo_root}/Sources/ExperimentPairComparison.cpp" \
    -o "${build_dir}/ExperimentReplicationPlanningTests"

"${build_dir}/ExperimentReplicationPlanningTests"

command_source="${repo_root}/Sources/ExperimentReplicationPlanningCommand.cpp"
scheduler_source="${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"

rg -q 'pqxx::read_transaction transaction' "${command_source}"
rg -q 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ' "${command_source}"
rg -q 'LoadAuthoritativeArmEvidence' "${command_source}"
if rg -n '\b(pqxx::work|INSERT|UPDATE|DELETE|Persist|Queue|CreateExperiment)\b' \
    "${command_source}"; then
    printf '%s\n' 'replication planner database adapter gained a write path' >&2
    exit 1
fi
rg -q -- '--plan-experiment-replications=SOURCE_A_ID:SOURCE_B_ID' \
    "${scheduler_source}"
rg -q -- '--replication-seeds=SEED\[,SEED\.\.\.\]' "${scheduler_source}"
rg -q 'ExperimentReplicationPlanning::ParseReplicationSeeds' \
    "${scheduler_source}"
rg -q 'ExperimentReplicationPlanning::RunPlanningCommand' \
    "${scheduler_source}"

printf '%s\n' 'Experiment replication planner CLI contract tests passed'
