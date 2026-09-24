#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/DerivedData/Development/ExperimentReplicationMaterialization/Tests"
mkdir -p "${build_dir}"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/ExperimentReplicationMaterializationTests.cpp" \
    "${repo_root}/Sources/ExperimentReplicationMaterialization.cpp" \
    "${repo_root}/Sources/ExperimentReplicationPlanningService.cpp" \
    "${repo_root}/Sources/ExperimentReplicationPlanning.cpp" \
    "${repo_root}/Sources/ExperimentReplicationComparison.cpp" \
    "${repo_root}/Sources/ExperimentPairComparisonService.cpp" \
    "${repo_root}/Sources/ExperimentPairComparison.cpp" \
    -o "${build_dir}/ExperimentReplicationMaterializationTests"

"${build_dir}/ExperimentReplicationMaterializationTests"

command_source="${repo_root}/Sources/ExperimentReplicationMaterializationCommand.cpp"
repository_source="${repo_root}/Sources/ExperimentReplicationMaterializationRepository.cpp"
scheduler_source="${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"
runtime_header="${repo_root}/Sources/SchedulerCore/ProductionSchedulerRuntimeInternal.hpp"

rg -q 'pqxx::work transaction' "${command_source}"
rg -q 'SET TRANSACTION ISOLATION LEVEL SERIALIZABLE' "${command_source}"
rg -q 'LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE' "${command_source}"
rg -q 'RunMaterializationInTransaction' "${command_source}"
rg -q "'paused','train'" "${repository_source}"
rg -Fq 'max(duplicate_nonce)' "${repository_source}"
rg -q -- '--materialize-experiment-replications' "${scheduler_source}"
rg -Fq 'arg == "--materialize-experiment-replications"' "${scheduler_source}"
rg -Fq 'arg, "--materialize-experiment-replications", value' "${scheduler_source}"
rg -q 'materializeExperimentReplications' "${runtime_header}"
rg -q 'RunMaterializationCommand' "${scheduler_source}"

commit_line="$(rg -n 'transaction\.commit\(\)' "${command_source}" | cut -d: -f1)"
abort_line="$(rg -n 'transaction\.abort\(\)' "${command_source}" | cut -d: -f1)"
publish_line="$(rg -n 'output << stagedOutput\.str\(\)' "${command_source}" | cut -d: -f1)"
[[ -n "${commit_line}" && -n "${abort_line}" && -n "${publish_line}" ]]
(( commit_line < publish_line ))
(( abort_line < publish_line ))
rg -Fq 'catch (const pqxx::in_doubt_error& error)' "${command_source}"
rg -q 'state=materialization_outcome_unknown' "${command_source}"
rg -q 'transaction=commit_outcome_unknown' "${command_source}"
rg -q 'commitSucceeded = true' "${command_source}"

if rg -n 'Queue|Schedule|Worker|kill\(|SIG[A-Z]+' \
    "${repo_root}/Sources/ExperimentReplicationMaterialization.cpp" \
    "${command_source}" "${repository_source}"; then
    printf '%s\n' 'materializer gained scheduler/worker action' >&2
    exit 1
fi

printf '%s\n' 'Experiment replication materialization tests passed'
