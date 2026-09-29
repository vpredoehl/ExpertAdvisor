#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/DerivedData/Development/ControlledReplicationStudySpecification/Tests"
mkdir -p "${build_dir}"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/ControlledReplicationStudySpecificationTests.cpp" \
    "${repo_root}/Sources/ControlledReplicationStudySpecification.cpp" \
    -o "${build_dir}/ControlledReplicationStudySpecificationTests"

"${build_dir}/ControlledReplicationStudySpecificationTests"

scheduler_source="${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"
rg -q -- '--validate-controlled-replication-study=PATH' "${scheduler_source}"
rg -q -- '--compare-controlled-replication-study=PATH' "${scheduler_source}"
rg -q 'RunValidateCommand' "${scheduler_source}"
rg -q 'RunCompareCommand' "${scheduler_source}"
rg -q 'EvaluateFamilyComparison' \
    "${repo_root}/Sources/ControlledReplicationStudySpecificationService.cpp"
if rg -n '\b(pqxx::work|INSERT|UPDATE|DELETE|Persist|Queue|CreateExperiment)\b' \
    "${repo_root}/Sources/ControlledReplicationStudySpecificationService.cpp"; then
    printf '%s\n' 'study specification service gained a write path' >&2
    exit 1
fi

printf '%s\n' 'Controlled replication study specification tests passed'
