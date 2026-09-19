#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_scheduler_core_boundary.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/SchedulerCore/SchedulerEngine.cpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerPolicy.cpp" \
    "${repo_root}/Tests/SchedulerCoreBoundaryTests.cpp" \
    -o "${test_dir}/SchedulerCoreBoundaryTests"

"${test_dir}/SchedulerCoreBoundaryTests"

production_source="${repo_root}/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"
daemon_cli="${repo_root}/Sources/SchedulerCore/SchedulerDaemonCli.cpp"
legacy_source="${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"
legacy_header="${repo_root}/Headers/ExperimentScheduler.hpp"
project_file="${repo_root}/ExpertAdvisor.xcodeproj/project.pbxproj"

rg -q 'int RunProductionSchedulerDaemon\(' "${production_source}"
rg -q '#include "ProductionSchedulerDaemon.hpp"' "${daemon_cli}"
rg -q 'return RunProductionSchedulerDaemon\(configuration\);' "${daemon_cli}"
if rg -q '\bRunSchedulerDaemon\(' "${legacy_source}" "${legacy_header}"; then
    printf '%s\n' "legacy production daemon runner remains exposed" >&2
    exit 1
fi
if rg -q \
    'IsExperimentSchedulerCommand|RunExperimentSchedulerCli|RegisterSchedulerWorkerAttempt' \
    "${production_source}"; then
    printf '%s\n' "production composition references a legacy compatibility symbol" >&2
    exit 1
fi
rg -q 'ProductionSchedulerDaemon\.cpp in Sources' "${project_file}"
printf '%s\n' "SchedulerCoreBoundaryTests passed"
