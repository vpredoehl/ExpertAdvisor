#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_scheduler_services.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/SchedulerCore/SchedulerPolicy.cpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerRepository.cpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerAdmissionService.cpp" \
    "${repo_root}/Sources/SchedulerCore/ReconciliationService.cpp" \
    "${repo_root}/Sources/SchedulerCore/WorkerProcessController.cpp" \
    "${repo_root}/Sources/SchedulerCore/WorkerControlService.cpp" \
    "${repo_root}/Tests/SchedulerOrchestrationServiceTests.cpp" \
    -o "${test_dir}/SchedulerOrchestrationServiceTests"

"${test_dir}/SchedulerOrchestrationServiceTests"
printf '%s\n' "SchedulerOrchestrationServiceTests passed"
