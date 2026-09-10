#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_scheduler_internal_seams.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

if rg -n 'pqxx|libpq' \
    "${repo_root}/Sources/SchedulerCore/SchedulerRepository.hpp" \
    "${repo_root}/Sources/SchedulerCore/WorkerProcessController.hpp" \
    "${repo_root}/Sources/SchedulerCore/WorkerProcessController.cpp"; then
    printf '%s\n' "internal seam unexpectedly exposes a database dependency" >&2
    exit 1
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/SchedulerCore/SchedulerPolicy.cpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerRepository.cpp" \
    "${repo_root}/Sources/SchedulerCore/WorkerProcessController.cpp" \
    "${repo_root}/Tests/SchedulerInternalSeamTests.cpp" \
    -o "${test_dir}/SchedulerInternalSeamTests"

"${test_dir}/SchedulerInternalSeamTests"
printf '%s\n' "SchedulerInternalSeamTests passed"
