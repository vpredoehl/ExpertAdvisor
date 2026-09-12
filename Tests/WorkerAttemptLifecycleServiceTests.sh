#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_worker_attempt_lifecycle.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/SchedulerCore/WorkerAttemptLifecycleService.cpp" \
    "${repo_root}/Tests/WorkerAttemptLifecycleServiceTests.cpp" \
    -o "${test_dir}/WorkerAttemptLifecycleServiceTests"

"${test_dir}/WorkerAttemptLifecycleServiceTests"
printf '%s\n' "WorkerAttemptLifecycleServiceTests passed"
