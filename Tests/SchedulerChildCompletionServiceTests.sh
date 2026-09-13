#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_child_completion.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

if rg -n 'pqxx|libpq|ExperimentScheduler.hpp|PostgresSchedulerRepository' \
    "${repo_root}/Sources/SchedulerCore/SchedulerChildCompletionService.hpp"; then
    printf '%s\n' \
        "scheduler child completion header exposes a forbidden dependency" >&2
    exit 1
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/SchedulerCore/SchedulerChildCompletionService.cpp" \
    "${repo_root}/Tests/SchedulerChildCompletionServiceTests.cpp" \
    -o "${test_dir}/SchedulerChildCompletionServiceTests"

"${test_dir}/SchedulerChildCompletionServiceTests"
printf '%s\n' "SchedulerChildCompletionServiceTests passed"
