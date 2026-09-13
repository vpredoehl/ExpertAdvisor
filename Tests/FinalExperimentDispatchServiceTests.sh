#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_final_dispatch.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

if rg -n 'pqxx|libpq|ExperimentScheduler.hpp|PostgresSchedulerRepository' \
    "${repo_root}/Sources/SchedulerCore/FinalExperimentDispatchService.hpp"; then
    printf '%s\n' "final dispatch service header exposes a forbidden dependency" >&2
    exit 1
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/SchedulerCore/FinalExperimentDispatchService.cpp" \
    "${repo_root}/Tests/FinalExperimentDispatchServiceTests.cpp" \
    -o "${test_dir}/FinalExperimentDispatchServiceTests"

"${test_dir}/FinalExperimentDispatchServiceTests"
printf '%s\n' "FinalExperimentDispatchServiceTests passed"
