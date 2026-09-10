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
printf '%s\n' "SchedulerCoreBoundaryTests passed"
