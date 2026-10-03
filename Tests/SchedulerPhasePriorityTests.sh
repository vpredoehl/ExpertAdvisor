#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_phase_priority.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT
"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror -I"${repo_root}/Sources" \
    "${repo_root}/Tests/SchedulerPhasePriorityTests.cpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerPolicy.cpp" \
    -o "${test_dir}/SchedulerPhasePriorityTests"
"${test_dir}/SchedulerPhasePriorityTests"
printf '%s\n' 'SchedulerPhasePriorityTests passed'
