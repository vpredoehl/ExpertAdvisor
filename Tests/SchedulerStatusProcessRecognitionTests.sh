#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_scheduler_status_process.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/SchedulerStatusProcessRecognitionTests.cpp" \
    -o "${test_dir}/SchedulerStatusProcessRecognitionTests"

"${test_dir}/SchedulerStatusProcessRecognitionTests"
printf '%s\n' 'SchedulerStatusProcessRecognitionTests passed'
