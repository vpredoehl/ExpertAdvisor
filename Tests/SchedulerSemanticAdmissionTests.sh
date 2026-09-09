#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_scheduler_semantic_admission.XXXXXX)"
trap 'rm -rf "${test_dir}"' EXIT
clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/SchedulerSemanticAdmissionTests.cpp" \
    -o "${test_dir}/SchedulerSemanticAdmissionTests"
"${test_dir}/SchedulerSemanticAdmissionTests"
printf '%s\n' "SchedulerSemanticAdmissionTests passed"
