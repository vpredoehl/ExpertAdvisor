#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_continuation_orchestration.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

if rg -n 'pqxx|libpq' \
    "${repo_root}/Sources/SchedulerCore/ContinuationOrchestrationService.hpp"; then
    printf '%s\n' "continuation service header exposes a database dependency" >&2
    exit 1
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/ContinuationPolicy.cpp" \
    "${repo_root}/Sources/ContinuationPolicyInheritance.cpp" \
    "${repo_root}/Sources/SchedulerCore/ContinuationOrchestrationService.cpp" \
    "${repo_root}/Tests/ContinuationOrchestrationServiceTests.cpp" \
    -o "${test_dir}/ContinuationOrchestrationServiceTests"

"${test_dir}/ContinuationOrchestrationServiceTests"
printf '%s\n' "ContinuationOrchestrationServiceTests passed"
