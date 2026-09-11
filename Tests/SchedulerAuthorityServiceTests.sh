#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_scheduler_authority.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

if rg -n '#include[[:space:]]*[<"]pqxx/' \
    "${repo_root}/Sources/SchedulerCore/SchedulerAuthorityService.hpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerAuthorityRepository.hpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerRepository.hpp"; then
    printf '%s\n' "pqxx leaked through a public SchedulerCore authority seam" >&2
    exit 1
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/SchedulerCore/SchedulerAuthorityService.cpp" \
    "${repo_root}/Tests/SchedulerAuthorityServiceTests.cpp" \
    -o "${test_dir}/SchedulerAuthorityServiceTests"

"${test_dir}/SchedulerAuthorityServiceTests"
printf '%s\n' "SchedulerAuthorityServiceTests passed"
