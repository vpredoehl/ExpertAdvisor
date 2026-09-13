#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_experiment_transition.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

if rg -n 'pqxx|libpq|exec_params|raw SQL' \
    "${repo_root}/Sources/SchedulerCore/ExperimentTransitionService.hpp"; then
    printf '%s\n' "experiment transition service header exposes a database dependency" >&2
    exit 1
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/SchedulerCore/ExperimentTransitionService.cpp" \
    "${repo_root}/Tests/ExperimentTransitionServiceTests.cpp" \
    -o "${test_dir}/ExperimentTransitionServiceTests"

"${test_dir}/ExperimentTransitionServiceTests"
printf '%s\n' "ExperimentTransitionServiceTests passed"
