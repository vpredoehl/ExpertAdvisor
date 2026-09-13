#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_checkpoint_evaluation.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

if rg -n 'pqxx|libpq|exec_params|raw SQL' \
    "${repo_root}/Sources/SchedulerCore/CheckpointEvaluationService.hpp"; then
    printf '%s\n' "checkpoint evaluation service header exposes a database dependency" >&2
    exit 1
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/CheckpointPolicy.cpp" \
    "${repo_root}/Sources/SchedulerCore/CheckpointEvaluationService.cpp" \
    "${repo_root}/Tests/CheckpointEvaluationServiceTests.cpp" \
    -o "${test_dir}/CheckpointEvaluationServiceTests"

"${test_dir}/CheckpointEvaluationServiceTests"
printf '%s\n' "CheckpointEvaluationServiceTests passed"
