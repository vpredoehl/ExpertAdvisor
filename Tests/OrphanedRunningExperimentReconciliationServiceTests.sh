#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea-orphan-reconciliation-service.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Sources" \
    "${repo_root}/Tests/OrphanedRunningExperimentReconciliationServiceTests.cpp" \
    "${repo_root}/Sources/SchedulerCore/ReconciliationService.cpp" \
    -o "${test_dir}/OrphanedRunningExperimentReconciliationServiceTests"

"${test_dir}/OrphanedRunningExperimentReconciliationServiceTests"
echo "OrphanedRunningExperimentReconciliationServiceTests passed"
