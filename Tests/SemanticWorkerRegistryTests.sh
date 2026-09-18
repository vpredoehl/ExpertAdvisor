#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_semantic_worker_registry.XXXXXX)"
trap 'rm -rf "${test_dir}"' EXIT
clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/SemanticWorkerRegistryTests.cpp" \
    "${repo_root}/Sources/SchedulerCore/SemanticWorkerRegistry.cpp" \
    -o "${test_dir}/SemanticWorkerRegistryTests"
"${test_dir}/SemanticWorkerRegistryTests"
printf '%s\n' "SemanticWorkerRegistryTests passed"
