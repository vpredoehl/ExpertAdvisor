#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_scheduler_daemon_config.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/SchedulerDaemonConfigurationTests.cpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerDaemonConfiguration.cpp" \
    "${repo_root}/Sources/SchedulerCore/SemanticWorkerRegistry.cpp" \
    -o "${test_dir}/SchedulerDaemonConfigurationTests"

"${test_dir}/SchedulerDaemonConfigurationTests"
printf '%s\n' "SchedulerDaemonConfigurationTests passed"
