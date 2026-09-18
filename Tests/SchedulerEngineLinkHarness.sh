#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release/libSchedulerCore.a"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_scheduler_engine_link.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

test -f "${archive}"
clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/SchedulerEngineLinkHarness.cpp" \
    "${archive}" \
    -Wl,-map,"${test_dir}/SchedulerEngineLinkHarness.map" \
    -o "${test_dir}/SchedulerEngineLinkHarness"

"${test_dir}/SchedulerEngineLinkHarness"
rg -q 'libSchedulerCore\.a\(SchedulerEngine\.o\)' \
    "${test_dir}/SchedulerEngineLinkHarness.map"
if rg -q 'libSchedulerCore\.a\(ExperimentScheduler\.o\)' \
    "${test_dir}/SchedulerEngineLinkHarness.map"; then
    printf '%s\n' "ExperimentScheduler.o was unexpectedly extracted" >&2
    exit 1
fi
printf '%s\n' \
    "SchedulerEngineLinkHarness passed; ExperimentScheduler.o not extracted"
