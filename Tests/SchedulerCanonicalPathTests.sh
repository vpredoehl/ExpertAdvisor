#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/scheduler-canonical-path.XXXXXX")"
cleanup() {
    rm -rf "${test_dir}"
}
trap cleanup EXIT

binary="${test_dir}/SchedulerOwnershipPolicyTests"
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror \
    "${repo_root}/Tests/SchedulerOwnershipPolicyTests.cpp" \
    -o "${binary}"
canonical_binary="$(realpath "${binary}")"

direct_output="$("${binary}")"
direct_path="$(printf '%s\n' "${direct_output}" |
    sed -n 's/^CANONICAL_EXECUTABLE_PATH=//p')"
test "${direct_path}" = "${canonical_binary}"

ln -s "${binary}" "${test_dir}/scheduler-policy-symlink"
symlink_output="$("${test_dir}/scheduler-policy-symlink")"
symlink_path="$(printf '%s\n' "${symlink_output}" |
    sed -n 's/^CANONICAL_EXECUTABLE_PATH=//p')"
test "${symlink_path}" = "${canonical_binary}"

PATH="${test_dir}:${PATH}" \
    SchedulerOwnershipPolicyTests >"${test_dir}/basename.out"
basename_path="$(sed -n \
    's/^CANONICAL_EXECUTABLE_PATH=//p' \
    "${test_dir}/basename.out")"
test "${basename_path}" = "${canonical_binary}"

printf '%s\n' \
    "SchedulerCanonicalPathTests passed" \
    "direct=${direct_path}" \
    "symlink=${symlink_path}" \
    "basename=${basename_path}"
