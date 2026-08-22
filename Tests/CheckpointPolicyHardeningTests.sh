#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_checkpoint_policy.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    "${repo_root}/Sources/CheckpointPolicy.cpp" \
    "${repo_root}/Tests/CheckpointPolicyHardeningTests.cpp" \
    -o "${test_dir}/CheckpointPolicyHardeningTests"

"${test_dir}/CheckpointPolicyHardeningTests"
