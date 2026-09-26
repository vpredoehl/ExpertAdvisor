#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_launch_arguments_ablation.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    -I"${repo_root}/Sources" \
    "${repo_root}/Tests/LaunchArgumentsFeatureAblationTests.cpp" \
    "${repo_root}/Sources/LaunchArguments.cpp" \
    -o "${test_dir}/LaunchArgumentsFeatureAblationTests"
"${test_dir}/LaunchArgumentsFeatureAblationTests"
printf '%s\n' "LaunchArgumentsFeatureAblationTests passed"
