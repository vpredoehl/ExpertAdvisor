#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_launch_arguments_seed.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    -I"${repo_root}/Sources" \
    "${repo_root}/Tests/LaunchArgumentsFreshInitializationSeedTests.cpp" \
    "${repo_root}/Sources/LaunchArguments.cpp" \
    -o "${test_dir}/LaunchArgumentsFreshInitializationSeedTests"
"${test_dir}/LaunchArgumentsFreshInitializationSeedTests"
printf '%s\n' "LaunchArgumentsFreshInitializationSeedTests passed"
