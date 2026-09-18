#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_directory="${1:?usage: $0 /path/to/published/worker-directory}"
runtime_directory="$(cd "${runtime_directory}" && pwd)"
test -f "${runtime_directory}/default.metallib"
test -f "${runtime_directory}/MetaNN.metallib"

test_dir="$(mktemp -d /tmp/ea_metal_runtime_resolution.XXXXXX)"
trap 'rm -rf "${test_dir}"' EXIT
ln -s "${runtime_directory}/default.metallib" "${test_dir}/default.metallib"
ln -s "${runtime_directory}/MetaNN.metallib" "${test_dir}/MetaNN.metallib"

xcrun clang++ -std=c++20 -fobjc-arc -Wall -Wextra -Werror \
    -framework Foundation -framework Metal \
    "${repo_root}/Tests/MetalRuntimeResolutionTests.mm" \
    -o "${test_dir}/MetalRuntimeResolutionTests"

(cd / && "${test_dir}/MetalRuntimeResolutionTests")
