#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_persisted_ablation.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

include_flags=()
while IFS= read -r include_dir; do
    include_flags+=(-isystem "${include_dir}")
done < <(find "${repo_root}/MetaNN/MetaNN/" -type d \
    -name DerivedData -prune -o -type d -print)

# Compile the real persistence component and its actual ResumeCheckpointConfig.
# Dead-strip unused database workflows; main calls only pure mask validators.
# Pin the existing libpqxx 7 headers ahead of incidental /usr/local headers.
"${CXX:-/usr/bin/clang++}" -std=c++20 -mmacosx-version-min=27.0 \
    -Wall -Wextra -Werror -Wno-deprecated-declarations \
    -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-unused-but-set-variable \
    -I/opt/homebrew/opt/libpqxx@7.10.1/include -I/opt/homebrew/opt/libpq/include \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${include_flags[@]}" -isystem /opt/homebrew/opt/libomp/include \
    "${repo_root}/Tests/TrainingWorkerPersistedAblationTests.cpp" \
    "${repo_root}/Sources/PersistedModelRuntimeConfig.cpp" \
    "${repo_root}/Sources/TrainingWorkerFeatureAblation.cpp" \
    -L/opt/homebrew/opt/libpqxx@7.10.1/lib -lpqxx -Wl,-dead_strip \
    -o "${test_dir}/TrainingWorkerPersistedAblationTests"
"${test_dir}/TrainingWorkerPersistedAblationTests"
printf '%s\n' 'TrainingWorkerPersistedAblationTests passed (pure validators; no database connection)'
