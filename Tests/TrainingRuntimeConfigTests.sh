#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_training_runtime_config.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    -I"${repo_root}/Sources" \
    "${repo_root}/Tests/TrainingRuntimeConfigTests.cpp" \
    -o "${test_dir}/TrainingRuntimeConfigTests"
"${test_dir}/TrainingRuntimeConfigTests"

rg -q 'const std::optional<EA::Training::RuntimeConfig> trainingRuntimeConfig' \
    "${repo_root}/LSTM/main.cpp"
rg -U -q 'SavePeriodicCheckpointIfDue[\s\S]{0,650}const EA::Training::RuntimeConfig& runtimeConfig' \
    "${repo_root}/LSTM/main.cpp"
printf '%s\n' "TrainingRuntimeConfigTests passed"
