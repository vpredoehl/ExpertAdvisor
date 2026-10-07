#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_training_worker_ablation.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    -I"${repo_root}/Sources" \
    "${repo_root}/Tests/TrainingWorkerFeatureAblationTests.cpp" \
    "${repo_root}/Sources/TrainingWorkerFeatureAblation.cpp" \
    -o "${test_dir}/TrainingWorkerFeatureAblationTests"
"${test_dir}/TrainingWorkerFeatureAblationTests"

# The managed resume path must call the extracted implementation; keeping a
# second file-local copy in main.cpp would silently defeat the boundary.
rg -q 'Training::ValidateExpandedResumeAblationComposition\(' \
    "${repo_root}/Sources/PersistedModelRuntimeConfig.cpp"
! rg -q '^void ValidateExpandedResumeAblationComposition\(' \
    "${repo_root}/LSTM/main.cpp"
printf '%s\n' "TrainingWorkerFeatureAblationTests passed"
