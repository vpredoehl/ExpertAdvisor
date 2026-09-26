#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_training_ablation_reconciliation.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/TrainingFeatureAblationReconciliationTests.cpp" \
    -o "${test_dir}/TrainingFeatureAblationReconciliationTests"
"${test_dir}/TrainingFeatureAblationReconciliationTests"
printf '%s\n' "TrainingFeatureAblationReconciliationTests passed"
