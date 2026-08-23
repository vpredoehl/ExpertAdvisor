#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_training_objective_tests.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/TrainingObjectiveTests.cpp" \
    -o "${test_dir}/TrainingObjectiveTests"

"${test_dir}/TrainingObjectiveTests"
