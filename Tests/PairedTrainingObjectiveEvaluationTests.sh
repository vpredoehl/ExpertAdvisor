#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_phase4d_tests.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT

"${CXX:-clang++}" \
  -std=c++20 \
  -Wall -Wextra -Werror \
  -I"$repo_root/Headers" \
  -I"$repo_root/Sources" \
  "$repo_root/Tests/PairedTrainingObjectiveEvaluationTests.cpp" \
  "$repo_root/Sources/PairedTrainingObjectiveEvaluation.cpp" \
  "$repo_root/Sources/InferenceProfitability.cpp" \
  -o "$build_dir/paired_training_objective_evaluation_tests"

"$build_dir/paired_training_objective_evaluation_tests"
