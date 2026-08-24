#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_phase4e_tests.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT

"${CXX:-clang++}" \
  -std=c++20 \
  -Wall -Wextra -Werror \
  -I"$repo_root/Headers" \
  -I"$repo_root/Sources" \
  "$repo_root/Tests/PostPairReplicationPolicyTests.cpp" \
  "$repo_root/Sources/PostPairReplicationPolicy.cpp" \
  "$repo_root/Sources/PairedTrainingObjectiveEvaluation.cpp" \
  "$repo_root/Sources/InferenceProfitability.cpp" \
  -o "$build_dir/post_pair_replication_policy_tests"

"$build_dir/post_pair_replication_policy_tests"
