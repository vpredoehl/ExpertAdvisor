#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_phase4f_tests.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT

"${CXX:-clang++}" \
  -std=c++20 \
  -Wall -Wextra -Werror \
  -I"$repo_root/Headers" \
  -I"$repo_root/Sources" \
  "$repo_root/Tests/ReplicationCandidatePlanningTests.cpp" \
  "$repo_root/Sources/ReplicationCandidatePlanning.cpp" \
  "$repo_root/Sources/PostPairReplicationPolicy.cpp" \
  "$repo_root/Sources/PairedTrainingObjectiveEvaluation.cpp" \
  "$repo_root/Sources/InferenceProfitability.cpp" \
  -o "$build_dir/replication_candidate_planning_tests"

"$build_dir/replication_candidate_planning_tests"

if rg -q '#include <pqxx|RunComparisonCommand|ExperimentScheduler|experimentId' \
  "$repo_root/Sources/ReplicationCandidatePlanning.hpp" \
  "$repo_root/Sources/ReplicationCandidatePlanning.cpp"; then
  echo "phase4f_forbidden_dependency_detected" >&2
  exit 1
fi
