#!/usr/bin/env bash
set -euo pipefail

# This test compiles only the pure in-memory family summarizer and its
# synthetic fixture. It has no PostgreSQL, process, scheduler, or executable
# invocation.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="${EA_PHASE7_TEST_ROOT:-${repo_root}/DerivedData/Development/EconomicEventFeaturesPhase7/Tests}"
build_dir="${test_root}/FeatureAblationReplicationFamilySummary"
mkdir -p "${build_dir}"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Tests/FeatureAblationReplicationFamilySummaryTests.cpp" \
    "${repo_root}/Sources/FeatureAblationReplicationFamilySummary.cpp" \
    -o "${build_dir}/FeatureAblationReplicationFamilySummaryTests"

"${build_dir}/FeatureAblationReplicationFamilySummaryTests"
