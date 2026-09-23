#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_tg4_production_adapter.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -DEA_TG3_SYNCHRONIZATION_WORK_INSTRUMENTATION \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/TG4ProductionStreamingPulseAdapterTests.cpp" \
    -o "${test_dir}/TG4ProductionStreamingPulseAdapterTests"

"${test_dir}/TG4ProductionStreamingPulseAdapterTests"

adapter="${repo_root}/Headers/TG4ProductionStreamingPulseAdapter.hpp"
! rg -q 'Artifacts/tg4-confirmation-2025-v1|Scripts/tg4_analysis_config' "$adapter"
! rg -q '#include "Tensor.hpp"|FeatureLayout|kCurrentModelInputWidth' "$adapter"
rg -q 'TG4ADerivedSourceUTLUpABOnlyV1' "$adapter"
rg -q 'newlyCreatedConfluenceObservations' "$adapter"

echo "TG4ProductionStreamingPulseAdapterTests source-boundary checks passed"
