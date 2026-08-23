#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_profitability_distribution.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/ExperimentRecommendation.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationScoring.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationEvaluation.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationRanking.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/ProfitabilityDistribution.cpp" \
    "${repo_root}/Tests/ProfitabilityDistributionTests.cpp" \
    -o "${test_dir}/ProfitabilityDistributionTests"

"${test_dir}/ProfitabilityDistributionTests"
