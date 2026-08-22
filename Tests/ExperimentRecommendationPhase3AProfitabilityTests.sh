#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_recommendation_phase3a.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${repo_root}/Sources/ExperimentRecommendation.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationCandidateGenerator.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationScoring.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationEvaluation.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationRanking.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationConversionWorkflow.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationCampaignPlanning.cpp" \
    "${repo_root}/Tests/ExperimentRecommendationPhase3AProfitabilityTests.cpp" \
    -o "${test_dir}/ExperimentRecommendationPhase3AProfitabilityTests"

"${test_dir}/ExperimentRecommendationPhase3AProfitabilityTests"
