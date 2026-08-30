#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_profitability_prospective_comparison.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT
pqxx="/opt/homebrew/Cellar/libpqxx@7.10.1/7.10.1"
libpq="/opt/homebrew/opt/libpq"

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    -I"${pqxx}/include" -I"${libpq}/include" \
    "${repo_root}/Sources/ExperimentRecommendation.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/InferenceProfitabilityRepository.cpp" \
    "${repo_root}/Sources/ProfitabilityDistribution.cpp" \
    "${repo_root}/Sources/ProfitabilityCalibration.cpp" \
    "${repo_root}/Sources/ProfitabilityVerification.cpp" \
    "${repo_root}/Tests/CampaignProfitabilityProspectiveComparisonTests.cpp" \
    -L"${pqxx}/lib" -L"${libpq}/lib" \
    -Wl,-rpath,"${pqxx}/lib" -Wl,-rpath,"${libpq}/lib" \
    -lpqxx -lpq \
    -o "${test_dir}/CampaignProfitabilityProspectiveComparisonTests"

"${test_dir}/CampaignProfitabilityProspectiveComparisonTests"
echo "CAMPAIGN_PROFITABILITY_PHASE13_COMPARISON_TESTS_PASS"
