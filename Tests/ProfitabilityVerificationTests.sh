#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="${repo_root}/DerivedData/Development/ProfitabilityPhase8/UnitTests"
mkdir -p "${test_dir}"

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a pqxx_cflags <<< "$(pkg-config --cflags libpqxx)"
    read -r -a pqxx_libs <<< "$(pkg-config --libs libpqxx)"
else
    pqxx_prefix="$(brew --prefix libpqxx)"
    libpq_prefix="$(brew --prefix libpq)"
    pqxx_cflags=("-I${pqxx_prefix}/include" "-I${libpq_prefix}/include")
    pqxx_libs=("-L${pqxx_prefix}/lib" "-L${libpq_prefix}/lib" -lpqxx -lpq)
fi

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${pqxx_cflags[@]}" \
    "${repo_root}/Sources/ExperimentRecommendation.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/InferenceProfitabilityRepository.cpp" \
    "${repo_root}/Sources/ProfitabilityDistribution.cpp" \
    "${repo_root}/Sources/ProfitabilityVerification.cpp" \
    "${repo_root}/Tests/ProfitabilityVerificationTests.cpp" \
    "${pqxx_libs[@]}" \
    -o "${test_dir}/ProfitabilityVerificationTests"

"${test_dir}/ProfitabilityVerificationTests"
