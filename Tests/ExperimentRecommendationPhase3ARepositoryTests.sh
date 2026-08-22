#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(mktemp -d /tmp/ea_recommendation_phase3a_repository.XXXXXX)"
trap 'rm -rf -- "${build_dir}"' EXIT

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a pqxx_cflags <<< "$(pkg-config --cflags libpqxx)"
    read -r -a pqxx_libs <<< "$(pkg-config --libs libpqxx)"
else
    pqxx_prefix="$(brew --prefix libpqxx)"
    libpq_prefix="$(brew --prefix libpq)"
    pqxx_cflags=("-I${pqxx_prefix}/include" "-I${libpq_prefix}/include")
    pqxx_libs=("-L${pqxx_prefix}/lib" "-L${libpq_prefix}/lib" -lpqxx -lpq)
fi

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror \
    "${pqxx_cflags[@]}" -I"${repo_root}/Headers" \
    -I"${repo_root}/Sources" \
    "${repo_root}/Sources/ExperimentRecommendation.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationScoring.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationEvaluation.cpp" \
    "${repo_root}/Sources/ExperimentRecommendationEvaluationRepository.cpp" \
    "${repo_root}/Tests/ExperimentRecommendationPhase3ARepositoryTests.cpp" \
    "${pqxx_libs[@]}" -o "${build_dir}/repository_tests"

cd "${repo_root}"
"${build_dir}/repository_tests"
