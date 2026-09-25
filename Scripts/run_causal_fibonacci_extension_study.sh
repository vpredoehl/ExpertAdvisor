#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/DerivedData/FibonacciExtensionStudy"
binary="${build_dir}/causal-fibonacci-extension-study"
mkdir -p "${build_dir}"

read -r -a pqxx_cflags <<< "$(pkg-config --cflags libpqxx)"
read -r -a pqxx_libs <<< "$(pkg-config --libs libpqxx)"

"${CXX:-c++}" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -pedantic \
    -Wno-c++23-attribute-extensions \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${pqxx_cflags[@]}" \
    "${repo_root}/Sources/TG4HistoricalEmpiricalEvaluation.cpp" \
    "${repo_root}/Sources/TG4HistoricalMarketDataRepository.cpp" \
    "${repo_root}/Common/HistoricalFxTimestamp.cpp" \
    "${repo_root}/Sources/CausalFibonacciExtensionHistoricalEvaluation.cpp" \
    "${repo_root}/Sources/CausalFibonacciExtensionHistoricalEvaluationCLI.cpp" \
    "${pqxx_libs[@]}" -o "${binary}"

cd "${repo_root}"
exec "${binary}" "$@"
