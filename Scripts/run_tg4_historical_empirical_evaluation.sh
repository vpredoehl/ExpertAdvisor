#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build/tg4"
binary="${build_dir}/tg4-historical-evaluation"
mkdir -p "${build_dir}"

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a pqxx_cflags <<< "$(pkg-config --cflags libpqxx)"
    read -r -a pqxx_libs <<< "$(pkg-config --libs libpqxx)"
else
    pqxx_prefix="$(brew --prefix libpqxx@7.10.1 2>/dev/null || brew --prefix libpqxx)"
    pqxx_cflags=("-I${pqxx_prefix}/include")
    pqxx_libs=("-L${pqxx_prefix}/lib" -lpqxx)
fi

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -Wno-c++23-attribute-extensions \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${pqxx_cflags[@]}" \
    "${repo_root}/Sources/TG4HistoricalEmpiricalEvaluation.cpp" \
    "${repo_root}/Sources/TG4HistoricalMarketDataRepository.cpp" \
    "${repo_root}/Sources/TG4HistoricalEmpiricalEvaluationCLI.cpp" \
    "${repo_root}/Common/HistoricalFxTimestamp.cpp" \
    "${pqxx_libs[@]}" -o "${binary}"

exec "${binary}" "$@"
