#!/usr/bin/env bash

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_binary="${1:-/tmp/price_level_phase5a_characterization}"

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a pqxx_cflags <<< "$(pkg-config --cflags libpqxx)"
    read -r -a pqxx_libs <<< "$(pkg-config --libs libpqxx)"
else
    pqxx_prefix="$(brew --prefix libpqxx@7.10.1 2>/dev/null || brew --prefix libpqxx)"
    libpq_prefix="$(brew --prefix libpq)"
    pqxx_cflags=("-I${pqxx_prefix}/include" "-I${libpq_prefix}/include")
    pqxx_libs=("-L${pqxx_prefix}/lib" "-L${libpq_prefix}/lib" -lpqxx -lpq)
fi

mkdir -p "$(dirname "${output_binary}")"
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -Wno-c++23-attribute-extensions \
    -I"${repo_root}/Headers" \
    "${repo_root}/Common/HistoricalFxTimestamp.cpp" \
    "${repo_root}/Sources/PriceLevelCharacterizationCLI.cpp" \
    "${pqxx_cflags[@]}" "${pqxx_libs[@]}" \
    -o "${output_binary}"
