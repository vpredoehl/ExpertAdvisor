#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_tg4_canonical_replay.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a pqxx_cflags <<< "$(pkg-config --cflags libpqxx)"
    read -r -a pqxx_libs <<< "$(pkg-config --libs libpqxx)"
else
    pqxx_prefix="$(brew --prefix libpqxx@7.10.1 2>/dev/null || brew --prefix libpqxx)"
    libpq_prefix="$(brew --prefix libpq)"
    pqxx_cflags=("-I${pqxx_prefix}/include" "-I${libpq_prefix}/include")
    pqxx_libs=("-L${pqxx_prefix}/lib" "-L${libpq_prefix}/lib" -lpqxx -lpq)
fi

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -Wno-c++23-attribute-extensions \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources/MarketDataCore" \
    "${pqxx_cflags[@]}" \
    "${repo_root}/Common/PricePoint.cpp" \
    "${repo_root}/Common/HistoricalFxTimestamp.cpp" \
    "${repo_root}/Common/db_cursor.cpp" \
    "${repo_root}/Sources/MarketDataCore/MarketDataCore.cpp" \
    "${repo_root}/Tests/TG4ProductionStreamingCanonicalReplayTests.cpp" \
    "${pqxx_libs[@]}" -o "${test_dir}/TG4ProductionStreamingCanonicalReplayTests"

PGOPTIONS="${PGOPTIONS:+${PGOPTIONS} }-c default_transaction_read_only=on" \
    "${test_dir}/TG4ProductionStreamingCanonicalReplayTests"
