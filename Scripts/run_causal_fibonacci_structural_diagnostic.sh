#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="${root}/DerivedData/CausalFibonacciStructuralDiagnostic"
mkdir -p "$dir"
read -r -a cflags <<< "$(pkg-config --cflags libpqxx)"
read -r -a libs <<< "$(pkg-config --libs libpqxx)"
"${CXX:-c++}" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Werror -pedantic -Wno-c++23-attribute-extensions -I"$root/Headers" -I"$root/Sources" "${cflags[@]}" "$root/Sources/TG4HistoricalEmpiricalEvaluation.cpp" "$root/Sources/TG4HistoricalMarketDataRepository.cpp" "$root/Common/HistoricalFxTimestamp.cpp" "$root/Sources/CausalFibonacciStructuralDiagnostic.cpp" "$root/Sources/CausalFibonacciStructuralDiagnosticCLI.cpp" "${libs[@]}" -o "$dir/diagnostic"
cd "$root"; exec "$dir/diagnostic" "$@"
