#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/Build/market_path_fixed_stop_loss_tests"
binary="${build_dir}/MarketPathFixedStopLossTests"
mkdir -p "${build_dir}"

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Sources" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/StrategyEvaluationCore/StrategyEvaluation.cpp" \
    "${repo_root}/Tests/MarketPathFixedStopLossTests.cpp" \
    -o "${binary}"

"${binary}"
