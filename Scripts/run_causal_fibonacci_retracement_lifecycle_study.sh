#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/DerivedData/FibonacciRetracementLifecycleStudy"
binary="${build_dir}/causal-fibonacci-retracement-lifecycle-study"

cd "${repo_root}"

# Historical research artifacts must identify the exact committed source tree
# that produced them. Untracked files are permitted because prior research
# artifacts may intentionally remain in the worktree, but tracked changes
# would make HEAD insufficient provenance.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked worktree changes prevent reproducible Fibonacci artifact provenance" >&2
    git status --short >&2
    exit 1
fi

export GIT_COMMIT="$(git rev-parse HEAD)"

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
    "${repo_root}/Sources/CausalFibonacciRetracementLifecycleHistoricalArtifact.cpp" \
    "${repo_root}/Sources/CausalFibonacciRetracementLifecycleHistoricalEvaluationCLI.cpp" \
    "${pqxx_libs[@]}" -o "${binary}"

exec "${binary}" "$@"

