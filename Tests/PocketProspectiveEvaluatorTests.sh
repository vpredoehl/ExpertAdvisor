#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_pocket_prospective.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT
"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -I"${repo_root}/Headers" \
  "${repo_root}/Tests/PocketProspectiveEvaluatorTests.cpp" -o "${test_dir}/PocketProspectiveEvaluatorTests"
(cd "${repo_root}" && "${test_dir}/PocketProspectiveEvaluatorTests")
"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -I"${repo_root}/Headers" \
  -I/opt/homebrew/opt/libpqxx@7.10.1/include -I/opt/homebrew/opt/libpq/include \
  "${repo_root}/Tests/PocketProspectiveEvaluatorCliTests.cpp" -L/opt/homebrew/opt/libpqxx@7.10.1/lib \
  -L/opt/homebrew/opt/libpq/lib -lpqxx -lpq -o "${test_dir}/PocketProspectiveEvaluatorCliTests"
"${test_dir}/PocketProspectiveEvaluatorCliTests"
rg -q -- 'command != "--pocket-prospective-evaluator"' "${repo_root}/Sources/PocketProspectiveEvaluatorCli.hpp"
rg -q -- 'kPreconfirmationEnd' "${repo_root}/Sources/PocketProspectiveEvaluatorCli.hpp"
rg -q -- 'READ ONLY' "${repo_root}/Sources/PocketProspectiveEvaluatorCli.hpp"
echo "PocketProspectiveEvaluatorTests.sh passed"
