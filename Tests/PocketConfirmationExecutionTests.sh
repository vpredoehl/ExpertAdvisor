#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_pocket_confirmation_execution.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT
"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -I"${repo_root}/Headers" \
  "${repo_root}/Tests/PocketConfirmationExecutionTests.cpp" -o "${test_dir}/PocketConfirmationExecutionTests"
(cd "${repo_root}" && "${test_dir}/PocketConfirmationExecutionTests")
rg -q -- 'resolutionEnd' "${repo_root}/Sources/PocketProspectiveEvaluatorCli.hpp"
rg -Fq -- 'LoadAndValidateConfiguration(options.configuration)' "${repo_root}/Sources/PocketConfirmationExecutionCli.hpp"
echo "PocketConfirmationExecutionTests.sh passed"
