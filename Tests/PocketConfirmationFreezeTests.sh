#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_pocket_confirmation_freeze.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT
"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -I"${repo_root}/Headers" \
  "${repo_root}/Tests/PocketConfirmationFreezeTests.cpp" -o "${test_dir}/PocketConfirmationFreezeTests"
(cd "${repo_root}" && "${test_dir}/PocketConfirmationFreezeTests")
rg -q -- 'pqxx|ReadOnlyBars|CausalPocketDetector' "${repo_root}/Headers/PocketConfirmationFreeze.hpp" && exit 1 || true
echo "PocketConfirmationFreezeTests.sh passed"
