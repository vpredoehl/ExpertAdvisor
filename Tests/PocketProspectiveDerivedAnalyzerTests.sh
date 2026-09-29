#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_pocket_derived.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT
"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -I"${repo_root}/Headers" \
  "${repo_root}/Tests/PocketProspectiveDerivedAnalyzerTests.cpp" -o "${test_dir}/PocketProspectiveDerivedAnalyzerTests"
(cd "${repo_root}" && "${test_dir}/PocketProspectiveDerivedAnalyzerTests")
echo "PocketProspectiveDerivedAnalyzerTests.sh passed"
