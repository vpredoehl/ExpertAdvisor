#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="$(mktemp -d /tmp/ea-build-identity.XXXXXX)"
trap 'rm -rf -- "$test_root"' EXIT

generator="$repo_root/Scripts/GenerateBuildProvenance.py"
fixture_repo="$test_root/fixture-repo"
mkdir -p "$fixture_repo"
git -C "$fixture_repo" init -q
git -C "$fixture_repo" config user.email build-identity-test@example.test
git -C "$fixture_repo" config user.name build-identity-test
printf '%s\n' fixture > "$fixture_repo/source.txt"
git -C "$fixture_repo" add source.txt
git -C "$fixture_repo" commit -q -m fixture

valid_header="$test_root/valid/GeneratedBuildProvenance.hpp"
python3 "$generator" --repository-root "$fixture_repo" --output \
  "$valid_header" --configuration Release
python3 - "$valid_header" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
match = re.search(r'#define EXPERTADVISOR_SOURCE_COMMIT "([0-9a-f]+)"', text)
if not match or not re.fullmatch(r'[0-9a-f]{40}', match.group(1)):
    raise SystemExit("generated Release provenance is not exactly 40 lowercase hex")
PY

printf '%s\n' dirty >> "$fixture_repo/source.txt"
if python3 "$generator" --repository-root "$fixture_repo" --output \
    "$test_root/dirty/GeneratedBuildProvenance.hpp" --configuration Release; then
    echo "dirty Release provenance generation unexpectedly succeeded" >&2
    exit 1
fi

if rg -n 'git (rev-parse HEAD|status --porcelain)|CommandOutput' \
    "$repo_root/Sources/CampaignOperationsProductionAdmissionService.cpp"; then
    echo "runtime Git discovery remains in CaptureActualManagerBuildContract" >&2
    exit 1
fi

cxx="${CXX:-clang++}"
common_flags=(-std=c++20 -Wall -Wextra -Werror
  -I "$repo_root/Sources" -I "$repo_root/Headers")
sources=(
  "$repo_root/Sources/CampaignOperations.cpp"
  "$repo_root/Sources/CampaignOperationsDispatch.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmission.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmissionRepository.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmissionService.cpp"
  "$repo_root/Sources/ExperimentRecommendation.cpp"
)
link_flags=(-L /opt/homebrew/opt/libpqxx@7.10.1/lib
  -L /opt/homebrew/opt/libpq/lib -lpqxx -lpq)

build_and_run() {
    local name="$1" expected="$2" provenance_include="$3" ndebug="$4"
    local binary="$test_root/$name"
    local defines=(-DEXPECT_VALID_BUILD_PROVENANCE="$expected")
    [[ "$ndebug" == 1 ]] && defines+=(-DNDEBUG)
    "$cxx" "${common_flags[@]}" "${defines[@]}" \
      -I "$provenance_include" \
      "$repo_root/Tests/CampaignOperationsBuildIdentityTests.cpp" \
      "${sources[@]}" "${link_flags[@]}" -o "$binary"
    "$binary" "$binary"
}

build_and_run valid-release 1 "$(dirname "$valid_header")" 1

invalid_include="$test_root/invalid"
mkdir -p "$invalid_include"
python3 - "$invalid_include/GeneratedBuildProvenance.hpp" <<'PY'
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(
    '#pragma once\n#define EXPERTADVISOR_SOURCE_COMMIT "not-a-commit"\n'
)
PY
build_and_run invalid-release 0 "$invalid_include" 1
build_and_run missing-release 0 "$test_root/missing" 1
build_and_run valid-debug 0 "$(dirname "$valid_header")" 0

echo "Campaign Operations build-identity provenance tests passed"
