#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${repo_root}/DerivedData/ExpertAdvisor/ReleaseFiniteValueValidationTests"
mkdir -p "${build_root}"
test_dir="$(mktemp -d "${build_root}/run.XXXXXX")"
trap 'rm -rf -- "${test_dir}"' EXIT

# Read the checked-in target policies, rather than testing only an unrelated
# debug/default compiler invocation. Dedicated TRAIN and INFER share these libs.
python3 - "${repo_root}" "${test_dir}/optimization.txt" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
project = json.loads(subprocess.check_output([
    '/usr/bin/plutil', '-convert', 'json', '-o', '-',
    str(root / 'ExpertAdvisor.xcodeproj/project.pbxproj')]))
objects = project['objects']
for name in ('ModelInputPreparation', 'ProfitabilityCore',
             'StrategyEvaluationCore', 'SchedulerCore'):
    target = next(value for value in objects.values()
                  if value.get('isa') == 'PBXNativeTarget' and value.get('name') == name)
    configuration_list = objects[target['buildConfigurationList']]
    release = next(objects[key] for key in configuration_list['buildConfigurations']
                   if objects[key]['name'] == 'Release')
    optimization = str(release['buildSettings'].get('GCC_OPTIMIZATION_LEVEL', ''))
    if optimization != '3':
        raise SystemExit(f'{name} must explicitly preserve Release finite-value semantics')
    print(f'{name}: -O{optimization}')
Path(sys.argv[2]).write_text('-O' + optimization + '\n')
PY
read -r optimization < "${test_dir}/optimization.txt"
compiler="${CXX:-$(/usr/bin/xcrun --find clang++)}"
flags=(-std=c++20 -Wall -Wextra -Werror -pedantic
       -isysroot "$(/usr/bin/xcrun --sdk macosx --show-sdk-path)"
       -I"${repo_root}/Headers" -I"${repo_root}/Sources")
sources=("${repo_root}/Tests/ReleaseFiniteValueValidationTests.cpp"
         "${repo_root}/Sources/InferenceProfitability.cpp"
         "${repo_root}/Sources/StrategyEvaluationCore/StrategyEvaluation.cpp"
         "${repo_root}/Sources/CheckpointPolicy.cpp"
         "${repo_root}/Sources/SchedulerCore/CheckpointEvaluationService.cpp")
# Inputs originate in argv so optimized code cannot prevalidate constant values.
inputs=(7ff8000000000000 7ff0000000000000 fff0000000000000
        7fc00000 7f800000 ff800000)
"${compiler}" "${flags[@]}" "${optimization}" -DNDEBUG \
    "${sources[@]}" -o "${test_dir}/finite-release"
"${test_dir}/finite-release" "${inputs[@]}"

# Negative control: the exact original optimization mode must be caught by
# runtime tests even with assertions disabled. Keep its expected diagnostics.
"${compiler}" "${flags[@]}" -Ofast -DNDEBUG \
    -Wno-error=deprecated-ofast -Wno-error=nan-infinity-disabled \
    "${sources[@]}" -o "${test_dir}/finite-original"
if "${test_dir}/finite-original" "${inputs[@]}" > "${test_dir}/original.log" 2>&1; then
    echo 'Original -Ofast defect was not detected' >&2
    exit 1
fi
cat "${test_dir}/original.log"
rg -Fq 'FAIL TG2 tolerance +infinity' "${test_dir}/original.log"
rg -Fq 'FAIL TG3 tolerance +infinity' "${test_dir}/original.log"
echo 'Original -Ofast finite-value defect detected'

# Existing feature-vector parity oracle compares training and inference rows
# byte-for-byte at all registered widths. Check it with the corrected Release
# policy and a conforming unoptimized reference; keep assertions enabled.
for level in "${optimization}" -O0; do
    "${compiler}" "${flags[@]}" "${level}" \
        "${repo_root}/Tests/LSTMFeatureVectorParityTests.cpp" \
        -o "${test_dir}/feature-parity"
    "${test_dir}/feature-parity" > "${test_dir}/parity${level}.log"
    cat "${test_dir}/parity${level}.log"
done
cmp "${test_dir}/parity${optimization}.log" "${test_dir}/parity-O0.log"

echo 'ReleaseFiniteValueValidationTests passed (layout 13, width 171, feature parity)'
