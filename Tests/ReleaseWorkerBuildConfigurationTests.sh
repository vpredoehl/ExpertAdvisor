#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project="${repo_root}/ExpertAdvisor.xcodeproj/project.pbxproj"
analyze_scheme="${repo_root}/ExpertAdvisor.xcodeproj/xcshareddata/xcschemes/LSTM Analyze Worker.xcscheme"

# Release map files must be unique per linker architecture invocation. Xcode
# links universal command-line tools once per architecture and otherwise
# reports duplicate map-file producers before it can emit the executable.
for target in lstm-scheduler lstm-analyze-worker lstm-infer-worker
do
    rg -Fq "LD_MAP_FILE_PATH = \"\$(TARGET_TEMP_DIR)/\$(CURRENT_ARCH)/${target}.map\";" \
        "${project}"
done

# The normal stable DerivedData Release build is also the deployment build for
# the default scheduler analyzer. Keep that dependency explicit; analysis is
# intentionally not selected through the semantic inference-worker registry.
rg -U -q '0A10000F2F70000100AAA001 /\* LSTM Release \*/ = \{[\s\S]*?dependencies = \([\s\S]*?0F6000203600000100AAA001 /\* PBXTargetDependency \*/' \
    "${project}"
rg -U -q '0F6000203600000100AAA001 /\* PBXTargetDependency \*/ = \{[\s\S]*?name = "lstm-analyze-worker";[\s\S]*?target = 0F6000143600000100AAA001 /\* lstm-analyze-worker \*/;' \
    "${project}"
rg -Fq 'BlueprintIdentifier="0F6000143600000100AAA001"' "${analyze_scheme}"
rg -Fq 'BuildableName="lstm-analyze-worker"' "${analyze_scheme}"

printf '%s\n' 'ReleaseWorkerBuildConfigurationTests passed'
