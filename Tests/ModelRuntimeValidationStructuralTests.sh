#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
main_file="${repo_root}/LSTM/main.cpp"
component_cpp="${repo_root}/Sources/ModelRuntimeValidation.cpp"
component_hpp="${repo_root}/Sources/ModelRuntimeValidation.hpp"

test -f "${component_cpp}"
test -f "${component_hpp}"
rg -q 'struct ModelConfigValidationResult' "${component_hpp}"
rg -q 'PrintModelConfigValidation\(' "${component_cpp}"
rg -q 'PrintMaterializedModelConfigValidation\(' "${component_cpp}"
rg -q 'ValidateLoadedModelSymbolForSelectedTable\(' "${component_cpp}"
! rg -q '^ModelConfigValidationResult PrintModelConfigValidation\(' "${main_file}"
! rg -q '^ModelConfigValidationResult PrintMaterializedModelConfigValidation\(' "${main_file}"
! rg -q '^void ValidateLoadedModelSymbolForSelectedTable\(' "${main_file}"
rg -q 'ModelRuntimeValidation.hpp' "${main_file}"
rg -q 'using EA::ModelRuntimeValidation::PrintModelConfigValidation' "${main_file}"
! rg -q 'TrainingWorkerApplication' "${component_cpp}" "${component_hpp}"
rg -q 'ModelRuntimeValidation.cpp in Sources' "${repo_root}/ExpertAdvisor.xcodeproj/project.pbxproj"

printf '%s\n' 'ModelRuntimeValidationStructuralTests passed'
