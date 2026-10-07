#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
main_file="${repo_root}/LSTM/main.cpp"
component_cpp="${repo_root}/Sources/PersistedModelRuntimeConfig.cpp"
component_hpp="${repo_root}/Sources/PersistedModelRuntimeConfig.hpp"

rg -q 'struct TrainConfigMeta' "${component_hpp}"
rg -q 'struct ResumeCheckpointConfig' "${component_hpp}"
! rg -q '^struct TrainConfigMeta' "${main_file}"
! rg -q '^struct ResumeCheckpointConfig' "${main_file}"
! rg -q '^TrainConfigMeta ParseTrainConfigMeta' "${main_file}"
! rg -q '^ResumeCheckpointConfig LoadResumeCheckpointConfig' "${main_file}"
rg -q '^TrainConfigMeta ParseTrainConfigMeta' "${component_cpp}"
rg -q '^ResumeCheckpointConfig LoadResumeCheckpointConfig' "${component_cpp}"
rg -q 'LoadResumeCheckpointConfig\(' "${main_file}"
rg -q 'ParseTrainConfigMeta\(' "${main_file}"
! rg -q 'TrainingWorkerApplication' "${component_cpp}" "${component_hpp}"

printf '%s\n' 'PersistedModelRuntimeConfigStructuralTests passed'
