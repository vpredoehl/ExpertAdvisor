#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
main="${root}/LSTM/main.cpp"
cpp="${root}/Sources/LstmRuntimeConstruction.cpp"
hpp="${root}/Sources/LstmRuntimeConstruction.hpp"
test -f "${cpp}"; test -f "${hpp}"
rg -q 'LSTM CreateLstmForRuntimeLogLevel\(const ::Tensor&' "${hpp}" "${cpp}"
! rg -q '^EA::LSTM CreateLstmForRuntimeLogLevel\(' "${main}"
rg -q 'EA::LstmRuntimeConstruction::CreateLstmForRuntimeLogLevel' "${main}"
rg -q 'RuntimeLogging.hpp' "${cpp}"
! rg -q 'TrainingWorkerApplication|gRuntimeInferenceMode|pqxx|ApplyPersisted|loadOptimizer|evaluationFacts|RunInferenceRuntime|scheduler' "${cpp}" "${hpp}"
printf '%s\n' 'LstmRuntimeConstructionStructuralTests passed'
