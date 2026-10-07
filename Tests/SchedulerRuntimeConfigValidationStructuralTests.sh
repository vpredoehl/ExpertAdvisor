#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
main="${root}/LSTM/main.cpp"
cpp="${root}/Sources/SchedulerRuntimeConfigValidation.cpp"
hpp="${root}/Sources/SchedulerRuntimeConfigValidation.hpp"
test -f "${cpp}"; test -f "${hpp}"
for symbol in LoadExperimentDonchian20Mode LoadExperimentFeatureWarmupScope LoadExperimentDonchianLookback ValidateSchedulerDonchian20Mode ValidateSchedulerFeatureWarmupScope ValidateSchedulerDonchianLookback; do
  test "$(rg -c "^.*${symbol}\(" "${cpp}")" -ge 1
  ! rg -q "^.*${symbol}\(" "${main}"
done
rg -q 'SELECT donchian20_mode FROM experiment WHERE experiment_id=\$1;' "${cpp}"
rg -q 'SELECT feature_warmup_scope FROM experiment WHERE experiment_id=\$1;' "${cpp}"
rg -q 'SELECT donchian_lookback FROM experiment WHERE experiment_id=\$1;' "${cpp}"
rg -q 'if \(!experimentId.has_value\(\)\)' "${cpp}"
rg -q 'SchedulerRuntimeConfigValidation.hpp' "${main}"
! rg -q 'TrainingWorkerApplication|gRuntimeInferenceMode|std::function|callback' "${cpp}" "${hpp}"
printf '%s\n' 'SchedulerRuntimeConfigValidationStructuralTests passed'
