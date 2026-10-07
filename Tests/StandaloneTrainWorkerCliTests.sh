#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
products_dir="${1:-${repo_root}/DerivedData/TrainWorkerExtraction/Build/Products/Debug}"
worker="${products_dir}/lstm-train-worker"

test -x "${worker}"

identity="$("${worker}" --build-identity)"
[[ "${identity}" == TRAIN_WORKER_BUILD_IDENTITY,* ]]
[[ "${identity}" == *",artifact_role=lstm-train-worker,"* ]]

output="$(mktemp)"
trap 'rm -f "${output}"' EXIT

if "${worker}" --train 2010-01-01 2010-01-02 >"${output}" 2>&1; then
    exit 1
fi
grep -q 'lstm-train-worker requires scheduler-managed --train work' "${output}"

if "${worker}" --infer --scheduler-experiment-id=1 \
    --scheduler-worker-attempt-id=1 2010-01-01 2010-01-02 >"${output}" 2>&1; then
    exit 1
fi
grep -q 'scheduler-managed final inference requires one explicit --model' "${output}"

# The dedicated entry is a standalone process boundary and invokes the
# dedicated managed-TRAIN application. Legacy LSTM_Release retains its
# independent compatibility implementation.
rg -q 'RunTrainingWorkerApplication\(argc, argv\)' \
    "${repo_root}/LSTM/TrainWorkerMain.cpp"
rg -q 'RunLegacyTrainingWorkerApplication\(argc, argv, false\)' \
    "${repo_root}/LSTM/main.cpp"
rg -q 'launchArgs = EA::ParseLaunchArgs\(argc, argv\)' \
    "${repo_root}/Sources/TrainingWorkerApplication.cpp"
rg -q 'RegisterSchedulerWorker' \
    "${repo_root}/Sources/TrainingWorkerApplication.cpp"
rg -q 'LoadSchedulerFeatureAblationMask' \
    "${repo_root}/Sources/PersistedModelRuntimeConfig.cpp"

printf '%s\n' 'StandaloneTrainWorkerCliTests passed'
