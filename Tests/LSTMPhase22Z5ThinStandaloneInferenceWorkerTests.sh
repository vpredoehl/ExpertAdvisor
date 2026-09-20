#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
products_dir="${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Debug"
worker="${products_dir}/lstm-infer-worker"
compatibility="${products_dir}/LSTM_Debug"
project_file="${repo_root}/ExpertAdvisor.xcodeproj/project.pbxproj"
application="${repo_root}/Sources/ManagedInferenceApplication.cpp"
adapter="${repo_root}/Sources/ManagedInferenceWorkerCli.cpp"
adapter_header="${repo_root}/Sources/ManagedInferenceWorkerCli.hpp"
worker_main="${repo_root}/LSTM/InferWorkerMain.cpp"
daemon="${repo_root}/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"

test -x "${worker}"
test -x "${compatibility}"
test -f "${adapter}"
test -f "${adapter_header}"

# The target remains a thin executable adapter and cannot compile main.cpp.
target_sources="$(sed -n '/0F8000063800000100AAA001 \/\* Sources \*\//,/runOnlyForDeploymentPostprocessing/p' "${project_file}")"
grep -q 'InferWorkerMain.cpp in Sources' <<<"${target_sources}"
grep -q 'ManagedInferenceWorkerCli.cpp in Sources' <<<"${target_sources}"
grep -q 'ManagedInferenceApplication.cpp in Sources' <<<"${target_sources}"
! grep -q 'main.cpp in Sources' <<<"${target_sources}"
! rg -q 'LSTM/main\.cpp' "${worker_main}" "${adapter}" "${adapter_header}"
rg -q 'RunStandaloneManagedInferenceWorkerCli' "${worker_main}"
rg -q 'RunManagedInferenceWorker\(' "${adapter}" "${adapter_header}"
rg -q 'ParseManagedInferenceWorkerArgs' "${repo_root}/LSTM/main.cpp" "${adapter}" "${adapter_header}"

# Scheduler command construction remains unchanged and receives a selected
# semantic executable rather than a sibling executable chosen by this phase.
final_command="$(sed -n '/std::vector<std::string> BuildInferCommand/,/^}/p' "${daemon}")"
checkpoint_command="$(sed -n '/std::vector<std::string> BuildCheckpointEvalInferCommand/,/^}/p' "${daemon}")"
grep -q 'argv.push_back(selectedWorkerExecutable)' <<<"${final_command}"
grep -q 'argv.push_back(selectedWorkerExecutable)' <<<"${checkpoint_command}"
for option in --infer --model --scheduler-experiment-id --donchian20-mode --feature-warmup-scope --donchian-lookback --log-level; do
    grep -q -- "${option}" <<<"${final_command}"
done
for option in --infer --model --scheduler-checkpoint-eval-id --donchian20-mode --feature-warmup-scope --donchian-lookback --log-level; do
    grep -q -- "${option}" <<<"${checkpoint_command}"
done
rg -q 'scheduler-worker-attempt-id' "${daemon}"

common=(--infer --model=11 --donchian20-mode=enabled --feature-warmup-scope=full_history_warmup --donchian-lookback=20 --log-level=summary 2020-01-01 2020-01-02 --scheduler-worker-attempt-id=33)
final=(--infer --model=11 --scheduler-experiment-id=22 --donchian20-mode=enabled --feature-warmup-scope=full_history_warmup --donchian-lookback=20 --log-level=summary 2020-01-01 2020-01-02 --scheduler-worker-attempt-id=33)
checkpoint=(--infer --model=11 --scheduler-checkpoint-eval-id=23 --donchian20-mode=enabled --feature-warmup-scope=full_history_warmup --donchian-lookback=20 --log-level=summary 2020-01-01 2020-01-02 --scheduler-worker-attempt-id=33)

out="$(mktemp)"
trap 'rm -f "${out}"' EXIT

# Deliberately nonexistent isolated database reaches application-owned
# registration and preserves status 125 for both final and checkpoint forms.
for command in "${final[*]}" "${checkpoint[*]}"; do
    read -r -a argv <<<"${command}"
    set +e
    LSTM_DB_NAME=phase22z5_nonexistent "${worker}" "${argv[@]}" >"${out}" 2>&1
    worker_rc=$?
    LSTM_DB_NAME=phase22z5_nonexistent "${compatibility}" "${argv[@]}" >"${out}.compat" 2>&1
    compatibility_rc=$?
    set -e
    test "${worker_rc}" -eq 125
    test "${compatibility_rc}" -eq 125
    grep -q 'SCHEDULER_WORKER_REGISTRATION_FAILED,worker_attempt_id=33' "${out}"
    grep -q 'SCHEDULER_WORKER_REGISTRATION_FAILED,worker_attempt_id=33' "${out}.compat"
done
rm -f "${out}.compat"

reject() {
    set +e
    "${worker}" "$@" >"${out}" 2>&1
    status=$?
    set -e
    test "${status}" -eq 1
    grep -q 'Argument error:' "${out}"
}

reject --infer --scheduler-experiment-id=22 --donchian20-mode=enabled --feature-warmup-scope=full_history_warmup --donchian-lookback=20 --log-level=summary 2020-01-01 2020-01-02 --scheduler-worker-attempt-id=33
reject --infer --model=11 --scheduler-experiment-id=22 --scheduler-checkpoint-eval-id=23 --donchian20-mode=enabled --feature-warmup-scope=full_history_warmup --donchian-lookback=20 --log-level=summary 2020-01-01 2020-01-02 --scheduler-worker-attempt-id=33
reject "${common[@]}"
reject --infer-all
reject --train
reject --resume-model-id=11
reject --analyze-experiment=22
reject --scheduler-daemon
reject --campaign-status=1
reject --import-economic-events

# Application owns exactly one registration and the detached/persistence
# transaction boundaries remain unchanged.
test "$(rg -c 'RegisterSchedulerWorker' "${application}")" -eq 1
! rg -q 'RegisterSchedulerWorker' "${adapter}" "${worker_main}"
rg -U -q 'REPEATABLE READ, READ ONLY;[\s\S]{0,16000}read\.commit\(\);[\s\S]{0,8000}PrepareInferenceInput\([\s\S]{0,8000}RunInferenceRuntime\(' "${application}"
rg -q 'AcquireCoordinationLock' "${application}"
rg -U -q 'PersistResult\([\s\S]{0,1600}PersistProfitability\([\s\S]{0,500}write\.commit\(\)' "${application}"
rg -q 'ON CONFLICT \(checkpoint_eval_id\)' "${application}"
rg -q 'ON CONFLICT \(model_id,symbol' "${application}"

# Phase 23A may publish the worker and add role-aware registry selection, but
# the Phase 22Z5 worker still has no scheduler lifecycle implementation.
rg -q 'selectInferenceWorker' "${repo_root}/Sources/SchedulerCore/SemanticWorkerRegistry.hpp"
! rg -q 'RegisterSchedulerWorker|PrepareInferenceInput|RunInferenceRuntime' \
    "${repo_root}/LSTM/InferWorkerMain.cpp"

"${worker}" --build-identity >"${out}"
grep -q 'artifact_role=lstm-infer-worker' "${out}"
grep -q 'canonical_executable=' "${out}"
grep -q 'executable_sha256=sha256:' "${out}"

printf '%s\n' 'LSTMPhase22Z5ThinStandaloneInferenceWorkerTests passed'
