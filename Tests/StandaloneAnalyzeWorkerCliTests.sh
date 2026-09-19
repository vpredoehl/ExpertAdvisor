#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
products_dir="${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release"
worker="${products_dir}/lstm-analyze-worker"
legacy="${products_dir}/LSTM_Release"

test -x "${worker}"
test -x "${legacy}"

worker_output="$(mktemp)"
legacy_output="$(mktemp)"
trap 'rm -f "${worker_output}" "${legacy_output}"' EXIT

if "${worker}" --pause-all >"${worker_output}" 2>&1; then
    exit 1
fi
grep -q "unsupported analyze worker option '--pause-all'" "${worker_output}"

if "${worker}" --analyze-experiment=1 >"${worker_output}" 2>&1; then
    exit 1
fi
grep -q 'DIRECT_CLI_MANAGED_WORK_REJECTED,operation=analyze,experiment_id=1,reason=exact_worker_attempt_required' "${worker_output}"

# Verify every option emitted by BuildAnalyzeCommand is accepted by the
# standalone worker parser. Registration is expected to fail against the
# deliberately nonexistent database, but argument parsing must succeed.
set +e
LSTM_DB_NAME=phase22b_nonexistent \
    "${worker}" \
    --analyze-experiment=1 \
    --scheduler-worker-attempt-id=1 \
    --auto-generate-reports \
    --experiment-report-dir=phase22b_reports \
    >"${worker_output}" 2>&1
registration_rc=$?
set -e
test "${registration_rc}" -eq 125
grep -q 'SCHEDULER_WORKER_REGISTRATION_FAILED,worker_attempt_id=1' "${worker_output}"
if grep -q 'unsupported analyze worker option' "${worker_output}"; then
    exit 1
fi

# Verify split-value forms accepted by the existing CLI contract as well.
set +e
LSTM_DB_NAME=phase22b_nonexistent \
    "${worker}" \
    --analyze-experiment 1 \
    --scheduler-worker-attempt-id 1 \
    --auto-generate-reports \
    --experiment-report-dir phase22b_reports \
    >"${worker_output}" 2>&1
split_registration_rc=$?
set -e
test "${split_registration_rc}" -eq 125
grep -q 'SCHEDULER_WORKER_REGISTRATION_FAILED,worker_attempt_id=1' "${worker_output}"
if grep -q 'unsupported analyze worker option' "${worker_output}"; then
    exit 1
fi

if "${legacy}" --analyze-experiment=1 >"${legacy_output}" 2>&1; then
    exit 1
fi
grep -q 'DIRECT_CLI_MANAGED_WORK_REJECTED,operation=analyze,experiment_id=1,reason=exact_worker_attempt_required' "${legacy_output}"

# Keep this list synchronized with ProductionSchedulerDaemon::BuildAnalyzeCommand.
for option in \
    '--analyze-experiment' \
    '--auto-generate-reports' \
    '--experiment-report-dir'
do
    rg -q -- "${option}" \
        "${repo_root}/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"
    rg -q -- "${option}" \
        "${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"
done
rg -q -- '--scheduler-worker-attempt-id' \
    "${repo_root}/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"
rg -q -- '--scheduler-worker-attempt-id' \
    "${repo_root}/Sources/SchedulerCore/ExperimentScheduler.cpp"

rg -q -- 'analyzeWorkerExecutablePath' \
    "${repo_root}/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"
rg -q -- 'lstm-analyze-worker' \
    "${repo_root}/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"

if git -C "${repo_root}" diff --name-only | rg -q '^Builds/SemanticWorkers/'; then
    echo "semantic-worker artifacts were unexpectedly modified" >&2
    exit 1
fi

printf '%s\n' 'StandaloneAnalyzeWorkerCliTests passed'
