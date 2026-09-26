#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
daemon="${repo_root}/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"
selection="${repo_root}/Sources/SchedulerCore/TrainingWorkerSelection.hpp"
command="${repo_root}/Sources/SchedulerCore/TrainingWorkerCommand.hpp"

test -f "${selection}"
test -f "${command}"
rg -q 'registry\.selectTrainingReferenceWorker\(persisted\)' "${selection}"

# Both preflight and reservation resolve TRAIN through the same loader.  The
# loader itself must retain the existing authoritative admission/identity path.
test "$(rg -c 'LoadTrainingWorkerSelection\(' "${daemon}")" -ge 4
rg -U -q 'LoadTrainingWorkerSelection[\s\S]{0,900}LoadSemanticWorkerAdmission\([\s\S]{0,200}"train"[\s\S]{0,500}SelectTrainingWorker' "${daemon}"
rg -U -q 'SemanticWorkerPreflight\([\s\S]{0,1800}phase == "train"[\s\S]{0,400}LoadTrainingWorkerSelection[\s\S]{0,700}validateRuntimeForExecutable\(executable\)' "${daemon}"
rg -U -q 'ReserveExperimentWorkerAttempt\([\s\S]{0,1800}phase == "train"[\s\S]{0,400}LoadTrainingWorkerSelection[\s\S]{0,1400}validateRuntimeForExecutable\(selectedWorkerExecutable\)[\s\S]{0,1400}findByCanonicalExecutable\(selectedWorkerExecutable\)' "${daemon}"
rg -U -q 'validateRuntimeForExecutable\(selectedWorkerExecutable\)[\s\S]{0,1000}hasCapacity\(phase, maximumCapacity\)' "${daemon}"

# Attempt provenance is copied from the selected registry artifact, not rebuilt
# from its path or from the current-worker options.
rg -U -q 'WorkerAttemptLifecycleService lifecycle[\s\S]{0,1600}selectedWorkerExecutable,[\s\S]{0,200}artifact->semanticLayoutVersion[\s\S]{0,200}artifact->modelInputWidth[\s\S]{0,400}artifact->sourceCommit[\s\S]{0,200}artifact->sha256[\s\S]{0,200}artifact->runtimeIdentity[\s\S]{0,200}artifact->canonicalManifestPath' "${daemon}"

# The reserved canonical executable is the launch argv executable.  This is
# the assertion that fails under the historical current-worker implementation.
rg -U -q 'BuildTrainCommand\([\s\S]{0,700}BeginTrainingWorkerCommand\([\s\S]{0,200}selectedWorkerExecutable' "${daemon}"
rg -U -q 'phase == FinalExperimentPhase::Train[\s\S]{0,300}BuildTrainCommand\([\s\S]{0,200}reservedAttempt->canonicalExecutablePath' "${daemon}"
if rg -U -q 'BuildTrainCommand\([\s\S]{0,700}argv\.push_back\(options\.currentWorkerExecutablePath\)' "${daemon}"; then
    echo "train launch still routes through the current worker" >&2
    exit 1
fi

# The selected executable and canonical persisted ablation identity initialize
# one shared argv before the resume branch. Empty masks are omitted by the
# command helper; nonempty masks use the training CLI's exact equals form.
rg -U -q 'BeginTrainingWorkerCommand\([\s\S]{0,200}selectedWorkerExecutable,[\s\S]{0,100}experiment\.featureAblationMask' "${daemon}"
rg -U -q 'BeginTrainingWorkerCommand\([\s\S]{0,1200}const std::optional<long long> resumeFrom[\s\S]{0,500}return argv' "${daemon}"
rg -U -q 'if \(!canonicalFeatureAblationMask\.empty\(\)\)[\s\S]{0,200}"--ablate-features=" \+ canonicalFeatureAblationMask' "${command}"
if rg -q -- '"--ablate-features=" \+ experiment\.' "${daemon}"; then
    echo "train ablation propagation bypasses the shared command prefix" >&2
    exit 1
fi

rg -q 'SCHEDULER_TRAIN_WORKER_SELECTED' "${daemon}"
rg -q 'worker_input_width=' "${daemon}"
rg -U -q 'AdmitStoppedExperimentWorker\([\s\S]{0,5000}LoadTrainingWorkerSelection[\s\S]{0,700}semantic_worker_selection_mismatch' "${daemon}"

# ANALYZE remains independently routed and INFER keeps its established loader.
rg -q 'options\.analyzeWorkerExecutablePath' "${daemon}"
rg -q 'LoadInferenceWorkerSelection' "${daemon}"

printf '%s\n' "SchedulerTrainingWorkerRoutingTests passed"
