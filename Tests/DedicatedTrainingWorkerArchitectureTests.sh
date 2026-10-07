#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
project="$root/ExpertAdvisor.xcodeproj/project.pbxproj"
legacy="$root/LSTM/main.cpp"
front="$root/LSTM/TrainWorkerMain.cpp"
application="$root/Sources/TrainingWorkerApplication.cpp"
header="$root/Sources/TrainingWorkerApplication.hpp"

test -f "$front"
test -f "$application"
test -f "$header"

# Each executable root and the dedicated application are ordinary translation
# units. No source file obtains another entry point by textual inclusion.
! rg -q '#include[[:space:]]+[<"][^>"]+\.cpp[>"]' "$root/LSTM" "$root/Sources"
rg -q '^int main\(int argc, const char\* argv\[\]\)' "$front"
rg -q 'RunTrainingWorkerApplication\(argc, argv\)' "$front"
rg -q 'int EA::Training::RunTrainingWorkerApplication\(int argc, const char\* argv\[\]\)' "$application"

# Train Worker owns a distinct source phase with its own root and application.
rg -U -q '0FA000043A00000100AAA001 /\* lstm-train-worker \*/ = \{[\s\S]*?buildPhases = \([[:space:]]*0FAC00003A00000100AAA001 /\* Sources \*/' "$project"
rg -U -q '0A10000F2F70000100AAA001 /\* LSTM Release \*/ = \{[\s\S]*?buildPhases = \([\s\S]*?0A10000E2F70000100AAA001 /\* Sources \*/' "$project"
train_phase="$(sed -n '/0FAC00003A00000100AAA001 \/\* Sources \*\/ = {/,/^[[:space:]]*};/p' "$project")"
grep -q 'TrainWorkerMain.cpp in Sources' <<<"$train_phase"
grep -q 'TrainingWorkerApplication.cpp in Sources' <<<"$train_phase"
! grep -q 'main.cpp in Sources' <<<"$train_phase"
! grep -q 'CheckpointModelPersistence.cpp' <<<"$train_phase"

# The dedicated application is managed TRAIN only.
for forbidden in RunInferenceRuntime RunInferenceEvaluation evaluationFacts \
    schedulerInferenceContext PersistCompletedInferenceResult \
    PersistCompletedCheckpointInferenceResult; do
    ! rg -q "$forbidden" "$application"
done
rg -q 'lstm-train-worker requires scheduler-managed --train work' "$application"
rg -q '!launchArgs.schedulerExperimentId.has_value\(\)' "$application"
rg -q '!launchArgs.schedulerWorkerAttemptId.has_value\(\)' "$application"
rg -q 'launchArgs.schedulerCheckpointEvalId.has_value\(\)' "$application"
rg -q 'launchArgs.inferenceMode.value_or\(true\)' "$application"

# The closed shared foundation remains the dedicated path's dependency seam.
for required in LaunchRuntimeConfig RuntimeEconomicCalendarIdentity \
    SchedulerWorkerOwnershipTestBoundary TrainingRuntimeConfig \
    SchedulerRuntimeConfigValidation TrainingWorkerFeatureAblation \
    PersistedModelRuntimeConfig ModelRuntimeValidation \
    LstmRuntimeConstruction RuntimeLogging RuntimeDatabaseConnection \
    LstmHotspotProfileFinalizer CheckpointTrainingControl; do
    rg -q "$required" "$application"
done

# TRAIN behavior remains represented: resume/load/validation, checkpoints,
# final persistence, lineage, and producer-attempt binding.
for required in LoadResumeCheckpointConfig ApplyResumeRuntimeConfig \
    LoadSchedulerFeatureAblationMask ValidateSchedulerResumeFeatureAblationMask \
    SavePeriodicCheckpointIfDue QueueCheckpointInferenceIfEligible \
    LoadCheckpointStopConfig RecordCheckpointStopReached \
    bindProducerWorkerAttempt LinkModelToSchedulerExperimentIfPresent \
    LinkModelParentIfPresent; do
    rg -q "$required" "$application"
done
rg -q 'runtimeDatabaseWork->commit\(\)' "$application"
rg -q 'DBIO::PgModelIO::saveAll' "$application"

# Strategy B deliberately leaves the legacy compatibility TRAIN body intact.
rg -q 'int RunLegacyTrainingWorkerApplication\(' "$legacy"
rg -q 'SavePeriodicCheckpointIfDue' "$legacy"
rg -q 'RunInferenceRuntime' "$legacy"
rg -q 'RunLegacyTrainingWorkerApplication\(argc, argv, false\)' "$legacy"

printf '%s\n' 'DedicatedTrainingWorkerArchitectureTests passed'
