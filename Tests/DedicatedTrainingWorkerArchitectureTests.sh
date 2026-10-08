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

# Resolve target-owned phases by ID. Provenance must precede TRAIN compilation;
# neither phase has to be first, and object/comment ordering is irrelevant.
python3 - "$project" <<'PY'
import json
from pathlib import PurePosixPath
import subprocess
import sys

objects = json.loads(subprocess.check_output(
    ["/usr/bin/plutil", "-convert", "json", "-o", "-", "--", sys.argv[1]]
))["objects"]
train_id = "0FA000043A00000100AAA001"
source_id = "0FAC00003A00000100AAA001"
provenance_id = "0FA000083A00000100AAA001"
release_id = "0A10000F2F70000100AAA001"
release_source_id = "0A10000E2F70000100AAA001"


def require(condition, message):
    if not condition:
        sys.exit("TRAIN architecture: " + message)


train = objects[train_id]
release = objects[release_id]
require(train["isa"] == release["isa"] == "PBXNativeTarget",
        "worker and Release must be native targets")
phases = train["buildPhases"]
require(phases.count(source_id) == 1, "TRAIN must own its Sources phase")
require(phases.count(provenance_id) == 1, "TRAIN must own its provenance phase")
require(objects[source_id]["isa"] == "PBXSourcesBuildPhase",
        "TRAIN Sources must be a source phase")
require(objects[provenance_id]["isa"] == "PBXShellScriptBuildPhase",
        "TRAIN provenance must be a shell-script phase")
require(phases.index(provenance_id) < phases.index(source_id),
        "TRAIN provenance must precede Sources")
require([p for p in phases if objects[p]["isa"] == "PBXSourcesBuildPhase"]
        == [source_id], "TRAIN must have only its dedicated Sources phase")
require(release["buildPhases"].count(release_source_id) == 1
        and objects[release_source_id]["isa"] == "PBXSourcesBuildPhase",
        "Release must retain its Sources phase")
for target_id, target in objects.items():
    if target_id != train_id and target.get("isa") == "PBXNativeTarget":
        require(not {source_id, provenance_id}.intersection(target["buildPhases"]),
                "TRAIN phases must not be shared with another target")

sources = [PurePosixPath(objects[objects[f]["fileRef"]]["path"]).name
           for f in objects[source_id]["files"]]
for required in ("TrainWorkerMain.cpp", "TrainingWorkerApplication.cpp"):
    require(sources.count(required) == 1, "TRAIN must compile exactly one " + required)
for forbidden in ("main.cpp", "CheckpointModelPersistence.cpp",
                  "InferenceRuntime.cpp", "ManagedInferenceApplication.cpp"):
    require(forbidden not in sources, "TRAIN must not compile " + forbidden)
PY

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
