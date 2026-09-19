# LSTM Phase 22B — Standalone Analyze Worker

## Baseline

Branch: lstm-feature-development
Baseline commit: 5352225

Phase 22A selected the analyze worker as the first standalone worker executable.

## Implementation

Phase 22B adds the standalone executable `lstm-analyze-worker`.

Production composition:

lstm-analyze-worker
  -> RunStandaloneAnalyzeWorkerCli
  -> RegisterSchedulerWorker
  -> AnalyzeExperimentById

The executable uses a thin main in LSTM/AnalyzeWorkerMain.cpp and reuses the existing ExperimentScheduler analysis implementation. Analysis logic was not duplicated.

## CLI Boundary

The standalone worker accepts only the managed scheduler analysis contract:
- --analyze-experiment
- --scheduler-worker-attempt-id
- --auto-generate-reports
- --experiment-report-dir

Value-bearing options accept the same split and equals forms supported by the existing CLI.

An exact scheduler worker-attempt ID is required before managed analysis can execute. Unsupported operator/admin arguments are rejected.

The existing compatibility path remains available:
LSTM_Release --analyze-experiment=...

## Scheduler Routing

Phase 22B does not change production scheduler routing.

ProductionSchedulerDaemon.cpp contains no reference to lstm-analyze-worker. Scheduler selection and reservation of the dedicated executable are deferred to Phase 22C.

## Semantic Worker Publication

Phase 22B does not change semantic-worker registry format, manifests, publication, layout routing, or immutable semantic-worker artifacts.

The standalone analyze executable is not introduced as a semantic inference artifact.

## Files Changed

- LSTM/AnalyzeWorkerMain.cpp
- Headers/ExperimentScheduler.hpp
- Sources/SchedulerCore/ExperimentScheduler.cpp
- ExpertAdvisor.xcodeproj/project.pbxproj
- Tests/StandaloneAnalyzeWorkerCliTests.sh
- this Phase 22B report

## Validation

Focused Release build of lstm-analyze-worker:
- BUILD SUCCEEDED
- BUILD_RC=0
- output: DerivedData/ExpertAdvisor/Build/Products/Release/lstm-analyze-worker

Focused CLI test:
- StandaloneAnalyzeWorkerCliTests passed
- TEST_RC=0

The test verifies dedicated worker existence, admin-command rejection, exact managed-worker-attempt enforcement, the complete scheduler-generated analysis option contract, registration failure exit code 125, split and equals value forms, legacy compatibility behavior, unchanged production scheduler routing, and unchanged semantic-worker artifacts.

git diff --check passed.

## Remaining Prerequisite

Because Phase 22B changes C++ source and Xcode linkage, the normal clean-tree full LSTM Release build remains required after the implementation is reviewed and committed.

Release provenance must not be bypassed while the source tree is dirty.

## Disposition

Phase 22B standalone analyze worker: GO WITH PREREQUISITES
