---
title: "LSTM Historical Training Semantic Worker Selection Regression"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_HistoricalTrainingSemanticWorkerSelection_Regression_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Historical Training Semantic Worker Selection Regression

Implemented the scheduler architecture fix. TRAIN admission, preflight, reservation, provenance, and launch now use the same exact registry-selected semantic worker.

1. Root cause confirmed

TRAIN loaded the authoritative persisted identity for admission but independently defaulted executable selection and command construction to `options.currentWorkerExecutablePath`. This allowed identity A—layout 8/width 80—to authorize executable B—layout 9/width 103.

2. Files changed

- [ProductionSchedulerDaemon.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ProductionSchedulerDaemon.cpp:1930>)
- [SemanticWorkerRegistry.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/SemanticWorkerRegistry.cpp:950>)
- [TrainingWorkerSelection.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/TrainingWorkerSelection.hpp:1>)
- [SemanticWorkerRegistryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/SemanticWorkerRegistryTests.cpp:408>)
- [SemanticWorkerRegistryTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/SemanticWorkerRegistryTests.sh:11>)
- [WorkerAttemptLifecycleServiceTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/WorkerAttemptLifecycleServiceTests.cpp:166>)
- [SchedulerTrainingWorkerRoutingTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerTrainingWorkerRoutingTests.sh:1>)
- [semantic-layout-inference-worker-routing.md](</Volumes/Developer SSD/ExpertAdvisor/docs/semantic-workers/semantic-layout-inference-worker-routing.md:72>)

3. Train-selection architecture

`TrainingWorkerSelection.hpp` provides the role-specific selection seam and delegates directly to:

```cpp
registry.selectTrainingReferenceWorker(persisted)
```

`LoadTrainingWorkerSelection()` first uses the existing `LoadSemanticWorkerAdmission(..., "train")` identity-loading path, preserving explicit persisted identity as authoritative. It then selects the exact registered TRAIN artifact without fallback or compatibility coercion.

Fresh experiments with both identity fields null retain existing legacy behavior: select the current published training worker without manufacturing an identity. Expected-but-missing, incomplete, unsupported, or mismatched identities fail closed.

4. Shared preflight and reservation semantics

Both paths call `LoadTrainingWorkerSelection()`:

- Preflight validates the selected artifact’s runtime.
- Reservation validates that same selected runtime before capacity consumption.
- Reservation resolves provenance from the selected registry artifact.
- `BuildTrainCommand()` receives the selected or reserved canonical executable.
- Actual launch uses `reservedAttempt->canonicalExecutablePath`.

Stopped TRAIN attempts are also protected: a retained worker whose executable differs from the exact selection is not sent `SIGCONT`; admission is deferred with `semantic_worker_selection_mismatch`.

`SCHEDULER_TRAIN_WORKER_SELECTED` now reports persisted and selected width/layout, canonical executable, experiment ID, and reason.

5. Historical 8/80 regression

Passed using the operational immutable registry:

- Persisted experiment identity: layout 8 / width 80.
- Current worker: layout 9 / width 103.
- Selected worker: historical TRAIN layout 8 / width 80.
- Reason: `immutable_historical_semantic_worker`.
- Historical executable differs from the current executable.
- Routing regression confirms the current 9/103 executable cannot become TRAIN launch argv.

6. Current 9/103 regression

Passed:

- Persisted identity: layout 9 / width 103.
- Selected worker: current TRAIN layout 9 / width 103.
- Reason: `current_published_semantic_worker`.

7. Negative/rejection coverage

Passed coverage for:

- Fresh null/null identity retaining legacy-current behavior.
- Expected-but-missing identity rejection.
- Width-only and layout-only identity rejection.
- Unsupported historical TRAIN layout rejection.
- Width/layout mismatch rejection.
- Missing train capability rejection.
- Selected runtime becoming unavailable after registry loading.
- Runtime validation occurring before capacity consumption.
- Mismatched stopped TRAIN worker not being resumed.

8. Inference and analyze results

Inference layout-8 and layout-9 selection assertions remain green and independently select INFER-role artifacts. Existing inference loading was not changed.

Analyze routing remains on `analyzeWorkerExecutablePath`; its focused regression passed.

9. Provenance

Reservation now copies these fields directly from the selected artifact:

- semantic layout version
- model input width
- semantic worker role
- source commit
- executable SHA-256
- runtime identity
- canonical manifest path
- canonical executable

The lifecycle service regression verifies those exact values are forwarded into the worker-attempt reservation request.

10. Tests and Debug build

All passed:

```text
bash Tests/SchedulerDaemonConfigurationTests.sh
bash Tests/FinalExperimentDispatchServiceTests.sh
bash Tests/SemanticWorkerRegistryTests.sh
bash Tests/SchedulerSemanticAdmissionTests.sh
bash Tests/WorkerAttemptLifecycleServiceTests.sh
bash Tests/SchedulerTrainingWorkerRoutingTests.sh
bash Tests/SchedulerAnalyzeWorkerRoutingTests.sh
```

Debug build:

```bash
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Debug" \
  -configuration Debug \
  -derivedDataPath DerivedData/ExpertAdvisor \
  build
```

Result: `** BUILD SUCCEEDED **`.

The build retained existing environmental warnings concerning the LLVM23 Info.plist and a libomp macOS deployment-version mismatch. No warning attributable to these changes was introduced.

An earlier Debug-configuration invocation using the `LSTM Release` scheme failed at its `lstm-analyze-worker` dependency because of pre-existing unresolved replication/comparison symbols. No Release-configuration build was performed; the repository’s established `LSTM Debug` target succeeded.

11. Diff checks

`git diff --check` passed.

The two new untracked files were also checked separately with `git diff --no-index --check`; both passed.

12. Safety confirmation

No live database queries or writes were performed. No experiment rows were modified. No scheduler or worker was started, stopped, restarted, resumed, or signaled. Experiments 668/669 and attempts 1196/1197 were not touched.

Only read-only OS process inspection was used to confirm the scheduler and stopped workers remained present before and after development.

13. Recommended manual acceptance for 668/669 — not executed

Perform this in an approved maintenance window, one experiment at a time:

1. Keep 668/669 paused. Gracefully stop the production scheduler; do not signal either worker yet.
2. Deploy an approved, provenance-valid build containing this fix. Verify the registry contains a TRAIN artifact at layout 8/width 80 and that its executable, manifest, hashes, and runtime validate.
3. Read-only verify the experiment still has layout 8/width 80, `current_epoch=NULL`, and `last_model_id=NULL`. Verify attempts 1196/1197 remain stopped and identify the old layout-9/103 executable.
4. Run the individual resume dry-run:

   ```bash
   <fixed-LSTM_Release> --resume-experiment=668 --dry-run
   ```

5. Queue the resume while the scheduler is stopped:

   ```bash
   <fixed-LSTM_Release> --resume-experiment=668 --yes
   ```

   This queues admission and sends no signal.

6. As the sole scheduler authority, run one dry scheduler cycle with the exact production registry and ordinary production options:

   ```bash
   <fixed-LSTM_Release> \
     --schedule-experiments \
     --scheduler-once \
     --dry-run \
     --semantic-worker-registry=/absolute/path/to/registry.json \
     <the remaining production scheduler options>
   ```

7. Do not proceed unless output proves all of the following:

   ```text
   SCHEDULER_TRAIN_WORKER_SELECTED
   experiment_id=668
   model_input_width=80
   model_input_semantic_layout_version=8
   worker_input_width=80
   worker_semantic_layout_version=8
   reason=immutable_historical_semantic_worker
   ```

   The emitted TRAIN argv must begin with the immutable historical layout-8 TRAIN executable, never the layout-9/103 executable.

8. Retire the old stopped layout-9 process using an approved identity-aware operational procedure. It must verify the exact attempt, PID, process group, process-start identity, executable, and command before signaling. A stopped group must be continued before termination, with identity revalidated before any escalation. Do not use direct SQL or an unvalidated `kill`.
9. Positively verify the old process group is absent. Start the fixed scheduler. Its authoritative missing-stopped-worker workflow should abandon and detach the old attempt while preserving the queued resume, then reserve a new attempt.
10. Before accepting continued training, verify the new attempt’s canonical executable and every provenance field exactly match the registered layout-8/80 TRAIN artifact. Also verify the native process executable is that same canonical path and no layout-9/103 TRAIN process was launched.
11. Repeat the full procedure independently for experiment 669.

If the old stopped process cannot be retired through an approved identity-safe procedure, leave the experiment deferred and paused; do not bypass the workflow with SQL.

Repository state:

```text
 M Sources/SchedulerCore/ProductionSchedulerDaemon.cpp
 M Sources/SchedulerCore/SemanticWorkerRegistry.cpp
 M Tests/SemanticWorkerRegistryTests.cpp
 M Tests/SemanticWorkerRegistryTests.sh
 M Tests/WorkerAttemptLifecycleServiceTests.cpp
 M docs/semantic-workers/semantic-layout-inference-worker-routing.md
?? Sources/SchedulerCore/TrainingWorkerSelection.hpp
?? Tests/SchedulerTrainingWorkerRoutingTests.sh
```

`git diff --stat`:

```text
 .../SchedulerCore/ProductionSchedulerDaemon.cpp    | 138 ++++++++++++++++----
 Sources/SchedulerCore/SemanticWorkerRegistry.cpp   |  37 ++++++
 Tests/SemanticWorkerRegistryTests.cpp              | 141 ++++++++++++++++++++-
 Tests/SemanticWorkerRegistryTests.sh               |   3 +-
 Tests/WorkerAttemptLifecycleServiceTests.cpp       |  19 ++-
 .../semantic-layout-inference-worker-routing.md    |  45 +++++--
 6 files changed, 345 insertions(+), 38 deletions(-)
```

The stat excludes the two untracked additions: 17 lines in `TrainingWorkerSelection.hpp` and 40 lines in `SchedulerTrainingWorkerRoutingTests.sh`.