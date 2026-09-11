# LSTM Phase 20E — Scheduler Remaining Orchestration Decomposition Output

## Summary

Phase 20E completed successfully.

Starting commit: `f9151f3 Extract scheduler authority service`

Implementation commit before this report is amended into it: `a4a9a76 Extract scheduler continuation orchestration service`

Phase 20E extracted continuation orchestration from `Sources/SchedulerCore/ExperimentScheduler.cpp` into a new `ContinuationOrchestrationService`. It added focused tests, updated continuation policy-isolation coverage to follow the moved implementation, integrated the new source into the SchedulerCore Xcode target, and preserved the existing scheduler executable architecture.

No standalone scheduler executable was introduced.

## Selected Responsibility

Continuation orchestration was selected because candidate discovery, ranking, automatic continuation evaluation, queue decisions, scan limits, dry-run behavior, and related continuation status handling formed a coherent scheduler-domain cluster.

The phase intentionally stopped after this one substantive extraction.

## Files Added

- `Sources/SchedulerCore/ContinuationOrchestrationService.cpp`
- `Sources/SchedulerCore/ContinuationOrchestrationService.hpp`
- `Tests/ContinuationOrchestrationServiceTests.cpp`
- `Tests/ContinuationOrchestrationServiceTests.sh`

## Files Modified

- `ExpertAdvisor.xcodeproj/project.pbxproj`
- `Sources/SchedulerCore/ExperimentScheduler.cpp`
- `Tests/ContinuationProfitabilityPolicyIsolationTests.py`

## Architecture

Before:

```text
SchedulerEngine
  +--> SchedulerAdmissionService
  +--> WorkerControlService
  +--> ReconciliationService
  +--> SchedulerAuthorityService
  +--> SchedulerRepository
  +--> WorkerProcessController
  +--> SchedulerPolicy
```

After:

```text
SchedulerEngine
  +--> SchedulerAdmissionService
  +--> WorkerControlService
  +--> ReconciliationService
  +--> SchedulerAuthorityService
  +--> ContinuationOrchestrationService
  +--> SchedulerRepository
  +--> WorkerProcessController
  +--> SchedulerPolicy
```

`ContinuationOrchestrationService` owns real continuation-domain behavior rather than acting as a wrapper.

## Compatibility

This was a structural refactor. It did not intentionally change experiment state semantics, priority/preemption, pause/resume, capacity accounting, worker-attempt identity, reconciliation/orphan recovery, authority/lease/fencing, database schema, worker launch semantics, CLI behavior, model/training/inference behavior, feature semantics, profitability semantics, or strategy semantics.

`Tests/ContinuationProfitabilityPolicyIsolationTests.py` was updated because source-level assertions needed to follow continuation behavior into the new service rather than assume it remained physically in `ExperimentScheduler.cpp`.

## Tests

The Phase 20E run completed focused and regression scheduler testing before exhausting its usage allowance. The tested set included:

- `ContinuationOrchestrationServiceTests`
- `SchedulerAuthorityServiceTests`
- `SchedulerCoreBoundaryTests`
- `SchedulerInternalSeamTests`
- `SchedulerOrchestrationServiceTests`
- `PostgresSchedulerRepositoryTests`
- `SchedulerSemanticAdmissionTests`
- continuation profitability/policy isolation coverage
- direct scheduler ownership-policy testing
- direct scheduler child-status testing

The new continuation test shell script is executable.

`git diff --check` passed before commit. The exact staged seven-file Phase 20E implementation set also passed `git diff --cached --check`.

## Implementation Commit

Implementation was committed as:

`a4a9a76 Extract scheduler continuation orchestration service`

Pre-report commit statistics:

- 7 files changed
- 834 insertions
- 383 deletions

## Release Build Validation

After the implementation commit, all required Release builds were rerun with stable `DerivedData/Release`:

```text
SchedulerCore Release:          rc=0 — BUILD SUCCEEDED
StrategyEvaluationCore Release: rc=0 — BUILD SUCCEEDED
LSTM Release:                   rc=0 — BUILD SUCCEEDED
```

The earlier full LSTM build failure during the Codex run was the expected Release provenance clean-tree guard. Once the implementation was committed and the worktree was clean, the full build succeeded.

The project build policy remains: stable DerivedData, Xcode's normal product path, no version-specific `CONFIGURATION_BUILD_DIR`, and post-build copying for retained builds.

## Incremental Build Result

The successful `LSTM Release` build was immediately repeated using the identical command and DerivedData path:

```text
repeat rc = 0
CompileC  = 1
Libtool   = 0
Ld        = 1
```

The only source recompiled was:

`Sources/CampaignOperationsProductionAdmissionService.cpp`

This exactly matches the Phase 20D baseline. SchedulerCore was reused without a new archive operation. The recurring single compile remains associated with generated build-provenance dependencies and stays outside Phase 20E scope.

## Monolith Reduction

Starting:

`28837 Sources/SchedulerCore/ExperimentScheduler.cpp`

Ending:

`28603 Sources/SchedulerCore/ExperimentScheduler.cpp`

Net reduction: 234 lines.

The architectural result matters more than raw line count: continuation orchestration now has an explicit SchedulerCore service boundary and focused tests.

## Remaining Responsibilities

`ExperimentScheduler.cpp` is still approximately 28.6k lines and remains substantially more than composition/compatibility glue. Significant responsibilities still include experiment queue/resume configuration, checkpoint-inference and checkpoint-policy orchestration, experiment transition/retry/requeue behavior, worker-attempt reservation/activation/spawn lifecycle, worker command construction, scheduler-loop integration, result/checkpoint-analysis coordination, large CLI parsing/dispatch paths, and operational/reporting logic.

## Phase 20F Recommendation

Do one more internal SchedulerCore decomposition phase before creating a standalone `lstm-scheduler`.

Phase 20F should inventory the remaining 28.6k-line file and choose one coherent high-value boundary. The strongest likely candidates are worker-attempt lifecycle orchestration, checkpoint-evaluation/checkpoint-policy orchestration, or experiment transition/retry/requeue orchestration. The choice should follow actual dependency clustering rather than a predetermined phase name.

A dedicated scheduler executable should wait until `ExperimentScheduler.cpp` is predominantly composition, CLI adaptation, and compatibility glue.

## Final Status

Phase 20E is complete: a substantive continuation responsibility was extracted, focused and regression tests passed, all required Release builds passed, the immediate repeat build confirmed expected incremental reuse, stable DerivedData policy was preserved, no schema change was introduced, and the implementation was committed cleanly as `a4a9a76`.
