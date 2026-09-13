# LSTM Phase 20G — Scheduler Remaining Domain Decomposition

## Summary

Phase 20G continued the conservative `SchedulerCore` decomposition by extracting one additional cohesive scheduler responsibility from `Sources/SchedulerCore/ExperimentScheduler.cpp`.

The selected responsibility was **checkpoint-analysis orchestration**. The phase introduced `CheckpointAnalysisOrchestrationService`, extended the scheduler repository seam with the persistence operations required by that boundary, added focused characterization/unit coverage, updated affected scheduler seam tests, and integrated the new service into the `SchedulerCore` Xcode target.

No standalone scheduler executable was introduced. The remaining `ExperimentScheduler.cpp` still contains substantial scheduler-domain logic, so another internal decomposition phase is recommended before considering an executable split.

## Starting State

- Branch: `lstm-feature-development`
- Starting HEAD: `6bc6a272385d2beab0fe4784e8a8fca616d9c255`
- Starting `ExperimentScheduler.cpp` line count: **28,418**
- Worktree at Phase 20G preflight: **clean**

The existing SchedulerCore architecture already included:

- `SchedulerEngine`
- `SchedulerAdmissionService`
- `WorkerControlService`
- `ReconciliationService`
- `SchedulerAuthorityService`
- `ContinuationOrchestrationService`
- `WorkerAttemptLifecycleService`
- `SchedulerRepository`
- `PostgresSchedulerRepository`
- `WorkerProcessController`
- `SchedulerPolicy`

## Inventory and Selected Boundary

The Phase 20G inventory found several substantial responsibilities still inside `ExperimentScheduler.cpp`, including:

- checkpoint evaluation / checkpoint-analysis orchestration;
- experiment transition, completion, retry, and requeue orchestration;
- checkpoint policy/evaluation logic;
- scheduler status/control/reporting logic;
- queue/control and remaining CLI composition.

Phase 20G selected **checkpoint-analysis orchestration** as the extraction boundary.

This boundary was preferred because it was cohesive, testable, and distinct from the already-extracted scheduler-launched worker-attempt lifecycle. Phase 20F intentionally left scheduler-in-process checkpoint analysis outside `WorkerAttemptLifecycleService`; Phase 20G preserved that distinction rather than forcing in-process analysis through the child-process lifecycle abstraction.

## Files Added

- `Sources/SchedulerCore/CheckpointAnalysisOrchestrationService.cpp`
- `Sources/SchedulerCore/CheckpointAnalysisOrchestrationService.hpp`
- `Tests/CheckpointAnalysisOrchestrationServiceTests.cpp`
- `Tests/CheckpointAnalysisOrchestrationServiceTests.sh`

## Files Modified

- `ExpertAdvisor.xcodeproj/project.pbxproj`
- `Sources/SchedulerCore/ExperimentScheduler.cpp`
- `Sources/SchedulerCore/PostgresSchedulerRepository.cpp`
- `Sources/SchedulerCore/PostgresSchedulerRepository.hpp`
- `Sources/SchedulerCore/SchedulerRepository.hpp`
- `Tests/PostgresSchedulerRepositoryTests.cpp`
- `Tests/SchedulerInternalSeamTests.cpp`
- `Tests/SchedulerOrchestrationServiceTests.cpp`
- `Tests/WorkerAttemptLifecycleServiceTests.cpp`

## Architectural Result

The scheduler architecture after Phase 20G includes:

```text
SchedulerEngine
    |
    +-- SchedulerAdmissionService
    +-- WorkerControlService
    +-- ReconciliationService
    +-- SchedulerAuthorityService
    +-- ContinuationOrchestrationService
    +-- WorkerAttemptLifecycleService
    +-- CheckpointAnalysisOrchestrationService
    |
    +-- SchedulerRepository
    +-- PostgresSchedulerRepository
    +-- WorkerProcessController
    +-- SchedulerPolicy
```

`CheckpointAnalysisOrchestrationService` now owns checkpoint-analysis claim/work/finalization behavior that was previously embedded in `ExperimentScheduler.cpp`.

The repository seam was extended with semantic operations needed by the new service. Public scheduler service APIs remain separated from pqxx/libpq/raw SQL implementation details; PostgreSQL-specific behavior remains in `PostgresSchedulerRepository`.

The extraction preserved the distinction between scheduler-launched child worker lifecycle, owned by `WorkerAttemptLifecycleService`, and scheduler-in-process checkpoint analysis, now owned by `CheckpointAnalysisOrchestrationService`.

## Behavior Preservation

Phase 20G was a structural refactor. No schema migration was introduced.

The implementation was intended to preserve scheduler admission and capacity semantics; priority and preemption behavior; pause/resume semantics; worker-attempt identity and lifecycle semantics; orphan/stopped-worker reconciliation; scheduler authority, lease, ownership, and fencing; continuation scanning and queueing; checkpoint inference, analysis, and stop behavior; experiment completion/failure/retry behavior; SQL ordering, locking, and affected-row requirements; worker arguments and canonical executable behavior; log paths; CLI behavior; and training, inference, profitability, and strategy-evaluation behavior.

## Focused and Regression Validation

The following tests passed during manual Phase 20G completion:

- `CheckpointAnalysisOrchestrationServiceTests.sh`
- `WorkerAttemptLifecycleServiceTests.sh`
- `PostgresSchedulerRepositoryTests.sh`
- `SchedulerInternalSeamTests.sh`
- `SchedulerOrchestrationServiceTests.sh`
- `SchedulerAuthorityServiceTests.sh`
- `SchedulerCoreBoundaryTests.sh`
- `ContinuationOrchestrationServiceTests.sh`
- `SchedulerSemanticAdmissionTests.sh`
- `CheckpointPolicyHardeningTests.sh`

`git diff --check` also completed successfully with return code 0.

No production-state-mutating scheduler CLI tests were run.

## Build Validation

The `SchedulerCore` Release target built successfully using the stable DerivedData path:

```bash
xcodebuild   -project ExpertAdvisor.xcodeproj   -scheme SchedulerCore   -configuration Release   -derivedDataPath "$PWD/DerivedData/Release"   build
```

Result:

```text
** BUILD SUCCEEDED **
```

After the Phase 20G implementation was committed, the full `LSTM Release` scheme was built twice using the established stable build path. Both builds succeeded.

Immediate repeated build counts were:

```text
CompileC = 1
Libtool  = 0
Ld       = 0
```

The only recompiled source was:

```text
Sources/CampaignOperationsProductionAdmissionService.cpp
```

This preserves the desired incremental-build behavior and is slightly better than the previous approximate repeat-build baseline of `CompileC=1`, `Libtool=0`, `Ld=1`.

No `CONFIGURATION_BUILD_DIR` override was used.

## Line Count

`Sources/SchedulerCore/ExperimentScheduler.cpp`:

```text
Before Phase 20G: 28,418 lines
After Phase 20G:  28,339 lines
Net reduction:         79 lines
```

The relatively small line-count reduction is acceptable because Phase 20G prioritized a coherent domain boundary over raw line-count reduction.

## Commit

Phase 20G implementation commit:

```text
b42bd60 Extract checkpoint analysis orchestration service
```

Commit summary:

```text
13 files changed, 884 insertions(+), 190 deletions(-)
```

New files created by the commit:

```text
Sources/SchedulerCore/CheckpointAnalysisOrchestrationService.cpp
Sources/SchedulerCore/CheckpointAnalysisOrchestrationService.hpp
Tests/CheckpointAnalysisOrchestrationServiceTests.cpp
Tests/CheckpointAnalysisOrchestrationServiceTests.sh
```

The worktree was clean after the two full Release builds.

## Deferred / Intentionally Not Performed

Phase 20G did not create a standalone scheduler executable, introduce a schema migration, redesign scheduler behavior, move scheduler code into a new subproject, alter training/inference/profitability/strategy semantics, vary `CONFIGURATION_BUILD_DIR`, or run production-mutating scheduler CLI tests.

## Phase 20H Recommendation

A standalone scheduler executable is **not yet justified**.

`ExperimentScheduler.cpp` remains approximately **28.3K lines** and still contains substantial scheduler-domain logic rather than being mostly composition/compatibility glue.

The preferred Phase 20H candidate is **experiment transition / completion / retry / requeue orchestration**. This appears to be the next high-value internal boundary because it can move state-transition ownership out of the monolith while preserving the already-separated responsibilities for admission/preemption, process control, reconciliation, scheduler authority/fencing, continuation orchestration, worker-attempt lifecycle, and checkpoint-analysis orchestration.

A secondary Phase 20H candidate is **checkpoint policy / checkpoint-evaluation orchestration**.

The Phase 20H prompt should inventory the remaining code again before selecting the final boundary rather than forcing either candidate in advance.

## Final Phase 20G Status

- Implementation committed: **yes**
- Focused tests: **passed**
- Scheduler regression tests: **passed**
- `SchedulerCore` Release build: **passed**
- Full `LSTM Release` build: **passed twice**
- Repeat-build cache behavior: **healthy**
- Final worktree: **clean**
- Standalone scheduler executable: **deferred**
- Recommended next step: **Phase 20H internal SchedulerCore decomposition**
