# LSTM Phase 20F — Scheduler Worker-Attempt Lifecycle Decomposition

## Result

Phase 20F completed the planned internal SchedulerCore decomposition by
extracting a coherent worker-attempt lifecycle service while preserving the
existing scheduler executable, CLI, database schema, scheduler semantics, and
stable incremental-build behavior.

The extracted responsibility is the scheduler-launched worker lifecycle
boundary covering reservation, lifecycle claim, child launch, and persistence
of launch/spawn/failure evidence.

No standalone scheduler executable was introduced.

## Starting State

- Branch: `lstm-feature-development`
- Starting HEAD: `47245a6 Add Phase 19C causal path predictability analysis`
- Starting `Sources/SchedulerCore/ExperimentScheduler.cpp`: 28,615 lines
- Worktree was clean at Phase 20F start.

Existing internal SchedulerCore decomposition included:

- `SchedulerAdmissionService`
- `WorkerControlService`
- `ReconciliationService`
- `SchedulerAuthorityService`
- `ContinuationOrchestrationService`
- `SchedulerRepository`
- `PostgresSchedulerRepository`
- `WorkerProcessController`
- `SchedulerPolicy`

## Extracted Boundary

Added:

- `Sources/SchedulerCore/WorkerAttemptLifecycleService.cpp`
- `Sources/SchedulerCore/WorkerAttemptLifecycleService.hpp`

The service owns the scheduler-launched worker-attempt lifecycle boundary,
including:

- generation of scheduler worker launch identities;
- experiment worker-attempt reservation;
- checkpoint inference worker-attempt reservation;
- reservation/lifecycle-claim result handling;
- child process launch coordination;
- persistence of spawned-worker evidence;
- persistence of launch failures;
- exact-attempt/fencing rejection diagnostics.

The extraction deliberately did not absorb unrelated scheduler responsibilities
such as:

- stopped-worker reconciliation;
- scheduler authority/lease/fencing;
- capacity and priority admission;
- preemption;
- pause/resume control;
- continuation orchestration;
- final experiment transition logic;
- in-process checkpoint analysis.

`ClaimCheckpointAnalysis()` remains in `ExperimentScheduler.cpp` because
checkpoint analysis is an in-process scheduler operation using
`ownership_origin='scheduler_in_process'`, not a scheduler-launched child
worker. Its launch identity continues to use the same scheduler invocation /
nonce / command-identity structure without routing the operation through
`WorkerAttemptLifecycleService`.

## Repository Seam

`SchedulerRepository` and `PostgresSchedulerRepository` were extended with
narrow worker-attempt reservation operations.

The public SchedulerCore seam does not expose pqxx or raw SQL.

Reservation operations return semantic result types instead of leaking
database implementation details.

The PostgreSQL implementation preserves atomic reservation/lifecycle-claim
semantics and scheduler fencing predicates.

## Tests Added

Added:

- `Tests/WorkerAttemptLifecycleServiceTests.cpp`
- `Tests/WorkerAttemptLifecycleServiceTests.sh`

The focused service test covers reservation/claim outcomes, spawned-worker
persistence, launch-failure persistence, exact-attempt predicate rejection,
fence rejection diagnostics, and required scheduler authority context.

## Validation

The following focused/regression tests passed during Phase 20F:

- `WorkerAttemptLifecycleServiceTests`
- `ContinuationOrchestrationServiceTests`
- `SchedulerAuthorityServiceTests`
- `SchedulerCoreBoundaryTests`
- `SchedulerInternalSeamTests`
- `SchedulerOrchestrationServiceTests`
- `PostgresSchedulerRepositoryTests`
- `SchedulerSemanticAdmissionTests`
- direct `SchedulerOwnershipPolicyTests`

`PostgresSchedulerRepositoryTests` was extended for the new reservation paths.
Its queue snapshot expectation was corrected after reservation of the additional
train attempt: the fixture then contains two running train experiments.

`git diff --check` passed.

Production scheduler CLI tests that could mutate active scheduler/database
state were intentionally not run.

## Release Builds

Stable DerivedData policy was preserved:

`DerivedData/Release`

No varying `CONFIGURATION_BUILD_DIR` was used.

Validated successfully:

- SchedulerCore Release build
- StrategyEvaluationCore Release build
- full `LSTM Release` build

After the Phase 20F implementation commit, two consecutive full
`LSTM Release` builds succeeded.

Immediate repeat-build counts:

- `CompileC = 1`
- `Libtool = 0`
- `Ld = 1`

The only repeated compilation was:

`Sources/CampaignOperationsProductionAdmissionService.cpp`

This matches the established generated-build-provenance behavior and confirms
that Phase 20F preserved the desired incremental-build cache behavior.

## Commit

Implementation commit:

`a6f3076 Extract scheduler worker-attempt lifecycle service`

Implementation change set:

- 12 files
- 918 insertions
- 243 deletions

## Monolith Reduction

`Sources/SchedulerCore/ExperimentScheduler.cpp`

- Start: 28,615 lines
- End: 28,418 lines
- Net reduction: 197 lines

The relatively modest line-count reduction is expected because Phase 20F moved
a narrow but semantically important reservation/launch/persistence boundary
while retaining compatibility and orchestration integration in the existing
scheduler.

## Phase 20G Recommendation

Do not split a standalone scheduler executable yet.

`ExperimentScheduler.cpp` remains approximately 28.4K lines, so another
internal SchedulerCore decomposition is warranted before introducing a new
scheduler process boundary.

Phase 20G should begin with an inventory of the remaining responsibilities and
extract the largest cleanly separable scheduler-domain responsibility rather
than selecting a service name in advance.

Likely candidates include:

1. checkpoint evaluation / checkpoint orchestration that remains outside the
   worker-attempt lifecycle service;
2. experiment transition, completion, retry, and requeue orchestration;
3. scheduler invocation/lifecycle orchestration not already owned by
   `SchedulerAuthorityService`;
4. another larger cohesive responsibility identified by inventory.

The same constraints should remain in force:

- one coherent extraction per phase;
- no raw pqxx in public service interfaces;
- no new global scheduler state;
- preserve SQL ordering/locking and fencing semantics;
- preserve stable DerivedData and default Xcode product paths;
- do not vary `CONFIGURATION_BUILD_DIR` for retained builds;
- avoid DB-mutating production scheduler CLI validation;
- measure the immediate repeated Release build.
