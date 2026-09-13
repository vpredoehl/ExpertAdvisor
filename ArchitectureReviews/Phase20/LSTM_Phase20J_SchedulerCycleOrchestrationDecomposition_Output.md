---
title: "LSTM Phase 20J Scheduler Cycle Orchestration Decomposition"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase20J_SchedulerCycleOrchestrationDecomposition_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Phase 20J Scheduler Cycle Orchestration Decomposition

Implemented exactly one extraction: `SchedulerCycleService`. All focused tests and the permitted `SchedulerCore` Release build pass. Changes remain unstaged.

## 1. Starting state

- Branch: `lstm-feature-development`
- HEAD: `8bba8ac9710d96de80823b2ecb71bedeb24c6f44`
- Starting `ExperimentScheduler.cpp`: 27,116 lines
- Worktree: clean

Initial log:

```text
8bba8ac Extract checkpoint policy evaluation service
bf294bd Extract scheduler experiment transition service
bf7366e Resolve legacy model identity for scheduler admission
7dccae9 Extract checkpoint analysis orchestration service
6bc6a27 Extract scheduler worker-attempt lifecycle service
47245a6 Add Phase 19C causal path predictability analysis
017700a Consume checkpoint stop request after checkpoint stop
a92b57c Fix stopped worker reconciliation after build cleanup
```

## 2. Remaining-responsibility inventory

| Candidate | Existing ownership | Coupling | Extraction assessment |
|---|---|---|---|
| Poll/cycle sequencing | Still in `ExperimentScheduler.cpp`; `SchedulerEngine` only forwards compatibility entrypoints | PostgreSQL high in preparation; process and authority behavior indirect/delegated | Cleanly extractable through a narrow orchestration port |
| Phase dispatch loops | Admission/preemption and worker-attempt behavior already belongs to established services | PostgreSQL/process/authority high | Possible later, but risks re-owning subordinate behavior |
| Child reaping/orphan recovery | Decision logic partly belongs to reconciliation/control/lifecycle services; persistence adapter remains local | High across PostgreSQL, process identity, and fencing | Cohesive but higher-risk |
| Queue/enqueue/materialization | Not extracted | PostgreSQL high; process/authority low | Requires substantial repository work |
| Status/intelligence reporting | Not extracted | PostgreSQL high, process moderate, authority low | Cohesive reporting extraction, but not the preferred cycle boundary |
| Daemon lifecycle | Authority and continuation semantics already have owners | Authority high | Unsafe to extract without risking ownership overlap |
| CLI/compatibility dispatch | Compatibility glue, not scheduler-domain orchestration | Very broad dependency fan-out | Not an appropriate service boundary |

## 3. Selected responsibility

```text
SELECTED PHASE 20J RESPONSIBILITY:
One scheduler poll/cycle’s control-flow orchestration.
```

`SchedulerEngine` does not own this behavior: it only delegates recognition, CLI execution, and worker-attempt registration to `ExperimentScheduler` compatibility entrypoints.

The new service owns:

- poll diagnostic lifecycle;
- preparation-result gating;
- normal versus cancellation-only cycle selection;
- exact phase-dispatch ordering;
- cycle return-code aggregation.

It does not own authority, admission, preemption, reservation, launch, persistence, checkpoint evaluation, continuation scanning, or reconciliation semantics.

## 4. Control flow

BEFORE:

```text
RunScheduler
  authority refresh
  RunSchedulerOnce
    begin poll diagnostics
    one preparation transaction:
      authority/fence validation
      schema validation
      lock/read global control
      cancellation reconciliation
      metadata backfill
      child reaping
      orphan recovery
      checkpoint-eval enqueue
      invalid-phase handling
      queue snapshot
      commit
    cancellation branch:
      checkpoint train
      checkpoint inference
    normal branch:
      train
      final inference
      final analysis
      checkpoint inference
      checkpoint analysis
    finish poll diagnostics
  continuation scan
```

AFTER:

```text
RunScheduler
  authority refresh
  SchedulerCycleService::runOnce
    beginPoll adapter
    prepare adapter:
      same transaction and same operation order as before
    service selects:
      cancellation branch in original order, or
      normal five-phase dispatch in original order
    finishPoll adapter
  continuation scan
```

## 5. Architectural boundary and files

Added:

- [SchedulerCycleService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/SchedulerCycleService.hpp>)
- [SchedulerCycleService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/SchedulerCycleService.cpp>)
- [SchedulerCycleServiceTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerCycleServiceTests.cpp>)
- [SchedulerCycleServiceTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerCycleServiceTests.sh>)

Modified:

- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ExperimentScheduler.cpp:21141>)
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj:331>)

`ExperimentScheduler.cpp`:

- Before: 27,116 lines
- After: 27,112 lines
- Net reduction: 4 lines

No repository seam changed. No SQL was moved.

The new public header includes only `<functional>`. It contains no:

- `ExperimentScheduler.hpp`
- other application/high-fan-out scheduler headers
- pqxx/libpq
- repository implementation headers
- raw SQL or transaction types

## 6. Tests

Passed:

```text
Tests/SchedulerCycleServiceTests.sh
Tests/SchedulerInternalSeamTests.sh
Tests/SchedulerOrchestrationServiceTests.sh
Tests/SchedulerAuthorityServiceTests.sh
Tests/WorkerAttemptLifecycleServiceTests.sh
Tests/CheckpointAnalysisOrchestrationServiceTests.sh
Tests/CheckpointEvaluationServiceTests.sh
Tests/ContinuationOrchestrationServiceTests.sh
Tests/ExperimentTransitionServiceTests.sh
Tests/SchedulerCoreBoundaryTests.sh
```

The new tests cover:

- empty/non-scheduling cycle with no dispatch;
- full normal-cycle ordering and result aggregation;
- cancellation-only train/inference ordering;
- authority/preparation exception preventing dispatch;
- unsuccessful preparation preventing dispatch;
- incomplete adapter rejection.

## 7. Focused build

Command:

```bash
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme SchedulerCore \
  -configuration Release \
  -derivedDataPath "$PWD/DerivedData/Release" \
  build
```

Result:

```text
** BUILD SUCCEEDED **
```

Build actions:

```text
CompileC: 2
Libtool: 1
Ld: 0
```

Compiled units:

```text
SchedulerCycleService.cpp
ExperimentScheduler.cpp
```

Warnings:

- Existing project-level `-Ofast` deprecation warning.
- Existing libpqxx `exec_params` deprecations from `ExperimentScheduler.cpp`; 342 warnings reported for that unit.
- No new cycle-service diagnostic other than the inherited `-Ofast` warning.

The full `LSTM Release` build and incremental benchmark were not run, as required.

`git diff --check`: passed with no output.

## 8. Risk assessment

TRANSACTION-BOUNDARY RISK:

No change. The same `pqxx::connection` and `pqxx::work` remain in the adapter, with the same authority validation, mutations, snapshot load, and commit location. Early schema/control failures still leave via transaction destruction without committing. Dispatch functions and their independent transaction scopes are unchanged.

ORDERING RISK:

No change. Preparation operations retain their original order, and the service tests lock down both normal and cancellation dispatch ordering. Admission, preemption, reservation, launch, checkpoint evaluation, and continuation operations remain in their prior relative positions.

AUTHORITY/FENCING RISK:

No lease, fence, invocation-authority, takeover, or displacement decision moved. The production adapter sets the cycle as ready only after `RequireAndRefreshSchedulerAuthority` and the preparation transaction completes. `SchedulerAuthorityLost` continues propagating to the existing outer handler.

## 9. Behavioral assessment

- CLI and schema semantics: unchanged.
- Migration-089 model identity enforcement: untouched.
- Legacy model-identity resolution: untouched.
- Transaction and locking behavior: unchanged.
- Operational log strings: preserved. The only new text is an internal invalid-adapter exception.
- Standalone scheduler extraction: not yet justified. The 27,112-line compatibility unit still contains substantial CLI, PostgreSQL adapter, phase-dispatch, reaping/recovery, and reporting behavior.

Recommended Phase 20K candidate: the active final-experiment phase-dispatch cluster—`RunTrainJobs`, `RunInferJobs`, and `RunAnalyzeJobs` around lines 20,510–21,028—using the cycle service’s new operations seam while preserving admission, preemption, lifecycle, and launch ownership.

## 10. Final worktree

Exact `git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/SchedulerCore/ExperimentScheduler.cpp
?? Sources/SchedulerCore/SchedulerCycleService.cpp
?? Sources/SchedulerCore/SchedulerCycleService.hpp
?? Tests/SchedulerCycleServiceTests.cpp
?? Tests/SchedulerCycleServiceTests.sh
```

Exact `git diff --stat`—which excludes the four untracked additions until staged:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj       |  6 +++
 Sources/SchedulerCore/ExperimentScheduler.cpp | 60 +++++++++++++--------------
 2 files changed, 34 insertions(+), 32 deletions(-)
```

Nothing was staged, committed, amended, or pushed.