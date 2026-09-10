# LSTM Phase 20B — Scheduler Internal Dependency Extraction

## Summary

Phase 20B completed the next conservative decomposition step for `SchedulerCore`.

The scheduler remains part of the existing `LSTM_Release` production executable. No dedicated scheduler or worker executable was introduced.

The primary architectural change was the introduction of explicit scheduler persistence and worker process-control seams:

- `SchedulerRepository`
- `PostgresSchedulerRepository`
- `WorkerProcessController`

`Sources/SchedulerCore/ExperimentScheduler.cpp` was updated to consume these seams while preserving existing scheduler behavior.

No database migration was introduced for this extraction.

## Starting point

Phase 20B began from:

    3141694 Extract scheduler core library

Branch:

    lstm-feature-development

The worktree contained the unrelated local file:

    run_phase19_batch.sh

That file was not included in the Phase 20B scheduler extraction commit.

## Phase 20A baseline

Phase 20A established:

- `SchedulerCore` static-library target
- `SchedulerEngine` executable-facing boundary
- scheduler implementation physically owned by `Sources/SchedulerCore`
- `StrategyEvaluationCore` retained as an independent static library
- successful full `LSTM Release` build

Phase 20B preserved that external boundary.

## Pre-refactor coupling inventory

`Sources/SchedulerCore/ExperimentScheduler.cpp` remained a very large implementation unit, approximately 29,000 lines.

The file directly contained extensive PostgreSQL/libpqxx interaction for scheduler authority/fencing, queue/admission state, worker-attempt lifecycle, capacity accounting, pause/resume, preemption, orphan recovery/reconciliation, checkpoint evaluation, continuation handling, scheduler reporting, and experiment lifecycle persistence.

It also directly contained POSIX/process-control behavior including worker PID handling, liveness checks, stop/resume signaling, child-status handling, and worker launch/process coordination.

Phase 20B intentionally did not attempt to remove all SQL or all scheduler orchestration from `ExperimentScheduler.cpp`.

## New scheduler persistence seam

Added:

    Sources/SchedulerCore/SchedulerRepository.hpp
    Sources/SchedulerCore/SchedulerRepository.cpp
    Sources/SchedulerCore/PostgresSchedulerRepository.hpp
    Sources/SchedulerCore/PostgresSchedulerRepository.cpp

`SchedulerRepository` defines scheduler-oriented persistence contracts using scheduler-domain value types instead of requiring scheduler policy/orchestration code to depend directly on raw PostgreSQL operations.

`PostgresSchedulerRepository` provides the PostgreSQL-backed implementation while preserving the existing database representation and transaction behavior for extracted operations.

No schema migration was required.

## New worker process-control seam

Added:

    Sources/SchedulerCore/WorkerProcessController.hpp
    Sources/SchedulerCore/WorkerProcessController.cpp

This component encapsulates scheduler-owned OS/process behavior used by scheduler orchestration, including process liveness, stop/resume signaling, child-state observation, and worker-launch/process operations.

POSIX/macOS behavior remains the concrete implementation. No new asynchronous scheduler behavior was introduced.

## ExperimentScheduler integration

Modified:

    Sources/SchedulerCore/ExperimentScheduler.cpp

The implementation now delegates selected persistence and process-control responsibilities through the new Phase 20B seams.

The extraction deliberately preserved priority ordering, resume precedence, preemption rules, capacity accounting, pause/resume admission semantics, ownership/authorization, scheduler authority/fencing, worker-attempt lifecycle, orphan recovery/reconciliation, phase distinctions, and continuation behavior.

The existing `SchedulerEngine` boundary introduced in Phase 20A remains intact.

## Xcode integration

`ExpertAdvisor.xcodeproj/project.pbxproj` was updated so the new implementation files are owned and compiled by the `SchedulerCore` static-library target.

`LSTM Release` continues to link `libSchedulerCore.a`, `libStrategyEvaluationCore.a`, and `libProfitabilityCore.a`.

## New focused tests

Added:

    Tests/SchedulerInternalSeamTests.cpp
    Tests/SchedulerInternalSeamTests.sh
    Tests/PostgresSchedulerRepositoryTests.cpp
    Tests/PostgresSchedulerRepositoryTests.sh

## Validation results

The following all passed:

    bash Tests/SchedulerCoreBoundaryTests.sh
    bash Tests/SchedulerInternalSeamTests.sh
    bash Tests/PostgresSchedulerRepositoryTests.sh
    SchedulerOwnershipPolicyTests
    SchedulerChildStatusTests

Aggregate result:

    Boundary=0
    InternalSeams=0
    PostgresRepo=0
    Ownership=0
    ChildStatus=0

## Build validation

`SchedulerCore` Release and `StrategyEvaluationCore` Release builds succeeded.

A full production Release build was then performed:

    xcodebuild       -project ExpertAdvisor.xcodeproj       -scheme "LSTM Release"       -configuration Release       -derivedDataPath "$PWD/DerivedData/Phase20B/LSTMReleaseFinal"       build

Result:

    ** BUILD SUCCEEDED **
    LSTM Release rc=0

The final linker command included:

    -lSchedulerCore
    -lStrategyEvaluationCore
    -lProfitabilityCore

This verifies the extracted scheduler seams compile and link through the production Release executable.

The normal `LSTM Release` build ran its build-provenance phase but did not invoke the separate `Publish LSTM Canonical` aggregate target.

## Scheduler extraction commit

Phase 20B scheduler extraction commit:

    1ea5bf8 Extract scheduler repository and process-control seams

## Behavioral deviations

No intentional scheduler behavioral deviations were introduced.

No database schema changes, scheduler CLI spelling changes, model/checkpoint changes, feature changes, training-objective changes, inference/profitability changes, or strategy-evaluation changes were introduced.

## Remaining coupling

`Sources/SchedulerCore/ExperimentScheduler.cpp` remains a large orchestration unit and still owns substantial scheduler orchestration, persistence-backed workflows, continuation/checkpoint coordination, authority/reconciliation logic, lifecycle transitions, reporting, and launch/admission sequencing.

Phase 20B establishes internal seams but does not yet make the scheduler sufficiently decomposed to justify immediately moving it into a separate executable.

## Recommended Phase 20C

Continue internal decomposition inside the existing `SchedulerCore` static library before introducing `lstm-scheduler`.

The best next candidates are:

- admission/capacity/preemption orchestration
- pause/resume worker control
- worker-attempt reconciliation/orphan recovery
- scheduler authority/lease handling

The goal should be to make `SchedulerEngine` call smaller domain services while preserving the same executable and CLI.

After that, reassess whether SchedulerCore is sufficiently isolated for a dedicated `lstm-scheduler` executable.
