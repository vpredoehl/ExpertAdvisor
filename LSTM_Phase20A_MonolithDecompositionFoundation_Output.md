# LSTM Phase 20A — Monolith Decomposition Foundation

## Summary

Phase 20A completed the first conservative architectural extraction from the
monolithic `LSTM_Release` executable.

The production executable remains `LSTM_Release`. No scheduler/train/infer/
analyze worker executable split was performed.

The primary completed extraction is `SchedulerCore`. The existing
`StrategyEvaluationCore` static-library boundary was retained and validated.

No database migrations, model-format changes, feature-semantic changes,
training-objective changes, inference-semantic changes, scheduler lifecycle
changes, or CLI spelling changes were intentionally introduced.

## Starting state

Branch:

    lstm-feature-development

Starting HEAD:

    b01970afb300e047d9a99dfbd2387dfbb493b1a8

The worktree contained pre-existing changes before Phase 20A, including:

    M ExpertAdvisor.xcodeproj/project.pbxproj
    ?? run_phase19_batch.sh

The untracked Phase 19 batch script was not included in the Phase 20A work.

## Architecture inventory

Before extraction, scheduler command handling and worker-attempt registration
were directly accessed by `LSTM/main.cpp` through the legacy
`ExperimentScheduler` interface.

The scheduler implementation was physically located at:

    Sources/ExperimentScheduler.cpp

The scheduler implementation contains scheduler CLI handling, persistence,
worker-process control, lifecycle transitions, admission, priority/preemption,
pause/resume, ownership/fencing, worker-attempt handling, orphan recovery,
reconciliation, continuation processing, and scheduler status support.

Strategy evaluation already had an extracted static-library foundation:

    StrategyEvaluationCore

Phase 20A preserved that existing boundary and validated it rather than
performing a semantic redesign.

## Source-layout changes

The scheduler implementation now physically lives under:

    Sources/SchedulerCore/

Files added:

    Sources/SchedulerCore/ExperimentScheduler.cpp
    Sources/SchedulerCore/SchedulerEngine.cpp
    Sources/SchedulerCore/SchedulerEngine.hpp
    Sources/SchedulerCore/SchedulerPolicy.cpp
    Sources/SchedulerCore/SchedulerPolicy.hpp

The previous path:

    Sources/ExperimentScheduler.cpp

is retained as a compatibility symbolic link to:

    SchedulerCore/ExperimentScheduler.cpp

This preserves existing source references while moving ownership of the
implementation into the scheduler component.

## SchedulerCore target

A new static-library target was added:

    SchedulerCore

Product:

    libSchedulerCore.a

The target contains:

    ExperimentScheduler.cpp
    SchedulerEngine.cpp
    SchedulerPolicy.cpp

`LSTM Release` links against `libSchedulerCore.a`.

The scheduler implementation is therefore no longer compiled directly as an
implementation source owned solely by the executable.

## SchedulerCore public boundary

The new typed application seam is:

    EA::SchedulerCore::SchedulerEngine

with:

    CommandInvocation
    WorkerAttemptRegistration

and operations equivalent to:

    recognizes(...)
    run(...)
    registerWorkerAttempt(...)

`LSTM/main.cpp` now delegates scheduler command recognition, scheduler command
execution, and scheduler worker-attempt registration through this component
boundary.

The compatibility `EA::ExperimentScheduler` entrypoints remain available
behind the new facade.

## Scheduler policy seam

The following scheduler ordering rules were given explicit component-level
helpers:

    PriorityRank(...)
    ResumeOriginRank(...)
    CanPreempt(...)

Existing persisted string values remain unchanged.

Priority rank remains:

    high   -> 0
    normal -> 1
    low    -> 2

Equal priority does not preempt because `CanPreempt()` requires a strictly
better candidate rank.

Resume-origin ordering remains explicitly represented with operator precedence
ahead of preemption and other origins.

## StrategyEvaluation boundary

`StrategyEvaluationCore` remains a static-library target.

No Pocket, candlestick, Smart Waves, Elliott Wave, or other new strategy was
implemented in Phase 20A.

No strategy semantics were intentionally changed.

The existing StrategyEvaluation foundation remains the reusable domain
boundary for future strategy research.

## LSTM Release integration

`LSTM/main.cpp` no longer directly dispatches scheduler CLI commands through
`EA::ExperimentScheduler`.

It now constructs:

    EA::SchedulerCore::SchedulerEngine

and delegates scheduler-related operations through that object.

CLI spellings and external command behavior were intentionally preserved.

The executable remains:

    LSTM_Release

No separate scheduler or worker executable was introduced.

## Tests

### SchedulerCore boundary

Command:

    bash Tests/SchedulerCoreBoundaryTests.sh

Result:

    PASS

### Scheduler ownership policy

Command:

    clang++ -std=c++20 -Wall -Wextra -Werror \
      -IHeaders \
      Tests/SchedulerOwnershipPolicyTests.cpp \
      -o <temporary>/SchedulerOwnershipPolicyTests

    <temporary>/SchedulerOwnershipPolicyTests

Result:

    PASS

### Scheduler child status

Command:

    clang++ -std=c++20 -Wall -Wextra -Werror \
      -IHeaders -ISources \
      Tests/SchedulerChildStatusTests.cpp \
      -o <temporary>/SchedulerChildStatusTests

    <temporary>/SchedulerChildStatusTests

Result:

    PASS

## Build validation

### SchedulerCore

Release static-library build:

    xcodebuild \
      -project ExpertAdvisor.xcodeproj \
      -target SchedulerCore \
      -configuration Release \
      SYMROOT=<isolated Phase20A path> \
      OBJROOT=<isolated Phase20A path> \
      build

Result:

    BUILD SUCCEEDED

### StrategyEvaluationCore

Command:

    xcodebuild \
      -project ExpertAdvisor.xcodeproj \
      -scheme StrategyEvaluationCore \
      -configuration Release \
      -derivedDataPath <isolated Phase20A path> \
      build

Result:

    BUILD SUCCEEDED

### LSTM Release

Command:

    xcodebuild \
      -project ExpertAdvisor.xcodeproj \
      -scheme "LSTM Release" \
      -configuration Release \
      -derivedDataPath <isolated Phase20A path> \
      build

Result:

    BUILD SUCCEEDED

This validates the final executable link against the extracted static
SchedulerCore library.

## Compiler observations

The scheduler build emits existing libpqxx `exec_params` deprecation warnings.

These were warnings only and did not prevent compilation or linking.

No Phase 20A cleanup of those calls was attempted because that would be
unrelated to the structural extraction.

## Behavioral deviations

None intentionally introduced.

No scheduler algorithms were intentionally rewritten.

No database schema changes were made.

No model/checkpoint serialization changes were made.

No feature-width or model-input identity behavior was changed.

No training objective or profitability semantics were changed.

No strategy semantics were changed.

## Current dependency direction

    LSTM_Release / CLI
          |
          +--> SchedulerCore
          |
          +--> StrategyEvaluationCore
          |
          +--> existing training / inference / model code

SchedulerCore still contains substantial persistence and OS process-control
coupling internally. That coupling was intentionally left in place for this
first structural cut to avoid changing scheduler behavior.

## Remaining coupling

`Sources/SchedulerCore/ExperimentScheduler.cpp` remains a large implementation
unit and still directly performs:

- PostgreSQL persistence
- process discovery/control/spawning
- scheduler authority and fencing
- worker-attempt persistence
- lifecycle reconciliation
- continuation handling
- scheduler CLI operations

Phase 20A establishes the component ownership and executable-facing seam but
does not yet fully separate scheduler policy, repository access, and process
control into independently replaceable services.

This is intentional.

## Recommended next decomposition

The next scheduler-focused phase should extract internal scheduler dependencies
behind explicit repository and worker-process-control seams without changing
scheduler behavior.

A useful target direction is:

    SchedulerEngine
        |
        +--> SchedulerRepository
        |
        +--> WorkerProcessController
        |
        +--> SchedulerPolicy

After those seams are stable and tested, a dedicated `lstm-scheduler`
executable can be introduced with considerably lower risk.

Separately, shared persistence/common infrastructure can be extracted before
splitting train/infer/analyze worker executables.

