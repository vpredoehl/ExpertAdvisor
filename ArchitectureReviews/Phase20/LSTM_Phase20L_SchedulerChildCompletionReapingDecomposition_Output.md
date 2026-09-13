---
title: "LSTM Phase 20L Scheduler Child Completion Reaping Decomposition"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase20L_SchedulerChildCompletionReapingDecomposition_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Phase 20L Scheduler Child Completion Reaping Decomposition

Implemented Phase 20L as one cohesive extraction: `SchedulerChildCompletionService`.

1. Starting baseline

- Branch: `lstm-feature-development`
- HEAD: `50afbeb658d487a7b416fc89ba69425de1e70405`
- Phase 20K: `50afbeb Extract scheduler final experiment dispatch service`
- Phase 20J retained: `99f8524 Extract scheduler cycle orchestration service`
- Starting `ExperimentScheduler.cpp`: 27,000 lines
- Pre-flight worktree: clean

2. Responsibility inventory

| Responsibility | Classification before extraction |
|---|---|
| Scheduler-owned child map and terminal bookkeeping | A — reaper-owned |
| `observedStatus` terminal caching/retry | A |
| Running-child retention | A |
| Exit/nonzero/signal/unexpected classification | A |
| Wait-error diagnostic and cleanup | A |
| Missing-attempt stale-child rejection | A |
| Checkpoint-stop-before-generic-finalization ordering | A |
| Stale-reaper failpoint ordering and scheduler-stop side effect | A |
| Exact-attempt verification sequencing | A; locking implementation D |
| One-time exit diagnostics | A |
| Checkpoint versus experiment persistence dispatch | A |
| Persistence-before-attempt-finalization ordering | A |
| Tracking removal timing | A |
| `waitpid(WNOHANG)` and POSIX status decoding | B, delegated to `WorkerProcessController` |
| Checkpoint-stop database handling | B/D |
| Experiment/checkpoint evidence and status persistence | B/D |
| Worker-attempt finalization and binding clearing | B/D |
| Authority/fencing policy | C, `SchedulerAuthorityService` |
| Pause/resume/process control | C, `WorkerControlService` |
| Missing/stopped/orphan reconciliation | C, `ReconciliationService` |
| Reserve/spawn/launch-failure lifecycle | C, `WorkerAttemptLifecycleService` |
| Operator retry/requeue/cancel | C, `ExperimentTransitionService` |
| Final train/infer/analyze dispatch | C, `FinalExperimentDispatchService` |
| Result/model/checkpoint SQL and row transitions | D, retained behind callbacks |
| Transaction creation, commit, rollback | D, unchanged in scheduler |

Selected Phase 20L responsibility: scheduler-owned child terminal-observation orchestration, including classification, cached evidence, ordering, diagnostics, and bookkeeping.

It was not already owned by another service: the process controller only observes; lifecycle service stops at launch failure; reconciliation handles non-owned missing/stopped workers; none owned the reaper’s terminal state machine.

3. Before control flow

- Running/stopped child: authority refresh → `waitpid(WNOHANG)` → running → retain tracking; no persistence.
- Exit 0/nonzero: observe → cache → checkpoint-stop check → failpoint → exact-attempt lock/verification → exit diagnostic → experiment/checkpoint persistence and evidence checks → worker-attempt finalization/binding clear → erase.
- Signal: identical, preserving `-signal`; checkpoint-stop retains its special `128+signal`.
- `WaitError`, including `ECHILD`: exact diagnostic → erase → no failure persistence; later reconciliation handles missing-process evidence.
- Missing attempt ID: cache terminal evidence → stale-child diagnostic → erase.
- Exact-attempt mismatch: reject → optional test scheduler-stop side effect → erase without touching replacement.
- Unexpected status: exact verification → dedicated persistence → finalization with `-1` and `unexpected_wait_status` → erase.
- Callback exception: transaction rolls back; cached terminal status and tracking remain for retry.

4. After control flow

The ordering is identical. `SchedulerChildCompletionService` now performs observation, caching, classification, diagnostics, callback ordering, and removal decisions. Existing scheduler adapters still perform:

- Checkpoint-stop exact-terminal handling.
- Stale replacement injection.
- Exact active-attempt locking.
- Result/evidence checks.
- SQL persistence and worker-attempt finalization.
- Authority and transaction management.
- Subsequent orphan reconciliation.

No transaction or process-observation ordering changed.

5. Architectural boundary and files

Added:

- [SchedulerChildCompletionService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/SchedulerChildCompletionService.hpp>)
- [SchedulerChildCompletionService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/SchedulerChildCompletionService.cpp>)
- [SchedulerChildCompletionServiceTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerChildCompletionServiceTests.cpp>)
- [SchedulerChildCompletionServiceTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerChildCompletionServiceTests.sh>)

Modified:

- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ExperimentScheduler.cpp>)
- [SchedulerChildStatusTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/SchedulerChildStatusTests.cpp>)
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj>)

`ExperimentScheduler.cpp`: 27,000 → 26,864 lines, net reduction 136.

6. Repository, SQL, and public-header boundary

- Repository seam changes: none.
- No SQL moved or modified.
- No unrelated SQL cleanup.
- Public header exposes no `pqxx`, raw SQL, PostgreSQL type, `ExperimentScheduler.hpp`, or Postgres implementation header.
- Direct public-header includes: `SchedulerChildStatus.hpp`, `<functional>`, `<iosfwd>`, `<map>`, `<optional>`, `<string>`, `<string_view>`, and `<sys/types.h>`.
- Production direct consumers: `ExperimentScheduler.cpp` and the service implementation. Test consumer: focused service test. Fan-out is low.

7. Validation

Passed:

- `Tests/SchedulerChildCompletionServiceTests.sh`
- Manual `SchedulerChildStatusTests.cpp` compile/run with `-Wall -Wextra -Werror`
- `Tests/WorkerAttemptLifecycleServiceTests.sh`
- `Tests/SchedulerInternalSeamTests.sh`
- `Tests/SchedulerOrchestrationServiceTests.sh`
- `Tests/SchedulerCycleServiceTests.sh`
- `Tests/SchedulerAuthorityServiceTests.sh`
- `Tests/SchedulerCoreBoundaryTests.sh`
- `git diff --check`, including separate checks for untracked files

Focused build:

```sh
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme SchedulerCore \
  -configuration Release \
  -derivedDataPath "$PWD/DerivedData/Release" \
  build
```

Result: `** BUILD SUCCEEDED **`

CompileC units touched:

- `SchedulerChildCompletionService.cpp`
- `ExperimentScheduler.cpp`

No full `LSTM Release` build or executable/database integration test was run, per policy.

8. Risk assessment

PROCESS-LIFECYCLE RISK — Low. `WNOHANG`, exit/signal interpretation, stopped-child retention, wait-error behavior, cached observations, checkpoint-stop handling, stale-attempt rejection, and removal timing are preserved and characterized.

TRANSACTION-BOUNDARY RISK — Low. Transactions, locks, authority refresh, persistence order, rollback behavior, result checks, and commit location did not move.

ORDERING RISK — Low. Tests verify checkpoint-stop precedence, stale injection before exact verification, persistence before finalization, cleanup after successful handling, and cached retry without a second observation.

AUTHORITY/FENCING RISK — Low. Authority decisions remain uncached and in existing scheduler/authority code. No authority semantics entered the service.

BUILD-FAN-OUT RISK — Low. The narrow header has two production consumers and no database dependencies.

9. Preservation and unresolved concerns

- CLI, schema, migrations, Phase 1–20K behavior, model identity, migration-089 fields, checkpoint behavior, dispatch, pause/resume, and reconciliation semantics were not intentionally changed.
- Operational diagnostic markers and field ordering are preserved.
- Focused build reported the existing warning set: 342 `ExperimentScheduler.cpp` warnings, predominantly deprecated `libpqxx::exec_params`, plus the existing `-Ofast` deprecation. The focused standalone test is warning-clean under `-Werror`.
- Database-backed stale-reaper/checkpoint-stop/result-recovery integration suites were not rerun because they require an LSTM executable and live scheduler/process/database coordination.
- A standalone scheduler is still not justified at 26,864 lines.

10. Git state

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/SchedulerCore/ExperimentScheduler.cpp
 M Tests/SchedulerChildStatusTests.cpp
?? Sources/SchedulerCore/SchedulerChildCompletionService.cpp
?? Sources/SchedulerCore/SchedulerChildCompletionService.hpp
?? Tests/SchedulerChildCompletionServiceTests.cpp
?? Tests/SchedulerChildCompletionServiceTests.sh
```

Exact `git diff --stat`—Git excludes the untracked new files:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj       |   6 +
 Sources/SchedulerCore/ExperimentScheduler.cpp | 330 ++++++++------------------
 Tests/SchedulerChildStatusTests.cpp           |  11 +
 3 files changed, 114 insertions(+), 233 deletions(-)
```

Nothing was staged, committed, pushed, or archived.

11. Recommended Phase 20M candidate

Inventory `RecoverOrphanedRunningExperiments`, currently beginning around line 17,217. The strongest candidate is its database-backed exact worker-attempt reconciliation orchestration, implemented by extending the existing `ReconciliationService` ownership rather than introducing a competing reconciliation service. Its current policy helpers are already extracted, but substantial locking, evidence evaluation, stopped-worker, and missing-process orchestration remains in `ExperimentScheduler.cpp`.