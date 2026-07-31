---
title: "Scheduler Restart Multi-Instance and Worker Ownership Correctness Implementation"
document_type: "architecture review"
status: "final"
generated_from: "Scheduler_Restart_MultiInstance_WorkerOwnership_Correctness_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Scheduler Restart Multi-Instance and Worker Ownership Correctness Implementation

## 1. Executive summary

Implemented a database-authoritative scheduler coordination architecture with:

- A fenced singleton scheduler lease and immutable invocation history.
- Durable worker attempts and global capacity reservations.
- Restart-safe observation of workers launched by prior schedulers.
- Exact process-identity validation resistant to PID reuse.
- Transactional duplicate-launch prevention and destructive-reconciliation fences.
- Child-supplied spawn evidence plus a parent-controlled exec gate.
- Canonical absolute self-executable resolution for every scheduler child.
- Explicit graceful shutdown, lease-loss, crash, and takeover semantics.
- Read-only status diagnostics for lease, ownership, capacity, and attempts.

The active production scheduler and workers were inspected but not stopped, signaled, migrated, or modified.

## 2. Architectural defect model

The prior design had four coupled defects:

- Scheduler authority was an operational convention, not a database fact.
- Capacity and worker ownership depended on `gSchedulerOwnedChildren` and lifecycle rows local to each scheduler.
- Recovery discovered workers through command/PID scanning and could fail experiments simply because a restarted scheduler lacked local parentage.
- Self-launch used potentially non-canonical argv spelling, allowing basename `exec` failures and exit 127.

Status commands also contained a legacy path that could backfill `worker_pid` from process scans. That mutation was removed.

The remaining process-local child collection is now used only to reap exit status for direct children. It has no authority over capacity, ownership, orphan classification, or duplicate launches.

## 3. Implemented ownership and worker-accounting architecture

Migration 051 adds:

- `experiment_scheduler_invocation`: immutable PID, PGID, kernel start identity, canonical executable, nonce, command, and historical disposition.
- `experiment_scheduler_lease`: singleton owner, monotonic fencing token, authority state, heartbeat, and expiry.
- `experiment_scheduler_worker_attempt`: immutable work/launch identity, scheduler fence, capacity class, lifecycle state, PID/PGID/start identity, canonical executable, command identity, and reconciliation evidence.
- Active-attempt foreign keys on `experiment` and `experiment_checkpoint_eval`.

The replacement scheduler observes valid prior workers without rewriting their original launch identity. Observation authority is recorded separately.

## 4. Startup, takeover, shutdown, and crash recovery

- Startup creates a historical invocation and locks the singleton lease before any claim, launch, reconciliation, or lifecycle mutation.
- A duplicate invocation receives exit code 3 and a deterministic `SCHEDULER_OWNERSHIP_REJECTED` record. It may add its rejected invocation history, but performs no lifecycle or worker mutation.
- Immediate acquisition is allowed for vacant or explicitly released ownership.
- Abandoned ownership takeover requires both:

  - lease expiration; and
  - proven owner-process absence or immutable identity mismatch.

- A fresh lease rejects takeover even when the owner has already died.
- An expired lease with a still-valid or ambiguously inspectable owner also rejects takeover.
- Lease duration is 90 seconds; refresh occurs before each poll and at most every 30 seconds during long sleeps.
- Graceful SIGINT/SIGTERM stops polling and launching, preserves workers, and releases only the exact invocation/fence.
- Database or fencing loss stops further claims, launches, and destructive reconciliation. A foreign lease cannot be released by the displaced process.
- Crashes require no cleanup for correctness.

## 5. Global capacity rules

Capacity-consuming attempt states are:

`reserved`, `spawned`, `running`, `observed`, `identity_ambiguous`.

Non-consuming terminal states are:

`completed`, `failed`, `launch_failed`, `abandoned`.

Allocation is:

- Train workers → train capacity.
- Final inference, checkpoint inference, and cancellation-authorized inference → infer capacity.
- Final analysis workers → analyze capacity.
- Checkpoint analysis remains synchronous scheduler work, runs under the fenced lease transaction, and is blocked when analyze capacity is exhausted.

Experiment status alone cannot remove a live or ambiguous attempt from capacity. Reservations are counted before process creation, and the lease serializes capacity decisions.

## 6. Orphan and process-identity rules

A worker is valid only when durable evidence matches the observed process:

- PID;
- process group;
- kernel process-start identity;
- canonical executable;
- experiment and phase identity;
- checkpoint identity where applicable;
- scheduler worker-attempt identity for post-051 launches.

PID presence alone is insufficient. PID reuse, executable mismatch, process-group mismatch, command/work mismatch, missing identity, or failed inspection produces `identity_ambiguous`. Such attempts continue consuming capacity and cannot be signaled or destructively reconciled.

Administrative stop was also hardened: it now requires the exact active durable attempt and revalidates all identity components immediately before signaling.

## 7. Interrupted-launch recovery

- Reservation committed, no child created: after the bounded recovery grace, the exact `reserved` attempt becomes `launch_failed` with `never_spawned`; the lifecycle update requires the same active attempt.
- Child created before parent persistence completes: the child idempotently persists its own complete PID/PGID/start/executable/command evidence before waiting on the exec gate. This narrow update cannot refresh the lease or claim work.
- The parent repeats the same fenced persistence before granting exec permission.
- Parent crash before permission closes the gate; the child exits 126, leaving durable spawned/exited evidence rather than being misclassified as never created.
- Exec failure exits 127 and remains tied to the exact attempt. It cannot create an uncontrolled replacement loop.
- Recovery and replay use exact compare-and-update predicates and are idempotent.

## 8. Canonical executable behavior

`_NSGetExecutablePath` is used with buffer retry, followed by `realpath`, absolute-path validation, and `X_OK`.

Every scheduler-created train, infer, analyze, checkpoint-infer, and cancellation worker uses `execv` with this canonical path. `PATH`, CWD, basename argv, symlink spelling, and Screen state are not launch authority.

A symlink invocation intentionally resolves to its canonical target. The live process test launched the scheduler from another directory through a basename symlink, created a real disposable child, and verified:

- canonical persisted path;
- canonical child command;
- registered worker attempt;
- no exit 127.

## 9. Database migration

[051_scheduler_ownership_and_worker_attempts.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql>) is transactional and replay-idempotent.

It includes:

- Constraints for authority states, attempt shapes, work classes, and terminal evidence.
- Unique partial indexes preventing concurrent active attempts for an experiment phase or checkpoint evaluation.
- Capacity and scheduler-history indexes.
- Foreign keys with restricted deletion.
- Runtime grants.
- Conservative conversion of existing active rows to `legacy_unverified` / `identity_ambiguous` attempts.

Ambiguous legacy rows are not asserted to be owned or alive. They consume capacity until exact evidence resolves them.

Migration 050’s `current_operation` contract remains unchanged: only `train`, `infer`, and `analyze`.

## 10. Files changed

Core implementation:

- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
- [GlobalExperimentControl.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/GlobalExperimentControl.cpp>)
- [main.cpp](</Volumes/Developer SSD/ExpertAdvisor/LSTM/main.cpp>)
- [ExperimentScheduler.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/ExperimentScheduler.hpp>)
- [SchedulerExecutablePath.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/SchedulerExecutablePath.hpp>)
- [SchedulerOwnershipPolicy.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/SchedulerOwnershipPolicy.hpp>)

Migration and tests:

- `Database/migrations/051_scheduler_ownership_and_worker_attempts.sql`
- `Tests/SchedulerCanonicalPathTests.sh`
- `Tests/SchedulerOwnershipIntegrationTests.sh`
- `Tests/SchedulerOwnershipMigrationTests.sql`
- `Tests/SchedulerOwnershipPolicyTests.cpp`
- `Tests/SchedulerOwnershipProcessIntegrationTests.sh`
- `Tests/GlobalExperimentControlIntegrationTests.sh`

Documentation:

- `Database/README.md`
- `docs/architecture/Volume_VII_Experiment_Lifecycle.md`
- `docs/architecture/Volume_XI_Scheduler.md`
- `docs/architecture/adr/ADR-0016-scheduler-atomic-claim-hardening.md`

## 11. Tests and build commands

Passed:

```bash
Tests/SchedulerCanonicalPathTests.sh
Tests/SchedulerOwnershipIntegrationTests.sh
Tests/SchedulerOwnershipProcessIntegrationTests.sh
```

The process integration independently covers four live worker classes, duplicate rejection, non-mutating rejection, status counts, graceful release, lease loss, foreign fencing, scheduler crash, fresh-lease rejection, expired/dead takeover, never-spawned recovery, global capacity, canonical launch, and exit-127 regression.

Broader compatibility:

```bash
Tests/GlobalExperimentControlIntegrationTests.sh \
  "$PWD/DerivedData/SchedulerOwnership/Build/Products/Release/LSTM_Release" \
  "$PWD/DerivedData/SchedulerOwnership/Tests/GlobalExperimentControlProcessTests"
```

Result: `GlobalExperimentControlIntegrationTests passed`, including crash-window, pause/resume/cancellation, selective resume, checkpoint race, current-operation, and scheduler-restart compatibility.

ASan/UBSan-backed tests passed:

- `SchedulerOwnershipPolicyTests`
- `SchedulerChildStatusTests`
- `GlobalExperimentControlTests`
- `GlobalExperimentControlProcessTests`
- `ContinuationPolicyInheritanceTests`
- `CampaignOperationsTests`

Leak detection was disabled because Apple’s ASan runtime reports it unsupported.

Final isolated build:

```bash
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/SchedulerOwnership \
  build
```

Result: `** BUILD SUCCEEDED **`.

Also passed:

```bash
bash -n Tests/*.sh
git diff --check
```

## 12. Exact results

- Release compilation and linking succeeded.
- Migration 051 applied twice successfully in a disposable database.
- Conservative legacy conversion and active-attempt linking passed.
- Duplicate active attempts were rejected by database constraints.
- Foreign lease refresh/release predicates affected zero rows.
- Four prior-scheduler worker classes were observed without lifecycle failure or relaunch.
- Global status reported train=1, infer=2, analyze=1 for the test fixture.
- Exact owner death plus fresh lease rejected takeover.
- Expired lease plus owner identity failure incremented the fence and permitted takeover.
- Forced lease displacement caused scheduler exit code 4 and did not release the foreign lease.
- A real canonical child launch through basename/symlink/CWD indirection did not return 127.
- No disposable scheduler/worker processes remained after final cleanup.
- Production scheduler PID 94420 and its observed production workers remained untouched.

## 13. Deferred verification and remaining risks

- Migration 051 was not applied to production.
- The corrected scheduler was not started against production.
- Production status was not invoked because the running executable predates this correction and its old status path could write metadata.
- Actual adoption of the currently running production workers remains a deployment verification step.
- Independent CEE review remains pending as requested.
- The build emits the repository’s existing libpqxx 7.10 `exec_params` deprecation backlog and pre-existing warnings—316 warnings in the final compile. Removing that repository-wide API backlog would materially exceed this bounded scheduler correction, but it remains technical debt under the project’s warning policy.

## 14. Production deployment prerequisites and ordering

1. Build and retain the corrected executable at a new path; do not overwrite the executable used by live processes.
2. Stop only the pre-051 scheduler and verify its immutable process identity has exited. Preserve live workers.
3. Back up the production database.
4. Confirm migration 050 is applied and no unsupported `current_operation` values exist.
5. Apply migration 051 as the administrative migration role while no pre-051 scheduler can claim work.
6. Inspect the conservative legacy attempts before starting dispatch.
7. Start exactly one corrected scheduler.
8. Verify lease owner/fence/freshness, global capacity, prior-worker observation, unresolved legacy attempts, and canonical executable through scheduler status.
9. Do not permit replacement dispatch until all ambiguous legacy live workers are either exactly observed or deterministically reconciled.

Applying migration 051 while an old scheduler remains able to claim work is unsupported because that executable does not honor the lease.

## 15. `git status --short`

```text
 M Database/README.md
 M Headers/ExperimentScheduler.hpp
 M LSTM/main.cpp
 M Sources/ExperimentScheduler.cpp
 M Sources/GlobalExperimentControl.cpp
 M Tests/GlobalExperimentControlIntegrationTests.sh
 M docs/architecture/Volume_VII_Experiment_Lifecycle.md
 M docs/architecture/Volume_XI_Scheduler.md
 M docs/architecture/adr/ADR-0016-scheduler-atomic-claim-hardening.md
?? Database/migrations/051_scheduler_ownership_and_worker_attempts.sql
?? Headers/SchedulerExecutablePath.hpp
?? Headers/SchedulerOwnershipPolicy.hpp
?? Tests/SchedulerCanonicalPathTests.sh
?? Tests/SchedulerOwnershipIntegrationTests.sh
?? Tests/SchedulerOwnershipMigrationTests.sql
?? Tests/SchedulerOwnershipPolicyTests.cpp
?? Tests/SchedulerOwnershipProcessIntegrationTests.sh
```

## 16. `git diff --stat`

`git diff --stat` reports tracked files only; the new untracked files are listed above.

```text
 Database/README.md                                 |   12 +
 Headers/ExperimentScheduler.hpp                    |    1 +
 LSTM/main.cpp                                      |   40 +-
 Sources/ExperimentScheduler.cpp                    | 3084 ++++++++++++++++++--
 Sources/GlobalExperimentControl.cpp                |   12 +-
 Tests/GlobalExperimentControlIntegrationTests.sh   |   45 +-
 .../Volume_VII_Experiment_Lifecycle.md             |    7 +
 docs/architecture/Volume_XI_Scheduler.md           |  119 +-
 .../ADR-0016-scheduler-atomic-claim-hardening.md   |    7 +
 9 files changed, 3033 insertions(+), 294 deletions(-)
```