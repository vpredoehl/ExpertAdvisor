# Scheduler Ownership Coordinated Remaining Defects
## Independent CEE Verification, Commit-Curation Review, and Cutover Report

Date: 2026-07-30
Repository: `/Volumes/Developer SSD/ExpertAdvisor`
Reviewer mode: independent, high-reasoning, fail-closed
Final transition reached: **STOP GATE 1**

## 1. Executive summary

The implementation is not commit-ready or production-ready. Production cutover
was not performed.

Independent source tracing and a disposable PostgreSQL reproduction found a
material exact-attempt defect in the production stop-at-checkpoint path. A
scheduler-owned train worker can persist its ordinary checkpoint-stop
transition from `running/train` to `pending/infer` or `pending/analyze` while
leaving its active train attempt bound and capacity-consuming. The scheduler
reaper subsequently rejects that attempt because the lifecycle is neither the
expected active `running/train` row nor terminal. Orphan recovery cannot clear
the attempt because the lifecycle binding still exists, and the next phase
cannot be claimed because the active-attempt column is not null. This can strand
capacity and the experiment indefinitely.

Scenario 37, stop-at-checkpoint compatibility, is therefore only partial. Its
test calls the checkpoint transition without a generation-52 durable attempt
and never asserts reaper finalization, binding clearance, capacity release, or
next-phase claimability. The claimed 40/40 coverage is not substantiated:
the independent result is **39 full, 1 partial, 0 missing, 0 manual-only**.

A fresh warning-as-error compilation also fails on an unused lambda capture in
the changed global-control source. The isolated Xcode Release build succeeds
but produces hundreds of warnings, contrary to the repository rule that
compiler warnings are defects.

No process was signaled. No production database row was changed. Nothing was
staged or committed. No migration, generation-52 cutover, corrected scheduler
launch, duplicate-scheduler probe, or legacy reconciliation was attempted.

During validation, the pathname used by the live scheduler disappeared from
disk while the scheduler and its seven workers remained alive from their
mapped executable vnode. The isolated build was directed to a distinct
DerivedData path, but the shared production DerivedData directory was modified
during the same validation interval. Causation was not proven. This is an
additional fail-closed condition: the exact running image can no longer be
re-checksummed through its recorded pathname, and the old scheduler cannot be
restarted from that pathname if it exits.

## 2. Independent review verdict

**FAIL — material ownership/lifecycle defect.**

The failure is independent of the implementation reports:

1. `RecordCheckpointStopReached` changes the experiment to a pending next phase
   and clears PID mirrors, but does not terminalize or clear the active attempt
   ([Sources/GlobalExperimentControl.cpp](Sources/GlobalExperimentControl.cpp),
   lines 2932-2992).
2. Migration 052 deliberately validates an unchanged attempt against
   `OLD.phase`, so the phase change is accepted while retaining the train
   attempt
   ([Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql](Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql),
   lines 267-304).
3. Exact-attempt verification accepts only an active `running` lifecycle in the
   required phase, or an allowed terminal lifecycle
   ([Sources/SchedulerOwnershipRepository.hpp](Sources/SchedulerOwnershipRepository.hpp),
   lines 258-275).
4. The child reaper verifies that predicate before finalization and discards
   its local child record when verification fails
   ([Sources/ExperimentScheduler.cpp](Sources/ExperimentScheduler.cpp),
   lines 16061-16149).
5. Missing-process recovery requires `running` in the original phase. Its
   lifecycle-changed path only terminalizes an attempt if no lifecycle row
   remains bound, and it does not require an affected row before counting the
   case reconciled
   ([Sources/ExperimentScheduler.cpp](Sources/ExperimentScheduler.cpp),
   lines 15150-15199).

Disposable reproduction after migrations 051 and 052:

```text
after_worker_checkpoint_transition:
  pending | infer | attempt 1 | lifecycle_phase train |
  lifecycle_state identity_ambiguous | capacity train
reaper_exact_predicate_matches = false
orphan_detached_terminalization_rows = 0
next_phase_claim_rows = 0
train_capacity_still_consumed = 1
```

Additional validation defect:

```text
Sources/GlobalExperimentControl.cpp:1544:10:
error: lambda capture 'target' is not used
[-Werror,-Wunused-lambda-capture]
```

No implementation source was patched.

## 3. Exact commit-curation inventory

### Tracked modified implementation candidates

```text
Database/README.md
Headers/ExperimentScheduler.hpp
LSTM/main.cpp
Sources/ExperimentScheduler.cpp
Sources/GlobalExperimentControl.cpp
Sources/GlobalExperimentControl.hpp
Tests/GlobalExperimentControlIntegrationTests.sh
Tests/GlobalExperimentControlProcessTests.cpp
docs/architecture/Volume_VII_Experiment_Lifecycle.md
docs/architecture/Volume_XI_Scheduler.md
docs/architecture/adr/ADR-0016-scheduler-atomic-claim-hardening.md
docs/architecture/adr/README.md
```

### Intended new implementation/migration/test/documentation candidates

```text
Database/migrations/051_scheduler_ownership_and_worker_attempts.sql
Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql
Headers/SchedulerExecutablePath.hpp
Headers/SchedulerOwnershipPolicy.hpp
Sources/SchedulerOwnershipRepository.hpp
Tests/SchedulerCanonicalPathTests.sh
Tests/SchedulerContinuationOwnershipIntegrationTests.sh
Tests/SchedulerOwnershipIntegrationTests.sh
Tests/SchedulerOwnershipMigrationTests.sql
Tests/SchedulerOwnershipPolicyTests.cpp
Tests/SchedulerOwnershipProcessIntegrationTests.sh
Tests/fixtures/scheduler_protocol_cutover_bin/ps
docs/architecture/adr/ADR-0018-scheduler-generation-52-exact-attempt-authority.md
```

These are only candidate inventory items. They must not be staged until the
material defect is corrected and the complete validation is rerun.

### Review reports to retain as evidence, separate from an implementation commit

```text
Scheduler_Ownership_Coordinated_Remaining_Defects_Implementation_Output.md
Scheduler_Restart_MultiInstance_WorkerOwnership_Architectural_Correction_Implementation_Output.md
Scheduler_Restart_MultiInstance_WorkerOwnership_Correctness_Implementation_Output.md
Scheduler_Restart_MultiInstance_WorkerOwnership_Focused_Independent_CEE_Review_Output.md
Scheduler_Ownership_Coordinated_Remaining_Defects_Independent_CEE_Cutover_Output.md
```

The first four reports contain historical claims and must not be treated as
implementation proof. This report records the current independent blocker.
Repository precedent places retained review reports in dedicated review
artifact directories; no relocation was attempted during a failed review.

### Excluded from any implementation commit

- All `*_Transcript.txt` files: historical audit input, ignored by repository
  policy, preserve locally but do not commit with implementation.
- `watch_20260729-142309`: external watch artifact.
- `DerivedData/`: generated build and sanitizer artifacts.
- Production logs, checkpoints, models, dumps, temporary databases, and worker
  state.
- Any unrelated historical prompts or reports not intentionally selected in a
  later review-artifact commit.

## 4. Git commit or deferred commands

No files were staged and no commit was created. This was controlled by STOP
GATE 1, not by a policy requiring operator-owned commits. Because the current
source is not commit-ready, no `git add`/`git commit` command is authorized or
recommended.

## 5. Source commit/worktree identity

```text
branch: campaign-operations
HEAD: 939e1265f47d57d131c217274026ad7e479856e6
HEAD subject: Harden current_operation canonicalization and migration safety
tracked binary-diff SHA-256:
f67fd90b1d642995cc148661184138be53a9eb3627322d6affb412291e6dfc9a
```

The branch differs from the `phase6` branch stated in `AGENTS.md`; this was not
changed.

Reviewed migration hashes:

```text
050 489e72eaa742683bf8552c85149232794e50c25a1802588afdbfa00ebaae417b
051 d9ab9387a7eefcfa31f90b9f6dc666906818e1a7fc2d103b3e6dfd3bfd70147f
052 e0c000eec73fd125dca1cd2cfc6f2d2cd471ba576ba72aba4377fc316a9a1a9c
```

The actual 052 filename is
`052_scheduler_protocol_and_exact_attempt_hardening.sql`.

## 6. Isolated build command, path, canonical path, and checksum

Review build command:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme 'LSTM Release' \
  -configuration Release \
  -derivedDataPath DerivedData/CEEVerificationGeneration52 \
  build
```

Result: `** BUILD SUCCEEDED **`, with substantial warnings.

```text
canonical path:
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/CEEVerificationGeneration52/Build/Products/Release/LSTM_Release
mode: -rwxr-xr-x
size: 6190800
SHA-256: 5f8faa08ee108d497b66c2ab3a304f0905551b67af3bc486b7679d9b4c7334dd
```

`--help` completed and printed usage without scheduling or production database
mutation.

This is a review candidate, not an approved corrected production executable.
At baseline, the on-disk old pathname hashed as:

```text
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
SHA-256: 0de622b981457129391f56df8c782ec699838c860629440afcd4c3f4eee65754
```

By final verification that pathname no longer existed, although `lsof` still
showed the live process mapped to inode 4147527, size 6079152, under that name.
The baseline pathname checksum therefore cannot now be reverified and was not
proven to describe the mapped running inode. STOP GATE 2 would also fail.

## 7. Production baseline process inventory

Scheduler:

```text
PID: 94420
PPID: 94419
PGID: 94419
kernel start identity: 1785092342:726185
start: Sun Jul 26 13:59:02 2026
canonical executable:
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
command:
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
  --schedule-experiments
  --max-train-procs=7
  --max-infer-procs=5
  --max-analyze-procs=1
  --scheduler-poll-seconds=30
  --scheduler-log-dir=/Volumes/Developer SSD/ExpertAdvisor/experiment_logs
```

Parent PID 94419 is a `login -pflq vjp /usr/bin/env ...` process with PGID
94419. The scheduler environment reported
`STY=94418.expertadvisor_scheduler`; `screen -ls` reported no socket.

Live workers:

| Experiment | PID | PPID | PGID | Kernel start identity | Phase |
|---:|---:|---:|---:|---|---|
| 460 | 67171 | 94420 | 67171 | `1785393183:547983` | train |
| 461 | 67818 | 94420 | 67818 | `1785393655:385605` | train |
| 462 | 71349 | 94420 | 71349 | `1785396431:475546` | train |
| 463 | 38330 | 94420 | 38330 | `1785450197:630969` | train |
| 482 | 38400 | 94420 | 38400 | `1785450231:746484` | train |
| 483 | 39259 | 94420 | 39259 | `1785450914:106400` | train |
| 484 | 39540 | 94420 | 39540 | `1785451144:205933` | train |

Each worker had its own PGID, exact `--scheduler-experiment-id`, and the same
old executable path. No worker was signaled.

An automatic restart configuration exists:

```text
label: com.vjp.lstm.scheduler
state: spawn scheduled
active count: 0
program:
/Users/vjp/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
working directory: /Users/vjp/ExpertAdvisor
last exit: 78 EX_CONFIG
configured limits: train 10, infer 9, analyze 9
```

Its configured executable does not exist. It is nevertheless an active
automatic-restart configuration and would have to be safely disabled and
reverified before a future migration.

## 8. Production baseline database inventory

All queries were inside read-only transactions.

```text
database: LSTM
role: pqxx
schema_migrations: 46 rows
maximum migration: 049
050 present: no
051 present: no
052 present: no
ownership/protocol tables: absent
```

Experiment totals by state:

```text
cancelled: 45
completed: 244
failed: 14
pending: 4
running: 7
total: 314
```

The seven running rows were all `running/train/train`, had the exact worker
PID/PGID/start/executable/command mirrors listed above, and had epochs:

```text
460 62/80
461 62/80
462 59/80
463 12/80
482 12/80
483 12/80
484 12/80
```

Checkpoint evaluations:

```text
completed/done: 231
failed/infer: 36
running: 0
total: 267
```

Other baselines:

```text
global desired_state: running
active admin request: NULL
current pause request: NULL
global revision: 8
admin requests: cancel_all 1, pause_all 1, resume_all 2 (all completed)
admin worker outcomes: 13 completed experiment outcomes
continuation decisions:
  continuation_queued 82
  eligible 5
  insufficient 1
  rejected_threshold 8
  rejected_trend 1
models: 1021
inference results: 469
analysis results: 475
noncanonical current_operation rows: 0
```

Migration 050 being absent is an independent Stage 6 prerequisite failure.

## 9. Backup command and result

No production backup command was run. STOP GATE 1 prohibited progression to
Stage 3. No production database mutation followed, so there is no new backup
path, size, checksum, or `pg_restore --list` result to report.

## 10. Exact old scheduler identity verification

The baseline identity record is in section 7. It was sufficient for read-only
observation but was never used to authorize a signal. Final verification still
found PID 94420 with the same PPID, PGID, start time, and command. The executable
pathname disappearance makes executable-image checksum correlation incomplete.

## 11. Exact stop command and result

Not executed. No signal of any kind was sent to PID 94420, its process group,
or any worker.

## 12. Proof no scheduler-dispatch process remained

Not applicable: the old scheduler intentionally remains active because Gate 1
failed. PID 94420 still owns the observed scheduler-dispatch command.

## 13. Migration 051 execution and verification

Production: not executed.

Disposable database: applied successfully and replayed successfully through
the ownership migration tests. SHA-256 is recorded in section 5.

## 14. Migration 052 execution and verification

Production: not executed.

Disposable database: applied successfully and replayed successfully. The
generation guard rejects scheduler claims without generation-52 session
authority after completed cutover. The stop-at-checkpoint reproduction also
proved that the exact-active-attempt trigger permits the defective unchanged
attempt binding across the phase transition.

## 15. Generation-52 cutover command and verification

Not executed against production. Production has no protocol table and remains
at migration 049.

## 16. Corrected scheduler launch command and identity

Not executed. No corrected scheduler process or durable invocation exists.

## 17. Duplicate corrected scheduler rejection

Not executed against production.

The disposable process integration suite verified exit 3,
`SCHEDULER_OWNERSHIP_REJECTED`, `mutations=0`, and unchanged active-attempt
counts for a second scheduler.

## 18. Lease, fence, protocol, invocation, and heartbeat verification

Production verification is impossible because migrations 051/052 were not
applied and the four ownership/protocol tables are absent. There is no
generation-52 invocation, fence, lease, or heartbeat.

## 19. Worker-attempt and capacity verification

Production has no durable attempt table. The seven live workers therefore
cannot be represented in generation-52 attempts or durable generation-52
capacity accounting. No adoption was attempted.

Disposable ownership/capacity tests passed, except for the independently
reproduced stop-at-checkpoint path, which remains capable of stranding train
capacity.

## 20. Legacy no-PID inventory and reconciliation

Production has no migration-051 attempt table, so no legacy no-PID attempt
inventory exists yet. No reconciliation was performed.

Disposable tests verified the bounded, fail-closed legacy path under their
fixtures. That does not authorize production reconciliation.

## 21. Pre/post identity table for surviving workers

There was no cutover. The table compares Stage-1 baseline with the final
read-only observation:

| Experiment | Pre PID/PGID/start | Final PID/PGID/start | Result |
|---:|---|---|---|
| 460 | `67171/67171/1785393183:547983` | same | alive, unchanged |
| 461 | `67818/67818/1785393655:385605` | same | alive, unchanged |
| 462 | `71349/71349/1785396431:475546` | same | alive, unchanged |
| 463 | `38330/38330/1785450197:630969` | same | alive, unchanged |
| 482 | `38400/38400/1785450231:746484` | same | alive, unchanged |
| 483 | `39259/39259/1785450914:106400` | same | alive, unchanged |
| 484 | `39540/39540/1785451144:205933` | same | alive, unchanged |

This proves survival during the review window, not post-cutover adoption.

## 22. Proof of no replacement or duplicate worker

No corrected scheduler was launched. Final process and database inventories
showed the same seven PID/PGID/start identities and the same seven running
experiment rows. No additional production worker was observed.

## 23. Current-operation canonicalization result

Production query:

```text
current_operation NOT NULL and not in (train,infer,analyze): 0 rows
```

Migration 050 canonical operation values remain exactly `train`, `infer`, and
`analyze`, but migration 050 is not yet recorded in production.

## 24. Forty-scenario independent traceability

The classifications below use the original required scenario order, not the
renumbered implementation-report table.

| # | Required scenario | Verdict | Independently traced evidence |
|---:|---|---|---|
| 1 | Restart with live train | Full | Durable-attempt migration fixture plus live-process recovery/adoption assertions |
| 2 | Restart with live infer | Full | Mixed attempt fixture and status/capacity assertions |
| 3 | Restart with live analyze | Full | Analyze attempt, interruption, and recovery assertions |
| 4 | Restart with live checkpoint-infer | Full | Checkpoint attempt uniqueness, lifecycle binding, and capacity assertions |
| 5 | Duplicate scheduler while lease valid | Full | Second process exit 3, rejection diagnostic, zero mutation |
| 6 | Duplicate rejection is nonmutating | Full | Invocation/attempt counts asserted unchanged |
| 7 | Expired dead-owner takeover | Full | Exact owner-death plus expiry process assertions |
| 8 | No takeover of valid owner | Full | Fresh live owner prevents takeover |
| 9 | Scheduler crash with live workers | Full | Restart process and durable observation fixtures |
| 10 | Graceful scheduler shutdown with live workers | Full | Shutdown ownership policy and process lifecycle assertions |
| 11 | PID reuse/start mismatch | Full | Real process mismatch stays ambiguous and capacity-consuming |
| 12 | Executable mismatch | Full | Real process canonical-executable mismatch assertions |
| 13 | Command/experiment mismatch | Full | Exact command/work identity mismatch assertions |
| 14 | Valid prior worker adoption | Full | Exact process identity becomes observed without relaunch |
| 15 | Invalid worker/orphan reconciliation | Full | Missing-process exact lifecycle and result/no-result recovery assertions |
| 16 | Ambiguous identity fails closed | Full | No signal; binding and capacity retained |
| 17 | Train capacity accounting | Full | Migration, policy, status, and process assertions |
| 18 | Infer capacity accounting | Full | Mixed capacity/status assertions |
| 19 | Analyze capacity accounting | Full | Short claim/work/finalize and status assertions |
| 20 | Checkpoint capacity accounting | Full | Checkpoint infer/analyze attempt fixtures |
| 21 | Mixed accounting | Full | Status asserts train=1, infer=2, analyze=1 in fixture |
| 22 | Concurrent last slot | Full | Reservation uniqueness/capacity policy assertions |
| 23 | Duplicate experiment launch | Full | Active-attempt uniqueness and claim predicates |
| 24 | Duplicate checkpoint launch | Full | Checkpoint active-attempt unique constraint assertions |
| 25 | Claim succeeds but no spawn | Full | Reserved attempt consumes capacity and bounded recovery |
| 26 | Child spawned, parent crashes before persist | Full | Gated launch identity/crash boundary and exact recovery |
| 27 | Exit 127/126 launch failure | Full | Canonical exec test plus interrupted-launch exit-126 recovery |
| 28 | Canonical executable from foreign CWD | Full | Real basename-symlink worker launched from different CWD |
| 29 | Basename invocation | Full | Canonical path shell tests and process test |
| 30 | Symlink invocation | Full | Direct/symlink/basename equality assertions |
| 31 | Lease loss during work | Full | Fence displacement stops before next claim; guarded release |
| 32 | Foreign refresh/release | Full | Invocation/fence predicates reject foreign authority |
| 33 | Destructive exact predicate | Full | Stale-reaper replacement race leaves newer attempt intact |
| 34 | Replay idempotency | Full | 051/052 direct DDL replay and cutover replay assertions |
| 35 | Interrupted recovery | Full | Three checkpoint-analysis interruption boundaries |
| 36 | Global pause/resume/cancel | Full | Isolated DB integration and process tests |
| 37 | Stop-at-checkpoint compatibility | **Partial** | Transition test omits generation-52 attempt/reaper/capacity/next-phase assertions; disposable reproduction fails |
| 38 | Cancellation inference | Full | Checkpoint cancellation ownership/replay/race assertions |
| 39 | Continuation and checkpoint-inference | Full | Continuation authority/replay and ordinary attempt-path integration |
| 40 | Status ownership counts | Full | Ownership state and per-capacity status output asserted |

Result: **39 full; 1 partial; 0 missing; 0 manual-only.**

## 25. Tests, builds, and audits executed

Passed:

```text
bash -n Tests/*.sh
git diff --check
git diff --cached --check (empty staging area)
Tests/SchedulerOwnershipIntegrationTests.sh
Tests/SchedulerCanonicalPathTests.sh
Tests/SchedulerOwnershipProcessIntegrationTests.sh \
  <isolated LSTM_Release> \
  DerivedData/CEEReviewSchedulerOwnership/Tests/GlobalExperimentControlProcessTests
Tests/SchedulerContinuationOwnershipIntegrationTests.sh <isolated LSTM_Release>
Tests/GlobalExperimentControlIntegrationTests.sh \
  <isolated LSTM_Release> \
  DerivedData/CEEReviewSchedulerOwnership/Tests/GlobalExperimentControlProcessTests
fresh ASan/UBSan SchedulerOwnershipPolicyTests
fresh ASan/UBSan SchedulerChildStatusTests
fresh ASan/UBSan ContinuationPolicyInheritanceTests
isolated Xcode Release build
safe isolated --help
```

Qualification: the global-control process helper used by the shell integration
suites was the pre-existing CEEReviewSchedulerOwnership ASan build. A fresh
current-source helper could not be produced because the current changed source
fails warning-as-error compilation.

Failed:

```bash
clang++ -std=c++20 -Wall -Wextra -Werror \
  -Wno-deprecated-declarations -IHeaders \
  $(pkg-config --cflags libpqxx) \
  -c Sources/GlobalExperimentControl.cpp \
  -o DerivedData/CEEVerificationGeneration52/ManualAudit/GlobalExperimentControl.o
```

Result: unused `target` lambda capture at line 1544.

The ordinary Xcode build produced 74 warnings in the main compile and 366 in
the scheduler compile, primarily libpqxx deprecations, plus project-source
warnings. The project instructions classify warnings as defects.

Mechanical audits:

```text
process-control audit: every fork/exec/waitpid/kill/killpg occurrence enumerated
authority/lease/fence audit: 115 matching source references inspected
FOR UPDATE/advisory-lock audit: 60 matching references inspected
```

No disposable test database remained. No disposable scheduler or worker
process remained.

## 26. Deviations, blockers, and stopped gates

Primary blocker: exact-attempt/capacity leak on ordinary stop-at-checkpoint.

Secondary blocker: fresh warning-as-error compilation fails.

Additional safety conditions:

- Production migration 050 is absent.
- The current branch is `campaign-operations`, not the documented `phase6`.
- A launchd scheduler KeepAlive/RunAtLoad configuration remains spawn-scheduled.
- The live scheduler executable pathname disappeared during validation.
- The isolated build succeeded with many warnings.

The process integration suite's process helper was not rebuilt from current
source because that compile failed.

## 27. Residual risks

- A stop-at-checkpoint event under generation 52 can strand one train attempt,
  one train capacity slot, and the experiment's next phase.
- The exact bytes of the currently mapped old scheduler/worker image are no
  longer checksum-verifiable through the missing pathname.
- If the old scheduler exits, its recorded command cannot restart from the
  missing executable.
- The spawn-scheduled launchd configuration can interfere with a future
  no-dispatch proof even though its current program path is missing.
- Production has no durable ownership data until migrations are safely applied;
  live-worker conversion remains unobserved in production.
- Cutover absence evidence is only tested under disposable fixtures; future
  operations must still positively inspect wrappers, launchd, Screen, cron, and
  exact process identities.

## 28. Commit-readiness verdict

**Not ready.** Do not stage or commit this implementation until the
stop-at-checkpoint ownership boundary and warning-as-error failure are corrected
and all 40 scenarios pass independent current-source validation.

## 29. Production-readiness verdict

**Not ready. No cutover occurred.** Production remains on migration 049 with
the old scheduler and seven old workers running.

## 30. Exact current production scheduler command

```text
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release --schedule-experiments --max-train-procs=7 --max-infer-procs=5 --max-analyze-procs=1 --scheduler-poll-seconds=30 --scheduler-log-dir=/Volumes/Developer SSD/ExpertAdvisor/experiment_logs
```

PID 94420 remained alive at final observation. The executable pathname in this
command was absent from disk at final verification.

## 31. Rollback boundary and prohibited actions

Database rollback boundary: no production migration was applied; production is
still at migration 049. No database rollback is required.

Do not:

- stop the current scheduler based on this report alone;
- signal any worker;
- attempt migrations 051/052 or generation-52 cutover;
- start the isolated review binary against production;
- restart the old scheduler from an unverified or recreated binary;
- overwrite the missing old executable pathname;
- use `pkill`, `killall`, group signaling, Screen destruction, or ad hoc SQL;
- alter production lifecycle/attempt rows to mask the defect.

If the current old scheduler exits before a corrected review, leave scheduling
stopped and preserve the live workers. Recovering or recreating the old binary
requires a separate, explicit, checksum-verified operational decision.

After any future generation-52 cutover, the old scheduler must never be
restarted.

## 32. `git status --short`

Before creation of this report:

```text
 M Database/README.md
 M Headers/ExperimentScheduler.hpp
 M LSTM/main.cpp
 M Sources/ExperimentScheduler.cpp
 M Sources/GlobalExperimentControl.cpp
 M Sources/GlobalExperimentControl.hpp
 M Tests/GlobalExperimentControlIntegrationTests.sh
 M Tests/GlobalExperimentControlProcessTests.cpp
 M docs/architecture/Volume_VII_Experiment_Lifecycle.md
 M docs/architecture/Volume_XI_Scheduler.md
 M docs/architecture/adr/ADR-0016-scheduler-atomic-claim-hardening.md
 M docs/architecture/adr/README.md
?? Database/migrations/051_scheduler_ownership_and_worker_attempts.sql
?? Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql
?? Headers/SchedulerExecutablePath.hpp
?? Headers/SchedulerOwnershipPolicy.hpp
?? Scheduler_Ownership_Coordinated_Remaining_Defects_Implementation_Output.md
?? Scheduler_Restart_MultiInstance_WorkerOwnership_Architectural_Correction_Implementation_Output.md
?? Scheduler_Restart_MultiInstance_WorkerOwnership_Correctness_Implementation_Output.md
?? Scheduler_Restart_MultiInstance_WorkerOwnership_Focused_Independent_CEE_Review_Output.md
?? Sources/SchedulerOwnershipRepository.hpp
?? Tests/SchedulerCanonicalPathTests.sh
?? Tests/SchedulerContinuationOwnershipIntegrationTests.sh
?? Tests/SchedulerOwnershipIntegrationTests.sh
?? Tests/SchedulerOwnershipMigrationTests.sql
?? Tests/SchedulerOwnershipPolicyTests.cpp
?? Tests/SchedulerOwnershipProcessIntegrationTests.sh
?? Tests/fixtures/
?? docs/architecture/adr/ADR-0018-scheduler-generation-52-exact-attempt-authority.md
?? watch_20260729-142309
```

This report is now an additional untracked file.

## 33. `git log -1 --oneline`

```text
939e126 Harden current_operation canonicalization and migration safety
```

No new commit was made.

## 34. Diff statistics

Unstaged tracked diff:

```text
12 files changed, 6532 insertions(+), 986 deletions(-)
```

Staged diff: empty.

Untracked files, including migrations/tests/docs/reports, are not represented
in `git diff --stat`.

## Explicit answers to final questions

- Independently verified without relying on its own report? **Yes; verdict is
  failure based on source tracing, current-source compilation, disposable
  database reproduction, process inspection, and read-only production SQL.**
- All 40 scenarios fully covered? **No. 39 full, scenario 37 partial.**
- Worktree curated to an intentional commit? **Inventory produced, but no:
  material defects prohibit staging/commit.**
- Corrected executable built at a distinct canonical path? **A distinct review
  candidate was built; it is not approved as corrected or production-ready.**
- Verified production backup created before mutation? **No production database
  or process mutation occurred; Gate 1 stopped before backup stage.**
- Only the exact old scheduler stopped? **No scheduler was stopped.**
- Did all existing workers survive? **Yes during the review window; no cutover
  occurred.**
- Absence of scheduler-dispatch authority positively proven? **No; old
  scheduler PID 94420 remains active.**
- Migrations 051 and 052 applied transactionally and correctly? **Only in
  disposable databases, not production.**
- Protocol generation 52 active? **No.**
- Database technically blocks old scheduler dispatch? **Disposable migration
  tests show the generation guard; production does not have it installed.**
- Exactly one corrected scheduler active? **No corrected scheduler is active.**
- Invocation/fence/heartbeat coherent? **Not applicable; production tables are
  absent.**
- Existing workers represented exactly once in durable attempts/capacity?
  **No; production is pre-051.**
- Any workers replaced or duplicated? **No during this review.**
- Ambiguous identities non-signalable? **The tested implementation enforces
  this in covered paths; no production signal was attempted.**
- Can legacy no-PID attempts strand capacity indefinitely? **The documented
  legacy path is bounded in covered fixtures, but the non-legacy
  stop-at-checkpoint defect can strand capacity indefinitely.**
- Repository ready to commit? **No.**
- Production scheduler ready for normal operation after cutover? **No cutover
  occurred; the existing old scheduler remains active, with its pathname
  missing.**
- Exact condition preventing completion? **STOP GATE 1: material
  stop-at-checkpoint exact-attempt/capacity leak, scenario 37 partial, and
  current-source warning-as-error validation failure.**
