---
title: "Scheduler Restart, Multi-Instance, Worker Ownership, and Durable Attempt Architectural Correction"
document_type: "implementation report"
status: "final"
reasoning_effort: "high"
date: "2026-07-29"
---

# Scheduler ownership architectural correction

## 1. Executive summary

This increment corrects the remaining coupled scheduler defects around fence
propagation, exact durable-attempt authority, process control, child exit
handling, direct CLI execution, checkpoint analysis, lock order, mixed-version
cutover, and legacy no-PID reconciliation.

The implementation uses one generation-52 authority model:

- Every authoritative scheduler transaction revalidates the exact invocation
  ID and fencing token.
- Every managed child or in-process checkpoint analysis has a durable attempt
  and capacity reservation before work begins.
- Signals, reaping, reconciliation, lifecycle clearing, and capacity release
  resolve and mutate the exact attempt/lifecycle binding.
- Checkpoint analysis is split into short claim and finalize transactions with
  work outside database locks.
- Corrected startup remains blocked until positive mixed-version cutover
  evidence is durably complete.
- Legacy no-PID rows remain fail-closed during a bounded grace period, then
  reconcile only with exact binding and positive process-absence evidence.

All production and integration validation used isolated DerivedData and
disposable databases. Production scheduler PID 94420 and its workers were
observed for safety but were not stopped, signaled, migrated, or mutated.

## 2. Confirmed defect model

Repository inspection confirmed the independent review's defects:

1. Continuation helpers reconstructed scheduler options without preserving
   durable authority and could commit after displacement.
2. Process identity validation and durable-attempt validation were separate,
   inconsistent implementations across signal and exit paths.
3. Reaping could reason from experiment/phase rather than the exact launching
   attempt.
4. Managed direct CLI work could enter train/infer/analyze execution without a
   durable reservation.
5. Checkpoint analysis performed expensive work in a broad transaction and had
   no durable in-process attempt.
6. Scheduler transactions contained reachable advisory/lease and
   attempt/lifecycle lock inversions.
7. Migration 051 described operational cutover but could not stop a corrected
   scheduler from starting while an old dispatcher remained possible.
8. Migrated `legacy_unverified` rows with no PID had no bounded, auditable
   terminal path.

These were one defect family: operations lacked a single exact authority tuple
`(protocol generation, scheduler invocation, fencing token, worker attempt,
lifecycle binding)`.

## 3. Shared invariants and authority model

The implementation preserves these invariants:

1. Only the invocation holding the exact active fence may mutate scheduler
   state.
2. Every scheduler-managed unit has one exact durable child or in-process
   attempt.
3. Every signal, finalization, reap, reconciliation, capacity release, and
   lifecycle transition is bound to that exact attempt.
4. Ambiguous identity is non-signalable, non-destructively-reconcilable, and
   capacity-consuming.
5. Scheduler locks follow one documented global order.
6. No managed child or checkpoint analysis begins without durable capacity.
7. Corrected dispatch cannot start until old dispatch authority is positively
   excluded.
8. Legacy no-PID rows fail closed but have a bounded, auditable reconciliation
   path.
9. Direct CLI cannot duplicate scheduler-managed work.
10. Destructive lifecycle changes use exact SQL compare-and-update predicates
    and affected-row verification.

`SchedulerAuthorityContext` carries invocation ID, fencing token, canonical
executable identity, protocol generation, and authority-held state. It is
propagated through scheduler options rather than reconstructed.

## 4. Continuation fencing correction

Automatic scanning, candidate evaluation, already-satisfied handling, durable
decision recording, child creation, and final queue mutation now carry the
same authority context.

Every mutating boundary calls exact lease/fence revalidation. Candidate
calculation may occur outside locks; its write transaction revalidates the
source predicates and exact authority before applying results. A displaced
scheduler exits with ownership-loss code 4 and cannot record a queued
decision, create a child, or change source continuation state.

Deterministic disposable-database failpoints cover:

- `continuation_before_evaluation_mutation`
- `continuation_after_evaluation_before_queue`
- `continuation_before_child_creation`
- `continuation_before_final_commit`

Each asserts no child, no queued decision, no foreign attempt, and no source
queue binding.

## 5. Exact-attempt verification architecture

`Sources/SchedulerOwnershipRepository.hpp` contains the shared primitive. Its
checks are intentionally separated:

- durable attempt row: exact attempt ID, kind, phase, capacity class, origin,
  allowed state, work identity, launch identity, invocation, and fence;
- lifecycle binding: exact experiment/checkpoint row, status, phase, and
  `active_scheduler_worker_attempt_id`;
- process identity: PID, PGID, kernel start identity, canonical executable,
  exact command/work options, and positive observation;
- authority: protocol generation, current lease owner, and fence;
- mutation: final SQL compare-and-update plus exactly-one affected-row check.

Terminal lifecycle rows may be verified only for parent reaping of a child
that already finalized its own work; they must remain bound to the same
attempt. That exception does not weaken signalability.

## 6. Signaling correction

Global pause/resume/cancel and scheduler stop paths use the same exact-attempt
verification. Before `kill` of a process group they require a signalable
attempt, complete immutable identity, exact lifecycle binding, and two process
observations—the second immediately before the signal.

PID/start, PGID, executable, command/work, lifecycle binding, attempt state, or
fence mismatch produces rejection without a signal. `identity_ambiguous` and
foreign attempts are never signalable. Outcomes store `worker_attempt_id` and
are updated only for that exact attempt/binding.

Process tests cover successful SIGSTOP/SIGCONT/SIGTERM/SIGKILL and rejection
for reused PID/start identity, PGID, executable, command, stale lifecycle,
replacement during signal, departed process, ambiguous identity, and stale
owner fence.

## 7. Reaping and finalization correction

Local child bookkeeping stores the durable attempt ID returned by reservation.
`waitpid(..., WNOHANG)` results are resolved through that ID, then verified
against PID, immutable attempt identity, lifecycle binding, state, invocation,
and fence.

Attempt terminalization and lifecycle clearing occur atomically. Duplicate
observations are idempotent. A restarted scheduler that is not the parent uses
process observation/reconciliation, not `waitpid`. Unknown local children and
wait failures cannot mutate durable work.

The process suite replaces the lifecycle binding immediately before exact
reaper verification and proves the stale reaper cannot terminalize or clear
the replacement attempt. It also forces a parent crash after child
self-registration but before gate permission: the child exits 126, the
complete spawned attempt remains capacity-consuming, and a new owner
reconciles only that attempt.

## 8. Direct CLI worker policy and implementation

Policy: standalone CLI modes remain available only when they do not identify
scheduler-managed work. Any invocation with a scheduler experiment or
checkpoint-evaluation identity must supply the exact durable
`--scheduler-worker-attempt-id`.

Train, final inference, checkpoint inference, cancellation inference, and
analysis register and verify the exact attempt before work. Missing or stale
attempts fail deterministically before work and do not create a reservation.
The process suite explicitly rejects direct managed train, infer, checkpoint
infer, and analyze invocations and verifies zero attempts are created.

## 9. Checkpoint analysis claim/work/finalize design

Checkpoint analysis remains in-process but is operationally equivalent to a
managed child:

1. **Claim:** canonical locks, current fence, eligibility, analyze capacity,
   unique `checkpoint_analyze` attempt, immutable work identity, and exact
   evaluation binding are committed quickly.
2. **Work:** filesystem/model/CPU analysis occurs without lease-row,
   lifecycle-row, or advisory locks.
3. **Finalize:** canonical locks, current fence, exact attempt, exact
   checkpoint binding, and unchanged sources are revalidated. Result/failure,
   attempt terminalization, lifecycle clearing, and capacity release commit
   atomically.

Crash-after-claim and lease-loss failpoints leave one recoverable consuming
attempt. A new proven owner abandons only the stale in-process attempt,
requeues the exact evaluation, and prevents duplicate effects. Tests cover
capacity exhaustion, duplicate claim, crash recovery, loss during work, loss
before finalize, stale finalizer, result-persistence failure behavior, and
bounded lock duration.

Final experiment analysis was also corrected: expensive parsing is read-only
and outside the short exact-attempt finalize transaction.

## 10. Canonical global lock order and cycle elimination

The code-level contract beside shared lock helpers and both architecture
volumes specify:

1. global coordination advisory lock;
2. scheduler protocol/cutover row;
3. scheduler lease row;
4. invocation, administrative request, and worker-outcome rows;
5. worker-attempt rows in ascending attempt ID;
6. experiment rows in ascending experiment ID;
7. checkpoint-evaluation rows in ascending evaluation ID;
8. continuation decision and checkpoint-analysis result rows;
9. campaign/recommendation rows in their existing service order.

A transaction may start at a later level but may never acquire an earlier
level afterward. Multi-row locks use deterministic `ORDER BY`. The
attempt/lifecycle, advisory/lease, checkpoint analysis, child registration,
parent finalization, and global-control inversions were removed.

The global-control integration test uses an explicit advisory-lock holder and
contender handshake: the contender is proven blocked, the holder commits, and
the contender then observes cancellation and inserts no checkpoint row.

## 11. Mixed-version migration/startup barrier

Migration 052 adds the singleton protocol generation/cutover record and
generation-aware write guards. Corrected lease acquisition requires:

- required generation 52;
- `cutover_state='complete'`;
- non-null cutover time, executable, operator, and positive process evidence;
- a generation-52 invocation.

The explicit cutover command inspects all scheduler-dispatch process
identities. Any old scheduler, inspection failure, pending/failed/partial
state, or missing evidence rejects non-destructively before invocation
creation. Completed cutover is replay-idempotent and cannot be rolled back
while proof fields remain populated.

Tests cover old scheduler present, corrected scheduler present, inspection
failure, stale metadata, pending/failed/partial cutover, safe explicit
cutover, replay, rollback rejection, and zero lifecycle mutation on rejection.

## 12. Legacy no-PID reconciliation

No-PID legacy attempts remain `identity_ambiguous` and capacity-consuming
until all positive predicates hold:

- generation-52 cutover completed;
- cutover grace elapsed;
- no foreign scheduler process;
- process enumeration succeeded without permission denial;
- no command/work match was observed;
- the exact legacy attempt remains in the expected state;
- the exact lifecycle row remains bound, or is positively proven detached.

Bound rows terminalize as `abandoned` with
`legacy_no_pid_proven_absent_after_cutover`, then fail and clear only the exact
lifecycle. Detached duplicate/old rows use
`legacy_no_pid_lifecycle_detached`. Checkpoint-infer uses its exact evaluation
binding; stale `checkpoint_analyze` attempts use owner-loss recovery. Repeated
reconciliation is idempotent.

Status reports unresolved no-PID attempts by capacity class with
`capacity_consumed=1`, diagnostic, grace, and cutover state.

## 13. Updated process-launch and process-control matrix

| Site | Mechanism | Classification | Authority |
|---|---|---|---|
| `RunProcessAndWait` | fork/execvp/waitpid | Infrastructure-only database backup | Not scheduler work |
| `LaunchReservedChildProcess` | fork/gate/execv | Scheduler-owned and fully controlled | Exact reserved attempt; child and parent registration; gate |
| `SchedulerChildStatus` / reaper | waitpid/WNOHANG | Scheduler-owned | Exact local child attempt, lifecycle, owner/fence |
| Native global control | process inspection/kill of PGID | Scheduler/global-control owned | Two exact observations plus exact attempt/binding |
| `TerminateUncommittedSchedulerChildren` | SIGTERM/SIGKILL | Legacy launch cleanup | Reachable only from unused legacy schedulers |
| `LaunchChildProcess` / `Run*Legacy` | fork/execv/waitpid | Inactive legacy implementation | No production caller; retained only to avoid unrelated removal |
| scheduler process discovery | popen `ps` | Infrastructure/read-only | Cutover or diagnostic evidence only |
| resource/status inspection | popen `ps`, `sysctl`, `vm_stat` | Infrastructure/read-only | Cannot claim or mutate work |
| `RunMetadata` | popen git command | Infrastructure/read-only | Metadata only |
| direct CLI train/infer/checkpoint/analyze | existing process | Safely prohibited for managed work unless exact attempt supplied | Same exact registration protocol |
| test fixture helpers | fork/exec/kill/wait | Test-only | Disposable PID/PGID/start/executable verification |

No production `vfork`, `posix_spawn`, `system()`, `NSTask`, or `killpg()`
site exists. Process-group signals use `kill(-pgid, signal)`.

## 14. Exact destructive-mutation predicate table

| Mutation | Final SQL authority predicate |
|---|---|
| Continuation decision/queue | exact source/decision IDs and source predicates; transaction revalidates exact current lease owner/fence immediately before commit |
| Child launch failure | exact attempt ID + invocation + fence + active state + `EXISTS` exact experiment/evaluation binding; lifecycle clears exact attempt and returns one row |
| Never-spawned recovery | exact attempt ID + `reserved` + `EXISTS` exact running lifecycle binding; lifecycle update requires same attempt |
| Parent reap | exact attempt ID + invocation + fence + PID + allowed active state + `EXISTS` exact lifecycle binding |
| Reconciliation with result/absence | exact attempt ID + allowed state + exact lifecycle binding; current fence locked/refreshed |
| Detached stale attempt | exact attempt ID/state plus `NOT EXISTS` any lifecycle binding |
| Global cancel/missing process | exact attempt ID/state + exact experiment/evaluation binding; outcome carries exact attempt ID |
| Administrative stop | exact attempt ID/kind/work/phase/state + exact lifecycle binding after second OS observation |
| Checkpoint analysis finalize | exact attempt ID + invocation + fence + kind/state + exact checkpoint evaluation binding |
| Final analysis child finalize | exact attempt ID/kind/phase/state + exact experiment status/phase/model/binding |
| Legacy no-PID terminalization | exact attempt ID/origin/state/no-PID + generation/cutover/grace + exact lifecycle binding |
| Lifecycle mirror clearing | exact lifecycle primary key + exact `active_scheduler_worker_attempt_id`; exactly one row |
| Capacity release | only the exact attempt transition to a terminal state; no separate counter mutation |
| Stale in-process analysis recovery | exact attempt ID/kind/state + exact evaluation binding, followed by exact requeue/clear |

Every scheduler-authoritative row above is in a transaction that locks and
refreshes the exact lease owner/fence. Exactly-one helpers turn stale predicates
into rollback, not best-effort success.

## 15. Migration details

Migration 051 remains the base ownership migration. New migration
`052_scheduler_protocol_and_exact_attempt_hardening.sql` is transactional,
replay-idempotent, and was not applied to production.

Migration 052 adds:

- protocol generation and cutover proof;
- generation-52 invocation capability;
- `checkpoint_analyze` durable-attempt shape;
- exact active-attempt lifecycle guards;
- attempt identity on administrative outcomes;
- constraints and indexes for kind/origin/identity/capacity;
- trigger guards against pre-generation scheduler write patterns after
  cutover;
- runtime grants;
- irreversible-complete cutover validation.

Migration 050 `current_operation` remains exactly `train`, `infer`, or
`analyze`.

## 16. Files changed

Core:

- `Headers/ExperimentScheduler.hpp`
- `Headers/SchedulerExecutablePath.hpp`
- `Headers/SchedulerOwnershipPolicy.hpp`
- `LSTM/main.cpp`
- `Sources/ExperimentScheduler.cpp`
- `Sources/GlobalExperimentControl.cpp`
- `Sources/GlobalExperimentControl.hpp`
- `Sources/SchedulerOwnershipRepository.hpp`

Migrations/tests:

- `Database/migrations/051_scheduler_ownership_and_worker_attempts.sql`
- `Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql`
- `Tests/GlobalExperimentControlIntegrationTests.sh`
- `Tests/GlobalExperimentControlProcessTests.cpp`
- `Tests/SchedulerCanonicalPathTests.sh`
- `Tests/SchedulerContinuationOwnershipIntegrationTests.sh`
- `Tests/SchedulerOwnershipIntegrationTests.sh`
- `Tests/SchedulerOwnershipMigrationTests.sql`
- `Tests/SchedulerOwnershipPolicyTests.cpp`
- `Tests/SchedulerOwnershipProcessIntegrationTests.sh`
- `Tests/fixtures/scheduler_protocol_cutover_bin/ps`

Documentation:

- `Database/README.md`
- `docs/architecture/Volume_VII_Experiment_Lifecycle.md`
- `docs/architecture/Volume_XI_Scheduler.md`
- `docs/architecture/adr/ADR-0016-scheduler-atomic-claim-hardening.md`
- `docs/architecture/adr/ADR-0018-scheduler-generation-52-exact-attempt-authority.md`
- `docs/architecture/adr/README.md`

## 17. Tests and build commands executed

Passed on the final source unless explicitly noted:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" -configuration Release \
  -derivedDataPath DerivedData/SchedulerOwnershipCorrection build

Tests/SchedulerOwnershipProcessIntegrationTests.sh \
  "$PWD/DerivedData/SchedulerOwnershipCorrection/Build/Products/Release/LSTM_Release" \
  "$PWD/DerivedData/SchedulerOwnership/Tests/GlobalExperimentControlProcessTests"

Tests/SchedulerOwnershipIntegrationTests.sh
Tests/SchedulerContinuationOwnershipIntegrationTests.sh \
  "$PWD/DerivedData/SchedulerOwnershipCorrection/Build/Products/Release/LSTM_Release"
Tests/SchedulerCanonicalPathTests.sh
Tests/GlobalExperimentControlIntegrationTests.sh \
  "$PWD/DerivedData/SchedulerOwnershipCorrection/Build/Products/Release/LSTM_Release" \
  "$PWD/DerivedData/SchedulerOwnership/Tests/GlobalExperimentControlProcessTests"
```

Sanitizer-backed pure tests passed with:

```bash
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  DerivedData/SchedulerOwnership/Tests/SchedulerOwnershipPolicyTests
ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  DerivedData/SchedulerOwnership/Tests/SchedulerChildStatusTests
```

The first sanitizer invocation with `detect_leaks=1` was executed and failed
before test execution because Apple's ASan runtime does not support leak
detection. It was rerun with supported halt-on-error settings and passed.

Shell syntax, `git diff --check`, launch/signal/waitpid search,
lease/fence mutation search, and `FOR UPDATE`/advisory-lock search were also
executed. The migration integration applies migrations 051/052 twice and runs
`SchedulerOwnershipMigrationTests.sql`.

## 18. Forty-scenario traceability

| # | Scenario | Test file | Exact test function or shell section | Type | Assertions | Status |
|---:|---|---|---|---|---|---|
| 1 | Restart with live train worker | `SchedulerOwnershipProcessIntegrationTests.sh` | `launch_worker train 920051` and prior-worker observation section | DB+process | exact PID/PGID/start/exe/command/attempt retained; no duplicate | Full |
| 2 | Restart with live inference worker | same | `launch_worker infer 920052` | DB+process | exact attempt observed; infer capacity remains one | Full |
| 3 | Restart with live final-analysis worker | same | `launch_worker analyze 920053` | DB+process | exact attempt observed; analyze capacity retained | Full |
| 4 | Restart with live checkpoint worker | same | `launch_worker checkpoint 920054 925051` | DB+process | exact checkpoint binding retained; no relaunch | Full |
| 5 | Duplicate scheduler | same | live owner then `duplicate.out` | process | exit 3, ownership rejected, one live scheduler | Full |
| 6 | Duplicate rejection is non-mutating | same | `duplicate.out` before/after counts | DB+process | `mutations=0`; attempts/lifecycles/capacity unchanged | Full |
| 7 | Expired dead-owner takeover | same | crash recovery/takeover section | DB+process | exact process absence + expiry increments fence once | Full |
| 8 | Valid/fresh owner prevents takeover | same; `SchedulerOwnershipPolicyTests.cpp` | fresh-dead-owner section; `DecideSchedulerTakeover` assertions | unit+process | fresh, valid, ambiguous owner all reject | Full |
| 9 | Scheduler crash preserves live workers | same | four-worker fixture plus forced scheduler termination | DB+process | worker identities remain live/bound/capacity-consuming | Full |
| 10 | Graceful scheduler exit preserves workers | same | graceful SIGTERM/released lease section | DB+process | exact lease released; workers not signaled/cleared | Full |
| 11 | PID/start reuse | same; `GlobalExperimentControlProcessTests.cpp` | identity mismatch fixture; `main` reused-PID signal matrix | DB+process | ambiguous, no signal, binding/capacity retained | Full |
| 12 | Executable mismatch | same files | executable mismatch fixture / `executableRejected` | DB+process | no signal or destructive reconciliation | Full |
| 13 | PGID mismatch | `GlobalExperimentControlProcessTests.cpp` | `groupRejected` in `main` | process | second exact validation rejects; zero signal | Full |
| 14 | Command/work mismatch | same; process integration | `staleCommand` / command-work mismatch fixture | DB+process | no signal; ambiguous and consuming | Full |
| 15 | Valid prior worker observation | process integration | `SCHEDULER_PRIOR_WORKER_OBSERVED` four-kind section | DB+process | observation updates diagnostic only; launch identity immutable | Full |
| 16 | Ambiguous identity fails closed | process integration | three mismatch fixtures | DB+process | all remain bound, non-signalable, consuming | Full |
| 17 | Train capacity | process integration; policy unit | status/capacity assertions; consuming-state loop | DB+unit | exact count, reservation consumes, terminal release | Full |
| 18 | Inference capacity | process integration | four-worker status section | DB+process | final+checkpoint infer count exactly two | Full |
| 19 | Analyze capacity | process integration | live analyze and checkpoint-analysis sections | DB+process | exhaustion prevents claim; stale claim consumes one | Full |
| 20 | Checkpoint capacity | process integration; migration SQL | checkpoint worker and unique-attempt section | DB+process | checkpoint infer shares infer capacity; duplicate rejected | Full |
| 21 | Mixed global accounting | process integration | `SCHEDULER_STATUS_GLOBAL_CAPACITY` assertions | DB+process | train=1, infer=2, analyze=1 from durable states | Full |
| 22 | Concurrent last-slot claim | process integration; migration SQL | singleton duplicate scheduler + unique active attempt constraints | concurrency+DB | contender cannot reach claim; duplicate insert rejected | Full |
| 23 | Duplicate experiment attempt | `SchedulerOwnershipMigrationTests.sql` | `duplicate active experiment attempt accepted` block | DB | unique violation; existing exact attempt unchanged | Full |
| 24 | Duplicate checkpoint attempt | same | checkpoint-infer and checkpoint-analyze duplicate blocks | DB | both kinds reject second active attempt | Full |
| 25 | Claim committed, no spawn | process integration | experiment 920056 never-spawned section | DB+process | exact reserved attempt -> launch_failed; exact binding cleared | Full |
| 26 | Child registered, parent crashes before gate | process integration | `scheduler_parent_crash_after_child_registration` | DB+process | exit 87; complete spawned identity; capacity one; exact recovery to zero | Full |
| 27 | Exit 126/127 and signal status | `SchedulerChildStatusTests.cpp`; process integration | `main`; canonical launch assertions | process+unit | actual missing exec=127, gate close=126, signaled exit decoded | Full |
| 28 | Canonical launch from foreign CWD | process integration | canonical child section | process | child exec succeeds outside repository CWD | Full |
| 29 | Basename invocation | `SchedulerCanonicalPathTests.sh`; process integration | basename case / canonical child section | process | resolves absolute executable; no 127 | Full |
| 30 | Symlink invocation | same | symlink case | process | canonical target persisted and executed | Full |
| 31 | Lease loss stops mutation | process integration; continuation integration | lease-loss-owner; four continuation boundaries | DB+process | exit 4; foreign owner retained; no later claim/queue | Full |
| 32 | Foreign fence cannot mutate/release | `SchedulerOwnershipMigrationTests.sql`; process integration | foreign refresh/release block; lease-loss section | DB+process | affected rows zero; foreign lease remains active | Full |
| 33 | Destructive actions require exact attempt | process integration; global process test | stale-reaper boundary; identity signal matrix | DB+process | replacement attempt remains active; stale signal/reaper mutate zero | Full |
| 34 | Migration and retry replay | `SchedulerOwnershipIntegrationTests.sh` | direct migration replay section | migration | 051/052 second application succeeds; rows/constraints stable | Full |
| 35 | Interrupted launch deterministic recovery | process integration | never-spawned and parent-crash sections | DB+process | exact result codes/diagnostics, binding clear, capacity release | Full |
| 36 | Global pause/resume/cancel | `GlobalExperimentControlIntegrationTests.sh`; process test | production command section; Pause/Resume/Cancel matrix | DB+process+concurrency | exact signals/outcomes/attempt IDs; stale rows rejected | Full |
| 37 | Checkpoint stop/control | global integration; process test | checkpoint request replay; `TestProductionCheckpointStopOwnership` | DB+process | exact checkpoint target/attempt; restart replay no duplicate | Full |
| 38 | Cancellation inference | global integration | deterministic checkpoint-inference/cancellation advisory-lock section | concurrency+DB | contender blocks, observes cancel, inserts zero rows | Full |
| 39 | Continuation and checkpoint-analysis compatibility | continuation integration; process integration | baseline+four boundaries; checkpoint claim/crash/loss/recovery sections | DB+process | replay-safe queue; no post-fence mutation; exact analyze recovery | Full |
| 40 | Ownership/status/cutover/legacy diagnostics | process integration | barrier, direct CLI, legacy no-PID, and status sections | DB+process | generation reason, no-PID capacity, attempts, owner/fence all exact | Full |

Summary: **Full 40/40; partial 0/40; missing 0/40; manual-only 0/40.**

## 19. Exact results

- Release build: succeeded.
- Process integration: succeeded on the final binary.
- Ownership migration/repository integration: succeeded, including replay.
- Continuation ownership integration: succeeded at all four fence boundaries.
- Global pause/resume/cancel integration: succeeded.
- Canonical-path integration: succeeded.
- Sanitizer-backed ownership policy and child-status tests: succeeded under
  supported Apple ASan/UBSan settings.
- Shell syntax and `git diff --check`: succeeded.
- Mechanical production-code process-control search found every fork/exec,
  popen, waitpid, and kill site and classified them in section 13.
- No disposable test database or exact-identity test process remained after
  cleanup.
- The build emitted 366 warnings, overwhelmingly libpqxx 7.10
  `exec_params` deprecations. There were no compiler errors in the successful
  final build.

## 20. Deferred verification and residual risks

- Migrations 051/052 were not applied to production.
- Generation-52 cutover was not performed against production.
- The currently running pre-correction scheduler was not replaced.
- Production adoption/reconciliation of its existing workers remains a
  deployment exercise.
- The repository-wide libpqxx deprecation backlog remains and conflicts with
  the project's warning-as-defect aspiration; converting that API surface is
  outside this scheduler increment.
- The inactive `Run*Legacy` implementations remain in source. Mechanical
  call-site audit found no production caller except the renamed checkpoint
  analysis entry point, whose body now implements claim/work/finalize.
- An independent post-correction CEE review has not yet been run.

## 21. Commit-readiness verdict

**Worktree ready to commit: No, as currently presented.**

The scheduler correction and its tests are internally consistent and passing,
but the worktree includes untracked historical review/report files plus the
external `watch_20260729-142309` artifact, and the final build still reports
the project-wide libpqxx deprecation warning backlog. Those items must be
curated or explicitly accepted before a commit is prepared.

## 22. Production-readiness verdict

**Ready for production deployment: No.**

The implementation is deployable only after backup, controlled old-scheduler
shutdown, generation-52 cutover proof, migration application, status review,
and post-cutover observation in production. Passing disposable tests cannot
substitute for those operational gates.

Explicit answers:

- Can continuation mutate after lease loss? **No.**
- Is every signal bound to the exact durable attempt? **Yes.**
- Can a stale reaper finalize a newer attempt? **No.**
- Can direct CLI execution bypass ownership or capacity? **No for
  scheduler-managed identities; standalone unrelated work remains allowed.**
- Is checkpoint analysis bounded and crash-recoverable? **Yes.**
- Is scheduler lock ordering documented, mechanically audited, and free of the
  identified reachable cycles? **Yes.**
- Can old and corrected scheduler versions coexist unsafely? **No after the
  generation-52 barrier; corrected startup fails closed before cutover.**
- Can legacy no-PID attempts strand capacity indefinitely? **No when positive
  cutover/process-absence predicates become available; otherwise they
  intentionally remain fail-closed and visible.**
- Are all 40 required scenarios fully covered? **Yes: 40/40 executed in
  disposable environments.**
- Is the current worktree ready to commit? **No, for the curation/warning
  reasons above.**
- Is it ready for production deployment now? **No.**

## 23. Production deployment prerequisites and ordering

1. Curate the intended commit and obtain independent review.
2. Build the corrected executable at a new path; do not overwrite the binary
   used by live processes.
3. Back up production.
4. Stop only the old scheduler after exact PID/PGID/start/executable
   verification; preserve workers.
5. Positively verify no old scheduler-dispatch process remains.
6. Apply migrations 051 then 052 using the authorized migration role.
7. Run explicit generation-52 cutover; do not manufacture proof metadata.
8. Start one corrected scheduler.
9. Verify protocol, invocation, lease/fence, exact attempts, capacity, prior
   worker observation, and unresolved no-PID diagnostics.
10. Wait the configured legacy grace before bounded reconciliation.
11. Reconcile only through scheduler workflows; never edit production
    experiment rows to make status look clean.
12. Roll forward on post-cutover failure; completed cutover must not silently
    re-enable old dispatch writes.

## 24. `git status --short`

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

## 25. `git diff --stat`

Plain `git diff --stat` excludes all untracked migration, test, header,
fixture, ADR, report, and watch files.

```text
 Database/README.md                                 |   39 +
 Headers/ExperimentScheduler.hpp                    |    9 +
 LSTM/main.cpp                                      |   60 +-
 Sources/ExperimentScheduler.cpp                    | 5637 +++++++++++++++++---
 Sources/GlobalExperimentControl.cpp                |  856 ++-
 Sources/GlobalExperimentControl.hpp                |    5 +
 Tests/GlobalExperimentControlIntegrationTests.sh   |  151 +-
 Tests/GlobalExperimentControlProcessTests.cpp      |  534 +-
 .../Volume_VII_Experiment_Lifecycle.md             |   32 +-
 docs/architecture/Volume_XI_Scheduler.md           |  179 +-
 .../ADR-0016-scheduler-atomic-claim-hardening.md   |   14 +
 docs/architecture/adr/README.md                    |    2 +
 12 files changed, 6532 insertions(+), 986 deletions(-)
```
