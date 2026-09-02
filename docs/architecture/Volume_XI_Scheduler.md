# Volume XI — Scheduler

Status: Generation-52 exact-attempt ownership implemented and independently reviewed; safe-window process regression remains a pre-enable gate
Version: 0.5.1
Last revised: 2026-08-03

## 1. Purpose

Define durable ownership of experiment work selection, capacity, process
launch, lifecycle reconciliation, recovery, and scheduler observability.

## 2. Scope

### 2.1 In scope

Polling, eligibility, phase claims, worker slots, launch commands, attempt/PID
state, completion handling, orphan recovery, operator controls, and shutdown.

### 2.2 Out of scope

Training/inference mathematics, recommendation policy/review, profitability,
and research policy decisions are owned by other volumes.

### 2.3 Current implementation status

The scheduler manages experiment training, inference, analysis, checkpoint, and
configured continuation workflows. Recommendation generation, scoring, and
review remain explicitly outside its polling loop.

Global administrative execution control is database-authoritative. Scheduler
claim/launch boundaries and pause, resume, or cancellation commands use one
PostgreSQL advisory-lock protocol. An active request or persistent paused state
suppresses ordinary train, infer, analyze, checkpoint, and continuation
launches. Unix suspension is separate from experiment lifecycle status. See
[`GlobalExperimentControls.rst`](../GlobalExperimentControls.rst) for operator,
process-identity, checkpoint, inference, dry-run, and restart semantics.

Migration 051 and scheduler version 0.3.0 implement the ADR-0016 boundary with
a fenced singleton lease, immutable invocation history, active worker
attempts, global capacity reservations, and gated child launch. Campaign
Operations production dispatch remains a separately governed decision; its
handoff produces only ordinary experiments and adds no scheduler work class.

Migration 052 and scheduler version 0.4.0 correct the remaining shared
ownership boundaries. Scheduler mutation receives one
`SchedulerAuthorityContext` containing the invocation ID, fencing token,
canonical executable, and held-authority state. Continuation evaluation and
queueing, exact-attempt finalization, checkpoint analysis, and recovery carry
that context without reconstructing ownership-free options.

Independent Phase G verification observed scheduler generation 52 active with
cutover complete and found no open scheduler architecture defect. It did not
rerun process-level scheduler/global-control suites because production workers
were active; those suites remain an ADR-0019 rollout prerequisite in a safe
window, not an unowned Campaign Operations workaround.

ADR-0019 permits Campaign Operations to consume exactly generation-52 protocol
and cutover evidence through a narrow pinned security-definer function. The
function returns a versioned exact canonical and, for a mutating caller, holds
the scheduler protocol row `FOR SHARE` through the caller transaction. It does
not expose general scheduler tables or liveness authority.

## 3. Responsibilities

### 3.1 Owned responsibilities

Own eligible-work polling, atomic claims, per-operation capacity, worker
process launch, durable attempt state, completion transitions, and recovery.

### 3.2 Dependencies

Consumes Volume VII lifecycle state and invokes bounded workers from Volumes
V/VI. PostgreSQL contracts are governed with Volume XII.

### 3.3 Prohibited responsibilities

The scheduler MUST NOT invent experiments, approve recommendations, alter
model math, infer policy from scores, or treat PID presence as sole truth.

## 4. Architecture

### 4.1 Components

Poller, eligibility/priority rules, capacity manager, database claim service,
command builder, process supervisor, completion reconciler, recovery logic,
and status/diagnostic CLI.

### 4.2 Control flow

Read eligible durable state → acquire bounded claim → commit claim → launch
worker outside the transaction → observe completion → persist authoritative
outcome → release capacity.

### 4.3 Ownership boundaries

Lifecycle services define legal transitions; scheduler selects and claims;
workers calculate and report; repositories persist. Campaign Operations ends
at accepted lifecycle handoff and may observe scheduler/lifecycle evidence
read-only. The scheduler does not read Campaign Operations policy tables.
It never polls the Campaign Manager or Campaign Operations request tables and
receives no Campaign Operations privilege. Pending experiments produced by
Phase H are ordinary work under unchanged scheduler claim/capacity policy.

## 5. Data model

### 5.1 Authoritative entities

Experiment phase/status, worker attempt, operation, PID/process group,
executable, kernel process-start identity, capacity class, exit/failure state,
scheduler version, and recovery metadata.

`experiment_scheduler_invocation` is immutable invocation history. Its nonce,
PID, process group, kernel start identity, and canonical executable distinguish
restart and PID reuse. PID and display text alone are diagnostic.

`experiment_scheduler_lease` is the sole current authority. Its owner and
monotonic fencing token are authoritative. Acquisition is permitted for a
vacant or explicitly released lease, or only when the lease is expired **and**
the former PID/start/process-group/executable identity is proven missing or
mismatched. Ambiguous inspection fails closed.

The lease duration is 90 seconds. The owner refreshes before every poll and at
most every 30 seconds while sleeping. A lease row remains locked throughout a
scheduler write transaction, so a long database operation serializes takeover
instead of silently allowing overlapping authority. A fresh lease rejects
takeover even when its process is already absent; expiry without proved
absence/mismatch also rejects takeover.

`experiment_scheduler_worker_attempt` is the worker and capacity authority.
The launch scheduler/identity is immutable; `observed_by_scheduler_invocation_id`
records observation without rewriting launch history. The active-attempt
foreign keys on experiment and checkpoint rows fence lifecycle mutation.

Authoritative identity fields are the invocation ID/fence, linked worker
attempt ID, work identity and phase, PID, process group, kernel start identity,
canonical executable, required command option identities, and attempt state.
Reservation/spawn/registration/completion timestamps are authoritative event
ordering evidence. The scheduler invocation display command, log path,
human-readable diagnostic, and status resource measurements are diagnostic.
The linked lifecycle `worker_*` columns are required identity mirrors for
global administrative controls; they cannot create ownership without the
active attempt.

Ordinary scheduler workers are authorized by active `experiment` execution
state. Checkpoint inference workers are authorized independently by active
`experiment_checkpoint_eval` state; completion of the parent experiment does
not invalidate an otherwise matching checkpoint worker.

### 5.2 Provenance and versions

Launches retain exact experiment/model IDs, operation, command/invocation
metadata, scheduler/build provenance, and attempt ownership.

### 5.3 Invariants and legacy data

Status, phase, operation, worker fields, and attempt state must form valid
shapes. Scheduler launches and conservative adoption persist the kernel
process-start identity. Administrative signaling requires an exact PID,
process-group, executable, experiment, phase, and process-start match. Orphan
recovery uses durable evidence plus process checks. Migration 051 represents
legacy active rows as `legacy_unverified`/`identity_ambiguous` attempts. They
consume capacity but assert neither liveness nor scheduler ownership until
exact OS evidence is validated.

States `reserved`, `spawned`, `running`, `observed`, and
`identity_ambiguous` consume capacity. `completed`, `failed`,
`launch_failed`, and `abandoned` do not. Train consumes train capacity; final,
checkpoint, and cancellation-authorized checkpoint inference share infer
capacity; final analysis consumes analyze capacity. In-process checkpoint
analysis runs only when global analyze capacity is available. Lifecycle status
alone never removes an active attempt from capacity.

``experiment.current_operation`` has exactly one canonical vocabulary:
``train``, ``infer``, or ``analyze``. These values name lifecycle work only.
Unix suspension belongs to ``worker_control_state``; cancellation, checkpoint,
retry, and recovery detail belongs to lifecycle status/phase, administrative
outcomes, and diagnostics. Those control labels must never be stored as
``current_operation``.

Migration 050 exactly maps legacy ``training``, ``inference``, and ``analysis``
values. It reconciles a historical control label only when status, phase, and
the owning cancellation or checkpoint fields prove the legacy writer shape;
unsupported or ambiguous rows abort the migration without being inferred from
phase. The final constraint enforces the canonical set.

The compatibility trigger canonicalizes only those same proved legacy writes
during a rolling deployment, so a still-running pre-050 worker cannot
reintroduce them. It is temporary: retain it until every pre-050 scheduler,
worker, and administrative executable has exited and validation shows no
legacy writes. A later reviewed migration may then drop the trigger and
function while retaining the constraint. Current readers normalize the three
noun-form aliases before migration, but preserve unsupported values as invalid
diagnostic evidence rather than concealing them.

## 6. Transactions

### 6.1 Read paths

Polling may read candidates optimistically, but eligibility is rechecked in the
claim transaction.

### 6.2 Write paths

Claims, start metadata, completion, recovery, and operator transitions have
short explicit transactions. No transaction remains open across process launch
or worker execution.

Launch is reservation/claim commit, gated fork, durable PID/process
group/start/executable/command persistence, and only then `execv`. Before it
waits on the gate, the child idempotently persists its own exact spawn evidence
against the pre-existing attempt; that narrow update cannot refresh the lease
or claim work. The parent repeats the same fenced persistence before granting
exec permission. Thus a parent crash after `fork` still leaves spawned/exited
evidence, while a reservation with no child evidence is provably never
created. The worker registers the attempt before bounded work. Exec failure
remains exit 127 with a durable terminal attempt and cannot cause an
unaccounted retry loop.

### 6.3 Failure semantics

Launch failure, worker failure, stale claim, and persistence failure remain
distinct. Recovery never marks work complete without authoritative evidence.
Database/lease refresh loss stops the owner before its next claim or launch.
Foreign identities cannot refresh or release the row. A malformed or
uninspectable identity becomes capacity-consuming `identity_ambiguous`;
operators receive a diagnostic, and no signal or destructive lifecycle change
is authorized.

## 7. Concurrency

### 7.1 Conflict domain

The same experiment phase/attempt and shared capacity class may conflict.

### 7.2 Locking and serialization

Claims use database-visible conditional state/locks. Capacity is enforced
without globally serializing unrelated status inspection or completed work.
Multi-row locks require stable order.

Global lock order is:

1. global coordination advisory lock;
2. scheduler protocol/cutover singleton;
3. scheduler lease singleton;
4. scheduler invocation and administrative request/outcome rows;
5. worker-attempt rows by ascending `worker_attempt_id`;
6. experiment rows by ascending `experiment_id`;
7. checkpoint-evaluation rows by ascending `checkpoint_eval_id`;
8. continuation and checkpoint-analysis result rows;
9. campaign/recommendation rows in their separately documented service order.

A transaction may start at the first lock it needs, but it must never acquire
an earlier lock afterward. Attempt-before-lifecycle is mandatory. Every
multi-row lock uses an explicit stable order. Process inspection normally
occurs outside database locks; the second identity check immediately before a
signal remains inside the short global-control critical section.

The lease row serializes the capacity conflict domain. A reservation
transaction counts all active attempts, locks the exact lifecycle row, inserts
one attempt, and compare-and-updates its active-attempt fence before commit.

### 7.3 Winner, loser, and retry outcomes

Exactly one scheduler/attempt claims an operation. Losers observe changed state
and skip deterministically. Retry and orphan recovery preserve attempt history
and never duplicate completion.

## 8. CLI

### 8.1 Commands and validation

Scheduler start/status/recovery and operator lifecycle controls are explicit.
Dry-run and once modes must accurately describe write behavior.

### 8.2 Machine output

Events identify scheduler cycle, experiment, phase, attempt, operation,
capacity, PID, transition, reason, and error.
`SCHEDULER_STATUS_CHECKPOINT_JOB` reports each active checkpoint inference row,
while the existing status and resource records include every validated
checkpoint worker in managed inference totals exactly once.
Ownership, global capacity, reservations, prior-launch observation, ambiguous
identity, and exact attempt records are exposed as
`SCHEDULER_STATUS_OWNERSHIP`, `SCHEDULER_STATUS_GLOBAL_CAPACITY`,
`SCHEDULER_STATUS_WORKER_OWNERSHIP`, and
`SCHEDULER_STATUS_WORKER_ATTEMPT`.
`SCHEDULER_STATUS_EXECUTABLE_IDENTITY` distinguishes the executable running
the status command from the lease-owning scheduler's canonical executable and
the executable observed for that exact scheduler process. The legacy
`SCHEDULER_STATUS_OWNERSHIP.canonical_executable_path` field remains the
lease-owning scheduler invocation path and is identical to the additive
`scheduler_canonical_executable_path` field.

`SCHEDULER_STATUS_WORKER` reports one coherent runtime identity result per
discovered or expected worker. `managed` is granted only after the exact
durable PID, process group, kernel start identity, command/work identity, and
canonical executable all validate. `execution_state` independently reports
`running`, `stopped`, `missing`, or `unknown`. Therefore an exact stopped
attempt bound to a paused experiment is `managed=1,execution_state=stopped`,
whereas a foreign stopped process remains unmanaged, an expected dead process
is authoritative/missing, and a live identity mismatch is reported separately
from unmanaged work.

### 8.3 Human output

Summaries distinguish queued, claimed, running, recovering, failed, and
completed work without overstating process observation.
Scheduler status lists active checkpoint inference jobs separately and reports
only processes that fail both experiment and checkpoint-evaluation ownership
validation as unmanaged.
Managed totals include both executing and intentionally stopped workers;
additive managed-running and managed-paused counters expose that split without
changing the legacy managed totals.

## 9. Testing

### 9.1 Pure tests

Eligibility, capacity, priority, command construction, status mapping, and
child-status interpretation.

### 9.2 Persistence and migration tests

Claim compare-and-set, state constraints, attempt metadata, and recovery.

### 9.3 Concurrency and integration tests

Multiple schedulers, capacity races, launch failure, completion races, orphan
recovery, shutdown, and unrelated work concurrency.

### 9.4 Regression boundaries

Scheduler changes preserve model math, evidence calculation, recommendation
status, continuation policy, and explicit operator semantics.

## 10. Operational safety

### 10.1 Runtime isolation

Tests do not start or signal a real scheduler without explicit authorization.
Documentation changes never affect running processes.

### 10.2 Permissions and destructive operations

Scheduler roles receive only required lifecycle/attempt privileges. Force-stop
and destructive recovery require explicit operator confirmation.

### 10.3 Observability and recovery

Durable state, logs, attempts, PIDs, timestamps, operations, exit codes, and
diagnostics support reconciliation after scheduler or worker failure.

SIGINT/SIGTERM stops polling and launch, preserves live child attempts, and
conditionally releases only the caller's exact lease/fence. Crash cleanup is
not required. Takeover observes valid prior workers without changing their
launch identity. PID reuse or PID/start/process-group/executable/work-ID/attempt
mismatch becomes `identity_ambiguous`, continues consuming capacity, is never
signaled, and blocks destructive reconciliation.

Operator stop follows the same rule: it requires the exact active durable
attempt and revalidates PID, process group, start identity, canonical
executable, experiment/phase command identity, and safety of the process group
immediately before signaling. Command/PID scanning cannot authorize a signal.

Parent reaping has the same exact-attempt rule. A `waitpid` observation is
local evidence only. Before applying it, the scheduler locks and validates the
exact attempt and exact lifecycle binding. The terminal attempt update and
lifecycle clear both repeat the attempt ID predicate. A stale parent, duplicate
observation, unknown child, or observation whose lifecycle binding has moved
cannot affect the replacement attempt.

Stopping training at a persisted checkpoint is a successful completion of that
train phase. In one fenced transaction the control workflow verifies the exact
active train attempt and checkpoint decision, records the epoch/model evidence,
sets that attempt to `completed` with reconciliation result
`checkpoint_stop_completed`, clears the exact lifecycle binding and process
mirrors, and advances to the policy-selected pending phase or cancellation
destination. The transaction must affect exactly one attempt and exactly one
lifecycle row. A later `waitpid` for the old child may append exact exit
evidence to that already-terminal attempt, but it never changes lifecycle state
or capacity and never requires the lifecycle still to reference the old
attempt. Replay recognizes the exact terminal reason and checkpoint evidence
without applying the phase transition again.

Direct CLI train, final inference, checkpoint inference, and analysis targeting
a scheduler-managed experiment or checkpoint evaluation are prohibited unless
the invocation carries the exact durable `worker_attempt_id` reserved by the
scheduler. Bulk direct analysis is prohibited. Legitimate standalone training
or inference without a scheduler experiment/evaluation identity remains
unmanaged and cannot target scheduler lifecycle rows.

Checkpoint analysis remains in-process but is no longer transaction-spanning.
Claim briefly revalidates the lease, reserves global analyze capacity by
creating a `checkpoint_analyze` attempt, and binds the checkpoint evaluation.
Work reads and computes outside write transactions and lease/global-control
locks. Finalize revalidates the current fence, exact attempt, lifecycle
binding, and inference-result identity, then atomically persists the result,
terminalizes that attempt, and clears that exact binding. Crash or fence loss
leaves a durable active attempt for the next owner to abandon and requeue
exactly once.

All self-children resolve `_NSGetExecutablePath` with buffer retry, `realpath`,
absolute/executable validation, then use `execv`. `argv[0]`, `PATH`, the
working directory, Screen, and symlink spelling are not authority. A symlink
invocation intentionally resolves to its canonical target.

Deployment ordering is: build the generation-52 executable at a new path;
stop only old scheduler dispatch processes while preserving positively
identified workers; back up; apply migrations 051 and 052; run the explicit
generation-52 cutover command; and start one corrected scheduler only after
the command records positive process-inspection evidence that no old or
corrected scheduler is running. Startup remains read-only and fails closed
while cutover is pending or failed. Repeated cutover/startup is idempotent.

Migration 052 triggers reject scheduler lifecycle write shapes unless the
database session declares protocol generation 52. This is defense in depth;
the cutover record and positive absence inspection are the startup authority.
Do not roll back the executable or migration independently after cutover.

Phase H approval is exact, not monotonic: it accepts generation 52 and one
byte-identical versioned protocol canonical only. `generation >= 52`, “52 or
newer,” and authorization from cutover state or liveness alone are prohibited.
A future generation does not inherit approval; Campaign Operations becomes
ineffective for new acquisition/handoff until its own disable/new-contract/
independent-verification/re-enable sequence completes. The scheduler neither
reads nor enforces that Campaign Operations enable chain.

Legacy no-PID attempts remain fail-closed during cutover and for a bounded
post-cutover grace interval. After cutover is complete, old dispatch authority
is positively absent, the exact lifecycle binding remains, no launch can still
publish identity, and the grace period has elapsed, reconciliation may
terminalize only that exact `legacy_unverified`/`identity_ambiguous` attempt.
It records `legacy_no_pid_proven_absent_after_cutover`; repeated reconciliation
is a no-op. Missing process-inspection permission never authorizes this path.
Unresolved legacy no-PID attempts remain visible in status and consume
capacity.

## 11. Future extensions

### 11.1 Approved extension points

Typed work classes, capacity policies, claim services, and worker adapters.

### 11.2 Deferred capabilities

Distributed schedulers, remote workers, priority budgets, and research-campaign
priority/capacity policy.

### 11.3 Required decisions

Any new work class requires an ADR defining eligibility, claim, capacity,
idempotency, recovery, operator control, and regression scope.

## 12. References

- [Volume I §§8–10](Volume_I_Foundation.md)
- [ADR-0004](adr/ADR-0004-scheduler-ownership-boundaries.md)
- [ADR-0016](adr/ADR-0016-scheduler-atomic-claim-hardening.md)
- [ADR-0018](adr/ADR-0018-scheduler-generation-52-exact-attempt-authority.md)
- [ADR-0019](adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md)
- [Volume VII](Volume_VII_Experiment_Lifecycle.md)
- [Volume XII](Volume_XII_Database.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established scheduler ownership and safety outline. | ADR-0004 |
| 0.2.0 | 2026-07-24 | Accepted atomic claim/attempt hardening, preserved ordinary experiment work classes, and gated Campaign Operations production dispatch pending implementation and independent verification. | ADR-0004, ADR-0016 |
| 0.2.1 | 2026-07-25 | Documented checkpoint-evaluation worker ownership and status accounting. | ADR-0004 |
| 0.2.2 | 2026-07-27 | Restored the canonical `current_operation` vocabulary (`train`, `infer`, `analyze`) across persistence, recovery, administrative controls, and status reporting. | ADR-0004 |
| 0.3.0 | 2026-07-27 | Implemented fenced ownership, durable global worker attempts/capacity, conservative recovery, gated canonical launch, shutdown, and diagnostics. | ADR-0004, ADR-0016 |
| 0.4.0 | 2026-07-29 | Added generation-52 authority propagation, exact signaling/reaping/finalization, bounded checkpoint analysis, direct-CLI ownership enforcement, canonical lock order, technical cutover barrier, and bounded legacy no-PID reconciliation. | ADR-0016, ADR-0018 |
| 0.4.1 | 2026-07-30 | Made stop-at-checkpoint atomically complete the exact train attempt, release capacity, clear its binding, and tolerate delayed reap and replay. | ADR-0018 |
| 0.5.0 | 2026-07-31 | Recorded generation-52 implementation/review status and the exact narrow Phase H evidence interface without scheduler polling, Campaign Operations privileges, or future-generation approval inheritance. | ADR-0018, ADR-0019 |
| 0.5.1 | 2026-08-03 | Completed H1 readiness reporting for scheduler canonical/hash evidence, independent verification, and approved-versus-actual Manager build comparison while preserving fail-closed separation of authority. | ADR-0019B |
