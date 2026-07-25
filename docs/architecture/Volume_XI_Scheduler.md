# Volume XI — Scheduler

Status: Foundation with accepted atomic-claim hardening; hardening not implemented
Version: 0.2.1
Last revised: 2026-07-25

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

ADR-0016 accepts the required hardening of ordinary scheduler claims: an exact
pending/phase recheck, durable scheduler-attempt identity, and conditional
lifecycle claim must commit atomically before process launch. Repository review
found that this target is not yet fully implemented. Campaign Operations
production dispatch therefore remains disabled; its accepted handoff produces
only ordinary experiments and adds no scheduler work class.

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

## 5. Data model

### 5.1 Authoritative entities

Experiment phase/status, worker attempt, operation, PID/process group,
executable, kernel process-start identity, capacity class, exit/failure state,
scheduler version, and recovery metadata.

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
recovery uses durable evidence plus process checks. Legacy active rows without
complete identity are rejected for signaling until conservatively adopted.

## 6. Transactions

### 6.1 Read paths

Polling may read candidates optimistically, but eligibility is rechecked in the
claim transaction.

### 6.2 Write paths

Claims, start metadata, completion, recovery, and operator transitions have
short explicit transactions. No transaction remains open across process launch
or worker execution.

### 6.3 Failure semantics

Launch failure, worker failure, stale claim, and persistence failure remain
distinct. Recovery never marks work complete without authoritative evidence.

## 7. Concurrency

### 7.1 Conflict domain

The same experiment phase/attempt and shared capacity class may conflict.

### 7.2 Locking and serialization

Claims use database-visible conditional state/locks. Capacity is enforced
without globally serializing unrelated status inspection or completed work.
Multi-row locks require stable order.

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

### 8.3 Human output

Summaries distinguish queued, claimed, running, recovering, failed, and
completed work without overstating process observation.
Scheduler status lists active checkpoint inference jobs separately and reports
only processes that fail both experiment and checkpoint-evaluation ownership
validation as unmanaged.

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
- [Volume VII](Volume_VII_Experiment_Lifecycle.md)
- [Volume XII](Volume_XII_Database.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established scheduler ownership and safety outline. | ADR-0004 |
| 0.2.0 | 2026-07-24 | Accepted atomic claim/attempt hardening, preserved ordinary experiment work classes, and gated Campaign Operations production dispatch pending implementation and independent verification. | ADR-0004, ADR-0016 |
| 0.2.1 | 2026-07-25 | Documented checkpoint-evaluation worker ownership and status accounting. | ADR-0004 |
