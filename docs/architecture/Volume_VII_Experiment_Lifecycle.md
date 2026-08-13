# Volume VII — Experiment Lifecycle

Status: Exact-attempt lifecycle contract implemented
Version: 0.2.0
Last revised: 2026-07-29

## 1. Purpose

Define experiment configuration, identity, state transitions, lineage,
checkpoint/continuation policy interaction, and lifecycle auditability.

## 2. Scope

### 2.1 In scope

Experiment creation, semantic/invocation identity, statuses/phases, attempts,
models, checkpoints, continuation decisions, lineage, and terminal outcomes.

### 2.2 Out of scope

Worker computation belongs to Volumes V/VI; scheduler polling and capacity to
Volume XI; advisory recommendations to Volume VIII.

An executing scheduler phase is also fenced by
`active_scheduler_worker_attempt_id`. The lifecycle row authorizes the work
unit; the linked Volume XI attempt authorizes the exact execution. Completion,
failure, recovery, and cancellation must compare expected phase/status and
attempt ID. A lifecycle status change alone does not release capacity while
that durable attempt remains active.

Every destructive lifecycle mutation repeats the exact attempt binding in its
SQL predicate. A C++ lookup is diagnostic, not authority. Terminalization,
capacity release, lifecycle identity clearing, checkpoint transition,
requeueing, adoption, and abandonment must either update the exact linked
attempt and lifecycle row atomically or affect zero rows and fail closed.

For a stop-at-checkpoint decision, the checkpoint is the successful boundary of
the train phase. The exact train attempt becomes `completed` with
`checkpoint_stop_completed`; the same transaction records checkpoint evidence,
clears the exact active-attempt binding and worker identity mirrors, and writes
the canonical next operation (`infer` or `analyze`) or the established
cancellation destination. A replay may recognize that exact terminal attempt,
but cannot repeat the transition or touch a replacement attempt.

Automatic continuation is scheduler-authoritative mutation. Evaluation
decision persistence, child creation, source/decision linkage, and final commit
all carry and revalidate the same invocation/fence context. Expensive
evaluation may use an immutable snapshot outside locks, but commit must
revalidate the fence and all source predicates. An ownership loss rolls back
the candidate transaction and stops the scan with no queued child.

Checkpoint analysis has its own durable `checkpoint_analyze` attempt and uses
claim/work/finalize. The checkpoint-evaluation row is the lifecycle binding;
the attempt is the execution/capacity authority. Result persistence and policy
evaluation occur only in an exact fenced finalize transaction.

### 2.3 Current implementation status

The platform implements persisted experiment lifecycle, checkpoint evaluation,
continuation policy/decisions, recovery metadata, and lineage. This outline
does not change those contracts.

## 3. Responsibilities

### 3.1 Owned responsibilities

Own experiment configuration provenance, lifecycle state machine, attempt and
model linkage, explicit operator transitions, and continuation lineage.

### 3.2 Dependencies

Receives configurations from explicit queue commands and, only in a future
accepted design, a recommendation conversion service. Supplies bounded work to
Volume XI.

### 3.3 Prohibited responsibilities

Lifecycle code MUST NOT treat a recommendation approval as an experiment,
infer configuration from incomplete provenance, or let workers self-queue.

## 4. Architecture

### 4.1 Components

Identity/domain rules, experiment repository, lifecycle service, continuation
policy/planner, operator CLI, and scheduler-facing claim interface.

### 4.2 Control flow

Validate identity/configuration → persist explicit experiment → scheduler
claims phase work → worker reports bounded outcome → lifecycle service advances
or terminates → optional explicit continuation evaluation/queueing.

### 4.3 Ownership boundaries

Services validate transitions; repositories update atomically; scheduler owns
operational claims; workers return facts; continuation policy cannot bypass
ordinary experiment creation and capacity rules.

## 5. Data model

### 5.1 Authoritative entities

Experiment, invocation configuration, attempt, model, checkpoint, lineage,
continuation policy, continuation decision, and operational state.

### 5.2 Provenance and versions

Row ID, semantic identity, invocation identity, lineage, implementation
provenance, and mutable state remain distinct per Volume I §5.

### 5.3 Invariants and legacy data

State/phase shapes, model ownership, lineage, attempt ownership, and terminal
semantics require database enforcement. Legacy records are preserved and
excluded where authoritative mapping is impossible.

The authoritative ``experiment.current_operation`` values are ``train``,
``infer``, and ``analyze``. The nouns ``training``, ``inference``, and
``analysis`` may describe activities in prose, but they are not persisted
operation values. Administrative and checkpoint state is not an operation and
must be represented by its owning lifecycle/control fields.

## 6. Transactions

### 6.1 Read paths

Status/lineage inspection is read-only; scheduler eligibility reads a
consistent set required by its claim operation.

### 6.2 Write paths

Creation, transition plus audit metadata, claim, completion, and continuation
decision/child creation each have explicit atomic boundaries.

### 6.3 Failure semantics

No transition may leave status, phase, worker ownership, model linkage, or
decision history partially updated.

## 7. Concurrency

### 7.1 Conflict domain

Conflicts are scoped to the same experiment, attempt, semantic duplicate key,
or continuation decision.

### 7.2 Locking and serialization

Use row locks, conditional updates, unique constraints, or transaction advisory
locks with stable ordering; unrelated experiments remain concurrent.

### 7.3 Winner, loser, and retry outcomes

Exactly one claimant owns a phase attempt. Duplicate creation and repeated
terminal transitions return defined results and do not create partial lineage.

## 8. CLI

### 8.1 Commands and validation

Queue, status, pause/resume/cancel/retry, checkpoint, and continuation commands
are explicit and mutually exclusive where state could conflict.

### 8.2 Machine output

Events identify experiment, phase, prior/resulting status, attempt, model,
reason, and whether a write occurred.

### 8.3 Human output

Summaries distinguish requested state, persisted state, running work, and
future eligibility.

## 9. Testing

### 9.1 Pure tests

Cover identity, transition, continuation planning, inheritance, and validation.

### 9.2 Persistence and migration tests

Cover state shapes, ownership, lineage, atomic transitions, and legacy rows.

### 9.3 Concurrency and integration tests

Exercise duplicate queues, simultaneous controls, claims, orphan recovery, and
continuation child creation.

### 9.4 Regression boundaries

Lifecycle changes preserve training math, inference evidence, recommendations,
and scheduler capacity unless explicitly scoped.

## 10. Operational safety

### 10.1 Runtime isolation

Tests never target real experiments; lifecycle mutations require exact IDs and
operator authorization.

### 10.2 Permissions and destructive operations

Runtime roles receive transition-specific access; history and lineage are not
silently deleted.

### 10.3 Observability and recovery

Durable attempts, PIDs, operation names, timestamps, exit status, and reasons
support reconciliation without treating process presence as sole truth.

## 11. Future extensions

### 11.1 Approved extension points

Versioned configuration contracts, explicit lifecycle services, and scheduler
claim interfaces.

### 11.2 Deferred capabilities

Recommendation conversion, experiment campaigns, budgets, and cross-machine
attempt execution.

### 11.3 Required decisions

Each needs identity, authorization, capacity, idempotency, audit, and rollback
ADRs before implementation.

## 12. References

- [Volume I §§5–10](Volume_I_Foundation.md)
- [ADR-0002](adr/ADR-0002-deterministic-experiment-identity.md)
- [Volume VIII](Volume_VIII_Recommendation_Engine.md)
- [Volume XI](Volume_XI_Scheduler.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established the experiment-lifecycle outline. | — |
| 0.1.1 | 2026-07-27 | Defined the sole canonical `current_operation` lifecycle vocabulary. | ADR-0004 |
| 0.2.0 | 2026-07-29 | Bound lifecycle mutation, continuation, and checkpoint analysis to generation-52 exact attempts and scheduler authority. | ADR-0018 |
| 0.2.1 | 2026-07-30 | Defined atomic exact-attempt completion and idempotent replay for stop-at-checkpoint transitions. | ADR-0018 |
