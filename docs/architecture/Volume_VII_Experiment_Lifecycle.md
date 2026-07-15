# Volume VII — Experiment Lifecycle

Status: Foundation outline
Version: 0.1.0
Last revised: 2026-07-15

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
