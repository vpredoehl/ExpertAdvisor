# Volume V — Training Engine

Status: Foundation outline
Version: 0.1.0
Last revised: 2026-07-15

## 1. Purpose

Define deterministic training semantics, optimizer state, checkpoint creation,
resume compatibility, and bounded worker execution.

## 2. Scope

### 2.1 In scope

Training inputs, batching, loss, optimization, epochs, checkpoint cadence,
model persistence, resume behavior, progress, and training-worker outcomes.

### 2.2 Out of scope

Experiment queue ownership belongs to Volumes VII/XI; inference and analysis to
Volume VI; label calculation to Volume III.

### 2.3 Current implementation status

Training, checkpointing, and resume paths are implemented. Exact replay is
limited by provenance such as unpersisted RNG and compiled configuration; this
outline does not strengthen those guarantees retroactively.

## 3. Responsibilities

### 3.1 Owned responsibilities

The engine owns mathematical updates, optimizer progress, epoch accounting,
checkpoint contents, and worker result reporting.

### 3.2 Dependencies

Consumes data/labels/model contracts from Volumes II–IV and receives a bounded
invocation from the scheduler.

### 3.3 Prohibited responsibilities

Workers MUST NOT choose the next experiment, approve recommendations, alter
continuation policy, or claim scheduler capacity independently.

## 4. Architecture

### 4.1 Components

Invocation validator, dataset assembler, model/optimizer loader, training loop,
checkpoint writer, progress reporter, and final model publisher.

### 4.2 Control flow

Validate invocation → load/construct compatible state → train bounded epochs →
publish checkpoints/progress → publish final outcome.

### 4.3 Ownership boundaries

Training owns model math. Scheduler owns process lifecycle and experiment
transitions. Repositories own durable model and progress writes.

## 5. Data model

### 5.1 Authoritative entities

Training invocation, optimizer state, model checkpoint, completed model,
training attempt, epoch progress, and outcome.

### 5.2 Provenance and versions

Persist architecture, feature/label, optimizer, epoch, source lineage, and
implementation compatibility sufficient for the claimed replay level.

### 5.3 Invariants and legacy data

Resume requires ownership and compatibility checks. Missing optimizer or model
metadata is not guessed. Checkpoint interval is invocation identity, not
ordinary semantic experiment identity under the current contract.

## 6. Transactions

### 6.1 Read paths

Load model metadata and parameters consistently before computation.

### 6.2 Write paths

Publish each model/checkpoint and its required metadata atomically; progress
updates are independent bounded transactions.

### 6.3 Failure semantics

Failed training preserves completed checkpoints and diagnostics but never marks
an incomplete final model as complete.

## 7. Concurrency

### 7.1 Conflict domain

One active training attempt owns one claimed experiment operation.

### 7.2 Locking and serialization

Database attempt/claim state, not PID observation alone, prevents duplicate
ownership. Model IDs are unique and completed artifacts are not rewritten.

### 7.3 Winner, loser, and retry outcomes

Only the authoritative attempt may publish lifecycle results. Retries create or
reuse lineage according to explicit resume/retry policy, never silently.

## 8. CLI

### 8.1 Commands and validation

Worker invocations require complete validated configuration and bounded ranges.

### 8.2 Machine output

Progress and completion records include experiment/model ownership and epochs.

### 8.3 Human output

Summaries distinguish checkpoint progress, final completion, and failure.

## 9. Testing

### 9.1 Pure tests

Cover loss, optimizer, batching, epoch limits, checkpoint policy inputs, and
resume compatibility.

### 9.2 Persistence and migration tests

Cover atomic model save/load, ownership, metadata, and progress writes.

### 9.3 Concurrency and integration tests

Verify one attempt owns publication and interrupted runs recover predictably.

### 9.4 Regression boundaries

Changes preserve label, inference, continuation, recommendation, and scheduler
contracts unless explicitly amended.

## 10. Operational safety

### 10.1 Runtime isolation

Training starts only through an explicit operator invocation or scheduler-owned
claim; documentation and diagnostics never launch it.

### 10.2 Permissions and destructive operations

Workers receive only model/progress privileges required for their attempt.

### 10.3 Observability and recovery

Heartbeat/progress, logs, exit status, checkpoints, and durable attempt identity
support orphan detection and recovery.

## 11. Future extensions

### 11.1 Approved extension points

Versioned optimizer, loss, checkpoint, and model interfaces.

### 11.2 Deferred capabilities

Distributed training, alternative optimizers, adaptive schedules, and automatic
resource tuning.

### 11.3 Required decisions

Extensions require identity, determinism, resource, recovery, and compatibility
ADRs before scheduler integration.

## 12. References

- [Volume I §§3–10](Volume_I_Foundation.md)
- [Volume IV](Volume_IV_Model_Architecture.md)
- [Volume VII](Volume_VII_Experiment_Lifecycle.md)
- [Volume XI](Volume_XI_Scheduler.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established the training-engine outline. | — |
