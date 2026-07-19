# ADR-0005: Manual recommendation conversion creates a paused experiment

Status: Accepted
Date: 2026-07-19
Deciders: Project architecture
Affected volumes: Volume VII; Volume VIII §11; Volume XI
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Phase 4C persists an exact conversion proposal and a separate append-only human
review history. A later explicit operation must be able to materialize an
approved proposal without treating approval as execution, producing duplicate
experiments under retries, or making the scheduler an implicit consumer.

## 2. Decision

An explicit manual command may convert one proposal only when its latest
serialized review decision is `approve`.

- Proposal integrity is revalidated from its authoritative canonical evidence.
- A transaction-scoped proposal-specific advisory lock serializes review writes
  and conversion without granting runtime UPDATE access to immutable proposals.
- One transaction resolves the current review, inserts one `paused`/`train`
  experiment, and inserts immutable conversion evidence.
- A unique proposal reference makes retries and concurrent identical requests
  converge on the same experiment.
- The audit row links the exact proposal, approving decision, and experiment.
- The experiment is not pending, queued, running, resumed, or scheduler-owned.
- A later explicit lifecycle command is required before any execution.
- A completed conversion remains historical fact if a later review decision
  reverses the proposal disposition.

## 3. Rationale and decision drivers

- Keep approval and materialization as separate durable events.
- Preserve complete recommendation-to-experiment provenance.
- Prevent ambiguous retry outcomes and duplicate experiment rows.
- Keep worker launch, capacity, and continuation behavior unchanged.

## 4. Consequences

### 4.1 Positive consequences

- Manual conversion is atomic, auditable, and concurrency-safe.
- Operators can inspect the created experiment before taking lifecycle action.
- Scheduler restart alone cannot execute a newly converted proposal.

### 4.2 Negative consequences and trade-offs

- Conversion creates a paused experiment that requires another explicit action.
- A conflicting pre-existing experiment identity prevents conversion rather
  than silently linking unrelated work.
- Review and conversion for the same proposal serialize on one advisory key.

### 4.3 Risks and mitigations

- Concurrent rejection versus conversion: both take the same transaction lock;
  lock order determines which explicit operation is observed first. A lock-hash
  collision only adds serialization and cannot establish proposal equality.
- Configuration drift: the repository reconstructs and compares canonical
  proposal and experiment invocation identities.
- Retry after an uncertain response: the immutable execution row is returned.

## 5. Compatibility and migration

Migration 038 is additive. It creates only conversion-owned audit state and a
composite review/proposal uniqueness index. It does not change scheduler,
continuation, recommendation scoring, ranking, or review semantics.

## 6. Verification and operational evidence

- Migration constraints, restrictive foreign keys, and runtime privileges.
- Pending/rejected denial and approved creation tests.
- Concurrent identical conversion and exact replay tests.
- Atomic rollback and immutable provenance tests.
- CLI mutual exclusion and proof that the experiment remains paused.

## 7. Alternatives considered

### 7.1 Create a pending experiment

Rejected because scheduler startup could then execute conversion output without
a separate explicit operational decision.

### 7.2 Convert as part of approval

Rejected because it collapses human governance and experiment creation into one
event and makes review retries operationally consequential.

### 7.3 Queue immediately after conversion

Rejected because queueing, capacity, worker launch, and automation are outside
Phase 4C Step 4 and remain under ADR-0004.

## 8. References

- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
- [ADR-0003](ADR-0003-advisory-recommendation-evaluation.md)
- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)

## 9. Revision history

| Date | Change |
|---|---|
| 2026-07-19 | Accepted explicit, idempotent conversion to one paused experiment. |
