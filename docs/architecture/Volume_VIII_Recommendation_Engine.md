# Volume VIII — Recommendation Engine

Status: Foundation outline aligned with Phase 4A Steps 1–5
Version: 0.1.0
Last revised: 2026-07-15

## 1. Purpose

Define the advisory research-recommendation subsystem: deterministic identity,
candidate generation, persistence, duplicate handling, scoring/ranking, and
explicit human review.

## 2. Scope

### 2.1 In scope

Recommendation policy and identities, pure candidates, scans, source evidence,
duplicates, immutable scores/components, deterministic ranks/explanations,
terminal review transitions, and immutable review events.

### 2.2 Out of scope

Recommendation-to-experiment conversion, profitability claims, queueing,
scheduler polling, worker execution, and automatic approval/rejection/expiration.

### 2.3 Current implementation status

Phase 4A Steps 1–5 implement the in-scope capabilities. Detailed current
contracts remain in the Phase 4 documents referenced in §12. This volume
organizes, but does not redesign, them.

## 3. Responsibilities

### 3.1 Owned responsibilities

Own deterministic proposal identity/generation, authoritative persistence and
duplicate provenance, advisory score history, explicit review disposition, and
their presentation.

### 3.2 Dependencies

Consumes completed experiment/final-analysis evidence from Volumes VI/VII.
There is no implemented downstream execution dependency.

### 3.3 Prohibited responsibilities

The subsystem MUST NOT create experiments, queue work, mutate experiment or
scheduler state, infer reviewer identity, use score thresholds for status, or
claim expected profitability/correctness.

## 4. Architecture

### 4.1 Components

- Pure identity/policy and candidate-generation domain components.
- Repository-owned PostgreSQL mapping, scans, duplicates, scores, and reviews.
- Services for explicit generation, scoring, inspection, and review use cases.
- CLI parsing/validation/presentation outside scheduler polling.

### 4.2 Control flow

Completed evidence → deterministic source selection → pure candidates →
collision-aware persistence → optional explicit score run → optional explicit
operator review. Every arrow is separately invoked and auditable.

### 4.3 Ownership boundaries

Canonical text decides identity; repository transactions decide persistence;
pure scoring policy decides numeric evidence; pure review rules decide legal
transitions; the operator supplies the review action. No stage owns conversion.

## 5. Data model

### 5.1 Authoritative entities

Recommendation policy, semantic/invocation configuration, recommendation scan,
recommendation, score run/result/component, and review event.

### 5.2 Provenance and versions

Canonical text is authoritative; tagged hashes accelerate lookup. Persisted
records retain source experiment/analysis/model, scan, policy, identity,
structural mutation, source rank, score policy/components, and review snapshot.

### 5.3 Invariants and legacy data

Scan-associated rows require positive source rank and cannot detach from their
originating scan. Proposed/approved identities remain active; rejected/expired
matches remain historical blockers. Legacy NULL/NULL scan/rank rows are not
backfilled and are excluded from operations requiring provenance.

Review transitions are only proposed → approved, rejected, or expired. Review
events are append-only to the runtime role. Approval is advisory and leaves
`approved_experiment_id` null under Step 5.

## 6. Transactions

### 6.1 Read paths

Evidence, list, detail, score history, and review history use typed read-only
repository paths.

### 6.2 Write paths

Candidate persistence uses short collision-aware transactions. Score result and
ordered components persist atomically. Review locks one recommendation and
updates status plus one immutable event in the same transaction.

### 6.3 Failure semantics

Invalid commands create no scan/event. Candidate failures are counted per scan.
Score retries compare complete immutable results. Review event failure rolls
back status; terminal retries create no event.

## 7. Concurrency

### 7.1 Conflict domain

Generation conflicts on semantic configuration plus policy; scoring retries on
run plus recommendation; review conflicts on one recommendation row.

### 7.2 Locking and serialization

Generation uses a transaction advisory key plus canonical rechecks and active
uniqueness. Scoring uses uniqueness and complete retry comparison. Review uses
`SELECT ... FOR UPDATE` and one-event uniqueness. Unrelated work is concurrent.

### 7.3 Winner, loser, and retry outcomes

Concurrent identical generation returns one created/one existing result.
Equivalent score retry reuses only an exact persisted result. Concurrent review
produces one terminal winner and one `recommendation_review_status_conflict`;
exactly one event exists.

## 8. CLI

### 8.1 Commands and validation

Generation, scoring, score inspection, review actions, and review inspection
are explicit families. Review mutation is mutually exclusive with generation,
scoring, scheduler, continuation, and experiment-lifecycle commands.

### 8.2 Machine output

`EXPERIMENT_RECOMMENDATION_*` records use stable fields, percent-escaped text,
explicit `NULL`, deterministic ordering, and separate conflict/failure events.

### 8.3 Human output

Human score/review summaries are concise, control-byte safe, and state that the
result is advisory and created/queued no experiment.

## 9. Testing

### 9.1 Pure tests

Cover canonical identity/collisions, eligibility, candidates, scoring formulas,
ranking, review parsing/reasons, and all transition outcomes.

### 9.2 Persistence and migration tests

Cover scans, duplicates, source rank/scan provenance, score ownership and
immutability, review constraints, runtime privileges, SQLSTATEs, and migration
repeatability.

### 9.3 Concurrency and integration tests

Cover same-candidate generation, score retries, review action races, unrelated
reviews, rollback, exact cleanup, and before/after experiment/identity/score
digests.

### 9.4 Regression boundaries

Steps 1–5 remain individually stable. Changes MUST NOT alter training,
inference, continuation, experiment lifecycle, or scheduler behavior.

## 10. Operational safety

### 10.1 Runtime isolation

No recommendation command enters scheduler polling. Integration uses disposable
rows and exact cleanup without disturbing genuine experiments.

### 10.2 Permissions and destructive operations

Runtime access to review history is SELECT/INSERT only. Owner/test connections
perform exact fixture cleanup. Other immutable histories follow Volume I §7.3.

### 10.3 Observability and recovery

Scans/runs expose completion and counters; scores/events retain immutable
provenance; explicit conflicts distinguish safe rejection from system failure.

## 11. Future extensions

### 11.1 Approved extension points

Versioned policies, pure candidate/scoring components, typed repository/service
APIs, and separate inspection commands.

### 11.2 Deferred capabilities

Recommendation conversion, multi-source attribution, automatic research
campaigns, profitability evidence, and scheduler-managed recommendation work.

### 11.3 Required decisions

Conversion requires a new ADR defining authorization, complete implementation
identity, idempotent experiment creation, budgets, transactions, audit history,
and scheduler capacity. Approval alone can never imply conversion.

## 12. References

- [Volume I §§5–10](Volume_I_Foundation.md)
- [ADR-0003](adr/ADR-0003-advisory-recommendation-evaluation.md)
- [Recommendation foundation](../Phase4AExperimentRecommendationFoundation.rst)
- [Recommendation persistence](../Phase4AExperimentRecommendationPersistence.rst)
- [Recommendation scoring](../Phase4AExperimentRecommendationScoring.rst)
- [Recommendation review](../Phase4AExperimentRecommendationReview.rst)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established the Step 1–5-aligned recommendation architecture outline. | ADR-0003 |
