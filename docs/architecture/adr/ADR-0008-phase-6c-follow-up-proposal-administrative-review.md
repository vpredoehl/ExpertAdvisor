# ADR-0008: Phase 6C records one immutable non-authorizing administrative review

Status: Accepted
Date: 2026-07-21
Deciders: Project architecture
Affected volumes: Volume VIII §§2–11; Volume XII §§2–11
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Phase 6B durably preserves an exact Phase 6A advisory follow-up proposal but
intentionally records no operator decision. An operator must be able to approve
or reject one persisted proposal without turning approval into activation,
execution, follow-up authorization, scheduling, queueing, experiment mutation,
or campaign-success evidence. Concurrent attempts, exact retries, corrupted
storage, and owner-level provenance drift must have deterministic fail-closed
behavior.

## 2. Decision

Phase 6C defines one pure immutable review value, migration 043 with one
append-only review-event table, an exact-binding repository, and read-only show
and list presentation.

- Contract version 1 binds persisted proposal ID, exact Phase 6A
  version/canonical/hash, approved or rejected decision, reviewer, and reason.
- Canonical text is authoritative and byte-length-prefixed; the tagged hash is
  an accelerator. Generated event ID and timestamp remain outside identity.
- The table has one restrictive Phase 6B foreign key, exact-provenance insert
  enforcement, bytewise canonical collation, and one-event-per-proposal
  uniqueness.
- A proposal-ID-scoped transaction advisory lock serializes review attempts.
  First persistence is ``recorded``, exact replay is ``existing_identical``,
  and every non-identical attempt for the same proposal conflicts.
- Reload reconstructs the immutable value, validates its full canonical
  identity, reloads Phase 6B, and compares the exact proposal binding.
- Show/list run repeatable-read and read-only, order lists by descending event
  ID, and render explicit negative authority fields.
- Runtime access is SELECT, payload-column INSERT, and sequence USAGE only.
  Existing events are never updated, deleted, expired, or superseded.

An approved event means only administrative approval for possible
consideration by a later explicitly authorized phase.

## 3. Rationale and decision drivers

- Make the operator decision explicit without weakening the Phase 6A/6B
  advisory boundary.
- Bind review to authoritative canonical proposal content rather than trusting
  a hash or row ID alone.
- Make rejected decisions as durable and auditable as approved decisions.
- Give exact retry and concurrent conflict behavior one deterministic outcome.
- Keep later action authority absent from both schema and API.

## 4. Consequences

### 4.1 Positive consequences

- One persisted proposal has at most one immutable administrative decision.
- Exact retries converge and differing reviewer/reason/decision payloads cannot
  silently replace history.
- Constraints, trigger provenance, typed hydration, canonical recomputation,
  and Phase 6B reload provide defense in depth.
- Operators can inspect review history without locks, writes, or sequence use.

### 4.2 Negative consequences and trade-offs

- Proposal canonical text is duplicated in the review event to preserve exact
  self-contained binding and may approach one MiB.
- Phase 6C cannot correct an operator mistake in place; a future explicitly
  designed phase would need a separate append-only policy if correction is
  ever authorized.
- List hydration reloads each referenced Phase 6B proposal to validate current
  durable agreement.

### 4.3 Risks and mitigations

- Authority creep: no action/lifecycle columns or adapters, plus fixed negative
  canonical and presentation fields.
- Concurrent disagreement: one proposal-scoped transaction advisory lock and
  unique proposal ID.
- Hash collision or corruption: canonical equality is authoritative and all
  hashes are recomputed on reload.
- Upstream drift: restrictive foreign key, insert trigger, and repository
  comparison against immutable Phase 6B storage.

## 5. Compatibility and migration

Migration 043 is additive after migration 042. It alters no Phase 6A/6B row,
experiment table, scheduler table, or lifecycle schema. Existing Phase 6A and
6B contract semantics and identities remain unchanged. Rollback, if ever
operationally approved, consists only of separately removing the new Phase 6C
objects; no production rollback is performed by the implementation work.

## 6. Verification and operational evidence

- Pure contract tests for both decisions, immutable types, deterministic and
  locale-independent identity, sensitivity, malformed inputs, and negative
  authority invariants.
- Migration and repository tests for repeatability, C collation, restrictive
  FK, least privilege, generated columns, exact round trip/replay, all conflict
  forms, concurrency, rollback, malformed storage, and provenance mismatch.
- Read-only show/list tests for not-found, ordering, limit validation, exact
  safety output, unchanged sequences/tables, and absence of advisory/tuple
  locks.
- Phase 6A/6B, Phase 5, relevant Phase 4D, project validation, and Release
  build regressions.

## 7. Alternatives considered

### 7.1 Add approved state to the Phase 6B proposal row

Rejected because it mutates advisory proposal evidence and obscures the
separate administrative decision.

### 7.2 Treat persistence or preview as approval

Rejected because observation and storage do not establish reviewer identity or
intent.

### 7.3 Let approval activate or queue follow-up work

Rejected because action authority requires a later explicit contract,
transaction, and operational safety decision.

### 7.4 Allow multiple current/superseding review states

Rejected for Phase 6C. One immutable effective decision gives deterministic
replay and conflict semantics without inventing expiration or correction
policy.

## 8. References

- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
- [Volume XII](../Volume_XII_Database.md)
- [Phase 6A proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
- [Phase 6B persistence and preview](../../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
- [Phase 6C administrative review](../../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
- [ADR-0001](ADR-0001-postgresql-source-of-truth.md)
- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
- [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md)
- [ADR-0007](ADR-0007-phase-6b-follow-up-proposal-persistence.md)

## 9. Revision history

| Date | Change |
|---|---|
| 2026-07-21 | Accepted exact append-only Phase 6C administrative review without action authority. |
