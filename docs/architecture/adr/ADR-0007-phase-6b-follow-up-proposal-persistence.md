# ADR-0007: Phase 6B persists exact advisory proposals for read-only preview

Status: Accepted
Date: 2026-07-20
Deciders: Project architecture
Affected volumes: Volume VIII §§2–9, 11; Volume XII
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Phase 6A produces a pure immutable, non-authorizing follow-up proposal with a
complete canonical identity. Operators need durable, repeatable inspection of
that exact value. Persistence must not turn eligibility into approval or
introduce activation, execution, scheduling, experiment creation, or follow-up
authority. Reload must also reject database corruption rather than returning a
mutable or partially validated approximation.

## 2. Decision

Phase 6B adds one append-only PostgreSQL manifest plus ordered member table,
one collision-aware repository, a private validated hydration seam, and a
read-only preview service.

- The manifest and members preserve every public Phase 6A field exactly.
- Canonical text is authoritative; the tagged hash only selects a lock/index
  bucket. Same-hash/different-canonical proposals receive different storage
  collision ordinals.
- Exact retries return the existing row. A hash-scoped transaction advisory
  lock serializes concurrent exact persistence.
- Upstream approval/materialization/member triggers validate copied immutable
  provenance, and a deferred trigger requires complete ordered membership.
- Reload reconstructs the non-assignable Phase 6A object through one private
  friend builder and reruns direct alignment plus full canonical/payload
  invariants.
- Preview runs under repeatable-read, read-only isolation and renders explicit
  negative authority fields.
- Runtime privileges are append-only and column-scoped. Phase 6B provides no
  approval, activation, execution, queue, scheduler, experiment, or follow-up
  authorization API or schema.

## 3. Rationale and decision drivers

- Preserve the identity produced by Phase 6A without parsing presentation
  output or recomputing Phase 5 science/policy.
- Make persisted corruption and duplicate exact identity fail closed.
- Keep collision safety aligned with existing canonical-first repositories.
- Make operator inspection useful while keeping it observably read-only.
- Prevent persistence metadata such as row ID, collision ordinal, and creation
  timestamp from changing proposal identity.

## 4. Consequences

### 4.1 Positive consequences

- A persisted proposal reloads as the same immutable typed value.
- Exact canonical and every upstream canonical/hash pair survive round trip.
- Exact retries are idempotent and hash collisions do not merge identities.
- Database constraints, triggers, privileges, and C++ hydration provide
  defense in depth against incomplete or malformed records.
- Preview cannot mutate lifecycle or scheduler state.

### 4.2 Negative consequences and trade-offs

- Full canonical provenance is duplicated and may approach the Phase 6A 1 MiB
  ceiling.
- Inserts depend on existing immutable Phase 4D approval/materialization rows.
- Owner-level database corruption remains possible, but repository reload
  detects it and fails closed.
- Previewing full authoritative canonical text can produce large output.

### 4.3 Risks and mitigations

- Hash collision: lock/index by hash, then compare canonical text exactly and
  retain a collision ordinal outside identity.
- Partial manifest: insert in one transaction and enforce member completeness
  with a deferred constraint trigger.
- Authority creep: no decision/lifecycle schema or API and explicit false
  safety fields in every preview record.
- Mutable reconstruction: return the original const-field Phase 6A type and a
  non-assignable persisted wrapper.

## 5. Compatibility and migration

Migration ``042`` is additive and depends on the existing Phase 4D approval
and materialization tables. It does not alter experiment or scheduler tables.
The Phase 6A public builder and canonical grammar remain version 1 and
unchanged; constructor validation is strengthened for exact hydration.

## 6. Verification and operational evidence

- Repeatable migration, exact privilege, FK, trigger, index, and collation
  assertions.
- Exact round-trip, immutable reload, canonical/hash preservation, idempotent
  duplicate replay, duplicate persisted identity rejection, and malformed row
  rejection.
- Preview content and repeatable-read/read-only behavior with an untouched
  sequence sentinel.
- Phase 6A, Phase 5A/B/C, relevant Phase 4D regression tests, Release build,
  and whitespace audit.

## 7. Alternatives considered

### 7.1 Store only the proposal hash

Rejected because hashes are not authoritative and cannot reconstruct or
inspect exact provenance.

### 7.2 Store only canonical text and parse it on every read

Rejected because Phase 6A intentionally defines canonical generation, not a
public parser, and normalized member rows support provenance constraints and
bounded typed reload without making presentation text authoritative input.

### 7.3 Return a mutable persistence DTO

Rejected because callers could mistake reconstructed fields for a validated
Phase 6A proposal and silently lose immutability guarantees.

### 7.4 Add operator approval or execution state

Rejected as a distinct later authority explicitly outside Phase 6B.

## 8. References

- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
- [Phase 6A proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
- [Phase 6B persistence and preview](../../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
- [ADR-0001](ADR-0001-postgresql-source-of-truth.md)
- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
- [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md)

## 9. Revision history

| Date | Change |
|---|---|
| 2026-07-20 | Accepted exact append-only Phase 6B proposal persistence and read-only preview. |
