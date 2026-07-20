# ADR-0006: Phase 6A follow-up proposal is advisory and identity-bound

Status: Accepted
Date: 2026-07-20
Deciders: Project architecture
Affected volumes: Volume VIII §§2–5, 9, 11
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Phase 5A produces one immutable campaign outcome assessment, and Phase 5C
produces one immutable advisory policy decision that may mark the exact
assessment eligible for later operator review. That eligibility is not itself
an approval, authorization, persisted review, experiment, or scheduler work
item. The first Phase 6 increment needs a deterministic candidate that preserves
the exact upstream identity without colliding conceptually with Phase 4D
campaign materialization or silently reopening the scientific decision.

## 2. Decision

Phase 6A introduces the distinct pure domain entity
`RecommendationCampaignFollowUpProposal` with contract version 1.

- It binds exactly one Phase 5A assessment v2 and one Phase 5C policy decision
  v1 containing one policy v1.
- Construction requires exact canonical-text, hash, campaign,
  materialization, ordered-member, count, and embedded-assessment alignment.
- Construction requires the Step 5C result to be eligible for operator review,
  favorable, and explicitly non-authorizing. Phase 6A does not recalculate
  eligibility.
- Canonical text is authoritative identity. The tagged hash is only an
  accelerator, and canonical text is compared even when hashes match.
- Embedded canonical and delimiter-capable strings are byte-length framed.
- Canonical text is independent of observation time and bounded to 1,048,576
  bytes before immutable construction.
- The proposal is read-only, database-free, non-persistent, advisory,
  non-authoritative, unapproved, inactive, non-executing, non-authorizing,
  non-scheduler work, and does not declare campaign success.
- Phase 6A adds no repository, SQL, migration, service, CLI, formatter,
  persisted operator review, scheduler consumer, queue, worker, or lifecycle
  mutation.
- Persistence, review, activation, and execution remain separate possible
  later Phase 6 steps and receive no authority from this ADR.

## 3. Rationale and decision drivers

- Preserve exact upstream scientific and policy provenance without copying or
  reinterpreting their truth tables.
- Keep operator-review eligibility visibly separate from approval and action.
- Make collision handling safe by retaining authoritative canonical text.
- Avoid the misleading implication that an advisory proposal is Phase 4D
  materialization or any experiment lifecycle request.
- Keep later persistence and operational designs free to establish their own
  authorities, transactions, idempotency, and concurrency contracts.

## 4. Consequences

### 4.1 Positive consequences

- Identical validated Phase 5 inputs yield an identical Phase 6A identity
  regardless of locale or observation time.
- Every relevant upstream identity and explicit non-authority semantic is
  reviewable in one immutable candidate.
- Unsupported, mismatched, non-favorable, noneligible, authorizing, or
  oversized inputs fail closed with stable reasons.
- No database, scheduler, or execution dependency enters the pure boundary.

### 4.2 Negative consequences and trade-offs

- Binding complete upstream canonical text duplicates data and can produce a
  sizable canonical value.
- The 1 MiB ceiling may reject exceptionally large but otherwise valid Phase 5
  decisions; changing that ceiling requires explicit versioned design.
- Phase 6A alone cannot record or enact an operator decision.

### 4.3 Risks and mitigations

- Hash collision: compare authoritative canonical text after any hash match.
- Upstream identity drift: validate exact version/canonical/hash and direct
  campaign, materialization, and ordered-member fields.
- Accidental authority creep: compile-time false safety flags, canonical
  bindings, pure tests, and absence of persistence/scheduler integration.
- Oversized future payload: reject above the fixed byte ceiling before
  proposal construction.

## 5. Compatibility and migration

The change is additive C++ domain code and documentation. It requires no
schema, data, privilege, repository, CLI, runtime, or scheduler migration and
does not modify Phase 5A, 5B, or 5C contracts.

## 6. Verification and operational evidence

- Golden canonical text and independently fixed tagged hash.
- Exact upstream alignment and stable refusal-reason tests.
- Favorable/non-favorable, invariance, identity-sensitivity, IEEE edge, size,
  malformed-version, and construction-access tests.
- Focused normal, ASan, and UBSan runs plus Phase 4D/5 regression tests.
- Both existing Xcode targets compile the source in an incremental Release
  build; no Debug or clean build is required.

## 7. Alternatives considered

### 7.1 Generic materialization entity

Rejected because it could be confused with the persisted Phase 4D operation
that materializes an approved campaign into Phase 4C proposals.

### 7.2 Build from Step 5B formatted output

Rejected because presentation text is not the immutable typed Phase 5A/5C
contract and would weaken exact identity alignment.

### 7.3 Recompute Step 5C eligibility in Phase 6A

Rejected because it would duplicate policy truth, risk divergence, and reopen
scientific interpretation outside the versioned Step 5C owner.

### 7.4 Persist or approve the proposal in Phase 6A

Rejected because persistence, operator review, approval, activation, and
execution require distinct later authority and operational designs.

## 8. References

- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
- [Phase 5 outcome assessment and policy](../../Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst)
- [Phase 6A follow-up proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
- [ADR-0003](ADR-0003-advisory-recommendation-evaluation.md)
- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
- [ADR-0005](ADR-0005-manual-recommendation-conversion.md)

## 9. Revision history

| Date | Change |
|---|---|
| 2026-07-20 | Accepted the pure identity-bound, non-authorizing Phase 6A follow-up proposal. |
