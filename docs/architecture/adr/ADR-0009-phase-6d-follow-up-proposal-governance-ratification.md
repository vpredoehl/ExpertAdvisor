# ADR-0009: Phase 6D ratifies governance advancement after approved review

Status: Accepted
Date: 2026-07-22
Deciders: Project architecture
Affected volumes: Volume VIII §§2–11; Volume XII §§2–11
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Phase 6C already records one immutable approved or rejected administrative
review of an exact persisted Phase 6B proposal. Its positive decision answers
whether the proposal is acceptable on its merits for possible advancement.
Repeating that approval in Phase 6D would create no distinct authority.

The next increment instead needs explicit governance evidence that an
independent authorized actor has inspected the exact approved review and full
provenance and ratified advancement into the next separately controlled
phase. That evidence must not itself grant the next phase's capabilities.

## 2. Decision

Phase 6D defines pure immutable governance ratification, additive migration
044, an exact-binding append-only repository, and one transactional service.

- Phase 6C remains the merits review with `approved` or `rejected` outcomes.
- Phase 6D records only `ratified`: advancement of the exact approved review
  into the next separately controlled phase.
- Ratifier identity MUST differ byte-for-byte from the Phase 6C reviewer
  identity. Self-ratification fails closed.
- The authority role is fixed as `follow_up_governance_ratifier`; callers do
  not supply arbitrary role text.
- Contract version 1 binds the exact review-event ID, review and proposal
  version/canonical/hash identities, approved review decision, reviewer,
  fixed role, ratified decision, ratifier, basis, separation rule, and fixed
  negative authority semantics.
- Canonical text is authoritative, byte-length framed, locale-independent,
  and bounded. The tagged hash is only an accelerator.
- Migration 044 stores one ratification per review/proposal with restrictive
  provenance, approved eligibility, fixed-role, ratified-decision, and
  separation-of-duties enforcement.
- A review-event-scoped transaction advisory lock serializes writes. First
  persistence is `recorded`, exact replay is `existing_identical`, and any
  changed payload conflicts.
- Runtime access is table `SELECT`, payload-column `INSERT`, and sequence
  `USAGE` only. Existing ratifications cannot be updated or removed.
- Phase 6D adds no CLI or operational adapter.

Ratification is governance evidence only. It grants no Campaign Operations
capability, follow-up authorization, execution authorization, or lifecycle
authority. Any later operational consumer requires its own accepted authority.

## 3. Rationale and decision drivers

- Give Phase 6D a materially distinct governance question and consequence.
- Require independent human accountability at the advancement boundary.
- Consume authoritative persisted evidence without recomputing Phase 5 or
  Phase 6A truth.
- Preserve canonical-first collision safety and deterministic retry behavior.
- Keep every operational capability absent until separately designed and
  accepted.

## 4. Consequences

### 4.1 Positive consequences

- The audit trail distinguishes proposal merits review from governance
  ratification of advancement.
- One actor cannot both approve the Phase 6C review and ratify it in Phase 6D.
- Full typed provenance and defense-in-depth database enforcement detect stale,
  forged, malformed, or self-ratified evidence.
- Equivalent retries and concurrent requests converge without replacing
  history.

### 4.2 Negative consequences and trade-offs

- A second governance actor is required before a proposal can be ratified.
- Complete review and proposal canonical text is duplicated and may exceed two
  MiB per event.
- A mistaken ratification cannot be changed in place; correction would require
  a separately accepted append-only design.
- No main-program CLI exists in this increment.

### 4.3 Risks and mitigations

- Semantic duplication: distinct question, `ratified` decision, fixed role,
  separation policy, and advancement-only canonical semantics.
- Self-ratification: pure, service, repository, check-constraint, and trigger
  enforcement.
- Hash collision: compare complete authoritative canonical and typed payload.
- Upstream drift: restrictive foreign keys, joined provenance trigger, and
  validated Phase 6C repository reload.
- Privilege escalation: explicit PUBLIC/runtime revokes, column grants,
  invoker rights, pinned search path, and NULL-ACL fallback tests.

## 5. Compatibility and migration

Migration 044 and its Phase 6D domain, repository, service, and tests were
implemented and committed before this acceptance alignment. Migration 044 is
additive after migration 043 and does not alter Phase 6A, 6B, or 6C rows,
identities, APIs, or privileges. It touches no experiment, model,
continuation, lifecycle, queue, Campaign Operations, or scheduler table.

## 6. Implementation implications

- The Phase 6D service validates and persists governance evidence only; it has
  no Campaign Operations, lifecycle, experiment, scheduler, or worker adapter.
- Consumers may validate the exact immutable ratification as prerequisite or
  provenance, but must also possess their own accepted operational authority.
- The historical negative canonical field remains byte-compatible and cannot
  be reinterpreted as a roadmap phase or positive capability.
- Existing migration 044 rows and canonical identities remain unchanged.

## 7. Verification and operational evidence

- Pure warnings-as-errors tests for immutable semantics, fixed role and
  decisions, golden canonical/hash, locale, UTF-8/DEL validation, separation
  of duties, identity sensitivity, and negative authority.
- Isolated PostgreSQL clean-install and 043-upgrade tests for repeatability,
  exact ACL/NULL-ACL behavior, provenance, fixed role, distinct actors,
  service syntax, replay/conflict, concurrency, rollback, malformed storage,
  collision handling, read-only queries, and upstream sentinels.
- Phase 6A/6B/6C regression tests, exact Xcode project membership, diff audit,
  and an incremental Release build only when scheduler-worker load is safe.

## 8. Alternatives considered

### 8.1 Keep a second administrative approval

Rejected because it repeats Phase 6C without a distinct question, role,
policy, or consequence.

### 8.2 Allow the Phase 6C reviewer to ratify

Rejected because no committed authority permits self-ratification and it
would remove the independent governance check that distinguishes Phase 6D.

### 8.3 Treat Phase 6C approval as automatic ratification

Rejected because merits review does not establish the separate governance
decision to advance.

### 8.4 Add activation or execution authority

Rejected because the next phase requires a separate accepted capability and
operational safety design.

## 9. Relationships to other ADRs

- ADR-0006 through ADR-0008 define the exact advisory proposal, persistence,
  and administrative merits-review chain consumed by Phase 6D.
- ADR-0010 permits a ratification to be optional Campaign Operations
  prerequisite/provenance while retaining Phase 4D as the only V1 origin.
- ADR-0011 remains the only operational permission to act.

## 10. References

- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
- [Volume XII](../Volume_XII_Database.md)
- [Phase 6A proposal](../../Phase6ARecommendationCampaignFollowUpProposal.rst)
- [Phase 6B persistence](../../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
- [Phase 6C review](../../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
- [Phase 6D ratification](../../Phase6DRecommendationCampaignFollowUpProposalRatification.rst)
- [ADR-0001](ADR-0001-postgresql-source-of-truth.md)
- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
- [ADR-0006](ADR-0006-phase-6a-follow-up-proposal.md)
- [ADR-0007](ADR-0007-phase-6b-follow-up-proposal-persistence.md)
- [ADR-0008](ADR-0008-phase-6c-follow-up-proposal-administrative-review.md)

## 11. Revision history

| Date | Change |
|---|---|
| 2026-07-22 | Proposed distinct Phase 6D governance ratification with mandatory separation of duties and no Phase 6E capability. |
| 2026-07-24 | Accepted the implemented governance-only contract, aligned migration history, and clarified that later operational authority must be separate. |
