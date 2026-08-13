# ADR-0010: Campaign Operations ownership and bounded V1 scope

Status: Accepted
Date: 2026-07-24
Deciders: Project architecture
Affected volumes: Volume VII §§2–7; Volume VIII §§2–5; Volume X §§1–11;
Volume XI §§2–7; Volume XII §§2–7
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Recommendation Governance already owns proposal policy, review, approval, and
one immutable Phase 4D materialization. Phase 5 can create and activate the
materialization's ordinary experiments through accepted lifecycle workflows,
and the scheduler owns claims and worker execution. No accepted owner existed
for the durable coordination between those boundaries.

Without an explicit owner, materialization, ratification, budget, lifecycle
state, and scheduler capacity could each be misread as execution authority.
Campaign identity and its relationship to the reserved Volume X automation
outline also require a fixed boundary.

## 2. Decision

Campaign Operations is the sole owner of operational campaign orchestration.
V1 has these mandatory boundaries:

- One operational campaign binds one exact, fully validated, immutable Phase
  4D materialization. The materialization ID is the V1 natural key; at most one
  operational campaign may bind it.
- The only V1 origin kind is `phase4d_materialization_v1`. Phase 6D
  ratification may be exact prerequisite or provenance evidence, but is never
  an executable origin, membership source, or operational authorization.
- Phase 4D ordered materialization members remain the sole membership
  authority. Campaign Operations MUST NOT add, remove, reorder, recompute, or
  reinterpret proposals or materializations.
- Campaign Operations owns campaign identity, operational grants, budget,
  reservations, durable requests, orchestration attempts, bindings, campaign
  controls, reconciliation observations, completion, and audit attribution.
- Recommendation Governance retains proposal, review, approval,
  materialization, Phase 5 assessment, and Phase 6 governance authority.
- Experiment Lifecycle and the accepted Phase 4C/5 services retain experiment
  creation and lifecycle-transition authority.
- The scheduler retains eligibility selection, capacity, claims, process
  launch/supervision, attempts, and execution. Experiments created through a
  Campaign Operations handoff are ordinary scheduler resources.
- Workers retain bounded computation only. PostgreSQL retains committed
  durable truth.
- Campaign Operations refines only the bounded operational-coordination portion
  of Volume X. It is not Recommendation Governance Phase 6E and does not
  authorize broader autonomous research.

The immutable campaign row is the creation fact. Its existence directly
derives the initial administrative state
`awaiting_operational_authorization`; there is no durable draft or competing
readiness authority.

## 3. Rationale and decision drivers

- Preserve the accepted Phase 4 and Phase 5 ownership chain.
- Prevent governance evidence, money, or scheduler capacity from becoming
  implied permission.
- Give long-lived coordination one accountable owner without transferring
  lifecycle or scheduler authority.
- Reuse the exact immutable materialization instead of creating a second
  membership model.
- Keep the first implementation bounded and extensible by versioned,
  separately accepted contracts.

## 4. Consequences

### 4.1 Positive consequences

- Every durable fact and transition has one owner.
- Campaign identity and membership are deterministic and replayable.
- Direct Phase 4C/5 use remains valid and unchanged.
- Campaign Operations can evolve without becoming a scheduler or autonomous
  recommendation subsystem.

### 4.2 Negative consequences and trade-offs

- V1 cannot dispatch a subset of a materialization.
- Phase 6 follow-up evidence cannot itself create executable scope.
- Cross-owner handoffs require explicit transactions, bindings, and
  privileges.

### 4.3 Risks and mitigations

- Authority overlap: enforce the ownership table in the accepted Campaign
  Operations specification and least-privilege roles under ADR-0017.
- Duplicate campaigns: enforce the materialization natural key and compare
  complete canonical identity on replay.
- Scope creep: require a later accepted ADR for new origins, partial dispatch,
  adaptive policy, or a scheduler work class.

## 5. Compatibility and migration

This decision preserves all Phase 4–6 identities, rows, services, and
privileges. Migration 045 and the Phase 1 implementation already provide the
V1 immutable campaign, optional governance provenance, and authorization
foundation consistent with this decision.

Later persistence is additive. Existing recommendation campaigns and direct
Phase 4C/5 workflows are not backfilled, adopted, or converted implicitly.

## 6. Implementation implications

- Layering remains CLI or operator adapter → service/workflow → repository →
  PostgreSQL.
- Campaign creation loads the authoritative Phase 4D materialization through
  its validating workflow, then inserts the immutable campaign and audit
  reference in one transaction.
- Canonical text is authoritative; tagged hashes are lookup accelerators.
- No migration, command, schema presence, or budget enables runtime work by
  itself.

## 7. Verification and operational evidence

- Pure identity and canonical golden-vector tests.
- Migration uniqueness, provenance, collision, append-only, and ACL tests.
- Exact replay and changed-scope conflict tests.
- Regression tests proving Phase 4C/4D/5 and scheduler state are unchanged.
- Read-only status tests deriving the initial state after restart without a
  readiness row.

## 8. Alternatives considered

### 8.1 Extend Recommendation Governance with Phase 6E

Rejected because governance evidence is intentionally non-operational and does
not own budgets, requests, lifecycle, or scheduler execution.

### 8.2 Let the scheduler own campaigns

Rejected because the scheduler owns execution capacity and attempts, not
proposal scope, authorization, budget, or orchestration policy.

### 8.3 Create a second materialization or mutable campaign membership

Rejected because it would compete with the immutable Phase 4D membership
authority and break existing Phase 4/5 provenance.

## 9. Relationships to other ADRs

- ADR-0001 supplies PostgreSQL durable authority.
- ADR-0002 supplies deterministic identity principles.
- ADR-0003 and ADR-0005 preserve advisory and explicit conversion boundaries.
- ADR-0009 permits Phase 6D evidence only as governance provenance.
- ADR-0011 through ADR-0015 define the owned Campaign Operations contracts.
- ADR-0016 and ADR-0017 preserve scheduler and privilege separation.

## 10. References

- [Accepted Campaign Operations specification](../../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [Volume X](../Volume_X_Research_Automation.md)
- [Volume XI](../Volume_XI_Scheduler.md)
- [Phase 4D materialization](../../Phase4DExperimentRecommendationCampaignMaterialization.rst)
- [Phase 5 launch](../../Phase5ExperimentRecommendationCampaignLaunch.rst)
- [Migration 045](../../../Database/migrations/045_campaign_operations_foundation.sql)

## 11. Revision history

| Date | Change |
|---|---|
| 2026-07-24 | Accepted Campaign Operations as the bounded owner of orchestration over one exact Phase 4D materialization. |
