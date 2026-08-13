# ADR-0013: Durable operational request and atomic lifecycle handoff

Status: Accepted
Date: 2026-07-24
Deciders: Project architecture
Affected volumes: Volume VII §§3–7; Volume VIII §§3–7; Volume X §§3–7;
Volume XI §§3–7; Volume XII §§5–7
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Campaign Operations must persist intent before attempting downstream work,
survive crashes and lost responses, and avoid duplicate Phase 5 mutations.
Request acceptance, dispatch selection, lifecycle handoff, scheduler claim,
and worker execution are distinct authority boundaries.

## 2. Decision

V1 uses a durable request/outbox followed by a separate atomic lifecycle
handoff.

### 2.1 Request identity and acceptance

- One system-derived logical-operation key binds the campaign canonical,
  `dispatch_full_materialization`, action version, and exact complete
  materialization canonical.
- Actor, reason, accepting grant, budget entry, timestamps, and dispatcher do
  not change the logical-operation key.
- The database admits one V1 request per campaign/action/version and one
  reservation per request.
- Request acceptance is one transaction that revalidates active
  authorization, active budget, campaign controls, exact full materialization,
  and available units, then inserts a `held` reservation, acquisition event,
  `ready` request, and audit references. All commit or none commit.
- Acceptance is durable intent only. It is not dispatch, experiment creation,
  scheduler admission, scheduler claim, worker launch, or execution.

### 2.2 Dispatch and handoff

- Candidate selection is advisory. Dispatch authority comes from the locked
  request state/version and lease plus the revalidated accepting grant.
- Dispatch acquisition and its outcome are separate immutable audit facts;
  attempt rows are never authority.
- The handoff transaction holds authorization → budget → campaign →
  reservation → request locks through completion.
- Inside that caller-owned PostgreSQL transaction, Campaign Operations invokes
  the existing transaction-bound Phase 5 launch workflow. Phase 5/Experiment
  Lifecycle remains the owner of proposal execution, activation, experiment
  creation/reuse, and transition to ordinary `pending/train`.
- The same transaction inserts the complete per-member immutable binding set,
  inserts permanent V1 downstream control ownership, changes request to
  `bound`, changes reservation to `committed`, and records outcome/audit
  evidence.
- A deferred completeness rule requires exactly the authoritative Phase 4D
  members before binding can commit.
- Exact pre-existing `pending/train` evidence may be adopted only with the
  additional ADR-0011 adoption/control capability, complete deterministic
  causality, and no existing control owner. Adoption consumes the same budget.
- V1 control ownership is permanent. There is no transfer, release, or
  implicit ownership through a binding.
- Progressed unbound evidence, partial binding, causality mismatch, or an
  unknown commit outcome fails closed into reconciliation; Campaign Operations
  never fabricates a binding or experiment.

The scheduler does not participate in this transaction. After commit, bound
experiments are ordinary scheduler resources; ADR-0016 governs their claims
and worker execution.

## 3. Rationale and decision drivers

- Persist intent before side effects.
- Reuse accepted Phase 4C/5 lifecycle authority rather than duplicate SQL.
- Make handoff, causality, and budget consumption indivisible.
- Permit deterministic recovery from crashes, retries, and direct Phase 5
  races.

## 4. Consequences

### 4.1 Positive consequences

- One logical operation cannot create duplicate requests or experiments.
- Request acceptance can be implemented without scheduler changes.
- Lost responses can reload canonical request or binding evidence.
- Bindings remain authoritative after experiments progress.

### 4.2 Negative consequences and trade-offs

- The handoff transaction spans multiple established lock domains.
- V1 requires same-database transaction composition with Phase 5.
- Direct Phase 5 overlap may require explicit adoption or fail closed.

### 4.3 Risks and mitigations

- Revocation race: hold the authorization lock through handoff commit.
- Partial binding: deferred cardinality/provenance validation.
- Duplicate dispatch: guarded request version/lease and exact canonical replay.
- Commit uncertainty: canonical result lookup before retry; uncertainty never
  authorizes a second handoff.

## 5. Compatibility and migration

The accepted Phase 4C/5 workflows remain unchanged and callable directly.
Campaign Operations adds only its request, reservation, attempt, binding,
control-owner, and audit evidence through additive migrations.

No scheduler table, work class, priority, capacity, experiment identity, or
Phase 5 materialization contract changes. Production dispatch remains disabled
until ADR-0016's implementation gate is satisfied.

## 6. Implementation implications

- Phase 2 may implement request acceptance only; it MUST NOT acquire dispatch
  leases or invoke Phase 5.
- Later dispatch uses bounded stable-ID selection and short acquisition
  transactions.
- The service owns the caller transaction; repositories do not commit
  independently.
- Serialization/deadlock retry restarts the whole operation with identical
  canonical input. Unique conflicts reload and compare complete canonical
  bytes. Other SQL errors are not recast as idempotency.
- No database transaction stays open across process launch or worker
  execution.

## 7. Verification and operational evidence

- Golden logical-operation, reservation, request, attempt, binding, and
  control-owner identities.
- Atomic acceptance rollback, exact replay, changed payload, collision, and
  lost-response tests.
- Independent-connection duplicate acceptance, dispatch, revocation, pause,
  cancellation, direct Phase 5, and adoption races.
- Deferred complete-binding and corruption tests.
- Phase 4C/5 regression tests proving accepted workflows and identities are
  unchanged.
- Negative tests proving acceptance and attempt rows do not grant scheduler or
  worker authority.

## 8. Alternatives considered

### 8.1 Invoke Phase 5 before persisting intent

Rejected because a crash can create unbound work with no durable request.

### 8.2 Reimplement Phase 5 with direct Campaign Operations SQL

Rejected because it bypasses authoritative lifecycle workflows and duplicates
accepted invariants.

### 8.3 Treat the request as a scheduler job

Rejected because the scheduler owns ordinary experiment execution, not
Campaign Operations orchestration.

## 9. Relationships to other ADRs

- ADR-0005 and Phase 5 own downstream experiment creation/activation.
- ADR-0010 defines campaign and materialization scope.
- ADR-0011 and ADR-0012 supply authorization and reservation.
- ADR-0014 owns campaign controls and completion.
- ADR-0015 owns cancellation and recovery.
- ADR-0016 owns execution claims; ADR-0017 owns the narrow Phase 5 adapter.

## 10. References

- [Accepted Campaign Operations specification §§12, 16–20](../../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [Phase 5 launch](../../Phase5ExperimentRecommendationCampaignLaunch.rst)
- [Phase 4C execution](../../Phase4CExperimentRecommendationConversionExecution.rst)

## 11. Revision history

| Date | Change |
|---|---|
| 2026-07-24 | Accepted one durable full-materialization request/outbox and atomic Phase 5 lifecycle handoff. |
