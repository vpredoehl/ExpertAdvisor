# ADR-0011: Campaign operational authorization chain

Status: Accepted
Date: 2026-07-24
Deciders: Project architecture
Affected volumes: Volume VIII §§2–5; Volume X §§3–7; Volume XII §§5–7
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Campaign approval, materialization, Phase 6D ratification, budget availability,
an accepted request, and scheduler capacity are distinct facts. None of the
existing accepted chains grants Campaign Operations permission to reserve,
accept, dispatch, or adopt work.

A durable, revocable, replay-safe authorization chain is required without
rewriting upstream governance or inventing implied authority.

## 2. Decision

An active exact operational authorization grant is the only permission for
Campaign Operations to reserve budget, accept a request, select it for
dispatch, or perform a downstream handoff.

- Authorization is an append-only, monotonically versioned chain per exact
  campaign, action contract, and scope contract.
- Persisted event kinds are exactly `granted`, `revoked`, and
  `expiry_observed`.
- `supersede` is a service transition that inserts one successor `granted`
  event naming the exact prior head and incrementing the chain once. There is
  no `superseded` event kind or second transition row.
- The initial fixed role is `campaign_operations_authorizer`; callers do not
  provide arbitrary role text.
- Each grant binds the exact campaign/materialization, named actions, complete
  scope, prerequisite policy and exact evidence, actor, reason, mandatory
  `not_before`, and optional `expires_at`.
- Supported V1 prerequisite policies are
  `phase4d_materialization_only_v1` and
  `phase4d_materialization_plus_exact_phase6d_ratification_v1`.
- The unique highest chain event is effective only when it is `granted`, its
  exact evidence still validates, and PostgreSQL transaction time is in
  `[not_before, expires_at)`. `expires_at=none` is canonical when absent.
- Chain versions and predecessor edges are unique and fork-resistant.
- Budget never substitutes for authorization. Ratification, approval,
  materialization, request acceptance, and scheduler capacity never substitute
  for authorization.
- Revocation or expiry blocks new reservation, acceptance, selection, and
  handoff. It does not erase committed bindings or control scheduler workers.
- A successor grant immediately becomes the effective head when valid, but an
  existing unbound request remains bound to its accepting grant and cannot
  borrow the successor. The owning settlement workflow must terminalize that
  request and release its hold after proving no binding.
- Adoption of exact pre-existing `pending/train` work additionally requires
  `adopt_existing_pending_and_control`. Ordinary dispatch authority is
  insufficient.
- Safety cancellation of a campaign before authorization, lifecycle-delegated
  safety cancellation of bound work, read-only inspection, budget
  administration, and reconciliation observation do not become dispatch
  authority.

Capability separation is mandatory. V1 adds no human-principal inequality
beyond the already accepted Phase 6C reviewer/Phase 6D ratifier separation.
The same authenticated person MAY hold multiple Campaign Operations
capabilities when institutional policy grants them explicitly; no
implementation may infer such grants.

## 3. Rationale and decision drivers

- Make permission explicit, durable, revocable, and auditable.
- Preserve immutable historical decisions and deterministic retries.
- Keep governance, funding, orchestration, and execution as separate gates.
- Avoid overlapping active chains and stale-grant races.

## 4. Consequences

### 4.1 Positive consequences

- Every action can cite one exact accepting grant.
- Revocation, expiry, and supersession have deterministic effects.
- Exact replay converges; stale heads and changed payloads conflict.
- Phase 6D remains usable as prerequisite evidence without gaining operational
  power.

### 4.2 Negative consequences and trade-offs

- Every write path must lock and revalidate the authorization domain.
- A successor grant cannot rescue an already accepted request.
- Time-dependent effectiveness requires PostgreSQL time on every protected
  mutation.

### 4.3 Risks and mitigations

- Forked chain: unique version/predecessor constraints plus a transaction
  advisory lock.
- Stale authorization through handoff: hold the authorization lock through
  downstream commit.
- Clock disagreement: use PostgreSQL transaction time, UTC normalization, and
  an exclusive expiry bound.
- Role confusion: fixed contract role and ADR-0017 capability grants.

## 5. Compatibility and migration

Migration 045 and Phase 1 already implement the immutable campaign
authorization chain. This ADR accepts that implemented contract; it does not
rewrite rows or change Phase 4–6 authority.

Future action kinds require supported contract versions and, when they expand
authority, a later accepted ADR. Existing events remain immutable.

## 6. Implementation implications

- Construct and validate canonical identities before opening a write
  transaction where practical.
- Acquire the authorization semantic lock first when a workflow requires
  active authority, load the unique head, and revalidate after locking.
- Compare complete canonical bytes after hash lookup.
- Exact retry returns the existing event; stale predecessor/version or changed
  payload returns a deterministic conflict.
- Authorization services may append authorization and audit evidence only;
  they receive no budget, request, experiment, scheduler, or worker mutation
  privilege.

## 7. Verification and operational evidence

- Golden canonical/hash and UTC/expiry tests.
- Grant/revoke/expiry/supersede transition matrices.
- Concurrent identical and competing-successor tests.
- Revocation/supersession versus reservation and handoff interleavings.
- Exact replay, changed payload, stale head, collision, malformed-chain, and
  restart-effectiveness tests.
- Negative tests proving upstream approvals, ratification, budget, and
  scheduler capacity cannot authorize an action.

## 8. Alternatives considered

### 8.1 Treat ratification or materialization as authorization

Rejected because both accepted contracts explicitly deny operational power.

### 8.2 Mutable current-authorization row

Rejected because it would erase decision history and make replay,
supersession, and audit ambiguous.

### 8.3 Two-event supersession

Rejected because a `superseded` row plus successor grant creates competing
version and lifecycle interpretations. One successor `granted` event is the
single authority.

## 9. Relationships to other ADRs

- ADR-0003, ADR-0005, and ADR-0009 define non-substitutable upstream evidence.
- ADR-0010 owns campaign scope.
- ADR-0012 and ADR-0013 require this grant for reservation and handoff.
- ADR-0015 defines safety cancellation and settlement.
- ADR-0017 assigns the separate capability roles.

## 10. References

- [Accepted Campaign Operations specification §§10, 12–13, 18–20](../../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [Migration 045](../../../Database/migrations/045_campaign_operations_foundation.sql)
- [ADR-0009](ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md)

## 11. Revision history

| Date | Change |
|---|---|
| 2026-07-24 | Accepted the explicit append-only operational authorization chain and its non-substitution boundaries. |
