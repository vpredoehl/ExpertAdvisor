# ADR-0016: Scheduler atomic claim hardening for production dispatch

Status: Accepted
Date: 2026-07-24
Deciders: Project architecture
Affected volumes: Volume VII §§6–10; Volume X §§6–10; Volume XI §§3–10;
Volume XII §§5–10
Supersedes: None
Superseded by: None

## 1. Context and problem statement

ADR-0004 requires an atomic durable scheduler claim before process launch.
Repository review found that current pending selection is an ordinary read,
the running update lacks a pending/expected-phase compare-and-set predicate,
and no durable scheduler-attempt identity closes multiple-scheduler or
cancellation races.

Campaign Operations does not create a new scheduler work class, but production
handoff will produce ordinary pending experiments. Production dispatch cannot
safely be enabled while their claim boundary remains ambiguous.

## 2. Decision

The scheduler remains the sole owner of experiment execution and must harden
ordinary claims before Campaign Operations production dispatch is enabled.

- Polling may remain optimistic, but claim occurs in one short PostgreSQL
  transaction that locks the exact experiment/operation conflict domain.
- Under lock, the scheduler rechecks exact experiment identity, expected
  phase, `pending` eligibility, global administrative control, cancellation
  precedence, and capacity authority.
- The transaction creates one durable, immutable scheduler-attempt identity
  and atomically performs the conditional lifecycle claim. Exactly one
  scheduler wins; losers observe changed state and skip deterministically.
- Attempt identity binds the exact experiment, phase/operation, scheduler
  instance, attempt ordinal, expected lifecycle version, and claim time
  metadata needed for launch/recovery.
- Capacity reservation and claim must commit consistently under the
  scheduler-owned contract. Process launch occurs only after claim commit and
  outside the transaction.
- Launch failure, running, completion, failure, orphaning, and recovery remain
  distinct durable scheduler-attempt outcomes.
- Cancellation versus claim is decided by the same database-visible lifecycle
  conflict domain. Campaign Operations only observes the result.
- Campaign Operations never receives scheduler claim, attempt, capacity,
  priority, process, or completion-transition privileges.
- The scheduler never reads Campaign Operations authorization, budget,
  request, or campaign-control tables.
- Campaign-bound experiments use the existing ordinary work classes and
  eligibility rules. No campaign priority, slot reservation, or new polling
  class is authorized.
- Existing global experiment pause/resume/cancel controls remain
  scheduler-owned administrative safety controls and are not Campaign
  Operations control events.

ADR acceptance alone does not enable production dispatch. Enablement requires
implementation, multi-connection verification, independent review, and an
explicit default-off configuration/privilege decision.

Implementation note (2026-07-27): migration 051 and scheduler version 0.3.0
implement this decision with a fenced lease, invocation history, globally
capacity-accounted attempts, lifecycle attempt fences, and gated canonical
launch. Independent verification and any Campaign Operations enablement remain
separate.

Correction note (2026-07-29): migration 052 and scheduler version 0.4.0 apply
the same attempt authority to continuation mutation, signals, parent reaping,
reconciliation, capacity release, and in-process checkpoint analysis. They
also establish one global lock order and a technical mixed-version cutover
barrier. ADR-0018 records the coordinated correction.

## 3. Rationale and decision drivers

- Satisfy the already accepted ADR-0004 claim invariant.
- Prevent duplicate workers and cancellation/claim lost updates.
- Keep scheduler execution independent from campaign orchestration.
- Avoid a new campaign-specific scheduler path.

## 4. Consequences

### 4.1 Positive consequences

- Multiple schedulers converge on one claim.
- Attempt ownership and launch/recovery become durable and auditable.
- Campaign Operations can hand off ordinary experiments without scheduler
  coupling.
- Global controls and campaign controls retain separate authority.

### 4.2 Negative consequences and trade-offs

- Scheduler schema and claim code require a separately tested hardening
  increment.
- Production Campaign Operations dispatch remains disabled until verification.
- Capacity and claim transactions need careful ordering.

### 4.3 Risks and mitigations

- Claim/cancellation race: one exact lifecycle conflict domain and conditional
  transition.
- Claim committed but launch failed: distinct attempt outcome and scheduler
  recovery.
- Stale process metadata: bind durable attempt and kernel process-start
  identity; never trust PID alone.
- Campaign coupling: deny scheduler access to Campaign Operations tables.

## 5. Compatibility and migration

Hardening is scheduler-owned and must preserve all existing experiment work
classes, lifecycle meanings, continuation behavior, capacity policy, and
global controls. Any scheduler schema is additive and independent of Campaign
Operations migrations.

Campaign Operations Phases A–D require no scheduler change. Its handoff phase
may be implemented default-off or isolated-test-only before hardening, but
production dispatch is prohibited.

## 6. Implementation implications

- The scheduler service owns claim/attempt transactions and their repository
  APIs.
- Claim uses an expected state/version predicate, not an unconditional running
  update.
- Process launch, supervision, and worker execution remain outside database
  transactions.
- Rollback of Campaign Operations production enablement disables dispatch
  first; it never deletes bindings or rewrites lifecycle truth.

## 7. Verification and operational evidence

- Independent-connection multiple-scheduler claim races.
- Claim versus global pause/cancel and lifecycle cancellation races.
- Capacity exhaustion, launch failure, stale claim, orphan recovery,
  completion race, shutdown, and restart tests.
- Exact attempt replay/corruption and process-start identity tests.
- Existing training, inference, analysis, checkpoint, continuation, and global
  control regressions.
- Independent verification before the production enablement gate changes.

## 8. Alternatives considered

### 8.1 Rely on a singleton scheduler convention

Rejected because it does not close cancellation versus unconditional running
updates and is not durable correctness.

### 8.2 Let Campaign Operations claim or launch bound experiments

Rejected because it would overlap scheduler execution authority.

### 8.3 Add a campaign-specific scheduler work class

Rejected because accepted handoff creates ordinary experiments and needs no
new capacity policy.

## 9. Relationships to other ADRs

- ADR-0004 remains the governing scheduler ownership boundary; this ADR
  specifies the missing claim hardening required by it.
- ADR-0013 ends Campaign Operations authority at lifecycle handoff.
- ADR-0014 keeps campaign pause distinct from scheduler/global controls.
- ADR-0015 observes cancellation/claim outcomes without process control.
- ADR-0017 enforces role isolation.

## 10. References

- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
- [Volume XI](../Volume_XI_Scheduler.md)
- [Accepted Campaign Operations specification §17 and §31.8](../../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [Global experiment controls](../../GlobalExperimentControls.rst)

## 11. Revision history

| Date | Change |
|---|---|
| 2026-07-24 | Accepted scheduler-owned atomic claim/attempt hardening and the production-dispatch gate. |
| 2026-07-27 | Recorded migration 051 and scheduler 0.3.0 implementation; independent verification remains separate. |
| 2026-07-29 | Recorded generation-52 exact-attempt correction and delegated its detailed recovery/cutover rules to ADR-0018. |
