# ADR-0015: Campaign cancellation, reconciliation, and restart recovery

Status: Accepted
Date: 2026-07-24
Deciders: Project architecture
Affected volumes: Volume VII §§6–10; Volume X §§6–10; Volume XI §§6–10;
Volume XII §§6–10
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Cancellation can race with reservation acquisition, dispatch, handoff,
scheduler claim, running work, and completion. Crashes can also leave a lease,
reservation, or commit outcome uncertain. A single mutable cancellation flag
or a reconciler with repair privileges would blur ownership and permit unsafe
refunds or lifecycle writes.

## 2. Decision

Cancellation intent and settlement are separate immutable facts.

### 2.1 Cancellation

- A cancellation request binds the exact campaign and optional request,
  binding, or control-owner target, expected target version, actor,
  capability, reason, scope, and canonical identity.
- A settlement separately records the exact disposition and authoritative
  Campaign Operations or lifecycle evidence.
- Before authorization or reservation, cancellation blocks future Campaign
  Operations work and requires no budget settlement.
- For an unbound held request, one transaction takes budget → campaign →
  reservation → request (authorization first only when the exact cause
  requires it), proves no binding/downstream commit, appends the cancellation
  request, changes request to `cancelled`, releases the reservation through its
  event, appends settlement/audit, and commits all or none.
- After binding, committed units are never released. Campaign Operations first
  records cancellation intent, commits, releases all Campaign Operations
  locks, then invokes only the accepted lifecycle cancellation service using
  the permanent control owner. A later Campaign Operations transaction records
  the lifecycle-backed settlement.
- Campaign Operations never directly updates an experiment, scheduler claim,
  PID, process group, or worker.
- If ordinary lifecycle authority rejects cancellation of claimed or running
  work, settlement records `running_cancellation_not_supported` (or the exact
  accepted terminal disposition) and Campaign Operations waits for ordinary
  terminal lifecycle evidence. No new stop or pause authority is inferred.
- Terminal lifecycle truth wins over cancellation; history is not rewritten.

### 2.2 Reconciliation and recovery

- Reconciliation detects and records; owning services perform every repair
  transition.
- The reconciler may perform bounded, stable-ID, read-only comparisons and
  append immutable observations. It has no budget, reservation, request,
  binding, cancellation-settlement, lifecycle, completion, or resolution-
  forging privilege.
- Every observation has a stable reason, exact point-in-time evidence,
  expected version, recommended owning service/action, and canonical identity.
- Every selected observation belongs to one durable cursor identity. Cursor
  and membership commit atomically; replay loads by cursor identity and never
  infers members from a run key, ID range, or current state.
- An observation is resolved only by an immutable resolution referencing exact
  evidence produced by the owning service or separately accepted repair
  authority.
- Unknown commit results are resolved by canonical lookup before retry.
  Proven absence permits whole-operation retry; ambiguity remains held and
  `reconciliation_required`.
- Expired leases or reservations are recoverable only after database time,
  locks, and exact no-binding/no-downstream-commit proof. PID or process absence
  is never proof.
- Restart scans are bounded by state and stable ID. They may re-drive only
  named owning-service transitions with identical canonical inputs.

## 3. Rationale and decision drivers

- Preserve owner-specific transitions and least privilege.
- Make cancellation causality and outcome independently auditable.
- Fail closed during the handoff crash window.
- Prevent premature budget release or invented experiment control.

## 4. Consequences

### 4.1 Positive consequences

- Cancellation races have deterministic winners and durable outcomes.
- Running-work limitations are explicit rather than hidden behind signals.
- Reconciliation cannot silently rewrite truth.
- Restart recovery is idempotent and bounded.

### 4.2 Negative consequences and trade-offs

- Bound cancellation requires at least two transactions around the lifecycle
  call.
- A campaign may remain cancelling until running work naturally terminates.
- Ambiguous evidence can retain budget holds and require operator attention.

### 4.3 Risks and mitigations

- Deadlock: all held-reservation settlement paths take budget before campaign,
  reservation, and request; IDs are ascending within a level.
- Lost lifecycle response: reload lifecycle request/experiment evidence.
- False repair: resolution must reference an accepted owning-service event.
- Unbounded restart: fixed batch size and stable cursor.

## 5. Compatibility and migration

This decision does not change the existing lifecycle cancellation contract or
global experiment controls. Future Campaign Operations cancellation,
observation, and resolution tables are additive.

No running-process termination feature is authorized. A future change to
running-work lifecycle authority requires its own accepted lifecycle/scheduler
ADR and does not alter this detection/delegation boundary silently.

## 6. Implementation implications

- Request and settlement have different canonical identities and natural keys.
- Bound lifecycle calls occur only after committing cancellation intent and
  releasing Campaign Operations locks.
- Exact retry returns the same request, transition, settlement, observation, or
  resolution; changed payload conflicts.
- Serialization/deadlock retry recreates all transaction-derived state for the
  whole canonical operation and publishes identifiers only after commit.
- Concurrent unbound cancellation replay rechecks settlement after taking the
  cancellation locks and converges on the exact durable settlement.
- Reconciliation observations never become authorization or completion
  evidence until an owning resolution edge exists.

## 7. Verification and operational evidence

- Cancellation tests at every authority, reservation, lease, handoff, claim,
  running, terminal, and completion boundary.
- Budget-first lock-order assertions and deadlock-sensitive races.
- Separate request/settlement replay and conflict tests.
- Crash-window matrix for before/after intent, handoff, binding, lifecycle
  call, settlement, and commit response.
- Bounded restart, stale lease, expiry, unknown commit, observation/resolution,
  privilege, and no-creative-repair tests.
- Negative tests proving no direct experiment write or process signal.

## 8. Alternatives considered

### 8.1 One mutable cancellation flag

Rejected because intent, application, refusal, and terminal evidence are
different facts.

### 8.2 Let reconciliation repair any inconsistency

Rejected because it would combine observation with every owning subsystem's
mutation authority.

### 8.3 Signal running workers directly

Rejected because Campaign Operations does not own scheduler processes and
process identity is not lifecycle authority.

## 9. Relationships to other ADRs

- ADR-0012 defines reservation settlement and budget locking.
- ADR-0013 defines handoff, bindings, and permanent control ownership.
- ADR-0014 defines pause and completion blockers.
- ADR-0016 owns scheduler claim/cancellation precedence.
- ADR-0017 enforces reconciler and cancellation least privilege.

## 10. References

- [Accepted Campaign Operations specification §§18–23](../../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [Volume VII](../Volume_VII_Experiment_Lifecycle.md)
- [Global experiment controls](../../GlobalExperimentControls.rst)

## 11. Revision history

| Date | Change |
|---|---|
| 2026-07-24 | Accepted separate cancellation/settlement, detection-only reconciliation, and bounded restart recovery. |
