# ADR-0014: Campaign lifecycle, controls, completion, and archival

Status: Accepted
Date: 2026-07-24
Deciders: Project architecture
Affected volumes: Volume VII §§3–7; Volume X §§3–11; Volume XII §§5–7
Supersedes: None
Superseded by: None

## 1. Context and problem statement

Campaign Operations needs a durable administrative lifecycle without competing
with experiment lifecycle state, scheduler attempts, or scientific outcome.
Pause/resume, operational completion, and archival were previously implied by
events and read models but lacked accepted authority.

## 2. Decision

Campaign administrative state is a deterministic derived summary over the
immutable campaign row, authoritative append-only events, guarded
reservation/request state, exact bindings, and current lifecycle evidence.
No mutable campaign status row or projection is mutation authority.

### 2.1 Lifecycle

The valid V1 administrative states are:

`awaiting_operational_authorization`, `authorized`, `budgeted`, `reserving`,
`ready`, `dispatching`, `active`, `paused`, `cancellation_requested`,
`cancelling`, `terminal_completed`, `terminal_cancelled`, `terminal_failed`,
`inconsistent`, and `reconciliation_required`.

Precedence is:

```text
inconsistent
> reconciliation_required
> terminal states
> cancelling
> cancellation_requested
> paused
> dispatching
> active
> ready
> reserving
> budgeted
> authorized
> awaiting_operational_authorization
```

Creation commits directly from no row to a row deriving
`awaiting_operational_authorization`. `drafted` may exist only in memory before
commit.

Campaign, member operational, reservation, request, downstream experiment,
scheduler attempt, and scientific result are separate dimensions and MUST NOT
be collapsed.

### 2.2 Pause and resume

- Pause and resume are append-only, versioned, alternating campaign-control
  events with exact expected versions, actors, capabilities, reasons, and
  canonical identities.
- Pause blocks new Campaign Operations reservation, dispatch selection, and
  handoff actions.
- Pause does not change existing experiment lifecycle state, scheduler claims,
  worker process state, global experiment controls, or budget history.
- Resume does not restore a remembered prior state. It removes only the
  Campaign Operations pause gate and derives the current eligible state from
  all other authority.
- Resume is invalid after terminal completion/cancellation/failure and while
  unresolved inconsistency makes derivation unsafe.

### 2.3 Completion

Operational completion is one immutable append-only decision that all
coordination obligations are settled at one exact evidence point. It is not
scientific success.

Completion requires:

- the campaign is not merely paused as a substitute for closure and no active
  grant retains an obligation able to create or dispatch work;
- no held reservation, active dispatch lease, unsettled request, incomplete
  binding, unresolved cancellation, or blocking unresolved reconciliation
  observation;
- exact budget equations;
- complete control-owner attribution where required; and
- terminal authoritative lifecycle evidence for every bound member.

The classification precedence is fixed:

1. contradiction or unsettled evidence blocks completion;
2. an unbound `permanently_failed` request with no binding →
   `terminal_failed / operational_request_failed`;
3. mixed failed and other outcomes → `terminal_failed /
   mixed_terminal_outcomes`;
4. all bound work failed → `terminal_failed / downstream_failure`;
5. completed plus cancelled/never-dispatched scope →
   `terminal_completed / terminal_partial_completion`;
6. all scope cancelled or never dispatched →
   `terminal_cancelled / all_scope_cancelled`;
7. all exact members bound and completed →
   `terminal_completed / all_downstream_completed`.

There is no force-complete, override, reopen, supersede, or delete operation in
V1. Later lifecycle retry/requeue does not rewrite completion; read models show
both the recorded decision and `post_completion_lifecycle_changed`.

### 2.4 Archival

A campaign is logically archived for default operational listings when its
immutable completion event exists. Archival is a read-model classification,
not a new authority, event, lifecycle transition, data move, deletion, or
privilege. Archived campaigns remain queryable by exact identity and retain
all history needed to reconstruct authorization, accounting, causality,
completion, and audit.

Physical retention, partition retirement, export, or deletion requires a later
accepted retention ADR. Until then, runtime deletion is prohibited. This
physical-retention decision is not an implementation prerequisite for V1
campaign operations.

## 3. Rationale and decision drivers

- Avoid a mutable campaign status competing with event history.
- Keep campaign controls separate from scheduler/global experiment controls.
- Make completion reproducible and independent of scientific interpretation.
- Provide useful archival behavior without inventing deletion policy.

## 4. Consequences

### 4.1 Positive consequences

- Restart can reconstruct administrative state from durable truth.
- Pause/resume cannot silently alter running work.
- Completion is idempotent, auditable, and classification-stable.
- Archived history remains available for audit and replay.

### 4.2 Negative consequences and trade-offs

- Status queries must derive across multiple authoritative dimensions.
- A later lifecycle change can make a completed snapshot historically stale,
  requiring explicit display.
- Append-only storage grows until a retention ADR is accepted.

### 4.3 Risks and mitigations

- Projection drift: projections are rebuildable and never authorize writes.
- Premature completion: one repeatable-read transaction takes all global locks
  and validates the exhaustive prerequisite matrix.
- Operator confusion: outputs distinguish operational completion, scientific
  outcome, current lifecycle, and logical archival.

## 5. Compatibility and migration

Migration 045's immutable campaign row already establishes the initial state.
Future control/completion persistence is additive. No existing experiment,
global-control, recommendation, or Phase 5 state is relabeled.

Logical archival requires no data migration. Physical retention remains
disabled.

## 6. Implementation implications

- Control events are append-only and use expected control versions.
- Completion runs in one `REPEATABLE READ` transaction with authorization →
  budget → campaign/completion → reservations → requests lock order and
  ascending IDs within each level.
- Exact completion replay returns the existing canonical event; different
  evidence/classification conflicts.
- Read-only status may use a projection only when it exposes source
  high-water marks and can be rebuilt from authority.

## 7. Verification and operational evidence

- Exhaustive campaign-state and resume-derivation matrices.
- Pause/resume versus reservation, dispatch, cancellation, and completion
  races.
- Completion classification, blocker, replay, concurrent-winner, and
  post-completion lifecycle-change tests.
- Restart derivation without mutable campaign status.
- Read-model tests proving archived rows remain exactly queryable and no
  delete/move occurs.
- Regression tests separating global experiment controls and scientific
  outcome policy.

## 8. Alternatives considered

### 8.1 Mutable campaign status column

Rejected because it would compete with immutable events and become vulnerable
to partial updates and projection drift.

### 8.2 Pause scheduler or workers when a campaign pauses

Rejected because Campaign Operations does not own scheduler or process control.

### 8.3 Delete or move rows on completion

Rejected because no accepted retention period or safe referential-retirement
contract exists.

## 9. Relationships to other ADRs

- ADR-0010 establishes the creation fact and state ownership.
- ADR-0011 through ADR-0013 supply authorization, accounting, requests, and
  bindings used in derivation.
- ADR-0015 supplies cancellation and reconciliation prerequisites.
- ADR-0016 preserves scheduler/global-control separation.
- ADR-0017 governs projections, audit, and append-only privileges.

## 10. References

- [Accepted Campaign Operations specification §§13, 21–23](../../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [Global experiment controls](../../GlobalExperimentControls.rst)
- [Phase 5 outcome assessment](../../Phase5ExperimentRecommendationCampaignOutcomeAssessment.rst)

## 11. Revision history

| Date | Change |
|---|---|
| 2026-07-24 | Accepted derived lifecycle, campaign-only pause/resume, immutable completion, and non-destructive logical archival. |
