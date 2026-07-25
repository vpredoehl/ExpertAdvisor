# Volume X — Research Automation

Status: Authoritative for bounded Campaign Operations; broader automation reserved
Version: 1.0.0
Last revised: 2026-07-24

## 1. Purpose

Define controlled research orchestration while preserving explicit scope,
authorization, budgets, audit, lifecycle ownership, scheduler ownership, and
human authority. ADR-0010 through ADR-0017 and the
[Campaign Operations specification](../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
are the authoritative V1 refinement of this volume.

## 2. Scope

### 2.1 In scope

Campaign Operations over one exact Phase 4D materialization: operational
grants, member-unit budgets, reservations, durable requests, accepted Phase 5
handoff orchestration, campaign controls, cancellation coordination,
reconciliation observations, completion, logical archival, audit, and
read-only status.

### 2.2 Out of scope

Unbounded autonomy, self-modifying production code, implicit experiment
creation, live trading, or bypassing scheduler/lifecycle ownership.

### 2.3 Current implementation status

Campaign Operations Phase 1 implements foundational pure values,
canonical/validation contracts, budget arithmetic, and completion
classification helpers. Migration 045 and its repositories persist only the
immutable campaign, optional governance provenance, append-only authorization,
and audit foundations. Later accepted persistence and services remain
unimplemented unless identified by migration, code, and tests. Broader
autonomous research remains reserved; existing recommendation and continuation
capabilities MUST NOT be composed into it informally.

## 3. Responsibilities

### 3.1 Owned responsibilities

Campaign Operations owns orchestration state, explicit operational authority,
budget accounting, reservations, requests, bindings, campaign controls,
reconciliation observations, completion, and audit. It does not own proposal
or materialization truth, experiment lifecycle transitions, scheduler
execution, workers, or scientific interpretation.

### 3.2 Dependencies

Consumes the exact Phase 4D materialization and optional Phase 6D governance
provenance from Volume VIII. It invokes work only through accepted Phase
4C/5 and Volume VII lifecycle interfaces, then observes ordinary lifecycle
evidence. It has no direct Volume IX profitability or scheduler interface.

### 3.3 Prohibited responsibilities

MUST NOT bypass review, invent authorization, exceed budgets, mutate model
math, acquire scheduler claims/capacity, or directly control workers.

## 4. Architecture

### 4.1 Components

V1 components are campaign, authorization, budget, reservation/request,
dispatch/binding, control/cancellation, reconciliation, completion/audit, and
read-model services and repositories. The accepted Phase 5 transaction
primitive is the lifecycle adapter; there is no scheduler adapter or campaign
scheduler work class.

### 4.2 Control flow

Exact Phase 4D materialization → operational campaign → explicit grant →
budget → held reservation and accepted request → accepted Phase 5 lifecycle
handoff and immutable binding → ordinary pending experiment → scheduler claim
and execution → lifecycle evidence → reconciliation → operational completion.

### 4.3 Ownership boundaries

Campaign Operations orchestrates and accounts. Recommendation Governance owns
proposals/materializations. Experiment Lifecycle and accepted Phase 4C/5
services create and activate ordinary experiments. The scheduler claims and
executes them. Operators act only through explicit campaign, lifecycle, or
scheduler-global control capabilities.

## 5. Data model

### 5.1 Authoritative entities

Operational campaign, governance provenance, authorization event, budget
ledger entry, reservation and transition event, request, dispatch attempt and
outcome, downstream binding/control owner, campaign control, cancellation
request/settlement, reconciliation observation/resolution, completion, and
audit reference. Exact authoritative shapes are fixed by the accepted Campaign
Operations specification and ADRs.

### 5.2 Provenance and versions

Every decision records its exact authoritative inputs, canonical contract
version, budget and request evidence, actor/capability, and downstream IDs.

### 5.3 Invariants and legacy data

Budget consumption is monotonic and auditable. Missing authority or provenance
blocks action. Advisory status never becomes implicit authorization.

## 6. Transactions

### 6.1 Read paths

Inspection consumes exact authoritative evidence through read-only,
repeatable snapshots where cross-row consistency matters.

### 6.2 Write paths

Each accepted mutation uses the exact transaction and global lock order in
ADR-0010 through ADR-0017. Reservation and request acceptance are atomic;
accepted Phase 5 handoff, complete bindings, and budget commitment are atomic.
External lifecycle cancellation is a separately committed, explicitly
reconciled call.

### 6.3 Failure semantics

Uncertain budget or request state fails closed and requires reconciliation.

## 7. Concurrency

### 7.1 Conflict domain

Same authorization chain, budget account, campaign, reservation, request,
binding/control owner, cancellation target, or completion decision.

### 7.2 Locking and serialization

The global Campaign Operations order is authorization → budget → campaign →
reservation → request, with ascending IDs within a level. Workflows omit only
inapplicable earlier levels and MUST NOT acquire an earlier level after a
later one.

### 7.3 Winner, loser, and retry outcomes

At most one accepted request consumes a reservation; retries return the same
request only when authoritative idempotency identity matches.

## 8. CLI

### 8.1 Commands and validation

Any later Campaign Operations mutation command requires explicit campaign and
expected identity/version, actor capability, reason, validation, and
confirmation appropriate to its effect. This volume does not itself add CLI
commands.

### 8.2 Machine output

Events expose exact campaign, grant, budget, reservation, request, binding,
control, cancellation, reconciliation, completion, and audit identities.

### 8.3 Human output

Summaries state current grant, accounting, controls, request/binding state,
ordinary lifecycle/scheduler evidence, completion, and logical archival
without claiming scientific success.

## 9. Testing

### 9.1 Pure tests

Canonical identity, authorization, budgets, lifecycle derivation, completion
classification, and failure/retry state matrices.

### 9.2 Persistence and migration tests

Ledger monotonicity, request idempotency, audit immutability, and permissions.

### 9.3 Concurrency and integration tests

Budget races, duplicate requests, pause/stop races, and scheduler isolation.

### 9.4 Regression boundaries

Recommendation, lifecycle, continuation, and scheduler semantics remain owned
by their volumes.

## 10. Operational safety

### 10.1 Runtime isolation

Automation defaults disabled and cannot run from documentation or mere schema
presence.

### 10.2 Permissions and destructive operations

Least-privilege roles, explicit grants and budgets, campaign controls,
lifecycle-delegated cancellation, scheduler-global safety controls, and audit
are mandatory before deployment.

### 10.3 Observability and recovery

Durable campaign status, reservations, requests, outcomes, and reconciliation
must survive process failure.

## 11. Future extensions

### 11.1 Approved extension points

New origin kinds, versioned action/scope contracts, and scheduler-independent
orchestration adapters require accepted ADRs. Existing accepted V1 extension
points do not authorize broader automation.

### 11.2 Deferred capabilities

Automated selection of new recommendations/materializations, partial-member
dispatch, adaptive search/budgets, profitability policy, and autonomous
experimentation.

### 11.3 Required decisions

ADR-0010 through ADR-0017 close V1 Campaign Operations authority. Physical
retention/deletion, partial-member dispatch, new executable origins, adaptive
budgets, autonomous selection, running-worker stop authority, and new
scheduler work classes each require a later accepted owning-domain ADR.

## 12. References

- [Volume I §17.4](Volume_I_Foundation.md)
- [Volume VII](Volume_VII_Experiment_Lifecycle.md)
- [Volume VIII](Volume_VIII_Recommendation_Engine.md)
- [Volume XI](Volume_XI_Scheduler.md)
- [Accepted Campaign Operations specification](../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [ADR-0010 through ADR-0017](adr/README.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Reserved the research-automation architecture and safety gates. | — |
| 1.0.0 | 2026-07-24 | Accepted the bounded Campaign Operations refinement while retaining broader autonomous research as reserved. | ADR-0010–ADR-0017 |
