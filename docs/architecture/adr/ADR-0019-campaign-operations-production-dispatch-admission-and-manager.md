# ADR-0019: Campaign Operations production dispatch admission and Manager

Status: Accepted
Date: 2026-07-31
Deciders: Project architecture
Affected volumes: Volume X §§2–11; Volume XI §§2–11; Volume XII §§2–11
Supersedes: None
Superseded by: None

## 1. Context and problem statement

ADR-0013 accepted durable request acquisition and an atomic Phase 5 lifecycle
handoff, but migration 048 intentionally restricted that path to isolated test
databases and kept `production_dispatch_enabled=false`. ADR-0016 and ADR-0018
subsequently established scheduler generation-52 atomic-claim and exact-attempt
authority. The accepted Campaign Operations roadmap still described Phase H
persistence, services, and CLI as scheduler-owned only and described production
enablement as a configuration/privilege action.

Production dispatch now requires an explicit durable authority boundary. Schema
presence, process liveness, role membership, a request Boolean, or scheduler
cutover state cannot serve as that authority. A Campaign Operations-owned
consumer is also needed to invoke the already accepted request-to-Phase-E
workflow without acquiring scheduler, lifecycle, worker, or scientific policy
ownership.

## 2. Decision

Campaign Operations owns durable global production-dispatch admission and the
Campaign Manager. Production deployment is default-off: in the absence of an
exact effective immutable enable event, production acquisition and handoff are
prohibited. Enable and disable events form one immutable, append-only,
fork-resistant, alternating chain. Disable-first is the rollback rule; history
is never deleted or rewritten.

The exact accepted contract is
[Campaign Operations Phase H — Production Dispatch Admission and Manager](../CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md).
Its identities, operation keys, replay rules, lock order, scheduler evidence,
privileges, readiness, rollout, and H1–H4 boundaries are normative.

### 2.1 Production authority hierarchy

Global authority is the current enable-chain head only when it is an enable
event and its complete scheduler protocol evidence exactly matches the current
pinned Phase H scheduler evidence. V1 accepts exactly generation 52 and exactly
`cutover_state=complete`; neither `generation >= 52`, “52 or newer,” nor cutover
state alone is sufficient.

Per-request authority is one immutable first production admission together
with a complete Attempt V2 and matching production audit. The existing
`production_dispatch_enabled` column becomes only a one-way serialization
witness. It never authorizes a read or mutation by itself.

### 2.2 Campaign Manager and Phase E reuse

The Campaign Manager belongs to Campaign Operations. It may perform exact
single-request production dispatch and bounded, sequential run-once processing.
All production and isolated-test paths use one internal Phase E engine.
Duplicated acquisition or handoff implementations are prohibited. Test hooks
exist only in the isolated-test adapter and are unreachable from production
constructors, CLI options, configuration, and environment variables.

The Manager selects candidates optimistically and read-only, then uses the
request version/lease compare-and-set under the accepted lock order. It does
not accept requests automatically and does not poll, claim, reserve capacity,
launch, signal, supervise, or control scheduler workers.

### 2.3 Scheduler boundary

Phase H consumes scheduler evidence through a narrow, pinned,
security-definer read interface. A mutating production path obtains a shared
lock on the scheduler protocol singleton through that interface before taking
the Campaign Operations production gate and holds both through commit.
Campaign Operations receives no raw scheduler-table access. The scheduler
does not poll Campaign Operations and receives no Campaign Operations
privilege.

Scheduler liveness is diagnostic only. Generation 53 or any protocol-canonical
mismatch immediately makes the current enable head ineffective for new
acquisition and handoff. Generation 53 requires a disable event, a new
enablement contract version, new independent verification, and a new enable
event; generation-52 approval is never inherited.

### 2.4 Scope limits

Campaign Operations does not own worker launch or control, scheduler claims,
attempts, leases, capacity or processes, scientific policy or interpretation,
or lifecycle mutation outside the existing Phase E handoff. Disable does not
cancel experiments, reverse reservations, signal workers, or mutate scheduler
state. Exact replay of an already committed binding remains valid after
disablement.

Physical archival, retention/deletion, autonomous research, adaptive budgets,
partial dispatch, scheduler work classes or capacity policy, running-worker
cancellation, automatic request acceptance or completion, and scientific
interpretation remain excluded.

Continuous Campaign Manager operation is excluded until H4 is separately
accepted by an operational ADR or an explicit later architecture acceptance.
Database-level multi-manager correctness does not authorize daemon operation,
polling, autostart, supervision, or deployment ownership.

## 3. Rationale and decision drivers

- Preserve PostgreSQL as committed authority and canonical text as semantic
  identity while using hashes only for lookup/integrity.
- Make production authorization explicit, immutable, replayable, and
  independently auditable.
- Reuse the accepted Phase E workflow without creating a second implementation
  or crossing scheduler/lifecycle ownership.
- Fail closed across disable/re-enable, scheduler upgrades, lease recovery, and
  uncertain commits.
- Permit canary-first, bounded rollout and disable-first rollback.

## 4. Consequences

### 4.1 Positive consequences

- No enable event and no production login membership leave H1–H3 inert.
- Exact scheduler/build approval cannot silently flow to a later protocol.
- One request operation is recoverable by a different authorized Manager
  without changing its logical attempt identity.
- Global disable serializes with acquisition/handoff without destroying
  evidence or claiming process-control authority.
- Completion V1 remains compatible because its request evidence nests each
  complete stored attempt canonical byte-for-byte.

### 4.2 Negative consequences and trade-offs

- Migration 055 requires owner-DML-safe immediate and deferred constraints,
  additional immutable evidence, and a detailed ACL surface.
- A scheduler upgrade can strand an acquired, unbound old-event lease until
  normal expiry and Phase F recovery.
- Operators must manage explicit enable/disable evidence and dedicated role
  membership; schema deployment alone cannot enable production.
- Initial Manager throughput is intentionally bounded and sequential.

### 4.3 Risks and mitigations

- Boolean becomes competing authority: guarded false-to-true plus deferred
  bidirectional admission consistency and negative authorization tests.
- Hash collision masks changed replay: full canonical comparison at every
  lookup and nested-evidence level.
- Unknown commit duplicates action: fresh-connection exact-operation lookup
  before bounded whole-transaction retry.
- Test capability reaches production: literal database prefix, acknowledgement,
  role graph, V1 shape, false Boolean, and absence-of-enable checks together.
- Operational daemon semantics are inferred: H4 is explicitly excluded.

## 5. Compatibility and migration

Migration 055 is additive design authority. It must not backfill existing
requests, reinterpret historical Attempt V1 or Completion V1 identities, grant
membership to a LOGIN role, or modify scheduler/lifecycle production state.
Attempt V1 remains the isolated-test shape; Attempt V2 adds the complete
production chain. Existing V1 bindings, outcomes, completions, and audits keep
their original canonical bytes and meaning.

Deployment remains disabled through migration and executable installation.
Dedicated login creation and role grants are reviewed operational actions after
migration, not migration side effects. Rollback is: record disable, stop the
Manager, revoke production roles, retain all evidence.

## 6. Implementation implications

H1 creates authority/persistence/readiness but no production handoff. H2 adds
enable/disable and caller-keyed exact canary dispatch through the common Phase E
engine. H3 adds bounded sequential run-once processing with deterministic
per-request operation keys. H4 continuous operation requires separate
acceptance and is not part of the initial implementation.

Canonical contracts, PostgreSQL reconstruction, C++ reconstruction, golden
vectors, same-hash/different-canonical rejection, owner-DML guards, full
canonical replay, bounded `40001`/`40P01` retry, and actual
`pqxx::in_doubt_error` injection are implementation acceptance gates.

## 7. Verification and operational evidence

Before the first enable event, readiness must prove migration 055 identity and
checksum, exact scheduler generation/protocol/cutover evidence, the independent
verification reference, approved Manager service/build contracts, actual
session principal and direct/inherited roles, Completion V1 nested-V2 proof,
and zero unresolved rollout blockers. The scheduler/global-control process
regressions deferred while production workers were active must run in a safe
window.

Rollout is disabled migration/deployment, role audit, readiness, explicit
enable, one exact canary, evidence inspection, one-request run-once, and only
then bounded reviewed increases. H4 is considered separately.

## 8. Alternatives considered

### 8.1 Configuration or role membership as global enablement

Rejected because neither produces immutable, replayable evidence tied to an
exact scheduler and Manager build contract.

### 8.2 Scheduler-owned campaign polling

Rejected because it would make the scheduler consume Campaign Operations
policy, add privileges or a work class, and blur claim versus orchestration
ownership.

### 8.3 Boolean-only request authorization

Rejected because a mutable witness would repeat the
`completion_boundary_closed` competing-authority defect corrected in Phase G.

### 8.4 Continuous Manager in the initial increment

Rejected because concurrency safety does not decide supervision, autostart,
cadence/backoff, shutdown, health, logging, deployment ownership, or churn.

## 9. Relationships to other ADRs and amended wording

- ADR-0010 continues to own Campaign Operations scope; this ADR adds only the
  bounded production consumer and retains every prohibited autonomy boundary.
- ADR-0011 authorization remains necessary and is revalidated; global enable
  and request admission do not replace it.
- ADR-0012 budget/reservation authority and accounting are unchanged.
- ADR-0013 request/lease/binding and Phase E handoff remain authoritative; this
  ADR adds production admission and Attempt V2 and requires one shared engine.
- ADR-0014 completion/lifecycle ownership remains unchanged; Completion V1 is
  retained and production evidence becomes nested request evidence.
- ADR-0015 alone owns expired-lease recovery, cancellation, and reconciliation;
  disable creates no new recovery or process-control authority.
- ADR-0016 scheduler claim/capacity/worker isolation remains unchanged.
- ADR-0017 least privilege/audit remains authoritative and is extended by the
  production roles fixed here.
- ADR-0018 generation-52 exact-attempt authority remains unchanged. This ADR
  requires its exact evidence and does not promote generation 52 approval to a
  future generation.

This ADR precisely amends the accepted Campaign Operations architecture
§31.8 statements that Phase H “persistence/services/CLI” are “scheduler-owned
only,” that migration effects are only those of ADR-0016, and that Phase E
enablement is merely a default-off configuration/privilege change. Scheduler
hardening remains scheduler-owned; durable production admission, exact
dispatch, and the Manager are Campaign Operations-owned. It also amends the
§37 traceability row that assigned production enablement only to scheduler
ownership plus operational deployment approval. No other ADR wording is
superseded.

## 10. References

- [Normative Phase H architecture](../CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md)
- [Accepted Campaign Operations architecture](../../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [ADR index](README.md)
- [Phase G final independent verification](../../../ArchitectureReviews/CampaignOperations/06_Phase5_Operational_Completion/2026-07-31/CampaignOperations_Phase5_FinalFour_IndependentCEEVerification_Output.md)

## 11. Revision history

| Date | Change |
|---|---|
| 2026-07-31 | Accepted durable production admission, Campaign Operations-owned Manager, exact generation-52 evidence, run-once-first rollout, disable-first rollback, and H4 exclusion. |
