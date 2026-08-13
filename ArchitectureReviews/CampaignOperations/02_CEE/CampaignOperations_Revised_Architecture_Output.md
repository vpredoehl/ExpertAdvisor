# Campaign Operations Architecture

Status: **Accepted Campaign Operations V1 architecture**

Architecture contract version: **1.3**

Operational implementation status: **Campaign Operations Phase 5 / architectural Phase G implemented; Phase H runtime not implemented**

Last amended: **2026-07-31**

# 1. Executive Decision, Authority, and Acceptance Status

This file is the repository's detailed normative Campaign Operations V1
specification and the accepted refinement of Volume X. Its authority comes
from accepted ADR-0010 through ADR-0019; it does not override those ADRs or
Volume I. A Proposed ADR remains non-authoritative.

The current authority facts are deliberately separate:

- **Operational implementation status:** Recommendation Governance Phases 4–6
  and Campaign Operations Phases 1–5 / architectural A–G are implemented
  through migration 054. Phase H production admission, Attempt V2, production
  dispatch, and Manager runtime remain unimplemented.
- **Architectural acceptance status:** ADR-0009 and ADR-0010 through ADR-0019
  are Accepted. Volume X incorporates this specification and the normative
  Phase H document as its detailed contract.
- **Production enablement status:** architecture acceptance authorizes bounded
  implementation under the per-increment gates in §31. It does not grant
  database-role membership, enable a runtime service, operate a scheduler, or
  permit production dispatch. ADR-0019 requires migration 055, exact readiness,
  reviewed role assignment, an explicit enable event, and canary-first rollout.

Campaign Operations is a bounded subsystem for durable, long-lived
coordination of explicitly authorized work over one exact Phase 4D
materialization. It is not a new Recommendation Governance phase and must
never be called Phase 6E.

Campaign Operations is a new, bounded subsystem for durable, long-lived coordination of explicitly authorized recommendation-campaign work. It does not extend Recommendation Governance, replace Experiment Lifecycle, allocate scheduler capacity, supervise workers, or interpret scientific results.

The governing sequence is:

```text
Exact Phase 4D materialization scope
+ optional Phase 6D governance prerequisite/provenance
+ explicit operational authorization
+ separately granted member-unit budget
→ reservation
→ accepted durable operational request
→ dispatch attempt acquisition evidence
→ accepted Phase 5/lifecycle handoff
→ immutable downstream binding and budget commitment
→ ordinary pending experiments
→ scheduler admission and claim
→ worker execution
→ lifecycle evidence
→ reconciliation
→ operational completion
```

The central decisions are:

- Phase 6D ratification is upstream governance evidence, never operational authorization.
- V1 has exactly one origin kind, `phase4d_materialization_v1`; Phase 6D is never an
  executable origin. At most one V1 operational campaign may bind a Phase 4D
  materialization, independent of provenance labels.
- An operational campaign is bound to one exact Phase 4D materialization manifest. Its persisted members remain the sole membership authority.
- Initial budget accounting uses integer materialized-member dispatch units, not speculative CPU time or monetary cost.
- Reservation and operational request commit first; the request is the durable dispatch outbox.
- One system-derived logical-operation key makes the complete V1
  campaign/action/materialization request unique; actor, reason, or a later grant cannot
  create a second logical request.
- A later dispatch transaction reuses the existing Phase 5 transaction-bound launch workflow and atomically records downstream bindings and reservation commitment.
- Request-to-experiment bindings remain authoritative after experiments progress beyond `pending`.
- Campaign pause and cancellation control future Campaign Operations actions; they do not directly control workers.
- Operational completion records settled coordination, not scientific success.
- Campaign Operations Phases A–D require no scheduler change. Phase E must remain
  disabled or isolated-test-only until scheduler claim hardening is accepted,
  implemented, and independently verified. Even after that prerequisite,
  production requires ADR-0019 H1/H2 implementation, exact readiness, reviewed
  role membership, and an explicit enable event.

Implementation remains incremental. Architecture acceptance never authorizes
production data mutation, deployment, scheduler operation, or role enablement
without a separately scoped implementation or operational task.

# 2. Repository-Grounded Context

Targeted repository evidence establishes these fixed contracts:

- `Volume_I_Foundation.md` requires PostgreSQL authority, canonical-text identity, hash collision checks, narrow transactions, explicit idempotency, global lock ordering, least privilege, and accepted ADRs before material architecture changes.
- ADR-0003 keeps recommendation generation, scoring, ranking, and review advisory.
- ADR-0005 permits explicit conversion of an approved Phase 4C proposal into one ordinary `paused/train` experiment. Approval alone never queues it.
- Migration 041 and the Phase 4D materialization implementation persist one immutable ordered campaign manifest. Those member rows are the sole membership authority.
- Phase 5 execution, activation, and launch reuse Phase 4C authority. The Phase 5 launch transaction can create paused experiments and activate them to ordinary `pending/train` atomically, but starts no scheduler or worker.
- `LaunchRecommendationCampaignInTransaction` is explicitly transaction-bound: its caller owns the transaction, and it reuses the established proposal, activation, and experiment lock domains.
- Phase 5 campaign status is read-only and follows exact materialization membership into ordinary lifecycle evidence.
- Phase 5 outcome assessment and policy are advisory and explicitly do not declare campaign success or authorize follow-up.
- Phases 6A–6D create a pure proposal, append-only proposal persistence, immutable administrative review, and immutable governance ratification. Every layer fixes negative operational-authority semantics.
- Migration 044, Phase 6D domain/repository/service sources, tests, and project
  membership in the committed repository tree prove Phase 6D implementation exists.
- ADR-0004 requires a durable atomic scheduler claim before process launch.
- The current scheduler polls ordinary `pending` experiment rows. `LoadPendingExperiments` performs an unlocked candidate read, while `MarkExperimentRunning` performs an unconditional update without a `pending` compare-and-set predicate or durable scheduler-attempt identity. No scheduler-attempt table was found. This is an implementation gap relative to ADR-0004.

PostgreSQL remains authoritative. Logs, process presence, PIDs, projections, and in-memory control state cannot supersede persisted workflow evidence.

# 3. Formal Phase 6 Closure

Phase 6D is implementation-complete within its deliberately non-operational scope:

- Migration 044 creates its append-only ratification event.
- The pure contract fixes the authority role, approved-review eligibility, negative operational powers, canonical identity, and reviewer/ratifier separation.
- The repository implements exact replay, changed-payload conflict, collision-safe canonical comparison, restrictive provenance, and bounded reads.
- The service owns one transaction from validated review loading through commit.
- Focused domain, migration, repository, service, concurrency, corruption, ACL, and regression tests exist.
- The Xcode project includes the implementation.
- Volume VIII states that Phases 6A–6D are implemented.

Formal architectural closure is now complete:

- ADR-0009 is Accepted and records the implemented migration 044 history.
- The ADR index and Volumes VIII/XII align Phase 6D as implemented,
  governance-only evidence.
- Phase 6D documentation states that any operational consumer requires its own
  accepted authority; it does not imply a Phase 6E roadmap.
- ADR-0010 through ADR-0019 accept the bounded Campaign Operations ownership,
  authorization, accounting, request/handoff, lifecycle, recovery, scheduler,
  privilege, and audit decisions.

Phase 6D remains optional prerequisite/provenance for the same exact Phase 4D
scope and never becomes operational authorization.

# 4. Scope and Architectural Rationale

## 4.1 In scope

V1 covers one operational coordination envelope per exact Phase 4D materialization;
explicit operational grants; a single integer member-dispatch budget ledger;
reservations; one full-materialization logical request; dispatch-attempt audit; atomic
accepted Phase 5 handoff; bindings and control ownership; pause/resume/cancellation;
bounded reconciliation/restart recovery; operational completion; PostgreSQL invariants,
privileges, migrations, audit, and read-only status.

## 4.2 Out of scope

V1 excludes recommendation selection/recomputation, a new Phase 6 materialization,
partial-member dispatch, automatic/adaptive budgeting, forecasting, autonomous or
profitability-driven research, scientific/statistical policy, new scheduler work
classes/capacity/priorities, worker launch/supervision/termination, direct experiment
SQL, and the broader reserved Volume X automation roadmap.

## 4.3 Rationale

Campaign Operations is the next bounded architectural concern after formal Phase 6 closure, but it is not another Recommendation Governance phase.

It refines the operational-coordination portions of Volume X Research Automation:

- durable campaigns;
- explicit operational authority;
- budgets and reservations;
- durable requests;
- audit;
- pause, cancellation, reconciliation, and restart recovery.

It does not supersede Volume X. Volume X’s broader proposal-policy, autonomous selection, adaptive research, and evidence-loop ambitions remain reserved. Campaign Operations should receive a dedicated accepted ownership ADR and either a dedicated architecture volume or a clearly bounded accepted refinement of Volume X.

The separation is necessary because the platform currently has several distinct forms of positive evidence:

```text
Recommendation approval
Campaign approval
Proposal review approval
Governance ratification
Budget availability
Pending experiment
Scheduler claim
Completed experiment
Favorable scientific interpretation
```

None is interchangeable with another. Combining them would turn advisory evidence, money, or capacity into implicit execution authority.

# 5. Terminology

| Term | Precise meaning |
|---|---|
| Recommendation campaign | The Phase 4D planning, review, approval, materialization, and exact-member workflow. It is governed by Recommendation Governance. |
| Campaign materialization | The immutable Phase 4D manifest and ordered member rows linking an approved campaign to exact Phase 4C proposals. It is the sole campaign-membership authority. |
| Follow-up proposal | The Phase 6A/6B immutable advisory value derived from exact Phase 5 assessment and policy evidence. It specifies no automatic downstream action. |
| Governance ratification | The Phase 6D immutable decision that an independent governance actor ratified advancement of an exact approved Phase 6C review. It is not operational authorization. |
| Operational campaign | A Campaign Operations-owned durable coordination envelope bound to one exact immutable executable materialization scope. Its successfully inserted immutable row is the authoritative creation fact; together with its same-transaction authoritative creation/audit evidence it establishes `awaiting_operational_authorization`, which the row deterministically derives after restart. |
| Campaign origin | V1 fixed literal `phase4d_materialization_v1`: the exact Phase 4D materialization is the executable origin and scope. Ratification and other provenance labels are not origin kinds. |
| Operational authorization grant | An explicit append-only grant, made by a fixed authorized role, permitting named Campaign Operations actions against one exact operational-campaign identity and scope. It is the only operational authority. |
| Budget grant | The first entry, or an explicit superseding entry after revocation, in the single append-only budget ledger. It grants integer member-dispatch units but no permission to act. |
| Reservation | A durable hold of budget units for one exact operational-request identity before downstream handoff. |
| Logical-operation key | The system-derived, canonical, versioned identity for the one V1 full-materialization action. It is the durable idempotency and database uniqueness domain; actor, reason, grant, and timestamp do not change it. |
| Accepted operational request | Campaign Operations-owned durable intent that committed with one held reservation and active operational grant for one named, versioned action over the complete materialization. Acceptance is not dispatch or downstream acceptance. |
| Dispatch | Campaign Operations’ bounded act of selecting an accepted request and invoking the accepted lifecycle handoff. It is not a scheduler claim. |
| Accepted downstream handoff | Invocation of the existing transaction-bound Phase 5/Phase 4C lifecycle authority, with all exact bindings and reservation commitment in the same PostgreSQL transaction. The historical Phase 5 name “launch” means lifecycle handoff to `pending/train`, not worker launch. |
| Downstream binding | Immutable causality from one request/member to exact Phase 4C execution, activation, and experiment evidence. A binding is not by itself a transferable control right. |
| Downstream control ownership | The separately persisted, one-controller V1 right to coordinate cancellation through the lifecycle service. V1 ownership is permanent and has no transfer or release operation. |
| Scheduler admission | The ordinary experiment lifecycle condition that makes work eligible for scheduler consideration, currently `pending` with a runnable phase. |
| Scheduler claim | The scheduler-owned atomic acquisition of one lifecycle attempt before worker launch. |
| Operational attempt | Audit-only acquisition/outcome evidence for one Campaign Operations dispatch attempt. Request state/version and its lease, not an attempt row, authorize selection. It is not a scheduler attempt or worker execution. |
| Cancellation | An explicit request to stop future Campaign Operations activity and, where separately authorized and lifecycle-valid, request ordinary downstream cancellation. |
| Reconciliation | Bounded comparison of authoritative Campaign Operations and lifecycle evidence. **Reconciliation detects and records; owning services perform every repair transition.** |
| Operational completion | An append-only decision that all Campaign Operations obligations, reservations, requests, bindings, cancellations, and terminal lifecycle evidence are settled. |
| Scientific outcome | The Phase 5 evidence classification and policy interpretation of experiment results. It is neither operational completion nor execution authority. |

# 6. Ownership Boundaries

| Owner | Authoritative responsibilities | Explicitly does not own |
|---|---|---|
| Recommendation Governance | Planning; recommendation/proposal policy; review and approval; immutable Phase 4D materialization and ordered membership; Phase 4C proposal/conversion authority; Phase 5 assessment evidence; Phase 6 proposal, review, approval, and governance ratification | Operational-campaign identity, operational grants, budgets, reservations, operational requests/dispatch/control/completion, scheduler policy, worker control |
| Campaign Operations | Operational-campaign identity; operational authorization grants; budget ledger; reservations; durable requests; dispatch-attempt audit; downstream bindings and control ownership; operational controls; reconciliation observations; operational completion and audit evidence | Recomputing or reinterpreting Recommendation Governance decisions; direct experiment/lifecycle writes; scheduler capacity/claims; worker computation; scientific interpretation |
| Experiment Lifecycle | Experiment creation through accepted workflows, lifecycle states and legal transitions, terminal experiment evidence | Campaign budget or authorization, scheduler selection, recommendation policy |
| Scheduler | Ordinary eligibility polling, capacity, atomic claims, attempts, launch, supervision, recovery, completion transitions | Campaign policy, campaign membership, operational authorization, budget |
| Workers | One bounded training, inference, or analysis computation; reporting facts | Selecting work, granting authority, creating campaigns, marking campaign completion |
| PostgreSQL | Committed durable truth, constraints, references, uniqueness, locks, immutable events, migration history | Inventing missing policy, provenance, or business meaning |

# 7. Campaign Operations Responsibilities

Campaign Operations owns:

- creating one immutable operational-campaign identity from exact accepted scope;
- validating upstream identities without recomputing governance truth;
- recording operational authorization grants, revocations, observed expiries, and one-event supersession transitions;
- maintaining an append-only member-unit budget ledger;
- acquiring, committing, releasing, and expiring reservations;
- persisting durable operational intent before dispatch;
- selecting dispatchable requests in bounded batches;
- invoking accepted lifecycle workflows without bypassing them;
- recording immutable request-to-downstream bindings;
- administratively pausing and resuming future dispatch;
- recording and coordinating cancellation;
- observing lifecycle truth without rewriting it;
- bounded reconciliation and restart recovery;
- append-only operational completion;
- read-only status and audit projections.

# 8. Prohibited Responsibilities

Campaign Operations must never:

- create experiments directly from scores, ranks, approval, or Phase 6D ratification;
- infer executable scope from a follow-up proposal that does not contain it;
- reconstruct campaign membership from current recommendations or queries;
- substitute budget availability for authorization;
- substitute process presence for execution or completion;
- assign scheduler capacity or priority;
- select runnable experiment phases;
- claim scheduler attempts;
- launch or supervise workers;
- signal processes as an implicit consequence of campaign control;
- update experiment results, models, inference evidence, or analysis evidence;
- rewrite completed, failed, or cancelled experiment outcomes;
- declare profitability, statistical validity, or scientific success;
- repair missing lifecycle facts by invention;
- bypass Phase 4C/5 workflows with direct repository or SQL writes.

The listed ownership is exact. In particular, Phase 6D ratification must never
authorize execution, create experiments or operational requests, reserve budget,
dispatch work, alter scheduler eligibility, or imply an operational roadmap stage.
Budget must never substitute for an operational grant. Campaign Operations must load
and validate the immutable upstream decision chain but must not score, rank, reselect,
recompute eligibility, or reinterpret any governance or scientific conclusion.

The scheduler must never interpret Campaign Operations or recommendation-governance policy.

# 9. Subsystem Decomposition

| Component | Responsibility | Authority |
|---|---|---|
| Pure domain contracts | Identity, canonicalization, validation, budget arithmetic, state transitions, completion classification, reason codes | Authoritative rules; no durable state |
| Authorization service | Validate exact upstream evidence and append one serialized grant, revoke, expiry-observation, or supersede transition; supersede appends one successor `granted` event | Authoritative authorization-event chain |
| Budget service | Append the single versioned grant/amend/revoke/supersede ledger and enforce integer invariants | Authoritative budget ledger; no snapshot or adjustment authority |
| Reservation service | Acquire, settle, release, or expire holds | Authoritative reservation state and events |
| Operational-request service | Create durable intent and bind one reservation | Authoritative request |
| Dispatch/handoff service | Select ready requests, invoke accepted Phase 5/lifecycle workflow, record binding | Authoritative dispatch result and binding |
| Campaign-control service | Append pause/resume events and immutable cancellation requests; coordinate lifecycle-owned cancellation | Authoritative control and cancellation-request evidence |
| Reconciliation service | Record bounded observations and request named owning-service operations only | Observations authoritative as evidence; never repair authority |
| Completion service | Validate settled evidence and append terminal operational decision | Authoritative completion event |
| Audit projection | Present causal actions across domain event tables | Derived; never current-state authority |
| Repositories | SQL, typed mapping, exact persistence primitives, locks, constraints | Persistence boundary |
| Current-state projection | Efficient campaign/request/member status | Derived and rebuildable |
| Scheduler-facing boundary | Expose only ordinary experiment IDs and lifecycle evidence after accepted handoff | No policy or campaign authority |
| Operator-facing boundary | Authenticate actors, request explicit mutations, and inspect durable state | Presentation/orchestration only; no embedded SQL or policy |

# 10. Operational Authorization

An operational grant is the only evidence that permits Campaign Operations to reserve budget, accept a request, or dispatch.

Required upstream evidence is:

1. One exact `operational_campaign` canonical identity.
2. One exact Phase 4D materialization ID, canonical text, hash, contract version, and ordered persisted membership.
3. The complete valid Phase 4D/Phase 4C eligibility chain required by the intended downstream workflow.
4. When the selected operational-grant prerequisite policy requires ratification, the exact Phase 6D event, review, proposal, canonical identities, and restrictive provenance. It remains prerequisite/provenance for the same Phase 4D scope.
5. A named, versioned downstream action and exact scope.
6. A fixed operational-authorizer role, actor identity, reason, and optional validity interval.

The initial role is `campaign_operations_authorizer`. It is a separately granted
database/application capability whose role text is fixed by the contract, not supplied
arbitrarily. The only human-principal inequality already fixed upstream is Phase 6C
reviewer versus Phase 6D ratifier. V1 does not silently invent additional human
separation rules: any required inequality between operational authorizer, budget
administrator, dispatcher, adopter, or canceller must be accepted in ADR-0011/0017.
Capabilities remain separate even when policy permits the same authenticated human to
hold more than one.

V1 campaign origin is always `phase4d_materialization_v1`. Ratification is a
prerequisite only where the operational-grant prerequisite policy requires it. It
never substitutes for the operational grant, changes campaign identity or membership,
or defines new executable work. A future follow-up proposal remains unsupported until
a separately accepted workflow produces another exact executable materialization.

The V1 prerequisite-policy domain is closed: `phase4d_materialization_only_v1` or
`phase4d_materialization_plus_exact_phase6d_ratification_v1`. The latter requires one
fully validated ratification whose transitive proposal/materialization identity equals
the campaign's exact Phase 4D materialization; the former records ratification as
optional provenance only. The chosen literal is immutable in the grant canonical and
cannot be inferred from row presence. Neither policy creates a different campaign
origin or membership.

Authorization events form one append-only, monotonically versioned chain per
`(campaign, action_contract, scope_contract)`:

- `granted`;
- `revoked`;
- `expiry_observed`, which is audit evidence only and does not replace the time check.

`supersede` is a transition/service operation, not a persisted event kind. It appends
exactly one successor event whose kind is `granted`; no separate `superseded` event row
exists.

Rules:

- A grant binds one campaign, action set, scope, actor, role, reason, and validity interval.
- The first grant is chain version 1. Every later event names the exact prior head and
  increments by one. A supersede operation appends exactly one new `granted` head that
  names the exact prior head and increments the chain version exactly once; the prior
  event remains immutable and is historically superseded only by that successor
  relationship. Revocation appends a `revoked` head. Unique `(campaign, action, scope,
  chain_version)` and `previous_event_id` constraints prohibit forks and competing
  successors.
- The effective grant is deterministic: load and lock the chain conflict key; select the
  unique highest chain version; it is active only when that head is `granted`, the
  transaction's PostgreSQL time is in `[not_before, expires_at)`, and its exact campaign,
  action, scope, and prerequisite evidence validate. A later row cannot backdate an
  overlapping active grant.
- `not_before` is required. `expires_at` is optional; canonical text renders no expiry
  exactly as `expires_at=none`. Supplied instants are UTC with six fractional digits,
  and no local-time or application-clock default may be inferred.
- Budget may exist before or after authorization but cannot activate it.
- Scheduler capacity is neither checked nor reserved by authorization.
- Exact supersede retry returns the same successor grant. A stale prior head, changed
  payload, or competing successor conflicts deterministically.
- Exact retry of every other transition returns the existing event.
- Reusing the same authorization request identity with changed scope, actor, reason, validity, or upstream canonical evidence conflicts.
- Revocation immediately blocks new reservations, requests, and dispatch selection.
- Revocation does not undo a committed binding or erase pending/running work.
- Existing uncommitted reservations must be released after authoritative revalidation.
- Existing bound work requires explicit cancellation handling.
- Expiration is checked transactionally using PostgreSQL time before reservation and dispatch.
- Supersession never rewrites the prior event and never inserts a separate inactive
  event; the successor `granted` row is the only effective active head when all
  validity, scope, prerequisite, and time checks pass.

An accepted unbound request remains bound to the exact grant that accepted it. If that
grant is revoked, expires, or ceases to be the head because a supersede operation
appended its successor grant before handoff, the request cannot borrow the successor:
the request-settlement service records permanent failure and releases its held
reservation after proving no binding. A new grant cannot resurrect that request or
create a second V1 logical operation.

Adoption of exact pre-existing `pending/train` Phase 5 evidence additionally requires
the active grant action `adopt_existing_pending_and_control`. Ordinary
`dispatch_full_materialization` authority is insufficient. Adoption consumes the same
one member-dispatch unit per member as creation because it satisfies and binds the
request. The handoff transaction rejects adoption if any experiment already has a
Campaign Operations control owner. V1 control ownership is permanent: there is no
transfer, release, or implicit acquisition through a binding.

Without active operational authorization, only read-only inspection, budget administration, safety cancellation of a campaign awaiting operational authorization, lifecycle-delegated safety cancellation of already-bound work, and reconciliation observation are possible.

A fixed V1 limitation remains: Phase 6D ratification does not define a new executable
materialization. No ratification-origin or follow-up-origin campaign exists in V1.

# 11. Data Model

Canonical text is authoritative for every semantic identity. Hashes may accelerate lookup, bucket locking, and diagnostics only.

## 11.1 Persistence principles

The names below are normative conceptual names; the accepted schema ADR may add a
project prefix without changing meaning. All canonical columns use PostgreSQL `C`
collation. Every foreign key is `ON DELETE RESTRICT`. Generated row IDs and
`created_at` values are locator/audit metadata, never semantic identity. Hash indexes
are non-unique accelerators; authoritative database uniqueness is enforced by bounded
typed natural columns. Potentially oversized canonical text MUST NOT be a direct
B-tree uniqueness key. Repositories always compare complete canonical bytes after a
natural-key/hash lookup, so hash collisions remain distinct and changed payloads
conflict deterministically.

Append-only tables deny runtime `UPDATE`, `DELETE`, and `TRUNCATE`, accept only
payload-column `INSERT`, protect generated columns, and use insert triggers only for
cross-row provenance that cannot be expressed by declarative constraints. Guarded
mutable projections (`reservation`, `request`, and optional read projection) are
updated only by named transition functions or column-scoped repositories with expected
state/version predicates. Every successful guarded transition inserts its immutable
event in the same transaction. No process-local mutex supplies durable correctness.

## 11.2 Authoritative tables and constraints

| Table | Purpose, primary/foreign keys | Natural uniqueness, checks, immutability, and indexes | Runtime access and projection relationship |
|---|---|---|---|
| `campaign_operations_campaign` | PK `operational_campaign_id`; restrictive FK to the exact Phase 4D materialization. Immutable materialization ID/version/canonical/hash, fixed origin literal, action-scope contract, campaign canonical/hash. The successfully inserted row is the authoritative creation fact and derives initial state `awaiting_operational_authorization`. | `UNIQUE(materialization_id)` enforces at most one V1 campaign independent of provenance. `CHECK(origin_kind='phase4d_materialization_v1')`; supported versions, non-empty canonical, tagged-hash shape, and exact member count checks. The canonical hash index is non-unique; exact canonical comparison decides replay/conflict. | Owned by `campaign_operations_owner`; payload insert only to `campaign_operations_campaign_creator`; all Campaign Operations services may read. It is not a mutable status row; its same-transaction audit reference proves creation causality but is not competing state authority. |
| `campaign_operations_governance_provenance_event` | PK event ID; FKs to campaign and exact Phase 6D ratification/review/proposal chain. Optional provenance/prerequisite evidence only. | Unique typed `(campaign_id, ratification_event_id)` natural key; non-unique hash index; exact copied canonical/hash fields validated by trigger, hydration, and full canonical replay comparison. Append-only. It cannot change origin, scope, grant, or request state. | Insert only to authorizer role; read to authorization/audit roles. No scheduler/lifecycle privileges. |
| `campaign_operations_authorization_event` | PK authorization event ID; FK campaign; self-FK `previous_event_id`; immutable effective event kind (`granted`, `revoked`, or `expiry_observed`), exact prerequisite evidence, action/scope, fixed role, actor/reason, `not_before`, nullable `expires_at`, canonical/hash. `supersede` is a service operation represented by one successor `granted` row. | Unique `(campaign_id, action_kind, scope_contract_version, chain_version)` and unique non-null `previous_event_id`; chain version positive and predecessor exactly `n-1`; event-shape checks. A unique prior edge plus the transaction rule prevents competing successors/heads. No `superseded` kind or row exists. Append-only; indexes on chain head and canonical hash. | Payload insert only through authorizer transition capability. Dispatch/reservation read. No budget, request, lifecycle, or scheduler mutation grant. |
| `campaign_operations_budget_ledger_entry` | PK budget entry ID; FK campaign; self-FK prior entry. The **only** budget authority. Immutable version, kind (`grant`, `amend`, `revoke`, `supersede`), signed delta, prior total, resulting total, unit literal, status, actor/reason, canonical/hash. | Unique `(campaign_id, ledger_version)` and unique non-null prior entry; `resulting_total=prior_total+delta`, all totals nonnegative, first entry is `grant`, `revoke` produces `revoked` head, `supersede` is required to leave a revoked head. Under-lock service check and deferred constraint require `resulting_total >= committed+held`. Index campaign/version descending and hash. Append-only. No separate snapshot or adjustment table exists. | Insert only to budget administrator. Reservation/dispatch/completion read. The latest entry is derived under lock, never copied into an authoritative mutable budget snapshot. |
| `campaign_operations_reservation` | PK reservation ID; FKs campaign, accepting authorization grant, and budget entry; immutable logical-operation canonical/hash, exact full scope, unit type/amount, optional semantic expiry. Guarded `state`, `state_version`, and settlement event FK. | Unique typed V1 logical-operation natural key and later unique consuming request; non-unique canonical-hash index; amount equals authoritative materialization member count; states `held|committed|released|expired|reconciliation_required`; terminal shape checks prohibit commitment after release/expiry and release/expiry after commitment. Index campaign/state/expiry and budget entry. | Insert/guarded transition only to reservation service; dispatcher gets only commit transition; cancellation/settlement gets release/expiry transition only through the §19 authorization-if-required → budget → campaign → reservation → request order. Current state is authoritative only with its same-transaction event history. |
| `campaign_operations_reservation_event` | PK event ID; FK reservation and optional binding-set/request/cancellation/reconciliation cause. Immutable acquisition, commitment, release, expiry, or inconsistency evidence with prior/resulting versions and canonical/hash. | Unique typed `(reservation_id, resulting_state_version)` natural key; non-unique hash index and full canonical comparison; shape checks tie event kind to state. Append-only; index reservation/version. | Insert only through the same named reservation transition that updates the reservation row. Read by audit/completion. |
| `campaign_operations_request` | PK request ID; FKs campaign, exact accepting authorization grant, and reservation. Immutable logical-operation key, action/version, exact materialization canonical/hash and full member count, prerequisite identities, accepting actor/reason, request canonical/hash. Guarded state/version and dispatch lease fields. | `UNIQUE(campaign_id, action_kind, action_contract_version)` is the V1 full-scope logical-operation uniqueness domain; `UNIQUE(reservation_id)` enforces one consumer. Canonical hash indexes are non-unique and full canonical comparison decides replay/conflict. States are `ready|dispatching|bound|permanently_failed|cancelled|reconciliation_required`; no undefined `recorded` state. Lease fields are present only in `dispatching`; terminal states cannot lease. Index dispatchable state/lease expiry. | Request service inserts; dispatcher may acquire/clear lease and bind through named transitions; cancellation/settlement may terminalize. Immutable payload never changes. |
| `campaign_operations_dispatch_attempt` | PK attempt ID; FK request. Immutable **acquisition** audit: ordinal, expected/resulting request version, lease token digest/expiry, dispatcher, canonical/hash. It contains no later outcome. | Unique typed `(request_id, attempt_ordinal)` natural key; non-unique hash index and full canonical comparison. Attempt ordinal is allocated under request lock. Append-only; index request/ordinal. It is audit-only and never dispatch authority. | Dispatcher inserts on successful lease acquisition. All transition authorization comes from locked request state/version and lease, not this table. |
| `campaign_operations_dispatch_attempt_outcome` | PK outcome ID; unique FK attempt; immutable terminal result, diagnostics, request/binding/reservation resulting versions, canonical/hash. | One outcome per attempt; outcome enum and causal-shape checks. Exact retry returns the row; changed outcome conflicts. Append-only; index result/reason. | Dispatcher or owning recovery transition inserts. It never changes request state by itself. |
| `campaign_operations_request_binding` | PK binding ID; FKs request, exact materialization member, proposal, approving review, conversion execution, activation, and experiment. Immutable per-member creation/reuse dispositions and canonical/hash. | Unique `(request_id, materialization_member_id)`; exact proposal/member FKs; deferred completeness trigger requires exactly the persisted ordered materialization membership before request can become `bound`. Dispositions freeze `created` versus `adopted_existing_pending` and exact Phase 5 execution/activation results. Append-only; indexes request/member and experiment. | Insert only by dispatcher inside the accepted Phase 5 transaction. Lifecycle/scheduler cannot write it. Later experiment progress never invalidates it. |
| `campaign_operations_downstream_control_owner` | PK control-owner ID; FKs binding, request, and experiment. Immutable V1 control attribution. | `UNIQUE(experiment_id)` and `UNIQUE(binding_id)`; mode is `created_control` or `authorized_adoption_control`. Adoption mode requires an authorization event containing `adopt_existing_pending_and_control`, enforced by service plus trigger. Append-only. V1 defines no transfer/release row or operation. | Dispatcher inserts atomically with binding. Cancellation may read and invoke lifecycle control but cannot update ownership. |
| `campaign_operations_control_event` | PK control event ID; FK campaign and optional prior control event. Immutable `pause` or `resume`, expected control version, actor/capability/reason, canonical/hash. | Unique typed `(campaign_id, control_version)` natural key and previous-event chain; alternating/legal-state checks; non-unique hash index and full canonical comparison. Append-only; index campaign/version descending. | Campaign-control role inserts. Optional state projection is derived from the unique latest event. |
| `campaign_operations_cancellation_request` | PK cancellation request ID; FKs campaign and optional request/binding/control-owner target. Immutable scope, expected target version, actor/capability/reason, canonical/hash. | Natural unique target/scope/logical cancellation key; exact replay returns existing and changed payload conflicts. Append-only; index unsettled target. It contains no mutable or future settlement reference. | Cancellation role inserts. It does not receive direct experiment update or process-signal privileges. |
| `campaign_operations_cancellation_settlement` | PK settlement ID; FK cancellation request and exact reservation/request transition or lifecycle control evidence. Immutable disposition and canonical/hash. | Unique cancellation request ID for terminal settlement; dispositions include `unbound_cancelled`, `lifecycle_request_accepted`, `already_terminal`, `running_cancellation_not_supported`, and `inconsistent`. Append-only; causal-shape checks and index disposition. | Insert by cancellation coordinator only after the Campaign Operations transition or accepted lifecycle service returns authoritative evidence. |
| `campaign_operations_reconciliation_observation` | PK observation ID; FK campaign and typed target. Immutable reconcile-run key, reason code, expected version, exact point-in-time evidence canonical/hash, recommended owning service/action, diagnostics, canonical/hash. | Unique bounded typed reconcile-run/target/evidence-version natural key; non-unique hash index and full canonical comparison; reason/action enum checks. Append-only; bounded-query indexes by unresolved target, state, and ID cursor. Observation time is metadata. | Reconciler may read authoritative tables and insert observations only; it receives no repair mutation capability. |
| `campaign_operations_reconciliation_resolution` | PK resolution ID; FK observation and exact event/transition produced by an owning service. Immutable result, resolver service/capability, canonical/hash. | Unique observation ID for effective resolution; exact causal FK is required; routine observation cannot self-resolve. Append-only. Unresolved state is `observation LEFT JOIN resolution WHERE resolution IS NULL`, so historical observations do not permanently dominate. | Insert only by the owning transition service or a narrow resolution recorder after that service commits. The reconciler cannot forge it. |
| `campaign_operations_completion_event` | PK completion ID; unique FK campaign. Immutable terminal administrative state, classification, exact budget/reservation/request/binding/cancellation summaries, point-in-time terminal lifecycle evidence, actor/service reason, canonical/hash. | One row per campaign; deferred validation requires all completion prerequisites and disjoint classification. Append-only; hash index. No override/supersession column exists in V1. | Insert only to completion service. Later lifecycle retry/requeue cannot update/delete it. |
| `campaign_operations_audit_reference_event` | PK audit ID; FKs the exact authoritative domain event/transition and causal identities. Immutable actor/capability/reason, expected/prior/resulting version, stable outcome/diagnostic, replay disposition, database time. | One same-transaction reference per material action/result where applicable; checks require at least one authoritative cause and prohibit invented unrelated IDs. Append-only; indexes campaign, request, actor, cause, time. | Insert by each narrow service only for its own actions. It is an index over authority, never a competing truth table. |
| `campaign_operations_current_projection` (optional) | PK campaign/member/request composite; source high-water marks and derived display state only. | Rebuildable; uniqueness mirrors projection key. It has no FK from authoritative mutation tables and cannot be consulted to authorize a write. | Owned/written only by `campaign_operations_projection_writer`; status roles may read. All mutation roles are denied `SELECT` if needed to prevent accidental authorization use. |

## 11.3 Immutability, deletion, and retention

Cross-row provenance triggers compare complete canonical content, never hashes alone.
Owner-only repair is not a runtime API and must use a separately reviewed exact repair
procedure that appends evidence; ordinary roles receive no history deletion. Archival
or partition retirement remains deferred to a retention ADR and must preserve every
restrictive reference and the ability to reconstruct authorization, accounting,
causality, and completion. Retention duration is not a Phase A blocker.

# 12. Identity and Deterministic Replay

V1 canonical prefixes are fixed, distinct, and versioned:

- `campaign_operations_campaign_v1`
- `campaign_operations_governance_provenance_v1`
- `campaign_operations_authorization_event_v1`
- `campaign_operations_budget_ledger_entry_v1`
- `campaign_operations_reservation_v1`
- `campaign_operations_reservation_event_v1`
- `campaign_operations_logical_operation_v1`
- `campaign_operations_request_v1`
- `campaign_operations_dispatch_attempt_v1`
- `campaign_operations_dispatch_attempt_outcome_v1`
- `campaign_operations_dispatch_binding_v1`
- `campaign_operations_downstream_control_owner_v1`
- `campaign_operations_control_event_v1`
- `campaign_operations_cancellation_request_v1`
- `campaign_operations_cancellation_settlement_v1`
- `campaign_operations_reconciliation_observation_v1`
- `campaign_operations_reconciliation_resolution_v1`
- `campaign_operations_completion_v1`

Every canonical value is serialized as the prefix followed by semicolon-delimited,
fixed-order ASCII field names. Variable UTF-8 or embedded canonical values use
`BYTE_LENGTH:VALUE`; enums use fixed lowercase ASCII literals; booleans use
`true|false`; integers use unsigned or signed base-10 as defined by the field; and
collections include a count followed by items in their contract order. The tagged V1
digest is the repository-standard `fnv1a64:<16 lowercase hex>` accelerator. Equality,
replay, and conflict always compare complete canonical text after any digest lookup.

Authoritative identity inputs, natural/database uniqueness, and replay behavior are:

| Durable concept | Canonical semantic inputs and provenance | Natural/database uniqueness; immutable/mutable boundary; retry/replay |
|---|---|---|
| Operational campaign | Exact materialization ID/version/canonical/hash and member count, fixed `phase4d_materialization_v1`, fixed full-materialization operation contract. Phase 6 evidence is excluded because it is provenance, not origin/scope. | Natural key and DB unique FK are materialization ID. Entire payload immutable. Exact canonical retry returns existing; any changed scope payload conflicts. |
| Governance provenance event | Campaign canonical; exact Phase 6D ratification/review/proposal IDs, versions, canonicals/hashes; prerequisite-policy version. | Unique campaign/ratification pair. Append-only. Retry exact or conflict; never changes campaign scope. |
| Operational authorization event/grant | Campaign canonical; chain version/prior event; effective event kind (`granted`, `revoked`, or `expiry_observed`); exact action/scope; prerequisite evidence; fixed role; actor/reason; required `not_before`; `expires_at=none` or normalized instant. A supersede request canonically identifies the single successor `granted` event and exact prior head; `supersede` is not serialized as an event kind. | Unique campaign/action/scope/version and prior-event edge. Append-only chain. A supersede increments the chain once and exact retry returns that same successor grant; stale prior/version, changed payload, or a competing successor conflicts. Effective-head evaluation considers only the unique highest event and treats it as active only when it is `granted` and all checks pass. Time passing changes effectiveness, never identity. |
| Budget grant/amendment/revocation/supersession | Campaign canonical; ledger version/prior entry; kind/status; unit literal; delta/prior/resulting totals; administrator/reason and exact cause. | Unique campaign/version and prior edge. Append-only single authority. Exact retry returns entry; stale version or changed payload conflicts. |
| Logical operation | Campaign canonical; action kind/version; exact materialization canonical/hash and full-scope literal. It intentionally excludes authorization grant, actor, reason, time, dispatcher, and budget-ledger entry. | Unique typed V1 campaign/action/version natural key; canonical bytes are authoritative but are not a direct oversized-text unique key. Immutable durable idempotency key. Changed request payload under this key conflicts; no later actor/grant creates a second operation. |
| Reservation | Logical-operation canonical; campaign; accepting grant; budget head/version; full member scope and count; integer amount/unit; `expires_at=none` or normalized instant. | Unique logical operation and unique later consuming request. Immutable payload plus guarded state/version. Exact create retry reloads; state-transition retry uses reservation-event identity. |
| Reservation transition/settlement | Reservation canonical; transition kind; expected/resulting state/version; exact request/binding/cancellation/reconciliation cause; amount. | Unique reservation/resulting version. Append-only event paired with guarded projection update. Exact retry returns event; changed or stale transition conflicts. |
| Operational request | Logical-operation canonical; campaign; accepting grant; reservation; action/version; exact materialization and ordered-scope digest/count; immutable accepting actor/reason and prerequisite canonicals. | Unique logical operation, campaign/action/version, and reservation. Immutable payload plus guarded state/version/lease. Exact replay returns request; any changed payload conflicts. |
| Dispatch attempt acquisition | Request canonical; attempt ordinal; expected/resulting request version; lease-token digest and semantic lease expiry; dispatcher. | Unique request/ordinal. Append-only audit header. Retry returns acquisition; no outcome is mutable into it. |
| Dispatch attempt outcome | Attempt canonical; terminal outcome/reason; resulting request/reservation versions; exact binding-set canonical when present. | Unique attempt FK. Append-only. Exact replay returns outcome; changed outcome conflicts and requires reconciliation rather than rewrite. |
| Request-to-downstream binding | Request canonical; materialization-member canonical/ordinal; exact proposal/review/execution/activation/experiment IDs and canonicals/hashes; frozen execution/activation/adoption disposition. | Unique request/member. Append-only and stable after lifecycle progress. Complete set required. Exact replay returns set; changed member evidence conflicts. |
| Downstream control owner | Binding canonical; request; experiment; fixed control mode; exact adoption authorization when applicable. | Unique experiment and binding. Append-only/permanent in V1. Replay exact or conflict; no transfer/release/reconstruction from binding alone. |
| Pause/resume control event | Campaign canonical; control version/prior event; action; expected effective control state/version; actor/capability/reason. | Unique campaign/control version. Append-only. Resume derives eligibility from current authority/budget/request facts; it does not remember a mutable prior state. |
| Cancellation request | Campaign and exact target canonical; scope; expected target state/version; actor/capability/reason; logical cancellation key. | Unique target/scope/logical key. Append-only. Exact retry converges; settlement is a different identity. |
| Cancellation settlement | Cancellation-request canonical; exact Campaign Operations or lifecycle service evidence canonical; disposition; affected reservation/request/binding IDs. | One terminal settlement per cancellation request. Append-only. Exact retry returns it; conflicting evidence enters reconciliation. |
| Reconciliation observation | Reconcile-run/request canonical; target canonical; expected state/version; reason code; exact **point-in-time** upstream/Campaign Operations/lifecycle evidence canonical; recommended owning service/action. | Unique complete observation canonical. Append-only and detection-only. Observation time is excluded. Exact retry converges; a new evidence snapshot creates a new observation. |
| Reconciliation resolution | Observation canonical; exact owning service/capability; exact repair-transition event canonical; resolution disposition. | Unique observation. Append-only. Only existing authoritative transition evidence may resolve; changed evidence conflicts. |
| Operational completion decision | Campaign canonical; disjoint terminal classification; exact settled budget ledger head and arithmetic; ordered reservation/request/binding/cancellation identities; per-member **point-in-time** terminal lifecycle evidence. | Unique campaign. Append-only, no override. Exact replay returns event; any different terminal payload conflicts. |

Canonicalization rules:

- compare bytewise under a specified byte-oriented grammar;
- use fixed field order and fixed ASCII field names;
- frame variable UTF-8 text with its byte length;
- reject malformed UTF-8, NUL, DEL, and prohibited controls;
- render integers as locale-independent base-10 without leading `+`;
- sort collections only by their contract-defined bytewise or numeric order;
- retain materialization members in persisted ordinal order;
- render semantic timestamps in normalized UTC with fixed precision;
- render all supplied semantic instants as `YYYY-MM-DDTHH:MM:SS.ffffffZ`; render an
  optional absent expiry only as the literal `none`; reject a missing required instant,
  offsets after normalization boundaries, more precise input that cannot round-trip,
  and every inferred local-time default;
- exclude database-generated IDs, `created_at`, observation time, PID, process identity, host, transaction ID, and mutable lifecycle state from campaign, logical-operation, request, reservation, and binding identity;
- permit exact point-in-time mutable lifecycle facts only in reconciliation-observation,
  cancellation-settlement, and completion-decision evidence identities because those
  entities identify an observation/decision at a fixed evidence point. Later lifecycle
  state creates new observation evidence and never changes an earlier identity;
- require a contract-version change for grammar or equivalence changes.

Collision handling:

- a hash match only selects candidate rows or a serialization bucket;
- complete canonical text decides equality;
- same hash/different canonical values remain distinct and observable;
- lock-key collisions add serialization only;
- potentially oversized canonical text is never a direct B-tree uniqueness
  key; bounded typed natural keys enforce database uniqueness and canonical
  hash indexes remain non-unique.

Replay:

- exact logical-operation key and exact stored request/result canonical: return `existing_identical`;
- same logical-operation key with any changed payload: `logical_operation_payload_conflict`;
- retry after lost response: reload by request canonical identity;
- if a binding exists, return it without invoking Phase 5 again;
- downstream lifecycle progress never invalidates the immutable binding;
- if no binding exists but downstream evidence has progressed beyond the handoff
  precondition, do not infer causality—record `progressed_unbound_evidence` and fail
  closed;
- serialization failure (`40001`) or deadlock (`40P01`) retries the entire transaction
  with the same canonical operation, a bounded attempt count, and jitter; uniqueness
  conflict reloads and compares complete canonical text before returning identical or
  conflict; other SQL errors are not reclassified as idempotency.

# 13. Lifecycle State Machine

Administrative campaign state is a derived summary over the authoritative campaign row,
authoritative events, and current guarded request/reservation state. A projection is
never mutation authority. Only an
**unresolved** reconciliation observation participates in precedence; an observation
with an exact `campaign_operations_reconciliation_resolution` is historical evidence.
The existence of a valid immutable campaign row is the durable authority for initial
state `awaiting_operational_authorization`; after restart that state is derived until a
later authoritative event changes the summary. Its precedence is:

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

Separate state dimensions must never be collapsed:

| Dimension | States |
|---|---|
| Administrative campaign | `awaiting_operational_authorization`, `authorized`, `budgeted`, `reserving`, `ready`, `dispatching`, `active`, `paused`, `cancellation_requested`, `cancelling`, `terminal_completed`, `terminal_cancelled`, `terminal_failed`, `inconsistent`, `reconciliation_required` |
| Member operational | `not_requested`, `reserved`, `request_ready`, `dispatching`, `bound_pending`, `claimed`, `running`, `terminal_completed`, `terminal_failed`, `terminal_cancelled`, `inconsistent`, `reconciliation_required` |
| Reservation | `held`, `committed`, `released`, `expired`, `reconciliation_required` |
| Request | `ready`, `dispatching`, `bound`, `permanently_failed`, `cancelled`, `reconciliation_required` |
| Downstream experiment | Existing authoritative `paused`, `pending`, `running`, `completed`, `failed`, `cancelled` with `train`, `infer`, `analyze`, `done` |
| Scheduler attempt | Target contract: `claimed`, `launching`, `running`, `completed`, `failed`, `orphaned`, `recovered`; not yet fully implemented in the repository |
| Scientific result | Existing not-ready, comparable success, context-changed, metric-gap, terminal failure/cancellation, inconsistent, plus separate advisory interpretation |

The following campaign transitions are exhaustive:

| Transition | Authority and preconditions | Transaction owner and durable result | Retry / invalid behavior |
|---|---|---|---|
| none → `awaiting_operational_authorization` | Campaign creator validates the complete exact Phase 4D materialization and every creation invariant before commit | Campaign service inserts the immutable campaign and its same-transaction creation/audit reference; no readiness/control event exists | Exact replay returns campaign; changed scope conflicts; rollback leaves no durable partial state |
| `awaiting_operational_authorization` → `authorized` | Operational authorizer; active exact grant | Authorization transaction appends grant | Exact replay returns event; changed grant conflicts |
| any unbound projection whose accepting grant becomes inactive → deterministically derived eligible state | Referenced accepting grant is revoked, expires, or ceases to be the head because a supersede operation appends its successor `granted` event | Authorization-chain and database-time evidence are authority; the predecessor is inactive, while a valid successor `granted` head is immediately active. Campaign state derives from that current effective head and the remaining authoritative facts. A bounded request-settlement transaction later terminalizes each affected unbound request and releases its reservation under the §19 authorization → budget → campaign → reservation → request order | The old request cannot borrow the successor and committed bindings are unaffected. Supersession continues operational authorization through the successor; `awaiting_operational_authorization` derives only when no active effective head remains |
| `authorized` → `budgeted` | Active budget-ledger head has positive reservable `V` | Budget service appends a ledger entry | Stale ledger version conflicts |
| `budgeted` → `reserving` | Active authorization, unpaused campaign, sufficient units | Reservation/request transaction locks budget and campaign | Retry uses request identity |
| `reserving` → `ready` | Reservation and request commit successfully | Request service persists held reservation and ready request | Rollback leaves neither; lost response reloads request |
| `ready` → `dispatching` | Exact accepting grant is still the active head, active budget head, valid reservation, no pause/cancellation, lease available | Dispatcher CAS-acquires request lease and appends attempt acquisition | Lease loser skips; changed expected version conflicts; attempt is audit-only |
| `dispatching` → `active` | Accepted Phase 5 handoff and complete binding set | Handoff transaction commits bindings, request `bound`, reservation `committed` | Lost response reloads binding |
| `dispatching` → `ready` | Owning dispatch service proves no binding/downstream commit and lease expired | Dispatch recovery transition clears lease, appends attempt outcome/reservation audit as applicable | Reconciler only records the observation and requests this transition; never reset on uncertainty |
| any nonterminal → `paused` | Campaign-control actor; expected control version | Append `campaign_operations_control_event(pause)`; blocks new reservation, lease acquisition, and handoff | Does not pause lifecycle work, scheduler claims, or workers |
| `paused` → deterministically derived eligible state | Resume actor; expected control version | Append `campaign_operations_control_event(resume)`; derived state is recomputed from current grant, budget, reservations, requests, bindings, cancellations, and unresolved observations after ignoring the pause event | No remembered “prior state”; invalid if terminal, cancelling, or inconsistent; inactive authority derives `awaiting_operational_authorization`, not active |
| any nonterminal → `cancellation_requested` | Cancellation-capable actor | Append immutable cancellation request | Exact retry converges; settlement remains a separate immutable fact |
| `cancellation_requested` → `cancelling` | Outstanding reservations, requests, or bound lifecycle work exist | Cancellation coordinator records settlement progress | No direct worker mutation |
| cancelling → `terminal_cancelled` | No unresolved reservation/request; all scope cancelled or never dispatched | Completion service appends completion event | Blocked by uncertainty/inconsistency |
| `active` → `terminal_completed` | All members terminal completed and all ledgers settled | Completion transaction | Exact replay returns completion |
| active/cancelling → `terminal_failed` | Terminal failed or mixed failed scope, fully settled | Completion transaction with exact summary | Scientific interpretation remains separate |
| any nonterminal → `reconciliation_required` | Uncertain commit, expired lease with ambiguous evidence, leaked reservation, or incomplete binding | Observation event and derived state | No creative repair |
| any → `inconsistent` | Contradictory authoritative identities or impossible cardinality | Observation records exact contradiction | Only validated owning-service correction or successor evidence can clear |
| `reconciliation_required` → derived valid state | Exact observation plus an accepted owning-service transition proves consistency | Owning service performs its normal transition and records the resolution edge | Reconciler cannot mutate or self-resolve |
| `inconsistent` → derived valid state | An owning service's accepted correction, authorization successor grant, budget supersession, or separately authorized exact repair has committed and a resolution event references it | Owning service/repair authority records the append-only successor/correction and resolution | Routine reconciliation can never clear inconsistency |

Unlisted transitions return a stable invalid-transition result and produce no mutation other than an optional rejected-attempt audit event.

`drafted`, if used by an implementation at all, is only a pre-commit in-memory
construction concept. It is never persisted and never participates in durable state
derivation, restart recovery, reconciliation, completion, or database authority. No
readiness table, generalized control event, or mutable campaign status column is
introduced.

Authorization revocation, expiry, or successor grant, plus budget revocation, pause,
and cancellation, affect all
unbound projections (`budgeted`, `ready`, `dispatching`, and derived active summaries),
not only `authorized`. The handoff transaction's lock/recheck decides the race. If the
handoff commits first, immutable bindings and committed units stand; if the control or
grant change commits first, handoff is denied and the unbound request is settled by its
owning service.

An immutable completion event is a decision about obligations and lifecycle evidence
at its recorded evidence point. Ordinary lifecycle authority may later retry a failed
experiment or requeue inference/analysis. That later action neither deletes nor reopens
Campaign Operations completion. Read models must display the original completion as
`completed_at_recorded_evidence` together with current lifecycle state and a
`post_completion_lifecycle_changed` observation; they must not claim that an experiment
remains terminal merely because completion was once recorded.

# 14. Budget Model

The initial budget unit is one exact materialization-member dispatch unit.

Justification:

- Phase 4D already provides an authoritative immutable member count.
- Each member links one exact Phase 4C proposal.
- Phase 5 launch maps each member to at most one conversion execution, activation, and experiment.
- It is deterministic without inventing hardware, duration, monetary-cost, or CPU estimates.
- Scheduler capacity remains independent.

Definitions:

- `G`: `resulting_total` of the unique latest budget-ledger entry; no other table defines it.
- `A`: cumulative units ever reserved.
- `C`: cumulative committed units.
- `L`: cumulative released or expired units.
- `H = A - C - L`: currently held units.
- `M = G - C - H = G - A + L`: arithmetically unallocated units.
- `V = M` only when the latest ledger entry is active; otherwise `V = 0`.
  `V`, not `M`, is reservable availability.

Required invariants:

```text
A = C + L + H
G = C + H + M
C + H ≤ G
G, A, C, L, H, M, V ≥ 0
```

`exhausted` means `V = 0`.

The ledger is the single write and calculation authority:

1. `grant` creates version 1 with `prior_total=0`, positive `delta`, active
   status, and `resulting_total=delta`.
2. `amend` requires an active head, exact prior version, and a nonzero signed
   delta. It remains active and may not yield `G < C + H`.
3. `revoke` requires an active head and sets `resulting_total` exactly to the
   then-current `C + H`, status `revoked`, and therefore reservable `V=0`.
   It does not uncommit or silently release holds.
4. No `amend` may follow a revoked head. A later allowance requires an explicit
   `supersede` entry naming that revoked head, recording a newly authorized total at
   least `C + H`, and returning the chain to active status. This is observable and
   cannot silently resurrect the revoked grant.
5. Every ledger mutation and reservation acquisition locks the same budget account,
   recomputes `C`, `H`, `G`, `M`, and `V` from authoritative rows, validates the
   equations, and then appends exactly one next-version entry or reservation.
6. Every held-reservation release, expiry, or other settlement locks that same budget
   account before campaign/reservation/request, recomputes the same equations, and
   commits its reservation event/state transition atomically.

Rules:

- Only a budget administrator may grant, amend, revoke, or explicitly supersede budget.
- Grant/amend/revoke/supersede entries are append-only and monotonically versioned.
- A negative amendment may not make `G < C + H`.
- Reservation acquisition and request acceptance are one atomic transaction; neither
  exists alone after commit.
- Consumption occurs when a complete durable downstream binding commits.
- Scheduler admission, scheduler claim, worker start, and experiment completion do not consume additional Campaign Operations units.
- Pre-dispatch cancellation releases held units.
- Permanent pre-handoff rejection releases held units through an explicit settlement.
- Once a downstream binding commits, units remain committed even if the experiment later fails or is cancelled.
- Failure or rollback before binding leaves the reservation held for retry or explicit release.
- An uncertain handoff result keeps units held until binding reconciliation.
- Expired reservations may be reclaimed only after proving no binding or downstream commit exists.
- PostgreSQL row locks/version checks on the campaign budget serialize every held
  release/expiry/settlement against budget grant, amendment, revocation, budget
  supersession, reservation acquisition, handoff commitment, and completion.
- Budget availability never grants operational authorization or scheduler capacity.
- One reservation funds exactly one request because request `reservation_id` and the
  reservation logical-operation canonical are unique. One request cannot consume a
  second reservation, and scope/count/unit checks reject incompatible pairing.
- A terminal unbound request (`permanently_failed` or `cancelled`) must settle its held
  reservation to `released` in the same transaction. A bound request must have a
  `committed` reservation. Completion rejects every other combination.

General CPU time, accelerator time, cloud cost, or profitability budgets are deferred.

# 15. Reservation Model

A reservation binds:

- operational campaign;
- active authorization identity;
- exact budget-ledger head/version;
- exact operational-request canonical identity;
- exact materialization/member scope;
- member-unit amount;
- semantic expiry;
- reservation canonical identity.

Acquisition:

1. Validate the pure request and reservation identities.
2. Lock authorization, budget, and campaign domains in global order.
3. Recheck authorization, pause/cancellation state, exact active budget-ledger head, and available units.
4. Insert the held reservation, its acquisition event, and the `ready` accepted
   operational request in one transaction.
5. Append same-transaction audit evidence.
6. Commit both or neither.

Rules:

- One reservation may be consumed by at most one accepted operational request.
- One accepted request must have exactly one reservation.
- The reservation amount must equal the exact member scope for the V1 action.
- Reservations are not scheduler capacity reservations.
- A held reservation may transition only to committed, released, expired, or reconciliation-required.
- `committed`, `released`, and `expired` are terminal. Release after commitment,
  commitment after release/expiry, expiry after commitment, or any amount mutation is
  rejected by guarded expected-state update, event uniqueness, and database checks.
- Commitment requires the complete immutable binding set.
- Cancellation before dispatch releases it.
- Cancellation release uses budget → campaign → reservation → request, preceded by
  authorization only when the exact workflow requires it.
- Cancellation after binding does not refund it.
- Expiration uses PostgreSQL time and cannot race past a dispatcher without locking the request/reservation.
- An active dispatch lease blocks automatic expiration.
- Reservation expiry is optional and canonicalized as `none`; when present it uses
  PostgreSQL time and is an exclusive upper bound. Automatic expiry also requires the
  budget, campaign, reservation, and request locks, no active lease, no binding, a
  nonterminal unbound request, and deterministic downstream evidence that no Phase 5
  mutation committed.
- Ambiguous downstream evidence blocks release.
- Restart recovery finds held reservations by bounded status/expiry queries.
- Leakage detection requires a missing or terminal request plus proof that no binding exists.
- No missing PID, scheduler absence, or process observation can release a reservation.
- Concurrent acquisition serializes on the budget account. The winner commits only if
  `amount <= V`; every loser reloads and either observes the identical logical request
  or receives `insufficient_member_dispatch_units`. No retry may increment `A` twice.

# 16. Operational Request and Handoff Model

The initial intended action is:

> Ensure the complete exact Phase 4D materialization has ordinary Phase 4C execution and activation evidence and becomes bound to its ordinary experiment lifecycle through the accepted Phase 5 launch workflow.

V1 scope is the complete materialization, not an arbitrary subset. Member-level bindings are still recorded. Per-member dispatch would require a later accepted workflow because current Phase 5 campaign launch is all-or-nothing.

The logical-operation key is derived once from campaign canonical identity, fixed action
`dispatch_full_materialization`, action contract version, and exact materialization
canonical identity. Request actor, reason, accepting grant, budget entry, and timestamps
remain immutable audit/provenance payload but do not change that key. The database
therefore admits one and only one V1 logical request for the materialization action.
Exact replay compares the entire stored request; a different actor, reason, grant, or
payload under the same logical key conflicts instead of creating another request.

An operational request is not:

- a scheduler claim;
- a worker attempt;
- an experiment result;
- scientific success;
- evidence that a process started.

Protocol choice: **Option B — durable request/outbox before handoff.**

Reason:

- it permits explicit cancellation before dispatch;
- it preserves durable operator intent across restart;
- it separates request acceptance from dispatcher availability;
- it permits bounded competing dispatchers;
- it gives reservations meaningful held state;
- it makes lost-response and lease recovery explicit;
- the later handoff can still be atomic because the Phase 5 primitive accepts a caller-owned PostgreSQL transaction.

Flow:

1. Reservation acquisition event, `held` reservation, and accepted `ready` request
   commit together.
2. A bounded selector locks authorization → budget → campaign → reservation →
   request, rechecks dispatchability, CAS-acquires a request lease, and inserts an
   immutable attempt-acquisition row. The request/lease remains selection authority.
3. The handoff transaction reacquires and **holds levels 1–5 in that order**,
   validates the lease and exact accepting grant through commit, and then enters the
   existing Phase 4C/5 levels 6–8.
4. It invokes `LaunchRecommendationCampaignInTransaction`; this historical Phase 5
   “launch” establishes ordinary `pending/train` lifecycle state but launches no worker.
5. It inserts one binding and one permanent control-owner row per exact ordered
   materialization member. A deferred check rejects an incomplete or extra set.
6. It transitions reservation `held→committed` and request
   `dispatching→bound`, inserts their events, and commits the attempt outcome and audit
   references in the same PostgreSQL transaction as the Phase 5 mutation.
7. Only after commit may the ordinary scheduler observe resulting `pending`
   experiments; scheduler admission, claim, worker execution, lifecycle completion, and
   scientific interpretation remain separate facts.

Results:

- `created_and_bound`: Phase 5 created execution/activation evidence.
- `adopted_existing_pending_and_bound`: exact valid Phase 5 evidence already existed,
  all experiments remained exactly `pending/train`, the grant explicitly authorized
  `adopt_existing_pending_and_control`, and no other control owner existed. Adoption
  consumes the full member-unit reservation.
- `existing_identical`: binding already exists for the request.
- `rejected`: an authoritative prerequisite failed before downstream mutation.
- `conflict`: changed payload, partial direct operation, competing control owner, or progressed unbound experiment.
- `reconciliation_required`: commit or causality cannot be proven.

If the handoff commit response is lost, lookup by logical-operation/request canonical
identity returns the complete binding set. If experiments have since progressed beyond
pending, the bindings remain sufficient; Phase 5 is not called again.

If no binding exists and experiments have already progressed, Campaign Operations must not infer that its request created or authorized them.

## 16.1 Crash-window and restart matrix

V1 deliberately uses the existing caller-owned Phase 5 transaction, so there is no
committed state in which Phase 5 created an experiment for this handoff but the matching
Campaign Operations binding failed to commit. The durable boundary nevertheless spans
the request-acceptance transaction, a later lease-acquisition transaction, and the
atomic handoff transaction:

| Observed after restart/retry | Authoritative conclusion and permitted action |
|---|---|
| No request | Acceptance did not commit. Retry the same logical operation; reservation and request again commit together. |
| `ready` request, held reservation, no attempt | Accepted but never selected. It remains eligible only after current grant/budget/control revalidation. |
| `dispatching`, unexpired lease, no binding | Another dispatcher may own selection; skip. Attempt rows are audit only. |
| `dispatching`, expired lease, no binding, and exact Phase 4C/5 evidence remains in the pre-handoff state | Record an observation; the dispatch service may append `no_downstream_commit`, clear the lease, and return the request to `ready`. |
| Handoff transaction failed before commit | PostgreSQL rolled back Phase 5 writes, bindings, reservation commitment, request transition, outcome, and audit together. Retry after authoritative reload. |
| Complete bindings exist, reservation/request projection is stale | The handoff committed or authoritative events prove it. Owning reservation/request services restore only their projections and record a reconciliation resolution; do not reinvoke Phase 5. |
| Complete binding and bound request already exist | Return `existing_identical` regardless of later lifecycle progress. |
| Exact `pending/train` Phase 5 artifacts exist but no binding | They are pre-existing direct-work evidence, not proof of this request. Attach only inside the normal handoff transaction with explicit adoption/control authority and a free control-owner key; otherwise conflict. |
| Any unbound artifact progressed beyond `pending/train`, partial Phase 5 evidence, incomplete binding set, or causality mismatch | Record `progressed_unbound_evidence`, `partial_downstream_evidence`, or `causality_ambiguous`; keep the reservation held/reconciliation-required and fail closed. Never fabricate an experiment or binding. |
| Attempt outcome exists but request/binding evidence disagrees | Attempt is not authority. Record contradiction and block completion until owning-service evidence resolves it. |
| Permanent authoritative prerequisite failure before any downstream mutation | Owning dispatch/request service records `permanently_failed`, releases the reservation in the same transaction, and records the attempt outcome. |

A future handoff that cannot join the downstream service's PostgreSQL transaction would
require a new accepted ADR and a downstream API keyed by the same logical-operation
canonical. Its recovery would query that API's authoritative result before retry. V1
must not simulate such an API with process presence, logs, or direct SQL.

# 17. Scheduler Interaction Contract

ADR-0004 remains unchanged.

The scheduler may:

- poll ordinary experiment lifecycle state;
- choose eligible phases;
- allocate scheduler capacity;
- acquire durable claims;
- launch and supervise workers;
- recover attempts;
- record lifecycle completion.

The scheduler must not:

- query recommendation-governance or Campaign Operations tables;
- interpret campaign policy;
- grant operational authorization;
- allocate Campaign Operations budget;
- reconstruct materialization membership;
- become the Campaign Operations state machine.

Campaign Operations may:

- use an accepted lifecycle workflow to create or activate ordinary experiments;
- retain exact bindings to those experiments;
- observe lifecycle and scheduler-attempt evidence read-only.

Campaign Operations must not:

- reserve scheduler slots;
- assign priorities or runnable phases;
- acquire scheduler claims;
- launch workers;
- mark experiments running or complete.

Current gap:

- pending experiments are selected with an ordinary read;
- the running transition lacks a `WHERE status='pending' AND phase=expected` compare-and-set;
- no durable scheduler-attempt identity was found;
- cancellation can race with the unconditional running update;
- multiple schedulers can select and launch the same experiment.

Therefore:

- Campaign Operations Phases A–D require no scheduler code, schema, privilege, polling,
  capacity, or behavior change.
- Phase E may be implemented only behind a default-off feature gate or in an isolated
  test environment. Its presence must not make a request dispatchable in production.
- Production dispatch enablement requires the separate scheduler-hardening ADR to be
  Accepted, the hardening implemented, multi-connection claim/cancellation tests
  passing, and an independent verification recording that the deployed claim is atomic.
- A singleton scheduler convention reduces duplicate-scheduler risk but does not fully resolve cancellation-versus-running-transition races.
- A new scheduler work class is not required: accepted handoff should yield ordinary experiment lifecycle state.
- Any future new work class remains a separate ADR.

Campaign Operations never creates scheduler attempts, chooses runnable phases, marks
experiments runnable outside the accepted Phase 4C/5 lifecycle, claims capacity,
launches or supervises processes, or treats PID/process presence as durable truth. The
scheduler never reads Campaign Operations policy tables. The scheduler-hardening track
belongs to scheduler/lifecycle ownership and is not hidden inside Campaign Operations.

# 18. Transaction Boundaries

Every mutation uses a short explicit libpqxx write transaction. The service owns the
transaction and commits only after all authoritative writes and audit references have
succeeded. Repository methods neither hide commits nor classify arbitrary SQL errors as
success. Default isolation may be `READ COMMITTED` because the named rows/semantic keys
are locked and revalidated; completion uses `REPEATABLE READ` plus the same mutation
locks to freeze its multi-table evidence snapshot. PostgreSQL `transaction_timestamp()`
is the only validity/expiry clock inside a mutation.

Common retry contract: validation/authorization failures do not retry; expected-version
and changed-payload conflicts return stable conflict; unique conflicts reload and
compare complete canonical text; `40001`/`40P01` retry the whole transaction a bounded
three times with the identical canonical request and jitter; connection/commit outcome
uncertainty is resolved by canonical lookup before any retry. A crash before commit
leaves no transaction writes; a crash after commit is an ordinary lost response.

Projection rule for every row below: the immutable campaign row is the initial-state
input for creation; thereafter the authoritative event and guarded state version are
the projection input. If a synchronous current projection is enabled, its narrow
writer updates the affected key and source high-water mark in the same transaction. If
it is disabled or asynchronously rebuilt, the mutation writes no projection and status
derives from authority; projection lag/failure can never roll back, authorize, or
reinterpret the domain result. The transaction descriptions therefore name
authoritative writes and treat projection maintenance as optional derived work.

| Workflow and transaction boundary | Authoritative reads, locks in §19 order, and under-lock validation | Same-transaction writes, uniqueness, and commit point | Retry and crash behavior |
|---|---|---|---|
| **Operational-campaign creation** — one transaction | Load the complete exact Phase 4D manifest and ordered members through its validator. Take level-3 materialization semantic lock; re-read and validate exact canonical/hash/count/provenance, fixed V1 origin, supported contracts, member cardinality/order, scope identity, and every creation invariant before any commit. | Insert the immutable campaign and its creation/audit reference. The campaign row is the authoritative creation fact and immediately derives `awaiting_operational_authorization`; no separate readiness/control write exists. Rely on unique materialization FK and campaign canonical. Commit after both. | Exact canonical conflict returns existing; same materialization/changed scope conflicts. Crash before commit leaves neither row and no durable draft; lost commit response reloads by materialization and deterministically derives `awaiting_operational_authorization`. |
| **Authorization grant/revoke/expiry observation or supersede** — one transaction per transition | Take level-1 campaign/action/scope authorization key, then level-3 campaign row. Load exact prerequisite governance evidence and current chain head. Validate expected head/version, one of the three persisted effective event kinds, actor capability, fixed role, `[not_before, expires_at)`, action/scope, and adoption capability when requested. A supersede operation additionally binds the exact prior head and successor grant payload. | Append exactly one authorization event and audit reference. Supersede writes one `granted` successor, names the prior head, and increments the chain version once; it writes no `superseded` row. Unique chain version/prior edge prevents forks and competing successors. No request/budget/lifecycle write. Commit after event/audit. | Exact supersede retry returns the same successor grant. Stale prior head/version, changed payload, or competing successor conflicts deterministically; other exact transition retries return their event. Revocation/handoff race is decided because handoff holds the same level-1 key through its commit. Affected unbound requests settle later in bounded owning-service transactions. |
| **Budget grant/amend/revoke/supersede** — one transaction per ledger entry | Take level-2 campaign budget semantic key, then level-3 campaign row. Lock current ledger head and relevant reservation rows in ascending reservation ID when totals must be verified. Recompute `C,H,G,M,V`; validate expected version, entry kind, delta equation, status transition, capability, and `resulting_total>=C+H`. | Append the sole next ledger entry and audit reference. Unique campaign/version and prior edge. Commit after invariant query confirms equations. | Exact retry returns entry; stale head conflicts. Concurrent reservation either commits first and changes `H` or waits and sees the new `V`; neither overcommits. No row is mutated on rollback. |
| **Reservation acquisition and request acceptance** — one indivisible transaction | Construct logical-operation/request/reservation canonicals before the transaction. Lock level 1 active authorization key, level 2 budget, level 3 campaign; revalidate unique latest grant, prerequisite policy, PostgreSQL time, active budget head, pause/cancel/completion absence, exact Phase 4D full scope, unit count, and `amount<=V`. New reservation/request have no pre-existing row locks. | Insert reservation `held`, acquisition event, request `ready`, and audit references. Rely on unique logical operation, campaign/action/version, and reservation consumption. Commit creates all or none; this commit is request acceptance. | Duplicate logical operation reloads exact request or conflicts; concurrent budget loser reports insufficient units. Crash/lost response reloads by logical-operation canonical; `A` cannot increment twice. |
| **Dispatch selection/lease acquisition** — one short transaction for one bounded candidate | Candidate query is advisory and uses stable request-ID order/`SKIP LOCKED`. For each chosen row acquire level 1 authorization, level 2 budget, level 3 campaign, level 4 reservation, level 5 request. Recheck exact accepting grant is active/head, active budget, pause/cancel/completion absence, held reservation/expiry, request `ready` and expected version, no binding, lease availability. | CAS request to `dispatching`, increment version, set opaque lease-token digest/expiry, allocate attempt ordinal, insert immutable attempt-acquisition/audit. Commit before Phase 5 work. | Lock/CAS loser skips. Validation denial leaves request unchanged or invokes a separate deterministic settlement after commit. Crash after commit leaves a recoverable lease; attempt row never substitutes for the request. |
| **Downstream handoff, bindings, and commitment** — one transaction after lease commit | Acquire and hold levels **1–5**: authorization, budget, campaign, reservation, request. Recheck accepting grant/head/time, active budget, controls, exact lease/token/version, held reservation, no cancellation/binding. Then acquire levels 6–8 in sorted Phase 5 order: proposals, conversion executions/activations, experiments. Reload exact materialization and Phase 4C/5 evidence. Existing pending adoption additionally validates explicit adoption/control grant and no control owner. | Invoke transaction-bound Phase 5; insert complete ordered bindings and control owners; deferred check proves exact cardinality; append reservation commitment event and set `committed`; set request `bound`/clear lease; insert attempt outcome and audit. Commit once after all downstream and Campaign Operations writes. | Any pre-commit failure rolls back Phase 5 and Campaign Operations writes together. Lost response reloads binding set and returns existing. Unique control owner or changed evidence conflicts. No stale authorization read can race revocation because level 1 is held through commit. |
| **Pause** — one transaction | Take level-3 campaign row/control semantic key. Validate expected control version, no completion, and latest control is not already paused. | Append pause control event and audit; optional projection update by narrow writer in same transaction. Commit blocks later reservation/selection/handoff rechecks. | Exact retry returns event; stale version conflicts. If handoff already holds campaign lock, pause waits and observes bound work; if pause commits first, handoff denies. No lifecycle/scheduler mutation. |
| **Resume** — one transaction | Take level-1 authorization key, level-2 budget, level-3 campaign/control key and relevant level-4/5 rows in ascending ID when deriving state. Validate expected control version, latest event paused, no completion/cancellation/inconsistency. Recompute eligible state from current facts; do not read remembered prior projection state. | Append resume control event/audit and optional derived projection. It does not mutate reservations, requests, lifecycle, or scheduler. | Exact retry converges; inactive authorization may validly resume to derived `awaiting_operational_authorization`. Stale or terminal target conflicts. |
| **Cancellation request without held-reservation settlement** — one transaction | Take only the required Campaign Operations levels in global order: campaign, then affected reservation/request rows in ascending IDs. This path is limited to campaign-level cancellation or a target already proven unable to release/expire/otherwise settle a held reservation. Validate capability, exact target/control ownership, expected state/version, scope, and no duplicate changed request. Never acquire budget after campaign; if under-lock evidence shows held-reservation settlement is required, abort and restart through the next row. Do not take experiment locks. | Append the immutable cancellation request and audit only. Bound work and committed reservations remain unchanged. The later settlement remains a separate immutable fact. | Exact retry returns the same request; changed payload conflicts. Dispatch race is decided by request state/version. Crash before commit leaves no partial cancellation. |
| **Unbound cancellation request and held-reservation release** — one transaction | Take level-1 authorization first only when the exact cancellation/settlement decision requires current authorization-head revalidation; then always acquire level-2 budget, level-3 campaign, level-4 reservation, and level-5 request in ascending ID order. Validate capability, exact target and expected state/version, held reservation, no binding/downstream commit, no ambiguous lease, and exact budget accounting. | Append the immutable cancellation request, set the request `cancelled`, append the reservation release event and guarded `released` update, append the separate immutable cancellation settlement, and append audit references. Commit all facts together; request and settlement retain distinct identities. | Exact retry returns the same request, release event, and settlement; changed payload conflicts. Dispatch/commit/release races are decided under ordered locks. Ambiguity leaves the hold or records reconciliation-required. Crash before commit leaves no partial cancellation; lost response uses canonical lookup. |
| **Bound downstream cancellation coordination** — separate lifecycle-owned transaction, never nested under Campaign Operations locks | First record the immutable cancellation request through the non-releasing path above, commit, and release every Campaign Operations lock before invoking the accepted lifecycle control service. The lifecycle service locks/rechecks the experiment under its own contract; no Campaign Operations SQL impersonates it. | Lifecycle service owns its transition/audit. A subsequent Campaign Operations transaction appends the separate cancellation settlement referencing exact lifecycle evidence; it never releases/refunds the committed reservation or rewrites lifecycle truth. | Lost lifecycle response is resolved by lifecycle request/experiment evidence. Pending cancellation, scheduler claim, running rejection, or terminal evidence produce explicit settlements. No process signal is sent by Campaign Operations. Retry, crash, ambiguity, and reconciliation behavior are unchanged. |
| **Reservation release/expiry/permanent-failure settlement** — one transaction per request/reservation | When the cause requires exact authorization-head revalidation, take level-1 authorization first; in every case take level-2 budget, level-3 campaign, level-4 reservation, then level-5 request. Load bindings, lease, cancellation, grant/budget status, and any exact observation. Require held state, exact accounting, and deterministic proof of no committed binding/downstream result; expiry also requires database time beyond expiry and no active lease. | Append reservation release/expiry event, guarded terminal reservation update, request `cancelled` or `permanently_failed`, attempt outcome/cancellation settlement as applicable, and audit. Commit all together. | Exact event retry converges. Budget mutation/acquisition and commitment/release races are decided by the shared budget domain plus expected states. Any ambiguity leaves held/reconciliation-required; no release after commitment. Crash/lost-response and reconciliation behavior remain unchanged. |
| **Reconciliation observation** — one read snapshot plus one short insert transaction per bounded item | Select maximum 1000 candidates by stable ID cursor. In a read-only consistent snapshot load exact target, Campaign Operations, Phase 4D/4C/5, lifecycle and scheduler-attempt evidence where available. The insert transaction locks only the target needed to confirm its expected version; it performs no transition. | Append observation and audit reference only, with reason, evidence canonical, recommended owning service/action. Commit detection before any service invocation. | Exact evidence retry returns observation. A changed snapshot creates a new observation. Crash cannot leave a repair without observation because no repair occurs here. |
| **Owning-service repair and reconciliation resolution** — normal named transition, then resolution | After observation commit, call the named owning service. Any transition that releases, expires, or otherwise settles a held reservation takes authorization first only if its exact proof requires it, then levels 2–5: budget, campaign, reservation, request. Other transitions take only their needed levels in global order. Reload exact observation and target and validate evidence still matches. | Owning service performs only an already accepted idempotent transition and its event/audit; resolution row references that committed transition in the same transaction when possible, or in a following exact-reference transaction. | Stale evidence yields no change/new observation. Lost response reloads owning event. Reconciler has no privilege to perform the transition or manufacture resolution. |
| **Operational completion** — one `REPEATABLE READ` transaction | Take level-1 authorization key, level-2 budget, level-3 campaign/completion conflict key, then all level-4 reservations and level-5 requests in ascending IDs. Load exact control, cancellation, unresolved observations, complete bindings/control owners, and authoritative terminal lifecycle evidence. Recompute budget equations and disjoint classification under lock. | Insert one immutable completion event and audit reference; optional projection becomes terminal. Unique campaign FK prevents a second decision. Commit only after every §22 prerequisite remains true. | Exact canonical replay returns completion; different evidence/classification conflicts. Concurrent settlement or cancellation either commits first and is included or waits and observes completion. Crash/lost response reloads by campaign. No administrative override exists. |

No transaction remains open across:

- process launch;
- worker execution;
- scheduler polling;
- operator interaction;
- filesystem commands;
- network calls;
- long-running reconciliation loops.

The in-process call to the existing transaction-bound Phase 5 repository is not an
external network/process call and is intentionally inside the handoff transaction. The
separate lifecycle cancellation call is outside Campaign Operations locks/transactions,
with durable request and settlement evidence on each side.

# 19. Global Lock Order

Every transaction must acquire only the levels it needs, in this order:

1. Operational authorization conflict keys, ordered by campaign ID then action/scope canonical bytes.
2. Budget account/version rows, ordered by campaign ID.
3. Operational campaign rows, ordered by operational-campaign ID.
4. Reservation rows, ordered by reservation ID.
5. Operational-request rows, ordered by request ID.
6. Existing proposal review/execution advisory domains, ordered by proposal ID.
7. Existing activation advisory domains, ordered by conversion-execution ID.
8. Experiment rows, ordered by experiment ID.

Rules:

- Advisory-lock namespaces must be distinct per domain.
- Hash collisions add serialization only.
- No transaction may acquire an earlier level after a later one.
- No transaction acquires the budget domain after the campaign domain. A workflow that
  discovers after locking campaign that it must settle a held reservation aborts and
  restarts from the required authorization key, when applicable, then budget.
- Every cancellation, expiry, permanent-failure, or reconciliation-owned transition
  that can release, expire, or otherwise settle a held reservation takes authorization
  only when required by that exact workflow, then budget → campaign → reservation →
  request. At minimum, unbound cancellation with release always takes budget → campaign
  → reservation → request.
- The shared budget domain serializes every held-reservation release/expiry/settlement
  against budget grant, amendment, revocation, budget supersession, reservation
  acquisition, handoff commitment, and operational completion.
- Dispatch selection and handoff acquire levels 1–5 in this exact order. Handoff holds
  all five through Phase 5 mutation, complete binding/control-owner insertion,
  reservation commitment, request binding, attempt outcome, audit, and commit.
- Bound-work cancellation releases every Campaign Operations lock before invoking the
  separate lifecycle-owned cancellation transaction.
- Direct Phase 4C/5 operations begin at level 6 and therefore remain compatible.
- Scheduler capacity/experiment/attempt locks form a scheduler-owned chain under its
  accepted ADR and never acquire Campaign Operations or Recommendation Governance
  locks. No Campaign Operations transaction acquires scheduler locks, so they are not
  interleaved into levels 1–8.
- Reconciliation records an observation, commits, then invokes an owning repair service; it does not reverse the order.
- Overlapping campaigns sort the union of proposal, execution, and experiment IDs before locking.

Advisory/semantic keys for new rows occupy the level the future row will occupy. For
example, campaign creation's materialization key is level 3 and a new reservation does
not require a later-to-earlier lock inversion. A transaction starting at a later level
may never subsequently discover that it needs an earlier one; it must abort, rebuild
its ordered lock set, and retry from the beginning.

# 20. Concurrency Analysis

| Race | Conflict domain and winner | Loser/retry result | Preserved invariant |
|---|---|---|---|
| Duplicate campaign creation | Materialization semantic key plus unique materialization FK; first insert | Exact replay returns existing; any changed campaign payload conflicts | At most one V1 campaign per exact materialization, independent of provenance |
| Budget grant race | Campaign budget account/head lock | Stale-ledger-version conflict; caller reloads | Single ledger chain and `C + H ≤ G` |
| Reservation race | Campaign budget lock | Insufficient funds or retry after reload | No over-reservation |
| Duplicate request | System-derived logical-operation key plus unique campaign/action/version and reservation | Exact full payload replay returns request; actor/reason/later-grant change conflicts | One V1 logical request and one reservation for complete materialization scope |
| Authorization revocation vs handoff | Shared level-1 authorization key held through handoff commit | Revocation first denies handoff and triggers unbound settlement; handoff first leaves committed binding intact | No stale grant authorizes downstream mutation |
| Budget revoke/amend vs reservation/handoff | Shared level-2 budget lock | Ledger winner fixes `V`; later transaction reloads active head/totals | No negative balance, over-reservation, or revoked-budget resurrection |
| Held-reservation settlement vs budget grant/amend/revoke/supersede, reservation acquisition, handoff commitment, or completion | Shared level-2 budget lock acquired before campaign/reservation/request; later row locks and expected states decide the same-target race | Loser reloads exact accounting and converges, conflicts, or keeps the hold on ambiguity | No released hold is counted concurrently as held/committed, no budget mutation observes a torn settlement, and no lock inversion occurs |
| Cancellation vs dispatch | Request row/version | Cancellation first blocks dispatch; dispatch first commits binding before cancellation settlement | No partial handoff or early release |
| Cancellation vs scheduler claim | Experiment lifecycle compare-and-set | Pending cancellation wins or durable claim wins | Requires scheduler hardening |
| Cancellation vs completion | Terminal lifecycle/control state | Later action records `already_terminal` or settles cancellation | Terminal history is not rewritten |
| Reconciliation vs dispatch | Request lease/version | Reconciler skips active lease; dispatcher loses if reconciliation transitioned state | No creative redispatch |
| Reconciliation vs cancellation | Request/campaign state version | Later transaction reloads cancellation evidence | No lost cancellation |
| Direct Phase 4C/5 vs Campaign Operations | Shared proposal/activation/experiment locks | Campaign request may adopt exact pending evidence only with explicit adoption/control grant and no owner; partial/progressed/unowned-causality evidence conflicts or reconciles | No duplicate execution/activation or implicit control acquisition |
| Overlapping operational campaigns | Downstream control-binding uniqueness | First control owner wins; second conflicts or remains observational | One active controller per experiment |
| Multiple Campaign Operations processes | PostgreSQL locks, constraints, leases | Losers skip, replay, or conflict deterministically | Process-local locks are unnecessary |
| Multiple scheduler processes | Current contract is insufficient | Unsupported until hardening | No claim of safety |
| Hash collision | Shared accelerator bucket | Canonical comparison retains distinct identities | No false equality |
| Lost response | Canonical request/result lookup | Retry returns committed row or safely retries uncommitted work | No duplicate intent |
| Transaction rollback | PostgreSQL transaction | Request remains ready/held or no rows exist, depending boundary | No partial handoff |
| Partial external progress | Scheduler/lifecycle evidence | Campaign Operations observes binding and lifecycle; never recreates based on process absence | Lifecycle remains authoritative |
| Completion vs lifecycle retry/requeue | Separate ownership; completion locks Campaign Operations evidence, lifecycle owns later experiment transition | Completion remains immutable; read model records current lifecycle/post-completion change without reopening campaign | Historical operational decision is not false current lifecycle truth |

## 20.1 Failure classification and retry ownership

| Class | Examples | Durable effect and retry rule |
|---|---|---|
| `validation_rejected` | Malformed canonical, unsupported version, wrong full scope, invalid transition | No authoritative mutation except optional rejected audit. Caller must correct input; automatic retry forbidden. |
| `authorization_denied` | No active exact grant, wrong action, missing adoption capability, expired/revoked grant, or accepting grant no longer head because of its successor | No new reservation/handoff. Existing unbound request is settled by owning service; a later grant never silently revives it. |
| `budget_denied` | Inactive budget head, insufficient `V`, amendment below `C+H` | No spend or partial ledger change. Operator may make an explicit new ledger entry; identical operation retry alone does not help. |
| `idempotent_existing` | Exact canonical request/event/binding/completion already committed | Return authoritative stored result without re-executing downstream work. This is success with replay disposition, not a new mutation. |
| `semantic_conflict` | Same natural/logical key with changed payload, stale expected version, control owner collision, partial/progressed unbound Phase 5 evidence | Fail deterministically and observably; no blind retry. Operator/owning service must resolve exact cause. |
| `transient_database` | Serialization failure `40001`, deadlock `40P01`, retryable connection loss before known commit | Retry whole transaction at most three times with identical canonical request and jitter. Never retry only a suffix of a logical transaction. |
| `commit_outcome_unknown` | Connection lost during/after commit | Perform canonical result lookup first. Existing exact result returns replay; proven absence permits whole-operation retry; ambiguity records reconciliation-required. |
| `downstream_temporarily_unavailable` | Accepted lifecycle service cannot be invoked before mutation | Request remains ready/held or lease is cleared by owning dispatch service. Bounded retry is allowed while grant/budget/control remain valid. |
| `permanent_pre_handoff_failure` | Authoritative prerequisite cannot ever satisfy the immutable V1 request and no downstream mutation committed | Same transaction marks request permanently failed and releases held reservation with events. No later grant/budget resurrects it. |
| `reconciliation_required` | Expired lease with uncertain evidence, missing projection, lost response not resolved by canonical lookup | Keep funds held and block redispatch/completion. Reconciler records; named owning service may transition only from exact evidence. |
| `inconsistent` | Contradictory canonical provenance, impossible cardinality/accounting, conflicting causal evidence | Fail closed and escalate exact evidence. Routine retry/reconciliation cannot clear; accepted owning-service correction or successor evidence plus resolution is required. |
| `lifecycle_terminal_or_control_rejected` | Pending cancellation lost to claim, running cancellation unsupported, already terminal | Preserve lifecycle truth and record cancellation settlement/disposition. Never rewrite or signal a process. |

Stable machine reason codes distinguish these classes. Diagnostics may add detail but
must not change classification or identity. A retry counter, host, PID, exception text,
or backoff time is metadata and never semantic identity.

# 21. Cancellation Semantics

| Timing | Required behavior |
|---|---|
| Before authorization | Append cancellation of the durable campaign in `awaiting_operational_authorization`; no budget or downstream action exists |
| After authorization, before reservation | Block new reservations and dispatch; retain authorization history |
| Before reservation | No budget settlement required |
| After reservation/accepted request but before binding | Under budget→campaign→reservation→request locks, append cancellation request, set request `cancelled`, release held units with a separate reservation event, and append cancellation settlement only after proving no binding/downstream commit |
| After dispatch selection | Cancellation and dispatcher serialize on request state; ambiguous lease goes to reconciliation |
| After dispatch/binding | Do not release committed units; require the immutable control-owner row and submit an ordinary lifecycle cancellation request if separately authorized |
| After experiment creation but paused | Use accepted lifecycle cancellation, never direct Campaign Operations update |
| While pending | Lifecycle cancellation may win before scheduler claim; requires claim compare-and-set hardening |
| After scheduler claim | Record Campaign Operations cancellation request; scheduler/lifecycle owns attempt handling, termination policy, and immutable attempt evidence |
| While running | Request cancellation only through an accepted lifecycle authority. Current ordinary cancellation may reject running work; Campaign Operations records that settlement and waits for terminal lifecycle evidence. It never signals the worker or scheduler process. |
| Concurrent with completion | Authoritative terminal evidence wins; cancellation is recorded as not applied or incorporated into settled classification |
| After terminal completion | Reject as `already_terminal`; never rewrite outcome |

Cancellation:

- is operator-authorized under a distinct capability;
- binds exact expected target identity and state version;
- records actor, role, reason, scope, and causal request;
- is exactly replayable;
- does not delete history;
- cannot release budget before downstream state is settled;
- cannot assume absence of a process means cancellation succeeded;
- cannot transform completed or failed experiments into cancelled experiments.

Every cancellation path that releases, expires, or otherwise settles a held
reservation acquires authorization only when required by that exact proof, then budget
→ campaign → reservation → request. An unbound cancellation with release always begins
at budget; it never locks campaign and later acquires budget. That budget lock
serializes release against budget grant, amendment, revocation, budget supersession,
reservation acquisition, handoff commitment, and completion. A request-only or bound
cancellation that cannot settle a held reservation may omit budget, but it must abort
and restart at budget if contrary under-lock evidence is discovered.

The immutable cancellation request and immutable cancellation settlement are different
facts. A request never contains a field that is filled later. Settlement references the
exact request and either the same-transaction unbound request/reservation events or the
separately committed lifecycle-owned control evidence. When terminal lifecycle evidence
wins a race, disposition is deterministically `already_terminal`; when a running
cancellation is unsupported it is `running_cancellation_not_supported` and the campaign
remains unsettled until terminal lifecycle evidence exists.

Cancelling a campaign means stop future Campaign Operations reservation/selection/
handoff. Cancelling an unbound request is Campaign Operations-owned. Cancelling a bound
but unclaimed `pending` lifecycle artifact and requesting cancellation of running work
are lifecycle-owned. Scheduler claim/attempt handling, worker termination, and terminal
lifecycle evidence remain scheduler/lifecycle-owned. These facts are never collapsed
into one `cancelled` boolean.

For bound work, Campaign Operations commits the immutable cancellation request and
releases all Campaign Operations locks before invoking the separate lifecycle-owned
cancellation service. Its later immutable settlement references the lifecycle evidence
and never releases committed units. This preserves the existing retry, crash,
ambiguity, and reconciliation contract.

Administrative campaign pause blocks new Campaign Operations actions only. It does not pause scheduler claims or workers already running.

# 22. Completion Semantics

Operational completion requires:

- the campaign is not paused as a substitute for closure and no active operational
  authorization remains with an obligation that can still create/dispatch work; the
  grant is exhausted by the one bound operation or has been explicitly revoked,
  has ceased to be head because a supersede operation appended its exact successor
  `granted` event, has expired, or is otherwise closed by its accepted contract;
- every reservation committed, released, or expired with authoritative evidence;
- every accepted request bound, permanently failed, or cancelled;
- no dispatch lease or uncertain commit;
- complete member binding cardinality where handoff occurred;
- terminal ordinary lifecycle evidence for every bound member;
- settled cancellation evidence;
- consistent budget arithmetic;
- no unresolved cancellation obligation;
- no unresolved reconciliation observation that blocks closure;
- no `inconsistent` or `reconciliation_required` member or request.

Classifications:

Classification uses the first matching row below, in order. The predicates are
disjoint; “member” includes never-dispatched scope when stated.

| Precedence | Exact settled evidence | Administrative terminal state | Completion classification |
|---|---|---|---|
| 0 | Any contradiction, incomplete binding, ambiguous attempt/commit, held reservation, nonterminal bound lifecycle, unresolved cancellation, or blocking unresolved observation | Nonterminal | `inconsistent` or `reconciliation_required`; completion forbidden |
| 1 | An unbound request is `permanently_failed` and no member was bound | `terminal_failed` | `operational_request_failed` |
| 2 | At least one bound member failed **and** at least one other scope member completed, cancelled, or was never dispatched | `terminal_failed` | `mixed_terminal_outcomes` (therefore failed-plus-cancelled is never `downstream_failure`) |
| 3 | One or more bound members failed and every other bound member also failed, with no completed/cancelled/never-dispatched scope | `terminal_failed` | `downstream_failure` |
| 4 | No failed member; at least one bound member completed and at least one scope member cancelled or was never dispatched | `terminal_completed` | `terminal_partial_completion` |
| 5 | Every exact scope member was cancelled or never dispatched; none completed or failed | `terminal_cancelled` | `all_scope_cancelled` |
| 6 | Every exact scope member is bound and every bound experiment completed | `terminal_completed` | `all_downstream_completed` |

Completion is recorded by a narrowly authorized completion service after reloading exact evidence. The event binds:

- campaign identity;
- budget summary;
- all reservation/request/binding identities;
- per-member lifecycle terminal state;
- cancellation settlement;
- classification;
- actor/service role and reason.

Committed member units are not refunded at completion. Exact replay returns the original event. A changed terminal payload conflicts.

Completion is append-only and idempotent, with one database-unique decision per
campaign. V1 provides **no administrative override, reopen, supersede, delete, or force
complete** path because no accepted repository authority permits one. Evidence defects
must be resolved by their owning service before completion, never waived.

Later lifecycle retry or requeue does not mutate the completion decision. It is shown as
a post-completion lifecycle change as specified in §13; it cannot retroactively change
the point-in-time completion classification or be misreported as proof that current
lifecycle state remains terminal.

A scientifically unfavorable result can still be operationally complete. A scientifically favorable result does not itself complete operations.

# 23. Reconciliation and Restart Recovery

Reconciliation is bounded, restart-safe, and non-creative.

**Reconciliation detects and records; owning services perform every repair transition.**
The reconciliation role cannot insert governance, grants, budget, reservations,
requests, bindings, lifecycle artifacts, reservation/request transitions, cancellation
settlements, or completion events. After an observation commits, orchestration may call
one named owning service using the observation as expected evidence. That service
independently reloads and validates current truth and records its own transition. A
separate resolution row attributes the observation and the owning transition.

Authoritative inputs:

- campaign and authorization events;
- the single budget ledger;
- reservations and requests;
- dispatch attempts and bindings;
- exact Phase 4D membership;
- Phase 4C execution/activation evidence;
- experiment lifecycle;
- scheduler-attempt evidence when available;
- cancellation and completion events.

Selection:

- require an explicit bounded limit;
- default small batch, hard maximum 1000;
- order by stable numeric ID/cursor;
- restrict to states needing reconciliation, expired leases/reservations, or unsettled cancellation;
- never perform an unbounded repository scan.

Each observation has a deterministic identity binding the reconcile request, target, expected state version, and exact observed evidence. `observed_at` is metadata.

V1 reason codes are fixed and versioned:

- `ready_request_not_dispatched`;
- `dispatch_lease_expired_no_downstream_evidence`;
- `dispatch_outcome_unknown`;
- `binding_projection_missing`;
- `reservation_projection_missing_commit`;
- `held_reservation_terminal_unbound_request`;
- `reservation_expired_no_downstream_evidence`;
- `cancellation_settlement_pending`;
- `terminal_lifecycle_completion_ready`;
- `progressed_unbound_evidence`;
- `partial_downstream_evidence`;
- `binding_cardinality_mismatch`;
- `control_owner_conflict`;
- `budget_accounting_mismatch`;
- `post_completion_lifecycle_changed`;
- `causality_ambiguous`.

Every observation stores the reconcile-run canonical, stable target type/ID/canonical,
expected target state/version, reason code, exact authorization/budget/reservation/
request/binding/control/cancellation identities where applicable, exact Phase 4D and
Phase 4C/5 evidence, lifecycle status/phase and lifecycle evidence version, scheduler
attempt identity when it exists, recommended owning service/action, diagnostic code,
and evidence canonical/hash. Fields are required where applicable; unrelated causal IDs
remain explicitly `none` and are never invented.

Permitted **owning-service** actions after detection:

- restore a request projection from an existing immutable binding;
- commit a held reservation when the complete binding proves the handoff committed;
- release a reservation only when no binding/downstream result exists and the request is terminal or validly expired;
- clear a stale dispatch lease only after proving no committed binding or downstream result;
- settle cancellation from authoritative lifecycle evidence;
- record operational completion when all prerequisites are proven.

Any permitted owning-service action that releases or expires a held reservation uses
the §19 order: authorization only if required by its exact proof, then budget → campaign
→ reservation → request. The reconciler still owns no such locks or mutation; it only
records the observation before requesting the owning service.

The observation itself performs none of these actions. In particular, the reconciler
does not create missing governance evidence, invent authority/budget/reservations,
fabricate an experiment or binding, rewrite lifecycle truth, reinterpret science, or
bypass the owning service. Attaching an existing downstream artifact is allowed only
through the normal handoff/adoption service when deterministic exact pending evidence,
explicit adoption/control authority, and the free control-owner key all validate.

Prohibited behavior:

- create a new operational request;
- recreate downstream work because a process is absent;
- infer completion from PID absence;
- invent missing bindings;
- release an uncertain reservation;
- mark an experiment or campaign complete without terminal lifecycle evidence;
- rewrite malformed evidence;
- scan without bounds.

Contradictory or causally ambiguous evidence enters `inconsistent` or `reconciliation_required` and is escalated to an operator with exact IDs and diagnostics.

Batch selection persists/returns a stable `(last_target_id, run_canonical)` cursor,
processes at most 1000 ordered targets, commits per target or small bounded group, and
can restart from the last committed cursor. Duplicate observations converge by exact
canonical identity. A crash before an observation commit records nothing; a crash after
commit leaves an unresolved observation that is safe to revisit. Historical
observations cease to block only when a resolution row references an accepted owning-
service transition; routine observation never clears `inconsistent`.

# 24. Security and Privilege Model

Capabilities must be separated:

| Capability | Minimum privilege shape |
|---|---|
| Operational authorization | Read exact upstream evidence; payload-column insert into authorization events; sequence usage |
| Budget administration | Read campaign/accounting summaries; payload insert into the single budget ledger; no request or lifecycle mutation |
| Reservation/request | Read authorization/budget/campaign; insert reservation/request/audit; narrowly scoped guarded state updates |
| Dispatch | Read exact upstream and request state; insert attempts/bindings; guarded reservation/request updates; invoke only the accepted lifecycle handoff capability |
| Adoption/control | No standalone writer. The dispatcher may exercise this additional capability only when the active grant contains `adopt_existing_pending_and_control`; it inserts one permanent control owner and cannot transfer one. |
| Cancellation | Insert cancellation requests/settlements; read and lock the budget account before invoking narrowly scoped held-reservation release/settlement transitions; invoke a narrowly scoped lifecycle control boundary where permitted; no budget-ledger mutation, process signal, or direct experiment update |
| Reconciliation | Read authoritative state and insert observations only; no repair, transition, binding, budget, lifecycle, cancellation-settlement, resolution-forging, or completion privilege |
| Completion | Read settled evidence; insert completion/audit events |
| Status | `SELECT` on approved read models or views only |
| Projection writer | Read event high-water marks; guarded rebuild/upsert/delete on the optional projection only; no authoritative mutation and no authority decision |
| Scheduler | Lifecycle claim/attempt privileges only; no recommendation-governance or Campaign Operations policy access |

Requirements:

- no broad `UPDATE`, `DELETE`, or `TRUNCATE` on Campaign Operations history;
- payload-column `INSERT` only where generated IDs/timestamps must be protected;
- sequence `USAGE`, not `UPDATE` or unnecessary `SELECT`;
- trigger/function execution revoked from `PUBLIC` and unrelated runtime roles;
- pinned safe `search_path`;
- security-definer functions only for narrowly scoped mutable lifecycle operations that cannot be safely expressed with column grants;
- explicit expected-state and provenance checks inside every privileged transition;
- append-only audit;
- explicit ACL and NULL-ACL tests.

`campaign_operations_owner` is the NOLOGIN schema/table/function owner. Each capability
is a separate NOLOGIN group role granted to the minimum runtime login; `PUBLIC` is
revoked on schema, tables, sequences, and functions before positive grants. Roles
receive payload-column inserts and sequence `USAGE`, never sequence `UPDATE` or
unnecessary `SELECT`. Transition functions are invoker-rights where possible; any
security-definer function has a pinned `pg_catalog,<exact_schema>,pg_temp` search path,
validates expected state and provenance internally, and has `EXECUTE` revoked from
`PUBLIC` and unrelated roles. Migration down/disable procedures revoke runtime grants
before disabling services or removing objects; authoritative history is not dropped as
a runtime rollback.

The dispatcher must not simply inherit the existing broad `pqxx` experiment
privileges. It receives only a dedicated transactional capability that calls the
accepted Phase 5 primitive for one exact validated materialization and permits the exact
same-transaction binding/settlement writes. It cannot issue arbitrary experiment SQL.
The scheduler role has no `SELECT` on Campaign Operations tables; Campaign Operations
roles have no scheduler-claim/attempt or worker-control privileges. The projection
writer is the only writer of the optional current projection, and no mutation service
may consult that projection for authorization.

# 25. Audit Requirements

Every authoritative action records the following **where applicable to that action**;
unrelated fields are explicit `none` and are never populated with invented causal IDs:

- actor identity;
- fixed capability/role;
- reason;
- operation contract version;
- exact expected canonical identity and hash;
- prior state and version;
- resulting state and version;
- operational campaign ID;
- authorization event ID;
- budget-ledger entry ID/version;
- reservation ID;
- request and dispatch-attempt IDs;
- cancellation or reconciliation cause;
- exact proposal, review, execution, activation, experiment, and scheduler-attempt IDs where applicable;
- outcome and stable diagnostic;
- database timestamp metadata;
- replay disposition: recorded, existing-identical, conflict, rejected, repaired, or no-change.

Audit causality must answer: who acted under which capability; which exact immutable
upstream evidence, grant, budget head, reservation, logical operation, request, attempt,
control/cancellation/observation caused the action; what expected state/version was
checked; what authoritative domain event/transition committed; which exact downstream
identities resulted; and whether the call recorded, replayed, conflicted, was rejected,
or made no change. Observation and repair are attributed separately. Actor-provided
reason text is evidence but never the sole operation identity.

Domain events remain authoritative. A consolidated audit stream is a derived index over them or an append-only same-transaction reference ledger. It must never become a mutable competing current-state table.

# 26. Component Interaction Diagrams

```text
Recommendation Governance
  Phase 4D immutable materialization ───────────────┐
  Phase 6D prerequisite/provenance, if required ──┤
                                                   v
                        validate complete exact Phase 4D scope
                                                   |
                          immutable Operational Campaign row
                         [awaiting_operational_authorization]
                                                   |
                                      Operational Authorization
                                                   |
                                     +-------------+-------------+
                                     |                           |
                                 Budget grant               Pause/cancel
                                     |
                              Reservation + Request
                                     |
                             Durable dispatch outbox
                                     |
                    Accepted Phase 5 transaction-bound handoff
                                     |
                    Phase 4C execution + activation + experiment
                                     |
                             ordinary pending/train
                                     |
                                  Scheduler
                             capacity + durable claim
                                     |
                                   Worker
                                     |
                         Experiment Lifecycle evidence
                                     |
                      Reconciliation + Operational Completion

Scientific outcome assessment reads lifecycle/result evidence separately.
Campaign creation has no durable drafted state or separate readiness event.
```

```text
Request acceptance transaction:
authorization → budget → campaign → reservation → request → audit → COMMIT

Dispatch transaction:
authorization
  → budget
  → campaign
  → reservation
  → request
  → proposal locks
  → activation/execution locks
  → experiment locks
  → Phase 5 handoff
  → member bindings
  → permanent control owners
  → reservation commit
  → request bound
  → attempt outcome
  → audit
  → COMMIT

Authorization through request locks remain held until COMMIT.

Authorization supersession transaction:
authorization → campaign → one successor `granted` event → audit → COMMIT
(the successor names the exact prior head; no `superseded` event row exists)
```

```text
Cancellation:
Campaign Operations immutable cancellation request
          |
          +-- unbound request with held reservation
          |       budget → campaign → reservation → request
          |       → immutable release event → separate immutable settlement → COMMIT
          |
          +-- bound experiment → COMMIT request + release Campaign Operations locks
                                      |
                                  ordinary lifecycle control request
                                      |
                          scheduler/worker ownership remains unchanged
                                      |
                           terminal lifecycle evidence
                                      |
                     separate cancellation settlement/completion
```

# 27. Repository Interface Responsibilities

Conceptual repositories are:

- Operational Campaign Repository:
  - create/find by canonical identity;
  - load and validate the complete exact materialization binding and every creation
    invariant before insert;
  - derive `awaiting_operational_authorization` from the immutable campaign row after
    commit/restart, without a readiness row or mutable status;
  - reject duplicate or malformed campaign rows.

- Authorization Repository:
  - append/find authorization events;
  - load the effective event chain;
  - serialize campaign authorization conflicts;
  - persist only `granted`, `revoked`, and `expiry_observed`; implement supersede by
    appending one `granted` successor to the exact prior head and return that same row
    on exact retry.

- Budget Repository:
  - load and lock effective budget;
  - append the single grant/amend/revoke/supersede ledger chain;
  - calculate held/committed/remaining totals;
  - expose the level-2 serialization primitive used before every held-reservation
    release, expiry, or settlement;
  - reject invariant violations.

- Reservation Repository:
  - insert held reservations;
  - guarded settlement/release/expiry only after the owning service has acquired budget
    → campaign → reservation → request (with authorization first when required);
  - bounded leakage queries.

- Operational Request Repository:
  - insert/find by canonical identity;
  - bounded dispatch candidate selection;
  - guarded lease and state transitions.

- Dispatch/Binding Repository:
  - append attempt acquisition and separate terminal outcome evidence;
  - persist complete request/member bindings;
  - persist permanent V1 control owners and enforce uniqueness;
  - reload lost-response results.

- Cancellation Repository:
  - append immutable cancellation requests and separate settlements;
  - never combine request/settlement identity or acquire a budget lock after campaign;
  - load unsettled cancellations.

- Campaign Control Repository:
  - append versioned pause/resume events;
  - load the exact latest control chain;
  - expose no mutable control authority.

- Reconciliation Repository:
  - append observations;
  - append resolution references only to exact owning-service transitions;
  - select bounded reconciliation candidates.

- Completion Repository:
  - append/find terminal decisions;
  - load exact settled summaries.

- Projection Repository:
  - read-only aggregate status;
  - rebuildable projection maintenance through the narrow projection-writer role.

Adapters to existing repositories must:

- load Phase 4D materializations through their authoritative validator;
- load Phase 6D evidence through its immutable repository;
- invoke the existing Phase 5 transaction-bound launch workflow;
- load lifecycle/result evidence through existing typed repositories;
- never expose raw PostgreSQL rows.

# 28. Service Layer Responsibilities

Services orchestrate use cases without embedding SQL:

- Create Operational Campaign:
  - validate the complete exact Phase 4D materialization and every creation invariant
    before commit;
  - construct identity once;
  - persist or replay the immutable row whose existence derives
    `awaiting_operational_authorization`; never persist a draft/readiness state.

- Record Operational Authorization:
  - reload campaign and upstream evidence;
  - enforce actor capability and separation;
  - append only `granted`, `revoked`, or `expiry_observed` events;
  - implement supersede as exactly one successor `granted` event naming the exact prior
    head, with deterministic retry/conflict behavior and no `superseded` row.

- Administer Budget:
  - validate integer member-unit changes;
  - enforce version and ledger invariants.

- Accept Operational Request:
  - validate grant, campaign controls, scope, and budget;
  - construct reservation/request identities;
  - commit both.

- Settle Held Reservation/Request:
  - take authorization only when required by the exact cause, then budget → campaign →
    reservation → request;
  - release, expire, or permanently fail the held reservation/request atomically with
    their immutable events and any separate cancellation settlement;
  - preserve the established exact-retry, crash, ambiguity, and reconciliation rules.

- Dispatch Request:
  - acquire lease;
  - revalidate the exact accepting grant and all authority while holding levels 1–5;
  - invoke accepted Phase 5 workflow;
  - record complete bindings/permanent control owners and settle reservation atomically.

- Control Campaign:
  - append pause/resume evidence for future actions;
  - record immutable cancellation request and separate settlement;
  - route every unbound cancellation/release through the budget-first
    reservation/request settlement service and never acquire budget after campaign;
  - release all Campaign Operations locks before calling the separate lifecycle-owned
    cancellation service for bound work;
  - coordinate but not perform worker control.

- Reconcile Campaign:
  - select bounded candidates;
  - record observations;
  - request named owning-service transitions without possessing their privileges;
  - record resolution only from exact accepted transition evidence.

- Complete Campaign:
  - validate terminal evidence;
  - classify operational completion;
  - append one terminal event.

- Read Status:
  - use repeatable-read, read-only snapshots;
  - distinguish authoritative events from derived summaries.

The operator-facing boundary authenticates and presents these services but does not define CLI commands in this architecture.

# 29. Test Strategy

Required test layers:

1. Pure contracts:
   - every identity and canonical grammar, including logical operation, dispatch
     acquisition/outcome, control, reservation transition, cancellation request/
     settlement, reconciliation resolution, and completion snapshot exceptions;
   - byte-length framing;
   - locale independence;
   - UTC timestamp normalization;
   - ordered members;
   - exhaustive state-transition matrices, deterministic resume derivation, and every
     invalid transition;
   - creation transitions directly from none to `awaiting_operational_authorization`,
     with `drafted` rejected as durable state;
   - authorization event enum limited to `granted`, `revoked`, and `expiry_observed`,
     with supersede producing one successor grant and one version increment;
   - supersession with an unbound request makes the predecessor inactive, makes the
     valid successor grant immediately active, settles the old request without borrowing,
     and derives `awaiting_operational_authorization` only when no active effective head
     remains;
   - budget arithmetic;
   - completion classification.

2. Golden vectors:
   - exact canonical text and tagged hash for every V1 entity;
   - collision fixtures proving canonical-first equality.
   - explicit `none` expiry and UTC-microsecond validity vectors.

3. Migration and persistence:
   - clean application, previous-version upgrade, repeatability, checksums;
   - campaign/materialization uniqueness, logical-operation/request uniqueness,
     one-reservation/one-request constraints, single budget chain, restrictive FKs,
     checks, indexes, deferred binding completeness, and corruption rejection;
   - campaign row and same-transaction creation/audit evidence commit together and
     establish the initial administrative state;
   - no campaign readiness table/generalized creation control event/mutable status, no
     persisted `drafted`, and no authorization `superseded` event kind or second row;
   - immutable history and generated-column protection.
   - upgrade with existing Phase 4–6 evidence preserved byte-for-byte and a downgrade/
     disable rehearsal that revokes services without deleting authoritative history.

4. ACL:
   - authorization, budget, dispatcher, cancellation, reconciliation, completion, status, and scheduler isolation;
   - no broad update/delete/truncate;
   - sequence and NULL-ACL behavior;
   - revoked function execution and pinned search paths.
   - prohibited direct experiment SQL, reconciliation repair, projection-based
     authorization, scheduler reads of Campaign Operations, and Campaign Operations
     scheduler-claim/process-control access.

5. Replay/conflict:
   - exact replay;
   - changed-payload conflict;
   - lost-response lookup;
   - supersede exact retry returns the same successor grant; stale prior head, changed
     payload, and competing successor conflict without a second event;
   - actor/reason/later-grant changes under one logical-operation key conflict;
   - duplicate request and duplicate dispatch converge;
   - progressed-experiment binding replay and progressed-unbound refusal.

6. Independent-connection concurrency:
   - duplicate campaign and authorization;
   - budget-version and reservation races;
   - duplicate requests;
   - dispatch/cancellation;
   - cancellation/completion;
   - reconciliation/dispatch;
   - direct Phase 4C/5 overlap;
   - grant revocation versus handoff through commit;
   - budget revocation/supersession versus reservation;
   - held-reservation cancellation/release/expiry versus budget grant, amendment,
     revocation, budget supersession, reservation acquisition, handoff commitment, and
     completion, proving budget-first acquisition and no campaign→budget inversion;
   - duplicate adoption/control-owner acquisition;
   - pause/resume versus dispatch, cancellation races, and completion races;
   - hash collisions.

7. Failure injection:
   - before request commit;
   - after request commit;
   - before/after Phase 5 mutation;
   - before binding;
   - commit response lost;
   - rollback and connection loss.
   - after lease acquisition/before handoff, after Phase 5 mutation/before binding
     insertion, after complete binding/before commit, and after commit/before response;
   - prove the atomic V1 handoff cannot commit an experiment without its binding.

8. Recovery:
   - restart immediately after campaign creation derives
     `awaiting_operational_authorization` from the immutable campaign row without a
     draft, readiness event, projection, or mutable status;
   - leaked held reservation;
   - expired lease;
   - uncertain binding;
   - restart with ready, dispatching, cancelling, and completion-ready campaigns;
   - bounded cursor progression.
   - binding reconstruction only from complete deterministic evidence and explicit
     adoption authority; refusal to infer from process/PID absence;
   - reconciliation idempotency, owning-service resolution attribution, restart at
     every durable request/reservation/control/cancellation state, and no creative repair.

9. Scheduler isolation:
   - no Campaign Operations table polling;
   - no capacity mutation;
   - no worker launch from Campaign Operations;
   - claim hardening tests under the scheduler ADR.
   - Phase E default-off/isolated-test-only enforcement and independent production-
     enablement verification.

10. Regression:
    - unchanged Phase 4–6 canonical identities, rows, privileges, and workflow behavior;
    - unchanged direct Phase 4C/5 operations;
   - unchanged scientific outcome interpretation.

11. Audit and completion:
    - causal completeness for each mutation with inapplicable fields explicitly absent;
    - cancellation-request/settlement separation;
    - unbound cancellation atomic request/release/settlement replay and bound-work
      lifecycle invocation only after Campaign Operations locks are released;
    - disjoint completed/failed/cancelled classification truth table;
    - completion rejection for every unsettled prerequisite;
    - exact completion replay, concurrent completion winner, no override, and later
      lifecycle retry/requeue displayed without reopening completion.

Every test group must include both a positive proof and a prohibited-behavior proof.
Canonical golden vectors must be approved before Phase B acceptance; exhaustive
transition matrices must be approved before the first persistence increment that uses
them. No test may launch a real scheduler/worker or mutate production experiment rows.

12. Production safety:
    - disposable non-production schemas/databases;
    - refuse known production database names;
    - no real scheduler or worker launch;
    - inspect production processes before any future controlled integration test.

# 30. Accepted ADR Authority

ADR-0010, Campaign Operations ownership and its bounded relationship to reserved Volume
X, is the first Campaign Operations authority. The accepted records are:

1. ADR-0010: ownership, V1 Phase 4D-only origin/scope, and Volume X boundary.
2. ADR-0011: operational authorization chain, prerequisite evidence, actions, actor
   capabilities, validity, revocation, the three persisted event kinds, supersession as
   one successor `granted` event, and explicit adoption authority.
3. ADR-0012: single integer member-unit budget ledger and reservation accounting.
4. ADR-0013: logical-operation/request uniqueness, durable dispatch outbox, atomic
   Phase 5 handoff, bindings, and permanent V1 control ownership.
5. ADR-0014: campaign creation directly into durable
   `awaiting_operational_authorization`, campaign control lifecycle, and operational
   completion.
6. ADR-0015: cancellation request/settlement, budget-first held-reservation settlement,
   reconciliation detection-only boundary, restart recovery, and resolution attribution.
7. ADR-0016: scheduler atomic-claim hardening and the absolute interaction boundary.
8. ADR-0017: least-privilege roles, narrow Phase 5/lifecycle capabilities, projections,
   audit, and migration privilege rollout.
9. ADR-0018: scheduler generation-52 exact-attempt authority and cutover evidence.
10. ADR-0019: durable production admission, exact generation-52 evidence consumption,
    one common Phase E engine, bounded run-once Manager, and H4 exclusion.

There is no blanket “all future ADRs before any implementation” gate. Each
increment below names its exact accepted prerequisites. A proposed later ADR
never governs its increment, but an unrelated future ADR does not block
already governed work. ADR-0009 is separately Accepted as governance-only
upstream evidence. ADR-0016/ADR-0018 implementation and verification, followed
by the separate ADR-0019 implementation/readiness/enablement gates, govern
**production dispatch only**, not documentation, pure domain work, or
non-scheduler persistence.

# 31. Recommended Implementation Phases

Each increment preserves existing Phase 4–6, lifecycle, scheduler, and production data
behavior at its commit boundary.

## 31.1 Phase A — Authority and documentation closure (complete)

- **Gate:** Satisfied on 2026-07-24: ADR-0009 and ADR-0010 through
  ADR-0017 are Accepted and the affected architecture documents are aligned.
- **Scope/contracts:** freeze the terminology, ownership table, Phase 4D-only V1 origin,
  and accepted status of this architecture.
- **Persistence/services/CLI:** none.
- **Tests:** link/status/authority consistency checks and an independent verification of
  all focused-CEE corrections plus the three final-verification amendments in §36.
- **Migration effect:** none.
- **Exclusions:** no code, schema, role, command, dispatch, campaign activation, or
  scheduler change.
- **Acceptance criteria:** Satisfied: ADR/index/Volumes VIII/XII/Phase 6D
  wording agree, the Campaign Operations ADR set is Accepted, and no current
  contract treats Phase 6D ratification as operational authority.
- **Rollback/recovery:** documentation corrections are normal reviewed changes; an
  accepted ADR is superseded, not silently reverted.

## 31.2 Phase B — Pure contracts and golden vectors

- **Gate:** ADR-0010 plus each governing ADR-0011–0015 accepted before implementing its
  corresponding contract slice. No schema depends on unfinished pure contracts.
- **Scope/contracts:** all V1 canonical identities, logical-operation uniqueness,
  authorization/budget/reservation/request/control/cancellation/reconciliation/
  completion transition rules, failure reason codes, and disjoint classification.
- **Persistence/services/CLI:** none; pure C++20 only.
- **Tests:** golden canonical/hash vectors for every durable concept, hash collision,
  `none` expiry, UTC precision, exhaustive state matrices, budget equations, invalid
  transitions, and prohibited authority substitutions.
- **Migration effect:** none.
- **Exclusions:** no database adapter, Phase 5 call, scheduler/process access, or future
  partial-member/adaptive-budget contract.
- **Acceptance criteria:** warnings-free tests; vectors independently reviewed and
  frozen; every enum/transition has stable positive and negative cases.
- **Rollback/recovery:** pure code may be reverted before persistence; after schema uses
  a contract version, incompatible change requires V2 rather than rewriting V1.

## 31.3 Phase C — Campaign and authorization evidence

- **Gate:** Accepted ADR-0010, ADR-0011, and authorization/audit portions of ADR-0017.
- **Scope/contracts:** one campaign per Phase 4D materialization, optional governance
  provenance events, direct durable creation into `awaiting_operational_authorization`,
  serialized authorization chain, three persisted authorization event kinds, one-event
  supersession, and effective-grant algorithm.
- **Persistence:** additive campaign, provenance, authorization, and audit-reference
  tables/roles; no experiment or scheduler table change.
- **Services/CLI/operator:** create/grant/revoke/supersede services and read-only show;
  mutation commands, if exposed, require exact IDs/expected versions and explicit actor/
  reason. They cannot accept requests or dispatch.
- **Tests:** migration clean/upgrade/repeatability, exact replay/conflict, chain forks,
  supersede exact retry/stale prior/competing successor/no second event, validity/expiry,
  campaign uniqueness, no durable draft/readiness/status authority, restart-derived
  initial state, restrictive FKs, corruption, concurrency, ACL, and unchanged Phase 4–6
  sentinels.
- **Migration effect:** one additive ledgered migration after current HEAD; feature
  remains disabled.
- **Exclusions:** budget, reservation, operational request, lifecycle, and scheduler.
- **Acceptance criteria:** at most one campaign per materialization; deterministic active
  grant from the one-event model; campaign row deterministically derives the initial
  state after restart; runtime immutability/least privilege proven; no operational side
  effect.
- **Rollback/recovery:** disable services and revoke grants first; preserve append-only
  rows. Schema removal, if ever approved before production data, is a separate next
  migration, never a rewrite of an applied migration.

## 31.4 Phase D — Budget, reservations, and request acceptance

- **Gate:** Accepted ADR-0012, request-acceptance portion of ADR-0013, and ADR-0017, in
  addition to Phase C acceptance.
- **Scope/contracts:** single ledger; grant/amend/revoke/supersede; exact accounting;
  reservation acquisition/events; one full-scope logical operation and accepted request.
- **Persistence:** additive budget ledger, reservation/event, and request tables,
  constraints/indexes/transition privileges. No dispatch/binding table is enabled.
- **Services/CLI/operator:** budget administration and request acceptance/status; request
  acceptance commits held reservation atomically but no component consumes the outbox.
- **Tests:** equations, negative amendment, revoked/superseded head, concurrent
  reservations, duplicate requests, changed-payload conflict, rollback/lost response,
  budget-first expiry/terminal settlement and prohibited campaign→budget inversion,
  ACL, and migration upgrade.
- **Migration effect:** additive, default-off request dispatch flag; existing schemas and
  rows untouched.
- **Exclusions:** Phase 5 invocation, experiment creation/activation, scheduler changes,
  adaptive/monetary budgets, partial-member requests.
- **Acceptance criteria:** no double spend/negative balance; one reservation per request;
  one logical request per materialization action; restart reloads ready requests without
  executing them.
- **Rollback/recovery:** disable acceptance and revoke writers; settle or retain existing
  held reservations explicitly. Never delete or fabricate settlements to downgrade.

## 31.5 Phase E — Durable dispatch and atomic lifecycle handoff

- **Gate:** Phase D accepted; ADR-0013 and dispatch/security portions of ADR-0017
  Accepted. Test-only work does not require scheduler code change, but production
  enablement additionally requires the ADR-0016 gate in §31.8.
- **Scope/contracts:** lease/attempt acquisition and separate outcome, levels 1–8,
  caller-owned Phase 5 transaction, complete bindings, explicit pending-work adoption,
  permanent V1 control owner, atomic reservation commitment/request binding.
- **Persistence:** additive attempt/outcome, binding, and control-owner tables plus
  deferred binding completeness. No scheduler table change.
- **Services/CLI/operator:** bounded dispatcher and explicit single-request test command;
  both are default-off and restricted to isolated test environments. Phase 5 “launch”
  remains lifecycle handoff, never worker launch.
- **Tests:** duplicate selection/dispatch, direct Phase 5 overlap, adoption grant/control
  conflict, revocation race, all crash windows, lost commit response, progressed-unbound
  refusal, restart recovery, Phase 5 regression, and no process/scheduler action.
- **Migration effect:** additive objects/roles; production feature flag and writer grant
  remain disabled.
- **Exclusions:** automatic polling in production, worker launch/supervision, scheduler
  capacity/claim, cancellation, reconciliation, scientific interpretation.
- **Acceptance criteria:** exact atomic binding/commit; no direct experiment SQL; no
  committed Phase 5 artifact without complete binding; dispatcher cannot run in
  production until §31.8 passes.
- **Rollback/recovery:** disable dispatcher/revoke its role; ready/held work remains
  durable for later explicit settlement. Committed bindings/units are never reversed.

## 31.6 Phase F — Controls, cancellation, and reconciliation

- **Gate:** Accepted ADR-0014, ADR-0015, and applicable ADR-0017 roles; Phase E storage
  accepted.
- **Scope/contracts:** pause/resume chain, cancellation request/settlement, permanent
  control ownership, budget-first held-reservation cancellation/settlement, bounded
  observations, owning-service resolution, restart cursors.
- **Persistence:** additive control, cancellation, observation, and resolution tables;
  narrow projection writer optional.
- **Services/CLI/operator:** explicit pause/resume/cancel/status/reconcile-observe
  operations. Lifecycle cancellation is called only through its accepted service.
- **Tests:** pause/resume/dispatch races, cancellation at every boundary, running refusal,
  every §20 held-reservation settlement race and prohibited campaign→budget inversion,
  bound-work lifecycle invocation only after Campaign Operations lock release,
  scheduler-claim race under test seam, observation idempotency, no creative repair,
  resolution attribution, bounded restart, privileges, and audit completeness.
- **Migration effect:** additive; reconcile and projection jobs default disabled.
- **Exclusions:** process signals, worker termination policy, lifecycle SQL, campaign
  completion, autonomous repair or selection.
- **Acceptance criteria:** controls block only future Campaign Operations actions;
  cancellation never refunds committed units or bypasses lifecycle; every held release
  is budget-first and request/settlement remain separate; reconciler can only
  detect/record.
- **Rollback/recovery:** stop jobs/revoke roles; events remain replayable. Unsettled
  cancellations/observations continue to block completion rather than being erased.

## 31.7 Phase G — Operational completion and audit views

- **Gate:** Accepted ADR-0014, cross-reviewed ADR-0015 cancellation semantics, and
  completion/audit portions of ADR-0017; Phase F accepted.
- **Scope/contracts:** completion prerequisites, disjoint precedence, no override,
  immutable point-in-time decision, post-completion lifecycle display.
- **Persistence:** completion event and optional audit/status projection/view only.
- **Services/CLI/operator:** explicit complete-if-settled service and read-only status;
  no force-complete command.
- **Tests:** each blocking prerequisite, every classification combination, concurrent
  completion/settlement/cancellation, exact replay/conflict, later retry/requeue display,
  privileges, and full causal audit.
- **Migration effect:** additive; no existing lifecycle result is changed.
- **Exclusions:** scientific success/profitability/statistical policy, lifecycle terminal
  transition, budget refund, administrative override.
- **Acceptance criteria:** completion is possible iff every §22 condition is proven;
  one immutable event; status never equates operational and scientific success.
- **Rollback/recovery:** disable completion writer and retain events; projections can be
  rebuilt. No completed event is deleted or reopened.

## 31.8 Phase H — Separate scheduler hardening and production enablement

- **Authority:** ADR-0019 and
  `docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md`
  amend the earlier Phase H wording in this section. Scheduler hardening remains
  scheduler-owned under ADR-0016/ADR-0018; durable production admission and the
  Manager are Campaign Operations-owned.
- **H1 — authority/persistence:** canonical contracts and golden vectors; migration
  055; immutable alternating enable/disable evidence; one first request admission;
  additive Attempt V1/V2; owner-DML guards; production roles; repeatable-read
  readiness/status. No production handoff or Manager batch.
- **H2 — exact production dispatch:** one common Phase E engine shared with the
  isolated-test adapter; enable/disable services; caller-keyed exact request canary;
  fresh-connection uncertain-commit recovery; disable/acquisition/handoff races. No
  Manager batch.
- **H3 — bounded run-once:** optimistic read-only selection, deterministic
  version-derived per-request keys, sequential bounded processing, request-local
  continuation, global-failure stop, and multi-manager correctness. No daemon,
  polling, autostart, or supervision.
- **H4 — continuous Manager:** excluded until a separate operational ADR or explicit
  later acceptance owns supervision, restart/autostart, cadence/backoff, shutdown,
  health, logging, duplicate-instance/churn, deployment, and rollback.
- **Scheduler contract:** exactly generation 52, exactly `cutover_state=complete`,
  exact versioned protocol canonical and independent verification reference through
  a narrow locked evidence function. Liveness is diagnostic; future generations do
  not inherit approval. Neither side receives the other's broad privileges.
- **Acceptance/rollout:** H1–H3 are inert without an enable event and production
  login grants. Deploy disabled, run safe-window scheduler/global-control
  regressions, audit a dedicated Manager login, run readiness, enable explicitly,
  dispatch one exact canary, inspect the complete evidence chain, then run a
  one-request Manager batch and increase only through reviewed bounds.
- **Exclusions:** all §8 boundaries plus physical archival/retention, autonomous
  research, adaptive budgets, partial dispatch, scheduler work classes/capacity,
  worker/process control, scientific interpretation, automatic request acceptance
  or completion, and initial continuous operation.
- **Rollback/recovery:** record disable first, stop Manager, revoke roles, and retain
  all evidence. Disable neither cancels experiments nor mutates scheduler/lifecycle
  state; old leases expire and use existing Phase F recovery.

# 32. Risks

Blocking production-dispatch risks:

- Phase H runtime and migration 055 are not implemented; no enable event or
  production Manager login may be treated as present.
- Process-level scheduler/global-control regressions were deferred while
  production workers were active and remain a safe-window pre-enable gate.

Implementation risks:

- canonical payload size and repeated provenance;
- complex cross-domain lock ordering;
- projection drift;
- incorrect attribution of pre-existing downstream work;
- accidentally coupling budget to scheduler capacity;
- existing broad `pqxx` experiment privileges.
- accidentally implementing an attempt table as dispatch authority or a projection as
  an authorization source.

Migration risks:

- large canonical indexes or copied provenance;
- FK validation cost;
- legacy rows lacking authoritative mapping;
- privilege rollout breaking existing runtime paths;
- retention growth for append-only evidence.

Operational risks:

- leaked reservations after uncertain commits;
- long-running cancellation with claimed/running experiments;
- clock and expiry misunderstandings;
- overlapping campaigns seeking control of one experiment;
- operator misreading operational completion as scientific success;
- prematurely enabling dispatch against the current scheduler.
- operators confusing an active budget, governance ratification, accepted request,
  pending experiment, scheduler claim, lifecycle completion, and scientific outcome.

# 33. Resolved Ambiguities and Deferred Extensions

No implementation-blocking V1 architectural ambiguity remains. ADR-0011,
ADR-0014, ADR-0016, and ADR-0017 resolve the five external questions recorded
by the candidate review:

1. Database/security administration explicitly assigns separate NOLOGIN
   capabilities to authenticated service principals. No additional human
   inequality is inferred beyond Phase 6 reviewer/ratifier separation.
2. The dispatcher uses an invoker-rights application adapter and the existing
   caller-owned Phase 5 transaction primitive under a dedicated narrow role;
   no security-definer business workflow or broad `pqxx` privilege is used.
3. ADR-0016 fixes scheduler-owned atomic conditional claim plus durable attempt
   identity. Its implementation/verification is a production-dispatch gate,
   not coding discretion for Campaign Operations.
4. Campaign Operations gains no power over running processes. If accepted
   lifecycle cancellation refuses claimed/running work, it records that
   settlement and waits for terminal lifecycle evidence.
5. Completion produces non-destructive logical archival in read models.
   Physical retention/deletion remains a future ADR; until then deletion is
   prohibited and the absence of a retention period does not block V1.

All listed objects are owned by `campaign_operations_owner`. In addition to each row's
write rule, the authorization, budget, reservation/request, dispatch, cancellation,
reconciliation, and completion services receive `SELECT` only on the exact upstream and
Campaign Operations tables named by their §18 workflow. The audit/completion services
may read all authoritative Campaign Operations event/state tables; reconciliation may
read them and accepted upstream/lifecycle evidence but may insert only observations.
The public status role reads approved views, not base tables. The projection writer
reads event high-water marks and writes only the projection. The scheduler and workers
receive no Campaign Operations table privilege. These grants are positive allowlists;
everything else is revoked as specified in §24.

A workflow that converts a Phase 6 follow-up into a new exact materialization, partial-
member dispatch, adaptive budgeting, forecasting, autonomous selection, profitability,
scientific policy, physical retention, running-worker stop authority, and new scheduler
work classes are excluded future capabilities, not unresolved V1 decisions.

# 34. Acceptance Preconditions

## 34.1 Baseline acceptance

The Campaign Operations Phase A baseline is Accepted. The focused review and
targeted amendments in §36 resolved the 17 recorded findings. ADR-0009 and
ADR-0010 through ADR-0019 are Accepted. ADR-0019 and the normative Phase H
document further amend only the production-admission/Manager wording identified
in §31.8 and §37; the ADR index and Volumes X/XI/XII are aligned.

## 34.2 Implementation acceptance

Implementation is authorized increment by increment only by the exact Accepted ADR
gates and acceptance criteria in §31. Before each increment closes, its canonical golden
vectors/state matrices, schema/ACL/concurrency/failure/recovery tests, isolated migration
upgrade, and Phase 4–6/lifecycle/scheduler regressions must pass. No proposed governing
ADR may be used as authority. ADR-0016 acceptance, implementation, and independent
verification are required only before production dispatch. Phase H additionally
requires its implemented H1/H2 gates, exact readiness, role audit, and explicit enable
event; until then Phase E remains disabled or isolated-test-only.

## 34.3 System-wide acceptance invariants

At every increment boundary: PostgreSQL remains durable authority; one campaign and one
logical operation bind the exact Phase 4D materialization; operational grant, budget,
reservation, accepted request, dispatch, binding, scheduler admission/claim, worker
execution, lifecycle completion, operational completion, and scientific interpretation
remain non-interchangeable; accepted Phase 4C/5 and lifecycle services are never
bypassed; scheduler isolation is absolute; and production data/processes remain
untouched by tests.

Baseline acceptance additionally fixes that campaign creation directly establishes
`awaiting_operational_authorization`, authorization supersession persists exactly one
successor `granted` event, and every held-reservation settlement follows the §19
budget-first order without changing the established retry, crash, ambiguity, or
reconciliation contracts.

# 35. Overall Readiness Assessment

This architecture is a complete, internally consistent Accepted specification
under ADR-0010 through ADR-0019. It authorizes bounded Campaign Operations
implementation increment by increment under §31; it does not authorize
deployment, production data mutation, role membership, or scheduler operation.

The fixed upstream architecture supplies strong foundations: immutable materialization membership, exact canonical provenance, reusable Phase 5 transaction primitives, PostgreSQL transactions, and read-only lifecycle projections.

The executable V1 scope question is closed: only one exact Phase 4D
materialization is an origin, and Phase 6D is optional
prerequisite/provenance. The architectural authority gates are satisfied for
Campaign Operations Phases 1–5 / architectural A–G implementation. Scheduler
generation-52 hardening is implemented and independently reviewed; the
deferred process-level suites remain a safe-window rollout gate. Phase H
runtime, migration 055, explicit enablement, and production role assignment
are still required before production dispatch.

Campaign Operations is the correct next bounded concern. It refines the operational parts of Volume X, does not supersede completed Phases 4–6, and must not be labeled Phase 6E.

# 36. Final Deliverable Verification

The focused-CEE correction trace is normative amendment evidence:

| # | Focused finding | Resolution in this document |
|---|---|---|
| 1 | Campaign origin | §§1, 5, 10, 12, and 35 freeze V1 to exact Phase 4D materialization; ratification is prerequisite/provenance only. |
| 2 | Campaign uniqueness | §§11–12 require one campaign per materialization independent of provenance. |
| 3 | Request idempotency | §§12 and 16 define one system logical-operation key and database campaign/action uniqueness; actor/reason/later grant conflict. |
| 4 | Budget authority | §§11 and 14 define one append-only ledger with no competing snapshot/adjustment authority. |
| 5 | Campaign-control evidence | §§11–13 define append-only versioned pause/resume events and deterministic resume derivation. |
| 6 | Dispatch attempts | §§11–12 and 16 split immutable acquisition/outcome evidence and make request lease/version authoritative. |
| 7 | Cancellation settlement | §§11–12, 18, and 21 separate immutable request from immutable settlement. |
| 8 | Binding/adoption/control | §§10–12 and 16 require explicit adoption authority, consume units, enforce one controller, and make V1 ownership permanent. |
| 9 | Identity/replay | §12 catalogs every replay-significant identity, exact null/time syntax, and point-in-time lifecycle evidence exceptions. |
| 10 | Locks/revocation | §§18–20 and 26 require levels 1–5 in order and held through atomic handoff commit. |
| 11 | Completion | §§13 and 22 define disjoint classification precedence, no override, and later lifecycle retry/requeue behavior. |
| 12 | Reconciliation | §§5, 18, and 23 fix detection-only ownership, reason codes, bounded restart, and owning-service resolution. |
| 13 | Scheduler enablement | §§1, 17, 29, and 31 make A–D scheduler-independent, Phase E disabled/test-only, and production conditional on accepted/implemented/independently verified hardening. |
| 14 | Implementation sequencing | §§30–31 replace blanket gating with bounded per-increment ADR gates, tests, migration effects, exclusions, acceptance, and recovery. |
| 15 | Cancellation settlement lock order | §§18–21 and 26–29 require every held-reservation cancellation/release/expiry/settlement to acquire budget before campaign/reservation/request, serialize against all budget and commitment operations, and release Campaign Operations locks before bound-work lifecycle cancellation. |
| 16 | Authorization supersession event model | §§9–13, 18, and 27–31 restrict persisted kinds to `granted`, `revoked`, and `expiry_observed`; supersede appends one successor `granted` event naming the exact prior head, with one version increment and deterministic replay/conflict. |
| 17 | Durable initial lifecycle authority | §§5, 11, 13, 18, and 26–31 make the immutable campaign row derive `awaiting_operational_authorization` directly; no durable `drafted`, readiness event/table, or mutable status exists. |

Verification of an implementation must additionally prove the transaction, PostgreSQL,
privilege, recovery, cancellation, completion, scheduler-isolation, lifecycle-ownership,
and audit invariants in §§11–29. This architecture amendment changes no production
code, migration, test, project configuration, scheduler behavior, experiment row, or
separate authority record.

## 36.1 Revision history

| Version | Date | Change | Authority status |
|---|---|---|---|
| 1.0-candidate | 2026-07-22 | Incorporated the focused-CEE candidate corrections and froze V1 authority, identity, persistence, concurrency, recovery, privilege, migration, test, and increment contracts. | Candidate later found by final verification to require the three targeted amendments in 1.1; not implementation authority. |
| 1.1-candidate | 2026-07-22 | Narrowly corrected held-reservation cancellation lock order, authorization supersession to one successor `granted` event, and campaign creation directly into durable `awaiting_operational_authorization`; no other architecture was redesigned. | Candidate awaiting focused verification of these three amendments, formal ADR-0009 alignment, and Accepted ADR-0010; not implementation authority. |
| 1.2 | 2026-07-24 | Accepted the verified V1 specification under ADR-0010 through ADR-0017, aligned Phase 6D, resolved institutional/adapter/scheduler/running-work/archival ambiguities, and recorded implementation traceability. | Accepted implementation authority subject to per-increment gates; production dispatch remains gated by implemented and independently verified ADR-0016 hardening. |
| 1.3 | 2026-07-31 | Accepted ADR-0019 and the normative Phase H correction for durable production admission, exact generation-52 evidence, common Phase E engine, bounded run-once Manager, disable-first rollback, and H4 exclusion. | Accepted H1–H3 implementation authority; Phase H runtime remains default-off and unimplemented. |

# 37. Architectural Authority Traceability Matrix

| Campaign Operations feature or boundary | Authoritative owner | Authorizing ADR(s) | Normative specification |
|---|---|---|---|
| Campaign ownership and one-campaign-per-materialization identity | Campaign Operations | ADR-0010 | §§1, 5–7, 11–13 |
| Campaign lifecycle and initial durable state | Campaign Operations, derived from authoritative facts | ADR-0010, ADR-0014 | §§5, 11, 13 |
| Recommendation/proposal ownership | Recommendation Governance | ADR-0003, ADR-0005, ADR-0010 | §§2, 6–8 |
| Immutable Phase 4D materialization and ordered membership | Recommendation Governance | ADR-0005, ADR-0010 | §§1–2, 5, 10–12 |
| Phase 6D governance prerequisite/provenance | Recommendation Governance; consumed read-only by Campaign Operations | ADR-0009, ADR-0010, ADR-0011 | §§1–3, 5, 10–12 |
| Operational authorization as the only permission to act | Campaign Operations authorizer | ADR-0011 | §§5, 10–13, 18–20 |
| Authorization revocation, expiry, and one-event supersession | Campaign Operations authorizer | ADR-0011 | §§10, 12–13, 18–20 |
| Capability assignment and human separation rule | Database/security administration and application authentication | ADR-0011, ADR-0017 | §§10, 24–25 |
| Budget ownership and versioned single ledger | Campaign Operations budget administrator | ADR-0012 | §§11–12, 14, 18–20 |
| Reservation accounting, commitment, release, and expiry | Campaign Operations reservation/request service | ADR-0012, ADR-0015 | §§11–15, 18–23 |
| Request acceptance and durable outbox | Campaign Operations request service | ADR-0011–ADR-0013 | §§12, 15–16, 18–20 |
| Logical-operation uniqueness and exact replay | Campaign Operations plus PostgreSQL constraints | ADR-0012, ADR-0013 | §§11–12, 15–16, 18–20 |
| Dispatch selection and lease authority | Campaign Operations dispatcher; request state/version remains authority | ADR-0013 | §§11–13, 16, 18–20 |
| Experiment creation/activation at handoff | Existing Phase 4C/5 and Experiment Lifecycle services | ADR-0005, ADR-0013 | §§6, 16, 18, 26–28 |
| Narrow invoker-rights Phase 5 transaction adapter | Campaign Operations dispatcher using accepted Phase 5 workflow | ADR-0013, ADR-0017 | §§16, 18, 24, 27–28 |
| Immutable request-to-experiment binding | Campaign Operations | ADR-0013 | §§11–13, 16, 18 |
| Adoption and permanent V1 downstream control ownership | Campaign Operations dispatcher under extra grant | ADR-0011, ADR-0013 | §§10–12, 16, 18–21 |
| Experiment resource ownership after handoff | Experiment Lifecycle; scheduler for execution | ADR-0004, ADR-0010, ADR-0013, ADR-0016 | §§6, 13, 16–17 |
| Scheduler polling, capacity, atomic claims, attempts, and workers | Scheduler | ADR-0004, ADR-0016 | §§6, 13, 17 |
| Production-dispatch global and per-request admission | Campaign Operations; scheduler supplies exact read-only protocol evidence | ADR-0016, ADR-0018, ADR-0019 | §31.8 and normative Phase H architecture §§3–8 |
| Bounded Campaign Manager run-once | Campaign Operations, ending at the existing Phase E handoff | ADR-0013, ADR-0017, ADR-0019 | §31.8 and normative Phase H architecture §§9, 11, 18 |
| Continuous Campaign Manager | No current runtime owner; separately deferred to H4 | ADR-0019 | §31.8 and normative Phase H architecture §18 |
| Campaign pause/resume | Campaign Operations; future orchestration only | ADR-0014 | §§11–13, 18, 21 |
| Scheduler-global pause/resume/cancel controls | Scheduler/global control subsystem | ADR-0004, ADR-0016 | §§6, 8, 17, 21 |
| Campaign cancellation intent and settlement | Campaign Operations cancellation coordinator | ADR-0015 | §§11–13, 18–23 |
| Bound experiment cancellation | Experiment Lifecycle; Campaign Operations coordinates and records | ADR-0015, ADR-0016 | §§18, 21, 23 |
| Running-work refusal/terminal wait | Experiment Lifecycle/Scheduler; no Campaign Operations process power | ADR-0015, ADR-0016 | §§17–18, 21–23 |
| Reconciliation observation | Campaign Operations reconciler, detection only | ADR-0015 | §§11–13, 18, 23 |
| Repair/resolution transition | Exact owning service; reconciler cannot repair | ADR-0015, ADR-0017 | §§18, 23–25, 27–28 |
| Restart recovery and unknown-commit handling | Exact owning service using canonical lookup | ADR-0013, ADR-0015 | §§12, 16, 18, 20, 23 |
| Global lock order and transactional guarantees | Owning services and PostgreSQL | ADR-0011–ADR-0015 | §§18–20 |
| Exact replay, changed-payload conflict, and idempotency | Every owning service/repository | ADR-0011–ADR-0015, ADR-0017 | §§12, 18–20, 25 |
| Operational completion and disjoint classification | Campaign Operations completion service | ADR-0014 | §§13, 18, 22 |
| Scientific outcome interpretation | Recommendation Governance/Phase 5 policy; never Campaign Operations completion | ADR-0003, ADR-0010, ADR-0014 | §§6, 13, 22 |
| Logical campaign archival | Read model derived from immutable completion | ADR-0014 | §§11, 13, 22–25 |
| Physical retention/deletion | Deferred; no runtime owner or permission in V1 | ADR-0014, ADR-0017 | §§11.3, 24, 33 |
| Auditability and causal attribution | Domain owners; audit stream is a derived causal index | ADR-0017 | §§11–12, 18, 24–25 |
| Least-privilege roles and deployment enablement | Database/security administration | ADR-0017 | §§24–25, 27–29 |
| Read-only status and rebuildable projections | Campaign Operations reader/projection writer | ADR-0014, ADR-0017 | §§11, 13, 24–29 |
| Future origins, partial dispatch, adaptive budget, autonomy, retention, or new work class | Deferred to later owning-domain ADR | ADR-0010, ADR-0012, ADR-0014, ADR-0016 | §§4, 14–17, 30–33 |

# 38. Authority Review Conflict Resolution

| Conflict or implied contract | Why it conflicted | Minimum documentation correction | Result |
|---|---|---|---|
| Phase 6D implementation and migration 044 existed while ADR-0009 and Volumes VIII/XII still said `Proposed`. | Under Volume I, implemented evidence cannot silently make a Proposed ADR authoritative. | Accept ADR-0009, align implementation history and affected volumes, and retain the absolute negative operational boundary. | Resolved; Phase 6D is accepted governance-only evidence. |
| The reviewed Campaign Operations specification was complete but explicitly a candidate, while migration 045/Phase 1 implemented part of it and the Phase 2 prompt assumed accepted authority. | Implementation history does not promote candidate text, and Phase 2 correctly stopped at the missing ADR gate. | Accept ADR-0010 through ADR-0017, incorporate the detailed specification through Volume X, and record actual Phase 1 versus later implementation status. | Resolved; future increments have exact accepted gates. |
| Some candidate table descriptions used “unique canonical” shorthand, while migration 045 and its tests prohibit direct B-tree uniqueness on potentially oversized canonical text. | Direct oversized-text uniqueness can fail PostgreSQL index-size limits and contradicted implemented collision-safe persistence. | Preserve canonical semantic identity but specify bounded typed natural uniqueness, non-unique hash indexes, and complete canonical comparison. | Resolved without changing any canonical identity. |
| The stopped Phase 2 implementation prompt described a budget authority payload as carrying an accepting operational-authorization event, while the reviewed budget model permits budget administration before or after authorization. | The prompt was an implementation request that explicitly deferred to missing accepted architecture; treating its suggested payload as authority would couple funding to permission and contradict the candidate's separate gates. | ADR-0012 makes budget entries campaign-scoped under a separate budget capability; ADR-0011 authorization is revalidated only when reserving, accepting, selecting, or handing off work. Audit may cite an authorization cause when applicable but it is not budget authority. | Resolved; budget and permission remain non-interchangeable. |
| “Launch” in the accepted Phase 5 workflow could be confused with scheduler worker launch. | Phase 5 launch means atomic lifecycle handoff to ordinary `pending/train`; scheduler launch is a later, separate authority. | ADR-0013 fixes Phase 5 as lifecycle handoff; ADR-0016 fixes scheduler claims/process execution. | Resolved; no execution-authority overlap. |
| Existing scheduler-global pause/resume/cancel controls could be confused with Campaign Operations pause/resume/cancellation. | Global controls suppress or signal scheduler-managed work; campaign controls govern future orchestration and delegate lifecycle cancellation. | ADR-0014 through ADR-0016 name the separate owners and prohibit reuse as campaign state. | Resolved. |
| Institutional role assignment, narrow Phase 5 privilege mechanism, running-work behavior, and archival were left external to the candidate. | Leaving them to implementation discretion could create privilege or control overlap. | ADR-0015 and ADR-0017 fix explicit administrator role assignment, invoker-rights Phase 5 reuse, no running-process power, logical archival, and no physical deletion. | Resolved for V1; later expansion requires a new ADR. |
| Volume XI originally stated the required atomic scheduler boundary while the target was not yet fully implemented. | This was an implementation gap against ADR-0004, not authority for Campaign Operations to repair or bypass it. | ADR-0016/ADR-0018 specified and migrations 051–052 implemented scheduler-owned hardening; safe-window process regression remains a rollout gate. | Resolved without granting Campaign Operations scheduler authority. |
| Original §31.8 made Phase H persistence/services/CLI scheduler-owned only and treated enablement as configuration/privilege, while the Phase H proposal introduced durable admission and a Campaign Operations Manager. | Implementation would otherwise invent durable and runtime ownership absent an accepted ADR. | ADR-0019 and the normative Phase H document freeze global/per-request admission, identities, operation keys, exact generation-52 evidence, common Phase E engine, run-once Manager, H1–H4 gates, privileges, readiness, and rollout. | Resolved; H1–H3 are implementation-authorized, default-off, and H4 remains excluded. |

The stale branch/development-status prose in `AGENTS.md` is not a Campaign
Operations authority record and was not changed by this documentation-only
closure. It does not override the ADR index, volumes, or accepted
specification.

# 39. Readiness Decision

The Phase 6D acceptance mismatch, every Campaign Operations V1 implied
contract, and every targeted Phase H authority gap identified by the review
chain are now explicit in Accepted ADRs. No H1–H3 identity, replay,
scheduler-generation, lock-order, privilege, or rollout choice is left for an
implementation pass to invent.

Campaign Operations implementation may continue increment by increment under
§31. H1–H3 are independently committable and inert while there is no enable
event and no production LOGIN membership. H4 cannot be inferred from
multi-manager database correctness.

Production dispatch remains prohibited until H1/H2 implementation, exact
readiness, safe-window scheduler/global-control regressions, reviewed role
assignment, and an explicit enable event. Documentation acceptance alone does
not enable production.

**READY FOR CAMPAIGN OPERATIONS IMPLEMENTATION**
