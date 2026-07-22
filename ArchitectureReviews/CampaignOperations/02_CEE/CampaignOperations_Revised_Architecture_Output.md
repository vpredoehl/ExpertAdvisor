# 1. Executive Summary

Campaign Operations is a new, bounded subsystem for durable, long-lived coordination of explicitly authorized recommendation-campaign work. It does not extend Recommendation Governance, replace Experiment Lifecycle, allocate scheduler capacity, supervise workers, or interpret scientific results.

The governing sequence is:

```text
Immutable campaign scope
+ optional Phase 6D governance evidence
+ explicit operational authorization
+ separately granted member-unit budget
→ reservation
→ durable operational request
→ accepted Phase 5/lifecycle handoff
→ ordinary pending experiments
→ scheduler admission and claim
→ worker execution
→ lifecycle evidence
→ reconciliation
→ operational completion
```

The central decisions are:

- Phase 6D ratification is upstream governance evidence, never operational authorization.
- An operational campaign is bound to one exact Phase 4D materialization manifest. Its persisted members remain the sole membership authority.
- Initial budget accounting uses integer materialized-member dispatch units, not speculative CPU time or monetary cost.
- Reservation and operational request commit first; the request is the durable dispatch outbox.
- A later dispatch transaction reuses the existing Phase 5 transaction-bound launch workflow and atomically records downstream bindings and reservation commitment.
- Request-to-experiment bindings remain authoritative after experiments progress beyond `pending`.
- Campaign pause and cancellation control future Campaign Operations actions; they do not directly control workers.
- Operational completion records settled coordination, not scientific success.
- Production dispatch must not be enabled until the scheduler claim race is addressed under a separate accepted ADR.

The first implementation increment must be documentation and ADR work only.

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
- Migration 044, Phase 6D domain/repository/service sources, tests, project membership, and HEAD commit `d78771fe836f18efa2b1119176394221366f8141` prove Phase 6D implementation exists.
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

Formal architectural acceptance is not currently demonstrable, however:

- ADR-0009 says `Status: Proposed`.
- Its decision section is titled “Proposed decision.”
- Its migration section says migration 044 is “uncommitted and undeployed,” despite migration 044 and the implementation being committed at HEAD.
- The ADR index lists ADR-0009 as `Proposed`.
- Volume VIII says “proposed Phase 6D recorded.”
- Volume XII says “proposed Phase 6D recorded.”
- Both affected volumes describe Phase 6D as implemented while their status and revision history still call it proposed.
- `AGENTS.md` also has stale development-status text: it names branch `phase6`, omits Phase 6D from completed work, while the actual branch is `campaign-operations`.

Under Volume I §16 and the ADR status definitions, “Proposed” is not implementation authority. The correct conclusion is therefore:

- Operational implementation status: complete for Phase 6D’s governance-only contract.
- Formal architectural closure: incomplete until project architecture records acceptance and aligns the documentation.

Required documentation-only corrections, after acceptance is confirmed:

1. ADR-0009:
   - change status to `Accepted`;
   - rename “Proposed decision” to “Decision”;
   - replace the stale “uncommitted and undeployed” statement with implemented migration history;
   - add an acceptance/implementation-alignment revision entry;
   - express the negative boundary as “no operational authorization or downstream capability,” without inventing a Phase 6E roadmap item.

2. ADR index:
   - change ADR-0009’s index status to `Accepted`;
   - add a revision-history entry recording acceptance.

3. Volume VIII:
   - change the volume status to aligned through accepted and implemented Phase 6D;
   - replace the revision-history word “proposed”;
   - retain the absolute prohibition against treating ratification as operational authority.

4. Volume XII:
   - make the equivalent status and revision-history corrections;
   - retain migration 044’s append-only and least-privilege boundaries.

5. Phase 6D implementation document:
   - replace prospective wording that “Phase 6E remains” with the neutral statement that any future operational capability requires its own accepted authority.

No Campaign Operations implementation may rely on the current Proposed/Accepted ambiguity. No Phase 6E should be invented merely to continue numbering.

# 4. Architectural Rationale

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
| Operational campaign | A Campaign Operations-owned durable coordination envelope bound to one exact immutable executable materialization scope. |
| Operational authorization | An explicit append-only grant, made by a fixed authorized role, permitting named Campaign Operations actions against one exact operational-campaign identity and scope. |
| Budget grant | A separately authorized append-only allowance of integer member-dispatch units. It is not permission to act. |
| Reservation | A durable hold of budget units for one exact operational-request identity before downstream handoff. |
| Operational request | Campaign Operations-owned durable intent to perform one named, versioned downstream action over an exact campaign/member scope. |
| Dispatch | Campaign Operations’ bounded act of selecting an accepted request and invoking the accepted lifecycle handoff. It is not a scheduler claim. |
| Scheduler admission | The ordinary experiment lifecycle condition that makes work eligible for scheduler consideration, currently `pending` with a runnable phase. |
| Scheduler claim | The scheduler-owned atomic acquisition of one lifecycle attempt before worker launch. |
| Operational attempt | One Campaign Operations dispatch attempt for an existing durable request. It is not a scheduler attempt or worker execution. |
| Cancellation | An explicit request to stop future Campaign Operations activity and, where separately authorized and lifecycle-valid, request ordinary downstream cancellation. |
| Reconciliation | Bounded comparison of authoritative Campaign Operations and lifecycle evidence to detect or repair non-creative state divergence. |
| Operational completion | An append-only decision that all Campaign Operations obligations, reservations, requests, bindings, cancellations, and terminal lifecycle evidence are settled. |
| Scientific outcome | The Phase 5 evidence classification and policy interpretation of experiment results. It is neither operational completion nor execution authority. |

# 6. Ownership Boundaries

| Owner | Authoritative responsibilities | Explicitly does not own |
|---|---|---|
| Recommendation Governance | Planning, review, approval, immutable materialization, Phase 4C proposal authority, Phase 5 advisory outcomes, Phase 6 proposal/review/ratification | Operational authorization, budgets, reservations, scheduler policy, worker control |
| Campaign Operations | Operational campaign identity, operational grants, member-unit budget, reservations, requests, dispatch attempts, bindings, administrative controls, reconciliation, operational completion | Recommendation policy, experiment truth, scheduler capacity, worker computation, scientific interpretation |
| Experiment Lifecycle | Experiment creation through accepted workflows, lifecycle states and legal transitions, terminal experiment evidence | Campaign budget or authorization, scheduler selection, recommendation policy |
| Scheduler | Ordinary eligibility polling, capacity, atomic claims, attempts, launch, supervision, recovery, completion transitions | Campaign policy, campaign membership, operational authorization, budget |
| Workers | One bounded training, inference, or analysis computation; reporting facts | Selecting work, granting authority, creating campaigns, marking campaign completion |
| PostgreSQL | Committed durable truth, constraints, references, uniqueness, locks, immutable events, migration history | Inventing missing policy, provenance, or business meaning |

# 7. Campaign Operations Responsibilities

Campaign Operations owns:

- creating one immutable operational-campaign identity from exact accepted scope;
- validating upstream identities without recomputing governance truth;
- recording operational authorization, revocation, expiration, and supersession;
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

The scheduler must never interpret Campaign Operations or recommendation-governance policy.

# 9. Subsystem Decomposition

| Component | Responsibility | Authority |
|---|---|---|
| Pure domain contracts | Identity, canonicalization, validation, budget arithmetic, state transitions, completion classification, reason codes | Authoritative rules; no durable state |
| Authorization service | Validate exact upstream evidence and append grant/control events | Authoritative authorization events |
| Budget service | Grant/amend member units and enforce ledger invariants | Authoritative budget ledger |
| Reservation service | Acquire, settle, release, or expire holds | Authoritative reservation state and events |
| Operational-request service | Create durable intent and bind one reservation | Authoritative request |
| Dispatch/handoff service | Select ready requests, invoke accepted Phase 5/lifecycle workflow, record binding | Authoritative dispatch result and binding |
| Campaign-control service | Pause, resume, request cancellation, and record control evidence | Authoritative control events |
| Reconciliation service | Record bounded observations and invoke only accepted repair transitions | Observations authoritative as evidence; repairs authoritative through owning services |
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
4. For a ratification-origin campaign, the exact Phase 6D event, review, proposal, canonical identities, and restrictive provenance.
5. A named, versioned downstream action and exact scope.
6. A fixed operational-authorizer role, actor identity, reason, and optional validity interval.

The initial role is conceptually `campaign_operations_authorizer`. It must be a separately granted capability. For a ratification-origin campaign, the operational authorizer must not be the Phase 6 reviewer or ratifier. The role text is fixed by the contract, not supplied arbitrarily.

Ratification is a prerequisite only where the operational-campaign origin policy requires it. It never substitutes for the operational grant or define new executable work.

Authorization events are append-only:

- `granted`;
- `revoked`;
- `superseded`;
- `expired` observation, when an explicit expiry has elapsed.

Rules:

- A grant binds one campaign, action set, scope, actor, role, reason, and validity interval.
- Budget may exist before or after authorization but cannot activate it.
- Scheduler capacity is neither checked nor reserved by authorization.
- Exact retry returns the existing event.
- Reusing the same authorization request identity with changed scope, actor, reason, validity, or upstream canonical evidence conflicts.
- Revocation immediately blocks new reservations, requests, and dispatch selection.
- Revocation does not undo a committed binding or erase pending/running work.
- Existing uncommitted reservations must be released after authoritative revalidation.
- Existing bound work requires explicit cancellation handling.
- Expiration is checked transactionally using PostgreSQL time before reservation and dispatch.
- Supersession names the prior event and never rewrites it.

Without active operational authorization, only read-only inspection, budget administration, safety cancellation of a draft, and reconciliation observation are possible.

A current architectural limitation remains: Phase 6D ratification does not define a new executable materialization. A new follow-up campaign cannot be created from it until a separately accepted workflow produces exact executable scope.

# 11. Data Model

Canonical text is authoritative for every semantic identity. Hashes may accelerate lookup, bucket locking, and diagnostics only.

| Entity | Owner and purpose | Identity, fields, provenance, uniqueness, replay, and retention |
|---|---|---|
| `operational_campaign` | Campaign Operations; immutable coordination envelope | Immutable contract version, exact Phase 4D materialization FK/canonical/hash, origin kind, optional Phase 6D provenance, purpose, and canonical identity. Generated row ID/time are metadata. V1 permits one operational campaign per exact materialization and origin mode. Exact replay returns existing; changed payload conflicts. Retained while any event, request, binding, or downstream row refers to it. |
| `operational_authorization_event` | Authorization service; grant/control history | Immutable event type, campaign FK/canonical, exact upstream evidence, fixed role, actor, reason, action scope, validity, prior/superseded event, canonical identity. Unique request identity and serialized campaign authorization chain. Never updated or deleted. |
| `campaign_budget_grant` / `budget_version` | Budget service; effective member-unit allowance | Immutable campaign FK, version, resulting granted total, administrator, reason, prior version, canonical identity. Unique campaign/version and request identity. A budget adjustment is an append-only version or adjustment event, never an in-place amount change. |
| `campaign_budget_adjustment` | Budget service; positive or negative change | Immutable delta, prior/resulting total, actor, role, reason, causal event, canonical identity. Negative change is rejected if resulting total is below committed plus held units. |
| `campaign_reservation` | Reservation service; durable hold | Immutable campaign, request canonical identity, scope, unit type, amount, budget version, expiry, canonical identity. Mutable only through guarded `state`, `state_version`, and lease/settlement metadata; every transition has append-only audit evidence. Unique reservation identity and unique consuming request. |
| `campaign_operational_request` | Request service; durable intent/outbox | Immutable campaign, authorization grant, reservation, exact materialization/member scope, intended action/version, expected upstream identities, actor/reason, canonical identity. Guarded mutable status, state version, dispatch lease owner/expiry. Unique canonical request and unique reservation consumption. |
| `campaign_dispatch_attempt` | Dispatch service; one operational attempt | Immutable request, attempt ordinal, expected request version, dispatcher identity, outcome, diagnostics, and canonical identity. It is not a worker attempt. Exact retry of an attempt identity returns its event. |
| `campaign_request_binding` | Dispatch service; immutable downstream result | One row per request/materialization member binding exact proposal, review, conversion execution, activation, and experiment IDs plus disposition such as created or reused-pending. Unique request/member. One active control owner per downstream experiment prevents ambiguous cancellation. Canonical binding remains authoritative after lifecycle progress. |
| `campaign_cancellation_event` | Campaign-control service | Immutable target type/ID, expected state version, actor, capability, reason, requested scope, prior state, canonical identity, and later settlement reference. Exact retry converges; changed payload conflicts. |
| `campaign_reconciliation_observation` | Reconciliation service | Immutable bounded run/request identity, target identity, exact observed authoritative evidence fingerprint/canonical, finding, recommended non-creative action, and timestamp metadata. Observations never themselves mutate lifecycle truth. |
| `campaign_completion_event` | Completion service | Immutable campaign identity, terminal operational state/classification, exact member/request/binding/reservation/budget summaries, lifecycle evidence, actor/service role, reason, canonical identity. One terminal decision per campaign; exact replay returns it, changed terminal payload conflicts. |
| Append-only audit event | Cross-cutting audit | References the authoritative domain event or transaction result, actor, role, reason, expected identity, prior/resulting state, causal IDs, replay disposition, and timestamp. It does not duplicate mutable current state. |
| Optional current-state projection | Projection writer | Mutable derived campaign/member/request summary with source-event high-water marks. Rebuildable and never accepted as authority for mutation. |

Foreign keys must be restrictive and prove provenance through exact parent IDs. Where copied canonical evidence is stored for fail-closed validation, triggers or repository reload must compare the complete canonical content, not only hashes.

Runtime deletion is prohibited. Archival or partition retirement requires a separate retention ADR and must preserve referential and audit reconstruction.

# 12. Identity and Deterministic Replay

V1 canonical prefixes should be distinct and versioned:

- `campaign_operations_campaign_v1`
- `campaign_operations_authorization_v1`
- `campaign_operations_budget_v1`
- `campaign_operations_reservation_v1`
- `campaign_operations_request_v1`
- `campaign_operations_cancellation_v1`
- `campaign_operations_dispatch_binding_v1`
- `campaign_operations_reconciliation_observation_v1`
- `campaign_operations_completion_v1`

Authoritative identity inputs are:

| Identity | Required semantic inputs |
|---|---|
| Operational campaign | Exact materialization ID/version/canonical/hash, origin mode, optional exact ratification provenance, fixed operation contract |
| Operational authorization | Campaign canonical, exact upstream evidence, action set/scope, fixed role, actor, reason, validity |
| Budget version | Campaign canonical, prior version, delta/resulting total, unit type, administrator, reason |
| Reservation | Campaign, budget version, request canonical, exact scope, unit amount, expiry |
| Operational request | Campaign, active authorization, action/version, exact materialization/member scope, expected canonical evidence, actor/reason |
| Cancellation | Target canonical, expected state version, scope, actor/capability, reason |
| Dispatch binding | Request canonical and ordered exact downstream proposal/review/execution/activation/experiment identities |
| Reconciliation observation | Reconciliation request identity, target, expected state version, observed authoritative evidence canonical, finding |
| Completion decision | Campaign canonical, terminal classification, exact settled ledger/request/member/lifecycle evidence |

Canonicalization rules:

- compare bytewise under a specified byte-oriented grammar;
- use fixed field order and fixed ASCII field names;
- frame variable UTF-8 text with its byte length;
- reject malformed UTF-8, NUL, DEL, and prohibited controls;
- render integers as locale-independent base-10 without leading `+`;
- sort collections only by their contract-defined bytewise or numeric order;
- retain materialization members in persisted ordinal order;
- render semantic timestamps in normalized UTC with fixed precision;
- exclude database-generated IDs, `created_at`, observation time, PID, process identity, host, transaction ID, and mutable lifecycle state unless the contract explicitly treats a supplied validity/expiry instant as semantic;
- require a contract-version change for grammar or equivalence changes.

Collision handling:

- a hash match only selects candidate rows or a serialization bucket;
- complete canonical text decides equality;
- same hash/different canonical values remain distinct and observable;
- lock-key collisions add serialization only.

Replay:

- exact request and exact stored canonical result: return `existing_identical`;
- same idempotency key with changed payload: deterministic conflict;
- retry after lost response: reload by request canonical identity;
- if a binding exists, return it without invoking Phase 5 again;
- downstream lifecycle progress never invalidates the immutable binding;
- if no binding exists but downstream evidence has progressed beyond the handoff precondition, do not infer causality—enter reconciliation or conflict.

# 13. Lifecycle State Machine

Campaign control state is a derived summary over authoritative events. Its precedence is:

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
> drafted
```

Separate state dimensions must never be collapsed:

| Dimension | States |
|---|---|
| Administrative campaign | `drafted`, `awaiting_operational_authorization`, `authorized`, `budgeted`, `reserving`, `ready`, `dispatching`, `active`, `paused`, `cancellation_requested`, `cancelling`, `terminal_completed`, `terminal_cancelled`, `terminal_failed`, `inconsistent`, `reconciliation_required` |
| Member operational | `not_requested`, `reserved`, `request_ready`, `dispatching`, `bound_pending`, `claimed`, `running`, `terminal_completed`, `terminal_failed`, `terminal_cancelled`, `inconsistent`, `reconciliation_required` |
| Reservation | `held`, `committed`, `released`, `expired`, `reconciliation_required` |
| Request | `recorded`, `ready`, `dispatching`, `bound`, `rejected`, `cancellation_requested`, `cancelled`, `reconciliation_required` |
| Downstream experiment | Existing authoritative `paused`, `pending`, `running`, `completed`, `failed`, `cancelled` with `train`, `infer`, `analyze`, `done` |
| Scheduler attempt | Target contract: `claimed`, `launching`, `running`, `completed`, `failed`, `orphaned`, `recovered`; not yet fully implemented in the repository |
| Scientific result | Existing not-ready, comparable success, context-changed, metric-gap, terminal failure/cancellation, inconsistent, plus separate advisory interpretation |

The following campaign transitions are exhaustive:

| Transition | Authority and preconditions | Transaction owner and durable result | Retry / invalid behavior |
|---|---|---|---|
| none → `drafted` | Campaign creator; exact valid materialization scope | Campaign service inserts immutable campaign and audit | Exact replay returns campaign; changed scope conflicts |
| `drafted` → `awaiting_operational_authorization` | Scope validation complete | Campaign service records readiness/control event | Exact replay no-op; invalid scope fails closed |
| awaiting → `authorized` | Operational authorizer; active exact grant | Authorization transaction appends grant | Exact replay returns event; changed grant conflicts |
| `authorized` → awaiting | Revocation, expiry, or supersession before binding | Authorization transaction appends control event | No history removal; committed work is unaffected |
| `authorized` → `budgeted` | Effective budget total greater than zero | Budget service appends budget version | Stale version conflicts |
| `budgeted` → `reserving` | Active authorization, unpaused campaign, sufficient units | Reservation/request transaction locks budget and campaign | Retry uses request identity |
| `reserving` → `ready` | Reservation and request commit successfully | Request service persists held reservation and ready request | Rollback leaves neither; lost response reloads request |
| `ready` → `dispatching` | Active grant, valid reservation, no cancellation, lease available | Dispatcher CAS-acquires request lease and appends attempt | Lease loser skips; changed expected version conflicts |
| `dispatching` → `active` | Accepted Phase 5 handoff and complete binding set | Handoff transaction commits bindings, request `bound`, reservation `committed` | Lost response reloads binding |
| `dispatching` → `ready` | Transactionally proven no binding/downstream commit and lease expired | Reconciliation/dispatch reset transaction | Never reset on uncertainty |
| any nonterminal → `paused` | Campaign-control actor | Control event; blocks new reservation/dispatch | Does not pause workers |
| `paused` → prior eligible active state | Resume actor; authorization and budget revalidated | Control event and derived projection update | Invalid if terminal or inconsistent |
| any nonterminal → `cancellation_requested` | Cancellation-capable actor | Append cancellation event | Exact retry converges |
| `cancellation_requested` → `cancelling` | Outstanding reservations, requests, or bound lifecycle work exist | Cancellation coordinator records settlement progress | No direct worker mutation |
| cancelling → `terminal_cancelled` | No unresolved reservation/request; all scope cancelled or never dispatched | Completion service appends completion event | Blocked by uncertainty/inconsistency |
| `active` → `terminal_completed` | All members terminal completed and all ledgers settled | Completion transaction | Exact replay returns completion |
| active/cancelling → `terminal_failed` | Terminal failed or mixed failed scope, fully settled | Completion transaction with exact summary | Scientific interpretation remains separate |
| any nonterminal → `reconciliation_required` | Uncertain commit, expired lease with ambiguous evidence, leaked reservation, or incomplete binding | Observation event and derived state | No creative repair |
| any → `inconsistent` | Contradictory authoritative identities or impossible cardinality | Observation records exact contradiction | Only validated repair/supersession can clear |
| reconciliation/inconsistent → valid prior state | Exact observation plus authorized non-creative repair proves consistency | Owning service performs normal transition; reconciler records result | No direct arbitrary state rewrite |

Unlisted transitions return a stable invalid-transition result and produce no mutation other than an optional rejected-attempt audit event.

# 14. Budget Model

The initial budget unit is one exact materialization-member dispatch unit.

Justification:

- Phase 4D already provides an authoritative immutable member count.
- Each member links one exact Phase 4C proposal.
- Phase 5 launch maps each member to at most one conversion execution, activation, and experiment.
- It is deterministic without inventing hardware, duration, monetary-cost, or CPU estimates.
- Scheduler capacity remains independent.

Definitions:

- `G`: current effective granted units.
- `A`: cumulative units ever reserved.
- `C`: cumulative committed units.
- `L`: cumulative released or expired units.
- `H = A - C - L`: currently held units.
- `M = G - C - H = G - A + L`: remaining/available units.

Required invariants:

```text
A = C + L + H
G = C + H + M
C + H ≤ G
G, A, C, L, H, M ≥ 0
```

`exhausted` means `M = 0`.

Rules:

- Only a budget administrator may grant or amend budget.
- Grants and amendments are append-only and versioned.
- A negative amendment may not make `G < C + H`.
- Reservation occurs before request acceptance.
- Consumption occurs when a complete durable downstream binding commits.
- Scheduler admission, scheduler claim, worker start, and experiment completion do not consume additional Campaign Operations units.
- Pre-dispatch cancellation releases held units.
- Permanent pre-handoff rejection releases held units through an explicit settlement.
- Once a downstream binding commits, units remain committed even if the experiment later fails or is cancelled.
- Failure or rollback before binding leaves the reservation held for retry or explicit release.
- An uncertain handoff result keeps units held until binding reconciliation.
- Expired reservations may be reclaimed only after proving no binding or downstream commit exists.
- PostgreSQL row locks/version checks on the campaign budget serialize reservation and amendment races.
- Budget availability never grants operational authorization or scheduler capacity.

General CPU time, accelerator time, cloud cost, or profitability budgets are deferred.

# 15. Reservation Model

A reservation binds:

- operational campaign;
- active authorization identity;
- budget version;
- exact operational-request canonical identity;
- exact materialization/member scope;
- member-unit amount;
- semantic expiry;
- reservation canonical identity.

Acquisition:

1. Validate the pure request and reservation identities.
2. Lock authorization, budget, and campaign domains in global order.
3. Recheck authorization, pause/cancellation state, budget version, and available units.
4. Insert the held reservation and operational request in one transaction.
5. Append audit evidence.
6. Commit both or neither.

Rules:

- One reservation may be consumed by at most one accepted operational request.
- One accepted request must have exactly one reservation.
- The reservation amount must equal the exact member scope for the V1 action.
- Reservations are not scheduler capacity reservations.
- A held reservation may transition only to committed, released, expired, or reconciliation-required.
- Commitment requires the complete immutable binding set.
- Cancellation before dispatch releases it.
- Cancellation after binding does not refund it.
- Expiration uses PostgreSQL time and cannot race past a dispatcher without locking the request/reservation.
- An active dispatch lease blocks automatic expiration.
- Ambiguous downstream evidence blocks release.
- Restart recovery finds held reservations by bounded status/expiry queries.
- Leakage detection requires a missing or terminal request plus proof that no binding exists.
- No missing PID, scheduler absence, or process observation can release a reservation.

# 16. Operational Request and Handoff Model

The initial intended action is:

> Ensure the complete exact Phase 4D materialization has ordinary Phase 4C execution and activation evidence and becomes bound to its ordinary experiment lifecycle through the accepted Phase 5 launch workflow.

V1 scope is the complete materialization, not an arbitrary subset. Member-level bindings are still recorded. Per-member dispatch would require a later accepted workflow because current Phase 5 campaign launch is all-or-nothing.

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

1. Reservation and request commit as `held` and `ready`.
2. A bounded selector acquires one request dispatch lease.
3. The handoff transaction rechecks grant, campaign controls, reservation, request version, and complete materialization evidence.
4. It acquires the existing Phase 4C/5 locks in their accepted order.
5. It invokes the existing transaction-bound Phase 5 launch workflow.
6. It inserts every exact member binding.
7. It commits the reservation, marks the request bound, records dispatch outcome, and appends audit evidence in the same transaction.
8. Only after commit may the ordinary scheduler observe resulting `pending` experiments.

Results:

- `created_and_bound`: Phase 5 created execution/activation evidence.
- `reused_pending_and_bound`: exact valid Phase 5 evidence already existed and all experiments remained pending.
- `existing_identical`: binding already exists for the request.
- `rejected`: an authoritative prerequisite failed before downstream mutation.
- `conflict`: changed payload, partial direct operation, competing control owner, or progressed unbound experiment.
- `reconciliation_required`: commit or causality cannot be proven.

If the handoff commit response is lost, lookup by request canonical identity returns the bindings. If experiments have since progressed beyond pending, the bindings remain sufficient; Phase 5 is not called again.

If no binding exists and experiments have already progressed, Campaign Operations must not infer that its request created or authorized them.

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

- Phases A–D of Campaign Operations require no scheduler behavior.
- The first production-enabled downstream handoff must be preceded by a separate accepted scheduler-hardening ADR and increment.
- A singleton scheduler convention reduces duplicate-scheduler risk but does not fully resolve cancellation-versus-running-transition races.
- A new scheduler work class is not required: accepted handoff should yield ordinary experiment lifecycle state.
- Any future new work class remains a separate ADR.

# 18. Transaction Boundaries

| Transaction | Reads and validation | Writes and locks | Commit, retry, and uncertain recovery |
|---|---|---|---|
| Campaign creation | Exact materialization, ordered members, optional ratification | Semantic campaign lock; insert immutable campaign and audit | Exact canonical replay returns existing |
| Authorization recording | Campaign, upstream governance evidence, current authorization chain | Authorization-chain lock; insert event/audit | Lost response reloads by canonical identity |
| Budget grant/amendment | Campaign, prior budget version, held/committed totals | Budget lock; insert version/adjustment/audit | Stale expected version conflicts |
| Reservation acquisition/request creation | Active grant, campaign controls, budget, request identity | Authorization → budget → campaign locks; insert reservation, request, audit | Both commit or neither; reload by request identity |
| Dispatch selection | Bounded ready requests and expired leases | Request CAS/row lock; insert dispatch attempt or lease metadata | Loser skips; expired lease requires reconciliation |
| Downstream handoff | Grant, campaign, reservation, request, materialization, Phase 4C/5 evidence | Request/reservation locks followed by existing proposal, activation, experiment locks; invoke Phase 5 primitive | Any failure rolls back downstream writes |
| Request binding | Complete Phase 5 result in same handoff transaction | Insert all member bindings; commit reservation; update guarded request state; audit | Complete binding set or rollback |
| Cancellation recording | Exact target and expected state | Campaign/request lock; append cancellation/audit | Exact retry converges |
| Downstream cancellation request | Existing binding and ordinary lifecycle state | Separate lifecycle-owned transaction using accepted transition service | Re-read lifecycle/audit after uncertain response |
| Reconciliation observation | Bounded authoritative snapshot | Insert immutable observation only | Exact observation retry converges |
| Reservation settlement/repair | Reservation, request, binding, exact observation | Reservation/request locks; guarded transition and audit | Never release on ambiguous evidence |
| Operational completion | All events, reservations, requests, bindings, lifecycle terminal evidence, budget totals | Campaign/completion conflict lock; insert completion/audit | Exact replay returns event; changed terminal decision conflicts |

No transaction remains open across:

- process launch;
- worker execution;
- scheduler polling;
- operator interaction;
- filesystem commands;
- network calls;
- long-running reconciliation loops.

# 19. Global Lock Order

Every transaction must acquire only the levels it needs, in this order:

1. Operational authorization conflict keys, ordered by campaign ID.
2. Budget account/version rows, ordered by campaign ID.
3. Operational campaign rows, ordered by operational-campaign ID.
4. Reservation rows, ordered by reservation ID.
5. Operational-request rows, ordered by request ID.
6. Existing proposal review/execution advisory domains, ordered by proposal ID.
7. Existing activation advisory domains, ordered by conversion-execution ID.
8. Scheduler capacity-class rows, if introduced under a separate ADR.
9. Experiment rows, ordered by experiment ID.
10. Scheduler-attempt rows, ordered by attempt ID.

Rules:

- Advisory-lock namespaces must be distinct per domain.
- Hash collisions add serialization only.
- No transaction may acquire an earlier level after a later one.
- Campaign Operations releases levels 1–5 before invoking a separate lifecycle cancellation transaction if the action cannot be composed safely.
- Direct Phase 4C/5 operations begin at level 6 and therefore remain compatible.
- Scheduler claims begin at their scheduler-owned suffix and never acquire Campaign Operations or governance locks.
- Reconciliation records an observation, commits, then invokes an owning repair service; it does not reverse the order.
- Overlapping campaigns sort the union of proposal, execution, and experiment IDs before locking.

# 20. Concurrency Analysis

| Race | Conflict domain and winner | Loser/retry result | Preserved invariant |
|---|---|---|---|
| Duplicate campaign creation | Canonical campaign key; first insert | Exact replay returns existing; changed scope conflicts | One campaign per V1 semantic scope |
| Budget grant race | Campaign budget version lock | Stale-version conflict; caller reloads | Version chain and `C + H ≤ G` |
| Reservation race | Campaign budget lock | Insufficient funds or retry after reload | No over-reservation |
| Duplicate request | Request canonical/unique reservation | Exact replay returns request; changed payload conflicts | One request consumes one reservation |
| Cancellation vs dispatch | Request row/version | Cancellation first blocks dispatch; dispatch first commits binding before cancellation settlement | No partial handoff or early release |
| Cancellation vs scheduler claim | Experiment lifecycle compare-and-set | Pending cancellation wins or durable claim wins | Requires scheduler hardening |
| Cancellation vs completion | Terminal lifecycle/control state | Later action records `already_terminal` or settles cancellation | Terminal history is not rewritten |
| Reconciliation vs dispatch | Request lease/version | Reconciler skips active lease; dispatcher loses if reconciliation transitioned state | No creative redispatch |
| Reconciliation vs cancellation | Request/campaign state version | Later transaction reloads cancellation evidence | No lost cancellation |
| Direct Phase 4C/5 vs Campaign Operations | Shared proposal/activation/experiment locks | Campaign request may reuse exact pending evidence, conflict on partial/progressed evidence, or require reconciliation | No duplicate execution/activation |
| Overlapping operational campaigns | Downstream control-binding uniqueness | First control owner wins; second conflicts or remains observational | One active controller per experiment |
| Multiple Campaign Operations processes | PostgreSQL locks, constraints, leases | Losers skip, replay, or conflict deterministically | Process-local locks are unnecessary |
| Multiple scheduler processes | Current contract is insufficient | Unsupported until hardening | No claim of safety |
| Hash collision | Shared accelerator bucket | Canonical comparison retains distinct identities | No false equality |
| Lost response | Canonical request/result lookup | Retry returns committed row or safely retries uncommitted work | No duplicate intent |
| Transaction rollback | PostgreSQL transaction | Request remains ready/held or no rows exist, depending boundary | No partial handoff |
| Partial external progress | Scheduler/lifecycle evidence | Campaign Operations observes binding and lifecycle; never recreates based on process absence | Lifecycle remains authoritative |

# 21. Cancellation Semantics

| Timing | Required behavior |
|---|---|
| Before authorization | Append cancellation of the draft; no budget or downstream action exists |
| After authorization, before reservation | Block new reservations and dispatch; retain authorization history |
| Before reservation | No budget settlement required |
| After reservation | Cancel request and release held units only after locking request/reservation and proving no binding |
| After request creation | Mark request cancellation-requested/cancelled; dispatcher must recheck before handoff |
| After dispatch selection | Cancellation and dispatcher serialize on request state; ambiguous lease goes to reconciliation |
| After dispatch/binding | Do not release committed units; submit an ordinary lifecycle cancellation request if separately authorized |
| After experiment creation but paused | Use accepted lifecycle cancellation, never direct Campaign Operations update |
| While pending | Lifecycle cancellation may win before scheduler claim; requires claim compare-and-set hardening |
| After scheduler claim | Record campaign cancellation request; scheduler/lifecycle owns attempt handling |
| While running | Do not erase or directly cancel the process. Existing lifecycle may reject cancellation or require explicit stop/pause authority |
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

Administrative campaign pause blocks new Campaign Operations actions only. It does not pause scheduler claims or workers already running.

# 22. Completion Semantics

Operational completion requires:

- every reservation committed, released, or expired with authoritative evidence;
- every request bound, rejected, or cancelled;
- no dispatch lease or uncertain commit;
- complete member binding cardinality where handoff occurred;
- terminal ordinary lifecycle evidence for every bound member;
- settled cancellation evidence;
- consistent budget arithmetic;
- no `inconsistent` or `reconciliation_required` member.

Classifications:

| Evidence | Administrative terminal state | Completion classification |
|---|---|---|
| Every bound experiment completed | `terminal_completed` | `all_downstream_completed` |
| Completed and cancelled members, none failed | `terminal_completed` | `terminal_partial_completion` |
| Every member cancelled or never dispatched | `terminal_cancelled` | `all_scope_cancelled` |
| Completed and failed members | `terminal_failed` | `mixed_terminal_outcomes` |
| Any terminal failed member | `terminal_failed` | `downstream_failure` |
| Permanent request failure before handoff | `terminal_failed` | `operational_request_failed` |
| Held/uncertain reservation or request | Nonterminal | `reconciliation_required` |
| Contradictory evidence | Nonterminal | `inconsistent` |

Completion is recorded by a narrowly authorized completion service after reloading exact evidence. The event binds:

- campaign identity;
- budget summary;
- all reservation/request/binding identities;
- per-member lifecycle terminal state;
- cancellation settlement;
- classification;
- actor/service role and reason.

Committed member units are not refunded at completion. Exact replay returns the original event. A changed terminal payload conflicts.

A scientifically unfavorable result can still be operationally complete. A scientifically favorable result does not itself complete operations.

# 23. Reconciliation and Restart Recovery

Reconciliation is bounded, restart-safe, and non-creative.

Authoritative inputs:

- campaign and authorization events;
- budget versions and adjustments;
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

Permitted repairs:

- restore a request projection from an existing immutable binding;
- commit a held reservation when the complete binding proves the handoff committed;
- release a reservation only when no binding/downstream result exists and the request is terminal or validly expired;
- clear a stale dispatch lease only after proving no committed binding or downstream result;
- settle cancellation from authoritative lifecycle evidence;
- record operational completion when all prerequisites are proven.

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

# 24. Security and Privilege Model

Capabilities must be separated:

| Capability | Minimum privilege shape |
|---|---|
| Operational authorization | Read exact upstream evidence; payload-column insert into authorization events; sequence usage |
| Budget administration | Read campaign/budget summaries; payload insert into budget versions/adjustments; no request or lifecycle mutation |
| Reservation/request | Read authorization/budget/campaign; insert reservation/request/audit; narrowly scoped guarded state updates |
| Dispatch | Read exact upstream and request state; insert attempts/bindings; guarded reservation/request updates; invoke only the accepted lifecycle handoff capability |
| Cancellation | Insert cancellation events; invoke a narrowly scoped lifecycle control boundary where permitted |
| Reconciliation | Read authoritative state and insert observations; repair capability separated and limited to named transitions |
| Completion | Read settled evidence; insert completion/audit events |
| Status | `SELECT` on approved read models or views only |
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

The new dispatcher must not simply inherit the existing broad `pqxx` experiment privileges. A dedicated transactional lifecycle capability is required without duplicating Phase 5 policy in SQL.

# 25. Audit Requirements

Every authoritative action records:

- actor identity;
- fixed capability/role;
- reason;
- operation contract version;
- exact expected canonical identity and hash;
- prior state and version;
- resulting state and version;
- operational campaign ID;
- authorization event ID;
- budget version;
- reservation ID;
- request and dispatch-attempt IDs;
- cancellation or reconciliation cause;
- exact proposal, review, execution, activation, experiment, and scheduler-attempt IDs where applicable;
- outcome and stable diagnostic;
- database timestamp metadata;
- replay disposition: recorded, existing-identical, conflict, rejected, repaired, or no-change.

Domain events remain authoritative. A consolidated audit stream is a derived index over them or an append-only same-transaction reference ledger. It must never become a mutable competing current-state table.

# 26. Component Interaction Diagrams

```text
Recommendation Governance
  Phase 4D immutable materialization ───────────────┐
  Phase 6D ratification, if required ──────────────┤
                                                   v
                                      Operational Campaign
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
```

```text
Request acceptance transaction:
authorization → budget → campaign → reservation → request → audit → COMMIT

Dispatch transaction:
request/reservation
  → proposal locks
  → execution locks
  → activation locks
  → experiment locks
  → Phase 5 handoff
  → member bindings
  → reservation commit
  → request bound
  → audit
  → COMMIT
```

```text
Cancellation:
Campaign Operations cancellation event
          |
          +-- unbound request → release reservation → settle
          |
          +-- bound experiment → ordinary lifecycle control request
                                      |
                          scheduler/worker ownership remains unchanged
                                      |
                           terminal lifecycle evidence
                                      |
                         cancellation settlement/completion
```

# 27. Repository Interface Responsibilities

Conceptual repositories are:

- Operational Campaign Repository:
  - create/find by canonical identity;
  - load exact materialization binding;
  - reject duplicate or malformed campaign rows.

- Authorization Repository:
  - append/find authorization events;
  - load the effective event chain;
  - serialize campaign authorization conflicts.

- Budget Repository:
  - load and lock effective budget;
  - append grants/adjustments;
  - calculate held/committed/remaining totals;
  - reject invariant violations.

- Reservation Repository:
  - insert held reservations;
  - guarded settlement/release/expiry;
  - bounded leakage queries.

- Operational Request Repository:
  - insert/find by canonical identity;
  - bounded dispatch candidate selection;
  - guarded lease and state transitions.

- Dispatch/Binding Repository:
  - append attempts;
  - persist complete request/member bindings;
  - enforce control-owner uniqueness;
  - reload lost-response results.

- Cancellation Repository:
  - append cancellation events;
  - load unsettled cancellations.

- Reconciliation Repository:
  - append observations;
  - select bounded reconciliation candidates.

- Completion Repository:
  - append/find terminal decisions;
  - load exact settled summaries.

- Projection Repository:
  - read-only aggregate status;
  - rebuildable projection maintenance.

Adapters to existing repositories must:

- load Phase 4D materializations through their authoritative validator;
- load Phase 6D evidence through its immutable repository;
- invoke the existing Phase 5 transaction-bound launch workflow;
- load lifecycle/result evidence through existing typed repositories;
- never expose raw PostgreSQL rows.

# 28. Service Layer Responsibilities

Services orchestrate use cases without embedding SQL:

- Create Operational Campaign:
  - validate immutable scope;
  - construct identity once;
  - persist or replay.

- Record Operational Authorization:
  - reload campaign and upstream evidence;
  - enforce actor capability and separation;
  - append the exact event.

- Administer Budget:
  - validate integer member-unit changes;
  - enforce version and ledger invariants.

- Accept Operational Request:
  - validate grant, campaign controls, scope, and budget;
  - construct reservation/request identities;
  - commit both.

- Dispatch Request:
  - acquire lease;
  - revalidate all authority;
  - invoke accepted Phase 5 workflow;
  - record bindings and settle reservation atomically.

- Control Campaign:
  - pause/resume future actions;
  - record cancellation;
  - coordinate but not perform worker control.

- Reconcile Campaign:
  - select bounded candidates;
  - record observations;
  - delegate repairs to owning services.

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
   - every identity and canonical grammar;
   - byte-length framing;
   - locale independence;
   - UTC timestamp normalization;
   - ordered members;
   - state transitions and invalid transitions;
   - budget arithmetic;
   - completion classification.

2. Golden vectors:
   - exact canonical text and tagged hash for every V1 entity;
   - collision fixtures proving canonical-first equality.

3. Migration and persistence:
   - clean application, previous-version upgrade, repeatability, checksums;
   - restrictive FKs, uniqueness, deferred completeness, corruption rejection;
   - immutable history and generated-column protection.

4. ACL:
   - authorization, budget, dispatcher, cancellation, reconciliation, completion, status, and scheduler isolation;
   - no broad update/delete/truncate;
   - sequence and NULL-ACL behavior;
   - revoked function execution and pinned search paths.

5. Replay/conflict:
   - exact replay;
   - changed-payload conflict;
   - lost-response lookup;
   - progressed-experiment binding replay.

6. Independent-connection concurrency:
   - duplicate campaign and authorization;
   - budget-version and reservation races;
   - duplicate requests;
   - dispatch/cancellation;
   - cancellation/completion;
   - reconciliation/dispatch;
   - direct Phase 4C/5 overlap;
   - hash collisions.

7. Failure injection:
   - before request commit;
   - after request commit;
   - before/after Phase 5 mutation;
   - before binding;
   - commit response lost;
   - rollback and connection loss.

8. Recovery:
   - leaked held reservation;
   - expired lease;
   - uncertain binding;
   - restart with ready, dispatching, cancelling, and completion-ready campaigns;
   - bounded cursor progression.

9. Scheduler isolation:
   - no Campaign Operations table polling;
   - no capacity mutation;
   - no worker launch from Campaign Operations;
   - claim hardening tests under the scheduler ADR.

10. Regression:
    - unchanged Phase 4–6 canonical identities, rows, privileges, and workflow behavior;
    - unchanged direct Phase 4C/5 operations;
    - unchanged scientific outcome interpretation.

11. Production safety:
    - disposable non-production schemas/databases;
    - refuse known production database names;
    - no real scheduler or worker launch;
    - inspect production processes before any future controlled integration test.

# 30. ADR Recommendations

Before implementation:

1. Complete ADR-0009 acceptance and documentation alignment.
2. Accept ADR-0010: Campaign Operations ownership and relationship to Volume X. This is the first new ADR that must be accepted.
3. ADR-0011: Operational authorization, upstream evidence, actor roles, revocation, expiry, and supersession.
4. ADR-0012: Member-unit budget and reservation ledger semantics.
5. ADR-0013: Durable operational request, dispatch outbox, Phase 5 handoff, and downstream binding.
6. ADR-0014: Campaign lifecycle and operational completion.
7. ADR-0015: Cancellation, reconciliation, restart recovery, and control ownership.
8. ADR-0016: Scheduler claim hardening and Campaign Operations interaction boundary.
9. ADR-0017: Least-privilege roles, privileged lifecycle transition boundary, and audit model.

ADRs may be combined only if the combined record remains independently reviewable. No runtime implementation may begin under merely proposed ADRs.

# 31. Recommended Implementation Phases

- Phase A — Formal Phase 6 closure and terminology:
  - accept/align ADR-0009;
  - correct affected volumes;
  - accept Campaign Operations ownership ADR;
  - no runtime behavior.

- Phase B — Pure domain:
  - identities, canonical grammars, budget arithmetic, state machine, classifications;
  - no database.

- Phase C — Campaign and authorization persistence:
  - immutable campaign manifest binding;
  - append-only operational authorization;
  - read-only inspection only.

- Phase D — Budget and reservations:
  - append-only budget versions;
  - reservation ledger and invariant tests;
  - no scheduler integration.

- Phase E — Operational request and durable handoff:
  - durable request/outbox;
  - dispatch attempts;
  - accepted Phase 5 workflow adapter;
  - immutable bindings;
  - no direct worker launch.
  - Production enablement is blocked until scheduler claim hardening is accepted and implemented.

- Phase F — Cancellation and reconciliation:
  - campaign controls;
  - bounded observations and non-creative repair;
  - restart recovery.

- Phase G — Operational completion:
  - terminal evidence validation;
  - append-only completion and summaries.

- Phase H — Separate scheduler track:
  - durable compare-and-set claim/attempt hardening under its own ADR;
  - new work-class integration only if later justified.
  - This is not Campaign Operations implementation and must not be hidden inside Phase E.

# 32. Risks

Blocking risks:

- ADR-0009 remains formally proposed.
- Phase 6D does not define a new executable follow-up materialization.
- Current scheduler claim behavior does not satisfy safe cancellation or multi-scheduler races.

Implementation risks:

- canonical payload size and repeated provenance;
- complex cross-domain lock ordering;
- projection drift;
- incorrect attribution of pre-existing downstream work;
- accidentally coupling budget to scheduler capacity;
- existing broad `pqxx` experiment privileges.

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

# 33. Open Architectural Questions

These decisions must not be made implicitly during coding:

1. What accepted workflow converts a ratified Phase 6 follow-up proposal into a new exact executable materialization? Until defined, ratification-origin work cannot create new experiments.
2. Must every operational campaign require Phase 6D ratification, or only campaigns explicitly classified as ratification-origin?
3. What institutional actor grants the operational-authorizer capability, and are additional human separation rules required between authorization and budget administration?
4. What narrowly scoped database capability will let the dispatcher reuse Phase 5 atomically without inheriting broad experiment mutation privileges?
5. What durable scheduler-attempt and capacity contract will satisfy ADR-0004?
6. How should an already-pending direct Phase 5 result be adopted, and who receives control ownership?
7. What operator authority may request stop or pause for a running experiment when ordinary cancellation is invalid?
8. What retention and archival period applies to canonical payloads, attempts, observations, and audit?
9. Are authorization and reservation expirations required in V1, and what default policy applies when no expiry is supplied?
10. Should later versions permit partial-member dispatch? V1 must remain all-members because the existing Phase 5 workflow is all-or-nothing.

# 34. Acceptance Preconditions

Implementation is not authorized until:

- ADR-0009 is formally accepted and its documentation is consistent.
- Campaign Operations ownership and Volume X relationship are accepted.
- Operational authorization and actor-role rules are accepted.
- Exact executable scope is identified without deriving it from scores or ratification.
- Member-unit budget and reservation invariants are accepted.
- The durable request/outbox and Phase 5 handoff protocol are accepted.
- Cancellation, reconciliation, completion, and control ownership are accepted.
- Global lock order is reviewed against direct Phase 4C/5 paths.
- Scheduler claim hardening is accepted before production dispatch.
- Least-privilege roles and transactional lifecycle capability are accepted.
- Canonical golden vectors and state-transition matrices are frozen.
- Test isolation and production-process safety requirements are approved.
- No Proposed/Accepted ambiguity remains in the authority chain.

# 35. Overall Readiness Assessment

The repository is ready for Critical Engineering Evaluation and ADR drafting, but not for Campaign Operations implementation.

The fixed upstream architecture supplies strong foundations: immutable materialization membership, exact canonical provenance, reusable Phase 5 transaction primitives, PostgreSQL transactions, and read-only lifecycle projections.

Three gates remain:

1. formal Phase 6D acceptance/documentation closure;
2. an accepted definition of executable scope for ratification-origin follow-up;
3. scheduler claim hardening before operational dispatch is enabled.

Campaign Operations is the correct next bounded concern. It refines the operational parts of Volume X, does not supersede completed Phases 4–6, and must not be labeled Phase 6E.

# 36. Final Deliverable Verification

- [x] All 36 required sections are present in the required order.
- [x] No code was implemented.
- [x] No files were modified.
- [x] No migrations were created.
- [x] No CLI commands were designed.
- [x] No scheduler integration was implemented.
- [x] Every CEE finding is addressed.
- [x] Repository inspection was targeted rather than an uncontrolled repository-wide listing.
- [x] This response is a complete architecture document, not a tool transcript.
- [x] The recommended first implementation increment is architecture/ADR-only.

Verification appendix:

- Files changed: none.
- Tests/builds run: none; this was an architecture-only, read-only task.
- Database accessed: none.
- Production scheduler, training, inference, or analysis processes accessed: none.
- Current branch: `campaign-operations`.
- HEAD: `d78771fe836f18efa2b1119176394221366f8141`.
- `git status --short`: clean; no output.
- `git diff --stat`: no differences.
- `git diff --check`: passed; no output.