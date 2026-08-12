# Campaign Operations Phase H — Production Dispatch Admission and Manager

Status: Accepted implementation authority
Version: 1.2.1
Date: 2026-08-10
Authority: ADR-0019 as narrowly amended by ADR-0019A and ADR-0019B, and later refined for external H4 supervision by ADR-0020; ADR-0010 through ADR-0018 remain governing as stated below

## 1. Decision and normative language

This document replaces the ambiguous Phase H proposal with the exact contract
authorized for incremental implementation. `MUST`, `MUST NOT`, `REQUIRED`, and
`PROHIBITED` are normative. No implementation pass may choose different
identity fields, operation-key semantics, scheduler compatibility, authority
hierarchy, lock order, privilege allocation, recovery behavior, or rollout
semantics without a later accepted ADR.

Canonical text is authority. Every canonical is ASCII field labels plus values
framed as `<UTF-8-octet-length>:<exact-bytes>` wherever this document says
“framed.” Integers use unsigned base-10 with no leading zero except `0`.
Timestamps use UTC microseconds as `YYYY-MM-DDTHH:MM:SS.ffffffZ`. Hashes are
non-unique lookup/integrity aids and never decide replay. The implementation
MUST compare complete canonical text after every hash/key match.

## 2. Scope, exclusions, and ownership

### 2.1 In scope

- durable global production enable/disable authority;
- immutable global and per-request production evidence;
- exact single-request production dispatch;
- Attempt V2 acquisition and the existing Phase E handoff;
- bounded sequential Campaign Manager run-once processing;
- read-only readiness/status;
- exact scheduler generation-52 protocol evidence;
- canary-first rollout and disable-first rollback.

### 2.2 Excluded

Physical archival, retention/deletion, autonomous research, adaptive budgets,
partial dispatch, scheduler work classes, scheduler capacity or priority
policy, scheduler claims/attempts/leases, worker/process launch or control,
running-worker cancellation, scientific interpretation, automatic request
acceptance, automatic completion, and an in-process continuous Manager
CLI/daemon are out of scope. ADR-0020 separately accepts external,
deployment-owned H4 supervision that repeatedly invokes H3 run-once; it adds
no scheduler polling or database singleton, heartbeat, lease, or leader-
election authority. Disable does not stop or signal work and does not mutate
lifecycle or scheduler state.

### 2.3 Component ownership

| Component/fact | Owner | Boundary |
|---|---|---|
| Enable/disable chain, request admission, Attempt V2, Manager, readiness | Campaign Operations | May invoke only accepted Phase E and read pinned scheduler evidence. |
| Request, lease, binding, reservation settlement | Existing Campaign Operations Phase E | One common internal engine; existing transaction boundaries remain. |
| Expired-lease recovery, cancellation, reconciliation | Existing Campaign Operations Phase F | Phase H never invents an alternate recovery path. |
| Completion and logical archival | Existing Campaign Operations Phase G | Completion V1 remains authoritative. |
| Protocol/cutover row, claims, capacity, attempts, launch, workers | Scheduler | Campaign Operations receives no raw table or mutation access. |
| Experiment lifecycle | Experiment Lifecycle / accepted Phase 5 | Phase H mutation ends at the existing Phase E handoff. |
| Scientific policy/results | Recommendation/scientific owners | Never interpreted by Campaign Operations. |

The scheduler does not poll Campaign Operations and gains no Campaign
Operations privileges. The Manager does not wake, signal, launch, supervise,
or reserve capacity for the scheduler.

## 3. Authoritative production hierarchy

Committed authority is evaluated in this order:

1. no enablement evidence means globally disabled;
2. the current exact effective enable event is global production authority;
3. immutable request admission plus a complete Attempt V2 and production audit
   is per-request production authority;
4. `production_dispatch_enabled` is only a one-way serialization witness.

Global authority is necessary but does not replace the exact ADR-0011
operational authorization, ADR-0012 budget/reservation authority, request
state/version, completion gate, or Phase E prerequisites. Per-request admission
is necessary but does not keep an old enable event effective.

An enable head is effective only when its event is complete/current, the live
§4.2 scheduler canonical is byte-identical, its independent verification
reference is present, the executing service contract is byte-identical to the
approved service contract, and the executing §4.3 build canonical is
byte-identical to the approved build. Capability membership is then a separate
necessary permission to call the mutation; it never makes an ineffective event
effective.

## 4. Frozen canonical identities

### 4.1 General rules

The column order below is the canonical field order. Prefixes and field labels
are literal. `none` is literal only for a nullable numeric/reference selector;
canonical text values are framed, including empty predecessor canonical at
genesis. The `*_contract_version` field is present even though the prefix is
versioned. Fixed capabilities are literal role names, not caller input.
Every identity is the prefix immediately followed by each
`;field_name=value` in listed order, with no whitespace or line break. Line
breaks in code blocks are presentation only. Genesis predecessor canonical is
exactly the empty framing `0:`.

Diagnostic-only fields MUST NOT appear in a logical canonical. They may be
stored in a separate append-only diagnostic/status record and include PID,
hostname, invocation timestamp, session UUID, process instance ID, retry count,
connection identity, recovering principal, and batch identity.

### 4.2 Scheduler protocol evidence canonical

The narrow scheduler-owned interface returns exactly:

```text
campaign_operations_scheduler_protocol_evidence_v1
;required_generation=52
;cutover_state=complete
;migration_contract=61:migration-052-scheduler-generation-52-exact-attempt-authority
;protocol_contract=50:scheduler-generation-52-exact-attempt-authority-v1
;cutover_completed_at=<framed UTC timestamp>
;cutover_completed_by=<framed exact stored value>
;cutover_executable_path=<framed exact stored value>
;cutover_process_evidence=<framed exact stored value>
```

The numeric lengths shown for the two fixed literals are normative and MUST be
golden-vector checked. `required_generation` and `cutover_state` must also be
returned as typed columns. The canonical uses the exact current migration-052
cutover row; scheduler liveness, current PID, current host, and current
invocation are diagnostic and excluded. Migration 055 MUST fail if the
implemented fixed literal lengths do not match their UTF-8 octet lengths.

The independent verification reference is not synthesized by this interface.
It is an exact, operator-supplied immutable artifact identifier that reviewed
this scheduler canonical and the approved deployment/build evidence. It is
framed into the enable event.

### 4.3 Manager build contract canonical

The approved and actual executable build identity is exactly:

```text
campaign_operations_manager_build_v1
;manager_service_contract=<framed service contract>
;source_commit=<40 lowercase hexadecimal Git commit bytes>
;source_tree_state=clean
;build_configuration=Release
;compiler_contract=<framed exact compiler/build-system version contract>
;executable_sha256=<literal sha256: followed by 64 lowercase hexadecimal bytes>
;build_contract_version=1
```

The build pipeline and executable must reconstruct the same canonical. A path,
mtime, host, PID, signing session, or deployment timestamp is diagnostic and
excluded. Readiness compares the running executable canonical byte-for-byte to
the enable event. A dirty source tree, non-Release configuration, missing
embedded contract, or digest mismatch is not ready.

### 4.4 Global enable event

| Order | Canonical field | Exact authority |
|---:|---|---|
| 0 | prefix | `campaign_operations_production_enable_event_v1` |
| 1 | `operation_key` | framed caller-supplied key |
| 2 | `predecessor_event_id` | `none` at genesis, otherwise exact positive ID |
| 3 | `predecessor_event_canonical` | framed empty at genesis, otherwise complete predecessor canonical |
| 4 | `expected_prior_version` | `0` at genesis, otherwise exact head version |
| 5 | `resulting_version` | `expected_prior_version + 1` |
| 6 | `scheduler_protocol_evidence` | framed complete §4.2 canonical |
| 7 | `independent_verification_reference` | framed exact reviewed artifact reference |
| 8 | `authorizing_actor` | framed application actor that authorized enablement |
| 9 | `capability` | literal `campaign_operations_production_enabler` |
| 10 | `manager_service_contract` | framed exact approved service contract; V1 value is `campaign-operations-production-dispatch-and-manager-run-once-v1` |
| 11 | `approved_build_contract` | framed complete §4.3 build canonical |
| 12 | `reason` | framed nonempty reason |
| 13 | `enablement_contract_version` | `1` |

The enable row stores its complete canonical/hash and exact typed mirrors.
Operation key, predecessor edge, and resulting version are unique conflict
domains in one alternating chain.

### 4.5 Global disable event

| Order | Canonical field | Exact authority |
|---:|---|---|
| 0 | prefix | `campaign_operations_production_disable_event_v1` |
| 1 | `operation_key` | framed caller-supplied key |
| 2 | `predecessor_event_id` | exact current enable event ID |
| 3 | `predecessor_event_canonical` | framed complete current enable canonical |
| 4 | `expected_prior_version` | exact current enable version |
| 5 | `resulting_version` | `expected_prior_version + 1` |
| 6 | `disabling_actor` | framed application actor |
| 7 | `capability` | literal `campaign_operations_production_disabler` |
| 8 | `reason` | framed nonempty reason |
| 9 | `enablement_contract_version` | same version as the predecessor enable event |

Scheduler evidence, Manager/build contract, stopped PID, deployment log,
hostname, and session/process identity are excluded. Disable must remain
available when scheduler evidence is unavailable or mismatched.

### 4.6 First request production admission

| Order | Canonical field | Exact authority |
|---:|---|---|
| 0 | prefix | `campaign_operations_request_production_admission_v1` |
| 1 | `operational_request_id` | exact positive request ID |
| 2 | `request_identity_canonical` | framed complete immutable request canonical |
| 3 | `expected_request_version` | exact version before the first production acquisition |
| 4 | `dispatch_operation_key` | framed exact first acquisition key |
| 5 | `enable_event_id` | exact authorizing enable event ID |
| 6 | `enable_event_canonical` | framed complete enable canonical |
| 7 | `requesting_actor` | framed actor requesting this dispatch |
| 8 | `original_executing_service_principal` | framed exact original `session_user` service principal |
| 9 | `approved_build_contract` | framed complete §4.3 canonical required by the enable event |
| 10 | `capability` | literal `campaign_operations_production_dispatcher` |
| 11 | `admission_contract_version` | `1` |

There is exactly one admission per request. Candidate scan/batch identity,
recovering principal, PID, host, connection, and invocation are excluded.

### 4.7 Attempt V2

| Order | Canonical field | Exact authority |
|---:|---|---|
| 0 | prefix | `campaign_operations_dispatch_attempt_v2` |
| 1 | `operational_request_id` | exact positive request ID |
| 2 | `request_identity_canonical` | framed complete request canonical |
| 3 | `request_production_admission_canonical` | framed complete §4.6 canonical |
| 4 | `enable_event_id` | exact authorizing current enable ID |
| 5 | `enable_event_canonical` | framed complete §4.4 canonical |
| 6 | `operation_key` | framed caller-supplied or deterministic per-request key |
| 7 | `attempt_ordinal` | next ordinal across V1 and V2 attempts |
| 8 | `expected_request_version` | exact acquisition input version |
| 9 | `resulting_request_version` | `expected_request_version + 1` |
| 10 | `lease_token_digest` | framed exact persisted digest; opaque token remains secret/process-local |
| 11 | `lease_expires_at` | framed exact UTC microsecond expiry |
| 12 | `requesting_actor` | framed actor requesting dispatch |
| 13 | `original_executing_service_principal` | framed original `session_user` service principal |
| 14 | `approved_build_contract` | framed complete §4.3 enabled build canonical |
| 15 | `capability` | literal `campaign_operations_production_dispatcher` |
| 16 | `attempt_contract_version` | `2` |

Attempt V1 retains its existing canonical exactly and has none of fields 3–6
or 12–16 above except its existing `dispatcher_identity` and contract version.
V1 and V2 share one attempt table and ordinal sequence; typed nullable columns
plus a version-shape constraint prohibit mixed/incomplete rows.

### 4.8 Recovery identity and conflicting replay

A different authorized Manager may recover the exact original unexpired
Attempt V2. It MUST load and return the stored attempt/admission/enable chain;
it MUST NOT rebuild or change the attempt canonical. The recovering
principal/process/build/time may appear only in separate diagnostic audit or
status evidence. It does not replace the original actor, service principal, or
build contract. Exact-operation lookup occurs before constructing any new
attempt candidate. A recovering caller does not resupply the stored original
service principal/build as new logical input, so a different current principal
or process/build is not itself a replay conflict. Handoff still requires the
current enable event and executing service/build readiness from §3; recovery
lookup alone cannot bypass that gate.

For a matching logical operation key, `conflicting_replay` is REQUIRED if any
caller-supplied logical input or asserted original canonical byte differs,
including predecessor edge/version, scheduler canonical, verification
reference, actor, reason, asserted original service principal/build,
capability, request identity/version, admission, enable event, ordinal, lease
digest/expiry, or contract version. Current recovery diagnostics are excluded
from that comparison. A same hash with different canonical is a conflict.
Partial, malformed, missing nested, mismatched typed/canonical, or contradictory
evidence fails closed and is never `existing_identical`.

## 5. Frozen operation-key contract

All operation keys match the literal contract
`^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$`. They are lookup keys, not substitutes
for the full canonical.

| Operation | Key and durability | Replay/recovery rule |
|---|---|---|
| Global enable | Caller-supplied; durable and unique in the global event chain. | Find exact event by key, hydrate chain, full-compare; exact returns original, any change conflicts. |
| Global disable | Caller-supplied; durable and unique in the same chain. | Same as enable; no scheduler-readiness precondition. |
| Exact single-request production dispatch | Caller-supplied and REQUIRED; persisted as `(operational_request_id, operation_key)` on Attempt V2. | Lookup never infers from current request state. Exact returns original admission/attempt/lease or binding; change conflicts. |
| First request admission | Uses the first production dispatch operation key and is uniquely owned by request ID. | Exact first attempt/admission replay returns original. A later attempt references the immutable admission; it does not create another. |
| Attempt V2 acquisition | Same per-request dispatch key supplied to the engine. | `(request_id,key)` finds the exact attempt even after request state/version changes. |
| Recovered unexpired lease | Reuses the original request/key, attempt ID, canonical, digest, and expiry. | No new attempt, key, ordinal, or canonical; recovering identity is diagnostic. |
| Handoff | The Attempt V2 `operation_key` remains the handoff operation key; exact attempt ID plus complete attempt canonical disambiguates the persisted acquisition. No second random key exists. | Lookup by `(request_id, operation_key)` and validate exact attempt/binding. Exact complete binding/outcome returns `existing_identical`; partial or different evidence fails closed. |
| Run-once batch selection | No durable batch key or row. The snapshot may differ when rerun. | Batch replay has no semantic identity; each request does. |
| Run-once per-request processing | Deterministic `mgr-v1:<fnv1a64(source-canonical)>:<expected-version>`, where source canonical is `campaign_operations_manager_request_operation_v1;request_identity_canonical=<framed complete request canonical>;expected_request_version=<N>`. | Persist the complete source canonical with Attempt V2 and full-compare it; the hash segment is only bounded lookup material. |
| Post-Phase-F-recovery reacquisition | New request version produces a new deterministic run-once key or requires a new caller key for exact CLI dispatch, and receives the next ordinal. | Phase F recovery advances request version. Reuse of the pre-recovery key is exact replay of the old attempt, never a new acquisition. |

Random process UUID, PID, hostname, session ID, batch UUID, or invocation time
MUST NOT be used as logical identity. For `pqxx::in_doubt_error`, enable and
disable look up by their caller key; acquisition looks up by `(request_id,
operation_key)`; handoff looks up by exact attempt ID/canonical and binding.
Mutable request state is validated after locating the operation and is never
used to guess which operation committed.

## 6. Exact scheduler-generation contract

Phase H V1 is compatible with exactly:

- `required_generation = 52`;
- `cutover_state = complete`;
- byte-identical §4.2 protocol evidence canonical;
- byte-identical independent verification reference recorded in the current
  enable event;
- the exact Manager service and build contracts recorded in that event.

`generation >= 52`, “52 or newer,” approval inherited by generation 53,
cutover state alone, scheduler liveness, and broad raw scheduler-table access
are prohibited.

Migration 055 must create two pinned `SECURITY DEFINER` functions owned by
`campaign_operations_scheduler_protocol_evidence_owner`: a read-only snapshot
function and a mutating-path lock function. They use an explicit `pg_catalog,
public` search path, reference the qualified scheduler singleton, accept no
caller-supplied SQL/identity fields, validate all non-null complete cutover
fields, reconstruct §4.2, and return typed generation/state plus canonical/hash.
The lock function executes `SELECT ... FOR SHARE`; its row lock survives the
function and is held by the caller transaction through commit. `PUBLIC`
execution and raw-table privileges are revoked. The owner has only the exact
scheduler columns required. The execute capability is a separate NOLOGIN role.

On any protocol-canonical mismatch or generation other than 52:

- enablement history remains immutable;
- the current enable event becomes ineffective immediately;
- new acquisition and handoff fail readiness;
- an acquired but unbound generation-52/E1 attempt cannot hand off;
- disable E1 must be recorded before re-enable because the chain alternates;
- generation 53 requires a new enablement contract version, new independent
  verification, and new enable event;
- E2 never authorizes a lease that cites E1;
- old leases must expire and be recovered through Phase F before new
  acquisition.

## 7. PostgreSQL committed-state equations and guards

The following equations are normative at transaction commit:

```text
production_dispatch_enabled
  == EXISTS(exact immutable request production admission)

first admission
  => exactly one matching first Attempt V2
  => exactly one matching production audit

Attempt V1
  => production_dispatch_enabled = false
  => no production admission
  => no production enablement/admission fields

Attempt V2
  => production_dispatch_enabled = true
  => exact admission evidence
  => exact enablement evidence
```

Migration 055 replaces only the existing false-only check. An immediate
owner-DML-safe guard permits `false -> true` only at the required nested trigger
depth inside the guarded production-acquisition workflow. `true -> false` is
always rejected. A guarded transaction-local production-acquisition context
must be set by the fixed security-definer transition, checked by triggers, and
cleared automatically at transaction end; application-settable GUCs are not
authority.

Deferred constraint triggers enforce Boolean/admission equivalence in both
directions and first-admission/Attempt-V2/production-audit completeness. Enable,
disable, admission, V2 attempt, and production-audit history reject update,
delete, and truncate under ordinary owner DML as well as runtime roles. The
request guard rejects delete/truncate or a Boolean mutation that would violate
the equation. Rollback tests must show Boolean, admission, attempt, and audit
all remain unchanged after any failure.

Existing false rows are not backfilled. No read or mutation path authorizes
from the Boolean alone. Admission, V2, the false-to-true transition, or a new
production audit after completion is rejected. Completion insertion must gate
on complete admission and audit for every V2 attempt.

This deliberately repeats the Phase G structural correction: as
`completion_boundary_closed` is only a mutex witness for one immutable
completion event, `production_dispatch_enabled` is only a mutex witness for one
immutable admission. Immediate guarded directionality plus deferred
bidirectional equivalence prevents either Boolean from becoming a competing
fact.

## 8. Enable/disable and concurrency state machine

Locks are held through transaction commit unless a row is only read in the
optimistic selection/readiness path.

| Race/state | Required committed result |
|---|---|
| Enable versus acquisition | Acquisition sees disabled or the fully committed exact enable event; it cannot observe a partial event. |
| Disable versus acquisition | Protocol shared lock precedes gate. Disable takes the exclusive gate. Disable-first blocks acquisition; acquisition-first may commit under its exact event. |
| Disable versus handoff | Handoff holds protocol and shared gate through commit. Handoff-first may commit; disable-first blocks handoff. |
| Handoff-first outcome | Complete Phase E binding/outcome commits and exact replay remains valid after later disable. |
| Disable-first outcome | No handoff occurs; the acquired lease/history remains immutable for expiry and Phase F recovery. |
| Re-enable E2 while lease cites E1 | E2 does not authorize E1. Handoff refuses. |
| Old-event lease expiry | Status marks blocked, reports expiry, and becomes Phase F recovery-eligible only under existing Phase F predicates. |
| Phase F recovery | Existing authority terminalizes/records the old attempt as required, clears the expired lease, advances request version, and returns the request to its existing eligible state. |
| New acquisition after recovery | Uses new expected/resulting version, next ordinal, new operation key, current enable event, and a new V2 attempt; admission remains the immutable first admission. |
| Exact existing binding replay after disable | Returns the original complete binding/outcome; no enablement is needed to acknowledge committed truth and Phase 5 is not reinvoked. |
| Protocol upgrade while acquired/unbound | Current enable is ineffective; handoff fails; lease expires and Phase F recovers it before any new-generation acquisition. |
| Cancellation/pause/completion versus acquisition/handoff | Existing earlier campaign/request gates and accepted lock order decide; no Phase H bypass. Completion wins by closing the campaign boundary; post-completion production evidence is rejected. |
| Authorization revocation versus acquisition/handoff | Existing authorization-domain lock/revalidation decides. |
| Reservation expiry versus acquisition | Existing budget/reservation/request locks and PostgreSQL transaction time decide. |

Disable never deletes or rewrites a lease, admission, attempt, binding, outcome,
or audit. It never cancels experiments, reverses committed reservations,
signals workers, or mutates scheduler state.

Status must expose current enable head, current-event leases, old-event blocked
leases, exact expiry, Phase F recovery eligibility, and
`reconciliation_required` state/counts.

## 9. Campaign Manager and common Phase E engine

One internal Phase E engine is shared by:

1. the existing isolated-test adapter;
2. the exact production single-request adapter;
3. bounded Campaign Manager run-once.

The engine owns the accepted two-transaction acquisition/handoff workflow,
full-canonical replay, retry classification, and binding recovery. Adapters
provide only their authority evidence and operation key. Duplicate SQL/service
implementations are prohibited.

Test hooks are constructor-sealed behind the isolated-test adapter and cannot
be selected by production CLI, Manager configuration, environment variables,
or production constructors.

Run-once uses one read-only repeatable-read candidate snapshot, orders by
`operational_request_id`, and selects at most a reviewed bound `N`. It takes no
row/advisory locks and performs no mutation. `FOR UPDATE SKIP LOCKED` and every
request-first selection lock are prohibited. Each candidate is then processed
sequentially in its own normal engine transactions with the deterministic key
in §5. Request-local semantic failure is recorded/reported and processing
continues. Global disable/protocol mismatch, privilege failure, or database-wide
failure stops the batch. No transaction spans requests, sleeps, process work,
filesystem access, scheduler waits, or worker operations.

## 10. Test-only Attempt V1 isolation

Database naming is supplemental defense only. V1 acquisition requires all of:

- literal byte-prefix comparison with
  `expertadvisor_campaign_operations_phase3_test_` (never SQL `LIKE`);
- the isolated-test capability role;
- the existing literal acknowledgement;
- exact Attempt V1 shape;
- `production_dispatch_enabled=false`;
- no enablement event exists;
- `session_user` has no direct or inherited membership in any production role.

The Manager login must have no direct or inherited membership in either
existing Phase E isolated-test role. The tests must prove underscore characters
are literal and similarly shaped non-test database names are rejected.

## 11. Repository and service boundaries

Repositories persist and hydrate complete canonical chains; they do not decide
business authority. Services validate identities, order transactions, compare
full canonicals, classify replay, and invoke the common Phase E engine.

Required repository surfaces are exact event-by-operation-key lookup,
current-head lookup, exact scheduler evidence snapshot/lock, immutable admission
lookup, V2 attempt lookup by `(request_id,key)` and attempt ID, complete
binding/outcome lookup, optimistic candidate selection, and readiness snapshot.
Every persisted load reconstructs and validates canonical/hash/typed mirrors
and every nested canonical before returning.

Enable transaction: exact scheduler evidence shared lock, exclusive production
gate, exact replay lookup/head reload, append event/audit, commit. Acquisition
and handoff additionally prove the executing service/build contract matches
the authorizing enable event. Disable:
exclusive production gate, replay/head validation, append event/audit, commit;
it has no scheduler readiness dependency. Acquisition/handoff: exact scheduler
evidence shared lock, shared production gate, then the existing Phase E locks.

## 12. Replay, retry, and uncertain commit

Whole transactions retry only SQLSTATE `40001` and `40P01`, at most three
attempts. A uniqueness result is reloaded and full-compared; it is not assumed
identical. Arbitrary `pqxx::sql_error`, `broken_connection`, or other failure
is never replay success.

For enable, disable, acquisition, and handoff, `pqxx::in_doubt_error` requires:

1. discard the uncertain connection;
2. open a fresh connection;
3. locate the exact durable operation by operation key or exact attempt
   identity, never mutable-state inference;
4. hydrate and validate the complete canonical chain;
5. return the original identical result only if complete and byte-exact;
6. retry only after proven absence and unchanged expected authority;
7. fail closed on partial, malformed, contradictory, or lookup-uncertain
   evidence.

Lookup itself receiving `pqxx::in_doubt_error` discards that lookup connection
and continues only within the same bounded fresh-connection proof policy. If
absence cannot be proven, the stable result is outcome ambiguous or
reconciliation required, never guessed success.

Actual `pqxx::in_doubt_error` injection tests are required before commit,
after commit/before response, and during lookup for every one of the four
workflow families.

## 13. Global lock order and path audit

The Campaign Operations order extended by Phase H is:

```text
0a exact scheduler protocol evidence, shared when mutating
0b global production-enable domain
1  authorization domains
2  budget
3  campaign/completion
4  reservations ascending
5  requests ascending
6+ existing Phase 5/lifecycle domains
```

| Path | Locks/order |
|---|---|
| Enable | 0a shared, then 0b exclusive; append under those locks. |
| Disable | 0b exclusive; no scheduler dependency, then head/event append. |
| Readiness/status | Read-only repeatable-read snapshot; no mutating/advisory/tuple locks. |
| Candidate selection | Read-only optimistic snapshot; no locks and no `SKIP LOCKED`. |
| Acquisition | 0a shared, 0b shared, then 1→2→3→4→5 and existing later needs. |
| Handoff | 0a shared, 0b shared, then 1→2→3→4→5→6+ exactly as current Phase E. |
| Phase F recovery | Existing 3→5 order; it does not acquire 0a/0b because it removes expired authority and never produces a handoff. |
| Cancellation | Existing authorization/campaign/reservation/request/lifecycle order; no new Phase H locks unless rejecting a production mutation at its normal campaign/request gate. |
| Completion | Existing campaign/completion-first Phase G order and post-completion guards; no production gate is acquired. |
| Scheduler readiness | Scheduler-owned protocol/cutover order only; it never takes Campaign Operations locks. |

Enable uses 0a because it authorizes exact evidence. Disable intentionally
starts at 0b so it remains emergency-safe. No transaction may hold Campaign
Operations locks while waiting on a scheduler process, filesystem operation,
worker launch/signal, sleep, network/external work, or other long-running work.

## 14. Privilege and deployment contract

Migration 055 creates or hardens one sealed infrastructure owner and the seven
frozen NOLOGIN production roles; migration 059 adds the NOLOGIN dispatch-service
capability. The exact role table in ADR-0019A §5 is
normative.

- `campaign_operations_h1_boundary_authority` is `NOLOGIN SUPERUSER`, has no
  membership in either direction, and owns every protected H1 table, sequence,
  view, trigger/helper, and fixed transition. It is not a production
  capability; superuser/cluster-administrator power is outside the supported
  threat model. PostgreSQL implicit owner authority is why no ordinary or
  reachable role may own the boundary;

- `campaign_operations_production_enabler`;
- `campaign_operations_production_disabler`;
- `campaign_operations_production_dispatcher`;
- `campaign_operations_production_phase5_transactional`;
- `campaign_operations_production_reader`;
- `campaign_operations_scheduler_protocol_evidence_owner`;
- `campaign_operations_scheduler_protocol_evidence_reader`.
- `campaign_operations_production_dispatch_service` (migration 059 only; the
  sole EXECUTE grantee for the final mutation-capable wrapper).

The enabler cannot disable or dispatch. The disabler cannot enable or dispatch.
The ordinary Manager login receives only `campaign_operations_production_dispatcher`,
`campaign_operations_production_phase5_transactional`,
`campaign_operations_production_reader`,
`campaign_operations_scheduler_protocol_evidence_reader`, and the existing
read capabilities explicitly required by the common Phase E engine. It lacks
enabler, disabler, every isolated-test role, and
all scheduler mutation capabilities, and it cannot reach the dispatch-service
capability. A separately provisioned dispatch-service LOGIN receives only
`campaign_operations_production_dispatch_service` and
`campaign_operations_production_phase5_transactional`; it must not receive
`campaign_operations_production_dispatcher` and is used only by the reviewed
C++ mutation path after the Manager-side actual-build preflight. The scheduler cannot read Campaign
Operations. The Manager cannot mutate scheduler claims, attempts, leases,
capacity, ownership, invocation, or processes. `pqxx` receives no new
membership. Migration 055 grants no LOGIN membership.

Functions have pinned qualified names/search paths, `PUBLIC` execute revoked,
exact expected-state predicates, and catalog-tested ownership/ACLs. History and
witness triggers reject ordinary owner DML. Before enablement, deployment must
audit direct and inherited membership for the actual `session_user`, including
nested role membership and prohibited test/production combinations.

### 14.1 Direct SQL readiness boundary correction

Migration 059 is the forward correction for the confirmed direct-SQL
readiness bypass. The deployed Manager/dispatcher login retains no
`EXECUTE` privilege on
`transition_campaign_operations_request_dispatch_production_v2`. That raw
SECURITY DEFINER function remains owned by
`campaign_operations_h1_boundary_authority` and is entered only by the sealed
`campaign_operations_production_dispatch_authorized_v3`
service boundary. Only the distinct dispatch-service capability can execute
that wrapper; the ordinary Manager/dispatcher login cannot. Before invoking the
unchanged atomic transition it evaluates the
existing readiness view, role graph, scheduler generation-52 evidence,
effective enablement, approved build identity, canonical Admission/Attempt
evidence, completion nested-V2 proof, and reconciliation blockers. PUBLIC,
`pqxx`, and inherited dispatcher membership cannot execute the raw function.

The application still performs the reviewed C++ readiness/build preflight through
the Manager connection, then routes the mutation transaction through the
dispatch-service connection. H3 Manager/H4 supervisor continue to use that
path. Migration 059 makes the
database-side authorization unavoidable, so a SQL caller cannot manufacture a
caller-settable GUC/token or skip readiness by invoking the raw transition.
Emergency disable remains independently executable by the disabler boundary;
replay, uncertain-commit recovery, valid lease recovery, and the existing lock
ordering remain in the unchanged transition and common dispatch engine.

The former unaccepted
`campaign_operations_production_transition_owner` is not part of the inventory
and is not created. The scheduler-protocol evidence owner retains its role name
for compatibility but owns no H1 boundary object after migration 055. Unsafe
pre-existing memberships or default ACLs cause migration failure; they are not
silently retained or repaired. ADR-0019A §§4–8 freeze the exact ownership,
context, current/default ACL, and `SECURITY DEFINER` rules.

ADR-0019B freezes the sealed role as exact `NOLOGIN SUPERUSER INHERIT
NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1`, with
no password, validity, role/database configuration, operational credential, or
membership/ADMIN OPTION edge in either direction. Migration 055 inspects every
attribute before mutation and fails closed rather than normalizing an existing
role. The other seven H1 roles are likewise exact inert NOLOGIN roles.

Ownership transfer is a literal schema-qualified object/signature manifest.
Name patterns and trigger-discovery transfers are prohibited. Every non-system
schema and relevant catalog class is audited for unexpected boundary ownership,
same-name entry points, overloads/default variants, wrappers, triggers,
operators, casts, rules, dependencies, PUBLIC execution, and grant options.
The minimal table/view/sequence inventory and exact function manifest are
normative in ADR-0019B §5 and migration 055's audit CTE.

Deployment uses the read-only versioned
`Scripts/CampaignOperationsH1DeploymentAudit.sh` at pre/post upgrade,
pre-restore, post-role recreation, post-database restore, and pre-enablement.
Its stable `H1A001`–`H1A011` failures block deployment. Supported restore
workflows and hostile cases A–J are frozen by ADR-0019B §9; migration checksum
and ledger success never substitute for role-graph, catalog, ACL, or historical
byte evidence.

Emergency rollback is: commit disable, stop Manager processes, revoke
production roles. No persisted evidence is deleted.

## 15. Readiness and status

Readiness/status is read-only and repeatable-read. It performs no advisory or
tuple locking, sequence advancement, repair, enable/disable, lease acquisition,
or scheduler mutation. It reports at least:

- migration 055 identity, recorded checksum, and all canonical contract
  versions;
- exact scheduler generation, cutover state/evidence, §4.2 canonical/hash, and
  any mismatch;
- exact independent verification reference;
- current enable head canonical/hash/version/kind and computed effective state;
- approved Manager service/build contract and actual running build contract;
- actual `session_user` and `current_user`;
- direct and inherited required/prohibited role readiness;
- ready admitted and ready unadmitted request counts;
- active current-event leases;
- old-event blocked leases with exact expiry and Phase F recovery eligibility;
- reconciliation-required counts/details;
- Completion V1 nested-V2 proof version/status.

“Ready” is true only when every required item is exact, current, and complete.
Scheduler liveness may be shown separately as diagnostic and cannot change the
result except that operators may impose a separate operational rollout stop.

## 16. Completion V1 nested-canonical proof

Completion V1 remains valid; Completion V2 is prohibited absent a later direct
repository trace showing a break in this chain:

```text
campaign_operations_completion_v1
  -> request_evidence_canonical
     -> exact attempt_identity_canonical
        -> complete request admission canonical
        -> complete enablement canonical
```

The proof is direct:

1. `campaign_operations_completion_evidence_text(..., 'request')` in migration
   054 emits `requests_v1` and inserts each stored
   `campaign_operations_dispatch_attempt.attempt_identity_canonical` using its
   exact octet length and exact text, ordered by ordinal/ID. It does not
   reconstruct attempt fields.
2. `BuildCompletionEventFromAuthority` in
   `Sources/CampaignOperationsCompletionRepository.cpp` calls that PostgreSQL
   function and constructs `CanonicalIdentity` from the complete returned
   evidence text.
3. `BuildCompletionEvent` in `Sources/CampaignOperationsCompletion.cpp` frames
   that complete request-evidence canonical into
   `campaign_operations_completion_v1`.
4. Migration 054 independently reconstructs the completion canonical and
   compares every evidence canonical/hash. Migration 055 adds V2 validation and
   completion blockers but does not change this byte-preserving nesting.

Because Attempt V2 itself frames the complete admission and enable canonicals,
the unchanged Completion V1 snapshot transitively binds both. Historical V1
attempt and completion bytes remain unchanged.

H1 tests must cover complete V2 nested canonicals; PostgreSQL and C++
reconstruction; byte-for-byte C++/PostgreSQL golden vectors;
same-hash/different-canonical rejection at attempt, admission, enablement,
request-evidence, and completion levels; malformed/incomplete V2 rejection
before commit; additive V1/V2 compatibility; unchanged historical V1 attempt
and completion identities; completion gates on admission and production audit;
and post-completion rejection of admission, V2, Boolean transition, and new
production audit.

## 17. Migration 055 architecture

Migration 055 is implemented by H1 as the disabled authority and persistence
foundation. Its implementation is authorized to add:

- `campaign_operations_production_enablement_event`, one append-only
  alternating global table with complete enable/disable typed mirrors,
  canonical/hash, and unique resulting-version, predecessor-edge, and
  operation-key constraints;
- `campaign_operations_production_enablement_audit_reference_event`, exactly one
  immutable matching audit per enable/disable event;
- `campaign_operations_request_production_admission`, exactly one immutable
  first admission per request, with exact request/enable FKs and complete
  canonical/hash;
- additive nullable V2 columns and V1/V2 shape constraints on the existing
  dispatch-attempt table;
- a nullable admission FK plus production cause/capability shapes on the
  existing `campaign_operations_dispatch_audit_reference_event`, so the first
  V2 acquisition has exactly one matching production audit;
- the guarded false-to-true request transition, deferred equations in §7,
  history/witness owner-DML guards, completion gates, and post-completion gates;
- `record_campaign_operations_production_enable_v1` and
  `record_campaign_operations_production_disable_v1` fixed transition
  functions;
- `transition_campaign_operations_request_dispatch_production_v2`, the only
  function allowed to establish first admission, Boolean true, Attempt V2, and
  its audit atomically;
- `campaign_operations_scheduler_protocol_evidence_snapshot_v1` and
  `campaign_operations_scheduler_protocol_evidence_lock_v1`, the narrow
  scheduler-evidence functions from §6;
- `campaign_operations_production_readiness_v1` and
  `campaign_operations_production_status_v1`, read-only views backed only by
  validated authoritative facts;
- roles and narrow scheduler evidence functions from §14.

Constraint/trigger families are frozen by behavior: alternating fork-resistant
head/version/predecessor/key uniqueness; typed-mirror/canonical/hash validity;
V1/V2 exclusive shape; one admission per request; one enablement audit per
event; one first-V2 production audit per admission; immediate guarded witness
direction; deferred witness/admission equivalence; deferred first-admission
completeness; post-completion rejection; and update/delete/truncate rejection
for every immutable history and witness boundary. Names may follow the existing
`campaign_operations_<object>_<rule>` convention, but no family may be omitted
or weakened.

It MUST be transactional, runner-checksummed, idempotent on supported replay,
and additive over 054. It MUST leave every existing request false, add no
admission/backfill, preserve every V1 canonical, grant no login role, and make
no scheduler/lifecycle/worker data mutation. Table/sequence/function ownership,
NULL ACL/default ACL interpretation, column privileges, search paths, and
PUBLIC revocation are tested from PostgreSQL catalogs.

The former acquisition spelling
`transition_campaign_operations_request_dispatching_production_v2` is
impossible at 64 UTF-8 bytes. The authoritative replacement
`transition_campaign_operations_request_dispatch_production_v2` is exactly 61
UTF-8 bytes. Tests compare `pg_proc.proname::text` and octet length exactly and
reject truncation notices; `regprocedure` resolution alone is insufficient.

The transaction context remains only as sealed, exact-operation evidence. It
binds PID, top-level XID, transition kind, request, and operation key; fixed
transitions remove it before return and a deferred constraint forbids a row at
commit. All protected relations, triggers, helpers, and transitions are owned
by the unreachable sealed authority, so an ordinary former owner cannot forge
the row, grant a writer, replace a guard, or install a recursive trigger. Exact
replay lookup and full immutable evidence comparison precede mutable head/CAS
validation for enable, disable, and acquisition. Acquisition establishes
0a→0b→authorization→budget→campaign/completion→reservations ascending→requests
ascending by calling the existing Phase E lock helpers. ADR-0019A §§4–10 are
normative for these corrections.

## 18. H1–H4 increments

### H1 — Authority and persistence foundation

ADR-0019; this canonical/golden-vector contract; migration 055 implementation;
immutable enablement/admission evidence; Attempt V1/V2 compatibility;
owner-DML guards; roles; readiness/status. H1 has no production handoff, no
Manager batch, and no continuous mode.

### H2 — Exact production dispatch

Refactor to one common Phase E engine; enable/disable services; exact
single-request production adapter; required caller-supplied operation key;
exact canary CLI; fresh-connection in-doubt recovery; disable/acquisition/
handoff concurrency tests. H2 has no run-once batch and no continuous mode.

### H3 — Bounded Campaign Manager run-once

Optimistic read-only candidate selection; deterministic per-request operation
keys; sequential bounded processing; request-local failure continuation;
global disable/protocol/privilege/database failure stop; multi-manager
correctness tests. H3 has no daemon, polling, sleep, autostart, or supervision.

### H4 — Continuous Campaign Manager

ADR-0019 excluded H4 from its initial implementation. ADR-0020 later accepts
H4 solely as external, deployment-owned (including launchd) supervision that
repeatedly invokes the bounded H3 run-once command. It owns the operational
supervision, restart/autostart, cadence/backoff, graceful shutdown,
health/status, log ownership/rotation, duplicate-instance limits, and rollback
contract. It does not authorize an in-process `LSTM_Release` continuous
CLI/daemon, scheduler-owned Campaign Manager polling, or database singleton,
heartbeat, lease, or leader-election authority. Database-level multi-manager
correctness remains only an H3 request-boundary backstop.

H1, H2, and H3 are independently committable and safe while no enable event
exists and no production LOGIN membership has been granted.

### 18.1 Frozen command boundary

H1 adds read-only `--campaign-operations-production-readiness` and
`--campaign-operations-production-status`. H2 adds explicit
`--campaign-operations-production-enable`,
`--campaign-operations-production-disable`, and
`--campaign-operations-dispatch-request`; every mutation uses the existing
`--campaign-operations-operation-key` option, with request ID/expected version,
actor, reason where applicable, and literal acknowledgement. H3 adds only
`--campaign-operations-manager-run-once` with a positive bounded dispatch
limit. Production mutation rejects dry-run; read-only commands reject
acknowledgement. No production flag exposes test hooks, and no continuous
Manager CLI/daemon command exists. ADR-0020 H4 is an external supervisor, not
an additional `LSTM_Release` command.

## 19. Required implementation verification

In addition to the tests already named:

- pure identity validation/golden vectors, UTF-8 framing, locale and UTC
  precision, exact/conflicting replay, and illegal transitions;
- clean install, 054→055, migration replay, no backfill, constraint/trigger/
  owner-DML/ACL/search-path/role catalog assertions;
- V1 test isolation including literal prefix and membership-graph negatives;
- rollback proof for Boolean/admission/attempt/audit;
- enable/disable/acquisition/handoff exact replay, collisions, partial evidence,
  bounded retries, and all uncertain-commit injection points;
- independent-connection races from §8 with `pg_blocking_pids()` evidence, not
  timing sleeps;
- two or more Managers contending for the same requests and unrelated-request
  concurrency;
- Phase A–G identity/replay/concurrency/ACL/CLI/integration regressions;
- scheduler generation-52 claim/cancellation/launch/orphan/capacity/shutdown
  regressions in disposable databases/fake-process seams and process suites only
  in an operationally safe window.

No test may alter production experiment rows, roles, scheduler state, or
workers.

Canonical contract-version readiness constrains relevant persisted evidence that
exists. During genesis, an entirely uninstantiated Admission V1 family and an
entirely uninstantiated production Attempt V2 family are represented by explicit
zero evidence counts and are ``genesis-empty`` rather than version mismatches.
After the first relevant row exists, the aggregate must be exactly Admission
V1 or Attempt V2 as applicable. Wrong or mixed versions, malformed or unreadable
rows, stale canonical/hash evidence, and broken required links remain
fail-closed. Genesis-empty does not waive enablement, build, role, scheduler,
lease, reconciliation, completion-proof, or any other readiness requirement.

## 20. Rollout, rollback, and implementation acceptance

Rollout order is fixed:

1. remain disabled through migration and executable deployment;
2. run scheduler/global-control process regressions in a safe window;
3. create and audit the dedicated Manager login;
4. run readiness/status;
5. record one explicit enable event;
6. dispatch one exact caller-keyed canary;
7. inspect admission, attempt, binding, lifecycle, scheduler claim, and audit;
8. run one-request run-once;
9. increase only through bounded reviewed limits;
10. when continuous operation is required, use only the ADR-0020 external H4
    deployment contract.

Rollback is always disable first, then stop Manager, then revoke roles. No
admission, attempt, binding, enablement, disablement, or audit evidence is
deleted or rewritten.

An increment is accepted only when compiler/tests and PostgreSQL constraints
prove its exact contract, its documentation is aligned, and no implementation
choice was invented. H1–H3 do not authorize production merely by being merged;
the explicit enable event and reviewed login grants are still required.

## 21. Relationships and amendments

ADR-0010 through ADR-0018 retain all prior guarantees: full-canonical replay,
bounded whole-transaction retry, fresh-connection uncertain-commit recovery,
PostgreSQL-enforced invariants, immutable evidence, owner-DML safety, accepted
lock ordering, least privilege, ownership separation, and unchanged historical
V1 identities.

This document and ADR-0019 amend only the Phase H roadmap/traceability wording
identified in ADR-0019 §9. The original Phase H proposal and focused
verification remain review inputs, not authority.

## 22. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-07-31 | Accepted exact Phase H authority, identities, operation keys, generation-52 contract, invariants, state machine, Completion V1 proof, increments, privileges, readiness, and rollout. |
| 1.1.0 | 2026-08-01 | Applied ADR-0019A's targeted 61-byte acquisition name, sealed owner-safe context, exact replay, complete lock order, and ACL/verification correction; all other Phase H boundaries remain unchanged. |
| 1.2.0 | 2026-08-01 | Applied ADR-0019B's exact sealed-role identity/graph, literal minimum ownership and all-schema entry-point allowlists, automatic deployment audit, restore A–J, default-ACL, lock, negative-test, and historical-byte acceptance contracts. |

## H1 final implementation-consistency correction

The frozen lifecycle is implemented as ``ready@v3/no admission ->
dispatching@v4/admission A/Attempt V2 #1 -> ready@v5/admission A`` after a
committed Phase F recovery, then ``dispatching@v6/admission A/Attempt V2 #2``.
Admission A is inserted only when absent and is never updated, replaced, or
deleted. Each later acquisition uses the current request version, enablement,
actor, service principal, and approved build while retaining A as immutable
first-admission evidence.

Exact replay locates an attempt by persisted request/version/key metadata and
hydrates its original principal and build; the recovering Manager's current
identity and build are not historical replay inputs. Repository hydration is
closed over ``Attempt V2 -> current enablement/audit -> immutable admission ->
first enablement/audit -> first Attempt V2/acquisition audit``. Missing,
duplicate, or inconsistent nodes fail as persistence corruption.

Readiness reports scheduler canonical/hash identity, independent verification
reference, enablement canonical/hash, approved and actual running build
canonical/hash, comparison result, and all blockers. Missing runtime evidence
remains blocking while the complete available evidence stays reportable.
