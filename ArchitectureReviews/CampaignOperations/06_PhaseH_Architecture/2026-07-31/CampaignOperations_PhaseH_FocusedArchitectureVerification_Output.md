---
title: "Campaign Operations Phase H Focused Architecture Verification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_FocusedArchitectureVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H Focused Architecture Verification

# Findings

## BLOCKER

### 1. ADR-0019 is required and does not yet exist

- Proposal: the plan recognizes the need for a new ADR in its [Executive Summary](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:46>) and implementation sequence at [line 627](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:627>).
- Evidence: the accepted roadmap currently defines Phase H persistence/services/CLI as “scheduler-owned only” and treats enablement as a default-off configuration/privilege action ([accepted architecture §31.8](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md:1961>)). No ADR-0019 is present.
- Requirement: durable global admission, per-request admission, and a Campaign Operations-owned Manager materially amend that accepted wording.
- Failure mode: implementation would invent authority not yet granted by an accepted decision.
- Type: architecture/documentation.
- Correction: accept ADR-0019 before persistence or service implementation. It must own production admission, Manager ownership, exact scheduler/build evidence, default-off deployment, disable semantics, run-once-first rollout, scheduler isolation, and H4 exclusion.

### 2. Actor, service-principal, build, and recovery identities are conflated

- Proposal: request admission records “manager actor/build identity” ([§2](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:146>)); Attempt V2 has “manager build identity” ([§3](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:173>)); any authorized manager may recover a lease ([Failure Recovery](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:492>)); changed actor/build conflicts ([Replay Analysis](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:480>)).
- Evidence: existing Attempt V1 makes `dispatcher_identity` part of exact canonical identity ([Dispatch.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatch.cpp:378>)), while recovery reloads the stored attempt rather than rebuilding it with the recovering process ([DispatchRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchRepository.cpp:450>)).
- Requirement: authorizing actor, per-request requesting actor, original executing service principal, database capability, build contract, and recovering service/process identity are distinct.
- Failure mode: a restarted replica or different authorized manager could produce `conflicting_replay` solely because its login, PID, host, or invocation UUID differs.
- Type: identity/replay.
- Correction: freeze distinct canonical fields as specified below. Recovery must reuse the original attempt unchanged; the recovering principal is diagnostic audit/status evidence only.

### 3. Per-request operation-key semantics are absent

- Proposal: enable/disable accept operation keys, but the exact production-dispatch command accepts only request ID/version/actor ([CLI](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:404>)); run-once similarly has no per-request key contract ([line 413](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:413>)). Acquisition uncertainty is described using request version, lease digest, attempt, admission, and audit, without a durable lookup key ([line 364](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:364>)).
- Evidence: Attempt V1 is unique by request/ordinal and request/resulting-version but has no operation key ([migration 048](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/048_campaign_operations_durable_dispatch_handoff.sql:35>)). The existing opaque lease token is process-random ([DispatchService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:42>)); only its persisted digest survives.
- Requirement: uncertain-commit lookup must locate the exact logical operation without inferring it from mutable current state.
- Failure mode: after a crash, a manager cannot distinguish replay of one acquisition from a new acquisition, especially after disable/re-enable or safe Phase F recovery.
- Type: identity/replay/recovery.
- Correction:
  - exact single-request CLI must require a caller-supplied operation key;
  - run-once must derive a deterministic key from immutable request identity plus expected request version;
  - persist `(request_id, operation_key)` on Attempt V2;
  - exact replay returns the original attempt; a later post-recovery attempt receives a new ordinal and a new version-derived key;
  - no process UUID, PID, hostname, or random batch ID may be the logical key.

### 4. The Boolean-witness invariant lacks an owner-DML-safe design

- Proposal: the plan correctly calls `production_dispatch_enabled` a witness and states false→true must commit with admission and Attempt V2 ([§2](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:161>)), but it does not specify the immediate guard and bidirectional deferred checks needed to prove that claim.
- Evidence: the current column has a hard `CHECK (production_dispatch_enabled = false)` ([migration 047](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/047_campaign_operations_budget_request_acceptance.sql:363>)). Phase G required an explicit owner-level guard and deferred equivalence check to stop `completion_boundary_closed` becoming competing authority ([migration 054](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql:1105>); [independent finding](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/06_Phase5_Operational_Completion/2026-07-31/CampaignOperations_Phase5_FinalFour_IndependentCEEVerification_Output.md:119>)).
- Requirement: Boolean, admission, Attempt V2, and audit must be atomically and bidirectionally consistent, including against ordinary owner DML.
- Failure mode: owner update/delete/truncate could leave true without admission, admission without true, or admission without the first complete V2 attempt.
- Type: migration/privilege/invariant.
- Correction:
  - immediate guard: only false→true, only from the guarded nested production-acquisition workflow; true→false always rejected;
  - deferred constraint: `production_dispatch_enabled = EXISTS(exact immutable admission)`;
  - deferred first-admission completeness: matching V2 attempt and audit must exist at commit;
  - immutable update/delete/truncate guards on enablement, admission, production-attempt evidence, and their audit;
  - rollback tests proving all four facts remain unchanged.

## HIGH

### 5. Scheduler-upgrade semantics are unspecified

- Proposal: V1 requires generation 52 ([Production gates](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:98>) and [Scheduler Design](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:234>)), but does not define generation 53 behavior.
- Evidence: migration 052 stores a mutable singleton protocol row and advances `required_generation` with `GREATEST` ([migration 052](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql:7>)). It does not provide a versioned protocol canonical.
- Requirement: future scheduler changes must not inherit generation-52 approval.
- Failure mode: an implementation using `>= 52`, or comparing only `cutover_state='complete'`, could silently authorize a later, independently unverified protocol.
- Type: scheduler-boundary/rollout.
- Correction:
  - Phase H V1 accepts exactly generation 52 and one exact versioned protocol-evidence canonical;
  - “52 or newer” is prohibited;
  - any canonical mismatch or generation other than 52 makes the current enable head ineffective immediately;
  - generation 53 requires both a new enablement-contract version and a new independently verified enable event;
  - because the chain alternates, record disable before the new enable;
  - an already-acquired unbound attempt is blocked after upgrade and must expire/recover before a new attempt.

### 6. Protocol/gate lock acquisition and old-event lease status need tightening

- Proposal: the global order is correct ([Concurrency Analysis](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:447>)), but acquisition is described as taking the gate and then validating generation 52 ([line 344](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:344>)). Status does not explicitly report old-event blocked leases or recovery eligibility.
- Evidence: accepted dispatch order is authorization→budget→campaign→reservation→request ([accepted architecture §19](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md:997>)); scheduler protocol precedes all Campaign Operations locks under ADR-0018 ([ADR-0018](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0018-scheduler-generation-52-exact-attempt-authority.md:36>)).
- Requirement: protocol evidence must be stable while the production gate and Campaign Operations locks are held.
- Failure mode: gate→protocol versus protocol→gate ambiguity creates a future inverse edge or protocol-upgrade TOCTOU.
- Type: concurrency/scheduler-boundary/status.
- Correction: acquisition and handoff take protocol `FOR SHARE` through the narrow function, then shared production gate, then levels 1–8, holding both through commit. Status must distinguish current-event leases, old-event blocked leases, expiry, and Phase F recovery eligibility.

### 7. Continuous operation is not sufficiently specified

- Proposal: continuous mode is in scope and described by bounded polling, interval, graceful shutdown, and multi-manager safety ([§Continuous mode](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:271>)).
- Evidence: accepted Phase E explicitly excluded automatic production polling ([accepted architecture](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md:1907>)).
- Requirement: database concurrency safety is not a daemon-operating contract.
- Failure mode: duplicate deployment, uncontrolled churn, unclear restart/autostart ownership, unbounded log growth, or rollback that stops processes before disabling authority.
- Type: architecture/rollout/operations.
- Correction: exclude continuous mode from initial Phase H. H4 must separately accept supervision, restart policy, deployment owner, cadence/backoff, health, shutdown, log ownership/rotation, duplicate-instance limits, automatic startup, and rollback.

### 8. The V1 database-name defense is incomplete and the shown `LIKE` is unsafe

- Proposal: V1 uses `current_database() LIKE 'expertadvisor_campaign_operations_phase3_test_%'` ([§Isolated-test compatibility](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:223>)).
- Evidence: `_` is a wildcard under SQL `LIKE`; the existing application performs a literal prefix comparison and literal acknowledgement ([DispatchService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:55>)).
- Requirement: database name is supplemental defense, not authority.
- Failure mode: a similarly shaped non-test database or a login holding both test and production capabilities could reach V1.
- Type: privilege/test/security.
- Correction: use literal prefix comparison, and require all of:
  - isolated-test capability role;
  - literal acknowledgement;
  - V1 attempt shape;
  - Boolean false;
  - no enablement event;
  - no direct or indirect production-role membership for `session_user`.

  The production manager login must have no direct or inherited membership in either existing Phase E test role.

## MEDIUM

### 9. Completion inclusion was left conditional; this review resolves it

- Proposal: it conditionally requires Completion V2 if any path reconstructs attempt fields ([§Production attempt V2](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:173>)).
- Evidence: PostgreSQL’s request evidence embeds each complete `attempt_identity_canonical` byte-for-byte ([migration 054](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql:304>)); C++ loads that complete evidence canonical and embeds it into `campaign_operations_completion_v1` ([CompletionRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionRepository.cpp:254>), [Completion.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletion.cpp:123>)).
- Requirement: resolve the version choice before implementation.
- Failure mode: treating this as a later detail could persist production attempts not covered by completion identity.
- Type: lifecycle/migration/test.
- Correction: explicitly retain Completion V1. H1 must prove SQL/C++ nested golden vectors and reject malformed V2 evidence before any row commits.

### 10. Readiness/status omits required rollout blockers

- Proposal: status lists enablement, scheduler match, counts, leases, reconciliation, and build identity ([§Status views](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md:188>)).
- Requirement: it must also expose migration/checksum presence, completion-chain proof, exact login membership graph, old-enable blocked leases, lease expiry/recovery eligibility, and the accepted independent verification reference.
- Failure mode: an operator can receive an apparently ready state while using the wrong login/build or while stranded old-event leases exist.
- Type: rollout/status/documentation.
- Correction: expand the non-mutating repeatable-read readiness contract described below.

## LOW

### 11. Architecture documentation status is stale

- Evidence: Volume XI still says generation-52 validation is pending ([Volume XI](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/Volume_XI_Scheduler.md:1>)), while the later independent report records generation 52 active/complete, subject to deferred safe process regression ([final verification](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/06_Phase5_Operational_Completion/2026-07-31/CampaignOperations_Phase5_FinalFour_IndependentCEEVerification_Output.md:18>)).
- Type: documentation.
- Correction: ADR-0019/H1 must reconcile Volume X, XI, XII, the ADR index, database migration documentation, and Phase H operational documentation.

## NONBLOCKING_FOLLOWUP

- Process-level scheduler/global-control suites were deliberately not rerun. The accepted final verification reports they were deferred because production workers were active ([evidence](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/06_Phase5_Operational_Completion/2026-07-31/CampaignOperations_Phase5_FinalFour_IndependentCEEVerification_Output.md:20>)). This does not block the architecture decision, but it is a mandatory pre-enable rollout gate.

# Architecture decision

## 1. Verdict

Production Dispatch Admission plus a separate Campaign Manager is the correct Phase H direction. It closes the accepted durable request→Phase 5 lifecycle handoff without giving Campaign Operations scheduler claims, capacity, process, worker, or scientific authority.

The proposal is not sufficiently precise for implementation as written. It becomes incrementally implementable after the targeted corrections above, beginning with accepted ADR-0019.

## 2. Baseline inspected

Reviewed directly:

- both Phase H proposal documents;
- accepted Campaign Operations architecture and Phase G final verification;
- ADR-0010 through ADR-0018;
- migrations 048–054, with request-column origin in migration 047;
- dispatch, binding, control/recovery, completion repositories/services and canonical types;
- scheduler protocol and exact-attempt implementation;
- Phase E/F/G migration, replay, concurrency, CLI, ACL, and integration-test patterns;
- Volume X, XI, XII, ADR index, database documentation;
- current Git state.

Repository baseline: branch `campaign-operations`, HEAD `283f6a5ca5cd39c26e0f3599eff246b0b12d32fd`.

## 3. Identity contract

| Fact | Immutable canonical identity | Diagnostic only |
|---|---|---|
| Global enable | operation key, predecessor/version, exact scheduler canonical, verification reference, authorizing actor, fixed enabler capability, approved Manager service contract, exact build contract, reason | operator session ID, host, PID, invocation time |
| Global disable | operation key, predecessor/version, disabling actor, fixed disabler capability, reason | stopped process IDs or deployment logs |
| First request admission | exact request/version, dispatch operation key, enable event canonical, requesting actor, original executing service principal, build contract, dispatcher capability | candidate-scan/batch identity |
| Attempt V2 | request/admission canonicals, current enable canonical, operation key, ordinal, expected/resulting version, lease digest/expiry, requesting actor, original service principal, build contract, capability | retry count, hostname, PID, ephemeral UUID |
| Recovered unexpired lease | exactly the original Attempt V2 canonical | recovering principal/process/build and recovery time may be audited separately |
| Handoff | exact attempt canonical/ID and resulting complete binding/outcome | recovering process instance |

A different authorized manager may recover the same unexpired attempt. Its identity must not rebuild or mutate the original attempt. Changed canonical actor, reason, service, build, capability, request version, or enablement event under the same operation key is `conflicting_replay`.

Global enablement approval and per-request dispatch authorization are separate necessary authorities. Neither substitutes for the campaign’s existing exact operational authorization.

## 4. Operation-key and replay contract

- Enable and disable require caller-supplied keys.
- Exact single-request production dispatch also requires a caller-supplied key.
- Run-once derives a stable key from a versioned canonical containing immutable request identity and expected request version. Including the request hash is appropriate; process randomness is prohibited.
- Request identity plus expected version is sufficient for run-once because every committed acquisition or Phase F recovery advances request version.
- The run-once batch itself remains non-durable and may select a different snapshot on replay.
- Every per-request operation is durable.
- Exact replay of an acquisition returns its original admission/attempt/lease evidence.
- Expired-lease Phase F recovery terminalizes the old attempt and increments request version; the next acquisition is a new ordinal fact with a new version-derived operation key.
- The random opaque lease token is acceptable only because its digest is persisted and recovery does not require reconstructing the raw token. It cannot replace the operation key.

## 5. Scheduler-generation contract

Phase H V1 authorizes exactly:

```text
scheduler_protocol_evidence_v1
required_generation = 52
cutover_state = complete
exact generation-52 cutover evidence
exact migration/protocol contract identity
```

The narrow scheduler-owned security-definer function must return and, for mutation transactions, share-lock this evidence. Campaign Operations receives no raw scheduler-table access.

If generation 53 or any different protocol canonical appears:

- current enablement history remains immutable;
- its effective status becomes inactive;
- acquisition and handoff fail readiness;
- an acquired but unbound generation-52 attempt cannot hand off;
- a disable event must be recorded;
- a new accepted enablement contract version, independent verification, and new enable event are required.

Scheduler liveness remains diagnostic and non-authoritative.

## 6. Completion V1/V2 decision

Completion V2 is not required.

The complete chain is:

```text
completion_v1
  -> request_evidence_canonical
     -> exact attempt_identity_canonical
        -> complete admission canonical
        -> complete enablement canonical
```

Required H1 proof:

- Attempt V2 contains both complete nested canonicals, not only IDs/hashes.
- PostgreSQL reconstructs and validates enablement, admission, and Attempt V2 canonicals.
- C++ hydration rebuilds the same canonicals.
- C++/PostgreSQL golden vectors cover the whole nested chain.
- Same-hash/different-canonical is rejected at each level.
- Migration 055 changes the attempt-version check additively to V1-or-V2 with nullable V2 fields; existing V1 rows and completion rows are untouched.
- Completion gates include the new admission table and production audit.
- Post-completion request admission, Attempt V2 insertion, or Boolean transition is impossible.

## 7. Boolean witness versus admission authority

Authoritative hierarchy:

1. no enable event means globally disabled;
2. current exact effective enable event is global authority;
3. immutable request admission plus Attempt V2 is per-request production authority;
4. `production_dispatch_enabled` only records that the request crossed the production serialization boundary.

Required committed-state equations:

```text
production_dispatch_enabled
  == EXISTS(exact request production admission)

first admission
  => exactly one matching first Attempt V2 and audit

Attempt V1
  => production flag false
  => no admission or enablement fields

Attempt V2
  => production flag true
  => exact admission and enablement evidence
```

Read models may display the Boolean. No acquisition, handoff, replay, or recovery decision may authorize from it.

## 8. Disable/re-enable state machine

| Ordering/state | Deterministic result |
|---|---|
| Handoff obtains shared gate before disable | Handoff may commit; disable waits and then records the new head |
| Disable obtains exclusive gate first | Handoff observes disabled and performs no mutation |
| Re-enable creates E2 while lease cites E1 | E1 lease remains blocked; E2 cannot authorize it |
| E1 lease expires | Existing Phase F recovery records the old attempt outcome and returns the request to ready |
| New acquisition after recovery | New ordinal/version/key under E2; original admission remains immutable |
| Exact binding replay after disable | Returns existing identical evidence; disable does not invalidate committed history |

Disablement never cancels experiments, releases committed reservations, signals workers, rewrites bindings, or deletes evidence.

## 9. Run-once versus continuous mode

Initial production scope must stop at:

- readiness/status;
- explicit enable/disable;
- exact single-request dispatch;
- bounded sequential run-once.

Continuous mode belongs in H4 or a separate operational ADR after canary acceptance. Multi-manager database safety proves request correctness, not supervision, deployment, logging, health, or churn policy.

## 10. Common Phase E engine and test isolation

The proposal is correct to require one internal Phase E engine shared by:

- isolated-test adapter;
- production single-request adapter;
- bounded Manager.

Duplicated acquisition or handoff implementations should be rejected.

Test hooks must exist only in the isolated-test API and must not be expressible through production CLI options, Manager configuration, environment variables, or production service constructors.

## 11. Lock order and concurrency

Required order:

```text
0a exact scheduler protocol evidence, shared when mutating
0b production-enable domain
1  authorization domains
2  budget
3  campaign/completion
4  reservations ascending
5  requests ascending
6+ Phase 5/lifecycle domains
```

- Enable: protocol share lock → exclusive production gate → append.
- Disable: exclusive production gate only.
- Acquisition/handoff: protocol share lock → shared production gate → existing levels.
- Readiness/candidate selection: read-only, optimistic, no row locks.
- Phase F recovery: no production gate; it must work while disabled and uses existing campaign/reservation/request serialization.
- Pause/cancel/completion retain existing orders and never acquire the production gate after later locks.
- Scheduler claim paths never acquire Campaign Operations locks.
- `FOR UPDATE SKIP LOCKED` request-first selection is prohibited.
- No transaction may wait for a scheduler process, filesystem operation, process launch/signal, or sleep while holding Campaign Operations locks.

Independent-connection tests must prove blockers with `pg_blocking_pids()`.

## 12. In-doubt recovery

For enable, disable, acquisition, and handoff:

1. discard the uncertain connection;
2. open a fresh connection;
3. locate the exact durable operation by its operation key or attempt identity;
4. hydrate and validate the full canonical chain;
5. return identical if complete and exact;
6. permit bounded whole-transaction retry only when complete absence and unchanged expected authority are proven;
7. return ambiguity/reconciliation on partial or contradictory evidence.

Actual `pqxx::in_doubt_error` must be injected before commit, after commit/before response, and during lookup. Tests must assert one authoritative row and one audit row. Arbitrary `sql_error` must never be treated as replay success.

## 13. Privilege and deployment model

The proposed capabilities are appropriate:

- production enabler;
- production disabler;
- production dispatcher;
- production Phase 5 transactional;
- production reader;
- narrow scheduler-protocol evidence function.

Required assignments:

- enabler cannot dispatch or disable;
- disabler cannot enable or dispatch;
- Manager login receives production dispatcher, production Phase 5 transactional, and approved read access only;
- Manager lacks enabler/disabler and all isolated-test roles;
- `pqxx` gains no membership;
- migration grants no login membership;
- scheduler cannot read Campaign Operations;
- Manager cannot mutate scheduler claims, attempts, capacity, leases, ownership, or processes;
- pre-enable readiness audits direct and inherited memberships;
- emergency rollback is disable → stop Manager → revoke roles;
- historical evidence is retained.

## 14. Readiness/status and rollout

Readiness must be read-only and report:

- migration 055 filename/checksum and contract versions;
- exact generation-52 protocol canonical/hash and cutover evidence;
- independent verification reference;
- current enablement head and effective state;
- approved Manager service/build contract and actual session principal;
- direct/indirect capability readiness;
- ready admitted/unadmitted counts;
- active current-event leases;
- old-event blocked leases, expiry, and recovery eligibility;
- reconciliation-required counts;
- Completion V1 nested-V2 proof status.

Accepted rollout:

1. remain disabled through migration and executable deployment;
2. run safe scheduler/global-control regression window;
3. create/audit dedicated Manager login;
4. run readiness/status;
5. record explicit enable;
6. dispatch one exact canary;
7. inspect admission, attempt, binding, lifecycle, scheduler claim, and audit evidence;
8. run one-request run-once;
9. increase only through bounded reviewed limits;
10. consider H4 separately.

## 15. Recommended increments

- **H1 — Authority and persistence foundation:** ADR-0019, canonical contracts/golden vectors, migration 055, immutable evidence, V1/V2 compatibility, owner-DML guards, roles, readiness/status. No production handoff.
- **H2 — Exact production dispatch:** common engine refactor, enable/disable services, exact canary adapter/CLI, full in-doubt recovery, disable/handoff concurrency tests. No Manager batch.
- **H3 — Bounded run-once:** read-only candidate selection, deterministic per-request keys, sequential bounded processing, request-local continuation, global-failure stop, multi-manager tests.
- **H4 — Continuous Manager:** separate operational acceptance covering daemon behavior and deployment.

H1–H3 are independently committable and safe while no enable event or production login membership exists. H4 is independently committable only after its operating contract is accepted and with autostart disabled.

## 16. Required documentation changes

ADR-0019 must amend the Phase H roadmap and update:

- accepted Campaign Operations roadmap/traceability;
- ADR index and ADR-0010/0013/0014/0016/0017 relationships;
- Volume X Manager ownership and boundaries;
- Volume XI exact protocol evidence and non-liveness rule;
- Volume XII migration 055, constraints, roles, and canonical chain;
- database migration documentation;
- Phase E isolated-test documentation;
- a new Phase H operator runbook covering readiness, canary, disable-first rollback, and H1–H4 gates.

## 17. Implementation blockers

Implementation must not begin beyond ADR drafting until these are frozen:

- accepted ADR-0019;
- distinct actor/service/build/capability schema;
- explicit and derived operation-key rules;
- exact generation-52 canonical and upgrade behavior;
- Boolean/admission bidirectional invariant;
- old-event lease state machine and lock order;
- Completion V1 nested-chain proof;
- supplemental V1 test isolation;
- H4 exclusion.

## 18. Residual risks

- A full-materialization handoff may create a large ordinary pending workload; staged bounds mitigate this without scheduler coupling.
- The five-minute lease may be tight for large materializations; measure before changing the contract.
- Canonical hashing supplies integrity, not authentication; authenticated principals and capability membership remain authority.
- Multiple Managers are correct but can create unnecessary database churn.
- Physical retention remains a later ADR.
- Database superusers can disable triggers; that remains outside the application-role trust boundary.

## 19. Verification record

No files, SQL, documentation, roles, database rows, scheduler state, workers, backups, or processes were modified. No build or tests were run; verification was static and read-only.

Final captured `git status --short`:

```text
 M screenlog.0
?? CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Output.md
```

The transcript is untracked and ignored by `*_Transcript.txt`. `screenlog.0` was pre-existing and continued growing during the review.

Final captured `git diff --stat`:

```text
screenlog.0 | 967 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
1 file changed, 967 insertions(+)
```

Cached diff: empty.

PHASE_H_ARCHITECTURE_READY_AFTER_TARGETED_CORRECTIONS