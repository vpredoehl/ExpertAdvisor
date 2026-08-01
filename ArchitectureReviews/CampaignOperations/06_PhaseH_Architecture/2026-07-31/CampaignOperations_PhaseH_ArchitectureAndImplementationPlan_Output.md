---
title: "Campaign Operations Phase H Architecture and Implementation Plan"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_ArchitectureAndImplementationPlan_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H Architecture and Implementation Plan

# Executive Summary

Phase H should be **Production Dispatch Admission and Campaign Manager**.

The obvious alternatives are wrong:

- Physical archival is not next. Phase G already provides logical archival, while deletion/retention lacks an accepted policy and remains deliberately deferred.
- The scheduler must not poll Campaign Operations requests. That would transfer campaign policy and budget semantics into the scheduler.
- Simply permitting `production_dispatch_enabled=true` is insufficient. A mutable boolean would not record who enabled production, which scheduler protocol was accepted, or which authority admitted a specific attempt.

Migrations 051–052 already implement generation-52 scheduler fencing and exact worker-attempt authority. Phase G is complete. The remaining gap is a durable, default-off production boundary that lets a separate Campaign Manager consume Phase E requests and atomically hand them to the existing Phase 5 lifecycle workflow.

The result should be:

```text
Durable ready request
        |
        v
Campaign Manager
  - verifies global production admission
  - acquires the existing request lease
  - invokes the existing Phase E handoff
        |
        v
ordinary pending/train experiments
        |
        v
existing generation-52 scheduler
  - claims capacity and exact attempts
  - launches and supervises workers
```

The scheduler receives no Campaign Operations privileges, reads no Campaign Operations tables, and gains no new work class.

This design requires a new ADR because the accepted roadmap described Phase H primarily as scheduler-owned hardening and configuration enablement. Scheduler hardening has since landed; adding durable production admission and a manager process is the correct amendment, but it should not be smuggled in as a flag change.

# Proposed Phase H Scope

## In scope

- Durable global production-dispatch enable/disable authority.
- Immutable scheduler-protocol and independent-verification evidence attached to enablement.
- Production-mode request and attempt admission.
- A production adapter around the existing Phase E dispatch engine.
- A bounded Campaign Manager consumer, initially one request at a time.
- Explicit single-request and bounded run-once CLI commands.
- Optional continuous manager mode after canary validation.
- Explicit handling of `pqxx::in_doubt_error`.
- Production readiness and dispatch status.
- Dedicated least-privilege runtime principals.
- Default-off deployment and reversible disablement.

## Out of scope

- Scheduler polling of Campaign Operations tables.
- A campaign-specific scheduler queue, priority, capacity reservation, or work class.
- Worker launch, supervision, signaling, or running-work termination by Campaign Operations.
- Automatic grants, budgets, request acceptance, cancellation, completion, or scientific assessment.
- Partial-materialization dispatch.
- Budget refunds after binding.
- Physical archival, deletion, retention, or partition retirement.
- Adaptive budgeting, autonomous campaign selection, forecasting, or optimization.
- Redesign of Phases A–G.

## Why this belongs in Phase H

The accepted Phase H roadmap already gates production dispatch on scheduler hardening and operational approval. Migrations 051–052 now supply the scheduler half of that gate. Leaving the Phase E handoff test-only would leave the entire operational pipeline inert despite all correctness prerequisites being present.

Retention belongs later because it needs legal/operational retention periods, export semantics, referential retirement, and restore policy. Production dispatch instead closes the already-designed request-to-lifecycle path without adding scientific or scheduling policy.

# Architecture

The architecture has five distinct authorities:

| Component | Authority |
|---|---|
| Production-admission service | Enables or disables production consumption and records scheduler-readiness evidence |
| Campaign Manager | Selects bounded ready requests and invokes production dispatch |
| Phase E dispatch service | Owns leases, attempts, atomic Phase 5 handoff, bindings, control owners, and reservation commitment |
| Experiment Lifecycle | Owns experiment creation/activation and legal lifecycle state |
| Scheduler | Owns capacity, claims, attempts, processes, recovery, and completion |

The Campaign Manager is not currently a named repository component. Phase H should introduce it explicitly rather than adding Campaign Operations polling to `ExperimentScheduler`.

If “Campaign Manager” is ultimately an external controller, it must use the same service interface and database roles described here. It must not receive raw SQL access.

## Production gates

A production dispatch transaction must satisfy all of these:

1. Migration 055 is installed.
2. The scheduler protocol is generation 52 and cutover is `complete`.
3. The latest immutable global dispatch-enablement event is `enable`.
4. The running manager build matches the build contract recorded by that event.
5. The request has an active exact dispatch authorization.
6. Budget, campaign, reservation, control, cancellation, and completion gates pass.
7. The request is ready or has an exactly recoverable lease.
8. The transaction records the exact production-admission evidence used.

No one condition substitutes for another.

# Database Design

Use additive migration:

`055_campaign_operations_production_dispatch_admission.sql`

## 1. Global enablement chain

Add `campaign_operations_dispatch_enablement_event`.

Important fields:

- `dispatch_enablement_event_id`
- `previous_event_id`
- `enablement_version`
- `event_kind`: `enable` or `disable`
- `expected_prior_version`
- caller-supplied `operation_key`
- `scheduler_required_generation`, fixed at 52 for V1
- exact scheduler protocol evidence canonical/hash
- independent verification reference
- approved Campaign Manager contract/build identity
- actor, capability, reason
- enablement contract version
- canonical identity and hash
- PostgreSQL-owned `recorded_at`

Absence of an event means disabled. The first event must be `enable`; later events alternate.

Uniqueness must cover version, predecessor edge, and operation key. Exact operation-key replay returns the original event after full-canonical comparison. Changed replay conflicts.

Add a one-to-one immutable enablement audit table. Do not force this environment-wide event into the campaign-specific audit table.

## 2. Request production admission

Add `campaign_operations_request_production_admission`.

It records the first production admission of a request:

- request and campaign IDs
- exact request canonical
- enabling event ID and canonical
- expected request state/version
- manager actor/build identity
- admission contract version
- canonical identity/hash
- database time

It is unique by request. It is inserted in the same transaction as the first production lease acquisition.

The existing `production_dispatch_enabled` column becomes a one-way, rebuildable witness:

- request insertion still requires `false`;
- `false → true` is allowed only during the first production acquisition;
- a matching immutable request-admission row and V2 attempt must commit with it;
- `true → false` is prohibited;
- the boolean alone never authorizes dispatch.

Existing false rows are not backfilled or rewritten.

## 3. Production dispatch attempt V2

Extend `campaign_operations_dispatch_attempt` additively with nullable production fields:

- `production_enablement_event_id`
- `request_production_admission_id`
- exact enablement/admission canonical values
- manager build identity

Existing V1 attempts retain null production fields and their exact canonical identity.

New production attempts use attempt contract V2. Their canonical identity includes the complete enablement and request-admission evidence. This is preferable to a detached attempt-admission table because Phase G already embeds `attempt_identity_canonical` into completion evidence. Thus new production evidence becomes transitively part of the canonical completion snapshot without redefining historical completion V1 identities.

The migration must prove that this remains true. If any completion path reconstructs attempt fields instead of embedding the attempt canonical, a completion V2 must be introduced. Existing V1 completion identities must never be reinterpreted.

## 4. Status views

Add approved read models exposing:

- current global enablement state and version;
- enablement event/hash and scheduler evidence;
- whether current scheduler protocol still matches;
- ready admitted/unadmitted request counts;
- active production leases;
- latest production attempt and admission;
- reconciliation-required counts;
- approved manager contract/build identity.

Views are presentation only and cannot authorize writes.

## 5. Function and trigger changes

Add guarded functions for:

- locking and reading the scheduler protocol readiness snapshot;
- locking the global enablement domain;
- appending enable/disable events;
- production request lease acquisition;
- production handoff revalidation;
- production readiness/status.

Update the existing dispatch completeness trigger to distinguish:

- V1 isolated-test acquisition: flag false, V1 attempt, test database only;
- V2 production acquisition: flag true, complete request admission, complete V2 attempt, current enablement event.

Update the bound transition so a production handoff requires the same enablement event that authorized its attempt to remain the current enabled head.

Add completion-boundary gates to request-admission inserts. A completed campaign cannot acquire new production evidence.

## 6. Isolated-test compatibility

The old false-flag acquisition path should become database-enforced test-only:

```text
current_database() LIKE
'expertadvisor_campaign_operations_phase3_test_%'
```

The application acknowledgement remains, but the database-name rule prevents an old executable with accidentally retained role membership from exercising the V1 test transition in production.

# Scheduler Design

No scheduler schema, polling, priority, or capacity changes belong in Phase H.

Before enablement, the admission service validates:

- `experiment_scheduler_protocol.required_generation = 52`;
- `cutover_state = 'complete'`;
- complete cutover evidence exists;
- the independent verification reference is recorded.

The scheduler need not be running at the instant production is enabled. Liveness is not correctness: if it is unavailable after handoff, experiments remain ordinary durable `pending/train` work.

After handoff:

- the scheduler discovers the experiments through its existing lifecycle query;
- generation-52 claim CAS and worker-attempt authority apply;
- Campaign Operations does not wake or signal the scheduler;
- launch failure, orphaning, capacity exhaustion, and worker completion remain scheduler-owned.

The scheduler role receives no Campaign Operations privileges. The Campaign Manager receives no scheduler claim, lease, attempt, capacity, or process privilege.

# Campaign Manager Interaction

The Campaign Manager should be a Campaign Operations-owned consumer with two modes.

## Run-once mode

- Select at most `N`, where `1 <= N <= 100`.
- Order candidates by `operational_request_id`.
- Process sequentially in the initial implementation.
- Open no transaction across requests or while sleeping.
- Continue past a request-specific semantic failure.
- Stop the batch on global disablement, protocol failure, privilege failure, or database-wide failure.

Candidate selection is optimistic and read-only. It must not use `FOR UPDATE SKIP LOCKED`, because locking a request before authorization, budget, campaign, and reservation would invert the accepted lock order. Lease CAS decides the winner.

## Continuous mode

Continuous operation may be added after run-once canary validation:

- bounded batch per poll;
- configurable minimum polling interval;
- graceful shutdown stops new selection;
- no manager singleton is required;
- multiple managers are safe because request lease/version remains authority.

A manager invocation log or heartbeat table is unnecessary in V1. Request leases and attempt evidence already make manager crashes recoverable. Process-local manager metrics are diagnostic, not authority.

The manager must not automatically:

- create campaigns;
- grant authorization or budget;
- accept requests;
- cancel or complete campaigns;
- interpret scientific results.

# Repository Changes

Add production-admission domain and persistence interfaces, likely as:

- `CampaignOperationsProductionDispatch.hpp/.cpp`
- `CampaignOperationsProductionDispatchRepository.hpp/.cpp`
- `CampaignOperationsManagerService.hpp/.cpp`

Required repository operations:

- load and validate the scheduler protocol readiness snapshot;
- find the current enablement head;
- append enable/disable event and audit;
- compare full canonical on replay;
- select bounded production candidates;
- load request production admission;
- acquire a production lease and V2 attempt atomically;
- load an exact recoverable production lease;
- revalidate the exact enablement event at handoff;
- expose production readiness/status.

Refactor `CampaignOperationsDispatchService` so one internal dispatch engine is shared by:

- the existing isolated-test adapter;
- the production single-request adapter;
- the Campaign Manager.

Test hooks remain reachable only through the isolated-test adapter.

All durable-outcome reads must hydrate and validate full canonicals, not hashes alone.

# Service Changes

## Production enablement service

One short transaction:

1. Load and validate exact existing replay, if any.
2. For `enable`, lock/read scheduler protocol evidence.
3. Take the exclusive global enablement lock.
4. Reload the head.
5. Validate expected version and transition.
6. Append event and audit.
7. Commit.

Retry whole transactions for SQLSTATE `40001` and `40P01`, maximum three attempts. Handle `pqxx::in_doubt_error` by reconnecting and looking up the exact operation key/full canonical before any retry.

Disablement does not require scheduler readiness and must be available through a narrower emergency capability.

## Production dispatch service

Keep the existing two-transaction Phase E structure.

Acquisition additionally:

- takes the shared production gate before authorization;
- validates generation 52 and current enablement;
- creates the request admission when needed;
- sets the request witness true;
- writes a V2 attempt containing the exact enablement evidence.

Handoff additionally:

- reacquires the shared production gate first;
- requires the V2 attempt’s enablement event to remain the active head;
- otherwise refuses the handoff and leaves the lease for Phase F recovery.

All existing authorization, budget, campaign, reservation, request, Phase 5, binding, control-owner, and completion checks remain.

## In-doubt handling

Productionization must explicitly catch `pqxx::in_doubt_error`; it is not a `sql_error` or `broken_connection`.

- Acquisition uncertainty: reconnect and validate exact request version, lease digest, V2 attempt, admission, and audit.
- Handoff uncertainty: reconnect and validate the complete binding/reservation/request/outcome canonical.
- Proven absence permits a bounded retry.
- Partial or contradictory evidence returns `commit_outcome_unknown` or `reconciliation_required`.
- Never classify arbitrary SQL errors as replay success.

# CLI Changes

Recommended commands:

```text
--campaign-operations-production-readiness
```

Read-only. Reports migration, scheduler protocol, enablement, build contract, role, and request-count blockers.

```text
--campaign-operations-production-enable OPERATION_KEY
--campaign-operations-expected-enablement-version N
--campaign-operations-scheduler-verification-reference REF
--campaign-operations-manager-build BUILD_ID
--campaign-operations-actor ACTOR
--campaign-operations-reason REASON
--yes
```

```text
--campaign-operations-production-disable OPERATION_KEY
--campaign-operations-expected-enablement-version N
--campaign-operations-actor ACTOR
--campaign-operations-reason REASON
--yes
```

```text
--campaign-operations-production-status
```

Read-only.

```text
--campaign-operations-dispatch-request REQUEST_ID
--campaign-operations-expected-request-version N
--campaign-operations-actor ACTOR
--yes
```

Production canary and exact operational recovery command.

```text
--campaign-operations-manager-run-once
--campaign-operations-dispatch-limit N
--campaign-operations-actor ACTOR
--yes
```

A continuous `--campaign-operations-manager` with polling interval should be introduced only after the run-once rollout passes.

Mutation commands reject `--dry-run`; readiness replaces speculative mutation. Status/readiness reject `--yes`.

# PostgreSQL Invariants

PostgreSQL must enforce:

1. Absence of enablement evidence means production disabled.
2. Enablement events form one fork-resistant alternating chain.
3. Exact operation-key replay returns one identical event; changed replay conflicts.
4. Enable requires generation-52 completed scheduler protocol evidence.
5. Enablement and its audit commit atomically.
6. Enablement history is insert-only and rejects update, delete, and truncate, including owner-level ordinary DML.
7. Requests still begin with `production_dispatch_enabled=false`.
8. The witness can transition only false→true and only with complete request admission plus a V2 attempt.
9. The witness can never return to false.
10. A V1 attempt has no production fields and is allowed only in the isolated-test database namespace.
11. A V2 attempt has complete request, enablement, build, canonical, and audit evidence.
12. Production handoff requires the attempt’s enablement event to remain the current enabled head.
13. Disablement blocks new acquisition and handoff but never deletes leases, bindings, experiments, or history.
14. Request admission and production attempts are rejected after campaign completion.
15. Canonical text is authoritative; hashes are lookup/integrity aids only.
16. Existing V1 request, attempt, binding, completion, and audit identities retain their original meaning.
17. Scheduler roles cannot read Campaign Operations tables.
18. Campaign Operations roles cannot claim scheduler work, reserve scheduler capacity, or mutate worker attempts.

# Concurrency Analysis

Add these levels ahead of the accepted Campaign Operations order:

```text
0a scheduler protocol row, read-only when required
0b global production-enable domain
1  authorization domains
2  budget
3  campaign/completion
4  reservations, ascending ID
5  requests, ascending ID
6+ existing Phase 5/lifecycle domains
```

Enable takes the protocol row and exclusive production-domain lock. Dispatch takes shared production-domain authority before authorization.

Race outcomes:

| Race | Result |
|---|---|
| Enable vs acquisition | Acquisition sees either disabled or the fully committed enable event |
| Disable vs acquisition | Winner is serialized; disable-first blocks acquisition |
| Disable vs handoff | Handoff-first may commit; disable-first blocks handoff |
| Re-enable vs old lease | Old lease remains tied to the old event and cannot hand off |
| Two managers | One request lease CAS wins; loser reloads authoritative evidence |
| Pause/cancel/completion vs dispatch | Existing campaign/request locks and gates decide |
| Authorization revoke vs dispatch | Existing authorization-domain order decides |
| Reservation expiry vs acquisition | Existing reservation/request locking and PostgreSQL time decide |
| Scheduler claim vs bound cancellation | Existing lifecycle/scheduler exact-attempt authority decides |

No transaction may hold Campaign Operations locks while waiting for a scheduler process or performing filesystem/process work.

# Replay Analysis

- Enable/disable replay is keyed by operation key and full canonical.
- Request production admission is uniquely identified by the exact request and first enablement event.
- Every production attempt is a new ordinal V2 fact under the request lock.
- A recovered lease reuses its exact attempt and enablement evidence.
- An existing complete binding returns `existing_identical` without Phase 5 reinvocation.
- Changed actor, reason, verification reference, scheduler evidence, build identity, request version, or enablement event conflicts.
- Hash equality never decides replay.
- A Campaign Manager batch is intentionally not a durable logical operation. Re-running it may select different currently ready requests. Determinism applies to each request operation, not to an opportunistic queue snapshot.
- Completion binds the V2 attempt canonical transitively. Historical V1 completion identities remain unchanged.

# Failure Recovery

- **Crash before acquisition commit:** no admission, flag, lease, attempt, or audit survives.
- **Crash after acquisition commit:** any authorized manager can recover the exact unexpired lease.
- **Disable after acquisition:** handoff is refused; Phase F later records and safely recovers the expired lease to ready.
- **Crash inside Phase 5 handoff:** the downstream execution, activation, experiments, bindings, owners, reservation commitment, request transition, outcome, and audit roll back together.
- **Lost handoff acknowledgement:** reload the complete authoritative binding before retry.
- **Partial downstream evidence:** fail closed as reconciliation-required; do not reconstruct or re-run Phase 5 creatively.
- **Scheduler unavailable after binding:** experiments remain pending; this is not a dispatch rollback condition.
- **Scheduler launch/worker failure:** scheduler lifecycle and attempt recovery remain authoritative.
- **Enable/disable uncertain commit:** stop production activity until full-canonical lookup proves the event or its absence.
- **Manager termination:** finish or abandon the current short transaction; select no new work. Existing leases supply restart authority.
- **Emergency rollback:** commit a disable event, stop manager processes, then revoke production roles. Never delete admissions or reverse committed bindings.

# Privilege Model Changes

Create separate NOLOGIN capabilities:

- `campaign_operations_production_enabler`
- `campaign_operations_production_disabler`
- `campaign_operations_production_dispatcher`
- `campaign_operations_production_phase5_transactional`
- `campaign_operations_production_reader`
- a narrowly owned security-definer scheduler-protocol evidence function

Rules:

- The disabler cannot enable.
- The enabler cannot dispatch.
- The manager login receives only the two production dispatch capabilities and approved reads.
- Production roles do not inherit the isolated-test transition capability.
- `pqxx` receives no new membership.
- No login membership is granted by migration 055.
- Scheduler-protocol access is through a pinned, narrowly scoped function; production roles receive no general scheduler table access.
- Functions use pinned `search_path`, expected-state predicates, and revoked PUBLIC execution.
- Deployment must audit existing direct and indirect membership in the Phase E test roles before enabling production.

# Migration Strategy

1. Accept a new ADR amending Phase H to cover durable admission and the Campaign Manager.
2. Freeze V1/V2 canonical grammars and golden vectors.
3. Back up the database through the normal operational process.
4. Apply migration 055 with production absent and disabled.
5. Leave every existing request false; do not backfill admissions.
6. Prove migration replay applies zero changes on the second run.
7. Deploy the new executable while no Campaign Manager is running.
8. Verify generation-52 cutover and run the deferred scheduler/global-control process suites in a safe window.
9. Create a dedicated manager login and grant only the reviewed production roles.
10. Run readiness and production status.
11. Commit an explicit enable event.
12. Dispatch one exact canary request.
13. Verify binding, pending lifecycle state, scheduler claim behavior, status, and audit.
14. Run a bounded one-request manager batch, then increase gradually.
15. Enable continuous manager operation only after canary evidence is accepted.

Rollback is disable-first. Do not roll back migration 055, scheduler generation 52, or persisted Phase H evidence.

# Testing Plan

## Pure tests

- Enablement and request-admission canonical golden vectors.
- Attempt V1/V2 shape and canonical distinction.
- UTF-8 byte framing, delimiters, large evidence, locale independence, UTC precision.
- Exact/conflicting enable, disable, admission, and dispatch replay.
- Illegal transition matrix.
- Completion canonical inclusion of V2 attempt evidence.

## Migration tests

- Clean supported installation and 054→055 upgrade.
- Idempotent migration replay.
- Existing V1 requests/attempts/completions remain byte-identical.
- No request flag backfill.
- Direct malformed insert/update/delete/truncate rejection.
- Deferred admission/attempt/audit completeness.
- Test-path rejection outside the test database prefix.
- Completion-boundary rejection.
- Role ownership, NOLOGIN, sequence, column, NULL ACL, function-owner, and search-path assertions.
- No grant to `pqxx`.

## Repository/service tests

- Enable/disable exact and changed replay.
- Actual `pqxx::in_doubt_error` before and after commit.
- Broken connection and unique-winner recovery.
- First production admission and later production retry.
- Old-enable lease rejected after disable/re-enable.
- Full binding replay without Phase 5 reinvocation.
- Partial downstream evidence and corruption rejection.
- Bounded `40001`/`40P01` retries and exhaustion.

## Independent-connection concurrency

- Enable versus acquisition.
- Disable versus acquisition and handoff.
- Two and more Campaign Managers selecting the same requests.
- Pause, cancellation, authorization revocation, reservation expiry, and completion versus production dispatch.
- PostgreSQL blocker assertions using `pg_blocking_pids()`, not timing sleeps.
- Lock-order and deadlock tests across scheduler protocol, production gate, and existing Campaign Operations locks.

## Scheduler integration

Using disposable databases and fake worker/process seams:

- handoff produces only ordinary `pending/train`;
- generation-52 scheduler claims exactly once;
- capacity accounting remains unchanged;
- claim versus cancellation;
- launch failure, orphan recovery, stale fence, shutdown, and restart;
- scheduler cannot read Campaign Operations;
- Campaign Manager cannot access scheduler claim/attempt/process mutation.

Do not run process-level suites against active production scheduler or workers without the required safety check.

## Regression

Run Phases A–G tests, Phase 4C/5 launch tests, scheduler ownership suites, global controls, migration replay, Release build, strict focused builds, CLI parser suites, and documentation/ADR consistency checks.

# Risks

- **Roadmap drift:** durable admission and a manager exceed the old “scheduler-owned only” Phase H wording. Mitigation: accept an ADR first.
- **Boolean-as-authority regression:** code may consult only `production_dispatch_enabled`. Mitigation: make admission rows and current enablement authoritative; test the boolean as insufficient.
- **Completion evidence omission:** new facts could escape the canonical snapshot. Mitigation: V2 attempt canonical must contain the complete admission chain; otherwise create Completion V2.
- **Disable misunderstood as worker stop:** disable blocks future Campaign Operations handoff only. It does not stop already-bound or running experiments.
- **Queue burst:** one full materialization may add many pending experiments. Mitigation: sequential manager, small batches, staged rollout; do not invent scheduler capacity coupling.
- **Mixed-version bypass:** an old executable may retain Phase E role membership. Mitigation: database-enforced test database restriction and pre-enable membership audit.
- **Long handoff:** large materializations may approach the five-minute lease. Measure before changing the lease; do not extend it casually.
- **Verification reference authenticity:** canonical hashing provides integrity, not identity authentication. The enabler database role remains the actual authority.
- **Active production processes:** final scheduler/global-control process regression remains operationally sensitive.
- **Manager overload:** multiple managers are correct but can create database churn. Correctness does not imply unlimited deployment concurrency.
- **Append-only growth:** real, but physical retention remains a later ADR.

# Step-by-step Implementation Plan

1. Draft and accept ADR-0019 for production admission, Campaign Manager ownership, default-off deployment, and rollback.
2. Freeze enablement, request-admission, and attempt-V2 canonical contracts.
3. Add pure C++ types, validation, transition rules, and golden tests.
4. Implement migration 055 tables, roles, functions, triggers, views, and V1 compatibility.
5. Prove Phase G completion evidence captures the complete V2 attempt canonical.
6. Add production-admission repository interfaces and corruption validation.
7. Refactor the Phase E service into one common engine with isolated-test and production adapters.
8. Add explicit `in_doubt_error` recovery using fresh connections and full-canonical lookup.
9. Implement the production enable/disable and readiness services.
10. Implement the exact single-request production command.
11. Implement a sequential bounded Campaign Manager run-once service.
12. Add CLI parsing, validation, help, machine output, and stable exit codes.
13. Complete pure, migration, ACL, repository, concurrency, replay, and fault-injection tests.
14. Run the complete Phase A–G and scheduler regression matrix.
15. Obtain an independent CEE architecture/implementation review.
16. Deploy migration and executable while disabled.
17. Verify scheduler generation 52 and process-level regressions in a safe window.
18. Grant dedicated production roles, enable explicitly, and dispatch one canary.
19. Review canary evidence before enabling bounded manager polling.
20. Add continuous manager mode only after the run-once operating contract is proven.

No files were changed for this design. No build or tests were run because implementation was explicitly out of scope. Current repository state is `M screenlog.0`; `git diff --stat` reports `156` appended lines in that pre-existing live scheduler log. `git diff --check` reports its existing CRLF/trailing-whitespace content.