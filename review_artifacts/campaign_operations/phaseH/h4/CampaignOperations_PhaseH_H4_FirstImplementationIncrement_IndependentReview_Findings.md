# Campaign Operations Phase H H4 First Implementation Increment — Independent Review Findings

Date: 2026-08-09
Review type: Independent implementation review
Result: READY_FOR_H4_FIRST_INCREMENT_TARGETED_CORRECTION

## Scope

Independent review of the first H4 implementation increment against accepted ADR-0020, using the bundled H4 supervisor, fixture tests, H4 runbook, deployment artifacts, H3 contract documentation/tests, and relevant H3/H1 source contract excerpts.

The supplied 10-test H4 fixture suite was independently executed in an isolated sandbox and passed. Additional adversarial probes were then run against the supervisor implementation.

## Overall assessment

The first increment preserves the selected H4 architecture: external deployment-owned supervision around the existing bounded H3 command, immediate H1 readiness preflight, no new database authority, no scheduler/worker control, and finite retry/shutdown policy.

It is not ready to commit. The independent review found implementation defects requiring a targeted correction. ADR-0020 itself does not need redesign.

## Findings

### HIGH — Readiness classification is not deterministic

`Supervisor.readiness()` searches the entire real readiness record for `principal` or `role` before checking build and scheduler failure.

The H1 readiness record normally contains fields including `session_principal` and `current_principal`. Therefore a blocked readiness result such as a build mismatch can be incorrectly classified as `privilege_failure`.

Adversarial reproduction: a valid readiness record containing `ready=false`, `build_comparison=mismatch`, `session_principal=manager`, and `current_principal=manager` was classified as `privilege_failure` instead of `manager_build_not_ready`.

Required correction: classify from stable structured readiness fields/blockers with explicit precedence matching ADR-0020. Do not infer privilege failure merely from field names that occur in every readiness record.

### HIGH — Restart state is not fully fail-closed

`restore()` converts `retry_count` using `int(...)` without validating its type/range. A malformed persisted value such as `"oops"` raises `ValueError` and crashes the supervisor rather than selecting `STOP_MALFORMED_RESULT`.

A record marked `complete=true` can also restore a valid `next_action` without proving that the required prior normalized classification is present and valid.

Required correction: define and validate the complete persisted-state schema, including classification, action, retry count, deployment/database/environment identities, terminal-result state, and any fields required by a restored action. Any malformed, missing, out-of-range, contradictory, or unknown value must fail closed to `STOP_MALFORMED_RESULT`.

### HIGH — Finite numeric configuration is not enforced

Python's JSON decoder accepts non-finite floating-point values. `NaN` and `Infinity` can pass the current positive-number checks for interval, drain timeout, or backoff.

Adversarial probes confirmed acceptance of non-finite values.

Required correction: explicitly require `math.isfinite()` for all floating-point timing/backoff values, in addition to positivity and any accepted bounds.

### HIGH — Shutdown has a pre-launch race

The main loop checks `shutdown_requested`, then performs state work before `_run()` creates the H3 process. A signal arriving after the check but before `Popen()` can therefore start a new H3 invocation after shutdown was requested.

Required correction: close the launch boundary so a shutdown request observed before child creation suppresses the launch. Add a deterministic test that injects shutdown at the last pre-Popen boundary.

### HIGH — Deployment execution/database identity is not sufficiently explicit

The launchd plist does not declare the intended execution identity/domain, and the example connection environment explicitly identifies host/database but not the reviewed PostgreSQL login/service identity.

Ambient/default connection identity must not decide the production Manager principal.

Required correction: make the reviewed deployment identity and exact database connection principal/service binding explicit and validated, without placing credentials in argv/logs. Preserve the existing H1-H3 role/ACL architecture; do not add database authority.

### MEDIUM — H3 request-set consistency is incomplete

The classifier verifies that each request ID occurs among candidate IDs, but does not require request IDs to be unique or to correspond exactly to the processed candidate prefix/set.

Adversarial reproduction: candidates `1:2`, processed `2`, with two request records both for request `1` was accepted as a valid work completion.

Required correction: enforce the exact H3 sequential result relationship required by the H3 contract: unique request records, exact processed count, and request IDs matching the first `processed` candidate IDs in order.

### MEDIUM — STOP/restart operator resolution is underspecified in implementation/runbook

Persisted STOP actions are restored and stop again, while the runbook says correction plus intentional restart is required. There is no reviewed mechanism described for clearing/resolving deployment-owned STOP state after operator resolution.

Required correction: define a narrow deployment-owned operator procedure for resolving/resetting H4 operational state after investigation. It must not infer or repair database truth and every subsequent launch must still pass configuration and immediate H1 readiness.

### MEDIUM — launchd crash-restart behavior is incomplete

The plist uses `RunAtLoad=true` and `KeepAlive=false`. This starts the supervisor when loaded but does not itself provide continuous recovery after an unexpected supervisor crash.

Required correction: align the launchd lifecycle configuration with ADR-0020's continuous-operation/restart policy, while preventing aggressive restart loops and preserving persisted fail-closed classification.

### MEDIUM — Health fields can overstate observed state

`duplicate_drift_detected` is always `false` even though this first increment does not actually observe duplicate drift. `service_alive` remains true in persisted STOP records. `next_scheduled_invocation` is recorded as `now()` rather than the actual future scheduled time.

Required correction: do not report an unobserved condition as verified false; make service/stopped state truthful; record an actual future scheduled time or a semantically accurate equivalent.

### MEDIUM — Durability/evidence implementation is weaker than the ADR wording

State file contents are fsynced before `os.replace`, but the containing directory is not fsynced. Captured child output is written without an explicit fsync. Classifier decisions/state transitions are represented primarily by overwritten current state rather than an append-only durable transition trail.

Required correction: either strengthen implementation durability to satisfy ADR-0020's durable capture/state-transition requirement or explicitly implement a reviewed durable transition/evidence mechanism. Do not create database evidence or authority.

## What passed

- Existing H4 fixture suite: 10/10 PASS in isolated execution.
- External supervisor architecture preserved.
- Exact bounded H3 command reused.
- Immediate readiness is attempted before permitted H3 invocations.
- No C++/schema/ACL/migration/scheduler authority was added.
- No database singleton, heartbeat, lease, or leader election was introduced.
- Finite retry budget/backoff structure exists.
- H3 remains the request/database correctness authority.
- Active experiment scheduler/training/inference processes are outside H4 control.

## Required next gate

Do not commit the first H4 implementation increment yet.

Perform a tightly scoped targeted correction addressing the findings above, add adversarial regression tests, rerun existing H4/H3 structural validation, and then perform focused independent reverification.

READY_FOR_H4_FIRST_INCREMENT_TARGETED_CORRECTION
