# ADR-0020: Campaign Manager continuous operation

Status: Accepted
Date: 2026-08-09
Deciders: Project architecture
Affected volumes: Volume I §16.1; Volume X §§2–11; Volume XI §§2–11; Volume XII §§2–11
Supersedes: None
Superseded by: None

## 1. Context and problem statement

ADR-0019 deliberately limits the Campaign Manager to bounded, sequential
operation and excludes continuous operation. H1 establishes default-off
production admission, readiness, immutable evidence, and sealed roles. H2
adds caller-driven production dispatch through the common Phase E engine. H3
adds exactly one bounded command:

    LSTM_Release --campaign-operations-manager-run-once LIMIT --yes

H3 selects one read-only REPEATABLE READ candidate snapshot, orders candidates
by operational_request_id, processes at most the supplied positive limit
sequentially, and exits. It has no daemon, polling/sleep loop, autostart,
signal handling, batch row, Manager claim, global singleton, or
process-lifetime database ownership. Existing deterministic request/version
operation identity, request-version/lease compare-and-set, immutable evidence,
exact replay, and transaction/lock contracts already make concurrent H3
Managers safe at the request boundary.

Those correctness contracts do not decide cadence, process ownership, exit
classification, restart limits, health, logging, shutdown, deployment identity,
or operational rollback. H4 therefore needs a separate operational contract.
This proposal supplies that contract without changing an H1–H3 runtime,
schema, command, role, ACL, migration, or database contract.

## 2. Decision

The following are new H4 decisions proposed by this ADR. They are
implementation authority only if this ADR is independently accepted.

### 2.1 Continuous-operation model and boundaries

Continuous Campaign Manager operation is an **external, deployment-owned
supervisor** repeatedly invoking only the existing H3 bounded command for one
reviewed deployment scope. One H3 invocation remains the fundamental bounded
work unit. The supervisor starts no later invocation until the preceding child
has exited.

The supervisor MUST NOT duplicate request selection, acquisition, handoff,
retry, replay, operation-key, lease, fencing, or scheduler-control logic. It
MUST NOT hold a database transaction, connection, lock, claim, or lease across
a wait or process lifetime. It introduces no in-process continuous command,
daemon, scheduler-owned campaign polling, database singleton, heartbeat,
Manager lease, invocation row, leader election, or process-lifetime fence.

H3 request/dispatch/database correctness remains authoritative. H4 neither
changes H3 candidate predicates/order/limit nor alters the common Phase E
engine, transaction retry, uncertain-commit recovery, request-local
continuation, global-stop behavior, or existing claim/fencing semantics.

### 2.2 Deployment configuration and scope

Before schedule activation, the deployment-owned configuration MUST validate:

- the exact existing executable, secure connection configuration, and command
  LSTM_Release --campaign-operations-manager-run-once LIMIT --yes;
- a positive LIMIT no greater than H3's accepted maximum of 100;
- a nonzero normal interval;
- finite graceful-drain timeout, finite retry budget/threshold, and bounded
  backoff policy;
- durable stdout/stderr capture and an external retention/rotation policy; and
- one deployment identity and reviewed scope.

Missing, malformed, unsafe, substituted, or out-of-range configuration MUST
fail closed as STOP_INVALID_CONFIGURATION; an application default or an
unreviewed environment substitution MUST NOT repair it. Credentials/secrets
MUST NOT appear in command lines or logs.

The normative scope is **exactly one H4 supervisor instance per target
PostgreSQL database, deployment environment, and reviewed Campaign Operations
deployment identity**. The service manager or deployment platform MUST use its
own unit/job/identity controls to prevent ordinary duplicate launch. This is a
deployment invariant, not database authority.

If duplicate drift is observed, the deployment owner MUST surface it in health
and logs and stop or suppress duplicate launching according to its defined
deployment action. Any overlap already in progress relies on H3's existing
multi-Manager correctness. Such tolerance is a correctness backstop, not
normal operation, and does not authorize a database singleton, heartbeat,
lease, leader-election row, PID table, advisory-lock ownership scheme, or
process-lifetime fence.

### 2.3 Readiness immediately before every invocation

Valid supervisor configuration is required before schedule activation. The
supervisor MUST perform existing **aggregate, read-only H1 production
readiness** immediately before every permitted H3 invocation, including the
first startup invocation and every invocation after normal sleep, a no-work
result, retry/backoff, supervisor restart, or a permitted prior abnormal exit.
A readiness failure prevents that invocation from starting and receives the
matrix action in §2.5. A failure while sleeping is discovered at the next
required pre-invocation check.

The three signals are distinct:

| Signal | Authority and use |
|---|---|
| H1 aggregate readiness | Supervisor pre-invocation authorization/preflight; read-only and required immediately before launch. |
| H3 transactional production gates | Request-time database authority for acquisition/handoff; final and fail-closed. |
| Supervisor health/liveness | Deployment diagnostic only; never authorization. |

Pre-invocation readiness does not replace H3 transactional authority, and H3
checks do not remove the pre-invocation readiness requirement. If readiness
changes after preflight but before or during H3, the H3 database gates remain
final and fail closed under H1–H3. The supervisor has no authority to change
enablement, roles, scheduler state, requests, or durable evidence.

### 2.4 Narrow deployment-owned output classifier

H4 authorizes a narrow deployment-owned **output classifier**. For each child,
it reads only the child exit status and captured existing H3 machine-readable
stdout/stderr. It validates the expected H3 terminal-result shape, maps the
invocation to exactly one row of §2.5, and durably records the classification
and selected next action.

The existing H3 terminal-result contract is:

- successful or classified-stop completion emits exactly one stdout
  CAMPAIGN_OPERATIONS_MANAGER_RUN_ONCE record with dispatch_limit,
  candidates_selected, processed, stopped_early, stop_reason, and
  candidate_request_ids;
- each processed candidate emits one stdout
  CAMPAIGN_OPERATIONS_MANAGER_REQUEST record; and
- a classified global stop emits exactly one stderr
  CAMPAIGN_OPERATIONS_MANAGER_STOPPED record with the stable reason and
  diagnostic, and exits 2; a normal completion exits 0 with stop_reason=none.

"Terminal-result" describes the completed H3 result, not a promise that the
summary is physically the final output line. A valid record requires the
literal record name, each required field once, a known stop_reason, exit status
consistent with that reason, and internally consistent counts and request
records. A positively identified process interruption is not a completed H3
result: absence of a normal terminal-result is expected and §2.5 classifies it
before terminal-result validation. Otherwise, when the child has reached a
state requiring a completed H3 result, the classifier MUST treat a missing
record, malformed fields, multiple summary/stop records, conflicting
stdout/stderr reason, inconsistent counts, or any unrecognized output/result as
fail-closed. It MUST NOT treat such output as no-work.

The classifier owns no request, dispatch, or database authority. It performs no
request retry, creates no operation key, mutates no H1–H3 evidence, and does
not infer durable correctness from logs where PostgreSQL evidence is
authoritative. It MUST NOT be a second dispatch engine, retry engine, database
authority, or state-repair mechanism.

### 2.5 Exhaustive supervisor outcome/action matrix

The finite action set is:

- CONTINUE_AFTER_NORMAL_INTERVAL — schedule only after the validated normal
  interval; the next launch still requires §2.3 readiness.
- BACKOFF_AND_RETRY_WITH_READINESS — consume the finite retry budget, wait the
  configured bounded backoff, then run §2.3 before any launch.
- STOP_DISABLED — stop launches pending a separately authorized effective
  enablement and operator/deployment restart.
- STOP_DEGRADED_OPERATOR_REQUIRED — stop launches and require operator
  investigation; configuration MUST NOT override it.
- STOP_INVALID_CONFIGURATION — stop before launch until corrected and
  revalidated.
- STOP_MALFORMED_RESULT — stop and require operator review of the child
  result/capture; it MUST NOT fall through to retry.
- RESTORE_PERSISTED_NEXT_ACTION — only on supervisor restart with a complete
  deployment-owned durable prior classification/action, reapply that action
  without reclassifying database truth; a resumed retry still requires
  configuration and §2.3 readiness.

Each outcome maps to exactly one action:

| Normalized outcome class, determined only from preflight, restart state, or H3/process result | Required next action |
|---|---|
| Valid exit 0, valid terminal-result, one or more requests processed, no request-local semantic failure | CONTINUE_AFTER_NORMAL_INTERVAL |
| Valid exit 0, valid terminal-result, zero candidates/zero processed (valid no-work) | CONTINUE_AFTER_NORMAL_INTERVAL |
| Valid exit 0, valid terminal-result containing one or more H3 request-local semantic failures | CONTINUE_AFTER_NORMAL_INTERVAL |
| Valid H3 global stop production_disabled, including aggregate readiness showing global production disablement | STOP_DISABLED |
| Valid H3 global stop scheduler_protocol_ineffective, including aggregate readiness effective-enable/scheduler-protocol mismatch | STOP_DEGRADED_OPERATOR_REQUIRED |
| Valid H3 global stop manager_build_not_ready, including aggregate readiness/build mismatch | STOP_DEGRADED_OPERATOR_REQUIRED |
| Valid H3 global stop privilege_failure, including aggregate readiness authorization/role failure | STOP_DEGRADED_OPERATOR_REQUIRED |
| Valid H3 database_failure that is not classified below as indeterminate, and aggregate readiness database-wide failure | BACKOFF_AND_RETRY_WITH_READINESS |
| Connectivity/transport failure before a valid H3 result, including connection loss or unavailable database | BACKOFF_AND_RETRY_WITH_READINESS |
| H3-exposed commit-unknown, indeterminate, reconciliation-required, or equivalent uncertainty diagnostic, including a database_failure result carrying that diagnostic | STOP_DEGRADED_OPERATOR_REQUIRED |
| Positively identified child crash, forced child termination, or equivalent externally proven process interruption; absence of a normal terminal-result is expected | BACKOFF_AND_RETRY_WITH_READINESS |
| Missing, malformed, duplicated, conflicting, or internally inconsistent H3 terminal-result where the child otherwise reached a state requiring a completed H3 result | STOP_MALFORMED_RESULT |
| Process state cannot be positively classified | STOP_MALFORMED_RESULT |
| Supervisor restart with valid configuration and a complete deployment-owned durable prior classification/action | RESTORE_PERSISTED_NEXT_ACTION |
| Supervisor restart with valid configuration but without a complete deployment-owned durable prior classification/action | STOP_MALFORMED_RESULT |
| Repeated identical or consecutive retryable failure reaches its configured finite threshold/budget | STOP_DEGRADED_OPERATOR_REQUIRED |
| Invalid supervisor configuration | STOP_INVALID_CONFIGURATION |
| Any otherwise unrecognized outcome | STOP_MALFORMED_RESULT |

The rows are ordered by specificity and are non-overlapping. Invalid
configuration takes precedence before any launch or restart restoration. With
valid configuration, a valid H3 classified result takes precedence over its
exit code, and an indeterminate diagnostic takes precedence over generic
database failure. A positively identified child crash, forced child
termination, or equivalent externally proven process interruption is its own
normalized outcome class: the absence of a normal terminal-result is expected
and that class takes precedence over terminal-result validation. Terminal-result
invalidity applies only when the child otherwise reached a state requiring a
completed H3 result. If the process state cannot be positively classified, it
is STOP_MALFORMED_RESULT. No MUST-stop class is retryable or overrideable by
configuration, and no outcome silently falls through to ordinary retry.

On supervisor restart, the supervisor MUST restore only a complete
deployment-owned durable prior classification/action. It MUST NOT reconstruct
workflow truth from logs, infer a prior durable action from partial output, or
silently auto-retry. If no complete durable prior classification/action exists,
it MUST select STOP_MALFORMED_RESULT and require operator review before any
further launch. After operator resolution, any later permitted launch still
requires validated configuration and immediate §2.3 readiness. This rule does
not create a database durability mechanism; the prior classification/action is
deployment-owned operational state.

Every retryable class requires a deployment-configured finite consecutive or
repeated-failure threshold/equivalent finite budget and bounded backoff. On
reaching it, the matrix's threshold row applies. The count resets only after a
valid, readiness-authorized H3 normal completion
(CONTINUE_AFTER_NORMAL_INTERVAL); no-work is such a completion. It does not
reset after malformed output, a preflight failure, or a merely restarted
supervisor. MUST-stop classes bypass the budget and stop immediately.

### 2.6 Emergency disable, H4 rollback, and planned maintenance

These are separate procedures:

| Procedure | Required sequence and purpose |
|---|---|
| Emergency production disable | Use the existing immutable global disable event first where accepted architecture requires it. Its purpose is immediate removal of production database authority. Then prevent future supervisor launches and stop/drain processes as appropriate. Process stopping is secondary and does not revoke database authority. |
| H4-only rollback | Disable/remove the external supervisor deployment and prevent its future launches. Leave H1–H3, data, schema, and evidence intact. A global disable event is not required merely to uninstall H4 if production is intended to remain otherwise operable; bounded H3 run-once remains available when readiness and authorization permit. |
| Planned maintenance | Prevent future supervisor launches, then gracefully drain/finish the active bounded H3 invocation. Resume only after configuration and immediate pre-invocation readiness revalidation. This is neither an emergency disable nor permanent H4 rollback. |

Emergency production disable remains consistent with ADR-0019's disable-first
rollback: when production authority itself is being withdrawn, commit disable
before process/role actions. None of these procedures deletes or rewrites
immutable evidence, and H4-only rollback does not create a data/schema rollback.

### 2.7 Shutdown and timeout semantics

On a shutdown or maintenance-stop request, no new H3 invocation MAY begin. If
no child is active, the supervisor stops promptly. If an H3 run-once child is
active, the supervisor allows normal completion. The graceful-drain timeout is
deployment-configured, finite, and validated. Forced termination is exceptional
rather than routine; after grace expires, the supervisor MAY escalate only
under the configured and accepted deployment policy.

Any forced child termination is classified exactly as an abnormal H3 process
interruption in §2.5. H4 introduces no partial-batch rollback, repair,
cancellation, or cleanup workflow. If termination occurs during candidate
snapshot, acquisition, handoff, or uncertain commit, existing H1–H3
transaction/rollback/replay/lease/recovery semantics remain authoritative. The
supervisor MUST NEVER rewrite durable Campaign Operations state to make a
shutdown appear clean.

### 2.8 Health, logging, and authority

The external supervisor record/health surface MUST expose:

- deployment identity and target database/environment/reviewed scope;
- service alive state, duplicate-instance drift state, and graceful
  shutdown/drain state when applicable;
- child invocation start/end time and exit status;
- readiness check time/result and readiness blocker/diagnostic when failed;
- whether the H3 terminal-result was valid, missing, malformed, or conflicting;
- normalized outcome classification, chosen next action, and next scheduled
  invocation time when applicable;
- whether work occurred, explicit last valid work result, and explicit last
  valid no-work result;
- consecutive/repeated failure count; and
- degraded/stopped state.

It MUST durably capture stdout/stderr, classifier decisions, and supervisor
state transitions. Credential secrecy is mandatory; rotation, retention,
aggregation, alert transport, and deployment health implementation remain
external responsibilities.

Supervisor records and logs are operational evidence only. PostgreSQL H1–H3
durable evidence remains authoritative for production authority, request truth,
commit, rollback, replay, and recovery. Logs cannot authorize dispatch or prove
rollback/commit by themselves. Lack of a supervisor record is a
deployment-health concern, not authority to invent database state.

### 2.9 Compatibility and non-goals

This proposal preserves all accepted H1–H3 default-off admission, exact
enablement, scheduler generation/cutover, build, sealed-role, Phase E,
transaction, immutable-evidence, request/lease, replay, recovery, and scheduler
boundary contracts. The external service remains separate from the experiment
scheduler and does not supervise training, inference, analysis, workers,
claims, capacity, or process lifecycle.

H4 does not decide high availability, dynamic scaling, cross-host singleton,
new database telemetry, automatic enablement/disablement, worker control,
partial interruption, in-process signal mechanics, numeric policy defaults, or
a new command. No database coordination is implied by the deployment
single-instance invariant.

## 3. Rationale and decision drivers

- External composition retains H3 as the only dispatch work unit and avoids a
  second authority/control plane.
- A deterministic classifier makes incomplete, malformed, or ambiguous process
  evidence fail closed without claiming database truth.
- Per-invocation aggregate readiness avoids treating no-work snapshots as a
  readiness proof while retaining H3 transactional authority as final.
- Deployment-only single-instance controls limit normal overlap without
  weakening H3's established correctness backstop.
- Bounded restart and shutdown rules avoid infinite churn and volatile repair.

## 4. Consequences

### 4.1 Positive consequences

- Continuous operation remains reviewable without a daemon or new database
  authority.
- Every observable H3/process result has a deterministic operational action.
- Operators can distinguish disable, H4 removal, maintenance, retryable
  degradation, malformed capture, and durable workflow evidence.
- The bounded H3 command remains available for controlled manual operation.

### 4.2 Trade-offs and risks

- The deployment platform must provide durable output capture, finite policy,
  duplicate prevention, and health/alerting.
- A malformed or unavailable result intentionally stops rather than guessing
  no-work; an operator may need to inspect existing database evidence.
- Graceful drain may wait for the bounded child; forced termination deliberately
  delegates recovery to existing H1–H3 contracts.

## 5. Compatibility and migration

Acceptance would authorize only a future additive deployment implementation
around the existing H3 command. It requires no database migration, data rewrite,
ACL/role change, CLI change, runtime source change, or backup by itself. H1–H3
remain usable without H4. Removing H4 supervision returns the deployment to
bounded H3 operation and is not a data rollback.

## 6. First implementation increment and verification

Before a proposed H4 implementation is accepted, focused evidence MUST prove:

1. exact H3 command construction and H3 LIMIT validation;
2. invalid supervisor configuration fails closed;
3. readiness is required immediately before every invocation;
4. valid work, valid no-work, and request-local failure classification;
5. every H3 global-stop classification;
6. missing, malformed, and conflicting terminal-result handling after a child
   otherwise reaches a state requiring a completed result;
7. positively identified process-interruption classification, including a crash
   or forced termination without a normal terminal-result, and fail-closed
   handling of an unclassifiable process state;
8. bounded retry/backoff and retry-budget exhaustion to operator-required
   degradation;
9. a MUST-stop class cannot be overridden by configuration;
10. supervisor restart restores only a complete persisted action, fails closed
    with operator review when no complete durable prior classification/action
    exists, and revalidates readiness before any later permitted launch;
11. ordinary single-instance deployment and duplicate-drift detection/handling;
12. accidental H3 overlap remains safe under existing H3 concurrency behavior;
13. graceful stop before child launch, graceful drain with an active child, and
    forced termination after finite grace;
14. existing H1–H3 recovery remains authoritative after forced termination;
15. the health record contains every §2.8 field and output/classifier/state
    capture is durable;
16. credential secrecy and absence of scheduler/worker interference; and
17. no database singleton, heartbeat, lease, schema, ACL, migration, or new
    database authority is created.

This is a required later implementation/review checklist, not authorization to
implement or run such tests in this documentation correction.

## 7. Alternatives considered

### 7.1 In-process daemon or continuous CLI

Rejected: it changes H3 lifecycle/signal surface and creates an unnecessary
application process-control plane.

### 7.2 Database singleton, heartbeat, Manager lease, or leader election

Rejected: H3 already tolerates overlap; adding database ownership would add
unaccepted authority, failure, privilege, and recovery semantics.

### 7.3 Retry based only on exit status or logs

Rejected: H3 exit codes alone cannot distinguish all global, database, and
incomplete outcomes; malformed evidence must fail closed.

## 8. References

- [ADR-0019](ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md)
- [ADR-0019A](ADR-0019A-h1-owner-safe-transaction-authorization.md)
- [ADR-0019B](ADR-0019B-h1-sealed-role-deployment-contract.md)
- [ADR-0019C](ADR-0019C-h2-privilege-deployment-contract.md)
- [Campaign Operations Phase H production contract](../CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md)
- [Campaign Operations Phase H1](../../CampaignOperationsPhaseH1.rst)
- [Campaign Operations Phase H2](../../CampaignOperationsPhaseH2.rst)
- [Campaign Operations Phase H3](../../CampaignOperationsPhaseH3.rst)
- [ADR index](README.md)

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 0.1.0 | 2026-08-09 | Proposed external-supervisor H4 operational contract. |
| 0.1.1 | 2026-08-09 | Targeted correction: deterministic H3 classifier/matrix, immediate pre-invocation readiness, distinct operational procedures, normative deployment scope, finite failures, shutdown, observability, validation, and index consistency. |
| 0.1.2 | 2026-08-09 | Residual targeted correction: process-interruption precedence and fail-closed restart recovery for an incomplete deployment-owned prior action. |
| 1.0.0 | 2026-08-09 | Accepted after independent architecture review, targeted correction, focused reverification, and direct H3 machine-output contract verification. |
