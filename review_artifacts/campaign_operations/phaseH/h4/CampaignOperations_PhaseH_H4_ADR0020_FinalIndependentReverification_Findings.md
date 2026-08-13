---
title: "Campaign Operations Phase H H4 ADR-0020 Final Independent Reverification"
document_type: "architecture review"
status: "final"
reviewer: "ChatGPT GPT-5.6 Sol"
date: "2026-08-09"
---

# Campaign Operations Phase H H4 ADR-0020 Final Independent Reverification

## Executive verdict

ADR-0020 is ready for formal acceptance.

The focused residual findings are closed, and the previously outstanding H3 machine-output evidence gap has been independently verified from the supplied implementation excerpts.

Final gate:

READY_FOR_H4_ADR0020_ACCEPTANCE

## Residual finding closure

### HIGH — Process interruption versus missing terminal result

Closed.

ADR-0020 now defines a positively identified child crash, forced termination, or equivalent externally proven process interruption as its own normalized outcome class.

For that class:

- absence of a normal H3 terminal result is expected;
- process-interruption classification takes precedence over terminal-result validation;
- the next action is `BACKOFF_AND_RETRY_WITH_READINESS`;
- the existing finite retry budget and bounded backoff still apply.

Terminal-result invalidity applies only when the child otherwise reached a state requiring a completed H3 result. Unknown or unclassifiable process state fails closed to `STOP_MALFORMED_RESULT`.

The outcome/action matrix is no longer overlapping on this boundary.

### MEDIUM — Supervisor restart without complete durable prior action

Closed.

ADR-0020 now distinguishes:

- restart with valid configuration and a complete deployment-owned durable prior classification/action -> `RESTORE_PERSISTED_NEXT_ACTION`;
- restart with valid configuration but without a complete durable prior classification/action -> `STOP_MALFORMED_RESULT`.

The corrected contract forbids reconstructing workflow truth from logs, inferring a prior action from partial output, or silently auto-retrying. Operator review is required before another launch, and every later permitted launch still requires validated configuration and immediate pre-invocation H1 readiness.

No database durability mechanism was added.

## Machine-output evidence verification

The supplied `CampaignOperationsManagerService.cpp` excerpt independently verifies the H3 machine-readable interface assumed by ADR-0020.

### Request records

`RenderRequest()` emits:

`CAMPAIGN_OPERATIONS_MANAGER_REQUEST`

with the request identity/version, operation key, outcome, dispatch classification, replay disposition, commit/replay flags, and diagnostic code.

Each element in `result.requests` is rendered exactly once by the run-once command path.

### Run-once summary

`RunCampaignOperationsManagerOnceCommand()` contains one summary emission site:

`CAMPAIGN_OPERATIONS_MANAGER_RUN_ONCE`

with:

- `dispatch_limit`;
- `candidates_selected`;
- `processed`;
- `stopped_early`;
- `stop_reason`;
- `candidate_request_ids`.

The summary is emitted before individual request records, confirming ADR-0020's clarification that "terminal-result" is a semantic completed result and not necessarily the physically final output line.

### Classified global-stop record

When `result.stoppedEarly` is true, the command contains one global-stop emission site:

`CAMPAIGN_OPERATIONS_MANAGER_STOPPED`

with:

- `reason`;
- `diagnostic`.

The same branch returns exit code `2`.

Normal completion returns exit code `0`.

The caller in `ExperimentScheduler.cpp` directly returns `RunCampaignOperationsManagerOnceCommand(...)`, so the H3 command's `0` / `2` result is not translated by an intermediate caller.

### Stable stop reasons

`ToText(ManagerGlobalStopReason)` verifies the exact stable literals:

- `none`
- `production_disabled`
- `scheduler_protocol_ineffective`
- `manager_build_not_ready`
- `privilege_failure`
- `database_failure`

### Failure before terminal-result creation

The implementation also confirms why ADR-0020 needs a separate process/connectivity-failure class.

Initial connection/candidate selection occurs before the completed-result rendering path, so failures before a valid H3 result can escape without a normal terminal record. H4 therefore must not assume that every failed invocation produces both a run-once summary and a stopped record.

### Commit-outcome uncertainty

When dispatch reports `commitOutcomeUnknown`, H3 sets:

- `stoppedEarly = true`;
- `stopReason = databaseFailure`;
- `stopDiagnostic = dispatched.diagnosticCode`;

and then exits through the classified global-stop path.

This supports ADR-0020's more specific H4 classification of an indeterminate/commit-unknown diagnostic above the generic database-failure class.

## Architecture status

No remaining BLOCKER, HIGH, or MEDIUM finding was identified.

The following architecture remains intact:

- external deployment-owned H4 supervisor;
- H3 run-once remains the only bounded Campaign Manager work primitive;
- aggregate H1 readiness immediately before every permitted H3 invocation;
- deterministic deployment-owned classifier;
- finite retry budget and bounded backoff;
- separate emergency production disable, H4-only rollback, and planned maintenance procedures;
- exactly one supervisor per target database/environment/reviewed deployment scope;
- H3 multi-Manager correctness remains the accidental-overlap backstop;
- no database singleton, heartbeat, Manager lease, leader election, advisory ownership, or process-lifetime fence;
- bounded graceful shutdown with exceptional forced termination;
- PostgreSQL H1-H3 durable evidence remains authoritative for workflow truth.

## Editorial note

ADR-0020 contains a non-blocking quotation-mark mismatch in the phrase:

`"Terminal-result”`

This is LOW/editorial only and may be corrected before acceptance.

## Database / implementation impact

This independent reverification required no:

- runtime source change;
- schema change;
- migration;
- ACL/role/grant change;
- production database mutation;
- database backup;
- H3 behavior change.

## Final gate

READY_FOR_H4_ADR0020_ACCEPTANCE
