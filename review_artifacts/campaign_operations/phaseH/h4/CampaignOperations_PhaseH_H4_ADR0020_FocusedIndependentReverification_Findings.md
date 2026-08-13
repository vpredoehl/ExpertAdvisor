---
title: "Campaign Operations Phase H H4 ADR-0020 Focused Independent Reverification Findings"
document_type: "architecture review"
status: "final"
basis: "CampaignOperations_PhaseH_H4_ADR0020_Reverification_Input.md"
reviewer: "ChatGPT GPT-5.6 Sol"
date: "2026-08-09"
---

# Campaign Operations Phase H H4 ADR-0020 Focused Independent Reverification Findings

## Executive verdict

The targeted correction materially closes the prior ADR-0020 review findings and preserves the selected external-supervisor architecture. However, ADR-0020 is not yet ready for acceptance because two residual operational-contract ambiguities remain.

Recommended gate:

READY_FOR_H4_ADR0020_TARGETED_CORRECTION

## Finding 1 — HIGH — Crash / forced-termination rows conflict with terminal-result invalidity precedence

ADR-0020 defines:

- abnormal nonzero process exit -> `BACKOFF_AND_RETRY_WITH_READINESS`;
- child process crash -> `BACKOFF_AND_RETRY_WITH_READINESS`;
- forced child termination -> `BACKOFF_AND_RETRY_WITH_READINESS`;

but also defines:

- missing H3 terminal-result record -> `STOP_MALFORMED_RESULT`;
- malformed H3 terminal-result record -> `STOP_MALFORMED_RESULT`;
- multiple/conflicting terminal records -> `STOP_MALFORMED_RESULT`;

and states that terminal-result invalidity takes precedence over any inferred process result.

A genuine crash or forced termination will commonly result in no valid H3 terminal record. Under the current precedence rule, such an interruption maps to `STOP_MALFORMED_RESULT`, which conflicts with the explicit crash/forced-termination retry rows.

This violates the ADR's own requirement that each observable outcome map deterministically to exactly one supervisor action.

### Minimum correction scope

Choose and document one explicit precedence rule.

Preferred narrow correction:

- positively identified crash or forced termination is a distinct process-interruption classification;
- for that classification, absence of a normal H3 terminal result is expected and the process-interruption row takes precedence over terminal-result validation;
- `STOP_MALFORMED_RESULT` applies when a terminal result is required for an otherwise completed child result but is missing, malformed, duplicated, or conflicting;
- unknown/unclassifiable process states still fail closed.

No H3 runtime change is required.

## Finding 2 — MEDIUM — Supervisor restart with no durably recorded prior action is unspecified

ADR-0020 defines:

`RESTORE_PERSISTED_NEXT_ACTION`

for supervisor restart after a prior failure, but it does not explicitly define the case where the supervisor itself fails before durably recording the prior classification/action.

A restart implementation must not guess, reclassify database truth from logs, or silently retry when no complete durable prior supervisor action exists.

### Minimum correction scope

Add a deterministic fail-closed rule for restart when there is no complete durable prior classification/action record.

For example:

- supervisor restart with no complete durable prior classification/action -> `STOP_MALFORMED_RESULT` or a dedicated equivalent operator-required fail-closed action;
- do not reconstruct durable workflow truth from logs;
- do not auto-retry until an operator resolves the incomplete supervisor record.

## Evidence limitation

The uploaded reverification bundle confirms the H3 bounded-command contract, frozen global-stop classification, request-local continuation, exact replay/recovery boundaries, and the corrected ADR/index/review outputs.

The bundle does not independently expose all implementation-level details for the exact H3 machine-record names, record cardinalities, and exit-code behavior asserted by ADR-0020. The targeted correction reports that `Tests/CampaignOperationsPhaseH3ContractTests.sh` passed, but a full independent proof of those exact machine-output assumptions would require the relevant H3 contract test or service/output source.

This is an evidence limitation in the uploaded bundle, not a separate ADR defect.

## Architecture assessment

The following remain sound and should not be redesigned:

- external deployment-owned supervisor;
- H3 run-once remains the only bounded work primitive;
- immediate aggregate readiness before every permitted H3 launch;
- deterministic deployment-owned classifier;
- finite retry budget and bounded backoff;
- emergency disable / H4-only rollback / planned maintenance separation;
- exactly one supervisor per target database/environment/reviewed deployment scope;
- no database singleton, heartbeat, Manager lease, leader election, advisory ownership, or process-lifetime fence;
- graceful shutdown with bounded exceptional forced termination;
- PostgreSQL H1-H3 evidence remains authoritative for workflow truth;
- ADR-0020 remains Proposed until independent acceptance.

## Required next step

Perform one very small documentation-only targeted correction for the two findings above, then run focused independent reverification.

READY_FOR_H4_ADR0020_TARGETED_CORRECTION
