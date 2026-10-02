# ADR-0021: Read-only scheduler operational observation

Status: Accepted
Date: 2026-10-01
Deciders: Project architecture
Affected volumes: Volume XI §§3–9; Volume XII §6
Supersedes: None
Superseded by: None

## 1. Context and problem statement

The legacy scheduler-status implementation is embedded in a scheduler
translation unit that also composes mutation and control services.  It cannot
be reused by a dedicated operational observer without either linking mutation
capability or exposing a mutable repository transaction to observation code.

Operational investigation needs durable scheduler/lifecycle evidence and
diagnostic process observation, but must not become scheduler authority.

## 2. Decision

SchedulerCore will expose a dedicated, typed operational read-model boundary.
It exposes only observation data required by scheduler and experiment status.
It does not expose generic SQL execution, scheduler claims, lifecycle
transitions, worker-attempt creation/finalization, administrative controls,
reconciliation, recovery, queueing, preemption, or process signaling.

`lstm-observer` consumes that boundary directly.  Its database snapshots use
`REPEATABLE READ, READ ONLY`; it does not acquire scheduler authority or
instantiate scheduler mutation service composition.  Operating-system process
observation remains diagnostic and cannot create authority or override the
ADR-0018 exact-attempt record.

The initial extraction preserves legacy `LSTM_Release --status` and
`--scheduler-status` output and transaction behavior.  Migration of legacy
`--scheduler-status` to the new read-only transaction is separate follow-up
work.

## 3. Rationale and decision drivers

This preserves one authoritative durable read model while preventing a
read-only evidence consumer from receiving incidental scheduler-control
capability.  A repeatable read-only snapshot gives coherent multi-query
evidence without adding write authority.

## 4. Consequences

The read model is a deliberate SchedulerCore interface with focused tests and
an observer executable.  It adds extraction work but no schema, privilege,
lifecycle, capacity, or scheduler-authority change.

## 5. Compatibility and migration

Existing status commands and machine records remain compatible.  No production
rows are rewritten and no new database privileges are granted.

## 6. Verification and operational evidence

Tests must prove observer command rejection for mutation vocabulary, read-only
snapshot setup, no scheduler authority acquisition, and parity for deterministic
status evidence.  Existing exact-attempt/process-distinction regression tests
remain required.

## 7. Alternatives considered

Linking the legacy scheduler monolith was rejected because it couples the
observer to mutation-capable composition.  Shelling out to `LSTM_Release` was
rejected because it is not a reusable typed read boundary.  Changing legacy
status transaction semantics during this extraction was deferred.

## 8. References

- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
- [ADR-0018](ADR-0018-scheduler-generation-52-exact-attempt-authority.md)
- [Volume XI](../Volume_XI_Scheduler.md)
- [Volume XII](../Volume_XII_Database.md)

## 9. Revision history

| Date | Change |
|---|---|
| 2026-10-01 | Accepted the narrow read-only operational observation boundary. |
