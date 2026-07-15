# ADR-0004: Scheduler ownership boundaries

Status: Accepted
Date: 2026-07-15
Deciders: Project architecture
Affected volumes: Volume I §10; Volume VII; Volume XI
Supersedes: None
Superseded by: None

## 1. Context and problem statement

The platform coordinates long-running training, inference, analysis,
checkpoint, and configured continuation work. Many other capabilities can be
invoked explicitly but are not scheduler work. Without a firm boundary, adding
a command could accidentally add polling, consume worker capacity, or create
implicit lifecycle transitions.

Process launch also cannot safely share one long database transaction, and PID
presence alone cannot establish durable ownership after failures.

## 2. Decision

The scheduler owns only explicitly assigned lifecycle work classes.

- Eligibility and lifecycle transitions are durable database contracts.
- The scheduler atomically claims bounded work before launching a worker.
- Worker launch and execution occur outside database transactions.
- Workers perform one bounded operation and report facts; they do not select or
  queue further work independently.
- Capacity, retries, recovery, shutdown, and operator controls are defined per
  work class.
- New scheduler-managed work requires an ADR defining eligibility, claim,
  capacity, idempotency, recovery, observability, and regression boundaries.
- Recommendation generation, scoring, and review remain outside scheduler
  polling under ADR-0003.

## 3. Rationale and decision drivers

- Prevent duplicate workers and lost lifecycle updates.
- Keep database transactions short.
- Make capacity and process ownership observable and recoverable.
- Prevent unrelated explicit commands from becoming background automation.

## 4. Consequences

### 4.1 Positive consequences

- Clear ownership between lifecycle services, scheduler, and workers.
- Narrow database claims support multiple processes safely.
- Scheduler capacity remains predictable.
- Orphan recovery can reconcile durable attempt state with processes.

### 4.2 Negative consequences and trade-offs

- Every work class needs explicit claim/recovery design.
- Process supervision and database reconciliation remain operationally complex.
- New automation cannot be added merely by placing a command in the poll loop.

### 4.3 Risks and mitigations

- Launch after claim fails: persist a distinct launch failure and reconcile.
- Multiple schedulers race: use database-visible conditional claims/locks.
- Stale PIDs: combine attempt identity and durable timestamps/state with process
  inspection; never trust PID alone.

## 5. Compatibility and migration

This ADR records existing scheduler boundaries and requires no schema change.
Future work classes may require additive attempt/capacity schema and must
preserve current experiment and continuation behavior.

## 6. Verification and operational evidence

- Pure eligibility, capacity, command, and status-mapping tests.
- Claim/concurrency tests with multiple database connections.
- Launch-failure, completion-race, orphan-recovery, and shutdown tests.
- Regression checks proving advisory commands do not enter scheduler polling.
- Controlled integration that does not disturb running experiments.

## 7. Alternatives considered

### 7.1 Workers self-select follow-up work

Rejected because it bypasses centralized capacity, durable lifecycle ownership,
and operator controls.

### 7.2 Hold a database transaction for worker lifetime

Rejected because long transactions harm concurrency and failure recovery.

### 7.3 Treat every explicit command as pollable work

Rejected because command availability does not define scheduler eligibility or
resource policy.

## 8. References

- [Volume I §10](../Volume_I_Foundation.md)
- [Volume VII](../Volume_VII_Experiment_Lifecycle.md)
- [Volume XI](../Volume_XI_Scheduler.md)
- [ADR-0003](ADR-0003-advisory-recommendation-evaluation.md)

## 9. Revision history

| Date | Change |
|---|---|
| 2026-07-15 | Recorded explicit scheduler work ownership and process/transaction boundaries. |
