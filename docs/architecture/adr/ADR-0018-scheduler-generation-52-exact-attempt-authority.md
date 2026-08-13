# ADR-0018: Scheduler generation-52 exact-attempt authority

Status: Accepted
Date: 2026-07-29
Deciders: Project architecture
Affected volumes: Volume VII §§4–10; Volume XI §§3–10; Volume XII §§5–10
Supersedes: None
Superseded by: None

## 1. Context

Migration 051 introduced a fenced scheduler lease and durable worker attempts,
but several paths retained weaker authority: continuation mutation, signals,
parent reaping, direct CLI work, and in-process checkpoint analysis. The old
and new paths also admitted lock-order inversions, relied on an operational
mixed-version convention, and had no bounded safe result for legacy no-PID
attempts.

## 2. Decision

- One scheduler authority context carries the invocation ID, fencing token,
  canonical executable, and held state through every scheduler mutation.
- One shared exact-attempt verifier separates durable row validation, OS
  process validation, current-fence validation, and exact SQL mutation.
- Signals require a signalable exact attempt, complete immutable process/work
  identity, matching lifecycle binding, and a second observation immediately
  before `kill`/`killpg`. Ambiguous attempts are never signalable.
- Parent reaping and all destructive reconciliation terminalize and clear only
  the exact attempt still bound to the lifecycle row. Capacity is released only
  by that exact attempt becoming terminal.
- Direct CLI work may not target scheduler-managed experiment or checkpoint
  lifecycle rows without the scheduler-reserved attempt ID. Bulk direct
  analysis is prohibited.
- Checkpoint analysis remains in-process and uses short claim, unlocked work,
  and exact fenced finalize phases with one durable analyze-capacity attempt.
- The canonical lock order is advisory coordination, protocol, lease,
  invocation/admin rows, attempts, experiments, checkpoint evaluations,
  continuation/analysis results, then campaign/recommendation service locks.
- Migration 052 establishes protocol generation 52. Corrected startup requires
  a completed cutover record created only after positive inspection proves all
  scheduler dispatch processes absent. Generation-aware triggers reject old
  scheduler lifecycle write shapes after migration.
- Legacy no-PID ambiguity consumes capacity until completed cutover evidence,
  absence of a possible identity-publication transition, exact lifecycle
  binding, and a bounded grace interval prove safe abandonment. Reconciliation
  is exact, audited, and idempotent.

## 3. Consequences

Ownership loss stops later mutation, stale observations cannot affect
replacement attempts, and checkpoint analysis is crash-recoverable without
holding the lease or lifecycle locks during computation. Deployment gains an
explicit irreversible coordination point: after cutover, executable and
database protocol generation must move together.

Ambiguous identity intentionally reduces available capacity. Operators must
inspect the durable diagnostic instead of bypassing it. A process-inspection
permission failure is ambiguity, never absence.

## 4. Rationale and decision drivers

The attempt row is the only object that can unite capacity, immutable process
identity, lifecycle binding, and scheduler fence without inferring ownership
from process-local state. Reusing it across all destructive paths avoids
slightly different pause, cancel, reap, and recovery authorities. A technical
cutover is required because an executable that predates the lease cannot
participate in a database-only mutual-exclusion protocol.

## 5. Compatibility and migration

Migration 052 is additive to 051 and preserves canonical
`current_operation` values `train`, `infer`, and `analyze`. Ordinary experiment
and checkpoint identities are unchanged. The new analyze-attempt kind uses
analyze capacity without adding a lifecycle operation. Existing ambiguous
workers remain alive and capacity-consuming. Production data is neither
rewritten nor terminalized by migration.

## 6. Verification and operational evidence

Required verification covers fence loss at every continuation mutation
boundary; exact signal rejection for PID, process group, start, executable,
command/work, lifecycle, ambiguity, and foreign fence; stale reaper replacement;
direct CLI rejection; checkpoint-analysis exhaustion/crash/fence loss/retry;
lock-order serialization; cutover rejection/replay; and legacy no-PID grace,
abandonment, replay, and capacity accounting.

## 7. Alternatives considered

Operator-only shutdown convention was rejected because it supplies no positive
evidence to corrected startup. PID-only signaling and lifecycle-only
finalization were rejected because PID reuse and replacement attempts remain
reachable. Keeping checkpoint analysis in one transaction was rejected because
CPU/filesystem work would retain database locks and an uncommitted capacity
decision. Treating no-PID as immediate absence was rejected because a legacy
launch could still be publishing identity.

## 8. Deployment

Build at a new path, stop old dispatch processes, preserve positively
identified workers, back up, apply migrations 051 and 052, execute the explicit
cutover command, and only then start one generation-52 scheduler. A pending or
failed cutover is a read-only startup rejection. Never synthesize cutover state
or roll back only one side of the protocol.

## 9. References

- [ADR-0004](ADR-0004-scheduler-ownership-boundaries.md)
- [ADR-0016](ADR-0016-scheduler-atomic-claim-hardening.md)
- [Volume VII](../Volume_VII_Experiment_Lifecycle.md)
- [Volume XI](../Volume_XI_Scheduler.md)

## 10. Revision history

| Date | Change |
|---|---|
| 2026-07-29 | Accepted generation-52 exact-attempt authority, cutover, recovery, and lock-order correction. |
