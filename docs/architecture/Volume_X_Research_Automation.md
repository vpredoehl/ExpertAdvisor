# Volume X — Research Automation

Status: Reserved outline; not implemented by this document
Version: 0.1.0
Last revised: 2026-07-15

## 1. Purpose

Define a future controlled automation layer for proposing and coordinating
bounded research while preserving explicit policy, budgets, audit, and human
authority.

## 2. Scope

### 2.1 In scope

Future campaign goals, proposal policies, budgets, stopping conditions,
selection/evaluation loops, audit, and operator control.

### 2.2 Out of scope

Unbounded autonomy, self-modifying production code, implicit experiment
creation, live trading, or bypassing scheduler/lifecycle ownership.

### 2.3 Current implementation status

Reserved. Existing recommendation and continuation capabilities do not form an
autonomous research agent and MUST NOT be composed as one informally.

## 3. Responsibilities

### 3.1 Owned responsibilities

Future automation owns declared campaign policy and budget accounting only.

### 3.2 Dependencies

Consumes advisory evidence from Volumes VI/VIII/IX and requests work only
through accepted Volume VII/XI interfaces.

### 3.3 Prohibited responsibilities

MUST NOT bypass review, invent authorization, exceed budgets, mutate model math,
or directly control workers.

## 4. Architecture

### 4.1 Components

Potential components are campaign policy, planner, budget ledger, proposal
adapter, decision/audit repository, operator controls, and scheduler adapter.

### 4.2 Control flow

Explicit campaign → bounded evidence snapshot → proposal → required review or
policy gate → lifecycle request → scheduler execution → evidence update → stop.

### 4.3 Ownership boundaries

Automation proposes and accounts. Domain services validate. Experiment
lifecycle creates. Scheduler claims and runs. Operators retain pause/stop power.

## 5. Data model

### 5.1 Authoritative entities

Future campaign, policy version, budget, proposal, decision, execution request,
evidence snapshot, and stop reason.

### 5.2 Provenance and versions

Every decision records exact inputs, policy canonical text/version, budgets,
software version, and downstream IDs.

### 5.3 Invariants and legacy data

Budget consumption is monotonic and auditable. Missing authority or provenance
blocks action. Advisory status never becomes implicit authorization.

## 6. Transactions

### 6.1 Read paths

Planning consumes named immutable evidence snapshots.

### 6.2 Write paths

Decision, budget reservation, and execution request require an explicit atomic
or compensating protocol before implementation.

### 6.3 Failure semantics

Uncertain budget or request state fails closed and requires reconciliation.

## 7. Concurrency

### 7.1 Conflict domain

Same campaign budget, proposal, or execution request.

### 7.2 Locking and serialization

Future ledgers use row locks/conditional updates and authoritative request IDs.

### 7.3 Winner, loser, and retry outcomes

At most one accepted request consumes a reservation; retries return the same
request only when authoritative idempotency identity matches.

## 8. CLI

### 8.1 Commands and validation

Future start/pause/stop/status commands require explicit campaign IDs, limits,
and confirmation for mutations.

### 8.2 Machine output

Events expose policy, budget, evidence, decision, request, and stop reason.

### 8.3 Human output

Summaries state remaining budget, current authority, and whether any work was
actually requested or queued.

## 9. Testing

### 9.1 Pure tests

Policy, budgets, stopping, selection, and deterministic planning.

### 9.2 Persistence and migration tests

Ledger monotonicity, request idempotency, audit immutability, and permissions.

### 9.3 Concurrency and integration tests

Budget races, duplicate requests, pause/stop races, and scheduler isolation.

### 9.4 Regression boundaries

Recommendation, lifecycle, continuation, and scheduler semantics remain owned
by their volumes.

## 10. Operational safety

### 10.1 Runtime isolation

Automation defaults disabled and cannot run from documentation or mere schema
presence.

### 10.2 Permissions and destructive operations

Least-privilege roles, explicit budgets, operator stop controls, and audit are
mandatory before deployment.

### 10.3 Observability and recovery

Durable campaign status, reservations, requests, outcomes, and reconciliation
must survive process failure.

## 11. Future extensions

### 11.1 Approved extension points

None are executable yet; interfaces named here are architectural placeholders.

### 11.2 Deferred capabilities

Campaign planning, automated conversion, adaptive search, and autonomous
experimentation.

### 11.3 Required decisions

Multiple ADRs are required for authority, budgets, safety, policy identity,
conversion, scheduler capacity, and shutdown before implementation.

## 12. References

- [Volume I §17.4](Volume_I_Foundation.md)
- [Volume VII](Volume_VII_Experiment_Lifecycle.md)
- [Volume VIII](Volume_VIII_Recommendation_Engine.md)
- [Volume XI](Volume_XI_Scheduler.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Reserved the research-automation architecture and safety gates. | — |
