# Volume [Roman numeral] — [Domain title]

Status: Draft | Foundation outline | Authoritative | Superseded
Version: 0.1.0
Last revised: YYYY-MM-DD
Owners: [Architecture/domain owners]

> Template instructions are enclosed in brackets and must be removed from a
> completed volume. Top-level section numbers are stable. Add subordinate
> headings such as §7.1, §7.2, and §7.3 without renumbering §§1–13.

## 1. Purpose

[State why this domain exists, what evidence or capability it provides, and
how it supports the research platform.]

## 2. Scope

### 2.1 In scope

[List the behaviors and contracts governed by this volume.]

### 2.2 Out of scope

[List adjacent responsibilities explicitly owned elsewhere.]

### 2.3 Current implementation status

[Distinguish implemented, partially implemented, planned, and deferred work.]

## 3. Responsibilities

### 3.1 Owned responsibilities

[Identify authoritative state and decisions owned here.]

### 3.2 Dependencies

[Identify upstream and downstream contracts by volume and section.]

### 3.3 Prohibited responsibilities

[State what this domain must never infer, mutate, or authorize.]

## 4. Architecture

### 4.1 Components

[Describe pure domain, repository, service, CLI, worker, and scheduler roles as
applicable.]

### 4.2 Control flow

[Describe the deterministic use-case flow.]

### 4.3 Ownership boundaries

[Name every state transition owner and cross-domain interface.]

## 5. Data model

### 5.1 Authoritative entities

[Define entities and identity categories.]

### 5.2 Provenance and versions

[Define canonical text, hashes, lineage, policy versions, and implementation
versions.]

### 5.3 Invariants and legacy data

[Define constraints, nullability, immutable history, and legacy behavior.]

Any SQL in this section is illustrative and non-executable unless it links to
an actual migration file.

## 6. Transactions

### 6.1 Read paths

[Define read-only snapshots and consistency requirements.]

### 6.2 Write paths

[Define atomic units, row changes, audit events, and rollback behavior.]

### 6.3 Failure semantics

[Define partial failure, retries, finalization, and error persistence.]

## 7. Concurrency

### 7.1 Conflict domain

[Name which records or semantic keys may conflict.]

### 7.2 Locking and serialization

[Define row locks, advisory locks, unique constraints, and lock ordering.]

### 7.3 Winner, loser, and retry outcomes

[Define exact concurrent outcomes, idempotency, and stable conflict reasons.]

## 8. CLI

### 8.1 Commands and validation

[List explicit command families, mutual exclusion, and early validation.]

### 8.2 Machine output

[Define stable event names, fields, escaping, nulls, and exit codes.]

### 8.3 Human output

[Define concise safe presentation and disclaimers.]

## 9. Testing

### 9.1 Pure tests

[Parsing, validation, calculation, identity, and ordering.]

### 9.2 Persistence and migration tests

[Constraints, SQLSTATEs, transactions, mapping, repeatability, and privileges.]

### 9.3 Concurrency and integration tests

[Independent connections, exact fixtures, before/after comparisons, and
cleanup.]

### 9.4 Regression boundaries

[Name adjacent volumes that must remain unchanged.]

## 10. Operational safety

### 10.1 Runtime isolation

[Define scheduler, worker, and live-data boundaries.]

### 10.2 Permissions and destructive operations

[Define runtime roles, owner operations, cleanup, and operator confirmation.]

### 10.3 Observability and recovery

[Define logs, durable status, alerts, dry runs, recovery, and rollback.]

## 11. Future extensions

### 11.1 Approved extension points

[Name interfaces designed for extension.]

### 11.2 Deferred capabilities

[Name non-implemented behavior explicitly.]

### 11.3 Required decisions

[List ADRs, schema contracts, compatibility work, and safety evidence required
before implementation.]

## 12. References

- [Volume I](Volume_I_Foundation.md)
- [Relevant volumes and ADRs]
- [Implementation specifications and tests]

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | YYYY-MM-DD | Initial domain outline. | — |
