# Volume XII — Database

Status: Foundation aligned through Phase 6C
Version: 0.3.0
Last revised: 2026-07-21

## 1. Purpose

Define PostgreSQL's durable ownership, schema/migration governance,
constraints, privileges, transaction conventions, and audit-history policy.

## 2. Scope

### 2.1 In scope

Schemas, tables, constraints, indexes, sequences, roles, grants, migrations,
transactional persistence, legacy compatibility, backup, and integrity.

### 2.2 Out of scope

Domain formulas, lifecycle policy, scheduler selection, and operator
authorization are specified by their owning volumes.

### 2.3 Current implementation status

PostgreSQL is the implemented durable source of truth. Ordered SQL migrations
are checksum-recorded by the project runner. Phase 6B migration 042 adds an
append-only exact follow-up-proposal manifest and ordered member table for
read-only operator preview. Phase 6C migration 043 adds one append-only
administrative review event per exact persisted proposal. Neither migration
adds lifecycle, experiment, execution, queue, or scheduler state.

## 3. Responsibilities

### 3.1 Owned responsibilities

Own durable records, referential integrity, enforceable state/value invariants,
uniqueness, runtime privileges, migration history, and transactional isolation.

### 3.2 Dependencies

Every persistent domain defines its semantic rules; repositories translate
those rules into typed SQL and transactions consistent with this volume.

### 3.3 Prohibited responsibilities

The database MUST NOT invent missing provenance, encode scattered application
policy without a domain contract, or rely on logs/process memory as authority.

## 4. Architecture

### 4.1 Components

PostgreSQL server, public domain tables, constraints/indexes/sequences,
runtime/owner roles, migration ledger/runner, repositories, and backup tools.

### 4.2 Control flow

Validated service request → repository transaction → database constraints and
locks → commit durable state → typed result. Migration flow is ordered file →
checksum verification → transaction → ledger insertion.

### 4.3 Ownership boundaries

Domain code defines meaning; repository owns SQL; PostgreSQL owns committed
truth; migrations own schema evolution; scheduler/workers use least privilege.

## 5. Data model

### 5.1 Authoritative entities

The schema represents experiments, models, evidence, policies, lifecycle,
continuations, recommendations, scoring, reviews, and migration history.
Exact inventories belong to migrations and domain volumes.
Phase 6B follow-up proposal rows durably preserve the complete immutable Phase
6A advisory value. Their row ID, hash-collision ordinal, and creation timestamp
are repository metadata outside the authoritative proposal identity.
Phase 6C review events preserve the exact proposal ID/version/canonical/hash,
approved or rejected administrative decision, reviewer, reason, and complete
review canonical/hash. Review-event ID and creation timestamp are metadata
outside review identity. Approval is not action authorization.

### 5.2 Provenance and versions

Durable contracts use migration version, canonical policy/configuration
versions, schema/model versions, timestamps, lineage, and implementation
metadata appropriate to the domain.

### 5.3 Invariants and legacy data

Primary/foreign keys, checks, uniqueness, ownership, status shapes, and
append-only permissions are enforced where safe. `NOT VALID` constraints may
protect new writes while preserving unmapped legacy rows; validation is a
separate deliberate operation.

## 6. Transactions

### 6.1 Read paths

Inspection uses read-only transactions. Multi-query evidence loading uses a
consistent snapshot when required by the domain.

### 6.2 Write paths

Repositories use short explicit read-write transactions. Related mutable state
and immutable audit/provenance commit atomically. External work is excluded.
Phase 6B inserts one proposal manifest and its complete ordered members under a
hash-scoped advisory lock; a deferred trigger rejects partial membership.
Phase 6C serializes one proposal-ID conflict domain with a transaction advisory
lock, verifies the exact Phase 6B row, and inserts at most one review event. It
never updates or supersedes an event or changes a Phase 6B row.

### 6.3 Failure semantics

Constraint and serialization failures roll back the complete transaction and
map to stable repository/service outcomes. Finalization does not hide the
initiating error.
Phase 6C returns ``recorded`` for the first valid event,
``existing_identical`` for exact replay, and a deterministic conflict for any
different decision, reviewer, reason, or review identity on that proposal.

## 7. Concurrency

### 7.1 Conflict domain

Conflicts are scoped by row, unique semantic key, attempt, or named advisory
key defined by the owning domain.

### 7.2 Locking and serialization

Use row locks, conditional updates, unique indexes, foreign keys, and
transaction-scoped advisory locks. Full canonical values are rechecked after
hash/advisory-key matches.
Phase 6C uses no proposal row lock: one proposal-ID-scoped advisory lock plus a
unique proposal foreign key serializes attempts without mutating Phase 6B.

### 7.3 Winner, loser, and retry outcomes

Owning volumes define exact outcomes. The database guarantees no partial
commit; applications do not reinterpret arbitrary SQL failure as idempotency.
Concurrent identical Phase 6C events converge on one row; concurrent differing
events produce one recorded winner and one deterministic conflict.

## 8. CLI

### 8.1 Commands and validation

Migration/backup commands are explicit operational tools. Domain CLIs never
embed raw SQL supplied as business behavior.

### 8.2 Machine output

Migration records expose apply/skip/error, version, filename, checksum result,
database, and host without leaking credentials.

### 8.3 Human output

Operational summaries distinguish schema change, data repair, backup, and
read-only inspection.
Phase 6C show/list records expose event/proposal IDs, identities, decision,
reviewer, reason, timestamp, and explicit negative action-authority fields.

## 9. Testing

### 9.1 Pure tests

Repository query construction is secondary to domain pure tests; identifiers,
mapping validation, and stable errors remain testable.

### 9.2 Persistence and migration tests

Use isolated schemas/rollback, direct reruns where supported, SQLSTATE checks,
constraint/index/grant inspection, and runner checksum repeatability.

### 9.3 Concurrency and integration tests

Use independent connections and disposable exact IDs to prove locks,
uniqueness, rollback, privileges, and unrelated-row concurrency.
Phase 6C additionally verifies concurrent identical/conflicting reviews and
read-only presentation without advisory/tuple locks or sequence advancement.

### 9.4 Regression boundaries

Schema changes verify unaffected experiment state, identities, history,
permissions, scheduler behavior, and repository compatibility.
Migration 043 tests preserve Phase 6B manifests/members and experiment and
scheduler sentinels byte-for-byte.

## 10. Operational safety

### 10.1 Runtime isolation

Tests prefer isolated schemas and do not operate on genuine research rows.
Migrations run transactionally through the project runner.

### 10.2 Permissions and destructive operations

Runtime roles receive minimum table/sequence privileges. Immutable audit tables
deny runtime update/delete. Owner cleanup and repair are explicit, exact, and
separate from application APIs.
Phase 6C grants only review-table ``SELECT``, payload-column ``INSERT``, and
sequence ``USAGE``. Generated ID/timestamp insertion and update, delete, and
truncate are denied.

### 10.3 Observability and recovery

Migration ledger, database backups, constraint names, stable SQLSTATEs, and
transaction logs support diagnosis and restoration. Credentials are never
emitted in normal output.

## 11. Future extensions

### 11.1 Approved extension points

Additive migrations, typed repositories, versioned canonical provenance, and
least-privilege roles.

### 11.2 Deferred capabilities

Dataset snapshots, distributed coordination, archival partitions, retention,
and additional operational roles.

### 11.3 Required decisions

Changes require data-volume estimates, compatibility, migration/rollback,
permissions, backup, concurrency, and observability decisions.

## 12. References

- [Volume I §§7–9, 15](Volume_I_Foundation.md)
- [ADR-0001](adr/ADR-0001-postgresql-source-of-truth.md)
- [ADR-0007](adr/ADR-0007-phase-6b-follow-up-proposal-persistence.md)
- [ADR-0008](adr/ADR-0008-phase-6c-follow-up-proposal-administrative-review.md)
- [Phase 6B persistence and preview](../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
- [Phase 6C administrative review](../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
- [`migrate_lstm_db.sh`](../../migrate_lstm_db.sh)
- [`Database/migrations`](../../Database/migrations)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established database ownership and migration-governance outline. | ADR-0001 |
| 0.2.0 | 2026-07-20 | Recorded the append-only exact Phase 6B follow-up-proposal schema, collision-safe transaction, and least-privilege read-only preview boundary. | ADR-0001, ADR-0007 |
| 0.3.0 | 2026-07-21 | Recorded one exact append-only Phase 6C administrative review event per proposal, deterministic replay/conflict, validated reload, and read-only presentation without action authority. | ADR-0001, ADR-0004, ADR-0008 |
