# ADR-0001: PostgreSQL as the durable source of truth

Status: Accepted
Date: 2026-07-15
Deciders: Project architecture
Affected volumes: Volume I §§7–9; Volume XII
Supersedes: None
Superseded by: None

## 1. Context and problem statement

The platform coordinates long-running training, inference, analysis,
continuation, and advisory recommendation workflows. Processes may terminate,
restart, overlap, or be inspected independently. Logs, PIDs, in-memory objects,
and filesystem artifacts cannot alone provide transactional ownership,
referential integrity, or a consistent lifecycle record.

The project already persists experiments, models, evidence, policies,
recommendations, scores, reviews, and migration history in PostgreSQL.

## 2. Decision

PostgreSQL is the authoritative durable source of truth for platform state
represented by the database schema.

- Repositories own SQL and map rows into typed domain records.
- Related state and audit/provenance writes commit atomically.
- Database constraints enforce durable invariants where safe.
- Process memory, logs, files, and PIDs are evidence or caches, not competing
  authority.
- Schema evolution uses ordered checksum-recorded migrations.
- A recorded migration is never rewritten; corrections are additive.
- Runtime roles receive least privilege, and append-only audit history denies
  ordinary runtime update/delete privileges.

This decision does not claim that PostgreSQL currently contains immutable
snapshots of every external market-data source. Exact data-replay identity
remains a separate future contract under Volume II.

## 3. Rationale and decision drivers

- Transactions provide atomic lifecycle and provenance updates.
- Constraints and locks provide cross-process correctness.
- Durable queries make recovery and audit independent of one process.
- Ordered migrations make schema history observable and repeatable.
- Typed repositories isolate storage details from domain policy.

## 4. Consequences

### 4.1 Positive consequences

- Restart-safe state and cross-process coordination.
- Enforceable references, uniqueness, status shapes, and ownership.
- Consistent inspection and recovery paths.
- Auditable schema and decision history.

### 4.2 Negative consequences and trade-offs

- Repository and migration discipline are mandatory.
- Database availability and backup quality become operational dependencies.
- Long transactions and broad locks can harm concurrency if poorly designed.

### 4.3 Risks and mitigations

- Schema drift: prevented by the migration ledger and checksum checks.
- Excess privilege: mitigated by role-specific grants and privilege tests.
- Legacy incompatibility: handled with additive migrations, explicit exclusion,
  and `NOT VALID` constraints where appropriate.

## 5. Compatibility and migration

This ADR records existing architecture and requires no schema change. Future
schema changes follow Volume XII. Existing legacy rows are preserved; missing
provenance is not guessed merely to satisfy a new contract.

## 6. Verification and operational evidence

- Migration runner repeatability and checksum mismatch detection.
- Isolated migration tests and constraint/index/grant inspection.
- Transaction rollback and concurrency tests using independent connections.
- Backup and restore procedures appropriate to deployed environments.
- Repository tests confirming raw rows do not escape persistence boundaries.

## 7. Alternatives considered

### 7.1 Logs and process state as authority

Rejected because they lack atomicity, referential integrity, durable ownership,
and reliable recovery semantics.

### 7.2 Filesystem-only state

Rejected as the primary lifecycle store because multi-process claims,
relational integrity, querying, and migrations would require rebuilding a
database poorly.

### 7.3 Per-subsystem independent stores

Not selected for current architecture because it adds distributed consistency
and operational complexity without a demonstrated need.

## 8. References

- [Volume I §7](../Volume_I_Foundation.md)
- [Volume XII](../Volume_XII_Database.md)
- [`migrate_lstm_db.sh`](../../../migrate_lstm_db.sh)

## 9. Revision history

| Date | Change |
|---|---|
| 2026-07-15 | Recorded the existing PostgreSQL ownership and migration strategy. |
