# Volume XII — Database

Status: Aligned through scheduler generation 52 and accepted Campaign Operations Phase H architecture; implemented through Campaign Operations Phase H H1 (migration 055)
Version: 0.15.0
Last revised: 2026-08-03

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
Phase 6D migration 044 adds one append-only governance-ratification event per
exact eligible approved Phase 6C review. The fixed ratifier role and mandatory
reviewer/ratifier separation distinguish it from merits review. It adds no
Campaign Operations, lifecycle, experiment, execution, queue, or scheduler
state.

Campaign Operations migration 045 adds the immutable operational-campaign,
optional governance-provenance, append-only authorization-chain, and audit
foundation. It adds no budget, reservation, request, dispatch, cancellation,
completion, lifecycle, scheduler, worker, UI, or CLI behavior and grants no
capability role to the runtime login.

Campaign Operations migration 047 adds the append-only budget ledger, guarded
reservation/request projections, immutable reservation transition evidence,
same-transaction audit completeness, and read-only status views. A held
full-materialization reservation and ready request cannot commit separately.
Request prerequisite/provenance evidence must exactly match its accepting
authorization, cause-specific audits must exactly match their authoritative
budget or request-acceptance mutation, and optional reservation expiry is
validated against PostgreSQL ``transaction_timestamp()``.
The new budget-administrator and request-acceptor roles are distinct NOLOGIN
capabilities and are not granted to ``pqxx``. No dispatch, lifecycle,
experiment, scheduler, worker, control, cancellation, completion, or
monitoring authority is added.

Campaign Operations migration 048 adds immutable dispatch attempts and
outcomes, complete per-member bindings, permanent V1 control owners,
reservation commitment evidence, guarded lease/request/reservation
transitions, commit-time binding completeness, and exact dispatch audit
references. Dedicated dispatcher and transactional Phase 5 roles are NOLOGIN,
are not members of or granted to `pqxx`, and receive only explicit allowlists.
Production dispatch remains constrained false; no scheduler tables, scheduler
attempts, claims, worker privileges, or production poller are added.

Migrations 049–052 remain the additive global-control and scheduler-hardening
history. They establish selective global resume, canonical
``current_operation`` values, scheduler ownership/worker attempts, and the
generation-52 exact-attempt protocol. Campaign Operations neither owns nor
duplicates those contracts.

Campaign Operations migration 053 adds append-only campaign control,
cancellation request/settlement, lifecycle cancellation,
reconciliation observation/resolution/cursor, and control-audit evidence.
Every observation has a non-null foreign key to one cursor batch, and cursor
plus exact membership commit atomically. Replay resolves the cursor identity
and never infers membership from request ranges or mutable request state.
Guarded transitions and deferred audit completeness make unbound cancellation
and expired-lease recovery atomic. Lifecycle-owned cancellation is separately
committed and replayed before bound settlement. Dedicated controller,
cancellation coordinator, reconciler, recovery, and lifecycle roles are
NOLOGIN, narrowly privileged, and not granted to `pqxx`. No scheduler or
worker-process authority, production dispatch, completion, or archival is
added.

Campaign Operations migration 054 implements architectural Phase G with one
unique append-only completion event per campaign, a deferred exact audit
reference, exhaustive blocker/classification functions, immutable-history and
post-completion action gates, and a derived status view. A dedicated NOLOGIN
writer has column-scoped inserts and narrow evidence/lock reads only. The
function owner receives only four read-only experiment lifecycle columns;
neither role can mutate lifecycle, scheduler, worker, recommendation,
cancellation-settlement, or reconciliation-resolution authority. Migration
054 rewrites no existing history. Phase H production enablement remains
separate.

ADR-0019, as narrowly amended by ADR-0019A and ADR-0019B, authorizes migration 055,
implemented by Phase H Step 1, as the Phase H
authority and persistence foundation. It adds the immutable alternating
production enable/disable chain, one immutable first production admission per
request, additive Attempt
V2 fields/shape, production audit completeness, owner-DML-safe witness/history
guards, readiness/status, one sealed boundary owner, dedicated NOLOGIN
production capabilities, and a
narrow scheduler-protocol evidence interface. Migration 055 must add no login
membership, backfill no request, reinterpret no V1 identity, and mutate no
scheduler, lifecycle, experiment, or worker row.

The frozen inventory is
`campaign_operations_production_enablement_event`,
`campaign_operations_production_enablement_audit_reference_event`,
`campaign_operations_request_production_admission`, additive V2/admission
columns on the existing dispatch-attempt/dispatch-audit tables, fixed
enable/disable/production-acquisition transition functions, narrow scheduler
protocol snapshot/lock functions, and
`campaign_operations_production_readiness_v1` /
`campaign_operations_production_status_v1` views. Constraints enforce one
alternating head, exact canonical/typed mirrors, V1/V2 exclusive shape, one
admission and first V2/audit, Boolean/admission equivalence, completion gates,
immutability, and owner-DML update/delete/truncate rejection.

The exact production-acquisition function is
`transition_campaign_operations_request_dispatch_production_v2` (61 UTF-8
bytes); the former 64-byte spelling is invalid. Exact catalog tests compare
`pg_proc.proname::text`, octet length, signature, and overload absence without
using truncating `regprocedure` input as sole proof.

The frozen role/function inventory is corrected by ADR-0019A. The seven
ordinary production roles remain `NOLOGIN NOSUPERUSER` with no H1 membership.
The additional infrastructure role
`campaign_operations_h1_boundary_authority` is `NOLOGIN SUPERUSER`, has no
memberships in either direction, and owns every table, sequence, view,
canonical/helper/trigger, scheduler-evidence function, and fixed transition
that can create or validate H1 authorization/evidence. It replaces the
unaccepted ordinary transition-owner role. Superuser/physical administrator
power is outside the threat model; every ordinary former owner, capability,
`pqxx`, generic LOGIN, and reachable role has no ownership, direct evidence
DML, context write, or H1 fixed-transition execution.

The sealed context binds backend PID, top-level XID, kind, request, and key.
Only the fixed transitions write it; success deletes it, rollback removes it,
and a deferred constraint prohibits persistence at commit. Protected relation
ownership prevents an ordinary owner from replacing guards or installing a
recursive trigger. Owner and current/default ACL rules are frozen in
ADR-0019A §§5–8 and catalog-tested with PostgreSQL NULL-ACL semantics.

ADR-0019B makes deployment evidence automatic and fail-closed. The sealed role
is exact down to INHERIT, connection limit, password, validity, role/database
configuration, BYPASSRLS, and an empty recursive graph with no ADMIN OPTION.
Migration preflight never repairs an incompatible existing role. A literal
minimum ownership/signature manifest replaces every wildcard transfer; catalog
audits scan all non-system schemas and object classes for extra ownership,
alternate-schema names/wrappers/overloads/defaults, dependencies, PUBLIC
execution, explicit/default ACL drift, and grant options. The versioned
read-only deployment audit is mandatory before/after upgrade and restore and
before enablement, with stable SQLSTATE/`H1A` diagnostics.

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
Phase 6D ratification events preserve the exact review-event ID and complete
review version/canonical/hash, reviewer, exact reviewed proposal identity,
approved eligibility, fixed authority role, ``ratified`` decision, distinct
ratifier, basis, and complete ratification canonical/hash. Ratification-event
ID and creation timestamp remain metadata outside identity, and ratification
grants no Campaign Operations or other operational authority.

Campaign Operations entities and exact privilege boundaries are authorized
incrementally by ADR-0010 through ADR-0019 and the accepted Campaign
Operations specification. Domain events remain authoritative; guarded
request/reservation projections and optional read projections never replace
their same-transaction event evidence.

For Phase H, the exact current effective enable event is global authority and
immutable admission plus complete Attempt V2/audit is per-request authority.
`production_dispatch_enabled` is only a serialization witness. At commit it
equals existence of exact admission; first admission implies exactly one
matching first V2 attempt and audit. V1 implies false/no production fields; V2
implies true/complete admission and enablement. Immediate false-to-true-only
guarding and deferred bidirectional consistency repeat the proven
`completion_boundary_closed` witness pattern rather than create a second
authority.

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
Phase 6D serializes one review-event-ID conflict domain with a transaction
advisory lock, validates the complete Phase 6C/6B chain and approved
eligibility, fixed role, and separation of duties, and inserts at most one
ratification event. It never changes any upstream row.

### 6.3 Failure semantics

Constraint and serialization failures roll back the complete transaction and
map to stable repository/service outcomes. Finalization does not hide the
initiating error.
Phase 6C returns ``recorded`` for the first valid event,
``existing_identical`` for exact replay, and a deterministic conflict for any
different decision, reviewer, reason, or review identity on that proposal.
Phase 6D returns the same recorded/existing-identical outcomes and conflicts on
any different ratification payload for the same review.

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
Phase 6D similarly uses no upstream row lock: one review-event-ID-scoped
advisory lock plus unique review/proposal foreign keys serializes attempts.

### 7.3 Winner, loser, and retry outcomes

Owning volumes define exact outcomes. The database guarantees no partial
commit; applications do not reinterpret arbitrary SQL failure as idempotency.
Concurrent identical Phase 6C events converge on one row; concurrent differing
events produce one recorded winner and one deterministic conflict.
Concurrent identical Phase 6D ratifications converge on one row; concurrent
differing ratifications produce one recorded winner and one deterministic
conflict.

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
Phase 6D adds no main-program CLI. Its repository lookup/list operations are
read-only, while its explicit typed service owns the governance write.

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
Phase 6D additionally verifies clean and upgrade paths, exact eligibility,
fixed role, separation of duties, concurrent identical/conflicting
ratifications, rollback, NULL-ACL fallback, and read-only lookup/list without
advisory/tuple locks or sequence advancement.

### 9.4 Regression boundaries

Schema changes verify unaffected experiment state, identities, history,
permissions, scheduler behavior, and repository compatibility.
Migration 043 tests preserve Phase 6B manifests/members and experiment and
scheduler sentinels byte-for-byte.
Migration 044 tests preserve Phase 6B manifests/members, Phase 6C reviews, and
experiment/scheduler sentinels byte-for-byte.

Migration 055 tests must preserve historical Attempt V1 and Completion V1
canonicals byte-for-byte, prove additive V1/V2 shape, reconstruct complete
nested enable/admission/attempt/request/completion canonicals in PostgreSQL and
C++, reject same-hash/different-canonical at every level, prove owner-DML
update/delete/truncate rejection and atomic rollback, and inspect exact ACL,
role ownership, search path, PUBLIC revocation, and absence of LOGIN grants.
The fixture rows must exist under schema 054 before migration 055, with exact
canonical bytes and hashes captured before upgrade. Tests also cover exact
replay/conflict for enable, disable, and acquisition before mutable head/CAS;
the full 0a→0b→1→2→3→4→5 lock order with independent connections and
`pg_blocking_pids()`; actual former-owner/SET ROLE/context threats; deferred
context cleanup; and future-object default ACLs.

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
Phase 6D grants the same narrow privilege shape for its ratification table and
sequence. PUBLIC/runtime trigger-function execution is revoked, the invoker-
rights trigger has a pinned safe context, and NULL ACLs are interpreted through
PostgreSQL default ACLs in privilege tests.

Migration 055 creates separate NOLOGIN enabler, disabler, dispatcher,
production Phase 5 transactional, reader, and scheduler-evidence
owner/capability roles plus the sealed, unreachable
`campaign_operations_h1_boundary_authority`. Enabler/disabler/dispatcher are mutually bounded;
`pqxx` receives no membership. The Manager has no test role or scheduler
mutation privilege. Global/admission/attempt/audit evidence and the Boolean
witness are sealed-owned and reject ordinary owner DML. The legacy-named
scheduler evidence owner owns no H1 object; the sealed boundary owner owns the
pinned scheduler helpers and callers receive only exact EXECUTE, not raw table
access.

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
- [ADR-0009](adr/ADR-0009-phase-6d-follow-up-proposal-governance-ratification.md)
- [Campaign Operations ADR-0010 through ADR-0019](adr/README.md)
- [ADR-0019](adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md)
- [Normative Phase H architecture](CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md)
- [Accepted Campaign Operations specification](../../ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md)
- [Phase 6B persistence and preview](../Phase6BRecommendationCampaignFollowUpProposalPersistence.rst)
- [Phase 6C administrative review](../Phase6CRecommendationCampaignFollowUpProposalReview.rst)
- [Phase 6D governance ratification](../Phase6DRecommendationCampaignFollowUpProposalRatification.rst)
- [`migrate_lstm_db.sh`](../../migrate_lstm_db.sh)
- [`Database/migrations`](../../Database/migrations)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established database ownership and migration-governance outline. | ADR-0001 |
| 0.2.0 | 2026-07-20 | Recorded the append-only exact Phase 6B follow-up-proposal schema, collision-safe transaction, and least-privilege read-only preview boundary. | ADR-0001, ADR-0007 |
| 0.3.0 | 2026-07-21 | Recorded one exact append-only Phase 6C administrative review event per proposal, deterministic replay/conflict, validated reload, and read-only presentation without action authority. | ADR-0001, ADR-0004, ADR-0008 |
| 0.4.0 | 2026-07-22 | Recorded proposed append-only Phase 6D governance ratification per eligible approved review, fixed role, mandatory separation of duties, deterministic replay/conflict, and least privilege without Phase 6E authority. | ADR-0001, ADR-0004, ADR-0009 |
| 0.5.0 | 2026-07-24 | Aligned accepted Phase 6D and recorded Campaign Operations' append-only, transactional, least-privilege authority and implemented Phase 1 foundation. | ADR-0009–ADR-0017 |
| 0.6.0 | 2026-07-24 | Added Campaign Operations Phase 2 budget, reservation, durable request, acquisition, audit, and least-privilege persistence contracts. | ADR-0010–ADR-0013, ADR-0017 |
| 0.7.0 | 2026-07-25 | Added isolated Phase 3 durable dispatch, atomic Phase 5 handoff evidence, complete bindings/control ownership, settlement, and narrow NOLOGIN capabilities without production or scheduler enablement. | ADR-0010–ADR-0017 |
| 0.8.0 | 2026-07-25 | Added Phase 4 append-only controls, cancellation coordination and settlement, deterministic reconciliation evidence, and guarded expired-lease recovery without scheduler or worker authority. | ADR-0010–ADR-0017 |
| 0.9.0 | 2026-07-30 | Integrated Phase F as migration 053 after the unchanged 049–052 global-control and scheduler-hardening history; reserved unimplemented Phase G for migration 054. | ADR-0010–ADR-0018 |
| 0.10.0 | 2026-07-31 | Added Phase G immutable operational completion, exact audit/blocker/status evidence, and least-privilege completion capability without lifecycle, scientific, scheduler, or worker authority. | ADR-0014, ADR-0015, ADR-0017 |
| 0.11.0 | 2026-07-31 | Accepted migration 055 architecture for immutable production admission, Attempt V1/V2 compatibility, Boolean/admission equations, Completion V1 nesting, owner-DML guards, least-privilege roles, and no login grants. | ADR-0019 |
| 0.12.0 | 2026-07-31 | Implemented migration 055 H1 authority/persistence, canonical reconstruction, fixed but ungranted production transitions, immutable/deferred constraints, inert production roles, hardened test-only V1 authority, and read-only readiness/status without an H1 production mutation surface. | ADR-0019 |
| 0.13.0 | 2026-08-01 | Applied ADR-0019A's targeted 61-byte acquisition identifier, sealed owner-safe context/ownership inventory, exact fixed-transition replay, complete acquisition lock order, default ACL and SECURITY DEFINER hardening, and genuine pre-055 fixture requirements. | ADR-0019A |
| 0.14.0 | 2026-08-01 | Applied ADR-0019B's exact sealed-role identity/recursive graph, literal minimum ownership and all-schema entry-point manifests, automatic deployment audits, restore A–J, complete ACL/lock/negative matrices, and post-restore historical-byte proof. | ADR-0019B |
| 0.15.0 | 2026-08-03 | Completed migration-055 fail-closed function tuples, post-recovery reacquisition, cross-principal replay, full audit/authority hydration, stable diagnostics, and recursive edge-preserving role evidence. | ADR-0019B |
