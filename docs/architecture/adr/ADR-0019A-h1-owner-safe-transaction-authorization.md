# ADR-0019A: H1 owner-safe transaction authorization correction

Status: Accepted
Date: 2026-08-01
Deciders: Project architecture
Amends: ADR-0019 and the Phase H normative architecture, only as listed in §16
Affected increment: Phase H Step 1 (H1)

Targeted successor: [ADR-0019B](ADR-0019B-h1-sealed-role-deployment-contract.md)
narrows this ADR's deployment assumptions with an exact role tuple, complete
role graph, literal ownership/all-schema allowlists, automatic deployment
audit, restore matrix, and complete acceptance evidence. Its correction does
not change the identifier, replay, context, lock order, or phase boundaries
accepted here.

## 1. Problem statement

Focused independent CEE reverification concluded
`PHASE_H_H1_NOT_READY_TO_COMMIT`. PostgreSQL cannot represent the frozen
64-byte acquisition-function identifier, the ordinary transition owner can
forge the persistent context row and grant access to its owned functions, the
three fixed transitions do not implement exact replay, and acquisition takes
the request lock before campaign/completion. Default privileges and several
`SECURITY DEFINER` boundaries are also insufficiently frozen.

The correction is deliberately limited to the H1 database authority boundary,
its exact identifier, replay, lock order, hydration, and verification evidence.
It does not authorize an H2 caller or any H2, H3, or H4 behavior.

## 2. Authoritative findings

The focused reverification findings are accepted as architecture input:

1. an ordinary owner always retains implicit privileges on its objects, so
   `REVOKE` cannot protect an owner-writable context or owner-alterable guard;
2. direct and inherited membership or `SET ROLE` makes a `NOLOGIN` ordinary
   owner reachable;
3. caller-controlled GUCs, `pg_trigger_depth()`, PID/XID alone, and current
   absence of LOGIN membership are not authority;
4. replay must be decided from complete immutable evidence before mutable head
   or request-CAS validation;
5. the fixed acquisition path must establish the complete global lock order;
6. PostgreSQL catalog identity requires exact `pg_proc.proname::text`, not a
   truncating parser or `regprocedure` lookup alone.

## 3. Exact acquisition identifier

The impossible former name is
`transition_campaign_operations_request_dispatching_production_v2` (64 UTF-8
bytes). Its authoritative replacement is
`transition_campaign_operations_request_dispatch_production_v2` (61 UTF-8
bytes). The replacement is explicit, unambiguous, versioned, follows the
existing `transition_campaign_operations_<object>_<action>_<mode>_vN`
convention, and does not share its first 63 bytes with any existing function.

Migration, C++, architecture, documentation, checksums, and tests MUST use the
61-byte text exactly. Migration and catalog tests MUST set notices to fail or
capture and reject identifier-truncation notices, compare
`pg_proc.proname::text` byte-for-byte, compare `octet_length(proname) = 61`,
and reject any same-prefix catalog function or overload. A successful
`regprocedure` lookup is necessary signature evidence but is not proof of the
exact catalog name.

## 4. Owner-safe transaction authorization

### 4.1 Sealed boundary owner

PostgreSQL has no mechanism that denies an object owner the ability to alter or
grant its object. Therefore no ordinary role may own a protected H1 table,
sequence, trigger function, canonical/hash helper, fixed transition, scheduler
evidence helper, completion gate, role-graph helper, or readiness/status
object. Migration 055 creates the exact cluster role
`campaign_operations_h1_boundary_authority` as `NOLOGIN SUPERUSER` with
`NOCREATEDB NOCREATEROLE NOREPLICATION`. It has no direct or inherited members,
is not a member of another role, and has no grantee with `ADMIN OPTION`.

This sealed role is infrastructure ownership, not a production capability.
It replaces the unauthorized ordinary
`campaign_operations_production_transition_owner`; that old role is not
created or used by migration 055. Migration 055 fails with SQLSTATE `42501`
if the sealed role or any H1 role has pre-existing membership, if the sealed
role has LOGIN, or if any ordinary role owns a protected boundary object.
Migration 055 never silently preserves or repairs unsafe membership.

Superuser and physical cluster-administrator power are explicitly outside the
supported threat model. This exception is necessary and exact: PostgreSQL
object ownership itself is grant/alter authority. No application login,
`pqxx`, capability role, ordinary object owner, or reachable `SET ROLE` target
may reach the sealed owner.

### 4.2 Exact context

The context table remains because operation identity must survive nested
trusted triggers while rejecting recursive mutation for another operation. It
is owned by the sealed authority and has no privileges for PUBLIC, `pqxx`, any
ordinary owner, or any capability role. Its exact key is backend PID plus
top-level transaction ID; its value binds transition kind, request ID where
applicable, and operation key. Only the three fixed sealed-owner transitions
may insert or delete it.

Each fixed transition inserts context only after replay, immutable-input, and
lock/CAS validation and before its first guarded mutation. Every guarded
trigger checks the exact PID, top-level XID, kind, request ID, and operation
key through a sealed, pinned, schema-qualified validator. A row for one request
or transition kind cannot authorize another. A deferred constraint trigger
requires the context table to be empty at commit. Success deletes the row
before return; statement/subtransaction rollback removes an insertion; caught
exceptions cannot retain it; top-level commit cannot persist it; rollback,
savepoint rollback, pooled-session reuse, and concurrent sessions therefore
carry no authority forward.

Ordinary roles cannot create equivalent evidence because they cannot own,
alter, trigger-modify, insert, update, delete, truncate, grant, or execute an
ungranted writer on any context or protected object. All protected triggers and
functions use an explicit `pg_catalog, public` search path, schema-qualified
references, static SQL, exact signatures, and no permissive wrapper or
overload. The sealed owner also prevents the owner of the former request,
attempt, audit, admission, enablement, or completion tables from replacing a
guard with a recursive trigger. Trusted recursive invocation remains bound to
the one exact context tuple and fails closed for any different operation.

`pg_trigger_depth()`, an application-settable GUC, implicit owner privilege,
absence of current LOGIN membership, and a persistent owner-writable row are
not used as authority.

## 5. Frozen corrected role inventory

All membership columns below mean direct and inherited membership. Migration
055 installs an empty graph and grants no LOGIN membership.

| Role | LOGIN | Memberships | Owned objects | Explicit privileges | Prohibited privileges | May grant | Fixed transitions | Context/evidence write | Deployment rule |
|---|---:|---|---|---|---|---:|---|---|---|
| `campaign_operations_h1_boundary_authority` | no | none in/out | all H1 protected tables/sequences/views and every boundary/helper function listed in §6 | implicit sealed ownership only | LOGIN, ordinary reachability, membership, application use | superuser only; outside threat model | owns; no H1 caller grant | owns; fixed functions only | role attributes and empty graph required; any mismatch fails |
| `campaign_operations_production_enabler` | no | none at H1 | none | none at H1 | direct DML, context, disable, dispatch, ownership | no | no at H1; H2 may grant enable only | no | later dedicated enable login only |
| `campaign_operations_production_disabler` | no | none at H1 | none | none at H1 | direct DML, context, enable, dispatch, ownership | no | no at H1; H2 may grant disable only | no | later dedicated disable login only |
| `campaign_operations_production_dispatcher` | no | none at H1 | none | none at H1 | direct DML, context, enable, disable, ownership | no | no at H1; H2 may grant acquisition only | no | later Manager login only |
| `campaign_operations_production_phase5_transactional` | no | none at H1 | none | none at H1 | context and H1 fixed transitions | no | no | no | later Manager login only with dispatcher |
| `campaign_operations_production_reader` | no | none at H1 | none | SELECT on H1 evidence/views | every mutation, context, ownership | no | no | no | may later be granted to read-only/Manager login |
| `campaign_operations_scheduler_protocol_evidence_owner` | no | none at H1 | none after 055 | none | scheduler DML, boundary ownership, transition execution | no | no | no | retained name for compatibility; sealed owner now owns helpers |
| `campaign_operations_scheduler_protocol_evidence_reader` | no | none at H1 | none | EXECUTE snapshot/lock helpers | direct scheduler DML, context, fixed transitions | no | no | no | later Manager/readiness membership only |

`pqxx`, generic LOGIN roles, isolated-test capabilities, direct/inherited
combined isolated/production memberships, and `SET ROLE` into any listed role
receive no H1 mutation authority. Before H2 enablement, deployment repeats the
complete graph audit and rejects prohibited combinations.

## 6. Exact ownership and ACL rules

| Object family | Owner | Allowed ordinary ACL | Required denial |
|---|---|---|---|
| context table and its empty-at-commit/validation functions | sealed authority | none | all table/function privileges to PUBLIC, `pqxx`, owners, capabilities |
| operational request; enablement, enablement-audit, admission, Attempt, dispatch-audit, completion tables and owned sequences | sealed authority | existing workflow grants only through exact functions; production reader SELECT only on H1 evidence | no ordinary direct H1 evidence DML; no trigger ownership |
| enable/disable/acquisition fixed transitions | sealed authority | none in H1 | PUBLIC, `pqxx`, every ordinary owner/capability/reader denied |
| canonical/hash, witness, audit, completion, scheduler-evidence, role-graph, readiness/status helpers used by H1 | sealed authority | exact reader/helper EXECUTE only where frozen | PUBLIC EXECUTE denied for every `SECURITY DEFINER`; no overload/wrapper |
| readiness/status views | sealed authority | production reader SELECT | no mutation or ownership |

Every fixed transition is `VOLATILE`, `PARALLEL UNSAFE`, non-leakproof, has no
defaults or variadic parameters, and has exactly one signature. ACL catalog
tests interpret NULL ACL with `acldefault`; implicit owner privileges are never
reported as revocable protection.

## 7. Default privileges

For the sealed owner and every ordinary owner that still creates Campaign
Operations objects in the H1 schema, migration 055 explicitly applies:

```sql
ALTER DEFAULT PRIVILEGES FOR ROLE <owner>
  REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE <owner> IN SCHEMA public
  REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE <owner> IN SCHEMA public
  REVOKE ALL ON SEQUENCES FROM PUBLIC;
```

Pre-existing default ACLs are catalog-audited. Any PUBLIC or ordinary-role
function execution, table write, sequence use, or grant option fails migration
rather than being silently retained. Tests cover both NULL ACL expansion and
explicit `pg_default_acl` rows, then create future objects as each owner in a
disposable cluster and prove no leakage.

## 8. `SECURITY DEFINER` rules

Every `SECURITY DEFINER` boundary involved in enablement, disablement,
admission, Attempt V2, witness mutation, audit insertion, completion gating,
scheduler evidence, role-graph evaluation, and readiness/status has:

- exact sealed ownership where mutation or authority is possible;
- explicit `SET search_path = pg_catalog, public`;
- schema-qualified types, relations, functions, operators where resolution can
  be caller-controlled;
- static SQL and no caller-provided identifier or dynamic SQL;
- PUBLIC execution revoked and an enumerated exact ACL;
- no default arguments, variadic signature, overload, wrapper, or same-prefix
  truncation collision.

## 9. Exact replay algorithm

Enable, disable, and acquisition use the same decision order:

1. validate operation-key bytes;
2. look up immutable evidence by the exact operation key (and request ID for
   acquisition) before reading a mutable head or applying request CAS;
3. require the complete root row, its exact typed mirrors, canonical and hash,
   nested predecessor/scheduler/build/admission/enable/request evidence, and
   the exactly one matching audit row;
4. reconstruct all canonical bytes from the supplied logical fields and stored
   immutable nested evidence;
5. if every typed value and canonical byte/hash is exact, return the original
   stored composite row without consulting current mutable state;
6. if the key exists with any changed field, or a hash matches while canonical
   bytes differ, raise SQLSTATE `23505` with the transition-specific
   `conflicting replay` diagnostic;
7. if evidence is partial, missing, malformed, contradictory, or cannot be
   reconstructed exactly, raise SQLSTATE `23514` with the transition-specific
   `replay evidence corrupt` diagnostic;
8. only proven absence proceeds to locks and mutable validation; after the
   serialization lock, repeat the lookup to close the concurrent-insert race.

Enable replay survives later head or scheduler advancement. Disable replay
survives later head advancement. Acquisition replay survives request state,
version, lease, completion, recovery, or enable-head advancement wherever the
stored immutable evidence is still complete. Mutable state is never evidence
that the operation committed.

## 10. Full lock order

The unchanged global order is:

```text
0a scheduler protocol evidence
0b production-enable domain
1  authorization
2  budget
3  campaign/completion
4  reservations ascending
5  requests ascending
6+ existing Phase 5/lifecycle domains
```

| Path | Exact order |
|---|---|
| enable new operation | 0a shared, 0b exclusive, replay recheck, append |
| disable new operation | 0b exclusive, replay recheck, append; no scheduler dependency |
| acquisition new operation | replay lookup, 0a shared, 0b shared, 1 authorization head, 2 budget head, 3 campaign/completion, 4 request reservation(s) ascending, 5 request, replay recheck/CAS, then existing 6+ |
| completion/recovery/cancellation/handoff | prior accepted order unchanged; they never acquire a lower-numbered lock while holding a higher one |

Acquisition may optimistically read request identifiers before locking, but it
MUST revalidate them after the request lock. It uses the existing authoritative
`lock_campaign_operations_authorization_head`,
`lock_campaign_operations_budget_head`, `lock_campaign_operations_campaign`,
`lock_campaign_operations_reservation`, and
`lock_campaign_operations_request` boundaries. Deterministic independent-
connection tests use `pg_blocking_pids()` catalog evidence at every boundary;
sleep timing is not proof.

## 11. Migration 055 amendment

Migration 055 is updated in place while uncommitted. It replaces the impossible
name, removes the ordinary transition-owner design, creates/hardens the sealed
owner, rejects unsafe pre-existing role/default-ACL state, assigns protected
ownership, pins and qualifies every boundary, implements complete exact replay,
establishes the full acquisition lock order, adds no backfill, and preserves
transactional/idempotent supported replay. Its filename remains
`055_campaign_operations_production_admission_foundation.sql`; SHA-256,
embedded C++ checksum, and ledger evidence are regenerated from the final
bytes.

## 12. H1/H2/H3/H4 boundary confirmation

H1 remains default-off and exposes no supported enable, disable, acquisition,
handoff, Manager, scheduler-control, lifecycle-control, environment switch, or
production role deployment. Fixed mutations exist only as ungranted database
boundaries. H2 remains responsible for callers and exact single dispatch; H3
for bounded run-once; H4 remains excluded pending separate acceptance.

## 13. Rollout and rollback

Install and all mutation/ACL/role tests run only in a disposable cluster. H1
rollout remains inert. Before H2, operators must verify the exact checksum,
catalog names, sealed-owner attributes/empty graph, default ACLs, fixed
function ACLs, scheduler generation-52 evidence, and safe-window process
tests. Operational rollback remains disable-first once H2 exists; H1 rollback
is transactional migration rollback before ledger acceptance. History is never
deleted or rewritten.

## 14. Compatibility and historical fixtures

An authoritative schema-054 fixture MUST persist at least one complete Attempt
V1 and Completion V1, including every nested evidence row, before migration 055
is applied. The harness records exact canonical text as bytea and exact hashes,
applies 055, and byte-compares all values. Migration 055 performs no backfill,
row rewrite, or V1 reinterpretation. Completion V1 continues nesting stored
attempt canonical bytes; V2 support is additive.

## 15. Acceptance matrix

| Requirement | Required evidence |
|---|---|
| exact identifier | notice-free migration; exact `proname::text`; 61 octets; no overload/prefix collision |
| owner safety | actual former ordinary owners, all capabilities, `pqxx`, generic LOGIN, direct/inherited/combined membership and `SET ROLE` fail at intended guard; unsafe pre-existing graph fails migration |
| context lifecycle | success, rollback, savepoint, subtransaction, caught exception, pooled reuse, concurrent sessions, collision, nested and recursive triggers |
| replay | exact and every-field conflict for enable/disable/acquisition; same-hash/different-canonical; partial/malformed/missing nested evidence; post-head/state advancement |
| lock order | independent connections plus `pg_blocking_pids()` for 0a, 0b, 1, 2, 3, 4, 5 and no reverse wait |
| ACL/default ACL | exact owner/current ACL/NULL ACL/default ACL catalog matrix and future-object probes |
| definers | owner, search path, qualification, static SQL, ACL, no overload/wrapper for every boundary |
| hydration | per-column stored enablement, admission, Attempt V2 corruption fails before repository/service return |
| history | genuine pre-055 Attempt V1 and Completion V1 canonical/hash byte equality after upgrade |
| scope | default-off and no H2/H3/H4 caller or process reachability |

Every negative test records the exact statement/function, expected and actual
SQLSTATE, expected constraint or function-specific diagnostic, and verifies
that no prerequisite ACL/shape failure masked the intended branch. The H1
implementation report carries the requirement-to-branch traceability table.

## 16. Exact amendment scope

This ADR amends only:

- ADR-0019 and Phase H text naming the 64-byte acquisition function;
- Phase H §§7, 9, 13, 14, 16, and 17 where they describe transaction context,
  ownership/ACLs, replay ordering, lock acquisition, the seven-role inventory,
  and migration-055 objects;
- Volume XII's migration-055 role/function/ownership/default-ACL inventory;
- the corresponding revision histories and ADR index.

All other ADR-0019 text remains unchanged, including canonical identities,
scheduler and lifecycle isolation, authority hierarchy, Completion V1 meaning,
H1–H4 boundaries, default-off deployment, disable-first rollback, and the
prohibition on production mutation or role deployment in H1.

## 17. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-01 | Accepted the targeted H1 identifier and owner-safe transaction-authorization correction after focused independent CEE reverification. |
| 1.1 | 2026-08-01 | Recorded ADR-0019B's narrow sealed-role deployment-contract correction; all transaction-authorization decisions remain otherwise unchanged. |
