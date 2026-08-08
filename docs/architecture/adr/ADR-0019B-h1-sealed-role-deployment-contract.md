# ADR-0019B: H1 sealed-role deployment contract

Status: Accepted
Date: 2026-08-01
Deciders: Project architecture
Amends: ADR-0019A only as listed in §17
Affected increment: Phase H Step 1 (H1)

## 1. Decision

H1 deployment is supported only when the sealed boundary role is exact,
unreachable, minimally privileged, and proven by the versioned automatic audit
defined here. Clean installation, upgrade, role recreation, database restore,
and pre-enablement use the same catalog contract. A checksum or `NOLOGIN`
alone is never deployment evidence.

This amendment corrects the deployment contract only. It adds no H2 caller,
H3 Manager, H4 behavior, production role membership, scheduler mutation,
lifecycle mutation, worker control, or production enablement.

## 2. Stable failure contract

Migration preflight and the deployment audit fail closed. Security invariant
failures use SQLSTATE `42501`; persisted-contract failures use `55000`; corrupt
persisted H1 evidence uses `23514`. Each failure includes exactly one stable
diagnostic prefix:

| Code | SQLSTATE | Meaning |
|---|---|---|
| `H1A001` | `42501` | boundary role missing where creation is not permitted |
| `H1A002` | `42501` | role attribute, password, validity, connection, or configuration mismatch |
| `H1A003` | `42501` | direct, recursive, inherited, or ADMIN OPTION role-graph edge |
| `H1A004` | `42501` | unexpected ownership, executable-as-boundary object, or grant into/out of the role |
| `H1A005` | `42501` | alternate-schema protected name, wrapper, overload, default variant, or dependency |
| `H1A006` | `42501` | explicit object ACL mismatch, PUBLIC execution, or grant option |
| `H1A007` | `42501` | global or schema-specific default ACL mismatch |
| `H1A008` | `55000` | fixed transition, trigger, search path, or supporting-definer mismatch |
| `H1A009` | `55000` | migration filename, SHA-256, or ledger mismatch |
| `H1A010` | `23514` | historical canonical/hash byte proof missing or unequal when required |
| `H1A011` | `55000` | unsupported audit stage or deployment workflow |

Tests assert SQLSTATE, code, the named catalog object or role, and the actual
failing statement. A test that fails earlier at an unrelated ACL or fixture
error is a failure, not a successful negative test.

## 3. Exact sealed-role identity

`campaign_operations_h1_boundary_authority` has exactly:

| Property | Required value |
|---|---|
| LOGIN | `false` (`NOLOGIN`) |
| SUPERUSER | `true` |
| INHERIT | `true` |
| CREATEDB | `false` |
| CREATEROLE | `false` |
| REPLICATION | `false` |
| BYPASSRLS | `false` |
| connection limit | `-1` |
| password verifier | NULL |
| valid-until | NULL |
| role configuration | NULL/empty |
| database-specific role configuration | no rows |
| memberships granted to role | none |
| role granted to members | none |
| ADMIN OPTION | none on every edge involving the role |
| operational credential | none |

`SUPERUSER` is the narrow infrastructure-ownership decision accepted by
ADR-0019A. Physical cluster administrators remain outside the application
threat model, but their changes are inside the deployment audit threat model.
`INHERIT` is frozen true because an empty graph makes it inert and exact catalog
identity is preferable to a release-dependent default assumption.

If the role does not exist during migration, migration 055 may create it with
the exact tuple above. If it already exists, migration 055 inspects the complete
identity before any mutation. One mismatch raises `42501 H1A002`; it never
alters, clears, revokes, resets, or otherwise normalizes the role. The same
fail-closed rule applies to restored and separately recreated roles.

The seven ordinary H1 roles are exact `NOLOGIN NOSUPERUSER INHERIT NOCREATEDB
NOCREATEROLE NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1` roles with no
password, validity, configuration, credential, or role-graph edge. Migration
may create a missing role; an incompatible pre-existing role fails closed.

## 4. Complete role-graph invariant

The graph over `pg_auth_members` is empty in both directions for all eight H1
roles. The invariant is recursive rather than a single direct-membership test:

- no H1 role is directly or indirectly granted to another role;
- no H1 role is directly or indirectly a member of another role;
- no role can reach the boundary role with `SET ROLE` or `SET LOCAL ROLE`;
- no LOGIN can inherit boundary authority through any `INHERIT` chain;
- no edge involving an H1 role has ADMIN OPTION;
- isolated-test and production capabilities cannot be combined;
- `pqxx`, scheduler, capability, owner, generic LOGIN, and test roles have no
  direct or transitive path to or from the boundary role.

The audit reports both the shortest role path and whether any traversed edge is
admin-grantable. Acceptance fixtures cover direct, two-level, three-level,
NOLOGIN-only intermediate, ADMIN OPTION, combined test/production, and
boundary-inherits-another-role graphs. Independent LOGIN sessions prove both
`SET ROLE` and inherited privilege failure.

## 5. Exact minimum owned-object manifest

The boundary role owns only the following H1 authority objects and the indexes,
row/composite types, and array types PostgreSQL couples to these exact tables.
Coupled objects are compared by dependency OID, never by a name wildcard.

### 5.1 Relations

| Type | Schema-qualified object | Prior owner | Reason |
|---|---|---|---|
| table | `public.campaign_operations_production_transition_context` | created by 055 | sealed transaction authorization |
| table | `public.campaign_operations_production_enablement_event` | created by 055 | immutable enable/disable head |
| table | `public.campaign_operations_production_enablement_audit_reference_event` | created by 055 | enable/disable audit completeness |
| table | `public.campaign_operations_request_production_admission` | created by 055 | immutable first admission |
| table | `public.campaign_operations_operational_request` | `campaign_operations_owner` | Boolean/admission witness and request CAS |
| table | `public.campaign_operations_dispatch_attempt` | `campaign_operations_owner` | additive Attempt V2 guard |
| table | `public.campaign_operations_dispatch_audit_reference_event` | `campaign_operations_owner` | production audit reference guard |
| table | `public.campaign_operations_completion_event` | `campaign_operations_owner` | post-completion production rejection and V2 nesting |
| table | `public.campaign_operations_completion_audit_reference_event` | `campaign_operations_owner` | completion evidence pairing |
| view | `public.campaign_operations_production_readiness_v1` | created by 055 | read-only readiness |
| view | `public.campaign_operations_production_status_v1` | created by 055 | read-only status |

The exact serial/identity sequences dependency-owned by the eight non-context
tables above are included. No schema, foreign table, materialized view,
standalone type/domain, operator, cast, publication/subscription, large object,
event trigger, or unrelated relation may be boundary-owned.

### 5.2 Functions and triggers

The authoritative function manifest is the literal, schema-qualified signature
array in migration 055's `campaign_operations_h1_owned_function_manifest_v1`
audit CTE. It contains only:

1. the three fixed transitions and their three exact replay helpers;
2. exact context validator/empty-at-commit functions;
3. exact H1 canonical/hash and scheduler-evidence snapshot/lock helpers called
   by those transitions;
4. the five accepted lock helpers required for stages 1 through 5;
5. exact trigger functions attached to the nine protected tables;
6. exact validation helpers reached by those trigger functions; and
7. the deployment audit itself.

Each literal entry records object kind, schema, `pg_get_function_identity_arguments`,
prior owner, new owner, architectural reason, required ACL, `prosecdef`, and
required `proconfig`. Missing expected entries raise `H1A008`; extra owned
entries raise `H1A004`. Cancellation, reconciliation, recovery, lifecycle,
recommendation, scheduler mutation, experiment, worker, completion transition,
and prior-phase functions not in the literal manifest remain with their prior
owners. The former `%campaign_operations%` and trigger-discovery ownership
loops are prohibited.

## 6. All-schema entry-point and object contract

Migration preflight and every database audit scan every schema except
`pg_catalog`, `information_schema`, `pg_toast`, and temporary/internal `pg_%`
schemas. They enumerate schemas, relations and partitions, sequences, ordinary
and materialized views, foreign tables, functions, procedures, aggregates and
every overload, types/domains, operators, casts, ordinary/event triggers,
rules, publications/subscriptions where supported, large objects, explicit
dependencies, ACLs, and ownership.

Only the exact `public` manifest may be boundary-owned or execute with the
boundary owner. The scan rejects:

- a protected name/signature or 63-byte prefix in another schema;
- extra overloads, default parameters, variadic forms, procedures, aggregates,
  operators, or casts sharing a protected entry point;
- any alternate-schema boundary-owned object;
- a PUBLIC-executable boundary-owned `SECURITY DEFINER` function;
- an ordinary- or event-trigger entry point outside the exact manifest;
- a wrapper with a catalog dependency on a protected function;
- a SQL/PL function body that references a protected schema-qualified name
  where PostgreSQL does not retain a body dependency;
- a rule, trigger, cast, or operator dependency exposing protected authority.

The source-reference check is a conservative supplement to catalog dependency
closure, not the primary allowlist. Dynamic SQL is forbidden in every allowed
boundary-owned function. An unexpected object fails; it is never transferred,
renamed, dropped, revoked, or repaired automatically.

## 7. Current and default ACL matrices

The explicit ACL matrix expands NULL ACLs through `acldefault` and compares
grantee, privilege, and grant option exactly. Boundary-owned tables/sequences
permit only owner-implicit access plus the exact existing workflow grants and
the frozen H1 reader grants. Fixed transitions and internal helpers are owner
executable only in H1. Scheduler-evidence readers receive EXECUTE only on the
two exact helpers. Production readers receive only the documented SELECT and
read-helper privileges. PUBLIC, `pqxx`, LOGIN, test, and other H1 capabilities
receive no boundary execution or evidence/context mutation. No allowed grant
has grant option.

For each of the boundary owner, `campaign_operations_owner`, and scheduler
evidence owner, global and `public`-specific defaults are exact for object kinds
`f`, `r`, `S`, `T`, and `n` where PostgreSQL supports the command. Future
functions/procedures have no PUBLIC execution; future tables, sequences,
schemas, and types/domains grant nothing to PUBLIC, `pqxx`, any H1 capability,
test role, or LOGIN. Extra `pg_default_acl` grantees or grant options raise
`H1A007`; migration never silently revokes an incompatible existing default.

Disposable probes create a function, procedure, table, sequence, schema, type,
and domain as every owner and expand the resulting ACL. Probe objects are never
created in a production database.

## 8. Automatic deployment audit

`Scripts/CampaignOperationsH1DeploymentAudit.sh` is the one supported command
surface. It is versioned `h1-deployment-audit-v1`, read-only, uses
`ON_ERROR_STOP`, and returns nonzero on any mismatch. Its stages are:

```text
pre-upgrade
post-upgrade
pre-restore
post-role-recreation
post-database-restore
pre-enablement
```

Role-only stages query cluster catalogs before a database restore. Database
stages invoke `public.campaign_operations_h1_deployment_audit_v1`, compare the
literal migration filename/SHA-256 to `schema_migrations`, and request the
historical-byte check when the fixture ledger says it applies. The script never
creates, alters, grants, revokes, or repairs a role or database object.

Exact invocation:

```bash
Scripts/CampaignOperationsH1DeploymentAudit.sh \
  --stage <stage> --host <socket> --port <port> \
  --user <cluster-auditor> --database <database>
```

The cluster auditor must already have catalog visibility and EXECUTE on the
audit function. Production application credentials are unsupported. Audit
success is one line: `H1_DEPLOYMENT_AUDIT_V1_OK stage=<stage>`.

## 9. Supported backup/restore matrix

| Scenario | Before restore | Permitted action | After restore | Result |
|---|---|---|---|---|
| A database-only dump, roles recreated separately | `pre-restore`, then exact role recreation | owner-preserving database restore | `post-role-recreation`, `post-database-restore` | supported only if both pass |
| B roles-only dump plus database restore | audit role dump in an empty disposable target | restore exact roles, then database | both post audits | supported only if exact |
| C safe pre-existing target roles | exact role-only audit | database restore | full post audit | supported |
| D incompatible pre-existing roles | `H1A002` | none | not applicable | rejected before restore |
| E ADMIN OPTION or graph edge | `H1A003` | none | not applicable | rejected before restore |
| F altered role attributes/settings | `H1A002` | none | not applicable | rejected before restore |
| G unexpected boundary-owned object | role audit may pass | restore only into disposable verification target | `H1A004` | deployment blocked |
| H differing default ACL | role audit may pass | disposable verification restore | `H1A007` | deployment blocked |
| I PUBLIC EXECUTE on protected signature | role audit may pass | disposable verification restore | `H1A006` | deployment blocked |
| J alternate-schema wrapper/overload | role audit may pass | disposable verification restore | `H1A005` | deployment blocked |

`pg_dump` database-only output does not contain the cluster role graph; valid
database bytes and a valid migration ledger therefore never waive either role
audit. Unsupported restore tooling or a skipped stage raises `H1A011` and is a
deployment blocker.

## 10. Upgrade contract

Before any DDL, migration 055 performs the exact role, graph, pre-existing
ownership, all-schema entry-point, and default-ACL preflight. It may create only
missing exact roles and the enumerated H1 objects. It uses only literal
ownership statements and then runs the same database audit used after restore.
Every hostile attribute and object-class fixture verifies transaction rollback
and proves the named `H1A` branch was reached.

## 11. Lock-path acceptance

The accepted order remains exactly:

```text
0a scheduler evidence
0b production-enable domain
1  authorization
2  budget
3  campaign/completion
4  reservations ascending
5  requests ascending
6+ lifecycle domains
```

Independent connections identify themselves with `application_name`; each
probe records `pg_blocking_pids()` and granted/waiting `pg_locks`. The matrix
covers acquisition versus enable, disable, completion, cancellation,
reconciliation/recovery, authorization revocation, budget mutation,
reservation expiry, same-request acquisition, same-campaign different
requests, unrelated campaigns, uncommitted replay, rollback, and absence of a
reverse wait. A sleep only holds a verified lock and is never itself proof.

## 12. Negative-test authenticity

The executable traceability table records requirement, fixture, complete
principal graph, exact statement/function, expected SQLSTATE, expected stable
diagnostic or constraint, observed failure point, prerequisite proof, and
result. It includes each role attribute; all catalog object classes; wrapper,
overload, graph, `SET ROLE`, default-ACL, upgrade/restore, context, recursive
trigger, replay conflict/corruption, all five canonical/hash levels, hydration,
post-completion, lock, and historical-byte cases. Missing evidence is reported
as missing, never PASS.

## 13. Historical bytes

A genuine schema-054 Attempt V1 and Completion V1, including nested Attempt V1
bytes in Completion V1 request evidence, is captured as `bytea` before 055.
The fixture compares all canonical and hash bytes after migration, supported
dump, exact role recreation, restore, and post-restore audit. Migration and
restore perform no rewrite or reinterpretation. The audit requires the
historical snapshot marker only in the disposable historical-proof workflow;
absence of application history in a clean production database is not an error.

## 14. Fixed transitions and definers

The exact three transitions remain owner-only, `VOLATILE`, `PARALLEL UNSAFE`,
non-leakproof, non-variadic, without defaults/overloads, with pinned
`pg_catalog, public`, exact arguments, exact return types, and PUBLIC revoked.
The acquisition name remains the 61-byte
`transition_campaign_operations_request_dispatch_production_v2`. Every allowed
supporting `SECURITY DEFINER` is similarly owner/ACL/search-path checked.

## 15. Deployment and rollback

H1 remains inert and default-off. No production role is granted. Before any H2
enablement, `pre-enablement` audit success is mandatory and separately retained
as deployment evidence. Audit failure blocks deployment; it does not authorize
automatic repair. Restore verification and destructive fixtures run only in
new disposable local clusters.

## 16. Acceptance disposition

Implementation may report only `READY_FOR_FOCUSED_INDEPENDENT_REVERIFICATION`
or `NOT_READY_FOR_REVERIFICATION`. Readiness to commit remains exclusively a
future independent CEE decision.

## 17. Amendment scope

This ADR narrows ADR-0019A §§4–8 and §§13–15 by replacing implicit clean-state
assumptions, wildcard transfers, partial default ACLs, and prose-only restore
review with the exact sealed-role deployment contract above. The 61-byte name,
replay semantics, context design, lock order, canonical identities, H1/H2/H3/H4
boundaries, scheduler/lifecycle isolation, historical V1 meaning, and all other
accepted ADR-0019/0019A decisions are unchanged.

## 18. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-01 | Accepted the exact sealed-role identity/graph, literal ownership and all-schema entry-point manifests, current/default ACL matrices, automatic deployment audit, restore A–J, complete lock/negative matrices, and post-restore byte proof. |
| 1.1 | 2026-08-03 | Recorded the architecture-preserving implementation-consistency correction for the protected-function tuple, diagnostic namespace, and recursive authority evidence. |

## 19. Final implementation-consistency reference

Migration 055 applies one exact protected-function catalog contract to all 48
protected signatures in preflight and in the read-only deployment audit. The
tuple covers schema-qualified identity, signature, owner phase, function kind,
language, security mode, volatility, parallel safety, default count, variadic
type, configuration/search path, and ACL inventory. An exact new H1 function
owned by an ordinary role is rejected before DDL; ``CREATE OR REPLACE`` never
normalizes it. The supported schema-054 form permits only the accepted safe
``pg_catalog, public, pg_temp`` path on an exact legacy function owned by
``campaign_operations_owner``.

The diagnostic namespace is exclusively ``H1A001`` through ``H1A011``.
Recursive role evidence reports every shortest-path edge with direction,
complete role sequence, total depth, edge depth, and ``ADMIN OPTION``.

The correction evidence and complete consistency matrix are retained in
``CampaignOperations_PhaseH_H1_FinalArchitectureConsistencyCorrection_Output.md``.
ADR-0019B's focused-reverification evidence enum in §16 remains frozen; the
correction report maps its zero-defect ready value to the request-level
``READY_FOR_INDEPENDENT_REVERIFICATION`` disposition without changing any
deployment, evidence, or authorization contract.
