# ADR-0019C: H2 privilege and deployment reachability contract

Status: Accepted targeted amendment
Date: 2026-08-08
Depends on: ADR-0019, ADR-0019A, ADR-0019B, and accepted Phase E roles from migrations 045–054

This amendment resolves the H2 reachability gap without changing H1. It is
the authority for migration 056 ACLs, migration 057's additive production
handoff transition, deployment-time LOGIN membership, and the H2 deployment
audit. Migration 055, its checksum, manifests, fixed
function signatures, sealed owner, and H1 audit meaning are immutable.

## 1. Exact role inventory

Migration 055 creates these NOLOGIN roles. Migration 056 neither creates
LOGINs nor grants any role to a LOGIN:

| Role | H2 purpose |
|---|---|
| `campaign_operations_production_enabler` | one enable transition only |
| `campaign_operations_production_disabler` | one disable transition only |
| `campaign_operations_production_dispatcher` | one production acquisition only |
| `campaign_operations_production_phase5_transactional` | existing Phase E transactional handoff |
| `campaign_operations_production_reader` | H1 read-only evidence/readiness/status |
| `campaign_operations_scheduler_protocol_evidence_reader` | H1 snapshot/lock helpers |
| `campaign_operations_h1_boundary_authority` | sealed H1 owner; never an application capability |

All capability roles remain `NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE
NOREPLICATION NOBYPASSRLS INHERIT`, with no password, expiry, database role
settings, or grant option. The sealed owner remains the only owner of the H1
protected boundary and is unreachable from every application role.

## 2. Exact H2 ACL contract

The only fixed-transition grants are:

| Principal | Exact function | Privilege |
|---|---|---|
| `campaign_operations_production_enabler` | `public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)` | `EXECUTE` only |
| `campaign_operations_production_disabler` | `public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)` | `EXECUTE` only |
| `campaign_operations_production_dispatcher` | `public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)` | `EXECUTE` only |

No role receives another fixed transition. No grant has `ADMIN OPTION`. The
three functions remain `SECURITY DEFINER`, `VOLATILE`, untrusted parallel
behavior, zero defaults/variadic arguments, fixed `pg_catalog, public` search
path, and sealed-owner ownership under the unchanged H1 catalog contract.

Supporting H2 grants are:

| Principal | Supporting authority |
|---|---|
| production dispatcher | `SELECT` on the three immutable production evidence tables |
| production Phase 5 transactional | the same three H2 evidence `SELECT` grants, `EXECUTE` on `campaign_operations_scheduler_protocol_evidence_lock_v1()`, and the established migration-048 Phase E transactional grants copied to this distinct production capability |
| production reader | unchanged H1 `SELECT` on enablement/admission/readiness/status objects, plus `EXECUTE` on the read-only `campaign_operations_production_readiness_snapshot_v1()` adapter |
| scheduler evidence reader | unchanged H1 `EXECUTE` on snapshot and lock helpers |

These are read/helper privileges only. Dispatcher and Phase 5 have no direct
DML on protected H1 production tables, no scheduler-table access, and no
enable/disable transition. Existing Phase E grants from migration 048 remain
the authority for common handoff; H2 adds no new Phase E DML.

## 3. Actor contracts

### Enable

The enabler invokes only the exact enable function. Enable validation reads
H1 readiness/evidence through `production_reader` and the snapshot helper
through `scheduler_protocol_evidence_reader`. Its deployment identity is
exactly `enabler + production_reader + scheduler_protocol_evidence_reader`.
It may not have disabler, dispatcher, Phase 5, isolated-test, recovery,
cancellation, completion-writer, generic `pqxx`, or sealed-owner reachability.

### Disable

The disabler invokes only the exact disable function. Disable intentionally
has no scheduler-readiness dependency. Its replay path needs immutable
enablement event/audit reads through `production_reader`, but not the
scheduler evidence reader. Its deployment identity is exactly
`disabler + production_reader`. It may not have enabler, dispatcher, Phase 5,
scheduler evidence, isolated-test, recovery, cancellation, completion-writer,
generic `pqxx`, or sealed-owner reachability.

### Acquisition and common Phase E handoff

The production dispatcher invokes only the exact acquisition function. The
common C++ engine switches with `SET LOCAL ROLE` to
`campaign_operations_production_phase5_transactional` for the normal Phase E
transaction and handoff. It never switches to the sealed owner or an
intermediary capability role.

The Manager/application LOGIN must have exactly:

`dispatcher + production_phase5_transactional + production_reader + scheduler_protocol_evidence_reader`.

The readiness adapter is a narrow H2 correction required by the frozen role
model. The unchanged H1 readiness view invokes the sealed role-membership
helper, but PostgreSQL evaluates a view with invoker privileges and the H2
application roles must not receive that helper's `EXECUTE` privilege. The
adapter is `SECURITY DEFINER`, owned by the sealed H1 owner, preserves
`session_user` for persisted role evidence, and is the only additional
read-only H2 function grant. It does not grant application access to the
sealed helper or any H1 transition.

Dispatcher is used for acquisition; Phase 5 is used for handoff and fresh
recovery. Reader and scheduler memberships are needed before switching for
readiness and after reconnect. Phase 5 has its own exact lock-helper and
evidence-read ACL because `SET LOCAL ROLE` uses the target role privileges.
This is not privilege escalation. The Manager has no enabler, disabler,
isolated-test, recovery, cancellation, completion-writer, scheduler mutation,
raw scheduler-table, or sealed-owner authority.

## 4. Role-combination matrix

The audit accepts no production capability membership through a NOLOGIN
intermediary and no `ADMIN OPTION` edge. The only mutation LOGIN tuples are:

| LOGIN tuple | Result |
|---|---|
| enabler + reader + scheduler evidence reader | allowed enable identity |
| disabler + reader | allowed disable identity |
| dispatcher + Phase 5 + reader + scheduler evidence reader | allowed Manager/dispatch identity |
| reader alone, scheduler reader alone, or reader + scheduler reader | optional read-only identity |
| any other tuple containing a production mutation role | rejected |

Explicitly rejected: enabler+disabler, enabler+dispatcher,
disabler+dispatcher, dispatcher without the exact Phase 5/reader/scheduler
combination, dispatcher with isolated-test capability, dispatcher with
recovery/cancellation/completion-writer capability, generic `pqxx` or generic
application-role membership, nested/transitive membership, capability-role
membership to another capability role, ADMIN OPTION, and any path to the
sealed owner. Membership is direct from a deployment-created LOGIN to a
NOLOGIN capability role. Disposable tests may create synthetic LOGINs solely
to prove these rules.

## 5. Recovery principal

Enable, disable, acquisition, and handoff recovery reconnect as the same
original LOGIN. A fresh connection is mandatory after `pqxx::in_doubt_error`
or a broken transport. The LOGIN retains the same direct membership tuple; no
new key, role, or principal is invented, and `SET LOCAL ROLE` is repeated
only to the same dispatcher or Phase 5 capability. Enable replay uses reader
and scheduler evidence; disable replay uses reader; acquisition/handoff uses
dispatcher or Phase 5 plus its exact supporting grants. Partial, malformed,
conflicting, or lookup-uncertain evidence fails closed as ambiguous or
reconciliation required.

## 6. Versioned deployment evidence and diagnostics

Migrations 056 and 057 and the H2 manifest are the versioned H2 deployment
authority. The H2
deployment audit is `Scripts/CampaignOperationsH2DeploymentAudit.sh`, version
`h2-deployment-audit-v1`. It verifies migration 055/056/057 filename, checksum,
and ledger; exact H2 function/table/helper ACL tuples; no PUBLIC, `pqxx`, or
grant option; H1 fixed-function owner/security-definer/volatility/search-path/
default/overload shape; exact capability attributes; recursive graph, direct
LOGIN edges, ADMIN OPTION, sealed-owner reachability; allowed LOGIN tuples;
and absence of direct protected-table DML or raw scheduler-table privileges.

The unchanged H1 audit runs on the 055 baseline before H2 extension and
remains the authority for H1 historical ownership, ACL-origin, canonical, and
protected-object evidence. H2 diagnostics are stable: `H2A001` missing or
wrong role attributes, `H2A002` incompatible role/ACL state, `H2A003` extra,
PUBLIC, grant-option, or direct-DML privilege, `H2A004` migration identity,
`H2A005` fixed-function catalog shape, and `H2A006` invalid LOGIN combination.
SQLSTATE is `42501` for privilege/deployment failures and `55000` for catalog
or migration identity failures.

## 7. Install, upgrade, restore, replay, and rollback

Clean install applies 055, 056, then 057; fixed transitions become reachable only
through the exact H2 roles, production remains disabled, and no application
LOGIN membership exists. An accepted 055 upgrade first proves unchanged H1
audit, then applies 056 and 057 transactionally. Partial, extra, PUBLIC, or
grant-option ACL state fails closed and is never silently normalized. Replay
is idempotent only when the exact final ACL is already present. Restore and
role-recreation workflows restore exact NOLOGIN roles, apply/verify 056/057,
recreate allowed LOGIN memberships operationally, and rerun H2 audit. Missing
roles, wrong attributes, unsafe graph, or missing Manager tuple block
enablement. No business rows, scheduler state, H1 evidence, or H1 checksum is
rewritten.

Rollback is disable first, stop the Manager/application caller, then revoke
production LOGIN memberships operationally. Migration 056 ACLs and immutable
history remain intact; capability roles remain NOLOGIN. Revoking function
EXECUTE is an emergency catalog downgrade action, not normal rollback, and no
rollback deletes or rewrites immutable rows.

This amendment stops at H2 exact caller-driven single-request dispatch. It
does not add H3 candidate selection, Manager run-once, generated operation
keys, bounded batch processing, or H4 daemon, polling, autostart,
supervision, worker control, or continuous behavior.
