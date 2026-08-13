---
title: "Campaign Operations Phase H H1 Targeted Architecture Correction and Implementation"
document_type: "implementation and assurance report"
status: "final"
date: "2026-08-01"
branch: "campaign-operations"
base_commit: "fec53d5"
---

# Campaign Operations Phase H H1 Targeted Architecture Correction and Implementation

## 1. Executive result

The narrowly scoped architecture correction was accepted in ADR-0019B before
the implementation was changed. Migration 055 now fails closed on incompatible
role identities, role-graph edges, unexpected ownership, protected alternate
names, and unsafe default ACLs; wildcard ownership transfer is gone; a
versioned read-only audit and genuine dump/role-recreation/restore harness are
implemented. No commit was created. No production process, database, role,
experiment row, or shared DerivedData was changed.

All tests actually run are passing. The result is nevertheless
`NOT_READY_FOR_REVERIFICATION`: §25 records remaining ADR-0019B acceptance
evidence that is still incomplete. In particular, the complete cross-operation
lock matrix, exact tuple-by-tuple ACL audit, rule/dependency negative fixtures,
and the full executable negative-test authenticity matrix have not all been
closed. This report does not claim that H1 is ready to commit.

## 2. Architecture amendment summary

Added accepted ADR-0019B, which freezes the sealed role tuple, empty recursive
graph, minimum owner inventory, all-schema entry-point contract, ACL/default-ACL
rules, stable H1A001--H1A011 diagnostics, staged deployment audit, restore
workflows A--J, lock order, historical bytes, and unchanged H1/H2/H3/H4
boundaries. ADR-0019/0019A, the ADR index, Phase H architecture, Volume XII,
revision histories, the operational H1 document, and Database README were
updated.

## 3. Exact boundary-role attribute contract

`campaign_operations_h1_boundary_authority` is exactly:

`NOLOGIN SUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION
NOBYPASSRLS CONNECTION LIMIT -1`, NULL password, NULL validity, no role or
database settings, no credential, no membership in either direction, no
members, and no ADMIN OPTION edge. A missing role may be created with that
tuple. A pre-existing mismatch raises SQLSTATE `42501`, diagnostic `H1A002`;
migration 055 does not normalize it. Disposable fixtures cover LOGIN,
NOSUPERUSER, NOINHERIT, CREATEDB, CREATEROLE, REPLICATION, BYPASSRLS,
connection limit, password, validity, and role configuration.

## 4. Recursive role-graph contract

The graph involving any of the eight H1 roles must be empty. Therefore every
direct, nested, inherited, SET ROLE/SET LOCAL ROLE, boundary-inherits-outward,
combined capability, and ADMIN OPTION path is rejected. Direct, two-level,
three-level, NOLOGIN-chain, ADMIN OPTION, boundary-inherits, and LOGIN SET ROLE
fixtures reached `42501 H1A003` rather than an unrelated earlier failure.

## 5. Minimal owned-object inventory

The post-migration exact inventory is 11 public relations (nine tables and two
views), eight serial/identity sequences dependency-coupled to those tables, 47
exactly-one public function names/signatures, and 62 exact
schema/relation/trigger triples. Relation ownership is limited to:

- `campaign_operations_production_transition_context`
- `campaign_operations_production_enablement_event`
- `campaign_operations_production_enablement_audit_reference_event`
- `campaign_operations_request_production_admission`
- `campaign_operations_operational_request`
- `campaign_operations_dispatch_attempt`
- `campaign_operations_dispatch_audit_reference_event`
- `campaign_operations_completion_event`
- `campaign_operations_completion_audit_reference_event`
- `campaign_operations_production_readiness_v1`
- `campaign_operations_production_status_v1`

PostgreSQL-coupled row/array types and indexes are accepted only through exact
relation dependencies. No schema, standalone type/domain, operator, cast,
event trigger, large object, publication, subscription, or unrelated relation
may be boundary-owned. Unrelated cancellation, lifecycle, reconciliation,
recovery, recommendation, scheduler-mutation, experiment, and worker functions
remain with their accepted prior owners.

## 6. All-schema ownership/execution allowlists

Migration preflight and the post-deployment audit scan every non-system schema.
They enumerate relations/partitions/sequences/views/foreign tables, functions,
procedures, aggregates and overloads, types/domains, operators, casts,
ordinary/event triggers, publications/subscriptions, large objects, ownership,
ACLs, and function dependencies. Protected names must occur exactly once in
`public`; default-argument variants, overloads, alternate schemas, casts, and
extra boundary trigger entries fail with H1A004/H1A005.

Remaining limitation: an exhaustive fixture and audit branch for every rule
and every dependency form is not yet present; see §25.

## 7. Wildcard transfer removal

The former `%campaign_operations%` and broad SECURITY DEFINER/trigger discovery
ownership loops were removed. Ownership changes now use literal
`ALTER TABLE`, `ALTER VIEW`, `ALTER SEQUENCE`, and exact `regprocedure`
signatures. Missing expected signatures abort; extra objects are never
transferred automatically.

## 8. Automated audit design and exact commands

The authoritative surfaces are
`campaign_operations_h1_deployment_audit_v1(text, boolean, boolean)` and
`Scripts/CampaignOperationsH1DeploymentAudit.sh`. The SQL function is
SECURITY INVOKER and is invoked inside a READ ONLY transaction by the command.
The shell command recomputes migration 055 SHA-256, uses `ON_ERROR_STOP`, and
returns nonzero with H1A001--H1A011 diagnostics.

```text
Scripts/CampaignOperationsH1DeploymentAudit.sh --stage pre-upgrade --host HOST --port PORT --user ADMIN --database DB
Scripts/CampaignOperationsH1DeploymentAudit.sh --stage post-upgrade --host HOST --port PORT --user ADMIN --database DB
Scripts/CampaignOperationsH1DeploymentAudit.sh --stage pre-restore --host HOST --port PORT --user ADMIN --database DB
Scripts/CampaignOperationsH1DeploymentAudit.sh --stage post-role-recreation --host HOST --port PORT --user ADMIN --database DB
Scripts/CampaignOperationsH1DeploymentAudit.sh --stage post-database-restore --host HOST --port PORT --user ADMIN --database DB
Scripts/CampaignOperationsH1DeploymentAudit.sh --stage pre-enablement --host HOST --port PORT --user ADMIN --database DB
```

## 9. Upgrade results

Clean installation, genuine 054-to-055 upgrade, migration replay, every
boundary attribute fixture, graph fixtures, hostile ownership, alternate
schema, and default-ACL fixtures ran in newly initialized temporary PostgreSQL
clusters. Migration rollback and stable SQLSTATE/diagnostic checks passed.
Post-upgrade audit returned
`H1_DEPLOYMENT_AUDIT_V1_OK stage=post-upgrade`.

## 10. Restore support matrix/results

| Scenario | Result in disposable restore target |
|---|---|
| A database-only dump plus separately recreated roles | pass |
| B roles-only dump plus database restore | pass |
| C safe exact pre-existing roles | pass |
| D incompatible pre-existing roles | rejected, H1A002 |
| E ADMIN OPTION/graph edge | rejected, H1A003; SET ROLE and SET LOCAL ROLE demonstrated |
| F altered role attribute | rejected, H1A002 |
| G unexpected boundary-owned object | post-restore rejected, H1A004 |
| H differing default ACL | post-restore rejected, H1A007 |
| I PUBLIC EXECUTE on protected signature | post-restore rejected, H1A006 |
| J alternate-schema protected overload | post-restore rejected, H1A005 |

The harness uses a real custom-format database dump, a roles-only dump, a
second initialized PostgreSQL cluster, role recreation, schema/data restore,
and post-restore audit. The synthetic pre-055 fixture requires
`--disable-triggers` only because its deliberately replica-loaded historical
rows are FK-invalid test history; that exception is local to the disposable
fixture and documented in the harness.

## 11. Default-ACL matrix/future-object evidence

Global and public-schema default privileges are set for the boundary owner,
`campaign_operations_owner`, and scheduler-evidence owner for functions,
tables, sequences, schemas, and types. The test creates a future function,
procedure, table, sequence, enum type, domain, and schema as each owner, expands
NULL ACLs with `acldefault`, and proves no PUBLIC privilege. Unsafe restored
defaults fail H1A007.

Remaining limitation: the audit currently rejects non-owner default grantees
and requires the global function rows, but does not compare every expected
global/schema-specific pg_default_acl row as a complete tuple matrix; see §25.

## 12. Membership, SET ROLE, and ADMIN OPTION evidence

The focused suite verified direct, two-hop, three-hop, NOLOGIN intermediary,
combined-role, boundary-inherits-other, and ADMIN OPTION graphs. An independent
LOGIN connection demonstrated both SET ROLE and SET LOCAL ROLE reachability in
the hostile fixture before the audit rejected the graph. Exact observed branch:
SQLSTATE 42501, H1A003.

## 13. Wrapper/overload rejection evidence

Alternate-schema same-name functions, overload multiplicity, protected source
references, boundary-owned wrappers, casts, trigger entries, and catalog
dependencies are rejected by preflight/audit branches. Scenario J and the
hostile alternate-schema upgrade fixture passed their exact H1A005 assertions.

Remaining limitation: the negative suite does not yet instantiate every
required object form (rule, operator, cast, procedure, event trigger, ordinary
trigger, dynamic wrapper) and prove its exact failure point; see §25.

## 14. Complete lock-path evidence

The implemented deterministic probe now verifies accepted acquisition order
0a scheduler evidence, 0b production-enable advisory domain, 1 authorization,
2 budget, 3 campaign/completion, 4 reservation, and 5 request. Every boundary
uses two independent connections, unique `application_name`,
`pg_blocking_pids()`, granted `pg_locks`, expected prior locks, absence of later
locks, and expected advisory-gate state. The focused suite passed all seven
boundaries.

This is not yet the complete requested cross-operation matrix (enable,
disable, completion, cancellation, reconciliation/recovery, revocation,
budget mutation, expiry, same/different request/campaign, uncommitted replay,
rollback, and reverse-wait absence). That missing matrix is a reverification
blocker in §25.

## 15. Negative-test authenticity matrix

| Requirement group | Fixture/failure proof | Result |
|---|---|---|
| 11 role attributes | one pre-055 role fixture each; exact migration statement | 42501 H1A002 |
| role graph/reachability | direct, 2/3 level, NOLOGIN, ADMIN, outward edge, SET ROLE | 42501 H1A003 |
| hostile ownership | boundary-owned table and restored extra object | 42501 H1A004 |
| alternate name/wrapper | second schema protected name/restore clone | 42501 H1A005 |
| PUBLIC execution | restored clone grant | 42501 H1A006 |
| unsafe defaults | upgrade and restored default ACL | 42501 H1A007 |
| fixed transitions/context/replay/hydration/completion | SQL suite checks named constraints/diagnostics and preceding fixture state | pass |
| historical bytes | required audit with exact bytea/hash comparisons | exact path passes; corrupt H1A010 fixture still missing |
| lock acquisition 0a--5 | catalog-confirmed blockers, not timing alone | pass |

The matrix is not complete for every requested object class and every
same-hash/different-canonical level. Missing rows remain a blocker; no PASS is
claimed for them.

## 16. Post-restore historical-byte proof

Attempt V1 ID 7001 and Completion V1 ID 7001 are created under schema 054.
Their canonical UTF-8 and hash bytes, including the nested Attempt V1 bytes in
Completion V1 request evidence, are captured before 055. Equality passed after
055, after custom dump, exact role recreation, restore into a second cluster,
and `campaign_operations_h1_deployment_audit_v1(..., true)`. No row was
rewritten or reinterpreted.

## 17. Fixed-transition and SECURITY DEFINER reverification

The three fixed transitions remain exact, owner-only, VOLATILE, PARALLEL
UNSAFE, non-leakproof, non-variadic, with no defaults/overloads and pinned
`search_path=pg_catalog, public`. PUBLIC and application execution are revoked.
Enable, disable, and acquisition replay; the exact 61-byte acquisition name;
Attempt V2 nested validation; independent hydration; and Phase F recovery
alignment all passed.

## 18. Migration checksum before/after

- Focused-review input: `5ce4d776b6d5be809297b44947f78b300543acab4d5bcf2f2ca1766c5de16d05`
- Current migration 055: `e2034ad30fe4f1487b577d9596610589493a75c5de72519e89371ac4ff2d2c72`

The current value was independently recomputed with `shasum -a 256`, matches
`kProductionAdmissionMigrationChecksum`, and passed the disposable migration
ledger and post-restore audit.

## 19. Strict compile and isolated Release build results

Passed with `-std=c++20 -Wall -Wextra -Werror`:

- `CampaignOperationsPhaseH1Tests.cpp`, then executed successfully;
- `CampaignOperationsTests.cpp`, then executed successfully;
- `CampaignOperationsPhaseH1RepositoryTests.cpp` source/link set;
- complete Phase 1--5 `CampaignOperationsRepositoryTests.cpp` source/link set,
  then executed successfully in a separate disposable cluster.

Isolated Release build:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath /tmp/ea-h1-reverify-deriveddata.W4fCSa build
```

Result: `** BUILD SUCCEEDED **`. The full project emitted existing warnings in
unchanged LSTM/libpqxx call sites (including deprecated `exec_params`); the
dedicated strict H1 and Campaign Operations compilations emitted none. Under
the repository rule that warnings are defects, the project-wide warning debt
is explicitly unclosed rather than reported as warning-free.

## 20. H1 inertness and H2/H3/H4 exclusion

Migration 055 remains default-off: no enable event, admission, Attempt V2,
production witness, LOGIN grant, scheduler/lifecycle/worker mutation, or
experiment mutation is created. The H1 CLI exposes read-only readiness/status
only. Parser/help tests prove no enable/disable command or C++ call to the fixed
mutation transitions. H2, H3, and H4 behavior was not implemented.

## 21. Skipped-suite classifications

- Live scheduler ownership/process integration suites: skipped to avoid active
  scheduler and seven training workers; **pre-production blocker**, not safe to
  run against the active environment.
- Audit against any production database/roles: skipped by authorization;
  **pre-enablement blocker**.
- Full cross-operation H1 lock matrix: not implemented/run; **focused
  reverification blocker**.
- Exhaustive negative authenticity rows for every catalog class and all five
  canonical collision levels: not implemented/run; **focused reverification
  blocker**.
- H2/H3/H4 behavioral suites: outside H1; **H2/H3/H4 prerequisite**, not H1
  evidence.
- Scheduler canonical-path/ownership policy test: safely run and passed.

## 22. Complete file list

Tracked working-tree files:

- `Database/README.md`
- `Database/migrations/055_campaign_operations_production_admission_foundation.sql`
- `ExpertAdvisor.xcodeproj/project.pbxproj`
- `Sources/CampaignOperations.hpp`
- `Sources/CampaignOperationsProductionAdmission.cpp`
- `Sources/CampaignOperationsProductionAdmission.hpp`
- `Sources/CampaignOperationsProductionAdmissionRepository.cpp`
- `Sources/CampaignOperationsProductionAdmissionRepository.hpp`
- `Sources/CampaignOperationsProductionAdmissionService.cpp`
- `Sources/CampaignOperationsProductionAdmissionService.hpp`
- `Sources/ExperimentScheduler.cpp`
- `Tests/CampaignOperationsPhaseH1CliTests.sh`
- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- `Tests/CampaignOperationsPhaseH1MigrationTests.sql`
- `Tests/CampaignOperationsPhaseH1RepositoryTests.cpp`
- `Tests/CampaignOperationsPhaseH1Tests.cpp`
- `Tests/CampaignOperationsRepositoryTests.cpp`
- `docs/CampaignOperationsPhaseH1.rst`
- `docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md`
- `docs/architecture/Volume_XII_Database.md`
- `docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md`
- `docs/architecture/adr/README.md`

Untracked files present:

- `Scripts/CampaignOperationsH1DeploymentAudit.sh`
- `docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md`
- `docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md`
- six H1 implementation/review report artifacts, including this report

The six pre-existing report artifacts were treated as review evidence/claims;
only this report was rewritten in this pass.

## 23. Commands and results

Passed:

- `bash Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- strict H1 unit compile and execution
- strict broad Campaign Operations unit compile and execution
- strict H1 repository compile; its runtime path is included in the focused suite
- strict Phase 1--5 repository/service/completion compile and disposable-cluster execution
- H1, Phase 2, Phase 4, and Phase 5 CLI scripts against the isolated binary
- `bash Tests/SchedulerCanonicalPathTests.sh`
- isolated Release `xcodebuild`
- `bash -n` on both H1 shell surfaces
- `git diff --check`
- `git diff --cached --check`

## 24. Final git status and diff statistics

Branch/base: `campaign-operations` at `fec53d5`.

Tracked diff: `22 files changed, 9037 insertions(+), 16 deletions(-)`.
Untracked line counts: reports 101/292/this file/597/610/374 lines; deployment
audit 217 lines; ADR-0019A 337 lines; ADR-0019B 352 lines. The exact final
`git status --short` is included in the handoff response and must be regenerated
after this report write.

## 25. Explicit disposition

`NOT_READY_FOR_REVERIFICATION`

Required before changing the disposition:

1. Freeze and compare the complete current explicit ACL tuple allowlist,
   including every accepted prior-phase helper grant and all column privileges.
2. Compare the complete global and schema-specific default-ACL row matrix, not
   only unsafe grantees plus future-object behavior.
3. Add exact rule and remaining operator/cast/procedure/event-trigger/dynamic
   wrapper/dependency fixtures with authentic failure-point evidence.
4. Complete the cross-operation lock matrix and reverse-wait proof required by
   ADR-0019B §11.
5. Complete the executable negative-test traceability matrix for every listed
   context, recursive-trigger, collision, corruption, lifecycle, and lock case.
6. Resolve or explicitly accept through architecture the warning-bearing full
   Release build; the strict H1 sources themselves are clean.

No claim is made that H1 is ready to commit.
