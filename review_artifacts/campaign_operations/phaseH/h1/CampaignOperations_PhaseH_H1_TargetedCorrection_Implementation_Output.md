---
title: "Campaign Operations Phase H H1 Targeted Architecture Correction and Implementation"
document_type: "implementation and reverification handoff"
status: "final"
date: "2026-08-01"
---

# Campaign Operations Phase H H1 Targeted Correction Implementation Report

## 1. Executive result

The two architecture blockers identified by the focused independent CEE were
corrected before migration 055 and the H1 implementation were amended.

- ADR-0019A is the accepted-style, narrowly scoped amendment.
- PostgreSQL acquisition uses an exact representable 61-byte identifier.
- The forgeable ordinary-owner context model is replaced by a sealed boundary
  authority whose superuser/physical-administrator power is explicitly outside
  the supported threat model.
- Exact operation-key replay precedes mutable-head/CAS validation for enable,
  disable, and acquisition.
- Acquisition observes the complete frozen lock order.
- Migration, role graph, default ACL, SECURITY DEFINER, hydration, historical
  fixture, negative, replay, rollback, and lock evidence passes in disposable
  PostgreSQL clusters.
- H1 remains default-off and exposes no H2, H3, or H4 mutation route.
- No commit was created.
- The live scheduler, seven live training workers, live database, production
  roles, experiment state, and shared DerivedData were observed but not changed.

## 2. Architecture decisions

The new authority order is ADR-0019, then ADR-0019A for the two corrected H1
points, then the remaining committed Phase H architecture. ADR-0019A amends
only the acquisition identifier, transaction authority, associated role/object
inventory, replay ordering, lock ordering, default ACLs, and verification
requirements. All other ADR-0019 H1-H4, lifecycle, scheduler-isolation,
identity, default-off, and Completion V1 boundaries remain unchanged.

The normative Phase H document is revision 1.1.0 and Volume XII is revision
0.13.0. ADR-0019 and the ADR index link the amendment and record the revision.

## 3. Exact acquisition identifier

Impossible former name:

`transition_campaign_operations_request_dispatching_production_v2`

Authoritative replacement:

`transition_campaign_operations_request_dispatch_production_v2`

The replacement is exactly 61 UTF-8 bytes. Migration and tests compare
`pg_proc.proname::text` exactly, require one exact row, reject the 63-byte
truncated former catalog spelling, and fail if migration 055 emits a truncation
notice. `regprocedure` is used only after the exact catalog-name proof and is
not treated as proof against truncation.

## 4. Corrected role and ownership model

| Role/class | Login | Membership | Boundary ownership/privilege | May grant | Fixed transition execution | Context/evidence write |
|---|---:|---|---|---:|---:|---:|
| `campaign_operations_h1_boundary_authority` | No | none in either direction | Owns protected H1/Phase E evidence, helpers, triggers, fixed transitions, views, and sequences | superuser is outside supported threat model | owner only; never reachable | owner only; never reachable |
| production enabler | No | none | no object ownership; frozen capability only | No | No in H1 | No |
| production disabler | No | none | no object ownership; frozen capability only | No | No in H1 | No |
| production dispatcher | No | none | no object ownership; frozen capability only | No | No in H1 | No |
| production Phase 5 transactional | No | none | no object ownership; frozen capability only | No | No in H1 | No |
| production reader | No | none | read-only readiness/status views | No | No | No |
| scheduler protocol evidence owner | No | none | compatibility name only; no raw protocol-table or helper access | No | No | No |
| scheduler protocol evidence reader | No | none | executes the exact snapshot helper | No | No | No |
| `campaign_operations_owner`, `pqxx`, generic LOGIN, or any inherited/reachable role | as pre-existing | cannot reach any role above | no protected ownership/write/execute | No boundary grants | No | No |

Migration 055 fails with SQLSTATE 42501 on prohibited pre-existing membership
or an unsafe default ACL. It does not silently preserve or normalize unsafe
reachability. It creates no LOGIN role and grants no membership.

## 5. Why ordinary authority cannot forge transaction evidence

The context relation remains, but its owner is the sealed, unreachable
`campaign_operations_h1_boundary_authority`. A row binds backend PID,
top-level transaction ID, exact transition kind, request identity where
applicable, and operation key. Only a fixed SECURITY DEFINER transition can
insert the matching row. Immediate triggers validate it and the transition
deletes it before return. A deferred constraint trigger rejects any context row
left at transaction end.

Consequently:

- table, evidence, helper, trigger, function, view, sequence, and context owners
  are the same unreachable sealed authority;
- ordinary former owners cannot bypass ACLs through implicit owner rights;
- direct or inherited membership, `SET ROLE`, combined isolated/production
  membership, nested SECURITY DEFINER calls, caller GUCs, trigger depth, and
  recursive triggers cannot create equivalent evidence;
- rollback, a rolled-back savepoint/subtransaction, or a caught exception
  removes the transaction-local row and every dependent write atomically;
- pooled-session reuse sees an empty context, and concurrent sessions have
  different PID/xid keys;
- false-to-true requires the complete fixed acquisition; true-to-false remains
  rejected; incomplete Boolean/admission/Attempt V2/audit combinations fail
  deferred consistency validation.

The database superuser, physical database administrator, and code execution as
that administrator remain explicitly outside the supported threat model. No
ordinary object owner or reachable role has equivalent power.

## 6. Exact replay implementation

Each fixed transition validates the operation key and performs an exact
operation-key lookup before mutable head and CAS validation, then repeats the
lookup after acquiring its locks.

| Transition | Exact replay | Conflict/corruption |
|---|---|---|
| enable | returns the original complete enablement and audit evidence even after head advancement | changed logical input is 23505; missing, partial, contradictory, canonical/hash, mirror, or audit evidence is 23514 |
| disable | returns the original complete disablement and audit evidence even after head advancement | same rules and diagnostics as enable |
| acquisition | returns the original admission, Attempt V2, request witness, and acquisition audit identity without consulting mutable request state as commit proof | changed logical input is 23505; incomplete or contradictory nested evidence is 23514 |

The acquisition replay helper reads the immutable request identity only; it
never infers commit from current request state/version/head. Tests cover exact
replay, changed-field replay, and same-hash/different-canonical evidence for all
three transition families and their required nested evidence.

## 7. Lock order and deterministic evidence

The implemented order is:

0a. scheduler protocol evidence;
0b. shared production-enable advisory domain;
1. authorization head;
2. budget head;
3. campaign/completion boundary;
4. reservations in ascending identity order;
5. requests in ascending identity order;
6+. existing Phase 5/lifecycle domains.

Acquisition makes only an optimistic request read before locks; it does not
lock the request before campaign/completion. It uses the existing Phase E lock
helpers and revalidates immutable request identifiers under the authoritative
locks. Three independent-connection probes block at campaign, reservation, and
request respectively. Each uses `pg_blocking_pids()` and `pg_locks` to prove
all earlier row locks and the shared production-gate advisory lock are held and
that later row locks are not yet held. No timing-only assertion is used.

## 8. Default ACL and SECURITY DEFINER hardening

Migration 055 revokes global default PUBLIC function execution (not merely a
schema-local default) for every involved owner and revokes default PUBLIC table
and sequence privileges in `public`. Tests interpret NULL ACLs with
`acldefault`, create future function/table/sequence probes as the sealed owner,
and prove no PUBLIC privilege. A pre-existing grant to `pqxx` is an exact
upgrade-failure test.

Every Campaign Operations SECURITY DEFINER function involved in the guarded
relations, scheduler snapshot, lifecycle cancellation, enable/disable,
admission, Attempt V2, witness, audit, completion, role graph, readiness, or
status is owned by the sealed authority, has `search_path=pg_catalog, public`,
and has PUBLIC execution revoked. All non-internal trigger functions on the
protected relation set receive and catalog-prove the same owner/search-path/ACL
policy. Fixed signatures have no permissive overload or wrapper. Bodies use
schema-qualified objects and no caller-controlled dynamic SQL.

## 9. Enablement hydration correction

Repository head selection now reads the stored `capability` and
`enablement_contract_version` columns independently. Mapping validates each
stored value before reconstructing the domain object. Per-column corruption
tests prove each mismatch fails before repository or service return.

## 10. Authentic negative and branch traceability matrix

| Requirement/branch | Exact observed failure |
|---|---|
| old table owner false-to-true/context forge | exact UPDATE/INSERT; 42501 permission-denied diagnostic names the target table |
| direct enable/admission/Attempt V2 | exact INSERT; 42501 and fixed-transition-context diagnostic |
| true-to-false and recursive/nested trigger spoof | exact UPDATE/trigger; 55000 `production admission witness is immutable` |
| protected evidence UPDATE/DELETE/TRUNCATE | exact statement; 55000 and evidence-specific immutable message |
| enable/disable/acquisition conflicting replay | exact fixed function; 23505 and transition-specific `conflicting replay` message |
| enable/admission/Attempt V2 same-hash changed canonical | exact fixed replay; 23514 and replay-evidence-corrupt message, not an earlier 42501 |
| incomplete Attempt V2 field/mixed V1-V2 | exact UPDATE; 23514 and `campaign_operations_dispatch_attempt_v1_v2_shape` constraint |
| post-completion Boolean/child evidence | exact UPDATE/INSERT; 55000 or 23514 and completion-specific diagnostic |
| unsafe membership/default ACL upgrade | exact migration; 42501 and migration-specific diagnostic; transaction fully rolled back |
| context persistence | exact INSERT/commit; 23514 and `production transition context must be empty at transaction end` |
| caught exception/savepoint/session reuse | forced inner exception and `ROLLBACK TO SAVEPOINT`; context count is zero before and after commit/reuse |
| role graph and `SET ROLE` | direct, inherited, nested, combined-role matrices verify exact role helper and deny boundary writes/execution |
| recovery status | positive case plus each binding, downstream execution, outcome, lease/state, stale-outcome, and transaction-race branch agrees with Phase F |
| lock order | independent blockers with catalog-confirmed wait edges and held/not-held lock sets |
| hydration corruption | each stored typed mirror/canonical/hash plus capability and contract version fails before service return |
| H2/H3/H4 reachability | CLI help/source scan rejects transition references and mutation commands |

## 11. Genuine pre-055 historical fixture proof

The disposable harness applies migration 054, then creates one authoritative
Attempt V1 and one Completion V1 plus all nested campaign, authorization,
budget, reservation, request, dispatch, and audit evidence. The Completion V1
canonical is checked by migration 054's validator and contains the Attempt V1
canonical. Exact UTF-8 canonical bytes and hashes are copied to the pre-055
snapshot before migration 055 runs.

After 055 and again after supported replay, bytea equality proves Attempt V1,
Completion V1 request evidence, Completion V1 canonical, and every hash are
unchanged. Completion V1 still validates. Separate request-evidence and
Completion V1 same-hash/different-canonical probes fail. Migration 055 performs
no V1 backfill or reinterpretation.

## 12. Files changed by the H1 implementation/correction

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
- `docs/CampaignOperationsPhaseH1.rst`
- `docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md`
- `docs/architecture/Volume_XII_Database.md`
- `docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md`
- `docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md`
- `docs/architecture/adr/README.md`
- `CampaignOperations_PhaseH_H1_TargetedCorrection_Implementation_Output.md`

Three other untracked H1 review/activity reports pre-existed this correction
and were not edited.

## 13. Migration checksum and replay evidence

- Focused-review input checksum: `9b272b286fe3244c9f74a79e87b4a4b66d9255b98eb11adea155910260c96b3c`
- Corrected checksum: `5ce4d776b6d5be809297b44947f78b300543acab4d5bcf2f2ca1766c5de16d05`

The corrected SHA-256 equals the embedded C++ constant and the disposable
migration-ledger value. Clean supported install, 054-to-055 upgrade,
checksum-stable replay, transaction rollback on incompatible schema, unsafe
role/default-ACL failure rollback, no backfill, no LOGIN grant, and no 055
identifier-truncation notice all pass. Older migrations 053/054 still emit
their pre-existing long-constraint/trigger notices; no such notice comes from
055.

## 14. Commands and results

Passed:

- `bash Tests/CampaignOperationsPhaseH1MigrationTests.sh` — migration/catalog,
  clean install, 054 upgrade, replay, negative, historical, deterministic lock,
  and H1 repository/service tests.
- Strict `clang++ -std=c++20 -Wall -Wextra -Werror` build and execution of
  `CampaignOperationsPhaseH1Tests.cpp`.
- Strict `clang++ -std=c++20 -Wall -Wextra -Werror` build and execution of
  `CampaignOperationsTests.cpp`.
- Strict `clang++ -std=c++20 -Wall -Wextra -Werror` build and isolated-cluster
  execution of `CampaignOperationsRepositoryTests.cpp`; this runs Phase 1-5
  migration SQL, repository/service/completion behavior, and concurrency races.
- `bash Tests/CampaignOperationsPhaseH1CliTests.sh <isolated binary>`.
- Phase 2, Phase 4, and Phase 5 CLI scripts against the isolated binary.
- `bash Tests/SchedulerCanonicalPathTests.sh` (includes scheduler ownership
  policy execution through direct, symlink, and basename paths).
- `git diff --check` and `git diff --cached --check`.

All PostgreSQL mutations occurred in temporary clusters/databases. Temporary
clusters were stopped and removed after execution.

## 15. Strict compilation and Release build

All H1 source/tests and the complete Campaign Operations
repository/service/completion regression source set compile cleanly with
`-Wall -Wextra -Werror`.

The isolated Release build command was:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath /tmp/ExpertAdvisor-H1-20260801-DD build
```

Result: `** BUILD SUCCEEDED **`.

The Xcode project as a whole emitted 660 warning lines: overwhelmingly existing
libpqxx `exec_params` deprecations, plus existing main/MetaNN/toolchain
diagnostics. None came from the new H1 source files. Eliminating repository-wide
legacy warnings is outside this narrowly scoped Phase H architecture correction;
the focused source sets are warning-free under `-Werror`.

## 16. H1 inertness and later-phase exclusion

Migration 055 creates no production evidence, never flips the request Boolean,
and grants no fixed transition to an H1 caller. C++ exposes readiness/status
only. CLI help and source scanning prove there is no enable, disable,
production-acquire, manager, canary, polling, claim/capacity, handoff, launch,
lifecycle, control, or signal route. H2 service/grant work, H3 Campaign Manager,
and H4 operational evidence remain future increments.

## 17. Skipped suites

The following process-interfering suites were not run because the production
scheduler and seven workers were active:

- `SchedulerOwnershipIntegrationTests.sh` — pre-enablement blocker.
- `SchedulerContinuationOwnershipIntegrationTests.sh` — pre-enablement blocker.
- `SchedulerOwnershipProcessIntegrationTests.sh` — pre-enablement blocker.
- global-control process integration suites — pre-enablement blocker.

They are not H1 commit blockers because H1 is default-off and adds no scheduler
or process-control execution path. H2 fixed-transition caller/canary tests are
H2 prerequisites; Campaign Manager/run-once tests are H3 prerequisites; H4
rollout/process observation remains pre-enablement operational evidence.

## 18. Final worktree evidence

`git status --short`:

```text
 M Database/README.md
 A Database/migrations/055_campaign_operations_production_admission_foundation.sql
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/CampaignOperations.hpp
 A Sources/CampaignOperationsProductionAdmission.cpp
 A Sources/CampaignOperationsProductionAdmission.hpp
 A Sources/CampaignOperationsProductionAdmissionRepository.cpp
 A Sources/CampaignOperationsProductionAdmissionRepository.hpp
 A Sources/CampaignOperationsProductionAdmissionService.cpp
 A Sources/CampaignOperationsProductionAdmissionService.hpp
 M Sources/ExperimentScheduler.cpp
 A Tests/CampaignOperationsPhaseH1CliTests.sh
 A Tests/CampaignOperationsPhaseH1MigrationTests.sh
 A Tests/CampaignOperationsPhaseH1MigrationTests.sql
 A Tests/CampaignOperationsPhaseH1RepositoryTests.cpp
 A Tests/CampaignOperationsPhaseH1Tests.cpp
 A docs/CampaignOperationsPhaseH1.rst
 M docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md
 M docs/architecture/Volume_XII_Database.md
 M docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md
 M docs/architecture/adr/README.md
?? CampaignOperations_PhaseH_H1_Implementation_Output.md
?? CampaignOperations_PhaseH_H1_IndependentCEEVerification_Output.md
?? CampaignOperations_PhaseH_H1_TargetedCorrection_FocusedIndependentCEEVerification_Output.md
?? CampaignOperations_PhaseH_H1_TargetedCorrection_Implementation_Output.md
?? docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md
```

Tracked diff stat before counting untracked artifacts:

```text
21 files changed, 7523 insertions(+), 16 deletions(-)
```

Untracked inventory at report generation:

```text
CampaignOperations_PhaseH_H1_Implementation_Output.md                         101 lines
CampaignOperations_PhaseH_H1_IndependentCEEVerification_Output.md            292 lines
CampaignOperations_PhaseH_H1_TargetedCorrection_FocusedIndependentCEEVerification_Output.md 610 lines
CampaignOperations_PhaseH_H1_TargetedCorrection_Implementation_Output.md     374 lines
docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md   329 lines
```

The five untracked files total 1,706 lines; the complete worktree therefore
contains the 21-file tracked diff above plus these five untracked artifacts.

## 19. Disposition

**READY_FOR_FOCUSED_INDEPENDENT_REVERIFICATION**

This is not a claim that H1 is ready to commit. Commit readiness belongs to the
next independent CEE verification.
