---
title: "Campaign Operations Phase H H1 Targeted Architecture Correction Focused Independent CEE Verification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_TargetedArchitectureCorrection_FocusedIndependentCEEVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H H1 Targeted Architecture Correction Focused Independent CEE Verification

# Campaign Operations Phase H H1 Focused Independent Verification

## 1. Executive verdict

**NOT VERIFIED**

The primary security claim is false for hostile upgrade and restore scenarios:

- Migration 055 can succeed while preserving prohibited boundary-role attributes.
- It can succeed with unexpected objects owned by the boundary role.
- It does not detect a pre-existing, cross-schema, PUBLIC-executable `SECURITY DEFINER` wrapper owned by the boundary role.
- An ordinary LOGIN successfully invoked such a wrapper and executed as `campaign_operations_h1_boundary_authority`.
- A database restore into a cluster with unsafe pre-existing membership retained the valid migration checksum while allowing the LOGIN to `SET ROLE` to the boundary role.

No production state was accessed or modified.

## 2. ADR-0019A authority and scope

**Verdict: narrowly scoped in intent, but its accepted-status evidence is not supportable by the implementation.**

ADR-0019A is currently an untracked working-tree document, not part of HEAD `fec53d5`. Per the supplied precedence, I nevertheless treated it as normative authority.

Semantic classification:

| Change | Classification |
|---|---|
| 64-byte acquisition-name correction | Authorized narrow amendment |
| Sealed owner and exact transaction context | Authorized narrow amendment |
| Exact replay for three transitions | Authorized narrow amendment |
| Full H1 lock-order requirement | Authorized narrow amendment |
| Default-ACL hardening | Authorized narrow amendment |
| Index/revision/status synchronization | Clarification |
| No H2/H3/H4 callers or mutation routes | Clarification |
| Wildcard transfer of 73 `SECURITY DEFINER` functions, including lifecycle/cancellation functions | Unauthorized architecture expansion |
| Acceptance claims unsupported by hostile-upgrade/restore tests | Blocker |

The stated threat model is coherent about excluding physical cluster administrators, but the implementation violates its own fail-closed requirements at [ADR-0019A](/Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md:59).

## 3. Exact acquisition-name verdict

**VERIFIED**

The authoritative catalog name was exactly:

`transition_campaign_operations_request_dispatch_production_v2`

Independent catalog evidence:

- UTF-8 byte length: **61**
- `pg_proc.proname::text`: exact match
- Exact function count: 1
- No overloads or default arguments
- No truncation notice
- Former 64-byte spelling absent
- Its silently truncated 63-byte catalog spelling absent
- Product migration, C++, tests, architecture, and Volume XII consistently use the 61-byte name

The migration definition is at [migration 055](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:2340).

## 4. Boundary-role clean-install reachability

**Conditionally verified for the tested clean graph; not sufficient for the overall verdict.**

Observed fresh-cluster catalog identity:

| Property | Value |
|---|---|
| OID | 19768, specific to the disposable cluster |
| LOGIN | false |
| SUPERUSER | true |
| CREATEDB | false |
| CREATEROLE | false |
| REPLICATION | false |
| BYPASSRLS | false |
| INHERIT | true |
| Connection limit | -1 |
| Password | null |
| Valid until | null |
| Role configuration | empty |
| Direct/inherited memberships | none |
| Members | none |
| ADMIN OPTION grantees | none |

A generic LOGIN received SQLSTATE `42501` for:

- `SET ROLE`
- `SET LOCAL ROLE`
- `SET SESSION AUTHORIZATION`
- execution of the fixed disable transition
- context-table insertion
- granting the boundary role
- altering the boundary role
- altering its default privileges
- replacing a protected function in `public`

This result depends entirely on the clean empty role graph. It does not survive the tested hostile restore.

## 5. Boundary-role hostile-upgrade verdict

**FAILED — blocker**

Migration 055 successfully accepted this hostile pre-existing role:

- LOGIN
- NOSUPERUSER
- CREATEDB
- CREATEROLE
- REPLICATION
- BYPASSRLS
- NOINHERIT
- connection limit 7
- password present
- validity through 2035
- role `search_path` configuration

After migration, it silently preserved:

- `BYPASSRLS`
- `NOINHERIT`
- connection limit 7
- password verifier
- validity
- role configuration

It normalized only the attributes explicitly present in [lines 66–73](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:66). The final audit at [lines 3045–3093](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3045) also omits those properties.

Migration 055 also succeeded when the boundary role owned an unexpected table.

## 6. Nested membership and SET ROLE verdict

**Partial pass, overall failed due post-install/restore reachability.**

Verified:

- Direct unsafe pre-migration membership fails with SQLSTATE `42501`.
- A two-level nested LOGIN membership graph fails with SQLSTATE `42501`.
- ADMIN OPTION on an edge is detected because the membership row itself is rejected.
- Clean generic LOGIN `SET ROLE` and `SET LOCAL ROLE` fail with `42501`.

However, after installation or restore, a cluster administrator can create an unsafe membership without any database-level audit preventing use. In the disposable restore fixture, the LOGIN immediately obtained the boundary role and superuser authority.

The checked-in test only covers direct membership in `campaign_operations_production_reader`, not direct/nested membership in the boundary role itself: [CampaignOperationsPhaseH1MigrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:306).

## 7. ADMIN OPTION and grant-authority verdict

**FAILED for supported restore/deployment safety.**

Clean installation had no ADMIN OPTION edge. A disposable post-install grant with ADMIN OPTION made the boundary immediately reachable:

- `SET ROLE campaign_operations_h1_boundary_authority` succeeded.
- `current_user` became the boundary role.
- `is_superuser` became true.
- The LOGIN created an object as boundary authority.

Migration checksum and database restore integrity do not detect this cluster-global change.

## 8. Minimal privilege and owned-object inventory

Observed clean-install ownership:

| Object class | Count | Classification |
|---|---:|---|
| Ordinary tables | 9 | Mixed: H1 evidence/context plus pre-existing operational/completion tables |
| Sequences | 8 | Associated with transferred tables |
| Views | 2 | H1 readiness/status |
| Functions | 73 | Excessive and dangerous |
| Procedures | 0 | None |
| Indexes | 34 | Ownership follows tables |
| Composite/array types | 22 | Ownership follows tables |
| Schemas | 0 | None |
| Operators/casts/event triggers/publications/large objects | 0 | None observed |
| Explicit ACLs on non-owned relations/functions | 0 | None observed |

The wildcard ownership loop at [lines 2904–2927](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:2904) transfers every public `SECURITY DEFINER` function matching `%campaign_operations%`, plus `apply_experiment_lifecycle_cancellation`.

That is not a minimum sealed H1 boundary. It includes cancellation, recovery, completion, and prior-phase functions. Because the role is SUPERUSER, reachability exposes the entire cluster, not merely those objects.

## 9. Context and authorization unforgeability

**Not verified.**

Clean-role tests correctly denied context DML at the intended ACL guard. Existing tests also cover former-owner DML, nested-trigger spoofing, exception/savepoint cleanup, and some session reuse.

The hostile wrapper invalidates the broader unforgeability claim:

1. A pre-existing schema was owned by the boundary role.
2. A PUBLIC-executable `SECURITY DEFINER` wrapper in that schema was owned by the boundary role.
3. Migration 055 succeeded.
4. An ordinary LOGIN invoked it.
5. The wrapper returned `campaign_operations_h1_boundary_authority` as `current_user`.
6. It successfully wrote to a boundary-owned table.

Migration 055 audits only qualifying functions in `public`; it does not globally reject alternate-schema wrappers or unexpected boundary-owned objects.

The complete recursive-trigger, caught-exception, retained-context, cross-operation, pooled-session, and concurrent-connection matrix requested was not independently established after this decisive failure.

## 10. Default-ACL verdict

**Partial pass; incomplete acceptance evidence.**

Verified on clean install:

- Owner-only future function defaults for the three audited H1 owners.
- No unsafe PUBLIC table or sequence grants.
- A hostile pre-existing default function ACL granting `pqxx` execution fails migration with SQLSTATE `42501`.

The implementation is at [lines 3010–3033](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3010).

Not fully verified:

- Future types and schemas
- Object creation probes for every involved owner
- Alternate-schema defaults
- Restore scenarios with differing default ACLs

## 11. Backup and restore verdict

**FAILED — blocker**

An owner-preserving database dump and restore retained:

- boundary ownership of protected objects;
- migration ledger row `055`;
- checksum `5ce4d776b6d5be809297b44947f78b300543acab4d5bcf2f2ca1766c5de16d05`.

The target cluster had unsafe pre-existing membership with ADMIN OPTION. Because database dumps do not enforce cluster-global role graphs, the ordinary LOGIN immediately executed `SET ROLE` after restore.

No automated pre-restore or post-restore role/ownership/wrapper audit is documented. [The deployment documentation](/Volumes/Developer%20SSD/ExpertAdvisor/docs/CampaignOperationsPhaseH1.rst:63) only states that LOGIN creation and memberships are reviewed deployment operations.

The remaining A–J restore variants cannot be classified as passing; the required invariant already failed.

## 12. Fixed-transition signature and ACL verdict

**Verified on a clean install.**

The three functions had:

- exact `public` schema and names;
- exact argument order/types;
- no defaults;
- no variadic arguments;
- no overloads;
- volatile classification;
- parallel unsafe;
- `SECURITY DEFINER`;
- owner `campaign_operations_h1_boundary_authority`;
- pinned `pg_catalog, public` search path;
- owner-only execution;
- no PUBLIC, `pqxx`, capability-role, or LOGIN execution.

Acquisition signature:

```text
(bigint, integer, text, timestamptz, text, text, text)
→ campaign_operations_dispatch_attempt
```

This verdict is limited by the alternate-schema wrapper blocker.

## 13. Supporting SECURITY DEFINER verdict

**FAILED.**

Functions selected by migration 055 in `public` were sealed, pinned, and had PUBLIC execution revoked. The audit did not cover every schema. A hostile boundary-owned `SECURITY DEFINER` wrapper in another schema survived migration and was publicly reachable.

## 14. Enable replay verdict

**Verified for the implemented and executed replay cases.**

The replay lookup precedes mutable-head checks, returns the stored identity and canonical bytes, creates no duplicate evidence, and rejects changed input/canonical evidence. Corrupt or partial evidence fails closed.

## 15. Disable replay verdict

**Verified for the implemented and executed replay cases.**

The stored disable event is returned before mutable-head validation. Same-key changed evidence conflicts and no duplicate event is created.

## 16. Acquisition replay verdict

**Verified for covered replay semantics.**

The replay path precedes mutable request/CAS checks and returns the original admission, Attempt V2, ordinal, versions, lease evidence, enablement chain, actor/build/capability, and audit identity. Covered mismatch paths use SQLSTATE `23505` for conflicts and `23514` for corrupt persisted evidence.

## 17. Full lock-order verdict

**Not fully verified.**

Source tracing shows the intended sequence:

`0a scheduler evidence → 0b production enablement → 1 authorization → 2 budget → 3 campaign/completion → 4 reservations → 5 requests → later lifecycle domains`

The deterministic test uses `pg_blocking_pids()` and `pg_locks`, which is valid evidence, but it blocks only at:

- campaign;
- reservation;
- request.

See [the three probes](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:484).

There is no complete independent acquisition-versus-completion/cancellation/revocation/expiry/disable/same-request/unrelated-request/replay matrix at all required boundaries.

## 18. Enablement hydration verdict

**Verified for capability and contract-version corruption.**

Repository hydration independently reads and validates stored capability, enablement contract version, canonical, hash, and typed mirrors before returning the object. Relevant validation begins in [CampaignOperationsProductionAdmissionRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:49).

## 19. Negative-test authenticity matrix

| Requirement | Actual branch/evidence | Result |
|---|---|---|
| Exact 61-byte name | Direct `proname::text` and `octet_length` | Pass |
| Clean generic `SET ROLE` | SQLSTATE `42501`, intended role guard | Pass |
| `SET LOCAL ROLE` | SQLSTATE `42501` | Pass |
| `SET SESSION AUTHORIZATION` | SQLSTATE `42501` | Pass |
| Fixed transition as generic LOGIN | SQLSTATE `42501`, function ACL | Pass |
| Context insertion | SQLSTATE `42501`, context-table ACL | Pass |
| Direct/nested hostile membership | SQLSTATE `42501`, migration diagnostic | Pass |
| Altered boundary attributes | Migration succeeded | **Fail** |
| Unexpected boundary-owned object | Migration succeeded | **Fail** |
| Alternate-schema SD wrapper | Migration and invocation succeeded | **Fail** |
| Restore unsafe ADMIN membership | Restore and `SET ROLE` succeeded | **Fail** |
| Former table-owner DML | SQLSTATE `42501`, intended table ACL | Pass |
| Replay changed canonical/input | `23505` conflict branches | Pass on covered cases |
| Partial/corrupt evidence | `23514` branches | Pass on covered cases |
| Same hash/different canonical at all five levels | Some levels covered, complete matrix absent | Incomplete |
| Recursive/nested trigger paths | Nested spoof covered; full recursion matrix absent | Incomplete |
| Lock order | Stages 3–5 only | Incomplete |
| Historical bytes after migration | Executed fixture passed | Pass |
| Historical bytes after restore | No complete byte-for-byte restore test | Incomplete |

The existing role-helper tests at [lines 560–601](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:560) do not prove inability to `SET ROLE` to the boundary.

## 20. Historical V1, Completion V1, and restore preservation

**Migration compatibility verified; restore preservation incomplete.**

Genuine pre-055 Attempt V1 and Completion V1 fixtures retained their canonical/hash bytes through migration 055. Completion V1 continued embedding stored Attempt V1 bytes, and V2 did not reinterpret V1.

The required byte comparison following every supported backup/restore scenario was not present and was not completed after restore safety failed.

## 21. H1 inertness and H2/H3/H4 boundary

**Verified for product reachability.**

No mutation CLI, run-once mode, continuous mode, production adapter, Manager, environment switch, constructor switch, worker launch/control, lifecycle mutation, or scheduler mutation was found.

Only read-only readiness/status CLI routes were added. H2, H3, and H4 were not implemented.

## 22. Migration, checksum, build, and repository

- Migration filename: `055_campaign_operations_production_admission_foundation.sql`
- Independently computed SHA-256: `5ce4d776b6d5be809297b44947f78b300543acab4d5bcf2f2ca1766c5de16d05`
- Embedded C++ checksum: match at [CampaignOperationsProductionAdmission.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmission.hpp:21)
- Disposable ledger checksum: match
- Clean installation: passed
- 054→055 upgrade: passed for supported fixture
- Migration replay: passed
- Transactional failure rollback: passed
- No backfill: passed
- `git diff --check`: passed
- `git diff --cached --check`: passed
- Isolated Release build: succeeded
- Xcode actually used: 26.6, not the documented 26.5
- Strict H1 and campaign C++ compilation with `-Wall -Wextra -Werror`: passed
- Release build emitted existing project/toolchain warnings; H1 strict compilation itself was clean
- Temporary cluster, roles, databases, logs, binaries, object files, and isolated DerivedData: removed
- Shared DerivedData: untouched
- Files changed by this verification: **none**
- Harness correction: only the disposable test database name was changed from an insufficiently marked name to one containing `disposable_test`; no source or production result changed

Executed successfully:

- `bash Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- Strict `CampaignOperationsPhaseH1Tests.cpp`
- Strict `CampaignOperationsTests.cpp`
- Full `CampaignOperationsRepositoryTests.cpp` against the disposable cluster
- `bash Tests/SchedulerCanonicalPathTests.sh`
- Phase H1, Phase 2, Phase 4, and Phase 5 CLI suites against the isolated Release binary
- Isolated `xcodebuild`
- Independent catalog, role, hostile-upgrade, wrapper, default-ACL, ownership, and restore probes

## 23. Findings ordered by severity

### F1 — Blocker: hostile attributes silently preserved

- **Requirement:** Exact frozen role identity; hostile-upgrade fail closed.
- **File:** [migration 055:66](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:66), [migration 055:3065](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3065)
- **Role:** `campaign_operations_h1_boundary_authority`
- **Clean impact:** Clean role happened to match expected state.
- **Upgrade impact:** Unsafe state is silently accepted.
- **Restore impact:** Altered global attributes survive outside database restore enforcement.
- **Scenario:** BYPASSRLS, NOINHERIT, connection limit, password, validity, and role configuration remained after successful migration.
- **Current reachability:** NOLOGIN alone prevents direct login, but does not satisfy the frozen contract and combines catastrophically with wrappers or restored membership.
- **Why tests missed it:** Existing hostile fixture tests only membership and one default ACL.
- **Correction:** Preflight every specified catalog property and fail `42501`; do not normalize a hostile pre-existing boundary role.
- **Reverification:** Run separate fixtures for every attribute, password, validity, and database/global role setting.

### F2 — Blocker: reachable alternate-schema SECURITY DEFINER wrapper

- **Requirement:** No wrapper, alternate entry point, or unexpected owned object.
- **File:** [migration 055:2904](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:2904)
- **Object:** Boundary-owned `hostile.exec_as_boundary(text)`.
- **Clean impact:** Not present in the clean fixture.
- **Upgrade impact:** Migration succeeds and preserves an ordinary LOGIN’s path to boundary superuser.
- **Restore impact:** Wrapper ownership and PUBLIC execution can be restored unchanged.
- **Scenario:** Ordinary LOGIN called the wrapper, became boundary `current_user`, and wrote a boundary-owned table.
- **Current reachability:** Fully reachable in the hostile upgrade fixture.
- **Why tests missed it:** Audit is restricted to qualifying functions in `public`; no cross-schema ownership/wrapper fixture exists.
- **Correction:** Add a fail-closed, all-schema ownership/ACL/dependency allowlist before any migration mutation.
- **Reverification:** Precreate wrapper, overload, same-name function, procedure, trigger, operator, cast, event trigger, and unexpected owned objects across alternate schemas.

### F3 — Blocker: restore does not enforce the role contract

- **Requirement:** Safe supported backup/restore.
- **File:** [CampaignOperationsPhaseH1.rst:63](/Volumes/Developer%20SSD/ExpertAdvisor/docs/CampaignOperationsPhaseH1.rst:63)
- **Role:** Restored boundary owner plus unsafe cluster-global LOGIN membership.
- **Clean impact:** None.
- **Upgrade impact:** Deployment can introduce the same state after migration.
- **Restore impact:** Valid checksum and ledger coexist with immediate boundary reachability.
- **Scenario:** Owner-preserving restore into an unsafe cluster; LOGIN successfully ran `SET ROLE`.
- **Current reachability:** Immediate in the restore fixture.
- **Why tests missed it:** No backup/restore suite or mandatory pre/post audit exists.
- **Correction:** Provide automated pre-restore and post-restore role, ownership, ACL, default-ACL, and wrapper audits that fail closed.
- **Reverification:** Execute all required A–J restore fixtures.

### F4 — High: boundary ownership is not minimal

- **Requirement:** Minimum sealed H1 ownership; no lifecycle/cancellation authority.
- **File:** [migration 055:2904](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:2904)
- **Objects:** 73 functions, including `apply_experiment_lifecycle_cancellation`.
- **Clean/upgrade/restore impact:** Enlarges the blast radius of any membership or wrapper defect.
- **Scenario:** Any reachable boundary role is superuser and also owns broad prior-phase functions.
- **Current reachability:** Clean graph blocks ordinary users; hostile upgrade and restore do not.
- **Why tests missed it:** Tests assert broad sealing rather than an exact ownership allowlist.
- **Correction:** Replace wildcard ownership transfer with the smallest reviewed H1 allowlist.
- **Reverification:** Exact owned-object snapshot and unrelated-domain mutation tests.

### F5 — High: acceptance claims exceed actual negative and lock tests

- **Requirement:** Requirement-to-branch authenticity and full lock-order matrix.
- **Files:** [migration test:306](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:306), [lock test:484](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:484), [implementation report:185](/Volumes/Developer%20SSD/ExpertAdvisor/CampaignOperations_PhaseH_H1_TargetedCorrection_Implementation_Output.md:185)
- **Impact:** False assurance across clean install, upgrade, and restore.
- **Scenario:** Report claims direct/inherited/nested/combined `SET ROLE` and full lock coverage; tests instead cover a production-reader membership fixture, role-helper reconstruction, and stages 3–5.
- **Current reachability:** Assurance defect; F2/F3 provide actual reachable failures.
- **Why tests missed it:** No executable requirement-to-branch matrix.
- **Correction:** Add exact principals, statements, SQLSTATEs, diagnostics, and failure-point assertions.
- **Reverification:** Run the complete matrix independently and compare logs to each claimed branch.

### F6 — Medium: future-object default-ACL matrix incomplete

- **Requirement:** Future functions, procedures, tables, sequences, schemas, and types for every H1 owner.
- **File:** [migration 055:3010](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3010)
- **Impact:** Future type/schema or alternate-schema defaults are not established.
- **Current reachability:** No clean unsafe default observed.
- **Why tests missed it:** Probes are limited to function/table/sequence coverage.
- **Correction:** Define and test safe defaults for every supported object kind and schema.
- **Reverification:** Create and inspect one future object of each kind as every owner.

## 24. Residual risks and deferred evidence

- Complete restore A–J matrix not run after a definitive restore blocker was reproduced.
- Full context recursion/subtransaction/concurrency matrix remains incomplete.
- Full lock-path matrix remains incomplete.
- Historical bytes after every restore form remain unverified.
- Xcode 26.6 was used instead of the documented 26.5.
- Release build contains pre-existing warnings outside the strict H1 compilation.
- Production role graph was intentionally not inspected or changed.

## 25. Skipped-suite classification

Skipped because a live scheduler and active workers were present:

| Suite | Classification |
|---|---|
| `SchedulerOwnershipIntegrationTests.sh` | Pre-enablement blocker requiring a safe process window |
| `SchedulerContinuationOwnershipIntegrationTests.sh` | Pre-enablement blocker requiring a safe process window |
| `SchedulerOwnershipProcessIntegrationTests.sh` | Pre-enablement blocker requiring a safe process window |
| `GlobalExperimentControlIntegrationTests.sh` | Pre-production blocker requiring a safe process window |

These skips are not the reason for the commit disposition; database authority blockers already prevent commit.

## 26. H1 disposition

- Safe to commit: **No**
- Safe to migrate into production while disabled: **No**
- Ready for H2: **No**
- Blocked pending correction: **Yes**
- Blocked only before enablement by safe-window regressions: **No**

## 27. Required correction and reverification commands

After correcting migration 055 and adding the missing fixtures:

```bash
bash Tests/CampaignOperationsPhaseH1MigrationTests.sh

bash Tests/SchedulerCanonicalPathTests.sh

EA_VERIFY_DD="$(mktemp -d /tmp/ea-h1-reverify-dd.XXXXXX)"
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath "$EA_VERIFY_DD" \
  build

bash Tests/CampaignOperationsPhaseH1CliTests.sh \
  "$EA_VERIFY_DD/Build/Products/Release/LSTM_Release"

shasum -a 256 \
  Database/migrations/055_campaign_operations_production_admission_foundation.sql

git diff --check
git diff --cached --check
git status --short
git diff --stat
```

During an approved safe process window:

```bash
bash Tests/SchedulerOwnershipIntegrationTests.sh
bash Tests/SchedulerContinuationOwnershipIntegrationTests.sh
bash Tests/SchedulerOwnershipProcessIntegrationTests.sh
bash Tests/GlobalExperimentControlIntegrationTests.sh
```

Required new test branches before reverification:

- every hostile boundary-role attribute;
- unexpected owned objects;
- every-schema wrapper/overload audit;
- all A–J restore scenarios;
- exact default ACLs for every owner/object kind;
- complete principal and SET ROLE matrix;
- complete lock-path matrix;
- post-restore historical byte comparison.

## Repository state

Branch: `campaign-operations`
HEAD: `fec53d5`
Working-tree entries: 27
Verifier-created repository changes: none

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
?? CampaignOperations_PhaseH_H1_TargetedArchitectureCorrectionAndImplementation_Output.md
?? CampaignOperations_PhaseH_H1_TargetedCorrection_FocusedIndependentCEEVerification_Output.md
?? CampaignOperations_PhaseH_H1_TargetedCorrection_Implementation_Output.md
?? docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md
```

`git diff --stat`:

```text
21 files changed, 7523 insertions(+), 16 deletions(-)
```

## 28. Final disposition token

`PHASE_H_H1_NOT_READY_TO_COMMIT`