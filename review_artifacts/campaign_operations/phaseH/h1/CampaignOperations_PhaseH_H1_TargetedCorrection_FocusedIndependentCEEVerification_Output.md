---
title: "Campaign Operations Phase H H1 Targeted Correction Focused Independent CEE Verification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_TargetedCorrection_FocusedIndependentCEEVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H H1 Targeted Correction Focused Independent CEE Verification

# Focused Independent CEE Reverification Report

## 1. Executive verdict

**NOT VERIFIED**

The targeted correction does not close all prior H1 findings. Independent catalog and transaction probes found two commit-blocking defects and several high-severity contract failures:

- The guarded transaction context is forgeable by its ordinary owner.
- PostgreSQL silently truncates the frozen 64-byte acquisition-function name.
- None of the three fixed transitions implements required exact replay.
- The acquisition path violates the frozen global lock order.
- Migration 055 introduces an undocumented eighth role with excessive ownership authority.
- Required default-ACL, role-graph, corruption, historical-fixture, and intended-branch tests remain incomplete.

Production mutation is not exposed through an H1 C++ or CLI surface, but default-off and current lack of LOGIN grants do not make the database authority safe.

## 2. H1 versus H2/H3/H4 scope boundary

**H1 inertness: verified.**

No production enable, disable, dispatch, canary, Manager, run-once, continuous-mode, scheduler-polling, worker-control, lifecycle-mutation, environment-switch, constructor-switch, or production-role deployment surface was found. The CLI test confirms only read-only readiness/status commands exist.

No H2, H3, or H4 behavior was implemented during this review, and I made no production-code changes.

**H1 correctness: not verified.** H2 must not begin while the blocker/high findings below remain open.

## 3. Prior-finding traceability

| Prior finding | Claimed correction | Independent result |
|---|---|---|
| Fixed transitions absent | Three fixed SQL functions added | Partial. Functions exist, but acquisition name is truncated, all lack exact replay, and ownership is unsafe. |
| Trigger-depth authority | Private table-based context | Not closed. Trigger depth is gone, but the table owner can forge context and invoke owner-controlled transitions. |
| H1 documentation crossed scope | Documentation corrected | Original scope text is largely restored, but the undocumented eighth transition-owner role creates a new frozen-authority conflict. |
| Leading-punctuation operation keys accepted | Explicit C++ and PostgreSQL leading-byte validation | Implementation closed; exhaustive required vector coverage remains incomplete. |
| Attempt V2 nested equality gaps | Added operation-key and expected-version comparisons | Implementation closed for the reported equalities; required per-field test traceability is incomplete. |
| Attempt V2 hydration reconstructed fields | Attempt-row fields loaded before comparison | Mostly closed for Attempt V2. Enablement hydration still reconstructs two stored typed mirrors. |
| Recovery eligibility drift | Status view aligned with migration 053 | Basic predicate alignment verified; implementation is mechanically duplicated and race/combination coverage is incomplete. |
| Warning/test gaps | Aggregate initialization fixed and tests expanded | Warning closed. Several negative tests reach the wrong branch or omit required threat actors and historical fixtures. |

## 4. Guarded-context security verdict

**NOT VERIFIED — blocker.**

The context object is `public.campaign_operations_production_transition_context`, defined at [migration 055](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:60).

Its structure is:

| Property | Actual value |
|---|---|
| Columns | `backend_pid integer`, `transaction_id bigint`, `transition_kind text`, `operational_request_id bigint`, `operation_key text` |
| Primary key | `backend_pid` |
| Transition kinds | `enable`, `disable`, `dispatch_v2` |
| Owner | `campaign_operations_production_transition_owner` |
| Sequence | None |
| RLS | Disabled; no forced RLS |
| PUBLIC/pqxx/H1 capability ACL | Revoked |
| Owner DML | Implicit INSERT/UPDATE/DELETE/TRUNCATE remains available |

The context validator checks PID, top-level transaction ID, transition kind, request ID, and optional operation key at [migration 055](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:839). Fixed enable, disable, and acquisition functions insert and explicitly delete context.

Context-dependent triggers include:

- `validate_campaign_operations_production_enablement_insert`
- `validate_campaign_ops_production_admission_insert`
- `validate_campaign_operations_dispatch_attempt_v2_insert`
- `guard_campaign_operations_production_admission_witness`
- `enforce_campaign_operations_production_admission_consistent`
- enablement audit completeness and post-completion guards

Independent disposable-database probing established:

- Direct ordinary DML without context reached the witness guard and failed.
- As `campaign_operations_production_transition_owner`, `has_table_privilege` was true for INSERT, UPDATE, DELETE, and TRUNCATE.
- That owner inserted a forged row containing its actual `pg_backend_pid()`, `txid_current()`, request ID, key, and `dispatch_v2`.
- A subsequent direct false→true request update passed the intended context check. The transaction was rolled back before deferred completeness evaluation.
- Without the forged row, the same statement failed with SQLSTATE `55000`.

This proves the intended guarded-context branch was reached and bypassed. It is not an unrelated ACL or row-shape failure.

The official “owner” test uses `campaign_operations_owner`, not the actual context owner, at [H1 migration tests](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:327). It therefore cannot detect this bypass.

## 5. Exception, savepoint, subtransaction, and session-reuse verdict

For legitimate fixed-function execution:

- Successful transitions explicitly delete the row before returning.
- A caught fixed-transition exception rolled back its subtransaction context.
- Savepoint rollback removed context created within the savepoint.
- A completed successful transition left zero context rows.
- Commit/rollback connection reuse therefore did not inherit legitimate context.

However, this is insufficient for the frozen automatic-clearing contract because:

- The protection depends on explicit deletion plus rollback, not an intrinsically transaction-local facility.
- The owner can create and commit forged or stale rows.
- No official test covers the true context owner, recursive triggers, successful transition followed by a second guarded operation in the same transaction, pooled-session reuse, or deterministic two-connection collision behavior.
- `backend_pid` as the primary key also lets the context owner create same-backend denial or collision state.

Accordingly, explicit deletion plus transaction rollback is sufficient only for current honest fixed-function paths. It is not sufficient for the required ordinary-owner threat model.

## 6. Transition-owner authority verdict

**NOT VERIFIED — blocker.**

Migration 055 creates `campaign_operations_production_transition_owner` at [lines 31–54](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:31).

The role is hardened to:

- NOLOGIN
- NOSUPERUSER
- NOCREATEDB
- NOCREATEROLE
- NOREPLICATION
- NOBYPASSRLS

Those attributes are correct. On a fresh disposable cluster it had no memberships.

Its effective authority is nevertheless excessive:

- Owns the context table.
- Owns the three fixed transition functions and context validator.
- Implicitly executes its owned functions even though ACLs show only owner execution.
- Can grant execution on its owned functions and DML on its owned context table.
- Receives full-table `SELECT, UPDATE` on operational requests.
- Receives `SELECT, INSERT` on enablement, enablement audit, admission, attempt, and dispatch audit tables.
- Receives sequence usage and helper-function execution at [lines 2352–2400](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:2352).

Migration hardening only revokes this role from `pqxx`. It neither removes nor rejects arbitrary pre-existing direct/inherited members, nor audits roles inherited by the transition owner. A pre-created role on upgrade can therefore retain unsafe reachability.

This is separate from unavoidable superuser power: the failure is reproducible under an ordinary NOLOGIN object owner reached through `SET ROLE` in the disposable threat fixture.

## 7. Frozen role-inventory compatibility verdict

**Unauthorized architecture change.**

The committed frozen architecture defines seven roles at [Phase H §14](/Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:539). Migration 055 creates eight.

The H1 documentation also says “seven NOLOGIN capabilities” and omits the transition owner at [CampaignOperationsPhaseH1.rst](/Volumes/Developer%20SSD/ExpertAdvisor/docs/CampaignOperationsPhaseH1.rst:41).

Because this additional role owns the protected boundary and changes its threat model, it is not merely an internal naming detail or documentation omission. Keeping it requires a separately accepted architecture amendment and a defensible owner-isolation design.

## 8. Exact transition signatures and ACL verdict

| Frozen function | Catalog result |
|---|---|
| `record_campaign_operations_production_enable_v1` | Exact name; 11 arguments; returns enablement-event row; VOLATILE; PARALLEL UNSAFE; SECURITY DEFINER; not leakproof; no defaults/variadic; pinned search path |
| `record_campaign_operations_production_disable_v1` | Exact name; 6 arguments; same security/runtime properties |
| `transition_campaign_operations_request_dispatching_production_v2` | **Not representable as written.** PostgreSQL truncates it to `transition_campaign_operations_request_dispatching_production_v` |

The acquisition identifier is 64 bytes while PostgreSQL permits 63-byte identifiers. Its declaration is at [migration 055](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1850). Migration output emits truncation notices.

Existing tests use the same overlong spelling. PostgreSQL truncates their `regprocedure` lookup too, so those tests falsely report the exact frozen name as present. Correct catalog verification must compare `pg_proc.proname::text` to the full expected text.

Existing fixed functions have:

- Qualified transition bodies and pinned `pg_catalog, public` search paths.
- No unsafe defaults, variadics, or permissive overloads.
- PUBLIC, `pqxx`, and H1 capability execution revoked.
- No C++ or CLI caller.

But:

- The transition owner retains implicit execution.
- No default ACL exists for future functions owned by this role. PostgreSQL’s default function ACL includes PUBLIC EXECUTE.
- Several SECURITY DEFINER trigger functions used by the boundary lack pinned search paths and use unqualified Campaign Operations relations, beginning at [line 860](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:860).

An accepted shorter acquisition name is required; this cannot be repaired solely in implementation while preserving the frozen literal name.

## 9. Enable transition verdict

**NOT VERIFIED.**

Verified behavior:

- Locks exact scheduler evidence, then the exclusive production gate.
- Requires complete generation-52 evidence and exact scheduler canonical.
- Validates the Manager service/build canonical.
- Appends one immutable event and matching audit atomically.
- Correctly rejects a non-disable predecessor.
- No scheduler liveness claim is used.

Failure:

- There is no operation-key replay lookup before predecessor validation at [lines 1678–1700](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1678).
- Independent exact replay of the committed enable returned SQLSTATE `40001`, rather than the original identical event.
- Same key/different canonical is not full-compared as required.

## 10. Disable transition verdict

**NOT VERIFIED.**

Verified behavior:

- Uses only the production gate and current enable head.
- Has no scheduler-readiness dependency, preserving emergency availability.
- Appends disable and audit atomically.
- Does not mutate scheduler, lifecycle, worker, admission, lease, binding, or prior audit evidence.

Failure:

- No operation-key replay lookup exists at [lines 1795–1807](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1795).
- Independent exact replay returned SQLSTATE `40001`.
- Same-key canonical conflict classification is absent.

## 11. Atomic production acquisition verdict

**NOT VERIFIED.**

A first successful invocation does atomically create admission, false→true witness, Attempt V2, lease fields, and production audit. Rollback removed all partial evidence in focused probes.

The following defects invalidate the boundary:

1. No `(request_id, operation_key)` replay lookup; replay fails CAS at [lines 1888–1905](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1888).
2. The context owner can forge the authorization row or invoke the owned function.
3. Lock order is inverted. The function locks scheduler/gate, then request. Admission insertion subsequently invokes the post-completion trigger, which locks campaign/completion after request. Frozen order requires campaign/completion before request.
4. The fixed function does not itself establish the complete frozen authorization→budget→campaign→reservation→request order.
5. No deterministic `pg_blocking_pids()` H1 transition test exists.

Independent acquisition replay returned SQLSTATE `40001`, not existing-identical evidence.

## 12. Operation-key equivalence verdict

**Implementation verified; test evidence incomplete.**

C++ validation at [CampaignOperationsProductionAdmission.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmission.cpp:28) explicitly checks:

- byte length 1–128;
- ASCII alphanumeric first byte;
- only ASCII alphanumeric plus `._:/-` thereafter.

PostgreSQL uses the frozen regex at [migration 055](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:66). Because every permitted byte is ASCII, regex character length cannot admit a multi-byte valid key.

Spaces, newline, tab, percent, underscore-leading, backslash, quotes, delimiters, UTF-8 non-ASCII, and embedded NUL are rejected by the C++ byte validator or PostgreSQL text representation.

The current tests cover representative keys only at [C++ test](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1Tests.cpp:128) and [SQL test](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:291). They do not provide the requested exhaustive ASCII-boundary, locale, all-first-byte, all-punctuation, and embedded-NUL trace.

## 13. Attempt V2 nested-validation verdict

**Implementation verified for the reported defects; coverage incomplete.**

The C++ builder validates operation key, request identity, admission expected version, request ID, enable ID/canonical, actor, original principal, approved build, resulting version, ordinal, capability, and contract versions before persistence.

The SQL validator independently compares operation key and expected version to admission at [lines 1006–1028](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1006).

The nullable V2 shape test is authentic: it enters replica mode specifically to reach the named `campaign_operations_dispatch_attempt_v1_v2_shape` constraint and verifies its constraint name.

Missing evidence:

- Every nested mismatch is not tested independently.
- Expected/resulting versions, actor, capability, contract version, and several IDs are absent from the repository corruption matrix.
- Combinations of malformed NULL and nested canonical discrepancies are not covered.

## 14. Repository hydration and corruption rejection verdict

**PARTIAL — medium finding remains.**

Attempt V2 hydration now selects all attempt-row fields before loading admission and enablement and compares duplicated rows before returning, at [repository lines 272–345](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:272). This closes the principal reconstruction defect.

Enablement hydration is still incomplete:

- `HeadColumns()` omits stored `capability` and `enablement_contract_version` at [lines 90–105](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:90).
- `MapHead()` reconstructs their fixed values via the C++ builder at [lines 49–87](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:49).
- Corruption of those two stored typed mirrors can therefore be masked when canonical/hash bytes remain unchanged.
- The enablement corruption loop does not test either field at [repository test](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:195).

Same-hash/different-canonical coverage exists for Attempt V2, admission, and enablement hydration, but not request evidence and Completion V1.

## 15. Phase F recovery-status alignment verdict

**Verified with drift risk.**

The H1 status predicate at [migration 055 lines 2154–2182](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:2154) mechanically matches migration 053’s authority at [migration 053 lines 1399–1427](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql:1399):

- dispatching state;
- expired lease;
- no request binding;
- no materialization-member conversion execution;
- no outcome for the maximum attempt ordinal.

Positive, binding, execution, and latest-outcome cases passed. The logic is duplicated rather than shared, so future Phase F changes can drift. Combination cases, stale non-latest outcomes, and deterministic binding/execution races remain unverified.

## 16. Negative-test intended-branch authenticity

| Requirement | Current test | Authenticity |
|---|---|---|
| Nested trigger cannot spoof context | Temporary trigger performs direct update | Reaches witness guard; authentic for one nested level only |
| Recursive trigger spoof | None | Missing |
| Context owner DML | Tests `campaign_operations_owner` | Wrong threat actor; false confidence |
| Forged PID/xid/request/key/kind | None | Missing; independent owner probe succeeded |
| Caught exception cleanup | Not in committed H1 suite | Independent disposable probe passed |
| Savepoint/subtransaction cleanup | Not in committed H1 suite | Independent disposable probe passed |
| Session reuse/concurrent collision | None | Missing |
| Exact transition replay | Final row counts labeled “rollback/replay” | Does not call a transition twice |
| Same hash/different enablement canonical | Insert catches `insufficient_privilege` | Wrong branch; context denial occurs before canonical/hash validation |
| Leading operation punctuation | Representative five-character set | Authentic but incomplete |
| Attempt nested mismatches | Request ID, key, version samples | Authentic but incomplete |
| Malformed NULL V2 shape | Replica mode plus exact constraint name | Authentic |
| Hydration corruption | Attempt/admission/enablement loops | Authentic for listed fields; matrix incomplete |
| Literal V1 prefix | Migration suite fixture | Authentic |
| Direct/inherited production exclusion | Partial role probes | Nested and combined membership matrix incomplete |
| Recovery predicates | Positive and main individual negatives | Authentic; combinations/races incomplete |
| Historical V1/Completion bytes | V1 row generated after 055 | Not historical evidence |
| Post-completion evidence | Several inserts | Most reach completion gate; Boolean case reaches witness immutability first |

## 17. Completion V1 and historical compatibility verdict

**PARTIAL; historical claim unverified.**

The current Completion V1 chain embeds the stored Attempt V2 canonical, and focused execution showed the persisted V2 bytes present in completion request evidence. Malformed/incomplete V2 evidence is rejected by shape and completion gates.

However:

- The “historical V1” test creates a new V1 row after migration 055 and rolls it back at [H1 migration test lines 342–415](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:342).
- No fixture persisted before migration 055 is upgraded and byte-compared.
- No pre-055 Completion V1 fixture is tested.
- Same-hash/different-canonical rejection is not independently proven at request-evidence and Completion V1 levels.

No migration 055 backfill or explicit rewrite was found, so accidental rewriting is unlikely, but the required historical proof is absent.

## 18. Migration, checksum, and no-backfill verdict

Migration filename and checksum are internally consistent:

- Filename: `055_campaign_operations_production_admission_foundation.sql`
- SHA-256: `9b272b286fe3244c9f74a79e87b4a4b66d9255b98eb11adea155910260c96b3c`
- Embedded C++ checksum: matches.
- Runner ledger checksum behavior: matches.

Verified in disposable databases:

- schema 054→055 upgrade;
- supported replay before H1 evidence;
- checksum-stable ledger record;
- mid-migration failure rollback;
- no H1 evidence backfill;
- no LOGIN grants;
- no scheduler/lifecycle/worker data mutation;
- clean schema-033 restoration followed by migrations 034–055.

A raw empty-database runner attempt failed at historical migration 004 because prerequisite role `pqxx` was absent; this was a harness/base-schema mismatch, not an H1 migration defect.

Migration is nevertheless not production-safe because of the transition-owner, context-forgery, exact-name, replay, and lock-order findings.

## 19. Build, warnings, and repository integration

Results:

- Strict H1 `clang++ -std=c++20 -Wall -Wextra -Werror`: passed.
- Aggregate initialization warning: genuinely fixed.
- No new warning suppression appears in the H1 diff.
- H1 source files are included in the correct Xcode target.
- Isolated Release build: `** BUILD SUCCEEDED **`.
- `git diff --check`: passed.
- `git diff --cached --check`: passed; there is no staged diff.
- Shared `DerivedData/ExpertAdvisor` was not used or modified.
- Disposable clusters, binaries, and temporary build paths created by this review were removed.
- Ignored implementation/review transcripts were inspected as claims/activity evidence, including the 28,795,765-byte targeted-correction transcript.

## 20. Regression execution and skipped suites

Executed:

| Suite/check | Result |
|---|---|
| `CampaignOperationsPhaseH1MigrationTests.sh` | Passed, but contains the authenticity gaps above |
| `CampaignOperationsPhaseH1Tests.cpp` | Strict compile and run passed |
| `CampaignOperationsPhaseH1RepositoryTests.cpp` | Passed inside the isolated H1 PostgreSQL harness |
| `CampaignOperationsPhaseH1CliTests.sh` | Passed against isolated Release binary |
| `CampaignOperationsTests.cpp` | Strict compile and run passed |
| `CampaignOperationsRepositoryTests.cpp` | Passed against disposable PostgreSQL |
| Phase 1–5 migrations | Applied through the repository/H1 isolation paths |
| Phase 2, 4, 5 CLI parser suites | Passed |
| `SchedulerCanonicalPathTests.sh` | Passed |
| `SchedulerOwnershipMigrationTests.sql` | Passed after isolated fixture adaptation |
| Release Xcode build | Passed |
| Git whitespace checks | Passed |

The unmodified scheduler ownership SQL first failed because no complete generation-52 protocol fixture was present. After adding isolated cutover evidence, it failed because `model.name` is now NOT NULL. A streamed, non-repository harness adaptation supplying cutover evidence and names passed. No production behavior or repository file was changed.

Live scheduler/training processes were present, so these were not run:

| Skipped suite | Classification |
|---|---|
| `SchedulerOwnershipIntegrationTests.sh` | Pre-enablement blocker; safe-window process regression |
| `SchedulerContinuationOwnershipIntegrationTests.sh` | Pre-enablement blocker; safe-window process regression |
| `SchedulerOwnershipProcessIntegrationTests.sh` | Pre-enablement blocker; safe-window process regression |
| `GlobalExperimentControlIntegrationTests.sh` | Pre-enablement blocker; safe-window process regression |

They are not, by themselves, H1 commit blockers. H1 is already blocked by independent database-contract defects.

## 21. Findings ordered by severity

### F1 — Blocker: guarded context is owner-forgeable

- **Invariant:** Direct table DML and the context owner must not reproduce fixed-transition authority.
- **Reference:** [Context definition and ownership](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:60); [validator](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:839).
- **Object:** `campaign_operations_production_transition_context`; `campaign_operations_production_transition_owner`.
- **Scenario:** Owner inserts current PID/xid plus dispatch identity, then direct false→true DML passes the guarded check.
- **Currently reachable:** Not through supported H1 CLI/C++; reachable to the ordinary owner or any retained member after migration.
- **Why tests missed it:** They test `campaign_operations_owner`, not the context owner.
- **Required correction:** Redesign so no ordinary object owner can forge, persist, grant, or indirectly reproduce context. Resolve role inventory through accepted architecture.
- **Reverification:** Repeat A–J as the actual owner and through direct/nested inherited LOGIN membership, proving the guarded SQLSTATE/diagnostic is reached.

### F2 — Blocker: frozen acquisition function name is impossible in PostgreSQL

- **Invariant:** Exact schema-qualified frozen function name.
- **Reference:** [Frozen declaration](/Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:665); [migration declaration](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1850).
- **Function:** Frozen 64-byte name; catalog contains a 63-byte truncated name ending in `_v`.
- **Scenario:** Tests and callers that rely on PostgreSQL’s identifier truncation appear to find a function that does not have the exact frozen name.
- **Currently reachable:** Function exists only under the truncated catalog name; no H1 caller.
- **Why tests missed it:** `regprocedure` test inputs are silently truncated too.
- **Required correction:** Accept an architecture amendment selecting a valid ≤63-byte name, then update migration, catalog tests, checksum, and documents.
- **Reverification:** Compare `pg_proc.proname::text` byte-for-byte and reject truncation notices.

### F3 — High: exact replay is absent from all fixed transitions

- **Invariant:** Exact operation-key replay returns original complete evidence; changed input conflicts.
- **References:** [Frozen replay contract](/Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:267); [enable](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1678); [disable](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1795); [acquisition](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1888).
- **Scenario:** Replaying enable, disable, or acquisition returns `40001` predecessor/CAS conflict.
- **Currently reachable:** Database functions exist; no H1 C++ caller.
- **Why tests missed it:** “rollback/replay stability” only counts rows at [test line 940](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:940).
- **Required correction:** Perform exact durable lookup before mutable-head/CAS validation and full-compare all logical fields and nested bytes.
- **Reverification:** Exact replay, same key/different field, same hash/different canonical, partial evidence, and in-doubt lookup for all three functions.

### F4 — High: acquisition violates frozen lock order

- **Invariant:** 0a→0b→authorization→budget→campaign/completion→reservation→request.
- **Reference:** [Frozen lock order](/Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:506); [request lock](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1888).
- **Scenario:** Acquisition locks the request and later admission triggers lock campaign/completion, opposing completion/cancellation order and permitting deadlock.
- **Currently reachable:** Only through direct database function invocation in H1.
- **Why tests missed it:** No deterministic independent-connection locking test.
- **Required correction:** Make the complete fixed acquisition boundary use the accepted order.
- **Reverification:** Independent connections with `pg_blocking_pids()` proving each lock boundary and no reverse wait.

### F5 — High: transition-owner/default-ACL and SECURITY DEFINER hardening incomplete

- **Invariant:** Frozen role graph, pinned search paths, no future PUBLIC execution, no unsafe inherited execution.
- **References:** [role creation](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:31); [unpinned trigger example](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:860).
- **Scenario:** Owned functions remain implicitly executable; future functions default to PUBLIC EXECUTE; unpinned trigger functions resolve unqualified objects through caller-influenced search paths.
- **Currently reachable:** No supported H1 caller, but unsafe after role membership deployment or retained upgrade membership.
- **Why tests missed it:** Only current explicit ACLs and `pqxx` membership are checked.
- **Required correction:** Resolve ownership design, default ACLs, role graph, and pin/qualify every SECURITY DEFINER boundary.
- **Reverification:** Full object/default-ACL/role graph catalog matrix with generic, nested, combined, owner, pqxx, and each capability role.

### F6 — High: required negative-test authenticity is incomplete

- **Invariant:** Every required negative test must reach the intended boundary.
- **References:** [owner test](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:327); [false same-hash test](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:856).
- **Scenario:** Same-hash canonical test passes on earlier `42501`; owner test uses wrong owner; replay is not tested; recursive/concurrent/session cases are absent.
- **Currently reachable:** Test-validity issue masks production defects.
- **Why tests missed it:** Expected exception class is too broad or wrong actor/statement is used.
- **Required correction:** Add expected SQLSTATE, exact diagnostic/constraint, and actual failure-point assertions for every branch.
- **Reverification:** Complete branch-coverage traceability table plus independent catalog/transaction probes.

### F7 — Medium: enablement hydration omits stored typed mirrors

- **Invariant:** Every stored field must be loaded from its row and independently compared.
- **Reference:** [HeadColumns](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:90).
- **Scenario:** Stored `capability` or `enablement_contract_version` is corrupt while canonical/hash remain unchanged; hydration reconstructs accepted constants and masks corruption.
- **Currently reachable:** Requires corruption authority; normal immutable paths should prevent it.
- **Why tests missed it:** Corruption loop omits both columns.
- **Required correction:** Select, validate, and corrupt-test both stored fields.
- **Reverification:** Independent per-column corruptions before repository/service return.

### F8 — Medium: historical byte preservation lacks historical fixtures

- **Invariant:** Attempt V1 and Completion V1 persisted before 055 remain byte-identical after upgrade.
- **Reference:** [post-055 generated V1 fixture](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:342).
- **Scenario:** Tests regenerate expected bytes using current functions, which cannot detect an upgrade-time compatibility regression.
- **Currently reachable:** Migration does not appear to rewrite rows, but proof is absent.
- **Why tests missed it:** Fixture is created after migration 055 and rolled back.
- **Required correction:** Persist authoritative pre-055 Attempt V1 and Completion V1 rows, upgrade, then byte-compare stored data.
- **Reverification:** 054→055 fixture upgrade plus nested same-hash/different-canonical corruption tests.

## 22. Residual risks and deferred evidence

- Exact owner-safe context design remains unresolved.
- A valid frozen acquisition identifier and role-inventory amendment are required.
- Fixed transition in-doubt behavior is not implemented or tested.
- Transition concurrency and deadlock behavior lacks deterministic evidence.
- Recovery predicate duplication can drift from Phase F.
- Exhaustive key-vector and per-field V2 mismatch coverage is absent.
- Safe-window scheduler/process suites remain required before enablement.
- Superuser power was excluded from the ordinary-owner assessment, as requested.

## 23. H1 disposition

| Decision | Determination |
|---|---|
| Safe to commit | **No** |
| Safe to migrate into production while disabled | **No** |
| Ready for H2 implementation | **No** |
| Blocked pending correction | **Yes** |
| Blocked only before enablement by process regressions | **No**; independent H1 blockers exist now |

## 24. Required correction verification commands

After the architecture/name/role decision and corrections:

```bash
git diff --check
git diff --cached --check

Tests/CampaignOperationsPhaseH1MigrationTests.sh

clang++ -std=c++20 -Wall -Wextra -Werror \
  -I Sources -I Headers \
  Tests/CampaignOperationsPhaseH1Tests.cpp \
  Sources/CampaignOperations.cpp \
  Sources/CampaignOperationsDispatch.cpp \
  Sources/CampaignOperationsProductionAdmission.cpp \
  Sources/ExperimentRecommendation.cpp \
  -o /tmp/CampaignOperationsPhaseH1Tests
/tmp/CampaignOperationsPhaseH1Tests

Tests/SchedulerCanonicalPathTests.sh

review_dd="$(mktemp -d /tmp/ea-h1-reverify-deriveddata.XXXXXX)"
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath "$review_dd" \
  build

Tests/CampaignOperationsPhaseH1CliTests.sh \
  "$review_dd/Build/Products/Release/LSTM_Release"
Tests/CampaignOperationsPhase2CliTests.sh \
  "$review_dd/Build/Products/Release/LSTM_Release"
Tests/CampaignOperationsPhase4CliTests.sh \
  "$review_dd/Build/Products/Release/LSTM_Release"
Tests/CampaignOperationsPhase5CliTests.sh \
  "$review_dd/Build/Products/Release/LSTM_Release"
```

The corrected disposable PostgreSQL harness must additionally run:

- context A–J as the actual context owner and nested LOGIN members;
- exact replay/conflict for enable, disable, and acquisition;
- exact catalog-name text comparison;
- default ACL and complete membership graph checks;
- `pg_blocking_pids()` lock-order tests;
- every Attempt V2 and enablement stored-field corruption;
- pre-055 Attempt V1 and Completion V1 upgrade fixtures.

At a scheduler-safe window, before enablement:

```bash
Tests/SchedulerOwnershipIntegrationTests.sh
Tests/SchedulerContinuationOwnershipIntegrationTests.sh
Tests/SchedulerOwnershipProcessIntegrationTests.sh
Tests/GlobalExperimentControlIntegrationTests.sh
```

## 25. Repository handoff

Reviewer changes: **none**.

Behavior changed by this review: **none**. No production database, scheduler, process, role, membership, lifecycle row, or shared DerivedData was modified.

Final worktree summary:

```text
Branch: campaign-operations
HEAD: fec53d5182c70249b26bce7d6c008b444b69bd89
Staged diff: none
Tracked diff: 19 files changed, 5931 insertions(+), 6 deletions(-)
```

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
?? CampaignOperations_PhaseH_H1_Implementation_Output.md
?? CampaignOperations_PhaseH_H1_IndependentCEEVerification_Output.md
?? CampaignOperations_PhaseH_H1_TargetedCorrection_Implementation_Output.md
```

Final disposition:

`PHASE_H_H1_NOT_READY_TO_COMMIT`