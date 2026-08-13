---
title: "Campaign Operations Phase H H1 Targeted Test Coverage Focused Independent Reverification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_TargetedTestCoverage_FocusedIndependentReverification_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# Campaign Operations Phase H H1 Targeted Test Coverage Focused Independent Reverification

# 1. Executive Summary

The H1 targeted corpus is substantial and authentic for most protected-function, ACL, replay, and readiness behaviors. Both disposable PostgreSQL test harnesses passed.

However, independent source-to-test mapping found material gaps:

- No hostile mutation tests for frozen `language`, `provariadic`, or `proconfig/search_path`.
- First-Attempt replay lacks corruption tests for typed relationship IDs and the approved-build hash mirror.
- No adversarial duplicate first-Attempt-row replay test.
- No readiness negative controls for unrelated role edges or unrelated completion constraints.
- Some readiness integrity tests accept a set of error codes rather than one exact stable diagnostic.

Therefore Step #4 is not independently closed.

# 2. Independent Verdict

`TARGETED_TEST_COVERAGE_REMAINING_GAPS_FOUND`

The passing harnesses prove that existing tests execute real persisted-state mutations and reach intended branches. They do not prove complete hostile-counterexample coverage.

# 3. One-to-One Targeted Coverage Matrix

| Counterexample family | Exact test/fixture | Exact mutation/setup | Intended branch | Expected SQLSTATE/result | Expected diagnostic | Reached intended branch? | Disposition |
|---|---|---|---|---|---|---|---|
| Alternate-schema protected name | `alternate_schema` in [ProtectedFunctionPreflightTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh:339>) | Moves protected function identity to another schema | Protected-function identity preflight | `42501` | `H1A005` | YES | COVERED |
| Extra overload | `overload`; registered `H1OBJ017` | Creates same-name overload | Exact protected identity resolution | `42501` | `H1A005` | YES | COVERED |
| Input signature mismatch | `input_signature` | Mutates input type signature | Signature contract validation | `42501` | `H1A005` | YES | COVERED |
| Identity-argument mismatch | `input_identity` | Mutates identity argument metadata | Catalog contract validation | `55000` | `H1A008` | YES | COVERED |
| Default-argument variant | `default_variant`; registered `H1OBJ018` | Creates same signature with an added default argument | Protected identity/default-argument validation | `42501` | `H1A005` | YES | COVERED |
| Procedure collision | Migration deployment audit fixture | Creates procedure with colliding protected name | Object-kind/identity validation | `42501` or `55000`, depending on fixture | `H1A005/H1A008` | YES | COVERED |
| Aggregate collision | `aggregate_kind` | Creates aggregate-shaped collision | Protected object-kind validation | `55000` | `H1A008` | YES | COVERED |
| Same-prefix collision | Alternate schema, overload, and default-variant cases | Colliding names with non-identical identities | Exact identity resolution | Exact failure above | Exact H1A005/H1A008 code | YES | SUBSUMED-BY-STRONGER-TEST |
| `prokind` mismatch | `object_kind` | Mutates function object kind | Catalog contract validation | `55000` | `H1A008` | YES | COVERED |
| Volatility mismatch | `volatility` | Mutates `provolatile` | Catalog contract validation | `55000` | `H1A008` | YES | COVERED |
| Leakproof mismatch | `leakproof` | Mutates `proleakproof` | Catalog contract validation | `55000` | `H1A008` | YES | COVERED |
| SECURITY INVOKER mismatch | `security_invoker` | Mutates security-definer flag to invoker | Catalog contract validation | `55000` | `H1A008` | YES | COVERED |
| SECURITY DEFINER mismatch | `security_definer` | Mutates security-invoker function to definer | Catalog contract validation | `55000` | `H1A008` | YES | COVERED |
| Language mismatch | No targeted mutation found | No alternate `prolang` mutation | Catalog contract validation | Expected `55000` / `H1A008` | Catalog-contract mismatch | NO | GAP |
| `proconfig` / search_path mismatch | No targeted mutation found | No hostile `ALTER FUNCTION ... SET search_path` mutation | Catalog contract validation | Expected `55000` / `H1A008` | Catalog-contract mismatch | NO | GAP |
| Variadic mismatch | No targeted mutation found | No `VARIADIC` or `provariadic` mutation | Catalog contract validation | Expected `55000` / `H1A008` | Catalog-contract mismatch | NO | GAP |
| Parallel-safety mismatch | `parallel_safety` | Mutates `proparallel` | Catalog contract validation | `55000` | `H1A008` | YES | COVERED |
| Strictness mismatch | Not frozen by accepted contract | `proisstrict` is not included in the frozen contract | Not applicable | Not applicable | Not applicable | N/A | SUBSUMED-BY-STRONGER-TEST |
| Return-type mismatch | `return_type` | Mutates return type | Return contract validation | `55000` | `H1A008` | YES | COVERED |
| OUT metadata mismatch | `out_argument` | Mutates OUT argument metadata | Return/argument metadata validation | `55000` | `H1A008` | YES | COVERED |
| INOUT metadata mismatch | `inout_argument` | Mutates INOUT metadata | Return/argument metadata validation | `55000` | `H1A008` | YES | COVERED |
| TABLE metadata mismatch | `table_return` | Mutates TABLE-return metadata | Return/argument metadata validation | `55000` | `H1A008` | YES | COVERED |
| Public EXECUTE | `public_execute` | Adds PUBLIC execute privilege | ACL validation | `42501` | `H1A006` | YES | COVERED |
| Grant option | `grant_option` | Adds grant option | ACL validation | `42501` | `H1A006` | YES | COVERED |
| Extra explicit grantee | `explicit_grantee` | Adds unauthorized explicit grantee role | ACL validation | `42501` | `H1A006` | YES | COVERED |
| Missing explicit grant | `missing_explicit_grant` | Removes expected explicit grant | ACL validation | `42501` | `H1A006` | YES | COVERED |
| Missing expected grantee role | `missing_expected_role` | Removes expected role/grant | ACL validation | `42501` | `H1A006` | YES | COVERED |
| Recreated/missing grantee role | `missing_recreated_expected_role` | Drops and recreates expected role path | ACL and role existence validation | `42501` | `H1A006` | YES | COVERED |
| NULL vs explicit ACL origin | `explicit_acl_origin` | Changes ACL origin between NULL and explicit ACL | ACL tuple/origin validation | `42501` | `H1A006` | YES | COVERED |
| Protected-function contract duplication | Six-family anti-drift loop | Extracts and compares all duplicated arrays | Source contract drift validator | Comparator must fail on mutation | Mutation-detected failure | YES | COVERED |
| Literal drift in `protected_functions` | Anti-drift mutation loop | Mutates protected literal | Duplicate-contract comparison | Nonzero/failure required | Contract mismatch | YES | COVERED |
| Literal drift in signatures | Anti-drift mutation loop | Mutates signature literal | Duplicate-contract comparison | Nonzero/failure required | Contract mismatch | YES | COVERED |
| Literal drift in contracts | Anti-drift mutation loop | Mutates contract literal | Duplicate-contract comparison | Nonzero/failure required | Contract mismatch | YES | COVERED |
| Literal drift in return contracts | Anti-drift mutation loop | Mutates return-contract literal | Duplicate-contract comparison | Nonzero/failure required | Contract mismatch | YES | COVERED |
| Literal drift in identity contracts | Anti-drift mutation loop | Mutates identity-contract literal | Duplicate-contract comparison | Nonzero/failure required | Contract mismatch | YES | COVERED |
| Literal drift in final ACL contracts | Anti-drift mutation loop | Mutates final ACL literal | Duplicate-contract comparison | Nonzero/failure required | Contract mismatch | YES | COVERED |
| Valid first-Attempt hydration | Repository loaded-path tests and SQL persisted graph | Hydrates persisted Attempt #1 and mirrors | Loaded hydration | Success | Valid persisted evidence | YES | COVERED |
| Valid Attempt #2 replay | SQL exact replay block | Replays persisted Attempt #2 after valid Attempt #1 | Production acquisition replay | Success, unchanged identity | Exact replay success | YES | COVERED |
| First-Attempt canonical corruption | `verify_first_attempt_corruption_rejected` | Changes canonical, retains hash | Replay evidence integrity | `23514` | `production acquisition replay evidence corrupt` | YES | COVERED |
| First-Attempt hash corruption | Same helper | Changes hash, retains canonical | Replay evidence integrity | `23514` | Same exact diagnostic | YES | COVERED |
| Typed request mirror canonical corruption | Same helper | Changes typed request mirror canonical | Replay evidence integrity | `23514` | Same exact diagnostic | YES | COVERED |
| Typed relationship-ID corruption | No first-Attempt replay mutation | No mutation of admission/event relationship IDs | Replay evidence integrity | Expected `23514` | Replay evidence corrupt | NO | GAP |
| Admission canonical/hash corruption | Same helper | Changes first admission canonical or hash | Replay evidence integrity | `23514` | Same exact diagnostic | YES | COVERED |
| Enablement canonical/hash corruption | Same helper | Changes first enablement canonical or hash | Replay evidence integrity | `23514` | Same exact diagnostic | YES | COVERED |
| Approved-build canonical corruption | Same helper | Changes approved-build canonical | Replay evidence integrity | `23514` | Same exact diagnostic | YES | COVERED |
| Approved-build hash corruption | No first-Attempt replay mutation | Hash mirror is not independently changed | Replay evidence integrity | Expected `23514` | Replay evidence corrupt | NO | GAP |
| Acquisition-audit corruption | Multiple first-Attempt audit mutations | Deletes, duplicates, or contradicts audit fields | Replay evidence integrity | `23514` | Same exact diagnostic | YES | COVERED |
| Missing first Attempt | Historical hydration/replay corruption helper | Deletes Attempt #1 | Historical evidence hydration | Corruption rejection | Corruption error / replay evidence corrupt | YES | COVERED |
| Duplicate first Attempt | No adversarial duplicate Attempt row | Only static uniqueness and duplicate audit are tested | Replay graph integrity | Expected rejection | Duplicate/constraint corruption | NO | GAP |
| Operation-key corruption | Same helper | Changes first Attempt operation key | Replay evidence integrity | `23514` | Replay evidence corrupt | YES | COVERED |
| Ordinal corruption | Same helper | Changes Attempt ordinal | Replay evidence integrity | `23514` | Replay evidence corrupt | YES | COVERED |
| Request-version corruption | Same helper | Contradicts expected/resulting request versions | Replay evidence integrity | `23514` | Replay evidence corrupt | YES | COVERED |
| Build relationship corruption | Canonical build mutation and audit-version mutations | Changes build evidence relationship | Replay evidence integrity | `23514` | Replay evidence corrupt | PARTIAL | GAP |
| Attempt #2 replay blocked by corrupt Attempt #1 | Same helper | Validates mutation, replays Attempt #2, compares before/after | Replay refusal and immutability | `23514`, Attempt #2 unchanged | Replay evidence corrupt | YES for tested fields | COVERED |
| Cross-principal exact recovery replay | SQL cross-principal recovery block | Authorized/unauthorized role replay with historical identity | Exact recovery replay | Success for authorized principal; permission failure for unauthorized | Historical exact replay; permission denied | YES | COVERED |
| Wrong Manager-build version | `AssertLoadedReadableVersionBlock` | Rewrites canonical and hash consistently to version 2 | Readiness semantic validation | `ready=false` | `manager_build_contract_version=2` | YES | COVERED |
| Manager-build structural corruption | `AssertReadinessIntegrityFailure` | Changes canonical without matching hash | Readiness integrity hydration | Rejection | Persistence/canonical integrity error | YES | COVERED |
| Wrong enablement version | `AssertLoadedReadableVersionBlock` | Rewrites canonical/hash and mirrors to version 2 | Readiness semantic validation | `ready=false` | `enablement_contract_version=2` | YES | COVERED |
| Enablement structural corruption | `AssertReadinessIntegrityFailure` | Raw canonical/hash or relationship corruption | Readiness integrity hydration | Rejection | Persistence/canonical integrity error | YES | COVERED |
| Admission wrong version | `AssertLoadedReadableVersionBlock` | Rewrites admission canonical/hash and Attempt mirrors | Readiness semantic validation | `ready=false` | `admission_contract_version=2` | YES | COVERED |
| Admission corruption | `AssertReadinessIntegrityFailure` | Appends canonical or falsifies hash | Readiness integrity hydration | Rejection | Persistence/canonical integrity error | YES | COVERED |
| Attempt V2 wrong version | `AssertLoadedReadableVersionBlock` | Rewrites Attempt V2 version and mirrors | Readiness semantic validation | `ready=false` | `production_attempt_contract_version=3` or `2|3` | YES | COVERED |
| Malformed one-sided Attempt rows | Readiness corruption loop | Nulls one relationship ID after dropping shape constraint | Readiness integrity hydration | Rejection | Persistence/canonical integrity error | YES | COVERED |
| Readable contradiction visibility | Readable version-block tests | Persists structurally valid but semantically wrong values | Readiness rendering | `ready=false` with blockers | Wrong persisted version remains visible | YES | COVERED |
| Read-only application/catalog snapshot | `ReadinessEvidenceJson` before/after command | Runs readiness command and compares persisted/catalog snapshots | Read-only proof | No changes/errors | Complete persisted/catalog evidence | YES | COVERED |
| Recursive role-path catalog snapshot | `AssertRoleHelperCatalogSnapshotDependency` | Adds relevant nested role membership | Catalog snapshot proof | Snapshot changes and includes edge | Recursive role closure evidence | YES | COVERED |
| Unrelated role-edge negative control | No test found | No unrelated role edge is added and proven irrelevant | Catalog snapshot negative control | Expected unchanged result | No false dependency | NO | GAP |
| Relevant Completion constraint mutation | Readiness constraint test | Drops relevant version constraint and adds replacement | Catalog snapshot/readiness proof | `ready=false` | `completion_nested_v2_proof_version=2` | YES | COVERED |
| Unrelated constraint negative control | No test found | No unrelated completion constraint mutation | Catalog snapshot negative control | Expected unchanged result | No false dependency | NO | GAP |

# 4. Protected-Function Catalog Coverage Assessment

The preflight test authentically covers:

- object kind;
- aggregate kind;
- volatility;
- parallel safety;
- leakproof;
- SECURITY DEFINER and SECURITY INVOKER;
- return type;
- OUT, INOUT, and TABLE metadata;
- identity arguments;
- overload and default-argument identity behavior.

The migration contract also freezes `language`, `provariadic`, and `proconfig`, as shown in [055_campaign_operations_production_admission_foundation.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:248>) and the corresponding validation logic at lines 786 onward.

Those three fields lack direct hostile mutation tests:

- no alternate-language mutation;
- no variadic mutation;
- no `proconfig` or search_path mutation.

`proisstrict` is not part of the accepted frozen contract, so strictness is not a coverage gap for this architecture.

# 5. Protected-Function ACL Coverage Assessment

ACL coverage is complete for the identified counterexamples.

The preflight cases independently exercise:

- PUBLIC execute;
- grant option;
- extra explicit grantee;
- missing explicit grant;
- missing expected role;
- recreated/missing expected role;
- NULL-versus-explicit ACL origin.

The helper verifies exact SQLSTATE, stable H1 diagnostic code, protected object identity, role identity where applicable, catalog preservation, and rollback. The migration harness additionally exercises production audit ACL-origin reconciliation.

# 6. Duplicate-Contract Drift Coverage Assessment

The anti-drift logic compares the complete current duplicated surface:

1. `protected_functions`;
2. `protected_function_signatures`;
3. `protected_function_contracts`;
4. `protected_function_return_contracts`;
5. `protected_function_identity_argument_contracts`;
6. `protected_function_final_extra_acl_contracts`.

It also compares both occurrences of the corresponding `allowed_*` arrays used by preflight and audit validation.

Each family is mutation-tested by changing a real literal and requiring the comparison to fail. The current migration contains no additional duplicated protected-function family outside this six-family set; the legacy ACL arrays and `new_h1_functions` are one-sided declarations rather than duplicated contract families.

This is an authentic source-contract mutation test, not a grep-only assertion. It does not, however, mutate the runtime migration validator itself; the proof is anti-drift equality rather than end-to-end semantic mutation.

Disposition: covered.

# 7. First-Attempt / Replay Coverage Assessment

The replay corpus is strong and uses authentic persisted PostgreSQL state.

The SQL helper:

- proves the hostile mutation occurred;
- invokes the actual replay path;
- expects exact `23514`;
- expects `production acquisition replay evidence corrupt`;
- verifies Attempt #2 was not repaired or partially changed;
- rolls back and proves restoration;
- validates successful replay before and after corruption where applicable.

Covered corruption includes canonical/hash mirrors, admission and enablement evidence, build canonical evidence, operation key, ordinal, request versions, acquisition audit presence/uniqueness, actor, capability, versions, and audit relationship mirrors.

Cross-principal exact recovery replay is also directly covered.

Remaining gaps are:

- first-Attempt admission relationship ID;
- first-Attempt enablement relationship ID;
- first-Attempt approved-build hash mirror;
- adversarial duplicate first-Attempt row.

The repository readiness tests that null relationship IDs do not subsume these gaps because they exercise readiness integrity hydration, not first-Attempt replay after a valid Attempt #1 has been established.

# 8. Readiness Regression-Coverage Assessment

Covered authentically:

- semantic wrong Manager-build version;
- Manager-build structural corruption;
- semantic wrong enablement version;
- enablement structural corruption;
- admission wrong version and corruption;
- Attempt V2 wrong version and corruption;
- malformed one-sided Attempt relationships;
- readable contradiction visibility;
- relevant Completion-constraint mutation;
- recursive relevant role-path snapshot;
- read-only persisted application/catalog snapshot.

Missing:

- unrelated role-edge negative control;
- unrelated Completion-constraint negative control.

The relevant-edge and relevant-constraint tests prove positive dependency, but do not prove that unrelated catalog edges or constraints are excluded.

# 9. Test Authenticity Assessment

The executed tests are generally non-vacuous:

- PostgreSQL state is created in disposable clusters.
- Mutations are applied to persisted rows or catalog objects.
- Mutation helpers verify the changed state.
- Exact SQLSTATEs and stable H1 diagnostic codes are checked for preflight/replay failures.
- Rollback and restoration are verified.
- Replay tests verify no repair or partial mutation.
- Readiness tests load through repository/application paths rather than relying only on in-memory fabricated objects.

One qualification: readiness integrity helpers accept a set of corruption-related error codes rather than asserting one exact stable diagnostic. This is weaker than the preflight and replay assertions, although it does not invalidate the underlying coverage.

# 10. Fixture / Traceability Registration Assessment

The migration-harness fixtures are registered consistently, including:

- overload and default-variant deployment fixtures;
- replay exact/conflict/corrupt fixtures;
- repository hydration fixtures;
- ACL-origin fixtures;
- runtime records, report entries, traceability, and evidence obligations.

The standalone [CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh:1>) cases are not separately present in the H1 reference graph. They appear to be ad hoc executable regression tests rather than evidence-graph generators, so I found no authoritative basis to require separate registration. Their lack of registration does not cure the missing mutations listed above.

# 11. Commands Executed

Read-only or disposable execution included:

- `git status --short`
- `git diff --check`
- `git diff --cached --check`
- process inspection for `LSTM_Release`, scheduler, training, inference, and analysis workers
- focused `rg`, `sed`, and `nl` source inspection
- `Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`
- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`

No source, SQL, test, manifest, script, documentation, generated evidence, index, staging area, or commit was modified.

# 12. Tests Executed and Exact Results

`Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`

- Exit: `0`
- Result: all listed preflight, ACL, rollback, and anti-drift cases passed.
- Summary: `duplicate_contract_families=ALL mutation_detection=PASS acl_tuple_components=PASS`

`Tests/CampaignOperationsPhaseH1MigrationTests.sh`

- Exit: `0`
- Migration tests passed.
- Lock-order tests passed.
- Repository/service tests passed.
- Phase 1–5 regression passed.
- ACL-origin production-audit tests passed.
- Trusted evidence generation passed.
- Traceability reconciliation passed.
- Reference graph semantic validation passed.

The harness reported partial final-artifact status because it was run with partial-final mode and did not produce the comprehensive final assurance bundle. That is not treated as a Step #4 coverage result.

# 13. Deferred Checks and Exact Reason

Shared Release/Xcode build and executable verification were deferred because active workers were detected:

- active `LSTM_Release` training processes;
- active scheduler processes.

Running or relinking the shared Release executable could interfere with active training/scheduler work. No production scheduler or experiment state was touched.

The requested final comprehensive H1 independent reverification was also not performed; this pass remained limited to targeted-test coverage.

# 14. Files Reviewed / Worktree Assessment

Primary files reviewed:

- [AGENTS.md](</Volumes/Developer SSD/ExpertAdvisor/AGENTS.md>)
- [CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh:19>)
- [CampaignOperationsPhaseH1MigrationTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:1>)
- [CampaignOperationsPhaseH1MigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1761>)
- [CampaignOperationsPhaseH1RepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:154>)
- [CampaignOperationsPhaseH1Tests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1Tests.cpp:1>)
- [055_campaign_operations_production_admission_foundation.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:248>)
- H1 fixture, requirement, runtime, report, traceability, and artifact registry files.

Worktree:

- The repository contains a large staged H1 baseline.
- No unstaged changes were introduced by this pass.
- `git diff --check`: passed.
- `git diff --cached --check`: passed.
- `git diff --stat`: no unstaged diff.
- `git diff --cached --stat`: `129 files changed, 34846 insertions(+), 20 deletions(-)`.

# 15. Remaining Coverage Gaps Ordered by Severity

High:

1. Missing direct mutation tests for frozen protected-function `language`, `provariadic`, and `proconfig/search_path`.
2. Missing first-Attempt replay tests for typed admission/enablement relationship IDs and approved-build hash.
3. Missing readiness negative controls for unrelated recursive role edges and unrelated completion constraints.

Medium:

4. No adversarial duplicate first-Attempt persisted row/replay test.
5. Readiness integrity tests do not consistently require one exact stable diagnostic.
6. Standalone preflight cases are not graph-registered; this is only a concern if the accepted evidence architecture later requires every ad hoc regression script to appear in the reference graph.

# 16. Final Disposition

`TARGETED_TEST_COVERAGE_REMAINING_GAPS_FOUND`