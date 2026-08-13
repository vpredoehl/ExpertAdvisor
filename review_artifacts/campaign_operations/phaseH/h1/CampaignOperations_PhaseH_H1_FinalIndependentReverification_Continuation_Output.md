---
title: "Campaign Operations Phase H H1 Final Independent Reverification Continuation"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_FinalIndependentReverification_Continuation_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H H1 Final Independent Reverification Continuation

# Final independent H1 reverification report

## 1. Executive summary

The staged H1 candidate does not satisfy the accepted ADR-0019/0019A/0019B contract.

Four material deficiencies remain:

1. Migration 055 does not validate the complete protected-function tuple before DDL mutation.
2. C++ repository hydration and SQL replay do not fully reconstruct and validate the admission’s first Attempt V2.
3. Readiness evaluates contract versions but does not report them.
4. The staged tests do not exercise those missing branches and therefore allowed the defects to pass.

The staged index remained unchanged throughout this continuation.

## 2. Independent verdict

`IMPLEMENTATION_CORRECTION_REQUIRED`

## 3. Findings ordered by severity

### High — Protected-function preflight is not the frozen exact tuple

The preflight contract in [migration 055](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:613>) compares only:

- signature
- language
- `SECURITY DEFINER`
- volatility
- parallel mode
- default count
- variadic type
- `proconfig`

It does not preflight:

- `prokind`
- `proleakproof`
- return type
- OUT-argument identity/modes
- explicit ACL inventory
- PUBLIC execution
- grant-option state

The later post-DDL audit adds `prokind` and `proleakproof` checks at [line 4617](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:4617>), but that is too late. ACLs are actively normalized by `REVOKE` at [line 3705](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3705>).

Concrete counterexamples:

- An exact-name/exact-input function with an incompatible return type passes preflight and reaches `CREATE OR REPLACE`, producing PostgreSQL’s raw return-type error rather than an H1A diagnostic.
- A pre-existing incompatible function ACL is revoked and normalized instead of being rejected before mutation.
- A procedure or aggregate sharing the exact identity can evade the tuple’s missing `prokind` validation until PostgreSQL or a later audit rejects it.

This contradicts ADR-0019B’s frozen tuple at [lines 357–365](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md:357>).

Minimal correction: encode and validate the complete per-signature tuple, including return/OUT identity and exact ACL expansion, before any `CREATE OR REPLACE`, `ALTER`, `REVOKE`, or ownership operation. Add targeted mutation fixtures for every attribute and require the appropriate H1A code/SQLSTATE.

### High — First Attempt V2 remains partially hydrated and partially replay-validated

`PersistedRequestProductionAdmission` stores only `firstAttemptId` and its audit, not the hydrated first Attempt V2, at [repository header line 57](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.hpp:57>).

`FindRequestProductionAdmission` selects only the first attempt ID using a partial predicate at [repository line 360](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:360>). It does not validate that first attempt’s:

- canonical text/hash
- ordinal and resulting version
- admission canonical/hash
- enablement hash
- approved-build hash
- capability/contract version
- lease evidence

A later Attempt V2 can therefore be returned while its nested first Attempt V2 is corrupt.

The SQL replay helper has the same gap. Its first-attempt query at [migration line 2728](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:2728>) checks only a subset of typed fields and audit metadata. It does not reconstruct the first attempt canonical/hash.

Counterexample: corrupt Attempt #1’s canonical while retaining its hash and subset fields, then replay Attempt #2. The current Attempt #2 validates, the partial first-attempt count remains one, and replay succeeds. Thus same-hash/different-canonical nested evidence can be treated as an exact existing operation.

This directly contradicts the closed graph claimed at [Phase H architecture line 880](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:880>).

Minimal correction: return and validate the complete first Attempt V2 object, its current enablement/audit, its canonical/hash and every typed mirror. Apply equivalent validation in SQL replay.

### Medium — Readiness omits required deployment evidence

The snapshot contains all five canonical contract versions and the Completion proof version at [repository header line 75](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.hpp:75>). Evaluation correctly blocks mismatches at [service line 155](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionService.cpp:155>).

However, CLI output beginning at [service line 244](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionService.cpp:244>) omits:

- `scheduler_evidence_contract_version`
- `manager_build_contract_version`
- `enablement_contract_version`
- `admission_contract_version`
- `production_attempt_contract_version`
- `completion_nested_v2_proof_version`
- an explicit Manager service-contract field

Only the proof validity string is emitted. This contradicts the required reporting contract at [Phase H lines 612–631](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:612>).

Missing runtime-build evidence is correctly blocking without suppressing other output, and the command is correctly read-only/repeatable-read.

### Medium — Material test coverage gaps

The staged tests do not cover the defects above:

- Protected-function tests cover ordinary ownership and default arguments only at [migration test line 407](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:407>).
- Repository mutation tests corrupt `dispatch-001`, which is also the only/first attempt, at [repository test line 146](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:146>). There is no second-attempt hydration test that corrupts the nested first attempt.
- Readiness assertions at [repository test line 269](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:269>) do not require any contract-version output.
- Detailed recursive edge assertions exist for the three-hop ADMIN OPTION scenario at [migration test line 840](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:840>), but direct, two-hop, and reverse fixtures only prove the generic H1A003 preflight result.

The retained evidence graph also maps numerous requirements to the same whole-suite migration transcript digest. That transcript proves the suite ran, but it is not independent per-branch proof.

### Low — Cached diff check fails

`git diff --cached --check` exited 2 with 287 trailing-whitespace findings, all in [CampaignOperationsH1Artifacts.tsv](</Volumes/Developer SSD/ExpertAdvisor/Tests/fixtures/CampaignOperationsH1Artifacts.tsv:12>). They are trailing empty TSV fields, but the required check is not clean.

## 4. Verification of the ten corrections

### 1. Post-recovery reacquisition — PASS for lifecycle mechanics

Implementation evidence:

- One admission per request: unique constraint at [migration line 1060](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1060>).
- One Attempt V2 per request/operation key: unique partial index at [line 1201](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1201>).
- Replay occurs before mutable-state gating at [line 3078](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3078>).
- Replay is repeated after authoritative lock acquisition at [line 3127](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3127>).
- Existing admission is reused at [line 3156](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3156>).
- Ordinal is `max + 1` under the locked request at [line 3146](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3146>).
- Request and Attempt versions advance at [lines 3201 and 3223](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3201>).

The staged lifecycle fixture at [migration SQL line 1471](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1471>) verifies `v3 → v4 → v5 → v6`, one admission, ordinal two, and two acquisition audits.

Concurrent insertions are serialized by the request lock and ultimately protected by the admission and operation-key unique constraints.

The incomplete nested replay validation is separately classified under corrections 3 and 8.

### 2. Cross-principal recovery — PASS

The replay helper loads stored admission, enablement, principal, and build at [migration line 2647](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:2647>). Existing replay compares caller-controlled version, lease, key, and actor, but deliberately does not require the caller’s current principal/build to equal historical values.

The cross-principal fixture at [migration SQL line 1439](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1439>) changes session authorization and supplies a false current build, then verifies the historical principal/build and attempt ID remain unchanged.

Caller-supplied build data cannot rewrite the stored replay result.

Test limitation: the fixture uses a disposable superuser because H1 intentionally grants no production mutation capability. It proves identity separation, not a future H2 authorization path.

### 3. Complete repository hydration — FAIL

Current and first enablement/audit hydration is substantive, but the first Attempt V2 is represented only by an ID and partial predicate. Missing or corrupt fields outside that predicate are not rejected.

Negative tests exercise many current-attempt and admission corruptions, but never hydrate Attempt #2 while independently corrupting Attempt #1.

### 4. Protected-function preflight — FAIL

Schema/signature/overload/default/owner/language/security/volatility/parallel/search-path checks exist, but the complete frozen tuple is absent before DDL. ACL normalization and raw non-H1A failures remain possible.

### 5. Readiness reporting — FAIL

Fail-closed evaluation, blockers, canonical/hash evidence, roles, principals, counts, actual-versus-approved build, and read-only behavior are implemented. Required contract and proof-version fields are not emitted.

### 6. Stable H1A diagnostics — FAIL

Explicit staged migration/deployment-audit diagnostics use H1A001–H1A011, and no stale H1M namespace was found on supported paths.

Nevertheless, incompatible return types or object kinds missed by preflight can escape through PostgreSQL-native errors, while incompatible ACLs can be normalized. Therefore the supported failure surface is not exclusively H1A001–H1A011.

### 7. Role-path evidence — PASS for implementation

The recursive audit at [deployment audit line 55](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1DeploymentAudit.sh:55>) reports:

- direction
- complete role sequence
- shortest path
- total depth
- every edge
- edge order/depth
- ADMIN OPTION per edge

`DISTINCT ON` ordered by depth and role sequence supplies deterministic shortest-path selection at [line 114](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1DeploymentAudit.sh:114>).

Fixtures include direct, two-hop with NOLOGIN intermediate, three-hop, ADMIN OPTION, and reverse direction at [migration test line 553](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:553>). The three-hop fixture proves complete edge output and actual `SET ROLE`/`SET LOCAL ROLE`.

### 8. Replay consistency — FAIL

Positive aspects:

- Exact replay is resolved before mutable request state.
- Current attempt/admission/enablement canonicals and hashes are reconstructed.
- Same-hash/different-canonical tests exist for the current admission and attempt at [migration SQL line 1083](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1083>).
- Historical schema-054 byte preservation and restore evidence were present in the retained disposable run.

Failure: replay of a later attempt does not reconstruct the first Attempt V2. Corrupting that first attempt’s canonical while replaying a later attempt can still return `existing_identical`.

### 9. Architecture consistency — FAIL

H1 remains default-off:

- No production LOGIN role is granted.
- Mutation and replay functions are revoked from ordinary H1 roles at [migration line 3488](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3488>).
- The C++ executable exposes only read-only readiness/status H1 commands.
- No H2/H3/H4 Manager loop, lifecycle transition caller, scheduler mutation, or continuous-production loop was found.

However, executable behavior contradicts accepted architecture regarding the protected tuple, complete hydration closure, and readiness reporting. Those disagreements are architectural findings.

### 10. Test quality — FAIL

The tests are not generally vacuous: SQL lifecycle/replay assertions, role reachability, canonical mutation, manifest validation, and C++ builders execute real logic.

They are materially inadequate for final acceptance because the exact counterexamples underlying findings 1–3 are untested. The evidence graph’s repeated use of a shared whole-suite transcript also cannot substitute for targeted branch evidence.

## 5. Remaining architectural risks

Beyond the verified defects:

- The post-recovery test invokes the real Phase F transition but fabricates unrelated recovery witness rows under replica mode at [migration SQL line 1492](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1492>). End-to-end Phase F service integration therefore remains unverified.
- Cross-principal behavior is tested using a superuser rather than a future ordinary authorized Manager. H2 authorization is out of H1 scope, but the eventual authorization integration remains unproven.
- No fresh restore A–J run was safe during this continuation; retained evidence was inspected instead.

## 6. Test assessment

Tests executed in this continuation:

- `Scripts/CampaignOperationsH1ManifestValidator.sh`
  - PASS
  - Manifest digest: `ec6e34b1dd3ee68bf5315f193baee233222822eb045437d154d4b29e9444fb9f`
  - 92 inventory rows, 70 explicit ACL rows, 27 default ACL states, 7 column ACL rows.
- `bash -n` on deployment audit, manifest validator, and migration harness
  - PASS
- Standalone H1 C++ compilation with C++20, `-Wall -Wextra -Werror`, followed by execution of `CampaignOperationsPhaseH1Tests`
  - PASS
- Eight staged Python assurance modules
  - 89 tests, PASS
- `git diff --cached --check`
  - FAIL: 287 trailing-whitespace findings.

Independently inspected retained evidence:

- Run ID: `h1-20260804T045405Z-90836`
- Migration/replay transcript reached “Campaign Operations Phase H1 migration tests passed.”
- Restore A–J, historical-byte, role scenario E, checksum, and graph receipts were present.
- Retained migration checksum matches the current staged migration.
- The shared transcript/digest pattern was identified and was not treated as independent proof for each claimed branch.

Not executed:

- Full `Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- Repository PostgreSQL integration test
- Restore workflow
- The prescribed Xcode build

Reason: an active scheduler and seven active training workers were confirmed. Rebuilding the shared Release product or launching database/executable integration workflows risked changing binaries used for new scheduler work or interfering with live activity. No scheduler, worker, experiment, or production database was altered.

## 7. Evidence supporting the verdict

Staged candidate identity:

- Cached diff SHA-256: `e12d59a6adcbf6582b06fdc98913ee73796ef431d292313de43bebbd9752a449`
- `git diff --cached --stat`: `113 files changed, 29337 insertions(+), 20 deletions(-)`
- `git diff --cached --check`: exit 2; 287 trailing-whitespace findings.
- Tracked unstaged files: zero.

Migration relationship:

- Migration SHA-256: `6d2a37552e95454083ce643e5b3dc2f7d2085cb1298cddbc8c5f0cf54e028056`
- Embedded C++ checksum at [header line 20](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmission.hpp:20>): identical.
- Retained checksum evidence: identical.
- Manifest aggregate digest: `ec6e34b1dd3ee68bf5315f193baee233222822eb045437d154d4b29e9444fb9f`.

## 8. Worktree integrity

The initial and final cached-diff hashes were identical:

`e12d59a6adcbf6582b06fdc98913ee73796ef431d292313de43bebbd9752a449`

No staged file was modified, unstaged, reset, or restaged during this review. No implementation source exists only as an unstaged or untracked file.

Untracked files consist of:

- prior review/output Markdown
- `.h1-five-finding-review.*`
- `Scripts/__pycache__/`
- `Tests/__pycache__/`

The evidence-report filenames referenced by staged evidence tooling are generated artifacts under an artifact root; they are not imported source dependencies and were correctly not staged.

Files changed by this review: none.

`git diff --stat` for tracked unstaged changes is empty. `git status --short` consists of the 113 staged candidate files plus the harmless untracked categories above.

## 9. Final disposition

IMPLEMENTATION_CORRECTION_REQUIRED