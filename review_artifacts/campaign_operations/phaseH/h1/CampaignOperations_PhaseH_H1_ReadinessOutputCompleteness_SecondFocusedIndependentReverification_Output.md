---
title: "Campaign Operations Phase H H1 Readiness Output Completeness Second Focused Independent Reverification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_ReadinessOutputCompleteness_SecondFocusedIndependentReverification_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Campaign Operations Phase H H1 Readiness Output Completeness Second Focused Independent Reverification

## 1. Executive Summary

Correction #3 is not independently closed. The new fields and tests improve version visibility, but readiness can still hide readable persisted evidence and bypass the integrity boundary for Manager-build, admission, and Attempt V2 evidence.

## 2. Independent Verdict

`READINESS_OUTPUT_COMPLETENESS_REMAINING_DEFECTS_FOUND`

## 3. Seven-Finding Closure Matrix

| Prior finding | Verdict | Evidence |
|---|---|---|
| 1. Authoritative version sourcing | REMAINS | Manager-build mismatch suppresses hydration and renders persisted service/canonical/hash as `none`. Admission/Attempt versions are read raw without evidence hydration. |
| 2. Repository test did not reach readiness | CLOSED | `Tests/CampaignOperationsPhaseH1MigrationTests.sh` completed its isolated suite, including the repository/service integration path. |
| 3. Read-only snapshot used nonexistent scheduler relation | REMAINS | Scheduler authority is correctly `experiment_scheduler_protocol`, but the before/after snapshot omits enablement-audit and completion-audit evidence. |
| 4. Contradictory evidence visibility unproven | REMAINS | The Manager-build “wrong version” test mutates canonical text without recomputing its hash; readiness suppresses hydration instead of failing at integrity. |
| 5. Empty/zero missing values | CLOSED | Nullable snapshot values render as `missing`, `none`, or `unavailable`; scheduler generation/cutover and missing enablement paths are exercised. |
| 6. Lease/reconciliation boundary | CLOSED | Accepted architecture assigns exact old-event expiry, Phase F eligibility, and reconciliation state to status; readiness carries aggregate counts. |
| 7. Release/CLI behavior unverified | OPERATIONALLY DEFERRED | Live scheduler and three live training workers make shared Release relinking/execution unsafe. |

## 4–7. Contract, Sourcing, Admission/Attempt, and Contradiction Assessment

The loaded path is:

`campaign_operations_production_readiness_v1` → `LoadProductionReadinessSnapshot` → `EvaluateProductionReadiness` → `RenderProductionReadiness`.

Scheduler version now comes from `experiment_scheduler_protocol.scheduler_evidence_contract_version`; migration SHA is correct; expected versions are separate C++ constants.

However, two material defects remain:

1. Manager-build mismatch bypasses integrity validation.
   The loader hydrates the enablement head only when both observed enablement and Manager-build versions equal the expected constants ([repository](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:574>)). A readable Manager-build mismatch therefore makes the renderer emit:

   - `manager_service_contract=none`
   - `enablement_canonical=none`
   - `enablement_hash=none`
   - `approved_build_canonical=none`
   - `approved_build_hash=none`

   despite persisted evidence being present. This is not complete observed-evidence output.

   Worse, the purported wrong-version test changes only `approved_build_contract_canonical` ([test](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:503>)); it does not recompute the stored hash. That is canonical/hash corruption. The mismatch gate prevents the existing hydrator from detecting it, contrary to the required preserved fail-closed integrity boundary.

2. Admission and Attempt V2 evidence are not hydrated or integrity-validated by readiness.
   The view aggregates raw typed fields from all admissions and attempts ([migration](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3939>)). The only V2 proof check is whether a V2 row has an admission ID ([migration](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:4017>)). It does not validate Attempt V2 canonical/hash/audit evidence, its enablement chain, or admission canonical/hash/audit evidence.

   The Attempt aggregate uses `admission_id IS NOT NULL OR enablement_event_id IS NOT NULL` ([migration](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3943>)). A malformed/non-production-shaped row with an admission but no enablement event can satisfy the version aggregate and the current completion proof predicate. This does not meet the exact production-Attempt scope.

## 8. Read-Only Integration Assessment

The command itself uses repeatable-read, read-only transactions. The repository test now snapshots the real scheduler table, not the former nonexistent relation.

But its persisted-state JSON excludes at least:

- `campaign_operations_production_enablement_audit_reference_event`
- completion audit-reference evidence

([test snapshot](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:39>)). Therefore its equality assertion cannot prove readiness left all correction-relevant evidence unchanged.

## 9. Migration / Checksum / Manifest Assessment

Closed.

- Actual migration SHA-256:
  `86a35844edd3cc233e8f72ff985c339474dc09d3cd79d354fcb3adeb902aa66f`
- Embedded C++ checksum matches.
- `Scripts/CampaignOperationsH1ManifestValidator.sh` passed.
- The manifest binds the ACL-manifest-set digest embedded in migration 055; it is not a whole-migration-byte manifest.

## 10. Missing-Value Representation Assessment

Closed for the exercised paths. Optional scheduler/version/proof values render explicitly as `missing`; absent hydrated evidence uses `none`; unavailable actual builds use `unavailable`. No zero/empty default remains for these snapshot fields.

## 11. Readiness / Status Detail Boundary Assessment

Closed. The authoritative architecture explicitly requires Status to expose exact expiry, Phase F recovery eligibility, and reconciliation state. The status renderer emits those per-request fields; readiness emits aggregate current/old lease and reconciliation counts.

## 12. Complete Output Contract Assessment

The normal renderer includes the requested field families, including expected-version fields, identities, roles, counts, proof status, and blockers.

It is incomplete under a readable Manager-build or enablement version contradiction because mismatch gating substitutes `none` for persisted enablement/build details. Thus the contract is not complete in the adversarial state it is intended to diagnose.

## 13. Test Authenticity / Counterexamples

Passed loaded-path cases in the repository integration test include wrong scheduler, Manager-build, enablement, admission, Attempt version, Completion version, missing scheduler generation/cutover, missing enablement, and one enablement-audit integrity failure.

But the test does not establish:

- mixed admission versions;
- mixed Attempt V2 versions;
- missing Attempt V2 evidence as a distinct loaded-path case;
- loaded-readiness rejection of admission/Attempt canonical/hash/audit corruption;
- correct exclusion of malformed rows that have only one production linkage;
- complete read-only preservation of enablement/completion audit evidence.

The existing direct hydrator corruption tests do not substitute for readiness hydration; readiness does not invoke those hydrators for admissions or Attempt V2 rows.

## 14. Regression Assessment

No regression found in the focused protected-function preflight, migration, scheduler-generation-52, recovery/replay, or H1 default-off suites exercised. The newly found defects are localized to readiness loading/rendering and its test coverage.

## 15–16. Commands and Test Results

- `git status --short` — broad staged H1 baseline plus nine unstaged Correction #3 files.
- `git diff --check` / cached check — known unrelated staged fixture trailing-whitespace findings; Correction #3-only diff check clean.
- `shasum -a 256 Database/migrations/055_campaign_operations_production_admission_foundation.sql` — matched claimed SHA.
- `Scripts/CampaignOperationsH1ManifestValidator.sh` — PASS.
- `Tests/CampaignOperationsPhaseH1MigrationTests.sh` — PASS on an isolated disposable PostgreSQL cluster.
- `Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh` — PASS.
- Strict compile: `clang++ -std=c++20 -Wall -Wextra -Werror -fsyntax-only …` for corrected readiness sources/tests — PASS.

## 17. Deferred Checks

Release build and CLI execution are deferred solely for worker safety. Active processes included:

- Scheduler PID 41599.
- Training workers PIDs 41106, 76855, and 79193.

No production process was stopped, signaled, relinked, or otherwise modified.

## 18. Files Reviewed / Worktree Assessment

Correction #3 files reviewed:

- `Database/migrations/055_campaign_operations_production_admission_foundation.sql`
- `Sources/CampaignOperationsProductionAdmission.hpp`
- `Sources/CampaignOperationsProductionAdmissionRepository.{hpp,cpp}`
- `Sources/CampaignOperationsProductionAdmissionService.{hpp,cpp}`
- `Tests/CampaignOperationsPhaseH1RepositoryTests.cpp`
- `Tests/CampaignOperationsPhaseH1Tests.cpp`
- `docs/CampaignOperationsPhaseH1.rst`

No repository files, index entries, or staging were modified.

`git diff --stat` for these unstaged correction files: 9 files, 602 insertions, 151 deletions.

## 19. Remaining Findings

1. **HIGH — readiness bypasses structural integrity validation for Manager-build mismatches and for all admission/Attempt V2 evidence.**
2. **HIGH — malformed Attempt rows can satisfy readiness’s production-version aggregate because its scope predicate uses `OR` and proof validation is incomplete.**
3. **MEDIUM — readable version contradiction hides required persisted Manager/enablement output by rendering `none`.**
4. **MEDIUM — read-only integration snapshot omits enablement and completion audit evidence.**
5. **LOW — Release/CLI verification remains deferred due active-worker safety.**

## 20. Final Disposition

`READINESS_OUTPUT_COMPLETENESS_REMAINING_DEFECTS_FOUND`