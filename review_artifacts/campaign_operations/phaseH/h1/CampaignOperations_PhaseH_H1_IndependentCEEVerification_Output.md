---
title: "Campaign Operations Phase H H1 Independent CEE Verification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_IndependentCEEVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H H1 Independent CEE Verification

# Independent CEE Verification — Phase H H1

## 1. Executive verdict

**H1 is not ready to commit.**

The primary blocker is architectural: frozen Phase H authority assigns three fixed transition functions to migration 055/H1, but the implementation omits them and edits the normative architecture documents to reassign them to H2 without separate accepted authority.

Additional high-severity C++/PostgreSQL equivalence and test-coverage defects remain.

## 2. Implementation artifacts and repository baseline

The actual artifacts were located without assuming fixed names:

- [Implementation output](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseH_H1_Implementation_Output.md>)
- `CampaignOperations_PhaseH_H1_Implementation_Transcript.txt` — ignored, 40 MB
- `CampaignOperations_PhaseH_H1_CEE_Verification_Prompt.txt` — ignored

The implementation output and transcript were treated as evidence of implementation activity, not architecture authorization.

Baseline:

- Branch: `campaign-operations`
- HEAD: `fec53d5 Add ADR-0019 and Phase H production dispatch architecture`
- No repository files were changed by this verification.
- The H1 implementation comprises 19 tracked/intended file changes plus the untracked implementation output.

## 3. Authoritative architecture

[ADR-0019](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md:39>) makes the Phase H document and H1–H4 boundaries normative. It states that H1 creates authority, persistence, and readiness without production handoff, while H2 adds dispatch services and the canary path ([line 166](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md:166>)).

Repository history shows only the original acceptance commit for the normative Phase H document. No later accepted ADR or architecture change authorizing the transition-function reassignment was found.

## 4. Required H1/H2 boundary

The frozen version of §17 explicitly authorized migration 055 to add:

- `record_campaign_operations_production_enable_v1`
- `record_campaign_operations_production_disable_v1`
- `transition_campaign_operations_request_dispatching_production_v2`

The latter was defined as the only function allowed to establish admission, Boolean true, Attempt V2, and production audit atomically.

The frozen increment boundary assigns:

- H1: migration 055, evidence, Attempt V1/V2 compatibility, guards, roles, readiness/status.
- H2: enable/disable **services**, the common engine, production adapter/CLI, and concurrency/recovery behavior.

Therefore, the functions were required to exist in H1. Their service/CLI reachability and production grants belong later.

## 5. Migration 055 behavioral change

The implementation adds:

- Immutable enablement and admission evidence.
- Attempt V2 schema and validation.
- Production audit relationships.
- NOLOGIN production roles.
- Scheduler evidence functions.
- Readiness and status views.
- Read-only CLI status/readiness paths.

It intentionally omits all three fixed transition functions. That omission conflicts with the frozen migration inventory.

## 6. Boolean/admission transition verification

The required condition is **not satisfied**.

Migration 055 currently establishes Boolean true from an admission trigger:

- Admission insert invokes the witness trigger at [migration 055:1246](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1246>).
- The trigger function directly sets `production_dispatch_enabled = true` at [line 945](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:945>).
- Its update guard authorizes the change using only `pg_trigger_depth() == 2` at [line 986](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:986>).

The isolated migration test constructs enablement, admission, Attempt V2, and audit records through privileged direct DML and commits Boolean true. Consequently:

> False → true is possible outside the architecturally required fixed guarded production transition.

The implementation does correctly withhold H1 service, CLI, role, and ordinary-caller access. That does not satisfy the owner-DML-safe fixed-transition requirement.

## 7. May the functions exist inertly in H1?

Yes.

They may exist in migration 055 while remaining:

- Revoked from `PUBLIC`.
- Ungranted to all H1 capabilities.
- Unreferenced by H1 services and CLI.
- Unreachable by ordinary callers.
- Incapable of establishing production state through any authorized H1 path.

Their presence supplies the frozen database boundary without implementing H2’s enablement or production-dispatch surface.

Deferring them to H2 is:

- Not fully compliant.
- Not an authorized implementation interpretation.
- An unauthorized architecture change.
- A blocker requiring correction or an independently accepted architecture amendment.

Thus classifications **(c)** and **(d)** apply.

## 8. Canonical operation-key defect

**High severity — C++/PostgreSQL contract mismatch.**

[ValidOperationKey](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmission.cpp:28>) permits `.`, `_`, `:`, `/`, or `-` as the first character. PostgreSQL requires the first character to be alphanumeric.

C++ can therefore construct canonical objects that PostgreSQL rejects. Existing tests cover spaces but not leading punctuation.

## 9. Attempt V2 nested-invariant defect

**High severity.**

[ValidateProductionDispatchAttemptV2](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmission.cpp:493>) does not verify:

- `attempt.operationKey == admission.dispatchOperationKey`
- `attempt.expectedRequestVersion == admission.expectedRequestVersion`

PostgreSQL enforces both at [migration 055:930](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:930>).

This permits C++ values that are internally inconsistent and cannot be persisted.

## 10. Repository hydration integrity

**Medium severity.**

[FindProductionDispatchAttemptV2](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:225>) does not load and independently validate all persisted nested canonical/hash/typed-mirror fields. It reconstructs several values from the separately loaded admission, which can mask corruption in the attempt row’s duplicated evidence.

This falls short of the frozen requirement that every persisted load validate the stored canonical, hash, typed mirrors, and nested evidence.

## 11. Privilege and inertness review

Positive findings:

- Production roles are NOLOGIN.
- No login membership is granted.
- No H1 service or CLI exposes enable, disable, or production acquisition.
- Production roles lack ordinary insert authority over the relevant evidence tables.
- Direct standalone Boolean updates are rejected.
- Immutable update/delete/truncate guards are present.

However, the current trigger-depth-only owner path is not equivalent to the frozen fixed `SECURITY DEFINER` transition plus guarded transaction context.

## 12. Readiness/status accuracy

**Medium severity.**

The H1 status view marks Phase F recovery eligible when a request is dispatching and the lease has expired ([migration 055:1624](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:1624>)).

The authoritative Phase F recovery transition additionally rejects requests having:

- A request binding.
- A downstream conversion execution.
- An outcome for the latest attempt.

Those predicates are visible in [migration 053:1383](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql:1383>). The H1 view can therefore report a request as recoverable when Phase F will reject recovery.

## 13. Service, CLI, and scheduler isolation

The H1 service and CLI remain read-only for production admission. No production handoff, Manager batch, or continuous-mode behavior was found.

Before testing, the active scheduler was inspected. It had managed workers and queued work, so live process-interfering suites were not launched. The Release build used an isolated DerivedData directory to avoid replacing the executable used by the active scheduler.

## 14. Test sufficiency

**High severity — mandatory coverage incomplete.**

Current tests do not adequately cover:

- Same-hash/different-canonical rejection at every nested evidence level.
- Complete Completion V1 nesting of Attempt V2.
- Malformed/incomplete V2 permutations.
- Post-completion admission, Boolean, Attempt V2, and audit rejection.
- Historical V1 byte preservation after migration 055.
- Positive isolated V1 operation under the permitted test authority.
- Cross-language UTF-8, delimiter, long-value, signed-boundary, and leading-zero vectors.
- Leading-punctuation operation keys.
- The missing transition-function inventory, ACLs, and H1 inertness.
- Recovery-status negative predicates.

The migration test passes, but it validates a privileged direct-DML construction that is contrary to the frozen fixed-transition design.

## 15. Architecture-authority integrity

Edits were classified as follows:

| Edit | Classification | Authority |
|---|---|---|
| “Migration 055 is implemented by H1” | Implementation-status documentation | Acceptable |
| “The following inventory remains accepted” | Clarification | Acceptable alone |
| “Mutation transition functions…are reserved to H2” | Normative architecture change | No independent acceptance found |
| Volume XII implementation/version entry | Implementation-status documentation | Acceptable |
| Removal of fixed transition functions from Volume XII’s frozen inventory | Normative architecture change | No independent acceptance found |

The relevant unauthorized edit appears at [Phase H architecture:643](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md:643>).

Neither the implementation report nor transcript authorizes this change.

## 16. Verification commands and results

Passed:

- Scheduler/process inspection and read-only scheduler status.
- Migration SHA-256 agreement with the compiled checksum.
- `git diff --check`
- `git diff --cached --check`
- Shell syntax checks.
- Xcode project plist validation.
- H1 C++ unit executable.
- Legacy Campaign Operations C++ regression executable.
- H1 CLI tests.
- Phase 2, Phase 4, and Phase 5 CLI tests.
- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- Release `xcodebuild` using `/tmp/ea-h1-xcodebuild.NZjQLK`
- Result: `** BUILD SUCCEEDED **`

Failed:

- Strict H1 compilation with `-Wall -Wextra -Werror`.
- New warning at [CampaignOperationsProductionAdmissionRepository.cpp:362](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:362>): missing aggregate field initializers.
- The Xcode project suppresses this warning, but repository policy treats compiler warnings as defects.

Skipped regression classification:

- Live scheduler/global-control/process suites: **pre-enablement blocker**, acceptable deferred operational evidence for the H1 commit itself.
- H2 enable/disable/acquisition/in-doubt/concurrency suites: **H2 prerequisite**.
- H3 Manager run-once and multi-manager suites: **H2/H3 prerequisite**.
- No skipped live-process suite was automatically treated as an H1 code blocker.

## 17. Required corrections and repository state

Required before H1 acceptance:

1. Restore the three fixed transition functions to migration 055 with frozen ownership, search-path, ACL, and guarded-context behavior.
2. Keep them ungranted and unreachable from H1 services/CLI.
3. Remove or separately authorize the normative H1→H2 documentation reassignment.
4. Correct operation-key validation.
5. Complete Attempt V2 nested validation and hydration checks.
6. Align recovery eligibility with the authoritative Phase F predicates.
7. Add the missing mandatory test matrix.
8. Fix the new strict-compilation warning.

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
```

`git diff --stat`:

```text
19 files changed, 4533 insertions(+), 9 deletions(-)
```

## 18. Architecture-authority integrity verdict

**The H1 implementation altered the frozen H1/H2 boundary without separate approval.**

Migration 055 was architecturally required to create the fixed enable, disable, and production-acquisition transition functions. Those functions could and should exist inertly in H1. Reassigning their creation to H2 is an unauthorized normative architecture change unless a new independently accepted ADR or equivalent architecture amendment approves it.

**Final disposition: `PHASE_H_H1_NOT_READY_TO_COMMIT`**