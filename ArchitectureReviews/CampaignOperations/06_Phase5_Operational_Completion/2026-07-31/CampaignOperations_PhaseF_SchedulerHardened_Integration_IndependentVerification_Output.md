---
title: "Campaign Operations Phase F Scheduler-Hardened Integration and Independent Verification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseF_SchedulerHardened_Integration_IndependentVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase F Scheduler-Hardened Integration and Independent Verification

## 1. Decision and scope

Accepted Campaign Operations Phase F was integrated into `campaign-operations` at HEAD `eae2fa83c447727fdd5789caabc5286f9cfe63e3`.

Source material:

- Branch: `github/phase4-cancellation-reconciliation`
- Implementation: `a0f9670480bfa226c28592c992d03e2bdc6ea8bb`
- Verified tip: `5db4c2003161e61d718d286094a3b9983158c953`
- Common ancestor: `eb11084cfe66c658b6ad97471f3292a11db896a1`

The verified tip contains the implementation commit plus archived correction and independent-verification artifacts. All were ported. Phase G, completion events, audit views, `complete-if-settled`, and migration `054` were not implemented.

## 2. Files changed

- Migration: [053 campaign controls/cancellation/reconciliation](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql>).
- Sources: six new `CampaignOperationsControl*` files; accepted modifications to `CampaignOperations`, dispatch repository/service, and service.
- CLI: accepted Phase F parsing, validation, help, and routing added to `Sources/ExperimentScheduler.cpp`.
- Xcode: all three new `.cpp` files and headers registered for both Release/Debug targets; project file passes `plutil`.
- Tests: new Phase 4 CLI and migration tests; accepted updates to Campaign Operations domain/repository/launch tests and Phase 2 CLI regression.
- Documentation: Database README, Phase 3/4 operator docs, architecture index/volumes, ADR-0015, and ADR-0017.
- Review artifacts: all eight accepted implementation, correction, integration, and independent-verification records under `ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/`.

The pre-existing untracked `CampaignOperations_Phase5_ArchitecturalPhaseG_Implementation_Output.md` was not modified.

## 3. Migration integration

The accepted migration was mechanically renumbered:

```text
049_campaign_operations_controls_cancellation_reconciliation.sql
→ 053_campaign_operations_controls_cancellation_reconciliation.sql
```

A normalized comparison against `5db4c20` returned `MATCH`; only the filename and embedded migration-number diagnostics changed.

Migrations `049`–`052` are byte-unchanged from HEAD:

```text
049_global_pause_selective_resume.sql
050_experiment_current_operation_canonicalization.sql
051_scheduler_ownership_and_worker_attempts.sql
052_scheduler_protocol_and_exact_attempt_hardening.sql
053_campaign_operations_controls_cancellation_reconciliation.sql
```

Upgrade verification used a disposable database restored from the authoritative backup, advanced to exactly `052`, then ran:

```bash
LSTM_DB_NAME=expertadvisor_phasef053_upgrade ./migrate_lstm_db.sh
```

Result:

```text
MIGRATION_APPLY,version=053,filename=053_campaign_operations_controls_cancellation_reconciliation.sql
MIGRATION_DONE,applied=1,skipped=48
```

A second run reported `applied=0,skipped=49`. The recorded `049`–`052` filenames remained unchanged. Focused catalog, trigger, constraint, role, ACL, append-only, and denial assertions passed afterward.

The supported fresh path was also covered by:

- a newly created database restored from the authoritative base and migrated through `053`;
- the synthetic Campaign Operations base fixture applying `045`, `047`, `048`, and `053`.

A completely empty PostgreSQL database is not a supported repository starting point: historical migration `012` expects the pre-migration foundational `model` schema.

## 4. Conflict resolution

Mechanical conflicts:

- Renumbered every active Phase F migration reference to `053`.
- Updated migration tests, CLI tests, documentation, review artifacts, and ordering descriptions.
- Merged Xcode file references and build-phase membership.
- Preserved historical global-control `049` references where they correctly identify `049_global_pause_selective_resume.sql`.

Semantic conflicts:

- `ExperimentScheduler.cpp`: retained the scheduler-hardened generation-52 ownership, exact-attempt, canonical-operation, and finalization implementation. Only Phase F CLI parsing/routing was added.
- `GlobalExperimentControlIntegrationTests.sh`: retained the current scheduler-hardened worker identity fixture. Its final content is unchanged from HEAD.
- `Database/README.md`: retained `049`–`052` as scheduler/global-control history and documented `053` as dependent but authority-separate.
- Canonical `current_operation` remains `train`, `infer`, and `analyze`; no Phase F control label was introduced into that vocabulary.

No scheduler implementation conflict was resolved in favor of the older Phase F branch.

## 5. Preserved Phase F semantics

Verification confirmed:

- Pause/resume are append-only, versioned Campaign Operations events and do not pause schedulers or signal workers.
- Cancellation request and settlement are immutable separate facts.
- Exact retries return persisted facts; changed payloads conflict.
- Unbound held cancellation follows budget → campaign → reservation → request locking and releases only held units.
- Committed reservations are never refunded.
- Bound lifecycle cancellation runs in separate transactions after Campaign Operations locks are released.
- Running lifecycle work returns `running_not_supported`; Campaign Operations receives no process-control authority.
- Reconciliation observation is detection-only; recovery uses separate owning-service evidence and role.
- Cursor scans and transaction retries are bounded and deterministic.
- Roles remain `NOLOGIN`, default-disabled, and narrowly granted; broad runtime or `pqxx` grants were not added.

## 6. Scheduler coexistence

- Migrations `051` and `052` are unchanged.
- Phase F runtime sources contain no reads or writes to scheduler lease, invocation, protocol, worker-attempt, capacity, or global-control tables.
- No signal, kill, process-group, claim, capacity, or worker-launch authority appears in Phase F.
- The only experiment mutation is through the accepted lifecycle-cancellation security-definer API for `pending`/`paused` work; running work is not mutated.
- Scheduler process verification passed generation fencing, foreign/stale owner rejection, exact attempt binding, delayed-reaper replacement protection, checkpoint-stop exact finalization, and restart retention.
- Production scheduler PID `94420` and the same seven training worker PIDs remained active after verification.

## 7. Verification performed

Principal commands and results:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor-PhaseFIntegration build
```

Result: `BUILD SUCCEEDED`, including a final incremental rebuild. The live `DerivedData/ExpertAdvisor` path was intentionally not overwritten because an active production scheduler launches workers from that binary.

```bash
Tests/CampaignOperationsPhase4CliTests.sh \
  DerivedData/ExpertAdvisor-PhaseFIntegration/Build/Products/Release/LSTM_Release
Tests/CampaignOperationsPhase2CliTests.sh \
  DerivedData/ExpertAdvisor-PhaseFIntegration/Build/Products/Release/LSTM_Release
```

Both passed. `bash -n` also passed.

Strict `-Wall -Wextra -Wpedantic -Werror` builds and executions passed for:

- `CampaignOperationsTests`
- `CampaignOperationsRepositoryTests`
- `ExperimentRecommendationCampaignLaunchRepositoryTests`
- `ExperimentCurrentOperationTests`
- `SchedulerOwnershipPolicyTests`

Database/integration results:

```text
CampaignOperationsRepositoryTests                         PASS
ExperimentRecommendationCampaignLaunchRepositoryTests     PASS
CampaignOperationsPhase4MigrationTests.sql                 PASS
052 → 053 upgrade and idempotent replay                    PASS
SchedulerOwnershipIntegrationTests.sh                      PASS
SchedulerOwnershipProcessIntegrationTests.sh               PASS
ExperimentCurrentOperationMigrationTests.sql               PASS
Unsupported current_operation rejection test               PASS
GlobalExperimentControlIntegrationTests.sh                  PASS
```

The global-control suite initially hit one timing-sensitive lease-owner assertion in unchanged baseline code. Immediate rerun and a separate confirmation rerun both passed completely, including crash-window tests.

Final audits passed:

```text
git diff --check                                           PASS
plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj        PASS
duplicate migration version scan                           PASS
accepted Phase F migration normalized comparison           MATCH
Phase F source scheduler/global-table scan                  CLEAN
Phase G/054 production-diff scan                            CLEAN
backup modification scan                                   CLEAN
```

All disposable databases were dropped.

## 8. Repository state

`git status --short`:

```text
 M Database/README.md
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/CampaignOperations.cpp
 M Sources/CampaignOperations.hpp
 M Sources/CampaignOperationsDispatchRepository.cpp
 M Sources/CampaignOperationsDispatchService.cpp
 M Sources/CampaignOperationsService.cpp
 M Sources/ExperimentScheduler.cpp
 M Tests/CampaignOperationsPhase2CliTests.sh
 M Tests/CampaignOperationsRepositoryTests.cpp
 M Tests/CampaignOperationsTests.cpp
 M Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp
 M docs/CampaignOperationsPhase3.rst
 M docs/architecture/CampaignOperations_Phase3_Durable_Dispatch.md
 M docs/architecture/README.md
 M docs/architecture/Volume_XII_Database.md
 M docs/architecture/Volume_X_Research_Automation.md
 M docs/architecture/adr/ADR-0015-cancellation-reconciliation-and-recovery.md
 M docs/architecture/adr/ADR-0017-campaign-privileges-and-audit.md
?? ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/
?? CampaignOperations_Phase5_ArchitecturalPhaseG_Implementation_Output.md
?? Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql
?? Sources/CampaignOperationsControl.cpp
?? Sources/CampaignOperationsControl.hpp
?? Sources/CampaignOperationsControlRepository.cpp
?? Sources/CampaignOperationsControlRepository.hpp
?? Sources/CampaignOperationsControlService.cpp
?? Sources/CampaignOperationsControlService.hpp
?? Tests/CampaignOperationsPhase4CliTests.sh
?? Tests/CampaignOperationsPhase4MigrationTests.sql
?? docs/CampaignOperationsPhase4.rst
?? docs/architecture/CampaignOperations_Phase4_Controls_Cancellation_Reconciliation.md
```

`git diff --stat`:

```text
19 files changed, 2773 insertions(+), 40 deletions(-)
```

That stat excludes the new untracked Phase F files. Nothing is staged. Database backups were untouched. No commit, push, reset, clean, stash, or amend was performed.

## 9. Residual risks

- The unchanged global-control lease-takeover test exhibited one transient race before passing twice consecutively; stabilizing that baseline assertion is a nonblocking follow-up.
- The full build reports existing external LLVM metadata, libpqxx deprecation, and legacy LSTM warnings; no warning was observed from the new Phase F translation units.
- Production protocol cutover remains `pending`, with seven legacy/unmanaged workers. This was deliberately not changed and prevents safely rebuilding over the live DerivedData binary.
- No Phase F integration issue prevents Phase G development, but Phase G still requires its separate reviewed `054` migration and must not be folded into this working tree increment.

## 10. Readiness verdict

`PHASE_F_INTEGRATION_COMPLETE_WITH_NONBLOCKING_FOLLOWUPS`