---
title: "Campaign Operations Phase 5 Architectural Phase G Independent Focused Verification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase5_ArchitecturalPhaseG_IndependentFocusedVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 5 Architectural Phase G Independent Focused Verification

# Independent verification findings

## BLOCKER

1. **Post-completion mutation gates are incomplete.**
   [Migration 054:946](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql:946>) gates only nine INSERT paths. It omits governance provenance, reservation events/commitments, dispatch outcomes, cancellation settlements, reconciliation observations/resolutions, and guarded reservation/request updates. Concrete ungated writers remain at [CampaignOperationsRepository.cpp:633](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:633>), [CampaignOperationsBindingRepository.cpp:727](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsBindingRepository.cpp:727>), and [CampaignOperationsControlRepository.cpp:803](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:803>).
   This violates ADR-0014’s immutable completion boundary and permits new Campaign Operations authority/history after completion. This is a Phase G defect.
   Required correction: gate every Phase B–F mutation workflow at its owning service/database boundary, while preserving exact historical replay.

2. **Lost-response/uncertain-commit recovery is absent.**
   [CampaignOperationsCompletionService.cpp:147](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionService.cpp:147>) retries only SQL states `40001`, `40P01`, and `23505` on the same connection. It does not handle `pqxx::broken_connection`, reconnect, or perform canonical lookup after an uncertain commit.
   A committed completion followed by connection loss can be reported as failure rather than converging to exact replay. This violates the §31.7 crash/lost-response contract.
   Required correction: connection-factory-based bounded whole-transaction retry, with lookup by campaign and exact canonical replay validation before another append attempt, plus deterministic lost-response tests.

3. **The persisted evidence does not bind every exact identity required by §22.**
   [Migration 054:239](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql:239>) records dispatch attempt/outcome counts, not their ordered identities/canonicals. Binding evidence records only a control-owner ID; cancellation and reconciliation evidence similarly omit exact settlement/resolution canonical identities.
   Different immutable evidence sets can therefore produce the same completion evidence text. This is a Phase G audit-integrity defect.
   Required correction: bind ordered attempt/outcome, owner, settlement, observation, and resolution identities/canonicals into the completion payload, and add golden/reconstruction tests.

## HIGH

4. **The mandatory blocker/classification test matrix is incomplete.**
   [CampaignOperationsTests.cpp:413](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsTests.cpp:413>) covers six aggregate examples, but not the required pairwise/precedence cases such as failed+completed, failed+cancelled, failed+never-dispatched, completed+never-dispatched, permanently-failed plus bound evidence, or duplicate/conflicting evidence. Repository coverage at [CampaignOperationsRepositoryTests.cpp:4930](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:4930>) lacks deterministic cases for several inconsistent-budget, cardinality, owner, lifecycle-conflict, and post-completion paths.
   This is a test gap that prevents the exhaustive proof requested for Phase G.

## NONBLOCKING_FOLLOWUP

5. `CampaignOperationsRepositoryTests` has a pre-existing schema-isolation defect: `CampaignOperationsPhase3MigrationTests.sql` counts trigger names across all schemas. It failed in the restored database when both public and test-schema triggers existed, but passed in an empty disposable database. This is a baseline test-harness issue, not a Phase G semantic regression.

6. The build still reports pre-existing libpqxx deprecation warnings when compiling the scheduler baseline and an external LLVM22 `Info.plist` warning. No warning remains in the new Phase G units.

# 1. Verdict

Phase G is not ready to commit. Migration structure, least privilege, locking, normal replay, and basic completion behavior are substantially implemented, but the post-completion authority boundary, exact evidence binding, uncertain-commit recovery, and mandatory coverage are incomplete.

# 2. Accepted Phase F baseline

- Branch: `campaign-operations`
- HEAD: `eae2fa83c447727fdd5789caabc5286f9cfe63e3`
- Authority: [Phase F scheduler-hardened independent verification](</Volumes/Developer SSD/ExpertAdvisor/CampaignOperations_PhaseF_SchedulerHardened_Integration_IndependentVerification_Output.md:14>)
- Source implementation: `a0f9670480bfa226c28592c992d03e2bdc6ea8bb`
- Verified source tip: `5db4c2003161e61d718d286094a3b9983158c953`
- Common ancestor: `eb11084cfe66c658b6ad97471f3292a11db896a1`
- Phase F is not committed or staged; it exists as unstaged/intent-to-add worktree content.
- Nothing is staged.

The accepted source delta contains 39 paths. The integrated worktree has 38 Phase F paths because `GlobalExperimentControlIntegrationTests.sh` correctly retained the scheduler-hardened HEAD version. The baseline comprises:

- Eight archived Phase F implementation/verification artifacts.
- Migration `053`.
- `CampaignOperations.cpp/.hpp`, six `CampaignOperationsControl*` units, dispatch repository/service, and Campaign Operations service.
- Phase 2/4 CLI and migration tests plus Campaign Operations domain/repository/launch tests.
- Phase 3/4 documentation, architecture volumes/index, ADR-0015, and ADR-0017.
- Merged Xcode, database README, and scheduler CLI integration.

Normalized comparison proved current migration `053` matches accepted source migration `049` apart from its required renumbering. Phase F core implementation files match the accepted verified tip byte-for-byte.

# 3. Phase G-only delta

Eleven new implementation files:

- Migration 054.
- Six `CampaignOperationsCompletion*` source/header files.
- Phase 5 CLI and migration tests.
- Phase 5 operator and architecture documentation.

Ten existing files contain both raw Phase F and later Phase G worktree edits:

- `Database/README.md`
- `ExpertAdvisor.xcodeproj/project.pbxproj`
- `Sources/ExperimentScheduler.cpp`
- Three C++ test files
- `docs/CampaignOperationsPhase4.rst`
- Architecture README and Volumes X/XII

The Phase G implementation report is a review artifact, not implementation.

An exact line-level Phase G stat is impossible because the accepted integrated Phase F/scheduler merge was never committed or captured in the index, and seven files mix both histories. Defensible isolation:

- 11 new Phase G files: **2,744 lines**
- Three exactly comparable test deltas against accepted Phase F: **691 insertions, 4 deletions**
- Total implementation paths: **21**
- Remaining seven mixed files were isolated semantically but cannot receive a defensible exact line stat.

# 4. Accidental Phase F change analysis

No Phase G regression was found in the accepted Phase F implementation units:

- Pause/resume and immutable control-owner behavior are unchanged.
- Cancellation intent remains separate from settlement.
- Budget-first lock ordering and committed-unit non-refund remain unchanged.
- Dispatch lease/attempt and exact-attempt behavior are unchanged.
- Reconciliation observation/resolution ownership remains unchanged.
- Scheduler ownership, generation-52 hardening, and lifecycle isolation remain intact.
- Phase F canonical identities and replay rules were not altered.

The post-completion defect is missing Phase G enforcement around these paths, not an alteration of their pre-completion Phase F semantics.

# 5. Migration 054 verification

Verified successfully:

- Correct ordering after `053`.
- Additive application from the authoritative backup.
- Migration-runner replay: `applied=0, skipped=50`.
- Direct SQL replay.
- No Campaign Operations, scheduler, recommendation, lifecycle, cancellation, reconciliation, or budget history rewrite/deletion.
- Unique completion per campaign and exact unique audit reference.
- Restrictive FKs, terminal/classification enums, budget/member equations, version checks, and blocker validation.
- Immutable UPDATE/DELETE rejection.
- Logical archival implemented only through a rebuildable view.
- Pinned `pg_catalog, public, pg_temp` search paths.
- Correct ownership and NOLOGIN roles.
- No production assignment to `pqxx`.
- Backup hashes remained unchanged.

I corrected missing TRUNCATE protection by adding statement-level guards at [migration 054:925](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql:925>) and corresponding catalog/runtime tests. Owner-role truncate now fails with SQLSTATE `55000`.

The incomplete post-completion trigger set remains a blocker.

# 6. Completion concurrency and invariants

Verified:

- CLI → service → repository → PostgreSQL layering is preserved.
- `SERIALIZABLE` supplies the consistent evidence snapshot and supplements explicit locks.
- Lock order is authorization → budget → campaign → reservations ascending → requests ascending at [CampaignOperationsCompletionRepository.cpp:173](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionRepository.cpp:173>).
- Transaction-derived evidence is rebuilt on SQL retry.
- `40001`, `40P01`, and `23505` retry is bounded to three attempts.
- Identical concurrent completions converge to one row and exact replay.
- Changed replay is non-mutating conflict.
- Cancellation and reconciliation races were exercised.
- Later lifecycle mutation changes current display without changing the stored event.

Not verified/satisfied: uncertain commit or broken connection recovery. Missing post-completion gates also permit later authority changes after the completion transaction has committed.

# 7. Privilege verification

Catalog and runtime checks established:

- `campaign_operations_completion_writer` is NOLOGIN, non-superuser, and has no memberships.
- It is not granted to `pqxx`.
- It has column-scoped INSERT, required SELECT, sequence USAGE, and specified lock/evidence function execution.
- It cannot update/delete/truncate completion history.
- It cannot mutate experiment lifecycle, scheduler, process, recommendation, cancellation-settlement, or reconciliation-resolution tables.
- Reader/auditor roles have read access only and cannot mutate completion authority.
- `PUBLIC`, `pqxx`, and unrelated roles cannot execute completion functions or insert completion/audit rows.
- Trigger functions are owned by `campaign_operations_owner` with pinned search paths.
- Positive completion/status operations and negative reader/owner mutation tests passed.

The migration’s production capability assignment remains disabled.

# 8. Blocker and classification coverage matrix

| Requirement | Implementation | Independent result |
|---|---|---|
| Pause is not closure | `campaign_paused` | Implemented and tested |
| Authorization closed/exhausted | missing/active-head blockers | Implemented; basic cases tested |
| Budget consistent | budget equations/blocker | Implemented; inconsistent cases under-tested |
| Reservation settled | held/reconciliation blockers | Implemented; reconciliation case under-tested |
| Request settled | ready/dispatching/reconciliation blockers | Implemented |
| No active/ambiguous lease | lease and attempt blockers | Implemented and tested |
| Exact attempt evidence | counts plus ambiguity checks | **Exact identity binding missing** |
| Complete bindings/owners | cardinality blockers | Implemented; conflict coverage incomplete |
| Terminal lifecycle | nonterminal blocker | Implemented |
| Cancellation settled | unsettled/inconsistent blockers | Implemented |
| Reconciliation resolved | unresolved blocker | Implemented |
| Contradiction before classification | blocker function precedes classifier | Implemented; coverage incomplete |

Classification order in SQL is correct:

1. `operational_request_failed`
2. `mixed_terminal_outcomes`
3. `downstream_failure`
4. `terminal_partial_completion`
5. `all_scope_cancelled`
6. `all_downstream_completed`

Operational completion remains separate from lifecycle and scientific success. The CLI labels scientific outcome `NOT_AUTHORITATIVE_NOT_EVALUATED`.

# 9. Post-completion mutation-gate matrix

| Mutation path | Required behavior | Current result |
|---|---|---|
| Governance provenance | Prohibit new evidence authority | **Missing gate** |
| Authorization transitions | Prohibit; identical replay only | Gated |
| Budget transitions | Prohibit; identical replay only | Gated |
| Reservation/request acceptance | Prohibit | Gated on initial INSERT |
| Dispatch candidate selection | Read-only advisory | Allowed |
| Lease acquisition | Prohibit | Attempt INSERT causes rollback |
| Binding/control owner | Prohibit | Gated |
| Reservation commitment/dispatch outcome | Prohibit | **Missing gates** |
| Pause/resume | Prohibit | Gated |
| Cancellation request | Prohibit | Gated |
| Cancellation settlement | Prohibit | **Missing gate** |
| Release/expiry/permanent failure | Prohibit | **State/event gates incomplete** |
| Reconciliation observation | Prohibit except explicitly approved display-only evidence | **Missing gate** |
| Reconciliation resolution/recovery | Prohibit | **Missing gate** |
| Ordinary experiment lifecycle retry | Allowed outside Campaign Operations | Correctly allowed and displayed |
| Exact historical replay | Return existing fact | Implemented |
| Changed replay | Conflict without mutation | Implemented |

# 10. Scope and scheduler/lifecycle isolation

No Phase H enablement, scheduler polling, work classes, capacity changes, claims, launches, supervision, signals, lifecycle terminal mutation, refunds, scientific policy, or automatic completion polling were introduced.

All scheduler/worker/process/production terms in the Phase G delta are:

- CLI/help statements explaining non-ownership.
- Documentation exclusions.
- Read-only scheduler/lifecycle isolation tests.
- Existing scheduler-hardened Phase F integration.

Lifecycle access is limited to four read-only experiment columns. Recommendation access is limited to materialization evidence.

# 11. Tests and builds performed

Passed:

- Migration `050 → 054` from a restored authoritative backup.
- Migration runner replay and direct migration 054 replay.
- Phase 5 migration/catalog/ACL tests.
- Direct owner-role truncate-denial test.
- Strict `-Wall -Wextra -Wpedantic -Werror` compilation of new Phase G units.
- Strict `CampaignOperationsTests`.
- Strict `CampaignOperationsRepositoryTests` on an empty disposable database, including the corrected truncate test.
- `ExperimentRecommendationCampaignLaunchRepositoryTests`.
- Phase 2, Phase 4, and Phase 5 CLI parser suites.
- Recommendation launch and outcome-policy pure tests.
- `GlobalExperimentControlTests`.
- `SchedulerCanonicalPathTests.sh`.
- `SchedulerChildStatusTests`.
- `ExperimentCurrentOperationTests`.
- `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj`.
- `bash -n` on Campaign Operations shell tests.
- `git diff --check`, `git diff --cached --check`, and untracked-file whitespace checks.
- Release build:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor-PhaseGIndependentVerification \
  -jobs 2 build

** BUILD SUCCEEDED **
```

The required live DerivedData path was not overwritten because four real unmanaged training workers were active. Scheduler status was read-only: authority vacant, scheduler absent, generation 52 cutover pending.

Not run: process-level scheduler/global-control integration suites, because they could interfere with active training workers.

Diagnostic failure: repository tests against the restored public schema exposed the pre-existing unqualified Phase 3 trigger-count test. The same suite passed in an empty disposable database.

All six disposable databases created during verification were removed.

# 12. Residual risks and deferred Phase H

Deferred correctly to Phase H:

- Production completion capability assignment.
- Production dispatch enablement.
- Scheduler integration or polling.
- Operational rollout/monitoring.

Remaining Phase G risks are the three blockers above and incomplete exhaustive test coverage.

# 13. Exact repository status and complete diff statistics

Nothing is staged. `git diff --cached --stat` is empty.

```text
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-25/CampaignOperations_Phase4_ArchitecturalPhaseF_Implementation_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-25/CampaignOperations_Phase4_ArchitecturalPhaseF_IndependentReview_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-25/CampaignOperations_ThreeFixWorktree_DualRemote_Integration_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-26/CampaignOperations_Phase4_FinalFocusedCorrection2_Implementation_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-26/CampaignOperations_Phase4_FinalFocusedCorrection2_IndependentCEEVerification_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-26/CampaignOperations_Phase4_FinalFocusedCorrection_Implementation_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-26/CampaignOperations_Phase4_FinalIndependentCEEVerification_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-26/CampaignOperations_ThreeFixWorktree_DualRemote_Integration_Output.md
 M Database/README.md
 A Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/CampaignOperations.cpp
 M Sources/CampaignOperations.hpp
 A Sources/CampaignOperationsControl.cpp
 A Sources/CampaignOperationsControl.hpp
 A Sources/CampaignOperationsControlRepository.cpp
 A Sources/CampaignOperationsControlRepository.hpp
 A Sources/CampaignOperationsControlService.cpp
 A Sources/CampaignOperationsControlService.hpp
 M Sources/CampaignOperationsDispatchRepository.cpp
 M Sources/CampaignOperationsDispatchService.cpp
 M Sources/CampaignOperationsService.cpp
 M Sources/ExperimentScheduler.cpp
 M Tests/CampaignOperationsPhase2CliTests.sh
 A Tests/CampaignOperationsPhase4CliTests.sh
 A Tests/CampaignOperationsPhase4MigrationTests.sql
 M Tests/CampaignOperationsRepositoryTests.cpp
 M Tests/CampaignOperationsTests.cpp
 M Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp
 M docs/CampaignOperationsPhase3.rst
 A docs/CampaignOperationsPhase4.rst
 M docs/architecture/CampaignOperations_Phase3_Durable_Dispatch.md
 A docs/architecture/CampaignOperations_Phase4_Controls_Cancellation_Reconciliation.md
 M docs/architecture/README.md
 M docs/architecture/Volume_XII_Database.md
 M docs/architecture/Volume_X_Research_Automation.md
 M docs/architecture/adr/ADR-0015-cancellation-reconciliation-and-recovery.md
 M docs/architecture/adr/ADR-0017-campaign-privileges-and-audit.md
?? CampaignOperations_Phase5_ArchitecturalPhaseG_Implementation_Output.md
?? CampaignOperations_PhaseF_SchedulerHardened_Integration_IndependentVerification_Output.md
?? Database/migrations/054_campaign_operations_completion_and_audit.sql
?? Sources/CampaignOperationsCompletion.cpp
?? Sources/CampaignOperationsCompletion.hpp
?? Sources/CampaignOperationsCompletionRepository.cpp
?? Sources/CampaignOperationsCompletionRepository.hpp
?? Sources/CampaignOperationsCompletionService.cpp
?? Sources/CampaignOperationsCompletionService.hpp
?? Tests/CampaignOperationsPhase5CliTests.sh
?? Tests/CampaignOperationsPhase5MigrationTests.sql
?? docs/CampaignOperationsPhase5.rst
?? docs/architecture/CampaignOperations_Phase5_Operational_Completion_Audit.md
```

`git diff --stat`:

```text
38 files changed, 11061 insertions(+), 187 deletions(-)
```

That excludes 13 untracked files containing 3,159 lines. Treating untracked files as additions, the complete visible worktree is 51 paths, 14,220 added-equivalent lines, and 187 deletions.

# 14. Corrections made

Narrow confirmed corrections:

- Fixed Phase G `pqxx::row` index types to remove `-Wshorten-64-to-32`.
- Corrected the canonical prefix to the authoritative `campaign_operations_completion_v1`.
- Added immutable TRUNCATE guards and catalog/runtime tests.
- Removed two trailing-whitespace defects in the Phase G architecture note.

No broad post-completion-gate, evidence-model, or retry redesign was attempted during this verification.

PHASE_G_NOT_READY_TO_COMMIT