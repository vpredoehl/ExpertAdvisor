---
title: "Campaign Operations Phase 5 Architectural Phase G Implementation"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase5_ArchitecturalPhaseG_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 5 Architectural Phase G Implementation

## 1. Decision and scope

Implemented Campaign Operations Phase 5, exactly architectural Phase G under accepted architecture §22/§31.7 and ADR-0014, ADR-0015, and ADR-0017.

No Phase H enablement, scheduler ownership, lifecycle mutation, scientific policy, refunds, automatic polling, force completion, reopening, or administrative override was added.

## 2. Files changed

Migration:

- [054_campaign_operations_completion_and_audit.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql>)
- [Database/README.md](</Volumes/Developer SSD/ExpertAdvisor/Database/README.md>)

Domain/repository/service:

- [CampaignOperationsCompletion.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletion.hpp>)
- [CampaignOperationsCompletion.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletion.cpp>)
- [CampaignOperationsCompletionRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionRepository.hpp>)
- [CampaignOperationsCompletionRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionRepository.cpp>)
- [CampaignOperationsCompletionService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionService.hpp>)
- [CampaignOperationsCompletionService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionService.cpp>)

CLI/build:

- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj>)

Tests:

- [CampaignOperationsPhase5MigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhase5MigrationTests.sql>)
- [CampaignOperationsPhase5CliTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhase5CliTests.sh>)
- [CampaignOperationsTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsTests.cpp>)
- [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp>)
- [ExperimentRecommendationCampaignLaunchRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp>)

Documentation:

- [CampaignOperationsPhase5.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignOperationsPhase5.rst>)
- [CampaignOperations Phase 5 architecture](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/CampaignOperations_Phase5_Operational_Completion_Audit.md>)
- Architecture indexes/Volumes X and XII and the Phase 4 reference were updated.

## 3. Completion semantics

Completion fails closed for:

- pause being used as closure;
- missing or still-obligating authorization;
- missing/inconsistent budget evidence;
- held or reconciliation-required reservations;
- missing, nonterminal, conflicting, or multiple requests;
- active leases or incomplete/ambiguous attempts;
- incomplete binding/control-owner cardinality;
- nonterminal bound lifecycle;
- unsettled/inconsistent cancellation;
- blocking unresolved reconciliation;
- any contradictory or ambiguous evidence.

Classification uses the accepted precedence:

1. `operational_request_failed`
2. `mixed_terminal_outcomes`
3. `downstream_failure`
4. `terminal_partial_completion`
5. `all_scope_cancelled`
6. `all_downstream_completed`

Contradiction or reconciliation-required evidence blocks before classification. Lifecycle and scientific outcomes remain separate dimensions.

## 4. Persistence and privileges

Migration 054 adds:

- One unique `campaign_operations_completion_event` per campaign.
- Immutable completion audit-reference rows.
- Exact evidence, blocker, and classification functions.
- Member, budget, terminal-state, classification, FK, canonical/hash-shape, and uniqueness constraints.
- Deferred same-transaction audit completeness.
- Database triggers rejecting completion/audit update or delete.
- Post-completion gates preventing new Campaign Operations authority or work.
- Rebuildable `campaign_operations_completion_status_v1`, including logical archival and post-completion lifecycle-change display.

`campaign_operations_completion_writer` is NOLOGIN and ungranted to `pqxx`. It receives only column-scoped inserts, required evidence reads, sequence usage, and lock functions. It cannot mutate lifecycle, scheduler, cancellation settlement, reconciliation resolution, or completion history.

## 5. Concurrency and recovery

Completion runs in PostgreSQL `SERIALIZABLE`, retaining repeatable-read evidence while adding SSI protection against cancellation/reconciliation child-row write skew.

Lock order is:

`authorization → budget → campaign/completion → reservations ascending → requests ascending`

Whole transactions retry boundedly for serialization failures, deadlocks, and concurrent uniqueness winners.

Verified:

- concurrent identical completion converges on one event;
- exact replay returns the original event;
- changed replay returns non-mutating `conflicting_replay`;
- completion races safely with cancellation settlement;
- completion races safely with reconciliation observation/resolution and lease recovery;
- restart/migration replay remains idempotent;
- lifecycle requeue changes current display but not recorded completion identity.

## 6. Audit/status behavior

The CLI adds:

- `--campaign-operations-complete-if-settled`
- `--campaign-operations-completion-status`

Output includes disposition, event identity/time/hash, terminal classification, exact blockers, cancellation/reconciliation evidence, current lifecycle evidence, logical archival, and post-completion lifecycle change.

Scientific output is explicitly labeled `NOT_AUTHORITATIVE_NOT_EVALUATED`.

## 7. Verification performed

Passed:

- Required Release build:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
```

- Warning-strict `-Wall -Wextra -Wpedantic -Werror` compilation of all new Phase G units and focused tests.
- `CampaignOperationsTests`
- `CampaignOperationsRepositoryTests` against a fresh disposable database.
- Migration 054 applied twice after 053, with catalog/ACL/constraint/trigger checks.
- `ExperimentRecommendationCampaignLaunchRepositoryTests`, including bound completion and later lifecycle requeue.
- Phase 2, Phase 4, and Phase 5 CLI parser suites.
- `ExperimentRecommendationCampaignLaunchTests`
- `ExperimentRecommendationCampaignOutcomePolicyTests`
- `GlobalExperimentControlTests`
- `SchedulerCanonicalPathTests.sh`
- `SchedulerChildStatusTests`
- `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj`
- `bash -n` for Campaign Operations CLI scripts.
- `git diff --check`
- `git diff --cached --check`

Both disposable databases were removed. Seven active training workers were inspected and left untouched; no scheduler process was running.

## 8. Residual risks or follow-ups

- The full Xcode build still emits pre-existing libpqxx `exec_params` deprecation warnings and the external LLVM22 toolchain `Info.plist` warning. New Phase G units compile warning-clean under `-Werror`.
- Process-level scheduler integration suites were not launched while real training workers were active. Scheduler ownership/path/attempt policy tests and the full application build passed.
- Phase H production capability assignment and rollout remain deliberately deferred.

Repository state remains mixed with the pre-existing Phase F worktree. The unrelated untracked verification report was untouched.

`git status --short` contains:

- the preserved 38-file Phase F baseline;
- 11 new untracked Phase G files;
- Phase G edits to existing build, CLI, tests, and documentation;
- the unrelated untracked Phase F verification report.

`git diff --stat`:

```text
38 files changed, 11046 insertions(+), 187 deletions(-)
```

This stat excludes untracked Phase G files until staged.

## 9. Readiness verdict

`PHASE_G_IMPLEMENTATION_COMPLETE_WITH_NONBLOCKING_FOLLOWUPS`