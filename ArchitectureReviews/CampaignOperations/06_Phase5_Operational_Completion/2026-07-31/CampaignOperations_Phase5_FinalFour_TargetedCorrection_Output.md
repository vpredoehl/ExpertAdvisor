---
title: "Campaign Operations Phase 5 Final Four-Defect Targeted Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase5_FinalFour_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 5 Final Four-Defect Targeted Correction

BLOCKER: None.

HIGH: Resolved — actual `pqxx::in_doubt_error` recovery now reconnects and proves the durable outcome before retrying.

HIGH: Resolved — every completion replay path performs complete canonical comparison against authoritative evidence.

HIGH: Resolved — PostgreSQL now validates all evidence hashes and the complete V1 completion identity before persistence.

HIGH: Resolved — lifecycle cancellation now serializes with the durable campaign completion boundary, preserves exact replay, and rejects changed/new facts after completion.

MEDIUM: None.

LOW: None.

NONBLOCKING_FOLLOWUP: A separate independent CEE review remains required. Process-level scheduler/worker suites were deliberately not run because one scheduler, seven training workers, and five inference workers were active. The Release build also reports a host-level unused LLVM22 toolchain warning; all touched standalone units compiled cleanly under `-Werror`.

## 1. Decision and exact scope

Decision: the four requested Phase G corrections are complete and ready for independent review.

No Phase H, scheduler signaling, process control, refunds, scientific policy, force completion, reopen, override, deletion, backup, or unrelated lifecycle behavior was added.

The live scheduler, production rows, production database, live DerivedData binary, and backups were not modified. Both disposable test databases were removed after verification.

## 2. Files changed for this correction

- [054_campaign_operations_completion_and_audit.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql>)
- [CampaignOperationsCompletionService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionService.hpp>)
- [CampaignOperationsCompletionService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionService.cpp>)
- [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp>)
- [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp>)
- [CampaignOperationsTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsTests.cpp>)
- [CampaignOperationsPhase5MigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhase5MigrationTests.sql>)
- [ExperimentRecommendationCampaignLaunchRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp>)

The worktree already contained substantial Phase F/G changes before this correction; the status and statistics below represent the complete shared worktree.

## 3. `pqxx::in_doubt_error` recovery proof

- `pqxx::in_doubt_error` is caught explicitly, separately from `broken_connection`.
- The uncertain connection is destroyed before recovery.
- A newly created connection runs a serializable, locked completion lookup before any append retry.
- Stored completion is compared against:
  - the complete retained attempted event, when present; and
  - a freshly rebuilt authoritative candidate and current blocker set.
- Exact equality returns `existing_identical`.
- Changed canonical returns `conflicting_replay`.
- Proven absence permits a bounded whole-transaction retry.
- Repeated inability to prove the result returns the closed `campaign_operations_completion_outcome_ambiguous` failure.
- `40001`, `40P01`, and `23505` remain bounded whole-transaction recovery paths.
- Tests throw actual `pqxx::in_doubt_error` before commit, after a persisted commit, and repeatedly during outcome lookup. They verify a single completion row and audit row.

## 4. Full-canonical replay proof

Normal early replay, fresh invocation, restart/no retained event, uncertain lookup, and uniqueness recovery now:

1. take the accepted domain locks under serializable isolation;
2. load current blockers;
3. rebuild the candidate from authoritative evidence;
4. compare the complete `CompletionEvent`, not hashes or selected request metadata.

Tests cover:

- fresh identical replay;
- changed lifecycle evidence;
- cancellation and reconciliation changes;
- changed actor, reason, and operation key;
- restart with no retained attempted event;
- same hash with a different canonical;
- evidence becoming non-completable;
- post-completion lifecycle retry/requeue remaining display-only while stored completion remains immutable.

## 5. PostgreSQL canonical/hash enforcement

Migration 054 now provides:

- `campaign_operations_tagged_fnv1a64(text)`, hashing UTF-8 bytes with exact unsigned modulo-2^64 FNV-1a arithmetic;
- `campaign_operations_completion_identity_valid(...)`, reconstructing the exact C++ V1 serialization;
- validation of all eight evidence canonical/hash pairs;
- exact validation of `completion_identity_canonical`;
- validation of `completion_identity_hash`;
- rejection before a malformed row can occupy the one-per-campaign key.

The serialization includes campaign, operation key, terminal state, classification, complete budget identity/arithmetic, counts, all evidence canonicals, actor, fixed capability, and reason.

Both internal validation functions are security-definer functions with pinned search paths. Execution is revoked from `PUBLIC` and runtime roles. No new direct update authority over the completion boundary was granted.

Runtime tests passed for every evidence hash, mismatched completion canonical, mismatched completion hash, same-hash/different-canonical, valid C++ persistence, immutability, idempotent migration replay, ACLs, owners, and search paths.

## 6. Lifecycle-cancellation gate behavior

- Lifecycle cancellation derives the operational campaign from the cancellation request, control owner, experiment, and actor.
- The security-definer transition acquires the campaign completion boundary before locking or mutating the experiment.
- Existing lifecycle rows compare both supplied canonical and hash before replay.
- Exact committed replay remains idempotent.
- Changed replay raises `23514`.
- A new lifecycle cancellation after completion raises `23514`.
- The repository replays the immutable stored lifecycle fact instead of deriving a changed candidate from the experiment’s post-cancellation state.
- Ordinary lifecycle-owned retry/requeue remains allowed and display-only.
- Direct inserts are independently protected by the migration trigger matrix.
- No general lifecycle mutation authority, scheduler signaling, or process control was introduced.

## 7. Locking and concurrency proof

The completion order remains:

`authorization → budget → campaign/completion → reservations ascending → requests ascending`

Lifecycle cancellation takes the campaign boundary before the experiment row. It does not acquire Campaign Operations authorization, budget, reservation, or request locks, so it does not introduce a reverse edge.

A durable `completion_boundary_closed` flag is stored on the already-authoritative campaign mutex row. This is necessary because an `INSERT` trigger that waited on a concurrent transaction can otherwise retain its pre-wait MVCC statement snapshot. Completion closes the flag atomically in its transaction; lifecycle gates inspect the locked tuple version directly.

The concurrency test proves:

- completion holds the campaign boundary;
- changed lifecycle replay waits;
- completion commits;
- lifecycle replay loses deterministically with `23514`;
- exactly one completion exists;
- lifecycle evidence remains complete and unchanged;
- no partial completion/lifecycle truth remains.

## 8. Exact tests and build results

Passed:

```text
xcrun clang++ -std=c++20 -O1 -Wall -Wextra -Wpedantic -Werror \
  -Wshorten-64-to-32 -Wno-c++23-attribute-extensions \
  ... Tests/CampaignOperationsTests.cpp ...
/tmp/CampaignOperationsTests-phaseg-final4
```

Result: exit 0.

```text
xcrun clang++ -std=c++20 -O0 -g -Wall -Wextra -Wpedantic -Werror \
  -Wno-deprecated-declarations -Wno-c++23-attribute-extensions \
  ... Tests/CampaignOperationsRepositoryTests.cpp ... libpqxx ...
LSTM_TEST_DB_NAME=phaseg_final4_test_20260731_44777 \
  /tmp/CampaignOperationsRepositoryTests-phaseg-final4
```

Result: exit 0. This covered:

- migration `053 → 054`;
- direct migration 054 replay;
- catalog, owner, ACL, and search-path checks;
- malformed canonical/hash inserts;
- actual `pqxx::in_doubt_error`;
- completion replay and uniqueness recovery;
- Phase F cancellation/reconciliation;
- Phase G blocker/classification tests.

```text
xcrun clang++ -std=c++20 -O0 -g -Wall -Wextra -Wpedantic -Werror \
  -Wno-deprecated-declarations -Wno-c++23-attribute-extensions \
  Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp \
  <CampaignOperations and ExperimentRecommendation source set> \
  ... libpqxx -pthread
LSTM_TEST_DB_NAME=expertadvisor_campaign_operations_phase3_test_phasegfinal4 \
LSTM_STEP3_RELEASE_BINARY=".../ExpertAdvisor-PhaseGFinalFour/.../LSTM_Release" \
  /tmp/ExperimentRecommendationCampaignLaunchRepositoryTests-phaseg-final4
```

Result: `Experiment recommendation campaign launch repository tests passed`.

This covered lifecycle-cancellation migration/runtime/concurrency, exact and changed replay, post-completion direct insertion, retry/requeue display behavior, Phase F cancellation/reconciliation, dispatch, and launch regressions.

```text
bash Tests/CampaignOperationsPhase5CliTests.sh \
  ".../ExpertAdvisor-PhaseGFinalFour/Build/Products/Release/LSTM_Release"
bash Tests/CampaignOperationsPhase4CliTests.sh <same-binary>
bash Tests/CampaignOperationsPhase2CliTests.sh <same-binary>
```

Result: all passed. Phase 5 was rerun after the final build.

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor-PhaseGFinalFour \
  -jobs 1 build
```

Result: `** BUILD SUCCEEDED **`.

```text
plutil -lint \
  ExpertAdvisor.xcodeproj/xcuserdata/vjp.xcuserdatad/xcschemes/xcschememanagement.plist
```

Result: `OK`.

```text
bash -n Tests/CampaignOperationsPhase2CliTests.sh \
  Tests/CampaignOperationsPhase4CliTests.sh \
  Tests/CampaignOperationsPhase5CliTests.sh
git diff --check
git diff --cached --check
```

Result: all exit 0.

During development, the broader suite first exposed the MVCC completion-boundary race and then the Phase F post-transition replay mismatch. Those failures drove the final corrections. An initial test database with the wrong safety prefix was rejected by the test gate, and incomplete/overbroad standalone source lists produced link-only failures before the passing strict source set was used.

## 9. Residual risks and unverified assumptions

- Process-level scheduler/worker tests were not run because active production scheduler, training, and inference processes were observed. This was the required safe choice.
- The alternate build emits a host configuration warning for `/Users/vjp/Library/Developer/Toolchains/LLVM22.xctoolchain` lacking `Info.plist`; no touched-source warnings occurred.
- PostgreSQL FNV validation is intentionally strict and can be noticeable for very large completion canonicals; the concurrency suite therefore uses a bounded 30-second lock timeout.
- Independent CEE verification is still required.
- `screenlog.0` is live and untracked, so its line/byte count can continue changing after the snapshot below.

## 10. Exact `git status --short`

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
?? CampaignOperations_Phase5_ArchitecturalPhaseG_IndependentFocusedVerification_Output.md
?? CampaignOperations_Phase5_ArchitecturalPhaseG_TargetedCorrection_Output.md
?? CampaignOperations_Phase5_TargetedCorrection_Independent_CEE_Review_Output.md
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
?? screenlog.0
```

## 11. Complete diff statistics, including untracked files

Tracked worktree:

```text
38 files changed, 11720 insertions(+), 187 deletions(-)
```

Cached diff:

```text
empty
```

Untracked snapshot:

```text
CampaignOperations_Phase5_ArchitecturalPhaseG_Implementation_Output.md lines=179 bytes=8449
CampaignOperations_Phase5_ArchitecturalPhaseG_IndependentFocusedVerification_Output.md lines=354 bytes=21279
CampaignOperations_Phase5_ArchitecturalPhaseG_TargetedCorrection_Output.md lines=439 bytes=22991
CampaignOperations_Phase5_TargetedCorrection_Independent_CEE_Review_Output.md lines=177 bytes=11447
CampaignOperations_PhaseF_SchedulerHardened_Integration_IndependentVerification_Output.md lines=236 bytes=11942
Database/migrations/054_campaign_operations_completion_and_audit.sql lines=1703 bytes=80205
Sources/CampaignOperationsCompletion.cpp lines=213 bytes=9900
Sources/CampaignOperationsCompletion.hpp lines=99 bytes=3346
Sources/CampaignOperationsCompletionRepository.cpp lines=428 bytes=20424
Sources/CampaignOperationsCompletionRepository.hpp lines=51 bytes=1759
Sources/CampaignOperationsCompletionService.cpp lines=450 bytes=17074
Sources/CampaignOperationsCompletionService.hpp lines=60 bytes=1776
Tests/CampaignOperationsPhase5CliTests.sh lines=84 bytes=3095
Tests/CampaignOperationsPhase5MigrationTests.sql lines=410 bytes=18590
docs/CampaignOperationsPhase5.rst lines=101 bytes=5084
docs/architecture/CampaignOperations_Phase5_Operational_Completion_Audit.md lines=76 bytes=4218
screenlog.0 lines=1837 bytes=201302

UNTRACKED_TOTAL files=17 lines=6897 bytes=442881
```

Combined snapshot: 55 files represented, 18,617 tracked insertions/untracked lines, and 187 tracked deletions.

PHASE_G_FINAL_FOUR_CORRECTIONS_COMPLETE_READY_FOR_INDEPENDENT_REVIEW