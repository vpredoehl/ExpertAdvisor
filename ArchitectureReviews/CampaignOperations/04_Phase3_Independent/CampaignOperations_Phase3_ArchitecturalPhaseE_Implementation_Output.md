---
title: "Campaign Operations Phase 3 Architectural Phase E Implementation"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase3_ArchitecturalPhaseE_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 3 Architectural Phase E Implementation

Implemented and staged Campaign Operations Phase 3 / architectural Phase E. Production dispatch remains disabled and no commit was created.

## Implementation

- Added deterministic V1 lease-digest, attempt, binding, control-owner, reservation-commitment, outcome, replay, conflict, and recovery domain contracts.
- Added bounded candidate selection and durable `ready → dispatching` lease acquisition with immutable attempt/audit evidence.
- Added caller-owned atomic handoff through the existing `LaunchRecommendationCampaignInTransaction` workflow.
- Added complete ordered bindings, permanent V1 control ownership, `held → committed` settlement, and `dispatching → bound`.
- Added lookup-before-retry, durable lease recovery, lost-response recovery, and fail-closed ambiguous-evidence handling.
- Added separately authorized exact `pending/train` adoption.
- Added an isolated single-request test adapter requiring an exact database prefix and acknowledgement. No production CLI or poller was added.

## Files

Added:

- [Migration 048](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/048_campaign_operations_durable_dispatch_handoff.sql>)
- [CampaignOperationsDispatch.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatch.hpp>)
- [CampaignOperationsDispatch.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatch.cpp>)
- [CampaignOperationsDispatchRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchRepository.hpp>)
- [CampaignOperationsDispatchRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchRepository.cpp>)
- [CampaignOperationsBindingRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsBindingRepository.hpp>)
- [CampaignOperationsBindingRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsBindingRepository.cpp>)
- [CampaignOperationsDispatchService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.hpp>)
- [CampaignOperationsDispatchService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp>)
- [Phase 3 migration tests](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhase3MigrationTests.sql>)
- [Phase 3 implementation document](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/CampaignOperations_Phase3_Durable_Dispatch.md>)
- [Phase 3 documentation index](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignOperationsPhase3.rst>)

Changed:

- Xcode project
- Campaign Operations pure and repository tests
- Phase 4C/5 launch repository tests
- Architecture Volumes X and XII

## Schema and authority

Migration 048 adds append-only dispatch attempts, outcomes, bindings, control owners, reservation commitments, and audit-reference events. It includes guarded request/reservation transitions, deferred complete-binding enforcement, stable indexes, pinned function search paths, and rerunnable privilege setup.

The migration performs no adoption or backfill, preserves existing rows, keeps `production_dispatch_enabled=false`, and adds no Phase F–H or scheduler-owned schema.

The lock order is:

1. Sorted authorization semantic domains
2. Budget
3. Campaign
4. Reservation
5. Request
6. Phase 4C proposal domains
7. Execution/activation domains
8. Experiments

Acquisition commits before handoff. The handoff transaction retains levels 1–5, classifies existing evidence, enters Phase 5 locks, invokes the existing transaction-bound workflow, and atomically commits lifecycle state, complete bindings, owners, settlement, request state, outcome, and audit evidence.

## Replay, recovery, and adoption

Complete bindings are validated and returned before any reinvocation. SQLSTATE `40001` and `40P01` retry the complete handoff up to three times. Unknown commit results succeed only when the complete canonical chain can be reloaded; partial or contradictory evidence returns reconciliation-required.

Adoption is distinct from replay and requires exact complete `pending/train` evidence plus an independently active `adopt_existing_pending_and_control` authorization. Partial, paused-only, progressed, mismatched, or ownership-colliding evidence fails closed.

## Privileges and safety

Migration 048 creates separate NOLOGIN dispatcher and Phase 5 transactional capability roles. Neither is granted to `pqxx` or any login principal. The transactional role receives only the required column-scoped lifecycle and Phase 3 rights, with no scheduler or worker privileges.

The test adapter requires:

- Database name prefix `expertadvisor_campaign_operations_phase3_test_`
- Acknowledgement `I_UNDERSTAND_PHASE3_TEST_ONLY`
- One explicit request ID

It does not poll, enable production dispatch, start a scheduler, launch workers, signal processes, or alter scheduler configuration. ADR-0016 remains unsatisfied.

## Verification performed

Passed:

- Pure identity/golden-vector suite under `-Wall -Wextra -Wpedantic -Werror`.
- Migration, ACL, acquisition, CAS, Phase 1, and Phase 2 repository suite:
  `LSTM_TEST_DB_NAME=campaign_operations_phase2_corrections_test /tmp/campaign_operations_repository_tests`
- Full Phase 4C/5 and Phase 3 atomic handoff suite:
  `LSTM_TEST_DB_NAME=expertadvisor_campaign_operations_phase3_test_local LSTM_STEP3_RELEASE_BINARY='DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release' /tmp/phase3_atomic_handoff_tests`
  Result: `Experiment recommendation campaign launch repository tests passed`
- Project validation: `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj`
- `git diff --cached --check`
- Release build:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
** BUILD SUCCEEDED **
```

Xcode emitted an unrelated warning about a missing `Info.plist` in the user-installed LLVM22 toolchain.

An externally started production scheduler and seven training workers were present at final inspection. They were not killed, paused, signaled, invoked, or otherwise disturbed.

## Remaining verification gaps

Dedicated tests are still missing for several explicitly required cases:

- Forced `40001` and `40P01` retry injection.
- Revocation/budget mutation versus handoff concurrency.
- Every individual rollback-injection hook.
- Full corruption permutations and ambiguous unknown-commit recovery.
- Control-owner collision and all progressed/causality-mismatch Phase 3 paths.
- Complete explicit default-ACL, sequence-negative, and concurrent duplicate-selection matrices.

Therefore the requested comprehensive verification threshold has not yet been met.

## Git state

- 18 intended files staged.
- No unstaged tracked changes.
- Cached diff: 18 files, 5,070 insertions, 11 deletions.
- Pre-existing review output remains untracked:
  `CampaignOperations_Phase3_Architecture_Readiness_CEE_Review_Output.md`
- HEAD remains `3a761d1`; no commit was made.

The implementation is staged, but the missing mandatory fault-injection and concurrency coverage prevents readiness.

NOT READY FOR INDEPENDENT PHASE 3 VERIFICATION