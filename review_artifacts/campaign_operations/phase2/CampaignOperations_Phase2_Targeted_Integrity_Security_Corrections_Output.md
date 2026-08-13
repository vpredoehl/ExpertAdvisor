---
title: "Campaign Operations Phase 2 Targeted Integrity and Security Corrections"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase2_Targeted_Integrity_Security_Corrections_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 2 Targeted Integrity and Security Corrections

Implemented and staged the targeted Phase 2 corrections. No commit was created, production dispatch remains disabled, and the existing scheduler binary was not replaced.

## 1. Findings resolved

- Request prerequisite binding: PostgreSQL now binds campaign, authorization identity/hash, action/scope contracts, prerequisite policy, and governance provenance exactly to the accepting authorization. Repository hydration rejects legacy/corrupt mismatches, and replay returns `existing_identical` only for exact evidence; changed prerequisite or provenance produces a deterministic conflict.
- Audit-to-cause binding: deferred constraints now validate budget audit actor, reason, capability, versions, outcome, replay disposition, and ledger ID. Request-acceptance audits are bound to the request, reservation, acquisition event, authorization, budget, actor, reason, capability, and versions.
- Expiry authority: non-null reservation expiry must be greater than PostgreSQL `transaction_timestamp()`. The C++ early check remains.
- Direct-capability coverage: actual NOLOGIN roles are exercised with `SET LOCAL ROLE`, including forged evidence, expired reservations, false audits, cross-capability operations, generated IDs, sequence mutation, and destructive operations.
- Deferred and concurrency coverage: added missing-evidence rollback tests and independent-connection races for authorization, budget, duplicate/changed acceptance, near-exhaustion, and successor ledger entries.
- ACL and CLI coverage: catalog tests now compare exact SELECT, column INSERT, sequence USAGE, and function EXECUTE allowlists and reject all other schema mutation privileges.
- Duplicate and output handling: all Phase 2 command/value options reject duplicates. Machine output now safely percent-encodes unsafe bytes and exposes complete acceptance/status evidence.

## 2. Schema and trigger changes

Migration 047 now provides:

- Exact request-to-authorization prerequisite/provenance validation.
- Cause-specific deferred audit enforcement.
- Database-owned expiry validation.
- Stricter audit cause shapes.
- Atomic rollback when reservation, request, acquisition, or audit evidence is absent or inconsistent.
- Preserved lock order:
  - Acceptance: authorization → budget → campaign.
  - Budget administration: budget → campaign.
- No dispatch, lifecycle transition, scheduler, worker, experiment, settlement, or automatic progression authority.

## 3. Repository and replay changes

`MapRequest` now reloads the reservation and accepting authorization and fails closed on any canonical evidence mismatch.

Replay comparison additionally checks:

- Acquisition-event relationship and amount.
- Authorization prerequisite and provenance evidence.
- Current authoritative budget entry identity and version.
- Exact reservation, request, authorization, and budget canonical identities.

Malformed persisted evidence returns `campaign_operations_request_authorization_evidence_mismatch`; changed replay evidence returns `campaign_operations_logical_operation_payload_conflict`.

## 4. CLI and output changes

- Duplicate command, actor, reason, expected-version, budget-value, expiry, and ID options are rejected deterministically.
- SQL and exception details are safely framed; raw `error.what()` is no longer inserted into comma-delimited output.
- Acceptance output includes campaign, authorization, budget entry/version, reservation, acquisition event, request, states/versions, hashes, replay disposition, and production-dispatch state.
- Request status includes its associated authorization, budget, and reservation evidence.

## 5. Files changed and tests added

Changed files include:

- [Migration and database documentation](/Volumes/Developer%20SSD/ExpertAdvisor/Database)
- [CampaignOperations.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperations.cpp)
- [CampaignOperations.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperations.hpp)
- [CampaignOperationsRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp)
- [CampaignOperationsRepository.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.hpp)
- [CampaignOperationsService.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsService.cpp)
- [CampaignOperationsService.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsService.hpp)
- [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp)
- [CampaignOperationsPhase2CliTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhase2CliTests.sh)
- [CampaignOperationsPhase2MigrationTests.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhase2MigrationTests.sql)
- [CampaignOperationsRepositoryTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp)
- [CampaignOperationsTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsTests.cpp)
- Campaign Operations and architecture documentation.
- Xcode project source membership.

## 6. Verification commands and results

- Pure domain tests:

```console
clang++ -std=c++20 -Wall -Wextra -Werror ... -o /tmp/campaign_operations_phase2_corrections_domain_test
/tmp/campaign_operations_phase2_corrections_domain_test
```

Passed.

- Repository/integration suite:

```console
clang++ -std=c++20 -Wall -Wextra -Werror ... -o /tmp/campaign_operations_phase2_corrections_repository_test
LSTM_TEST_DB_NAME=campaign_operations_phase2_corrections_test /tmp/campaign_operations_phase2_corrections_repository_test
```

Passed. The disposable harness applied migration 045 twice and migration 047 twice, proving migration ordering and safe rerun. Direct-role, deferred rollback, ACL, replay, and concurrency tests passed.

- CLI suite:

```console
Tests/CampaignOperationsPhase2CliTests.sh \
  /tmp/ExpertAdvisorPhase2CorrectionsDerivedData/Build/Products/Release/LSTM_Release
```

Passed.

- Isolated Release build:

```console
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath /tmp/ExpertAdvisorPhase2CorrectionsDerivedData \
  build
```

`BUILD SUCCEEDED`.

- Project and diff validation:

```console
plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj
git diff --check
git diff --cached --check
```

All passed.

## 7. Scheduler safety

The configured Release build was intentionally not replaced. Scheduler PID 43031 and two inference processes were active. `--scheduler-status --log-level=quiet` confirmed the scheduler was running.

The Release build used an isolated `/tmp` DerivedData path. No production experiment rows or scheduler state were modified.

## 8. Remaining risks

- The full Xcode build still reports pre-existing warnings in unrelated legacy sources, primarily deprecated libpqxx `exec_params` usage and existing unreachable/internal-declaration warnings. The corrected Campaign Operations targets compile cleanly under `-Wall -Wextra -Werror`.
- Production ACL assignment was not changed; the bounded roles remain NOLOGIN and ungranted to `pqxx`.
- Existing untracked verification artifacts were preserved and not staged.

## 9. `git status --short`

```text
M  Database/README.md
A  Database/migrations/047_campaign_operations_budget_request_acceptance.sql
M  ExpertAdvisor.xcodeproj/project.pbxproj
M  Sources/CampaignOperations.cpp
M  Sources/CampaignOperations.hpp
M  Sources/CampaignOperationsRepository.cpp
M  Sources/CampaignOperationsRepository.hpp
A  Sources/CampaignOperationsService.cpp
A  Sources/CampaignOperationsService.hpp
M  Sources/ExperimentScheduler.cpp
A  Tests/CampaignOperationsPhase2CliTests.sh
A  Tests/CampaignOperationsPhase2MigrationTests.sql
M  Tests/CampaignOperationsRepositoryTests.cpp
M  Tests/CampaignOperationsTests.cpp
A  docs/CampaignOperationsPhase2.rst
M  docs/architecture/Volume_XII_Database.md
M  docs/architecture/Volume_X_Research_Automation.md
?? CampaignOperations_Phase2_BudgetReservationRequest_Independent_Staged_Verification_Output.md
?? CampaignOperations_Phase2_Budget_Reservation_Request_Implementation_Output.md
?? review_artifacts/campaign_operations/authority_closure/
```

## 10. `git diff --cached --stat`

```text
17 files changed, 6989 insertions(+), 20 deletions(-)
```

CORRECTIONS COMPLETE — READY FOR FOCUSED VERIFICATION