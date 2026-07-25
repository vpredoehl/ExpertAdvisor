---
title: "Campaign Operations Phase 2 Budget Reservation and Request Implementation"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase2_Budget_Reservation_Request_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 2 Budget Reservation and Request Implementation

Implemented and staged Campaign Operations Phase 2. No commit was created.

### Architecture and behavior

- Preserves ADR-0010/11/12/13/17 authority boundaries.
- Implements append-only member-unit budget grants, amendments, revocations, and supersession.
- Atomically persists a `held` reservation, acquisition event, `ready` request, and audit evidence.
- Enforces authorization → budget → campaign locking and budget → campaign mutation locking.
- Exact retries return the existing request; changed actor, reason, grant, budget head, expiry, or payload conflicts.
- Reconciles accounting from authoritative ledger/reservation evidence and fails closed on incomplete evidence.
- Uses separate hardened budget-administrator, request-acceptor, and read-only capabilities.
- Does not dispatch, invoke Phase 5, modify experiments, claim scheduler work, launch workers, or implement campaign controls.

### Files changed

- Domain: [CampaignOperations.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperations.hpp>), [CampaignOperations.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperations.cpp>)
- Repository/service: [CampaignOperationsRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.hpp>), [CampaignOperationsRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp>), [CampaignOperationsService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsService.hpp>), [CampaignOperationsService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsService.cpp>)
- Schema: [047_campaign_operations_budget_request_acceptance.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/047_campaign_operations_budget_request_acceptance.sql>)
- CLI/build: [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>), [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj>)
- Tests: [CampaignOperationsTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsTests.cpp>), [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp>), [CampaignOperationsPhase2MigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhase2MigrationTests.sql>)
- Documentation: [CampaignOperationsPhase2.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignOperationsPhase2.rst>), database README and Volumes X/XII.

### Verification

Passed:

- Domain tests compiled with `-Wall -Wextra -Werror`.
- Repository/integration suite against a disposable PostgreSQL database.
- Migration 045 and 047 repeatability and ACL tests.
- Atomic rollback, authorization denial, budget enforcement, exact/conflicting replay, concurrent duplicate acceptance, accounting, and read-only status tests.
- CLI help, mandatory `--yes`, and prohibited `--dry-run` behavior.
- `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj`.
- Required Release build:
  `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build`
  — **BUILD SUCCEEDED**.

The build still reports existing `exec_params` deprecation warnings from unchanged legacy scheduler code; the new Phase 2 units pass warning-clean compilation.

### Intentionally deferred

Dispatch leases, Phase 5 handoff, experiment activation, scheduler claiming/polling, runtime execution, campaign pause/resume/cancellation/archival, automatic progression, analytics, forecasting, and monitoring remain unimplemented.

### Repository state

`git status --short` shows all 16 implementation files staged. The pre-existing `review_artifacts/campaign_operations/authority_closure/` directory remains untracked and untouched.

`git diff --stat` is empty because implementation changes are staged.

`git diff --cached --stat`:

```text
16 files changed, 5147 insertions(+), 16 deletions(-)
```