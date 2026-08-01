---
title: "Campaign Operations Phase 4 Architectural Phase F Implementation"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase4_ArchitecturalPhaseF_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 4 Architectural Phase F Implementation

Implemented Campaign Operations Phase 4 / architectural Phase F on `campaign-operations`.

> **Correction history (2026-07-25):** The final staged verification later
> showed that the original exact-cursor, concurrent-replay, whole-transaction
> retry, and readiness claims below were incomplete. The four findings were
> retained in the final staged verification artifact and corrected without
> adding later-phase scope: reconciliation now locks all campaigns before all
> requests, cursor and cursor-ID observation membership commit atomically,
> pending unbound replay rechecks settlement after locking, and cancellation
> retry state is attempt-local until commit. The corrected tree was subsequently subjected to a new
> final independent staged verification before commit.

### Behavioral change

- Durable, versioned pause/resume controls gate request acceptance and Phase E dispatch selection, acquisition, and handoff.
- Cancellation intent and settlement are separate immutable facts.
- Unbound cancellation atomically cancels the request and releases held units.
- Active leases are preserved until database-confirmed expiry.
- Bound cancellation delegates through lifecycle ownership; committed units are never refunded.
- Completed lifecycle work settles as `already_terminal`; running cancellation remains explicitly unsupported.
- Reconciliation durably records one cursor identity and its exact
  foreign-key-linked observation members in a single transaction before
  repair. Replay loads only by that identity.
- Expired held reservations are observed and delegated to the reservation owner; Phase F does not settle them.
- Exact and concurrent replays, including two pending-intent cancellation
  replays after lease expiry, converge on authoritative durable evidence.
- Cancellation transaction retries recreate all database-derived state after
  rollback and publish identifiers only after commit.
- Deferred database constraints reject unaudited or incomplete request transitions.
- No scheduler signaling, worker control, completion, archival, analytics, budgeting extensions, or later phases were added.

### Files changed

Primary additions:

- [Migration 053](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql>)
- [CampaignOperationsControl.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControl.hpp>)
- [CampaignOperationsControl.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControl.cpp>)
- [CampaignOperationsControlRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.hpp>)
- [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp>)
- [CampaignOperationsControlService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.hpp>)
- [CampaignOperationsControlService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp>)
- [Phase 4 CLI tests](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhase4CliTests.sh>)
- [Phase 4 migration tests](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhase4MigrationTests.sql>)
- [Operator documentation](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignOperationsPhase4.rst>)
- [Architecture documentation](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/CampaignOperations_Phase4_Controls_Cancellation_Reconciliation.md>)

Existing domain, dispatch, CLI, project, regression-test, database, and architecture files were updated for Phase F integration.

### Validation

Passed:

- Exact required Release build:
  `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build`
- `CampaignOperationsTests`
- `CampaignOperationsRepositoryTests`
- Migration 053 application twice plus catalog/ACL checks
- `CampaignOperationsPhase2CliTests.sh`
- `CampaignOperationsPhase4CliTests.sh`
- `ExperimentRecommendationCampaignLaunchRepositoryTests`
- `git diff --check`

Coverage includes cancellation, concurrent exact cancellation, pending-intent
replay after lease expiry, deterministic post-intent retry injection,
observation-versus-cancellation and observation-versus-recovery lock races,
atomic cursor-batch crash/restart, overlapping cursors, later changed request
state, later same-run-key observations, exact empty and nonempty replay,
terminal lifecycle cancellation, active dispatch ownership, pause/resume,
blocked acceptance and dispatch, reservation expiry, rollback completeness,
and CLI failures.

The active production scheduler and seven training workers were detected and left untouched. Database tests used only the disposable `expertadvisor_campaign_operations_phase3_test_local` database.

### Remaining risks

- Production dispatch remains disabled as required.
- Running-experiment cancellation intentionally records `running_cancellation_not_supported`.
- The Release build still reports 226 pre-existing libpqxx `exec_params` deprecation warnings in legacy `ExperimentScheduler.cpp`; no new Phase F source warnings were emitted. Fixing those would be an unrelated broad rewrite.

### `git status --short`

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
?? ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/
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

### `git diff --stat`

```text
 Database/README.md                                 |  18 +
 ExpertAdvisor.xcodeproj/project.pbxproj            |  24 ++
 Sources/CampaignOperations.cpp                     |  49 ++
 Sources/CampaignOperations.hpp                     |  10 +
 Sources/CampaignOperationsDispatchRepository.cpp   |  48 ++-
 Sources/CampaignOperationsDispatchService.cpp      |  10 +
 Sources/CampaignOperationsService.cpp              |  14 +
 Sources/ExperimentScheduler.cpp                    | 352 ++++++++++++-
 Tests/CampaignOperationsPhase2CliTests.sh          |   4 +-
 Tests/CampaignOperationsRepositoryTests.cpp        | 570 ++++++++++++++++++++-
 Tests/CampaignOperationsTests.cpp                  | 147 ++++++
 ...RecommendationCampaignLaunchRepositoryTests.cpp | 227 +++++++-
 docs/CampaignOperationsPhase3.rst                  |   4 +-
 .../CampaignOperations_Phase3_Durable_Dispatch.md  |   5 +-
 docs/architecture/README.md                        |   6 +
 docs/architecture/Volume_XII_Database.md           |  16 +-
 docs/architecture/Volume_X_Research_Automation.md  |  19 +-
 17 files changed, 1488 insertions(+), 35 deletions(-)
```

The stat excludes the fourteen untracked files represented by the twelve
untracked paths in `git status`, including the three implementation-review
artifacts under `ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/`.
