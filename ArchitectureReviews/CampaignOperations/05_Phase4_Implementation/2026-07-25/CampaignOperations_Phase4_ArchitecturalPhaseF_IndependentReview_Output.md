---
title: "Campaign Operations Phase 4 Architectural Phase F Independent Review"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase4_ArchitecturalPhaseF_IndependentReview_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 4 Architectural Phase F Independent Review

## Review outcome

Phase F is ready to commit after correcting eight concrete issues. No later-phase behavior or architectural redesign was introduced.

> **Subsequent review history (2026-07-25):** The readiness sentence above is
> preserved as the conclusion of this review, but it was superseded by the
> Campaign Operations Phase 4 Final Staged Verification. That verification
> found four additional commit-blocking defects: reconciliation lock inversion,
> non-atomic/range-inferred cursor membership, pending unbound cancellation
> replay convergence, and cancellation retry state escaping an aborted
> attempt. The corrected implementation now uses campaign-before-request batch
> locking, cursor-ID foreign-key membership committed atomically, a post-lock
> settlement recheck, and attempt-local retry state published only after
> commit. A new final independent staged verification was subsequently completed;
> this historical review alone was not commit approval.

### Issues found and corrected

1. **High — cancellation lock-order inversion and stale version race**
   - Held cancellation acquired the budget head before the budget advisory lock and did not revalidate the caller’s expected request version after locking.
   - Impact: possible deadlocks with budget writers and cancellation against a concurrently changed request.
   - Correction: centralized the authoritative budget advisory → budget head → campaign → reservation → request order and revalidated exact version/ownership under lock. Concurrent identical operations now converge. See [CampaignOperationsControlService.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp:330) and [CampaignOperationsControlRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:154).

2. **High — reconciliation cursor was not restart-safe**
   - Cursor identity used `(run_key,last_target_id)`. Empty successor batches could collide, and recovery replay could select a different batch after state changed.
   - Impact: nondeterministic restart behavior and incorrect replay accounting.
   - Correction: cursor identity is now `(run_key,prior_target_id)`; existing batches reload their durable observations, and the cursor is committed before recovery begins. Exact empty and nonempty replays are tested. See [CampaignOperationsControlService.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp:540) and [migration 053](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql:360).
   - Subsequent correction: `(run_key,prior_target_id)` remains the replay
     lookup key, but it resolves a durable cursor ID. Every observation is
     linked to that ID and commits with the cursor in one transaction. The
     earlier range-based reload was not exact or crash-safe.

3. **High — lease recovery predicate was incomplete**
   - Recovery did not bind all current reservation and expiry evidence into its predicate.
   - Impact: a stale observation could return a request to `ready` after reservation or cancellation evidence changed.
   - Correction: canonical evidence now includes reservation state/version/expiry, lease expiry, bindings, downstream executions, current-attempt outcome, and unresolved cancellation. Recovery reloads and compares the complete evidence and rejects expired reservations. See [CampaignOperationsControlRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:904) and [recovery validation](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:1114).

4. **High — pause gate had a database-level check/write race**
   - The trigger evaluated campaign control state without locking the campaign.
   - Impact: direct capability use could race pause against request acceptance or dispatch mutation despite service-level locking.
   - Correction: the authoritative gate now locks the campaign row before checking control state. See [migration 053](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql:585).

5. **Medium — expired reservations were absent from reconciliation**
   - The accepted architecture requires bounded selection of expired reservations, but only expired dispatch leases were selected. Lease expiry was also inferred from selection rather than explicitly observed.
   - Impact: leaked held reservations were invisible, and a request selected for another reason could be misclassified as lease-expired.
   - Correction: explicit reservation/lease expiry evidence and selection were added. Expired held reservations remain observational and are delegated to the reservation owner; Phase F does not settle them. See [CampaignOperationsControlService.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp:100).

6. **Medium — concurrent lifecycle/cancellation replay lacked transient retry**
   - Deadlock, serialization, or uniqueness races could escape instead of converging.
   - Impact: identical concurrent operator commands could fail nondeterministically.
   - Correction: bounded whole-transaction retry was added for `40001`, `40P01`, and `23505`, with exact replay lookup after serialization.
   - Subsequent correction: all transaction-derived cancellation state is now
     attempt-local and copied to the result only after commit. Pending unbound
     replay also rechecks settlement after acquiring cancellation-domain locks.

7. **Medium — unresolved cancellation selection and recovery ACL were inconsistent**
   - Candidate loading selected the first cancellation even if already settled, potentially hiding a later unresolved request. The stricter recovery query also lacked permission to read settlement evidence.
   - Impact: missed reconciliation work or runtime permission failure.
   - Correction: candidate loading selects the first unsettled cancellation, and the recovery role receives only the required settlement `SELECT`. Catalog tests enforce the grant.

8. **Low — tests and documentation overstated or omitted behavior**
   - Concurrent cancellation, exact cursor restart, expired reservation observation, and the recovery boundary were not fully covered. The implementation report contained stale status/stat claims.
   - Correction: added production-reachable concurrent and clock-expiry tests, cursor replay assertions, migration constraint/ACL checks, and corrected operator/architecture documentation.

## Verification

Passed:

- Required Release build:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
```

- Strict `-Wall -Wextra -Wpedantic -Werror` compilation of the changed repository and service sources.
- `CampaignOperationsTests`
- `CampaignOperationsRepositoryTests` against disposable database `expertadvisor_campaign_operations_phase3_test_local`
- Migration 053 applied twice, including catalog, ACL, constraint, and replay checks
- `CampaignOperationsPhase4CliTests.sh`
- `CampaignOperationsPhase2CliTests.sh`
- `ExperimentRecommendationCampaignLaunchRepositoryTests`
- `git diff --check`

The original repository suite included two-thread initial cancellation,
observe-only followed by recovery, cursor replay, migration replay, lifecycle
ownership, and real clock-based reservation expiry. Subsequent corrections add
the previously missing pending-intent replay race, observation-versus-
cancellation/recovery lock races, atomic-batch crash injection, overlapping
cursor identities, changed-state/same-run-key exact replay, and
post-intent retry injection. Dispatch lease expiry uses a test-only timestamp
adjustment to represent the otherwise time-reachable state.

The active scheduler and seven training workers were inspected and left untouched.

## Files changed by this review

- [CampaignOperationsControlRepository.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.hpp)
- [CampaignOperationsControlRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp)
- [CampaignOperationsControlService.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp)
- [migration 053](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql)
- [CampaignOperationsRepositoryTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp)
- [CampaignOperationsPhase4MigrationTests.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhase4MigrationTests.sql)
- [CampaignOperationsPhase4.rst](/Volumes/Developer%20SSD/ExpertAdvisor/docs/CampaignOperationsPhase4.rst)
- [Phase F architecture documentation](/Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/CampaignOperations_Phase4_Controls_Cancellation_Reconciliation.md)
- [implementation report](/Volumes/Developer%20SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-25/CampaignOperations_Phase4_ArchitecturalPhaseF_Implementation_Output.md)

## Remaining risks and assumptions

- Production dispatch remains disabled, as required.
- Running cancellation still records `running_cancellation_not_supported`; no scheduler or worker signal is sent.
- Pending-cancellation versus scheduler-claim hardening remains the accepted scheduler-ownership limitation and is outside Phase F.
- The Release build emits 226 pre-existing `exec_params` deprecation warnings from legacy `ExperimentScheduler.cpp`. No reviewed Phase F source emits warnings; resolving the legacy set would exceed this review’s scope.

## `git status --short`

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

## `git diff --stat`

```text
 Database/README.md                                 |  18 +
 ExpertAdvisor.xcodeproj/project.pbxproj            |  24 +
 Sources/CampaignOperations.cpp                     |  49 ++
 Sources/CampaignOperations.hpp                     |  10 +
 Sources/CampaignOperationsDispatchRepository.cpp   |  48 +-
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

The stat excludes the fourteen untracked files.
