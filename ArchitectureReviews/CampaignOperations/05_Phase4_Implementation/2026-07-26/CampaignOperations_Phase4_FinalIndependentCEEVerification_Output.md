---
title: "Campaign Operations Phase 4 Final Independent CEE Verification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase4_FinalIndependentCEEVerification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 4 Final Independent CEE Verification

## Verdict

The staged Campaign Operations Phase 4 implementation is **not ready to commit**. The four recently corrected defects remain resolved, and all focused build/tests pass, but five concrete issues remain.

## Findings

### 1. High — Distinct cancellation operation keys can leave a durable, permanently unsettled intent

Locations:

- [049 migration](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/049_campaign_operations_controls_cancellation_reconciliation.sql:106>)
- [CampaignOperationsControlService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp:324>)
- [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:217>)
- [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:393>)
- [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:1531>)

The schema permits multiple cancellation requests for the same campaign/request when their operation keys differ. Candidate loading, however, selects only the oldest unsettled cancellation.

Reproducible sequence:

1. Cancellation A is recorded while a dispatch lease is active.
2. Cancellation B, with a different operation key, is also recorded for the same request.
3. After lease expiry, replaying B loads A as the candidate cancellation.
4. Settlement transitions are attempted for B, but exact-observation validation rejects the A/B identity mismatch, rolling back.
5. Replaying A settles and cancels the request.
6. B is now durable but cannot settle because the request is already cancelled; it returns `reconciliation_required` indefinitely.
7. Status also examines only the earliest cancellation, potentially reporting the campaign settled while B remains unresolved.

Existing concurrency tests cover identical operation keys only: [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:3620>) and [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:3792>).

Recommended correction: under the complete cancellation lock domain, reject or converge a second distinct cancellation operation while an unresolved cancellation already targets that request. Add an active-lease regression using two different operation keys and verify every durable intent reaches a terminal settlement.

### 2. High — Reconciliation evidence identity depends on the PostgreSQL session time zone

Locations:

- [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:188>)
- [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:1103>)
- [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:1453>)

`reservation.expires_at::text` and `request.lease_expires_at::text` are incorporated directly into canonical reconciliation evidence. PostgreSQL renders `timestamptz::text` in the session time zone.

The same instant was verified to render as:

```text
2026-07-26 12:34:56.123456+00
2026-07-26 07:34:56.123456-05
```

Consequently, two reconcilers using different session time zones generate different canonical/hash identities for identical durable evidence. Overlapping-cursor recovery then cannot locate the already-accepted outcome because it requires byte-identical evidence. This breaks deterministic identity and cross-cursor replay idempotency.

Recommended correction: canonicalize timestamps in SQL using a fixed UTC representation with fixed microsecond precision, or use an integer epoch representation. Add reconciliation and cancellation replay tests where observation and recovery connections use different time zones.

### 3. High — Resolution ownership and causal-transition integrity are not database-enforced

Locations:

- [accepted architecture](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md:435>)
- [resolution schema](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/049_campaign_operations_controls_cancellation_reconciliation.sql:330>)
- [cancellation coordinator grant](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/049_campaign_operations_controls_cancellation_reconciliation.sql:1475>)
- [recovery grant](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/049_campaign_operations_controls_cancellation_reconciliation.sql:1581>)
- [completeness trigger](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/049_campaign_operations_controls_cancellation_reconciliation.sql:835>)

The accepted design requires a resolution to reference an exact authoritative owning-service event/transition. The implemented resolution stores only caller-provided transition canonical/hash text—there is no causal FK.

Both cancellation and recovery roles can insert every resolution column, including either `owning_capability`. The deferred trigger requires only a matching audit row; it does not prove:

- that the named transition exists;
- that it belongs to the stated capability;
- that the disposition matches the observation reason/action;
- that cancellation cannot manufacture recovery attribution or vice versa.

This violates the persisted ownership boundary even though the current repository code normally constructs valid rows.

Recommended correction: introduce enforceable typed causal references and shape constraints, or restrict inserts through capability-specific guarded functions that verify the authoritative transition and observation before inserting. Add direct-role negative tests for cross-capability attribution and nonexistent transition evidence.

### 4. Medium — Cursor membership completeness is checked by the service only after corruption is durable

Locations:

- [cursor schema](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/049_campaign_operations_controls_cancellation_reconciliation.sql:359>)
- [reconciler grants](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/049_campaign_operations_controls_cancellation_reconciliation.sql:1524>)
- [service replay check](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp:585>)
- [migration tests](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhase4MigrationTests.sql:214>)

The reconciler role can commit a cursor with `selected_count > 0` and no observations, or with inconsistent `last_target_id`, membership count, run key, or request bounds. The FK and unique constraint only validate each observation’s cursor reference; they do not validate the cursor as a complete batch manifest.

The service detects a count mismatch on later replay, but by then the authoritative cursor key is permanently occupied and the run cannot be repaired through normal append-only workflows.

Recommended correction: add a deferred cursor-completeness constraint trigger validating count, run key, bounds, ordering, and last-target semantics at commit. Add malformed direct-role transaction tests.

### 5. Medium — Phase 4 reconciliation does not implement the accepted retry contract

Locations:

- [accepted retry contract](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md:948>)
- [retry classification/backoff](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp:30>)
- [reconciliation transaction](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp:572>)
- [individual recovery transaction](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp:626>)

Control and cancellation mutations retry, but reconciliation batch persistence and each recovery mutation execute only once. A `40001` or `40P01` escapes directly despite the accepted bounded three-attempt contract. Existing backoff is also deterministic rather than jittered, and commit-outcome uncertainty is not resolved by canonical lookup.

Recommended correction: apply bounded, whole-transaction retries to cursor persistence and each recovery transaction, keeping all transaction-derived state attempt-local. Resolve uncertain commits through exact cursor/resolution lookup before retrying. Add injected serialization/deadlock tests.

## Focused corrections verified

The four requested corrections are present:

- Global reconciliation lock ordering sorts and deduplicates all campaign IDs, locks every campaign first, then locks request IDs in ascending order: [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:1268>).
- Durable replay resolves the exact `(run_key, prior_target_id)` cursor and loads members exclusively by cursor identity: [CampaignOperationsControlService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp:585>).
- Concurrent cancellation replay checks settlement both before and after acquiring the complete cancellation lock domain: [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp:686>).
- Cancellation retry state is attempt-local and published only after commit: [CampaignOperationsControlService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlService.cpp:300>).

## Verification performed

Passed:

```bash
git diff --cached --check

xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor \
  build

bash -n Tests/CampaignOperationsPhase2CliTests.sh \
  Tests/CampaignOperationsPhase4CliTests.sh

Tests/CampaignOperationsPhase4CliTests.sh \
  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release

Tests/CampaignOperationsPhase2CliTests.sh \
  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release

/tmp/CampaignOperationsTests-staged
```

Also passed in fresh disposable databases:

- `CampaignOperationsRepositoryTests`
- `ExperimentRecommendationCampaignLaunchRepositoryTests`

The initial repository-suite run against a previously reused test database encountered unrelated leftover schemas; the fresh isolated rerun passed. All disposable databases were removed. No scheduler or `LSTM_Release` process was active.

The Release build produced only the existing malformed external LLVM toolchain `Info.plist` warning.

## Files and behavioral change

The staged change comprises 33 files:

- 3 architecture-review records.
- Migration 049 and `Database/README.md`.
- Xcode project integration.
- 12 source files implementing Phase 4 controls, cancellation, reconciliation, CLI integration, dispatch gates, and scheduler integration.
- 6 test files.
- 9 user/architecture/ADR documentation files.

No repository files were modified by this review.

## Repository state

`git status --short`:

```text
AM ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-25/CampaignOperations_Phase4_ArchitecturalPhaseF_Implementation_Output.md
AM ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-25/CampaignOperations_Phase4_ArchitecturalPhaseF_IndependentReview_Output.md
A  ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-25/CampaignOperations_ThreeFixWorktree_DualRemote_Integration_Output.md
M  Database/README.md
A  Database/migrations/049_campaign_operations_controls_cancellation_reconciliation.sql
M  ExpertAdvisor.xcodeproj/project.pbxproj
M  Sources/CampaignOperations.cpp
M  Sources/CampaignOperations.hpp
A  Sources/CampaignOperationsControl.cpp
A  Sources/CampaignOperationsControl.hpp
A  Sources/CampaignOperationsControlRepository.cpp
A  Sources/CampaignOperationsControlRepository.hpp
A  Sources/CampaignOperationsControlService.cpp
A  Sources/CampaignOperationsControlService.hpp
M  Sources/CampaignOperationsDispatchRepository.cpp
M  Sources/CampaignOperationsDispatchService.cpp
M  Sources/CampaignOperationsService.cpp
M  Sources/ExperimentScheduler.cpp
M  Tests/CampaignOperationsPhase2CliTests.sh
A  Tests/CampaignOperationsPhase4CliTests.sh
A  Tests/CampaignOperationsPhase4MigrationTests.sql
M  Tests/CampaignOperationsRepositoryTests.cpp
M  Tests/CampaignOperationsTests.cpp
M  Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp
M  docs/CampaignOperationsPhase3.rst
A  docs/CampaignOperationsPhase4.rst
M  docs/architecture/CampaignOperations_Phase3_Durable_Dispatch.md
A  docs/architecture/CampaignOperations_Phase4_Controls_Cancellation_Reconciliation.md
M  docs/architecture/README.md
M  docs/architecture/Volume_XII_Database.md
M  docs/architecture/Volume_X_Research_Automation.md
M  docs/architecture/adr/ADR-0015-cancellation-reconciliation-and-recovery.md
M  docs/architecture/adr/ADR-0017-campaign-privileges-and-audit.md
?? CampaignOperations_Phase4_FinalFocusedCorrection_Implementation_Output.md
```

Staged diff:

```text
33 files changed, 8466 insertions(+), 38 deletions(-)
```

Unstaged `git diff --stat`, excluded from this staged review:

```text
2 files changed, 53 insertions(+), 9 deletions(-)
```