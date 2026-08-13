---
title: "Campaign Operations Phase H H1 First Attempt V2 Hydration and Replay Targeted Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_FirstAttemptV2HydrationReplay_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H H1 First Attempt V2 Hydration and Replay Targeted Correction

## 1. Executive summary

Correction #2 is implemented. Every hydrated production admission now contains a fully reconstructed first Attempt V2, including its enablement and acquisition audit. Later Attempt V2 SQL replay now validates that complete historical graph before returning the current attempt.

Attempt #2 fails closed with SQLSTATE `23514` and diagnostic `production acquisition replay evidence corrupt` when Attempt #1 evidence is corrupted.

## 2. Root cause

`FindRequestProductionAdmission` previously used a partial typed predicate to locate Attempt #1, retained only its ID and audit, and never reconstructed its canonical identity.

Likewise, `campaign_operations_production_acquire_replay_v2` treated a filtered audit/attempt count as sufficient proof. Corruption outside those selected predicates could therefore be masked by a valid Attempt #2.

## 3. Repository hydration before/after

Before:

```text
admission
  -> partial predicate
  -> firstAttemptId
  -> acquisition audit
```

After:

```text
admission
  -> unique ordinal-1 attempt
  -> shared Attempt V2 reconstruction
  -> canonical/hash validation
  -> exact nested admission
  -> exact enablement/build/scheduler/audit
  -> exact acquisition audit
```

The persisted admission now carries `PersistedProductionDispatchAttemptV2Evidence`, not merely an ID: [CampaignOperationsProductionAdmissionRepository.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.hpp:57).

Both historical and current attempts use the same reconstruction path: [CampaignOperationsProductionAdmissionRepository.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:245).

## 4. SQL replay before/after

Before, replay used `admission_audit_count` with a partial join predicate.

After, replay:

1. Requires exactly one ordinal-1 attempt for the admission.
2. Loads the complete row regardless of its contract-version predicate.
3. Loads its exact enablement and predecessor.
4. Reconstructs Attempt V2, admission, enablement, scheduler, and build canonicals.
5. Validates hashes against reconstructed bytes.
6. Requires exactly one complete acquisition audit and enablement audit.
7. Validates all historical relationships before classifying caller input.

Implementation: [migration 055](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:2933).

## 5–7. Complete graph and validations

The validated graph is now:

```text
current Attempt V2
  -> current admission mirrors
  -> current enablement/build evidence
  -> immutable admission
     -> admission enablement/build/audit
     -> complete first Attempt V2
        -> admission canonical/hash
        -> enablement canonical/hash
        -> scheduler/build canonical/hash
        -> complete acquisition audit
```

First-attempt checks include:

- Request ID and request canonical
- Ordinal exactly `1`
- Admission expected version and resulting `expected + 1`
- Operation key
- Lease digest and expiry through canonical reconstruction
- Dispatcher/requesting-actor equality
- Original service principal
- Capability and Attempt V2 contract version
- Admission ID/canonical/hash
- Enablement ID/canonical/hash
- Approved-build canonical/hash
- Attempt canonical byte equality and reconstructed hash
- Explicit null/partial-evidence rejection

Acquisition-audit validation now checks exact cardinality, campaign, request, attempt, null outcome ID, cause, actor, capability, versions, outcome, replay disposition, diagnostic, admission, and enablement.

## 8. Adversarial tests

The SQL fixture adds 18 Attempt-#2/first-Attempt-#1 corruption cases: [CampaignOperationsPhaseH1MigrationTests.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1766).

They cover canonical, hash, typed request mirror, admission canonical/hash, enablement canonical/hash, approved build, operation key, versions, ordinal, missing/duplicate audit, and audit actor/capability/version/admission/enablement mismatches.

Every case proves:

- Attempt #2 is unchanged.
- The mutation took effect.
- Replay returns exact `23514` corruption classification.
- No existing-identical result is accepted.
- Replay does not repair evidence.
- Fixture restoration occurs through subtransaction rollback.

Repository tests load Attempt #2 while independently corrupting Attempt #1, for both admission and attempt hydration paths: [CampaignOperationsPhaseH1RepositoryTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:68).

## 9–10. Regression safety

Cross-principal exact recovery remains intact. New checks use stored historical principal/build evidence only; no comparison against `session_user`, `current_user`, or the current recovery build was introduced.

Post-recovery reacquisition remains:

```text
ready@v3
-> Attempt #1, dispatching@v4
-> recovered ready@v5
-> same Admission A, Attempt #2, dispatching@v6
```

The final isolated suite proved Admission A reuse, ordinal `2`, versions `5 -> 6`, exact Attempt #2 replay, and unchanged historical evidence.

Protected-function preflight, default-off deployment, role ACLs, lock ordering, scheduler generation 52, and H1/H2/H3/H4 boundaries also passed their existing checks.

## 11–12. Files and checksum

Files changed by this correction:

- `Database/migrations/055_campaign_operations_production_admission_foundation.sql`
- `Sources/CampaignOperationsProductionAdmission.hpp`
- `Sources/CampaignOperationsProductionAdmissionRepository.cpp`
- `Sources/CampaignOperationsProductionAdmissionRepository.hpp`
- `Tests/CampaignOperationsPhaseH1MigrationTests.sql`
- `Tests/CampaignOperationsPhaseH1RepositoryTests.cpp`

Migration SHA-256 and embedded C++ checksum both equal:

```text
70e2324ac4ed45106fe86353eda4f36ba021bdb11140bf585c06e8018094c4d7
```

The manifest-set digest did not require modification and remains valid.

## 13–14. Verification and results

Passed:

- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`
  - Focused corruption replay cases
  - Repository hydration tests
  - Post-recovery reacquisition
  - Cross-principal exact replay and unauthorized ACL rejection
  - Same-hash/different-canonical regressions
  - Default-off, generation-52, lock-order, H-boundary checks
  - Phase 1–5 repository/service/completion regression
- `Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`
- `Scripts/CampaignOperationsH1ManifestValidator.sh`
- Strict C++20 `clang++ -Wall -Wextra -Werror -fsyntax-only`
- Migration SHA versus embedded checksum
- `git diff --check`

The H1 reference graph reported semantic validation `PASS`; its broader `NOT_READY_FOR_REVERIFICATION` remains expected because this pass intentionally did not regenerate final evidence or address the separate readiness-output correction.

## 15. Remaining risks or deferred verification

Release `xcodebuild` was deferred. The safety inspection found an active scheduler and seven managed training workers using the shared DerivedData executable. Rebuilding that product could affect subsequent worker launches.

The read-only `--scheduler-status` check succeeded and confirmed generation 52, an active scheduler owner, and no unmanaged LSTM processes.

## 16. `git status --short`

The large staged H1 baseline remains preserved. This correction appears as unstaged modifications (`AM`) to the six files listed above. Existing untracked reverification Markdown files remain unmodified and unstaged.

## 17. `git diff --stat`

```text
 ..._operations_production_admission_foundation.sql | 255 +++++++++++++--
 Sources/CampaignOperationsProductionAdmission.hpp  |   2 +-
 ...aignOperationsProductionAdmissionRepository.cpp | 223 +++++++------
 ...aignOperationsProductionAdmissionRepository.hpp |  11 +-
 Tests/CampaignOperationsPhaseH1MigrationTests.sql  | 347 +++++++++++++++++++++
 Tests/CampaignOperationsPhaseH1RepositoryTests.cpp | 151 ++++++++-
 6 files changed, 868 insertions(+), 121 deletions(-)
```

## 18. Final disposition

READY_FOR_FIRST_ATTEMPT_V2_FOCUSED_INDEPENDENT_REVERIFICATION