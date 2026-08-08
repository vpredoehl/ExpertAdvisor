---
title: "Campaign Operations Phase H H1 Cross-Principal Exact Recovery Replay Targeted Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_CrossPrincipalExactRecoveryReplay_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H H1 Cross-Principal Exact Recovery Replay Targeted Correction

## 1. Root cause

The original contradiction conflated immutable acquisition evidence—original `session_user` and approved build—with the current recovering Manager.

The staged migration already removed those invalid comparisons. The remaining proof gap was that cross-principal recovery used a superuser, bypassing the accepted function-ACL authorization boundary and leaving unauthorized recovery untested.

## 2. File changed

- [CampaignOperationsPhaseH1MigrationTests.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1439)

Correction delta: 183 insertions, 22 deletions. Migration 055 and generated artifacts were unchanged.

## 3. Behavioral correction

The permanent regression now proves:

- Principal A’s immutable Attempt V2/admission evidence is preserved.
- Authorized, non-superuser principal B replays successfully twice.
- B is authorized independently through `campaign_operations_production_dispatcher` and function EXECUTE authority.
- B’s current build does not replace or conflict with the historical approved build.
- Unauthorized principal C is rejected by PostgreSQL function ACL before execution.
- Same-principal A replay remains exact.
- Entire Attempt/admission rows and canonical UTF-8 bytes remain unchanged.
- Request state/version and attempt/admission/audit counts remain unchanged.
- No admission, Attempt, audit, or mutable-state change is fabricated.
- Temporary deployment-equivalent authorization is fully revoked before the post-upgrade ACL audit.

Existing tamper/corruption tests continue proving malformed historical evidence raises `23514` rather than being normalized.

## 4. Historical versus current identity

Historical identity remains sourced exclusively from persisted admission and Attempt V2:

- original service principal;
- requesting actor;
- approved build canonical/hash;
- request identity/version;
- operation key;
- lease digest/expiry;
- admission and enablement identities;
- complete Attempt V2 canonical/hash and audit evidence.

Current recovery identity is independently represented by `session_user` and validated through role membership/function ACL. It does not enter the historical canonical or replace its principal/build.

Current build readiness remains applicable to new acquisition/handoff, not acknowledgment of an already committed exact replay.

## 5. Recovery flow

```text
Original Manager A
  → transition_campaign_operations_request_dispatch_production_v2
  → immutable admission + Attempt V2 + acquisition audit persisted

Recovery Manager B
  → PostgreSQL validates current EXECUTE/dispatcher authority
  → lookup by (request_id, operation_key)
  → campaign_operations_production_acquire_replay_v2
  → reconstruct and validate request/admission/enablement/Attempt/audits
  → repository hydration validates the same complete canonical chain
  → original Attempt V2 returned unchanged
```

H1 intentionally contains no C++ H2 production mutation adapter. Its repository read path hydrates all historical values through `FindProductionDispatchAttemptV2`; no service or CLI performs direct SQL writes.

## 6. Validation

Passed:

- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`
  - focused cross-principal SQL checks;
  - migration/replay/restore tests;
  - deterministic lock tests;
  - repository/service hydration;
  - Phase 1–5 regression;
  - ACL-origin, pre-enablement, and uniqueness checks.
- `Scripts/CampaignOperationsH1ManifestValidator.sh`
  - 92 inventory rows; manifest digest `ec6e34b1…9444fb9f`.
- Required Release `xcodebuild`
  - `** BUILD SUCCEEDED **`.
- `bash -n Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- Correction-scoped staged `git diff --check`.
- Migration SHA and embedded SHA match:
  `2de39b9929cd605465a81f09608929e2ef802c073f0832622e0f5ee3ebd0dabc`.

The active scheduler and seven training worker PIDs remained running and were not interrupted.

## 7. Remaining boundaries

- Live production authorization/deployment was not exercised; the disposable test temporarily models the later accepted grant and then restores H1’s default-off ACL.
- Final evidence/documentation regeneration was intentionally excluded.
- Full staged `git diff --cached --check` still reports pre-existing terminal-tab warnings in TSV registry rows; the correction-scoped diff is clean.

## 8. Git status

All 114 files are staged. There are no unstaged or untracked files. The correction file is staged as:

```text
A  Tests/CampaignOperationsPhaseH1MigrationTests.sql
```

The other 113 incoming staged H1 files remain preserved.

## 9. Diff statistics

```text
git diff --stat
# empty

git diff --cached --stat
114 files changed, 30727 insertions(+), 20 deletions(-)
```

READY_FOR_CROSS_PRINCIPAL_RECOVERY_INDEPENDENT_REVERIFICATION