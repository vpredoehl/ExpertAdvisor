---
title: "Campaign Operations Phase H H1 Cross-Principal Exact Recovery Replay Independent Reverification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_CrossPrincipalExactRecoveryReplay_IndependentReverification_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase H H1 Cross-Principal Exact Recovery Replay Independent Reverification

## 1. Executive summary

The correction is complete and architecture-consistent. Independent source tracing and disposable PostgreSQL execution confirm that an authorized, non-superuser recovery principal can repeatedly acknowledge the exact committed Attempt V2 without matching the original service principal or historical build.

No correction-required findings were identified. No files were modified by this review.

## 2. Independent recovery call/data-flow

The actual flow is:

1. Original principal A invokes [`transition_campaign_operations_request_dispatch_production_v2`](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3340>).
2. After replay lookup misses, new acquisition validates scheduler/build readiness and lock order.
3. It persists:
   - immutable admission, including original `session_user`, at [migration 055:3442](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3442>);
   - Attempt V2 at [migration 055:3503](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3503>);
   - acquisition audit at [migration 055:3542](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3542>).
4. Recovery principal B invokes the same protected transition.
5. PostgreSQL checks B’s exact function `EXECUTE` authorization.
6. The transition calls [`campaign_operations_production_acquire_replay_v2`](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:2933>) before scheduler, enablement, current-build, locks, or request CAS.
7. The helper loads the stored request, admission, enablement, Attempt V2, and acquisition audit; reconstructs the canonical chain; and returns the stored composite row.
8. The transition immediately returns that row at [migration 055:3364](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3364>).

H1 repository hydration is a complementary read path, not an H2 mutation adapter:

- [`FindRequestProductionAdmission`](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:304>)
- [`FindProductionDispatchAttemptV2` by ID](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:391>)
- [`FindProductionDispatchAttemptV2` by request/key](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:472>)

These reconstruct the historical build, admission, enablement, Attempt V2, and audit entirely from persisted state.

## 3. Historical-versus-current identity assessment

Verified:

- The replay helper contains no comparison between `session_user`/`current_user` and the stored original service principal.
- B is different from both `campaign_manager_login` and `manager@example.test`.
- `requesting_actor` remains an exact logical replay input, but B is not required to equal that actor.
- `original_executing_service_principal` is loaded from the stored admission/attempt and returned unchanged.
- No current recovery identity is inserted into the admission or Attempt V2 canonical.

Historical build handling is also correct:

- The replay helper ignores the current build argument when an existing exact attempt is found.
- Attempt, admission, and enablement build canonical/hash values are checked against stored immutable evidence at [migration 055:3030](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3030>) and [migration 055:3082](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3082>).
- Repository hydration reconstructs the build from stored service contract, commit, compiler, and executable digest at [repository:18](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsProductionAdmissionRepository.cpp:18>).
- Current build equality remains enforced only after replay misses, on the new-acquisition path at [migration 055:3380](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3380>).

## 4. Authorization assessment

The corrected regression at [migration test:1439](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1439>) proves the real PostgreSQL ACL boundary:

- Principal B is explicitly `LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS`.
- B receives dispatcher membership and inherited `EXECUTE` on the exact transition signature.
- B’s superuser status, session identity, current identity, and dispatcher membership are asserted before replay.
- Principal C has no dispatcher membership and receives SQLSTATE `42501` with `permission denied for function …` before the transition body can execute.
- The self-raised “unexpected success” exception is not swallowed by the `insufficient_privilege` handler.
- The reader membership given to B is read-only, is part of the accepted Manager/readiness role surface, and cannot confer transition execution or DML authority.
- The function grant and temporary memberships are revoked, both roles are dropped, and absence of residual dispatcher execution is asserted at [migration test:1574](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1574>).
- The subsequent post-upgrade deployment audit passed.

## 5. Exact immutable replay assessment

The regression proves:

- B replays twice with two deliberately different current-build strings.
- Both returned composites are full-row `to_jsonb`-identical to the original Attempt V2.
- Same-principal replay remains full-row exact.
- Attempt and admission canonical UTF-8 bytes remain unchanged.
- Full Attempt V2 and admission rows remain unchanged.
- Request state and version remain unchanged.
- Attempt V2, admission, and production acquisition-audit counts remain unchanged.
- Historical principal, requesting actor, build canonical/hash, operation key, request/version evidence, lease digest/expiry, and nested admission/enablement identities are covered by the full-row comparisons.
- The replay helper contains no DML; it cannot replace enablement, admission, Attempt V2, audit, or request state.

Existing conflict/corruption coverage also passed:

- conflicting acquisition replay: SQLSTATE `23505`;
- malformed admission/attempt canonical evidence: SQLSTATE `23514`;
- same-hash/different-canonical evidence is rejected;
- repository hydration rejects tampered Attempt, admission, build, enablement, and audit fields at [repository tests:146](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:146>).

No normalization or repair path was found.

## 6. Regression and validation evidence

Passed:

- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`
  - cross-principal B replay twice;
  - unauthorized C rejection;
  - same-principal replay;
  - corruption/conflict regressions;
  - post-upgrade deployment audit;
  - deterministic lock tests;
  - repository/service hydration;
  - Phase 1–5 repository/service/completion regression;
  - ACL-origin and traceability checks.
- `Scripts/CampaignOperationsH1ManifestValidator.sh`
  - digest `ec6e34b1dd3ee68bf5315f193baee233222822eb045437d154d4b29e9444fb9f`;
  - 92 inventory rows.
- Migration/embedded SHA match:
  - `2de39b9929cd605465a81f09608929e2ef802c073f0832622e0f5ee3ebd0dabc`.
- `bash -n Tests/CampaignOperationsPhaseH1MigrationTests.sh`.
- Required Release `xcodebuild`: `** BUILD SUCCEEDED **`.
- Correction-scoped `git diff --check`: passed.

The build emitted one host configuration warning for a stale `LLVM22.xctoolchain` lacking `Info.plist`; it produced no source/compiler warning and succeeded.

Seven training workers and the active scheduler were observed before validation. No `LSTM_Release` command, production database mutation, or process-control test was run.

## 7. Findings by severity

- Critical: None.
- High: None.
- Medium: None.
- Low: None.

## 8. Files requiring changes

None.

Comparison against the recoverable prior staged review tree shows exactly one correction file:

- [`Tests/CampaignOperationsPhaseH1MigrationTests.sql`](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sql:1439>): 183 insertions, 22 deletions.

Migration 055, production C++, readiness, repository hydration, protected-function preflight, lock logic, diagnostics, and evidence-generation files are byte-identical to the prior staged state. Therefore the test-only correction is valid.

## 9. Remaining limitations/deferred boundaries

- H1 intentionally has no C++ H2 production mutation adapter; repository hydration and SQL transition execution are separate H1 verification surfaces.
- Live production deployment authorization was not exercised; the test uses a disposable cluster and removes the temporary deployment-equivalent grant.
- Final evidence/documentation regeneration was not performed.
- Full staged `git diff --cached --check` still reports 287 pre-existing terminal-empty-field whitespace findings in `CampaignOperationsH1Artifacts.tsv`; the correction-scoped diff is clean.

## 10. Git status summary

- Branch: `campaign-operations`
- HEAD: `1cca34c`
- Review-created changes: none
- Unstaged tracked diff: empty
- Staged state: 114 Phase H1 files, `30,727 insertions`, `20 deletions`
- Untracked at review start and finish:
  - `CampaignOperations_PhaseH_H1_CrossPrincipalExactRecoveryReplay_TargetedCorrection_Output.md`

CROSS_PRINCIPAL_RECOVERY_INDEPENDENTLY_VERIFIED