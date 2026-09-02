---
title: "LSTM Campaign Manager Phase 3B Scoring Policy Ranking Homogeneity Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_Phase3B_ScoringPolicy_RankingHomogeneity_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Phase 3B Scoring Policy Ranking Homogeneity Implementation

Implemented Campaign Manager Phase 3B fail-closed ranking semantic homogeneity. The worktree remains uncommitted, and migration 077 was tested only in an isolated temporary PostgreSQL cluster—not applied operationally.

## Implementation report

1. Files changed

Production:

- `Sources/ExperimentRecommendationScoring.hpp/.cpp`
- `Sources/ExperimentRecommendationEvaluation.hpp/.cpp`
- `Sources/ExperimentRecommendationEvaluationRepository.hpp/.cpp`
- `Sources/ExperimentRecommendationEvaluationService.cpp`
- `Sources/ExperimentRecommendationRanking.hpp/.cpp`
- `Sources/ExperimentRecommendationRankingRepository.hpp/.cpp`
- `Sources/ExperimentRecommendationRankingService.cpp`
- `Sources/ExperimentRecommendationCampaignPlanningRepository.cpp`

Tests and documentation:

- `Tests/ExperimentRecommendationRankingTests.cpp`
- `Tests/ExperimentRecommendationRankingRepositoryTests.cpp`
- `Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp`
- `Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp`
- New `Tests/CampaignManagerPhase3BRankingSemanticMigrationTests.sql`
- New `docs/CampaignManagerPhase3BRankingSemanticHomogeneity.rst`

2. Migration added

- `Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql`
- Confirmed 077 was the next unused number.
- Migration 076 was not modified.

3. Pre-change defect

Broad ranking scopes loaded evaluation results from multiple evaluation runs and directly sorted their numeric `final_score` values. No full-population proof established that scoring policies, scoring algorithms, evaluators, or disposition semantics matched before ordering and persistence.

4. Scoring semantic identity

Version 1 commits to:

- Full authoritative scoring-policy canonical text.
- Scoring version.
- Fixed algorithm contract.
- Component names and order.
- Component normalization.
- Weighted aggregation and score clamping.
- Missing-value behavior.
- Structural-distance behavior.
- Source-metric behavior.

5. Evaluation semantic identity

Version 1 commits to:

- Evaluation and evaluator versions.
- Evaluator algorithm.
- Disposition precedence.
- Eligibility semantics.
- Complete scoring semantic canonical.

6. Coherence rules

- Canonical text is authoritative.
- Policy canonical text must parse and exactly reconstruct.
- Hash must equal FNV-1a of its own canonical.
- Versions must match embedded versions and supported contracts.
- Same hash/different canonical and same canonical/inconsistent hash are rejected.
- Missing or malformed provenance is never filled from current defaults.

7. Full-population validation

Every loaded member is examined before sorting, limiting, membership construction, or snapshot creation. The validator:

- Validates each scoring semantic identity and embedded policy.
- Validates each evaluation semantic identity and result identity hash.
- Counts distinct authoritative `(version, canonical)` identities.
- Requires exactly one scoring and one evaluation identity for non-empty input.
- Returns deterministic result/run IDs, hashes, counts, and failure reason.

8. Protected scopes

All seven scopes are protected:

- `evaluation_run`
- `recommendation_scan`
- `symbol`
- `horizon`
- `family`
- `symbol_horizon`
- `global`

9. Empty populations

Existing empty-snapshot behavior is preserved. Empty snapshots record state `empty`, zero distinct identities, and no fabricated scoring/evaluation semantic identity.

10. Historical snapshots

Migration 077 classifies historical snapshots from full `source_membership_canonical`, not persisted top-N members:

- `verified_homogeneous`
- `empty`
- `legacy_heterogeneous`
- `legacy_unverified`

Historical rank, membership, scores, evaluations, and materialized campaigns are unchanged.

11. Snapshot schema additions

Added:

- Snapshot identity version.
- Population semantic state.
- Scoring canonical/hash/version.
- Evaluation canonical/hash/version.
- Distinct semantic counts.
- Homogeneity validation result.
- Shape constraints, indexes, triggers, and application-role immutability.

12. Snapshot identity

New snapshots use `experiment_recommendation_ranking_snapshot_identity_v2` and commit to population state plus both common semantic canonicals. Changing semantics while holding membership and numeric scores fixed changes the identity.

13. Ranking member database enforcement

A member insert must resolve through its evaluation result/run to the exact scoring and evaluation identities stored on the snapshot. Direct mismatched insertion fails with SQLSTATE `23514`.

14. Evaluation coherence

Application and database enforcement now require evaluation results to match their parent run’s:

- Evaluation policy canonical/hash/version.
- Evaluator version.
- Scoring policy canonical/hash/version.
- Evaluation result identity construction.

15. Score-history hardening

General score-history redesign was deliberately deferred. Phase 3B added only evaluation and ranking history enforcement required for ranking trust. Existing score-history weaknesses outside this semantic boundary remain out of scope.

16. Comparison behavior

Pairwise comparison now returns dedicated invalid/incomparable semantic states for:

- Malformed scoring provenance.
- Malformed evaluation provenance.
- Different scoring semantics.
- Different evaluation semantics.

Verified-compatible score-delta behavior is unchanged.

17. Campaign planning

Campaign planning accepts verified homogeneous snapshots and valid empty snapshots. It rejects heterogeneous or unverified snapshots before selection. Ordering and materialization semantics are unchanged.

18. Observability and CLI

Ranking status/list output includes state, semantic hashes/versions, distinct counts, and validation result. Explicit status includes canonicals. Rejections emit:

`EXPERIMENT_RECOMMENDATION_RANKING_REJECTED_HETEROGENEOUS_SEMANTICS`

with scope, member count, result IDs, conflicting run IDs, hashes, counts, and reason.

19. Requested-limit proof

Tests create a limit of one where the visible top member is compatible but an incompatible member exists outside the limit. Ranking rejects before limiting and persists no snapshot.

20. Same numeric score proof

Tests give incompatible evaluations identical `final_score` values. Ranking still rejects with `heterogeneous_scoring_semantics`.

21–25. Profitability and Phase 3C

- `kPhase3AProfitabilityScoringWeight == 0.0`
- `kPhase3AProfitabilityScoreContribution == 0.0`
- Profitability remains absent from scoring/evaluation/ranking semantic identities.
- Available, unavailable, positive, negative, and zero-actionable profitability remain decision-neutral.
- Homogeneous ranking order is unchanged; existing deterministic ordering tests pass.
- Migration 076 and FINAL inference-profitability binding are unchanged.
- No Phase 3C threshold, gate, weight, normalization, ranking, tie-break, or campaign-selection behavior was added.

## Verification

All targeted compilation used C++20 with `-Wall -Wextra -Werror`.

Passed:

- Recommendation scoring pure tests.
- Recommendation evaluation pure tests.
- Recommendation ranking pure tests.
- Campaign planning pure tests.
- Scoring repository tests.
- Evaluation repository tests.
- Ranking repository tests.
- Campaign planning repository tests.
- Scoring migration tests.
- Evaluation migration tests.
- Ranking migration tests.
- New Phase 3B migration replay/classification/enforcement test.
- Phase 3A profitability pure, repository, source-repository, and migration tests.
- Inference profitability pure and repository tests.
- Continuation profitability policy and migration tests.
- Checkpoint policy hardening and migration tests.

Representative exact database invocation:

```bash
psql -X -q -h 127.0.0.1 -p 55439 -d phase3b_test \
  -f Tests/CampaignManagerPhase3BRankingSemanticMigrationTests.sql
```

The final scoring → evaluation → ranking → campaign-planning repository sequence passed serially with exit 0. One evaluation concurrency assertion failed while database-heavy suites ran concurrently; the same executable passed immediately alone and again in the final serial sequence.

26. Test result

All final focused and regression runs: PASS.

27. `git diff --check`

PASS, exit 0.

28. Debug build

The normal command initially failed before compilation because Xcode’s user-script sandbox denied the existing provenance generator’s random sibling temporary file.

The provenance generator remained active, and Debug passed with:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Debug \
  -derivedDataPath DerivedData/ExpertAdvisor \
  ENABLE_USER_SCRIPT_SANDBOXING=NO \
  build
```

Result: `** BUILD SUCCEEDED **`

The build reports existing project-wide libpqxx `exec_params` deprecation warnings; targeted changed sources compile cleanly under `-Werror`.

29. Release build

Pending. The canonical Release provenance script explicitly requires a clean committed worktree. Per instruction, Release protections were not bypassed.

30. `git status --short`

```text
 M Sources/ExperimentRecommendationCampaignPlanningRepository.cpp
 M Sources/ExperimentRecommendationEvaluation.cpp
 M Sources/ExperimentRecommendationEvaluation.hpp
 M Sources/ExperimentRecommendationEvaluationRepository.cpp
 M Sources/ExperimentRecommendationEvaluationRepository.hpp
 M Sources/ExperimentRecommendationEvaluationService.cpp
 M Sources/ExperimentRecommendationRanking.cpp
 M Sources/ExperimentRecommendationRanking.hpp
 M Sources/ExperimentRecommendationRankingRepository.cpp
 M Sources/ExperimentRecommendationRankingRepository.hpp
 M Sources/ExperimentRecommendationRankingService.cpp
 M Sources/ExperimentRecommendationScoring.cpp
 M Sources/ExperimentRecommendationScoring.hpp
 M Tests/ExperimentRecommendationCampaignPlanningRepositoryTests.cpp
 M Tests/ExperimentRecommendationEvaluationRepositoryTests.cpp
 M Tests/ExperimentRecommendationRankingRepositoryTests.cpp
 M Tests/ExperimentRecommendationRankingTests.cpp
?? Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql
?? LSTM_CampaignManager_Phase3B_ScoringPolicy_RankingHomogeneity_Inspection_Output.md
?? Tests/CampaignManagerPhase3BRankingSemanticMigrationTests.sql
?? docs/CampaignManagerPhase3BRankingSemanticHomogeneity.rst
```

The inspection output was pre-existing and untouched.

31. `git diff --stat`

```text
 ...entRecommendationCampaignPlanningRepository.cpp |  45 ++++-
 Sources/ExperimentRecommendationEvaluation.cpp     | 132 ++++++++++++
 Sources/ExperimentRecommendationEvaluation.hpp     |  44 ++++
 ...xperimentRecommendationEvaluationRepository.cpp |  77 ++++++-
 ...xperimentRecommendationEvaluationRepository.hpp |   2 +
 .../ExperimentRecommendationEvaluationService.cpp  |  56 +++++-
 Sources/ExperimentRecommendationRanking.cpp        | 219 ++++++++++++++++++--
 Sources/ExperimentRecommendationRanking.hpp        |  46 ++++-
 .../ExperimentRecommendationRankingRepository.cpp  | 164 ++++++++++++++-
 .../ExperimentRecommendationRankingRepository.hpp  |  10 +
 Sources/ExperimentRecommendationRankingService.cpp | 119 ++++++++++-
 Sources/ExperimentRecommendationScoring.cpp        | 103 ++++++++++
 Sources/ExperimentRecommendationScoring.hpp        |  29 +++
 ...commendationCampaignPlanningRepositoryTests.cpp |  54 ++++-
 ...mentRecommendationEvaluationRepositoryTests.cpp |  27 ++-
 ...erimentRecommendationRankingRepositoryTests.cpp | 224 ++++++++++++++++++---
 Tests/ExperimentRecommendationRankingTests.cpp     | 166 +++++++++++++--
 17 files changed, 1423 insertions(+), 94 deletions(-)
```

Standard `git diff --stat` excludes the new untracked migration, migration test, and documentation.

32. Unresolved issues

- Canonical Release remains pending until independent verification and commit.
- The existing Xcode provenance script’s sandbox/temp-file incompatibility affects the normal Debug invocation.
- Existing project-wide libpqxx deprecation warnings remain; changed Phase 3B sources are warning-clean under targeted `-Werror` compilation.
- General score-run/result/component immutability hardening was deliberately deferred.