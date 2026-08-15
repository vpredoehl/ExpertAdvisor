---
title: "Phase 4A Scoring Approved Recommendation Workflow Fix"
document_type: "architecture review"
status: "final"
generated_from: "Phase4A_Scoring_Approved_Recommendation_Workflow_Fix_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Phase 4A Scoring Approved Recommendation Workflow Fix

Implemented the Phase 4A lifecycle fix.

- Root cause: scoring was restricted to `proposed` in the CLI, service, repository, and pure scorer. Phase 4D also relied solely on an approval-event score ID, which is null when scoring occurs after approval.
- Lifecycle change: scoreable statuses are now exactly `proposed` and `approved`; `rejected`/`expired` are rejected.
- Phase 4D now preserves an approval-linked score when present; otherwise it deterministically selects the latest completed valid score for that recommendation.

Production scoring succeeded:

| Recommendation | Score ID | Run ID |
|---:|---:|---:|
| 225 | 306 | 88 |
| 234 | 307 | 89 |
| 237 | 308 | 90 |
| 245 | 309 | 91 |
| 258 | 310 | 92 |
| 281 | 311 | 93 |
| 285 | 312 | 94 |
| 289 | 313 | 95 |
| 293 | 314 | 96 |
| 302 | 315 | 97 |

All ten remain `approved`; every run is `completed`.

Tests/build:

- Pure scoring regression test passed.
- Added repository coverage for proposed/approved/rejected/expired lifecycle behavior, review immutability, deterministic scoring, and post-approval score selection.
- Clean Release build passed:
  `xcodebuild ... clean build`

Materialization result:

- The original `score_missing_or_incomplete` failure is gone.
- Materialization now stops at `inconsistent_provenance`, before creating anything.
- Remaining blocker: scan-129 recommendations use semantic identity v4 with `donchian20_mode=enabled`, but Phase 4C source reconstruction currently emits v3 identity without the Donchian field. This is an existing identity-model compatibility gap, distinct from recommendation scoring sequencing.
- No Phase 4D materialization/proposal IDs were created.

Safety confirmation:

- Experiment count stayed 331.
- Conversion proposal count stayed 3.
- Active scheduler and training worker were not changed; no experiments were executed, activated, queued, or started.

Files changed:

- `Sources/ExperimentRecommendationScoring.cpp`
- `Sources/ExperimentRecommendationScoringService.cpp`
- `Sources/ExperimentRecommendationRepository.cpp`
- `Sources/ExperimentRecommendationService.hpp`
- `Sources/ExperimentScheduler.cpp`
- `Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp`
- Relevant scoring/materialization tests and Phase 4A/4D architecture docs.

`git status --short` and `git diff --stat` show 12 modified files, 251 insertions, 14 deletions.