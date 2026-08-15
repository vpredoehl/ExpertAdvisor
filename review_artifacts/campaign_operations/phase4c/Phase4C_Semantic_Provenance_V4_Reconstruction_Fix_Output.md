---
title: "Phase 4C Semantic Provenance V4 Reconstruction Fix"
document_type: "architecture review"
status: "final"
generated_from: "Phase4C_Semantic_Provenance_V4_Reconstruction_Fix_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Phase 4C Semantic Provenance V4 Reconstruction Fix

Implemented and validated the Phase 4C provenance reconstruction fix.

Root cause: Phase 4C rebuilt source/proposed identities with the legacy hard-coded v3 semantic form, while persisted campaign recommendations use v4, where `donchian20_mode` is part of the canonical identity.

Changes:
- Added v3/v4 semantic identity selection.
  - New/default identities are v4 and include `donchian20_mode`.
  - Persisted v3 recommendations reconstruct with the v3 contract and never acquire v4-only meaning.
- Phase 4C derives the identity version from persisted recommendation canonical text.
- v4 source reconstruction reads and parses the persisted `experiment.donchian20_mode`; it does not substitute a missing value.
- Unsupported semantic versions fail closed.
- Exact canonical/hash equality checks remain unchanged.
- Added durable `donchian20_mode` schema migration and mode type.

Files changed:
- `Sources/ExperimentRecommendation.cpp`
- `Sources/ExperimentRecommendation.hpp`
- `Sources/ExperimentRecommendationConversion.cpp`
- `Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp`
- `Headers/Donchian20Mode.hpp`
- `Database/migrations/060_donchian20_mode.sql`
- `Tests/ExperimentRecommendationTests.cpp`
- `Tests/ExperimentRecommendationCampaignMaterializationRepositoryTests.cpp`

Validation:
- Identity regression test passed: v4 determinism, v3 preservation, and invocation hash coverage.
- Phase 4C repository regression passed in a disposable database:
  - legitimate v4 reconstruction accepts `donchian20_mode=enabled`;
  - v4 source-mode mismatch rejects as `inconsistent_provenance`;
  - historical v3 provenance remains reconstructable without a mode field.
- Release build passed:

  `xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build`

Production validation:
- Read-only inspection first showed approval 3 had no materialization.
- Materialization of approval 3 succeeded:
  - `campaign_materialization_id=3`
  - 10 selected members
  - 10 conversion proposals created
  - all 10 member recommendations were v4 and all reconstructed sources were `enabled`.
- Exact replay returned `RECOMMENDATION_CAMPAIGN_MATERIALIZATION_ALREADY_RECORDED` with zero new proposals.

Safety counts:
- Experiments: `331 → 331`
- Conversion proposals: `3 → 13` (the intended ten materialized proposals)
- Activations: `1 → 1`
- Conversion executions: unchanged by the command (`3` afterward)
- No experiments were created/modified; no proposals activated; no scheduler/workers started by materialization.
- An existing scheduler and training worker were already active before validation and remained untouched.

No independent blocker appeared. No commit or push performed.

`git status --short` includes the intended changes above plus the pre-existing untracked `Phase4A_Scoring_Approved_Recommendation_Workflow_Fix_Output.md`.

`git diff --stat`: 6 tracked files changed, `244 insertions(+), 32 deletions(-)`; it excludes the two new untracked intended files until staged.