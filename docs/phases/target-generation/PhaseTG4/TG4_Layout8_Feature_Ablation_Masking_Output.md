---
title: "TG4 Layout 8 Feature Ablation Masking"
document_type: "architecture review"
status: "final"
generated_from: "TG4_Layout8_Feature_Ablation_Masking_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# TG4 Layout 8 Feature Ablation Masking

Implemented TG4 Layout‑8 feature ablation support.

Files changed:

- [FeatureAblation.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/FeatureAblation.hpp)
- [LSTMModelInputCompatibilityTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMModelInputCompatibilityTests.cpp)
- [FeatureAblationPairEvaluationRepositoryTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/FeatureAblationPairEvaluationRepositoryTests.sh)

Mask names:

- `tg4_inner_break_any`
- `tg4_source_tg3_structurally_eligible`
- `tg4_source_tg3_confluent`

`kTG4AblationMaskText` provides the canonical all-three text. No new parser alias was introduced; persisted provenance remains the concrete comma-separated names.

Masking occurs at the established `CopyTensorFeaturesForModelInput` boundary, after layout-specific tensor projection. On Layout 8, individual masks clear only their selected column; the all-three control clears columns 73–75 while preserving 0–72 and 76–79 exactly. No TG4 causal or hierarchy logic was changed.

Historical Layout 7 rejects TG4 masks before modifying output (`FEATURE_ABLATION_MASK_FEATURE_ABSENT_FROM_MODEL_INPUT`), preventing aliasing onto its return suffix.

Tests passed:

- `Tests/LSTMModelInputCompatibilityTests.sh`
- `Tests/LSTMInputWidthExpansionTests.sh`
- `Tests/LSTMFeatureVectorParityTests.sh`
- `Tests/TG4TensorModelInputIntegrationTests.sh`
- `Tests/TG4ProductionStreamingPulseAdapterTests.sh`
- `Tests/TG4ProductionStreamingCanonicalReplayTests.sh`
- `Tests/FeatureAblationPairEvaluationTests.sh`
- `Tests/FeatureAblationReplicationEvaluationTests.sh`
- `Tests/FeatureAblationPairEvaluationRepositoryTests.sh`
- `git diff --check`

Subsequent fixture repair:

`Tests/FeatureAblationPairEvaluationRepositoryTests.sh` originally failed
because its hand-built disposable `experiment` schema had not applied
migration 094 and therefore lacked `fresh_initialization_seed`. The test
harness now applies the idempotent owner migration 094 only to its isolated
temporary database. The repaired repository test passes.

The prescribed Release build reached compilation but failed its provenance
phase because the source tree is intentionally dirty: `Release provenance
requires a clean source tree`.

No production database, scheduler, semantic-worker registry, workers,
migrations, or experiments were modified.

Final state:

- Branch/HEAD: `lstm-feature-development` / `99767962a47dc33fb94b68710669b2aed22628c0`
- Worktree: the TG4 implementation files above plus the disposable test-fixture repair; no implementation code changed during the repair

Recommended next step: construct two otherwise identical Layout‑8 experiment definitions with the same seed/configuration: treatment mask empty; control mask `tg4_inner_break_any,tg4_source_tg3_structurally_eligible,tg4_source_tg3_confluent`. Do not queue them until normal experiment-review approval.

Suggested commit message: `Add Layout-8 TG4 feature ablation masks`
