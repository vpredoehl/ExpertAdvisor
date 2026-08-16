---
title: "LSTM Formal Feature Ablation Mechanism Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FormalFeatureAblationMechanism_Implementation_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Formal Feature Ablation Mechanism Implementation

Implemented formal feature-ablation infrastructure.

- Added canonical named registry and exact-zero mask application after model-input projection.
- Preserved Tensor width, `feature_size=45`, current `n_in=49`, historical mappings, and return features.
- Wired shared masking through training, normal inference, checkpoint inference, and infer-all.
- Added queue CLI: `--ablate-features=name1,name2`, canonicalized for persistence and deduplication.
- Added migration `069_feature_ablation_mask.sql` with default empty mask and updated identity index.
- Resume/inference derive masks from persisted experiment/model lineage; continuation children preserve the source mask.
- Added focused maintained compatibility tests.

Files changed include [FeatureAblation.hpp](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/Headers/FeatureAblation.hpp), [ModelInputContract.hpp](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/Headers/ModelInputContract.hpp), [LSTM.cpp](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM/LSTM.cpp), [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM/main.cpp), [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/Sources/ExperimentScheduler.cpp), [migration](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/Database/migrations/069_feature_ablation_mask.sql), tests, and [implementation report](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM_FormalFeatureAblationMechanism_Implementation_Output.md).

Verification:

- Passed: standalone `LSTMModelInputCompatibilityTests`.
- Passed: `git diff --check`.
- Release build attempted with required DerivedData path; stopped at the expected dirty-tree provenance guard before source compilation, not a compiler failure.
- Database-backed campaign/retry suites remain unrun.

`git status --short` shows the nine intended modified/new files. `git diff --stat` reports 131 tracked insertions / 11 deletions; untracked migration, registry header, and report are not included until staged.