---
title: "LSTM Warmup Query-Scope Semantic Provenance Assurance"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Warmup_QueryScope_SemanticProvenance_Assurance_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Warmup Query-Scope Semantic Provenance Assurance

## Verdict: VERSIONING REQUIRED AND IMPLEMENTED

The warmup fix is scientifically distinct: it changes feature values while leaving dates, order, and `n_in` unchanged. Existing semantic v4 was introduced for Donchian provenance before this fix; it cannot distinguish cold-boundary from full-history semantics.

Implemented `feature_warmup_scope`:

- `legacy_cold_boundary`
- `full_history_warmup`

Propagation now covers experiment persistence, model metadata, resume/inference reconstruction, recommendation semantic v5 canonical/hash, duplicate detection/evaluation, campaign materialization, Phase 4C conversion/execution, and scheduler child commands.

Historical compatibility:

- Pre-migration experiment rows receive durable `legacy_cold_boundary` via migration 067.
- Models lacking `feature_warmup_scope_meta` reconstruct as legacy.
- Recommendation semantic v3/v4 reconstruct as legacy; v5 carries the explicit field.
- Unknown canonical versions/scopes fail closed.
- New experiments/models default to `full_history_warmup`.
- Resumes reload and preserve the model’s scope.

`n_in` is not an identity discriminator: model metadata stores only schema version, `n_in`, and hidden size, while the fix preserves width/order. Train/inference date ranges also do not distinguish the algorithms.

Changed files include:

- [067_feature_warmup_scope.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/067_feature_warmup_scope.sql)
- [FeatureWarmupScope.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/FeatureWarmupScope.hpp)
- [PgModelIO.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/PgModelIO.hpp)
- [ExperimentRecommendation.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentRecommendation.hpp)
- [ExperimentRecommendation.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentRecommendation.cpp)
- [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp)
- [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp)
- Conversion, evaluation, repository, materialization, and regression-test sources.

Validation completed:

- Focused `ExperimentRecommendationTests`: passed.
- Release build: `xcodebuild ... build` passed.
- `git diff --check`: passed.

Not run:

- The requested destructive clean build (`rm -rf DerivedData/ExpertAdvisor`) was not safe: an active production scheduler and training worker are currently executing from that exact path.
- Database-backed campaign/conversion tests were not run because no isolated `LSTM_TEST_DB_NAME` is configured and production work is active.

Residual historical ambiguity: a model produced by a corrected-but-unversioned binary before migration 067 cannot be scientifically classified from durable state alone. The implementation treats missing metadata as legacy by the explicit compatibility contract; it does not rewrite historical rows.

`git status --short` contains this work plus the pre-existing warmup working-tree changes. `git diff --stat` reports 16 tracked files changed (426 insertions, 155 deletions), with migration 067 and existing untracked warmup artifacts additionally present.

Final gate: semantic provenance is closed in code, but operational closure awaits migration application and the deferred clean build/integration tests after the active experiment is finished.