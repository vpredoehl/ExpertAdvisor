---
title: "LSTM Profitability Phase 4A Objective Provenance Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase4A_ObjectiveProvenance_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 4A Objective Provenance Implementation

Implemented Phase 4A. The full implementation record is in [LSTM_Profitability_Phase4A_ObjectiveProvenance_Implementation_Output.md](</Volumes/Developer SSD/ExpertAdvisor/LSTM_Profitability_Phase4A_ObjectiveProvenance_Implementation_Output.md>).

Key outcome:

- Frozen objective: `legacy_first_hit_weighted_ce_v1`
- Deterministic identity: `fnv1a64:65818f2e1fa1a324`
- Experiment, recommendation, model, and checkpoint provenance now includes the objective.
- Ordinary resume requires exact objective compatibility.
- Marker-less historical models resolve deterministically to legacy.
- Migration 079 provides safe historical experiment defaults.
- No loss, gradient, target, architecture, inference, profitability, or Campaign Manager scoring behavior changed.

The architecture documentation is [LSTMTrainingObjectiveProvenance.md](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/LSTMTrainingObjectiveProvenance.md>).

Validation passed:

- `Tests/TrainingObjectiveTests.sh`
- `Tests/TrainingObjectiveMigrationTests.sh` against a disposable database, including two migration applications
- `ExperimentRecommendationTests`
- `InferenceProfitabilityTests.sh`
- Syntax-only warning-as-error compilation of all changed production translation units and the persistence test
- Xcode project enumeration

A full Xcode build and live-schema-derived persistence test were not run because the production scheduler, training workers, and inference worker remain active. Migration 079 was not applied to the live database, and no scheduler, experiment, campaign, backfill, or production binary state was changed.

Files changed include the new typed objective module, migration 079, scheduler and recommendation identity paths, model persistence/resume logic, focused tests, Xcode project integration, documentation, and the implementation report. The existing untracked inspection report was not modified.

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Headers/PgModelIO.hpp
 M LSTM/ExperimentMetaAnalyzer.cpp
 M LSTM/LSTM.cpp
 M LSTM/main.cpp
 M Sources/ExperimentRecommendation.cpp
 M Sources/ExperimentRecommendation.hpp
 M Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp
 M Sources/ExperimentRecommendationCampaignOutcomeAssessmentRepository.cpp
 M Sources/ExperimentRecommendationConversionExecutionRepository.cpp
 M Sources/ExperimentRecommendationEvaluationRepository.cpp
 M Sources/ExperimentRecommendationRepository.cpp
 M Sources/ExperimentScheduler.cpp
 M Tests/ExperimentRecommendationTests.cpp
 M Tests/LSTMInputWidthExpansionPersistenceTests.cpp
?? Database/migrations/079_training_objective_provenance.sql
?? Headers/TrainingObjective.hpp
?? LSTM_Profitability_Phase4A_ObjectiveProvenance_Implementation_Output.md
?? LSTM_Profitability_Phase4_TrainingObjective_Inspection_Output.md
?? Tests/TrainingObjectiveMigrationTests.sh
?? Tests/TrainingObjectiveMigrationTests.sql
?? Tests/TrainingObjectiveTests.cpp
?? Tests/TrainingObjectiveTests.sh
?? docs/architecture/LSTMTrainingObjectiveProvenance.md
```

`git diff --stat`—which excludes untracked new files:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj            |   2 +
 Headers/PgModelIO.hpp                              |  50 +++++++-
 LSTM/ExperimentMetaAnalyzer.cpp                    |  27 +++-
 LSTM/LSTM.cpp                                      |  25 ++++
 LSTM/main.cpp                                      |  78 +++++++++++-
 Sources/ExperimentRecommendation.cpp               |  30 ++++-
 Sources/ExperimentRecommendation.hpp               |  18 +--
 ...mmendationCampaignMaterializationRepository.cpp |  14 ++-
 ...endationCampaignOutcomeAssessmentRepository.cpp |  28 ++++-
 ...RecommendationConversionExecutionRepository.cpp |  18 ++-
 ...xperimentRecommendationEvaluationRepository.cpp |  10 +-
 Sources/ExperimentRecommendationRepository.cpp     |   8 ++
 Sources/ExperimentScheduler.cpp                    | 137 ++++++++++++++-------
 Tests/ExperimentRecommendationTests.cpp            |  19 ++-
 Tests/LSTMInputWidthExpansionPersistenceTests.cpp  |  13 ++
 15 files changed, 395 insertions(+), 82 deletions(-)
```