---
title: "LSTM Profitability Phase 4C Paired Objective Experiment Design"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase4C_PairedObjectiveExperiment_Design_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 4C Paired Objective Experiment Design

Phase 4C design and implementation are complete. No live experiments were queued, and the scheduler/backfill remained untouched.

The recommended initial pair is `usdcadrmp`, horizon 6, at 80 epochs. The report contains the exact runnable control/treatment commands, canonical identities, evaluation SQL, and decision rubric:

[Phase 4C Paired Objective Experiment Design](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_Profitability_Phase4C_PairedObjectiveExperiment_Design_Output.md)

### Behavioral change

- Legacy remains the default.
- Objective selection is explicit and unknown values are rejected.
- Queue/dry-run output includes objective ID and hash.
- Scheduler training children receive the persisted objective explicitly.
- Runtime verifies child argv against the authoritative experiment row and emits `TRAINING_OBJECTIVE_ACTIVE`.
- Duplicate and resume identity remain objective-sensitive.
- Campaign Manager, recommendations, and continuation defaults remain legacy.
- No migration or objective-math change was required.

### Files changed

- [TrainingObjective.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/TrainingObjective.hpp)
- [ExperimentScheduler.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp)
- [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp)
- [TrainingObjectiveTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/TrainingObjectiveTests.cpp)
- [TrainingObjectiveSchedulerIntegrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/TrainingObjectiveSchedulerIntegrationTests.sh)
- [LSTMTrainingObjectiveProvenance.md](/Volumes/Developer%20SSD/ExpertAdvisor/docs/architecture/LSTMTrainingObjectiveProvenance.md)
- Required Phase 4C report linked above.

### Validation

Passed:

- `TrainingObjectiveTests.sh`
- `TrainingObjectiveMigrationTests.sh`
- New Phase 4C disposable-database scheduler integration
- Input-width scheduler and persistence/resume tests
- Strict `ExperimentRecommendationTests` compile/run with `-Werror`
- `git diff --check`
- Isolated Debug Xcode build with `DerivedData/Phase4C`

The canonical production Release binary was not overwritten. The isolated build succeeded with an existing unrelated unreachable-code warning; no new warning was introduced by this change.

Remaining limitations: the pair has not been run, USD/CAD H6 historical profitability is still awaiting backfill, and separate classification/auxiliary epoch loss series are not persisted.

`git status --short`:

```text
 M Headers/TrainingObjective.hpp
 M LSTM/main.cpp
 M Sources/ExperimentScheduler.cpp
 M Tests/TrainingObjectiveTests.cpp
 M docs/architecture/LSTMTrainingObjectiveProvenance.md
?? LSTM_Profitability_Phase4C_PairedObjectiveExperiment_Design_Output.md
?? Tests/TrainingObjectiveSchedulerIntegrationTests.sh
```

`git diff --stat`:

```text
 Headers/TrainingObjective.hpp                      | 11 +++++
 LSTM/main.cpp                                      | 55 +++++++++++++++++++++-
 Sources/ExperimentScheduler.cpp                    | 41 +++++++++-------
 Tests/TrainingObjectiveTests.cpp                   |  7 +++
 .../LSTMTrainingObjectiveProvenance.md             | 17 ++++++-
 5 files changed, 112 insertions(+), 19 deletions(-)
```

The stat excludes the two untracked new files.