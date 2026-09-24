---
title: "LSTM Experiment Pair Comparison Reporting Provenance Polish"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_ExperimentPairComparison_ReportingProvenancePolish_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Experiment Pair Comparison Reporting Provenance Polish

Implemented the experiment-pair reporting and provenance polish without weakening authoritative evidence rules.

### Behavior

- Concise mode syntax:

  ```text
  --compare-experiment-pair=EXPERIMENT_A_ID:EXPERIMENT_B_ID --summary
  ```

- Existing verbose mode remains the default.
- Model-derived identity fields now use explicit optional presence semantics. Missing model evidence renders as unquoted `NULL`; persisted `0`, `false`, and empty strings remain real values.
- Unavailable identity values do not fabricate scientific mismatches. Comparisons remain incomplete when final evidence is absent.
- Summary output deterministically reports arm order, B−A convention, status, scientific comparability, identity differences, per-arm evidence/provenance availability, reasons, and available metrics/deltas. It adds no winner, recommendation, ranking, or synthetic score.
- A NULL persisted model producer link now reports:

  ```text
  experiment_<ID>_train_producer_worker_attempt_id_missing
  ```

  The equivalent inference diagnostic is `infer_producer_worker_attempt_id_missing`. Invalid and ambiguous resolution remain separate failures; no producer inference or legacy fallback was added.

Core implementation: [ExperimentPairComparison.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentPairComparison.hpp:20>), [ExperimentPairComparison.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentPairComparison.cpp:274>), [PairedTrainingObjectiveEvaluationRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/PairedTrainingObjectiveEvaluationRepository.cpp:91>), and [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ExperimentScheduler.cpp:12870>).

### Files changed by this increment

- `Sources/ExperimentPairComparison.hpp`
- `Sources/ExperimentPairComparison.cpp`
- `Sources/ExperimentPairComparisonService.hpp`
- `Sources/ExperimentPairComparisonService.cpp`
- `Sources/PairedTrainingObjectiveEvaluationRepository.cpp`
- `Sources/SchedulerCore/ExperimentScheduler.cpp`
- `Sources/SchedulerCore/ProductionSchedulerRuntimeInternal.hpp`
- `Tests/ExperimentPairComparisonTests.cpp`
- `Tests/ExperimentPairComparisonServiceTests.cpp`
- `Tests/ExperimentPairComparisonServiceTests.sh`
- `Tests/PairedTrainingObjectiveEvaluationRepositoryTests.cpp`

### Verification

All passed:

```text
Tests/ExperimentPairComparisonTests.sh
Tests/ExperimentPairComparisonServiceTests.sh
bash Tests/FeatureAblationPairEvaluationTests.sh
bash Tests/PairedTrainingObjectiveEvaluationTests.sh
bash Tests/PairedTrainingObjectiveEvaluationRepositoryTests.sh
```

The repository test used a disposable synthetic PostgreSQL database and confirmed it was dropped. It exercised both model and inference NULL producer-link diagnostics.

Debug build passed:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Debug" \
  -configuration Debug \
  -derivedDataPath "/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development" \
  build
```

The build retained pre-existing `exec_params`, toolchain metadata, and libomp deployment-version warnings; no changed comparison source produced a new warning.

`git diff --check` passed.

No live database, scheduler/worker, or experiment was queried or altered. No migration, training, inference, analysis, Release build, or publication occurred.

A genuinely ambiguous producer link cannot be constructed under the current single-valued producer FK/worker-attempt primary-key schema without violating authoritative schema semantics. The loader still has a distinct ambiguity diagnostic, and the existing supported ambiguous-final-inference fixture remains passing.

### Inherited worktree state

Already modified/untracked before this increment:

```text
M ExpertAdvisor.xcodeproj/project.pbxproj
M Sources/PairedTrainingObjectiveEvaluation.hpp
M Sources/PairedTrainingObjectiveEvaluationRepository.cpp
M Sources/SchedulerCore/ExperimentScheduler.cpp
M Sources/SchedulerCore/ProductionSchedulerRuntimeInternal.hpp
?? LSTM_ExperimentPairComparisonReporting_Output.md
?? LSTM_ExperimentPairComparison_ReadOnlyCLI_Output.md
?? Sources/ExperimentPairComparison.cpp
?? Sources/ExperimentPairComparison.hpp
?? Sources/ExperimentPairComparisonCommand.cpp
?? Sources/ExperimentPairComparisonService.cpp
?? Sources/ExperimentPairComparisonService.hpp
?? Tests/ExperimentPairComparisonServiceTests.cpp
?? Tests/ExperimentPairComparisonServiceTests.sh
?? Tests/ExperimentPairComparisonTests.cpp
?? Tests/ExperimentPairComparisonTests.sh
```

Some inherited files were necessarily extended by this increment; unrelated inherited edits were preserved.

Final `git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/PairedTrainingObjectiveEvaluation.hpp
 M Sources/PairedTrainingObjectiveEvaluationRepository.cpp
 M Sources/SchedulerCore/ExperimentScheduler.cpp
 M Sources/SchedulerCore/ProductionSchedulerRuntimeInternal.hpp
 M Tests/PairedTrainingObjectiveEvaluationRepositoryTests.cpp
?? LSTM_ExperimentPairComparisonReporting_Output.md
?? LSTM_ExperimentPairComparison_ReadOnlyCLI_Output.md
?? Sources/ExperimentPairComparison.cpp
?? Sources/ExperimentPairComparison.hpp
?? Sources/ExperimentPairComparisonCommand.cpp
?? Sources/ExperimentPairComparisonService.cpp
?? Sources/ExperimentPairComparisonService.hpp
?? Tests/ExperimentPairComparisonServiceTests.cpp
?? Tests/ExperimentPairComparisonServiceTests.sh
?? Tests/ExperimentPairComparisonTests.cpp
?? Tests/ExperimentPairComparisonTests.sh
```

Final `git diff --stat`—which does not include untracked files:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj            | 22 +++++++++++++
 Sources/PairedTrainingObjectiveEvaluation.hpp      |  4 +++
 ...PairedTrainingObjectiveEvaluationRepository.cpp | 34 +++++++++++++++++--
 Sources/SchedulerCore/ExperimentScheduler.cpp      | 38 ++++++++++++++++++++++
 .../ProductionSchedulerRuntimeInternal.hpp         |  2 ++
 ...dTrainingObjectiveEvaluationRepositoryTests.cpp | 24 ++++++++++++++
 6 files changed, 122 insertions(+), 2 deletions(-)
```