---
title: "LSTM Experiment Pair Comparison Read-Only CLI"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_ExperimentPairComparison_ReadOnlyCLI_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Experiment Pair Comparison Read-Only CLI

Implemented the generic read-only experiment-pair comparison CLI.

1. Repository state

- Actual branch: `lstm-feature-development`.
- Preserved all preceding uncommitted comparison-core work.
- No reset, clean, restore, checkout, or commit performed.
- The project file already contained preceding-increment edits; this increment only added the new service/command build entries.

2. Architecture reused

The command reuses:

- `FeatureAblationPairEvaluation::LoadAuthoritativeArmEvidence`
- `PairedTrainingObjectiveEvaluation` evidence types/loaders
- `InferenceProfitability::ResolveExactFinalInferenceResult`
- Existing authoritative profitability selection
- `ExperimentPairComparison::MakeArmResultSet`, `Compare`, and `Render`

No new SQL interpretation or metric definition was introduced.

3. Implemented behavior

- Strict positive, distinct `A:B` parsing.
- Argument order remains arm A then arm B.
- One repeatable-read `pqxx::read_transaction`.
- Missing final inference, analysis, or profitability remains incomplete/`NULL`.
- Deltas remain `arm_b - arm_a`.
- The existing deterministic renderer is used unchanged.
- Only `feature_ablation_mask` is automatically intentional, and only when exactly one mask is empty.
- Other scientific differences remain unexpected and incompatible.
- Successful reports—including incomplete/incompatible reports—exit `0`; evidence load failures exit `3`; database failures propagate to the established exit-`2` handler.

4. Command syntax

```text
--compare-experiment-pair=EXPERIMENT_A_ID:EXPERIMENT_B_ID
```

The separated option/value form is also supported consistently with existing CLI conventions.

5. Files changed by this increment

- [ExperimentPairComparisonService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentPairComparisonService.hpp:1>)
- [ExperimentPairComparisonService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentPairComparisonService.cpp:45>)
- [ExperimentPairComparisonCommand.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentPairComparisonCommand.cpp:41>)
- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ExperimentScheduler.cpp:487>)
- [ProductionSchedulerRuntimeInternal.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ProductionSchedulerRuntimeInternal.hpp:134>)
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj>)
- [ExperimentPairComparisonServiceTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentPairComparisonServiceTests.cpp:182>)
- [ExperimentPairComparisonServiceTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentPairComparisonServiceTests.sh:1>)

The preceding core, paired-objective files, report, and core tests were inherited and not changed by this increment.

6. Tests

Passed:

```text
Tests/ExperimentPairComparisonServiceTests.sh
Tests/ExperimentPairComparisonTests.sh
bash Tests/FeatureAblationPairEvaluationTests.sh
bash Tests/PairedTrainingObjectiveEvaluationTests.sh
git diff --check
```

Coverage includes parsing failures, missing experiments, exact-FINAL availability, missing analysis/profitability, zero actionable count, arm order, B−A deltas, deterministic rendering, no winner/recommendation, identity mismatches, feature-ablation recognition, and a no-write adapter contract.

7. Build validation

Passed:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Debug" \
  -configuration Debug \
  -derivedDataPath DerivedData/Development \
  build
```

Result: `BUILD SUCCEEDED`.

The new files produced no warnings. Existing project/environment warnings remain for libpqxx deprecations, LLVM23 toolchain metadata, and the libomp deployment version. No Release build was run.

8. Safety confirmation

No live PostgreSQL access, experiment lookup, worker/process inspection, scheduler action, training, inference, analysis, queueing, database write, migration, semantic-layout change, training-path change, Release publication, or executable invocation against experiments occurred.

9. Remaining limitation

The database-backed repository tests and the command against experiments 658/659 were intentionally not run. Therefore, their actual persisted evidence availability and provenance contracts remain unverified until a later authorized read-only execution.

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/PairedTrainingObjectiveEvaluation.hpp
 M Sources/PairedTrainingObjectiveEvaluationRepository.cpp
 M Sources/SchedulerCore/ExperimentScheduler.cpp
 M Sources/SchedulerCore/ProductionSchedulerRuntimeInternal.hpp
?? LSTM_ExperimentPairComparisonReporting_Output.md
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

`git diff --stat`—tracked files only; Git omits untracked new files:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj            | 22 ++++++++++++++++
 Sources/PairedTrainingObjectiveEvaluation.hpp      |  4 +++
 ...PairedTrainingObjectiveEvaluationRepository.cpp |  8 ++++++
 Sources/SchedulerCore/ExperimentScheduler.cpp      | 30 ++++++++++++++++++++++
 .../ProductionSchedulerRuntimeInternal.hpp         |  1 +
 5 files changed, 65 insertions(+)
```