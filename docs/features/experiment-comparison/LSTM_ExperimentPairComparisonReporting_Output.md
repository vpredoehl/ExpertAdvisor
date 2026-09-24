---
title: "LSTM Experiment Pair Comparison Reporting"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_ExperimentPairComparisonReporting_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Experiment Pair Comparison Reporting

Implemented the reusable experiment-pair comparison/reporting layer without interacting with experiments 658/659 or live infrastructure.

1. Repository state

- Actual branch: `lstm-feature-development`
- Worktree was clean before editing.
- Existing architecture inspected:
  - `FeatureAblationPairEvaluation` repository/service/renderer
  - `PairedTrainingObjectiveEvaluation`
  - `InferenceProfitability` result definitions
  - Existing fixture test patterns
- Existing specialized commands were not changed.

2. Implementation

Added a structured, deterministic comparison core that:

- Validates scientific identity before calculating deltas.
- Supports explicitly declared intentional differences such as `feature_ablation_mask`.
- Reports all unexpected identity differences.
- Clearly represents experiment A and B.
- Calculates all deltas as `arm_b - arm_a`.
- Includes inference, acceptance, leader, count, percentage, and profitability metrics.
- Preserves missing observations as `NULL`/incomplete—never zero.
- Avoids fabricated win percentage or average return when actionable count is zero.
- Emits stable identity, metric, reason, and output ordering.
- Produces no winner, score, or recommendation.
- Reuses the existing feature-ablation and profitability evidence types.

3. Files changed

- [ExperimentPairComparison.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentPairComparison.hpp>)
- [ExperimentPairComparison.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentPairComparison.cpp>)
- [ExperimentPairComparisonTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentPairComparisonTests.cpp>)
- [ExperimentPairComparisonTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentPairComparisonTests.sh>)
- [PairedTrainingObjectiveEvaluation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/PairedTrainingObjectiveEvaluation.hpp>)
- [PairedTrainingObjectiveEvaluationRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/PairedTrainingObjectiveEvaluationRepository.cpp>)
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj>)

4. Validation

Passed:

- `Tests/ExperimentPairComparisonTests.sh`
- `bash Tests/FeatureAblationPairEvaluationTests.sh`
- `PairedTrainingObjectiveEvaluationTests`, compiled with `-Wall -Wextra -Werror`
- `git diff --check`, including separate checks for untracked files
- Isolated Debug build:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Debug" \
  -configuration Debug \
  -derivedDataPath "/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development" \
  build

** BUILD SUCCEEDED **
```

The prescribed Release build was attempted in the same isolated directory but correctly stopped at the existing provenance guard:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
** BUILD FAILED **
```

No publication phase ran. The successful Debug build compiled and linked the new comparison source. The build reported existing environment warnings concerning the unavailable `LLVM23.xctoolchain` metadata and the libomp macOS deployment version.

5. Safety confirmation

No live database connection, SQL execution, process inspection, worker interaction, scheduler command, training, inference, experiment evaluation, queue operation, or Release publication occurred.

No schema, migration, semantic-layout, feature-ordering, tensor, objective, checkpoint, scheduler, dispatch, continuation, or training-runtime code was changed.

6. Remaining limitation

This increment intentionally adds the reusable downstream core, not a new database-backed CLI. After 658/659 finish, a read-only service/command can load their completed evidence through the existing repository workflow and pass it to this comparison layer.

7. Final Git state

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/PairedTrainingObjectiveEvaluation.hpp
 M Sources/PairedTrainingObjectiveEvaluationRepository.cpp
?? Sources/ExperimentPairComparison.cpp
?? Sources/ExperimentPairComparison.hpp
?? Tests/ExperimentPairComparisonTests.cpp
?? Tests/ExperimentPairComparisonTests.sh
```

Literal `git diff --stat`—which excludes the four untracked files:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj                 | 8 ++++++++
 Sources/PairedTrainingObjectiveEvaluation.hpp           | 4 ++++
 Sources/PairedTrainingObjectiveEvaluationRepository.cpp | 8 ++++++++
 3 files changed, 20 insertions(+)
```

The four new untracked files contain 1,041 lines total.