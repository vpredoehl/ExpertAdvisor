---
title: "LSTM Experiment Replication-Wave Planner Read-Only CLI"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_ExperimentReplicationWavePlanner_ReadOnlyCLI_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Experiment Replication-Wave Planner Read-Only CLI

Implemented the controlled read-only replication-wave planner.

1. Exact CLI syntax

```bash
LSTM_Release --plan-experiment-replications=656:657 --replication-seeds=44,45,46
```

Separated forms are also supported:

```bash
LSTM_Release --plan-experiment-replications 656:657 --replication-seeds 44,45,46
```

Argument and seed order are preserved. Options are mutually exclusive with other scheduler command modes.

2. Source-pair validation

The planner requires positive, distinct IDs and loads both arms through `LoadAuthoritativeArmEvidence`. It then requires:

- scientific comparability under `ExperimentPairComparison`;
- an explicit supported intervention;
- identical source seeds;
- complete required scientific identity;
- no unexpected A/B identity differences.

Outcome metrics are not used.

3. Seed semantics

Seeds are strict canonical positive `uint32` values. Empty input, whitespace, signs, leading zeroes, overflow, and duplicates are rejected before database access.

Source-seed reuse is allowed and labeled `same_seed_repeated_pairs`. It has no changed field and is never presented as independent evidence. Every report states `statistical_independence=not_inferred`.

4. Scientific preflight

Each proposal copies the authoritative source-arm identity and replaces only `fresh_initialization_seed`. Exact preflight verifies:

- both arms receive the requested seed;
- only the seed changed relative to each source arm;
- the source intervention is preserved exactly;
- no unexpected A/B difference exists;
- missing required identity produces `undetermined_due_to_missing_evidence`;
- numeric comparisons use exact canonical values.

Implementation: [ExperimentReplicationPlanning.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationPlanning.cpp:225>), domain: [ExperimentReplicationPlanning.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationPlanning.hpp:15>).

5. Authoritative reuse

The planner reuses:

- `FeatureAblationPairEvaluation::LoadAuthoritativeArmEvidence`;
- `ExperimentPairComparison::MakeArmResultSet`;
- `ExperimentPairComparison::MakeComparisonRequest`;
- `ExperimentPairComparison::Compare`;
- `ExperimentReplicationComparison::SeedReplicationModeText`.

It introduces no experiment-row schema, defaults, INSERT definition, or persisted configuration type.

6. Equivalent-experiment detection

The PostgreSQL adapter preselects non-cancelled candidates by authoritative symbol, seed, and exact ablation mask, then compares the existing `ExperimentPairComparison` configured identity exactly. It reports:

- `no_equivalent_experiment_found`;
- `equivalent_experiment_found=<id>`;
- `equivalent_experiment_ambiguous=<ids>`.

Multiple exact candidates or candidates with unavailable identity remain ambiguous. Equivalence is redundancy only and does not invalidate the plan. See [ExperimentReplicationPlanningCommand.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationPlanningCommand.cpp:109>).

7. Read-only transaction behavior

One `pqxx::read_transaction` with repeatable-read isolation covers source loading and all equivalence checks. The adapter contains no work transaction or mutation path. See [ExperimentReplicationPlanningCommand.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationPlanningCommand.cpp:173>).

8. States and exits

- `valid`
- `invalid`
- `undetermined_due_to_missing_evidence`

All successfully rendered states exit `0`. Input/evidence load failures exit `3`. Database exceptions propagate to the established exit-`2` handler.

9. Output

Deterministic line-oriented machine output includes planner/version, source IDs, source seed, ordered requested seeds, exact intervention values, proposed identities, changed fields, preflight results, unexpected differences, equivalence states, and read-only/materialization status. It emits no winner, ranking, recommendation, profitability forecast, significance claim, continuation decision, or queue decision.

CLI help is documented at [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ExperimentScheduler.cpp:12951>).

10. Files changed by this increment

New:

- [ExperimentReplicationPlanning.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationPlanning.hpp>)
- [ExperimentReplicationPlanning.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationPlanning.cpp>)
- [ExperimentReplicationPlanningService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationPlanningService.hpp>)
- [ExperimentReplicationPlanningService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationPlanningService.cpp>)
- [ExperimentReplicationPlanningCommand.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationPlanningCommand.cpp>)
- [ExperimentReplicationPlanningTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentReplicationPlanningTests.cpp:183>)
- [ExperimentReplicationPlanningTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentReplicationPlanningTests.sh>)

Pre-existing dirty files additionally edited for integration:

- `ExpertAdvisor.xcodeproj/project.pbxproj`
- `Sources/SchedulerCore/ExperimentScheduler.cpp`
- `Sources/SchedulerCore/ProductionSchedulerRuntimeInternal.hpp`

All other initial dirty files were preserved.

11. Tests

All passed:

```text
Tests/ExperimentReplicationPlanningTests.sh
Tests/ExperimentReplicationComparisonTests.sh
Tests/ExperimentPairComparisonTests.sh
Tests/ExperimentPairComparisonServiceTests.sh
bash Tests/FeatureAblationPairEvaluationTests.sh
Tests/PairedTrainingObjectiveEvaluationTests.sh
bash Tests/PairedTrainingObjectiveEvaluationRepositoryTests.sh
```

The repository test used and dropped disposable database `ea_phase4d_loader_1792_24924`.

12. Build

Exact requested command succeeded:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Debug" \
  -configuration Debug \
  -derivedDataPath "/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development" \
  build
```

Result: `BUILD SUCCEEDED`.

The build still reports inherited libpqxx deprecation warnings plus local toolchain/libomp environment warnings; the new planner translation units compile cleanly.

13. `git diff --check`

Passed with no output.

14. Live-system safety

No live PostgreSQL connection, experiment query, process inspection, worker signal, scheduler action, or interaction with experiments 656–659 occurred. No training, inference, analysis, queueing, or Release publication occurred.

15. Limitations

The scheduler’s private write-oriented `DuplicateWhereClause` is not directly callable from the read-only adapter. Equivalent detection therefore uses the broader authoritative pair configured identity and is intentionally stricter than queue admission duplicate gating. Missing materialized architecture identity makes equivalence ambiguous rather than guessed.

16. Later materialization boundary

A separately authorized future command should consume the immutable `Plan`, rerun `RecomputePreflight`, translate it through an extracted pure `SchedulerOptions` construction seam, rerun authoritative duplicate checks transactionally, and only then invoke the existing atomic queue workflow. No such path was added here.

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/PairedTrainingObjectiveEvaluation.hpp
 M Sources/PairedTrainingObjectiveEvaluationRepository.cpp
 M Sources/SchedulerCore/ExperimentScheduler.cpp
 M Sources/SchedulerCore/ProductionSchedulerRuntimeInternal.hpp
 M Tests/PairedTrainingObjectiveEvaluationRepositoryTests.cpp
 M Tests/PairedTrainingObjectiveEvaluationRepositoryTests.sh
?? LSTM_ExperimentPairComparisonReporting_Output.md
?? LSTM_ExperimentPairComparison_ReadOnlyCLI_Output.md
?? LSTM_ExperimentPairComparison_ReportingProvenancePolish_Output.md
?? LSTM_ExperimentReplicationComparison_ReadOnlyCLI_Output.md
?? Sources/ExperimentPairComparison.cpp
?? Sources/ExperimentPairComparison.hpp
?? Sources/ExperimentPairComparisonCommand.cpp
?? Sources/ExperimentPairComparisonService.cpp
?? Sources/ExperimentPairComparisonService.hpp
?? Sources/ExperimentReplicationComparison.cpp
?? Sources/ExperimentReplicationComparison.hpp
?? Sources/ExperimentReplicationComparisonCommand.cpp
?? Sources/ExperimentReplicationComparisonService.cpp
?? Sources/ExperimentReplicationComparisonService.hpp
?? Sources/ExperimentReplicationPlanning.cpp
?? Sources/ExperimentReplicationPlanning.hpp
?? Sources/ExperimentReplicationPlanningCommand.cpp
?? Sources/ExperimentReplicationPlanningService.cpp
?? Sources/ExperimentReplicationPlanningService.hpp
?? Tests/ExperimentPairComparisonServiceTests.cpp
?? Tests/ExperimentPairComparisonServiceTests.sh
?? Tests/ExperimentPairComparisonTests.cpp
?? Tests/ExperimentPairComparisonTests.sh
?? Tests/ExperimentReplicationComparisonTests.cpp
?? Tests/ExperimentReplicationComparisonTests.sh
?? Tests/ExperimentReplicationPlanningTests.cpp
?? Tests/ExperimentReplicationPlanningTests.sh
```

`git diff --stat`—which excludes untracked files:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj            |  66 ++++++++++
 Sources/PairedTrainingObjectiveEvaluation.hpp      |   4 +
 ...PairedTrainingObjectiveEvaluationRepository.cpp |  34 +++++-
 Sources/SchedulerCore/ExperimentScheduler.cpp      | 134 +++++++++++++++++++++
 .../ProductionSchedulerRuntimeInternal.hpp         |   7 ++
 ...dTrainingObjectiveEvaluationRepositoryTests.cpp |  26 +++-
 ...edTrainingObjectiveEvaluationRepositoryTests.sh |  12 +-
 7 files changed, 278 insertions(+), 5 deletions(-)
```