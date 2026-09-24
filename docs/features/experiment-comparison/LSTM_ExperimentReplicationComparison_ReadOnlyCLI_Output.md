---
title: "LSTM Experiment Replication Comparison Read-Only CLI"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_ExperimentReplicationComparison_ReadOnlyCLI_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Experiment Replication Comparison Read-Only CLI

Implemented the read-only multi-pair replication comparison facility.

## CLI

```text
--compare-experiment-replications=A_ID:B_ID,C_ID:D_ID
```

Separated option/value syntax is also supported. Pair and arm ordering remain exactly as supplied; deltas are always `arm_b - arm_a`. Experiment IDs must be positive and globally unique across all pairs.

## Behavior

- Reuses `ExperimentPairComparison`, its authoritative loaders, exact-final resolution, provenance validation, and metric definitions.
- Loads all experiments in one repeatable-read transaction.
- Compatibility states:
  - `compatible`
  - `incompatible`
  - `undetermined_due_to_missing_evidence`
- Requires each pair to be individually scientifically comparable.
- Requires identical intentional fields and exact arm-A/arm-B intervention values across pairs.
- Compares invariant configured scientific identities exactly; known materialized model/execution identities are also compared exactly.
- Missing configured input identity without a final-model fallback produces `undetermined_due_to_missing_evidence`.
- Missing final outcome evidence remains incomplete/`NULL`, not zero or incompatible.
- Aggregates are suppressed unless compatibility is established.

Seed handling follows the discovered persisted identity contract:

- A/B seed mismatch makes that pair scientifically incompatible.
- Seeds may differ across pairs.
- Output distinguishes:
  - `different_seed_replications`
  - `same_seed_repeated_pairs`
  - `unavailable_or_not_applicable`
- Output always states `statistical_independence=not_inferred`.

For every existing pair metric, compatible reports include supplied count, available count, ordered deltas, unweighted descriptive mean, minimum, maximum, and positive/zero/negative counts. Output includes `subjective_winner=NONE`.

## Files changed by this increment

New:

- [ExperimentReplicationComparison.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationComparison.hpp>)
- [ExperimentReplicationComparison.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationComparison.cpp>)
- [ExperimentReplicationComparisonService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationComparisonService.hpp>)
- [ExperimentReplicationComparisonService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationComparisonService.cpp>)
- [ExperimentReplicationComparisonCommand.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentReplicationComparisonCommand.cpp>)
- [ExperimentReplicationComparisonTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentReplicationComparisonTests.cpp>)
- [ExperimentReplicationComparisonTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentReplicationComparisonTests.sh>)

Extended:

- [ExperimentPairComparison.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentPairComparison.hpp>) and [ExperimentPairComparison.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentPairComparison.cpp>): shared authoritative metric catalog.
- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ExperimentScheduler.cpp>) and [ProductionSchedulerRuntimeInternal.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/SchedulerCore/ProductionSchedulerRuntimeInternal.hpp>): CLI parsing, help, exclusivity, dispatch.
- [project.pbxproj](</Volumes/Developer SSD/ExpertAdvisor/ExpertAdvisor.xcodeproj/project.pbxproj>): Debug target sources.
- Repository test harness files: support safe PostgreSQL socket paths under the required derived-data directory containing spaces.

All pre-existing pair-comparison, reporting/provenance, and paired-evidence changes were preserved.

## Verification

Passed:

```text
Tests/ExperimentReplicationComparisonTests.sh
Tests/ExperimentPairComparisonTests.sh
Tests/ExperimentPairComparisonServiceTests.sh
Tests/FeatureAblationPairEvaluationTests.sh
Tests/PairedTrainingObjectiveEvaluationTests.sh
Tests/PairedTrainingObjectiveEvaluationRepositoryTests.sh
git diff --check
```

The repository test used and dropped an isolated disposable PostgreSQL database. Initial harness attempts exposed socket-path quoting/length problems; those were corrected, and the final run passed with `DISPOSABLE_DATABASE_DROPPED=true`.

Debug build passed:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Debug" \
  -configuration Debug \
  -derivedDataPath DerivedData/Development \
  build
```

Result: `BUILD SUCCEEDED`. Existing unrelated libpqxx, LLVM23 metadata, and libomp deployment warnings remain.

No live database connection, experiment 656–659 lookup, scheduler/process inspection, migration, training, inference, analysis, queueing, or Release publication occurred.

Remaining limitation: the motivating live experiments were intentionally not invoked, so their current evidence and compatibility remain unverified.

## Final worktree

Branch: `lstm-feature-development`

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
?? Tests/ExperimentPairComparisonServiceTests.cpp
?? Tests/ExperimentPairComparisonServiceTests.sh
?? Tests/ExperimentPairComparisonTests.cpp
?? Tests/ExperimentPairComparisonTests.sh
?? Tests/ExperimentReplicationComparisonTests.cpp
?? Tests/ExperimentReplicationComparisonTests.sh
```

Tracked-file `git diff --stat`—Git omits untracked new files:

```text
7 files changed, 188 insertions(+), 5 deletions(-)
```