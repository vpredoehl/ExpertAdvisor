---
title: "LSTM Input-Width Expansion Provenance Ancestry Final Correction and Commit"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_InputWidthExpansion_Provenance_Ancestry_FinalCorrection_and_Commit_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Input-Width Expansion Provenance Ancestry Final Correction and Commit

Implemented, verified, and committed the append-only input-width expansion increment.

### Result

The defect was real. `ParseInputWidthExpansionProvenance()` required both:

- `semantic_layout == kModelInputSemanticLayoutVersion`
- reconstruction through a plan builder restricted to today’s target width

Both prevented historical expansion provenance from surviving a later append-only generation.

The correction now:

- Validates `(semantic generation, expanded width)` through the shared semantic-generation registry and explicit ancestry.
- Requires the expanded width to match that generation’s registered maximum.
- Reconstructs historical events using registered widths and feature semantics, independent of today’s target.
- Preserves the original event generation; it is never rewritten.
- Validates persisted model width, source model width, canonical metadata, source ancestry, experiment ownership, and retry lineage.
- Validates existing expansion provenance before re-expanding a marker-bearing source.
- Leaves ordinary resume/inference unchanged.

Canonical format:

```text
schema=1;source_model_id=...;source_input_width=...;expanded_input_width=...;new_tensor_columns=BEGIN:END;new_tensor_features=NAME|...;initialization=zero;semantic_layout=V
```

### Compatibility proof

Deterministic tests simulate:

```text
V1 / width 52 → V2 / width 53 → V3 / width 54
```

They prove:

- V1/52 provenance parses under V2 and V3.
- A V1/52 model can source a V2/53 re-expansion.
- The descendant records V2 while the source retains its exact V1 text.
- Historical provenance remains retry/source-valid through the common PgModelIO validation path.
- Unknown, impossible-width, non-ancestor, malformed, noncanonical, wrong-column, wrong-feature, wrong-policy, and broken-lineage metadata fail closed.

The actual repository remains V1/current width 52. No return-autocorrelation or V2/width-53 implementation exists in this worktree.

### Consumers reviewed

- Parser construction and every direct parser call
- PgModelIO save/load and source validation
- Checkpoint and final-model persistence
- Explicit resume/re-expansion
- Scheduler queue validation and current-width retry
- Experiment ownership and recursive parent/source lineage
- Ordinary resume/inference
- Recommendation-source exclusion, which uses `resume_expand_input_width`, not the metadata payload

### Verification

Passed:

- Input-width expansion unit tests
- PostgreSQL persistence/lineage tests
- Scheduler expansion integration tests
- Scheduler retry checkpoint-promotion tests
- Model-input compatibility tests
- Feature-vector parity tests
- Recommendation repository exclusion test
- Historical-level, rolling-range, and multi-bar feature regressions
- Shell syntax checks
- Migration 071 contract tests
- `git diff --check`
- `git diff --cached --check`
- Isolated Debug build
- Exact canonical post-commit Release build:

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  build
```

Result: `BUILD SUCCEEDED`.

The builds still report the repository’s pre-existing warning backlog, primarily deprecated libpqxx `exec_params` calls; the new provenance queries use the current API.

### Commit

Commit:

```text
2b77bb0930c0d75e5725ce17d3004446a9aaf893
Add append-only LSTM input-width expansion continuation
```

The exact staged set was 24 files:

```text
Database/migrations/071_resume_input_width_expansion.sql
Headers/ModelInputContract.hpp
Headers/ModelInputExpansion.hpp
Headers/PgModelIO.hpp
LSTM/main.cpp
LSTM_AppendOnlyInputWidthExpansion_Continuation_Implementation_Output.md
LSTM_InputWidthExpansion_SemanticMetadata_ForwardCompatibility_Review_Output.md
Sources/ExperimentRecommendationRepository.cpp
Sources/ExperimentScheduler.cpp
Tests/ExperimentRecommendationRepositoryTests.cpp
Tests/InputWidthExpansionMigrationTests.sql
Tests/LSTMInputWidthExpansionPersistenceTests.cpp
Tests/LSTMInputWidthExpansionPersistenceTests.sh
Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh
Tests/LSTMInputWidthExpansionTests.cpp
Tests/LSTMInputWidthExpansionTests.sh
Tests/SchedulerAnalyzeOrphanRecoveryTests.sh
Tests/SchedulerContinuationOwnershipIntegrationTests.sh
Tests/SchedulerControlWorkerIdentityIntegrationTests.sh
Tests/SchedulerOwnershipProcessIntegrationTests.sh
Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh
Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh
Tests/WorkerAttemptReconciliationIntegrationTests.sh
docs/LSTMInputWidthExpansion.rst
```

### Remaining operational steps

Not performed:

- Apply migration 071 to production
- Restart/cut over the active scheduler
- Queue an expansion experiment
- Modify production experiment/model rows
- Push the commit

The scheduler and two production training workers remain active and were not interrupted.

Final `git status --short` and `git diff --stat` are both empty.