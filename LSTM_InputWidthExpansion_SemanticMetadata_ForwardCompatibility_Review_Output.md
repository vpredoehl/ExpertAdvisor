---
title: "LSTM Input-Width Expansion Semantic Metadata Forward-Compatibility Review"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_InputWidthExpansion_SemanticMetadata_ForwardCompatibility_Review_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Input-Width Expansion Semantic Metadata Forward-Compatibility Review

## Result

The forward-compatibility defect was real.

Previously, marker-bearing models required:

```text
schema_version == compiled schema version
layout_version == compiled current layout version
```

A future change from width 52/version 1 to width 53/version 2 would reject a valid width-52/version-1 model. Marker-less width-52 models would still pass through the historical-width registry, creating inconsistent behavior.

## Corrected contract

`model_input_semantics_meta` is a logical 1×2 matrix stored under that parameter name in `matrix`:

```text
[metadata_schema_version, semantic_layout_version]
```

Current values are `[1, 1]`.

The semantic layout registry now records:

```text
version 1 → maximum input width 52 → root generation
```

For each future append-only feature, developers must:

1. Preserve all existing width and semantic-layout entries.
2. Register the new input width.
3. Increment `kModelInputSemanticLayoutVersion`.
4. Add a new registry entry pointing to the previous version as its append-only predecessor.

Expansion now requires:

- A registered persisted width.
- Valid integral 1×2 metadata.
- A known semantic version.
- A width that existed in that semantic generation.
- A semantic generation in the current layout’s explicit append-only ancestry.

Marker-less historical models retain their existing registered-width rule. Marker-bearing models receive the stronger width-plus-semantic validation.

Ordinary resume and inference remain unchanged: default `loadAll` does not validate this marker. Validation remains limited to explicit expansion paths.

## Forward proof

The deterministic test creates this simulated future registry:

```text
V1 / W → V2 / W+1 → V3 / W+2
```

It proves that a persisted `V1/W` model remains eligible under both future generations.

It also proves rejection of:

- Unknown semantic version.
- `V1` paired with width `W+1`.
- A later layout not linked to `V1` through append-only ancestry.
- Wrong matrix shape.
- Fractional/malformed metadata.
- Unsupported schema version.

Existing unknown-width, widening-shape, zero-initialization, lineage, retry provenance, ablation, and checkpoint persistence protections remain unchanged.

## Expansion-provenance follow-up

The same exact-current-generation defect also existed in
`input_width_expansion_meta`. Its canonical `semantic_layout` field is now
validated as immutable event-time evidence: the recorded generation must be
known, its registered maximum width must equal the event's expanded width, and
it must occur in the current generation's explicit append-only ancestry. The
parser reconstructs historical events using the registered widths and feature
semantics rather than the current-target-only widening entry point.

Marker-bearing sources now additionally prove that their persisted model width
equals the provenance target, the recorded source model's authoritative width
equals the provenance source width, and the model parent chain reaches that
source. This validation runs for re-expansion as well as current-width retry;
ordinary marker-less resume and inference paths remain unchanged.

Synthetic deterministic coverage proves `V1/52 -> V2/53 -> V3/54`: V1
provenance remains accepted under V2 and V3, a V1/52 model can be re-expanded
under V2, and the descendant records V2 while the source's V1 text remains
unchanged. Unknown, width-incompatible, non-ancestor, noncanonical, wrong
column/name, wrong-policy, and broken-lineage cases remain rejected.

## Review changes

- [ModelInputContract.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/ModelInputContract.hpp:53) — explicit registered historical-width list.
- [ModelInputExpansion.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/ModelInputExpansion.hpp:26) — semantic generation registry, parser, and ancestry validator.
- [PgModelIO.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/PgModelIO.hpp:334) — validates persisted width together with parsed metadata.
- [LSTMInputWidthExpansionTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMInputWidthExpansionTests.cpp:260) — simulated future-width regression and rejection cases.
- [LSTMInputWidthExpansionPersistenceTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMInputWidthExpansionPersistenceTests.cpp:95) — exact persisted 1×2 payload assertion.
- [LSTMInputWidthExpansion.rst](/Volumes/Developer%20SSD/ExpertAdvisor/docs/LSTMInputWidthExpansion.rst:26) — lifecycle and compatibility contract.

## Verification

Passed:

- `bash Tests/LSTMInputWidthExpansionTests.sh`
- `bash Tests/LSTMInputWidthExpansionPersistenceTests.sh`
- `bash Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh DerivedData/ExpertAdvisorExpansion/Build/Products/Debug/LSTM_Debug`
- `bash Tests/LSTMModelInputCompatibilityTests.sh`
- `bash Tests/LSTMFeatureVectorParityTests.sh`
- Isolated Debug `xcodebuild`
- `git diff --check`

The exact required Release build was run without bypasses. It stopped as designed with exit 65:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
```

The current changes compiled successfully in Debug. Release verification remains pending until the repository is legitimately clean.

No production database rows were changed, and no production migrations, experiments, commits, or pushes were performed. Database-backed tests used temporary databases and cleaned them up.

## Repository state

The following status was the snapshot captured by the earlier semantic-marker
review. It predates the expansion-provenance follow-up and final commit; the
final task report is authoritative for the delivered repository state.

`git status --short`:

```text
 M Headers/ModelInputContract.hpp
 M Headers/PgModelIO.hpp
 M LSTM/main.cpp
 M Sources/ExperimentRecommendationRepository.cpp
 M Sources/ExperimentScheduler.cpp
 M Tests/ExperimentRecommendationRepositoryTests.cpp
 M Tests/SchedulerAnalyzeOrphanRecoveryTests.sh
 M Tests/SchedulerContinuationOwnershipIntegrationTests.sh
 M Tests/SchedulerControlWorkerIdentityIntegrationTests.sh
 M Tests/SchedulerOwnershipProcessIntegrationTests.sh
 M Tests/SchedulerRetryFailedCheckpointPromotionIntegrationTests.sh
 M Tests/SchedulerTrainOrphanCheckpointRecoveryTests.sh
 M Tests/WorkerAttemptReconciliationIntegrationTests.sh
?? Database/migrations/071_resume_input_width_expansion.sql
?? Headers/ModelInputExpansion.hpp
?? LSTM_AppendOnlyInputWidthExpansion_Continuation_Implementation_Output.md
?? Tests/InputWidthExpansionMigrationTests.sql
?? Tests/LSTMInputWidthExpansionPersistenceTests.cpp
?? Tests/LSTMInputWidthExpansionPersistenceTests.sh
?? Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh
?? Tests/LSTMInputWidthExpansionTests.cpp
?? Tests/LSTMInputWidthExpansionTests.sh
?? docs/LSTMInputWidthExpansion.rst
```

`git diff --stat`—which excludes untracked files:

```text
13 files changed, 533 insertions(+), 66 deletions(-)
```
