---
title: "Phase 22U Resume + Direct-Inference Detached Materialization Migration"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase22U_ResumeDirectInferenceDetachedMaterializationMigration_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Phase 22U Resume + Direct-Inference Detached Materialization Migration

Implemented Phase 22U migration.

- Resume and direct selected-model inference now materialize one `REPEATABLE READ, READ ONLY` detached snapshot, commit before Tensor/LSTM work, then use the query-free applier.
- Snapshot diagnostics/validation no longer reread selected-model correctness state.
- Added owned model name, ancestry, linked experiment objective, and detached calendar identity use.
- Infer-all and scheduler final/checkpoint paths remain unchanged.

Files changed:

- [PgModelIO.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/PgModelIO.hpp)
- [main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp)
- [LSTMPhase22UResumeDirectInferenceMigrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMPhase22UResumeDirectInferenceMigrationTests.sh)
- [Phase 22U report](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_Phase22U_ResumeDirectInferenceDetachedMaterializationMigration_Output.md)

Validation passed:

- Debug `LSTM Release` build
- `Tests/LSTMInputWidthExpansionPersistenceTests.sh`
- `bash Tests/LSTMInputWidthExpansionTests.sh`
- `bash Tests/LSTMModelInputCompatibilityTests.sh`
- `Tests/LSTMPhase22UResumeDirectInferenceMigrationTests.sh`
- `git diff --check`

An active scheduler was detected; no production worker or experiment row was touched. The Debug build emitted one pre-existing unrelated warning in `CampaignOperationsProductionAdmissionService.cpp`.

Final disposition:

`Phase 22U resume + direct-inference detached materialization migration: IMPLEMENTATION COMPLETE — PENDING POST-COMMIT RELEASE VALIDATION`

`git status --short`:

```text
 M Headers/PgModelIO.hpp
 M LSTM/main.cpp
?? LSTM_Phase22U_ResumeDirectInferenceDetachedMaterializationMigration_Output.md
?? Tests/LSTMPhase22UResumeDirectInferenceMigrationTests.sh
```

`git diff --stat`:

```text
 Headers/PgModelIO.hpp |  65 +++++--
 LSTM/main.cpp         | 488 ++++++++++++++++++++++++++++++++++++++++----------
 2 files changed, 445 insertions(+), 108 deletions(-)
```