---
title: "Phase 22T Detached Persisted-Model Materialization Foundation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase22T_DetachedPersistedModelMaterializationFoundation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Phase 22T Detached Persisted-Model Materialization Foundation

Implemented the detached materialization foundation and report.

- Added owned reader/applier APIs in [PgModelIO.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/PgModelIO.hpp).
- Extended the focused persistence fixture with post-commit application, heads/objective/optimizer/marker/provenance/calendar checks, and deterministic two-connection repeatable-read coverage.
- Updated its build-product path for stable DerivedData.
- Created [Phase 22T report](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_Phase22T_DetachedPersistedModelMaterializationFoundation_Output.md).

Passed:

- `Tests/LSTMInputWidthExpansionPersistenceTests.sh`
- `bash Tests/LSTMInputWidthExpansionTests.sh`
- `bash Tests/LSTMModelInputCompatibilityTests.sh`
- Debug `LSTM Release` build
- `git diff --check`

The mandated clean Release build was blocked by the existing provenance script because Release builds require a clean Git tree and this implementation is uncommitted:

`GenerateBuildProvenance.py: Release provenance requires a clean source tree`

Final disposition is therefore:

`Phase 22T detached persisted-model materialization foundation: NO-GO`

`git status --short`:

```text
 M Headers/PgModelIO.hpp
 M Tests/LSTMInputWidthExpansionPersistenceTests.cpp
 M Tests/LSTMInputWidthExpansionPersistenceTests.sh
?? LSTM_Phase22T_DetachedPersistedModelMaterializationFoundation_Output.md
```