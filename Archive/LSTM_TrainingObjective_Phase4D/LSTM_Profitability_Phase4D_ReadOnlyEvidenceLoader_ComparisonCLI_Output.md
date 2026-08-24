---
title: "LSTM Profitability Phase 4D Read-Only Evidence Loader and Comparison CLI"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Profitability_Phase4D_ReadOnlyEvidenceLoader_ComparisonCLI_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Profitability Phase 4D Read-Only Evidence Loader and Comparison CLI

Implemented the read-only Phase 4D evidence loader, comparison service, and CLI.

The complete audit is in [LSTM_Profitability_Phase4D_ReadOnlyEvidenceLoader_ComparisonCLI_Output.md](</Volumes/Developer SSD/ExpertAdvisor/LSTM_Profitability_Phase4D_ReadOnlyEvidenceLoader_ComparisonCLI_Output.md>).

Key behavior:

- Adds `--compare-training-objective-pair=CONTROL_ID:TREATMENT_ID`.
- Requires an explicit primary metric and all seven materiality thresholds.
- Uses one repeatable-read `pqxx::read_transaction`.
- Loads exact experiment, model, inference, final analysis, and profitability provenance.
- Performs no writes and persists no comparison result.
- Returns exit 0 for every scientific disposition, including `INVALID_COMPARISON` and `INCOMPLETE`; exit 3 indicates missing, ambiguous, or invalid persisted evidence.
- Removes the requirement for the non-durable `TRAINING_OBJECTIVE_ACTIVE` log event while still requiring matching objective metadata throughout the materialized model lineage.
- Replaces fixture-only canonical identities with actual persisted metadata primitives.

Validation:

- Pure Phase 4D tests: passed.
- Disposable-database repository/service tests: passed.
- Deterministic output: passed.
- Complete before/after database digest: unchanged.
- Training objective regressions: passed.
- Inference profitability regressions: passed.
- Shell syntax, project lint, focused scheduler compilation, and `git diff --check`: passed.
- Disposable database `ea_phase4d_loader_41848_7733`: dropped successfully.
- Release build used only `DerivedData/Development`; it stopped at the expected dirty-worktree provenance gate with exit 65. The gate was not bypassed.

Remaining verification limits:

- A complete Release build requires a clean worktree.
- Production evidence was not exercised against a live database, as prohibited.
- Operators must supply an approved materiality policy before comparing 599/600.

Files changed include the Xcode project, scheduler CLI, evaluator contract/tests, and the new repository/service source files. Current `git diff --stat`:

```text
7 files changed, 630 insertions(+), 364 deletions(-)
```

The four new production source files and report are untracked and therefore excluded from that Git statistic. Nothing was staged or committed.

```text
LIVE_DATABASE_ACCESSED=false
LSTM_DATABASE_ACCESSED=false
FOREX_DATABASE_ACCESSED=false
DISPOSABLE_DATABASE_USED=ea_phase4d_loader_41848_7733
DISPOSABLE_DATABASE_DROPPED=true
PRODUCTION_DERIVEDDATA_TOUCHED=false
OPERATOR_COMMAND_NOT_EXECUTED=true
```

The exact guarded canonical and Development operator commands are included in the report and clearly marked `OPERATOR_COMMAND_NOT_EXECUTED`.