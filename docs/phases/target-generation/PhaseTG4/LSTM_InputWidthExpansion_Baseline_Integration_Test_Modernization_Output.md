---
title: "LSTM Input-Width Expansion Baseline Integration Test Modernization"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_InputWidthExpansion_Baseline_Integration_Test_Modernization_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Input-Width Expansion Baseline Integration Test Modernization

Modernized the isolated scheduler integration harness; production code was not changed.

- Start: branch `lstm-feature-development`, HEAD `3fa902297e01dea9cfe8a1514fe02b8f518d2c97`, clean worktree.
- Root causes:
  - It replayed 071/078/089/091/092 after cloning an already-current schema; 091 then collided on existing columns.
  - Its current-baseline layout assertions were stale (`77:6` / `6,6` instead of layout 7).
  - `LSTMModelInputIdentityMigrationTests.sql` asserts 089’s pre-092 index shape, so it belongs only in its dedicated predecessor-schema migration harness.
  - The production schema source lacks additive migration 094, which the current queue binary requires; the harness now applies 094 only to its disposable clone.

Changed:

- [LSTMInputWidthExpansionSchedulerIntegrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh)

Behavior:
- Production schema is dumped with a read-only safeguard.
- No replay of incompatible historical migrations.
- Keeps `InputWidthExpansionMigrationTests.sql` as a current-schema contract check.
- Preserves the intentionally historical marker-less width-51/layout-5 fixture.
- Verifies resumed queue records use current layout 7; fresh records require `77:7`.
- Rejects a supplied Release binary unless it embeds the current HEAD commit.

Width remains **77** and semantic layout remains **7**. No TG4 Tensor/model-input work was introduced.

Validation passed:

- `bash Tests/LSTMInputWidthExpansionTests.sh`
- `bash Tests/LSTMModelInputCompatibilityTests.sh`
- `bash Tests/LSTMFeatureVectorParityTests.sh`
- `bash Tests/LSTMModelInputIdentityMigrationTests.sh`
- `bash Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release`
- TG4 pre-integration and pulse-adapter boundary tests
- `git diff --check`

Release build passed using the required DerivedData path. It emitted three existing unrelated unused-symbol warnings in `Sources/LegacyDiagnosticCli.cpp`.

Database safety: only uniquely named temporary databases were mutated and were dropped; production `LSTM` was schema-source only. The active production scheduler was observed but not touched.

Remaining operational item: production lacks migration 094, so it must be applied through the normal migration workflow before using this current queue binary against production. TG4 Tensor/layout-8 integration remains intentionally unimplemented.

`git status --short`:

```text
 M Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh
```

`git diff --stat`:

```text
 Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh | 42 ++++++++++++++--------
 1 file changed, 28 insertions(+), 14 deletions(-)
```

Recommended commit message:

```text
test: modernize input-width expansion scheduler baseline fixture
```