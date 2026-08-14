---
title: "LSTM Return Feature Train Inference Equivalence Correction Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FeaturePipeline_ReturnFeature_TrainInferenceEquivalence_Correction_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Return Feature Train Inference Equivalence Correction Implementation

## Implementation status: blocked only on Release provenance

Root cause confirmed: return lookbacks used local window offsets. They now use the observation’s Tensor-global position in training, all inference variants, and baseline diagnostics.

Changed:

- `Headers/ReturnFeatureHistory.hpp` — shared global-position causal return helper.
- `Headers/LSTM.hpp`, `LSTM/LSTM.cpp` — training, direction, regression, and relative-move paths pass `w.begin() - t.begin() + rowIdx`.
- `LSTM/main.cpp` — diagnostic baseline uses `ex.globalStart + windowRow`.
- `Tests/LSTMFeatureVectorParityTests.cpp/.sh` — new focused parity regression.

The authoritative coordinate is the zero-based position in the bound `Tensor`; return `h` reads only positions `globalPosition` and `globalPosition - h`. Startup rows remain zero. Return order remains `1, 4, 8, 16`; formulas, scaling, and downstream clamp behavior are unchanged.

Tests passed:

- `Tests/LSTMFeatureVectorParityTests.sh`
- `Tests/LSTMModelInputCompatibilityTests.sh`
- `Tests/DonchianFeatureTests.cpp` standalone compile/run
- `xcodebuild -scheme "LSTM Debug" -configuration Debug ... build` — succeeded, compiling the changed production sources.

The new test covers the prior offset-16 failure with all four nonzero returns, startup boundaries, no look-ahead, ordering, clamp boundary behavior, byte-identical 38-wide rows, and 36/38 projection compatibility.

Release validation command was run exactly as requested, but failed before compilation:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
```

The worktree was already dirty from the operator’s untracked baseline-review file, and this patch is intentionally uncommitted. I did not alter the worktree or commit to bypass the release-provenance guard. Therefore `RELEASE_BUILD=BLOCKED`; all other listed acceptance items are established as PASS.

No new feature, schema, Campaign Operations, or H4 change was made. Existing query-start prehistory behavior for EMA/ATR/Donchian remains a separate limitation; the appended-return coordinate defect is corrected.

`git status --short`:

```text
 M Headers/LSTM.hpp
 M LSTM/LSTM.cpp
 M LSTM/main.cpp
?? Headers/ReturnFeatureHistory.hpp
?? LSTM_FeaturePipeline_CurrentBaseline_and_MinimalIncrement_Review_Output.md
?? Tests/LSTMFeatureVectorParityTests.cpp
?? Tests/LSTMFeatureVectorParityTests.sh
```

`git diff --stat`:

```text
 Headers/LSTM.hpp |  5 ++---
 LSTM/LSTM.cpp    | 67 +++++++++++++++++++++++---------------------------------
 LSTM/main.cpp    | 31 +++++++++++++-------------
 3 files changed, 45 insertions(+), 58 deletions(-)
```

(The new helper and tests are untracked, so excluded from Git’s standard diff stat.)