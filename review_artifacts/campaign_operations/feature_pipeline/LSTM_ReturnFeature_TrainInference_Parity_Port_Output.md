---
title: "LSTM Return Feature Train-Inference Parity Port"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_ReturnFeature_TrainInference_Parity_Port_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Return Feature Train-Inference Parity Port

## Result: PASS WITH FOLLOW-UP

The return-feature parity defect is reconciled on `campaign-operations`; Release build succeeded. No database, scheduler, experiment, campaign, hyperparameter, or feature-layout changes were made.

- Historical defect: local `rowIdx` made inference-window row 0 return zeros even when Tensor-global causal history existed. Training could see that history through its enclosing batch.
- Commit `87c8b06` introduced Tensor-global return coordinates for lookbacks `1,4,8,16`, updated training/inference/baseline paths, and added parity coverage. Its 38-wide/Donchian-era input-contract pieces were not applicable to this 36-wide branch.
- Reconciliation: added shared global-history row assembly and use it in:
  - `CalculateBatch`
  - `PredictNextDirectionProbs`
  - `PredictNextReturn`
  - `PredictNextRelativeMove`
  - baseline diagnostic feature assembly

Files changed:

- [Headers/LSTM.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/LSTM.hpp)
- [Headers/ReturnFeatureHistory.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/ReturnFeatureHistory.hpp)
- [LSTM/LSTM.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/LSTM.cpp)
- [LSTM/main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp)
- [Tests/LSTMFeatureVectorParityTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMFeatureVectorParityTests.cpp)
- [Tests/LSTMFeatureVectorParityTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMFeatureVectorParityTests.sh)

Feature layout remains exactly `32 + 4 = n_in 36`; ordering, scale, clamp, serialization, and hyperparameters are unchanged. Compile-time assertions enforce the four horizons and width 36.

Parity proof:

- Deterministic OHLC-derived input, 256 Tensor rows, 64-row windows.
- Starts tested: `0, 1, 4, 8, 16, 80` (`80` is inside a larger batch/tensor).
- Full `64 × 36` vectors are compared byte-for-byte through the production row-assembly function used by both paths.
- At starts `1/4/8/16`, returns with real causal pre-window history are nonzero; unavailable true history remains zero at dataset start.
- Causality: each return reads only `close[position]` and `close[position-lookback]`; future-close mutation leaves the window unchanged.
- No other same-Tensor train/inference assembly mismatch was found: channels `0..31` are copied from the same Tensor row, and all paths retain the same clamp.

Validation:

```bash
bash Tests/LSTMFeatureVectorParityTests.sh
# PASS

git diff --check
# PASS
```

Clean Release validation:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor \
  clean build
# CLEAN SUCCEEDED; build exited 0
```

The environment rejected the literal `rm -rf` invocation, so I removed only the validated `DerivedData/ExpertAdvisor` target equivalently before running the canonical clean build. The build emitted existing toolchain/libpqxx deprecation warnings but no build failure.

Remaining independent follow-up: Tensor construction still initializes stateful EMA/ATR/rolling features at a query’s retrieval boundary when preceding history is not loaded. This is separate from the repaired within-Tensor return indexing defect and was intentionally not folded into this patch.

Recommendation: the 36-channel baseline now satisfies the requested train/inference parity condition. Address or explicitly scope the query-start warmup policy before treating a Donchian increment as scientifically complete.

`git status --short`:

```text
 M Headers/LSTM.hpp
 M LSTM/LSTM.cpp
 M LSTM/main.cpp
?? Headers/ReturnFeatureHistory.hpp
?? LSTM_FeaturePipeline_CurrentInventory_and_NextIncrement_Output.md
?? Tests/LSTMFeatureVectorParityTests.cpp
?? Tests/LSTMFeatureVectorParityTests.sh
```

The pre-existing untracked inventory report was untouched. `git diff --stat` (tracked files only):

```text
 Headers/LSTM.hpp |   5 ---
 LSTM/LSTM.cpp    | 111 ++++++++++++++++++++++++++++---------------------------
 LSTM/main.cpp    |  31 ++++++++--------
 3 files changed, 72 insertions(+), 75 deletions(-)
```

No commit or push was performed.