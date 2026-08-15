---
title: "LSTM Stateful Feature Warmup Query-Scope Fix"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Stateful_Feature_Warmup_QueryScope_Fix_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Stateful Feature Warmup Query-Scope Fix

## Verdict

**DEFECT CONFIRMED AND FIXED.** `Tensor::Add` previously began EMA/ATR/rolling state at the requested query start, so a boundary timestamp differed from that same timestamp in a wider query.

## Production path

`fromDate/toDate` → `candlestick(...)` → `Tensor::Add` → batches → train `CalculateBatch` or inference `ProcessBatchPredict`.

Now the runtime:

1. Retrieves `candlestick(..., '-infinity', toDate)`.
2. Counts rows with `dt < fromDate`.
3. Builds/warmups `Tensor` from full source history.
4. Starts training, normal inference, and `--infer-all` batching at that logical output index.

The live PostgreSQL `candlestick` function accepts `-infinity`; this was verified read-only.

## Active stateful features

- Columns 0–5: previous-close-relative OHLC/body/range.
- 6–7: 32-bar rolling volatility and cumulative return.
- 14–18, 24–29: EMA 8/21/50 derived values; recursive, so exact values require all available predecessor history.
- 19–23: ATR-14 plus EMA/ATR-derived values; recursive.
- 31: 32-bar rolling raw-range mean.
- Appended 32–35: 1/4/8/16-bar returns.

Columns 8–13 and 30 are formula-stateless, but previously also became zero at a query boundary because `Tensor::Add` zeroes its first row. Full-history warmup fixes that boundary artifact too.

Training and inference share `Tensor` base features and the common global-position return assembler.

## Files changed

- [LSTM/main.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/main.cpp): full-history source query; SQL-derived output boundary; applies it to training, inference, and infer-all.
- [Headers/Tensor.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/Tensor.hpp): `ForEachBatchFrom` keeps warmup rows out of emitted batches.
- [FeatureWarmupScope.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/FeatureWarmupScope.hpp): centralizes the complete-history requirement.
- [LSTMStatefulFeatureWarmupTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMStatefulFeatureWarmupTests.cpp) and its script: boundary invariance, train/inference vector parity, output exclusion, reset isolation, and 36-wide contract.
- [LSTMStatefulFeatureWarmup.rst](/Volumes/Developer%20SSD/ExpertAdvisor/docs/LSTMStatefulFeatureWarmup.rst): operational contract.

No feature order, width, formulas, labels, clamping, or persisted model schema changed.

## Verification

Passed:

```bash
Tests/LSTMStatefulFeatureWarmupTests.sh
Tests/LSTMFeatureVectorParityTests.sh
```

Release build passed:

```bash
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor clean build
```

The build emitted existing toolchain and pqxx deprecation warnings, but completed successfully.

## Residual risk

Full-history loading increases source rows and memory/work for long-lived symbols. The authoritative `candlestick` source is the boundary of available predecessor history; exact recursive EMA/ATR state begins at its earliest returned row.

The running scheduler/training worker remained active; no worker command was launched.

## Repository state

`git status --short` includes pre-existing return-parity work plus this task’s new files. `git diff --stat` reports tracked changes:

```text
 Headers/LSTM.hpp   |   5 ---
 Headers/Tensor.hpp |  14 +++++++
 LSTM/LSTM.cpp      | 111 +++++++++++++++++++++++++++--------------------------
 LSTM/main.cpp      |  66 ++++++++++++++++++++++---------
 4 files changed, 118 insertions(+), 78 deletions(-)
```

Untracked files are omitted from that command’s stat.

It is now safe to proceed with the next feature increment.