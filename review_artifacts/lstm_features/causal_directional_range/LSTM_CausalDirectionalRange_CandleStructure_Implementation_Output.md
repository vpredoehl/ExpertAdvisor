---
title: "LSTM Causal Directional Range Candle Structure Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CausalDirectionalRange_CandleStructure_Implementation_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Causal Directional Range Candle Structure Implementation

Implemented the append-only causal directional body/range feature and v11 recommendation semantics.

- Column 39 uses prior completed OHLC only; `feature_size=40`, current `n_in=44`.
- Historical model prefixes through `n_in=43` remain exact.
- v3–v10 canonical reconstruction remains explicit; v11 is default.
- Added focused [directional-range tests](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/Tests/LSTMCausalDirectionalRangeTests.cpp) and report: [implementation output](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM_CausalDirectionalRange_CandleStructure_Implementation_Output.md).

Passed:

- `Tests/LSTMCausalDirectionalRangeTests.sh`
- All six requested LSTM regression suites
- Strict standalone `ExperimentRecommendationTests.cpp` compile/execution
- `git diff --check`

Build:

- Debug `xcodebuild` was attempted but blocked before source compilation by the existing provenance script’s DerivedData sandbox write denial. No source compile error was reported. No Release qualification claimed.

Remaining caveat: full Xcode build requires resolving the local DerivedData/provenance-script permission issue.

`git status --short`: 17 modified files plus 4 new files (feature header, focused test/source script, output report); no commit created.

`git diff --stat`: 17 tracked files changed, 99 insertions, 41 deletions (untracked additions excluded by Git’s default diff stat).