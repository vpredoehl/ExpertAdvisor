---
title: "LSTM Phase 21B MarketDataCore Read-Only Candlestick Extraction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase21B_MarketDataCore_ReadOnlyCandlestickExtraction_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Phase 21B MarketDataCore Read-Only Candlestick Extraction

Implemented MarketDataCore extraction and did not commit.

- Added [MarketDataCore.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/MarketDataCore/MarketDataCore.hpp) and [MarketDataCore.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/MarketDataCore/MarketDataCore.cpp).
- Added static Xcode target/linkage to `LSTM Release` (and Debug integration target); scheduler remains independent.
- Routed production and matching diagnostic candlestick discovery/query/materialization through MarketDataCore.
- Preserved caller-owned read-only transactions, warmup counting, `logicalOutputStartIndex`, ordering, quoted SQL values, and outcome coverage checks.
- Added [MarketDataCoreBoundaryTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/MarketDataCoreBoundaryTests.sh).
- Added required report: [LSTM_Phase21B_MarketDataCore_ReadOnlyCandlestickExtraction_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_Phase21B_MarketDataCore_ReadOnlyCandlestickExtraction_Output.md).

Validation:

- MarketDataCore structural test: passed.
- Focused MarketDataCore Release build: passed, exit 0.
- LSTM Debug Release-config integration build: passed, exit 0; linked `-lMarketDataCore`.
- Incremental proof: only `MarketDataCore.cpp`, `libMarketDataCore.a`, and executable link rebuilt; main/Tensor/LSTM/SchedulerCore did not.
- Normal `LSTM Release` build: expected provenance failure because the intentional changes are uncommitted.
- `git diff --check`: passed.

Remaining prerequisite: commit the intentional changes, then run the provenance-gated `LSTM Release` build. No database-backed candlestick fixture was run to avoid touching production data.

`git status --short`:

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M LSTM/main.cpp
?? LSTM_Phase21B_MarketDataCore_ReadOnlyCandlestickExtraction_Output.md
?? Sources/MarketDataCore/
?? Tests/MarketDataCoreBoundaryTests.sh
```

`git diff --stat`:

```text
 ExpertAdvisor.xcodeproj/project.pbxproj | 110 ++++++++++++++++++++++++++++++++
 LSTM/main.cpp                           | 103 ++++++++++++------------------
 2 files changed, 150 insertions(+), 63 deletions(-)
```

Phase 21B MarketDataCore extraction: GO WITH PREREQUISITES