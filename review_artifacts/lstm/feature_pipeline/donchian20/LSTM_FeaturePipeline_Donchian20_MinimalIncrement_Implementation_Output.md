---
title: "LSTM Feature Pipeline Donchian-20 Minimal Increment Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_FeaturePipeline_Donchian20_MinimalIncrement_Implementation_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-luna"
---

# LSTM Feature Pipeline Donchian-20 Minimal Increment Implementation

## Disposition

PASS WITH FINDINGS — implementation and required build pass; existing unrelated Release warnings remain.

- Base width: 32 → 34
- Effective width: 36 → 38
- Donchian channels:
  - column 32: upper-envelope distance
  - column 33: lower-envelope distance

Definitions use prior raw highs/lows only and existing `kFeatureScale`.

## Files changed

- [Params.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/Params.hpp)
- [FeatureLayout.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/FeatureLayout.hpp)
- [DonchianFeatures.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/DonchianFeatures.hpp)
- [Tensor.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM/Tensor.cpp)
- [DonchianFeatureTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/DonchianFeatureTests.cpp)

## Behavior

Startup uses the available prior observations. The first row remains zero because no prior extrema exist. Invalid/non-positive inputs produce neutral zero values.

Causality is enforced because `Tensor::Add` computes Donchian values before appending the current row to `raw_high` and `raw_low`.

Training and classification inference both consume the tensor’s base columns in order; existing return-channel appending remains unchanged.

Old models with `n_in = 36` are rejected by existing `model_meta` feature-count validation because runtime width is now 38. No migration logic was added.

## Verification

- Focused test: passed with `-Wall -Wextra -Werror`
- Release build: `** BUILD SUCCEEDED **`
- `git diff --check`: passed
- Existing build warnings: unrelated pre-existing `pqxx::exec_params` deprecations and other warnings

`git status --short`:

```text
 M Headers/Params.hpp
 M LSTM/Tensor.cpp
?? Headers/DonchianFeatures.hpp
?? Headers/FeatureLayout.hpp
?? Tests/DonchianFeatureTests.cpp
```

`git diff --stat`:

```text
 Headers/Params.hpp | 2 +-
 LSTM/Tensor.cpp    | 8 ++++++++
 2 files changed, 9 insertions(+), 1 deletion(-)
```

No additional indicators, architecture changes, tuning, scheduler changes, database schema changes, or model migration logic were introduced.