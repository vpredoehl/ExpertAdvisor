---
title: "LSTM Causal Rolling Range Expansion Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CausalRollingRangeExpansion_Implementation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Causal Rolling Range Expansion Implementation

Implemented `rolling_range_expansion` as tensor column 46.

- Base tensor width: 46 → 47
- Full model input width: 50 → 51
- Width-50 models retain a 46-column prefix and cannot consume the new feature.
- Registered formal ablation name: `rolling_range_expansion`.

Focused tests passed, including the new feature, tensor integration/causality, feature-vector parity, model-input compatibility, ablation behavior, volatility regime, and all affected causal-feature runners.

Release build was attempted but correctly blocked by the clean-worktree provenance safeguard; no bypass was used. `git diff --check` passes.

Full implementation report: [LSTM_CausalRollingRangeExpansion_Implementation_Output.md](</Volumes/Developer SSD/ExpertAdvisor-next-feature/LSTM_CausalRollingRangeExpansion_Implementation_Output.md>)

Final assessment: **PASS WITH CAVEAT** — only a clean-worktree Release build remains unverified. No commit created.