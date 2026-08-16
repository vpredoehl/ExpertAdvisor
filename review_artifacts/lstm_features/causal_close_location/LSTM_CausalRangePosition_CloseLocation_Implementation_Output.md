---
title: "LSTM Causal Range Position Close Location Implementation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CausalRangePosition_CloseLocation_Implementation_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# LSTM Causal Range Position Close Location Implementation

Implemented the isolated causal prior-bar close-location feature.

- Column 40 added; `feature_size=41`, current `n_in=45`.
- Historical model prefixes through `n_in=44` remain explicit and unchanged.
- Recommendation semantics advanced v11 → v12; v3–v11 remain independently reconstructable.
- No DB/schema, scheduler, campaign-policy, or experiment-config changes.
- No commit made.

Focused LSTM regressions and direct recommendation semantic test passed. `git diff --check` passed. Release build was attempted but stopped at the expected provenance guard because the implementation tree is uncommitted; no source compile failure was reached.

Full implementation record: [LSTM_CausalRangePosition_CloseLocation_Implementation_Output.md](/Volumes/Developer%20SSD/ExpertAdvisor-next-feature/LSTM_CausalRangePosition_CloseLocation_Implementation_Output.md)

`git status --short`: modified implementation/test files plus new close-location header/tests/report; working tree is intentionally dirty.

`git diff --stat`: 18 tracked files changed, 100 insertions, 43 deletions (new untracked files are not included in Git’s diff stat).