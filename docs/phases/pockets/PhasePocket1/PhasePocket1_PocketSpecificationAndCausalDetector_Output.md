---
title: "Phase Pocket 1 Pocket Specification and Causal Detector"
document_type: "architecture review"
status: "final"
generated_from: "PhasePocket1_PocketSpecificationAndCausalDetector_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Phase Pocket 1 Pocket Specification and Causal Detector

Completed the source-grounded rerun. The manual supports a 15-candle default indicator lookback and directional high/low references, but not a complete causal creation predicate, price-endpoint formula, or confirmation timing. No Pocket detector was implemented.

Changed and staged (not committed):

- [Pocket specification](/Volumes/Developer%20SSD/ExpertAdvisor/docs/phases/pockets/PhasePocket1/PocketSpecificationAndCausalDetector.md)
- [Phase output](/Volumes/Developer%20SSD/ExpertAdvisor/PhasePocket1_PocketSpecificationAndCausalDetector_Output.md)

Key finding: the prior 21-candle / fixed 30-pip / ten-level interpretation is unsupported. The manual’s touch/close labels are documented with page traceability, while their exact executable semantics remain unspecified. Reversion and breakout are explicitly kept separate from Pocket structure.

Validation:

- `git diff --check` — passed
- `git diff --cached --check` — passed
- No code changed, so no build or synthetic code tests were run, avoiding TG4 contention.
- `ResearchSources/Pockets/Pockets.pdf` remains ignored and unstaged.
- No TG4 artifacts/processes, database objects, historical data, scheduler, or Phase 21C files were touched.

`git status --short`:

```text
A  PhasePocket1_PocketSpecificationAndCausalDetector_Output.md
A  docs/phases/pockets/PhasePocket1/PocketSpecificationAndCausalDetector.md
```

`git diff --cached --stat`:

```text
2 files changed, 278 insertions(+)
```

Recommended Phase Pocket 2: obtain an authoritative clarified rule for trigger, boundaries, confirmation latency, and touch/close semantics; then implement a pure causal supplied-bar detector with focused synthetic tests.