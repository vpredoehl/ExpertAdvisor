---
title: "Price-Level Structure Phase 5 — Frozen Raw Producer and Replay-Parity Gate"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_PriceLevelStructure_Phase5_FrozenRawProducerReplayParity_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Price-Level Structure Phase 5 — Frozen Raw Producer and Replay-Parity Gate

Stopped at the required design gate; no files were changed.

- Starting/ending HEAD: `2d6ddaef113d51a52dd97fa03ace7bdd3af039af`
- Starting/ending `git status --short`: clean
- Required ancestry verified: all four specified commits are ancestors of HEAD.

The repository has no defensible frozen production Price-Level configuration. `CausalPriceLevelEngine` explicitly has no defaults, and its only concrete configurations are test/bridge fixtures (`{ pivotRadius=1, zoneHalfWidth=0.5, maxActive=4, maxAge=20, evidence=3, duration=900s }`). There is no training/inference producer or documented production selection establishing those as authoritative. The completed Phase 4 materials present in this checkout concern recommendation campaigns, not a Price-Level raw producer/configuration decision.

Therefore freezing those fixture values as model-input production configuration would be guessing, contrary to the gate. The missing prerequisite is an authoritative, versioned production decision naming all six detector fields (or a repository artifact establishing the existing fixture as production policy), including its rationale independent of experiment outcomes.

No producer, Layout 12, Tensor/model changes, scheduler/DB/experiment changes, tests, build, staging, or commit were performed. Layout 11, Tensor width 112, and model width 116 remain unchanged. Phase 6 append-only layout integration is not yet justified.