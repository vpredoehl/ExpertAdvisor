---
title: "TG4 Isolated Production Streaming Adapter"
document_type: "architecture review"
status: "final"
generated_from: "TG4_Isolated_Production_Streaming_Adapter_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# TG4 Isolated Production Streaming Adapter

Implemented the isolated production TG1→TG3 streaming adapter.

- New [adapter](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/TG4ProductionStreamingPulseAdapter.hpp) consumes canonical completed `Feature` bars and returns same-bar `[inner_break_any, eligible, confluent]` pulses.
- TG3 now exposes immutable same-call observation snapshots in [CausalFibonacciConfluenceIntegration.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/CausalFibonacciConfluenceIntegration.hpp), preventing retention/outcome changes from affecting emitted pulses.
- Added focused adapter and read-only canonical historical replay tests, plus [adapter documentation](/Volumes/Developer%20SSD/ExpertAdvisor/docs/phases/target-generation/PhaseTG4/TG4_ISOLATED_PRODUCTION_STREAMING_ADAPTER.md).

Semantics are exactly:

- No break: `[0,0,0]`
- Ineligible Inner break: `[1,0,0]`
- Eligible non-confluent: `[1,1,0]`
- Eligible confluent: `[1,1,1]`

Pulses are keyed to the completed bar’s start timestamp `T`, never `T+1`. Multiple events on one bar are OR-aggregated, order-independently. UTL-only policy is retained; DTL events are structurally ineligible and cannot become confluent.

Determinism findings: repeated replay has byte-identical serialized pulse rows; prefix pulses remain identical when replayed with a future tail. Outcome retention/finalization cannot rewrite returned pulse values.

Historical replay: the new read-only test loads `eurusdrmp` through the canonical shared reader for `[2025-03-05 05:00Z, 2025-03-06 05:00Z)`, verifies 96 canonical bars across two reads, and produces identical pulses. No evaluator parity assertion was manufactured because the frozen evaluator emits terminal outcome records rather than one pulse row per input bar.

Tests/builds passed:

- `TG4ProductionStreamingPulseAdapterTests.sh`
- `TG4ProductionStreamingCanonicalReplayTests.sh`
- TG1A, TG1B, TG2, TG3, TG4 causal-boundary, TG4 pre-integration, and canonical market-data parity tests
- LSTM feature-vector parity and model-input compatibility tests
- `xcodebuild -target MarketDataCore -configuration Release build`

The full Release scheme compiled in an isolated derived-data directory and then failed only at the expected clean-worktree provenance guard. No executable was launched.

Remaining non-causal issue: `LSTMInputWidthExpansionTests.sh`’s compiled assertions pass, but its final literal source-text check expects an obsolete CLI parser spelling. This predates and is unrelated to this change.

Confirmed: width remains **77**, semantic layout remains **7**, and no Tensor channels, frozen artifacts, database writes, scheduler/worker actions, or experiment changes were made. The repository is ready for a separately approved Tensor layout-8/width-80 integration phase, but none was implemented here.

`git status --short` shows the intended 9-file change set; `git diff --stat` reports tracked edits of `35 insertions, 8 deletions` (new files are untracked and therefore excluded from that command).

Proposed commit message:

`TG4: add isolated production streaming pulse adapter`