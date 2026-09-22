---
title: "TG4 Causal Boundary Closure"
document_type: "architecture review"
status: "final"
generated_from: "TG4_Causal_Boundary_Closure_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# TG4 Causal Boundary Closure

Boundary closure is documented in [TG4_CAUSAL_BOUNDARY_CLOSURE.md](/Volumes/Developer%20SSD/ExpertAdvisor/docs/TG4_CAUSAL_BOUNDARY_CLOSURE.md) and the existing contract was narrowly updated.

Findings:

- An event for completed bar `T` maps to Tensor row `i` for that same bar-start timestamp `T`; never `i+1`.
- Normal LSTM and TG4 bars are not yet integration-identical: normal loading includes the right endpoint, TG4 is half-open. The DST probe found 197 vs. 196 rows, differing only at the endpoint.
- Production must use a source-owned immutable TG1–TG3 configuration payload/hash; no runtime dependency on frozen TG4 artifacts.
- TG1 fractal/candidate and TG3 A/B retention bounds can change future pulse bits, so they are semantic configuration. Outcome-record eviction cannot rewrite an emitted row.
- Pulse replay is deterministic and invariant to future outcome/finalization processing when captured immediately from current-bar TG3 updates.
- The repository is not ready for a production-connected adapter until market-range parity and production configuration ownership are implemented. Tensor remains width=77/layout=7.

Files changed:

- [TG4_CAUSAL_BOUNDARY_CLOSURE.md](/Volumes/Developer%20SSD/ExpertAdvisor/docs/TG4_CAUSAL_BOUNDARY_CLOSURE.md)
- [TG4_CAUSAL_LSTM_FEATURE_CONTRACT.md](/Volumes/Developer%20SSD/ExpertAdvisor/docs/TG4_CAUSAL_LSTM_FEATURE_CONTRACT.md)
- [TG4CausalBoundaryClosureTests.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/TG4CausalBoundaryClosureTests.cpp)
- [TG4CausalBoundaryClosureTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/TG4CausalBoundaryClosureTests.sh)
- [TG4MarketDataParityProbe.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/TG4MarketDataParityProbe.sh)

Passed focused tests:

- `Tests/TG4CausalBoundaryClosureTests.sh`
- `Tests/TG4MarketDataParityProbe.sh`
- `Tests/TG1ACausalFractalTrendLineGeometryTests.sh`
- `Tests/TG2TrendLineBreakRetestBehaviorTests.sh`
- `Tests/TG3FibonacciConfluenceIntegrationTests.sh`
- `Tests/TG4HistoricalEmpiricalEvaluationTests.sh`
- `git diff --check`

No Xcode build was run because the active scheduler daemon was observed; these isolated tests compile their own fixtures and do not launch LSTM workers.

No frozen artifact, production database state, scheduler/worker state, experiment state, Tensor/layout, or model/checkpoint semantics changed.

`git status --short`:

```text
 M docs/TG4_CAUSAL_LSTM_FEATURE_CONTRACT.md
?? Tests/TG4CausalBoundaryClosureTests.cpp
?? Tests/TG4CausalBoundaryClosureTests.sh
?? Tests/TG4MarketDataParityProbe.sh
?? docs/TG4_CAUSAL_BOUNDARY_CLOSURE.md
```

`git diff --stat`:

```text
docs/TG4_CAUSAL_LSTM_FEATURE_CONTRACT.md | 34 ++++++++++++++++++++------------
1 file changed, 21 insertions(+), 13 deletions(-)
```

(The standard diff stat excludes the new untracked files above.)

Proposed commit message:

```text
docs: close TG4 causal integration boundaries
```