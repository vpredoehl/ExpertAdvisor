---
title: "TG4 Pre-Integration Causal Boundary Closure"
document_type: "architecture review"
status: "final"
generated_from: "TG4_PreIntegration_CausalBoundary_Closure_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# TG4 Pre-Integration Causal Boundary Closure

Implemented TG4 pre-integration closure. Ready: **YES** for a separately approved, isolated TG1→TG3 streaming-adapter phase only.

- Canonical absolute UTC `[start,end)` contract: [`CanonicalMarketDataRange.hpp`](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/CanonicalMarketDataRange.hpp). Exact right endpoint is excluded; 15-minute NY-civil bar starts are converted at the SQL boundary.
- Legacy model loader behavior is unchanged (still inclusive-right civil-string path). New callers use `MarketData::LoadCanonicalHalfOpenCandlesticks`.
- TG4’s read-only repository now uses the same shared canonical CTE.
- Production configuration: [`ProductionTG1TG3PulseConfiguration.hpp`](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/ProductionTG1TG3PulseConfiguration.hpp), factory `tg4a-derived-source-utl-up-ab-only-v1`; deterministic newline payload + classic-locale hex-float serialization + `fnv1a64:` hash.
- Semantic identity includes TG1 retention/candidate capacities, TG3 retained fractal/A-B capacities, policies, ratios, pip map/tolerance, timeframe/range contract, etc. Outcome retention is separately serialized as operational, not pulse-semantic.
- Frozen TG4A constants are encoded in source; no runtime frozen-artifact dependency.

Files changed include the two new headers, MarketDataCore/TG4 repository, two focused test suites, and the three requested TG4 documents.

Tests passed:

- `TG4PreintegrationCausalBoundaryClosureTests.sh`
- `TG4CanonicalMarketDataParityTests.sh` (read-only DB): ordinary 96, spring DST 196, fall DST 188, weekend-gap 192; zero mismatches/endpoints
- TG1/TG2/TG3 causal regressions
- TG4 causal-boundary regression
- width-77 feature-vector parity, input-width expansion, and model-input compatibility tests
- `xcodebuild -target MarketDataCore ... build`
- TG4 repository syntax check

The full `LSTM Release` build compiled affected targets but failed only at its expected provenance guard because the source tree is uncommitted. No executable, workers, scheduler, experiments, database writes, or frozen artifacts were changed.

Confirmed: width remains 77, layout remains 7, and no Tensor TG channels were added.

`git status --short` shows only this intended worktree change set; `git diff --stat` reports 75 insertions / 74 deletions for tracked files (new untracked files listed in status).

Proposed commit message:

`TG4: close canonical range and production config prerequisites`