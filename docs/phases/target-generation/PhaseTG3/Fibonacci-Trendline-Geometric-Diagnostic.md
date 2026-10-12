# Fibonacci–trendline geometric diagnostic

`Headers/CausalFibonacciTrendlineConfluenceDiagnostic.hpp` provides the standalone,
read-only `EA::TG3::TrendlineDiagnostic::Evaluate` function. It relates numeric
geometry from existing TG3 AB structures and TG1A trendline candidates; it does
not infer relations through equality of unrelated structure keys. No production
consumer, TG4 pulse, tensor column, layout, scheduler, registry, database schema,
or persisted workflow is changed.

## Measurements and policies

Callers must supply an explicit `Configuration(intersectionAtrThreshold)`.
Thresholds are diagnostic hypotheses, with no established profitability or
predictive significance. Default retracement ratios are
`{0.382, 0.5, 0.6180339887498949}`; the evaluator sorts and deduplicates ratios.
Each measurement reports its AB identity, trendline identity, retracement ratio,
and completed decision bar, together with:

- `fibPrice = bPrice + (aPrice - bPrice) * ratio`.
- `linePrice = line.ProjectedPrice(decisionBar)`.
- `normalizedStructureDistance = abs(fibPrice - linePrice) / ATR`.
- `normalizedMarketDistance = abs(close - (fibPrice + linePrice) / 2) / ATR`.

Overflow-safe `std::lerp` and `std::midpoint` compute the corresponding prices.
Nonrepresentable normalized values are unavailable, never NaN/Inf. With missing,
zero, negative, or nonfinite ATR, both normalized values are unavailable and all
configured threshold flags are false. Prices remain available. Thresholds use
inclusive `<=` comparisons. There is no hidden tolerance or rounding of ratios.

`intersectionThresholdMatch` tests structural distance alone.
`marketProximityThresholdMatch` is absent unless a market threshold is supplied;
when configured, unavailable market distance yields false.
`configuredThresholdMatch` requires the intersection threshold and, if supplied,
the market threshold. Even passing both tests describes geometric proximity;
it does not establish market interaction or an outcome.

The default `SourceUTLUpABOnly` policy considers UTL/up-AB pairs.
`SymmetricDirectionalDiagnostic` explicitly adds DTL/down-AB pairs as a research
hypothesis. Cross-direction pairs are omitted. Directions remain in the reported
identities. The production TG4 configuration continues to use its single
`0.6180339887498949` ratio and `SourceUTLUpABOnly` policy.

## Causality, input validation, and capacity

AB availability and both confirmations must be no later than the decision bar.
Trendline creation and both confirmations must be no later than the decision
bar, and `line.currentBar` must equal it exactly. Anchor bars must precede or
equal their confirmations, confirmations must precede or equal availability or
creation, and each pair of anchors must be ordered. AB prices must agree with
their reported direction. Invalid structures, nonfinite geometry, and
nonfinite projected prices are omitted. A nonfinite close, invalid configuration,
or conflicting eligible snapshots sharing one identity throws
`std::invalid_argument`. The caller is responsible for supplying the close and
ATR from the same completed bar and one series; no cross-series or wall-clock
timestamp validation is inferred from this bar-index API.

AB identities include direction, anchors, availability, and their existing
timestamps. Trendline identities reuse `TG2::CandidateIdentity`, including
direction, anchors, creation, and timestamps. Duplicate eligible snapshots with
identical measurement inputs collapse to one identity. Conflicting prices,
slopes, or confirmations for an eligible identity are rejected. Signed zero
ratios canonicalize to positive zero. Repeated calls return the same identities;
exactly-once output applies within a call, not across replay invocations.

Output is ordered lexicographically by AB identity fields (direction, A bar/time,
B bar/time, availability bar/time), then trendline identity fields (direction,
anchor 1 bar/time, anchor 2 bar/time, creation bar/time), then ascending ratio.
The decision bar is common to the call. This is identity order, not a relevance
ranking. It preserves a deterministic prefix under capacity limits.

All bounds must be positive. Oversized input spans and ratio lists are rejected
before filtering or allocation, including duplicates and invalid records. Defaults
allow 512 AB records, 4096 trendlines, and 16 ratios. Pair work is capped at 65536
unique causal pairs examined, including unsupported direction pairs. Output is
capped at 4096 measurements. `pairsExamined`, `pairCapacityReached`, and
`measurementCapacityReached` expose work and truncation; flags indicate a
further pair or ratio was encountered beyond the bound, not merely that the
bound was filled. Check these before interpreting a diagnostic sample.
No product-size allocation is performed. Sorting memory is proportional to the
bounded input spans; pairing work is bounded by the pair and ratio limits.

## Replay example and verification

The confirmed accessors match the implementation prompt. Create one existing
`CausalFibonacciConfluenceIntegration` with explicit TG3/geometry/behavior
configuration, and for each chronological completed candle:

```cpp
const auto update = integration.AddCompletedBar(candle); // exactly once
const auto& geometry = integration.Behavior().Classification().Geometry();
EA::TG3::TrendlineDiagnostic::Configuration diagnostic(0.25); // example hypothesis
diagnostic.marketProximityAtrThreshold = 0.5;                // example hypothesis
const auto result = EA::TG3::TrendlineDiagnostic::Evaluate(
    update.bar, candle.close, geometry.CurrentAtr(),
    integration.ABStructures(), geometry.Candidates(), diagnostic);
// Inspect measurements and capacity flags, or pass them to a research sink.
```

`Tests/CausalFibonacciTrendlineConfluenceDiagnosticTests.cpp` contains an
executable in-memory replay example using this sequence. Run it with:

```sh
bash Tests/CausalFibonacciTrendlineConfluenceDiagnosticTests.sh
```

It verifies arithmetic and threshold edges, direction policies, missing/invalid
ATR, future/stale input suppression, numeric overflow, configuration validation,
identity uniqueness and conflicts, deterministic bounds, unchanged input state,
historical-prefix versus streaming equivalence, and TG4 pulse equality with a
production adapter throughout a 5000-bar replay covering all four pulse states.
Replay uses explicit small work/output caps to verify bounded prefix parity.
It uses no database and starts no workers. Diagnostic consumers require normal
IEEE floating-point semantics; `-Ofast`/`-ffast-math` compilation is unsupported
because it can invalidate finite-input checks. Production build flags are unchanged.

Validation on 2026-10-11 used the focused script above and these existing
regression commands, all passing with their existing warning-as-error settings:

```sh
bash Tests/TG1ACausalFractalTrendLineGeometryTests.sh
bash Tests/TG1BTrendLineAngleClassificationTests.sh
bash Tests/TG2TrendLineBreakRetestBehaviorTests.sh
bash Tests/TG3FibonacciConfluenceIntegrationTests.sh
bash Tests/TG4PreintegrationCausalBoundaryClosureTests.sh
bash Tests/TG4CausalBoundaryClosureTests.sh
bash Tests/TG4ProductionStreamingPulseAdapterTests.sh
```

The Release build was attempted with the installed Xcode 27.0 in a separate
directory to preserve the active scheduler's executable:

```sh
xcodebuild -project ExpertAdvisor.xcodeproj -scheme 'LSTM Release' -configuration Release -derivedDataPath /tmp/ea_fibonacci_trendline_build.IzoznC/DerivedData -jobs 2 build
```

The first attempt lacked prebuilt `MetaNN`/`MetalBuffer` libraries in that fresh
directory. After copying those existing static libraries and Metal assets from
the repository's normal Release product directory, the retry reached the
existing provenance guard and failed with
`Release provenance requires a clean source tree`. No commit was authorized;
the new diagnostic files and the pre-existing `layout13_replication.sql` remain
untracked. The guard was preserved. Full Release build verification remains
blocked by that contract. Existing code/settings also emit Xcode 27 warnings
including deprecated `-Ofast`/libpqxx usage, disabled NaN/Inf semantics, and unused
symbols; these are outside this diagnostic increment. Logs are in
`/tmp/ea_fibonacci_trendline_build.IzoznC/build.log` and `build-retry.log`.

## Proposed h=4 research follow-up

Freeze the ratios, thresholds, directional policy, capacity handling, universe,
and a four-completed-bar outcome definition before examining held-out data.
Select at most one event per series per nonoverlapping four-bar outcome window,
with a predetermined rule for multiple pairs and ratios; avoid treating each
measurement as an independent event. Use chronological training/validation/test
periods with a four-bar purge at boundaries and a final untouched out-of-sample
period. Compare intersection-only and market-near subsets with suitable
nonmatched controls, reporting event counts, uncertainty, and multiplicity from
threshold/ratio exploration. Use existing authoritative evaluation workflows
for any later persisted study. This increment does not perform that study or
claim predictive significance.
