# LSTM TG2 trend-line break, retest, and empirical behavior measurement

## Disposition

**IMPLEMENTATION COMPLETE — PENDING CLEAN RELEASE VALIDATION**

Baseline branch and commit: `lstm-feature-development` at
`ed7d2e909015ea1177d0fdf6dc37c64aedecf6a0` (TG1B), above TG1A
`7058413`.

The implementation and source-level validation are complete. A normal
provenance-valid Release build must follow a commit so the final clean-source
gate can validate the new source state. No provenance check was weakened or
bypassed.

## Files

- `Headers/CausalTrendLineBreakRetestBehavior.hpp`
- `Tests/TG2TrendLineBreakRetestBehaviorTests.cpp`
- `Tests/TG2TrendLineBreakRetestBehaviorTests.sh`
- `docs/PhaseTG/TG2-Causal-Trend-Line-Break-Retest-Behavior.md`
- `docs/PhaseTG/LSTM_TG2_TrendLineBreakRetestBehavior_Output.md`

TG1A and TG1B production headers were not modified.

## Audited semantics

TG1A/TG1B provide causal candidate geometry, directionally signed live
distance, stable lifecycle identity, and frozen creation-time class, but
explicitly do not define break, retest, reversal, S/R break, or formation
takeout. The PostgreSQL candlestick function is OHLC aggregation only. The
historical-level proximity feature produces a private aggregate proximity
value, not stable S/R level events. The legacy `pocket` database routine is
retrospective and future-searching.

Consequently TG2 exposes named neutral measurement policies rather than
silently claiming supplied semantics. S/R break-to-reversal and candlestick
formation-to-takeout-to-reversal are deferred. Their exact prerequisites are
stable causal level/formation identities, known/confirmation time, directional
break or takeout component/threshold, reversal component/threshold, finite
horizon, and lifecycle/invalidation rules.

## Implemented behavior

The default break is completed close strictly beyond the projected line and
absolute nonnegative tolerance; an explicitly named completed-wick alternative
is available. A new candidate's first observation establishes state without a
break. One event is emitted on valid-to-broken transition, consecutive broken
bars do not double count, and a later completed valid-side bar explicitly
re-arms the candidate.

A retest is wick contact with the same candidate's current projection,
strictly after the break, within a finite inclusive horizon. It records contact
and latency only; it does not infer rejection, continuation, entry, or trade.

Inner-to-Outer pairing is same-direction, coexisting, frozen-Outer, nearest on
the broken side, and still beyond the complete break candle at pairing time.
Tie-breaking is stable candidate identity. Target contact uses the appropriate
wick extreme, explicit absolute tolerance, a finite inclusive horizon, and no
future information. Unpaired is structurally ineligible, never failure.

Overall Inner-to-Outer and retest-then-Outer are separate. The conditioned
outcome requires Outer contact on a completed bar later than the retest, so a
single candle cannot supply an invented intrabar ordering.

## Counts and probabilities

Every study reports eligible, resolved, successes, failures, censored,
pending, and structurally ineligible counts as applicable. The only rate is:

```text
successes / (successes + failures)
```

and it is absent for a zero denominator. Censored/pending/unpaired counts are
not silently dropped or counted as failures. No supplied 90% figure exists in
code as a probability, prior, threshold, expected result, or acceptance
criterion. No empirical market-population claim was made because this
increment supplies the causal measurement machinery, not an authorized
symbol/timeframe/date sample study.

## Causality, determinism, and bounds evidence

The focused tests prove completed-bar UTL/DTL break timing, no pre-break event,
episode deduplication and explicit re-arm, no break-bar retest, exact and
adjacent retest tolerance behavior, exact retest horizon, causal deterministic
pairing, exact Outer target contact, success/failure/censor/unpaired separation,
conditioned aggregation, raw numerator/denominator math, UTL/DTL symmetry,
immutable event identity/class/time under future bars, historical/streaming
event-prefix parity, and unchanged TG1B identity at every integrated prefix.

State is bounded by explicit active and retained observation caps. Capacity
eviction is deterministic explicit censoring and cumulative aggregates remain
complete. The 50,000-bar focused stream retained no more than 16 observations,
no more than 8 active observations, and produced the deterministic expected
25,000 episode count.

## Validation record

Completed:

- `Tests/TG2TrendLineBreakRetestBehaviorTests.sh` — PASS under C++20,
  `-Wall -Wextra -Werror -pedantic`; representative 50,000-bar run passed in
  119 ms with 16 retained, 1 active, and 25,000 deterministic break episodes.
- `Tests/TG1ACausalFractalTrendLineGeometryTests.sh` — PASS, including its
  50,000-bar bound/performance test and database fractal semantic audit.
- `Tests/TG1BTrendLineAngleClassificationTests.sh` — PASS.
- `Tests/LSTMInputWidthExpansionTests.sh` — PASS.
- `Tests/LSTMModelInputCompatibilityTests.sh` — PASS; compile-time checks
  retain `feature_size == 73` and current model input width 77.
- `Tests/FeatureAblationPairEvaluationTests.sh` — PASS.
- `Tests/FeatureAblationReplicationEvaluationTests.sh` — PASS.
- `Tests/SemanticWorkerRegistryTests.sh` — PASS.
- `Tests/SchedulerStatusProcessRecognitionTests.sh` — PASS.
- `Tests/LSTMCausalHistoricalLevelProximityTests.sh` — its stock invocation
  could not link because its hard-coded
  `DerivedData/Development/Build/Products/Debug` directory is absent. The same
  strict compile/link/test command was rerun against the existing
  `DerivedData/ExpertAdvisor/Build/Products/Debug` MetaNN/MetalBuffer products
  and passed. This is a pre-existing test-harness product-path dependency, not
  a TG2 failure.
- TG2 AddressSanitizer + UndefinedBehaviorSanitizer run — PASS. Apple ASan
  reports leak detection unsupported on this platform, so `detect_leaks=0`
  was used; address and undefined-behavior instrumentation remained enabled.
- Clang static analyzer (`clang++ --analyze`) — PASS with an empty diagnostics
  array.
- Repository/source audit and read-only scheduler status — PASS; active
  scheduler PID 61531 and two managed training workers were observed. No
  process or experiment was altered.
- Final `git diff --check` and explicit trailing-whitespace scan of all new
  files — PASS.
- `clang-format --dry-run --Werror` — unavailable because `clang-format` is
  not installed; strict compiler warnings and static analysis supplied the
  available source validation.

## Semantic and non-production evidence

The implementation is an isolated header and standalone test. It has no
include or call-site in `Tensor`, `FeatureLayout`, model input preparation,
feature ablation, scheduler, worker, registry, or publication code. Semantic
layout remains 7 (`kModelInputSemanticLayoutVersion == 7`) by construction;
physical `feature_size` remains 73 and current model input width remains 77.
No TG2 include exists outside the focused test. The related regressions above
passed, and `git diff` shows no scheduler, worker, registry, model-input, layout,
or ablation source change. The read-only scheduler inspection showed normal
active ownership and managed workers before validation; final process
inspection found the same scheduler PID 61531 and worker PIDs 95926/95985.
No production process was intentionally disturbed.

## Release build and repository state

Release build status: not run, pending the required committed clean-source
provenance validation. The Release provenance generator rejects a dirty source
tree, and the requested product path is also the path from which the active
production scheduler is running. Running the build now could replace live
products while two managed training workers are active, contrary to the
non-production constraint. After commit and a safe production window, run the
normal command:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
```

Final `git status --short`:

```text
?? Headers/CausalTrendLineBreakRetestBehavior.hpp
?? Tests/TG2TrendLineBreakRetestBehaviorTests.cpp
?? Tests/TG2TrendLineBreakRetestBehaviorTests.sh
?? docs/PhaseTG/
```

Final `git diff --stat` and `git diff --numstat` are empty because every TG2
file is new and remains untracked pending review/commit. The untracked TG2
footprint is five files and 2,100 lines. No pre-existing tracked file changed,
and the intentional stash list was left untouched.

## TG3 and other deferrals

Fibonacci/confluence, including whether an Outer UTL is within an up-AB
Fibonacci region, remains TG3. Orders, “best entry,” exits, stops, position
sizing, profitability, model labels/targets/features, input-width or layout
changes, scheduler/worker changes, registry changes, and publication remain
strictly outside TG2.
