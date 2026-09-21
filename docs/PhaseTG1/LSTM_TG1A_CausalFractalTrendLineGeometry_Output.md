---
title: "LSTM TG1A Causal Fractal Trend-Line Geometry"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_TG1A_CausalFractalTrendLineGeometry_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM TG1A Causal Fractal Trend-Line Geometry

TG1A is implemented as causal, bounded geometry infrastructure. Focused tests pass, but I cannot claim the full Release acceptance criterion because the repository’s clean-tree provenance gate rejected the requested build before compilation.

## Architecture and behavior

TG1A is a header-only C++20 geometry engine, intentionally separate from `Tensor` and production model features.

- Historical input is timestamp-sorted and then processed through the same streaming path.
- Five-candle fractals use strict comparisons.
- Anchor `i` is unavailable until completed bar `i+2`.
- UTLs require strict higher fractal lows; DTLs require strict lower fractal highs.
- All qualifying retained pairs coexist.
- Intervening-price and touch tolerances are separate, explicit absolute price-unit values.
- Candle and confirmed-fractal touches are counted independently.
- Raw projection uses bar-index separation.
- Optional ATR-normalized diagnostics use the existing Wilder recursive convention.
- No later cross deletes a candidate; break interpretation remains out of scope.

Default deterministic bounds:

- 512-bar anchor lookback
- 64 confirmed fractals per kind
- 512-bar candidate lifetime
- 4096 live candidates
- Oldest deterministic eviction only if the explicit cap is exceeded

A 50,000-bar optimized validation completed in 109 ms with 16 retained fractals and 128 configured live candidates.

## Files added

- [CausalFractalTrendLineGeometry.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/CausalFractalTrendLineGeometry.hpp>)
- [TG1ACausalFractalTrendLineGeometryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/TG1ACausalFractalTrendLineGeometryTests.cpp>)
- [TG1ACausalFractalTrendLineGeometryTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/TG1ACausalFractalTrendLineGeometryTests.sh>)
- [TG1A-Causal-Fractal-Trend-Line-Geometry.md](</Volumes/Developer SSD/ExpertAdvisor/docs/TG1A-Causal-Fractal-Trend-Line-Geometry.md>)

No database, Fibonacci, model-layout, Tensor, ablation, or semantic-worker files were changed.

## Database audit

- `Database/indicators/fractal.plpgsql` uses strict `max(high) < high` and `min(low) > low`.
- Its window is chronological: `ORDER BY dt ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING EXCLUDE CURRENT ROW`.
- The source field is `dt` in `Database/forex/candlestick.plpgsql`; the positional fractal output field is named `ts`.
- The requested candlestick file is therefore under `Database/forex`, not `Database/indicators`.
- `current_high` and `current_low` call an unbounded historical fractal calculation and filter afterward, so they are not suitable as point-in-time causal contracts. TG1A does not consume them.
- Legacy fixed-year `fractal_mv` behavior was documented but left unchanged.

## Test coverage

The new suite covers:

- Strict high/low fractals and equal-value rejection
- Explicit `i+1` unavailability and `i+2` confirmation
- Preservation of the original anchor bar
- Timestamp ordering independent of row insertion order
- Valid/invalid UTL and DTL creation
- Intervening-price rejection and tolerated boundary cases
- Exact projection
- Separate candle/fractal touch counts
- Third-fractal availability only upon confirmation
- No trend-geometry look-ahead
- Historical/streaming prefix parity
- Repeated-run ordering determinism
- Multiple simultaneous candidates
- Configured cap eviction and candidate expiration
- 50,000-bar bounded streaming behavior
- Database fractal semantic assertions

Representative output:

```text
symbol=EURUSD,timeframe=1h,direction=UTL,anchor1_bar=2,anchor1_price=5,anchor1_confirmation_timestamp=2023-11-15T02:13:20Z,anchor2_bar=7,anchor2_price=7,anchor2_confirmation_timestamp=2023-11-15T07:13:20Z,separation_bars=5,raw_slope_per_bar=0.4,atr_normalized_slope=0.0374554000438,age_bars=0,candle_touch_count=0,fractal_touch_count=2,projected_price=7.8,raw_price_to_line_distance=2.2,atr_normalized_distance=0.206004700241
```

## Validation results

Passed:

- `Tests/TG1ACausalFractalTrendLineGeometryTests.sh`
- Optimized `-O3 -Wall -Wextra -Werror` TG1A build and run
- Clang static analyzer
- `SchedulerSemanticAdmissionTests.sh`
- `SemanticWorkerRegistryTests.sh`
- `FeatureAblationPairEvaluationTests.sh`
- Compiled `LSTMInputWidthExpansionTests` binary
- TG1A-scoped `git diff --check`
- Semantic-layout non-change check

The older `LSTMInputWidthExpansionTests.sh` wrapper fails only because it greps the obsolete `Sources/ExperimentScheduler.cpp` path; the code now resides under `Sources/SchedulerCore`. Its compiled test passes.

Requested Release command:

```text
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
```

Result:

```text
GenerateBuildProvenance.py: Release provenance requires a clean source tree
** BUILD FAILED **
```

The failure occurred before compilation. The installed environment reports Xcode 27.0 rather than the documented 26.5. Two production training workers were active and were not interrupted; no publication or scheduler restart occurred.

## Scope confirmations

- No angle transformation or Long-Term/Outer/Inner classification
- No Fibonacci changes
- No trading, break/retest, probability, or label logic
- Semantic layout remains version 7
- Existing feature layout, input width, registry, and worker identities are unchanged

Remaining TG1B considerations are calibration of explicit tolerances by symbol/timeframe and collection of representative real-data geometry. A clean committed worktree is still required for final Release-build acceptance.

## Repository status

```text
 M Database/utilities/check_tables.plpgsql
?? .dukascopy-cache/
?? AUDCAD_2023_Dukascopy_Data_Integrity_Certification.zip
?? Headers/CausalFractalTrendLineGeometry.hpp
?? LSTM_Phase23D_CleanReleasePublicationAndPhase23ClosureAcceptance_Output.md
?? LSTM_Phase23E_CanonicalReleasePublicationHarnessRepairAndPhase23Closure_Output.md
?? Tests/TG1ACausalFractalTrendLineGeometryTests.cpp
?? Tests/TG1ACausalFractalTrendLineGeometryTests.sh
?? docs/Phase23/LSTM_Phase23D_CleanReleasePublicationAndPhase23ClosureAcceptance_Output.md
?? docs/TG1A-Causal-Fractal-Trend-Line-Geometry.md
```

Exact `git diff --stat`—which excludes untracked TG1A files:

```text
 Database/utilities/check_tables.plpgsql | 430 ++++++++++++++++++++++++++++++--
 1 file changed, 413 insertions(+), 17 deletions(-)
```

TG1A adds four files totaling 1,229 lines. The existing database modification and other untracked artifacts were preserved untouched.