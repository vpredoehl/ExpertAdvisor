# TG1B trend-line angle calibration and classification

TG1B is a diagnostic classification layer over the TG1A causal fractal
trend-line geometry. Its implementation is
`Headers/CausalFractalTrendLineAngleClassification.hpp`. It does not feed
`Tensor`, alter `FeatureLayout`, change feature ablation, change semantic
layout 7, or publish/register a worker.

## TG1A audit

The TG1B implementation audited the TG1A types, construction path, ATR update,
retention, diagnostics, and focused tests. No TG1A defect was found and no
TG1A field or behavior was changed.

TG1A's canonical slope is price change per sequential completed-bar index:

```text
rawSlopePerBar = (anchor2Price - anchor1Price) /
                 (anchor2Bar - anchor1Bar)
```

Low anchors must rise strictly to form a UTL, and high anchors must fall
strictly to form a DTL. The second anchor remains unavailable until its
five-candle fractal confirms at `anchor2Bar + 2`. Candidate creation occurs on
that confirmation bar. TG1A's optional ATR-normalized slope is:

```text
atrNormalizedSlope = rawSlopePerBar / currentAtr
```

`currentAtr` is a completed-bar Wilder recursive ATR with `alpha = 1 / period`,
seeded by the first true range. TG1A updates it with the current completed bar
before detecting confirmations and creating candidates. TG1A intentionally
recomputes that diagnostic on every later observation, so its live value is
dynamic. Candidate retention and deterministic eviction remain exactly as
documented by TG1A.

## Normalized coordinate convention

A chart-screen angle is not an intrinsic market quantity: it changes with
quote precision, panel size, axis range, zoom, and pixel aspect ratio. TG1B
therefore defines an angle only in explicit normalized coordinates.

For signed creation-time ATR-normalized slope `n` and explicitly supplied
positive finite reference-bar scale `S`:

```text
normalized rise over reference horizon = abs(n) * S
calibrated angle magnitude              = atan(abs(n) * S) * 180 / pi
```

The vertical unit is one ATR as known on the candidate-creation bar. The
horizontal unit represents `S` sequential completed bars. Thus `S` is a
documented coordinate calibration, not an empirical claim or hidden display
constant. `S = 1` means one completed bar per horizontal unit, but callers must
still pass that choice explicitly; TG1B has no implicit calibration default.
Changing `S` deterministically changes the reported angle and can change its
class. The scale is emitted in every diagnostic so results with different
calibrations cannot be mistaken for each other.

The input to `atan` is dimensionless: raw price units cancel against ATR and
the remaining per-bar rate is evaluated over the explicit reference-bar
horizon. The result is independent of quote magnitude, chart pixels, zoom,
and display state. It remains timeframe-specific in the scientifically honest
sense that a bar is the selected timeframe's bar; diagnostics include the
timeframe, and empirical calibration should stratify or compare timeframes.

UTL/DTL direction stays separate from steepness. Diagnostics preserve the
signed normalized slope, while angle and class use its absolute magnitude.
Equal-magnitude positive and negative normalized slopes therefore produce the
same angle/class. A missing, NaN, or infinite normalized slope produces an
unavailable angle and `Unclassified`. A nonpositive or non-finite calibration
scale is rejected.

## Exact classification contract

Endpoints are inclusive and no nearest-band assignment occurs:

| Classification | Angle magnitude |
|---|---:|
| `LongTerm` | 12 degrees through 20 degrees |
| `Outer` | 25 degrees through 40 degrees |
| `Inner` | 45 degrees through 85 degrees |
| `Unclassified` | below 12, `(20,25)`, `(40,45)`, above 85, or unavailable/invalid |

These are geometric labels only. In particular, `Unclassified` is a normal
outcome rather than an error or an instruction to choose the nearest class.

## Causal freeze convention

TG1B classifies once, on the second anchor's confirmation/creation bar. It
copies the raw slope, that completed bar's ATR, signed ATR-normalized slope,
explicit calibration scale, angle magnitude, and class into an immutable
classification record. The ATR includes bars only through the confirmation
bar, inclusive. No earlier classification record exists because no causal
candidate exists before that confirmation.

Later completed bars continue to update TG1A's observation-time distance,
ATR, and normalized-slope diagnostics, but they do not rewrite the TG1B
creation record. This stable convention treats steepness class as part of the
candidate's geometric identity. TG1B mirrors TG1A candidate expiry and
deterministic cap eviction, without changing those policies.

Historical ingestion stable-sorts by timestamp and calls the same TG1B
streaming update path. Duplicate timestamps remain rejected by TG1A. Focused
tests compare classifications after every common prefix, verify that future
ATR changes do not alter a fixed classification, and independently reproduce
the creation-bar Wilder ATR.

## Calibration/distribution support

`CausalFractalTrendLineAngleClassification` requires a global
`CalibrationConfiguration` at construction. It intentionally does not add a
symbol/timeframe override table. A future empirical layer can select scales by
symbol/timeframe before constructing an engine, with its own persisted and
versioned fallback policy, without changing the transformation.

For deterministic distribution collection, consumers can read
`Update::newlyClassifiedCandidates` as an append-only event stream at candidate
creation. `ClassifiedCandidates()` provides the currently live, TG1A-bounded
view. `FormatDiagnostic` emits:

- symbol and timeframe;
- UTL/DTL direction;
- both anchor bars, timestamps, prices, and confirmation bars/timestamps;
- creation bar/timestamp and anchor separation;
- unchanged TG1A raw slope per bar;
- creation ATR and signed creation ATR-normalized slope;
- calibration reference-bar scale;
- calibrated angle magnitude and classification; and
- the explicit `second_anchor_confirmation` timing convention.

Representative output:

```text
symbol=EURUSD,timeframe=1h,direction=UTL,...,raw_slope_per_bar=0.4,creation_atr=10.6793679825,creation_atr_normalized_slope=0.0374554000438,calibration_reference_bar_scale=20,calibrated_angle_magnitude_degrees=36.8371746562,classification=Outer,classification_timing=second_anchor_confirmation
```

The example scale demonstrates the mechanism; it is not a recommended global,
symbol, or timeframe calibration. Choosing or validating scales requires
empirical distribution analysis outside this increment. Point-in-time use must
select a predeclared calibration and must never estimate it from future bars.

## Scope boundary

TG1B introduces no break/retest behavior, Inner-to-Outer retracement rule,
probability claim, support/resistance or candlestick reversal rule, Fibonacci
confluence, label, target, entry, exit, profitability logic, model-input
feature, scheduler behavior, or worker publication. Those remain TG2/TG3 or
later empirical integration work.
