# Causal Fibonacci extension and conditional-transition research contract

## Scope and freeze status

This document freezes the measuring instrument identified in code as
`causal-fibonacci-extension-h1-h2-policy-frozen-v2`. Version 2 retains the
version-1 geometry and prospectively freezes the first H1/H2 evaluation
policy before any Fibonacci-extension outcome rate is inspected. It is research
infrastructure only. It does not add a Tensor field, alter model input width or
semantic layout, change training/inference/analyze/scheduler behavior, define a
trade, or authorize a historical outcome run.

The hypotheses came from an experienced trader with decades of market
experience. They are expert priors, not measured facts:

- **H1 — 1.272 continuation to 1.618:** if price moves beyond the 1.272
  Fibonacci extension, it will continue to the 1.618 extension. This is a
  qualitative directional prior; it has no invented numeric benchmark.
- **H2 — 1.272 rejection to 0.382:** if price reaches 1.272 and subsequently
  reverses, the expert benchmark probability of reaching the 0.382 pullback is
  approximately 80 percent.
- **H3 — post-D, post-1.272 pullback:** after a valid D extension is causally
  established and price subsequently reaches 1.272, the expert benchmark
  probability of a pullback to 0.382 or 0.500 is approximately 80 percent.

The 80 percent values are retained only as `expert_benchmark=0.8` for H2 and
the three H3 reporting endpoints. They are not truth, a threshold, an
acceptance test, or an expected result. A later measured estimate and its
measured-minus-benchmark difference remain separate from this prior.

No outcome-bearing historical extension result or rate was inspected to choose
this contract. Ratios, formulas, tolerance semantics, confirmation rules,
event ordering, horizons, and failure rules were not optimized. The extension
ratios are exactly `1.272` and `1.618`; pullbacks are exactly `0.382` and
`0.500`; `0.618` is retained only as the explicitly secondary descriptive
endpoint already grounded in the retained harmonic/TG3 methodology.

## Repository audit

### TG1A confirmation

`Headers/CausalFractalTrendLineGeometry.hpp` defines strict radius-two
fractals over five completed candles. A pivot centered at bar `i` is emitted
only on completed bar `i+2`. `ConfirmedFractal` records anchor bar/time/price
and confirmation bar/time. A candidate made from two same-kind confirmed
fractals is created at the second anchor's confirmation time. Inputs require
finite valid OHLC and unique increasing timestamps.

### TG3 A/B selection and availability

`Headers/CausalFibonacciConfluenceIntegration.hpp` defines the named
`MostRecentPriorOppositeConfirmedFractal` convention. A newly confirmed high B
selects the most recent earlier confirmed low A with `B > A` for an UpAB; a
newly confirmed low B selects the most recent earlier confirmed high A with
`B < A` for a DownAB. A must have an earlier anchor and be confirmed no later
than B. The immutable A/B identity contains direction, both anchor bars/times,
and B's availability bar/time. The A/B becomes available only at B
confirmation. Later pivots do not rewrite it. Selection for an observation may
use only a matching structure whose availability is no later than the
observation bar.

This increment consumes `TG3::ABStructure` directly. It does not add a second
swing selector or retrospectively choose a better leg.

### TG3 Fibonacci and tolerance behavior

TG3 intentionally has only `FibonacciLevelType::Retracement`. For
`R = abs(B-A)` it calculates:

```text
UpAB retracement(r)   = B - rR
DownAB retracement(r) = B + rR
```

Retracement ratios must be finite in `[0,1]`; the A/B range must be finite and
positive. TG3 confluence uses an inclusive finite nonnegative absolute-price
tolerance: `abs(observed-level) <= tolerance`. Its default directional study
is source UTL/up-AB only, with down-AB symmetry explicitly opt-in. Existing TG3
retracement code and tests are unchanged.

### TG4/TG4A reusable research conventions

TG4 separates frozen causal capture fields from later outcomes, uses explicit
score/outcome periods, treats end-of-input and capacity loss as censoring,
retains raw numerators and denominators, and computes two-sided Wilson 95
percent intervals on `successes/(successes+failures)`. Its event identities and
configuration fingerprints use deterministic FNV-1a-64 identity checksums;
artifacts use stable schemas and maximum round-trip floating precision. TG4A's
pre-study freeze records provenance and forbids confirmation data from
parameter selection. This increment reuses TG4's `Wilson95` implementation and
those artifact principles without connecting extensions to model features.

### Reusable horizon and tolerance conventions

The audit found no extension-specific horizon or structural invalidation rule.
It did find one directly reusable prospective outcome convention: TG2 defines
both retest and target windows as 20 completed bars after causal eligibility,
inclusive of the twentieth bar, and the frozen TG4A configuration retains both
20-bar values. TG2 checks contact before deadline failure, treats an unfinished
window as right-censored, and has no price-based structural invalidation for
the comparable target outcome. Version 2 therefore reuses exactly that narrow
convention for H1 and every H2 endpoint. It does not infer a new market-cycle,
volatility, opposite-anchor, or stop boundary.

The first study also reuses TG4A's prospectively frozen one-canonical-FX-pip
inclusive tolerance convention: `0.0001` for the five canonical non-JPY pairs
and `0.01` for `usdjpyrmp`. This is an existing measurement convention, not a
value selected from extension outcomes. A different instrument universe or
tolerance requires a new prospective contract version.

### D/C-D terminology

No authoritative D point, D extension, C-D leg, XABCD structure, or causal D
confirmation contract exists in the audited source, headers, tests, scripts,
or target-generation documentation. Other uses of the English word
“extension” refer to unrelated stop, campaign, or architectural extensions.
The trading-method context alone is not specific enough to select C, D, or an
anchor convention. H3 therefore fails closed as described below.

### Historical-study and artifact conventions

TG4 uses terminal per-event CSV rows, stable schema versions, deterministic
ordering, explicit structural eligibility/pending/censored/resolved counts,
event-weighted and cohort views, equal-symbol views, Wilson intervals, data
quality artifacts, and immutable causal snapshots. TG2/TG3 use strictly
increasing event sequences and explicit capacity censoring rather than silent
drops. TG4's historical event identity includes symbol, timeframe, candidate,
event timestamp, and sequence. This increment instead keys one structural
extension opportunity by the stable TG3 A/B identity so repeated bars from the
same move cannot become independent extension observations.

## Geometry contract

Let `R = abs(B-A)` and `s = +1` for UpAB and `-1` for DownAB. The typed
extension formula is:

```text
extension(r) = A + s * rR
```

Equivalently:

```text
UpAB extension(r)   = A + r(B-A) = B + (r-1)R
DownAB extension(r) = A - r(A-B) = B - (r-1)R
```

This is the direct continuation of the directed TG3 A-to-B leg: ratio `1.0`
is exactly B and ratios greater than one extend past B in the leg direction.
It does not overload TG3 retracement semantics. Each level records its type,
ratio, exact price, tolerance zone, source A/B identity, and source
availability bar/time. Non-finite anchors, ratios, prices or zones,
direction-inconsistent anchors, a degenerate range, a range inconsistent with
the anchors, and causally inconsistent confirmations are rejected.

Pullback levels deliberately call TG3's existing retracement calculation, so:

```text
UpAB pullback(r)   = B - rR
DownAB pullback(r) = B + rR
```

The names `0.382`, `0.500`, and `0.618` therefore mean percentage retraced
from B under the already accepted TG3 convention. They are not relabeled
coordinates from a different charting package.

## Causal event-state contract

All input bars are completed OHLC bars. An event is assigned the completed
bar's timestamp; no intrabar ordering is claimed. Nothing is observable before
the source A/B availability bar/time. As in TG3's same-call ingestion order,
the B-confirmation bar may be classified at its completed-bar timestamp,
because both the structure and the completed OHLC are then observable.

The primary definitions are intentionally minimal:

- **1.272 touched/reached:** for UpAB, completed-bar high is at or above the
  lower edge `level-tolerance`; for DownAB, completed-bar low is at or below
  the upper edge `level+tolerance`. This inclusive directional reach also
  handles a completed-bar gap through the zone.
- **1.272 moved beyond:** completed close is strictly beyond the far edge: for
  UpAB, `close > level+tolerance`; for DownAB,
  `close < level-tolerance`. A touch and a close-beyond are not synonyms.
- **1.272 rejection/reversal confirmed:** only after a 1.272 touch, a later
  completed bar closes through the near edge back toward B: for UpAB,
  `close < level-tolerance`; for DownAB, `close > level+tolerance`. A touch bar
  cannot confirm its own reversal.
- **1.618 reached after beyond:** the directional high/low reach rule is
  applied only on a completed bar strictly later than the eligible
  close-beyond bar. A same-bar 1.618 wick is not counted because its order
  relative to the close-beyond condition is unknowable.
- **0.382/0.500/0.618 reached after rejection:** UpAB uses
  `low <= level+tolerance`; DownAB uses `high >= level-tolerance`, only on a
  completed bar strictly later than rejection confirmation.

These are primary contract definitions, not variants selected from results.
No alternative close/wick grid has been added. A future sensitivity study must
name and freeze a new contract version prospectively rather than select an
alternative after comparing outcomes.

H1 becomes eligible only at close-beyond and succeeds only at a later 1.618
reach. H2 becomes eligible only at confirmed rejection and succeeds only at a
later 0.382 reach. H2's 0.500 and secondary 0.618 contacts are separately
retained as descriptive endpoints and cannot replace H2.

## H3 D dependency and fail-closed behavior

The schema reserves typed D point/confirmation fields and separate H3 0.382,
0.500, and `0.382 OR 0.500` outcomes. However, code supplies no operation that
can assert a valid D. `d_contract_status` is
`missing_authoritative_definition`, D fields remain empty, and all H3
endpoints are `ambiguous_or_ineligible` with the reason
`missing_authoritative_d_extension_contract`.

Human review must first freeze the source legs, D price formula, causal D
confirmation observation, post-D 1.272 anchor/formula, invalidation, and
deduplication relation. A later version must preserve the original H3 expert
benchmark alongside measured estimates. A later D cannot be backfilled into a
prior 1.272 event.

## Frozen H1 evaluation policy

- **Eligibility/event start:** the completed bar that first closes strictly
  beyond the far 1.272 tolerance edge establishes eligibility. Outcome
  observation begins on the next completed bar.
- **Success:** a directional 1.618 reach at the inclusive near tolerance edge
  on bars `eligibility+1` through `eligibility+20`. A 1.618 wick on the
  close-beyond bar is not a success because its intrabar order is unknowable.
- **Failure and horizon:** failure means no qualifying 1.618 reach by the close
  of bar `eligibility+20`. The target is evaluated before deadline failure, so
  an exact-boundary hit on bar 20 succeeds; a first hit on bar 21 is too late
  and is not retained as an endpoint hit.
- **Structural invalidation:** none. No opposite anchor, B crossing, rejection,
  volatility state, or later pivot terminates H1.
- **Right censoring:** an eligible event whose dataset or study window ends
  before bar 20, without success, is right-censored and never counted as a
  failure.
- **Ambiguity:** the frozen policy has no invalidation boundary, so
  target/invalidation ambiguity cannot arise. The generic synthetic-test
  infrastructure retains fail-closed `ambiguous_or_ineligible` behavior if a
  separately versioned future policy supplies an invalidation and the same OHLC
  bar reaches both it and the target.
- **Direction symmetry:** UpAB uses `high >= 1.618 - tolerance`; DownAB uses
  `low <= 1.618 + tolerance`.
- **Latency:** bars to target is `target_bar - close_beyond_bar`; the minimum is
  one and the maximum under this policy is 20.
- **Boundary finalization:** dataset exhaustion uses `end_of_dataset`; the
  exclusive study boundary uses `study_window_boundary`. Both are censoring.

## Frozen H2 evaluation policy

- **Qualifying rejection:** after a directional 1.272 touch, a strictly later
  completed bar must close through the near edge toward B. UpAB requires
  `close < 1.272 - tolerance`; DownAB requires
  `close > 1.272 + tolerance`.
- **Outcome start:** the rejection-confirmation bar establishes eligibility;
  observation begins on the next completed bar. A pullback wick on the
  rejection bar does not count.
- **Endpoints:** primary 0.382, descriptive 0.500, and secondary/descriptive
  0.618 are evaluated and retained independently. UpAB uses
  `low <= level + tolerance`; DownAB uses `high >= level - tolerance`.
- **Failure and horizon:** each endpoint fails independently if it has not been
  reached by the close of `rejection_bar+20`. Exact-boundary contact on bar 20
  succeeds; first contact on bar 21 is too late and is not retained as an
  endpoint hit. A deeper endpoint does not
  replace, optimize, or redefine the primary 0.382 endpoint.
- **Structural invalidation:** none, for the same prospective reason as H1.
- **Right censoring and ambiguity:** an unfinished eligible endpoint is
  right-censored at dataset or study end. The frozen no-invalidation policy has
  no target/invalidation ambiguity; generic future-policy infrastructure still
  fails closed on a same-bar target and invalidation.
- **Latency:** bars to pullback is `target_bar - rejection_bar`, from one
  through 20.

The cited 80 percent remains an external/expert benchmark for primary H2 only.
It did not select the horizon, tolerance, invalidation, rejection, A/B policy,
or endpoint and is not an acceptance threshold.

## Temporal and first-study boundary policy

The first separate empirical evaluation must reuse TG4A's preconfirmation
range: causal warmup begins at `2010-01-01T00:00:00Z`, while both score end and
outcome end are exclusive at `2025-01-01T00:00:00Z`. It must not load a 2025
bar to resolve a pre-2025 event. An otherwise unfinished 20-bar window at that
boundary is right-censored.

The established half-open exploratory, calibration, locked-rule validation,
and untouched-confirmation partitions remain unchanged. H1 cohort membership
is assigned from the close-beyond eligibility timestamp; H2 membership is
assigned from the rejection-confirmation eligibility timestamp. A/B
availability, later target time, and eventual outcome cannot rewrite that
assignment. Version 2 artifacts serialize both endpoint partitions.

The generic tracker remains usable for deterministic synthetic policy tests.
Aggregate evidence creation rejects configurations that are not the named
20-bar, no-invalidation, one-canonical-pip prospective configuration. This
freeze makes a separate historical study methodologically runnable; it does
not execute or validate that study here.

## Identity, dependence, and cohorts

`event_identity` is an FNV-1a-64 digest over schema version, symbol, timeframe,
TG3 A/B direction, both anchor bars/timestamps, and A/B availability
bar/timestamp. It is an identity checksum, not a security primitive. Every bar
within that A/B move updates one record. The accumulator supports both raw
observation-row and unique-structural-event summaries; inconsistent rows with
one identity are rejected.

Records retain symbol, direction, caller-frozen calendar period, source event
identity, and optional pre-existing volatility-regime and TG3 0.618-confluence
labels. Optional labels are accepted only if their availability is no later
than the source A/B availability, so they cannot alter or leak into the
extension event definition. No new volatility regime is defined here.

## Statistical and artifact contract

The observation schema is
`causal-fibonacci-extension-observation-v2`; the aggregate schema is
`causal-fibonacci-extension-aggregate-v2`. Deterministic CSV fields preserve:

- source A/B anchors, confirmations, direction, and availability;
- D and D-confirmation fields, empty while the D contract is unresolved;
- exact 1.272, 1.618, 0.382, 0.500, and 0.618 prices;
- frozen policy identity, tolerance, horizon, no-invalidation values, and
  endpoint-specific temporal partition;
- touch, beyond, rejection, and target-hit bars/timestamps;
- primary and descriptive outcome state, censoring state, and bars to target;
  and
- causal grouping labels.

For each endpoint and predefined cohort, aggregate output records eligible N,
successes, failures, censored, pending, ambiguous/ineligible, the resolved
denominator, measured success probability, TG4-standard Wilson 95 percent
interval, separately labeled expert benchmark when applicable,
measured-minus-benchmark, and minimum/median/maximum bars to target. H1 has no
numeric expert benchmark. H3 exposes 0.382, 0.500, and their union separately.
Censored, pending, and ineligible rows never enter the binary probability
denominator.

No database schema or persistence path is added. Artifact writing and a
historical market-data runner are intentionally outside this increment's stop
condition.

## Review gate

Before the separate historical evaluation, reviewers must verify that its
runner uses this exact version-2 policy and the frozen preconfirmation temporal
range. H3 remains ineligible. Any change to ratios, tolerance, horizon,
invalidation, eligibility, endpoint hierarchy, score/outcome boundaries, or
partition assignment requires a new prospective configuration identity before
outcomes are accessed.
