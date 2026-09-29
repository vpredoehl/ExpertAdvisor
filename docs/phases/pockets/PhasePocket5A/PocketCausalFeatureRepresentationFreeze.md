# Phase Pocket 5A — causal recent-confirmed-observation feature representation freeze

**Status: frozen specification only, before any Pocket Tensor integration or
Pocket/LSTM incremental-information evaluation.**

**Specification identifier:** `phase-pocket-5a-causal-recent-confirmed-observations-v1`<br>
**Implementation baseline:** `31b0f6b3f8975c52b0fc70070ca4d8e0134dedc1`

This is an outcome-blind representation freeze for a later question only:

> Does causally available Pocket information provide incremental predictive
> information to the LSTM beyond the existing layout-9 feature set?

It does not establish predictive information, profitability, causation, LSTM
utility, or a reason to implement a model feature.  It creates no Tensor
column, semantic layout, model-input-width change, feature-ablation mask,
experiment, scheduler action, or database row.

## Authority and inherited semantics

The immutable Phase Pocket 2 project operational detector remains authoritative
(`CausalPocketDetector.hpp` and `PocketOperationalDefinitionFreeze.md`).  It
is not an exact mathematical transcription of the MTI manual and is not an
execution, profitability, active-Pocket, expiry, invalidation, touch, fill, or
lifecycle model.

This representation consumes exactly one detector stream per symbol:

```text
CausalPocketDetector("15m_completed", 15)
```

`15` is inherited from the Phase Pocket 2 source-default baseline.  It is not
selected from Phase Pocket 4's separate `{10,15,20}` sensitivity reporting,
and those Phase 4 lookbacks are not mixed or optimized here.  The 11-channel
proposal has one detector stream; adding the three streams would be a different
feature family requiring a new freeze.

For a detector observation, its immutable information cutoff is exactly its
confirmation bar: `informationCutoffBar == confirmationBar == c`.  A feature
row for completed current bar `t` is calculated only after supplying that bar
to the detector.  Consequently an observation emitted while processing `t`
may contribute at age zero, and no observation is visible before its cutoff.
All observation endpoints, direction, event/cutoff coordinates, timestamps,
and timeframe are immutable after emission.

## New Phase 5A representation choice: recent confirmed observations

This phase freezes a recent-observation horizon of **20 completed bars**.  An
emitted observation is eligible at current completed bar `t` exactly when:

```text
t >= c  and  t - c <= 20
```

Thus ages are the inclusive integer range `0..20`.  This is an engineering
choice aligned with the already productionized causal Fibonacci structural
event-relevance horizon.  It is not derived from Pocket Phase 4 detector
lookbacks, outcome horizons, touch/fill outcomes, or confirmation results.
It does not cause an observation to expire, mutate, be replaced, or cease to
exist in the detector; it limits only this future representation's current-row
aggregation.

For each symbol, a future producer retains detector emissions append-only in
confirmation-bar order and aggregates every eligible one.  It must not choose
a nearest, newest, or otherwise privileged observation.  The detector emits
at most one observation while accepting one completed bar, but up to 21 prior
or current emissions can coexist in this aggregation horizon.  A producer
must reject duplicate emitted identities rather than silently deduplicating
them.  A sufficient identity is the symbol plus source timeframe, event and
information-cutoff bar/timestamp coordinates, and direction; normal detector
replay makes those coordinates unique within a symbol stream.

## Frozen 11-channel family

The future append-only family, in this exact order, is:

```text
pocket_recent_price_scale_valid

pocket_bull_recent_count_log
pocket_bull_youngest_age20
pocket_bull_median_touch_distance
pocket_bull_median_close_distance
pocket_bull_median_width

pocket_bear_recent_count_log
pocket_bear_youngest_age20
pocket_bear_median_touch_distance
pocket_bear_median_close_distance
pocket_bear_median_width
```

Bullish and bearish populations are always separate.  Let `E_d(t)` be all
eligible observations of direction `d` at bar `t`, and let `n_d=|E_d(t)|`.

```text
count_log_d = log1p(n_d)
youngest_age20_d = min(t - c for E_d(t)) / 20.0, when n_d > 0
```

An empty directional population has zeros for all five directional channels.
Age zero is valid; `count_log_d` distinguishes age-zero presence from an empty
population.

For current completed close `C`, observation boundary `P`, and denominator
`S`, the signed directional distance is:

```text
bullish: +(P - C) / S
bearish: -(P - C) / S
```

For each bullish observation, `TouchPrice()` is `upper` and `ClosePrice()` is
`lower`; for each bearish observation, `TouchPrice()` is `lower` and
`ClosePrice()` is `upper`.  The touch and close distance channels are the
exact nearest-rank p50 of their respective eligible directional values.  The
one-based p50 rank is `ceil(0.5*N)`—the lower middle for an even population;
there is no arithmetic interpolation.

For every eligible observation:

```text
normalized_width = (upper - lower) / S
```

The width channel is the same exact nearest-rank p50.  Detector-valid ranges
are strictly positive, so normalized widths are nonnegative.  Directional
close/touch distances deliberately remain signed in a common directional
coordinate system; this does not import Fibonacci's `upAB` API or semantics.

## Causal normalization and missingness

`S` uses precisely the existing layout-9 causal Fibonacci denominator policy,
without coupling this family to Fibonacci structure APIs:

```text
S = current causal ATR, when finite and ATR > canonicalPipSize;
    canonicalPipSize otherwise.
```

The canonical pip size is the canonical-symbol pip mapping already used by
`CausalFibonacciStructuralFeatureConfiguration`; it is not inferred from a
Pocket outcome or a market-data future.  A future producer must take its
current causal ATR from `TG1A::CausalFractalTrendLineGeometry::CurrentAtr()`
after accepting the same completed current bar through the existing layout-9
`CausalFibonacciStructuralFeatureConfiguration::geometry()` configuration
(the established Wilder recursive ATR with period `14`).  It may include that
completed bar and never a later bar.  This shares only the established
normalization convention; it does not consume Fibonacci `upAB`, event, or
structure state.

`pocket_recent_price_scale_valid` is `1` exactly when current close is finite,
canonical pip size is finite and positive, and the resulting finite positive
denominator can be formed under the fallback rule; otherwise it is `0`.
This is **not** an ATR-availability bit: a missing, non-finite, or too-small
ATR uses the valid pip fallback and can still yield `1`.

When scale is invalid, all normalized geometry channels (touch distance, close
distance, width) remain zero.  Counts and ages are still causally available
and are computed normally.  This follows the existing layout-9 convention:
scale invalidity is observed, not a reason to erase an otherwise confirmed
population.  A malformed/non-finite immutable detector observation, invalid
range, non-monotone producer bar coordinate, or non-finite value emitted while
scale is declared valid is a fail-closed producer error, never a hidden
missingness encoding.

## Determinism, causality, and startup invariants

1. The producer accepts strictly ordered completed bars only and replays the
   same prefix to the same row values.
2. It calls the Phase-2 detector on each completed bar before aggregating that
   row; no future bar or Phase-4 outcome label participates.
3. It retains every eligible immutable emission in both directions; later
   touch, fill, traversal, price action, H64 resolution, or another Pocket
   cannot mutate, remove, or prioritize an observation.
4. Eligibility uses integer completed-bar coordinates, not elapsed time.
5. Exact p50 sorting makes aggregate results permutation-invariant.  A
   deterministic producer order is confirmation coordinate followed by the
   immutable identity, although no aggregate selects the first identity.
6. Before sufficient detector history there are no emissions.  Before ATR is
   available, the canonical-pip fallback remains usable when close and pip are
   valid.  Every output must be finite.
7. No `Database/indicators/pocket.plpgsql` traversal/future logic is used.

## Explicit non-goals and future boundary

This freeze creates no active/living Pocket collection; does not make touch or
fill terminate a Pocket; does not use the Phase 4 H64 horizon as a lifetime;
does not choose a recent or nearest Pocket; and does not add a price-action,
trading, profitability, selection, threshold, or winner rule.

A later separately authorized layout-10 task may implement a distinct causal
producer, append exactly these 11 channels after layout 9 and before the
stable four-return model-input suffix, register `pockets.*` channels, expand
the model input, and add producer/Tensor tests.  A later separately frozen
paired CONTROL(layout 9) versus TREATMENT(layout 10) protocol may evaluate
incremental predictive information.  Neither implementation nor experiment
materialization is authorized by this specification.
