# Causal price-level structure engine — Phase 1

## Definition and scope

`Headers/CausalPriceLevelEngine.hpp` freezes the detector-only definition
`causal-price-level/v1`. It accepts one validated, completed OHLC bar at a time
and returns immutable observations plus a bounded active-zone snapshot. It is
not a Tensor feature, a MarketStructure producer, a Confluence definition, a
trading recommendation, a learned score, or a persistence workflow.

The engine uses the repository's completed-bar convention explicitly: input
`barStart` identifies `[barStart, barStart + completedBarDuration)`, and every
event observed on that bar is available at its completed end. Callers must
supply `completedBarDuration`; production use should supply the already
established canonical duration rather than assume a new timeframe.

## Domain evidence and engineering translation

The available project trading reference inspected for this phase is
`ResearchSources/Pockets/Pockets.pdf`. Its surrounding project documentation
uses directional ranges and distinguishes touch/close behavior, but it does not
specify a general support/resistance pivot, merge, or retest algorithm. Existing
TG2/TG3 documentation also uses causal break/retest language and inclusive
fixed-tolerance zones for its own frozen trend-line/Fibonacci semantics. Those
are domain concepts and repository precedents, not a specification for this
engine.

Accordingly, Phase 1 defines only the following deliberately narrow engineering
translation. The words `support_like` and `resistance_like` are state labels
derived from a pivot origin or a later cross. They do not claim that price will
hold, reverse, or produce a profitable trade.

## Representation and configuration

A level is a fixed canonical anchor `A` and a bounded inclusive zone:

```text
[A - W, A + W]
```

where `W = zoneHalfWidth` is a finite, nonnegative, explicit configuration
value. It is in raw symbol price units; Phase 1 deliberately does not infer
pip metadata, ATR, percentage, or a volatility scale. `W` is also the fixed
merge tolerance. A configuration has no implicit default and its canonical
identity includes every field: pivot radius, `W`, active cap, age cap, retained
pivot-evidence cap, and completed-bar duration.

Invalid configuration, nonfinite OHLC/width, inconsistent OHLC, nonpositive
duration, and non-increasing/duplicate bar starts reject. Equal high or low
ties are not pivots.

## Causal establishment and availability

For radius `R`, bar `i` is a high pivot only when its high is strictly greater
than the highs in `[i-R, i+R]` excluding itself. A low pivot is defined with
strictly lower lows. The candidate becomes known only on completion of bar
`i+R`:

```text
observedAt  = completion(i)
availableAt = completion(i + R)
```

No level exists before `availableAt`; the current confirmation bar does not
also generate a level interaction. This avoids treating an earlier geometric
point as knowledge available before its later confirmation.

## Merge, identity, and retained evidence

A confirmed pivot price `P` merges only when `abs(P - A) <= W` for an existing
fixed anchor. If several anchors qualify, select the smallest absolute distance,
then canonical level identity. The anchor, zone, primary origin, and identity
never recenter. Consequently `10.0` can merge `10.5` at `W=0.5`, but that does
not make a later `11.0` merge through a moving average or chain.

A new level identity includes the canonical symbol, complete definition
identity, and the first pivot's kind, bar coordinate, bar start, and exact
canonical price. Its provenance contains the same frozen definition and symbol.
The snapshot records a saturating pivot-observation count and only the first
configured number of canonical pivot evidence identities. Raw touch history is
not retained.

## Interaction vocabulary

All interaction events are emitted only at the completion of their current
input bar (`observedAt == availableAt`):

- `level_established`: the causal confirmation creates a new zone.
- `level_reinforced`: a later causally confirmed pivot merges into it.
- `touch`: bar range inclusively overlaps the zone after the immediately prior
  bar did not overlap it. The retained touch count increments then.
- `cross_up`: prior close is strictly below the lower boundary and current close
  is strictly above the upper boundary. `cross_down` mirrors this. A gap can
  therefore cross; a boundary close alone cannot.
- `retest`: an inclusive touch on a strictly later bar following the most recent
  cross, emitted once per cross. It makes no claim about follow-through.
- `role_reversal`: emitted with a cross only when the cross changes the current
  state from resistance-like to support-like (up) or support-like to
  resistance-like (down).
- `level_expired` and `level_evicted`: explicit removal diagnostics, respectively
  for age and capacity.

An initial high pivot is resistance-like and an initial low pivot support-like.
No `approach`, `rejection`, “successful breakout,” prediction, confidence, or
universal strength score is defined in Phase 1. `cross_*` is intentionally not
called a breakout: no extra criterion has been justified for that term.

## Bounds, ordering, and replay

The input history is retained only until the latest pivot window can be tested
(`2R + 1` bars after an update; at most `2R + 2` temporarily while accepting the
next bar). Active levels are limited to `maxActiveLevels`. A level remains active
while `currentBar - availableBar <= maxAgeBars`; older levels expire before
current-bar interactions. If a new distinct level arrives at capacity, evict the
oldest availability time, then lexicographically smallest canonical identity.
Active snapshots sort by anchor price then identity; observations sort by
identity. There are at most `3 * maxActiveLevels + 4` observations per update:
at most one expiration or three interactions per prior active level, plus two
candidate outcomes and at most two capacity evictions. State holds no raw touch
history and no more than `maxRetainedPivotEvidence` pivot IDs per active level.

The engine accepts only chronological unique bars. Its state contains no future
bar and neither alters old return values nor revises a level anchor. Replaying
an identical prefix produces the identical canonical representation; appending
later bars cannot change a result retained from that prefix.

## Integration boundary and non-goals

This phase intentionally adds no `MarketStructure::Observation` adapter. The
generic boundary already permits opaque pre-Tensor family identities, but a
bridge should be a separately reviewed phase that maps the exact frozen
PriceLevel observation vocabulary into roles without changing detector output.
The engine does not call `ConfluenceEngine`; future confluence may consume
copied bridged observations and must not suppress, mutate, or reinterpret these
raw observations.

There is no Tensor adapter, layout change, FeatureAblation entry, model-width
change, LSTM change, database access, scheduler interaction, worker change, or
experiment materialization in Phase 1. Existing TG/Fibonacci/Pocket/General
Confluence semantics are untouched.
