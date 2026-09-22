# TG3 causal Fibonacci confluence and trend-line integration

TG3 is reusable diagnostic and empirical research infrastructure in
`Headers/CausalFibonacciConfluenceIntegration.hpp`. It composes over TG2 and
does not feed Tensor, alter model inputs, change semantic layout 7, change
feature-ablation behavior, or participate in scheduler, worker, registry, or
publication paths.

## Source and repository audit

The audit covered the complete TG1A, TG1B, and TG2 headers, focused tests,
phase documentation, and prompts; all repository text matches for Fibonacci,
retracement, extension, swing, AB/ABC, and pivot terminology; the PostgreSQL
candlestick and strict five-candle fractal functions; the causal historical
weekly-level feature; strategy-evaluation diagnostics; and available project
files, manuals, PDFs, documents, and diagrams outside the economic-calendar
source archive.

The only supplied Fibonacci instruction found is:

```text
See if Outer UTL is within the up AB Fibonacci.
```

No available source defines A or B, the process for selecting one swing among
several, a ratio set, retracement versus extension, the meaning of "within,"
the exact observation time, or a DTL/down-AB counterpart. No Fibonacci or AB
implementation or PostgreSQL function exists in the repository. The existing
`fractal.plpgsql` contract does authoritatively define strict five-candle
fractals, and TG1A authoritatively makes a center at bar `i` available only at
`i+2`.

TG3 therefore preserves the supplied UTL/up-AB direction but labels every
missing semantic as an implementation convention. It does not represent the
conventions below as source-derived trading methodology. It implements no
extension levels because no extension instruction or ratios were found.

## Causal AB convention

The named neutral anchor policy is
`MostRecentPriorOppositeConfirmedFractal`. For each newly confirmed pivot B:

- an UpAB uses B as a confirmed fractal high and A as the most recent earlier
  confirmed fractal low with `B.price > A.price`;
- a DownAB uses B as a confirmed fractal low and A as the most recent earlier
  confirmed fractal high with `B.price < A.price`;
- A must have an earlier anchor bar and must already be confirmed no later
  than B's confirmation;
- B and the AB structure become available only at B's TG1A confirmation bar;
  and
- a finite zero range is rejected by the directional inequality, while
  non-finite or causally inconsistent input is rejected.

The stable identity is:

```text
(direction,
 A anchor bar/time,
 B anchor bar/time,
 B confirmation/availability bar/time)
```

It also records both prices, both confirmation times, range, and policy. Each
new B produces at most one structure under this convention. Multiple AB
structures coexist. Selection at a confluence observation uses the most
recently available matching-direction structure; stable identity is the tie
breaker. Neither structure creation nor selection uses later contact or TG2
outcome. Later pivots never rewrite an emitted structure or copied
observation.

This policy is a configurable measurement convention, not a claim that the
source defines swings this way. It deliberately reuses the causal TG1A pivot
contract and does not use ZigZag finalization, future extrema, or retrospective
"best swing" selection.

## Fibonacci configuration and equations

TG3 requires the caller to supply a non-empty ratio vector. There is no
default ratio set because the audited source supplies none. Ratios must be
finite and in `[0, 1]`; exact duplicates are sorted and removed
deterministically. Negative zero is canonicalized. Only
`FibonacciLevelType::Retracement` exists in TG3; extensions remain unsupported
and cannot be confused with retracements.

Let `R = abs(B - A)` and let `r` be a configured retracement ratio:

```text
UpAB level(r)   = B.price - r * R
DownAB level(r) = B.price + r * R
```

Thus ratio zero is at B and ratio one is at A in both orientations. Returned
levels are ordered by ascending price and then ratio. A non-finite or
non-positive range is rejected.

## Observation and Outer-line integration

The confluence observation is made on the same completed bar as an already
emitted TG2 Inner break. TG3:

1. copies TG2's paired Outer identity and geometry without changing or
   recomputing the pairing;
2. selects only an AB structure available by that completed break bar;
3. uses TG2's frozen Outer projection at the break bar;
4. calculates the explicitly configured retracement levels; and
5. freezes confluence state, selected AB, levels, distances, and policies.

The default `SourceUTLUpABOnly` study measures only a UTL Inner break, its TG2
paired UTL Outer, and an UpAB, matching the sole source statement. A DTL break
is structurally ineligible under this default. The separately named
`SymmetricDirectionalDiagnostic` can opt into DTL/down-AB measurement; it is
explicitly a diagnostic hypothesis, not source evidence.

An Inner break is classified as one of:

- confluence: paired Outer, eligible directional AB, and at least one matched
  configured level;
- no confluence: paired Outer and eligible AB, but no level match; or
- structurally ineligible: no TG2 paired Outer, no eligible AB, or a direction
  excluded by the selected study policy.

Structural ineligibility is never converted into non-confluence or behavioral
failure.

## Neutral "within" convention

The named policy is
`AbsolutePriceToleranceAroundExactRetracementLevel`. For Outer projection
`P`, level `L`, and configured finite nonnegative tolerance `T`:

```text
raw distance = abs(P - L)
match        = raw distance <= T
zone         = [L - T, L + T]
```

Both boundaries are inclusive. Tests use adjacent representable
floating-point values immediately inside and outside the boundary. TG3 records
every ratio, level, zone, raw distance, match flag, all matched ratios, and the
minimum raw distance. If the paired Outer's TG1A ATR is causally available on
the observation bar, it additionally records:

```text
minimum ATR-normalized distance = minimum raw distance / observation ATR
```

The tolerance is caller configuration and is never estimated from later
success. No default ratio or outcome-calibrated tolerance is supplied.

## Outcome synchronization and empirical groups

Confluence classification is immutable. TG3 only synchronizes the associated
TG2 retest, Outer-target, and retest-then-Outer outcome snapshots as TG2's
finite windows resolve. A future success, failure, censor, or pivot cannot
change the frozen confluence group, paired Outer, AB identity, ratios, levels,
or distances.

The aggregate exposes separate groups for confluence, no confluence, and
structural ineligibility. Each group retains raw Outer-target counts. For
observations whose TG2 retest succeeded, it separately retains raw
retest-then-Outer counts. Counts include eligible, resolved, successes,
failures, censored, pending, and structurally ineligible states as applicable.

For resolved outcomes only:

```text
empirical rate = successes / (successes + failures)
```

The rate is absent for a zero denominator. Censored, pending, and structurally
ineligible observations do not enter the denominator. These are observed
sample frequencies, not probabilities, causal effects, trading signals, or
profitability claims.

## Bounds, eviction, ordering, and complexity

Configuration separately bounds retained confirmed fractals per kind, active
AB structures, AB age, active TG3 observations, and retained TG3
observations. Defaults are 128 fractals per kind, 512 AB structures, 2,048
bars of AB age, 4,096 active observations, and 4,096 retained observations.

AB expiry and capacity eviction use deterministic availability/identity
ordering. A copied AB in an existing confluence observation remains immutable
after active-structure expiry. Observation capacity evicts the oldest event
sequence; every pending copied outcome becomes explicit
`censored/capacity_eviction`, its counts are accumulated, and the detailed
record is removed. Disappearance caused by TG2 capacity is detected and
represented the same way rather than becoming failure.

For bounded retained fractals `F`, AB structures `A`, TG2 candidates `C`, and
TG3 observations `O`, normal per-bar work is `O(F + A + O + C)` plus TG2's
existing bounded work. Storage is `O(F + A + O)` and does not grow with total
stream length. The focused 50,000-bar test exercises deterministic AB and
observation eviction and cumulative counts.

## Diagnostics and scope

Per-observation diagnostics include symbol/timeframe, TG2 event and Inner
candidate identity, direction and frozen class, break/observation time, paired
Outer identity and projection, AB identity/direction/anchors/prices/all
confirmation times, policy and provenance labels, ratios/levels/zones,
tolerance, raw and normalized distances, confluence/ineligibility, TG2 outcome
states, censoring, and an observed-frequency warning. Summary diagnostics
include every group numerator, denominator, rate, censor/pending count, and
structural reason count.

TG3 creates no order, entry, exit, stop, target, sizing rule, profitability
gate, model label, training objective, Tensor feature, input-width or semantic
layout change, feature-ablation change, database write, scheduler behavior,
worker selection, registry entry, or publication path. Turning any TG3 result
into a feature or trading decision requires a separate reviewed phase.
