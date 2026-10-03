# Price-Level Phase 5C bounds-confirmation study

## Purpose

`--bounds-confirmation-study` is a read-only, research-only study of the
bounded state of the isolated `price-level-adaptive-width-study-v1` detector.
Its manifest contract is `price-level-phase5c-bounds-confirmation-study-v1`.
Neither identifier is a production detector definition; in particular,
`causal-price-level/v1` is unchanged.

Phase 5C asks only whether the two bounded collections truncate material
state. It does not inspect `target`, future returns, profitability, prediction
quality, model performance, retests, reinforcement count, or interaction
frequency to choose a configuration. Structural event counts are diagnostic
parity information, not an optimization score. The executable emits no
recommended winner.

## Fixed provisional core and grid

Every one of the exactly nine candidates uses:

- pivot radius: `3` bars;
- scale: median of preceding completed-bar `high-low` ranges;
- scale lookback: `64` bars;
- scale multiplier: `1`;
- scale timing: `pivot_time`;
- maximum age: `512` bars; and
- completed-bar duration: `900` seconds.

The Cartesian grid varies only bounded state:

| `maxActiveLevels` | `maxRetainedPivotEvidence` |
|---:|---:|
| 32, 48, 64 | 8, 16, 32 |

The existing adaptive semantics remain fixed: scale uses only preceding
completed bars; the pivot-time snapshot precedes the originating pivot bar;
established width is frozen; reinforcing pivots must be inside the existing
frozen zone; a new candidate width cannot expand, recenter, or chain a level;
and an eligible merge resolves by nearest anchor then identity. A level remains
active through `available_bar + maxAgeBars` and is forcibly expired before
interactions at the following bar.

## Outputs and interpretation

Existing deterministic `candidate_detector_metrics.csv` retains per-symbol,
per-year, and aggregate event-time diagnostics, including
`evidence_saturated_reinforcements`, active-population distributions, and
capacity evictions. `candidate_lifecycle_metrics.csv` supplies the terminal
age-expiration, capacity-eviction, and right-censoring accounting by
establishment cohort.

Phase 5C additionally writes `bounds_confirmation_summary.csv`, exactly one
aggregate row per configuration. It contains bar count, candidate pivots,
establishments, reinforcements, forced age expirations, capacity evictions,
right-censored levels, capacity evictions per 1,000 bars, active p50/p90/max,
the count/fraction of requested symbols with one or more capacity evictions,
evidence saturation, and ended levels whose retained evidence reached the cap.

`evidence_saturation_fraction` is explicitly
`evidence_saturated_reinforcements / level_reinforced_events`. Its denominator
is emitted as `evidence_saturation_reinforcement_denominator`; it is zero when
there were no reinforcement events. The symbol/year rows in
`candidate_detector_metrics.csv` provide the saturation distribution rather
than concealing it in an aggregate.

Interpret the active-cap comparison as an engineering-capacity check: inspect
capacity evictions, their normalized rate, and their symbol coverage. Interpret
the evidence-cap comparison as a retained-evidence truncation check: inspect
saturated reinforcing pivots, their explicitly defined fraction, saturated
ended levels, and the symbol/year distribution. Do not rank candidates by
structural event rates and do not infer an outcome-optimal configuration.

## Manual long run (outside CEE)

Build the standalone read-only executable:

```bash
Scripts/build_price_level_characterization.sh /tmp/price_level_phase5c_bounds_characterization
```

Then run the authoritative 28-symbol universe over the half-open interval
`[2010-01-01, 2026-01-01)`. Do not run this long study in CEE.

```bash
EA_PRICE_LEVEL_SOURCE_ID="$(git rev-parse HEAD)" \
  /tmp/price_level_phase5c_bounds_characterization \
  --start 2010-01-01 --end 2026-01-01 \
  --pivot-radii 3 --scale-lookbacks 64 \
  --output-dir /tmp/price_level_phase5c_bounds_all28_2010_2025 \
  --bounds-confirmation-study
```

The output directory must not exist. Successful results publish atomically;
inspect `manifest.txt`, all three candidate CSVs, and the Phase 5C summary
before making any later bounded-memory decision.

## Authoritative historical result

The authoritative bounds-confirmation run used all 28 symbols over
`[2010-01-01, 2026-01-01)`, processing 11,050,517 completed 15-minute bars.
The source Git identity was
`0ea8314bd4ce01d8ca55c8a7c693381805993dc6`.

The active-level bound is resolved at `maxActiveLevels=48`. A cap of 32
produced 86 capacity evictions across 18 of 28 symbols
(`0.0077824413` evictions per 1,000 bars). Caps of 48 and 64 produced zero
capacity evictions. With either nonbinding cap, the maximum observed active
population was 37, so 48 retains 11 levels of headroom above the historical
maximum. The 48- and 64-level configurations produced identical establishment,
reinforcement, age-expiration, and right-censoring counts.

The retained-evidence bound is resolved at `maxRetainedPivotEvidence=32`.
Evidence caps of 8, 16, and 32 saturated 27.1862%, 3.89128%, and 0.0357958%
of reinforcement events respectively. At cap 32, 623 of 1,740,429
reinforcement events encountered already-saturated retained evidence, and
210 ended levels reached the retained-evidence cap.

The cap-32 evidence tail does not reveal material concentration requiring a
larger bound. The highest whole-history symbol saturation fraction was
EURCHF at 0.135656%. Across symbol/year rows, 127 had nonzero saturation;
the largest observed annual fraction was EURUSD 2016 at 0.556539%
(22 saturated reinforcements of 3,953), followed by EURCHF 2012 at
0.530839%. Evidence-saturation counts were identical across active caps
32, 48, and 64.

Phase 5C therefore closes with the following bounded-state engineering
parameters:

- `maxActiveLevels=48`
- `maxRetainedPivotEvidence=32`

These are engineering bounds, not outcome-optimized parameters. No target,
future return, profitability, prediction-quality, or model-performance data
was used to select them.

Together with the preceding characterization studies, the resulting
production-contract candidate is:

- pivot radius: `3`;
- scale statistic: median preceding completed-bar `high-low`;
- scale lookback: `64`;
- scale multiplier: `1`;
- scale timing: `pivot_time`;
- maximum age: `512` bars;
- maximum active levels: `48`;
- maximum retained pivot evidence: `32`; and
- completed-bar duration: `900` seconds.

This closes empirical characterization of the provisional core and bounded
state. A subsequent implementation phase may define an explicit
`causal-price-level/v2`; this study does not itself change the production
detector definition.
