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
