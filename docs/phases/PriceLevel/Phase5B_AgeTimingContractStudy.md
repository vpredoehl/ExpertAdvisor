# Price-Level Phase 5B age / timing contract study

## Scope

This is a read-only, research-only measurement pass for the already isolated
`price-level-adaptive-width-study-v1` detector. It does not alter
`causal-price-level/v1`, Layout11, Tensor/model inputs, experiments 702/703,
the scheduler, inference, profitability, database schema, historical workers,
or Phase 3B keyed confluence semantics. It reads only completed OHLC bars from
the canonical half-open loader; `target` is not selected or used.

The Phase 5B study manifest uses
`price-level-phase5b-age-timing-study-v1`. This describes the measurement
contract, not a production detector version or decision.

## Exact primary grid

`--age-timing-study` constructs exactly 24 candidates:

| Setting | Values |
|---|---|
| Pivot radius | `3`, `4` |
| Scale | median of preceding completed-bar `high-low` ranges |
| Scale lookback / multiplier | `64` / `1` |
| Scale timing | `pivot_time`, `confirmation_time` |
| Maximum age bars | `128`, `256`, `512`, `1024`, `2048`, `4096` |
| Active-level cap / retained evidence cap | `64` / `32` |
| Completed-bar duration | 900 seconds |

No active-cap or evidence-cap expansion is implied by this command.

`pivot_time` freezes the range-median snapshot immediately before the origin
pivot bar, excluding that bar and all confirmation bars. `confirmation_time`
freezes the snapshot immediately before the later confirmation bar; it may
contain the pivot and intervening bars, never the confirmation bar. A level's
width never changes after establishment. Reinforcement uses only the existing
level's frozen zone; nearest anchor then identity resolves an eligible tie.

## Lifecycle and censoring semantics

Lifetime is measured in completed-bar indices from the confirmation/
availability bar, not the origin pivot. For a level available at bar `A`, it
remains active through bar `A + maxAgeBars`; it is emitted as
`level_expired` before interactions at bar `A + maxAgeBars + 1`. Thus a forced
expiration's recorded ended lifetime is `maxAgeBars + 1` under the inherited
v1-compatible boundary.

The study never calls a level still active at the final requested bar an ended
level. It is recorded as `right_censored_end_of_requested_window`, with an
observed lifetime of `last_requested_bar - A`. Capacity eviction is separately
reported. `levels_other_terminated` is explicit and currently remains zero
unless a future research-only detector adds a distinct termination path.

`candidate_lifecycle_metrics.csv` contains complete-window (`establishment_year = 0`)
and establishment-year cohorts for every symbol and aggregate candidate.
It includes establishment/reinforcement, each termination reason, censored
count, forced-age-expiration fraction, observed survival counts/fractions at
the six age values, ended-lifetime distribution, and censored-lifetime
distribution. Its year is the confirmation/availability year so that a cohort
has one unambiguous terminal/censoring accounting at the requested-window end.

`candidate_detector_metrics.csv` remains the event-time structural output,
with all symbol/year/aggregate bars, touches, crosses, retests, reversals,
active p50/p90/max, evidence saturation, and normalized-rate inputs.

`age_curve_summary.csv` is the primary comparison file. It is deterministically
ordered by `(pivot_radius, scale_timing, max_age_bars)` and has aggregate rows
in the required age sequence. It provides counts/rates and the immediately
preceding age plus percent change for establishment rate, reinforcement
fraction, retest rate, and active p50. It intentionally contains no composite
score.

## Manual long run (outside CEE)

Build the standalone read-only executable:

```bash
Scripts/build_price_level_characterization.sh /tmp/price_level_phase5b_age_timing_characterization
```

Then run the 2010–2026 all-28-symbol primary grid. The output directory must
not already exist; successful publication is atomic.

```bash
EA_PRICE_LEVEL_SOURCE_ID="$(git rev-parse HEAD)" \
  /tmp/price_level_phase5b_age_timing_characterization \
  --start 2010-01-01 --end 2026-01-01 \
  --pivot-radii 3,4 --scale-lookbacks 64 \
  --output-dir /tmp/price_level_phase5b_age_timing_all28_2010_2025 \
  --age-timing-study
```

The default symbols are exactly the authoritative 28-symbol universe. The
manifest records read-only status, target omission, requested interval,
processed bars, exact grid, causal definitions, source identity when supplied,
and elapsed wall-clock seconds.

Phase 5A's 24-candidate, three-year all-28 run took approximately 126 seconds
for 1,998,797 bars. This 16-year study has the same candidate count but much
more history; its actual runtime depends on the available data and host. Run it
outside CEE and do not treat that estimate as a runtime guarantee.

After it completes:

1. Inspect `manifest.txt` and output file sizes, ensuring the output directory
   was atomically published rather than left as `.incomplete`.
2. Return the manifest and the CSV summaries to ChatGPT/CEE.
3. Only then perform the Phase 5B interpretation and any freeze-decision
   tranche. No production radius, timing, age, or cap is selected here.
