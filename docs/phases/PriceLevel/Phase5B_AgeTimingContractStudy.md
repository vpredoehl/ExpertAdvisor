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

## Completed study results

The primary all-history run completed over the authoritative 28-symbol
universe for `[2010-01-01, 2026-01-01)`. It processed 11,050,517 completed
15-minute bars in 444.295 seconds. The manifest reported
`read_only=true`, `target_column_used=false`, 24 candidate configurations,
and source Git identity
`7bf233d010b205896ef1ca9dfd597c295acec06a`.

The aggregate and cross-symbol results do not identify a natural level
lifetime. At maximum ages 256, 512, and 1024, no tested configuration had a
capacity eviction, yet essentially all levels that could be observed through
their terminal age ended by forced age expiration. Median establishment-year
age-expiration fraction was 1.000 for every radius/timing combination at
those ages. `levels_other_terminated` was zero.

Increasing maximum age materially changes detector structure rather than
approaching an evident plateau. For radius 3 with pivot-time scaling, aggregate
establishments per 1000 bars changed from 35.881 at age 256, to 26.301 at
512, to 19.178 at 1024, while reinforcement fraction increased from 0.805,
to 0.857, to 0.896 and retests per 1000 bars increased from 9.154, to
10.722, to 12.120. Active-population p50 increased from 9, to 13, to 19.
The same monotonic behavior was present for radius 4 and for confirmation-time
scaling.

The cross-symbol and calendar-year analyses confirm that this behavior is not
an aggregate artifact. Establishment and reinforcement distributions are
comparatively tight across symbols, and the same age progression persists
across the 2010-2025 calendar-year aggregates.

Age 2048 is no longer a clean age-only observation under the fixed
`maxActiveLevels=64` study bound. Radius 3 encountered capacity evictions in
16 of 28 symbols and 11 calendar years. Radius 4 encountered capacity
evictions in 7 of 28 symbols, with capacity eviction present in 6
confirmation-time calendar years and 7 pivot-time calendar years. The 4096
rows are still more strongly capacity-confounded. Consequently, extending
the age sweep with the same active-level cap would increasingly characterize
the capacity-eviction policy rather than unconstrained level age.

The Phase 5B conclusion is therefore that `maxAgeBars` is detector memory and
a first-class detector semantic, not a natural lifetime inferred from market
structure. The current detector has no endogenous level-death mechanism:
absent age expiration, capacity eviction, or end-window censoring, an
established level persists.

Pivot-time and confirmation-time scaling remain structurally close throughout
the full-history study. Because no material structural advantage for
confirmation-time was observed, `pivot_time` is the preferred contract
semantics: the frozen width is derived from information available before the
originating pivot bar rather than incorporating bars occurring after that
pivot. This is a semantic selection, not an optimization on interaction count.

Radii 3 and 4 are both structurally stable. Radius 4 is consistently more
selective, but this characterization provides no outcome-based criterion by
which lower structural density should be preferred. Radius 3 therefore remains
the provisional candidate carried forward from Phase 5A; Phase 5B does not
claim that radius 3 is empirically optimal.

For subsequent bounded-state confirmation, age 512 is the provisional
engineering candidate. It is entirely free of capacity eviction in this study
and keeps the active population substantially below the 64-level bound.
Selection of 512 is a bounded-memory design choice, not a discovered natural
lifetime or profitability optimum.

The resulting provisional adaptive detector core carried into the next
research tranche is:

- pivot radius: 3
- causal scale: median of the preceding 64 completed-bar `high-low` ranges
- scale multiplier: 1
- scale timing: `pivot_time`
- maximum age: 512 bars
- completed-bar duration: 900 seconds
- established width remains permanently frozen
- reinforcement remains constrained to the existing frozen zone

`maxActiveLevels=64` and `maxRetainedPivotEvidence=32` remain study bounds,
not yet independently frozen contract values. A subsequent bounds-confirmation
study should vary only those bounds while holding the provisional detector core
fixed. It must characterize capacity/evidence truncation rather than optimize
market outcomes or profitability.

