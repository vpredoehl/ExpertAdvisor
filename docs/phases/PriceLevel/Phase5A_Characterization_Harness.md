# Price-Level Phase 5A characterization harness

`Scripts/build_price_level_characterization.sh` builds the standalone,
read-only Phase 5A executable. It uses the canonical absolute half-open
15-minute candlestick query and processes only completed OHLC bars. It never
selects `target`, writes the database, or loads Tensor/LSTM code.

Build outside the repository products:

```bash
Scripts/build_price_level_characterization.sh /tmp/price_level_phase5a_characterization
```

Run an explicit interval; `--end` is exclusive. The default symbol list is
the 28 production FX pairs and the default pivot-radius study is
`1,2,3,4,6,8`.

```bash
/tmp/price_level_phase5a_characterization \
  --start 2023-01-01 --end 2026-01-01 \
  --output-dir /tmp/price_level_phase5a_2023_2025
```

Optional controls are `--symbols`, `--pivot-radii`, and
`--scale-lookbacks`; comma-separated values are required. The default scale
lookbacks (`32,96,384`) are research probes, not approved configuration
values.

`--v1-candidate radius,width,max_active,max_age,max_evidence` enables a
separate replay of the unchanged `causal-price-level/v1` engine and writes
structural interaction metrics. It is deliberately opt-in: the option does
not approve or persist the supplied values. `completedBarDuration` is fixed at
900 seconds for this harness because the source is the 15-minute production
candlestick layer.

Successful runs atomically publish the requested directory after writing:

- `pivot_radius.csv` — symbol and aggregate strict-pivot counts and rates.
- `scale.csv` — yearly raw-range and preceding-range scale distributions,
  including pivot-time and confirmation-time samples.
- `detector_metrics.csv` — populated only when a v1 candidate was requested;
  it records interactions, active-level distribution, ended lifetimes,
  pivot observations per ended level, and retained-evidence saturation.
- `manifest.txt` — source contract, causal-scale boundary, date range, bar
  count, elapsed runtime, and the explicitly requested candidate identity.

## Candidate detector study (research only)

`--adaptive-study` enables an isolated candidate detector study. It does not
alter `causal-price-level/v1`, Layout11, Tensor/LSTM code, database schema, or
any worker. Its separate configuration identity begins
`price-level-adaptive-width-study-v1`; it is not a production definition.

```bash
/tmp/price_level_phase5a_characterization \
  --start 2023-01-01 --end 2026-01-01 \
  --output-dir /tmp/price_level_phase5a_candidate_study \
  --pivot-radii 2,3,4 \
  --adaptive-study \
  --adaptive-lookbacks 32,64,128 \
  --adaptive-multipliers 0.5,1,2 \
  --fixed-control-widths 0.0005,0.05
```

The candidate detector studies radii 2, 3, and 4. Its scale is the exact
rolling median of the available preceding completed-bar `high-low` ranges,
using a startup prefix when fewer than the configured lookback bars exist. A
candidate's width is `multiplier * median`.

- `pivot_time` selects the scale snapshot taken before the pivot bar; it
  excludes the pivot and all later confirmation bars.
- `confirmation_time` selects the snapshot before the confirming bar; it can
  include the pivot and intervening completed bars, but never the confirmation
  bar itself.
- A newly established level freezes that calculated width permanently.
- A later pivot reinforces only when it is within an existing level's own
  frozen width. Among eligible levels the nearest anchor, then identity, wins.
  The reinforcing pivot's current candidate width is not used to expand,
  recenter, or chain levels.

The study uses deliberately generous bounded research caps of 64 active
levels, 512 age bars, and 32 retained pivot-evidence entries. They are probes
for saturation and capacity behavior, not a recommended v1 or production-v2
configuration. `--fixed-control-widths` runs the unchanged v1 engine at the
same radius and bounds, solely as limited raw-price controls.

In addition to the existing files, an adaptive-study run writes
`candidate_detector_metrics.csv`. It has symbol, aggregate, and yearly slices
for every control and adaptive candidate, including strict candidate outcomes,
interactions, active-count/lifetime/retest-latency distributions, evidence
saturation, expirations, and evictions. All distributions are deterministic;
the file is intentionally machine-readable rather than a recommendation.

## Phase 5B age/timing preparation (research only)

`--age-timing-study` is a separate, fixed Phase 5B study mode. It does not run
the Phase 5A controls or sweep any capacity/evidence settings. Its exact grid
is 24 adaptive research candidates: radii `3,4`; a 64-bar preceding-range
median; multiplier `1`; both scale timings; ages
`128,256,512,1024,2048,4096`; 64 active levels; and 32 retained pivot-evidence
entries. The corresponding study contract is
`price-level-phase5b-age-timing-study-v1`; the detector identity remains the
separate `price-level-adaptive-width-study-v1` identity.

The mode adds two deterministic CSVs:

- `candidate_lifecycle_metrics.csv` has complete-window and establishment-year
  cohorts, separating forced age expiration, capacity eviction, any other
  termination, and right-censoring at the requested window end. Lifetime is
  measured from the confirmation/availability bar. A level is active through
  `availableBar + maxAgeBars` and is force-expired before interactions at
  `availableBar + maxAgeBars + 1`.
- `age_curve_summary.csv` has one aggregate row per `(radius, timing, age)` in
  ascending age order, including structural rates, active population,
  censored/ended lifetime distributions, and adjacent-age percent changes.

See `Phase5B_AgeTimingContractStudy.md` for the exact manual long-run command
and interpretation handoff. Do not infer a production setting from this mode.
