# Price-Level Phase 5A candidate detector study

## Scope and frozen boundary

This is a read-only characterization of an isolated research detector. It did
not change `causal-price-level/v1`, Layout11, Tensor/LSTM inputs, experiments
702/703, scheduler behavior, database schema, historical workers, or Phase 3B
keyed confluence semantics. No `target`, model, or trading outcome was read.

The code uses `price-level-adaptive-width-study-v1`, a research-only identity.
It is not an implementation or approval of a production v2 contract.

## Causal candidate definition

For a configured lookback `L`, the scale immediately before a completed bar is
the exact median of the prior `min(L, available predecessor count)` completed
15-minute `high-low` ranges. The current bar is sampled before its own range is
inserted. Startup uses the available predecessor prefix; a level cannot be
established until at least one predecessor exists.

For a strict pivot at bar `i`, confirmed by completed bar `i + R`:

- pivot-time width is `multiplier * median(ranges before i)`;
- confirmation-time width is `multiplier * median(ranges before i + R)`.

Thus the former excludes the pivot and later bars; the latter can include the
pivot and intervening completed bars but excludes the confirmation bar. A new
level freezes that calculated width permanently. A later pivot reinforces only
when it is inside an existing level's frozen zone (`abs(pivot - anchor) <=
existing_width`); the nearest anchor then canonical identity wins. The later
candidate width never expands, contracts, or recenters an existing zone.

All research candidates used bounded probes: 64 active levels, age 512 bars,
and 32 retained pivot evidence entries. These are measurement caps, not a
recommendation.

## Runs and machine-readable output

All runs used the canonical absolute half-open completed-bar query on
`[2023-01-01, 2026-01-01)` in repeatable-read/read-only mode.

| Run | Universe | Candidates | Bars | Wall time | Output |
|---|---:|---:|---:|---:|---|
| breadth grid | EURGBP, GBPJPY, CADCHF, AUDCAD, EURUSD, USDJPY | 60 | 428,556 | 40.400 s | `/tmp/price_level_phase5a_candidate_subset_3` |
| all-universe confirmation | all 28 symbols | 24 | 1,998,797 | 126.246 s | `/tmp/price_level_phase5a_candidate_all28` |

The breadth grid contains radii `{2,3,4}`, lookbacks `{32,64,128}`,
multipliers `{0.5,1,2}`, both scale timings (54 adaptive candidates), plus
two raw fixed-width v1 controls (`0.0005`, `0.05`) at each radius (6 controls).
The lookback alternatives were structurally close on that deliberately diverse
subset, so the all-universe confirmation retained 64 bars while preserving all
three radii, all multipliers, both timings, and both controls.

A 2010–2025, 12-candidate follow-up (the multiplier-1 candidates and both
controls for all three radii) was started, then stopped after more than ten
minutes while active scheduler training was present. It did not atomically
publish results and is not used here. The existing all-universe yearly slices
provide the practical regime check; a future long-history pass should omit the
known-expensive raw controls or use an explicitly approved resource window.

Each directory contains `manifest.txt`, `pivot_radius.csv`, `scale.csv`,
`detector_metrics.csv`, and `candidate_detector_metrics.csv`. The last file is
the candidate-study output: one deterministic CSV row per candidate for every
symbol, every calendar year, and the whole symbol history, plus aggregate rows.
It contains candidate pivots, establishment/reinforcement, all interaction
counts, expiration/eviction/evidence saturation, and distributions for active
levels, lifetime, retest latency, and pivot observations per ended level.

## All-universe structural results

The table uses aggregate rows from `candidate_detector_metrics.csv`. `new` and
`retest` are per 1,000 bars; all candidates here use the 64-bar median.

| Radius / candidate | Pivots / 100 bars | New | Reinforcement | Retest | Active p50/p90 | Evictions / 1k |
|---|---:|---:|---:|---:|---:|---:|
| 2, multiplier 1, pivot time | 25.835 | 28.970 | 0.888 | 13.781 | 14 / 21 | 0 |
| 3, multiplier 1, pivot time | 18.414 | 27.449 | 0.851 | 12.647 | 14 / 19 | 0 |
| 4, multiplier 1, pivot time | 14.250 | 26.108 | 0.817 | 11.755 | 13 / 18 | 0 |
| 3, multiplier 0.5, pivot time | 18.414 | 44.693 | 0.757 | 71.674 | 22 / 31 | 0 |
| 3, multiplier 1, confirmation time | 18.414 | 27.144 | 0.853 | 12.286 | 13 / 19 | 0 |
| 3, multiplier 2, pivot time | 18.414 | 16.108 | 0.913 | 1.758 | 8 / 12 | 0 |
| v1 control, radius 3, raw width 0.0005 | 18.414 | 68.030 | 0.631 | 219.530 | 17 / 64 | 44.701 |
| v1 control, radius 3, raw width 0.05 | 18.414 | 12.580 | 0.932 | 16.730 | 1 / 24 | 0.006 |

For the radius-3, 64-bar, multiplier-1 pivot-time candidate, all levels that
ended did so at the 512-bar age boundary (aggregate lifetime p50/p90 513/513
bars); retest latency p50 was 1 bar and p90 37 bars. Evidence saturation was
0.015% of candidate pivots. This demonstrates that the 512-bar age probe, not
natural disappearance, determines typical observed lifetime.

## Cross-symbol and yearly stability

For radius 3 / 64 bars / multiplier 1 / pivot time, the cross-symbol
distribution was:

| Metric | p10 | p50 | p90 | min–max |
|---|---:|---:|---:|---:|
| New levels / 1,000 bars | 25.872 | 27.285 | 29.318 | 25.157–30.096 |
| Reinforcement fraction | 0.840 | 0.851 | 0.860 | 0.835–0.862 |
| Retests / 1,000 bars | 9.245 | 11.639 | 15.525 | 8.711–17.231 |
| Median active levels | 13 | 13 | 15 | 12–15 |
| Evictions / 1,000 bars | 0 | 0 | 0 | 0–0 |

The same metrics reveal the raw-control failure clearly. At radius 3, raw
`0.0005` has new-level p10/p50/p90 of 21.420/30.281/179.315 per 1,000 bars,
reinforcement 0.030/0.818/0.880, and active p50 10/15/64. Raw `0.05` instead
has new-level p10/p50/p90 1.945/1.947/46.844, reinforcement
0.742/0.989/0.989, and active p50 1/1/23. The controls respectively fragment
JPY-scale symbols or collapse non-JPY-scale symbols; neither is cross-symbol
defensible.

For the central adaptive candidate, yearly cross-symbol new-level-rate
p10/p50/p90 was 26.81/28.40/30.79 (2023), 25.72/27.53/29.85 (2024), and
24.42/25.80/28.68 (2025). Reinforcement p10/p50/p90 was
0.836/0.844/0.858, 0.837/0.847/0.862, and 0.844/0.858/0.867 respectively.
This is stable enough to justify further contract investigation, but not a
production selection.

## Bounds and unresolved choices

The 64-active-level probe did not evict radius-3 multiplier-1 candidates; the
largest observed active count was 42 across all multiplier-1 candidates. It
also did not evict multiplier-2 candidates (largest active count 26).
Multiplier 0.5 reached 64 at radius 2 and produced four evictions, so it cannot
support a smaller capacity claim. The 32-evidence cap almost never bound at
multiplier 1 but did bind more often at multiplier 2 (about 0.99% of candidate
pivots at radius 3); a future contract should characterize that cap directly.
The 512-bar age bound is clearly active and requires a separate age-sensitivity
study before it can be frozen.

Pivot-time and confirmation-time are causally distinct and tests prove the
difference. Their all-universe radius-3/multiplier-1 aggregate behavior is
close (new 27.449 vs 27.144 and retest 12.647 vs 12.286 per 1,000 bars), so
this study does not choose between them. Pivot time has the simpler immutable
origin interpretation; confirmation time is still plausible.

## Conclusions

- Rejected: a global raw fixed width as a cross-symbol v1 configuration.
- Plausible for further study: causal preceding-range median scales, especially
  64 bars with multiplier 1, at radii 3 or 4; multiplier 0.5 is capacity-heavy
  and multiplier 2 is interaction-sparse.
- Unresolved: pivot-time versus confirmation-time freezing, age below/above
  512 bars, a final active-level cap, and retained-evidence cap.

The evidence does not defensibly support retaining a single frozen raw-width
v1 configuration across the 28-symbol universe. It does support a subsequent,
narrowly scoped design decision about a separately versioned adaptive-width
detector contract. This tranche deliberately does not create that contract or
expose it to a model.
