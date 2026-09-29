# Phase Pocket 5B Continuation Handoff

## Checkpoint

- Repository checkpoint: `c2165828672412898785ead15cc2a3da33e0f37e`
- Subject: `Implement causal Pocket feature producer`
- Expected starting condition for the next tranche: clean worktree at that commit.

## Tranche 1 completed

Added exactly these files:

- `Headers/CausalPocketFeatures.hpp`
- `Tests/CausalPocketFeaturesTests.cpp`
- `Tests/CausalPocketFeaturesTests.sh`

The header-only producer consumes the source-default Phase Pocket 2 L=15
`CausalPocketDetector` stream and immutable confirmed observations. It freezes
the 11-channel Phase Pocket 5A representation: cutoff-age eligibility 0..20,
independent bull/bear populations, `log1p` counts, normalized youngest age,
lower-middle p50 medians, direction-oriented boundary distances,
nonnegative normalized width, causal completed-bar Wilder ATR-14 with canonical
pip fallback, invalid-scale geometry zeroing, deterministic retention/eviction,
and duplicate identity guarding. It introduces no Pocket lifecycle, outcome,
touch, fill, or traversal behavior.

Tests run and passed:

- `bash Tests/CausalPocketFeaturesTests.sh`
- `bash Tests/CausalPocketDetectorTests.sh`
- `git diff --check`

An adversarial review found no defect requiring a change. It specifically
reviewed causality, confirmation availability, ages 0/20/21, detector
integrity, retention, orientation, median behavior, scale fallback/validity,
streaming behavior, duplicate handling, startup behavior, and test coverage.
The worktree was clean after commit.

## Tranche 2 implementation map: semantic contract/layout only

Tranche 2 production scope only:

- `Headers/FeatureLayout.hpp`
- `Headers/ModelInputContract.hpp`
- `Headers/ModelInputExpansion.hpp`
- `Headers/MarketStructureRegistry.hpp`
- `Headers/FeatureAblation.hpp`

### Layout and widths

- Layout 9 Tensor width remains `99`; model-input width remains `103`.
- The stable return suffix remains four columns.
- Layout 10 appends 11 Pocket Tensor columns: Tensor width `110`; model-input
  width `114`.
- Layout 10 predecessor is layout 9: `{10, 114, 9}`.
- Layout-9-to-layout-10 expansion copies Tensor rows `0..98`, zero initializes
  Pocket rows `99..109`, and relocates return rows `99..102` to `110..113`.
- Historical layout-9 width 103 must remain valid and project only Tensor
  columns `0..98`.

### Frozen Pocket columns

```text
99  pocket_recent_price_scale_valid
100 pocket_bull_recent_count_log
101 pocket_bull_youngest_age20
102 pocket_bull_median_touch_distance
103 pocket_bull_median_close_distance
104 pocket_bull_median_width
105 pocket_bear_recent_count_log
106 pocket_bear_youngest_age20
107 pocket_bear_median_touch_distance
108 pocket_bear_median_close_distance
109 pocket_bear_median_width
```

### `FeatureLayout.hpp`

Append constants beginning at the existing Fibonacci feature size 99. Set
`feature_size` to 110. Preserve every prior offset. Add assertions pinning
Fibonacci size 99, first Pocket column 99, last Pocket column 109, exactly 11
new columns, and total feature size 110.

### `ModelInputContract.hpp`

Add `kCausalPocketRecentObservationModelInputWidth = 114`. Keep
`kCausalFibonacciStructuralModelInputWidth = 103` registered. Append 114 to
the registered-width table and make it current. Keep explicit projection
contracts: 103 maps to Tensor width 99, and 114 maps to Tensor width 110.
Do not use the global new `feature_size` for the historical 103 projection.

### `ModelInputExpansion.hpp`

Set current semantic layout to 10. Append `{10,
kCausalPocketRecentObservationModelInputWidth, 9}` to the layout registry.
Increase appended semantic names from 67 to 78; retain prior 67 byte/order
stable and append the 11 frozen underscore names in physical order above.
The existing generic expansion mechanism should provide the required row copy,
zero initialization, and suffix relocation without changing historical edges.

### `MarketStructureRegistry.hpp`

Keep the existing Pocket family record unchanged:

```text
{"pockets", 1, "causal-pocket-detector-phase2-v1",
 "confirmation bar / information cutoff timestamp"}
```

Increase channel records from 26 to 37 and append these layout-10 identities,
with family `pockets` and corresponding persisted names/columns above:

```text
pockets.recent.price_scale_valid
pockets.bull.recent.count_log
pockets.bull.recent.youngest_age_20
pockets.bull.recent.median_touch_distance
pockets.bull.recent.median_close_distance
pockets.bull.recent.median_width
pockets.bear.recent.count_log
pockets.bear.recent.youngest_age_20
pockets.bear.recent.median_touch_distance
pockets.bear.recent.median_close_distance
pockets.bear.recent.median_width
```

Availability must be explicit: Pocket channels unavailable in layouts 8 and 9,
available in layout 10. Preserve existing TG/Fibonacci availability. Update
default prefix resolution to semantic layout 10; do not use a broad comparison
that would expose future channels to old layouts.

### `FeatureAblation.hpp`

Increase the ablatable mapping table from 47 to 58. Append the frozen
underscore persisted identities mapped to columns 99..109 in exact physical
order. Keep every historic ablation identity unchanged. Update the default
semantic layout resolution from 9 to 10 so current layout registry resolution
can expose Pocket channels. No new Pocket ablation group constant was required
by the established tranche-2 map.

## Focused tranche-2 tests

- `Tests/MarketStructureRegistryTests.cpp`
  - `pockets.*` resolves to no channels at layouts 8 and 9 and exactly 11 at
    layout 10.
  - Verify order, identities, columns, family, and introduced layout.
  - Verify existing TG/Fibonacci layout-8/layout-9 behavior is unchanged.
  - Verify current 110/114 contract and Pocket ablation affects only Pocket
    columns, not return suffix columns.

- `Tests/LSTMModelInputCompatibilityTests.cpp`
  - Width 103 remains valid with Tensor projection 99.
  - Width 114 projects 110 Tensor columns.
  - Layout-9 projection ignores columns 99..109.
  - Current width is 114.

- `Tests/LSTMInputWidthExpansionTests.cpp`
  - Direct 103-to-114 expansion verifies all 11 semantic names, zero Pocket
    rows, and return relocation 99..102 to 110..113.
  - Adjust synthetic future-layout simulations because layout 10 becomes real.

- `Tests/LSTMFeatureVectorParityTests.cpp`
  - Update current width assertions and verify the return suffix is 110..113.

Suggested focused commands after implementation:

```bash
bash Tests/MarketStructureRegistryTests.sh
bash Tests/LSTMModelInputCompatibilityTests.sh
bash Tests/LSTMInputWidthExpansionTests.sh
bash Tests/LSTMFeatureVectorParityTests.sh
git diff --check
```

## Known hard-coded current-layout/current-width fallout

Direct expected fallout includes tests and code that hard-code current layout 9,
Tensor width 99, model width 103, or reserve layout 10 as hypothetical.

Known deferred callers:

- `Tests/TG4TensorModelInputIntegrationTests.cpp`
- `Tests/SemanticWorkerRegistryTests.cpp`
- `Sources/FeatureAblationReplicationEvaluation.cpp`

The latter needs a later narrow historical compatibility repair; it must not be
broadened into tranche 2. This handoff retains only the audit findings already
established; other possible hard-coded callers are uncertain until a future
scoped implementation review.

## Explicit deferred work

Do not include in tranche 2:

- Tensor integration or changes to `Tensor.hpp` / `LSTM/Tensor.cpp`
- `TG4TensorModelInputIntegrationTests`
- `SemanticWorkerRegistryTests`
- historical `FeatureAblationReplicationEvaluation` compatibility repair
- semantic-worker deployment artifacts
- scheduler, database, experiments, profitability, or ranking work
- Phase Pocket 4 artifacts or Phase Pocket 5A freeze changes
- CausalPocketDetector or Pocket producer changes
- training/inference execution or experiment materialization

## Scientific and architectural constraints

- Phase Pocket 5A freeze remains authoritative; do not reinterpret or optimize
  its semantics.
- Leave the Phase Pocket 2 detector unchanged, and retain source-default L=15
  producer semantics.
- Preserve the 11-channel order exactly.
- Preserve causal confirmed-observation semantics; do not introduce an active
  Pocket lifecycle, touch/fill termination, outcome dependency, nearest/newest
  selection, or future information.
- Preserve historical layout-9 compatibility and all historic feature offsets,
  semantic names, registry identities, model widths, and expansion behavior.
- Preserve the stable four-return suffix.
- Do not alter Tensor implementation, scheduler, database, experiments,
  ranking/profitability, or frozen Phase Pocket 4/5A artifacts in tranche 2.

## NEXT ACTION

The next session should start from clean HEAD `c2165828` and implement Phase
Pocket 5B tranche 2 only, following this semantic-contract/layout map. It must
not automatically continue into Tensor integration.
