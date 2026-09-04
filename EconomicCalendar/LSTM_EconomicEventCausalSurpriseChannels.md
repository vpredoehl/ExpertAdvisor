# LSTM causal economic-event surprise channels

## Contract

Semantic layout 6 appends two point-in-time first-release surprise channels to
the complete layout-5 Tensor prefix:

| Zero-based Tensor column | Stable feature name | Encoding |
| ---: | --- | --- |
| 71 | `causal_first_release_surprise_available` | 1 only for a PIT-visible, compatible scalar pair; otherwise 0 |
| 72 | `causal_first_release_surprise` | bounded normalized signed difference; 0 when unavailable |

The model input includes four return features, so the total width changes from
75 to 77. Width 75 remains registered as semantic layout 5 and projects the
unchanged Tensor prefix at columns 0 through 70. Layout 6 has append-only
predecessor 5. Resume expansion from width 75 to 77 zero-initializes exactly
Tensor columns 71 through 72 and records:

```text
new_tensor_features=causal_first_release_surprise_available|causal_first_release_surprise
new_tensor_columns=71:73
```

Migration 089 already supplies the durable experiment columns for model input
width and semantic layout. Phase 2 does not change that schema: new experiments
persist width 77/layout 6, while historical width 75/layout 5 identities remain
immutable. Migration 090 supplies the first-release actual provenance API.

## Surprise calculation

For the same relevant event selected by the existing economic-event pipeline:

```text
difference = first_release_actual.canonical_value_low
           - selected_consensus.forecast.canonical_value_low

normalized_surprise = difference
                    / EconomicEventNormalizationScale(event_family, unit)

causal_first_release_surprise = clamp(normalized_surprise, -10, +10)
```

Both intermediate and output values must be finite. The normalization function
uses only fixed family/unit constants:

| Canonical family | Required canonical unit | Divisor |
| --- | --- | ---: |
| CPI, PPI, PCE, GDP, DURABLE_GOODS, RETAIL_SALES, FOMC | percent | 10 |
| EMPLOYMENT, EMPLOYMENT_ANNUAL | count | 1,000,000 |
| JOLTS | count | 10,000,000 |

These constants contain no fitted or rolling data. Appending future events or
observations therefore cannot change a historical normalized value.

## PIT source and visibility

`LoadEconomicEventsForFeatureRange` performs one ordered range query. It joins
the existing `economic_event_selected_consensus` view and bulk-loads actuals
only through `economic_event_first_release_actual_at(endUtc)`. The
cutoff-independent `economic_event_first_release_actual` view contributes only
provenance state and selection reason; its value columns are not projected.
The feature engine then gates each bulk-loaded value for every completed bar:

```text
state == proven_first_release
AND proven_available_at <= information_cutoff
```

The inclusive boundary means the channel is unavailable immediately before
proved publication and available exactly at publication. A later revision can
change the audit-only canonical actual but cannot replace the selected first
release. An unproved late backfill remains `provenance_unavailable`. A directly
proved historical backfill becomes visible at its exact source publication
instant, not its observation or ingestion time. Conflicting possible earliest
values are `ambiguous` and unavailable; same-value earliest corroboration is
usable and deterministically selected.

The query retains the existing currency filter, event range/seed behavior,
same-timestamp importance and event-ID ordering, and relevant-event selection.
Having a valid actual never causes a different event to be selected.

## Compatibility and missingness

Phase 2 supports scalar/scalar surprise only. Forecast and actual must each
have a valid finite scalar shape, the same canonical unit, the same positive
persisted scale, and the same optional qualifier. No unit conversion, range
midpoint, or range subtraction is inferred. Missing consensus, missing or
not-yet-visible actual, provenance-unavailable or ambiguous state, range
values, incompatible semantics, and non-finite arithmetic all fail closed:

```text
available = 0
surprise = 0
```

A genuine zero difference is distinct:

```text
available = 1
surprise = 0
```

The diagnostic counters retain separate provenance-unavailable, ambiguous,
not-yet-available, missing-consensus, incompatible, and available dispositions,
plus available row counts by authoritative first-release source.

## Ablation handoff

Both names are independently registered in `FeatureAblation`. The recommended
Phase-3 controlled comparison uses identical data, seed, hyperparameters,
training objective, and checkpoint policy:

- CONTROL: no feature ablation;
- SURPRISE-ABLATION: ablate
  `causal_first_release_surprise_available,causal_first_release_surprise`.

Phase 2 prepares this identity but does not queue or run an experiment.
