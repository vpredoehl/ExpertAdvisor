# Market-structure registry and confluence boundary

## Scope of the current registry

`Headers/MarketStructureRegistry.hpp` is the common catalog for model-input
market-structure families. It currently registers only channels produced by
real, existing Tensor producers:

- `tg_structure` — the three layout-8 TG4 production pulse channels.
- `fibonacci` — the 23 layout-9 causal Fibonacci structural aggregate
  channels.
- `pockets` — the existing Phase Pocket 2 causal detector, which has no Tensor
  feature channel yet.

`elliott_wave`, `trendline`, `candlestick`, and generic `confluence` are
deliberately not registered yet. `pockets.*` fails today because its detector
does not yet emit a model-input channel; it is not silently accepted as an
empty ablation. Registering a feature channel without a causal producer would
create a misleading experimental arm.

Each catalog channel has a stable hierarchical feature ID, a family ID,
detector version, causal-availability contract, Tensor column, and the layout
in which it first appears. The old flat feature name remains the persisted
identity for existing channels because those names already appear in immutable
experiments. For example, the configuration ID
`fibonacci.up.recent.h1_count_log` resolves to the persisted channel name
`fib_up_recent_h1_count_log`.

The layout-9 Fibonacci producer is an immutable legacy composition over the
pre-existing TG1/TG3 research primitives. Its current implementation is not a
new independent Fibonacci detector, and this phase does not refactor it: doing
so would change the semantics of existing layout-9 experiments. Future
independent Fibonacci work must introduce a separate detector version and a
new semantic layout rather than repurposing layout 9.

## Hierarchical ablation

`--ablate-features` accepts existing flat names as before and now accepts
hierarchical exact names and trailing-prefix wildcards:

```text
fibonacci.*
fibonacci.up.recent.*
tg_structure.*
fibonacci.*,tg_structure.*
```

Only a trailing `.*` prefix wildcard is valid. A misspelled family, sub-family,
or channel fails explicitly; it is never ignored. The registry only exposes
sub-families that map to actual current channels. Thus
`fibonacci.retracement.*` is not an alias for a future, unimplemented feature
set.

The requested form is normalized for queue diagnostics without expanding a
wildcard. After the queue has resolved its authoritative
`model_input_semantic_layout_version`, the request is resolved, deduplicated,
and ordered by Tensor column. The resolved concrete flat list is written to
`experiment.feature_ablation_mask`; it remains the existing duplicate-
detection, continuation, model-lineage, training, and inference identity. No
schema migration is necessary because that immutable resolved list already
answers exactly which channels the experiment disabled.

Consequently, a historical request for `fibonacci.*` persists the 23 concrete
layout-9 Fibonacci names, not a wildcard. A later Fibonacci channel cannot
silently alter what that experiment means. Equivalent ordering of requests
produces the same resolved persisted identity.

Masks are applied after model-input prefix projection. They zero selected
canonical positions but do not change `model_input_width` or
`model_input_semantic_layout_version`. A family is also checked against the
experiment's semantic layout, so a layout-8 experiment cannot be given a
layout-9 Fibonacci mask. Training and every inference route use the same
`FeatureAblationMask` and `CopyTensorFeaturesForModelInput` implementation.

At new-experiment materialization the scheduler emits one bounded diagnostic:

```text
MARKET_STRUCTURE_REGISTRY_ACTIVE,semantic_layout=...,input_width=...,
registered_families=...,requested_ablation=...,resolved_ablation=...,
disabled_channel_count=...
```

## Detection, causality, and confluence

`MarketStructure::Observation` separates the timestamp at which a structure
occurred from `availableAt`, when the detector could causally know it. The
registry validates that availability is never earlier than observation, and
`CausallyAvailableObservations` filters inputs at the decision time. A future
detector must attach its source provenance and detector version and must not
backfill a later-confirmed state onto prior rows.

`ConfluenceEngine` is a separate interface. It consumes a const collection of
causally available detector observations and returns distinct descriptive
confluence observations. It has no mechanism to alter detector output. There
are no generic confluence Tensor channels or trading scores in this phase, and
therefore no generic `confluence.*` ablation arm yet. When concrete confluence
channels are introduced, they must be registered as a separate `confluence`
family in a new semantic layout so `confluence.*` can be masked without
masking Fibonacci, TG, or any other component family.

## Adding a family or a channel

1. Implement a causal independent detector and its availability/provenance
   contract. Do not make it call another detector.
2. Append actual Tensor channels and create a new semantic layout according to
   `ModelInputExpansion.hpp`; do not repurpose a layout number or infer
   compatibility from width.
3. Add the family and its channels to `MarketStructureRegistry.hpp`, including
   hierarchical IDs, detector version, layout membership, and causal timing.
4. Add feature-construction, causal-leakage, fixed-width masking,
   training/inference parity, and semantic-worker admission tests.
5. If explicit relationships are useful, implement them through a separate
   `ConfluenceEngine` and register its resulting channels under `confluence`.
   Do not put confluence logic into a detector.

The Fibonacci ratios and producer are unchanged by this registry. Layout-9
Fibonacci experiments continue to use their original flat persisted identities
and existing semantic-worker compatibility policy.
