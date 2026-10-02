# Price-level MarketStructure observation bridge — Phase 2

## Definition and scope

`Headers/PriceLevelMarketStructureObservationAdapter.hpp` defines the
stateless `price-level-market-structure-observation-bridge/v2` adapter. It
copies immutable `causal-price-level/v1` observations into the generic
`MarketStructure::Observation` boundary. Its opaque, pre-Tensor family is
`price_level`; the Tensor-only `MarketStructureRegistry::kFamilies` catalog is
intentionally unchanged.

The bridge is one-way. It neither owns nor invokes `CausalPriceLevelEngine`,
does not alter an `Update`, source observation, or detector state, and never
feeds MarketStructure or confluence results back to the detector. There is no
database, scheduler, worker, experiment, model, or Tensor dependency.

## Source and event mapping

The source detector remains the frozen `causal-price-level/v1` contract. Every
included source record is copied with its exact `observedAt`, `availableAt`,
and complete canonical source observation identity. The bridge descriptor role
is the exact canonical source event kind:

- `level_established`
- `level_reinforced`
- `touch`
- `cross_up`
- `cross_down`
- `retest`
- `role_reversal`

`level_expired` and `level_evicted` are deliberately excluded. They are
explicit Phase-1 removal diagnostics, not affirmative level/interactions for a
future relationship definition. Exclusion only affects this copied
MarketStructure stream: the detector continues to emit the immutable removal
records unchanged.

## Timing, polarity, and confidence

For every emitted observation:

```text
marketStructure.observedAt  = source.observedAt
marketStructure.availableAt = source.availableAt
```

No pivot, confirmation, interaction, removal, or role transition is backdated.
Decision-time consumers must use `CausallyAvailableObservations`, which admits
only records with `availableAt <= decisionTime`; its filter precedes validation
so later source records cannot alter an earlier decision prefix.

All descriptors use `neutral` polarity. `support_like`, `resistance_like`,
`cross_up`, and `cross_down` are typed detector state/event labels, not an
exact bullish, bearish, positive, or negative MarketStructure polarity. Phase
1 has no normalized confidence, so confidence is always absent. The bridge
also supplies no numeric descriptor or ML channel; source fixed-anchor/bound
information remains traceable through the copied canonical source identity and
frozen configuration provenance rather than being normalized or reinterpreted.

## Correlation key

Bridge v2 adds immutable generic relationship metadata without changing a
Phase-1 source observation or its descriptor. Every included observation has:

```text
type  = price_level.level_identity.v1
value = exact source.levelIdentity
```

The value is the complete frozen Phase-1 level identity: canonical symbol,
complete detector configuration identity, and the first pivot's frozen kind,
bar coordinate, bar start, and exact canonical price. It is neither a current
price nor a nearest-level/proximity lookup, insertion-order value, mutable zone
center, role/family label, fuzzy floating-point comparison, future evidence, or
trading outcome. Generic validation applies before output; a malformed or
oversized key fails closed.

## Identity, provenance, validation, and ordering

The MarketStructure detector version is `causal-price-level/v1`. Its source
observation ID is the complete Phase-1 canonical observation identity, which
contains the level identity, source evidence identity, event, times, and any
role transition. Bridge provenance contains its own version, normalized
canonical symbol, and a length-prefixed copy of the real Phase-1 provenance;
that provenance includes the complete frozen detector configuration identity.
The existing `CanonicalObservationIdentity` then provides the canonical,
locale-independent identity for copied observations. Because a correlation key
is identity-bearing generic metadata, every v2 bridge output has
`observation-v2` identity and v2 bridge provenance. The unchanged Phase-1
source observation identity remains exactly in `sourceObservationId`; the
descriptor schema remains `price-level-market-structure-observation-v1`.

The adapter validates the canonical Phase-1 source identity, source/level
provenance linkage, canonical symbol, timing, known event kind, and valid role
transitions. Its `Update` overload additionally rejects non-finite or
inconsistent active-level anchor/bounds. Duplicate source identities and
duplicate output identities reject. Outputs sort ascending by canonical
MarketStructure observation identity. The adapter retains no state; identical
source replay therefore yields canonical-equivalent output, and adding future
source events cannot change the causally available result at an earlier time.

## Confluence relationship and non-goals

The resulting observations are structurally acceptable inputs for a later,
explicit generic `DescriptiveConfluenceEngine` definition. Phase 2 defines no
such relationship and invokes no confluence engine. Neutral polarity prevents
this bridge from implicitly creating directional support or contradiction.

This phase does not modify Phase-1 detector semantics, Tensor features,
FeatureLayout, Layout 11, model-input width 116, FeatureAblation,
ModelInputExpansion/ModelInputContract, LSTM architecture, existing TG,
Fibonacci, Pocket, economic, or General-Confluence semantics. It does not
publish semantic workers, touch the database/scheduler, materialize or alter
experiments, or launch training/inference.

Later work, if separately designed and approved, may define exact price-level
confluence relationships and independently decide whether any fixed projection
deserves a new Tensor/layout contract. Neither is part of this bridge.
