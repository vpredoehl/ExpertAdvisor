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

The catalog validates its lookup namespace before use. Duplicate family IDs,
duplicate Tensor columns, an unknown channel family, and collisions between a
hierarchical ID and any persisted flat ID reject explicitly rather than relying
on declaration order.

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
When a persisted experiment has an explicit semantic-layout identity,
scheduler admission and model materialization revalidate the concrete mask
against that stored layout. The intentional legacy NULL/NULL identity retains
only its historical concrete-mask compatibility; it is not a wildcard
fallback.

At new-experiment materialization the scheduler emits one bounded diagnostic:

```text
MARKET_STRUCTURE_REGISTRY_ACTIVE,semantic_layout=...,input_width=...,
registered_families=...,requested_ablation=...,resolved_ablation=...,
disabled_channel_count=...
```

## Detection, causality, and confluence

`MarketStructure::Observation` separates the timestamp at which a structure
occurred from `availableAt`, when the detector could causally know it. Its
immutable value contract includes source provenance, producer-owned source
observation ID, detector version, and a typed descriptor. A descriptor has a
nonempty schema version and role, an explicit `positive`, `negative`, or
`neutral` polarity, and an optional `[0,1]` normalized confidence. Confidence
is accepted only when a producer genuinely uses that normalized contract; the
descriptive engine never ranks or compares it across detector families.

The observation family is opaque at this boundary. It is intentionally not
restricted to the registry's `kFamilies`, because that catalog represents only
families with already-materialized Tensor channels. This permits an independent
causal producer to describe evidence before any Tensor integration. The
canonical observation identity length-prefixes all provenance, timing, and
descriptor fields. Duplicate causally available identities reject explicitly.

`CausallyAvailableObservations` first excludes observations with
`availableAt > decisionTime`, then validates the remaining decision prefix.
Thus a later observation, including a malformed one, cannot backfill or alter
the result at an earlier decision time. Every participating observation must
still satisfy `observedAt <= availableAt` once it becomes available. The
returned causal prefix is ordered by canonical observation identity.

`DescriptiveConfluenceEngine` is the Phase 2 generic implementation of the
`ConfluenceEngine` interface. It copies and freezes a validated immutable
`ConfluenceDefinition` at construction. Definitions contain an ID, version,
relation (`support` or `contradiction`), exact left/right family-role
selectors (which must differ), and a positive per-selector candidate cap.
Their canonical identity contains every one of those fields, so a version or
semantic change has a different identity.

For an evaluation, the engine filters the causal prefix, selects each role,
sorts candidates by canonical observation identity, and retains the canonical
prefix up to the declared cap. It reports matching, retained, and overflow
counts per selector. It emits at most two aggregate observations per
definition: one for positive-left and one for negative-left. SUPPORT requires
the same non-neutral polarity on both sides; CONTRADICTION requires the
opposite polarity. This is bounded descriptive aggregation, not an all-pairs
component product. Component ordering is canonical; output availability is
the latest component availability; output identity contains definition,
decision time, relation/polarities, availability, and canonical components.

`ConfluenceReplay::CanonicalRepresentation()` is the in-memory diagnostic and
replay record. It exposes definition identity, decision time, candidate and
overflow accounting, selected identities, output identity, relation result,
and full canonical component provenance/availability. Source observations are
copied into derived observations for replay only; the engine neither mutates,
suppresses, replaces, nor reinterprets raw detector evidence.

## Keyed neutral descriptive relationships (Phase 2C)

Phase 2C adds an optional `CorrelationKey` to an observation. It is immutable
relationship metadata, deliberately separate from the descriptor (meaning and
polarity) and provenance/source observation ID (origin). A key has a producer
chosen lower-case identifier type and an opaque printable value. Types are at
most 64 bytes, values at most 4096 bytes, and neither may be empty. The engine
does not parse, inspect, derive, or fuzzily compare a value. Equality means
the complete `(type, value)` pair is exactly equal, so equal text under two
different types never joins.

A definition opts in through `exact_key_equality` and supplies a positive
`maxCorrelationKeys`. Missing keys never match: one missing key and two
missing keys both fail the join. The keyed path first filters the causal prefix
and validates it, then selects family/role candidates, groups them by the full
canonical typed key, discards groups absent on either side, and only then
checks the relation's polarity and temporal condition. It retains qualifying
canonical key groups up to `maxCorrelationKeys` and applies the existing
selector cap independently inside each retained key group. Thus an unrelated
earlier candidate, non-qualifying key, or key cannot consume a global selector
cap and discard a valid same-key relationship. Duplicate canonical observation
identities still reject before this process. There is no component Cartesian
product.

`co_occurrence` is the only new relation kind. It means only that the two
selected neutral observations co-occur under a definition's exact key and
optional temporal condition. It is not support, contradiction, direction,
causation, confirmation, confidence, strength, prediction, profitability, or
trading action. In this narrow initial contract it accepts neutral observations
on both sides only; positive and negative observations do not silently acquire
neutral meaning. It does not synthesize confidence, alter polarity, or mutate
or suppress raw observations. SUPPORT and CONTRADICTION retain their prior
same/opposite non-neutral polarity semantics. They may also opt into a keyed
definition in a future reviewed definition, but no existing definition does.

Keyed results are one aggregate per retained key and qualifying polarity. A
neutral `co_occurrence` therefore emits at most `maxCorrelationKeys` outputs;
a keyed directional relation remains bounded by at most two outputs per
retained key. Every non-temporal result includes at most
`maxCandidatesPerSelector` components from each side of its own key group,
canonically retained. Output identity includes the key, relation, predicate,
definition, decision time, availability, and components.

The optional temporal predicate is
`left_available_at_before_right_available_at`. It is strict (`<`): equality
and reverse order fail. It is defined on `availableAt`, not `observedAt`, so it
describes causal knowledge and cannot backdate a relationship. It requires
exact key equality and a selector cap of one. For each retained key/polarity,
the engine deterministically selects one pair: the earliest available right
candidate for which a left candidate was already available, then the earliest
such left candidate (canonical identity breaks equal-time ties). This is a
linear scan of time-sorted candidates, not a pair product. Its output contains
exactly that pair and is available at the right component's time, which is the
latest component availability. Causal filtering still occurs before key or
temporal processing, so a future (even malformed) observation cannot affect an
earlier decision prefix.

Canonical identity is versioned without changing historical bytes. An
observation without a key keeps its exact `observation-v1` identity; a keyed
one uses `observation-v2` and length-prefixes the canonical key. Definitions
using none of the new options keep their exact `confluence-definition-v1`
identity, legacy output identity, and `confluence-replay-v1` representation.
Keyed definitions, outputs, and replays use their respective v2 forms and
include key constraint, predicate, and key cap. This preserves existing
Layout-11 definition and replay identities while preventing new semantic
definitions from colliding with one another.

This is descriptive infrastructure only. It adds no Tensor channels, model
input width/layout, ablation behavior, persistence, scheduler/worker behavior,
or production price-level definition. In particular, the Phase-2 price-level
bridge remains unkeyed in Phase 2C; a separately reviewed phase must explicitly
adopt this generic contract before defining any price-level relationship.

There are no generic confluence Tensor channels, trading scores, database
persistence, or `confluence.*` ablation arm in this phase. A later Tensor phase
must register any materialized channels as a separate `confluence` family in a
new semantic layout so raw evidence remains independently available.

## Production observation bridge (Phase 2B)

Phase 2B introduces `MarketStructureProductionObservationAdapter.hpp`, a
non-destructive bridge between the existing `TG4Pulse::ProductionStreamingAdapter`
and the generic observation/engine contract. It consumes an already-created
`TG4Pulse::Pulse` value; it neither owns nor invokes the TG1/TG3 detector and
does not alter Tensor construction, raw pulse bits, or any historical feature.
Pocket is deliberately not a participant in this bridge.

The sole participating producer is the existing `tg_structure` TG4 production
pulse (`detectorVersion=tg4-production-pulse-v1`). A pulse is the completed
canonical 15-minute bar `[barStart, barStart + 900s)`. Every resulting generic
observation uses `observedAt == availableAt == barStart + 900s`; source IDs
include the bar start and role, and provenance includes the canonical symbol
and the frozen TG4 production configuration hash. Consequently a completed-bar
classification cannot affect a decision prefix before the bar completion time.

The mapping is exact and categorical:

- `inner_break` is positive exactly when `inner_break_any == 1`.
- With an inner break, `structural_eligibility` is positive exactly when
  `structurally_eligible == 1`, otherwise negative.
- With structural eligibility, `fibonacci_retracement_relation` is positive
  exactly when `confluent == 1`, otherwise negative.

The source hierarchy `confluent <= structurally_eligible <= inner_break_any`
is validated. Absent/non-applicable source states emit no observation rather
than an invented negative signal. The categorical pulse has no established
within-producer normalized confidence, so every bridge observation leaves
confidence absent. This bridge is additive: its copied observations and engine
components never suppress, replace, mutate, or reinterpret the raw TG4 pulse.

The following immutable descriptive definitions are frozen at version `v1`.
Both use canonical observation-identity ordering and a cap of one candidate per
selector; their per-pulse input is bounded and the generic engine has no
pairwise expansion.

- `tg4-structural-fibonacci-retracement-support/v1`: `SUPPORT` between
  `tg_structure/structural_eligibility` and
  `tg_structure/fibonacci_retracement_relation`. It describes only the
  existing positive/positive completed-bar state.
- `tg4-structural-fibonacci-retracement-contradiction/v1`: `CONTRADICTION`
  between the same selectors. It describes only the existing positive/negative
  completed-bar state.

`TG4ProductionConfluenceBridge` returns the source observations, bounded
generic replay records, and bounded descriptive outputs for a supplied pulse
and decision time. Output availability is computed by the generic engine and
is no earlier than each retained component. The definitions are not a trading
rule, score, recommendation, gate, or outcome-trained interpretation; they are
only provenance-bearing descriptions of established source states.

These results are not Tensor/model input, are not persisted, and do not create
`confluence.*` ablation. Tensor width, layouts 8/9/10, existing raw detector
semantics, and LSTM architecture remain immutable. A later, separately gated
phase may consider a fixed Tensor adapter only after this descriptive bridge is
validated; Phase 2B adds no semantic layout or Tensor registration.

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
