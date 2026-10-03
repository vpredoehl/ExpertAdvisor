# Price-Level Phase 6 — production `causal-price-level/v2`

## Contract and provenance

Phase 6 promotes the characterized Phase 5A/5B/5C adaptive detector into the
explicit production definition `causal-price-level/v2`.  The fixed production
configuration is:

- pivot radius `3`;
- median of preceding completed-bar `high-low` ranges;
- scale lookback `64`, multiplier `1`, and `pivot_time` scale timing;
- maximum age `512` bars;
- maximum active levels `48`;
- maximum retained pivot evidence `32`; and
- completed-bar duration `900` seconds.

The `48` active-level and `32` retained-evidence limits are Phase 5C
engineering bounds, not optimized outcome parameters.  No target, return,
profitability, prediction, or model-performance information is involved.

For a pivot P, v2 snapshots its scale before P is consumed: it uses the
causally available predecessor prefix (up to 64 ranges), never P or later
bars.  Its width is that snapshot times the multiplier and is immutable after
establishment.  Reinforcing candidates must fall in the existing frozen zone;
they cannot expand, recenter, or chain it.  Eligible merges select nearest
anchor and then canonical identity.  A level is active through
`available_bar + maxAgeBars` and expires before interactions on the following
bar.

## Architecture and compatibility

`Headers/CausalPriceLevelV2Engine.hpp` is a separate header-only production
engine.  It owns the causal scale window and emits a v2-specific configuration,
level, observation, update, provenance, and identity representation.  Keeping
it separate avoids changing the frozen fixed-width
`Headers/CausalPriceLevelEngine.hpp` implementation or making production
depend on the research harness.  v1 remains `causal-price-level/v1` with its
existing configuration, identities, observations, merge and expiry behavior.

The deterministic v2 configuration identity includes the definition id and
version plus every relevant scale, pivot, age, capacity, evidence, and bar
duration value.  Research identifiers such as
`price-level-adaptive-width-study-v1` are never used as v2 production
identities.

## Parity evidence and tests

`Tests/CausalPriceLevelV2EngineTests.cpp` replays focused synthetic sequences
through both v2 and the Phase 5 adaptive detector.  It compares causal scale
samples, strict-pivot counts, semantic event payloads, and active state on
every bar.  The sequences cover startup prefix behavior, confirmation-delay
independence, strict confirmation, frozen width and reinforcement,
nearest-anchor/identity ties, no chaining, touch/cross/retest/role behavior,
the Phase 5B expiry boundary, active-cap eviction, and retained-evidence
saturation.

The existing v1 and characterization suites remain regression coverage.

## Deferred boundary

This phase adds no Layout 11 or model feature exposure, tensor-width change,
training/inference semantic change, database change, scheduler behavior, or
experiment activation.  A later explicitly scoped phase may bridge immutable
v2 detector observations into a model-facing boundary.
