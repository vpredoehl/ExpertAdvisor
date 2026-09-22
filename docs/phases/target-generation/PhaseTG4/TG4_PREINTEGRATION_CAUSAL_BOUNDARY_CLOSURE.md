# TG4 pre-integration causal-boundary closure

Status: complete prerequisite increment. This closes the range and
configuration blockers identified in
`TG4_CAUSAL_BOUNDARY_CLOSURE.md`; it does not implement a streaming adapter or
connect any TG state to Tensor, training, inference, checkpoints, workers or
the scheduler.

## Scope and non-goals

This increment adds only a canonical market-range reader contract and a
production-owned TG1--TG3 pulse configuration identity. It adds no Tensor
field, no layout, no feature-width change, no schema migration, no persistence
write, no model/checkpoint reinterpretation, no adapter, and no TG4 rerun.
`feature_size` remains 77 and semantic layout remains 7.

## Canonical market-range contract

`EA::CanonicalMarketData::AbsoluteHalfOpenRange` in
`Headers/CanonicalMarketDataRange.hpp:21-39` is the canonical API. Callers
pass absolute UTC `PriceTP` instants only, with `start < end`; civil timestamps
are not accepted from callers. It returns completed canonical 15-minute bar
starts satisfying:

```text
[start, end)     start inclusive; end exclusive
```

The source `candlestick.dt` remains a naive America/New_York civil bar start.
`CanonicalHalfOpenCandlestickCte` converts the absolute bounds to New York
civil only at the `candlestick(...)` call and filters `dt AT TIME ZONE
'America/New_York'` back against the original absolute bounds
(`CanonicalMarketDataRange.hpp:57-76`). Thus a bar at exactly `end` is
excluded, every emitted bar is ordered `ORDER BY dt`, and an empty range with
no source bars returns no rows. The interval identity is the 15-minute
bar-start instant (`kCanonicalIntervalSeconds == 900`), not a close time.

DST changes are ordinary absolute-time comparisons. The reader neither accepts
ambiguous/nonexistent civil input nor fabricates DST, weekend, or other gap
bars; it retains available source bars in timestamp order. This is consistent
with the existing New York civil-to-UTC `PriceTP` parser in
`Common/HistoricalFxTimestamp.cpp:199-255`.

### Compatibility and shared API

`MarketData::LoadCanonicalHalfOpenCandlesticks` is the normal production
market-data component's explicit new API
(`Sources/MarketDataCore/MarketDataCore.hpp:34-38`,
`MarketDataCore.cpp:49-64`). `TG4::HistoricalMarketDataRepository` uses the
same CTE in both preflight and streaming reads
(`Sources/TG4HistoricalMarketDataRepository.cpp:16-26,42-77`). This provides
the single future Tensor/TG reader definition.

`MarketData::LoadCandlesticks` is deliberately unchanged: it accepts legacy
New-York civil strings and preserves its inclusive-right `candlestick` behavior
for existing `ModelInputPreparation::Prepare`
(`Sources/ModelInputPreparation/ModelInputPreparation.cpp:19-31`). Changing
that behavior would alter existing training/inference range materialization.
An approved future model-input migration must opt into
`LoadCanonicalHalfOpenCandlesticks`, establish persisted-range compatibility,
and then remove the legacy path only under its own approval.

### Parity evidence

`Tests/TG4PreintegrationCausalBoundaryClosureTests.cpp` runs two independent
consumer loops against the same canonical contract. It asserts exact ordered
timestamp/OHLC/900-second interval equality and deterministic repeated reads
for ordinary, endpoint, empty, single-row, DST spring, DST fall and
weekend-gap examples. It also verifies that the generated shared SQL uses the
exclusive endpoint. The test is a hermetic focused fixture: it does not write
or require a database. `Tests/TG4CanonicalMarketDataParityTests.sh` is the
complementary read-only PostgreSQL proof: ordinary=96, spring=196,
fall=188, and weekend-gap=192 rows; each case had equal normal/TG canonical
count, timestamp/OHLC/order and zero exact-endpoint rows. The legacy
production/TG4 historical endpoint mismatch is therefore preserved rather than
concealed.

## Production TG1--TG3 configuration

`EA::ProductionTG1TG3Pulse::Configuration::TG4ADerivedSourceUTLUpABOnlyV1()`
in `Headers/ProductionTG1TG3PulseConfiguration.hpp:345-357` is the
source-owned immutable factory. It has no filesystem/artifact loader. The
factory is field-for-field equivalent to the frozen TG4A configuration evidence
in `Scripts/tg4_analysis_config.frozen_v1.conf`, while keeping that file out of
the production runtime path.

The semantic inventory in its canonical payload (`:178-243`) is:

- canonical range contract, 15-minute interval, UTC bar-start identity,
  ascending order and no-synthetic-gap policy;
- TG1 strict five-completed-candle/radius-2 confirmation, geometry tolerances,
  anchor/candidate lookbacks and capacities, and ATR period;
- TG1B reference scale 14, inclusive 12--20/25--40/45--85 degree bands, and
  creation-time ATR-normalized classification snapshot;
- TG2 close-beyond-line break, rearm, nearest coexisting same-direction Outer
  pairing, and break tolerance;
- TG3 most-recent prior opposite fractal A/B policy, ratio set
  `0.6180339887498949`, `source_utl_up_ab_only`, exact-retracement absolute
  tolerance, canonical-pip conversion/map, and retained fractal/A-B bounds.

There is no DTL/down-Fibonacci option in the factory. The TG3 implementation's
diagnostic enum remains outside this production factory selection.

TG2 outcome/retest and TG3 confluence-observation retention limits are
immutable operational settings, separately serialized by
`OperationalRetentionPayload` (`:247-273`) but excluded from semantic pulse
identity because they cannot change a pulse captured immediately from a newly
created observation. In contrast TG1 confirmed-fractal/candidate limits and
TG3 confirmed-fractal/A-B limits are semantic and are included.

### Payload and identity

The canonical payload is newline-delimited `key=value`, stable-key ordered
text. It uses classic-locale hexadecimal floating-point representation and
sorts retracement ratios and the symbol/pip map before serializing. The hash is
the repository's established deterministic `fnv1a64:` convention
(`ProductionTG1TG3PulseConfiguration.hpp:60-74`), over the complete semantic
payload. For example, the identity takes this shape:

```text
configuration_schema=tg1-tg3-causal-pulse-configuration-v1
name=tg4a-derived-source-utl-up-ab-only-v1
market_range_contract=canonical-absolute-half-open-candlestick-v1
...
tg3_directional_policy=source_utl_up_ab_only
```

The exact `fnv1a64:<16 lowercase hex digits>` is intentionally derived at
runtime from those bytes rather than copied from a research artifact. A changed
semantic value produces a changed payload/hash; tests mutate every configurable
pulse-affecting tolerance, capacity, policy/value group and verify this.

## Future persistence requirement

When—and only when—a later approved TG layout/model phase exists, persist the
tuple `(configuration_schema, configuration_name, canonical_payload,
configuration_hash)` with the experiment materialization, model metadata and
each checkpoint. Training, resume and inference must require byte/hash
equality. This is additive to existing width/layout identity; no field, schema,
or write path is modified here.

## Tests and result

`Tests/TG4PreintegrationCausalBoundaryClosureTests.sh` passed. It compiles the
focused C++ fixture with `-Wall -Wextra -Werror -pedantic`, verifies contract
parity/DST/gap/endpoint/repeat behavior, payload/hash stability, semantic
capacity participation, operational-capacity separation, frozen-TG4A constant
equivalence, and absence of production runtime references to frozen artifacts
or Tensor/layout symbols.

`Tests/TG4CanonicalMarketDataParityTests.sh` passed under
`default_transaction_read_only=on`; it made no database writes. Existing TG1,
TG2, TG3, causal-boundary, width-expansion and model-input compatibility tests
also passed. `MarketDataCore` built successfully in Release. The whole
`LSTM Release` scheme compiled the affected targets but its final provenance
script correctly refused to run on this uncommitted worktree; no executable was
launched.

## Remaining blockers and readiness

There are no remaining blockers for a separately approved **isolated**
TG1->TG3 streaming-adapter phase. That phase must consume the canonical API and
factory above and remain disconnected from Tensor/model paths. Tensor feature
integration, persistence/schema work, layout 8, width 80, and any experiment
remain separate blocked work requiring explicit approval.
