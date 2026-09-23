# TG4 isolated production streaming adapter

Status: complete isolated prerequisite. This increment creates only a
production TG1->TG3 pulse adapter. It makes no Tensor, model, layout, width,
persistence, database, scheduler, worker, experiment, or frozen-artifact
change.

## Architecture and ownership

`EA::TG4Pulse::ProductionStreamingAdapter`
(`Headers/TG4ProductionStreamingPulseAdapter.hpp`) owns one
`TG3::CausalFibonacciConfluenceIntegration`, which already composes TG1A
geometry, TG1B frozen classification, TG2 break/pairing, and TG3 A/B and
confluence.

```text
canonical completed Feature bar
  -> ProductionStreamingAdapter::AddCompletedCanonicalBar
  -> existing TG1A -> TG1B -> TG2 -> TG3 streaming pipeline
  -> immutable same-call TG3 observation snapshot
  -> deterministic OR aggregation
  -> Pulse {barStart, [three bits]}
```

The adapter does not read TG4 CSVs, terminal evaluator output, outcomes, or
research configuration files. `TG3::Update` now carries
`newlyCreatedConfluenceObservations`, copied immediately after creation before
later events, outcome synchronization, finalization, or retention can affect
tracker-visible storage. The adapter consumes only that snapshot.

## Input, output, and timestamp contracts

The input is one `Feature` obtained from
`MarketData::LoadCanonicalHalfOpenCandlesticks`: a completed canonical
15-minute OHLC bar with absolute UTC `PriceTP` start `T`, selected under
`[start,end)`, in strict chronological order. DST/weekend gaps retain available
bars without synthesis. The adapter creates no competing reader; legacy
`LoadCandlesticks` remains unchanged.

Each input returns exactly one `Pulse`, keyed by the supplied `bar.time`:

```text
[tg4_inner_break_any,
 tg4_source_tg3_structurally_eligible,
 tg4_source_tg3_confluent]
```

The only valid values are `[0,0,0]`, `[1,0,0]`, `[1,1,0]`, and `[1,1,1]`, with
`confluent <= eligible <= inner_break_any`. A break generated on completed
bar-start `T` is returned for `T`, never `T+1`. The signals are one-bar pulses:
there is no carry-forward, decay, geometry, outcome, or duration channel.

## Aggregation, determinism, and retention

Multiple same-bar TG3 events are OR-reduced, so ordering cannot change the
result. Any Inner break sets bit 0; any non-ineligible event sets bit 1; any
confluence event sets bit 2. Returned pulses are values, not references to
TG2/TG3 outcome records. Later bars may update/censor outcomes, but cannot
rewrite a returned pulse. Prefix replay is consequently invariant to an
arbitrary later tail.

## Configuration, capacity, and directional policy

The adapter constructs only
`ProductionTG1TG3Pulse::Configuration::TG4ADerivedSourceUTLUpABOnlyV1()` and
uses its immutable payload/hash identity plus symbol-specific pip tolerance.
It duplicates no settings. TG1 fractal/candidate and TG3 fractal/A-B limits are
pulse-semantic and remain in that identity; TG2/TG3 outcome retention is
operational and cannot rewrite a captured row.

The configured TG3 policy is `source_utl_up_ab_only`. DTL structures may be
recognized by the underlying pipeline but are structurally ineligible and
cannot set confluence. No down-Fibonacci policy was introduced.

## Tests and results

`Tests/TG4ProductionStreamingPulseAdapterTests.sh` compiles with warnings as
errors and drives a deterministic 5,000-bar fixture through the real production
TG1->TG3 composition. It proves all four states, hierarchy, exact bar-start
mapping, byte-identical repeated replay serialization, prefix/future-tail equality, OR
commutativity, factory-owned capacity/configuration identity, and DTL rejection.

`Tests/TG4ProductionStreamingCanonicalReplayTests.sh` is a read-only historical
replay. It reads `eurusdrmp` twice using
`LoadCanonicalHalfOpenCandlesticks` over absolute UTC
`[2025-03-05 05:00:00, 2025-03-06 05:00:00)`, validates 96 ordered canonical
bars and exact half-open membership, then proves identical OHLCV/bar identity
and adapter pulses. It uses both `PGOPTIONS=default_transaction_read_only=on`
and `SET TRANSACTION READ ONLY`.

No exact comparison with the frozen evaluator is asserted: that evaluator owns
study partition/outcome-finalization records and does not emit a three-bit row
for every input bar. Equating terminal research records to adapter rows would
manufacture equivalence across distinct contracts. The bounded canonical replay
proves the relevant production property: deterministic direct causal replay.

## Readiness

The isolated prerequisite is complete. A separately approved layout-8/width-80
phase must still decide Tensor placement, warmup sharing, train/inference
parity, configuration persistence, and checkpoint/model compatibility. None is
implemented here. The repository is ready for that separate design/integration
phase; its current contract remains width 77 and semantic layout 7.
