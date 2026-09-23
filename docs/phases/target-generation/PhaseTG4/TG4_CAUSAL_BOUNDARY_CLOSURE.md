# TG4 causal boundary closure

Status: the audit/test increment is closed. Its two pre-integration blockers
were implemented by `TG4_PREINTEGRATION_CAUSAL_BOUNDARY_CLOSURE.md`, and the
isolated prerequisite adapter is recorded in
`TG4_ISOLATED_PRODUCTION_STREAMING_ADAPTER.md`. Tensor/model integration is
still separate. This document makes no Tensor, model, schema, scheduler,
worker, experiment, database, or frozen-artifact change.

## 1. Executive summary

The row key is now unambiguous: `Feature::time` is the **New-York-civil bar
start converted to a UTC `PriceTP` instant**. `Tensor::Add` appends one row in
source order and retains that same `Feature::time`; therefore a TG pulse made
from completed bar `T` belongs in the Tensor row appended for the same
canonical bar-start instant `T`, at source/Tensor index `i`—never at `i+1` and
never at a fractal's anchor bar. The row represents the completed interval
`[T, T + 15 minutes)`.

TG1--TG3 can deterministically emit the proposed `[inner_break_any,
structurally_eligible, confluent]` pulse on that completed break row. Outcome
synchronization, finalization, and outcome-record eviction cannot revise an
already captured pulse. Some retained-state bounds *can*, however, change
which later candidates/A-B structures exist and hence later pulse bits. Those
bounds are semantic configuration, not hidden operational tuning.

The market readers are **not currently identical at their right endpoint**.
Normal `MarketData::LoadCandlesticks` passes direct New-York civil timestamps
to `candlestick(...)`, whose materialized-view path uses inclusive `BETWEEN`.
TG4 converts UTC instants to New-York civil bounds but then applies an explicit
half-open UTC filter. The read-only DST probe found 197 normal rows versus 196
TG rows for equivalent 2025-03-07/11 bounds; the sole mismatch is normal's
`2025-03-11 00:00:00` bar. This is a production integration blocker until one
authoritative endpoint/time-basis contract is selected and both paths use it.

Production must own a new immutable TG1--TG3 configuration object and its
canonical fingerprint. It must not load
`Artifacts/tg4-confirmation-2025-v1` or the frozen research configuration at
runtime. The current 77-column/layout-7 model contract remains unchanged.

## 2. Paths inspected

| Concern | Evidence |
| --- | --- |
| Tensor row construction/timestamp retention | `Headers/PricePoint.hpp:31-38`; `LSTM/Tensor.cpp:102-180,443-449`; `Headers/Tensor.hpp:156-186` |
| Normal input loading | `Sources/ModelInputPreparation/ModelInputPreparation.cpp:19-31,76-85`; `Sources/MarketDataCore/MarketDataCore.cpp:23-46` |
| Database bar definition | `Database/forex/candlestick.plpgsql:21-37,46-108,113-161`; live read-only `pg_get_functiondef` audit on 2026-09-22 |
| DB-to-`Feature` conversion | `Common/db_cursor.cpp:76-105`; `Common/HistoricalFxTimestamp.cpp:199-255` |
| TG historical loading | `Sources/TG4HistoricalMarketDataRepository.cpp:15-75,101-123` |
| Completed-bar/feature time semantics | `Sources/EconomicEventFeatures.hpp:197-215`; `Sources/EconomicEventBarAlignment.cpp:47-109` |
| TG1--TG3 creation/confirmation | `Headers/CausalFractalTrendLineGeometry.hpp:147-181,321-345`; `Headers/CausalFibonacciConfluenceIntegration.hpp:1234-1269` |
| TG configuration/fingerprint | `Headers/TG4HistoricalEmpiricalEvaluation.hpp:38-57,257-281`; `Sources/TG4HistoricalEmpiricalEvaluation.cpp:915-1102,1128-1171,1459-1546` |
| Capacity behavior | `Headers/CausalFractalTrendLineGeometry.hpp:533-604`; `Headers/CausalTrendLineBreakRetestBehavior.hpp:803-860`; `Headers/CausalFibonacciConfluenceIntegration.hpp:741-760,862-915` |
| Existing model identity | `Headers/ModelInputContract.hpp:16-71,198-225`; `Headers/ModelInputExpansion.hpp:22-59`; `Headers/PgModelIO.hpp:359-407` |

## 3. Exact timestamp contract

`candlestick_cur` labels intraday aggregates with `date_trunc(higher_tf,time)
+ mn * interval` (`Database/forex/candlestick.plpgsql:21-37`), and the
production feature engine explicitly calls input 15-minute timestamps “bar
starts” and defines the completed information interval as `[barStart,
barStart + 15m)` (`Sources/EconomicEventFeatures.hpp:199-205`). `cst.dt` is a
naive historical New-York civil timestamp. Both normal loading and TG4 parse
that civil timestamp with `HistoricalFxTimestamp::ParseNewYorkCivilTimestamp`:
normal through `db_cursor_stream<Feature>` (`Common/db_cursor.cpp:86-105`) and
TG4 directly (`Sources/TG4HistoricalMarketDataRepository.cpp:66-75`). The
parser rejects nonexistent/ambiguous New-York civil DST times
(`Common/HistoricalFxTimestamp.cpp:251-255`). Thus `Feature::time` and
`TG1A::Candle.timestamp` are the same absolute UTC epoch-second representation
of a **bar start**, not a close timestamp.

`Tensor::Add` is invoked once per loaded `Feature` in source order and pushes
the feature time into `raw_time` at the same row index
(`ModelInputPreparation.cpp:76-85`; `Tensor.cpp:167-172,443-449`). It computes
the row only after the bar is complete—the code explicitly describes a row as
a “just-completed bar” (`Tensor.cpp:102-141`).

## 4. TG-to-Tensor row mapping

For ordered market bars `B[0..n)`, let `T = B[i].timestamp`, the bar-start
instant. After that physical interval completes:

```text
Tensor::Add(Feature for B[i])             -> Tensor row i, RawTimeAt(i) = T
TG1--TG3 AddCompletedBar(Candle for B[i]) -> only new break observations at i
new TG3 observation for B[i]              -> OR pulse into Tensor row i
```

TG3 sets `Update.bar` from TG2 and creates confluence observations only when
the break event has that current bar/timestamp
(`CausalFibonacciConfluenceIntegration.hpp:1234-1261`; `FibonacciConfluenceTracker::ObserveInnerBreak`,
`:284-345`). TG1's radius-two fractal at anchor `b` is not confirmed until
completed `b+2` (`CausalFractalTrendLineGeometry.hpp:321-345`). If it helps
create a candidate or A/B, its effect may first appear on the confirmation row
or a later break row; it must never backfill rows `b` or `b+1`.

`Tests/TG4CausalBoundaryClosureTests.cpp` fixes this key at replay time: each
new observation's `innerBreakBar` equals the supplied completed-candle index,
and its timestamp equals that input candle's timestamp. It also asserts
`confluent <= eligible <= inner_break_any`.

## 5. Market-data parity

The common pieces are real: both paths call the same database function with
`symbol, 15, 'minute'`, select OHLC/volume in `dt` order, and parse `dt` as
New-York civil time. TG4 additionally bounds the result after converting each
UTC instant back to New York (`TG4HistoricalMarketDataRepository.cpp:39-51`).
For all intersecting rows, the read-only probe compares row ordinal, `dt`,
open, close, high, low, and volume.

They are nevertheless not parity-equivalent:

| Path | Bound rule | Result for NY 2025-03-07 00:00 through 2025-03-11 00:00 |
| --- | --- | --- |
| Normal `LoadCandlesticks` | direct bounds to `candlestick`; MV uses `dt BETWEEN fromdt AND todt` | 197 rows, includes `2025-03-11 00:00:00` |
| TG4 historical | converted call plus `instant >= start AND instant < end` | 196 rows, ends `2025-03-10 23:45:00` |

The probe is [`Tests/TG4MarketDataParityProbe.sh`](../Tests/TG4MarketDataParityProbe.sh).
It runs under `default_transaction_read_only=on`, reports the difference, and
fails if the expected one-endpoint relationship changes. No conversion or
source change was made here. A future integration must choose one canonical
range API (absolute start/end plus a documented half-open policy is the
recommended shape), then have both normal and TG reads consume that one API.

## 6. Timezone/DST findings

The timestamp parser is host-timezone independent and covers standard/DST
offsets; its existing test proves 08:30 New York is 13:30 UTC in January and
12:30 UTC in July while rejecting both DST-invalid civil times
(`Tests/HistoricalFxTimestampTests.cpp:34-65`). The live parity probe crossed
the 2025 spring DST boundary and found the same OHLC/order for every common
row. It also exposed why a plain date is insufficient as an integration
identity: TG4 ranges are UTC instants (`ParseUtcDateOrTimestamp` in
`TG4HistoricalEmpiricalEvaluation.cpp:1223+`), whereas normal model input
passes unzoned strings directly to a database function taking `timestamp
without time zone` (`MarketDataCore.cpp:30-39`). The right conversion must be
made once at the future authoritative reader boundary, not independently by an
adapter.

## 7. Production TG configuration inventory

The following is the required production TG1--TG3 configuration inventory;
it is derived from `EvaluationConfiguration` and its parser, not from outcome
results. “Semantic” means a changed value can change a pulse. “Operational
invariant” is still immutable for reproducibility but cannot alter a newly
emitted pulse when semantic state is held fixed. “Research/output-only” must
not be in the production adapter configuration.

| Group | Values | Classification |
| --- | --- | --- |
| Input identity | canonical reader version; symbol; timeframe/period/unit; bar-start UTC semantics; interval/gap rejection policy | Semantic |
| TG1A | strict five completed candles/radius 2; intervening/touch tolerance; fractal anchor lookback; confirmed-fractal bound; candidate age; candidate bound; ATR period | Semantic (all retention/age bounds can alter later candidates) |
| TG1B | reference-bar scale; fixed LongTerm `[12,20]`, Outer `[25,40]`, Inner `[45,85]` inclusive bands; creation-time ATR/slope snapshot rule | Semantic |
| TG2 event creation/pairing | close-vs-wick break policy; rearm policy; break tolerance; nearest coexisting same-direction Outer beyond candle; Outer tolerance; deterministic identity tie-break | Semantic |
| TG3 | most-recent prior opposite confirmed fractal A/B; ratio set; absolute-tolerance confluence; directional policy; canonical pip convention and symbol map; TG3 confirmed-fractal bound; A/B age; active-A/B bound | Semantic |
| TG2/TG3 outcome tracking | retest contact/tolerance/horizon; Outer-target tolerance/horizon; active/retained break observations; active/retained confluence observations | Operational invariant for this pulse family: affects outcomes/retention, not break classification; keep immutable and test it |
| TG4 reporting | material-gap report threshold, minimum human report `N`, pending TG4 record bound, partitions/range score policy, artifact provenance/name | Research/output-only |

The current frozen values are evidence for constructing the eventual production
configuration, not a runtime dependency. The production factory must name its
own version and encode the chosen values in source-owned data.

## 8. Immutable configuration identity/persistence contract

Before the adapter is written, add a source-owned, immutable
`tg1-tg3-causal-pulse-v1` configuration factory—not a file loader—and a
canonical serialization with: schema/version; canonical market-reader/range
contract version; every semantic field above; every operational-invariant
bound; complete canonical-pip map; and the TG implementation behavior version.
Serialize enums by stable names, floating values with `max_digits10`, sorted
ratios/maps, and hash the resulting bytes (a cryptographic content hash is
preferred). The tuple `(configuration_schema, configuration_name,
configuration_hash)` is the production identity. Reject unknown hashes and
mismatched config bytes; never resolve it by a mutable “current TG config.”

There is useful prior art but it is research-scoped: `ConfigurationFingerprint`
already canonicalizes all TG4 fields and effective pip tolerances
(`TG4HistoricalEmpiricalEvaluation.cpp:1128-1171`). Do not reuse its
`EvaluationConfiguration` loader or read `Scripts/tg4_analysis_config.frozen_v1.conf`
from production. A production identity must exclude research-only reporting
fields and include the canonical reader/range version that the research
fingerprint does not carry.

When a future layout is actually introduced, persist the immutable production
config identity and canonical payload/hash with the experiment materialization,
model and every checkpoint; verify equality at training, resume and inference.
That is additive to—not a substitute for—the existing width/layout identity:
`experiment.model_input_width`,
`experiment.model_input_semantic_layout_version`, `model_meta`, and
`model_input_semantics_meta` (`PgModelIO.hpp:359-407`). The current semantic
registry is append-only (`ModelInputExpansion.hpp:35-59`). No persistence or
schema work is authorized in this closure increment.

## 9. Capacity/eviction analysis

Not all capacities are alike.

- TG1's fractal/candidate lookback, age, and maximum-candidate bounds prune
  live inputs used by later line creation/break detection
  (`CausalFractalTrendLineGeometry.hpp:348-380,533-580`). They can change
  `inner_break_any`.
- TG3's stored fractal, A/B age, and active-A/B bounds prune selections used at
  a later break (`CausalFibonacciConfluenceIntegration.hpp:719-760`). They can
  change `structurally_eligible` and `confluent`, and may also affect which
  event is eligible. They are semantic.
- TG2 break-observation and TG3 confluence-observation retention censors and
  archives outcome records (`CausalTrendLineBreakRetestBehavior.hpp:834-860`;
  `CausalFibonacciConfluenceIntegration.hpp:862-915`). It does not remove
  live candidates or A/B structures. It cannot rewrite a pulse captured from
  `newConfluenceObservations` on the current bar.

The new closure test exercises the latter with capacity one: an emitted row is
captured, the detailed outcome observation is evicted by the next event, and
the captured pulse remains `[1,0,0]`. Capacity changes must never be hidden:
any bound in the first two bullets belongs in the semantic config identity.

## 10. Deterministic streaming invariants

For identical ordered valid completed bars and identical immutable
configuration identity, the adapter must produce identical per-row pulse bits.
TG1 rejects non-increasing timestamps; TG2/TG3 enforce consecutive indexes and
strictly increasing timestamps (`CausalFractalTrendLineGeometry.hpp:147-181`;
`CausalTrendLineBreakRetestBehavior.hpp:204-229`; `CausalFibonacciConfluenceIntegration.hpp:213-224`). Candidate, pairing, fractal, A/B, and observation
orders have explicit deterministic sorting/tie-breaks.

The streaming algorithm must consume `TG3::Update.newConfluenceObservations`
immediately, look up their immutable classification, OR the three bits for
that same bar, then discard raw identity from the row output. Do not derive
pulses from `Observations()` later, terminal TG4 CSV, outcome fields, or
finalization. Future tails only synchronize outcomes (`CausalFibonacciConfluenceIntegration.hpp:347-379,1264-1269`), so they cannot revise emitted rows.

## 11. Tests/results

Focused tests run successfully:

- `Tests/TG4CausalBoundaryClosureTests.sh` — replay determinism; event-to-row
  timestamp/index mapping; future-tail prefix invariance; same-bar OR
  commutativity; outcome-record eviction cannot rewrite an emitted pulse.
- `Tests/TG4MarketDataParityProbe.sh` — real read-only New-York/DST boundary
  comparison; reports the one right-endpoint mismatch above.
- `Tests/TG1ACausalFractalTrendLineGeometryTests.sh` — confirmation boundary
  and bounded-state regression.
- `Tests/TG2TrendLineBreakRetestBehaviorTests.sh` and
  `Tests/TG3FibonacciConfluenceIntegrationTests.sh` — completed-bar break,
  deterministic pairing/confluence, prefix and capacity regression.
- `Tests/TG4HistoricalEmpiricalEvaluationTests.sh` — frozen research code's
  deterministic replay/bounded-stream regression only; it did not run an
  evaluation or produce artifacts.

No LSTM executable, worker, scheduler command, experiment, or historical
evaluation was started. A scheduler was observed active before tests; the
tests compile/run isolated C++ fixtures and read-only SQL only.

## 12. Pre-integration closure

The former blockers are closed without changing legacy model behavior:

1. `Headers/CanonicalMarketDataRange.hpp` defines the absolute UTC
   `[start,end)` contract and its shared canonical SQL CTE. The new normal
   `MarketData::LoadCanonicalHalfOpenCandlesticks` API
   (`Sources/MarketDataCore/MarketDataCore.cpp`) and the TG4 read-only
   repository use it. Legacy `LoadCandlesticks` remains inclusive-right for
   compatibility.
2. `Headers/ProductionTG1TG3PulseConfiguration.hpp` provides the source-owned
   `tg4a-derived-source-utl-up-ab-only-v1` immutable factory, deterministic
   semantic payload, and `fnv1a64:` identity. It has no runtime artifact
   dependency.

The detailed contract, field inventory, tests, persistence requirement and
readiness decision are recorded in
`TG4_PREINTEGRATION_CAUSAL_BOUNDARY_CLOSURE.md`. These closures are not a
reason to modify frozen TG4 artifacts or reinterpret 2025 results.

## 13. Explicitly deferred non-blocking questions

Continuous aggregation for future continuous TG fields remains unspecified.
Persistence/decay for future persistent TG fields remains unspecified. Neither
has an implementation, default, or performance-derived convention here.

## 14. Readiness decision for an isolated TG1→TG3 adapter

**Ready for a separately approved isolated TG1→TG3 streaming adapter, and not
ready for Tensor/model integration.** The canonical range and production
configuration prerequisites are now implemented and tested. Any future adapter
must remain disconnected from Tensor while width is 77/layout is 7.

## 15. Recommended next sequence

1. Specify and test one canonical absolute half-open market-data range API;
   make normal and TG consumers prove byte-equivalent bars over DST and gaps.
2. Add the production-owned TG1--TG3 configuration factory, canonical payload,
   hash, and non-research tests—without any Tensor channel.
3. Implement an isolated streaming adapter consuming completed canonical bars,
   emitting only the three same-bar OR pulses, with no TG4 artifact dependency.
4. In a separately approved increment, decide/add persistence for the config
   identity, then only later consider the append-only Tensor/layout migration.
