# TG4 causal point-in-time LSTM feature contract

Status: design/audit only. This document specifies the causal contract for a
future TG-derived feature implementation. It does not authorize a feature,
schema, model-width, semantic-layout, worker, scheduler, or experiment change.

## 1. Executive summary

Repository truth is that the active model-input contract is **77 columns** and
semantic-layout **7**, not a TG-dependent layout. Its physical `Tensor` prefix
is 73 columns and its last four columns are model-materialization-time return
features. The current path is:

```text
canonical market/economic reads -> ModelInputPreparation::Prepare -> Tensor::Add
-> Tensor row (73) -> CopyTensorFeaturesForModelInput + returns (4) -> LSTM
```

The TG pipeline is causal when driven one completed bar at a time. A five-bar
fractal centered at `b` is knowable only after completed `b+2`; a trend line
and TG1B class become knowable at second-anchor confirmation. On a completed
Inner-break bar, TG2 freezes the break and selects the coexisting paired Outer.
TG3 then chooses only an already available A/B and freezes confluence on that
same completed break bar. Retest and Outer outcomes mutate on later bars and
may be censored at finalization; they cannot be LSTM features.

The recommended first ablation is a small **three-channel, break-bar pulse
family**:

1. `tg4_inner_break_any`;
2. `tg4_source_tg3_structurally_eligible`;
3. `tg4_source_tg3_confluent`.

It exposes the frozen TG4A event definition without geometry, outcomes, or an
invented exposure lifetime. Compute it with a streaming completed-bar TG1--TG3
engine, never by joining terminal TG4 `observations.csv` rows.

The 2025 confirmation supports considering a future controlled ablation only.
It must not select the channels, persistence, scaling, aggregation, ratio,
tolerance, or any other representation here.

## 2. Repository paths inspected

| Concern | Primary authority |
| --- | --- |
| Tensor layout/construction | `Headers/FeatureLayout.hpp`; `LSTM/Tensor.cpp` (`Tensor::Add`) |
| Model projection/return suffix | `Headers/ModelInputContract.hpp`; `Headers/ReturnFeatureHistory.hpp`; `LSTM/LSTM.cpp` |
| Semantic naming/identity | `Headers/ModelInputFeatureSemantics.hpp`; `Headers/ModelInputExpansion.hpp` |
| Model-input warmup/preparation | `Sources/ModelInputPreparation/ModelInputPreparation.cpp` |
| TG1A/TG1B/TG2/TG3 | `Headers/CausalFractalTrendLineGeometry.hpp`; `Headers/CausalFractalTrendLineAngleClassification.hpp`; `Headers/CausalTrendLineBreakRetestBehavior.hpp`; `Headers/CausalFibonacciConfluenceIntegration.hpp` |
| TG4 records/read-only query | `Headers/TG4HistoricalEmpiricalEvaluation.hpp`; `Sources/TG4HistoricalEmpiricalEvaluation.cpp`; `Sources/TG4HistoricalMarketDataRepository.cpp` |
| Frozen study | `Scripts/tg4_analysis_config.frozen_v1.conf`; `docs/phases/target-generation/PhaseTG4/TG4-Historical-Empirical-Evaluation-Schema.md` |
| Persistence/worker admission | `Headers/PgModelIO.hpp`; `Sources/SchedulerCore/SchedulerSemanticAdmission.hpp`; `Sources/SchedulerCore/SemanticWorkerRegistry.{hpp,cpp}`; `Scripts/PublishSemanticWorker.py` |
| Existing test infrastructure | `Tests/TG1ACausalFractalTrendLineGeometryTests.cpp`; `Tests/TG1BTrendLineAngleClassificationTests.cpp`; `Tests/TG2TrendLineBreakRetestBehaviorTests.cpp`; `Tests/TG3FibonacciConfluenceIntegrationTests.cpp`; `Tests/TG4HistoricalEmpiricalEvaluationTests.cpp`; `Tests/LSTMFeatureVectorParityTests.cpp`; `Tests/LSTMInputWidthExpansionTests.cpp`; `Tests/SchedulerSemanticAdmissionIntegrationTests.sh` |

## 3. Current LSTM feature-layout baseline

### Identity and construction

`EA::kCurrentModelInputWidth` is 77 (`Headers/ModelInputContract.hpp:55-60`),
`feature_size` is 73 (`Headers/FeatureLayout.hpp:137-151`), and
`EA::kModelInputSemanticLayoutVersion` is 7
(`Headers/ModelInputExpansion.hpp:22-24`). `Tests/LSTMFeatureVectorParityTests.cpp:86-95`
asserts the same baseline.

`ModelInputPreparation::Prepare` opens read-only market/economic transactions,
loads an owned range, and invokes `Tensor::Add` for every row
(`Sources/ModelInputPreparation/ModelInputPreparation.cpp:16-85`). Training
and inference both resolve the persisted width, project exactly its Tensor
prefix, and append the four return features:

- training: `LSTM/LSTM.cpp:3350-3419`;
- return inference: `LSTM/LSTM.cpp:7397-7447`;
- relative-move inference: `LSTM/LSTM.cpp:7491-7540`.

Projection is an exact prefix copy, never a copy-and-zero compatibility shim
(`Headers/ModelInputContract.hpp:198-225`). A TG increment must retain that
rule.

### Exact ordered current layout

Columns 0--72 are Tensor features; 73--76 are return suffixes. The names are
authoritative in `Headers/ModelInputFeatureSemantics.hpp:29-75` and
`Headers/ModelInputExpansion.hpp:196-248`.

| Columns | Features |
| --- | --- |
| 0--3 | log open/close/high/low from previous close, scaled |
| 4--7 | candle body/range, rolling log-return volatility, rolling cumulative log return |
| 8--11 | legacy candle-cycle sin/cos; day-of-week sin/cos |
| 12--18 | upper/lower wick fraction; close-to-EMA 8/21/50 and EMA 8--21/21--50 by guarded range |
| 19--23 | close-to-EMA 8/21/50 and EMA 8--21/21--50 by ATR14 |
| 24--31 | EMA 8/21/50 log slopes by guarded range; EMA slopes by ATR14; body strength; range expansion |
| 32--33 | `donchian_up`, `donchian_down` |
| 34--35 | `session_phase_sin`, `session_phase_cos` |
| 36--48 | relative tick volume; return surprise; volatility regime; directional range; close location; directional efficiency; sign persistence; direction imbalance; adverse excursion; multi-bar range pressure; rolling range expansion; historical-level proximity; return autocorrelation |
| 49--58 | five economic-event flags and five recency decays |
| 59--62 | relevant-event consensus flag, low, high, range flag |
| 63--66 | released-event surprise flag/value/absolute/direction |
| 67--70 | authoritative-initial surprise flag/value/absolute/direction |
| 71--72 | causal first-release surprise availability/value |
| 73--76 | scaled log returns at lookbacks 1, 4, 8, 16 |

Exact economic names/order are in `Sources/EconomicEventFeatureLayout.hpp:10-78`.
Direct Tensor assignments are in `LSTM/Tensor.cpp:183-441`.

### Availability, scaling, warmup, and point-in-time safeguards

Ordinary unavailable history is finite, normally zero. The first Tensor row is
explicitly zero-filled before causally available values are assigned
(`LSTM/Tensor.cpp:143-180`); return suffixes are zero until their lookback is
available (`Tests/LSTMFeatureVectorParityTests.cpp:141-160`). Categorical
economic fields are recognized by names, not a global null mask
(`Headers/ModelInputFeatureSemantics.hpp:77-84`).

There is no global fitted feature normalizer. Price/return quantities use
`kFeatureScale`; EMA geometry uses guarded current range or causal ATR14
(`LSTM/Tensor.cpp:183-187, 259-396`). Inputs are finite-checked and inference
clamps materialized values to `[-10,10]` (`LSTM/LSTM.cpp:7433-7447`).

Rows are completed decision bars. Most appended structural engines are called
before retaining the current bar; documented range features include current
completed data (`LSTM/Tensor.cpp:102-141`). Donchian explicitly reads only
prior highs/lows (`LSTM/Tensor.cpp:202-215`).

Warmup is persisted request policy: full-history starts at
`kTensorFeatureHistoryQueryStart`; otherwise at output start. The prefix is
retained and `logicalOutputStartIndex` returned
(`Sources/ModelInputPreparation/ModelInputPreparation.cpp:19-31,76-85`). TG
state must receive the same physical warmup and not reset at score start.

### Persistence/checkpoint/worker baseline

Models store `model_meta` (including width) and
`model_input_semantics_meta` `[schemaVersion,layoutVersion]`
(`Headers/PgModelIO.hpp:359-407`). Save verifies experiment width/layout and
registered compatibility (`Headers/PgModelIO.hpp:169-215`). The semantic
registry is append-only; layout 7 is a same-width correction branching from 5,
not 6 (`Headers/ModelInputExpansion.hpp:35-59`).

Scheduler selection uses the persisted layout/width to select an immutable
historical worker or a compatible current worker
(`Sources/SchedulerCore/SemanticWorkerRegistry.cpp:878-927`), backed by
registered-chain admission (`SchedulerSemanticAdmission.hpp:50-73`). Do not
replace this with a TG-specific legacy worker mechanism.

## 4. TG1 -> TG4 causal dataflow

All bars here are valid, timestamp-ordered **completed** bars. TG1A rejects
duplicate/out-of-order bars (`CausalFractalTrendLineGeometry.hpp:147-181`);
TG2/TG3 independently enforce consecutive bar indexes and increasing
timestamps (`CausalTrendLineBreakRetestBehavior.hpp:204-230`,
`CausalFibonacciConfluenceIntegration.hpp:213-224`).

```text
completed candle t
  -> TG1A updates ATR and sees the 5-bar window
  -> fractals centered at t-2 become confirmed
  -> TG1A creates candidates; TG1B freezes class/angle at creation
  -> TG2 detects an armed candidate's break and pairs its Outer at t
  -> TG3 ingests t's confirmed fractals/A-Bs, then classifies t's new breaks
  -> TG4 captures immutable causal facts; later bars update outcomes only.
```

### TG1A: fractals and trend lines

TG1A uses radius 2 and a five-bar window (`CausalFractalTrendLineGeometry.hpp:244-246`).
At completed `t`, it evaluates a center at `t-2`; a strict high/low produces a
`ConfirmedFractal` with `anchorBar=t-2` and `confirmationBar=t`
(`:321-345`). Its existence and price therefore may not enter a feature at
the anchor or first following row.

When a same-kind fractal becomes confirmed, TG1A pairs it with qualifying prior
confirmed fractals, validates intervening history, and creates a line at the
second-anchor confirmation (`:348-409`). Candidate geometry updates from the
current completed candle/current Wilder ATR; lifecycle is causal/bounded
(`:499-581`). Live distance, touch count, age and ATR are mutable; a creation
field must use its creation snapshot.

### TG1B: frozen classification

TG1B computes its angle from **creation-time** ATR-normalized slope and fixed
LongTerm (12--20), Outer (25--40), Inner (45--85) bands
(`CausalFractalTrendLineAngleClassification.hpp:53-89`).
`ClassifyAtCreation` copies class, angle, creation ATR, normalized slope and
reference scale once (`:241-267`); later lifecycle synchronization removes
dead candidates but does not reclassify them (`:269-289`).

### TG2: break, pairing, outcomes

TG2 resolves older observations first and then checks transitions of live
candidates (`CausalTrendLineBreakRetestBehavior.hpp:204-230`). A break requires
an armed-valid-side to broken transition. A candidate first discovered already
broken is deliberately not a break (`:515-542`). Frozen TG4A uses
completed-close-beyond-line (`Scripts/tg4_analysis_config.frozen_v1.conf:33-44`);
implementation applies the configured component to current projection
(`CausalTrendLineBreakRetestBehavior.hpp:475-494`).

`EmitBreak` freezes OHLC, projection, penetration, directional distance, class,
inner slope and anchor. For an Inner break it selects the nearest qualifying
coexisting Outer; otherwise Outer target is structurally ineligible
(`:568-619`). Pairing requires same direction, Outer classification, creation
at/before break, and an Outer beyond the break candle; ties are deterministic
(`:621-665`).

Only post-break rows update outcome fields. Retest and Outer check
`break+1..deadline`; retest-then-Outer rejects unknown same-bar ordering
(`:713-800`). Finalization/capacity pressure censors pending outcomes
(`:810-860`).

### TG3: A/B and confluence

TG3 accepts a fractal only at exactly its confirmation bar/timestamp
(`CausalFibonacciConfluenceIntegration.hpp:226-281`). It chooses A as the most
recent prior confirmed opposite-kind directional pivot; B is the new confirmed
pivot. A/B is thus available at **B confirmation**, not B anchor. The A
selection excludes candidates confirmed after B (`:685-710`). A break can
select only a structure with `availabilityBar <= observationBar` (`:751-791`).

The integrated order is critical: advance TG3, ingest this bar's confirmed
fractals (making same-bar A/B available), synchronize old outcomes, then
observe new breaks (`CausalFibonacciConfluenceIntegration.hpp:1234-1261`). A
later fractal/A-B cannot be backfilled onto an earlier break.

TG3 requires paired Outer, source-supported direction and causal A/B. It
compares the Outer projection **at break** against configured levels using
inclusive absolute tolerance (`:793-844`). TG4A freezes
`source_utl_up_ab_only`, ratio `0.6180339887498949`, and one canonical FX pip
(`Scripts/tg4_analysis_config.frozen_v1.conf:46-59`). Confluence/A-B/levels/
distances are immutable; only copied TG2 outcome fields synchronize later
(`CausalFibonacciConfluenceIntegration.hpp:135-167,347-379`).

### TG4: capture versus terminal artifact

TG4 captures a record only for each new, scored TG3 Inner break and freezes
Inner/Outer classified candidate snapshots (`Sources/TG4HistoricalEmpiricalEvaluation.cpp:801-862`).
It holds records only to synchronize outcomes and emits them once terminal
(`:755-799,865-890`). The artifact schema independently says geometry/class/
pairing/A-B/levels/confluence freeze on break; only three outcome groups use
future bars (`docs/phases/target-generation/PhaseTG4/TG4-Historical-Empirical-Evaluation-Schema.md:103-107`).

## 5. Field-by-field causal classification

Definitions: **CAUSAL_AT_BREAK** is known after the completed event bar and
before its successor; **CAUSAL_BEFORE_BREAK** already existed and may be
snapshotted; **STRUCTURAL_METADATA** is identity/configuration/diagnostic;
**FUTURE_OUTCOME_ONLY** needs a later bar; **CENSORED/POST_OUTCOME** is created
or changed by terminalization/capacity; **NOT_SUITABLE_AS_MODEL_INPUT** should
not be a first feature; **REQUIRES_FURTHER_PROOF** is causal in research code
but needs model-input train/inference parity. Every future channel needs that
proof.

| Field(s) | Classification | Earliest availability/evidence | Disposition |
| --- | --- | --- | --- |
| direction; Inner candidate identity | CAUSAL_AT_BREAK; identity is STRUCTURAL_METADATA | `EmitBreak` copies both on transition (`CausalTrendLineBreakRetestBehavior.hpp:568-600`). | Direction excluded initially; identity never input. |
| frozen Inner class/angle | CAUSAL_BEFORE_BREAK | TG1B snapshots at candidate creation (`CausalFractalTrendLineAngleClassification.hpp:241-267`). | Optional later geometry only. |
| Inner creation ATR/normalized slope | CAUSAL_BEFORE_BREAK | Same creation snapshot. | Optional later; never recompute with later ATR. |
| frozen Outer class/angle | CAUSAL_AT_BREAK | Values existed at creation, but selected Outer is known only on pairing (`CausalTrendLineBreakRetestBehavior.hpp:621-665`). | Optional later only. |
| Outer creation ATR/normalized slope | CAUSAL_AT_BREAK | Available if paired Outer is selected; itself frozen at Outer creation. | Optional later only. |
| Inner anchors; creation bar/time | CAUSAL_BEFORE_BREAK; STRUCTURAL_METADATA | Stored at line creation from confirmed anchors (`CausalFractalTrendLineGeometry.hpp:390-407`). | Raw ID/time/price not suitable. |
| break bar/time | CAUSAL_AT_BREAK; STRUCTURAL_METADATA | Set in `EmitBreak`. | Index/time not suitable. |
| projected break line price | CAUSAL_AT_BREAK | Current-bar projection at `EmitBreak` (`CausalTrendLineBreakRetestBehavior.hpp:575-595`). | Raw price not suitable. |
| penetration/directional distance | CAUSAL_AT_BREAK | Derived from completed component/projection (`:575-595`). | Optional later with causal scale/clipping. |
| break OHLC/observed component | CAUSAL_AT_BREAK | Copied from completed candle (`:581-595`). | Redundant with current candle columns. |
| Outer pairing eligibility; paired Outer | CAUSAL_AT_BREAK | Deterministic same-bar live scan (`:621-665`). | Eligibility used in first family; identity not. |
| Outer projection at break | CAUSAL_AT_BREAK | Captured by successful pairing (`:661-664`). | Used inside TG3; raw value excluded. |
| TG3 A/B policy/direction | STRUCTURAL_METADATA | Frozen configuration; required direction resolves from line (`CausalFibonacciConfluenceIntegration.hpp:766-775`). | Do not encode constants. |
| A/B bars/prices | CAUSAL_AT_BREAK; STRUCTURAL_METADATA | Selected from already available A/B (`:777-820`). | Raw geometry excluded initially. |
| A/B confirmations/availability | CAUSAL_AT_BREAK | B confirmation creates availability; selection requires it by break (`:71-96,226-281,777-791`). | Audit/control fact, not first signal. |
| ratio set/directional/confluence policy/tolerance | STRUCTURAL_METADATA | Frozen configuration (`Scripts/tg4_analysis_config.frozen_v1.conf:46-59`). | No fixed-constant column. |
| confluence state | CAUSAL_AT_BREAK | Classifies causal paired-Outer/A-B/levels (`CausalFibonacciConfluenceIntegration.hpp:793-844`). | First-family conditional bit. |
| ineligible reason | CAUSAL_AT_BREAK | No pair, unsupported direction or no A/B (`:793-813`). | Collapse to eligibility initially. |
| matched ratio(s), nearest ratio, raw distance | CAUSAL_AT_BREAK | Computed over levels (`:816-843`). | Not first; singleton ratio and distance need a scale contract. |
| TG3 observation ATR / normalized distance | CAUSAL_AT_BREAK | ATR comes from live paired Outer; distance divided only when available (`:1321-1339,836-839`). | Optional later; needs unavailable rule. |
| level diagnostics/zones | CAUSAL_AT_BREAK; NOT_SUITABLE_AS_MODEL_INPUT | Variable diagnostic expansion (`:822-835`). | Never expose variable vector. |
| retest state/resolution/latency | FUTURE_OUTCOME_ONLY | Later bars only (`CausalTrendLineBreakRetestBehavior.hpp:713-755`). | Never input. |
| Outer-target state/resolution/latency | FUTURE_OUTCOME_ONLY | Later bars/deadline (`:757-775`). | Never input. |
| retest-then-Outer state/resolution/latency | FUTURE_OUTCOME_ONLY | Requires post-retest later ordering (`:777-799`). | Never input. |
| censor reasons/final record state | CENSORED/POST_OUTCOME; NOT_SUITABLE_AS_MODEL_INPUT | Capacity/finalization and terminal emission (`:810-860`; `TG4HistoricalEmpiricalEvaluation.cpp:865-890`). | Never input/reconstruct from final row. |
| schema/event identity/symbol/timeframe/partition/provenance | STRUCTURAL_METADATA; NOT_SUITABLE_AS_MODEL_INPUT | Artifact grouping. | Never input. |

## 6. Leakage-risk audit

| Risk | Repository path | Required invariant/test |
| --- | --- | --- |
| Current/in-progress candle | Tensor rows and TG entry points use completed bars (`LSTM/Tensor.cpp:102-141`). | Drive TG only after the exact market row accepted by `Tensor::Add`; never revise a materialized row. |
| Fractal backdating | Center `b` emits only at `b+2` (`CausalFractalTrendLineGeometry.hpp:321-345`). | Bars `b,b+1` cannot change when `b` later confirms. |
| A/B backdating | Exact confirmation-time ingestion/availability (`CausalFibonacciConfluenceIntegration.hpp:226-281`). | A/B whose B confirms after a break cannot affect it. |
| Later ATR reclassification | Creation snapshot is immutable (`CausalFractalTrendLineAngleClassification.hpp:241-267`). | Candidate class/angle remains byte-identical under later tail. |
| Later Outer pairing | Outer must have `creationBar <= break` (`CausalTrendLineBreakRetestBehavior.hpp:621-665`). | Future Outer cannot change historical eligibility/confluence. |
| Post-break state mutation | TG3 later synchronizes outcomes only (`CausalFibonacciConfluenceIntegration.hpp:347-379`). | Adapter reads only `newConfluenceObservations` on the break row, never mutable/terminal observations. |
| Retest/target outcomes | Resolution begins after break (`CausalTrendLineBreakRetestBehavior.hpp:713-800`). | No outcome state, latency, contact price, final state or censor reason in Tensor. |
| Finalization/censoring | Finalization/capacity censors (`:232-241,810-860`); TG4 output is terminal (`TG4HistoricalEmpiricalEvaluation.cpp:865-890`). | Prefix features identical with/without future tail/finalization. |
| Terminal CSV reconstruction | CSV contains later outcome fields (`TG4HistoricalEmpiricalEvaluation.cpp:1295-1431`). | Training feature materialization executes streaming TG state only; no artifact joins. |
| Future normalization | Tensor uses recursive/current calculation, not whole-sample fit (`LSTM/Tensor.cpp:217-251,353-396`). | Ban whole-dataset z-score/quantile fitting; use constants or prefix-only state. |
| Query/window leakage | TG4 streams ordered range in read-only transaction (`TG4HistoricalMarketDataRepository.cpp:101-123`). | Consume warmup through output end only; assert monotonicity; no SQL lead/future window. |
| Warmup reset | Preparation retains prefix/logical start (`ModelInputPreparation.cpp:19-31,76-85`). | Full-prefix and prefix-then-score replay must agree. |
| Same-bar order dependence | Candidate/A-B sorting and pairing tie break (`CausalFractalTrendLineGeometry.hpp:592-604`; `CausalFibonacciConfluenceIntegration.hpp:732-739`; `CausalTrendLineBreakRetestBehavior.hpp:651-659`). | Aggregation is commutative/deterministic under equivalent discovery order. |

## 7. Minimal first-ablation feature contract

### A. Recommended representation

Append exactly three Tensor columns in a future increment. They are pulses on
the completed bar where TG3 creates an Inner-break observation; they never
persist. This choice follows causal clarity and frozen TG4A semantics, not
2025 results.

| Channel | Exact completed-row definition | Domain/normalization | No event, ineligibility, unavailable | Lifetime/validity |
| --- | --- | --- | --- | --- |
| `tg4_inner_break_any` | 1 iff TG2 emits an Inner-class break at `t` and TG3 creates its observation; else 0. | Binary; no scale. | no event=0; all ineligible Inner breaks=1; a valid completed bar has no NA. Invalid stream input fails, never imputes. | One-row pulse; no extra validity bit. |
| `tg4_source_tg3_structurally_eligible` | 1 iff any same-bar Inner observation has state Confluence or NoConfluence; else 0. This means paired Outer + source-supported direction + causal A/B passed. | Binary; no scale. | no event=0; structurally ineligible=0; eligible states=1. Channel 1 distinguishes absent event from ineligible event. | One-row pulse; channel 1 is event guard. |
| `tg4_source_tg3_confluent` | 1 iff any same-bar structurally eligible observation is Confluence; else 0. | Binary; no scale; invariant `confluent <= eligible <= inner_break_any`. | no event/ineligible/eligible non-confluent=0; no NA. | One-row pulse; channel 2 is eligibility guard. |

| Row state | `inner_break_any` | `structurally_eligible` | `confluent` |
| --- | ---:| ---:| ---:|
| no Inner break | 0 | 0 | 0 |
| structurally ineligible Inner break | 1 | 0 | 0 |
| eligible non-confluent Inner break | 1 | 1 | 0 |
| eligible confluent Inner break | 1 | 1 | 1 |

The family is distinct from existing price/range/return/calendar inputs because
the 77-column baseline contains no confirmed fractal trend line, Inner-to-Outer
pairing or causally available A/B Fibonacci relation. It deliberately omits
raw price/anchors/angle/distance/outcomes.

No direction channel belongs to this initial family. With frozen
`source_utl_up_ab_only`, every structurally eligible TG3 event is UTL; DTL
Inner breaks become `DirectionUnsupportedByStudy`
(`CausalFibonacciConfluenceIntegration.hpp:766-775,793-813`). Channel 1 keeps
that a break occurred without falsely calling it non-confluent.

The future adapter must own an exact frozen-config TG1--TG3 streaming state and
consume the per-bar `TG3::Update` immediately. It must not use TG4 terminal
records; TG4 is a read-only research/artifact path.

### B. Optional later geometry, excluded from first experiment

Do not add initially: signed UTL/DTL break direction; frozen Inner/Outer class;
angle or normalized slope; causally normalized penetration; candidate age;
Outer separation in causal ATR units; A/B age/range; or capped nearest-level
distance. Each needs its own availability, aggregation, scale and clipping
contract. Raw anchors/times/prices/IDs, ratio/policy constants, variable level
diagnostics and all outcome/censor fields remain excluded.

## 8. Event-to-bar semantics

- A confluence value exists only on the **completed qualifying break bar**.
  TG3 creates it at that break (`CausalFibonacciConfluenceIntegration.hpp:284-345`).
- It does **not** persist. Existing TG code defines no feature exposure
  interval. Persisting through retest, target, expiry, N rows, next break, or
  a decay would invent semantics and can invite outcome leakage. The pulse is
  minimal, not an empirical-performance claim.
- No break emits `[0,0,0]`; a DTL or other structural-ineligibility event emits
  `[1,0,0]`; eligible non-confluence `[1,1,0]`; confluence `[1,1,1]`.
- Multiple same-bar Inner breaks aggregate by **bitwise OR** over all new TG3
  observations on that bar. This is commutative and deterministic; confluence
  is 1 if any eligible same-bar break is confluent. Retain no raw identity.
- A later event writes only its own later-row pulse; it never overwrites an old
  row. TG3 explicitly says later synchronization does not alter classification
  (`CausalFibonacciConfluenceIntegration.hpp:163-167`).

The repository does not define persistence nor aggregation for a future
continuous distance. Those are open design choices, not places to choose a
performance-driven convention.

## 9. Semantic-layout migration impact (future work only)

A future implementation of this three-column family must:

1. Append three Tensor columns after current column 72 in `FeatureLayout.hpp`,
   leaving columns 0--72 byte/semantic identical. The model width would be 80
   because the four return suffixes remain last
   (`Headers/ModelInputContract.hpp:16-60`). This document does not make it so.
2. Append three named semantic-registry entries and retain contiguous/unique
   registry assertions (`Headers/ModelInputExpansion.hpp:187-269`), allowing
   `ModelInputFeatureSemantics` to report them.
3. Register width 80, make it current, and append a semantic layout entry
   (expected layout 8; maximum width 80; predecessor 7). Never change layout
   7 or its predecessor chain.
4. Review width-expansion/provenance. Existing 77 columns must be preserved;
   zero initialization of newly expanded learned input weights is not proof
   that a 77/7 model became semantically TG-compatible.
5. Keep experiment/model/checkpoint identity coherent: experiment
   `model_input_width`/`model_input_semantic_layout_version`, `model_meta`,
   and `model_input_semantics_meta`; save/load checks are authoritative
   (`Headers/PgModelIO.hpp:169-215,359-407`). Review duplicate identity,
   resume, campaigns, CLI, inference/evaluation/reporting contexts carrying
   both fields.
6. Build/publish canonical role-aware train and inference workers using the
   existing content-addressed registry/manifests/runtime validation. The
   registry requires one current inference and one current training artifact
   for current layout (`SemanticWorkerRegistry.cpp:665-753`).
7. Preserve all historical worker entries. Layout-7/77 models keep selecting
   immutable historical workers. A new-layout experiment without a matching
   worker must be rejected, never silently run as layout 7
   (`SemanticWorkerRegistry.cpp:878-914`). Do not create an ad-hoc TG legacy
   mechanism.
8. Update model/checkpoint, expansion, admission, publication, feature parity,
   TG causality, migration, CLI and reporting tests together. Existing schema
   identity columns carry positive width/layout; inspect constraints before
   proposing any SQL migration.

## 10. Required test plan before training admission

Extend existing TG fixtures and add a small pure streaming-adapter test; do not
duplicate market-test infrastructure.

1. **Point-in-time causality:** extend `TG3FibonacciConfluenceIntegrationTests.cpp`;
   snapshot each output row after `AddCompletedBar`, prove prefix-only behavior,
   and assert `confluent <= eligible <= inner_break_any`.
2. **Future-tail mutation:** follow `LSTMFeatureVectorParityTests.cpp:169-177`:
   compute through T, mutate/delete bars after T, replay, and require
   byte-identical TG rows through T.
3. **Fractal confirmation boundary:** extend `TG1ACausalFractalTrendLineGeometryTests.cpp`:
   a pivot at b cannot affect b/b+1 and may first affect state at b+2.
4. **A/B confirmation boundary:** extend TG3 tests: B anchor before a break but
   confirmation after it is unavailable; same-bar B confirmation is available
   because integration ingests fractals before breaks.
5. **Break-bar availability:** extend TG2/TG3 tests: bits appear on completed
   break row, never one row later; a candidate discovered already broken emits
   no break (`CausalTrendLineBreakRetestBehavior.hpp:524-530`).
6. **Structural ineligibility:** cover no paired Outer, no causal A/B and DTL
   source-policy rejection; all are `[1,0,0]`, never a hidden missing value.
7. **No-event/missingness:** assert `[0,0,0]` before candidate warmup, on normal
   rows and after expiry; invalid/non-monotonic bars fail rather than impute.
8. **Multiple structures:** construct simultaneous ineligible/non-confluent/
   confluent breaks and assert bitwise OR independent of candidate ordering.
9. **Warmup boundary:** prepare full-prefix and prefix-then-score streams;
   compare logical output rows and prove no reset at `outputStart`.
10. **Train/inference parity:** extend `LSTMFeatureVectorParityTests.cpp` and
    `TensorPhase19CCausalAdapter` tests; matching global rows must be byte
    identical for training, direct inference and strategy evaluation.
11. **Checkpoint incompatibility:** extend `LSTMModelInputCompatibilityTests.cpp`
    and `LSTMInputWidthExpansionTests.cpp`: a 77/7 checkpoint cannot load as
    80/new layout; old semantics stay unchanged.
12. **Legacy-worker admission:** extend `SchedulerSemanticAdmissionIntegrationTests.sh`
    and `SemanticWorkerRegistryTests.cpp`: historical layout selects immutable
    worker; new layout is rejected until matching artifact exists; mismatch
    consumes no capacity.
13. **Deterministic replay:** extend `TG4HistoricalEmpiricalEvaluationTests.cpp`
    or adapter tests: identical ordered bars/configuration yield identical
    bits/event order, including expiry/capacity behavior without outcome input.

Run focused C++ tests first, then input/worker suites. Do not start
`LSTM_Release`, scheduler, training, inference or TG4 historical evaluation
without scheduler-safety checks and explicit authorization.

## 11. Open questions / ambiguities

1. **Resolved—row timestamp contract:** `Feature::time` and TG
   `Candle.timestamp` are New-York-civil **bar starts** converted to UTC
   `PriceTP` instants. An event for completed bar `T` maps to the Tensor row
   appended for `T`, not its successor or a fractal anchor. Evidence and test:
   `docs/phases/target-generation/PhaseTG4/TG4_CAUSAL_BOUNDARY_CLOSURE.md` sections 3--4;
   `Common/db_cursor.cpp:76-105`; `LSTM/Tensor.cpp:102-180,443-449`.
2. **Resolved pre-integration—market-data parity:**
   `Headers/CanonicalMarketDataRange.hpp` now specifies absolute UTC
   `[start,end)` canonical bars and provides the shared SQL CTE used by TG4 and
   exposed through `MarketData::LoadCanonicalHalfOpenCandlesticks`
   (`Sources/MarketDataCore/MarketDataCore.cpp:49-64`). Legacy model loading
   remains inclusive-right until an explicit future migration. See
   `TG4_PREINTEGRATION_CAUSAL_BOUNDARY_CLOSURE.md`.
3. **Resolved pre-integration—configuration ownership:**
   `Headers/ProductionTG1TG3PulseConfiguration.hpp` supplies the immutable,
   source-owned `tg4a-derived-source-utl-up-ab-only-v1` payload/hash without a
   runtime research-artifact dependency. Persistence remains explicitly
   deferred until an approved layout/model migration.
4. **Resolved—capacity effects:** TG1 retained fractals/candidates and TG3
   retained A/B state can alter later pulse bits and are semantic identity.
   Outcome-observation retention can censor outcomes but cannot rewrite a
   pulse already emitted. See boundary closure sections 9--10 and its focused
   replay/eviction test.
5. **Continuous aggregation:** only binary OR is specified. Min/mean/nearest
   choices for future continuous fields are absent from repository semantics.
6. **Persistence:** TG defines immutable break labels, not feature duration.
   Any decay/persistence requires new causal review and cannot use 2025 results.

## 12. Explicit non-goals

This document does not implement features; alter source, schema, migrations,
Xcode project, model width/layout, TG1--TG4 parameters, frozen artifacts,
database rows, scheduler/worker state or experiment queue; rerun TG4/
alternative confirmation analysis; infer causality or profitability; or select
representation from confirmation outcomes. TG4A remains descriptive evidence.

## 13. Recommended implementation sequence

1. Resolve the six open questions with input-data parity/configuration-ownership
   notes while preserving frozen TG4A artifacts.
2. Add a pure completed-bar TG1--TG3 adapter plus focused causality tests, but
   do not attach it to Tensor/model inputs.
3. Add the three pulse channels with warmup and train/inference parity tests;
   verify no terminal TG4 artifact path is invoked.
4. Apply the append-only width/layout/registry/persistence migration as one
   coherent change and pass compatibility/checkpoint tests.
5. Publish role-aware canonical workers through the existing semantic-worker
   mechanism and prove historic worker admission remains intact.
6. Only then prepare a separately approved, outcome-blind controlled LSTM
   ablation plan. Do not queue/run it as part of this design work.
