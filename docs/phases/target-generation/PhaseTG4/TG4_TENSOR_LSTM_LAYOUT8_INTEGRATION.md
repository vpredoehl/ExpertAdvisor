# TG4 Tensor/LSTM integration: semantic layout 8

## Contract

Layout 8 changes the current model-input identity from width 77/layout 7 to
width 80/layout 8. Layout 7 remains a valid historical identity with its
unchanged 73-column Tensor prefix and four-return suffix.

The physical Tensor columns are unchanged through column 72
(`causal_first_release_surprise`). Layout 8 adds exactly these completed-bar
TG4 pulse columns:

| Tensor / layout-8 model column | Name |
| ---: | --- |
| 73 | `tg4_inner_break_any` |
| 74 | `tg4_source_tg3_structurally_eligible` |
| 75 | `tg4_source_tg3_confluent` |

The four return features remain the final model-input columns, moved from
layout-7 positions 73–76 to layout-8 positions 76–79:
`lookback_log_return_{1,4,8,16}_scaled`.

## Causal path

`ModelInputPreparation` supplies canonical completed `Feature` bars in order
to `Tensor::Add`. Tensor owns one `TG4Pulse::ProductionStreamingAdapter` for
that same streaming lifecycle and calls `AddCompletedCanonicalBar` once for
each accepted bar. Tensor rejects a returned pulse whose `barStart` differs
from the `Feature` timestamp and rejects a hierarchy violation before it
materializes the row. The three returned bits are written directly as exact
`float` 0 or 1 values; no scaling, normalization, imputation, persistence, or
lookback is applied.

The pulse is same-bar: a completed bar beginning at `T` receives its pulse in
the Tensor row whose raw timestamp is `T`. Full-history warmup builds the
adapter state before the scored window, so training and inference consume the
same already-warmed Tensor row.

The frozen TG4A policy remains `source_utl_up_ab_only`. A DTL inner break is
recognized structurally but ineligible for TG4A and is therefore represented
as `[1,0,0]`, never as an eligible non-confluent event. The legal states are
`[0,0,0]`, `[1,0,0]`, `[1,1,0]`, and `[1,1,1]`, with
`confluent <= eligible <= inner_break_any`.

## Identity and expansion

`kModelInputSemanticLayoutRegistry` retains layout 7 at width 77 and adds
layout 8 at width 80 with layout 7 as its append-only semantic predecessor.
Compatibility therefore requires both persisted layout and width; width 80 is
not accepted as independent evidence of layout 8.

The existing input-width expansion workflow is reused. Its fused LSTM matrix
expansion copies rows 0–72 byte-for-byte, zero-initializes rows 73–75 for the
new TG4 inputs, moves the four learned return rows from 73–76 to 76–79, and
moves recurrent rows after the new complete input block. Biases and heads are
unchanged. This is a semantic remap, not append-only physical matrix growth.

No database migration is needed: migration 089 already persists arbitrary
`model_input_width` and `model_input_semantic_layout_version` values and its
identity immutability/duplicate constraints cover 80/8.

## Semantic workers and publication boundary

The operational semantic-worker registry already retains immutable layout-7
inference and training/reference artifacts outside DerivedData, including
their source commit and SHA-256 identities. Layout-8 publication must use the
existing content-addressed publisher after this source is committed and a
clean Release build exists. The publisher rejects dirty sources; this phase
does not alter the registry, a canonical worker symlink, or any retained
layout-7 artifact.

## Evidence

`TG4TensorModelInputIntegrationTests` covers layout ordering, canonical
timestamp equality, adapter/Tensor replay equality, exact binary values,
hierarchy, all four legal aggregate states, train/inference row parity, and
the full-history score-boundary path. `LSTMInputWidthExpansionTests` uses
sentinel fused-LSTM weights to prove the width-77/layout-7 to
width-80/layout-8 remap, including relocated returns and untouched non-input
parameters. Existing TG4 replay tests continue to cover deterministic replay,
future-tail invariance, same-bar aggregation, and DTL rejection.

Validation in this dirty, non-published tree passed:

- `TG4ProductionStreamingPulseAdapterTests.sh`
- `TG4ProductionStreamingCanonicalReplayTests.sh`
- `TG4TensorModelInputIntegrationTests.sh`
- `LSTMFeatureVectorParityTests.sh`
- `LSTMModelInputCompatibilityTests.sh`
- `LSTMInputWidthExpansionTests.sh`
- `LSTMInputWidthExpansionPersistenceTests.sh` (against a disposable cloned
  database)
- `LSTMModelInputIdentityMigrationTests.sh` (against a disposable cloned
  database)
- `SchedulerSemanticAdmissionTests.sh`
- `SemanticWorkerRegistryTests.sh`
- the directly affected causal-surprise observability unit and repository
  tests.

The ordinary `LSTM Release` build was intentionally attempted with the stable
`DerivedData/ExpertAdvisor` path. It reached the clean-tree provenance guard
and stopped before compiling the `LSTM Release` target:
`GenerateBuildProvenance.py: Release provenance requires a clean source tree`.
This was an expected non-publication guard, not a source compiler diagnostic.
For the same reason the scheduler width-expansion integration harness could
not obtain a matching layout-8 executable; the pre-existing product failed its
embedded-source-commit check before using its disposable database. After a
clean commit, run:

```sh
xcodebuild -project ExpertAdvisor.xcodeproj -scheme "LSTM Release" \
  -configuration Release -derivedDataPath DerivedData/ExpertAdvisor build
bash Tests/LSTMInputWidthExpansionSchedulerIntegrationTests.sh \
  DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release
```

No database migration was executed against production. No canonical or
semantic worker was published, and the retained layout-7 worker registry and
artifacts were read-only inspected only. TG4 remains deliberately absent from
`FeatureAblationMask`; a controlled TG4 ablation requires a separate approved
feature-mask/provenance design.
