# Volume II — Data Pipeline

Status: Foundation outline
Version: 0.1.1
Last revised: 2026-08-15

## 1. Purpose

Define how market observations become validated, ordered, versioned model
inputs while preserving source provenance and preventing look-ahead leakage.

## 2. Scope

### 2.1 In scope

Source ingestion, symbol/time normalization, validation, missing-data policy,
feature construction, dataset boundaries, and data/feature provenance.

### 2.2 Out of scope

Label semantics belong to Volume III; model topology and tensor consumption
belong to Volume IV; experiment scheduling belongs to Volumes VII and XI.

### 2.3 Current implementation status

The platform has working data and feature paths embedded in current training
and inference code. A complete durable feature/data identity contract is not
yet represented by this outline and MUST NOT be inferred from it.

## 3. Responsibilities

### 3.1 Owned responsibilities

The pipeline owns ordered observations, feature values, input-shape metadata,
source range provenance, validation outcomes, and versioned transformation
definitions.

### 3.2 Dependencies

It consumes authoritative source data and supplies aligned inputs to Volumes
III–VI. Experiment identity depends on its future version contract through
Volume VII.

### 3.3 Prohibited responsibilities

The pipeline MUST NOT choose labels, rank models, change experiment lifecycle
state, or trigger scheduler work.

## 4. Architecture

### 4.1 Components

Adapters read sources; pure transformations normalize and derive features;
validators reject ambiguous inputs; dataset assemblers produce ordered windows.

### 4.2 Control flow

Read source → normalize → validate → transform → align → construct windows →
emit provenance. Every order-dependent stage requires an explicit total order.

### 4.3 Ownership boundaries

Database repositories own source queries. Pure feature code owns calculations.
Training and inference consume immutable outputs rather than redefining them.

## 5. Data model

### 5.1 Authoritative entities

Conceptual entities are source series, observation, feature definition,
feature-set version, dataset range, and constructed sample.

### 5.2 Provenance and versions

Future exact replay requires source revision/snapshot identity, timezone and
calendar rules, feature-set canonical identity, normalization version, and
ordered feature names. A Git commit alone is insufficient.

### 5.3 Invariants and legacy data

Timestamps, symbols, ordering, duplicates, missing values, and finite numeric
domains require explicit policy. Legacy data lacking snapshot identity may be
usable for compatible research but MUST NOT be claimed as exact replay.

### 5.4 Implemented LSTM relative tick-volume feature

The canonical LSTM base-feature layout is append-only. Column 36 is the fixed
causal 32-bar relative tick-volume feature:

``log((vol_t + 1) / (mean(vol_(t-32)..vol_(t-1)) + 1))``.

The reference uses up to 32 completed predecessor observations, excluding the
current completed bar. The bootstrap row is zero; partial predecessor history
is used as-is; and zero, invalid, or non-finite direct inputs have a finite,
deterministic fallback. The active `candlestick` source's `vol` field is the
authoritative tick-volume input. The fixed lookback is not an experiment
parameter. Historical base prefixes and their model input widths remain 32/36,
34/38, and 36/40; the current 37-column base tensor maps to `n_in=41` after
the existing four appended return channels.

## 6. Transactions

### 6.1 Read paths

Dataset reads SHOULD use a consistent read-only snapshot for one construction.

### 6.2 Write paths

Any future materialization writes data plus its version/provenance atomically.

### 6.3 Failure semantics

Invalid or ambiguous source data fails with a stable diagnostic; partial
materializations are not published as complete datasets.

## 7. Concurrency

### 7.1 Conflict domain

Conflicts are scoped to the same source partition or materialized dataset key.

### 7.2 Locking and serialization

Future writers SHOULD use unique version keys or narrow database claims rather
than globally serializing readers.

### 7.3 Winner, loser, and retry outcomes

Equivalent materialization retries may reuse an identical completed artifact;
non-equivalent results under one identity are an integrity failure.

## 8. CLI

### 8.1 Commands and validation

Future commands must identify source, range, and version explicitly.

### 8.2 Machine output

Events report counts, ranges, versions, skips, and validation reasons.

### 8.3 Human output

Summaries distinguish completeness from compatibility and never imply model
quality.

## 9. Testing

### 9.1 Pure tests

Cover calendars, ordering, missing values, feature formulas, and boundary cases.

### 9.2 Persistence and migration tests

Cover snapshot/version constraints and atomic publication when introduced.

### 9.3 Concurrency and integration tests

Use fixed miniature datasets and verify identical outputs across retries.

### 9.4 Regression boundaries

Label, training, inference, experiment, and scheduler behavior remain unchanged
unless an accepted cross-volume ADR says otherwise.

## 10. Operational safety

### 10.1 Runtime isolation

Validation must not mutate live experiment or scheduler state.

### 10.2 Permissions and destructive operations

Source repair and deletion require owner procedures, never model-worker rights.

### 10.3 Observability and recovery

Materializations expose durable completeness and can be rebuilt from named
authoritative inputs when the contract supports it.

## 11. Future extensions

### 11.1 Approved extension points

Versioned source adapters, feature registries, and immutable dataset manifests.

### 11.2 Deferred capabilities

Durable data snapshots, alternative vendors, feature discovery, and streaming.

### 11.3 Required decisions

Each requires an ADR covering identity, retention, leakage controls, storage,
and compatibility before implementation.

## 12. References

- [Volume I §§3–5](Volume_I_Foundation.md)
- [Volume III](Volume_III_Label_Generation.md)
- [Volume VII](Volume_VII_Experiment_Lifecycle.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.1 | 2026-08-15 | Recorded the fixed causal 32-bar relative tick-volume feature and 37/41 input contract. | — |
| 0.1.0 | 2026-07-15 | Established the data-pipeline architecture outline. | — |
