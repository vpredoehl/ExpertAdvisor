# Volume II — Data Pipeline

Status: Foundation outline
Version: 0.1.3
Last revised: 2026-08-20

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

### 5.4 Implemented LSTM appended causal features

The canonical LSTM base-feature layout is append-only. Column 36 is the fixed
causal 32-bar relative tick-volume feature:

``log((vol_t + 1) / (mean(vol_(t-32)..vol_(t-1)) + 1))``.

The reference uses up to 32 completed predecessor observations, excluding the
current completed bar. The bootstrap row is zero; partial predecessor history
is used as-is; and zero, invalid, or non-finite direct inputs have a finite,
deterministic fallback. The active `candlestick` source's `vol` field is the
authoritative tick-volume input. The fixed lookback is not an experiment
parameter.

Column 37 is the fixed causal 32-bar RMS-normalized close-return surprise.
For completed bar `t`, `r_t = log(close_t / close_(t-1))`; its reference is up
to 32 immediately preceding valid close-to-close returns, excluding `r_t`.
The feature is:

``clamp(r_t / sqrt(mean(r_j^2)), -10, 10)``.

This denominator is root-mean-square return magnitude, with **no mean
subtraction**. Both closes must be finite and strictly positive for a return to
be valid. The bootstrap row, an empty predecessor set, a non-finite or
non-positive denominator, and an invalid or non-finite current return each
produce `0`. The result is always finite and intrinsically clamped to `[-10,
10]`. Calculation precedes retention: the current valid return enters the
rolling state only after its own feature has been computed, then is available
to future rows. The lookback is a fixed feature definition, not an experiment
parameter.

Subsequent established causal increments occupy columns 38 through 46. Column
47 is causal historical-level proximity. Completed source bars are aggregated
into UTC Monday-based weekly OHLC bars independently inside each symbol's
Tensor. A week is a high or low candidate only when it is the strict extremum
of a five-week window and both following weeks are already complete. Its
log-price bandwidth is one quarter of the median weekly log high/low range over
the 26 weeks ending at the pivot, floored at `1e-6`. Candidate formation also
requires that bandwidth history and the two confirmation weeks be consecutive;
no missing week is synthesized.

Candidates are retained for 260 calendar weeks and taper linearly over their
final 26 weeks. Candidate importance is a smooth Gaussian corroboration by
other retained pivots at similar log prices. Current proximity is the bounded
transform `1 - exp(-density / 2)`, where density sums the age-weighted,
corroboration-weighted Gaussian distance from the current close to every
candidate. Fewer than 104 completed weekly bars, invalid inputs, or no
corroborated candidates produce `0`. Candidate discovery uses only weeks
completed before the current bar; no symbol-independent levels or complete-data
level discovery are permitted.

The current base tensor has 48 columns; its four existing appended return
channels yield current `n_in=52`. Every historical persisted width remains an
exact prefix projection, including the immediately preceding `51 -> 47`
contract; the current mapping is `52 -> 48`.

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
| 0.1.3 | 2026-08-20 | Added causal historical-level proximity at column 47 and the 48/52 input contract. | — |
| 0.1.2 | 2026-08-15 | Added causal 32-bar RMS-normalized close-return surprise at column 37 and the 38/42 input contract. | — |
| 0.1.1 | 2026-08-15 | Recorded the fixed causal 32-bar relative tick-volume feature and 37/41 input contract. | — |
| 0.1.0 | 2026-07-15 | Established the data-pipeline architecture outline. | — |
