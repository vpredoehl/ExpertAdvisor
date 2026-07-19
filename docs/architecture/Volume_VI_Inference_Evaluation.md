# Volume VI — Inference and Evaluation

Status: Foundation outline
Version: 0.1.0
Last revised: 2026-07-15

## 1. Purpose

Define how compatible models produce predictions, metrics, rankings, and
immutable research evidence without changing model parameters.

## 2. Scope

### 2.1 In scope

Model loading, inference ranges, prediction outputs, evaluation metrics,
checkpoint/final analysis, evidence completeness, and deterministic ranking.

### 2.2 Out of scope

Training updates belong to Volume V; profitability simulation to Volume IX;
recommendation scoring to Volume VIII.

### 2.3 Current implementation status

Inference, checkpoint evaluation, and experiment analysis exist. Their exact
current schema and command contracts remain implementation references until
this volume is promoted from outline status.

## 3. Responsibilities

### 3.1 Owned responsibilities

Own compatible model evaluation, prediction/metric calculation, evidence
scope, completeness, deterministic comparison, and result provenance.

### 3.2 Dependencies

Consumes Volumes II–V contracts and supplies evidence to continuation,
recommendation, reporting, and future profitability domains.

### 3.3 Prohibited responsibilities

Evaluation MUST NOT update model weights, approve recommendations, create
experiments, or infer scheduler authorization from a metric.

## 4. Architecture

### 4.1 Components

Compatibility validator, inference runner, metric calculators, evidence
repository, analysis service, and deterministic ranker.

### 4.2 Control flow

Validate model/data contract → infer → persist scoped predictions/results →
calculate analysis → persist immutable evidence → present.

### 4.3 Ownership boundaries

Workers calculate; repositories persist; services determine evidence
completeness; scheduler owns when a worker runs.

## 5. Data model

### 5.1 Authoritative entities

Inference attempt/result, evaluation scope, analysis result, metric component,
rank, source model, and evidence watermark.

### 5.2 Provenance and versions

Results identify model, experiment, data/label scope, metric version, analysis
version, and completeness status.

### 5.3 Invariants and legacy data

Final and checkpoint evidence remain distinct. Missing or mismatched evidence
is not treated as zero or final. Legacy results may be excluded explicitly.

## 6. Transactions

### 6.1 Read paths

Inference reads immutable compatible model state and one defined data range.

### 6.2 Write paths

Result components that define one evidence item commit atomically; lifecycle
finalization is owned by its service/scheduler contract.

### 6.3 Failure semantics

Partial predictions are not promoted to completed evidence. Analysis failure
does not rewrite the source model or experiment configuration.

## 7. Concurrency

### 7.1 Conflict domain

Conflicts concern the same model, evaluation scope, and attempt identity.

### 7.2 Locking and serialization

Unique scope keys or attempt claims prevent duplicate authoritative results
while allowing unrelated models/scopes concurrently.

### 7.3 Winner, loser, and retry outcomes

Equivalent retries compare complete immutable results; mismatches fail visibly
or create a distinct versioned attempt.

## 8. CLI

### 8.1 Commands and validation

Commands identify model/experiment and evaluation scope explicitly.

### 8.2 Machine output

Events expose scope, completeness, versions, metrics, ranks, and stable errors.

### 8.3 Human output

Summaries label metrics as evidence, not profitability or guaranteed quality.

## 9. Testing

### 9.1 Pure tests

Cover metrics, ranking, ties, empty classes, nonfinite inputs, and explanations.

### 9.2 Persistence and migration tests

Cover result ownership, scope uniqueness, completeness, and immutable history.

### 9.3 Concurrency and integration tests

Exercise same-scope retries, different scopes, failure rollback, and model
immutability.

### 9.4 Regression boundaries

Inference changes preserve training, experiment lifecycle, recommendation
status, and scheduler capacity semantics.

## 10. Operational safety

### 10.1 Runtime isolation

Evaluation runs only through explicit commands or scheduler claims.

### 10.2 Permissions and destructive operations

Workers may append owned results but do not rewrite completed evidence.

### 10.3 Observability and recovery

Attempts expose durable phase, scope, progress, completion, and failure reason.

## 11. Future extensions

### 11.1 Approved extension points

Versioned metric and analysis components over immutable inference evidence.

### 11.2 Deferred capabilities

Calibration, robustness evaluation, walk-forward analysis, and profitability
integration.

### 11.3 Required decisions

New evidence requires comparability, versioning, completeness, and scheduler
ownership decisions.

## 12. References

- [Volume I §§3–10](Volume_I_Foundation.md)
- [Volume V](Volume_V_Training_Engine.md)
- [Volume VII](Volume_VII_Experiment_Lifecycle.md)
- [Volume IX](Volume_IX_Trading_Profitability.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established the inference/evaluation outline. | — |
