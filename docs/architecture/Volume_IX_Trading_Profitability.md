# Volume IX — Trading and Profitability Evaluation

Status: Reserved outline; not implemented by this document
Version: 0.1.0
Last revised: 2026-07-15

## 1. Purpose

Define a future evidence layer for translating predictions into reproducible
trading simulations and risk-aware profitability measurements.

## 2. Scope

### 2.1 In scope

Future signal mapping, execution assumptions, transaction costs, position/risk
rules, portfolio accounting, benchmarks, and profitability evidence.

### 2.2 Out of scope

Live trading, brokerage control, capital authorization, recommendation review,
and model training are not authorized by this outline.

### 2.3 Current implementation status

Reserved. Existing evaluation metrics MUST NOT be relabeled as profitability.

## 3. Responsibilities

### 3.1 Owned responsibilities

Future deterministic simulation and immutable financial-evidence provenance.

### 3.2 Dependencies

Consumes Volume VI predictions and Volume II market data under explicitly
versioned assumptions.

### 3.3 Prohibited responsibilities

MUST NOT place orders, move funds, guarantee returns, auto-approve research, or
hide costs and data leakage.

## 4. Architecture

### 4.1 Components

Proposed components are policy canonicalization, simulator, cost/slippage model,
risk engine, benchmark calculator, repository, and reporting service.

### 4.2 Control flow

Immutable predictions + market data + simulation policy → deterministic event
simulation → accounting → metrics → immutable evidence.

### 4.3 Ownership boundaries

Profitability evaluates evidence; it does not own experiment creation,
recommendation status, scheduler work, or external execution.

## 5. Data model

### 5.1 Authoritative entities

Future simulation policy/run, signal, fill, position, equity point, metric,
benchmark, and evidence component.

### 5.2 Provenance and versions

Identity must include prediction set, price-data snapshot, timing, costs,
slippage, risk, capital, benchmark, calendar, and simulator version.

### 5.3 Invariants and legacy data

No result is comparable without compatible policy and data provenance. Missing
costs or ambiguous fills fail closed.

## 6. Transactions

### 6.1 Read paths

Simulation consumes immutable snapshots.

### 6.2 Write paths

Run metadata and complete evidence publish atomically or through an explicit
running/completed state machine.

### 6.3 Failure semantics

Partial simulations are not profitability results.

## 7. Concurrency

### 7.1 Conflict domain

Same prediction/data/policy identity.

### 7.2 Locking and serialization

Future uniqueness or claims remain narrowly scoped per simulation identity.

### 7.3 Winner, loser, and retry outcomes

Exact retry compares all immutable outputs; mismatch creates failure, not reuse.

## 8. CLI

### 8.1 Commands and validation

Any future command requires explicit policy, evidence source, and bounded range.

### 8.2 Machine output

Events expose assumptions, evidence IDs, completeness, costs, risk, and metrics.

### 8.3 Human output

Reports state that simulated historical evidence is not a return guarantee.

## 9. Testing

### 9.1 Pure tests

Hand-verifiable fills, costs, accounting, risk, and metric cases.

### 9.2 Persistence and migration tests

Future ownership, completeness, immutability, and policy constraints.

### 9.3 Concurrency and integration tests

Snapshot-based deterministic reruns and exact rollback/cleanup.

### 9.4 Regression boundaries

No change to predictions, recommendation states, experiments, or scheduling.

## 10. Operational safety

### 10.1 Runtime isolation

Simulation is offline and non-executing.

### 10.2 Permissions and destructive operations

No broker credentials or order privileges belong in this domain.

### 10.3 Observability and recovery

Every result exposes assumptions, version, completeness, and failure reason.

## 11. Future extensions

### 11.1 Approved extension points

Versioned simulation, cost, risk, and benchmark interfaces.

### 11.2 Deferred capabilities

Portfolio optimization, walk-forward capital allocation, paper/live trading.

### 11.3 Required decisions

Each needs dedicated ADRs, independent safety review, authorization, audit,
secrets management, and kill-switch design.

## 12. References

- [Volume I §§2, 10, 17](Volume_I_Foundation.md)
- [Volume VI](Volume_VI_Inference_Evaluation.md)
- [Volume VIII](Volume_VIII_Recommendation_Engine.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Reserved the profitability architecture with strict non-execution boundaries. | — |
