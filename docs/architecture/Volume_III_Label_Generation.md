# Volume III — Label Generation

Status: Foundation outline
Version: 0.1.0
Last revised: 2026-07-15

## 1. Purpose

Define deterministic target and label semantics so training and evaluation use
the same versioned research question without look-ahead ambiguity.

## 2. Scope

### 2.1 In scope

Target modes, class definitions, horizons, thresholds, first-hit/look-ahead
rules, edge handling, target transformations, and label provenance.

### 2.2 Out of scope

Feature construction belongs to Volume II, loss implementation to Volume V,
and prediction metrics to Volume VI.

### 2.3 Current implementation status

Current binaries implement established label behavior and persist portions of
the configuration. This outline does not claim that every historical label
input can be reconstructed from an experiment row.

## 3. Responsibilities

### 3.1 Owned responsibilities

This domain owns the mapping from aligned observations and an explicit label
policy to targets, validity masks, and class/evidence counts.

### 3.2 Dependencies

It consumes Volume II observations and provides targets to Volume V plus
evaluation semantics to Volume VI.

### 3.3 Prohibited responsibilities

Label code MUST NOT select experiments, tune thresholds implicitly, change
lifecycle state, or infer missing policy versions.

## 4. Architecture

### 4.1 Components

Pure policy parsing, canonicalization, label calculation, boundary validation,
and summary calculation form the core.

### 4.2 Control flow

Validate policy and aligned range → calculate future-dependent target under the
named rule → mask unavailable boundaries → emit deterministic label and counts.

### 4.3 Ownership boundaries

Repositories may load persisted policy/provenance; pure label code alone owns
the mathematical mapping.

## 5. Data model

### 5.1 Authoritative entities

Label policy, label-policy version, target value, target class, validity mask,
and label summary.

### 5.2 Provenance and versions

Canonical identity includes every behavior-affecting rule: target mode, horizon,
thresholds, class mapping, transformation, and boundary policy.

### 5.3 Invariants and legacy data

Thresholds and transformed targets must be finite; horizons positive; classes
closed and stable. Missing historical version data is reported, not guessed.

## 6. Transactions

### 6.1 Read paths

Label computation is pure over a stable input dataset and policy.

### 6.2 Write paths

Future persisted label artifacts publish values and their policy identity
atomically.

### 6.3 Failure semantics

Invalid inputs produce no partially authoritative label set.

## 7. Concurrency

### 7.1 Conflict domain

Only identical dataset/policy artifact identities conflict.

### 7.2 Locking and serialization

Use unique canonical identities or narrow claims for future materializations.

### 7.3 Winner, loser, and retry outcomes

Identical retries must compare complete results; mismatches are not idempotent.

## 8. CLI

### 8.1 Commands and validation

Future diagnostics accept explicit data range and label policy only.

### 8.2 Machine output

Records expose policy identity, valid/invalid counts, class counts, and reasons.

### 8.3 Human output

Summaries describe label distribution without claiming predictive usefulness.

## 9. Testing

### 9.1 Pure tests

Cover exact boundary, first-hit, class, horizon, leap/calendar, and invalid-data
cases with hand-verifiable fixtures.

### 9.2 Persistence and migration tests

Cover policy ownership and immutable materialization if persistence is added.

### 9.3 Concurrency and integration tests

Compare training and inference/evaluation label interpretation on identical
fixtures.

### 9.4 Regression boundaries

Feature values, model math, scheduling, and experiment status do not change.

## 10. Operational safety

### 10.1 Runtime isolation

Label diagnostics do not queue or mutate experiments.

### 10.2 Permissions and destructive operations

Immutable label artifacts, if added, use least-privilege runtime access.

### 10.3 Observability and recovery

Policy/version mismatches fail visibly before training or comparison.

## 11. Future extensions

### 11.1 Approved extension points

Versioned policies and pure label calculators.

### 11.2 Deferred capabilities

Alternative target types, adaptive thresholds, multi-horizon and multi-task
labels.

### 11.3 Required decisions

New labels require an ADR addressing identity, comparability, metrics, model
compatibility, and migration.

## 12. References

- [Volume I §§3–5](Volume_I_Foundation.md)
- [Volume II](Volume_II_Data_Pipeline.md)
- [Volume V](Volume_V_Training_Engine.md)
- [Volume VI](Volume_VI_Inference_Evaluation.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established the label-generation architecture outline. | — |
