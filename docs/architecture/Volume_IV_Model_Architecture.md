# Volume IV — Model Architecture

Status: Foundation outline
Version: 0.1.0
Last revised: 2026-07-15

## 1. Purpose

Define model topology, tensor contracts, serialization compatibility, and
versioned architectural identity independently of training orchestration.

## 2. Scope

### 2.1 In scope

Input/output shapes, recurrent topology, hidden/layer dimensions, heads,
parameter layout, initialization contract, serialization, and compatibility.

### 2.2 Out of scope

Feature meanings belong to Volume II, labels to Volume III, optimizer behavior
to Volume V, and lifecycle ownership to Volumes VII and XI.

### 2.3 Current implementation status

The current LSTM and model metadata provide operational compatibility checks.
Not every behavior-affecting architectural constant is reconstructible from an
experiment row; this outline does not invent such provenance.

## 3. Responsibilities

### 3.1 Owned responsibilities

Model construction, parameter shape/name contract, forward-state interface,
initialization policy, serialization schema, and load-time compatibility.

### 3.2 Dependencies

Consumes Volume II input tensors and Volume III target/head requirements;
supplies executable models to Volumes V and VI.

### 3.3 Prohibited responsibilities

Model code MUST NOT queue experiments, select checkpoints, mutate database
lifecycle state, or reinterpret incompatible metadata.

## 4. Architecture

### 4.1 Components

Versioned configuration, model builder, recurrent core, output head, state
management, parameter registry, and serializer/loader.

### 4.2 Control flow

Validate configuration → construct or load compatible parameters → execute
forward contract → expose typed outputs and state.

### 4.3 Ownership boundaries

Architecture owns tensor and parameter semantics; training owns updates;
inference owns evaluation execution; persistence owns durable bytes/metadata.

## 5. Data model

### 5.1 Authoritative entities

Model-architecture configuration, model schema version, parameter set, model
state, and serialized model record.

### 5.2 Provenance and versions

Architecture identity must eventually include ordered features, dimensions,
layers, head/target mode, execution/state semantics, and initializer version.

### 5.3 Invariants and legacy data

Loaders validate shapes, schema, target compatibility, and required metadata.
Unknown legacy configurations fail closed rather than being relabeled.

## 6. Transactions

### 6.1 Read paths

Model loads read one internally consistent parameter and metadata snapshot.

### 6.2 Write paths

Model metadata and all required parameter matrices publish atomically from the
consumer's perspective.

### 6.3 Failure semantics

Incomplete saves are not eligible models; failed loads do not mutate lifecycle
state except through the owning service transaction.

## 7. Concurrency

### 7.1 Conflict domain

Conflicts concern one model record or named immutable artifact.

### 7.2 Locking and serialization

Model creation uses unique row ownership and avoids modifying completed models.

### 7.3 Winner, loser, and retry outcomes

Duplicate save retries require authoritative attempt identity; incompatible
content cannot be treated as the same model.

## 8. CLI

### 8.1 Commands and validation

Inspection commands may expose architecture and compatibility metadata without
altering model state.

### 8.2 Machine output

Records use stable schema/version, dimensions, target, and ownership fields.

### 8.3 Human output

Summaries distinguish topology, learned parameters, lineage, and compatibility.

## 9. Testing

### 9.1 Pure tests

Cover construction, shapes, state semantics, initialization, and compatibility.

### 9.2 Persistence and migration tests

Cover complete save/load round trips and rejection of partial/incompatible data.

### 9.3 Concurrency and integration tests

Verify independent model loads and atomic publication under concurrent readers.

### 9.4 Regression boundaries

Model changes must not silently alter labels, optimizer behavior, scoring, or
scheduler ownership.

## 10. Operational safety

### 10.1 Runtime isolation

Architecture inspection never launches training or inference.

### 10.2 Permissions and destructive operations

Completed model artifacts are protected from ordinary runtime deletion or
rewrite according to database policy.

### 10.3 Observability and recovery

Compatibility failures report exact schema/field mismatches and preserve the
source artifact.

## 11. Future extensions

### 11.1 Approved extension points

Versioned model builders, head interfaces, and serialization adapters.

### 11.2 Deferred capabilities

Attention, alternative recurrent cells, ensembles, and multi-head architectures.

### 11.3 Required decisions

Each extension requires an ADR defining semantic identity, serialization,
resume compatibility, inference compatibility, and comparison validity.

## 12. References

- [Volume I §§4–5](Volume_I_Foundation.md)
- [Volume II](Volume_II_Data_Pipeline.md)
- [Volume III](Volume_III_Label_Generation.md)
- [Volume V](Volume_V_Training_Engine.md)

## 13. Revision history

| Version | Date | Change | ADR |
|---|---|---|---|
| 0.1.0 | 2026-07-15 | Established the model-architecture outline. | — |
