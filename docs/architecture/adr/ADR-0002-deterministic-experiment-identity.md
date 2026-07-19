# ADR-0002: Deterministic experiment identity

Status: Accepted
Date: 2026-07-15
Deciders: Project architecture
Affected volumes: Volume I §§3–5; Volume VII; Volume VIII
Supersedes: None
Superseded by: None

## 1. Context and problem statement

The platform must distinguish a research question from a particular execution.
Database row IDs locate lifecycle records but do not describe semantic
equivalence. Resume lineage and checkpoint cadence can distinguish invocations
without necessarily changing the semantic hyperparameter/data question.

Hashes are useful lookup accelerators, but finite hashes can collide and cannot
serve as authoritative equality without an additional collision contract.

## 2. Decision

Experiment-related identity is deterministic, versioned, and content-based.

- Semantic configuration canonical text is authoritative for the named
  research equivalence.
- Invocation identity is separately defined when lineage or operational inputs
  must distinguish executions.
- Row IDs, lifecycle status, timestamps, PIDs, logs, and result metrics do not
  define semantic identity.
- Canonicalization defines field order, nulls, numeric formatting, dates,
  symbols, collections, and a version prefix.
- Hashes accelerate lookup and diagnostics only. Every possible hash match is
  resolved by canonical-text comparison, and collisions remain distinct and
  observable.
- Any identity-semantic change requires a new canonical version, compatibility
  analysis, migration strategy, volume update, and ADR.

## 3. Rationale and decision drivers

- Deterministic duplicate detection across processes and time.
- Separation of scientific equivalence from operational execution.
- Explicit collision safety.
- Stable provenance for recommendations, scores, and future comparisons.

## 4. Consequences

### 4.1 Positive consequences

- Equivalent configurations compare consistently independent of row IDs.
- Resume and operational differences can remain visible without corrupting
  semantic duplicate rules.
- Hash collisions cannot silently merge research configurations.

### 4.2 Negative consequences and trade-offs

- Canonical grammar and versions require careful maintenance.
- Historical rows lacking complete configuration may be unmappable.
- Adding a behavior-affecting setting may require schema and identity changes.

### 4.3 Risks and mitigations

- Hidden compiled settings: document as unreconstructible and fail closed where
  required until a durable implementation contract exists.
- Locale/timezone drift: use locale-independent numbers and explicit Gregorian
  date mapping.
- Hash-only shortcuts: prevent through repository tests and schema design.

## 5. Compatibility and migration

This ADR records the identity approach reflected in Phase 4. It does not
retroactively assign canonical identities to legacy rows. Existing canonical
versions remain stable. New versions must coexist or migrate through an
explicit accepted plan.

## 6. Verification and operational evidence

- Pure canonicalization and strict parsing tests.
- Stable known canonical text and hash fixtures.
- Adjacent floating-value, null, date, collection, and symbol cases.
- Collision tests proving equal hash/unequal text remains distinct.
- Repository duplicate and concurrency tests using full canonical values.

## 7. Alternatives considered

### 7.1 Database row ID as experiment identity

Rejected because it identifies one record, not semantic equivalence.

### 7.2 Hash-only equality

Rejected because collisions would silently merge or block distinct research.

### 7.3 One identity for semantics and invocation

Rejected because lineage and checkpoint controls have different equivalence
meaning from the underlying research configuration.

## 8. References

- [Volume I §5](../Volume_I_Foundation.md)
- [Volume VII](../Volume_VII_Experiment_Lifecycle.md)
- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
- [Phase 4 recommendation foundation](../../Phase4AExperimentRecommendationFoundation.rst)

## 9. Revision history

| Date | Change |
|---|---|
| 2026-07-15 | Recorded canonical semantic and invocation identity as the durable experiment-equivalence foundation. |
