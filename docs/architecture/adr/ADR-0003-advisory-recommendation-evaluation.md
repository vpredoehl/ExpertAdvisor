# ADR-0003: Advisory recommendation evaluation

Status: Accepted
Date: 2026-07-15
Deciders: Project architecture
Affected volumes: Volume I §§2.5, 10.3; Volume VIII
Supersedes: None
Superseded by: None

## 1. Context and problem statement

The recommendation subsystem can generate deterministic research proposals,
persist duplicate/provenance evidence, score and rank proposals, and record an
explicit operator review. These facts are useful for research prioritization,
but they do not provide a complete implementation identity, execution budget,
profitability guarantee, or experiment-creation authorization.

Combining proposal, evaluation, approval, conversion, and execution would make
scores or status changes implicit operational authority.

## 2. Decision

Recommendation generation, scoring, ranking, explanation, and review are
advisory and non-executing.

- Scores and ranks do not change recommendation status.
- Review actions are explicit operator inputs, never threshold-derived.
- Approval means only reviewed for possible later use.
- Approval does not create an experiment, queue work, authorize execution,
  predict profitability, or establish scheduler eligibility.
- Recommendation commands remain outside scheduler polling.
- Review history is immutable to the normal runtime role.
- Recommendation-to-experiment conversion is a separate deferred capability
  requiring its own accepted ADR and service boundary.

## 3. Rationale and decision drivers

- Preserve provenance between proposal, evidence, decision, and execution.
- Prevent accidental work creation from heuristic scores.
- Allow operator review without overstating model or financial confidence.
- Keep scheduler and experiment lifecycle ownership explicit.

## 4. Consequences

### 4.1 Positive consequences

- Operators can inspect and disposition proposals safely.
- Scores remain reproducible evidence rather than hidden policy triggers.
- No Phase 4 command consumes scheduler capacity or changes experiments.
- Future conversion can be designed with complete identity and audit.

### 4.2 Negative consequences and trade-offs

- Approved recommendations require a separate future workflow before use.
- There is no automatic throughput from recommendation to experiment.
- Operators must interpret advisory evidence without a profitability guarantee.

### 4.3 Risks and mitigations

- Misreading approval as authorization: mitigated by CLI disclaimers,
  documentation, and absence of conversion code.
- Silent status changes: prevented by explicit commands, terminal transitions,
  row locks, and immutable events.
- Score-driven automation by accident: prevented by scheduler isolation and
  service boundaries.

## 5. Compatibility and migration

This ADR records Phase 4A Steps 1–5. It requires no new behavioral migration.
Existing recommendation identities, duplicate rules, score history, review
events, and experiment rows remain unchanged.

## 6. Verification and operational evidence

- Pure candidate, scoring, ranking, and review-transition tests.
- Persistence and concurrency tests for duplicates, scores, and review events.
- Before/after experiment-state and score-history comparisons.
- CLI mutual-exclusion and scheduler-isolation tests.
- Runtime privilege checks for immutable review history.

## 7. Alternatives considered

### 7.1 Automatically queue the highest score

Rejected because a deterministic prioritization score is not execution
authority or evidence of expected profitability.

### 7.2 Treat approval as conversion authorization

Rejected because Step 5 lacks a conversion identity, budget, lifecycle request,
and scheduler-capacity contract.

### 7.3 Scheduler-managed recommendation polling

Rejected for Phase 4A because it would collapse advisory evaluation into
automation without explicit ownership and safety decisions.

## 8. References

- [Volume I §10.3](../Volume_I_Foundation.md)
- [Volume VIII](../Volume_VIII_Recommendation_Engine.md)
- [Phase 4 recommendation review](../../Phase4AExperimentRecommendationReview.rst)

## 9. Revision history

| Date | Change |
|---|---|
| 2026-07-15 | Recorded the advisory, non-executing recommendation boundary. |
