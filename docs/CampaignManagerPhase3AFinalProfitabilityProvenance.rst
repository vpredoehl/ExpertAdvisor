Campaign Manager Phase 3A: FINAL Profitability Provenance
========================================================

Phase 3A adds observational evidence only.  Recommendation scoring, source
eligibility, ranking, tie-breaking, selection, and campaign actions retain
their existing behavior.  The profitability scoring weight and contribution
are both exactly zero.

Authoritative source
--------------------

Recommendation source loading resolves the one exact completed ``final``
``inference_eval_result`` for the source experiment's ``last_model_id`` by
using the persisted experiment range and the model's authoritative training
configuration.  The result must have no ``checkpoint_eval_id`` and no
``parent_experiment_id``.  It then selects the one
``inference_profitability_observation`` with the same experiment, model,
inference result, ``final`` scope, no checkpoint identity, and the current
metric-definition canonical/hash.  Missing, ambiguous, or mismatched evidence
is recorded as unavailable; checkpoint evidence is never a fallback.

Immutable evidence
------------------

Migration 076 adds nullable FINAL-profitability fields to
``experiment_recommendation`` and
``experiment_recommendation_evaluation_result``.  All-null recommendation
fields identify a legacy pre-Phase-3A snapshot.  A Phase-3A-aware snapshot is
either explicitly unavailable with a reason, or stores the exact inference
result ID, observation ID, inference range, actionable count, aggregate and
average return, and metric/source/observation hashes.  Zero actionable
predictions are represented by an available observation with count zero,
aggregate return zero, and a NULL average; this is not unavailable evidence.

Database constraints validate FINAL scope and exact source provenance, and an
immutability trigger prevents profitability fields on a recommendation
snapshot from being updated.  Evaluation rows duplicate the frozen evidence
and add an observational canonical/hash for audit and idempotent retry checks.

Identity boundary
-----------------

The profitability canonical/hash is intentionally separate from the existing
recommendation policy, scoring policy, evaluation evidence, evaluation
identity, ranking snapshot, ranking member, and campaign plan identities.
Those existing identities remain decision-bearing.  The Phase 3A hash is
observational only, so profitability availability, sign, magnitude, or a later
observation cannot alter a score, invalidate a candidate, change a tie-break,
or change a campaign action.

Observability
-------------

Existing recommendation status, evaluation status/listing, and campaign plan
candidate output expose availability/reason, exact result and observation IDs,
scope, actionable count, aggregate and average return, and the explicit
``profitability_weight=0`` and
``profitability_score_contribution=0`` declarations.

Deferred work
-------------

Phase 3A does not change ranking-population or scoring-policy homogeneity
(Phase 3B).  It also does not define thresholds, normalization, a nonzero
weight, or any profitability-based decision behavior (Phase 3C).
