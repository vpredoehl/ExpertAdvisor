Campaign Manager Phase 10: Profitability Calibration
====================================================

Phase 10 adds a read-only, advisory calibration command over one explicit,
immutable recommendation ranking snapshot::

  LSTM_Release --calibrate-campaign-profitability=RANKING_SNAPSHOT_ID

The command opens one repeatable-read PostgreSQL read transaction.  It does
not persist a calibration artifact, change a recommendation or ranking, create
or queue an experiment, or modify scheduler/worker state.  Profitability-aware
production ranking remains disabled and the live profitability weight remains
zero.

Coverage audit
--------------

Every ranking member is joined to its exact evaluation result and frozen
recommendation profitability provenance.  The audit reports the member,
recommendation, evaluation result, source experiment/model, symbol/horizon,
frozen FINAL result/observation identity, exact unavailable reason, current
existence probes, and a recovery classification.

``no_profitability_observation`` with an exact FINAL result is classified as a
recoverable historical absence only in the limited sense that a future,
separately authorized exact-FINAL replay could produce a new profitability
artifact.  Phase 10 does not perform that replay and cannot insert the result
into an already frozen recommendation/evaluation snapshot.  Context mismatch,
missing exact FINAL inference, legacy/incomplete provenance, checkpoint
evidence, and cross-context observations are never substituted.

Calibration grid and decision policy
------------------------------------

The command always evaluates the control plus the complete fixed grid from
``0.0025`` through ``0.05`` in ``0.0025`` increments.  It directly compares
the mandatory ``0.01``, ``0.025``, and ``0.05`` anchors and reports population,
movement (including median and nearest-rank p90), sign directionality, top-5,
top-10, top-20, pairwise ordinal differences, exact membership intervals,
discontinuities, and materially sensitive recommendations.

The minimum-effective assessment is deliberately advisory.  It selects the
smallest swept weight that reaches the best observed top-5 positive count,
comes within one candidate of the best top-10 count while reaching at least
nine, strictly improves top-20 over control, moves no positive member down and
no negative member up, and has less mean absolute movement than ``0.05``.  The
report exposes this policy and the contiguous grid interval satisfying it; it
does not hard-code ``0.025``.

Production readiness
--------------------

The final readiness record reports overall and top-N valid-evidence coverage.
No repository coverage threshold currently authorizes activation, so the
command requests a later human-approved policy and reports
``activation_ready=false``.  Frozen-population calibration is in-sample: it
does not prove improved future realized profitability.  A later temporal,
out-of-sample validation must choose a weight from information available at
selection time and evaluate subsequent realized profitability before any
activation review.

Machine-readable records include
``CAMPAIGN_PROFITABILITY_CALIBRATION_START``,
``CAMPAIGN_PROFITABILITY_COVERAGE_SUMMARY``,
``CAMPAIGN_PROFITABILITY_COVERAGE_MEMBER``,
``CAMPAIGN_PROFITABILITY_CALIBRATION_SWEEP_POINT``,
``CAMPAIGN_PROFITABILITY_CALIBRATION_TOP_N``,
``CAMPAIGN_PROFITABILITY_CALIBRATION_PAIRWISE``,
``CAMPAIGN_PROFITABILITY_CALIBRATION_RESPONSE_CURVE``,
``CAMPAIGN_PROFITABILITY_CALIBRATION_STABILITY_REGION``,
``CAMPAIGN_PROFITABILITY_CALIBRATION_MINIMUM_EFFECTIVE_WEIGHT``, and
``CAMPAIGN_PROFITABILITY_PRODUCTION_READINESS``.
