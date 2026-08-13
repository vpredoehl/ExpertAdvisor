Phase 5 Step 4 — Recommendation Campaign Operational Status
=============================================================

Purpose and authority
---------------------

Phase 5 Step 4 is a read-only point-in-time projection for one persisted Phase
4D campaign materialization. It reports every exact member's ordinary Phase 4C
review, execution, activation, experiment lifecycle, worker metadata, model,
final inference, and final analysis evidence. It creates no campaign status or
lifecycle authority and persists nothing.

The immutable ordered rows in
``experiment_recommendation_campaign_materialization_member`` are the only
membership source. The production Phase 4D loader validates the manifest,
canonical identity and hash, selected count, contiguous ordinals, unique links,
and exact proposal provenance before downstream status is projected.

Snapshot and query contract
---------------------------

One PostgreSQL ``REPEATABLE READ``, ``READ ONLY`` transaction owns the complete
snapshot. The operation takes no advisory or row locks, performs no write or
sequence call, and uses bounded exact-ID queries for the materialized proposals
and their linked experiments. Unrelated workflow and experiment rows cannot
become campaign members.

``observed_at`` is one PostgreSQL transaction timestamp shared by every output
row. It is metadata, not identity. The deterministic version-1 snapshot
identity binds the validated materialization and every ordered member's durable
review, execution, activation, experiment lifecycle, worker, model, phase-
completion, exact completed/failed final-result counts, consistency, and
diagnostic projection.
Persisted lifecycle timestamps are evidence and are bound; ``observed_at`` and
local wall-clock time are not identity inputs. The identity contains no random
value, local command process ID, transaction ID, or unordered iteration. A
persisted ``worker_pid`` is bound as database evidence but is not treated as a
live-process observation.

Member and aggregate status
---------------------------

Execution, activation, operational state, terminal result, and consistency are
separate dimensions. Operational states distinguish proposal-only, executed
paused, pending scheduler claim, running train/infer/analyze, completed, failed,
cancelled, paused after progress, and inconsistent evidence. An execution keeps
its immutable original authorizing approval even when the greatest current
review-decision ID later records rejection.

Final inference is configured only when both persisted inference dates exist.
For ordinary recommendation-conversion experiments, the existing scheduler
requires that range after training and advances completed final inference to
final analysis. Successful ``completed/done`` therefore requires the exact
final inference row selected by the scheduler (current model, symbol, horizon,
threshold, and dates) and completed final analysis for the current model. A
persisted ``completed/done`` recommendation-conversion experiment without that
configured pipeline is reported as inconsistent rather than treated as a
train-only success.

Unknown lifecycle values, invalid cardinality or provenance, activation without
execution, missing experiments, incompatible activation/lifecycle state,
incoherent worker evidence, and missing configured terminal results remain
visible as deterministic member diagnostics. A valid materialization still
returns every ordered member when downstream evidence is malformed. An invalid
materialization rejects the command because exact membership cannot be trusted.
Any inconsistent member forces aggregate ``inconsistent`` status.

CLI and output
--------------

::

   LSTM_Release --recommendation-campaign-status=MATERIALIZATION_ID
   LSTM_Release --recommendation-campaign-status MATERIALIZATION_ID

The positive ID is required. Duplicate options, ``--dry-run``, ``--yes``, and
combinations with any other standalone operation are rejected before database
dispatch because the command is intrinsically read-only.

``RECOMMENDATION_CAMPAIGN_STATUS`` reports snapshot/materialization identities,
aggregate state and counts, percentages and epoch aggregates where meaningful,
and explicit no-write/no-lock/no-scheduler safety fields. It is followed by one
``RECOMMENDATION_CAMPAIGN_STATUS_MEMBER`` record per persisted ordinal with
exact provenance IDs, independent state dimensions, persisted lifecycle and
worker fields, phase-completion flags, consistency, and diagnostics. Optional
values use ``null`` and variable text uses the production percent encoder.

The snapshot is advisory and may become stale immediately after its transaction
ends. It observes persisted scheduler/worker evidence only; it does not inspect
processes, poll or signal the scheduler, launch workers, or control experiments.

Schema and boundaries
---------------------

No migration or privilege change is required. Runtime access uses existing
``SELECT`` privileges. Step 1 remains execute-only, Step 2 activate-only, Step 3
atomic execute-and-activate, and Step 4 observational only. Step 4 invokes none
of the preceding mutation commands and performs no automatic follow-up.

References
----------

* ``docs/Phase4DExperimentRecommendationCampaignHandoff.rst``
* ``docs/Phase5ExperimentRecommendationCampaignExecution.rst``
* ``docs/Phase5ExperimentRecommendationCampaignActivation.rst``
* ``docs/Phase5ExperimentRecommendationCampaignLaunch.rst``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
