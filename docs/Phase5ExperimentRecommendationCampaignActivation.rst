Phase 5 Step 2 — Atomic Campaign Conversion Activation
========================================================

Purpose and authority
---------------------

Phase 5 Step 2 is one separately confirmed convenience over the existing
Phase 4C conversion-activation authority. Given one persisted Phase 4D
materialization, it applies the ordinary Phase 4C activation to every exact
member in one transaction. It adds no campaign lifecycle row, activation
authority, scheduler queue, polling consumer, or worker behavior.

Membership and eligibility
--------------------------

The only membership is the ordered
``experiment_recommendation_campaign_materialization_member`` rows belonging to
the requested ``experiment_recommendation_campaign_materialization``. The
existing materialization and handoff validators establish the supported
version, canonical identity and hash, complete count, contiguous ordinals,
unique proposal links, and exact Phase 4C provenance.

Every member must have one valid immutable Phase 4C execution and its exact
experiment. First activation reuses the Phase 4C requirement that the
experiment is ``paused/train``, has ``invocation_mode``
``recommendation_conversion`` and ``duplicate_nonce=0``, and has no worker,
operation, epoch, start, completion, exit, error, or result metadata. The only
transition is the existing ``paused/train`` to ``pending/train`` transition.

Transaction, locks, and retry
-----------------------------

A write owns one PostgreSQL transaction. It resolves immutable execution and
experiment IDs, acquires the existing execution-scoped activation advisory
locks in ascending execution-ID order, then locks experiment rows in ascending
experiment-ID order. It reloads and validates all evidence before the first
insert or update. The shared Phase 4C transaction-bound primitive starts and
commits no transaction. One member failure rolls back every activation row and
experiment transition in the request.

An exact retry is ``already_satisfied`` only when every exact member has its
valid ordinary activation and every experiment remains in the exact
``pending/train`` post-activation state. A mix of activated and unactivated
members is a deterministic partial-operation conflict and is not completed.
Malformed or conflicting execution, activation, experiment, handoff, or
materialization evidence fails closed. Existing Phase 4C activation identities
and rows remain the durable authority; no campaign activation table is needed.

CLI and dry run
---------------

::

   LSTM_Release \
     --activate-recommendation-campaign-materialization=MATERIALIZATION_ID \
     --yes

The separated option form is also accepted. ``--yes`` is required for a write.
``--dry-run`` instead performs complete validation in one repeatable-read,
read-only transaction, takes no advisory or row lock, writes no row, changes no
experiment, and advances no sequence. Its result is advisory because another
explicit operation may change evidence after the snapshot.

Output and boundary
-------------------

``RECOMMENDATION_CAMPAIGN_ACTIVATION`` is followed by one
``RECOMMENDATION_CAMPAIGN_ACTIVATION_MEMBER`` row per persisted ordinal. Output
includes the materialization and operation hashes; exact recommendation,
proposal, original authorizing review, execution, experiment, and activation
IDs; pre/post lifecycle state; counts; diagnostics; and explicit all-or-nothing,
scheduler, worker, direct-process, and automatic-follow-up safety fields.
Optional IDs use ``null`` and variable text uses the production percent encoder.

Activation does not execute a worker or complete an experiment. It only makes
the ordinary experiments ``pending/train``. Any later scheduler claim and
worker lifecycle remain the existing scheduler's responsibility. The command
does not automatically invoke Phase 5 Step 1 or any later operation.
Phase 5 Step 3 is a distinct, separately confirmed command that may combine
the existing execution and activation transaction-bound authorities under one
outer transaction. Step 2 remains activate-only and never invokes Step 3.
Phase 5 Step 4 can observe activated members and their later persisted
lifecycle evidence in one read-only snapshot; it never invokes Step 2.

Schema
------

No migration or privilege change is required. Campaign scope is the immutable
Phase 4D materialization; outcomes are the existing immutable Phase 4C
activation rows and authorized experiment lifecycle transitions.

References
----------

* ``docs/Phase4CExperimentRecommendationConversionActivation.rst``
* ``docs/Phase4DExperimentRecommendationCampaignHandoff.rst``
* ``docs/Phase5ExperimentRecommendationCampaignExecution.rst``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
* ``docs/Phase5ExperimentRecommendationCampaignLaunch.rst``
