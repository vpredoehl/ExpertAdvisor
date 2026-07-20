Phase 5 Step 3 — Atomic Campaign Conversion Launch
====================================================

Purpose and authority
---------------------

Phase 5 Step 3 is one explicitly confirmed orchestration convenience over the
existing Phase 4C conversion-execution and conversion-activation authorities.
For one persisted Phase 4D campaign materialization, it establishes the
ordinary execution and activation outcome for every exact member in one
PostgreSQL transaction. A successful first launch leaves every resulting
experiment in the ordinary ``pending/train`` state. It does not start or signal
the scheduler, poll for work, launch a worker, or start training.

The operation adds no launch table, campaign status, lifecycle authority,
scheduler queue, privilege, or migration. Durable authority remains the
immutable Phase 4D materialization, ordinary Phase 4C proposal reviews,
executions, activations, and experiment rows.

Membership and identity
-----------------------

The sole member set is the persisted ordinal order of
``experiment_recommendation_campaign_materialization_member`` rows for the
requested ``experiment_recommendation_campaign_materialization``. Existing
Phase 4D loaders and validators establish the supported materialization
contract, canonical identity and hash, exact count, contiguous ordinals,
unique links, and proposal provenance. No recommendation, ranking, score,
symbol, horizon, policy, or current query can substitute membership.

The version-1 launch operation identity binds the operation type and contract,
the exact materialization ID, contract, canonical identity and hash, ordered
member count, and launch-semantics version. The validated immutable
materialization identity already binds the ordered exact members. The launch
identity contains no timestamp, process value, random input, or sequence ID;
ordinary Phase 4C execution and activation rows remain the durable retry
authority.

Transaction and lock hierarchy
------------------------------

One write transaction owns the complete launch. It:

1. acquires existing proposal review/execution advisory locks in ascending
   proposal-ID order;
2. reloads and validates the complete execution phase;
3. creates every required paused experiment and ordinary execution row;
4. resolves the complete execution and experiment set;
5. acquires existing activation advisory locks in ascending execution-ID
   order, then experiment row locks in ascending experiment-ID order;
6. reloads and validates the complete activation phase;
7. creates every required activation row and applies the existing
   ``paused/train`` to ``pending/train`` transition; and
8. commits once.

The reused Step 1, Step 2, and Phase 4C transaction-bound primitives begin and
commit no transaction. No activation mutation occurs before every member
passes activation validation. An execution-stage or activation-stage failure
rolls back every experiment, execution, activation, and lifecycle update made
by the request. The shared lock domains serialize direct Phase 4C review,
execution, and activation commands and overlapping campaign commands; locks
are not taken in materialization order.

Supported state matrix
----------------------

* No executions and no activations: create all ordinary executions and
  activations, returning ``newly_launched``.
* All valid executions and no activations: reuse the immutable executions,
  activate all experiments, and return ``activated_existing_executions``.
* All valid executions and activations, with all experiments still exactly
  ``pending/train``: change nothing and return ``already_satisfied``.
* Some but not all executions: reject with a partial-execution conflict.
* Some but not all activations: reject with a partial-activation conflict.
* Activation without exact execution provenance, malformed evidence, or an
  invalid lifecycle state: fail closed without repair.
* An activated experiment that has progressed beyond ``pending/train`` is not
  rewound and is not reported as an ordinary idempotent retry.

An existing execution keeps its original approving review as immutable
authorization even if a later ordinary review reverses current disposition.
A proposal that needs first execution must still have a current valid approve
decision.

CLI and dry run
---------------

Write::

   LSTM_Release \
     --launch-recommendation-campaign-materialization=MATERIALIZATION_ID \
     --yes

Dry run::

   LSTM_Release \
     --launch-recommendation-campaign-materialization MATERIALIZATION_ID \
     --dry-run

Both option forms require one positive ID. Duplicate or malformed values and
combinations with another standalone operation are rejected before database
dispatch. ``--yes`` is required for a write and is not required for dry run.

Dry run uses one repeatable-read, read-only transaction and takes no advisory
or row lock. It writes nothing and advances no sequence. When executions do not
yet exist, their future execution, experiment, and activation IDs are reported
as ``null`` rather than allocated. Results are advisory because another
explicit operation may change workflow evidence after the snapshot.

Output and operational boundary
-------------------------------

``RECOMMENDATION_CAMPAIGN_LAUNCH`` is followed by one
``RECOMMENDATION_CAMPAIGN_LAUNCH_MEMBER`` record per persisted ordinal. The
records contain materialization and operation hashes, aggregate create/reuse
counts, exact provenance IDs when available, execution and activation
dispositions, pre/post lifecycle state, diagnostics, dry-run state, and
explicit all-or-nothing and safety fields. Optional identifiers use ``null``;
variable text uses the production machine-text encoder.

Step 1 remains execute-only and leaves experiments ``paused/train``. Step 2
remains activate-only and requires existing executions. Step 3 combines those
existing authorities atomically without calling either public command and
without adding campaign lifecycle state. Pending experiments are merely
eligible for a later claim by the existing scheduler; launch does not imply
that training has started.

References
----------

* ``docs/Phase4CExperimentRecommendationConversionExecution.rst``
* ``docs/Phase4CExperimentRecommendationConversionActivation.rst``
* ``docs/Phase4DExperimentRecommendationCampaignHandoff.rst``
* ``docs/Phase5ExperimentRecommendationCampaignExecution.rst``
* ``docs/Phase5ExperimentRecommendationCampaignActivation.rst``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
