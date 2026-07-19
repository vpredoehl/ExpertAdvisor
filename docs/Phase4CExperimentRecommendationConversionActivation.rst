Phase 4C Step 5 Manual Conversion Activation
==============================================

Purpose and boundary
--------------------

Step 5 adds a second explicit operator action after Phase 4C Step 4. It
activates the one existing experiment linked by an immutable conversion-
execution record. Activation does not create an experiment and does not start
a worker. It changes the existing experiment from ``paused/train`` to
``pending/train``, the scheduler-eligible lifecycle state already used by the
project.

The scheduler remains unaware of Phase 4C audit tables. It continues to select
ordinary pending experiments through its existing lifecycle query.

Eligibility and integrity
-------------------------

The activation repository accepts one positive conversion-execution ID and
loads the exact persisted Step 4 execution. Before first activation it verifies
that:

* the execution, proposal, approving review decision, and created experiment
  retain the complete Step 4 provenance and invocation contract;
* the experiment is ``paused/train``;
* ``worker_pid``, ``current_operation``, and ``current_epoch`` are NULL;
* the Step 4 experiment-identity ``duplicate_nonce`` remains zero;
* no start, completion, exit, error, or model-result metadata is present; and
* ``invocation_mode`` remains ``recommendation_conversion``.

The repository rejects an invalid state; it does not repair lifecycle fields.
An existing exact activation record remains replayable after the experiment
later advances through its ordinary lifecycle.

Identity, transaction, and concurrency
---------------------------------------

The version-1 activation identity binds the conversion execution, proposal,
approving review decision, experiment, Step 4 execution identity hash, and the
exact ``paused/train`` to ``pending/train`` transition. Canonical text is
authoritative; its tagged hash is an accelerator.

Migration 039 adds
``experiment_recommendation_conversion_activation``. One short transaction:

1. takes an execution-scoped transaction advisory lock;
2. loads and validates the Step 4 execution;
3. locks only the linked experiment row;
4. inserts the immutable activation record; and
5. conditionally transitions that experiment to ``pending/train``.

The audit insert and lifecycle update commit or roll back together. Unique
execution and experiment references guarantee one activation through this
workflow. Concurrent identical requests converge on the same record. A replay
returns ``existing_identical`` without repeating the lifecycle update; a
different canonical identity conflicts without mutation.

Phase 5 Step 2 reuses this exact assessment, identity, execution-scoped lock,
experiment-row lock, insert, and conditional lifecycle update inside one outer
all-member transaction. The original single-execution command retains its own
transaction and commit ownership.

Database privileges
-------------------

Runtime access is limited to ``SELECT`` and column-limited ``INSERT`` on the
activation table plus sequence ``USAGE``. Runtime ``UPDATE``, ``DELETE``, and
``TRUNCATE`` are revoked, as are PUBLIC privileges. Restrictive foreign keys
preserve the execution, proposal, approving review decision, and experiment
provenance.

CLI
---

The explicit standalone commands are::

  --activate-recommendation-conversion-execution=EXECUTION_ID
  --recommendation-conversion-activation-status=ACTIVATION_ID

Successful output reports the activation, execution, proposal, approving
decision, experiment, previous and resulting lifecycle states, and activation
identity hash. It explicitly reports that no experiment or worker was created
or started and that the scheduler was not started.

Deferred work
-------------

Worker launch, scheduler polling of Phase 4C tables, automatic activation,
capacity decisions, budgets, and autonomous recommendation execution remain
outside Step 5. The existing scheduler may later claim the ordinary pending
experiment exactly as it would any other eligible experiment.

Phase 4C Step 6 provides a separate read-only aggregate for observing proposal,
review, execution, activation, and current experiment lifecycle provenance.
Phase 5 Step 2 provides a separately confirmed exact-materialization aggregate
convenience; it introduces no alternate activation or scheduler authority.

References
----------

* ``docs/architecture/adr/ADR-0004-scheduler-ownership-boundaries.md``
* ``docs/architecture/adr/ADR-0005-manual-recommendation-conversion.md``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
* ``docs/Phase4CExperimentRecommendationConversionExecution.rst``
