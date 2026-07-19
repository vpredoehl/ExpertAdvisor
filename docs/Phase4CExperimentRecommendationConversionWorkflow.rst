Phase 4C Step 6 Conversion Workflow Observability
===================================================

Purpose and boundary
--------------------

Step 6 provides a read-only aggregate over the durable manual conversion
chain::

  recommendation -> proposal -> review -> paused execution -> activation
  -> ordinary scheduler lifecycle

Observation never approves, rejects, executes, activates, queues, starts, or
changes an experiment. The scheduler does not query the workflow read model or
any Phase 4C audit table.

Authoritative ordering and state
--------------------------------

The greatest review-decision primary key remains the authoritative latest
manual disposition. The execution separately retains the exact approving
decision used by Step 4; a later review does not rewrite that history.

The stable workflow states are:

* ``pending_review``: no review decision;
* ``rejected``: latest decision rejects the proposal, including a deliberate
  reversal after execution or activation; durable downstream evidence remains
  visible but does not override the current manual disposition;
* ``approved_not_executed``: latest decision approves and no execution exists;
* ``executed_paused``: execution exists and its experiment remains pristine
  ``paused/train`` without activation;
* ``activated_pending``: activation exists and the experiment remains
  ``pending/train`` without worker ownership;
* ``scheduler_claimed_or_running``: the activated experiment has worker
  ownership, is running, or has advanced into a later pending phase;
* ``completed``, ``failed``, or ``cancelled``: terminal experiment lifecycle;
* ``inconsistent``: persisted provenance, identity, cardinality, or lifecycle
  violates the Phase 4C contract.

``proposed`` is reserved as a stable vocabulary value; an existing persisted
proposal without review is reported more specifically as ``pending_review``.

Integrity diagnostics
---------------------

The read model checks proposal, execution, and activation canonical/hash
pairs; supported contract versions; exact approving-review provenance; exact
execution/activation/proposal/experiment identifiers; uniqueness cardinality;
linked experiment existence; and lifecycle compatibility. Diagnostic codes are
emitted in a fixed order. Inconsistency is observed and reported—it is never
repaired.

Repository behavior
-------------------

One PostgreSQL read transaction supplies a consistent snapshot. A single
bounded query loads each requested workflow, including the greatest review ID,
the exact execution review, execution and activation evidence, and current
experiment lifecycle. Listing orders by proposal ID descending. State-filtered
listing scans at most 1000 deterministic candidates and returns at most the
requested limit; older matching rows outside that bounded window are not
scanned.

No migration or additional privilege is required. Existing proposal,
review-history, execution, activation, and experiment indexes support the
joins.

CLI
---

Inspect one proposal::

  LSTM_Release --recommendation-conversion-workflow=42

List a bounded set::

  LSTM_Release --list-recommendation-conversion-workflows \
      --conversion-workflow-state=activated_pending \
      --conversion-workflow-limit=25

A healthy machine record begins with
``RECOMMENDATION_CONVERSION_WORKFLOW`` and includes IDs, current lifecycle,
separate proposal/execution/activation hashes, integrity, diagnostics, and
explicit read-only safety fields. A missing proposal emits
``RECOMMENDATION_CONVERSION_WORKFLOW_NOT_FOUND`` with a nonzero exit code. An
observed inconsistent workflow is still a successful read and reports
``integrity_status=inconsistent``.

Isolated PostgreSQL verification
--------------------------------

Repository tests require an explicit non-production database and refuse the
database name ``LSTM``::

  createdb phase4c_workflow_test
  LSTM_TEST_DB_NAME=phase4c_workflow_test \
      /tmp/phase4c_step6_repository_tests
  dropdb phase4c_workflow_test

The test creates and drops an exact per-process schema. Step 2–5 repository
regressions use the same ``LSTM_TEST_DB_NAME`` guard.

Deferred work
-------------

Automatic scans, campaign control, profitability-aware selection, workflow
mutation, and scheduler integration remain outside Step 6.

References
----------

* ``docs/Phase4CExperimentRecommendationConversionPersistence.rst``
* ``docs/Phase4CExperimentRecommendationConversionProposalReview.rst``
* ``docs/Phase4CExperimentRecommendationConversionExecution.rst``
* ``docs/Phase4CExperimentRecommendationConversionActivation.rst``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
