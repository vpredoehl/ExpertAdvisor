Phase 4D Step 1 Deterministic Recommendation Campaign Planning
================================================================

Purpose and boundary
--------------------

Step 1 produces a reproducible, read-only answer to: which members of one
explicit persisted recommendation-ranking snapshot would be selected for a
bounded future campaign, and why? It creates no campaign row, proposal,
review, execution, activation, experiment, scheduler action, or worker.

The command is an explicit operator request. The policy defaults to disabled
in the domain; ``--plan-recommendation-campaign`` is the opt-in that enables
it. The scheduler does not poll plans or Phase 4C audit tables.

Authoritative evidence
----------------------

The planner requires one completed Phase 4B ranking snapshot by primary key.
Its immutable member order and final scores are authoritative. Recommendation
rows supply persisted source metrics and display/configuration identities.
Phase 4C workflow state is loaded through the existing Step 6 derivation, so
campaign planning does not redefine review or lifecycle semantics.
Copied ranking-member source experiment, symbol, horizon, and family values are
also checked against the authoritative recommendation row. A mismatch is
retained and excluded as ``identity_validation_failed``.

The repository uses one PostgreSQL read transaction and a fixed number of
set-based queries. It reads at most 1001 ranking members to enforce the hard
1000-candidate boundary. No production statement uses mutation SQL, row locks,
advisory locks, or sequences. Existing indexes are sufficient; Step 1 adds no
migration or privilege.

Policy version 1
----------------

The versioned policy contains:

* an explicit enabled flag;
* maximum selected and considered counts;
* minimum persisted leader score and inference accuracy;
* maximum persisted predicted-neutral proportion;
* an optional minimum profitability threshold;
* optional per-symbol and per-horizon limits;
* a per-source-experiment limit, conservatively one by default;
* explicit rejected, failed, and cancelled-workflow reconsideration flags;
* completed and inconsistent-workflow exclusion rules; and
* the fixed ranking-global-ordinal, recommendation-ID, ranking-member-ID, then
  canonical-evidence tie rule.

All counts are positive and at most 1000. Unit-interval source metrics and all
other numbers must be finite. The policy canonical text is versioned,
locale-independent, and includes every behavior-affecting field.

Profitability status
--------------------

The current schema has no authoritative durable profitability metric. The
planner never substitutes inference accuracy, leader score, or another proxy.
When a minimum profitability threshold is requested, current repository input
has no profitability value and each otherwise considered candidate receives
``profitability_metric_unavailable``. Durable profitability evidence and its
identity remain a deferred Volume IX capability.

Ordering, selection, and reasons
--------------------------------

Candidates retain the ranking snapshot's ``global_ordinal`` order. A
recommendation ID, ranking member ID, and finally complete canonical evidence
are deterministic fallbacks if malformed evidence duplicates an earlier key.
Filters preserve the original durable positions.

Policy, metric, ranking-bucket, identity, workflow, and candidate-limit rules
are evaluated before group and campaign limits. Group limits are filled by
higher-ranked included candidates. The stored recommendation invocation
canonical is the pre-conversion target identity used to prevent two selected
candidates from describing identical future work; it is not conversion
authorization. Every considered row remains in the plan with exactly one
``include`` or ``exclude`` decision and a deterministic, de-duplicated reason
sequence. A zero-selection plan is successful.

Leader score and inference accuracy are required persisted fields. The source
neutral proportion is nullable in historical storage; a missing value fails
closed with ``predicted_neutral_metric_unavailable`` instead of silently
passing the configured maximum.

Workflow exclusions distinguish pending review, rejection, approval without
execution, paused execution, activation/scheduler ownership, completion,
failure, cancellation, and inconsistent evidence. Reconsideration flags only
relax their named rejected/failed/cancelled exclusions. Inconsistent workflows
are always excluded.

Canonical plan identity
-----------------------

The authoritative campaign-plan canonical text length-frames variable text and
contains the contract version, complete campaign-policy canonical, scope
canonical, ranking-snapshot canonical/hash, every bounded candidate's persisted
evidence and Phase 4C workflow provenance, final decision, and ordered reason
codes. Its hash uses the existing recommendation canonical hash helper and is
only an accelerator.

The display generation timestamp is deliberately excluded. Input row order,
connection information, host, process, and locale do not affect identity.
Changing policy, scope, ranking membership/evidence, workflow evidence,
decision, or reasons changes the canonical plan identity.

CLI
---

Create a read-only plan from an explicit snapshot::

  LSTM_Release --plan-recommendation-campaign \
      --campaign-ranking-snapshot=42 \
      --campaign-limit=5 \
      --campaign-candidate-limit=100 \
      --campaign-min-leader-score=0.60 \
      --campaign-min-inference-accuracy=0.55 \
      --campaign-max-neutral-proportion=0.70

Optional symbol/horizon filters and per-group limits use the documented help
options. Reconsideration flags are explicit and default off.

Output starts with ``RECOMMENDATION_CAMPAIGN_PLAN``, emits one
``RECOMMENDATION_CAMPAIGN_PLAN_CANDIDATE`` per considered row, and ends with
``RECOMMENDATION_CAMPAIGN_PLAN_COMPLETE``. Text uses the existing percent-
escaping convention and missing values are ``NULL``. Every record states
``read_only=true`` plus false proposal/experiment/activation/scheduler/worker
safety fields. Invalid policy, missing schema, and missing/incomplete snapshot
produce stable nonzero outcomes.

Isolated PostgreSQL verification
--------------------------------

Repository tests require an explicit non-production database and reject the
database name ``LSTM``::

  createdb phase4d_campaign_test
  LSTM_TEST_DB_NAME=phase4d_campaign_test \
      /tmp/ExperimentRecommendationCampaignPlanningRepositoryTests
  dropdb phase4d_campaign_test

The test owns and removes an exact per-process schema. It proves row counts,
experiment ``updated_at`` values, and a sentinel sequence remain unchanged.

Deferred work
-------------

Phase 4D Step 2 consumes this plan through a separate deterministic, read-only
review contract. Step 3 may persist one explicit operator decision only after
reconstructing the plan and review from an explicit snapshot and matching the
expected review identity. Plan persistence, proposal creation, bulk review, campaign
execution, automatic activation, background scans, budgets, durable
profitability policy, and scheduler integration remain outside Step 1.

References
----------

* ``docs/Phase4BExperimentRecommendationRanking.rst``
* ``docs/Phase4CExperimentRecommendationConversionWorkflow.rst``
* ``docs/Phase4DExperimentRecommendationCampaignReview.rst``
* ``docs/Phase4DExperimentRecommendationCampaignApproval.rst``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
* ``docs/architecture/Volume_IX_Trading_Profitability.md``
