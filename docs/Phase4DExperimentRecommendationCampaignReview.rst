Phase 4D Step 2 Read-Only Recommendation Campaign Review
=========================================================

Purpose and boundary
--------------------

Step 2 reviews the deterministic in-memory campaign plan produced by Step 1.
It does not create or persist a campaign, proposal, review decision, execution,
activation, experiment, scheduler action, or worker. It adds no migration and
uses the same bounded read transaction and authoritative Phase 4B/4C evidence
as the planner.

Review contract
---------------

Review contract version 1 validates the campaign plan before presentation. A
review is rejected if the plan canonical/hash pair, policy canonical/hash,
scope canonical, summary counts, candidate shapes, or deterministic candidate
order is inconsistent. Successful review therefore reports
``deterministic_ordering_verified=true``.

The structured result contains:

* the plan and review canonical identities and hashes;
* selected candidates in campaign order;
* excluded candidates in campaign order with the planner's ordered reason
  codes unchanged;
* exact duplicate groups based on recommendation invocation canonical text;
* deterministic parameter-family, symbol, and horizon coverage; and
* considered, selected, excluded, duplicate, and diversity counts.

Duplicate grouping compares authoritative canonical text, not hashes. The
reported invocation-identity hash is a display accelerator. All coverage rows
are sorted by bytewise ``std::string`` order for family/symbol and numeric order
for horizon. Selected and excluded rows retain their original campaign
ordinals.

Canonical review identity
-------------------------

The versioned review canonical text length-frames variable text and binds the
complete authoritative plan identity, policy hash, scope, validated summary,
selected and excluded review rows, ordered reasons, exact duplicate grouping,
and coverage rows. The display generation timestamp is excluded. Identical
plans therefore produce identical review identities regardless of input
arrival order, locale, process, host, or observation time.

CLI
---

Review one explicit ranking snapshot under the same Step 1 policy options::

  LSTM_Release --review-recommendation-campaign \
      --campaign-ranking-snapshot=42 \
      --campaign-limit=5 \
      --campaign-min-leader-score=0.60

Output begins with ``RECOMMENDATION_CAMPAIGN_REVIEW``, emits selected and
excluded ``RECOMMENDATION_CAMPAIGN_REVIEW_CANDIDATE`` rows, optional duplicate
rows, family/symbol/horizon coverage rows, and a final
``RECOMMENDATION_CAMPAIGN_REVIEW_COMPLETE`` record. All events carry explicit
read-only safety fields. A valid review with zero selected candidates succeeds.

Database safety and deferred work
---------------------------------

The service calls the existing Step 1 repository loader and planner. It adds no
SQL and no database privileges. Repository verification uses only an explicit
non-``LSTM`` disposable database and proves proposal/audit/experiment row
counts, experiment ``updated_at``, and a sentinel sequence do not change.

Campaign persistence, campaign approval, proposal creation, bulk execution,
automatic activation, background scans, and scheduler integration remain
deferred.

References
----------

* ``docs/Phase4DExperimentRecommendationCampaignPlanning.rst``
* ``docs/Phase4CExperimentRecommendationConversionWorkflow.rst``
* ``docs/architecture/Volume_VIII_Recommendation_Engine.md``
