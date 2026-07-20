Phase 4D Step 5 — Campaign Handoff Status
===========================================

Purpose and boundary
--------------------

Step 5 projects one immutable Phase 4D campaign materialization onto the
current Phase 4C proposal lifecycle. The materialization manifest and its
one-based ordered member links remain the sole authority for campaign
membership. Current proposal review, execution, activation, and experiment
lifecycle rows are observed; they are never changed or repaired.

The projection is not persisted authority and is not authorization. In
particular, ``ready_for_phase4c_execution`` and
``ready_for_phase4c_activation`` mean only that the observed durable
prerequisites appear satisfied. Existing explicit Phase 4C commands remain
responsible for review, execution, and activation.

State and precedence
--------------------

Every materialization member is reported in its stored ordinal order with its
linked proposal, greatest-ID review decision, execution, activation, existing
Phase 4C workflow state, and deterministic integrity diagnostics. Aggregate
state uses this precedence:

1. any integrity diagnostic: ``inconsistent``;
2. any latest rejection: ``review_rejected``;
3. no approvals: ``awaiting_phase4c_review``;
4. mixed approvals and unreviewed members: ``partially_reviewed``;
5. all approved with no executions: ``ready_for_phase4c_execution``;
6. some executions: ``partially_executed``;
7. all executed with no activations: ``ready_for_phase4c_activation``;
8. some activations: ``partially_activated``;
9. all activated: ``fully_activated``.

Review timestamps do not select the current decision. Consistent with Phase
4C Step 3, the greatest review-decision primary key is authoritative.
``awaiting_review`` and ``no_authoritative_review`` count members for which no
valid current review can be established, including a member whose linked
proposal is missing. Such malformed members also carry an integrity diagnostic,
so these counts cannot cause an aggregate awaiting/ready state.

Integrity diagnostics
---------------------

Malformed immutable materialization evidence—unsupported contract version,
member-count mismatch, non-contiguous or reordered ordinals, invalid canonical
hashes, or duplicate immutable links—causes detailed inspection to fail
closed. Downstream inconsistencies remain observable with stable codes. These
include a missing linked proposal; proposal ID, recommendation, source
experiment, canonical identity, or hash mismatch; invalid latest review;
execution without valid approval; duplicate execution or activation rows;
activation without execution; execution/activation provenance or identity
mismatch; missing linked experiment; and incompatible experiment lifecycle.
No diagnostic path mutates data.

Read consistency and queries
----------------------------

Each command uses one PostgreSQL ``read_transaction`` set to
``REPEATABLE READ`` before its first query, so all manifest, member, and
workflow reads use one consistent snapshot. Show uses the existing validated
Step 4 mapping for one manifest and its members, then one bounded Phase 4C
workflow query for its exact linked proposal IDs. List first loads its bounded
manifest set, then performs one Step 4 member query and one workflow query per
returned materialization. The default list limit is 100 and the enforced
maximum is 1000. This avoids per-member workflow queries and preserves the
authoritative member order. No migration, view, lock, sequence access, or new
privilege is added.

CLI
---

::

   LSTM_Release --show-recommendation-campaign-handoff=ID
   LSTM_Release --list-recommendation-campaign-handoffs \
     --campaign-handoff-limit=100

The show command emits one ``RECOMMENDATION_CAMPAIGN_HANDOFF`` summary followed
by one ``RECOMMENDATION_CAMPAIGN_HANDOFF_MEMBER`` row per stored member. The
bounded list emits deterministic materialization-ID order and a final count.
Output states ``read_only=true`` and explicitly reports that no
materialization, proposal, review, execution, activation, experiment,
scheduler, or worker was created or started. Summary, list-completion, and
error rows carry the complete safety field set; member rows repeat
``read_only=true`` and are contextual children of the show summary.

Verification and deferred work
------------------------------

Use an explicit disposable non-``LSTM`` database::

   createdb phase4d_campaign_test 2>/dev/null || true
   LSTM_TEST_DB_NAME=phase4d_campaign_test \
     /tmp/ExperimentRecommendationCampaignHandoffRepositoryTests

Step 6 adds a separate explicit manual operation that can append the same
approve/reject decision to every exact materialized proposal through ordinary
Phase 4C review rows. This projection observes those rows immediately; it does
not invoke Step 6. Phase 5 Step 1 adds a separately confirmed atomic convenience
over the existing Phase 4C paused execution for the exact membership; the
handoff observes those rows and experiments but does not invoke Phase 5.
Phase 5 Step 2 adds a separately confirmed atomic convenience over ordinary
Phase 4C activation for the same exact membership; the handoff observes its
activation rows and lifecycle states but does not invoke it. Phase 5 Step 3
adds a distinct, separately confirmed atomic convenience that can establish
both ordinary Phase 4C executions and activations for the exact membership;
the handoff observes those durable results but never invokes it. Automatic
progression, repair, background scans, and scheduler integration remain
separate. The scheduler does not query the handoff projection or any Phase 4D
audit table.
