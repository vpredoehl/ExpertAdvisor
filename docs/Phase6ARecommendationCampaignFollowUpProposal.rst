Phase 6A Recommendation Campaign Follow-Up Proposal
====================================================

Purpose and entity
------------------

Phase 6A defines ``RecommendationCampaignFollowUpProposal``: one pure,
immutable, deterministic advisory candidate for later operator review.  The
name deliberately distinguishes this value from Phase 4D campaign
materialization, which is an explicit persisted operation over an approved
campaign.  A Phase 6A proposal materializes nothing.

Exact immutable inputs
----------------------

The builder accepts exactly one Phase 5A
``RecommendationCampaignOutcomeAssessment`` contract version 2 and one Phase
5C ``RecommendationCampaignOutcomePolicyDecision`` contract version 1.  The
decision must embed a Phase 5C policy contract version 1.  Phase 6A consumes
only their public immutable typed values; it does not consume Step 5B formatted
output, a database row, repository metadata, a scheduler value, a CLI request,
or mutable operator state.

Eligibility prerequisite
------------------------

Construction requires the validated Step 5C decision to report all of:

* ``followUpEligibility == EligibleForOperatorReview``;
* ``followUpAuthorized == false``; and
* ``campaignInterpretation == Favorable``.

Phase 6A does not derive those values, inspect Step 5A metric truth tables, or
recalculate Step 5C evidence sufficiency, direction, interpretation, or
eligibility.  It copies the exact decision classifications after validation.
Eligibility for operator review remains neither approval nor authorization.

Alignment and fail-closed validation
------------------------------------

Before construction, the builder validates supported assessment, policy, and
policy-decision versions and each authoritative-canonical/tagged-hash pair.
The decision's embedded assessment version, canonical text, and hash must
equal the supplied assessment exactly.  Hash equality is never substituted
for canonical-text equality.

Campaign approval ID, campaign canonical identity and hash, complete
materialization identity and approval linkage, member counts, and every
ordered member identity must match exactly.  Member identity includes ordinal,
materialization-member ID, ranking-member ID, recommendation ID, source-
experiment ID, proposal ID, and optional expected-experiment ID.  Mismatch,
malformed identity, unsupported version, non-eligibility, non-favorable
interpretation, or authorization fails closed with a stable explicit reason.
The narrow plain-data validation view exists only for safe malformed-boundary
tests and cannot construct any upstream value, proposal identity, or proposal.
Stable boundary reasons include ``assessment_contract_unsupported``,
``policy_contract_unsupported``, ``policy_decision_contract_unsupported``,
``assessment_identity_mismatch``, ``policy_identity_mismatch``,
``policy_decision_identity_mismatch``, ``campaign_identity_mismatch``,
``materialization_identity_mismatch``, ``member_count_mismatch``,
``member_identity_mismatch``, ``decision_not_eligible_for_operator_review``,
``favorable_campaign_interpretation_required``, and
``follow_up_authorization_must_be_false``.

Immutable output and non-authority
----------------------------------

The output exposes the proposal contract version; exact assessment, policy,
and policy-decision version/canonical/hash triples; complete campaign and
materialization identities; exact ordered members and count; copied Step 5C
evidence sufficiency, campaign interpretation, eligibility, and false
authorization; and the stable provenance reason
``eligible_favorable_policy_decision``.

The type fixes these semantics at compile time and in canonical identity:
``readOnly=true``, ``databaseFree=true``, ``persistent=false``,
``advisory=true``, ``authoritative=false``, ``approved=false``,
``activated=false``, ``executionAuthorized=false``,
``followUpAuthorized=false``, ``schedulerWork=false``, and
``declaresCampaignSuccess=false``.

Canonical identity grammar
--------------------------

Contract version 1 canonical text begins with
``experiment_recommendation_campaign_follow_up_proposal_v1``.  Semicolon-
named fields bind, in order:

* proposal version and every fixed safety semantic;
* assessment, policy, and policy-decision version/canonical/hash triples;
* campaign approval ID/canonical/hash;
* materialization ID, approval linkage, contract version, member count,
  canonical text, and hash;
* each upstream-canonical ordered member identity, including
  ``expected_experiment_id``;
* Step 5C evidence sufficiency, campaign interpretation, follow-up
  eligibility, false decision authorization, and proposal reason.

Every embedded canonical string and other delimiter-capable text is framed as
``BYTE_LENGTH:VALUE``.  Integer and length formatting uses the classic locale.
Canonical text is authoritative; the existing tagged FNV-1a hash is only an
accelerator.  ``observedAt``, clocks, PID, CLI formatting, repository metadata,
and any future approval or activation state are excluded.  Consequently,
observation time cannot change identity, while every behaviorally relevant
upstream identity can.

Canonical-size limit
--------------------

Version 1 permits at most 1,048,576 canonical-text bytes.  The builder checks
the completed canonical byte count before identity and proposal construction
and throws ``proposal_canonical_size_exceeded`` above that inclusive ceiling.
The bound limits a future review/persistence payload deliberately; it neither
adds persistence nor predicts a later database schema.  Any later change to
the ceiling or grammar requires a new contract version or an explicitly
compatible architecture decision.

Non-goals and later Phase 6 boundaries
--------------------------------------

Phase 6A adds no SQL, migration, table, repository, service, formatter, CLI,
preview command, operator-review persistence, approval, rejection, activation,
execution, scheduler work, queue, worker, campaign, recommendation, experiment,
or Phase 4D materialization.  It does not claim profitability, statistical
significance, repeatability, campaign success, or follow-up authorization.

Persistence, explicit operator review, approval/rejection, activation, and
execution remain distinct possible Phase 6B--6F concerns.  This contract does
not authorize, prescribe, or partially implement any of them.
