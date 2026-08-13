---
title: "Campaign Operations Phase H H2 Fixture Completion"
document_type: "verification report"
status: "incomplete"
---

# Campaign Operations Phase H / H2 — Fixture Completion

## Scope and result

The disposable H2 fixture was corrected far enough to exercise the real
production Phase E handoff with authoritative conversion workflow evidence.
The first complete attempt exposed a deterministic production-contract
contradiction after Phase E execution and activation. No workflow or migration
redesign was made.

## Exact fixture deficiency and correction

The former H2 setup linked a materialization member to a proposal identity that
did not exist in `experiment_recommendation_conversion_proposal`. It therefore
had no approved conversion review and the common downstream classifier
correctly returned `reconciliation_required` before launch.

The new `phase-e-fixture-setup` mode in
`Tests/CampaignOperationsPhaseH1WorkflowLockTests.cpp` adds, for disposable
campaigns 71 and 72:

1. a source experiment, analysis provenance, and approved recommendation;
2. a conversion proposal created through `PersistRecommendationConversionProposal`;
3. an approved conversion review created through
   `PersistRecommendationConversionProposalReviewDecision`;
4. materialization/member linkage using the persisted proposal identity; and
5. explicit pre-dispatch assertions for proposal, approved review,
   materialization-member identity, zero execution rows, and zero activation
   rows.

Observed fixture evidence:

`H2_PHASE_E_FIXTURE_ASSERTIONS campaign_id=71 proposal_id=1 review_id=1 proposal=1 approved_review=1 execution=0 activation=0 materialization_member=1`

The same assertion passed for campaign 72 with proposal/review IDs 2.

## Canonical production attempt

The disposable H2 script continued through:

- migration 056 apply and replay;
- H2 manifest and deployment/privilege checks;
- production enable with fresh-connection uncertain-commit recovery;
- real production acquisition into the common Phase E engine;
- conversion execution and activation from the corrected fixture.

The attempt then failed in the common production handoff finalization with:

`campaign operations bind compare-and-set lost`

The failure is not a fixture-state classification. The production acquisition
function in migration 055 sets
`production_dispatch_enabled = true` while moving the request to
`dispatching`. The authoritative `transition_campaign_operations_request_bound`
predicate in migration 048 requires
`production_dispatch_enabled = false`. The finalization update therefore
cannot match the request after a real production acquisition.

Evidence locations:

- `Database/migrations/055_campaign_operations_production_admission_foundation.sql:3776-3786`
- `Database/migrations/048_campaign_operations_durable_dispatch_handoff.sql:519-532`
- `Tests/CampaignOperationsPhaseH2WorkflowTests.sh`

This is a reproducible production contradiction exposed only after the
downstream fixture was completed. Per the requested scope discipline, H2
completion stops here before changing the production contract.

## Not executed after the blocker

The following acceptance evidence remains blocked and was not relabeled as
passing:

- completed handoff and durable canonical identities;
- production disable after completed handoff;
- exact dispatch replay and conflicting-key behavior after completion;
- handoff uncertain-commit recovery;
- complete duplicate-prevention count/identity matrix;
- required disable/acquisition/handoff concurrency matrix with
  `pg_blocking_pids()`;
- post-correction H1 regression rerun and final evidence graph regeneration.

Migration replay/checksum and H2 privilege/manifest evidence remained passing
in the attempted run. The prior clean-baseline H1REG027 result remains
`PRE_EXISTING` per established evidence; this correction introduced no basis
to relabel it.

## Smallest next corrective action

Resolve the production state-contract mismatch for the H2 handoff transition
under the established ADR-0019C privilege boundary, then rerun the disposable
H2 workflow from a clean fixture and complete the remaining replay, recovery,
disable, duplicate, concurrency, regression, and evidence checks.

H2_COMPLETION_INCOMPLETE
