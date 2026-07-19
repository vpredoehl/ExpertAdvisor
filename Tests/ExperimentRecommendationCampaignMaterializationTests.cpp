#include "../Sources/ExperimentRecommendationCampaignMaterialization.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <cassert>
#include <stdexcept>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationCampaignCandidateInput Candidate(long long id, int position)
{
    RecommendationCampaignCandidateInput value;
    value.rankingSnapshotId = 7;
    value.rankingSnapshotIdentityCanonical = "materialization-snapshot";
    value.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        value.rankingSnapshotIdentityCanonical);
    value.rankingPolicyCanonical = "materialization-ranking-policy";
    value.rankingPolicyHash = RecommendationCanonicalHash(
        value.rankingPolicyCanonical);
    value.rankingVersion = 1;
    value.rankingMemberId = 100 + id;
    value.rankingPosition = position;
    value.rankingBucket = "advisory_ready";
    value.rankingScore = 0.8;
    value.recommendationId = id;
    value.sourceExperimentId = 500 + id;
    value.symbol = "eurusd";
    value.predictionHorizon = 12;
    value.family = "core_lr_mult";
    value.targetEpochs = 120;
    value.leaderScore = 0.75;
    value.inferenceAccuracy = 0.70;
    value.predictedNeutralProportion = 0.30;
    value.recommendationSemanticCanonical =
        "materialization-semantic-" + std::to_string(id);
    value.recommendationSemanticHash = RecommendationCanonicalHash(
        value.recommendationSemanticCanonical);
    value.recommendationInvocationCanonical =
        "materialization-invocation-" + std::to_string(id);
    value.recommendationInvocationHash = RecommendationCanonicalHash(
        value.recommendationInvocationCanonical);
    return value;
}

RecommendationCampaignPlanningPolicy Policy()
{
    RecommendationCampaignPlanningPolicy value;
    value.enabled = true;
    value.maximumPerSourceExperiment.reset();
    return value;
}

struct Fixture
{
    RecommendationCampaignPlan plan;
    RecommendationCampaignApprovalEvidence approval;
};

Fixture BuildFixture()
{
    RecommendationCampaignPlanInput input;
    input.policy = Policy();
    input.scope.rankingSnapshotId = 7;
    input.rankingSnapshotIdentityCanonical = "materialization-snapshot";
    input.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        input.rankingSnapshotIdentityCanonical);
    input.generatedAt = "display-only";
    input.candidates = {Candidate(1, 1), Candidate(2, 2)};
    Fixture fixture;
    fixture.plan = PlanRecommendationCampaign(input);
    const auto review = ReviewRecommendationCampaignPlan(fixture.plan);
    RecommendationCampaignApprovalRequest request;
    request.decision = RecommendationCampaignApprovalDecision::approved;
    request.expectedCampaignReviewIdentityHash = review.identityHash;
    request.reviewerIdentity = "reviewer";
    request.reasonText = "Approved bounded campaign.";
    fixture.approval = BuildRecommendationCampaignApprovalEvidence(
        7, input.rankingSnapshotIdentityCanonical,
        input.rankingSnapshotIdentityHash, fixture.plan, review, request);
    return fixture;
}

ProposedExperimentSpecification Proposal(long long id)
{
    ProposedExperimentSpecification value;
    value.recommendationId = id;
    value.sourceExperimentId = 500 + id;
    value.conversionIdentityCanonical =
        "conversion-proposal-" + std::to_string(id);
    value.conversionIdentityHash = RecommendationCanonicalHash(
        value.conversionIdentityCanonical);
    return value;
}

} // namespace

int main()
{
    const auto policy = Policy();
    const auto policyCanonical =
        RecommendationCampaignPlanningPolicyCanonicalText(policy);
    assert(RecommendationCampaignPlanningPolicyCanonicalText(
               ParseRecommendationCampaignPlanningPolicyCanonicalText(
                   policyCanonical)) == policyCanonical);
    RecommendationCampaignPlanningScope scope;
    scope.rankingSnapshotId = 7;
    scope.symbol = "eurusd";
    scope.horizon = 12;
    const auto scopeCanonical =
        RecommendationCampaignPlanningScopeCanonicalText(scope);
    assert(RecommendationCampaignPlanningScopeCanonicalText(
               ParseRecommendationCampaignPlanningScopeCanonicalText(
                   scopeCanonical)) == scopeCanonical);

    const Fixture fixture = BuildFixture();
    RecommendationCampaignMaterializationRequest request;
    request.campaignApprovalId = 42;
    request.operatorIdentity = " operator@example ";
    request.reasonText = " Materialize exact proposal set. ";
    const std::vector proposals{Proposal(1), Proposal(2)};
    const auto evidence = BuildRecommendationCampaignMaterializationEvidence(
        request, fixture.approval, fixture.plan, proposals);
    ValidateRecommendationCampaignMaterializationEvidence(evidence);
    assert(evidence.operatorIdentity == "operator@example");
    assert(evidence.selectedMemberCount == 2);
    assert(evidence.members[0].memberOrdinal == 1);
    assert(evidence.members[1].memberOrdinal == 2);
    assert(evidence.materializationIdentityHash == RecommendationCanonicalHash(
        evidence.materializationIdentityCanonical));

    const auto repeated = BuildRecommendationCampaignMaterializationEvidence(
        request, fixture.approval, fixture.plan, proposals);
    assert(repeated.materializationIdentityCanonical ==
           evidence.materializationIdentityCanonical);
    assert(repeated.materializationIdentityHash ==
           evidence.materializationIdentityHash);

    auto changed = request;
    changed.operatorIdentity = "another-operator";
    assert(BuildRecommendationCampaignMaterializationEvidence(
               changed, fixture.approval, fixture.plan, proposals)
               .materializationIdentityHash != evidence.materializationIdentityHash);
    changed = request;
    changed.reasonText = "Different reason.";
    assert(BuildRecommendationCampaignMaterializationEvidence(
               changed, fixture.approval, fixture.plan, proposals)
               .materializationIdentityHash != evidence.materializationIdentityHash);
    RecommendationCampaignApprovalRequest changedApprovalRequest;
    changedApprovalRequest.decision =
        RecommendationCampaignApprovalDecision::approved;
    changedApprovalRequest.expectedCampaignReviewIdentityHash =
        fixture.approval.campaignReviewIdentityHash;
    changedApprovalRequest.reviewerIdentity = "another-reviewer";
    changedApprovalRequest.reasonText = "Approved bounded campaign.";
    const auto changedApproval = BuildRecommendationCampaignApprovalEvidence(
        fixture.approval.rankingSnapshotId,
        fixture.approval.rankingSnapshotIdentityCanonical,
        fixture.approval.rankingSnapshotIdentityHash, fixture.plan,
        ReviewRecommendationCampaignPlan(fixture.plan),
        changedApprovalRequest);
    assert(BuildRecommendationCampaignMaterializationEvidence(
               request, changedApproval, fixture.plan, proposals)
               .materializationIdentityHash != evidence.materializationIdentityHash);
    auto reordered = proposals;
    std::swap(reordered[0], reordered[1]);
    try
    {
        (void)BuildRecommendationCampaignMaterializationEvidence(
            request, fixture.approval, fixture.plan, reordered);
        assert(false);
    }
    catch (const std::invalid_argument&) {}

    auto duplicate = proposals;
    duplicate[1] = duplicate[0];
    duplicate[1].recommendationId = 2;
    duplicate[1].sourceExperimentId = 502;
    try
    {
        (void)BuildRecommendationCampaignMaterializationEvidence(
            request, fixture.approval, fixture.plan, duplicate);
        assert(false);
    }
    catch (const std::invalid_argument&) {}

    RecommendationCampaignApprovalRequest rejectedApprovalRequest;
    rejectedApprovalRequest.decision =
        RecommendationCampaignApprovalDecision::rejected;
    rejectedApprovalRequest.expectedCampaignReviewIdentityHash =
        fixture.approval.campaignReviewIdentityHash;
    rejectedApprovalRequest.reviewerIdentity = "reviewer";
    rejectedApprovalRequest.reasonText = "Rejected bounded campaign.";
    const auto rejectedApproval = BuildRecommendationCampaignApprovalEvidence(
        fixture.approval.rankingSnapshotId,
        fixture.approval.rankingSnapshotIdentityCanonical,
        fixture.approval.rankingSnapshotIdentityHash, fixture.plan,
        ReviewRecommendationCampaignPlan(fixture.plan),
        rejectedApprovalRequest);
    try
    {
        (void)BuildRecommendationCampaignMaterializationEvidence(
            request, rejectedApproval, fixture.plan, proposals);
        assert(false);
    }
    catch (const std::invalid_argument&) {}

    auto malformedEvidence = evidence;
    malformedEvidence.members[0].selectedMemberIdentityHash =
        RecommendationCanonicalHash("different-member");
    try
    {
        ValidateRecommendationCampaignMaterializationEvidence(
            malformedEvidence);
        assert(false);
    }
    catch (const std::invalid_argument&) {}

    auto nul = request;
    nul.operatorIdentity = std::string{"operator\0x", 10};
    try
    {
        (void)NormalizeRecommendationCampaignMaterializationRequest(nul);
        assert(false);
    }
    catch (const std::invalid_argument&) {}
    auto oversized = request;
    oversized.reasonText.assign(
        kRecommendationCampaignMaterializationReasonMaximum + 1, 'x');
    try
    {
        (void)NormalizeRecommendationCampaignMaterializationRequest(oversized);
        assert(false);
    }
    catch (const std::invalid_argument&) {}

    return 0;
}
