#include "../Sources/ExperimentRecommendationCampaignApproval.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <cassert>
#include <stdexcept>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationCampaignPlanningPolicy Policy()
{
    RecommendationCampaignPlanningPolicy policy;
    policy.enabled = true;
    policy.maximumPerSourceExperiment.reset();
    return policy;
}

RecommendationCampaignCandidateInput Candidate(long long id)
{
    RecommendationCampaignCandidateInput value;
    value.rankingSnapshotId = 7;
    value.rankingSnapshotIdentityCanonical = "approval-snapshot";
    value.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        value.rankingSnapshotIdentityCanonical);
    value.rankingPolicyCanonical = "approval-ranking-policy";
    value.rankingPolicyHash = RecommendationCanonicalHash(
        value.rankingPolicyCanonical);
    value.rankingVersion = 1;
    value.rankingMemberId = 100 + id;
    value.rankingPosition = static_cast<int>(id);
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
        "approval-semantic-" + std::to_string(id);
    value.recommendationSemanticHash = RecommendationCanonicalHash(
        value.recommendationSemanticCanonical);
    value.recommendationInvocationCanonical =
        "approval-invocation-" + std::to_string(id);
    value.recommendationInvocationHash = RecommendationCanonicalHash(
        value.recommendationInvocationCanonical);
    return value;
}

struct PlanAndReview
{
    RecommendationCampaignPlan plan;
    RecommendationCampaignReview review;
};

PlanAndReview Build(
    std::vector<RecommendationCampaignCandidateInput> candidates,
    const std::string& generatedAt = "display-time")
{
    RecommendationCampaignPlanInput input;
    input.policy = Policy();
    input.scope.rankingSnapshotId = 7;
    input.rankingSnapshotIdentityCanonical = "approval-snapshot";
    input.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        input.rankingSnapshotIdentityCanonical);
    input.generatedAt = generatedAt;
    input.candidates = std::move(candidates);
    PlanAndReview result;
    result.plan = PlanRecommendationCampaign(input);
    result.review = ReviewRecommendationCampaignPlan(result.plan);
    return result;
}

RecommendationCampaignApprovalRequest Request(
    const RecommendationCampaignReview& review,
    RecommendationCampaignApprovalDecision decision =
        RecommendationCampaignApprovalDecision::approved)
{
    RecommendationCampaignApprovalRequest request;
    request.decision = decision;
    request.expectedCampaignReviewIdentityHash = review.identityHash;
    request.reviewerIdentity = "operator-1";
    request.reasonText = "Explicit campaign decision.";
    return request;
}

RecommendationCampaignApprovalEvidence Evidence(
    const PlanAndReview& value,
    RecommendationCampaignApprovalRequest request)
{
    return BuildRecommendationCampaignApprovalEvidence(
        7,
        "approval-snapshot",
        RecommendationCanonicalHash("approval-snapshot"),
        value.plan,
        value.review,
        request);
}

} // namespace

int main()
{
    assert(ParseRecommendationCampaignApprovalDecision("approved") ==
           RecommendationCampaignApprovalDecision::approved);
    assert(ParseRecommendationCampaignApprovalDecision("rejected") ==
           RecommendationCampaignApprovalDecision::rejected);
    assert(!ParseRecommendationCampaignApprovalDecision("approve"));

    const PlanAndReview campaign = Build({Candidate(1)});
    auto request = Request(campaign.review);
    const auto approval = Evidence(campaign, request);
    ValidateRecommendationCampaignApprovalEvidence(approval);
    assert(approval.summary.selectedCount == 1);
    assert(approval.decision == RecommendationCampaignApprovalDecision::approved);
    assert(approval.approvalIdentityHash == RecommendationCanonicalHash(
        approval.approvalIdentityCanonical));

    const auto repeated = Evidence(campaign, request);
    assert(approval.approvalIdentityCanonical ==
           repeated.approvalIdentityCanonical);
    assert(approval.approvalIdentityHash == repeated.approvalIdentityHash);
    const auto displayTimeChanged = Build({Candidate(1)}, "another-time");
    assert(approval.approvalIdentityHash ==
           Evidence(displayTimeChanged, Request(displayTimeChanged.review))
               .approvalIdentityHash);

    auto changed = request;
    changed.decision = RecommendationCampaignApprovalDecision::rejected;
    assert(Evidence(campaign, changed).approvalIdentityHash !=
           approval.approvalIdentityHash);
    changed = request;
    changed.reviewerIdentity = "operator-2";
    assert(Evidence(campaign, changed).approvalIdentityHash !=
           approval.approvalIdentityHash);
    changed = request;
    changed.reasonText = "Different deliberate reason.";
    assert(Evidence(campaign, changed).approvalIdentityHash !=
           approval.approvalIdentityHash);

    auto normalized = request;
    normalized.reviewerIdentity = "  operator-1\t";
    normalized.reasonText = "\nExplicit campaign decision.  ";
    assert(Evidence(campaign, normalized).approvalIdentityHash ==
           approval.approvalIdentityHash);

    auto stale = request;
    stale.expectedCampaignReviewIdentityHash = "fnv1a64:0000000000000000";
    try
    {
        (void)Evidence(campaign, stale);
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_approval_review_identity_mismatch");
    }

    auto malformedHash = request;
    malformedHash.expectedCampaignReviewIdentityHash = "not-a-hash";
    try
    {
        (void)Evidence(campaign, malformedHash);
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_approval_expected_review_hash_invalid");
    }

    auto badReviewer = request;
    badReviewer.reviewerIdentity = "operator\nname";
    try
    {
        (void)Evidence(campaign, badReviewer);
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_approval_reviewer_invalid");
    }
    auto nulReviewer = request;
    nulReviewer.reviewerIdentity = std::string{"operator\0name", 13};
    try
    {
        (void)Evidence(campaign, nulReviewer);
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_approval_reviewer_invalid");
    }
    auto nulReason = request;
    nulReason.reasonText = std::string{"reason\0text", 11};
    try
    {
        (void)Evidence(campaign, nulReason);
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_approval_reason_invalid");
    }
    auto emptyReason = request;
    emptyReason.reasonText = " \t ";
    try
    {
        (void)Evidence(campaign, emptyReason);
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_approval_reason_invalid");
    }
    auto oversizedReason = request;
    oversizedReason.reasonText.assign(
        kRecommendationCampaignApprovalReasonMaximum + 1, 'x');
    try
    {
        (void)Evidence(campaign, oversizedReason);
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_approval_reason_invalid");
    }

    const PlanAndReview zero = Build({});
    try
    {
        (void)Evidence(zero, Request(zero.review));
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_approval_zero_selection_forbidden");
    }
    const auto zeroRejected = Evidence(
        zero,
        Request(zero.review, RecommendationCampaignApprovalDecision::rejected));
    assert(zeroRejected.summary.selectedCount == 0);

    auto tamperedReview = campaign;
    tamperedReview.review.summary.duplicateGroupCount = 1;
    try
    {
        (void)Evidence(tamperedReview, Request(tamperedReview.review));
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_approval_review_reconstruction_mismatch");
    }

    return 0;
}
