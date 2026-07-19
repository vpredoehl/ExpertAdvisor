#include "../Sources/ExperimentRecommendationCampaignProposalReview.hpp"

#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationCampaignMaterialization.hpp"

#include <cassert>
#include <stdexcept>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationCampaignProposalReviewRequest Request(
    RecommendationConversionProposalReviewDecision decision =
        RecommendationConversionProposalReviewDecision::approve)
{
    RecommendationCampaignProposalReviewRequest request;
    request.materializationId = 7;
    request.decision = decision;
    request.operatorIdentity = "  operator@example  ";
    request.reasonText = "  reviewed exact campaign  ";
    return request;
}

RecommendationCampaignProposalReviewInput Input(int count = 2)
{
    RecommendationCampaignProposalReviewInput input;
    input.materializationId = 7;
    input.materializationContractVersion =
        kRecommendationCampaignMaterializationContractVersion;
    input.materializationIdentityCanonical = "materialization-canonical";
    input.materializationIdentityHash = RecommendationCanonicalHash(
        input.materializationIdentityCanonical);
    input.selectedMemberCount = count;
    input.materializationComplete = true;
    input.workflowEvidenceConsistent = true;
    for (int ordinal = 1; ordinal <= count; ++ordinal)
    {
        RecommendationCampaignProposalReviewMemberInput member;
        member.memberOrdinal = ordinal;
        member.materializationMemberId = 100 + ordinal;
        member.proposalId = 200 + ordinal;
        input.members.push_back(member);
    }
    return input;
}

PersistedRecommendationConversionProposalReviewDecision Review(
    const RecommendationCampaignProposalReviewPlan& plan,
    int memberIndex,
    RecommendationConversionProposalReviewDecision decision)
{
    PersistedRecommendationConversionProposalReviewDecision review;
    review.reviewDecisionId = 300 + memberIndex;
    review.proposalId = plan.members[memberIndex].proposalId;
    review.decision = decision;
    review.requestId = plan.phase4cRequestId;
    review.operatorIdentity = plan.request.operatorIdentity;
    review.reasonText = plan.request.reasonText;
    review.decidedAt = "2026-07-19";
    review.createdAt = "2026-07-19";
    return review;
}

template <typename Function>
void ExpectInvalid(Function function, const char* expected)
{
    try
    {
        function();
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string{error.what()} == expected);
    }
}

} // namespace

int main()
{
    auto approve = BuildRecommendationCampaignProposalReviewPlan(
        Request(), Input());
    assert(approve.state ==
           RecommendationCampaignProposalReviewPlanState::ready);
    assert(approve.members.size() == 2);
    assert(approve.members[0].memberOrdinal == 1);
    assert(approve.members[1].memberOrdinal == 2);
    assert(approve.request.operatorIdentity == "operator@example");
    assert(approve.request.reasonText == "reviewed exact campaign");
    assert(approve.phase4cRequestId.find("campaign-review-v1-m7-") == 0);

    auto reject = BuildRecommendationCampaignProposalReviewPlan(
        Request(RecommendationConversionProposalReviewDecision::reject),
        Input());
    assert(reject.state == RecommendationCampaignProposalReviewPlanState::ready);
    assert(reject.operationIdentityHash != approve.operationIdentityHash);

    assert(!ParseRecommendationConversionProposalReviewDecision("approved"));
    auto invalidRequest = Request();
    invalidRequest.materializationId = 0;
    ExpectInvalid(
        [&] { (void)NormalizeRecommendationCampaignProposalReviewRequest(
                   invalidRequest); },
        "campaign_proposal_review_materialization_id_invalid");
    invalidRequest = Request();
    invalidRequest.materializationId = -1;
    ExpectInvalid(
        [&] { (void)NormalizeRecommendationCampaignProposalReviewRequest(
                   invalidRequest); },
        "campaign_proposal_review_materialization_id_invalid");
    invalidRequest = Request();
    invalidRequest.decision =
        static_cast<RecommendationConversionProposalReviewDecision>(999);
    ExpectInvalid(
        [&] { (void)NormalizeRecommendationCampaignProposalReviewRequest(
                   invalidRequest); },
        "invalid_recommendation_conversion_proposal_review_decision");
    invalidRequest = Request();
    invalidRequest.operatorIdentity = " \t ";
    ExpectInvalid(
        [&] { (void)NormalizeRecommendationCampaignProposalReviewRequest(
                   invalidRequest); },
        "recommendation_conversion_proposal_review_operator_empty");
    invalidRequest = Request();
    invalidRequest.reasonText = " \n ";
    ExpectInvalid(
        [&] { (void)NormalizeRecommendationCampaignProposalReviewRequest(
                   invalidRequest); },
        "recommendation_conversion_proposal_review_reason_empty");

    const auto deterministic = BuildRecommendationCampaignProposalReviewPlan(
        Request(), Input());
    assert(deterministic.operationIdentityCanonical ==
           approve.operationIdentityCanonical);
    assert(deterministic.operationIdentityHash == approve.operationIdentityHash);
    assert(deterministic.phase4cRequestId == approve.phase4cRequestId);

    auto duplicate = Input();
    duplicate.members[1].proposalId = duplicate.members[0].proposalId;
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), duplicate); },
        "campaign_proposal_review_duplicate_member_or_proposal");
    auto duplicateMember = Input();
    duplicateMember.members[1].materializationMemberId =
        duplicateMember.members[0].materializationMemberId;
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), duplicateMember); },
        "campaign_proposal_review_duplicate_member_or_proposal");
    auto reordered = Input();
    std::swap(reordered.members[0], reordered.members[1]);
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), reordered); },
        "campaign_proposal_review_member_identity_invalid");

    auto satisfiedInput = Input();
    satisfiedInput.members[0].authoritativeReview = Review(
        approve, 0, RecommendationConversionProposalReviewDecision::approve);
    satisfiedInput.members[1].authoritativeReview = Review(
        approve, 1, RecommendationConversionProposalReviewDecision::approve);
    auto satisfied = BuildRecommendationCampaignProposalReviewPlan(
        Request(), satisfiedInput);
    assert(satisfied.state ==
           RecommendationCampaignProposalReviewPlanState::alreadySatisfied);
    assert(satisfied.alreadySatisfiedCount == 2);
    assert((satisfied.diagnosticCodes ==
            std::vector<std::string>{"campaign_review_already_satisfied"}));

    auto opposite = Input();
    opposite.members[0].authoritativeReview = Review(
        approve, 0, RecommendationConversionProposalReviewDecision::reject);
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), opposite); },
        "campaign_proposal_review_opposite_decision_conflict");

    auto unrelatedSameDecision = Input();
    unrelatedSameDecision.members[0].authoritativeReview = Review(
        approve, 0, RecommendationConversionProposalReviewDecision::approve);
    unrelatedSameDecision.members[0].authoritativeReview->requestId =
        "unrelated-manual-review";
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), unrelatedSameDecision); },
        "campaign_proposal_review_existing_decision_conflict");

    auto partial = Input();
    partial.members[0].authoritativeReview = Review(
        approve, 0, RecommendationConversionProposalReviewDecision::approve);
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), partial); },
        "campaign_proposal_review_partial_operation_conflict");

    auto malformed = Input();
    malformed.members[0].authoritativeReview = Review(
        approve, 0, RecommendationConversionProposalReviewDecision::approve);
    malformed.members[0].authoritativeReview->reviewDecisionId = 0;
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), malformed); },
        "campaign_proposal_review_authoritative_review_invalid");

    auto zero = Input(0);
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), zero); },
        "campaign_proposal_review_zero_members");
    auto countMismatch = Input();
    countMismatch.selectedMemberCount = 1;
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), countMismatch); },
        "campaign_proposal_review_member_count_mismatch");
    auto hashMismatch = Input();
    hashMismatch.materializationIdentityHash = "fnv1a64:0000000000000000";
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), hashMismatch); },
        "campaign_proposal_review_materialization_identity_invalid");
    auto versionMismatch = Input();
    versionMismatch.materializationContractVersion = 999;
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), versionMismatch); },
        "campaign_proposal_review_materialization_identity_invalid");
    auto workflowMismatch = Input();
    workflowMismatch.workflowEvidenceConsistent = false;
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), workflowMismatch); },
        "campaign_proposal_review_workflow_evidence_invalid");
    auto overLimit = Input(
        kMaximumRecommendationCampaignProposalReviewMembers + 1);
    ExpectInvalid(
        [&] { (void)BuildRecommendationCampaignProposalReviewPlan(
                   Request(), overLimit); },
        "campaign_proposal_review_member_limit_exceeded");

    return 0;
}
