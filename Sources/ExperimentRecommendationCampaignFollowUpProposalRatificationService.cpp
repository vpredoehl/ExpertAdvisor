#include "ExperimentRecommendationCampaignFollowUpProposalRatificationService.hpp"

#include "ExperimentRecommendationCampaignFollowUpProposalReviewRepository.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace EA::ExperimentRecommendation
{
namespace
{

bool IsValidCanonicalHash(const std::string& value)
{
    constexpr std::string_view prefix = "fnv1a64:";
    return value.size() == prefix.size() + 16U &&
        value.compare(0, prefix.size(), prefix) == 0 &&
        std::all_of(value.begin() +
                static_cast<std::ptrdiff_t>(prefix.size()),
            value.end(),
            [](unsigned char character)
            {
                return (character >= '0' && character <= '9') ||
                    (character >= 'a' && character <= 'f');
            });
}

} // namespace

RecommendationCampaignFollowUpProposalRatificationRequest
ValidateRecommendationCampaignFollowUpProposalRatificationRequest(
    const RecommendationCampaignFollowUpProposalRatificationRequest& request)
{
    if (request.reviewEventId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_proposal_ratification_review_event_id_invalid");
    if (!IsValidCanonicalHash(request.expectedReviewIdentityHash))
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_proposal_ratification_expected_review_hash_invalid");
    ValidateRecommendationCampaignFollowUpProposalRatifierIdentity(
        request.ratifierIdentity);
    ValidateRecommendationCampaignFollowUpProposalRatificationBasis(
        request.ratificationBasis);
    return request;
}

RecommendationCampaignFollowUpProposalRatificationPersistResult
RatifyRecommendationCampaignFollowUpProposal(
    pqxx::connection& connection,
    const RecommendationCampaignFollowUpProposalRatificationRequest& request)
{
    const auto validated =
        ValidateRecommendationCampaignFollowUpProposalRatificationRequest(request);
    pqxx::work transaction{connection};
    if (!RecommendationCampaignFollowUpProposalReviewSchemaExists(transaction) ||
        !RecommendationCampaignFollowUpProposalRatificationSchemaExists(transaction))
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_schemas_required");
    const auto persistedReview =
        FindRecommendationCampaignFollowUpProposalReview(
            transaction, validated.reviewEventId);
    if (!persistedReview)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_review_not_found");
    if (persistedReview->review.identity.hash !=
        validated.expectedReviewIdentityHash)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_review_identity_mismatch");
    if (persistedReview->review.decision !=
        RecommendationCampaignFollowUpProposalReviewDecision::approved)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_review_not_eligible");
    if (persistedReview->review.reviewerIdentity ==
        validated.ratifierIdentity)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_separation_of_duties_violation");
    const auto ratification =
        BuildRecommendationCampaignFollowUpProposalRatification(
            persistedReview->reviewEventId, persistedReview->review,
            validated.ratifierIdentity, validated.ratificationBasis);
    auto result = PersistRecommendationCampaignFollowUpProposalRatification(
        transaction, ratification);
    transaction.commit();
    return result;
}

} // namespace EA::ExperimentRecommendation
