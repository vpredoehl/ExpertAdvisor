#include "ExperimentRecommendationCampaignFollowUpProposalReviewPresentation.hpp"

#include <iomanip>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string ReviewMachineText(const std::string& value)
{
    std::ostringstream escaped;
    escaped.imbue(std::locale::classic());
    escaped << std::uppercase << std::hex;
    for (const unsigned char character : value)
    {
        const bool alphanumeric =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9');
        const bool safe = alphanumeric || character == '-' ||
            character == '_' || character == '.' || character == ':' ||
            character == '/' || character == ';';
        if (safe) escaped << static_cast<char>(character);
        else
            escaped << '%' << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(character);
    }
    return escaped.str() == "NULL" ? "%4E%55%4C%4C" : escaped.str();
}

void PrintSafety(std::ostream& output)
{
    output << "read_only=true,persisted=true,administrative_review=true,"
              "activated=false,execution_authorized=false,"
              "follow_up_authorized=false,queued=false,scheduled=false,"
              "scheduler_started=false,scheduler_signaled=false,"
              "workers_started=false,experiments_created=false,"
              "experiments_modified=false,campaign_success_declared=false";
}

void UseReadOnlyRepeatableRead(pqxx::read_transaction& transaction)
{
    transaction.exec(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
}

void PrintFailure(
    std::ostream& errors,
    const std::string& recordType,
    const std::string& reason)
{
    errors << recordType << ",reason="
           << ReviewMachineText(reason) << ',';
    PrintSafety(errors);
    errors << '\n';
}

} // namespace

void WriteRecommendationCampaignFollowUpProposalReview(
    std::ostream& output,
    const PersistedRecommendationCampaignFollowUpProposalReview& persisted)
{
    const auto& review = persisted.review;
    output
        << "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_REVIEW"
        << ",review_event_id=" << std::to_string(persisted.reviewEventId)
        << ",follow_up_proposal_id="
        << std::to_string(review.followUpProposalId)
        << ",proposal_identity_hash="
        << ReviewMachineText(review.proposalIdentityHash)
        << ",review_contract_version="
        << std::to_string(review.identity.contractVersion)
        << ",review_identity_hash="
        << ReviewMachineText(review.identity.hash)
        << ",decision="
        << RecommendationCampaignFollowUpProposalReviewDecisionText(
               review.decision)
        << ",reviewer="
        << ReviewMachineText(review.reviewerIdentity)
        << ",reason=" << ReviewMachineText(review.reasonText)
        << ",created_at="
        << ReviewMachineText(persisted.createdAt) << ',';
    PrintSafety(output);
    output << '\n';
}

int RunShowRecommendationCampaignFollowUpProposalReview(
    const std::string& connectionString,
    long long reviewEventId,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        UseReadOnlyRepeatableRead(transaction);
        if (!RecommendationCampaignFollowUpProposalReviewSchemaExists(
                transaction))
            throw std::runtime_error(
                "recommendation_campaign_follow_up_proposal_review_schema_required");
        const auto persisted =
            FindRecommendationCampaignFollowUpProposalReview(
                transaction, reviewEventId);
        if (!persisted)
        {
            errors << "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_REVIEW_NOT_FOUND"
                   << ",review_event_id=" << std::to_string(reviewEventId)
                   << ',';
            PrintSafety(errors);
            errors << '\n';
            return 1;
        }
        WriteRecommendationCampaignFollowUpProposalReview(
            output, *persisted);
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        PrintFailure(errors,
            "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_REVIEW_INVALID",
            error.what());
        return 1;
    }
    catch (const std::exception& error)
    {
        PrintFailure(errors,
            "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_REVIEW_FAILED",
            error.what());
        return 2;
    }
}

int RunListRecommendationCampaignFollowUpProposalReviews(
    const std::string& connectionString,
    int limit,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        UseReadOnlyRepeatableRead(transaction);
        if (!RecommendationCampaignFollowUpProposalReviewSchemaExists(
                transaction))
            throw std::runtime_error(
                "recommendation_campaign_follow_up_proposal_review_schema_required");
        const auto reviews =
            ListRecommendationCampaignFollowUpProposalReviews(
                transaction, limit);
        for (const auto& review : reviews)
            WriteRecommendationCampaignFollowUpProposalReview(
                output, review);
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        PrintFailure(errors,
            "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_REVIEW_LIST_INVALID",
            error.what());
        return 1;
    }
    catch (const std::exception& error)
    {
        PrintFailure(errors,
            "RECOMMENDATION_CAMPAIGN_FOLLOW_UP_PROPOSAL_REVIEW_LIST_FAILED",
            error.what());
        return 2;
    }
}

} // namespace EA::ExperimentRecommendation
