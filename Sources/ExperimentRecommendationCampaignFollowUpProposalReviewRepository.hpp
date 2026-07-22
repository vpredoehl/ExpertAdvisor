#pragma once

#include "ExperimentRecommendationCampaignFollowUpProposalReview.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

inline constexpr int
    kMaximumRecommendationCampaignFollowUpProposalReviewListLimit = 1000;

struct PersistedRecommendationCampaignFollowUpProposalReview
{
    const long long reviewEventId;
    const RecommendationCampaignFollowUpProposalReview review;
    const std::string createdAt;

    PersistedRecommendationCampaignFollowUpProposalReview(
        long long reviewEventId,
        RecommendationCampaignFollowUpProposalReview review,
        std::string createdAt);
    PersistedRecommendationCampaignFollowUpProposalReview(
        const PersistedRecommendationCampaignFollowUpProposalReview&) =
        default;
    PersistedRecommendationCampaignFollowUpProposalReview(
        PersistedRecommendationCampaignFollowUpProposalReview&&) = default;
    bool operator==(
        const PersistedRecommendationCampaignFollowUpProposalReview&) const =
        default;
};

enum class RecommendationCampaignFollowUpProposalReviewPersistOutcome
{
    recorded,
    existingIdentical
};

std::string RecommendationCampaignFollowUpProposalReviewPersistOutcomeText(
    RecommendationCampaignFollowUpProposalReviewPersistOutcome outcome);

struct RecommendationCampaignFollowUpProposalReviewPersistResult
{
    const RecommendationCampaignFollowUpProposalReviewPersistOutcome outcome;
    const PersistedRecommendationCampaignFollowUpProposalReview persisted;

    RecommendationCampaignFollowUpProposalReviewPersistResult(
        RecommendationCampaignFollowUpProposalReviewPersistOutcome outcome,
        PersistedRecommendationCampaignFollowUpProposalReview persisted);
};

bool RecommendationCampaignFollowUpProposalReviewSchemaExists(
    pqxx::connection& connection);
bool RecommendationCampaignFollowUpProposalReviewSchemaExists(
    pqxx::transaction_base& transaction);

RecommendationCampaignFollowUpProposalReviewPersistResult
PersistRecommendationCampaignFollowUpProposalReview(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignFollowUpProposalReview& review);

std::optional<PersistedRecommendationCampaignFollowUpProposalReview>
FindRecommendationCampaignFollowUpProposalReview(
    pqxx::connection& connection,
    long long reviewEventId);
std::optional<PersistedRecommendationCampaignFollowUpProposalReview>
FindRecommendationCampaignFollowUpProposalReview(
    pqxx::transaction_base& transaction,
    long long reviewEventId);

std::optional<PersistedRecommendationCampaignFollowUpProposalReview>
FindRecommendationCampaignFollowUpProposalReviewByProposalId(
    pqxx::connection& connection,
    long long followUpProposalId);
std::optional<PersistedRecommendationCampaignFollowUpProposalReview>
FindRecommendationCampaignFollowUpProposalReviewByProposalId(
    pqxx::transaction_base& transaction,
    long long followUpProposalId);

std::vector<PersistedRecommendationCampaignFollowUpProposalReview>
ListRecommendationCampaignFollowUpProposalReviews(
    pqxx::connection& connection,
    int limit);
std::vector<PersistedRecommendationCampaignFollowUpProposalReview>
ListRecommendationCampaignFollowUpProposalReviews(
    pqxx::transaction_base& transaction,
    int limit);

} // namespace EA::ExperimentRecommendation
