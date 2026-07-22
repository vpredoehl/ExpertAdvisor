#pragma once

#include "ExperimentRecommendationCampaignFollowUpProposalRatification.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

inline constexpr int
    kMaximumRecommendationCampaignFollowUpProposalRatificationListLimit = 1000;

struct PersistedRecommendationCampaignFollowUpProposalRatification
{
    const long long ratificationEventId;
    const RecommendationCampaignFollowUpProposalRatification ratification;
    const std::string createdAt;

    PersistedRecommendationCampaignFollowUpProposalRatification(
        long long ratificationEventId,
        RecommendationCampaignFollowUpProposalRatification ratification,
        std::string createdAt);
    PersistedRecommendationCampaignFollowUpProposalRatification(
        const PersistedRecommendationCampaignFollowUpProposalRatification&) =
        default;
    PersistedRecommendationCampaignFollowUpProposalRatification(
        PersistedRecommendationCampaignFollowUpProposalRatification&&) = default;
    bool operator==(
        const PersistedRecommendationCampaignFollowUpProposalRatification&) const =
        default;
};

enum class RecommendationCampaignFollowUpProposalRatificationPersistOutcome
{
    recorded,
    existingIdentical
};

std::string RecommendationCampaignFollowUpProposalRatificationPersistOutcomeText(
    RecommendationCampaignFollowUpProposalRatificationPersistOutcome outcome);

struct RecommendationCampaignFollowUpProposalRatificationPersistResult
{
    const RecommendationCampaignFollowUpProposalRatificationPersistOutcome outcome;
    const PersistedRecommendationCampaignFollowUpProposalRatification persisted;

    RecommendationCampaignFollowUpProposalRatificationPersistResult(
        RecommendationCampaignFollowUpProposalRatificationPersistOutcome outcome,
        PersistedRecommendationCampaignFollowUpProposalRatification persisted);
};

bool RecommendationCampaignFollowUpProposalRatificationSchemaExists(
    pqxx::connection& connection);
bool RecommendationCampaignFollowUpProposalRatificationSchemaExists(
    pqxx::transaction_base& transaction);

RecommendationCampaignFollowUpProposalRatificationPersistResult
PersistRecommendationCampaignFollowUpProposalRatification(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignFollowUpProposalRatification& ratification);

std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatification(
    pqxx::connection& connection,
    long long ratificationEventId);
std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatification(
    pqxx::transaction_base& transaction,
    long long ratificationEventId);

std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatificationByReviewEventId(
    pqxx::connection& connection,
    long long reviewEventId);
std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatificationByReviewEventId(
    pqxx::transaction_base& transaction,
    long long reviewEventId);

std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatificationByProposalId(
    pqxx::connection& connection,
    long long followUpProposalId);
std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatificationByProposalId(
    pqxx::transaction_base& transaction,
    long long followUpProposalId);

std::vector<PersistedRecommendationCampaignFollowUpProposalRatification>
ListRecommendationCampaignFollowUpProposalRatifications(
    pqxx::connection& connection,
    int limit);
std::vector<PersistedRecommendationCampaignFollowUpProposalRatification>
ListRecommendationCampaignFollowUpProposalRatifications(
    pqxx::transaction_base& transaction,
    int limit);

} // namespace EA::ExperimentRecommendation
