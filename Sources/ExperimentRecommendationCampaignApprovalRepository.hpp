#pragma once

#include "ExperimentRecommendationCampaignApproval.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

inline constexpr int kMaximumRecommendationCampaignApprovalListLimit = 1000;

struct PersistedRecommendationCampaignApproval
{
    long long campaignApprovalId = -1;
    int reviewHashCollisionOrdinal = 0;
    RecommendationCampaignApprovalEvidence evidence;
    std::string createdAt;
};

enum class RecommendationCampaignApprovalPersistOutcome
{
    recorded,
    existingIdentical
};

std::string RecommendationCampaignApprovalPersistOutcomeText(
    RecommendationCampaignApprovalPersistOutcome outcome);

struct RecommendationCampaignApprovalPersistResult
{
    RecommendationCampaignApprovalPersistOutcome outcome =
        RecommendationCampaignApprovalPersistOutcome::recorded;
    PersistedRecommendationCampaignApproval approval;
};

bool RecommendationCampaignApprovalSchemaExists(pqxx::connection& connection);
bool RecommendationCampaignApprovalSchemaExists(
    pqxx::transaction_base& transaction);

RecommendationCampaignApprovalPersistResult PersistRecommendationCampaignApproval(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignApprovalEvidence& evidence);

std::optional<PersistedRecommendationCampaignApproval>
FindRecommendationCampaignApproval(
    pqxx::connection& connection,
    long long campaignApprovalId);
std::optional<PersistedRecommendationCampaignApproval>
FindRecommendationCampaignApproval(
    pqxx::transaction_base& transaction,
    long long campaignApprovalId);

std::optional<PersistedRecommendationCampaignApproval>
FindRecommendationCampaignApprovalByReviewIdentity(
    pqxx::connection& connection,
    const std::string& campaignReviewIdentityCanonical);

std::vector<PersistedRecommendationCampaignApproval>
ListRecommendationCampaignApprovals(
    pqxx::connection& connection,
    std::optional<RecommendationCampaignApprovalDecision> decision,
    int limit);

} // namespace EA::ExperimentRecommendation
