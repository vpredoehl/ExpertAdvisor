#pragma once

#include "ExperimentRecommendationCampaignFollowUpProposal.hpp"

#include <optional>
#include <string>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

struct PersistedRecommendationCampaignFollowUpProposal
{
    const long long followUpProposalId;
    const int identityHashCollisionOrdinal;
    const RecommendationCampaignFollowUpProposal proposal;
    const std::string createdAt;

    PersistedRecommendationCampaignFollowUpProposal(
        long long followUpProposalId,
        int identityHashCollisionOrdinal,
        RecommendationCampaignFollowUpProposal proposal,
        std::string createdAt);
    PersistedRecommendationCampaignFollowUpProposal(
        const PersistedRecommendationCampaignFollowUpProposal&) = default;
    PersistedRecommendationCampaignFollowUpProposal(
        PersistedRecommendationCampaignFollowUpProposal&&) = default;
    bool operator==(
        const PersistedRecommendationCampaignFollowUpProposal&) const =
        default;
};

enum class RecommendationCampaignFollowUpProposalPersistOutcome
{
    recorded,
    existingIdentical
};

std::string RecommendationCampaignFollowUpProposalPersistOutcomeText(
    RecommendationCampaignFollowUpProposalPersistOutcome outcome);

struct RecommendationCampaignFollowUpProposalPersistResult
{
    const RecommendationCampaignFollowUpProposalPersistOutcome outcome;
    const PersistedRecommendationCampaignFollowUpProposal persisted;

    RecommendationCampaignFollowUpProposalPersistResult(
        RecommendationCampaignFollowUpProposalPersistOutcome outcome,
        PersistedRecommendationCampaignFollowUpProposal persisted);
};

bool RecommendationCampaignFollowUpProposalSchemaExists(
    pqxx::connection& connection);
bool RecommendationCampaignFollowUpProposalSchemaExists(
    pqxx::transaction_base& transaction);

RecommendationCampaignFollowUpProposalPersistResult
PersistRecommendationCampaignFollowUpProposal(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignFollowUpProposal& proposal);

std::optional<PersistedRecommendationCampaignFollowUpProposal>
FindRecommendationCampaignFollowUpProposal(
    pqxx::connection& connection,
    long long followUpProposalId);
std::optional<PersistedRecommendationCampaignFollowUpProposal>
FindRecommendationCampaignFollowUpProposal(
    pqxx::transaction_base& transaction,
    long long followUpProposalId);

std::optional<PersistedRecommendationCampaignFollowUpProposal>
FindRecommendationCampaignFollowUpProposalByIdentity(
    pqxx::connection& connection,
    const std::string& proposalIdentityCanonical);
std::optional<PersistedRecommendationCampaignFollowUpProposal>
FindRecommendationCampaignFollowUpProposalByIdentity(
    pqxx::transaction_base& transaction,
    const std::string& proposalIdentityCanonical);

} // namespace EA::ExperimentRecommendation
