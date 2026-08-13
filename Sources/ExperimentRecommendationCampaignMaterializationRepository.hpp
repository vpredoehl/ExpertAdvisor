#pragma once

#include "ExperimentRecommendationCampaignMaterialization.hpp"
#include "ExperimentRecommendationConversionRepository.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

inline constexpr int kMaximumRecommendationCampaignMaterializationListLimit =
    1000;

struct PersistedRecommendationCampaignMaterializationMember
{
    long long materializationMemberId = -1;
    int memberOrdinal = 0;
    long long rankingMemberId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    int rankingPosition = 0;
    std::string selectedMemberIdentityCanonical;
    std::string selectedMemberIdentityHash;
    long long conversionProposalId = -1;
    std::string proposalIdentityCanonical;
    std::string proposalIdentityHash;
    std::string createdAt;
};

struct PersistedRecommendationCampaignMaterialization
{
    long long materializationId = -1;
    long long campaignApprovalId = -1;
    int contractVersion = 0;
    std::string approvalIdentityHash;
    long long rankingSnapshotId = -1;
    std::string rankingSnapshotIdentityHash;
    std::string planningPolicyHash;
    std::string campaignPlanIdentityHash;
    std::string campaignReviewIdentityHash;
    std::string materializedBy;
    std::string reasonText;
    int selectedMemberCount = 0;
    int initiallyCreatedProposalCount = 0;
    int initiallyReusedProposalCount = 0;
    std::string identityCanonical;
    std::string identityHash;
    std::string createdAt;
    std::vector<PersistedRecommendationCampaignMaterializationMember> members;
};

enum class RecommendationCampaignMaterializationPersistOutcome
{
    recorded,
    existingIdentical
};

struct RecommendationCampaignMaterializationPersistResult
{
    RecommendationCampaignMaterializationPersistOutcome outcome =
        RecommendationCampaignMaterializationPersistOutcome::recorded;
    PersistedRecommendationCampaignMaterialization materialization;
    int newlyCreatedProposalCount = 0;
    int reusedProposalCount = 0;
};

bool RecommendationCampaignMaterializationSchemaExists(
    pqxx::connection& connection);
bool RecommendationCampaignMaterializationSchemaExists(
    pqxx::transaction_base& transaction);

RecommendationConversionRequest
LoadRecommendationCampaignConversionRequest(
    pqxx::transaction_base& transaction,
    long long rankingSnapshotId,
    long long rankingMemberId,
    long long recommendationId,
    std::optional<Donchian20Mode> campaignDonchian20Mode = std::nullopt);

RecommendationCampaignMaterializationPersistResult
PersistRecommendationCampaignMaterialization(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignMaterializationEvidence& evidence);

std::optional<PersistedRecommendationCampaignMaterialization>
FindRecommendationCampaignMaterialization(
    pqxx::connection& connection,
    long long materializationId);
std::optional<PersistedRecommendationCampaignMaterialization>
FindRecommendationCampaignMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId);
std::optional<PersistedRecommendationCampaignMaterialization>
FindRecommendationCampaignMaterializationByApproval(
    pqxx::transaction_base& transaction,
    long long campaignApprovalId);

std::vector<PersistedRecommendationCampaignMaterialization>
ListRecommendationCampaignMaterializations(
    pqxx::connection& connection,
    std::optional<long long> campaignApprovalId,
    int limit);
std::vector<PersistedRecommendationCampaignMaterialization>
ListRecommendationCampaignMaterializations(
    pqxx::transaction_base& transaction,
    std::optional<long long> campaignApprovalId,
    int limit);

} // namespace EA::ExperimentRecommendation
