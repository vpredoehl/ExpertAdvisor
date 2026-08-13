#pragma once

#include "ExperimentRecommendationCampaignMaterialization.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationCampaignActivationContractVersion = 1;
inline constexpr int kMaximumRecommendationCampaignActivationMembers = 1000;

struct RecommendationCampaignActivationRequest
{
    long long materializationId = -1;
    bool dryRun = false;
};

enum class RecommendationCampaignActivationEvidenceState
{
    eligible,
    existingIdentical,
    invalid
};

struct RecommendationCampaignActivationMemberInput
{
    int memberOrdinal = 0;
    long long materializationMemberId = -1;
    long long recommendationId = -1;
    long long proposalId = -1;
    std::optional<long long> authorizationReviewDecisionId;
    std::optional<long long> executionId;
    std::optional<long long> experimentId;
    std::optional<long long> activationId;
    RecommendationCampaignActivationEvidenceState evidenceState =
        RecommendationCampaignActivationEvidenceState::invalid;
    std::string experimentStatus;
    std::string experimentPhase;
    std::string evidenceDiagnostic;
};

struct RecommendationCampaignActivationInput
{
    long long materializationId = -1;
    int materializationContractVersion = 0;
    std::string materializationIdentityCanonical;
    std::string materializationIdentityHash;
    int selectedMemberCount = 0;
    bool materializationComplete = false;
    bool workflowEvidenceConsistent = false;
    std::vector<RecommendationCampaignActivationMemberInput> members;
};

enum class RecommendationCampaignActivationMemberAction
{
    activate,
    alreadySatisfied
};

enum class RecommendationCampaignActivationPlanState
{
    ready,
    alreadySatisfied
};

struct RecommendationCampaignActivationMemberPlan
{
    int memberOrdinal = 0;
    long long materializationMemberId = -1;
    long long recommendationId = -1;
    long long proposalId = -1;
    long long authorizationReviewDecisionId = -1;
    long long executionId = -1;
    long long experimentId = -1;
    std::optional<long long> previousActivationId;
    std::string preActivationStatus;
    std::string preActivationPhase;
    std::string postActivationStatus;
    std::string postActivationPhase;
    RecommendationCampaignActivationMemberAction action =
        RecommendationCampaignActivationMemberAction::activate;
    std::vector<std::string> diagnosticCodes;
};

struct RecommendationCampaignActivationPlan
{
    RecommendationCampaignActivationRequest request;
    std::string materializationIdentityHash;
    std::string operationIdentityCanonical;
    std::string operationIdentityHash;
    RecommendationCampaignActivationPlanState state =
        RecommendationCampaignActivationPlanState::ready;
    int membersValidated = 0;
    int alreadyActivatedCount = 0;
    std::vector<RecommendationCampaignActivationMemberPlan> members;
    std::vector<std::string> diagnosticCodes;
};

RecommendationCampaignActivationRequest NormalizeRecommendationCampaignActivationRequest(
    const RecommendationCampaignActivationRequest& request);

RecommendationCampaignActivationPlan BuildRecommendationCampaignActivationPlan(
    const RecommendationCampaignActivationRequest& request,
    const RecommendationCampaignActivationInput& input);

} // namespace EA::ExperimentRecommendation
