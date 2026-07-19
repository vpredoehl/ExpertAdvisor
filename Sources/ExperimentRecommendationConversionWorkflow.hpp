#pragma once

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kDefaultRecommendationConversionWorkflowListLimit = 100;
inline constexpr int kMaximumRecommendationConversionWorkflowListLimit = 1000;

enum class RecommendationConversionWorkflowState
{
    proposed,
    pendingReview,
    rejected,
    approvedNotExecuted,
    executedPaused,
    activatedPending,
    schedulerClaimedOrRunning,
    completed,
    failed,
    cancelled,
    inconsistent
};

enum class RecommendationConversionWorkflowIntegrity
{
    consistent,
    inconsistent
};

struct RecommendationConversionWorkflowReviewFact
{
    long long reviewDecisionId = -1;
    long long proposalId = -1;
    std::string decision;
};

struct RecommendationConversionWorkflowExecutionFact
{
    long long executionId = -1;
    long long proposalId = -1;
    long long reviewDecisionId = -1;
    long long experimentId = -1;
    int contractVersion = 0;
    std::string authorizationDecision;
    std::string identityCanonical;
    std::string identityHash;
};

struct RecommendationConversionWorkflowActivationFact
{
    long long activationId = -1;
    long long executionId = -1;
    long long proposalId = -1;
    long long reviewDecisionId = -1;
    long long experimentId = -1;
    int contractVersion = 0;
    std::string previousStatus;
    std::string previousPhase;
    std::string resultingStatus;
    std::string resultingPhase;
    std::string identityCanonical;
    std::string identityHash;
};

struct RecommendationConversionWorkflowExperimentFact
{
    long long experimentId = -1;
    std::string status;
    std::string phase;
    std::optional<int> workerPid;
    std::optional<std::string> currentOperation;
    std::optional<int> currentEpoch;
};

struct RecommendationConversionWorkflowFacts
{
    long long proposalId = -1;
    int proposalContractVersion = 0;
    std::string proposalIdentityCanonical;
    std::string proposalIdentityHash;
    std::optional<RecommendationConversionWorkflowReviewFact> latestReview;
    std::optional<RecommendationConversionWorkflowReviewFact>
        executionReview;
    std::optional<RecommendationConversionWorkflowExecutionFact> execution;
    std::optional<RecommendationConversionWorkflowActivationFact> activation;
    std::optional<RecommendationConversionWorkflowExperimentFact> experiment;
    long long executionCount = 0;
    long long activationCount = 0;
};

struct RecommendationConversionWorkflowDerivation
{
    RecommendationConversionWorkflowState state =
        RecommendationConversionWorkflowState::inconsistent;
    RecommendationConversionWorkflowIntegrity integrity =
        RecommendationConversionWorkflowIntegrity::inconsistent;
    std::vector<std::string> diagnosticCodes;
};

std::string RecommendationConversionWorkflowStateText(
    RecommendationConversionWorkflowState state);

std::optional<RecommendationConversionWorkflowState>
ParseRecommendationConversionWorkflowState(const std::string& text);

std::string RecommendationConversionWorkflowIntegrityText(
    RecommendationConversionWorkflowIntegrity integrity);

RecommendationConversionWorkflowDerivation
DeriveRecommendationConversionWorkflow(
    const RecommendationConversionWorkflowFacts& facts);

} // namespace EA::ExperimentRecommendation
