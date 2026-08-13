#pragma once

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationCampaignStatusContractVersion = 1;
inline constexpr int kMaximumRecommendationCampaignStatusMembers = 1000;

struct RecommendationCampaignStatusRequest
{
    long long materializationId = -1;
};

enum class RecommendationCampaignStatusExecutionState
{
    notExecuted,
    executed,
    malformed,
    conflicting
};

enum class RecommendationCampaignStatusActivationState
{
    notActivated,
    activated,
    malformed,
    conflicting
};

enum class RecommendationCampaignStatusOperationalState
{
    proposalOnly,
    paused,
    pending,
    runningTrain,
    runningInfer,
    runningAnalyze,
    completed,
    failed,
    cancelled,
    pausedAfterProgress,
    inconsistent
};

enum class RecommendationCampaignStatusTerminalResult
{
    notTerminal,
    succeeded,
    failed,
    cancelled,
    unknown
};

enum class RecommendationCampaignStatusConsistency
{
    consistent,
    inconsistent
};

enum class RecommendationCampaignAggregateStatus
{
    notExecuted,
    partiallyExecuted,
    executedPaused,
    partiallyActivated,
    queued,
    inProgress,
    completed,
    completedWithFailures,
    cancelled,
    mixedTerminal,
    inconsistent
};

struct RecommendationCampaignStatusExperimentEvidence
{
    long long experimentId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    std::string status;
    std::string phase;
    int targetEpochs = 0;
    std::optional<int> currentEpoch;
    std::optional<int> workerPid;
    std::optional<std::string> currentOperation;
    std::optional<std::string> workerStartedAt;
    std::optional<std::string> startedAt;
    std::optional<std::string> completedAt;
    std::optional<int> exitCode;
    std::optional<std::string> errorMessage;
    std::optional<long long> modelId;
    std::string invocationMode;
    long long duplicateNonce = 0;
    std::string invocationIdentityCanonical;
    bool invocationProvenanceValid = false;
    bool inferStartPresent = false;
    bool inferEndPresent = false;
    bool inferConfigured = false;
    int modelLinkCount = 0;
    int inferenceResultCount = 0;
    int completedInferenceResultCount = 0;
    int failedInferenceResultCount = 0;
    int analysisResultCount = 0;
    int completedAnalysisResultCount = 0;
    int failedAnalysisResultCount = 0;
};

struct RecommendationCampaignStatusMemberInput
{
    int memberOrdinal = 0;
    long long materializationMemberId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    long long rankingMemberId = -1;
    long long proposalId = -1;
    std::optional<long long> currentReviewDecisionId;
    std::optional<std::string> currentReviewDecision;
    std::optional<long long> authorizationReviewDecisionId;
    std::optional<long long> executionId;
    std::optional<long long> activationId;
    std::optional<long long> experimentId;
    long long executionEvidenceCount = 0;
    long long activationEvidenceCount = 0;
    long long executionCount = 0;
    long long activationCount = 0;
    bool workflowConsistent = true;
    std::vector<std::string> workflowDiagnostics;
    std::optional<RecommendationCampaignStatusExperimentEvidence> experiment;
};

struct RecommendationCampaignStatusInput
{
    long long materializationId = -1;
    int materializationContractVersion = 0;
    std::string materializationIdentityCanonical;
    std::string materializationIdentityHash;
    int selectedMemberCount = 0;
    std::string observedAt;
    std::vector<RecommendationCampaignStatusMemberInput> members;
};

struct RecommendationCampaignStatusMember
{
    int memberOrdinal = 0;
    long long materializationMemberId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    long long rankingMemberId = -1;
    long long proposalId = -1;
    std::optional<long long> currentReviewDecisionId;
    std::optional<std::string> currentReviewDecision;
    std::optional<long long> authorizationReviewDecisionId;
    std::optional<long long> executionId;
    std::optional<long long> activationId;
    std::optional<long long> experimentId;
    long long executionEvidenceCount = 0;
    long long activationEvidenceCount = 0;
    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    RecommendationCampaignStatusExecutionState executionState =
        RecommendationCampaignStatusExecutionState::notExecuted;
    RecommendationCampaignStatusActivationState activationState =
        RecommendationCampaignStatusActivationState::notActivated;
    RecommendationCampaignStatusOperationalState operationalState =
        RecommendationCampaignStatusOperationalState::proposalOnly;
    RecommendationCampaignStatusTerminalResult terminalResult =
        RecommendationCampaignStatusTerminalResult::notTerminal;
    RecommendationCampaignStatusConsistency consistency =
        RecommendationCampaignStatusConsistency::consistent;
    std::optional<std::string> status;
    std::optional<std::string> phase;
    std::optional<int> currentEpoch;
    std::optional<int> targetEpochs;
    std::optional<int> workerPid;
    std::optional<std::string> currentOperation;
    std::optional<std::string> workerStartedAt;
    std::optional<std::string> startedAt;
    std::optional<std::string> completedAt;
    std::optional<int> exitCode;
    std::optional<std::string> errorMessage;
    std::optional<long long> modelId;
    std::optional<std::string> invocationIdentityHash;
    int modelLinkCount = 0;
    int finalInferenceResultCount = 0;
    int completedFinalInferenceResultCount = 0;
    int failedFinalInferenceResultCount = 0;
    int finalAnalysisResultCount = 0;
    int completedFinalAnalysisResultCount = 0;
    int failedFinalAnalysisResultCount = 0;
    bool trainComplete = false;
    bool inferConfigured = false;
    bool inferComplete = false;
    bool analyzeConfigured = false;
    bool analyzeComplete = false;
    std::vector<std::string> diagnosticCodes;
};

struct RecommendationCampaignStatusSnapshot
{
    RecommendationCampaignStatusRequest request;
    int materializationContractVersion = 0;
    std::string materializationIdentityHash;
    std::string observedAt;
    std::string snapshotIdentityCanonical;
    std::string snapshotIdentityHash;
    RecommendationCampaignAggregateStatus aggregateStatus =
        RecommendationCampaignAggregateStatus::notExecuted;
    RecommendationCampaignStatusConsistency consistency =
        RecommendationCampaignStatusConsistency::consistent;
    int proposalOnlyCount = 0;
    int executedCount = 0;
    int activatedCount = 0;
    int pausedCount = 0;
    int pendingCount = 0;
    int runningCount = 0;
    int runningTrainCount = 0;
    int runningInferCount = 0;
    int runningAnalyzeCount = 0;
    int completedCount = 0;
    int failedCount = 0;
    int cancelledCount = 0;
    int inconsistentCount = 0;
    int terminalCount = 0;
    int nonterminalCount = 0;
    int activeWorkerCount = 0;
    std::optional<int> minimumCurrentEpoch;
    std::optional<int> maximumCurrentEpoch;
    std::optional<long long> totalCurrentEpochs;
    std::optional<long long> totalTargetEpochs;
    std::optional<double> completedPercent;
    std::optional<double> terminalPercent;
    std::vector<std::string> diagnosticCodes;
    std::vector<RecommendationCampaignStatusMember> members;
};

RecommendationCampaignStatusRequest NormalizeRecommendationCampaignStatusRequest(
    const RecommendationCampaignStatusRequest& request);
RecommendationCampaignStatusSnapshot BuildRecommendationCampaignStatusSnapshot(
    const RecommendationCampaignStatusRequest& request,
    RecommendationCampaignStatusInput input);

std::string RecommendationCampaignStatusExecutionStateText(
    RecommendationCampaignStatusExecutionState value);
std::string RecommendationCampaignStatusActivationStateText(
    RecommendationCampaignStatusActivationState value);
std::string RecommendationCampaignStatusOperationalStateText(
    RecommendationCampaignStatusOperationalState value);
std::string RecommendationCampaignStatusTerminalResultText(
    RecommendationCampaignStatusTerminalResult value);
std::string RecommendationCampaignStatusConsistencyText(
    RecommendationCampaignStatusConsistency value);
std::string RecommendationCampaignAggregateStatusText(
    RecommendationCampaignAggregateStatus value);

} // namespace EA::ExperimentRecommendation
