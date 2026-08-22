#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace EA::ExperimentScheduler
{

struct ContinuationSourceExperiment
{
    std::string symbol;
    int predictionHorizon = 0;
    double cNextThreshold = 0.0;
    int checkpointInterval = 20;
    int targetEpochs = 0;
    std::string trainStart;
    std::string trainEnd;
    std::optional<std::string> inferStart;
    std::optional<std::string> inferEnd;
    std::optional<long long> lastModelId;
};

struct ContinuationPolicyConfig
{
    long long sourceExperimentId = -1;
    ContinuationSourceExperiment source;
    std::string status;
    std::string phase;
    bool enabled = false;
    std::optional<int> targetEpochs;
    int minEvals = 2;
    int patience = 2;
    std::optional<double> minLeaderScore;
    std::optional<double> minInferAccuracy;
    std::optional<double> minImprovement;
    std::optional<double> maxDegradation;
    std::optional<int> topN;
    std::string scope = "symbol_horizon";
    std::string trendMode = "none";
    std::string sourceMode = "best_checkpoint";
    bool includeExcluded = false;
    bool candidateExcluded = false;
    bool inheritToChild = false;
    std::optional<std::string> progressionMode;
    std::optional<int> targetIncrement;
    std::optional<int> maxTargetEpochs;
    std::optional<std::vector<int>> targetSequence;
    bool policyInherited = false;
    std::optional<long long> inheritedFromExperimentId;
    std::optional<long long> inheritedFromRevision;
    std::optional<std::string> inheritedFromHash;
    std::string inheritanceStatus = "not_requested";
    long long policyRevision = 1;
    std::optional<std::string> lastDecision;
    std::optional<std::string> lastReason;
    std::optional<long long> selectedModelId;
    std::optional<long long> queuedExperimentId;
    std::optional<long long> continuationSourceExperimentId;
    std::optional<long long> continuationSourceModelId;
    std::optional<int> continuationSourceEpoch;
    std::optional<long long> continuationDecisionId;
    int continuationGeneration = 1;
};

struct ContinuationPolicyUpdate
{
    std::set<std::string> keys;
    std::optional<int> targetEpochs;
    int minEvals = 2;
    int patience = 2;
    std::optional<double> minLeaderScore;
    std::optional<double> minInferAccuracy;
    std::optional<double> minImprovement;
    std::optional<double> maxDegradation;
    std::optional<int> topN;
    std::string scope = "symbol_horizon";
    std::string trendMode = "none";
    std::string sourceMode = "best_checkpoint";
    bool includeExcluded = false;
    bool candidateExcluded = false;
    bool inheritToChild = false;
    std::optional<std::string> progressionMode;
    std::optional<int> targetIncrement;
    std::optional<int> maxTargetEpochs;
    std::optional<std::vector<int>> targetSequence;
};

struct ContinuationProfitabilityEvidence
{
    long long observationId = -1;
    std::string observationIdentityHash;
    long long inferenceEvalResultId = -1;
    std::string inferenceScope;
    std::optional<long long> checkpointEvalId;
    std::string metricDefinitionHash;
    std::string sourceContentHash;
    std::uint64_t predictionCount = 0;
    std::uint64_t actionableCount = 0;
    std::uint64_t winningActionableCount = 0;
    std::uint64_t losingActionableCount = 0;
    double grossPositiveTerminalHorizonLogReturnSum = 0.0;
    double grossNegativeTerminalHorizonLogReturnSum = 0.0;
    double aggregateTerminalHorizonLogReturnSum = 0.0;
    std::optional<double>
        averageTerminalHorizonLogReturnPerActionablePrediction;
};

struct ContinuationEvidence
{
    long long analysisId = -1;
    std::optional<long long> checkpointEvalId;
    std::optional<long long> inferenceEvalResultId;
    long long modelId = -1;
    int completedEpoch = 0;
    std::optional<double> leaderScore;
    std::optional<double> inferAccuracy;
    std::optional<bool> acceptModel;
    std::string analysisScope;
    std::string completedAt;
    std::string updatedAt;
    std::optional<ContinuationProfitabilityEvidence> profitability;
    std::string profitabilityUnavailableReason =
        "no_selected_continuation_source";
};

struct ContinuationEvaluation
{
    bool persisted = false;
    bool reused = false;
    bool alreadyQueued = false;
    long long decisionId = -1;
    std::string decision = "skipped";
    std::string reason;
    ContinuationEvidence selected;
    int evidenceCount = 0;
    std::optional<int> rankValue;
    std::optional<std::string> trendMetric;
    std::optional<double> trendValue;
    std::string policyHash;
    std::string currentPolicyHash;
    std::string persistedDecisionPolicyHash;
    std::string evidenceWatermark;
    std::optional<long long> queuedExperimentId;
};

struct ContinuationPolicyIdentityMaterial
{
    std::string semanticConfiguration;
    bool inherited = false;
    std::optional<long long> inheritedFromExperimentId;
    std::string inheritanceStatus = "not_requested";
};

bool ValidContinuationScope(const std::string& value);
bool ValidContinuationTrendMode(const std::string& value);
bool ValidContinuationSourceMode(const std::string& value);
bool ValidContinuationProgressionMode(const std::string& value);

std::optional<std::string> EffectiveContinuationProgressionMode(
    const ContinuationPolicyConfig& config);
std::vector<int> ParseContinuationTargetSequence(const std::string& value);
std::string ContinuationTargetSequenceText(
    const std::optional<std::vector<int>>& sequence,
    const std::string& nullText = "NULL");

ContinuationPolicyUpdate ParseContinuationPolicyUpdate(const std::string& text);
void ApplyContinuationPolicyUpdate(
    ContinuationPolicyConfig& config,
    const ContinuationPolicyUpdate& update);

std::optional<std::string> ContinuationPolicyConfigurationError(
    const ContinuationPolicyConfig& config,
    bool requireSelectionConfig);
std::optional<std::string> ContinuationPolicyEnablementError(
    const ContinuationPolicyConfig& config);
bool ContinuationPolicySourceCompletionReady(
    const ContinuationPolicyConfig& config);

std::string ContinuationPolicySemanticCanonicalText(
    const ContinuationPolicyConfig& config);
std::string StableContinuationPolicyHash(const std::string& canonicalPolicy);
std::string SemanticContinuationPolicyHash(
    const ContinuationPolicyIdentityMaterial& material);
std::string ContinuationPolicySemanticHash(
    const ContinuationPolicyConfig& config);
std::string ContinuationOptionalDoubleText(
    const std::optional<double>& value);
std::string ContinuationOptionalIntText(
    const std::optional<int>& value);
// Stable diagnostic fields. Profitability is intentionally excluded from
// policy identity, evidence watermarks, eligibility, ordering and trends.
std::string ContinuationProfitabilityEvidenceLogFields(
    const ContinuationEvidence& evidence);
bool BetterBestContinuationSource(
    const ContinuationEvidence& lhs,
    const ContinuationEvidence& rhs);
bool IsCheckpointContinuationSource(const ContinuationEvidence& evidence);
bool IsFinalContinuationSource(
    const ContinuationPolicyConfig& config,
    const ContinuationEvidence& evidence);
std::optional<ContinuationEvidence> SelectContinuationSourceEvidence(
    const ContinuationPolicyConfig& config,
    const std::vector<ContinuationEvidence>& evidence);
bool PreferContinuationEvidenceAtSameEpoch(
    const ContinuationEvidence& candidate,
    const ContinuationEvidence& current);
std::vector<ContinuationEvidence> DeduplicateContinuationEvidence(
    const std::vector<ContinuationEvidence>& evidence);
std::string ContinuationEvidenceWatermark(
    const std::vector<ContinuationEvidence>& evidence);

enum class ContinuationTrendResult
{
    Pass,
    Reject,
    Insufficient
};

ContinuationTrendResult EvaluateContinuationTrend(
    const ContinuationPolicyConfig& config,
    const std::vector<ContinuationEvidence>& evidence,
    std::optional<std::string>& metric,
    std::optional<double>& trendValue,
    std::string& reason);
std::string ContinuationPolicyDisplayText(
    const ContinuationPolicyConfig& config);

} // namespace EA::ExperimentScheduler
