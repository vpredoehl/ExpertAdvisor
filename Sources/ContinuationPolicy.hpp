#pragma once

#include <optional>
#include <set>
#include <string>

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
    std::optional<int> targetIncrement;
    std::optional<int> maxTargetEpochs;
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
    std::optional<int> targetIncrement;
    std::optional<int> maxTargetEpochs;
};

struct ContinuationEvidence
{
    long long analysisId = -1;
    std::optional<long long> checkpointEvalId;
    long long modelId = -1;
    int completedEpoch = 0;
    std::optional<double> leaderScore;
    std::optional<double> inferAccuracy;
    std::optional<bool> acceptModel;
    std::string analysisScope;
    std::string completedAt;
    std::string updatedAt;
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

ContinuationPolicyUpdate ParseContinuationPolicyUpdate(const std::string& text);
void ApplyContinuationPolicyUpdate(
    ContinuationPolicyConfig& config,
    const ContinuationPolicyUpdate& update);

std::optional<std::string> ContinuationPolicyConfigurationError(
    const ContinuationPolicyConfig& config,
    bool requireSelectionConfig);

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
std::string ContinuationPolicyDisplayText(
    const ContinuationPolicyConfig& config);

} // namespace EA::ExperimentScheduler
