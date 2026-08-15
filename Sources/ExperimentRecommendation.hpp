#pragma once

#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "Donchian20Mode.hpp"
#include "FeatureWarmupScope.hpp"

namespace EA::ExperimentRecommendation
{

inline constexpr const char* kCoreLrMult = "core_lr_mult";
inline constexpr const char* kHeadLrMult = "head_lr_mult";
inline constexpr const char* kLabelThreshold = "label_threshold";
inline constexpr const char* kPredictionHorizon = "prediction_horizon";

enum class RecommendationSourceScope
{
    symbolHorizon,
    symbol,
    global
};

enum class RecommendationStatus
{
    proposed,
    rejected,
    expired,
    approved
};

enum class RecommendationDuplicateType
{
    noDuplicate,
    existingExperiment,
    activeRecommendation,
    excludedTerminalExperiment
};

struct RecommendationPolicy
{
    bool enabled = true;
    double minimumLeaderScore = 0.0;
    double minimumInferenceAccuracy = 0.0;
    std::optional<double> maximumPredictedNeutralProportion = 0.80;
    int topSourcesPerScope = 3;
    int maximumRecommendationsPerSource = 4;
    int maximumRecommendationsPerScan = 20;
    int minimumEvidenceCount = 1;
    std::set<std::string> allowedParameters{
        kCoreLrMult, kHeadLrMult, kLabelThreshold};
    std::vector<double> coreLrOffsets{-0.25, 0.25};
    std::vector<double> headLrOffsets{-0.5, 0.5};
    std::vector<double> labelThresholdOffsets{-0.0001, 0.0001};
    bool allowHorizonChanges = false;
    std::vector<int> permittedHorizons;
    RecommendationSourceScope sourceScope =
        RecommendationSourceScope::symbolHorizon;
    bool terminalExperimentsAreDuplicates = true;
    std::optional<int> expirationDays;
    int policyVersion = 1;
};

// These are the experiment-row fields that define the reproducible research
// question supported by Phase 4A. Runtime state, orchestration policy, lineage,
// and provenance are deliberately excluded. The schema audit in the Phase 4A
// foundation document records persisted model settings that cannot yet be
// round-tripped through an experiment row; Phase 4A must not guess them.
struct EffectiveExperimentConfiguration
{
    std::string symbol;
    int predictionHorizon = 0;
    double labelThreshold = 0.0;
    std::optional<double> coreLrMult;
    std::optional<double> headLrMult;
    int targetEpochs = 0;
    std::string trainStartDate;
    std::string trainEndDate;
    std::optional<std::string> inferStartDate;
    std::optional<std::string> inferEndDate;
    Donchian20Mode donchian20Mode = kDefaultDonchian20Mode;
    FeatureWarmupScope featureWarmupScope = kDefaultFeatureWarmupScope;
};

// Persisted recommendation provenance is immutable.  Version selection is
// therefore explicit when reconstructing historical identities rather than
// being inferred from the current default representation.
enum class RecommendationSemanticConfigurationVersion
{
    v3,
    v4,
    v5
};

// Invocation identity is deliberately distinct from semantic configuration
// identity. A resume model supplies initial weights and optimizer progress, so
// it materially distinguishes executions without changing the hyperparameter
// and data-selection question represented above. Checkpoint cadence controls
// persistence/evaluation opportunities rather than ordinary training math.
struct ExperimentInvocationConfiguration
{
    EffectiveExperimentConfiguration configuration;
    int checkpointInterval = 20;
    std::optional<long long> resumeModelId;
};

struct RecommendationSource
{
    long long experimentId = -1;
    std::optional<long long> modelId;
    std::optional<long long> analysisId;
    ExperimentInvocationConfiguration invocation;
    std::optional<double> leaderScore;
    std::optional<double> inferenceAccuracy;
    std::optional<double> predictedNeutralProportion;
    long long evidenceCount = 0;
};

struct RecommendationCandidateIdentity
{
    EffectiveExperimentConfiguration configuration;
    std::string canonicalText;
    std::string hash;
};

struct RecommendationInvocationIdentity
{
    ExperimentInvocationConfiguration invocation;
    std::string canonicalText;
    std::string hash;
};

RecommendationPolicy ParseRecommendationPolicy(const std::string& text);
std::optional<std::string> ValidateRecommendationPolicy(
    const RecommendationPolicy& policy);
std::string RecommendationPolicyCanonicalText(
    const RecommendationPolicy& policy);
std::string RecommendationPolicyHash(const RecommendationPolicy& policy);

std::string CanonicalRecommendationDouble(double value);
// Tagged deterministic accelerator for recommendation-owned canonical text.
// Canonical text remains authoritative and must be compared after a hash match.
std::string RecommendationCanonicalHash(const std::string& canonicalText);
std::optional<RecommendationSemanticConfigurationVersion>
RecommendationSemanticConfigurationVersionFromCanonicalText(
    const std::string& canonicalText);
FeatureWarmupScope RecommendationFeatureWarmupScopeFromCanonicalText(
    const std::string& canonicalText);
std::string RecommendationSemanticConfigurationFromInvocationCanonicalText(
    const std::string& canonicalText);
std::string CanonicalExperimentDateText(const std::string& value);
std::string EffectiveExperimentConfigurationCanonicalText(
    const EffectiveExperimentConfiguration& configuration,
    RecommendationSemanticConfigurationVersion version =
        RecommendationSemanticConfigurationVersion::v5);
std::string RecommendationCandidateHash(
    const EffectiveExperimentConfiguration& configuration,
    RecommendationSemanticConfigurationVersion version =
        RecommendationSemanticConfigurationVersion::v5);
RecommendationCandidateIdentity BuildRecommendationCandidateIdentity(
    const EffectiveExperimentConfiguration& configuration,
    RecommendationSemanticConfigurationVersion version =
        RecommendationSemanticConfigurationVersion::v5);
std::string ExperimentInvocationCanonicalText(
    const ExperimentInvocationConfiguration& invocation,
    RecommendationSemanticConfigurationVersion version =
        RecommendationSemanticConfigurationVersion::v5);
std::string ExperimentInvocationHash(
    const ExperimentInvocationConfiguration& invocation,
    RecommendationSemanticConfigurationVersion version =
        RecommendationSemanticConfigurationVersion::v5);
RecommendationInvocationIdentity BuildRecommendationInvocationIdentity(
    const ExperimentInvocationConfiguration& invocation,
    RecommendationSemanticConfigurationVersion version =
        RecommendationSemanticConfigurationVersion::v5);

std::string RecommendationSourceScopeText(RecommendationSourceScope value);
std::optional<RecommendationSourceScope> ParseRecommendationSourceScope(
    const std::string& value);
std::string RecommendationStatusText(RecommendationStatus value);
std::optional<RecommendationStatus> ParseRecommendationStatus(
    const std::string& value);
std::string RecommendationDuplicateTypeText(RecommendationDuplicateType value);

} // namespace EA::ExperimentRecommendation
