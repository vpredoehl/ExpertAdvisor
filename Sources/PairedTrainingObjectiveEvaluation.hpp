#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace EA::PairedTrainingObjectiveEvaluation
{

enum class Disposition
{
    Promising,
    Mixed,
    NotPromising,
    InvalidComparison,
    Incomplete
};

enum class ProfitabilityPrimaryMetric
{
    AggregateTerminalHorizonLogReturnSum,
    AverageTerminalHorizonLogReturnPerActionablePrediction
};

struct ObjectiveProvenance
{
    std::string canonical;
    std::string hash;
    bool operator==(const ObjectiveProvenance&) const = default;
};

struct RunProvenance
{
    std::string gitCommit;
    std::string gitBranch;
    std::optional<bool> gitDirty;
    std::string buildConfiguration;
    std::string compilerVersion;
    std::string schemaVersion;
    std::string schedulerVersion;
    std::string binaryName;
    bool operator==(const RunProvenance&) const = default;
};

// Exact persisted scientific identity. Canonical strings are used for
// contracts that already have a versioned representation in model metadata.
// No floating-point tolerance is applied between paired arms.
struct ScientificConfiguration
{
    long long experimentId = 0;
    std::string symbol;
    int predictionHorizon = 0;
    std::string trainStart;
    std::string trainEnd;
    std::string inferenceStart;
    std::string inferenceEnd;
    int targetEpochs = 0;
    double threshold = 0.0;
    std::optional<double> coreLearningRateMultiplier;
    std::optional<double> headLearningRateMultiplier;
    int checkpointInterval = 0;
    int inputWidth = 0;
    int hiddenSize = 0;
    int layerCount = 0;
    int windowSize = 0;
    int modelMetadataSchemaVersion = 0;
    int trainConfigurationSchemaVersion = 0;
    int normalizationVersion = 0;
    double classWeightDown = 0.0;
    double classWeightNeutral = 0.0;
    double classWeightUp = 0.0;
    std::string featureWarmupScope;
    std::string donchianMode;
    int donchianLookback = 0;
    std::string featureAblationMask;
    int optimizerMetadataSchemaVersion = 0;
    int optimizerType = 0;
    std::uint64_t optimizerUpdateCount = 0;
    int optimizerFirstMomentBufferCount = 0;
    int optimizerSecondMomentBufferCount = 0;
    double persistedCoreLearningRateMultiplier = 0.0;
    double persistedHeadWeightLearningRateMultiplier = 0.0;
    double persistedHeadBiasLearningRateMultiplier = 0.0;
    int labelRuleId = 0;
    int targetType = 0;
    double targetScale = 0.0;
    double targetBias = 0.0;
    bool targetUseZScore = false;
    double targetMean = 0.0;
    double targetStandardDeviation = 0.0;
    int modelInputMetadataSchemaVersion = 0;
    int modelInputLayoutVersion = 0;
    std::string persistedTrainingSymbol;
    std::string persistedTrainingStart;
    std::string persistedTrainingEnd;
    std::optional<std::string> inputWidthExpansionCanonical;
    std::optional<long long> resumeModelId;
    bool resumeExpandInputWidth = false;
    ObjectiveProvenance experimentObjective;
    RunProvenance runProvenance;
};

struct MaterializedModelObjective
{
    long long modelId = 0;
    bool isFinalModel = false;
    ObjectiveProvenance objective;
};

struct RuntimeObjectiveEvidence
{
    std::string eventName;
    std::string objectiveIdentifier;
    std::string objectiveHash;
};

struct ClassificationEvidence
{
    long long inferenceResultId = 0;
    long long modelId = 0;
    std::string inferenceScope;
    std::optional<long long> checkpointEvalId;
    std::optional<long long> parentExperimentId;
    std::string status;
    std::string symbol;
    int predictionHorizon = 0;
    double threshold = 0.0;
    int windowSize = 0;
    int labelRuleId = 0;
    int targetType = 0;
    std::string inferenceStart;
    std::string inferenceEnd;
    int completedEpochs = 0;
    std::optional<double> accuracy;
    std::optional<double> predictedDownProportion;
    std::optional<double> predictedNeutralProportion;
    std::optional<double> predictedUpProportion;

    long long analysisId = 0;
    long long analysisExperimentId = 0;
    long long analysisModelId = 0;
    std::string analysisScope;
    std::optional<long long> analysisCheckpointEvalId;
    std::optional<long long> analysisParentExperimentId;
    std::string analysisStatus;
    std::optional<double> inferenceAccuracy;
    std::optional<double> acceptAccuracy;
    std::optional<double> acceptRate;
    std::optional<double> leaderScore;
    std::optional<std::uint64_t> predictedDownCount;
    std::optional<std::uint64_t> predictedNeutralCount;
    std::optional<std::uint64_t> predictedUpCount;
    std::optional<std::uint64_t> acceptedPredictionCount;
};

struct ProfitabilityEvidence
{
    long long observationId = 0;
    long long experimentId = 0;
    long long modelId = 0;
    long long inferenceResultId = 0;
    std::string inferenceScope;
    std::optional<long long> checkpointEvalId;
    std::string inferenceStart;
    std::string inferenceEnd;
    std::uint64_t predictionCount = 0;
    std::uint64_t actionableCount = 0;
    double aggregateTerminalHorizonLogReturnSum = 0.0;
    std::optional<double>
        averageTerminalHorizonLogReturnPerActionablePrediction;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
    std::string sourceContentHash;
    std::string observationIdentityCanonical;
    std::string observationIdentityHash;
};

struct ArmEvidence
{
    ScientificConfiguration configuration;
    std::string experimentStatus;
    std::string experimentPhase;
    std::optional<long long> finalModelId;
    std::vector<MaterializedModelObjective> materializedModelObjectives;
    std::optional<RuntimeObjectiveEvidence> runtimeObjective;
    std::optional<ClassificationEvidence> classification;
    std::optional<ProfitabilityEvidence> profitability;
};

struct ClassificationDegradationPolicy
{
    double maximumInferenceAccuracyDecrease = 0.0;
    double maximumAcceptAccuracyDecrease = 0.0;
    double maximumAcceptRateDecrease = 0.0;
    double maximumLeaderScoreDecrease = 0.0;
    double maximumNeutralProportionIncrease = 0.0;
};

struct MaterialityPolicy
{
    ProfitabilityPrimaryMetric primaryProfitabilityMetric =
        ProfitabilityPrimaryMetric::
            AggregateTerminalHorizonLogReturnSum;
    // Absolute directional-log-return units. Zero is valid only when the
    // operator deliberately chooses sign-only screening.
    double minimumProfitabilityImprovement = 0.0;
    double maximumProfitabilityWorsening = 0.0;
    std::optional<ClassificationDegradationPolicy> classification;
};

struct MetricDelta
{
    std::optional<double> control;
    std::optional<double> treatment;
    std::optional<double> treatmentMinusControl;
    // Defined as (treatment-control)/abs(control); absent when control is zero.
    std::optional<double> relativeToAbsoluteControl;
    bool operator==(const MetricDelta&) const = default;
};

struct ComparisonResult
{
    Disposition disposition = Disposition::Incomplete;
    std::vector<std::string> invalidReasons;
    std::vector<std::string> incompleteReasons;
    std::vector<std::string> interpretationReasons;
    MetricDelta actionableCount;
    MetricDelta aggregateProfitability;
    MetricDelta averageProfitability;
    MetricDelta inferenceAccuracy;
    MetricDelta acceptAccuracy;
    MetricDelta acceptRate;
    MetricDelta predictedNeutralProportion;
    MetricDelta leaderScore;
};

ComparisonResult Compare(
    const ArmEvidence& control,
    const ArmEvidence& treatment,
    const MaterialityPolicy& policy);

std::string MaterialityPolicyCanonicalText(const MaterialityPolicy& policy);
std::string MaterialityPolicyIdentity(const MaterialityPolicy& policy);

std::string DispositionText(Disposition value);

} // namespace EA::PairedTrainingObjectiveEvaluation
