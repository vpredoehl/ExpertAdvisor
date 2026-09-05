#pragma once

#include "PairedTrainingObjectiveEvaluation.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace EA::FeatureAblationPairEvaluation
{

namespace SharedEvidence = PairedTrainingObjectiveEvaluation;

enum class Disposition
{
    ComparableComplete,
    ComparableIncomplete,
    IncompatibleConfiguration,
    MissingFinalInference,
    AmbiguousFinalInference,
    ProfitabilityEvidenceUnavailable,
    InvalidAblationPair
};

// Experiment-level scientific fields that are not currently part of the
// shared paired-evidence loader. They remain separate from dynamic checkpoint
// decisions: only the configured scientific policy is compared.
struct ExtendedScientificConfiguration
{
    // Immutable experiment-level input identity introduced by migration 089.
    // Legacy experiments may have NULL/NULL here; completed legacy pairs are
    // still validated from their final model metadata below.
    std::optional<int> configuredModelInputWidth;
    std::optional<int> configuredModelInputLayoutVersion;

    // These runtime constants are not separate database columns. They are
    // part of the executable scientific contract and are paired with exact
    // run provenance. Keeping them explicit lets deterministic fixtures prove
    // that a future persisted/runtime seed or batch change fails closed.
    double baseLearningRate = 1.0e-3 / 3.0;
    int batchSize = 256;
    std::optional<unsigned int> freshInitializationSeed = 42U;

    int trainingObjectiveVersion = 0;
    int lossDefinitionVersion = 0;
    std::string auxiliaryLossMode;
    double auxiliaryLossCoefficient = 0.0;
    std::optional<std::string> regressionTargetDefinition;
    std::optional<std::string> regressionNormalizationIdentity;
    std::optional<std::string> robustLossDefinition;
    std::optional<double> robustLossDelta;
    std::string targetClippingDefinition;
    std::string objectiveNormalizationIdentity;

    bool checkpointInferenceEnabled = false;
    std::optional<int> checkpointInferenceMinimumEpoch;
    std::optional<int> checkpointInferenceInterval;
    bool checkpointPolicyEnabled = false;
    std::optional<double> checkpointPolicyMinimumLeaderScore;
    std::optional<double> checkpointPolicyMinimumInferenceAccuracy;
    std::optional<int> checkpointPolicyTopN;
    std::string checkpointPolicyScope;
    std::string checkpointPolicyStopMode;
    int checkpointPolicyGraceEvaluations = 0;
    long long checkpointPolicyRevision = 0;
    std::optional<std::string> checkpointPolicyHash;

    bool continuationPolicyEnabled = false;
    std::string continuationPolicyScientificIdentity = "disabled";
    bool operator==(const ExtendedScientificConfiguration&) const = default;
};

// Explicitly non-scientific scheduler state. It is loaded for auditability but
// deliberately excluded from every pair compatibility comparison.
struct OperationalMetadata
{
    std::string schedulerPriority;
    std::optional<int> workerPid;
    bool operator==(const OperationalMetadata&) const = default;
};

struct ResumeCheckpointProvenance
{
    long long resumeModelId = 0;
    long long modelExperimentId = 0;
    bool ownExperimentCheckpoint = false;
    std::optional<int> checkpointEpoch;
    bool operator==(const ResumeCheckpointProvenance&) const = default;
};

struct ArmEvidence
{
    SharedEvidence::ArmEvidence authoritative;
    ExtendedScientificConfiguration extended;

    // Loaded only when resume_model_id is present. This does not replace the
    // shared final-model lineage validation; it lets the feature-ablation
    // comparator distinguish a genuinely different initialization lineage
    // from matched arms that each resumed from their own checkpoint at the
    // same continuation epoch.
    std::optional<ResumeCheckpointProvenance> resumeCheckpointProvenance;

    OperationalMetadata operational;

    // Present when the exact FINAL inference resolver found a unique row,
    // even if the corresponding final analysis is still absent.
    std::optional<long long> exactFinalInferenceResultId;
};

struct MetricDelta
{
    std::optional<double> control;
    std::optional<double> ablation;
    // Positive means the metric was higher with the requested features
    // available than with those features removed.
    std::optional<double> controlMinusAblation;
    bool operator==(const MetricDelta&) const = default;
};

struct ComparisonResult
{
    Disposition disposition = Disposition::ComparableIncomplete;
    std::vector<std::string> invalidReasons;
    std::vector<std::string> incompleteReasons;
    std::string canonicalAblatedFeatureSet;
    std::string ablationIdentityCanonical;
    std::string ablationIdentityHash;

    MetricDelta predictionCount;
    MetricDelta predictedDownCount;
    MetricDelta predictedNeutralCount;
    MetricDelta predictedUpCount;
    MetricDelta acceptedPredictionCount;
    MetricDelta actionableCount;
    MetricDelta aggregateProfitability;
    MetricDelta averageProfitability;
    MetricDelta inferenceAccuracy;
    MetricDelta acceptAccuracy;
    MetricDelta acceptRate;
    MetricDelta downProportion;
    MetricDelta neutralProportion;
    MetricDelta upProportion;
    MetricDelta leaderScore;
};

ComparisonResult Compare(const ArmEvidence& control,
                         const ArmEvidence& ablation,
                         std::string_view expectedAblationMask);

// Compatibility adapter for the Phase-6/7 consensus command and replication
// tooling, whose historical argument order was ABLATED:ENABLED and whose
// labels were control:treatment. The generic evaluator remains authoritative.
ComparisonResult CompareLegacyConsensusPair(
    const ArmEvidence& legacyAblatedControl,
    const ArmEvidence& legacyEnabledTreatment);

// Stable Phase 6 identity used both by the pair CLI and by Phase 7
// replication aggregation. Keeping one authority prevents the aggregate
// evaluator from reconstructing or approximating pair evidence identity.
std::string EvaluationIdentityCanonical(const ArmEvidence& control,
                                        const ArmEvidence& ablation,
                                        const ComparisonResult& result);
std::string EvaluationIdentityHash(const ArmEvidence& control,
                                   const ArmEvidence& ablation,
                                   const ComparisonResult& result);

std::pair<long long, long long> ParseExperimentIdPair(std::string_view text);
std::string DispositionText(Disposition value);

// Exit contract used by the CLI service. Database exceptions remain exit 2 in
// the established scheduler top-level handler.
int ExitCode(Disposition value);

} // namespace EA::FeatureAblationPairEvaluation
