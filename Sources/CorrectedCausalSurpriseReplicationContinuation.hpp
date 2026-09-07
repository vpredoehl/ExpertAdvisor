#pragma once

#include "FeatureAblationReplicationEvaluation.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace EA::CorrectedCausalSurpriseReplicationContinuation
{

namespace Replication = FeatureAblationReplicationEvaluation;

inline constexpr int kPlanSemanticVersion = 1;
inline constexpr int kScientificPolicyVersion = 1;
inline constexpr std::size_t kFollowOnPairCount = 2;

struct ScientificConfiguration
{
    std::string initialSymbol;
    int initialPredictionHorizon = 0;
    int targetEpochs = 0;
    double threshold = 0.0;
    std::optional<double> coreLearningRateMultiplier;
    std::optional<double> headLearningRateMultiplier;
    int checkpointInterval = 0;
    std::string trainingObjectiveCanonical;
    std::string trainingObjectiveHash;
    std::string trainStart;
    std::string trainEnd;
    std::string inferenceStart;
    std::string inferenceEnd;
    std::string donchianMode;
    std::string featureWarmupScope;
    int donchianLookback = 0;
    int modelInputWidth = 0;
    int semanticLayoutVersion = 0;
    long long economicCalendarSnapshotId = 0;
    std::string economicCalendarSnapshotHash;
    double baseLearningRate = 0.0;
    int batchSize = 0;
    std::optional<unsigned int> freshInitializationSeed;
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
    std::string continuationPolicyScientificIdentity;
    std::string schedulerPriority = "high";
    bool operator==(const ScientificConfiguration&) const = default;
};

struct Arm
{
    std::string role;
    std::string featureAblationMask;
    std::string scientificIdentityCanonical;
    std::string scientificIdentityHash;
    bool operator==(const Arm&) const = default;
};

struct Pair
{
    std::size_t ordinal = 0;
    std::string symbol;
    int predictionHorizon = 0;
    std::string independentDimension;
    std::string replicationUnitCanonical;
    std::string replicationUnitHash;
    Arm control;
    Arm treatment;
    bool operator==(const Pair&) const = default;
};

struct Plan
{
    int semanticVersion = kPlanSemanticVersion;
    int scientificPolicyVersion = kScientificPolicyVersion;
    ScientificConfiguration configuration;
    std::vector<Pair> pairs;
    std::string canonical;
    std::string hash;
    bool operator==(const Plan&) const = default;
};

enum class NextAction
{
    AwaitPairCompletion,
    InvalidPairRequiresReview,
    PrepareAdditionalReplications,
    ReplicationThresholdSatisfied
};

struct Gate
{
    std::size_t correctedValidPairCount = 0;
    std::size_t historicalPreFixPairCount = 0;
    std::size_t invalidOrIncompatiblePairCount = 0;
    int minimumValidReplications =
        Replication::kMinimumValidReplications;
    bool correctedReplicationMinimumSatisfied = false;
    bool anchorPairComplete = false;
    bool anchorPairValidCorrected = false;
    NextAction nextAction = NextAction::AwaitPairCompletion;
    std::optional<std::size_t> nextPlanPairOrdinal;
    std::vector<std::string> reasons;
};

Plan MakePlan(const ScientificConfiguration& configuration);
void ValidatePlan(const Plan& plan);
Gate EvaluateGate(const Replication::ReplicationEvaluation& evaluation,
                  const Replication::MemberEvaluation& anchorPair);

std::string NextActionText(NextAction value);
std::string RenderPlan(const Plan& plan);
std::string RenderGate(const Gate& gate);

} // namespace EA::CorrectedCausalSurpriseReplicationContinuation
