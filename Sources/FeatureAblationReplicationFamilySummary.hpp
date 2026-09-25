#pragma once

#include "FeatureAblationPairEvaluation.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace EA::FeatureAblationReplicationFamilySummary
{

namespace Pair = FeatureAblationPairEvaluation;

// These categories preserve the pair evaluator's distinct observation types.
enum class MetricCategory
{
    ModelPerformance,
    BehavioralCoverage,
    Profitability
};

enum class Metric
{
    PredictionCount,
    PredictedDownCount,
    PredictedNeutralCount,
    PredictedUpCount,
    AcceptedPredictionCount,
    ActionableCount,
    AggregateProfitability,
    AverageProfitability,
    InferenceAccuracy,
    AcceptAccuracy,
    AcceptRate,
    DownProportion,
    NeutralProportion,
    UpProportion,
    LeaderScore
};

// This is an opaque identity supplied by the later evidence adapter. It is
// deliberately not an experiment ID and cannot be used to discover or mutate
// experiments. Equality is exact byte equality.
struct ReplicationMember
{
    unsigned int freshInitializationSeed = 0;
    std::string controlExperimentIdentity;
    std::string ablationExperimentIdentity;
    std::string pairMemberIdentity;
    std::string interventionIdentity;
    std::string scientificConfigurationIdentity;
    bool pairValid = false;
    bool pairReady = false;
    bool pairComplete = false;
    Pair::ComparisonResult pairEvidence;
};

struct DescriptiveSummary
{
    std::size_t memberCount = 0;
    std::size_t positiveCount = 0;
    std::size_t zeroCount = 0;
    std::size_t negativeCount = 0;
    std::optional<double> meanDelta;
    std::optional<double> medianDelta;
    std::optional<double> minimumDelta;
    std::optional<double> maximumDelta;

    // Population standard deviation: sqrt(sum((x - mean)^2) / N).
    std::optional<double> populationStandardDeviation;
};

struct LeaveOneOutSummary
{
    unsigned int omittedFreshInitializationSeed = 0;
    std::string omittedPairMemberIdentity;
    DescriptiveSummary summary;
};

struct MetricFamilySummary
{
    Metric metric = Metric::PredictionCount;
    MetricCategory category = MetricCategory::BehavioralCoverage;
    // No value means this pair metric was unavailable for every included
    // member. Partially available metrics are rejected before this point.
    std::optional<DescriptiveSummary> descriptive;
    std::vector<LeaveOneOutSummary> leaveOneOut;
};

struct FamilySummary
{
    std::string interventionIdentity;
    std::string scientificConfigurationIdentity;
    std::vector<ReplicationMember> members;
    std::vector<MetricFamilySummary> metrics;
};

// Fail-closed, in-memory evaluation of already evaluated pair evidence.
// Throws std::invalid_argument for an empty or inconsistent family, missing
// identity/readiness, duplicate evidence, partially available metrics, or a
// non-finite available delta.
FamilySummary Summarize(std::vector<ReplicationMember> members);

std::string MetricText(Metric metric);
std::string MetricCategoryText(MetricCategory category);

} // namespace EA::FeatureAblationReplicationFamilySummary
