#include "FeatureAblationReplicationFamilySummary.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace EA::FeatureAblationReplicationFamilySummary
{
namespace
{

struct MetricDefinition
{
    Metric metric;
    MetricCategory category;
    const Pair::MetricDelta Pair::ComparisonResult::* delta;
};

constexpr MetricDefinition kMetricDefinitions[] = {
    {Metric::PredictionCount, MetricCategory::BehavioralCoverage,
     &Pair::ComparisonResult::predictionCount},
    {Metric::PredictedDownCount, MetricCategory::BehavioralCoverage,
     &Pair::ComparisonResult::predictedDownCount},
    {Metric::PredictedNeutralCount, MetricCategory::BehavioralCoverage,
     &Pair::ComparisonResult::predictedNeutralCount},
    {Metric::PredictedUpCount, MetricCategory::BehavioralCoverage,
     &Pair::ComparisonResult::predictedUpCount},
    {Metric::AcceptedPredictionCount, MetricCategory::BehavioralCoverage,
     &Pair::ComparisonResult::acceptedPredictionCount},
    {Metric::ActionableCount, MetricCategory::Profitability,
     &Pair::ComparisonResult::actionableCount},
    {Metric::AggregateProfitability, MetricCategory::Profitability,
     &Pair::ComparisonResult::aggregateProfitability},
    {Metric::AverageProfitability, MetricCategory::Profitability,
     &Pair::ComparisonResult::averageProfitability},
    {Metric::InferenceAccuracy, MetricCategory::ModelPerformance,
     &Pair::ComparisonResult::inferenceAccuracy},
    {Metric::AcceptAccuracy, MetricCategory::ModelPerformance,
     &Pair::ComparisonResult::acceptAccuracy},
    {Metric::AcceptRate, MetricCategory::BehavioralCoverage,
     &Pair::ComparisonResult::acceptRate},
    {Metric::DownProportion, MetricCategory::BehavioralCoverage,
     &Pair::ComparisonResult::downProportion},
    {Metric::NeutralProportion, MetricCategory::BehavioralCoverage,
     &Pair::ComparisonResult::neutralProportion},
    {Metric::UpProportion, MetricCategory::BehavioralCoverage,
     &Pair::ComparisonResult::upProportion},
    {Metric::LeaderScore, MetricCategory::ModelPerformance,
     &Pair::ComparisonResult::leaderScore},
};

DescriptiveSummary Describe(std::vector<double> values)
{
    if (values.empty())
        throw std::invalid_argument("replication_family_describe_empty_values");

    DescriptiveSummary result;
    result.memberCount = values.size();
    long double sum = 0.0L;
    for (const double value : values)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("replication_family_metric_nonfinite");
        sum += static_cast<long double>(value);
        if (value > 0.0) ++result.positiveCount;
        else if (value < 0.0) ++result.negativeCount;
        else ++result.zeroCount;
    }
    if (!std::isfinite(sum))
        throw std::invalid_argument("replication_family_metric_sum_nonfinite");

    const long double count = static_cast<long double>(values.size());
    const long double mean = sum / count;
    const double publicMean = static_cast<double>(mean);
    if (!std::isfinite(mean) || !std::isfinite(publicMean))
        throw std::invalid_argument("replication_family_metric_mean_nonfinite");
    result.meanDelta = publicMean;
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    result.medianDelta = values.size() % 2 == 0
        ? values[middle - 1] / 2.0 + values[middle] / 2.0
        : values[middle];
    result.minimumDelta = values.front();
    result.maximumDelta = values.back();

    long double squaredDeviationSum = 0.0L;
    for (const double value : values)
    {
        const long double deviation = static_cast<long double>(value) - mean;
        squaredDeviationSum += deviation * deviation;
    }
    if (!std::isfinite(squaredDeviationSum))
        throw std::invalid_argument("replication_family_metric_dispersion_nonfinite");
    const long double populationStandardDeviation =
        std::sqrt(squaredDeviationSum / count);
    const double publicPopulationStandardDeviation =
        static_cast<double>(populationStandardDeviation);
    if (!std::isfinite(populationStandardDeviation) ||
        !std::isfinite(publicPopulationStandardDeviation))
        throw std::invalid_argument("replication_family_metric_dispersion_nonfinite");
    result.populationStandardDeviation = publicPopulationStandardDeviation;
    return result;
}

void RequireIdentity(const std::string& value, const char* reason)
{
    if (value.empty()) throw std::invalid_argument(reason);
}

} // namespace

FamilySummary Summarize(std::vector<ReplicationMember> members)
{
    if (members.empty())
        throw std::invalid_argument("replication_family_empty");

    const std::string intervention = members.front().interventionIdentity;
    const std::string configuration =
        members.front().scientificConfigurationIdentity;
    RequireIdentity(intervention, "replication_family_intervention_identity_missing");
    RequireIdentity(configuration,
                    "replication_family_scientific_configuration_identity_missing");

    std::set<unsigned int> seeds;
    std::set<std::string> memberIdentities;
    std::set<std::pair<std::string, std::string>> armPairs;
    std::set<std::string> experimentIdentities;
    for (const auto& member : members)
    {
        RequireIdentity(member.controlExperimentIdentity,
                        "replication_family_control_identity_missing");
        RequireIdentity(member.ablationExperimentIdentity,
                        "replication_family_ablation_experiment_identity_missing");
        RequireIdentity(member.pairMemberIdentity,
                        "replication_family_member_identity_missing");
        RequireIdentity(member.interventionIdentity,
                        "replication_family_intervention_identity_missing");
        RequireIdentity(member.scientificConfigurationIdentity,
                        "replication_family_scientific_configuration_identity_missing");
        if (member.controlExperimentIdentity == member.ablationExperimentIdentity)
            throw std::invalid_argument("replication_family_arm_identity_not_distinct");
        if (!seeds.insert(member.freshInitializationSeed).second)
            throw std::invalid_argument("replication_family_duplicate_seed");
        if (!memberIdentities.insert(member.pairMemberIdentity).second ||
            !armPairs.insert({member.controlExperimentIdentity,
                              member.ablationExperimentIdentity}).second)
            throw std::invalid_argument("replication_family_duplicate_member");
        if (!experimentIdentities.insert(member.controlExperimentIdentity).second ||
            !experimentIdentities.insert(member.ablationExperimentIdentity).second)
            throw std::invalid_argument("replication_family_duplicate_member");
        if (member.interventionIdentity != intervention)
            throw std::invalid_argument("replication_family_intervention_identity_mismatch");
        if (member.scientificConfigurationIdentity != configuration)
            throw std::invalid_argument(
                "replication_family_scientific_configuration_identity_mismatch");
        if (!member.pairValid)
            throw std::invalid_argument("replication_family_member_invalid");
        if (!member.pairReady)
            throw std::invalid_argument("replication_family_member_not_ready");
        if (!member.pairComplete)
            throw std::invalid_argument("replication_family_member_incomplete");
    }

    std::sort(members.begin(), members.end(),
              [](const ReplicationMember& left, const ReplicationMember& right)
              {
                  return left.freshInitializationSeed <
                      right.freshInitializationSeed;
              });

    FamilySummary result;
    result.interventionIdentity = intervention;
    result.scientificConfigurationIdentity = configuration;
    result.members = std::move(members);

    for (const auto& definition : kMetricDefinitions)
    {
        std::vector<double> values;
        values.reserve(result.members.size());
        std::size_t available = 0;
        for (const auto& member : result.members)
        {
            const auto& delta = member.pairEvidence.*definition.delta;
            if (delta.controlMinusAblation)
            {
                ++available;
                if (!std::isfinite(*delta.controlMinusAblation))
                    throw std::invalid_argument("replication_family_metric_nonfinite");
                values.push_back(*delta.controlMinusAblation);
            }
        }
        if (available != 0 && available != result.members.size())
            throw std::invalid_argument(
                "replication_family_metric_availability_inconsistent:" +
                MetricText(definition.metric));

        MetricFamilySummary summary;
        summary.metric = definition.metric;
        summary.category = definition.category;
        if (available != 0)
        {
            summary.descriptive = Describe(values);
            for (std::size_t omitted = 0; omitted < result.members.size(); ++omitted)
            {
                std::vector<double> remaining;
                remaining.reserve(result.members.size() - 1);
                for (std::size_t index = 0; index < values.size(); ++index)
                    if (index != omitted) remaining.push_back(values[index]);
                // A one-member family has no remaining evidence. The omitted
                // member is still reported, with an explicitly empty summary.
                LeaveOneOutSummary leaveOneOut;
                leaveOneOut.omittedFreshInitializationSeed =
                    result.members[omitted].freshInitializationSeed;
                leaveOneOut.omittedPairMemberIdentity =
                    result.members[omitted].pairMemberIdentity;
                if (!remaining.empty()) leaveOneOut.summary = Describe(remaining);
                summary.leaveOneOut.push_back(std::move(leaveOneOut));
            }
        }
        result.metrics.push_back(std::move(summary));
    }
    return result;
}

std::string MetricText(Metric metric)
{
    switch (metric)
    {
        case Metric::PredictionCount: return "prediction_count";
        case Metric::PredictedDownCount: return "predicted_down_count";
        case Metric::PredictedNeutralCount: return "predicted_neutral_count";
        case Metric::PredictedUpCount: return "predicted_up_count";
        case Metric::AcceptedPredictionCount: return "accepted_prediction_count";
        case Metric::ActionableCount: return "actionable_count";
        case Metric::AggregateProfitability: return "aggregate_terminal_horizon_log_return";
        case Metric::AverageProfitability: return "average_terminal_horizon_log_return_per_actionable_prediction";
        case Metric::InferenceAccuracy: return "inference_accuracy";
        case Metric::AcceptAccuracy: return "accept_accuracy";
        case Metric::AcceptRate: return "accept_rate";
        case Metric::DownProportion: return "down_proportion";
        case Metric::NeutralProportion: return "neutral_proportion";
        case Metric::UpProportion: return "up_proportion";
        case Metric::LeaderScore: return "leader_score";
    }
    throw std::invalid_argument("replication_family_unknown_metric");
}

std::string MetricCategoryText(MetricCategory category)
{
    switch (category)
    {
        case MetricCategory::ModelPerformance: return "model_performance";
        case MetricCategory::BehavioralCoverage: return "behavioral_coverage";
        case MetricCategory::Profitability: return "profitability";
    }
    throw std::invalid_argument("replication_family_unknown_metric_category");
}

} // namespace EA::FeatureAblationReplicationFamilySummary
