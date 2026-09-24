#pragma once

#include "FeatureAblationPairEvaluation.hpp"

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace EA::ExperimentPairComparison
{

enum class Status
{
    ComparableComplete,
    ComparableIncomplete,
    IncompatibleScientificIdentity,
    InvalidEvidence
};

struct IdentityField
{
    std::string name;
    // nullopt means that the evidence object from which this field is
    // derived does not exist.  An engaged value remains authoritative even
    // when it is "0", "false", empty, or the persisted NULL token.
    std::optional<std::string> value;
    bool operator==(const IdentityField&) const = default;
};

struct MetricObservation
{
    std::optional<double> inferenceAccuracy;
    std::optional<double> acceptRate;
    std::optional<double> acceptAccuracy;
    std::optional<double> leaderScore;
    std::optional<double> predictionCount;
    std::optional<double> actionableCount;
    std::optional<double> winningActionableCount;
    std::optional<double> losingActionableCount;
    std::optional<double> actionablePercentage;
    std::optional<double> winPercentage;
    std::optional<double> grossPositiveReturn;
    std::optional<double> grossNegativeReturn;
    std::optional<double> aggregateReturn;
    std::optional<double> averageReturnPerAction;
};

struct ArmResultSet
{
    long long experimentId = 0;
    std::vector<IdentityField> scientificIdentity;
    bool finalModelAvailable = false;
    bool trainingProvenanceAvailable = false;
    bool finalInferenceAvailable = false;
    bool inferenceProvenanceAvailable = false;
    bool finalAnalysisAvailable = false;
    bool profitabilityObservationAvailable = false;
    MetricObservation metrics;
};

struct Request
{
    std::string armALabel = "experiment_a";
    std::string armBLabel = "experiment_b";
    // Every named field must exist in both arms and actually differ. All
    // other identity fields must compare exactly equal.
    std::vector<std::string> intentionalDifferenceFields;
};

struct IdentityDifference
{
    std::string field;
    std::string armA;
    std::string armB;
    bool intentional = false;
    bool operator==(const IdentityDifference&) const = default;
};

struct MetricDelta
{
    std::optional<double> armA;
    std::optional<double> armB;
    // Positive means the metric is higher for B than for A.
    std::optional<double> armBMinusArmA;
    bool operator==(const MetricDelta&) const = default;
};

struct ComparisonResult
{
    Status status = Status::ComparableIncomplete;
    long long experimentAId = 0;
    long long experimentBId = 0;
    std::string armALabel;
    std::string armBLabel;
    std::vector<IdentityField> armAScientificIdentity;
    std::vector<IdentityField> armBScientificIdentity;
    std::vector<IdentityDifference> intentionalDifferences;
    std::vector<IdentityDifference> unexpectedDifferences;
    std::vector<std::string> invalidReasons;
    std::vector<std::string> incompleteReasons;

    bool armAFinalModelAvailable = false;
    bool armATrainingProvenanceAvailable = false;
    bool armAFinalInferenceAvailable = false;
    bool armAInferenceProvenanceAvailable = false;
    bool armAFinalAnalysisAvailable = false;
    bool armAProfitabilityObservationAvailable = false;
    bool armBFinalModelAvailable = false;
    bool armBTrainingProvenanceAvailable = false;
    bool armBFinalInferenceAvailable = false;
    bool armBInferenceProvenanceAvailable = false;
    bool armBFinalAnalysisAvailable = false;
    bool armBProfitabilityObservationAvailable = false;

    MetricDelta inferenceAccuracy;
    MetricDelta acceptRate;
    MetricDelta acceptAccuracy;
    MetricDelta leaderScore;
    MetricDelta predictionCount;
    MetricDelta actionableCount;
    MetricDelta winningActionableCount;
    MetricDelta losingActionableCount;
    MetricDelta actionablePercentage;
    MetricDelta winPercentage;
    MetricDelta grossPositiveReturn;
    MetricDelta grossNegativeReturn;
    MetricDelta aggregateReturn;
    MetricDelta averageReturnPerAction;
};

// The ordered metric catalog is shared by pair rendering and replication
// aggregation so neither layer can silently diverge from the other.
struct MetricDefinition
{
    std::string_view name;
    MetricDelta ComparisonResult::* member;
};

const std::array<MetricDefinition, 14>& MetricDefinitions();

// Adapts the existing authoritative feature-ablation evidence without loading
// or mutating external state. Scientific identity values retain exact
// persisted equality semantics.
ArmResultSet MakeArmResultSet(
    const FeatureAblationPairEvaluation::ArmEvidence& evidence);

ComparisonResult Compare(const ArmResultSet& armA,
                         const ArmResultSet& armB,
                         const Request& request);

std::string Render(const ComparisonResult& result);
std::string RenderSummary(const ComparisonResult& result);
std::string StatusText(Status status);

} // namespace EA::ExperimentPairComparison
