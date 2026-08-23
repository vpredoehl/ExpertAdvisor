#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

// Phase 3C is advisory only. These constants are intentionally not policy
// inputs and there is no API for enabling a nonzero contribution.
inline constexpr double kPhase3CProfitabilityWeight = 0.0;
inline constexpr double kPhase3CProfitabilityScoreContribution = 0.0;
static_assert(kPhase3CProfitabilityWeight == 0.0);
static_assert(kPhase3CProfitabilityScoreContribution == 0.0);

struct ProfitabilitySemanticIdentity
{
    std::string canonical;
    std::string hash;
    int version = 0;

    bool operator==(const ProfitabilitySemanticIdentity&) const = default;
};

// An immutable analytical projection of one authoritative profitability row
// plus the material semantics obtained from its exact inference/evaluation
// context. It does not replace the persisted observation contract.
struct ProfitabilityObservation
{
    long long profitabilityObservationId = -1;
    long long experimentId = -1;
    long long modelId = -1;
    long long inferenceEvalResultId = -1;

    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
    std::string sourceContentHash;
    std::string observationIdentityCanonical;
    std::string observationIdentityHash;

    std::string inferenceScope;
    std::string inferenceStart;
    std::string inferenceEnd;
    std::string symbol;
    int predictionHorizon = 0;
    int modelInputWidth = 0;

    ProfitabilitySemanticIdentity featureClassIdentity;
    ProfitabilitySemanticIdentity inferenceEvaluationSemanticIdentity;
    ProfitabilitySemanticIdentity scoringSemanticIdentity;
    ProfitabilitySemanticIdentity evaluationSemanticIdentity;

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

std::optional<std::string> ValidateProfitabilityObservation(
    const ProfitabilityObservation& observation);

// The exact comparability cohort. Source-content and observation identities
// distinguish members but are deliberately not scientific grouping keys.
struct ProfitabilityPopulationIdentity
{
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
    std::string inferenceScope;
    std::string inferenceStart;
    std::string inferenceEnd;
    std::string symbol;
    int predictionHorizon = 0;
    int modelInputWidth = 0;
    ProfitabilitySemanticIdentity featureClassIdentity;
    ProfitabilitySemanticIdentity inferenceEvaluationSemanticIdentity;
    ProfitabilitySemanticIdentity scoringSemanticIdentity;
    ProfitabilitySemanticIdentity evaluationSemanticIdentity;

    bool operator==(const ProfitabilityPopulationIdentity&) const = default;
};

std::string ProfitabilityPopulationIdentityCanonicalText(
    const ProfitabilityPopulationIdentity& identity);
std::string ProfitabilityPopulationIdentityHash(
    const ProfitabilityPopulationIdentity& identity);

struct ProfitabilityNormalizationPolicy
{
    int version = 1;
    std::size_t minimumAnalyzablePopulationSize = 5;
    std::uint64_t supportHalfSaturationActionableCount = 100;
    std::vector<double> quantiles{0.05, 0.25, 0.50, 0.75, 0.95};
};

std::optional<std::string> ValidateProfitabilityNormalizationPolicy(
    const ProfitabilityNormalizationPolicy& policy);
std::string ProfitabilityNormalizationPolicyCanonicalText(
    const ProfitabilityNormalizationPolicy& policy);
std::string ProfitabilityNormalizationPolicyHash(
    const ProfitabilityNormalizationPolicy& policy);

enum class ProfitabilityPopulationState
{
    valid,
    insufficientPopulation,
    noAnalyzableObservations,
    invalid
};

std::string ProfitabilityPopulationStateText(
    ProfitabilityPopulationState state);

struct ProfitabilityDistributionQuantile
{
    double probability = 0.0;
    double value = 0.0;

    bool operator==(const ProfitabilityDistributionQuantile&) const = default;
};

struct ProfitabilityDistributionSummary
{
    ProfitabilityPopulationState state = ProfitabilityPopulationState::invalid;
    std::string reason;
    std::optional<ProfitabilityPopulationIdentity> populationIdentity;
    std::string populationIdentityCanonical;
    std::string populationIdentityHash;
    std::string membershipCanonical;
    std::string membershipHash;
    std::string normalizationPolicyCanonical;
    std::string normalizationPolicyHash;

    std::size_t populationCount = 0;
    std::size_t analyzablePopulationCount = 0;
    std::size_t zeroActionableCount = 0;
    std::uint64_t totalPredictionCount = 0;
    std::uint64_t totalActionableCount = 0;
    std::size_t negativeCount = 0;
    std::size_t zeroCount = 0;
    std::size_t positiveCount = 0;

    std::optional<double> minimum;
    std::optional<double> maximum;
    std::optional<double> mean;
    std::optional<double> median;
    std::optional<double> populationStandardDeviation;
    std::optional<double> medianAbsoluteDeviation;
    std::vector<ProfitabilityDistributionQuantile> quantiles;

    std::string canonical;
    std::string hash;
};

enum class ProfitabilityNormalizationState
{
    available,
    insufficientPopulation,
    zeroActionable,
    invalidPopulation
};

std::string ProfitabilityNormalizationStateText(
    ProfitabilityNormalizationState state);

struct ProfitabilityNormalizationResult
{
    static constexpr double profitabilityWeight =
        kPhase3CProfitabilityWeight;
    static constexpr double profitabilityScoreContribution =
        kPhase3CProfitabilityScoreContribution;

    long long profitabilityObservationId = -1;
    ProfitabilityNormalizationState state =
        ProfitabilityNormalizationState::invalidPopulation;
    std::string reason;
    std::optional<double> rawProfitabilityMetric;
    std::string populationIdentityHash;
    std::string membershipHash;
    std::size_t populationSize = 0;
    std::size_t analyzablePopulationSize = 0;
    std::uint64_t actionableCount = 0;
    std::optional<double> empiricalMidrankPercentile;
    std::optional<double> boundedCandidateMetric;
    double supportReliability = 0.0;
    std::string canonical;
    std::string hash;
};

struct ProfitabilityDistributionAnalysis
{
    ProfitabilityDistributionSummary summary;
    std::vector<ProfitabilityNormalizationResult> normalizationResults;
    std::string canonical;
    std::string hash;
};

// Validates the entire requested population before analysis. Incompatible or
// malformed rows invalidate the whole request; no row is silently discarded.
ProfitabilityDistributionAnalysis AnalyzeProfitabilityDistribution(
    std::vector<ProfitabilityObservation> observations,
    const ProfitabilityNormalizationPolicy& policy = {});

} // namespace EA::ExperimentRecommendation
