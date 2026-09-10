#pragma once

#include "StrategyEvaluation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace EA::StrategyEvaluation
{

inline constexpr int kPhase19AnalysisConfigurationVersion = 1;
inline constexpr std::size_t kPhase19FeatureCount = 7;
inline constexpr const char* kPhase19AnalysisIdentity =
    "causal_state_interaction_analysis_v1";

enum class Phase19Feature : std::size_t
{
    volatilityRegime = 0,
    rollingRangeExpansion = 1,
    directionalRange = 2,
    directionalEfficiency = 3,
    returnSignPersistence = 4,
    returnDirectionImbalance = 5,
    historicalLevelProximity = 6
};

struct Phase19FeatureIdentity
{
    Phase19Feature feature;
    std::string family;
    std::string semanticName;
    std::size_t tensorColumn = 0;
    std::string runtimeIdentity;
    std::string entryTiming;

    bool operator==(const Phase19FeatureIdentity&) const = default;
};

const std::array<Phase19FeatureIdentity, kPhase19FeatureCount>&
Phase19FeatureIdentities();

// This type deliberately contains only the predeclared entry-time feature
// vector. Post-entry outcomes cannot be represented as Phase 19 state.
struct Phase19EntryState
{
    std::uint64_t observationOrdinal = 0;
    std::array<double, kPhase19FeatureCount> values{};

    bool operator==(const Phase19EntryState&) const = default;
};

struct Phase19QuartileBoundaries
{
    double q1 = 0.0;
    double q2 = 0.0;
    double q3 = 0.0;

    bool operator==(const Phase19QuartileBoundaries&) const = default;
};

Phase19QuartileBoundaries ComputePhase19QuartileBoundaries(
    std::vector<double> values);
std::size_t Phase19QuartileStratum(
    double value, const Phase19QuartileBoundaries& boundaries);
const char* Phase19SupportLabel(std::uint64_t activatedCount) noexcept;

// Authoritative categorical features, when present, are grouped by sorted
// exact category identity. Phase 19's current predeclared features are all
// continuous, so this is retained as the deterministic categorical contract.
std::vector<std::string> ComputePhase19CategoricalGroups(
    std::vector<std::string> categories);
std::size_t Phase19CategoricalStratum(
    const std::string& category,
    const std::vector<std::string>& sortedGroups);

enum class Phase19CohortRole
{
    primary,
    diagnostic
};

enum class Phase19WindowLabel
{
    discoveryHistory2025,
    temporalValidationHistory2026
};

struct Phase19AnalysisContext
{
    bool modelAcceptanceQualified = false;
};

struct Phase19StrategyMetrics
{
    std::uint64_t observationCount = 0;
    std::uint64_t actionableCount = 0;
    std::uint64_t activatedCount = 0;
    std::uint64_t winningCount = 0;
    std::uint64_t losingCount = 0;
    std::uint64_t zeroCount = 0;
    std::uint64_t stopHitCount = 0;
    std::uint64_t terminalExitCount = 0;
    double aggregateDirectionalLogReturn = 0.0;
    std::optional<double> averageDirectionalLogReturn;
    std::optional<double> winningRate;
    std::optional<double> losingRate;
    std::optional<double> zeroRate;
    std::optional<double> stopHitRate;
    std::optional<double> terminalExitRate;
    std::optional<double> averageHoldingDurationSeconds;
    std::optional<double> medianHoldingDurationSeconds;
    std::optional<double> maximumAdverseExcursion;
    std::optional<double> maximumFavorableExcursion;
    std::optional<double> maximumDrawdown;
    std::optional<double> averageEffectiveStopLogarithmicDistance;
    std::optional<double> averageStopMultiplier;
};

struct Phase19PairwiseDelta
{
    double aggregateReturnDelta = 0.0;
    double averageReturnDelta = 0.0;
    double winningRateDelta = 0.0;
    double losingRateDelta = 0.0;
    double stopHitRateDelta = 0.0;
    double terminalExitRateDelta = 0.0;
    double averageHoldingDurationSecondsDelta = 0.0;
    double maximumDrawdownDelta = 0.0;
};

struct Phase19PopulationResult
{
    std::string populationIdentity;
    Phase19StrategyMetrics fixed;
    Phase19StrategyMetrics extension;
    Phase19PairwiseDelta extensionMinusFixed;
};

struct Phase19StateStratumResult
{
    Phase19FeatureIdentity featureIdentity;
    Phase19QuartileBoundaries boundaries;
    std::size_t stratumOrdinal = 0;
    std::uint64_t activatedCount = 0;
    double shareOfActivated = 0.0;
    std::string supportLabel;
    Phase19StrategyMetrics fixed;
    Phase19StrategyMetrics extension;
    Phase19PairwiseDelta extensionMinusFixed;
};

struct Phase19StateInteractionAnalysisResult
{
    Phase19CohortRole cohortRole = Phase19CohortRole::primary;
    Phase19WindowLabel windowLabel =
        Phase19WindowLabel::discoveryHistory2025;
    bool modelAcceptanceQualified = false;
    std::string phase18BStrategyConfigurationHash;
    std::string analysisConfigurationCanonical;
    std::string analysisConfigurationHash;
    std::vector<Phase19PopulationResult> populations;
    std::vector<Phase19StateStratumResult> stateStrata;
    bool requiresReadOnlyTransaction = true;
    bool productionRowsModified = false;
    std::string canonicalLines;
    std::string resultHash;
};

struct Phase19InvocationContext
{
    bool inferenceMode = false;
    bool hasExplicitModel = false;
    bool inferAll = false;
    bool schedulerExperiment = false;
    bool schedulerCheckpointEvaluation = false;
    bool schedulerWorkerAttempt = false;
    bool frozenOutcome = false;
};

void ValidatePhase19Invocation(const Phase19InvocationContext& context);

Phase19StateInteractionAnalysisResult EvaluatePhase19StateInteractionAnalysis(
    const AuthoritativeMarketPath& marketPath,
    const std::vector<Phase19EntryState>& entryStates,
    Phase19AnalysisContext context = {});

} // namespace EA::StrategyEvaluation
