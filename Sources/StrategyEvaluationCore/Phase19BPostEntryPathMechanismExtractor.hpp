#pragma once

#include "StrategyEvaluation.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace EA::StrategyEvaluation
{

inline constexpr int kPhase19BPathMechanismSchemaVersion = 1;
inline constexpr const char* kPhase19BPathMechanismSchemaIdentity =
    "phase19b_post_entry_path_mechanism_v1";
inline constexpr const char* kPhase19BComparisonSemantics =
    "exact_binary64_delta_sign_v1";

enum class Phase19BOutcomeClass
{
    saved,
    harmed,
    unchanged
};

const char* Phase19BOutcomeClassText(Phase19BOutcomeClass value) noexcept;

struct Phase19BInvocationContext
{
    bool inferenceMode = false;
    bool hasExplicitModel = false;
    bool inferAll = false;
    bool schedulerExperiment = false;
    bool schedulerCheckpointEvaluation = false;
    bool schedulerWorkerAttempt = false;
    bool frozenOutcome = false;
};

void ValidatePhase19BInvocation(const Phase19BInvocationContext& context);

struct Phase19BExtractionContext
{
    std::optional<long long> experimentId;
    bool readOnlyTransactionEnforced = false;
};

struct Phase19BPathMechanismObservation
{
    std::string observationId;
    std::uint64_t observationOrdinal = 0;
    std::uint64_t inferenceWindowStartRow = 0;
    std::uint64_t entrySourceRow = 0;
    std::uint64_t terminalSourceRow = 0;
    std::int64_t entryTimestampUnixSeconds = 0;
    std::int64_t terminalTimestampUnixSeconds = 0;
    int predictedClass = InferenceProfitability::kNeutralClass;
    PositionDirection direction = PositionDirection::flat;
    double directionalProbability = 0.0;
    double normalizedDirectionalConfidence = 0.0;
    bool activated = false;
    double baseStopLogarithmicDistance = 0.0;
    double extensionMultiplier = 0.0;
    double extensionStopLogarithmicDistance = 0.0;
    double fixedStrategyReturn = 0.0;
    double extensionStrategyReturn = 0.0;
    double extensionMinusFixedReturn = 0.0;
    Phase19BOutcomeClass outcomeClass = Phase19BOutcomeClass::unchanged;
    bool fixedStopHit = false;
    bool extensionStopHit = false;
    StrategyExitReason fixedExitReason = StrategyExitReason::noAction;
    StrategyExitReason extensionExitReason = StrategyExitReason::noAction;
    double maximumAdverseExcursion = 0.0;
    double maximumFavorableExcursion = 0.0;
    std::optional<std::uint64_t> maximumAdversePathPointOrdinal;
    std::optional<std::uint64_t> maximumFavorablePathPointOrdinal;
    std::optional<std::uint64_t> maximumAdverseSourceRow;
    std::optional<std::uint64_t> maximumFavorableSourceRow;
    std::optional<std::int64_t> maximumAdverseTimestampUnixSeconds;
    std::optional<std::int64_t> maximumFavorableTimestampUnixSeconds;
    std::optional<std::uint64_t> fixedStopPathPointOrdinal;
    std::optional<std::uint64_t> extensionStopPathPointOrdinal;
    std::optional<std::uint64_t> fixedStopSourceRow;
    std::optional<std::uint64_t> extensionStopSourceRow;
    std::optional<std::int64_t> fixedStopTimestampUnixSeconds;
    std::optional<std::int64_t> extensionStopTimestampUnixSeconds;
    std::optional<std::uint64_t> barsFromEntryToFixedStop;
    std::optional<std::uint64_t> barsFromEntryToExtensionStop;
    std::optional<double> fixedStopHorizonFraction;
    std::optional<double> extensionStopHorizonFraction;
    std::optional<std::uint64_t> barsRemainingAfterFixedStop;
    bool recoveredToEntry = false;
    std::optional<std::uint64_t> recoveryToEntryPathPointOrdinal;
    std::optional<std::uint64_t> recoveryToEntrySourceRow;
    std::optional<std::int64_t> recoveryToEntryTimestampUnixSeconds;
    std::optional<std::uint64_t> barsFromFixedStopToRecovery;
    std::optional<double> recoveryToEntryHorizonFraction;
    bool achievedFavorableBaseStopAfterFixedBreach = false;
    std::optional<std::uint64_t> favorableBaseStopPathPointOrdinal;
    std::optional<std::uint64_t> favorableBaseStopSourceRow;
    std::optional<std::int64_t>
        favorableBaseStopTimestampUnixSeconds;
    double terminalDirectionalReturn = 0.0;
    std::string terminalDirectionalClass;
    std::uint64_t barsInAuthoritativePath = 0;
    double adverseExcursionBeyondFixedStop = 0.0;
    std::optional<double> requiredExtraRoomBeforeRecovery;
    std::optional<double> maximumAdverseHorizonFraction;
    std::optional<double> maximumFavorableHorizonFraction;
};

struct Phase19BPathTracePoint
{
    std::string observationId;
    std::uint64_t observationOrdinal = 0;
    std::uint64_t pathPointOrdinal = 0;
    std::uint64_t sourceRow = 0;
    std::int64_t timestampUnixSeconds = 0;
    double horizonFraction = 0.0;
    double directionalOpenLogReturn = 0.0;
    double directionalAdverseExtremeLogReturn = 0.0;
    double directionalFavorableExtremeLogReturn = 0.0;
    double directionalCloseLogReturn = 0.0;
};

struct Phase19BPathMechanismResult
{
    std::string schemaIdentity;
    int schemaVersion = kPhase19BPathMechanismSchemaVersion;
    std::optional<long long> experimentId;
    long long modelId = 0;
    std::string symbol;
    std::string cohortRole;
    std::uint64_t predictionHorizon = 0;
    std::string windowStart;
    std::string windowEnd;
    std::uint64_t totalObservationCount = 0;
    std::uint64_t totalActionableCount = 0;
    std::uint64_t activatedCount = 0;
    std::uint64_t savedCount = 0;
    std::uint64_t harmedCount = 0;
    std::uint64_t unchangedCount = 0;
    double activatedFixedReturnSum = 0.0;
    double activatedExtensionReturnSum = 0.0;
    double perObservationDeltaSum = 0.0;
    double frozenPhase18BAggregateDelta = 0.0;
    double aggregateReconciliationResidual = 0.0;
    bool requiresReadOnlyTransaction = true;
    bool productionRowsModified = false;
    std::string marketPathHash;
    std::string phase18BExperimentIdentityHash;
    std::string metadataTsv;
    std::string observationsTsv;
    std::string pathTraceTsv;
    std::string resultHash;
    std::vector<Phase19BPathMechanismObservation> observations;
    std::vector<Phase19BPathTracePoint> pathTrace;
};

// Extracts post-entry descriptive outcomes only. Activation and both economic
// outcomes are delegated to the frozen Phase 18B controlled evaluation.
Phase19BPathMechanismResult ExtractPhase19BPostEntryPathMechanism(
    const AuthoritativeMarketPath& marketPath,
    Phase19BExtractionContext context);

} // namespace EA::StrategyEvaluation
