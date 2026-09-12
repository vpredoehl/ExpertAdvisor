#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace EA::StrategyEvaluation
{

inline constexpr int kPhase19CCausalDatasetSchemaVersion = 1;
inline constexpr const char* kPhase19CCausalDatasetSchemaIdentity =
    "phase19c_causal_path_predictability_dataset_v1";
inline constexpr const char* kPhase19CAnalysisPlanIdentity =
    "phase19c_frozen_analysis_plan_v1";

struct Phase19CPredictorIdentity
{
    std::string name;
    std::string source;
    std::string causalTiming;
    std::uint64_t modelInputColumn = 0;
    bool categorical = false;

    bool operator==(const Phase19CPredictorIdentity&) const = default;
};

struct Phase19CCausalEntryState
{
    std::uint64_t observationOrdinal = 0;
    std::uint64_t entrySourceRow = 0;
    std::int64_t entryTimestampUnixSeconds = 0;
    int predictedClass = 1;
    std::string direction;
    double directionalProbability = 0.0;
    double normalizedDirectionalConfidence = 0.0;
    std::vector<double> predictorValues;
};

struct Phase19CCausalDatasetContext
{
    long long modelId = 0;
    std::optional<long long> experimentId;
    std::string symbol;
    std::string cohortRole;
    std::uint64_t predictionHorizon = 0;
    std::string windowStart;
    std::string windowEnd;
    std::size_t modelInputWidth = 0;
    int semanticLayoutVersion = 0;
    std::string featureAblationIdentity;
    bool readOnlyTransactionEnforced = false;
};

struct Phase19CCausalDatasetResult
{
    std::string schemaIdentity;
    int schemaVersion = kPhase19CCausalDatasetSchemaVersion;
    std::uint64_t activatedCount = 0;
    std::uint64_t savedCount = 0;
    std::uint64_t harmedCount = 0;
    std::uint64_t unchangedCount = 0;
    std::uint64_t joinedCount = 0;
    std::uint64_t joinFailureCount = 0;
    std::size_t predictorCount = 0;
    std::string featureLayoutIdentity;
    std::string featureLayoutHash;
    std::string sourcePhase19BObservationsHash;
    std::string sourcePhase19BResultHash;
    std::string datasetHash;
    std::string resultHash;
    std::string datasetTsv;
    std::string metadataTsv;
    std::string predictorsTsv;
};

void ValidatePhase19CInvocation(bool inferenceMode,
                                bool hasExplicitModel,
                                bool inferAll,
                                bool schedulerContext,
                                bool hasArtifactDirectory);

// Reads only Phase 19B identity/decision columns needed to locate the causal
// Tensor state. Outcome and post-entry columns are deliberately not copied
// into this join request type.
std::vector<Phase19CCausalEntryState> ParsePhase19BCausalJoinRequests(
    const std::string& phase19BObservationsTsv);

Phase19CCausalDatasetResult BuildPhase19CCausalDataset(
    const std::string& phase19BObservationsTsv,
    const std::string& phase19BMetadataTsv,
    const std::vector<Phase19CCausalEntryState>& entryStates,
    const std::vector<Phase19CPredictorIdentity>& predictors,
    const Phase19CCausalDatasetContext& context);

} // namespace EA::StrategyEvaluation
