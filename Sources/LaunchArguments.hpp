#pragma once

#include "Donchian20Mode.hpp"
#include "FeatureWarmupScope.hpp"
#include "LstmRuntimeLogging.hpp"
#include "TrainingObjective.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace EA
{
struct LaunchArgs
{
    std::string fromDate;
    std::string toDate;
    bool positionalDateRangeSupplied = false;
    std::optional<std::string> symbol;
    std::optional<long long> modelId;
    std::optional<long long> resumeModelId;
    std::optional<int> targetEpochs;
    std::optional<std::string> newModelName;
    std::optional<bool> inferenceMode;
    std::optional<size_t> predictionHorizon;
    std::optional<double> thresholdLogret;
    std::optional<size_t> windowSize;
    std::optional<size_t> hiddenSize;
    std::optional<size_t> numLayers;
    std::optional<int> epochs;
    std::optional<double> coreLrMult;
    std::optional<double> headWeightLrMult;
    std::optional<double> headBiasLrMult;
    std::optional<int> checkpointEvery;
    std::optional<long long> schedulerExperimentId;
    std::optional<long long> schedulerCheckpointEvalId;
    std::optional<long long> schedulerWorkerAttemptId;
    std::optional<EA::TrainingObjective::Configuration> trainingObjective;
    std::optional<unsigned int> freshInitializationSeed;
    std::optional<long long> inferStartAfterModelId;
    std::optional<EA::FeatureWarmupScope> featureWarmupScope;
    std::optional<Donchian20Mode> donchian20Mode;
    std::optional<std::size_t> donchianLookback;
    std::optional<RuntimeLogLevel> logLevel;
    std::optional<std::string> lstmProfileOutputPath;
    bool evalTrading = false;
    bool inferAll = false;
    bool forceInfer = false;
    bool lstmProfileHotspots = false;
    bool resumeExpandInputWidth = false;
    std::optional<std::string> controlledFixedStopEvaluationPath;
    std::optional<std::string>
        probabilityConditionedStopEvaluationPath;
    std::optional<std::string>
        probabilityConditionedStopExtensionEvaluationPath;
    std::optional<std::string>
        probabilityStopExtensionStateAnalysisPath;
    std::optional<std::string>
        probabilityStopExtensionPathMechanismPath;
    std::optional<std::string> phase19CCausalPathPredictabilityDirectory;
    std::optional<std::string> phase19BPathArtifactDirectory;
    struct FrozenOutcomeSpec
    {
        std::string cohortHash;
        long long sourceExperimentId = -1;
        long long sourceModelId = -1;
        std::string outcomeStart;
        std::string outcomeEnd;
        std::string jobHash;
    };
    std::optional<FrozenOutcomeSpec> frozenOutcome;
};


bool HasControlledStrategyEvaluation(const LaunchArgs& launchArgs);
long long ParseModelIdArg(const std::string& value);
LaunchArgs ParseLaunchArgs(int argc, const char* argv[]);

} // namespace EA
