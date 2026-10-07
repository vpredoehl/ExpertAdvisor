#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include <pqxx/pqxx>

#include "Donchian20Mode.hpp"
#include "FeatureWarmupScope.hpp"
#include "ModelInputExpansion.hpp"
#include "LSTM.hpp"
#include "LaunchArguments.hpp"
#include "PgModelIO.hpp"
#include "TrainingObjective.hpp"

namespace EA::PersistedModelRuntimeConfig
{

struct TrainConfigMeta
{
    std::optional<std::string> symbol;
    int schemaVersion = DBIO::PgModelIO::kTrainConfigMetaSchemaVersion;
    std::size_t predictionHorizon = static_cast<std::size_t>(prediction_horizon);
    float thresholdLogret = c_next_threshold;
    std::size_t windowSize = static_cast<std::size_t>(window_size);
    int labelRuleId = DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId;
    float classWeightDown = kClassWeightDown;
    float classWeightNeutral = kClassWeightNeutral;
    float classWeightUp = kClassWeightUp;
    std::size_t numLayers = static_cast<std::size_t>(num_layers);
    int normalizationVersion = normalization_version;
    std::optional<std::size_t> epochsTrained;
    std::optional<float> coreLrMult;
    std::optional<float> headWeightLrMult;
    std::optional<float> headBiasLrMult;
};

struct ResumeCheckpointConfig
{
    long long sourceModelId = -1;
    std::string symbol;
    std::string fromDate;
    std::string toDate;
    TrainConfigMeta trainConfig;
    int modelInputWidth = 0;
    std::size_t modelHiddenSize = 0;
    LSTM::TargetType targetType = LSTM::TargetType::UpNeutralDownReturn;
    std::size_t completedEpoch = 0;
    std::size_t optimizerUpdateCount = 0;
    FeatureWarmupScope featureWarmupScope = FeatureWarmupScope::LegacyColdBoundary;
    Donchian20Mode donchian20Mode = kDefaultDonchian20Mode;
    std::size_t donchianLookback = kDefaultDonchianLookback;
    FeatureAblationMask featureAblationMask;
    bool expandInputWidthRequested = false;
    bool parameterExpansionRequired = false;
    std::optional<InputWidthExpansionProvenance> inputWidthExpansionProvenance;
    TrainingObjective::Configuration trainingObjective = TrainingObjective::Legacy();
};

struct SchedulerModelInputIdentity
{
    long long experimentId = -1;
    std::size_t width = 0;
    int semanticLayoutVersion = 0;
};

using ModelSymbolReporter = void (*)(long long, const std::string&);

bool ValidateResumeLaunchArgs(const LaunchArgs& launchArgs);
TrainingObjective::Configuration LoadExperimentTrainingObjective(
    pqxx::work& work, long long experimentId);
void ConfigureInputWidthExpansionForResume(
    const DBIO::PgModelIO::PersistedModelMaterialization& persisted,
    ResumeCheckpointConfig& config, bool requested,
    std::optional<long long> schedulerExperimentId);
FeatureAblationMask LoadModelFeatureAblationMask(pqxx::work& work,
                                                 long long modelId);
FeatureAblationMask LoadSchedulerFeatureAblationMask(
    pqxx::work& work, const LaunchArgs& launchArgs);
std::optional<SchedulerModelInputIdentity> LoadSchedulerModelInputIdentity(
    pqxx::work& work, const LaunchArgs& launchArgs);
void ValidateSchedulerModelFeatureAblationMask(
    const FeatureAblationMask& modelMask,
    const FeatureAblationMask& schedulerMask, long long modelId);
void ValidateSchedulerResumeFeatureAblationMask(
    const ResumeCheckpointConfig& resume,
    const FeatureAblationMask& schedulerMask);
TrainConfigMeta ParseTrainConfigMeta(const DBIO::PgModelIO::PersistedMatrix& matrix,
                                     bool requireExtended,
                                     const char* failure);
ResumeCheckpointConfig LoadResumeCheckpointConfig(
    const DBIO::PgModelIO::PersistedModelMaterialization& persisted,
    const TrainingObjective::Configuration& requestedObjective,
    ModelSymbolReporter reportDatabaseModelSymbol);
void ApplyResumeRuntimeConfig(const ResumeCheckpointConfig& config,
                              int targetEpochs,
                              bool& runtimeInferenceMode);

} // namespace EA::PersistedModelRuntimeConfig
