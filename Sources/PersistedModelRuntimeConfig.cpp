#include "PersistedModelRuntimeConfig.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "ModelInputContract.hpp"
#include "TrainingWorkerFeatureAblation.hpp"

namespace EA::PersistedModelRuntimeConfig
{
namespace
{

bool PrintResumeOverrideRejected(const char* param)
{
    std::cerr << "RESUME_CONFIG_OVERRIDE_REJECTED"
              << ",param=" << param
              << ",reason=resume_uses_database_config_only"
              << std::endl;
    return false;
}

bool SchedulerExperimentColumnExists(pqxx::work& work,
                                     const std::string& columnName)
{
    pqxx::result rows = work.exec_params(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'experiment' AND column_name = $1 LIMIT 1;",
        columnName);
    return !rows.empty();
}

} // namespace

bool ValidateResumeLaunchArgs(const LaunchArgs& launchArgs)
{
    if (!launchArgs.resumeModelId.has_value()) return true;
    if (launchArgs.inferenceMode.has_value() && *launchArgs.inferenceMode) return PrintResumeOverrideRejected("--infer");
    if (launchArgs.modelId.has_value()) return PrintResumeOverrideRejected("--model");
    if (launchArgs.symbol.has_value()) return PrintResumeOverrideRejected("--symbol");
    if (launchArgs.positionalDateRangeSupplied) return PrintResumeOverrideRejected("date_range");
    if (launchArgs.evalTrading) return PrintResumeOverrideRejected("--eval-trading");
    if (launchArgs.predictionHorizon.has_value()) return PrintResumeOverrideRejected("--prediction-horizon");
    if (launchArgs.thresholdLogret.has_value()) return PrintResumeOverrideRejected("--threshold");
    if (launchArgs.windowSize.has_value()) return PrintResumeOverrideRejected("--window-size");
    if (launchArgs.hiddenSize.has_value()) return PrintResumeOverrideRejected("--hidden-size");
    if (launchArgs.numLayers.has_value()) return PrintResumeOverrideRejected("--num-layers");
    if (launchArgs.epochs.has_value()) return PrintResumeOverrideRejected("--epochs");
    if (launchArgs.coreLrMult.has_value()) return PrintResumeOverrideRejected("--core-lr-mult");
    if (launchArgs.headWeightLrMult.has_value()) return PrintResumeOverrideRejected("--head-weight-lr-mult");
    if (launchArgs.headBiasLrMult.has_value()) return PrintResumeOverrideRejected("--head-bias-lr-mult");
    if (!launchArgs.targetEpochs.has_value()) return PrintResumeOverrideRejected("--target-epochs");
    return true;
}

TrainingObjective::Configuration LoadExperimentTrainingObjective(
    pqxx::work& work, long long experimentId)
{
    const pqxx::result rows = work.exec(
        "SELECT training_objective_canonical,training_objective_hash "
        "FROM experiment WHERE experiment_id=$1;", pqxx::params{experimentId});
    if (rows.empty()) throw std::runtime_error("training_objective_experiment_not_found");
    return TrainingObjective::ResolvePersisted(
        rows[0][0].as<std::string>(), rows[0][1].as<std::string>());
}

void ConfigureInputWidthExpansionForResume(
    const DBIO::PgModelIO::PersistedModelMaterialization& persisted,
    ResumeCheckpointConfig& config, bool requested,
    std::optional<long long> schedulerExperimentId)
{
    config.expandInputWidthRequested = requested;
    if (!requested) return;
    const std::size_t sourceWidth = static_cast<std::size_t>(config.modelInputWidth);
    if (sourceWidth < kCurrentModelInputWidth)
    {
        const InputWidthExpansionPlan plan = BuildInputWidthExpansionPlan(sourceWidth);
        config.parameterExpansionRequired = true;
        config.inputWidthExpansionProvenance = MakeInputWidthExpansionProvenance(config.sourceModelId, plan);
        return;
    }
    if (sourceWidth > kCurrentModelInputWidth)
    {
        (void)BuildInputWidthExpansionPlan(sourceWidth);
        return;
    }
    if (!schedulerExperimentId.has_value())
        throw std::runtime_error("MODEL_INPUT_EXPANSION_NOT_REQUIRED,source_n_in=" +
                                 std::to_string(sourceWidth) + ",target_n_in=" +
                                 std::to_string(kCurrentModelInputWidth));
    if (persisted.identity.experimentId != schedulerExperimentId)
        throw std::runtime_error("MODEL_INPUT_EXPANSION_RETRY_EXPERIMENT_LINEAGE_MISMATCH");
    if (!persisted.inputWidthExpansionProvenance.has_value())
        throw std::runtime_error("MODEL_INPUT_EXPANSION_RETRY_PROVENANCE_MISSING");
    config.inputWidthExpansionProvenance = persisted.inputWidthExpansionProvenance;
    if (config.inputWidthExpansionProvenance->expandedInputWidth != kCurrentModelInputWidth)
        throw std::runtime_error("MODEL_INPUT_EXPANSION_RETRY_PROVENANCE_TARGET_MISMATCH");
    if (std::find(persisted.identity.ancestryModelIds.begin(),
                  persisted.identity.ancestryModelIds.end(),
                  config.inputWidthExpansionProvenance->sourceModelId) ==
        persisted.identity.ancestryModelIds.end())
        throw std::runtime_error("MODEL_INPUT_EXPANSION_RETRY_SOURCE_LINEAGE_MISMATCH");
}

FeatureAblationMask LoadModelFeatureAblationMask(pqxx::work& work, long long modelId)
{
    const pqxx::result rows = work.exec_params(
        "SELECT m.experiment_id, e.feature_ablation_mask FROM model m "
        "LEFT JOIN experiment e ON e.experiment_id=m.experiment_id "
        "WHERE m.model_id=$1;", modelId);
    if (rows.empty()) throw std::runtime_error("model_not_found_for_feature_ablation_mask");
    if (rows[0][0].is_null()) return {};
    if (rows[0][1].is_null()) throw std::runtime_error("model_experiment_lineage_missing_feature_ablation_mask");
    return FeatureAblationMask::Parse(rows[0][1].as<std::string>());
}

FeatureAblationMask LoadSchedulerFeatureAblationMask(
    pqxx::work& work, const LaunchArgs& launchArgs)
{
    std::optional<long long> experimentId = launchArgs.schedulerExperimentId;
    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        const pqxx::result rows = work.exec_params(
            "SELECT COALESCE(parent_experiment_id, experiment_id) FROM "
            "experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            *launchArgs.schedulerCheckpointEvalId);
        if (rows.empty()) throw std::runtime_error("checkpoint_eval_not_found_for_feature_ablation_mask");
        experimentId = rows[0][0].as<long long>();
    }
    if (!experimentId.has_value()) return {};
    const pqxx::result rows = work.exec_params(
        "SELECT feature_ablation_mask FROM experiment WHERE experiment_id=$1;",
        *experimentId);
    if (rows.empty()) throw std::runtime_error("experiment_not_found_for_feature_ablation_mask");
    return FeatureAblationMask::Parse(rows[0][0].as<std::string>());
}

std::optional<SchedulerModelInputIdentity> LoadSchedulerModelInputIdentity(
    pqxx::work& work, const LaunchArgs& launchArgs)
{
    std::optional<long long> experimentId = launchArgs.schedulerExperimentId;
    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        const pqxx::result rows = work.exec_params(
            "SELECT COALESCE(parent_experiment_id,experiment_id) FROM "
            "experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            *launchArgs.schedulerCheckpointEvalId);
        if (rows.empty()) throw std::runtime_error("checkpoint_eval_not_found_for_model_input_identity");
        experimentId = rows[0][0].as<long long>();
    }
    if (!experimentId.has_value()) return std::nullopt;
    const bool hasWidth = SchedulerExperimentColumnExists(work, "model_input_width");
    const bool hasLayout = SchedulerExperimentColumnExists(work, "model_input_semantic_layout_version");
    if (!hasWidth && !hasLayout) return std::nullopt;
    if (hasWidth != hasLayout) throw std::runtime_error("experiment_model_input_identity_schema_incomplete");
    const pqxx::result rows = work.exec_params(
        "SELECT model_input_width,model_input_semantic_layout_version "
        "FROM experiment WHERE experiment_id=$1;", *experimentId);
    if (rows.empty()) throw std::runtime_error("experiment_not_found_for_model_input_identity");
    if (rows[0][0].is_null() && rows[0][1].is_null()) return std::nullopt;
    if (rows[0][0].is_null() || rows[0][1].is_null()) throw std::runtime_error("experiment_model_input_identity_incomplete");
    SchedulerModelInputIdentity identity{*experimentId, rows[0][0].as<std::size_t>(), rows[0][1].as<int>()};
    (void)ContractForModelInputWidth(identity.width);
    if (!IsModelInputSemanticLayoutWidthCompatible(
            identity.semanticLayoutVersion, identity.width,
            kModelInputSemanticLayoutRegistry, kRegisteredModelInputWidths,
            kModelInputSemanticLayoutVersion, kCurrentModelInputWidth, false))
        throw std::runtime_error("experiment_model_input_identity_incompatible");
    std::cout << "MODEL_INPUT_IDENTITY_ACTIVE"
              << ",experiment_id=" << identity.experimentId
              << ",model_input_width=" << identity.width
              << ",semantic_layout=" << identity.semanticLayoutVersion << std::endl;
    return identity;
}

void ValidateSchedulerModelFeatureAblationMask(
    const FeatureAblationMask& modelMask, const FeatureAblationMask& schedulerMask,
    long long modelId)
{
    if (modelMask.CanonicalText() == schedulerMask.CanonicalText()) return;
    throw std::runtime_error("FEATURE_ABLATION_MASK_LINEAGE_MISMATCH:model_id=" +
                             std::to_string(modelId) + ",model=" + modelMask.CanonicalText() +
                             ",scheduler=" + schedulerMask.CanonicalText());
}

void ValidateSchedulerResumeFeatureAblationMask(
    const ResumeCheckpointConfig& resume, const FeatureAblationMask& schedulerMask)
{
    if (!resume.expandInputWidthRequested)
    {
        ValidateSchedulerModelFeatureAblationMask(
            resume.featureAblationMask, schedulerMask, resume.sourceModelId);
        return;
    }
    Training::ValidateExpandedResumeAblationComposition(
        resume.featureAblationMask, schedulerMask,
        static_cast<std::size_t>(resume.modelInputWidth), feature_size);
}

TrainConfigMeta ParseTrainConfigMeta(const DBIO::PgModelIO::PersistedMatrix& matrix,
                                     bool requireExtended, const char* failure)
{
    const std::size_t required = requireExtended
        ? static_cast<std::size_t>(DBIO::PgModelIO::kTrainConfigMetaExtendedFieldCount)
        : static_cast<std::size_t>(DBIO::PgModelIO::kTrainConfigMetaFieldCount);
    if (matrix.rows != 1 || matrix.cols < required || matrix.values.size() < required)
        throw std::runtime_error(failure);
    const auto& vals = matrix.values;
    TrainConfigMeta meta;
    meta.schemaVersion = static_cast<int>(std::llround(vals[0]));
    meta.predictionHorizon = static_cast<std::size_t>(std::llround(vals[1]));
    meta.thresholdLogret = static_cast<float>(vals[2]);
    meta.windowSize = static_cast<std::size_t>(std::llround(vals[3]));
    meta.labelRuleId = static_cast<int>(std::llround(vals[4]));
    meta.classWeightDown = static_cast<float>(vals[5]);
    meta.classWeightNeutral = static_cast<float>(vals[6]);
    meta.classWeightUp = static_cast<float>(vals[7]);
    meta.numLayers = static_cast<std::size_t>(std::llround(vals[8]));
    meta.normalizationVersion = static_cast<int>(std::llround(vals[9]));
    meta.epochsTrained = static_cast<std::size_t>(std::llround(vals[10]));
    meta.coreLrMult = static_cast<float>(vals[11]);
    meta.headWeightLrMult = static_cast<float>(vals[12]);
    meta.headBiasLrMult = static_cast<float>(vals[13]);
    return meta;
}

ResumeCheckpointConfig LoadResumeCheckpointConfig(
    const DBIO::PgModelIO::PersistedModelMaterialization& persisted,
    const TrainingObjective::Configuration& requestedObjective,
    ModelSymbolReporter reportDatabaseModelSymbol)
{
    ResumeCheckpointConfig config;
    config.sourceModelId = persisted.identity.modelId;
    config.trainingObjective = persisted.trainingObjective;
    TrainingObjective::RequireResumeCompatible(config.trainingObjective, requestedObjective);
    if (persisted.experimentTrainingObjective.has_value())
        TrainingObjective::RequireResumeCompatible(config.trainingObjective,
                                                   *persisted.experimentTrainingObjective);
    config.featureWarmupScope = persisted.featureWarmupScope;
    config.donchian20Mode = persisted.donchian20Mode;
    config.donchianLookback = persisted.donchianLookback;
    config.featureAblationMask = persisted.identity.featureAblationMask;
    if (!persisted.trainConfigMeta.has_value())
        throw std::runtime_error("resume requires complete train_config_meta with 14 fields");
    config.trainConfig = ParseTrainConfigMeta(*persisted.trainConfigMeta, true,
        "resume requires complete train_config_meta with 14 fields");
    config.completedEpoch = config.trainConfig.epochsTrained.value_or(0);
    if (!persisted.trainSymbol.has_value() || !persisted.trainRange.has_value())
        throw std::runtime_error("resume requires persisted training symbol and range");
    config.symbol = *persisted.trainSymbol;
    reportDatabaseModelSymbol(config.sourceModelId, config.symbol);
    config.fromDate = persisted.trainRange->first;
    config.toDate = persisted.trainRange->second;
    config.modelInputWidth = static_cast<int>(persisted.modelMeta.inputWidth);
    config.modelHiddenSize = persisted.modelMeta.hiddenSize;
    if (!persisted.targetMeta.has_value()) throw std::runtime_error("resume requires valid target_meta");
    config.targetType = persisted.targetMeta->targetType;
    if (!persisted.optimizerMeta.has_value()) throw std::runtime_error("resume requires valid optimizer_meta");
    const int optimizerSchema = persisted.optimizerMeta->schemaVersion;
    const int optimizerType = persisted.optimizerMeta->optimizerType;
    if (optimizerSchema != DBIO::PgModelIO::kOptimizerMetaSchemaVersion ||
        optimizerType != DBIO::PgModelIO::kOptimizerTypeSgd)
        throw std::runtime_error("resume optimizer_meta is not supported by this binary");
    config.optimizerUpdateCount = persisted.optimizerMeta->updateCount;
    return config;
}

void ApplyResumeRuntimeConfig(const ResumeCheckpointConfig& config,
                              int targetEpochs, bool& runtimeInferenceMode)
{
    (void)TrainingObjective::ParseSupportedCanonicalText(
        TrainingObjective::CanonicalText(config.trainingObjective));
    if (config.trainConfig.schemaVersion != DBIO::PgModelIO::kTrainConfigMetaSchemaVersion)
        throw std::runtime_error("resume train_config_meta schema_version unsupported");
    if (config.trainConfig.labelRuleId != DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId)
        throw std::runtime_error("resume label_rule_id unsupported by this binary");
    if (config.trainConfig.numLayers != 1)
        throw std::runtime_error("resume num_layers unsupported by this binary");
    if (std::fabs(config.trainConfig.classWeightDown - kClassWeightDown) > 1e-7f ||
        std::fabs(config.trainConfig.classWeightNeutral - kClassWeightNeutral) > 1e-7f ||
        std::fabs(config.trainConfig.classWeightUp - kClassWeightUp) > 1e-7f)
        throw std::runtime_error("resume class weights differ from this binary");
    if (static_cast<std::size_t>(targetEpochs) <= config.completedEpoch)
        throw std::runtime_error("target epochs is an absolute final epoch and must be greater than checkpoint completed epoch");
    runtimeInferenceMode = false;
    prediction_horizon = config.trainConfig.predictionHorizon;
    c_next_threshold = config.trainConfig.thresholdLogret;
    window_size = config.trainConfig.windowSize;
    hidden_size = config.modelHiddenSize;
    n_out = hidden_size;
    num_layers = config.trainConfig.numLayers;
    normalization_version = config.trainConfig.normalizationVersion;
    core_lr_mult = config.trainConfig.coreLrMult.value();
    head_weight_lr_mult = config.trainConfig.headWeightLrMult.value();
    head_bias_lr_mult = config.trainConfig.headBiasLrMult.value();
    epoch_count = targetEpochs;
}

} // namespace EA::PersistedModelRuntimeConfig
