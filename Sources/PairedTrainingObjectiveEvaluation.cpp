#include "PairedTrainingObjectiveEvaluation.hpp"

#include "InferenceProfitability.hpp"
#include "TrainingObjective.hpp"

#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>

namespace EA::PairedTrainingObjectiveEvaluation
{
namespace
{

using TrainingObjective::CanonicalText;
using TrainingObjective::DeterministicHash;
using TrainingObjective::ParseSupportedCanonicalText;

void Add(std::vector<std::string>& values, std::string value)
{
    values.push_back(std::move(value));
}

bool Finite(double value)
{
    return std::isfinite(value);
}

// The persisted model train_config_meta stores threshold_logret as float,
// while experiment.c_next_threshold is double precision.  This is the
// existing repository/scheduler compatibility boundary (including exact
// final-inference resolution), not a pair-specific scientific tolerance.
constexpr double kPersistedModelThresholdTolerance = 1.0e-7;

bool SamePersistedModelThreshold(double experimentThreshold,
                                 double modelThreshold)
{
    return Finite(experimentThreshold) && Finite(modelThreshold) &&
        std::fabs(experimentThreshold - modelThreshold) <=
            kPersistedModelThresholdTolerance;
}

bool TaggedHash(const std::string& value)
{
    if (value.size() != 24 || !value.starts_with("fnv1a64:")) return false;
    for (std::size_t index = 8; index < value.size(); ++index)
        if (!((value[index] >= '0' && value[index] <= '9') ||
              (value[index] >= 'a' && value[index] <= 'f')))
            return false;
    return true;
}

MetricDelta Delta(std::optional<double> control,
                  std::optional<double> treatment)
{
    MetricDelta result{control, treatment, std::nullopt, std::nullopt};
    if (!control || !treatment || !Finite(*control) || !Finite(*treatment))
        return result;
    result.treatmentMinusControl = *treatment - *control;
    if (*control != 0.0)
        result.relativeToAbsoluteControl =
            *result.treatmentMinusControl / std::abs(*control);
    return result;
}

MetricDelta CountDelta(std::uint64_t control, std::uint64_t treatment)
{
    return Delta(static_cast<double>(control), static_cast<double>(treatment));
}

void RequireText(const std::string& value,
                 std::string_view field,
                 std::vector<std::string>& reasons)
{
    if (value.empty()) Add(reasons, "missing_" + std::string(field));
}

bool ValidObjective(const ObjectiveProvenance& value)
{
    if (value.canonical.empty() || !TaggedHash(value.hash) ||
        DeterministicHash(value.canonical) != value.hash)
        return false;
    try
    {
        return CanonicalText(ParseSupportedCanonicalText(value.canonical)) ==
            value.canonical;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

void ValidatePersistedFinalModelConfiguration(
    const ScientificConfiguration& value,
    std::string_view arm,
    std::vector<std::string>& reasons)
{
    const std::string prefix = std::string(arm) + '_';
    if (value.inputWidth <= 0 || value.hiddenSize <= 0 ||
        value.layerCount <= 0 || value.windowSize <= 0)
        Add(reasons, prefix + "architecture_incomplete");
    if (value.modelMetadataSchemaVersion <= 0 ||
        value.trainConfigurationSchemaVersion <= 0 ||
        value.optimizerMetadataSchemaVersion <= 0 ||
        value.optimizerType <= 0 || value.modelInputMetadataSchemaVersion <= 0 ||
        value.modelInputLayoutVersion <= 0 || value.labelRuleId <= 0 ||
        value.donchianLookback <= 0)
        Add(reasons, prefix + "persisted_model_semantics_incomplete");
    const double persistedValues[] = {
        value.classWeightDown,
        value.classWeightNeutral,
        value.classWeightUp,
        value.persistedCoreLearningRateMultiplier,
        value.persistedHeadWeightLearningRateMultiplier,
        value.persistedHeadBiasLearningRateMultiplier,
        value.targetScale,
        value.targetBias,
        value.targetMean,
        value.targetStandardDeviation};
    for (double persisted : persistedValues)
        if (!Finite(persisted))
            Add(reasons, prefix + "persisted_model_value_nonfinite");
    RequireText(value.persistedTrainingSymbol,
                prefix + "persisted_training_symbol", reasons);
    RequireText(value.persistedTrainingStart,
                prefix + "persisted_training_start", reasons);
    RequireText(value.persistedTrainingEnd,
                prefix + "persisted_training_end", reasons);
    if (value.persistedTrainingSymbol != value.symbol ||
        value.persistedTrainingStart != value.trainStart ||
        value.persistedTrainingEnd != value.trainEnd)
        Add(reasons, prefix + "persisted_training_context_mismatch");
    if (value.coreLearningRateMultiplier &&
        !SamePersistedModelThreshold(
            *value.coreLearningRateMultiplier,
            value.persistedCoreLearningRateMultiplier))
        Add(reasons, prefix + "persisted_core_lr_context_mismatch");
    if (value.headLearningRateMultiplier &&
        !SamePersistedModelThreshold(
            *value.headLearningRateMultiplier,
            value.persistedHeadWeightLearningRateMultiplier))
        Add(reasons, prefix + "persisted_head_lr_context_mismatch");
    if (value.resumeExpandInputWidth !=
        value.inputWidthExpansionCanonical.has_value())
        Add(reasons, prefix + "input_width_expansion_provenance_mismatch");
}

void ValidateScientificCompleteness(
    const ScientificConfiguration& value,
    std::string_view arm,
    std::vector<std::string>& reasons)
{
    const std::string prefix = std::string(arm) + '_';
    if (value.experimentId <= 0) Add(reasons, prefix + "experiment_id_invalid");
    if (value.predictionHorizon <= 0) Add(reasons, prefix + "horizon_invalid");
    if (value.targetEpochs <= 0) Add(reasons, prefix + "target_epochs_invalid");
    if (value.checkpointInterval <= 0)
        Add(reasons, prefix + "checkpoint_interval_invalid");
    if (!Finite(value.threshold) ||
        (value.coreLearningRateMultiplier &&
         !Finite(*value.coreLearningRateMultiplier)) ||
        (value.headLearningRateMultiplier &&
         !Finite(*value.headLearningRateMultiplier)))
        Add(reasons, prefix + "nonfinite_configuration");
    RequireText(value.symbol, prefix + "symbol", reasons);
    RequireText(value.trainStart, prefix + "train_start", reasons);
    RequireText(value.trainEnd, prefix + "train_end", reasons);
    RequireText(value.inferenceStart, prefix + "inference_start", reasons);
    RequireText(value.inferenceEnd, prefix + "inference_end", reasons);
    RequireText(value.featureWarmupScope,
                prefix + "feature_warmup_scope", reasons);
    RequireText(value.donchianMode, prefix + "donchian_mode", reasons);
    RequireText(value.runProvenance.gitCommit,
                prefix + "git_commit", reasons);
    if (!value.runProvenance.gitDirty)
        Add(reasons, prefix + "git_dirty_missing");
    RequireText(value.runProvenance.buildConfiguration,
                prefix + "build_configuration", reasons);
    RequireText(value.runProvenance.compilerVersion,
                prefix + "compiler_version", reasons);
    RequireText(value.runProvenance.schemaVersion,
                prefix + "schema_version", reasons);
    RequireText(value.runProvenance.schedulerVersion,
                prefix + "scheduler_version", reasons);
    RequireText(value.runProvenance.binaryName,
                prefix + "binary_name", reasons);
    if (!ValidObjective(value.experimentObjective))
        Add(reasons, prefix + "experiment_objective_invalid");
}

#define EA_COMPARE_FIELD(field, name)                                         \
    do                                                                        \
    {                                                                         \
        if (control.field != treatment.field) Add(reasons, name);             \
    } while (false)

void ValidatePairIdentity(const ScientificConfiguration& control,
                          const ScientificConfiguration& treatment,
                          std::vector<std::string>& reasons)
{
    EA_COMPARE_FIELD(symbol, "symbol_mismatch");
    EA_COMPARE_FIELD(predictionHorizon, "prediction_horizon_mismatch");
    EA_COMPARE_FIELD(trainStart, "train_start_mismatch");
    EA_COMPARE_FIELD(trainEnd, "train_end_mismatch");
    EA_COMPARE_FIELD(inferenceStart, "inference_start_mismatch");
    EA_COMPARE_FIELD(inferenceEnd, "inference_end_mismatch");
    EA_COMPARE_FIELD(targetEpochs, "target_epochs_mismatch");
    EA_COMPARE_FIELD(threshold, "threshold_mismatch");
    EA_COMPARE_FIELD(coreLearningRateMultiplier, "core_lr_mismatch");
    EA_COMPARE_FIELD(headLearningRateMultiplier, "head_lr_mismatch");
    EA_COMPARE_FIELD(checkpointInterval, "checkpoint_interval_mismatch");
    EA_COMPARE_FIELD(inputWidth, "input_width_mismatch");
    EA_COMPARE_FIELD(hiddenSize, "hidden_size_mismatch");
    EA_COMPARE_FIELD(layerCount, "layer_count_mismatch");
    EA_COMPARE_FIELD(windowSize, "window_size_mismatch");
    EA_COMPARE_FIELD(modelMetadataSchemaVersion,
                     "model_metadata_schema_mismatch");
    EA_COMPARE_FIELD(trainConfigurationSchemaVersion,
                     "train_configuration_schema_mismatch");
    EA_COMPARE_FIELD(normalizationVersion, "normalization_mismatch");
    EA_COMPARE_FIELD(classWeightDown, "class_weight_down_mismatch");
    EA_COMPARE_FIELD(classWeightNeutral, "class_weight_neutral_mismatch");
    EA_COMPARE_FIELD(classWeightUp, "class_weight_up_mismatch");
    EA_COMPARE_FIELD(featureWarmupScope, "feature_warmup_scope_mismatch");
    EA_COMPARE_FIELD(donchianMode, "donchian_mode_mismatch");
    EA_COMPARE_FIELD(donchianLookback, "donchian_lookback_mismatch");
    EA_COMPARE_FIELD(featureAblationMask, "feature_ablation_mask_mismatch");
    EA_COMPARE_FIELD(optimizerMetadataSchemaVersion,
                     "optimizer_metadata_schema_mismatch");
    EA_COMPARE_FIELD(optimizerType, "optimizer_type_mismatch");
    EA_COMPARE_FIELD(optimizerFirstMomentBufferCount,
                     "optimizer_first_moment_mismatch");
    EA_COMPARE_FIELD(optimizerSecondMomentBufferCount,
                     "optimizer_second_moment_mismatch");
    EA_COMPARE_FIELD(persistedCoreLearningRateMultiplier,
                     "persisted_core_lr_mismatch");
    EA_COMPARE_FIELD(persistedHeadWeightLearningRateMultiplier,
                     "persisted_head_weight_lr_mismatch");
    EA_COMPARE_FIELD(persistedHeadBiasLearningRateMultiplier,
                     "persisted_head_bias_lr_mismatch");
    EA_COMPARE_FIELD(labelRuleId, "label_rule_id_mismatch");
    EA_COMPARE_FIELD(targetType, "target_type_mismatch");
    EA_COMPARE_FIELD(targetScale, "target_scale_mismatch");
    EA_COMPARE_FIELD(targetBias, "target_bias_mismatch");
    EA_COMPARE_FIELD(targetUseZScore, "target_zscore_mismatch");
    EA_COMPARE_FIELD(targetMean, "target_mean_mismatch");
    EA_COMPARE_FIELD(targetStandardDeviation,
                     "target_standard_deviation_mismatch");
    EA_COMPARE_FIELD(modelInputMetadataSchemaVersion,
                     "model_input_metadata_schema_mismatch");
    EA_COMPARE_FIELD(modelInputLayoutVersion,
                     "model_input_layout_mismatch");
    EA_COMPARE_FIELD(persistedTrainingSymbol,
                     "persisted_training_symbol_mismatch");
    EA_COMPARE_FIELD(persistedTrainingStart,
                     "persisted_training_start_mismatch");
    EA_COMPARE_FIELD(persistedTrainingEnd,
                     "persisted_training_end_mismatch");
    EA_COMPARE_FIELD(inputWidthExpansionCanonical,
                     "input_width_expansion_mismatch");
    EA_COMPARE_FIELD(resumeModelId, "resume_model_id_mismatch");
    EA_COMPARE_FIELD(resumeExpandInputWidth,
                     "resume_expand_input_width_mismatch");
    EA_COMPARE_FIELD(runProvenance, "run_provenance_mismatch");

    if (control.experimentId == treatment.experimentId)
        Add(reasons, "experiment_id_reused");
    if (control.experimentObjective == treatment.experimentObjective)
        Add(reasons, "training_objective_same");
}

#undef EA_COMPARE_FIELD

void ValidateObjectiveExecution(const ArmEvidence& arm,
                                std::string_view name,
                                std::vector<std::string>& invalid,
                                std::vector<std::string>& incomplete)
{
    const std::string prefix = std::string(name) + '_';
    if (!arm.finalModelId)
    {
        Add(incomplete, prefix + "final_model_missing");
        return;
    }
    if (*arm.finalModelId <= 0)
        Add(invalid, prefix + "final_model_id_invalid");

    int finalCount = 0;
    std::set<long long> modelIds;
    for (const auto& model : arm.materializedModelObjectives)
    {
        if (!modelIds.insert(model.modelId).second)
            Add(invalid, prefix + "model_objective_identity_reused");
        if (model.modelId <= 0 || !ValidObjective(model.objective))
        {
            Add(invalid, prefix + "model_objective_invalid");
            continue;
        }
        if (model.objective != arm.configuration.experimentObjective)
            Add(invalid, prefix + "experiment_model_objective_mismatch");
        if (model.isFinalModel)
        {
            ++finalCount;
            if (model.modelId != *arm.finalModelId)
                Add(invalid, prefix + "final_model_identity_mismatch");
        }
    }
    if (arm.materializedModelObjectives.empty())
        Add(incomplete, prefix + "model_objective_evidence_missing");
    else if (finalCount != 1)
        Add(invalid, prefix + "final_model_objective_not_unique");

    // Production does not persist the diagnostic TRAINING_OBJECTIVE_ACTIVE
    // log event.  Exact experiment provenance plus objective metadata on
    // every materialized model is the durable execution-lineage contract.
    // Retain validation for callers that do have independently persisted
    // runtime evidence, but do not manufacture or require it.
    if (arm.runtimeObjective)
    {
        const auto& runtime = *arm.runtimeObjective;
        if (runtime.eventName != "TRAINING_OBJECTIVE_ACTIVE" ||
            runtime.objectiveHash !=
                arm.configuration.experimentObjective.hash)
            Add(invalid, prefix + "runtime_objective_mismatch");
        try
        {
            if (runtime.objectiveIdentifier !=
                ParseSupportedCanonicalText(
                    arm.configuration.experimentObjective.canonical)
                    .objectiveIdentifier)
                Add(invalid, prefix + "runtime_objective_identifier_mismatch");
        }
        catch (const std::exception&)
        {
            Add(invalid, prefix + "runtime_objective_unrecognized");
        }
    }
}

void ValidateClassification(const ArmEvidence& arm,
                            std::string_view name,
                            std::vector<std::string>& invalid,
                            std::vector<std::string>& incomplete)
{
    const std::string prefix = std::string(name) + '_';
    if (!arm.classification)
    {
        Add(incomplete, prefix + "final_classification_missing");
        return;
    }
    const auto& value = *arm.classification;
    if (!arm.finalModelId) return;
    if (value.inferenceResultId <= 0 || value.analysisId <= 0)
        Add(invalid, prefix + "classification_identity_invalid");
    if (value.modelId != *arm.finalModelId ||
        value.analysisModelId != *arm.finalModelId ||
        value.analysisExperimentId != arm.configuration.experimentId)
        Add(invalid, prefix + "classification_model_experiment_mismatch");
    if (value.inferenceScope != "final" || value.analysisScope != "final" ||
        value.checkpointEvalId || value.parentExperimentId ||
        value.analysisCheckpointEvalId || value.analysisParentExperimentId)
        Add(invalid, prefix + "classification_not_exact_final_scope");
    if (value.status != "completed" || value.analysisStatus != "completed")
        Add(incomplete, prefix + "classification_not_completed");
    if (value.symbol != arm.configuration.symbol ||
        value.predictionHorizon != arm.configuration.predictionHorizon ||
        !SamePersistedModelThreshold(
            arm.configuration.threshold, value.threshold) ||
        value.windowSize != arm.configuration.windowSize ||
        value.labelRuleId != arm.configuration.labelRuleId ||
        value.targetType != arm.configuration.targetType ||
        value.inferenceStart != arm.configuration.inferenceStart ||
        value.inferenceEnd != arm.configuration.inferenceEnd ||
        value.completedEpochs != arm.configuration.targetEpochs)
        Add(invalid, prefix + "classification_context_mismatch");

    const std::optional<double> values[] = {
        value.accuracy, value.predictedDownProportion,
        value.predictedNeutralProportion, value.predictedUpProportion,
        value.inferenceAccuracy, value.acceptAccuracy, value.acceptRate,
        value.leaderScore};
    for (const auto& metric : values)
        if (metric && !Finite(*metric))
            Add(invalid, prefix + "classification_metric_nonfinite");
    if (value.accuracy && value.inferenceAccuracy &&
        *value.accuracy != *value.inferenceAccuracy)
        Add(invalid, prefix + "inference_analysis_accuracy_mismatch");
}

void ValidateProfitability(const ArmEvidence& arm,
                           std::string_view name,
                           std::vector<std::string>& invalid,
                           std::vector<std::string>& incomplete)
{
    const std::string prefix = std::string(name) + '_';
    if (!arm.profitability)
    {
        Add(incomplete, prefix + "final_profitability_missing");
        return;
    }
    const auto& value = *arm.profitability;
    if (!arm.finalModelId || !arm.classification) return;
    if (value.observationId <= 0 || value.experimentId !=
            arm.configuration.experimentId ||
        value.modelId != *arm.finalModelId ||
        value.inferenceResultId != arm.classification->inferenceResultId)
        Add(invalid, prefix + "profitability_provenance_mismatch");
    if (value.inferenceScope != "final" || value.checkpointEvalId)
        Add(invalid, prefix + "profitability_not_exact_final_scope");
    if (value.inferenceStart != arm.configuration.inferenceStart ||
        value.inferenceEnd != arm.configuration.inferenceEnd)
        Add(invalid, prefix + "profitability_inference_range_mismatch");
    if (value.actionableCount > value.predictionCount ||
        !Finite(value.aggregateTerminalHorizonLogReturnSum) ||
        (value.averageTerminalHorizonLogReturnPerActionablePrediction &&
         !Finite(*value.averageTerminalHorizonLogReturnPerActionablePrediction)))
        Add(invalid, prefix + "profitability_values_invalid");
    if ((value.actionableCount == 0) ==
        value.averageTerminalHorizonLogReturnPerActionablePrediction.has_value())
        Add(invalid, prefix + "profitability_average_presence_mismatch");
    if (value.actionableCount == 0 &&
        value.aggregateTerminalHorizonLogReturnSum != 0.0)
        Add(invalid, prefix + "zero_actionable_nonzero_aggregate");
    if (value.actionableCount > 0 &&
        value.averageTerminalHorizonLogReturnPerActionablePrediction &&
        *value.averageTerminalHorizonLogReturnPerActionablePrediction !=
            value.aggregateTerminalHorizonLogReturnSum /
                static_cast<double>(value.actionableCount))
        Add(invalid, prefix + "profitability_average_value_mismatch");
    if (value.metricDefinitionCanonical !=
            InferenceProfitability::kMetricDefinitionCanonical ||
        value.metricDefinitionHash !=
            InferenceProfitability::MetricDefinitionHash())
        Add(invalid, prefix + "profitability_metric_definition_mismatch");
    if (!TaggedHash(value.sourceContentHash) ||
        value.observationIdentityCanonical.empty() ||
        !TaggedHash(value.observationIdentityHash) ||
        value.observationIdentityHash != DeterministicHash(
            value.observationIdentityCanonical))
        Add(invalid, prefix + "profitability_identity_invalid");
}

bool PolicyFiniteAndNonnegative(const MaterialityPolicy& policy)
{
    if (!Finite(policy.minimumProfitabilityImprovement) ||
        policy.minimumProfitabilityImprovement < 0.0 ||
        !Finite(policy.maximumProfitabilityWorsening) ||
        policy.maximumProfitabilityWorsening < 0.0)
        return false;
    if (!policy.classification) return true;
    const auto& c = *policy.classification;
    const double values[] = {
        c.maximumInferenceAccuracyDecrease,
        c.maximumAcceptAccuracyDecrease,
        c.maximumAcceptRateDecrease,
        c.maximumLeaderScoreDecrease,
        c.maximumNeutralProportionIncrease};
    for (double value : values)
        if (!Finite(value) || value < 0.0) return false;
    return true;
}

bool UnacceptableDecrease(const MetricDelta& metric, double maximumDecrease)
{
    return metric.treatmentMinusControl &&
        *metric.treatmentMinusControl < -maximumDecrease;
}

} // namespace

ComparisonResult Compare(const ArmEvidence& control,
                         const ArmEvidence& treatment,
                         const MaterialityPolicy& policy)
{
    ComparisonResult result;
    if (!PolicyFiniteAndNonnegative(policy))
        Add(result.invalidReasons, "materiality_policy_invalid");

    ValidateScientificCompleteness(control.configuration, "control",
                                   result.invalidReasons);
    ValidateScientificCompleteness(treatment.configuration, "treatment",
                                   result.invalidReasons);
    if (control.experimentStatus != "completed" ||
        control.experimentPhase != "done")
        Add(result.incompleteReasons, "control_experiment_not_complete");
    if (treatment.experimentStatus != "completed" ||
        treatment.experimentPhase != "done")
        Add(result.incompleteReasons, "treatment_experiment_not_complete");

    // A running arm has no authoritative final-model identity yet.  Report
    // that scientific state without comparing checkpoint metadata as though
    // it were final evidence.
    if (!result.invalidReasons.empty())
    {
        result.disposition = Disposition::InvalidComparison;
        return result;
    }
    if (!result.incompleteReasons.empty())
    {
        result.disposition = Disposition::Incomplete;
        return result;
    }

    if (!control.finalModelId)
        Add(result.incompleteReasons, "control_final_model_missing");
    if (!treatment.finalModelId)
        Add(result.incompleteReasons, "treatment_final_model_missing");
    if (!result.incompleteReasons.empty())
    {
        result.disposition = Disposition::Incomplete;
        return result;
    }

    ValidatePairIdentity(control.configuration, treatment.configuration,
                         result.invalidReasons);
    ValidatePersistedFinalModelConfiguration(
        control.configuration, "control", result.invalidReasons);
    ValidatePersistedFinalModelConfiguration(
        treatment.configuration, "treatment", result.invalidReasons);

    ValidateObjectiveExecution(control, "control", result.invalidReasons,
                               result.incompleteReasons);
    ValidateObjectiveExecution(treatment, "treatment", result.invalidReasons,
                               result.incompleteReasons);
    ValidateClassification(control, "control", result.invalidReasons,
                           result.incompleteReasons);
    ValidateClassification(treatment, "treatment", result.invalidReasons,
                           result.incompleteReasons);
    ValidateProfitability(control, "control", result.invalidReasons,
                          result.incompleteReasons);
    ValidateProfitability(treatment, "treatment", result.invalidReasons,
                          result.incompleteReasons);

    if (!result.invalidReasons.empty())
    {
        result.disposition = Disposition::InvalidComparison;
        return result;
    }
    if (!result.incompleteReasons.empty())
    {
        result.disposition = Disposition::Incomplete;
        return result;
    }

    const auto& cc = *control.classification;
    const auto& tc = *treatment.classification;
    const auto& cp = *control.profitability;
    const auto& tp = *treatment.profitability;
    result.actionableCount = CountDelta(cp.actionableCount, tp.actionableCount);
    result.aggregateProfitability = Delta(
        cp.aggregateTerminalHorizonLogReturnSum,
        tp.aggregateTerminalHorizonLogReturnSum);
    result.averageProfitability = Delta(
        cp.averageTerminalHorizonLogReturnPerActionablePrediction,
        tp.averageTerminalHorizonLogReturnPerActionablePrediction);
    result.inferenceAccuracy = Delta(cc.inferenceAccuracy, tc.inferenceAccuracy);
    result.acceptAccuracy = Delta(cc.acceptAccuracy, tc.acceptAccuracy);
    result.acceptRate = Delta(cc.acceptRate, tc.acceptRate);
    result.predictedNeutralProportion = Delta(
        cc.predictedNeutralProportion, tc.predictedNeutralProportion);
    result.leaderScore = Delta(cc.leaderScore, tc.leaderScore);

    const MetricDelta& primary = policy.primaryProfitabilityMetric ==
            ProfitabilityPrimaryMetric::
                AggregateTerminalHorizonLogReturnSum
        ? result.aggregateProfitability
        : result.averageProfitability;
    if (!primary.treatmentMinusControl)
    {
        Add(result.interpretationReasons,
            "primary_profitability_delta_unavailable");
        result.disposition = Disposition::Mixed;
        return result;
    }

    if (*primary.treatmentMinusControl < 0.0 &&
        -*primary.treatmentMinusControl >=
            policy.maximumProfitabilityWorsening)
    {
        Add(result.interpretationReasons,
            "primary_profitability_materially_worse");
        result.disposition = Disposition::NotPromising;
        return result;
    }

    if (!policy.classification)
    {
        Add(result.interpretationReasons,
            "classification_degradation_policy_not_configured");
        result.disposition = Disposition::Mixed;
        return result;
    }
    const auto& classificationPolicy = *policy.classification;
    if (!result.inferenceAccuracy.treatmentMinusControl ||
        !result.acceptAccuracy.treatmentMinusControl ||
        !result.acceptRate.treatmentMinusControl ||
        !result.predictedNeutralProportion.treatmentMinusControl ||
        !result.leaderScore.treatmentMinusControl)
    {
        Add(result.interpretationReasons,
            "classification_policy_metric_unavailable");
        result.disposition = Disposition::Incomplete;
        return result;
    }
    if (UnacceptableDecrease(result.inferenceAccuracy,
            classificationPolicy.maximumInferenceAccuracyDecrease) ||
        UnacceptableDecrease(result.acceptAccuracy,
            classificationPolicy.maximumAcceptAccuracyDecrease) ||
        UnacceptableDecrease(result.acceptRate,
            classificationPolicy.maximumAcceptRateDecrease) ||
        UnacceptableDecrease(result.leaderScore,
            classificationPolicy.maximumLeaderScoreDecrease) ||
        *result.predictedNeutralProportion.treatmentMinusControl >
            classificationPolicy.maximumNeutralProportionIncrease)
    {
        Add(result.interpretationReasons,
            "classification_degradation_unacceptable");
        result.disposition = Disposition::NotPromising;
        return result;
    }

    const bool aggregatePositive =
        result.aggregateProfitability.treatmentMinusControl &&
        *result.aggregateProfitability.treatmentMinusControl > 0.0;
    const bool averagePositive =
        result.averageProfitability.treatmentMinusControl &&
        *result.averageProfitability.treatmentMinusControl > 0.0;
    const bool aggregateNegative =
        result.aggregateProfitability.treatmentMinusControl &&
        *result.aggregateProfitability.treatmentMinusControl < 0.0;
    const bool averageNegative =
        result.averageProfitability.treatmentMinusControl &&
        *result.averageProfitability.treatmentMinusControl < 0.0;
    if ((aggregatePositive && averageNegative) ||
        (aggregateNegative && averagePositive) ||
        !result.averageProfitability.treatmentMinusControl)
    {
        Add(result.interpretationReasons,
            "profitability_primitives_disagree_or_are_ambiguous");
        result.disposition = Disposition::Mixed;
        return result;
    }

    if (*primary.treatmentMinusControl >=
        policy.minimumProfitabilityImprovement &&
        *primary.treatmentMinusControl > 0.0)
    {
        Add(result.interpretationReasons,
            "material_profitability_improvement_within_classification_policy");
        result.disposition = Disposition::Promising;
        return result;
    }

    Add(result.interpretationReasons,
        "profitability_effect_below_materiality_or_directionally_ambiguous");
    result.disposition = Disposition::Mixed;
    return result;
}

std::string MaterialityPolicyCanonicalText(const MaterialityPolicy& policy)
{
    if (!PolicyFiniteAndNonnegative(policy))
        throw std::invalid_argument("materiality_policy_invalid");

    std::string result = "paired_objective_materiality_policy_v1;";
    const auto append = [&result](std::string_view name,
                                  std::string_view value)
    {
        result.append(name);
        result.push_back('=');
        result.append(value);
        result.push_back(';');
    };
    switch (policy.primaryProfitabilityMetric)
    {
        case ProfitabilityPrimaryMetric::
                AggregateTerminalHorizonLogReturnSum:
            append("primary_profitability_metric",
                   "aggregate_terminal_horizon_log_return_sum");
            break;
        case ProfitabilityPrimaryMetric::
                AverageTerminalHorizonLogReturnPerActionablePrediction:
            append("primary_profitability_metric",
                   "average_terminal_horizon_log_return_per_actionable_prediction");
            break;
        default:
            throw std::invalid_argument(
                "materiality_policy_primary_metric_invalid");
    }
    append("minimum_profitability_improvement",
           TrainingObjective::CanonicalDouble(
               policy.minimumProfitabilityImprovement));
    append("maximum_profitability_worsening",
           TrainingObjective::CanonicalDouble(
               policy.maximumProfitabilityWorsening));
    append("classification_policy",
           policy.classification ? "configured" : "not_configured");
    if (policy.classification)
    {
        append("maximum_inference_accuracy_decrease",
            TrainingObjective::CanonicalDouble(
                policy.classification->maximumInferenceAccuracyDecrease));
        append("maximum_accept_accuracy_decrease",
            TrainingObjective::CanonicalDouble(
                policy.classification->maximumAcceptAccuracyDecrease));
        append("maximum_accept_rate_decrease",
            TrainingObjective::CanonicalDouble(
                policy.classification->maximumAcceptRateDecrease));
        append("maximum_leader_score_decrease",
            TrainingObjective::CanonicalDouble(
                policy.classification->maximumLeaderScoreDecrease));
        append("maximum_neutral_proportion_increase",
            TrainingObjective::CanonicalDouble(
                policy.classification->maximumNeutralProportionIncrease));
    }
    else
    {
        append("maximum_inference_accuracy_decrease", "NULL");
        append("maximum_accept_accuracy_decrease", "NULL");
        append("maximum_accept_rate_decrease", "NULL");
        append("maximum_leader_score_decrease", "NULL");
        append("maximum_neutral_proportion_increase", "NULL");
    }
    return result;
}

std::string MaterialityPolicyIdentity(const MaterialityPolicy& policy)
{
    return TrainingObjective::DeterministicHash(
        MaterialityPolicyCanonicalText(policy));
}

std::string DispositionText(Disposition value)
{
    switch (value)
    {
        case Disposition::Promising: return "PROMISING";
        case Disposition::Mixed: return "MIXED";
        case Disposition::NotPromising: return "NOT_PROMISING";
        case Disposition::InvalidComparison: return "INVALID_COMPARISON";
        case Disposition::Incomplete: return "INCOMPLETE";
    }
    throw std::invalid_argument("unknown_paired_objective_disposition");
}

} // namespace EA::PairedTrainingObjectiveEvaluation
