#include "FeatureAblationPairEvaluation.hpp"

#include "FeatureAblation.hpp"
#include "InferenceProfitability.hpp"
#include "TrainingObjective.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>

namespace EA::FeatureAblationPairEvaluation
{
namespace
{

using SharedEvidence::ArmEvidence;
using SharedEvidence::ScientificConfiguration;

void Add(std::vector<std::string>& values, std::string value)
{
    values.push_back(std::move(value));
}

bool Finite(double value)
{
    return std::isfinite(value);
}

MetricDelta Delta(std::optional<double> control,
                  std::optional<double> treatment)
{
    MetricDelta result{control, treatment, std::nullopt};
    if (control && treatment && Finite(*control) && Finite(*treatment))
        result.treatmentMinusControl = *treatment - *control;
    return result;
}

MetricDelta CountDelta(std::uint64_t control, std::uint64_t treatment)
{
    return Delta(static_cast<double>(control), static_cast<double>(treatment));
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

void RequireText(const std::string& value,
                 std::string_view field,
                 std::vector<std::string>& reasons)
{
    if (value.empty()) Add(reasons, "missing_" + std::string(field));
}

void ValidateExperimentConfiguration(
    const FeatureAblationPairEvaluation::ArmEvidence& arm,
    std::string_view role,
    std::vector<std::string>& reasons)
{
    const auto& value = arm.authoritative.configuration;
    const auto& extended = arm.extended;
    const std::string prefix = std::string(role) + '_';
    if (value.experimentId <= 0) Add(reasons, prefix + "experiment_id_invalid");
    if (value.predictionHorizon <= 0) Add(reasons, prefix + "horizon_invalid");
    if (value.targetEpochs <= 0) Add(reasons, prefix + "target_epochs_invalid");
    if (value.checkpointInterval <= 0)
        Add(reasons, prefix + "checkpoint_interval_invalid");
    if (value.donchianLookback <= 0)
        Add(reasons, prefix + "donchian_lookback_invalid");
    if (!Finite(value.threshold) ||
        (value.coreLearningRateMultiplier &&
         !Finite(*value.coreLearningRateMultiplier)) ||
        (value.headLearningRateMultiplier &&
         !Finite(*value.headLearningRateMultiplier)) ||
        !Finite(extended.auxiliaryLossCoefficient) ||
        (extended.robustLossDelta && !Finite(*extended.robustLossDelta)))
        Add(reasons, prefix + "nonfinite_configuration");
    RequireText(value.symbol, prefix + "symbol", reasons);
    RequireText(value.trainStart, prefix + "train_start", reasons);
    RequireText(value.trainEnd, prefix + "train_end", reasons);
    RequireText(value.inferenceStart, prefix + "inference_start", reasons);
    RequireText(value.inferenceEnd, prefix + "inference_end", reasons);
    RequireText(value.featureWarmupScope, prefix + "feature_warmup_scope", reasons);
    RequireText(value.donchianMode, prefix + "donchian_mode", reasons);
    RequireText(value.experimentObjective.canonical,
                prefix + "training_objective_canonical", reasons);
    if (!TaggedHash(value.experimentObjective.hash) ||
        TrainingObjective::DeterministicHash(
            value.experimentObjective.canonical) !=
            value.experimentObjective.hash)
        Add(reasons, prefix + "training_objective_hash_invalid");
    if (extended.trainingObjectiveVersion <= 0 ||
        extended.lossDefinitionVersion <= 0)
        Add(reasons, prefix + "training_objective_version_invalid");
    RequireText(extended.auxiliaryLossMode,
                prefix + "auxiliary_loss_mode", reasons);
    RequireText(extended.targetClippingDefinition,
                prefix + "target_clipping_definition", reasons);
    RequireText(extended.objectiveNormalizationIdentity,
                prefix + "objective_normalization_identity", reasons);
    RequireText(extended.checkpointPolicyScope,
                prefix + "checkpoint_policy_scope", reasons);
    RequireText(extended.checkpointPolicyStopMode,
                prefix + "checkpoint_policy_stop_mode", reasons);
    if (extended.checkpointPolicyGraceEvaluations <= 0 ||
        extended.checkpointPolicyRevision <= 0)
        Add(reasons, prefix + "checkpoint_policy_identity_invalid");
    RequireText(value.runProvenance.gitCommit, prefix + "git_commit", reasons);
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
}

#define EA_COMPARE_FIELD(field, name)                                        \
    do                                                                       \
    {                                                                        \
        if (control.field != treatment.field) Add(reasons, name);            \
    } while (false)

void ValidateExperimentPair(const ScientificConfiguration& control,
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
    EA_COMPARE_FIELD(featureWarmupScope, "feature_warmup_scope_mismatch");
    EA_COMPARE_FIELD(donchianMode, "donchian_mode_mismatch");
    EA_COMPARE_FIELD(donchianLookback, "donchian_lookback_mismatch");
    EA_COMPARE_FIELD(resumeExpandInputWidth,
                     "resume_expand_input_width_mismatch");
    EA_COMPARE_FIELD(experimentObjective, "training_objective_mismatch");
    EA_COMPARE_FIELD(runProvenance, "run_provenance_mismatch");
    if (control.experimentId == treatment.experimentId)
        Add(reasons, "experiment_id_reused");
}

bool ValidOwnCheckpointResume(
    const FeatureAblationPairEvaluation::ArmEvidence& arm)
{
    const auto& configuration = arm.authoritative.configuration;
    const auto& provenance = arm.resumeCheckpointProvenance;
    return configuration.resumeModelId &&
        provenance &&
        provenance->resumeModelId == *configuration.resumeModelId &&
        provenance->modelExperimentId == configuration.experimentId &&
        provenance->ownExperimentCheckpoint &&
        provenance->checkpointEpoch &&
        *provenance->checkpointEpoch > 0;
}

void ValidateResumeCompatibility(
    const FeatureAblationPairEvaluation::ArmEvidence& control,
    const FeatureAblationPairEvaluation::ArmEvidence& treatment,
    std::vector<std::string>& reasons)
{
    const auto& controlResume =
        control.authoritative.configuration.resumeModelId;
    const auto& treatmentResume =
        treatment.authoritative.configuration.resumeModelId;

    // Preserve the original behavior when the persisted initialization
    // identity is literally equal, including fresh/fresh.
    if (controlResume == treatmentResume) return;

    // A fresh arm paired with a resumed arm is not scientifically equivalent.
    if (!controlResume || !treatmentResume)
    {
        Add(reasons, "resume_model_id_mismatch");
        return;
    }

    // Different model IDs are compatible only for the narrowly defined case
    // where each ID is the arm's own persisted checkpoint and both checkpoints
    // represent the same continuation epoch.
    if (!ValidOwnCheckpointResume(control) ||
        !ValidOwnCheckpointResume(treatment))
    {
        Add(reasons, "resume_model_id_mismatch");
        return;
    }

    if (control.resumeCheckpointProvenance->checkpointEpoch !=
        treatment.resumeCheckpointProvenance->checkpointEpoch)
        Add(reasons, "resume_checkpoint_epoch_mismatch");
}

void ValidateFinalModelPair(const ScientificConfiguration& control,
                            const ScientificConfiguration& treatment,
                            std::vector<std::string>& reasons)
{
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
}

#undef EA_COMPARE_FIELD

void ValidateExtendedPair(const ExtendedScientificConfiguration& control,
                          const ExtendedScientificConfiguration& treatment,
                          std::vector<std::string>& reasons)
{
    if (control == treatment) return;
#define EA_COMPARE_EXTENDED(field, name)                                     \
    do                                                                       \
    {                                                                        \
        if (control.field != treatment.field) Add(reasons, name);            \
    } while (false)
    EA_COMPARE_EXTENDED(trainingObjectiveVersion,
                        "training_objective_version_mismatch");
    EA_COMPARE_EXTENDED(lossDefinitionVersion,
                        "loss_definition_version_mismatch");
    EA_COMPARE_EXTENDED(auxiliaryLossMode, "auxiliary_loss_mode_mismatch");
    EA_COMPARE_EXTENDED(auxiliaryLossCoefficient,
                        "auxiliary_loss_coefficient_mismatch");
    EA_COMPARE_EXTENDED(regressionTargetDefinition,
                        "regression_target_definition_mismatch");
    EA_COMPARE_EXTENDED(regressionNormalizationIdentity,
                        "regression_normalization_identity_mismatch");
    EA_COMPARE_EXTENDED(robustLossDefinition,
                        "robust_loss_definition_mismatch");
    EA_COMPARE_EXTENDED(robustLossDelta, "robust_loss_delta_mismatch");
    EA_COMPARE_EXTENDED(targetClippingDefinition,
                        "target_clipping_definition_mismatch");
    EA_COMPARE_EXTENDED(objectiveNormalizationIdentity,
                        "objective_normalization_identity_mismatch");
    EA_COMPARE_EXTENDED(checkpointInferenceEnabled,
                        "checkpoint_inference_enabled_mismatch");
    EA_COMPARE_EXTENDED(checkpointInferenceMinimumEpoch,
                        "checkpoint_inference_minimum_epoch_mismatch");
    EA_COMPARE_EXTENDED(checkpointInferenceInterval,
                        "checkpoint_inference_interval_mismatch");
    EA_COMPARE_EXTENDED(checkpointPolicyEnabled,
                        "checkpoint_policy_enabled_mismatch");
    EA_COMPARE_EXTENDED(checkpointPolicyMinimumLeaderScore,
                        "checkpoint_policy_minimum_leader_score_mismatch");
    EA_COMPARE_EXTENDED(checkpointPolicyMinimumInferenceAccuracy,
                        "checkpoint_policy_minimum_inference_accuracy_mismatch");
    EA_COMPARE_EXTENDED(checkpointPolicyTopN,
                        "checkpoint_policy_top_n_mismatch");
    EA_COMPARE_EXTENDED(checkpointPolicyScope,
                        "checkpoint_policy_scope_mismatch");
    EA_COMPARE_EXTENDED(checkpointPolicyStopMode,
                        "checkpoint_policy_stop_mode_mismatch");
    EA_COMPARE_EXTENDED(checkpointPolicyGraceEvaluations,
                        "checkpoint_policy_grace_evaluations_mismatch");
    EA_COMPARE_EXTENDED(checkpointPolicyRevision,
                        "checkpoint_policy_revision_mismatch");
    EA_COMPARE_EXTENDED(checkpointPolicyHash,
                        "checkpoint_policy_hash_mismatch");
#undef EA_COMPARE_EXTENDED
}

bool ValidateAblationIdentity(const ScientificConfiguration& control,
                              const ScientificConfiguration& treatment,
                              ComparisonResult& result)
{
    const std::size_t invalidReasonCountBefore =
        result.invalidReasons.size();

    try
    {
        const FeatureAblationMask controlMask =
            FeatureAblationMask::Parse(control.featureAblationMask);
        const FeatureAblationMask treatmentMask =
            FeatureAblationMask::Parse(treatment.featureAblationMask);
        const FeatureAblationMask expected = FeatureAblationMask::Parse(
            std::string(kEconomicEventConsensusAblationMaskText));
        const std::string controlCanonical = controlMask.CanonicalText();
        const std::string treatmentCanonical = treatmentMask.CanonicalText();
        const std::string expectedCanonical = expected.CanonicalText();
        result.canonicalAblatedFeatureSet = expectedCanonical;

        if (controlMask.empty() && treatmentCanonical == expectedCanonical)
            Add(result.invalidReasons, "reversed_control_treatment_order");
        else
        {
            if (controlCanonical != expectedCanonical)
                Add(result.invalidReasons,
                    "control_ablation_not_consensus_feature_family");
            if (!treatmentMask.empty())
                Add(result.invalidReasons,
                    "treatment_contains_feature_ablations");
        }
        if (result.invalidReasons.size() != invalidReasonCountBefore)
            return false;

        result.ablationIdentityCanonical =
            "feature_ablation_pair_identity_v1;control_mask=" +
            controlCanonical + ";treatment_mask=EMPTY;ablated_features=" +
            expectedCanonical +
            ";direction=control_ablated_treatment_enabled;";
        result.ablationIdentityHash = TrainingObjective::DeterministicHash(
            result.ablationIdentityCanonical);
        return true;
    }
    catch (const std::exception&)
    {
        Add(result.invalidReasons, "feature_ablation_mask_invalid");
        return false;
    }
}

void ValidateClassification(const FeatureAblationPairEvaluation::ArmEvidence& arm,
                            std::string_view role,
                            std::vector<std::string>& invalid,
                            std::vector<std::string>& incomplete)
{
    const std::string prefix = std::string(role) + '_';
    const auto& shared = arm.authoritative;
    if (!shared.classification)
    {
        Add(incomplete, arm.exactFinalInferenceResultId
            ? prefix + "final_analysis_missing"
            : prefix + "final_inference_missing");
        return;
    }
    const auto& value = *shared.classification;
    const auto& configuration = shared.configuration;
    if (!shared.finalModelId || value.inferenceResultId <= 0 ||
        value.analysisId <= 0 || value.modelId != *shared.finalModelId ||
        value.analysisModelId != *shared.finalModelId ||
        value.analysisExperimentId != configuration.experimentId)
        Add(invalid, prefix + "final_classification_provenance_mismatch");
    if (value.inferenceScope != "final" || value.analysisScope != "final" ||
        value.checkpointEvalId || value.parentExperimentId ||
        value.analysisCheckpointEvalId || value.analysisParentExperimentId)
        Add(invalid, prefix + "classification_not_exact_final_scope");
    if (value.status != "completed" || value.analysisStatus != "completed")
        Add(incomplete, prefix + "final_classification_not_completed");
    if (value.symbol != configuration.symbol ||
        value.predictionHorizon != configuration.predictionHorizon ||
        std::fabs(value.threshold - configuration.threshold) > 1.0e-7 ||
        value.windowSize != configuration.windowSize ||
        value.labelRuleId != configuration.labelRuleId ||
        value.targetType != configuration.targetType ||
        value.inferenceStart != configuration.inferenceStart ||
        value.inferenceEnd != configuration.inferenceEnd ||
        value.completedEpochs != configuration.targetEpochs)
        Add(invalid, prefix + "final_classification_context_mismatch");
    const std::optional<double> requiredMetrics[] = {
        value.inferenceAccuracy, value.acceptAccuracy, value.acceptRate,
        value.predictedNeutralProportion, value.leaderScore};
    for (const auto& metric : requiredMetrics)
    {
        if (!metric) Add(incomplete, prefix + "required_metric_missing");
        else if (!Finite(*metric))
            Add(invalid, prefix + "classification_metric_nonfinite");
    }
    // experiment_analysis_result.infer_accuracy is persisted at six
    // decimal places, while inference_eval_result.accuracy retains the
    // underlying full-precision ratio. Values representing the same result
    // may therefore differ by at most half of one unit in the sixth decimal.
    constexpr double kAnalysisAccuracyRoundingTolerance = 5.0e-7;
    if (value.accuracy && value.inferenceAccuracy &&
        std::fabs(*value.accuracy - *value.inferenceAccuracy) >
            kAnalysisAccuracyRoundingTolerance)
    {
        Add(invalid, prefix + "inference_analysis_accuracy_mismatch");
    }
}

void ValidateProfitability(const FeatureAblationPairEvaluation::ArmEvidence& arm,
                           std::string_view role,
                           std::vector<std::string>& invalid,
                           std::vector<std::string>& incomplete)
{
    const std::string prefix = std::string(role) + '_';
    const auto& shared = arm.authoritative;
    if (!shared.profitability)
    {
        Add(incomplete, prefix + "final_profitability_evidence_unavailable");
        return;
    }
    const auto& value = *shared.profitability;
    if (!shared.finalModelId || !shared.classification ||
        value.observationId <= 0 ||
        value.experimentId != shared.configuration.experimentId ||
        value.modelId != *shared.finalModelId ||
        value.inferenceResultId != shared.classification->inferenceResultId)
        Add(invalid, prefix + "profitability_provenance_mismatch");
    if (value.inferenceScope != "final" || value.checkpointEvalId)
        Add(invalid, prefix + "profitability_not_exact_final_scope");
    if (value.inferenceStart != shared.configuration.inferenceStart ||
        value.inferenceEnd != shared.configuration.inferenceEnd)
        Add(invalid, prefix + "profitability_inference_range_mismatch");
    if (value.actionableCount > value.predictionCount ||
        !Finite(value.aggregateTerminalHorizonLogReturnSum) ||
        (value.averageTerminalHorizonLogReturnPerActionablePrediction &&
         !Finite(*value.averageTerminalHorizonLogReturnPerActionablePrediction)))
        Add(invalid, prefix + "profitability_values_invalid");
    if ((value.actionableCount == 0) ==
        value.averageTerminalHorizonLogReturnPerActionablePrediction.has_value())
        Add(invalid, prefix + "profitability_average_presence_mismatch");
    if (value.metricDefinitionCanonical !=
            InferenceProfitability::kMetricDefinitionCanonical ||
        value.metricDefinitionHash !=
            InferenceProfitability::MetricDefinitionHash())
        Add(invalid, prefix + "profitability_metric_definition_mismatch");
}

} // namespace

ComparisonResult Compare(const FeatureAblationPairEvaluation::ArmEvidence& control,
                         const FeatureAblationPairEvaluation::ArmEvidence& treatment)
{
    ComparisonResult result;
    ValidateExperimentConfiguration(control, "control", result.invalidReasons);
    ValidateExperimentConfiguration(treatment, "treatment", result.invalidReasons);
    const auto& controlConfiguration = control.authoritative.configuration;
    const auto& treatmentConfiguration = treatment.authoritative.configuration;
    ValidateExperimentPair(controlConfiguration, treatmentConfiguration,
                           result.invalidReasons);
    ValidateResumeCompatibility(control, treatment, result.invalidReasons);
    ValidateExtendedPair(control.extended, treatment.extended,
                         result.invalidReasons);
    const bool validAblation = ValidateAblationIdentity(
        controlConfiguration, treatmentConfiguration, result);
    if (!validAblation)
    {
        result.disposition = Disposition::InvalidAblationPair;
        return result;
    }
    if (!result.invalidReasons.empty())
    {
        result.disposition = Disposition::IncompatibleConfiguration;
        return result;
    }

    const bool controlComplete =
        control.authoritative.experimentStatus == "completed" &&
        control.authoritative.experimentPhase == "done";
    const bool treatmentComplete =
        treatment.authoritative.experimentStatus == "completed" &&
        treatment.authoritative.experimentPhase == "done";
    if (!controlComplete)
        Add(result.incompleteReasons, "control_experiment_not_complete");
    if (!treatmentComplete)
        Add(result.incompleteReasons, "treatment_experiment_not_complete");
    if (!controlComplete || !treatmentComplete)
    {
        Add(result.incompleteReasons,
            "final_model_input_contract_provenance_unavailable");
        result.disposition = Disposition::ComparableIncomplete;
        return result;
    }

    if (!control.authoritative.finalModelId)
        Add(result.incompleteReasons, "control_final_model_missing");
    if (!treatment.authoritative.finalModelId)
        Add(result.incompleteReasons, "treatment_final_model_missing");
    if (!result.incompleteReasons.empty())
    {
        result.disposition = Disposition::MissingFinalInference;
        return result;
    }

    ValidateFinalModelPair(controlConfiguration, treatmentConfiguration,
                           result.invalidReasons);
    if (!result.invalidReasons.empty())
    {
        result.disposition = Disposition::IncompatibleConfiguration;
        return result;
    }

    ValidateClassification(control, "control", result.invalidReasons,
                           result.incompleteReasons);
    ValidateClassification(treatment, "treatment", result.invalidReasons,
                           result.incompleteReasons);
    if (!result.invalidReasons.empty())
    {
        result.disposition = Disposition::IncompatibleConfiguration;
        return result;
    }
    if (!result.incompleteReasons.empty())
    {
        result.disposition = Disposition::MissingFinalInference;
        return result;
    }

    ValidateProfitability(control, "control", result.invalidReasons,
                          result.incompleteReasons);
    ValidateProfitability(treatment, "treatment", result.invalidReasons,
                          result.incompleteReasons);
    if (!result.invalidReasons.empty())
    {
        result.disposition = Disposition::IncompatibleConfiguration;
        return result;
    }
    if (!result.incompleteReasons.empty())
    {
        result.disposition = Disposition::ProfitabilityEvidenceUnavailable;
        return result;
    }

    const auto& cc = *control.authoritative.classification;
    const auto& tc = *treatment.authoritative.classification;
    const auto& cp = *control.authoritative.profitability;
    const auto& tp = *treatment.authoritative.profitability;
    result.predictionCount = CountDelta(cp.predictionCount, tp.predictionCount);
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
    result.neutralProportion = Delta(
        cc.predictedNeutralProportion, tc.predictedNeutralProportion);
    result.leaderScore = Delta(cc.leaderScore, tc.leaderScore);
    result.disposition = Disposition::ComparableComplete;
    return result;
}

std::string EvaluationIdentityCanonical(
    const FeatureAblationPairEvaluation::ArmEvidence& control,
    const FeatureAblationPairEvaluation::ArmEvidence& treatment,
    const ComparisonResult& result)
{
    const auto optionalId = [](const std::optional<long long>& value)
    {
        return value ? std::to_string(*value) : std::string("NULL");
    };
    const auto observationId = [](
        const FeatureAblationPairEvaluation::ArmEvidence& arm)
    {
        return arm.authoritative.profitability
            ? std::to_string(arm.authoritative.profitability->observationId)
            : std::string("NULL");
    };
    return "feature_ablation_pair_evaluation_v1;control_experiment_id=" +
        std::to_string(control.authoritative.configuration.experimentId) +
        ";treatment_experiment_id=" +
        std::to_string(treatment.authoritative.configuration.experimentId) +
        ";ablation_identity_hash=" + result.ablationIdentityHash +
        ";control_final_inference_result_id=" +
        optionalId(control.exactFinalInferenceResultId) +
        ";treatment_final_inference_result_id=" +
        optionalId(treatment.exactFinalInferenceResultId) +
        ";control_profitability_observation_id=" + observationId(control) +
        ";treatment_profitability_observation_id=" + observationId(treatment) +
        ";disposition=" + DispositionText(result.disposition) + ";";
}

std::string EvaluationIdentityHash(
    const FeatureAblationPairEvaluation::ArmEvidence& control,
    const FeatureAblationPairEvaluation::ArmEvidence& treatment,
    const ComparisonResult& result)
{
    return TrainingObjective::DeterministicHash(
        EvaluationIdentityCanonical(control, treatment, result));
}

std::pair<long long, long long> ParseExperimentIdPair(std::string_view text)
{
    const std::size_t separator = text.find(':');
    if (separator == std::string_view::npos || separator == 0 ||
        separator + 1 >= text.size() ||
        text.find(':', separator + 1) != std::string_view::npos)
        throw std::invalid_argument(
            "--compare-feature-ablation-pair requires CONTROL_ID:TREATMENT_ID");
    const auto parse = [](std::string_view value) -> long long
    {
        long long parsed = 0;
        const auto [end, error] = std::from_chars(
            value.data(), value.data() + value.size(), parsed);
        if (error != std::errc{} || end != value.data() + value.size() ||
            parsed <= 0)
            throw std::invalid_argument(
                "feature-ablation pair experiment IDs must be positive integers");
        return parsed;
    };
    return {parse(text.substr(0, separator)),
            parse(text.substr(separator + 1))};
}

std::string DispositionText(Disposition value)
{
    switch (value)
    {
        case Disposition::ComparableComplete: return "comparable_complete";
        case Disposition::ComparableIncomplete: return "comparable_incomplete";
        case Disposition::IncompatibleConfiguration:
            return "incompatible_configuration";
        case Disposition::MissingFinalInference:
            return "missing_final_inference";
        case Disposition::AmbiguousFinalInference:
            return "ambiguous_final_inference";
        case Disposition::ProfitabilityEvidenceUnavailable:
            return "profitability_evidence_unavailable";
        case Disposition::InvalidAblationPair: return "invalid_ablation_pair";
    }
    throw std::invalid_argument("unknown_feature_ablation_pair_disposition");
}

int ExitCode(Disposition value)
{
    switch (value)
    {
        case Disposition::ComparableComplete: return 0;
        case Disposition::ComparableIncomplete:
        case Disposition::MissingFinalInference:
        case Disposition::ProfitabilityEvidenceUnavailable:
            return 4;
        case Disposition::IncompatibleConfiguration:
        case Disposition::AmbiguousFinalInference:
        case Disposition::InvalidAblationPair:
            return 3;
    }
    return 3;
}

} // namespace EA::FeatureAblationPairEvaluation
