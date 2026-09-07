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
                  std::optional<double> ablation)
{
    MetricDelta result{control, ablation, std::nullopt};
    if (control && ablation && Finite(*control) && Finite(*ablation))
        result.controlMinusAblation = *control - *ablation;
    return result;
}

MetricDelta CountDelta(std::uint64_t control, std::uint64_t ablation)
{
    return Delta(static_cast<double>(control), static_cast<double>(ablation));
}

MetricDelta CountDelta(const std::optional<std::uint64_t>& control,
                       const std::optional<std::uint64_t>& ablation)
{
    return Delta(control ? std::optional<double>{static_cast<double>(*control)}
                         : std::nullopt,
                 ablation ? std::optional<double>{static_cast<double>(*ablation)}
                          : std::nullopt);
}

std::optional<std::uint64_t> PredictionCount(
    const SharedEvidence::ClassificationEvidence& value)
{
    if (!value.predictedDownCount || !value.predictedNeutralCount ||
        !value.predictedUpCount)
        return std::nullopt;
    return *value.predictedDownCount + *value.predictedNeutralCount +
        *value.predictedUpCount;
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
        !Finite(extended.baseLearningRate) ||
        (extended.robustLossDelta && !Finite(*extended.robustLossDelta)))
        Add(reasons, prefix + "nonfinite_configuration");
    if (extended.batchSize <= 0)
        Add(reasons, prefix + "batch_size_invalid");
    if ((extended.configuredModelInputWidth.has_value() !=
         extended.configuredModelInputLayoutVersion.has_value()) ||
        (extended.configuredModelInputWidth &&
         *extended.configuredModelInputWidth <= 0) ||
        (extended.configuredModelInputLayoutVersion &&
         *extended.configuredModelInputLayoutVersion <= 0))
        Add(reasons, prefix + "configured_model_input_identity_invalid");
    if (extended.economicCalendarSnapshotId.has_value() !=
            extended.economicCalendarSnapshotHash.has_value() ||
        (extended.economicCalendarSnapshotId &&
         *extended.economicCalendarSnapshotId <= 0) ||
        (extended.economicCalendarSnapshotHash &&
         !TaggedHash(*extended.economicCalendarSnapshotHash)))
        Add(reasons, prefix + "economic_calendar_snapshot_identity_invalid");
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
    RequireText(extended.continuationPolicyScientificIdentity,
                prefix + "continuation_policy_scientific_identity", reasons);
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
        if (control.field != ablation.field) Add(reasons, name);            \
    } while (false)

void ValidateExperimentPair(const ScientificConfiguration& control,
                            const ScientificConfiguration& ablation,
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
    if (control.experimentId == ablation.experimentId)
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
    const FeatureAblationPairEvaluation::ArmEvidence& ablation,
    std::vector<std::string>& reasons)
{
    const auto& controlResume =
        control.authoritative.configuration.resumeModelId;
    const auto& ablationResume =
        ablation.authoritative.configuration.resumeModelId;

    // Preserve the original behavior when the persisted initialization
    // identity is literally equal, including fresh/fresh.
    if (controlResume == ablationResume) return;

    // A fresh arm paired with a resumed arm is not scientifically equivalent.
    if (!controlResume || !ablationResume)
    {
        Add(reasons, "resume_model_id_mismatch");
        return;
    }

    // Different model IDs are compatible only for the narrowly defined case
    // where each ID is the arm's own persisted checkpoint and both checkpoints
    // represent the same continuation epoch.
    if (!ValidOwnCheckpointResume(control) ||
        !ValidOwnCheckpointResume(ablation))
    {
        Add(reasons, "resume_model_id_mismatch");
        return;
    }

    if (control.resumeCheckpointProvenance->checkpointEpoch !=
        ablation.resumeCheckpointProvenance->checkpointEpoch)
        Add(reasons, "resume_checkpoint_epoch_mismatch");
}

void ValidateFinalModelPair(const ScientificConfiguration& control,
                            const ScientificConfiguration& ablation,
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

void ValidateConfiguredAndFinalInputIdentity(
    const FeatureAblationPairEvaluation::ArmEvidence& arm,
    std::string_view role,
    std::vector<std::string>& reasons)
{
    const auto& configured = arm.extended;
    const auto& model = arm.authoritative.configuration;
    if (configured.configuredModelInputWidth &&
        *configured.configuredModelInputWidth != model.inputWidth)
        Add(reasons, std::string(role) +
            "_configured_final_model_input_width_mismatch");
    if (configured.configuredModelInputLayoutVersion &&
        *configured.configuredModelInputLayoutVersion !=
            model.modelInputLayoutVersion)
        Add(reasons, std::string(role) +
            "_configured_final_model_input_layout_mismatch");
}

#undef EA_COMPARE_FIELD

void ValidateExtendedPair(const ExtendedScientificConfiguration& control,
                          const ExtendedScientificConfiguration& ablation,
                          std::vector<std::string>& reasons)
{
    if (control == ablation) return;
#define EA_COMPARE_EXTENDED(field, name)                                     \
    do                                                                       \
    {                                                                        \
        if (control.field != ablation.field) Add(reasons, name);            \
    } while (false)
    EA_COMPARE_EXTENDED(configuredModelInputWidth,
                        "configured_model_input_width_mismatch");
    EA_COMPARE_EXTENDED(configuredModelInputLayoutVersion,
                        "configured_model_input_layout_mismatch");
    EA_COMPARE_EXTENDED(economicCalendarSnapshotId,
                        "economic_calendar_snapshot_id_mismatch");
    EA_COMPARE_EXTENDED(economicCalendarSnapshotHash,
                        "economic_calendar_snapshot_hash_mismatch");
    EA_COMPARE_EXTENDED(baseLearningRate, "base_learning_rate_mismatch");
    EA_COMPARE_EXTENDED(batchSize, "batch_size_mismatch");
    EA_COMPARE_EXTENDED(freshInitializationSeed,
                        "fresh_initialization_seed_mismatch");
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
    EA_COMPARE_EXTENDED(continuationPolicyEnabled,
                        "continuation_policy_enabled_mismatch");
    EA_COMPARE_EXTENDED(continuationPolicyScientificIdentity,
                        "continuation_policy_scientific_identity_mismatch");
#undef EA_COMPARE_EXTENDED
}

bool ValidateAblationIdentity(const ScientificConfiguration& control,
                              const ScientificConfiguration& ablation,
                              std::string_view expectedAblationMask,
                              ComparisonResult& result)
{
    const std::size_t invalidReasonCountBefore =
        result.invalidReasons.size();

    try
    {
        const FeatureAblationMask controlMask =
            FeatureAblationMask::Parse(control.featureAblationMask);
        const FeatureAblationMask ablationMask =
            FeatureAblationMask::Parse(ablation.featureAblationMask);
        const FeatureAblationMask expected = FeatureAblationMask::Parse(
            std::string(expectedAblationMask));
        const std::string controlCanonical = controlMask.CanonicalText();
        const std::string ablationCanonical = ablationMask.CanonicalText();
        const std::string expectedCanonical = expected.CanonicalText();
        result.canonicalAblatedFeatureSet = expectedCanonical;

        if (expected.empty())
            Add(result.invalidReasons, "expected_ablation_mask_empty");
        if (!controlMask.empty())
            Add(result.invalidReasons, "control_feature_ablation_mask_not_empty");
        if (ablationCanonical != expectedCanonical)
            Add(result.invalidReasons, "ablation_mask_does_not_match_expected");
        if (result.invalidReasons.size() != invalidReasonCountBefore)
            return false;

        result.ablationIdentityCanonical =
            "feature_ablation_pair_identity_v2;control_mask=EMPTY;ablation_mask=" +
            ablationCanonical + ";expected_ablation_mask=" +
            expectedCanonical +
            ";direction=control_minus_ablation;";
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

EvidenceClassification ClassifyCausalSurpriseEvidence(
    const FeatureAblationPairEvaluation::ArmEvidence& control,
    const FeatureAblationPairEvaluation::ArmEvidence& ablation,
    ComparisonResult& result)
{
    const auto validateIdentity = [&result](
        const FeatureAblationPairEvaluation::ArmEvidence& arm,
        std::string_view role)
    {
        const std::string prefix = std::string(role) + '_';
        if (!arm.extended.configuredModelInputWidth ||
            !arm.extended.configuredModelInputLayoutVersion)
            Add(result.invalidReasons,
                prefix + "causal_surprise_model_input_identity_missing");
        if (arm.extended.configuredModelInputWidth &&
            *arm.extended.configuredModelInputWidth !=
                kCausalSurpriseModelInputWidth)
            Add(result.invalidReasons,
                prefix + "causal_surprise_model_input_width_not_77");
    };
    validateIdentity(control, "control");
    validateIdentity(ablation, "ablation");

    const auto controlLayout =
        control.extended.configuredModelInputLayoutVersion;
    const auto ablationLayout =
        ablation.extended.configuredModelInputLayoutVersion;
    if (!controlLayout || !ablationLayout)
        return EvidenceClassification::IncompatibleOrInvalidEvidence;
    if (*controlLayout != *ablationLayout)
    {
        Add(result.invalidReasons,
            "causal_surprise_semantic_layout_mismatch");
        return EvidenceClassification::IncompatibleOrInvalidEvidence;
    }
    if (*controlLayout == kPreFixCausalSurpriseSemanticLayoutVersion)
    {
        Add(result.classificationReasons,
            "semantic_layout_6_uses_pre_fix_exact_cutoff_contract");
        return EvidenceClassification::PreFixCausalSurpriseEvidence;
    }
    if (*controlLayout != kCorrectedCausalSurpriseSemanticLayoutVersion)
    {
        Add(result.invalidReasons,
            "causal_surprise_semantic_layout_not_supported");
        return EvidenceClassification::IncompatibleOrInvalidEvidence;
    }

    const auto requireSnapshot = [&result](
        const ExtendedScientificConfiguration& extended,
        std::string_view role)
    {
        if (!extended.economicCalendarSnapshotId ||
            !extended.economicCalendarSnapshotHash)
            Add(result.invalidReasons, std::string(role) +
                "_corrected_causal_surprise_snapshot_identity_missing");
    };
    requireSnapshot(control.extended, "control");
    requireSnapshot(ablation.extended, "ablation");
    if (!result.invalidReasons.empty())
        return EvidenceClassification::IncompatibleOrInvalidEvidence;

    Add(result.classificationReasons,
        "semantic_layout_7_uses_corrected_strict_cutoff_contract");
    return EvidenceClassification::CorrectedCausalSurprisePairEvidence;
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
    const std::optional<double> optionalProportions[] = {
        value.predictedDownProportion, value.predictedUpProportion};
    for (const auto& proportion : optionalProportions)
        if (proportion && (!Finite(*proportion) || *proportion < 0.0 ||
                           *proportion > 1.0))
            Add(invalid, prefix + "class_proportion_invalid");
    if (value.predictedNeutralProportion &&
        (*value.predictedNeutralProportion < 0.0 ||
         *value.predictedNeutralProportion > 1.0))
        Add(invalid, prefix + "class_proportion_invalid");
    const auto predictionCount = PredictionCount(value);
    if (value.acceptedPredictionCount && predictionCount &&
        *value.acceptedPredictionCount > *predictionCount)
        Add(invalid, prefix + "accepted_prediction_count_invalid");
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
                         const FeatureAblationPairEvaluation::ArmEvidence& ablation,
                         std::string_view expectedAblationMask)
{
    ComparisonResult result;
    ValidateExperimentConfiguration(control, "control", result.invalidReasons);
    ValidateExperimentConfiguration(ablation, "ablation", result.invalidReasons);
    const auto& controlConfiguration = control.authoritative.configuration;
    const auto& ablationConfiguration = ablation.authoritative.configuration;
    ValidateExperimentPair(controlConfiguration, ablationConfiguration,
                           result.invalidReasons);
    ValidateResumeCompatibility(control, ablation, result.invalidReasons);
    ValidateExtendedPair(control.extended, ablation.extended,
                         result.invalidReasons);
    const bool validAblation = ValidateAblationIdentity(
        controlConfiguration, ablationConfiguration, expectedAblationMask,
        result);
    const bool causalSurpriseEvaluation =
        result.canonicalAblatedFeatureSet ==
        kCausalEconomicEventSurpriseAblationMaskText;
    if (causalSurpriseEvaluation)
        result.evidenceClassification = ClassifyCausalSurpriseEvidence(
            control, ablation, result);
    if (!validAblation)
    {
        if (causalSurpriseEvaluation)
            result.evidenceClassification =
                EvidenceClassification::IncompatibleOrInvalidEvidence;
        result.disposition = Disposition::InvalidAblationPair;
        return result;
    }
    if (!result.invalidReasons.empty())
    {
        if (causalSurpriseEvaluation)
            result.evidenceClassification =
                EvidenceClassification::IncompatibleOrInvalidEvidence;
        result.disposition = Disposition::IncompatibleConfiguration;
        return result;
    }

    const bool controlComplete =
        control.authoritative.experimentStatus == "completed" &&
        control.authoritative.experimentPhase == "done";
    const bool ablationComplete =
        ablation.authoritative.experimentStatus == "completed" &&
        ablation.authoritative.experimentPhase == "done";
    if (!controlComplete)
        Add(result.incompleteReasons, "control_experiment_not_complete");
    if (!ablationComplete)
        Add(result.incompleteReasons, "ablation_experiment_not_complete");
    if (!controlComplete || !ablationComplete)
    {
        Add(result.incompleteReasons,
            "final_model_input_contract_provenance_unavailable");
        result.disposition = Disposition::ComparableIncomplete;
        return result;
    }

    if (!control.authoritative.finalModelId)
        Add(result.incompleteReasons, "control_final_model_missing");
    if (!ablation.authoritative.finalModelId)
        Add(result.incompleteReasons, "ablation_final_model_missing");
    if (!result.incompleteReasons.empty())
    {
        result.disposition = Disposition::MissingFinalInference;
        return result;
    }

    ValidateFinalModelPair(controlConfiguration, ablationConfiguration,
                           result.invalidReasons);
    ValidateConfiguredAndFinalInputIdentity(
        control, "control", result.invalidReasons);
    ValidateConfiguredAndFinalInputIdentity(
        ablation, "ablation", result.invalidReasons);
    if (!result.invalidReasons.empty())
    {
        if (causalSurpriseEvaluation)
            result.evidenceClassification =
                EvidenceClassification::IncompatibleOrInvalidEvidence;
        result.disposition = Disposition::IncompatibleConfiguration;
        return result;
    }

    ValidateClassification(control, "control", result.invalidReasons,
                           result.incompleteReasons);
    ValidateClassification(ablation, "ablation", result.invalidReasons,
                           result.incompleteReasons);
    if (!result.invalidReasons.empty())
    {
        if (causalSurpriseEvaluation)
            result.evidenceClassification =
                EvidenceClassification::IncompatibleOrInvalidEvidence;
        result.disposition = Disposition::IncompatibleConfiguration;
        return result;
    }
    if (!result.incompleteReasons.empty())
    {
        result.disposition = Disposition::MissingFinalInference;
        return result;
    }

    const auto& cc = *control.authoritative.classification;
    const auto& ac = *ablation.authoritative.classification;
    result.predictionCount = CountDelta(PredictionCount(cc),
                                        PredictionCount(ac));
    result.predictedDownCount = CountDelta(
        cc.predictedDownCount, ac.predictedDownCount);
    result.predictedNeutralCount = CountDelta(
        cc.predictedNeutralCount, ac.predictedNeutralCount);
    result.predictedUpCount = CountDelta(
        cc.predictedUpCount, ac.predictedUpCount);
    result.acceptedPredictionCount = CountDelta(
        cc.acceptedPredictionCount, ac.acceptedPredictionCount);
    result.inferenceAccuracy = Delta(cc.inferenceAccuracy, ac.inferenceAccuracy);
    result.acceptAccuracy = Delta(cc.acceptAccuracy, ac.acceptAccuracy);
    result.acceptRate = Delta(cc.acceptRate, ac.acceptRate);
    result.downProportion = Delta(
        cc.predictedDownProportion, ac.predictedDownProportion);
    result.neutralProportion = Delta(
        cc.predictedNeutralProportion, ac.predictedNeutralProportion);
    result.upProportion = Delta(
        cc.predictedUpProportion, ac.predictedUpProportion);
    result.leaderScore = Delta(cc.leaderScore, ac.leaderScore);

    ValidateProfitability(control, "control", result.invalidReasons,
                          result.incompleteReasons);
    ValidateProfitability(ablation, "ablation", result.invalidReasons,
                          result.incompleteReasons);
    if (!result.invalidReasons.empty())
    {
        if (causalSurpriseEvaluation)
            result.evidenceClassification =
                EvidenceClassification::IncompatibleOrInvalidEvidence;
        result.disposition = Disposition::IncompatibleConfiguration;
        return result;
    }
    if (!result.incompleteReasons.empty())
    {
        result.disposition = Disposition::ProfitabilityEvidenceUnavailable;
        return result;
    }

    const auto& cp = *control.authoritative.profitability;
    const auto& ap = *ablation.authoritative.profitability;
    result.actionableCount = CountDelta(cp.actionableCount, ap.actionableCount);
    result.aggregateProfitability = Delta(
        cp.aggregateTerminalHorizonLogReturnSum,
        ap.aggregateTerminalHorizonLogReturnSum);
    result.averageProfitability = Delta(
        cp.averageTerminalHorizonLogReturnPerActionablePrediction,
        ap.averageTerminalHorizonLogReturnPerActionablePrediction);
    result.disposition = Disposition::ComparableComplete;
    return result;
}

ComparisonResult CompareLegacyConsensusPair(
    const FeatureAblationPairEvaluation::ArmEvidence& legacyAblatedControl,
    const FeatureAblationPairEvaluation::ArmEvidence& legacyEnabledTreatment)
{
    return Compare(legacyEnabledTreatment, legacyAblatedControl,
                   kEconomicEventConsensusAblationMaskText);
}

std::string EvaluationIdentityCanonical(
    const FeatureAblationPairEvaluation::ArmEvidence& control,
    const FeatureAblationPairEvaluation::ArmEvidence& ablation,
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
    const auto optionalInt = [](const std::optional<int>& value)
    {
        return value ? std::to_string(*value) : std::string("NULL");
    };
    return "feature_ablation_pair_evaluation_v3;control_experiment_id=" +
        std::to_string(control.authoritative.configuration.experimentId) +
        ";ablation_experiment_id=" +
        std::to_string(ablation.authoritative.configuration.experimentId) +
        ";ablation_identity_hash=" + result.ablationIdentityHash +
        ";control_final_inference_result_id=" +
        optionalId(control.exactFinalInferenceResultId) +
        ";ablation_final_inference_result_id=" +
        optionalId(ablation.exactFinalInferenceResultId) +
        ";control_profitability_observation_id=" + observationId(control) +
        ";ablation_profitability_observation_id=" + observationId(ablation) +
        ";control_economic_calendar_snapshot_id=" +
        optionalId(control.extended.economicCalendarSnapshotId) +
        ";ablation_economic_calendar_snapshot_id=" +
        optionalId(ablation.extended.economicCalendarSnapshotId) +
        ";economic_calendar_snapshot_hash=" +
        control.extended.economicCalendarSnapshotHash.value_or("NULL") +
        ";control_model_input_width=" +
        optionalInt(control.extended.configuredModelInputWidth) +
        ";control_model_input_layout=" +
        optionalInt(control.extended.configuredModelInputLayoutVersion) +
        ";ablation_model_input_width=" +
        optionalInt(ablation.extended.configuredModelInputWidth) +
        ";ablation_model_input_layout=" +
        optionalInt(ablation.extended.configuredModelInputLayoutVersion) +
        ";evidence_classification=" +
        EvidenceClassificationText(result.evidenceClassification) +
        ";disposition=" + DispositionText(result.disposition) + ";";
}

std::string EvaluationIdentityHash(
    const FeatureAblationPairEvaluation::ArmEvidence& control,
    const FeatureAblationPairEvaluation::ArmEvidence& ablation,
    const ComparisonResult& result)
{
    return TrainingObjective::DeterministicHash(
        EvaluationIdentityCanonical(control, ablation, result));
}

std::pair<long long, long long> ParseExperimentIdPair(std::string_view text)
{
    const std::size_t separator = text.find(':');
    if (separator == std::string_view::npos || separator == 0 ||
        separator + 1 >= text.size() ||
        text.find(':', separator + 1) != std::string_view::npos)
        throw std::invalid_argument(
            "--compare-feature-ablation-pair requires CONTROL_ID:ABLATION_ID");
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

std::string EvidenceClassificationText(EvidenceClassification value)
{
    switch (value)
    {
        case EvidenceClassification::GenericFeatureAblationEvidence:
            return "generic_feature_ablation_evidence";
        case EvidenceClassification::PreFixCausalSurpriseEvidence:
            return "pre_fix_causal_surprise_evidence";
        case EvidenceClassification::CorrectedCausalSurprisePairEvidence:
            return "corrected_causal_surprise_pair_evidence";
        case EvidenceClassification::IncompatibleOrInvalidEvidence:
            return "incompatible_or_invalid_evidence";
    }
    throw std::invalid_argument(
        "unknown_feature_ablation_evidence_classification");
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
