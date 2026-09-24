#include "ExperimentPairComparison.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace EA::ExperimentPairComparison
{
namespace
{

using FeatureArm = FeatureAblationPairEvaluation::ArmEvidence;
using Shared = PairedTrainingObjectiveEvaluation::ArmEvidence;

std::string Number(double value)
{
    char buffer[128] {};
    const auto converted = std::to_chars(
        std::begin(buffer), std::end(buffer), value,
        std::chars_format::general);
    if (converted.ec != std::errc{})
        throw std::runtime_error("experiment_pair_number_format_failed");
    return std::string(buffer, converted.ptr);
}

template <typename Value>
std::string OptionalInteger(const std::optional<Value>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalNumber(const std::optional<double>& value)
{
    return value ? Number(*value) : "NULL";
}

std::string OptionalText(const std::optional<std::string>& value)
{
    return value.value_or("NULL");
}

std::string Boolean(bool value)
{
    return value ? "true" : "false";
}

std::optional<std::string> ExecutionIdentity(
    const std::optional<PairedTrainingObjectiveEvaluation::
        ScientificExecutionProvenance>& value)
{
    if (!value) return std::nullopt;
    return value->phase + '|' +
        std::to_string(value->semanticLayoutVersion) + '|' +
        std::to_string(value->modelInputWidth) + '|' +
        value->executableSha256 + '|' + value->runtimeIdentity;
}

std::string InitializationLineageIdentity(const FeatureArm& evidence)
{
    const auto& configuration = evidence.authoritative.configuration;
    if (!configuration.resumeModelId) return "fresh_initialization";
    const auto& provenance = evidence.resumeCheckpointProvenance;
    if (provenance && provenance->resumeModelId == *configuration.resumeModelId &&
        provenance->modelExperimentId == configuration.experimentId &&
        provenance->ownExperimentCheckpoint && provenance->checkpointEpoch &&
        *provenance->checkpointEpoch > 0)
        return "own_checkpoint_epoch:" +
            std::to_string(*provenance->checkpointEpoch);
    return "resume_model_id:" + std::to_string(*configuration.resumeModelId);
}

void AddIdentity(std::vector<IdentityField>& fields,
                 std::string name,
                 std::string value)
{
    fields.push_back({std::move(name), std::move(value)});
}

void AddIdentity(std::vector<IdentityField>& fields,
                 std::string name,
                 std::optional<std::string> value)
{
    fields.push_back({std::move(name), std::move(value)});
}

void AddModelIdentity(std::vector<IdentityField>& fields,
                      bool finalModelAvailable,
                      std::string name,
                      std::string value)
{
    AddIdentity(fields, std::move(name),
                finalModelAvailable
                    ? std::optional<std::string>{std::move(value)}
                    : std::nullopt);
}

std::optional<double> Count(const std::optional<std::uint64_t>& value)
{
    return value
        ? std::optional<double>{static_cast<double>(*value)}
        : std::nullopt;
}

MetricDelta Delta(std::optional<double> armA,
                  std::optional<double> armB,
                  bool computeDelta)
{
    MetricDelta result{armA, armB, std::nullopt};
    if (computeDelta && armA && armB && std::isfinite(*armA) &&
        std::isfinite(*armB))
        result.armBMinusArmA = *armB - *armA;
    return result;
}

std::map<std::string, std::optional<std::string>> IdentityMap(
    const ArmResultSet& arm,
    std::string_view role,
    std::vector<std::string>& invalid)
{
    std::map<std::string, std::optional<std::string>> result;
    for (const auto& field : arm.scientificIdentity)
    {
        if (field.name.empty())
        {
            invalid.push_back(std::string(role) +
                              "_scientific_identity_field_name_empty");
            continue;
        }
        if (!result.emplace(field.name, field.value).second)
            invalid.push_back(std::string(role) +
                              "_scientific_identity_field_duplicated:" +
                              field.name);
    }
    return result;
}

void AddMissingMetricReasons(const MetricObservation& metrics,
                             std::string_view role,
                             std::vector<std::string>& reasons)
{
    const auto add = [&](std::string_view name,
                         const std::optional<double>& value)
    {
        if (!value)
            reasons.push_back(std::string(role) +
                              "_metric_unavailable:" + std::string(name));
    };
    add("inference_accuracy", metrics.inferenceAccuracy);
    add("accept_rate", metrics.acceptRate);
    add("accept_accuracy", metrics.acceptAccuracy);
    add("leader_score", metrics.leaderScore);
    add("prediction_count", metrics.predictionCount);
    add("actionable_count", metrics.actionableCount);
    add("winning_actionable_count", metrics.winningActionableCount);
    add("losing_actionable_count", metrics.losingActionableCount);
    add("actionable_percentage", metrics.actionablePercentage);
    add("win_percentage", metrics.winPercentage);
    add("gross_positive_return", metrics.grossPositiveReturn);
    add("gross_negative_return", metrics.grossNegativeReturn);
    add("aggregate_return", metrics.aggregateReturn);
    add("average_return_per_action", metrics.averageReturnPerAction);
}

void ValidateMetricValues(const MetricObservation& metrics,
                          std::string_view role,
                          std::vector<std::string>& invalid)
{
    const std::optional<double> values[] = {
        metrics.inferenceAccuracy, metrics.acceptRate,
        metrics.acceptAccuracy, metrics.leaderScore,
        metrics.predictionCount, metrics.actionableCount,
        metrics.winningActionableCount, metrics.losingActionableCount,
        metrics.actionablePercentage, metrics.winPercentage,
        metrics.grossPositiveReturn, metrics.grossNegativeReturn,
        metrics.aggregateReturn, metrics.averageReturnPerAction};
    for (const auto& value : values)
        if (value && !std::isfinite(*value))
        {
            invalid.push_back(std::string(role) + "_metric_nonfinite");
            break;
        }

    if (metrics.predictionCount && *metrics.predictionCount < 0.0)
        invalid.push_back(std::string(role) + "_prediction_count_invalid");
    if (metrics.actionableCount &&
        (*metrics.actionableCount < 0.0 ||
         (metrics.predictionCount &&
          *metrics.actionableCount > *metrics.predictionCount)))
        invalid.push_back(std::string(role) + "_actionable_count_invalid");
    if (metrics.winningActionableCount && metrics.actionableCount &&
        (*metrics.winningActionableCount < 0.0 ||
         *metrics.winningActionableCount > *metrics.actionableCount))
        invalid.push_back(std::string(role) +
                          "_winning_actionable_count_invalid");
    if (metrics.losingActionableCount && metrics.actionableCount &&
        (*metrics.losingActionableCount < 0.0 ||
         *metrics.losingActionableCount > *metrics.actionableCount))
        invalid.push_back(std::string(role) +
                          "_losing_actionable_count_invalid");
    if (metrics.winningActionableCount && metrics.losingActionableCount &&
        metrics.actionableCount &&
        *metrics.winningActionableCount + *metrics.losingActionableCount >
            *metrics.actionableCount)
        invalid.push_back(std::string(role) +
                          "_actionable_outcome_counts_invalid");
}

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-' || character == '.';
        if (!safe) character = '_';
    }
    return value.empty() ? "EMPTY" : value;
}

std::string Reasons(const std::vector<std::string>& reasons)
{
    if (reasons.empty()) return "NONE";
    std::ostringstream output;
    for (std::size_t index = 0; index < reasons.size(); ++index)
    {
        if (index != 0) output << '|';
        output << MachineText(reasons[index]);
    }
    return output.str();
}

void PrintMetric(std::ostringstream& output,
                 std::string_view name,
                 const MetricDelta& metric)
{
    output << "EXPERIMENT_PAIR_METRIC"
           << ",metric=" << name
           << ",arm_a=" << OptionalNumber(metric.armA)
           << ",arm_b=" << OptionalNumber(metric.armB)
           << ",arm_b_minus_arm_a=" << OptionalNumber(metric.armBMinusArmA)
           << '\n';
}

void PrintSummaryMetric(std::ostringstream& output,
                        std::string_view name,
                        const MetricDelta& metric)
{
    if (!metric.armA && !metric.armB && !metric.armBMinusArmA) return;
    output << "EXPERIMENT_PAIR_SUMMARY_METRIC"
           << ",metric=" << name
           << ",arm_a=" << OptionalNumber(metric.armA)
           << ",arm_b=" << OptionalNumber(metric.armB)
           << ",arm_b_minus_arm_a=" << OptionalNumber(metric.armBMinusArmA)
           << '\n';
}

std::string ScientificComparability(const ComparisonResult& result)
{
    if (!result.invalidReasons.empty()) return "invalid_evidence";
    if (!result.unexpectedDifferences.empty()) return "incompatible";
    return "comparable";
}

} // namespace

const std::array<MetricDefinition, 14>& MetricDefinitions()
{
    static constexpr std::array<MetricDefinition, 14> definitions {{
        {"inference_accuracy", &ComparisonResult::inferenceAccuracy},
        {"accept_rate", &ComparisonResult::acceptRate},
        {"accept_accuracy", &ComparisonResult::acceptAccuracy},
        {"leader_score", &ComparisonResult::leaderScore},
        {"prediction_count", &ComparisonResult::predictionCount},
        {"actionable_count", &ComparisonResult::actionableCount},
        {"winning_actionable_count",
         &ComparisonResult::winningActionableCount},
        {"losing_actionable_count",
         &ComparisonResult::losingActionableCount},
        {"actionable_percentage", &ComparisonResult::actionablePercentage},
        {"win_percentage", &ComparisonResult::winPercentage},
        {"gross_positive_return", &ComparisonResult::grossPositiveReturn},
        {"gross_negative_return", &ComparisonResult::grossNegativeReturn},
        {"aggregate_return", &ComparisonResult::aggregateReturn},
        {"average_return_per_action",
         &ComparisonResult::averageReturnPerAction},
    }};
    return definitions;
}

ArmResultSet MakeArmResultSet(const FeatureArm& evidence)
{
    ArmResultSet result;
    const Shared& shared = evidence.authoritative;
    const auto& configuration = shared.configuration;
    const auto& extended = evidence.extended;
    result.experimentId = configuration.experimentId;
    result.finalModelAvailable = shared.finalModelId.has_value();
    result.trainingProvenanceAvailable = shared.trainingExecution.has_value();
    result.inferenceProvenanceAvailable = shared.inferenceExecution.has_value();

    auto& fields = result.scientificIdentity;
    AddIdentity(fields, "symbol", configuration.symbol);
    AddIdentity(fields, "prediction_horizon",
                std::to_string(configuration.predictionHorizon));
    AddIdentity(fields, "train_start", configuration.trainStart);
    AddIdentity(fields, "train_end", configuration.trainEnd);
    AddIdentity(fields, "inference_start", configuration.inferenceStart);
    AddIdentity(fields, "inference_end", configuration.inferenceEnd);
    AddIdentity(fields, "target_epochs",
                std::to_string(configuration.targetEpochs));
    AddIdentity(fields, "threshold", Number(configuration.threshold));
    AddIdentity(fields, "core_lr_mult",
                OptionalNumber(configuration.coreLearningRateMultiplier));
    AddIdentity(fields, "head_lr_mult",
                OptionalNumber(configuration.headLearningRateMultiplier));
    AddIdentity(fields, "checkpoint_interval",
                std::to_string(configuration.checkpointInterval));
    AddModelIdentity(fields, result.finalModelAvailable, "model_input_width",
                     std::to_string(configuration.inputWidth));
    AddModelIdentity(fields, result.finalModelAvailable, "hidden_size",
                     std::to_string(configuration.hiddenSize));
    AddModelIdentity(fields, result.finalModelAvailable, "layer_count",
                     std::to_string(configuration.layerCount));
    AddModelIdentity(fields, result.finalModelAvailable, "window_size",
                     std::to_string(configuration.windowSize));
    AddModelIdentity(fields, result.finalModelAvailable,
                     "model_metadata_schema_version",
                     std::to_string(configuration.modelMetadataSchemaVersion));
    AddModelIdentity(
        fields, result.finalModelAvailable,
        "train_configuration_schema_version",
        std::to_string(configuration.trainConfigurationSchemaVersion));
    AddModelIdentity(fields, result.finalModelAvailable,
                     "normalization_version",
                     std::to_string(configuration.normalizationVersion));
    AddModelIdentity(fields, result.finalModelAvailable, "class_weight_down",
                     Number(configuration.classWeightDown));
    AddModelIdentity(fields, result.finalModelAvailable,
                     "class_weight_neutral",
                     Number(configuration.classWeightNeutral));
    AddModelIdentity(fields, result.finalModelAvailable, "class_weight_up",
                     Number(configuration.classWeightUp));
    AddIdentity(fields, "feature_warmup_scope",
                configuration.featureWarmupScope);
    AddIdentity(fields, "donchian_mode", configuration.donchianMode);
    AddIdentity(fields, "donchian_lookback",
                std::to_string(configuration.donchianLookback));
    AddIdentity(fields, "feature_ablation_mask",
                configuration.featureAblationMask);
    AddModelIdentity(
        fields, result.finalModelAvailable,
        "optimizer_metadata_schema_version",
        std::to_string(configuration.optimizerMetadataSchemaVersion));
    AddModelIdentity(fields, result.finalModelAvailable, "optimizer_type",
                     std::to_string(configuration.optimizerType));
    AddModelIdentity(
        fields, result.finalModelAvailable,
        "optimizer_first_moment_buffer_count",
        std::to_string(configuration.optimizerFirstMomentBufferCount));
    AddModelIdentity(
        fields, result.finalModelAvailable,
        "optimizer_second_moment_buffer_count",
        std::to_string(configuration.optimizerSecondMomentBufferCount));
    AddModelIdentity(
        fields, result.finalModelAvailable, "persisted_core_lr_mult",
        Number(configuration.persistedCoreLearningRateMultiplier));
    AddModelIdentity(
        fields, result.finalModelAvailable, "persisted_head_weight_lr_mult",
        Number(configuration.persistedHeadWeightLearningRateMultiplier));
    AddModelIdentity(
        fields, result.finalModelAvailable, "persisted_head_bias_lr_mult",
        Number(configuration.persistedHeadBiasLearningRateMultiplier));
    AddModelIdentity(fields, result.finalModelAvailable, "label_rule_id",
                     std::to_string(configuration.labelRuleId));
    AddModelIdentity(fields, result.finalModelAvailable, "target_type",
                     std::to_string(configuration.targetType));
    AddModelIdentity(fields, result.finalModelAvailable, "target_scale",
                     Number(configuration.targetScale));
    AddModelIdentity(fields, result.finalModelAvailable, "target_bias",
                     Number(configuration.targetBias));
    AddModelIdentity(fields, result.finalModelAvailable, "target_use_zscore",
                     Boolean(configuration.targetUseZScore));
    AddModelIdentity(fields, result.finalModelAvailable, "target_mean",
                     Number(configuration.targetMean));
    AddModelIdentity(fields, result.finalModelAvailable,
                     "target_standard_deviation",
                     Number(configuration.targetStandardDeviation));
    AddModelIdentity(
        fields, result.finalModelAvailable,
        "model_input_metadata_schema_version",
        std::to_string(configuration.modelInputMetadataSchemaVersion));
    AddModelIdentity(
        fields, result.finalModelAvailable,
        "model_input_semantic_layout_version",
        std::to_string(configuration.modelInputLayoutVersion));
    AddModelIdentity(fields, result.finalModelAvailable,
                     "persisted_training_symbol",
                     configuration.persistedTrainingSymbol);
    AddModelIdentity(fields, result.finalModelAvailable,
                     "persisted_training_start",
                     configuration.persistedTrainingStart);
    AddModelIdentity(fields, result.finalModelAvailable,
                     "persisted_training_end",
                     configuration.persistedTrainingEnd);
    AddModelIdentity(fields, result.finalModelAvailable,
                     "input_width_expansion_canonical",
                     OptionalText(configuration.inputWidthExpansionCanonical));
    AddIdentity(fields, "initialization_lineage",
                InitializationLineageIdentity(evidence));
    AddIdentity(fields, "resume_expand_input_width",
                Boolean(configuration.resumeExpandInputWidth));
    AddIdentity(fields, "training_objective_canonical",
                configuration.experimentObjective.canonical);
    AddIdentity(fields, "training_objective_hash",
                configuration.experimentObjective.hash);

    AddIdentity(fields, "configured_model_input_width",
                OptionalInteger(extended.configuredModelInputWidth));
    AddIdentity(fields, "configured_model_input_semantic_layout_version",
                OptionalInteger(extended.configuredModelInputLayoutVersion));
    AddIdentity(fields, "economic_calendar_snapshot_id",
                OptionalInteger(extended.economicCalendarSnapshotId));
    AddIdentity(fields, "economic_calendar_snapshot_hash",
                OptionalText(extended.economicCalendarSnapshotHash));
    AddIdentity(fields, "base_learning_rate", Number(extended.baseLearningRate));
    AddIdentity(fields, "batch_size", std::to_string(extended.batchSize));
    AddIdentity(fields, "fresh_initialization_seed",
                OptionalInteger(extended.freshInitializationSeed));
    AddIdentity(fields, "training_objective_version",
                std::to_string(extended.trainingObjectiveVersion));
    AddIdentity(fields, "loss_definition_version",
                std::to_string(extended.lossDefinitionVersion));
    AddIdentity(fields, "auxiliary_loss_mode", extended.auxiliaryLossMode);
    AddIdentity(fields, "auxiliary_loss_coefficient",
                Number(extended.auxiliaryLossCoefficient));
    AddIdentity(fields, "regression_target_definition",
                OptionalText(extended.regressionTargetDefinition));
    AddIdentity(fields, "regression_normalization_identity",
                OptionalText(extended.regressionNormalizationIdentity));
    AddIdentity(fields, "robust_loss_definition",
                OptionalText(extended.robustLossDefinition));
    AddIdentity(fields, "robust_loss_delta",
                OptionalNumber(extended.robustLossDelta));
    AddIdentity(fields, "target_clipping_definition",
                extended.targetClippingDefinition);
    AddIdentity(fields, "objective_normalization_identity",
                extended.objectiveNormalizationIdentity);
    AddIdentity(fields, "checkpoint_inference_enabled",
                Boolean(extended.checkpointInferenceEnabled));
    AddIdentity(fields, "checkpoint_inference_minimum_epoch",
                OptionalInteger(extended.checkpointInferenceMinimumEpoch));
    AddIdentity(fields, "checkpoint_inference_interval",
                OptionalInteger(extended.checkpointInferenceInterval));
    AddIdentity(fields, "checkpoint_policy_enabled",
                Boolean(extended.checkpointPolicyEnabled));
    AddIdentity(fields, "checkpoint_policy_minimum_leader_score",
                OptionalNumber(extended.checkpointPolicyMinimumLeaderScore));
    AddIdentity(fields, "checkpoint_policy_minimum_inference_accuracy",
                OptionalNumber(extended.checkpointPolicyMinimumInferenceAccuracy));
    AddIdentity(fields, "checkpoint_policy_top_n",
                OptionalInteger(extended.checkpointPolicyTopN));
    AddIdentity(fields, "checkpoint_policy_scope",
                extended.checkpointPolicyScope);
    AddIdentity(fields, "checkpoint_policy_stop_mode",
                extended.checkpointPolicyStopMode);
    AddIdentity(fields, "checkpoint_policy_grace_evaluations",
                std::to_string(extended.checkpointPolicyGraceEvaluations));
    AddIdentity(fields, "checkpoint_policy_revision",
                std::to_string(extended.checkpointPolicyRevision));
    AddIdentity(fields, "checkpoint_policy_hash",
                OptionalText(extended.checkpointPolicyHash));
    AddIdentity(fields, "continuation_policy_enabled",
                Boolean(extended.continuationPolicyEnabled));
    AddIdentity(fields, "continuation_policy_scientific_identity",
                extended.continuationPolicyScientificIdentity);
    AddIdentity(fields, "training_execution_identity",
                ExecutionIdentity(shared.trainingExecution));
    AddIdentity(fields, "inference_execution_identity",
                ExecutionIdentity(shared.inferenceExecution));

    result.finalInferenceAvailable =
        evidence.exactFinalInferenceResultId.has_value();
    result.finalAnalysisAvailable = shared.classification.has_value();
    result.profitabilityObservationAvailable = shared.profitability.has_value();

    if (shared.classification)
    {
        const auto& value = *shared.classification;
        result.metrics.inferenceAccuracy = value.inferenceAccuracy;
        result.metrics.acceptRate = value.acceptRate;
        result.metrics.acceptAccuracy = value.acceptAccuracy;
        result.metrics.leaderScore = value.leaderScore;
    }
    if (shared.profitability)
    {
        const auto& value = *shared.profitability;
        result.metrics.predictionCount =
            static_cast<double>(value.predictionCount);
        result.metrics.actionableCount =
            static_cast<double>(value.actionableCount);
        result.metrics.winningActionableCount =
            Count(value.winningActionableCount);
        result.metrics.losingActionableCount =
            Count(value.losingActionableCount);
        if (value.predictionCount != 0)
            result.metrics.actionablePercentage =
                100.0 * static_cast<double>(value.actionableCount) /
                static_cast<double>(value.predictionCount);
        if (value.actionableCount != 0 && value.winningActionableCount)
            result.metrics.winPercentage =
                100.0 * static_cast<double>(*value.winningActionableCount) /
                static_cast<double>(value.actionableCount);
        result.metrics.grossPositiveReturn =
            value.grossPositiveTerminalHorizonLogReturnSum;
        result.metrics.grossNegativeReturn =
            value.grossNegativeTerminalHorizonLogReturnSum;
        result.metrics.aggregateReturn =
            value.aggregateTerminalHorizonLogReturnSum;
        result.metrics.averageReturnPerAction =
            value.averageTerminalHorizonLogReturnPerActionablePrediction;
    }
    return result;
}

ComparisonResult Compare(const ArmResultSet& armA,
                         const ArmResultSet& armB,
                         const Request& request)
{
    ComparisonResult result;
    result.experimentAId = armA.experimentId;
    result.experimentBId = armB.experimentId;
    result.armALabel = request.armALabel;
    result.armBLabel = request.armBLabel;
    result.armAFinalModelAvailable = armA.finalModelAvailable;
    result.armATrainingProvenanceAvailable =
        armA.trainingProvenanceAvailable;
    result.armAFinalInferenceAvailable = armA.finalInferenceAvailable;
    result.armAInferenceProvenanceAvailable =
        armA.inferenceProvenanceAvailable;
    result.armAFinalAnalysisAvailable = armA.finalAnalysisAvailable;
    result.armAProfitabilityObservationAvailable =
        armA.profitabilityObservationAvailable;
    result.armBFinalModelAvailable = armB.finalModelAvailable;
    result.armBTrainingProvenanceAvailable =
        armB.trainingProvenanceAvailable;
    result.armBFinalInferenceAvailable = armB.finalInferenceAvailable;
    result.armBInferenceProvenanceAvailable =
        armB.inferenceProvenanceAvailable;
    result.armBFinalAnalysisAvailable = armB.finalAnalysisAvailable;
    result.armBProfitabilityObservationAvailable =
        armB.profitabilityObservationAvailable;

    if (armA.experimentId <= 0)
        result.invalidReasons.push_back("arm_a_experiment_id_invalid");
    if (armB.experimentId <= 0)
        result.invalidReasons.push_back("arm_b_experiment_id_invalid");
    if (armA.experimentId == armB.experimentId)
        result.invalidReasons.push_back("experiment_id_reused");
    if (request.armALabel.empty())
        result.invalidReasons.push_back("arm_a_label_empty");
    if (request.armBLabel.empty())
        result.invalidReasons.push_back("arm_b_label_empty");

    const auto identityA = IdentityMap(armA, "arm_a", result.invalidReasons);
    const auto identityB = IdentityMap(armB, "arm_b", result.invalidReasons);
    for (const auto& [name, value] : identityA)
        result.armAScientificIdentity.push_back({name, value});
    for (const auto& [name, value] : identityB)
        result.armBScientificIdentity.push_back({name, value});
    std::set<std::string> intentional;
    for (const std::string& field : request.intentionalDifferenceFields)
    {
        if (!intentional.insert(field).second)
            result.invalidReasons.push_back(
                "intentional_difference_field_duplicated:" + field);
        if (!identityA.contains(field) || !identityB.contains(field))
            result.invalidReasons.push_back(
                "intentional_difference_field_missing:" + field);
        else if (!identityA.at(field) || !identityB.at(field))
            result.invalidReasons.push_back(
                "intentional_difference_field_unavailable:" + field);
        else if (identityA.at(field) == identityB.at(field))
            result.invalidReasons.push_back(
                "intentional_difference_field_equal:" + field);
    }

    std::set<std::string> fieldNames;
    for (const auto& [name, unused] : identityA)
    {
        (void)unused;
        fieldNames.insert(name);
    }
    for (const auto& [name, unused] : identityB)
    {
        (void)unused;
        fieldNames.insert(name);
    }
    for (const std::string& field : fieldNames)
    {
        const auto left = identityA.find(field);
        const auto right = identityB.find(field);
        if (left != identityA.end() && right != identityB.end())
        {
            // An unavailable evidence-derived value is not a scientific
            // observation and therefore cannot establish a mismatch.
            if (!left->second || !right->second) continue;
        }
        const std::string leftValue = left == identityA.end()
            ? "<MISSING>"
            : left->second.value_or("<UNAVAILABLE>");
        const std::string rightValue = right == identityB.end()
            ? "<MISSING>"
            : right->second.value_or("<UNAVAILABLE>");
        if (leftValue == rightValue) continue;
        IdentityDifference difference{
            field, leftValue, rightValue, intentional.contains(field)};
        if (difference.intentional)
            result.intentionalDifferences.push_back(std::move(difference));
        else
            result.unexpectedDifferences.push_back(std::move(difference));
    }

    if (!armA.finalInferenceAvailable)
        result.incompleteReasons.push_back("arm_a_final_inference_unavailable");
    if (!armB.finalInferenceAvailable)
        result.incompleteReasons.push_back("arm_b_final_inference_unavailable");
    if (!armA.finalAnalysisAvailable)
        result.incompleteReasons.push_back("arm_a_final_analysis_unavailable");
    if (!armB.finalAnalysisAvailable)
        result.incompleteReasons.push_back("arm_b_final_analysis_unavailable");
    if (!armA.profitabilityObservationAvailable)
        result.incompleteReasons.push_back(
            "arm_a_profitability_observation_unavailable");
    if (!armB.profitabilityObservationAvailable)
        result.incompleteReasons.push_back(
            "arm_b_profitability_observation_unavailable");
    AddMissingMetricReasons(armA.metrics, "arm_a", result.incompleteReasons);
    AddMissingMetricReasons(armB.metrics, "arm_b", result.incompleteReasons);
    ValidateMetricValues(armA.metrics, "arm_a", result.invalidReasons);
    ValidateMetricValues(armB.metrics, "arm_b", result.invalidReasons);

    const bool identityCompatible = result.invalidReasons.empty() &&
        result.unexpectedDifferences.empty();
    result.inferenceAccuracy = Delta(
        armA.metrics.inferenceAccuracy, armB.metrics.inferenceAccuracy,
        identityCompatible);
    result.acceptRate = Delta(
        armA.metrics.acceptRate, armB.metrics.acceptRate, identityCompatible);
    result.acceptAccuracy = Delta(
        armA.metrics.acceptAccuracy, armB.metrics.acceptAccuracy,
        identityCompatible);
    result.leaderScore = Delta(
        armA.metrics.leaderScore, armB.metrics.leaderScore, identityCompatible);
    result.predictionCount = Delta(
        armA.metrics.predictionCount, armB.metrics.predictionCount,
        identityCompatible);
    result.actionableCount = Delta(
        armA.metrics.actionableCount, armB.metrics.actionableCount,
        identityCompatible);
    result.winningActionableCount = Delta(
        armA.metrics.winningActionableCount,
        armB.metrics.winningActionableCount, identityCompatible);
    result.losingActionableCount = Delta(
        armA.metrics.losingActionableCount,
        armB.metrics.losingActionableCount, identityCompatible);
    result.actionablePercentage = Delta(
        armA.metrics.actionablePercentage, armB.metrics.actionablePercentage,
        identityCompatible);
    result.winPercentage = Delta(
        armA.metrics.winPercentage, armB.metrics.winPercentage,
        identityCompatible);
    result.grossPositiveReturn = Delta(
        armA.metrics.grossPositiveReturn, armB.metrics.grossPositiveReturn,
        identityCompatible);
    result.grossNegativeReturn = Delta(
        armA.metrics.grossNegativeReturn, armB.metrics.grossNegativeReturn,
        identityCompatible);
    result.aggregateReturn = Delta(
        armA.metrics.aggregateReturn, armB.metrics.aggregateReturn,
        identityCompatible);
    result.averageReturnPerAction = Delta(
        armA.metrics.averageReturnPerAction,
        armB.metrics.averageReturnPerAction, identityCompatible);

    if (!result.invalidReasons.empty())
        result.status = Status::InvalidEvidence;
    else if (!result.unexpectedDifferences.empty())
        result.status = Status::IncompatibleScientificIdentity;
    else if (!result.incompleteReasons.empty())
        result.status = Status::ComparableIncomplete;
    else
        result.status = Status::ComparableComplete;
    return result;
}

std::string Render(const ComparisonResult& result)
{
    std::ostringstream output;
    output << "EXPERIMENT_PAIR_COMPARISON"
           << ",version=1"
           << ",experiment_a_id=" << result.experimentAId
           << ",experiment_b_id=" << result.experimentBId
           << ",arm_a_label=" << MachineText(result.armALabel)
           << ",arm_b_label=" << MachineText(result.armBLabel)
           << ",delta_sign_convention=arm_b_minus_arm_a"
           << ",percentage_unit=percent"
           << ",percentage_delta_unit=percentage_points\n";
    const auto printIdentity = [&](std::string_view role,
                                   const std::vector<IdentityField>& fields)
    {
        for (const auto& field : fields)
        {
            output << "EXPERIMENT_PAIR_ARM_IDENTITY"
                   << ",role=" << role
                   << ",field=" << MachineText(field.name)
                   << ",value=";
            if (field.value)
                output << std::quoted(*field.value);
            else
                output << "NULL";
            output << '\n';
        }
    };
    printIdentity("arm_a", result.armAScientificIdentity);
    printIdentity("arm_b", result.armBScientificIdentity);
    for (const auto& difference : result.intentionalDifferences)
        output << "EXPERIMENT_PAIR_IDENTITY_DIFFERENCE"
               << ",field=" << MachineText(difference.field)
               << ",arm_a=" << std::quoted(difference.armA)
               << ",arm_b=" << std::quoted(difference.armB)
               << ",intentional=true\n";
    for (const auto& difference : result.unexpectedDifferences)
        output << "EXPERIMENT_PAIR_IDENTITY_DIFFERENCE"
               << ",field=" << MachineText(difference.field)
               << ",arm_a=" << std::quoted(difference.armA)
               << ",arm_b=" << std::quoted(difference.armB)
               << ",intentional=false\n";
    for (const auto& definition : MetricDefinitions())
        PrintMetric(output, definition.name, result.*definition.member);
    output << "EXPERIMENT_PAIR_RESULT"
           << ",status=" << StatusText(result.status)
           << ",invalid_reasons=" << Reasons(result.invalidReasons)
           << ",incomplete_reasons=" << Reasons(result.incompleteReasons)
           << ",subjective_winner=NONE\n";
    return output.str();
}

std::string RenderSummary(const ComparisonResult& result)
{
    std::ostringstream output;
    output << "EXPERIMENT_PAIR_SUMMARY"
           << ",version=1"
           << ",experiment_a_id=" << result.experimentAId
           << ",experiment_b_id=" << result.experimentBId
           << ",arm_a_label=" << MachineText(result.armALabel)
           << ",arm_b_label=" << MachineText(result.armBLabel)
           << ",arm_order=argument_order"
           << ",delta_sign_convention=arm_b_minus_arm_a"
           << ",status=" << StatusText(result.status)
           << ",scientific_comparability="
           << ScientificComparability(result) << '\n';

    const auto printEvidence = [&output](
        std::string_view role,
        bool finalModel,
        bool trainingProvenance,
        bool finalInference,
        bool inferenceProvenance,
        bool finalAnalysis,
        bool profitability)
    {
        output << "EXPERIMENT_PAIR_SUMMARY_EVIDENCE"
               << ",role=" << role
               << ",final_model_available=" << Boolean(finalModel)
               << ",training_provenance_available="
               << Boolean(trainingProvenance)
               << ",final_inference_available=" << Boolean(finalInference)
               << ",inference_provenance_available="
               << Boolean(inferenceProvenance)
               << ",final_analysis_available=" << Boolean(finalAnalysis)
               << ",profitability_observation_available="
               << Boolean(profitability) << '\n';
    };
    printEvidence(
        "arm_a", result.armAFinalModelAvailable,
        result.armATrainingProvenanceAvailable,
        result.armAFinalInferenceAvailable,
        result.armAInferenceProvenanceAvailable,
        result.armAFinalAnalysisAvailable,
        result.armAProfitabilityObservationAvailable);
    printEvidence(
        "arm_b", result.armBFinalModelAvailable,
        result.armBTrainingProvenanceAvailable,
        result.armBFinalInferenceAvailable,
        result.armBInferenceProvenanceAvailable,
        result.armBFinalAnalysisAvailable,
        result.armBProfitabilityObservationAvailable);

    const auto printDifferences = [&output](
        std::string_view kind,
        const std::vector<IdentityDifference>& differences)
    {
        if (differences.empty())
        {
            output << "EXPERIMENT_PAIR_SUMMARY_IDENTITY_DIFFERENCE"
                   << ",kind=" << kind << ",field=NONE\n";
            return;
        }
        for (const auto& difference : differences)
            output << "EXPERIMENT_PAIR_SUMMARY_IDENTITY_DIFFERENCE"
                   << ",kind=" << kind
                   << ",field=" << MachineText(difference.field)
                   << ",arm_a=" << std::quoted(difference.armA)
                   << ",arm_b=" << std::quoted(difference.armB) << '\n';
    };
    printDifferences("intentional", result.intentionalDifferences);
    printDifferences("unexpected", result.unexpectedDifferences);

    output << "EXPERIMENT_PAIR_SUMMARY_REASONS"
           << ",invalid_reasons=" << Reasons(result.invalidReasons)
           << ",incomplete_reasons=" << Reasons(result.incompleteReasons)
           << '\n';
    for (const auto& definition : MetricDefinitions())
        PrintSummaryMetric(output, definition.name,
                           result.*definition.member);
    return output.str();
}

std::string StatusText(Status status)
{
    switch (status)
    {
        case Status::ComparableComplete: return "comparable_complete";
        case Status::ComparableIncomplete: return "comparable_incomplete";
        case Status::IncompatibleScientificIdentity:
            return "incompatible_scientific_identity";
        case Status::InvalidEvidence: return "invalid_evidence";
    }
    throw std::invalid_argument("unknown_experiment_pair_comparison_status");
}

} // namespace EA::ExperimentPairComparison
