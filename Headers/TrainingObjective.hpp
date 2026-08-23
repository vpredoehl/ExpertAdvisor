#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace EA::TrainingObjective
{

inline constexpr int kConfigurationSchemaVersion = 1;
inline constexpr int kLegacyObjectiveVersion = 1;
inline constexpr int kLegacyLossDefinitionVersion = 1;
inline constexpr std::string_view kLegacyObjectiveIdentifier =
    "legacy_first_hit_weighted_ce_v1";
inline constexpr double kLegacyClassWeightDown = 1.0;
inline constexpr double kLegacyClassWeightNeutral = 1.0;
inline constexpr double kLegacyClassWeightUp = 1.0;
inline constexpr double kLegacySoftmaxLossProbabilityFloor = 1.0e-12;
inline constexpr double kLegacyClassificationLogitGradientScale = 0.1;
inline constexpr double kLegacySharedCoreClassificationGradientScale = 4.0;
inline constexpr double kLegacyGradientClipThreshold = 10.0;

enum class Mode
{
    LegacyFirstHitClassification
};

enum class AuxiliaryLossMode
{
    Disabled
};

enum class OptimizerFamily
{
    Sgd
};

enum class GradientClippingMode
{
    ComponentwiseAfterNormalization
};

struct Configuration
{
    int schemaVersion = kConfigurationSchemaVersion;
    int objectiveVersion = kLegacyObjectiveVersion;
    int lossDefinitionVersion = kLegacyLossDefinitionVersion;
    Mode mode = Mode::LegacyFirstHitClassification;

    std::string objectiveIdentifier =
        std::string(kLegacyObjectiveIdentifier);
    std::string objectiveFamily =
        "up_neutral_down_first_hit_classification";
    std::string classificationLoss =
        "true_class_weighted_softmax_cross_entropy_v1";
    std::string classificationTarget =
        "up_neutral_down_return_high_low_first_hit_strict_threshold_up_tie_v1";
    std::string classIndexOrder = "down_0_neutral_1_up_2";
    std::string classWeightSemantics =
        "true_class_weight_multiplies_loss_and_all_logit_components_v1";
    std::array<double, 3> classWeights {
        kLegacyClassWeightDown,
        kLegacyClassWeightNeutral,
        kLegacyClassWeightUp};
    double softmaxLossProbabilityFloor =
        kLegacySoftmaxLossProbabilityFloor;
    double classificationLogitGradientScale =
        kLegacyClassificationLogitGradientScale;
    double sharedCoreClassificationGradientScale =
        kLegacySharedCoreClassificationGradientScale;

    std::string internalLossNormalization =
        "weighted_loss_sum_divided_by_true_class_weight_sum_v1";
    std::string calculateBatchReturnNormalization =
        "weighted_loss_sum_divided_by_example_count_v1";
    std::string gradientNormalization =
        "all_calculate_batch_gradients_divided_by_true_class_weight_sum_v1";
    std::string batchWindowBoundary =
        "overlapping_windows_do_not_cross_outer_tensor_batch_v1";

    OptimizerFamily optimizer = OptimizerFamily::Sgd;
    std::string optimizerUpdate = "parameter_minus_learning_rate_times_gradient_v1";
    std::string learningRateContract =
        "base_rate_and_parameter_group_multipliers_persisted_in_training_config_v1";
    GradientClippingMode gradientClipping =
        GradientClippingMode::ComponentwiseAfterNormalization;
    double gradientClipThreshold = kLegacyGradientClipThreshold;
    std::string nonfiniteGradientPolicy = "skip_parameter_update_v1";
    std::string weightDecay = "none";
    std::string gradientAccumulationPrecision =
        "core_gradient_accumulation_double_head_gradient_accumulation_float_loss_accumulation_double_v1";

    AuxiliaryLossMode auxiliaryLossMode = AuxiliaryLossMode::Disabled;
    double auxiliaryLossCoefficient = 0.0;
    std::optional<std::string> regressionTargetDefinition;
    std::optional<std::string> regressionNormalizationIdentity;
    std::optional<std::string> robustLossDefinition;
    std::optional<double> robustLossDelta;
    std::string targetClippingDefinition = "none";
    std::string sharedGradientCombination = "classification_only_v1";

    bool operator==(const Configuration&) const = default;
};

inline const Configuration& Legacy()
{
    static const Configuration value;
    return value;
}

inline std::string_view ModeText(Mode mode)
{
    switch (mode)
    {
        case Mode::LegacyFirstHitClassification:
            return "legacy_first_hit_classification";
    }
    throw std::invalid_argument("training_objective_unknown_mode");
}

inline std::string_view AuxiliaryLossModeText(AuxiliaryLossMode mode)
{
    switch (mode)
    {
        case AuxiliaryLossMode::Disabled: return "disabled";
    }
    throw std::invalid_argument("training_objective_unknown_auxiliary_mode");
}

inline std::string_view OptimizerFamilyText(OptimizerFamily optimizer)
{
    switch (optimizer)
    {
        case OptimizerFamily::Sgd: return "sgd";
    }
    throw std::invalid_argument("training_objective_unknown_optimizer");
}

inline std::string_view GradientClippingModeText(GradientClippingMode mode)
{
    switch (mode)
    {
        case GradientClippingMode::ComponentwiseAfterNormalization:
            return "componentwise_after_normalization_before_update";
    }
    throw std::invalid_argument("training_objective_unknown_gradient_clipping_mode");
}

inline std::string CanonicalDouble(double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("training_objective_nonfinite_number");
    if (value == 0.0) value = 0.0;
    std::array<char, 128> buffer{};
    const auto result = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::general);
    if (result.ec != std::errc{})
        throw std::runtime_error("training_objective_number_format_failed");
    return std::string(buffer.data(), result.ptr);
}

inline std::optional<std::string> Validate(const Configuration& value)
{
    if (value.schemaVersion != kConfigurationSchemaVersion)
        return "unsupported_configuration_schema_version";
    if (value.objectiveVersion <= 0 || value.lossDefinitionVersion <= 0)
        return "version_must_be_positive";
    if (value.objectiveIdentifier.empty() || value.objectiveFamily.empty() ||
        value.classificationLoss.empty() ||
        value.classificationTarget.empty() || value.classIndexOrder.empty() ||
        value.classWeightSemantics.empty())
        return "classification_identity_required";
    for (const double weight : value.classWeights)
        if (!std::isfinite(weight) || weight <= 0.0)
            return "class_weights_must_be_finite_and_positive";
    if (!std::isfinite(value.softmaxLossProbabilityFloor) ||
        value.softmaxLossProbabilityFloor <= 0.0 ||
        value.softmaxLossProbabilityFloor >= 1.0)
        return "softmax_probability_floor_out_of_range";
    if (!std::isfinite(value.classificationLogitGradientScale) ||
        value.classificationLogitGradientScale <= 0.0 ||
        !std::isfinite(value.sharedCoreClassificationGradientScale) ||
        value.sharedCoreClassificationGradientScale <= 0.0)
        return "gradient_scales_must_be_finite_and_positive";
    if (!std::isfinite(value.gradientClipThreshold) ||
        value.gradientClipThreshold <= 0.0)
        return "gradient_clip_threshold_must_be_finite_and_positive";
    if (!std::isfinite(value.auxiliaryLossCoefficient) ||
        value.auxiliaryLossCoefficient != 0.0 ||
        value.auxiliaryLossMode != AuxiliaryLossMode::Disabled ||
        value.regressionTargetDefinition.has_value() ||
        value.regressionNormalizationIdentity.has_value() ||
        value.robustLossDefinition.has_value() ||
        value.robustLossDelta.has_value())
        return "phase4a_auxiliary_objective_must_be_disabled";
    if (value.targetClippingDefinition != "none" ||
        value.sharedGradientCombination != "classification_only_v1")
        return "phase4a_classification_only_contract_required";
    return std::nullopt;
}

inline void Append(std::string& output,
                   std::string_view name,
                   std::string_view value)
{
    output.append(name);
    output.push_back('=');
    output.append(value);
    output.push_back(';');
}

inline std::string OptionalText(const std::optional<std::string>& value)
{
    return value.value_or("NULL");
}

inline std::string CanonicalText(const Configuration& value)
{
    if (const auto error = Validate(value))
        throw std::invalid_argument("invalid_training_objective:" + *error);

    std::string output = "training_objective_configuration_v1;";
    Append(output, "schema_version", std::to_string(value.schemaVersion));
    Append(output, "objective_id", value.objectiveIdentifier);
    Append(output, "objective_family", value.objectiveFamily);
    Append(output, "objective_version", std::to_string(value.objectiveVersion));
    Append(output, "loss_definition_version",
           std::to_string(value.lossDefinitionVersion));
    Append(output, "mode", ModeText(value.mode));
    Append(output, "classification_loss", value.classificationLoss);
    Append(output, "classification_target", value.classificationTarget);
    Append(output, "class_index_order", value.classIndexOrder);
    Append(output, "class_weight_semantics", value.classWeightSemantics);
    Append(output, "class_weight_down", CanonicalDouble(value.classWeights[0]));
    Append(output, "class_weight_neutral", CanonicalDouble(value.classWeights[1]));
    Append(output, "class_weight_up", CanonicalDouble(value.classWeights[2]));
    Append(output, "softmax_loss_probability_floor",
           CanonicalDouble(value.softmaxLossProbabilityFloor));
    Append(output, "classification_logit_gradient_scale",
           CanonicalDouble(value.classificationLogitGradientScale));
    Append(output, "shared_core_classification_gradient_scale",
           CanonicalDouble(value.sharedCoreClassificationGradientScale));
    Append(output, "internal_loss_normalization", value.internalLossNormalization);
    Append(output, "calculate_batch_return_normalization",
           value.calculateBatchReturnNormalization);
    Append(output, "gradient_normalization", value.gradientNormalization);
    Append(output, "batch_window_boundary", value.batchWindowBoundary);
    Append(output, "optimizer_family", OptimizerFamilyText(value.optimizer));
    Append(output, "optimizer_update", value.optimizerUpdate);
    Append(output, "learning_rate_contract", value.learningRateContract);
    Append(output, "gradient_clipping_mode",
           GradientClippingModeText(value.gradientClipping));
    Append(output, "gradient_clip_threshold",
           CanonicalDouble(value.gradientClipThreshold));
    Append(output, "nonfinite_gradient_policy", value.nonfiniteGradientPolicy);
    Append(output, "weight_decay", value.weightDecay);
    Append(output, "gradient_accumulation_precision",
           value.gradientAccumulationPrecision);
    Append(output, "auxiliary_loss_mode",
           AuxiliaryLossModeText(value.auxiliaryLossMode));
    Append(output, "auxiliary_loss_coefficient",
           CanonicalDouble(value.auxiliaryLossCoefficient));
    Append(output, "regression_target_definition",
           OptionalText(value.regressionTargetDefinition));
    Append(output, "regression_normalization_identity",
           OptionalText(value.regressionNormalizationIdentity));
    Append(output, "robust_loss_definition",
           OptionalText(value.robustLossDefinition));
    Append(output, "robust_loss_delta",
           value.robustLossDelta ? CanonicalDouble(*value.robustLossDelta) : "NULL");
    Append(output, "target_clipping_definition", value.targetClippingDefinition);
    Append(output, "shared_gradient_combination", value.sharedGradientCombination);
    return output;
}

inline std::string Hex64(std::uint64_t value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::array<char, 16> encoded{};
    for (std::size_t index = 0; index < encoded.size(); ++index)
    {
        const unsigned shift = static_cast<unsigned>(
            (encoded.size() - 1 - index) * 4);
        encoded[index] = digits[(value >> shift) & 0x0fU];
    }
    return std::string(encoded.begin(), encoded.end());
}

inline std::string DeterministicHash(std::string_view canonicalText)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : canonicalText)
    {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return "fnv1a64:" + Hex64(hash);
}

inline std::string Identity(const Configuration& value)
{
    return DeterministicHash(CanonicalText(value));
}

inline Configuration ParseSupportedCanonicalText(const std::string& canonical)
{
    const Configuration legacy = Legacy();
    if (canonical == CanonicalText(legacy)) return legacy;
    throw std::invalid_argument(
        "unsupported_training_objective_canonical_configuration");
}

// Marker-less historical experiments/models predate configurable objectives;
// repository evidence therefore resolves them to the one objective that could
// have produced them. A partial marker fails closed.
inline Configuration ResolvePersisted(
    const std::optional<std::string>& canonical,
    const std::optional<std::string>& hash)
{
    if (!canonical.has_value() && !hash.has_value()) return Legacy();
    if (!canonical.has_value() || !hash.has_value())
        throw std::invalid_argument(
            "incomplete_training_objective_provenance");
    if (DeterministicHash(*canonical) != *hash)
        throw std::invalid_argument(
            "training_objective_provenance_hash_mismatch");
    return ParseSupportedCanonicalText(*canonical);
}

inline bool ResumeCompatible(const Configuration& persisted,
                             const Configuration& requested)
{
    return CanonicalText(persisted) == CanonicalText(requested) &&
           Identity(persisted) == Identity(requested);
}

inline void RequireResumeCompatible(const Configuration& persisted,
                                    const Configuration& requested)
{
    if (!ResumeCompatible(persisted, requested))
        throw std::invalid_argument("training_objective_resume_incompatible");
}

struct LegacyClassificationBatchSummary
{
    double weightedLossSum = 0.0;
    double trueClassWeightSum = 0.0;
    std::size_t exampleCount = 0;

    double InternallyNormalizedLoss() const
    {
        return trueClassWeightSum > 0.0
            ? weightedLossSum / trueClassWeightSum
            : 0.0;
    }

    double CalculateBatchReturnValue() const
    {
        return weightedLossSum /
            static_cast<double>(exampleCount > 0 ? exampleCount : 1);
    }
};

inline LegacyClassificationBatchSummary SummarizeLegacyClassificationBatch(
    std::span<const std::array<double, 3>> probabilities,
    std::span<const int> trueClasses,
    const Configuration& configuration = Legacy())
{
    if (probabilities.size() != trueClasses.size())
        throw std::invalid_argument("classification_batch_size_mismatch");
    LegacyClassificationBatchSummary result;
    for (std::size_t index = 0; index < probabilities.size(); ++index)
    {
        const int trueClass = trueClasses[index];
        if (trueClass < 0 || trueClass >= 3)
            throw std::invalid_argument("classification_class_out_of_range");
        const double probability = probabilities[index][trueClass];
        if (!std::isfinite(probability) || probability < 0.0 ||
            probability > 1.0)
            throw std::invalid_argument("classification_probability_out_of_range");
        const double weight =
            configuration.classWeights[static_cast<std::size_t>(trueClass)];
        result.weightedLossSum += weight * -std::log(std::max(
            configuration.softmaxLossProbabilityFloor, probability));
        result.trueClassWeightSum += weight;
        ++result.exampleCount;
    }
    return result;
}

inline std::array<double, 3> LegacyClassificationLogitGradient(
    const std::array<double, 3>& probabilities,
    int trueClass,
    const Configuration& configuration = Legacy())
{
    if (trueClass < 0 || trueClass >= 3)
        throw std::invalid_argument("classification_class_out_of_range");
    const double weight =
        configuration.classWeights[static_cast<std::size_t>(trueClass)];
    std::array<double, 3> result{};
    for (std::size_t classIndex = 0; classIndex < result.size(); ++classIndex)
    {
        result[classIndex] =
            configuration.classificationLogitGradientScale * weight *
            (probabilities[classIndex] -
             (classIndex == static_cast<std::size_t>(trueClass) ? 1.0 : 0.0));
    }
    return result;
}

} // namespace EA::TrainingObjective
