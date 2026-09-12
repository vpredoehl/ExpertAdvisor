#ifndef ModelInputFeatureSemantics_hpp
#define ModelInputFeatureSemantics_hpp

#include "FeatureLayout.hpp"
#include "ModelInputContract.hpp"
#include "ModelInputExpansion.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace EA
{

struct ModelInputFeatureSemantic
{
    std::string name;
    std::string source;
    std::string causalTiming;
    std::size_t modelInputColumn = 0;
    bool categorical = false;

    bool operator==(const ModelInputFeatureSemantic&) const = default;
};

// These names describe the original 32-column Tensor prefix in the exact
// assignment order in Tensor::Add. They complete the existing authoritative
// append-only semantic registry rather than guessing columns at the Phase 19C
// extraction site.
inline constexpr std::array<std::string_view, legacy_feature_size>
    kLegacyTensorFeatureNames{{
        "log_open_from_previous_close_scaled",
        "log_close_from_previous_close_scaled",
        "log_high_from_previous_close_scaled",
        "log_low_from_previous_close_scaled",
        "candle_body_scaled",
        "candle_range_scaled",
        "rolling_log_return_volatility_scaled",
        "rolling_cumulative_log_return_scaled",
        "legacy_candle_cycle_sin",
        "legacy_candle_cycle_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "upper_wick_fraction",
        "lower_wick_fraction",
        "close_minus_ema8_by_guarded_range",
        "close_minus_ema21_by_guarded_range",
        "close_minus_ema50_by_guarded_range",
        "ema8_minus_ema21_by_guarded_range",
        "ema21_minus_ema50_by_guarded_range",
        "close_minus_ema8_by_atr14",
        "close_minus_ema21_by_atr14",
        "close_minus_ema50_by_atr14",
        "ema8_minus_ema21_by_atr14",
        "ema21_minus_ema50_by_atr14",
        "ema8_log_slope_by_guarded_range",
        "ema21_log_slope_by_guarded_range",
        "ema50_log_slope_by_guarded_range",
        "ema8_slope_by_atr14",
        "ema21_slope_by_atr14",
        "ema50_slope_by_atr14",
        "candle_body_strength",
        "candle_range_expansion",
    }};

inline constexpr std::array<std::string_view, kModelReturnFeatureCount>
    kModelReturnFeatureNames{{
        "lookback_log_return_1_scaled",
        "lookback_log_return_4_scaled",
        "lookback_log_return_8_scaled",
        "lookback_log_return_16_scaled",
    }};

inline bool IsCategoricalModelInputFeature(std::string_view name)
{
    return name.ends_with("_event") ||
        name.ends_with("_has_consensus") ||
        name.ends_with("_consensus_is_range") ||
        name.ends_with("_has_surprise") ||
        name.ends_with("_surprise_available");
}

inline std::vector<ModelInputFeatureSemantic> ModelInputFeatureSemantics(
    std::size_t modelInputWidth)
{
    const ModelInputContract contract =
        ContractForModelInputWidth(modelInputWidth);
    std::vector<ModelInputFeatureSemantic> result;
    result.reserve(modelInputWidth);
    for (std::size_t column = 0; column < contract.tensorFeatureCount; ++column)
    {
        std::string name;
        if (column < kLegacyTensorFeatureNames.size())
            name = kLegacyTensorFeatureNames[column];
        else
        {
            const std::size_t appended = column - legacy_feature_size;
            if (appended >= kAppendedTensorFeatureSemantics.size() ||
                kAppendedTensorFeatureSemantics[appended].column != column)
                throw std::runtime_error(
                    "model_input_feature_semantic_layout_ambiguous");
            name = kAppendedTensorFeatureSemantics[appended].name;
        }
        const bool categorical = IsCategoricalModelInputFeature(name);
        result.push_back({
            std::move(name),
            "authoritative_model_input_tensor_decision_row",
            "completed_decision_row_at_or_before_entry",
            column,
            categorical});
    }
    for (std::size_t index = 0; index < kModelReturnFeatureCount; ++index)
    {
        result.push_back({
            std::string{kModelReturnFeatureNames[index]},
            "authoritative_model_input_return_suffix",
            "computed_from_completed_closes_ending_at_decision_row",
            contract.tensorFeatureCount + index,
            false});
    }
    if (result.size() != modelInputWidth)
        throw std::runtime_error("model_input_feature_semantic_count_mismatch");
    return result;
}

} // namespace EA

#endif /* ModelInputFeatureSemantics_hpp */
