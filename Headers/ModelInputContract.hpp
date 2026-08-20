#ifndef ModelInputContract_hpp
#define ModelInputContract_hpp

#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#include "FeatureLayout.hpp"
#include "FeatureAblation.hpp"

namespace EA
{

inline constexpr std::size_t kModelReturnFeatureCount = 4;
inline constexpr std::size_t kLegacyModelInputWidth =
    legacy_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kDonchianModelInputWidth =
    donchian_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kSessionPhaseModelInputWidth =
    session_phase_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kRelativeTickVolumeModelInputWidth =
    relative_tick_volume_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kCausalReturnSurpriseModelInputWidth =
    causal_return_surprise_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kCausalVolatilityRegimeModelInputWidth =
    causal_volatility_regime_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kCausalDirectionalRangeModelInputWidth =
    causal_directional_range_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kCausalCloseLocationModelInputWidth =
    causal_close_location_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kCausalDirectionalPersistenceModelInputWidth =
    causalDirectionalPersistenceCol + 1 + kModelReturnFeatureCount;
inline constexpr std::size_t kCausalReturnSignPersistenceModelInputWidth =
    causalReturnSignPersistenceCol + 1 + kModelReturnFeatureCount;
inline constexpr std::size_t kCausalReturnDirectionImbalanceModelInputWidth =
    causalReturnDirectionImbalanceCol + 1 + kModelReturnFeatureCount;
inline constexpr std::size_t kCausalDirectionalAdverseExcursionModelInputWidth =
    causalMultiBarRangePressureCol + kModelReturnFeatureCount;
inline constexpr std::size_t kCausalMultiBarRangePressureModelInputWidth =
    causal_multi_bar_range_pressure_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kCausalRollingRangeExpansionModelInputWidth =
    causal_rolling_range_expansion_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kHistoricalLevelProximityModelInputWidth =
    historical_level_proximity_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kReturnAutocorrelationModelInputWidth =
    return_autocorrelation_feature_size + kModelReturnFeatureCount;
inline constexpr std::size_t kCurrentModelInputWidth =
    kReturnAutocorrelationModelInputWidth;

// Every persisted width whose Tensor portion has a registered, stable
// semantic prefix.  Append-only feature additions must retain these entries
// and append their new width.
inline constexpr std::array<std::size_t, 16> kRegisteredModelInputWidths{{
    kLegacyModelInputWidth,
    kDonchianModelInputWidth,
    kSessionPhaseModelInputWidth,
    kRelativeTickVolumeModelInputWidth,
    kCausalReturnSurpriseModelInputWidth,
    kCausalVolatilityRegimeModelInputWidth,
    kCausalDirectionalRangeModelInputWidth,
    kCausalCloseLocationModelInputWidth,
    kCausalDirectionalPersistenceModelInputWidth,
    kCausalReturnSignPersistenceModelInputWidth,
    kCausalReturnDirectionImbalanceModelInputWidth,
    kCausalDirectionalAdverseExcursionModelInputWidth,
    kCausalMultiBarRangePressureModelInputWidth,
    kCausalRollingRangeExpansionModelInputWidth,
    kHistoricalLevelProximityModelInputWidth,
    kReturnAutocorrelationModelInputWidth,
}};

struct ModelInputContract
{
    std::size_t modelInputWidth;
    std::size_t tensorFeatureCount;
    std::size_t physicalTensorFeatureCount;
};

inline ModelInputContract ContractForModelInputWidth(std::size_t modelInputWidth)
{
    switch (modelInputWidth)
    {
        case kLegacyModelInputWidth:
            return {modelInputWidth, legacy_feature_size, 0};
        case kDonchianModelInputWidth:
            return {modelInputWidth, donchian_feature_size, 0};
        case kSessionPhaseModelInputWidth:
            return {modelInputWidth, session_phase_feature_size, 0};
        case kRelativeTickVolumeModelInputWidth:
            return {modelInputWidth, relative_tick_volume_feature_size, 0};
        case kCausalReturnSurpriseModelInputWidth:
            return {modelInputWidth, causal_return_surprise_feature_size, 0};
        case kCausalVolatilityRegimeModelInputWidth:
            return {modelInputWidth, causal_volatility_regime_feature_size, 0};
        case kCausalDirectionalRangeModelInputWidth:
            return {modelInputWidth, causal_directional_range_feature_size, 0};
        case kCausalCloseLocationModelInputWidth:
            return {modelInputWidth, causal_close_location_feature_size, 0};
        case kCausalDirectionalPersistenceModelInputWidth:
            return {modelInputWidth, causalReturnSignPersistenceCol, 0};
        case kCausalReturnSignPersistenceModelInputWidth:
            return {modelInputWidth, causalReturnDirectionImbalanceCol, 0};
        case kCausalReturnDirectionImbalanceModelInputWidth:
            return {modelInputWidth, causalDirectionalAdverseExcursionCol, 0};
        case kCausalDirectionalAdverseExcursionModelInputWidth:
            return {modelInputWidth, causalMultiBarRangePressureCol, 0};
        case kCausalMultiBarRangePressureModelInputWidth:
            return {modelInputWidth, causal_multi_bar_range_pressure_feature_size, 0};
        case kCausalRollingRangeExpansionModelInputWidth:
            return {modelInputWidth,
                    causal_rolling_range_expansion_feature_size, 0};
        case kHistoricalLevelProximityModelInputWidth:
            return {modelInputWidth,
                    historical_level_proximity_feature_size, 0};
        case kCurrentModelInputWidth:
            return {modelInputWidth, feature_size, 0};
        default:
            throw std::runtime_error(
                "MODEL_INPUT_WIDTH_UNSUPPORTED,model_n_in=" +
                std::to_string(modelInputWidth) +
                ",supported=" + std::to_string(kLegacyModelInputWidth) +
                ":" + std::to_string(kDonchianModelInputWidth) +
                ":" + std::to_string(kSessionPhaseModelInputWidth) +
                ":" + std::to_string(kRelativeTickVolumeModelInputWidth) +
                ":" + std::to_string(kCausalReturnSurpriseModelInputWidth) +
                ":" + std::to_string(kCausalVolatilityRegimeModelInputWidth) +
                ":" + std::to_string(kCausalDirectionalRangeModelInputWidth) +
                ":" + std::to_string(kCausalCloseLocationModelInputWidth) +
                ":" + std::to_string(kCausalDirectionalPersistenceModelInputWidth) +
                ":" + std::to_string(kCausalReturnSignPersistenceModelInputWidth) +
                ":" + std::to_string(kCausalReturnDirectionImbalanceModelInputWidth) +
                ":" + std::to_string(kCausalDirectionalAdverseExcursionModelInputWidth) +
                ":" + std::to_string(kCausalMultiBarRangePressureModelInputWidth) +
                ":" + std::to_string(kCausalRollingRangeExpansionModelInputWidth) +
                ":" + std::to_string(kHistoricalLevelProximityModelInputWidth) +
                ":" + std::to_string(kReturnAutocorrelationModelInputWidth));
    }
}

inline ModelInputContract ResolveModelInputContract(
    std::size_t modelInputWidth,
    std::size_t physicalTensorFeatureCount)
{
    ModelInputContract contract = ContractForModelInputWidth(modelInputWidth);
    if (physicalTensorFeatureCount < contract.tensorFeatureCount)
    {
        throw std::runtime_error(
            "MODEL_INPUT_TENSOR_TOO_NARROW,model_n_in=" +
            std::to_string(modelInputWidth) +
            ",required_tensor_features=" +
            std::to_string(contract.tensorFeatureCount) +
            ",runtime_tensor_features=" +
            std::to_string(physicalTensorFeatureCount));
    }
    contract.physicalTensorFeatureCount = physicalTensorFeatureCount;
    return contract;
}

// Copy exactly the tensor prefix required by the persisted model contract.
// In particular, this is a structural projection: it does not copy, then
// zero, columns that are absent from a legacy model's learned input matrix.
template <typename T>
inline void CopyTensorFeaturesForModelInput(
    T* destination,
    const T* source,
    const ModelInputContract& contract)
{
    std::memcpy(destination,
                source,
                contract.tensorFeatureCount * sizeof(T));
}

// This is the sole Tensor-derived model-input materialization hook.  The mask
// is applied after historical-prefix projection, so width and Tensor state are
// unchanged and unavailable historical features fail closed.
template <typename T>
inline void CopyTensorFeaturesForModelInput(
    T* destination,
    const T* source,
    const ModelInputContract& contract,
    const FeatureAblationMask& ablationMask)
{
    CopyTensorFeaturesForModelInput(destination, source, contract);
    ablationMask.ApplyToProjectedTensorFeatures(destination,
                                                 contract.tensorFeatureCount);
}

} // namespace EA

#endif /* ModelInputContract_hpp */
