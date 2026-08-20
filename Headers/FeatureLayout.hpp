#ifndef FeatureLayout_hpp
#define FeatureLayout_hpp

#include <cstddef>

#include "DonchianLookback.hpp"

// The first 32 Tensor columns are the pre-Donchian, persisted model prefix.
// Subsequent feature increments append columns so historical model prefixes
// retain their exact meaning.
inline constexpr std::size_t legacy_feature_size = 32;
// Closed Donchian-20 compatibility constant; runtime lookback is persisted
// separately and supplied to Tensor.
inline constexpr std::size_t donchian_lookback = kDefaultDonchianLookback;
inline constexpr std::size_t donchianUpCol = legacy_feature_size;
inline constexpr std::size_t donchianDownCol = legacy_feature_size + 1;
inline constexpr std::size_t donchian_feature_size = legacy_feature_size + 2;
inline constexpr std::size_t sessionPhaseSinCol = donchian_feature_size;
inline constexpr std::size_t sessionPhaseCosCol = donchian_feature_size + 1;
inline constexpr std::size_t session_phase_feature_size = donchian_feature_size + 2;
inline constexpr std::size_t relativeTickVolumeCol = session_phase_feature_size;
inline constexpr std::size_t relative_tick_volume_feature_size = session_phase_feature_size + 1;
inline constexpr std::size_t causalReturnSurpriseCol = relative_tick_volume_feature_size;
inline constexpr std::size_t causal_return_surprise_feature_size =
    relative_tick_volume_feature_size + 1;
inline constexpr std::size_t causalVolatilityRegimeCol =
    causal_return_surprise_feature_size;
inline constexpr std::size_t causal_volatility_regime_feature_size =
    causal_return_surprise_feature_size + 1;
inline constexpr std::size_t causalDirectionalRangeCol =
    causal_volatility_regime_feature_size;
inline constexpr std::size_t causal_directional_range_feature_size =
    causal_volatility_regime_feature_size + 1;
inline constexpr std::size_t causalCloseLocationCol =
    causal_directional_range_feature_size;
inline constexpr std::size_t causal_close_location_feature_size =
    causal_directional_range_feature_size + 1;
inline constexpr std::size_t causalDirectionalPersistenceCol =
    causal_close_location_feature_size;
inline constexpr std::size_t causalReturnSignPersistenceCol =
    causalDirectionalPersistenceCol + 1;
inline constexpr std::size_t causalReturnDirectionImbalanceCol =
    causalReturnSignPersistenceCol + 1;
inline constexpr std::size_t causalDirectionalAdverseExcursionCol =
    causalReturnDirectionImbalanceCol + 1;
inline constexpr std::size_t causalMultiBarRangePressureCol =
    causalDirectionalAdverseExcursionCol + 1;
inline constexpr std::size_t causal_multi_bar_range_pressure_feature_size =
    causalMultiBarRangePressureCol + 1;
inline constexpr std::size_t causalRollingRangeExpansionCol =
    causal_multi_bar_range_pressure_feature_size;
inline constexpr std::size_t causal_rolling_range_expansion_feature_size =
    causalRollingRangeExpansionCol + 1;
inline constexpr std::size_t historicalLevelProximityCol =
    causal_rolling_range_expansion_feature_size;
inline constexpr std::size_t historical_level_proximity_feature_size =
    historicalLevelProximityCol + 1;
inline constexpr std::size_t feature_size =
    historical_level_proximity_feature_size;

#endif /* FeatureLayout_hpp */
