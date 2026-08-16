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
inline constexpr std::size_t feature_size = causal_return_surprise_feature_size + 1;

#endif /* FeatureLayout_hpp */
