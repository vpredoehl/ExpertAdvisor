#ifndef FeatureLayout_hpp
#define FeatureLayout_hpp

#include <cstddef>

#include "DonchianLookback.hpp"

inline constexpr std::size_t legacy_feature_size = 32;
// Closed Donchian-20 compatibility constant; runtime lookback is persisted
// separately and supplied to Tensor.
inline constexpr std::size_t donchian_lookback = kDefaultDonchianLookback;
inline constexpr std::size_t donchianUpCol = legacy_feature_size;
inline constexpr std::size_t donchianDownCol = legacy_feature_size + 1;
inline constexpr std::size_t feature_size = legacy_feature_size + 2;

#endif /* FeatureLayout_hpp */
