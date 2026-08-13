#ifndef FeatureLayout_hpp
#define FeatureLayout_hpp

#include <cstddef>

inline constexpr std::size_t legacy_feature_size = 32;
inline constexpr std::size_t donchian_lookback = 20;
inline constexpr std::size_t donchianUpCol = legacy_feature_size;
inline constexpr std::size_t donchianDownCol = legacy_feature_size + 1;
inline constexpr std::size_t feature_size = legacy_feature_size + 2;

#endif /* FeatureLayout_hpp */
