#ifndef DonchianFeatures_hpp
#define DonchianFeatures_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "FeatureLayout.hpp"

// The supplied vectors contain completed prior bars only. Partial startup
// history is used as-is; no future or synthetic bars are introduced.
inline std::pair<float, float> ComputeCausalDonchian(
    const std::vector<float>& priorHighs,
    const std::vector<float>& priorLows,
    float currentClose,
    float featureScale,
    std::size_t lookback = kDefaultDonchianLookback)
{
    ValidateDonchianLookback(lookback);
    if (!std::isfinite(currentClose) || currentClose <= 0.0f)
        return {0.0f, 0.0f};

    const std::size_t available = std::min(priorHighs.size(), priorLows.size());
    const std::size_t start = available > lookback
        ? available - lookback
        : 0;

    float priorMaximumHigh = 0.0f;
    float priorMinimumLow = 0.0f;
    bool haveHigh = false;
    bool haveLow = false;
    for (std::size_t i = start; i < available; ++i)
    {
        if (std::isfinite(priorHighs[i]) && priorHighs[i] > 0.0f)
        {
            priorMaximumHigh = haveHigh ? std::max(priorMaximumHigh, priorHighs[i])
                                         : priorHighs[i];
            haveHigh = true;
        }
        if (std::isfinite(priorLows[i]) && priorLows[i] > 0.0f)
        {
            priorMinimumLow = haveLow ? std::min(priorMinimumLow, priorLows[i])
                                      : priorLows[i];
            haveLow = true;
        }
    }

    const float upper = haveHigh ? std::log(currentClose / priorMaximumHigh) * featureScale : 0.0f;
    const float lower = haveLow ? std::log(currentClose / priorMinimumLow) * featureScale : 0.0f;
    return {std::isfinite(upper) ? upper : 0.0f,
            std::isfinite(lower) ? lower : 0.0f};
}

// Closed Donchian-20 compatibility entry point.  New callers use the
// explicit lookback overload above; this preserves established test and API
// behavior for 20 exactly.
inline std::pair<float, float> ComputeCausalDonchian20(
    const std::vector<float>& priorHighs,
    const std::vector<float>& priorLows,
    float currentClose,
    float featureScale)
{
    return ComputeCausalDonchian(priorHighs, priorLows, currentClose,
                                 featureScale, kDefaultDonchianLookback);
}

#endif /* DonchianFeatures_hpp */
