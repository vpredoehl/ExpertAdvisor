#ifndef DonchianFeatures_hpp
#define DonchianFeatures_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "FeatureLayout.hpp"

// Returns the two causal Donchian-20 distances for the current close. The
// input vectors contain prior observations only; their current-bar values are
// intentionally not accepted by this calculation.
inline std::pair<float, float> ComputeCausalDonchian20(
    const std::vector<float>& priorHighs,
    const std::vector<float>& priorLows,
    float currentClose,
    float featureScale)
{
    if (!std::isfinite(currentClose) || currentClose <= 0.0f)
        return {0.0f, 0.0f};

    const size_t available = std::min(priorHighs.size(), priorLows.size());
    const size_t start = available > donchian_lookback
        ? available - donchian_lookback
        : 0;

    float priorMaximumHigh = 0.0f;
    float priorMinimumLow = 0.0f;
    bool haveHigh = false;
    bool haveLow = false;
    for (size_t i = start; i < available; ++i)
    {
        if (std::isfinite(priorHighs[i]) && priorHighs[i] > 0.0f)
        {
            priorMaximumHigh = haveHigh
                ? std::max(priorMaximumHigh, priorHighs[i])
                : priorHighs[i];
            haveHigh = true;
        }
        if (std::isfinite(priorLows[i]) && priorLows[i] > 0.0f)
        {
            priorMinimumLow = haveLow
                ? std::min(priorMinimumLow, priorLows[i])
                : priorLows[i];
            haveLow = true;
        }
    }

    const float upper = haveHigh
        ? std::log(currentClose / priorMaximumHigh) * featureScale
        : 0.0f;
    const float lower = haveLow
        ? std::log(currentClose / priorMinimumLow) * featureScale
        : 0.0f;
    return {
        std::isfinite(upper) ? upper : 0.0f,
        std::isfinite(lower) ? lower : 0.0f
    };
}

#endif /* DonchianFeatures_hpp */
