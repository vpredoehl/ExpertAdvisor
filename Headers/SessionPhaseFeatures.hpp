#ifndef SessionPhaseFeatures_hpp
#define SessionPhaseFeatures_hpp

#include <chrono>
#include <cmath>
#include <numbers>
#include <utility>

#include "PricePoint.hpp"

// The input is the canonical UTC system-clock instant produced by the price
// pipeline. Epoch arithmetic deliberately avoids host-local calendar and TZ
// conversion APIs, so the phase has no process-local or DST dependency.
inline std::pair<float, float> ComputeUtcSessionPhase(PriceTP timestamp)
{
    constexpr long long kSecondsPerUtcDay = 24 * 60 * 60;
    const long long epochSeconds = timestamp.time_since_epoch().count();
    const long long secondsOfDay =
        ((epochSeconds % kSecondsPerUtcDay) + kSecondsPerUtcDay) %
        kSecondsPerUtcDay;
    const double phase = 2.0 * std::numbers::pi *
        static_cast<double>(secondsOfDay) /
        static_cast<double>(kSecondsPerUtcDay);
    return {static_cast<float>(std::sin(phase)),
            static_cast<float>(std::cos(phase))};
}

#endif /* SessionPhaseFeatures_hpp */
