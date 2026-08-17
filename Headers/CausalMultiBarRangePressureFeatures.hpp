#ifndef CausalMultiBarRangePressureFeatures_hpp
#define CausalMultiBarRangePressureFeatures_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>

// Fixed, causal close position in the high/low range of the most recent 16
// completed bars, including the current bar. Startup uses the available causal
// prefix; no future or synthetic bars are introduced.
class CausalMultiBarRangePressure
{
public:
    static constexpr std::size_t kBarWindow = 16;

    float AddCompletedBar(float high, float low, float close)
    {
        completedBars.push_back({high, low, close});
        if (completedBars.size() > kBarWindow)
            completedBars.pop_front();

        if (completedBars.empty()) return 0.0f;

        float rollingHigh = 0.0f;
        float rollingLow = 0.0f;
        bool haveRange = false;
        for (const Bar& bar : completedBars)
        {
            if (!std::isfinite(bar.high) || !std::isfinite(bar.low) ||
                !std::isfinite(bar.close))
                return 0.0f;
            rollingHigh = haveRange ? std::max(rollingHigh, bar.high) : bar.high;
            rollingLow = haveRange ? std::min(rollingLow, bar.low) : bar.low;
            haveRange = true;
        }

        const double highValue = static_cast<double>(rollingHigh);
        const double lowValue = static_cast<double>(rollingLow);
        const double range = highValue - lowValue;
        const double scale = std::max({1.0, std::fabs(highValue), std::fabs(lowValue)});
        const double effectivelyZeroRange =
            static_cast<double>(std::numeric_limits<float>::epsilon()) * scale;
        if (!std::isfinite(range) || range <= effectivelyZeroRange)
            return 0.0f;

        const double result =
            2.0 * (static_cast<double>(close) - lowValue) / range - 1.0;
        if (!std::isfinite(result) ||
            result > static_cast<double>(std::numeric_limits<float>::max()) ||
            result < -static_cast<double>(std::numeric_limits<float>::max()))
            return 0.0f;

        const float value = static_cast<float>(result);
        return std::isfinite(value) ? value : 0.0f;
    }

    std::size_t RetainedBarCount() const { return completedBars.size(); }

private:
    struct Bar
    {
        float high;
        float low;
        float close;
    };

    std::deque<Bar> completedBars;
};

#endif /* CausalMultiBarRangePressureFeatures_hpp */
