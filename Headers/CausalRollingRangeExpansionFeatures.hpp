#ifndef CausalRollingRangeExpansionFeatures_hpp
#define CausalRollingRangeExpansionFeatures_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>

// Fixed, causal comparison of the current eight-bar high/low envelope with
// the current thirty-two-bar envelope. Both windows include the completed bar
// passed to AddCompletedBar; startup uses the available completed prefix.
class CausalRollingRangeExpansion
{
public:
    static constexpr std::size_t kShortLookback = 8;
    static constexpr std::size_t kLongLookback = 32;

    float AddCompletedBar(float high, float low)
    {
        completedBars.push_back({high, low});
        if (completedBars.size() > kLongLookback)
            completedBars.pop_front();

        return CurrentValue();
    }

    std::size_t RetainedBarCount() const { return completedBars.size(); }

private:
    struct Bar
    {
        float high;
        float low;
    };

    static bool AccumulateRange(const std::deque<Bar>& bars,
                                std::size_t start,
                                double& high,
                                double& low)
    {
        bool haveRange = false;
        for (std::size_t index = start; index < bars.size(); ++index)
        {
            const Bar& bar = bars[index];
            if (!std::isfinite(bar.high) || !std::isfinite(bar.low))
                return false;
            const double barHigh = static_cast<double>(bar.high);
            const double barLow = static_cast<double>(bar.low);
            high = haveRange ? std::max(high, barHigh) : barHigh;
            low = haveRange ? std::min(low, barLow) : barLow;
            haveRange = true;
        }
        return haveRange;
    }

    float CurrentValue() const
    {
        if (completedBars.empty()) return 0.0f;

        double longHigh = 0.0;
        double longLow = 0.0;
        if (!AccumulateRange(completedBars, 0, longHigh, longLow))
            return 0.0f;

        const std::size_t shortCount = std::min(kShortLookback, completedBars.size());
        double shortHigh = 0.0;
        double shortLow = 0.0;
        if (!AccumulateRange(completedBars, completedBars.size() - shortCount,
                             shortHigh, shortLow))
            return 0.0f;

        const double longRange = longHigh - longLow;
        const double shortRange = shortHigh - shortLow;
        const double scale = std::max({1.0, std::fabs(longHigh), std::fabs(longLow)});
        const double effectivelyZeroRange =
            static_cast<double>(std::numeric_limits<float>::epsilon()) * scale;
        if (!std::isfinite(longRange) || !std::isfinite(shortRange) ||
            longRange <= effectivelyZeroRange || shortRange < 0.0)
            return 0.0f;

        double ratio = shortRange / longRange;
        if (!std::isfinite(ratio)) return 0.0f;

        // A contained window is mathematically bounded by [0, 1]. Permit
        // only double-rounding-sized excursions at that boundary; anything
        // larger is an invalid range state, not a value to conceal by clipping.
        const double roundingTolerance = 8.0 * std::numeric_limits<double>::epsilon();
        if (ratio < 0.0)
        {
            if (ratio < -roundingTolerance) return 0.0f;
            ratio = 0.0;
        }
        else if (ratio > 1.0)
        {
            if (ratio > 1.0 + roundingTolerance) return 0.0f;
            ratio = 1.0;
        }

        const double result = 2.0 * ratio - 1.0;
        if (!std::isfinite(result) ||
            result > static_cast<double>(std::numeric_limits<float>::max()) ||
            result < -static_cast<double>(std::numeric_limits<float>::max()))
            return 0.0f;
        const float value = static_cast<float>(result);
        return std::isfinite(value) ? value : 0.0f;
    }

    std::deque<Bar> completedBars;
};

#endif /* CausalRollingRangeExpansionFeatures_hpp */
