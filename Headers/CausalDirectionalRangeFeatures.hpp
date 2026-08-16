#ifndef CausalDirectionalRangeFeatures_hpp
#define CausalDirectionalRangeFeatures_hpp

#include <cmath>

// Fixed causal candle-structure feature.  The value for a Tensor row is
// derived solely from the one completed bar retained before that row.
class CausalDirectionalRangeFeatures
{
public:
    float PriorDirectionalBodyRange() const
    {
        if (!hasPriorBar || !std::isfinite(priorOpen) ||
            !std::isfinite(priorHigh) || !std::isfinite(priorLow) ||
            !std::isfinite(priorClose))
            return 0.0f;

        const float range = priorHigh - priorLow;
        if (!std::isfinite(range) || range <= 0.0f) return 0.0f;

        const float result = (priorClose - priorOpen) / range;
        return std::isfinite(result) ? result : 0.0f;
    }

    void RetainCompletedBar(float open, float high, float low, float close)
    {
        hasPriorBar = true;
        priorOpen = open;
        priorHigh = high;
        priorLow = low;
        priorClose = close;
    }

private:
    bool hasPriorBar = false;
    float priorOpen = 0.0f;
    float priorHigh = 0.0f;
    float priorLow = 0.0f;
    float priorClose = 0.0f;
};

#endif /* CausalDirectionalRangeFeatures_hpp */
