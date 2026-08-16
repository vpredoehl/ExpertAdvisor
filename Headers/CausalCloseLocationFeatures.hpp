#ifndef CausalCloseLocationFeatures_hpp
#define CausalCloseLocationFeatures_hpp

#include <cmath>

// Fixed causal candle-position feature. The value for a Tensor row is derived
// solely from the one completed bar retained before that row.
class CausalCloseLocationFeatures
{
public:
    float PriorCloseLocation() const
    {
        if (!hasPriorBar || !std::isfinite(priorHigh) ||
            !std::isfinite(priorLow) || !std::isfinite(priorClose))
            return 0.0f;

        const float range = priorHigh - priorLow;
        if (!std::isfinite(range) || range <= 0.0f) return 0.0f;

        const float result = 2.0f * (priorClose - priorLow) / range - 1.0f;
        return std::isfinite(result) ? result : 0.0f;
    }

    void RetainCompletedBar(float high, float low, float close)
    {
        hasPriorBar = true;
        priorHigh = high;
        priorLow = low;
        priorClose = close;
    }

private:
    bool hasPriorBar = false;
    float priorHigh = 0.0f;
    float priorLow = 0.0f;
    float priorClose = 0.0f;
};

#endif /* CausalCloseLocationFeatures_hpp */
