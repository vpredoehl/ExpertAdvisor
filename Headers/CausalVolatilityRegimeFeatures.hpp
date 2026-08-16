#ifndef CausalVolatilityRegimeFeatures_hpp
#define CausalVolatilityRegimeFeatures_hpp

#include <cmath>
#include <cstddef>
#include <deque>

// Fixed by the persisted feature definition; neither lookback is an
// experiment parameter. The current bar's return is deliberately excluded
// from the value emitted for that bar.
inline constexpr std::size_t causalVolatilityRegimeShortLookback = 8;
inline constexpr std::size_t causalVolatilityRegimeLongLookback = 32;

class CausalVolatilityRegime8x32
{
public:
    float AddCompletedClose(float close)
    {
        float result = 0.0f;
        if (!priorReturns.empty())
        {
            const std::size_t longCount = priorReturns.size();
            const std::size_t shortCount =
                longCount < causalVolatilityRegimeShortLookback
                    ? longCount : causalVolatilityRegimeShortLookback;
            const double longRms = std::sqrt(
                priorReturnSquareSum / static_cast<double>(longCount));
            const double shortRms = std::sqrt(
                shortReturnSquareSum / static_cast<double>(shortCount));
            if (std::isfinite(longRms) && longRms > 0.0 &&
                std::isfinite(shortRms))
            {
                const double regime = shortRms / longRms - 1.0;
                if (std::isfinite(regime)) result = static_cast<float>(regime);
            }
        }

        const bool currentReturnValid =
            hasPreviousClose && std::isfinite(previousClose) &&
            previousClose > 0.0f && std::isfinite(close) && close > 0.0f;
        if (currentReturnValid)
        {
            const double currentReturn = std::log(
                static_cast<double>(close) / static_cast<double>(previousClose));
            if (std::isfinite(currentReturn)) RetainReturn(currentReturn);
        }

        // Invalid closes break return continuity so later bars cannot form a
        // synthetic return across an invalid bar.
        hasPreviousClose = true;
        previousClose = close;
        return std::isfinite(result) ? result : 0.0f;
    }

    std::size_t PriorReturnCount() const { return priorReturns.size(); }

private:
    void RetainReturn(double value)
    {
        if (priorReturns.size() == causalVolatilityRegimeLongLookback)
        {
            const double oldest = priorReturns.front();
            priorReturnSquareSum -= oldest * oldest;
            if (priorReturns.size() <= causalVolatilityRegimeShortLookback)
                shortReturnSquareSum -= oldest * oldest;
            priorReturns.pop_front();
        }
        priorReturns.push_back(value);
        priorReturnSquareSum += value * value;
        shortReturnSquareSum += value * value;
        if (priorReturns.size() > causalVolatilityRegimeShortLookback)
        {
            const double firstOutsideShortWindow =
                priorReturns[priorReturns.size() - causalVolatilityRegimeShortLookback - 1];
            shortReturnSquareSum -= firstOutsideShortWindow * firstOutsideShortWindow;
        }
    }

    bool hasPreviousClose = false;
    float previousClose = 0.0f;
    std::deque<double> priorReturns;
    double priorReturnSquareSum = 0.0;
    double shortReturnSquareSum = 0.0;
};

#endif /* CausalVolatilityRegimeFeatures_hpp */
