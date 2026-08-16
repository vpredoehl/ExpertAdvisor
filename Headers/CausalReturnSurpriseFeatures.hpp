#ifndef CausalReturnSurpriseFeatures_hpp
#define CausalReturnSurpriseFeatures_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>

// Fixed by the persisted feature definition; this is not an experiment
// parameter. The current return is deliberately excluded from the reference.
inline constexpr std::size_t causalReturnSurpriseLookback = 32;

class CausalReturnSurprise32
{
public:
    float AddCompletedClose(float close)
    {
        const bool currentReturnValid =
            hasPreviousClose && std::isfinite(previousClose) &&
            previousClose > 0.0f && std::isfinite(close) && close > 0.0f;

        double currentReturn = 0.0;
        if (currentReturnValid)
            currentReturn = std::log(static_cast<double>(close) /
                                     static_cast<double>(previousClose));

        float result = 0.0f;
        if (currentReturnValid && std::isfinite(currentReturn) &&
            !priorReturns.empty())
        {
            const double meanSquare = priorReturnSquareSum /
                static_cast<double>(priorReturns.size());
            const double rms = std::sqrt(meanSquare);
            if (std::isfinite(rms) && rms > 0.0)
            {
                const double surprise = currentReturn / rms;
                if (std::isfinite(surprise))
                    result = static_cast<float>(std::clamp(surprise, -10.0, 10.0));
            }
        }

        // Retain a close even when it is invalid so a later close cannot form
        // a synthetic return across an invalid bar. Valid returns alone enter
        // the causal reference state after this row's value is finalized.
        hasPreviousClose = true;
        previousClose = close;
        if (currentReturnValid && std::isfinite(currentReturn))
        {
            if (priorReturns.size() == causalReturnSurpriseLookback)
            {
                const double oldest = priorReturns.front();
                priorReturnSquareSum -= oldest * oldest;
                priorReturns.pop_front();
            }
            priorReturns.push_back(currentReturn);
            priorReturnSquareSum += currentReturn * currentReturn;
        }

        return std::isfinite(result) ? result : 0.0f;
    }

    std::size_t PriorReturnCount() const { return priorReturns.size(); }

private:
    bool hasPreviousClose = false;
    float previousClose = 0.0f;
    std::deque<double> priorReturns;
    double priorReturnSquareSum = 0.0;
};

#endif /* CausalReturnSurpriseFeatures_hpp */
