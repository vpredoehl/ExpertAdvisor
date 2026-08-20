#ifndef CausalReturnAutocorrelationFeatures_hpp
#define CausalReturnAutocorrelationFeatures_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>

// Fixed by the persisted feature definition. Thirty-two adjacent return pairs
// require thirty-three completed close-to-close log returns. The return formed
// by AddCompletedClose is legal for the current completed Tensor bar and is
// included before that row's correlation is evaluated.
inline constexpr std::size_t causalReturnAutocorrelationPairCount = 32;

class CausalReturnAutocorrelation32
{
public:
    static constexpr double kVarianceEpsilon = 1.0e-24;

    float AddCompletedClose(float close)
    {
        const bool currentReturnValid =
            hasPreviousClose && std::isfinite(previousClose) &&
            previousClose > 0.0f && std::isfinite(close) && close > 0.0f;
        bool retainedCurrentReturn = false;
        if (currentReturnValid)
        {
            const double currentReturn = std::log(
                static_cast<double>(close) /
                static_cast<double>(previousClose));
            if (std::isfinite(currentReturn))
            {
                completedReturns.push_back(currentReturn);
                if (completedReturns.size() >
                    causalReturnAutocorrelationPairCount + 1)
                {
                    completedReturns.pop_front();
                }
                retainedCurrentReturn = true;
            }
            else
            {
                completedReturns.clear();
            }
        }
        else
        {
            // Pearson lag pairs must remain adjacent in bar time. An invalid
            // return therefore invalidates the rolling pair window rather
            // than joining returns from opposite sides of a broken bar.
            completedReturns.clear();
        }

        // Retaining the close, including an invalid close, prevents a later
        // valid close from forming a synthetic return across an invalid bar.
        hasPreviousClose = true;
        previousClose = close;
        return retainedCurrentReturn ? CurrentValue() : 0.0f;
    }

    std::size_t CompletedReturnCount() const
    {
        return completedReturns.size();
    }

private:
    float CurrentValue() const
    {
        constexpr std::size_t requiredReturns =
            causalReturnAutocorrelationPairCount + 1;
        if (completedReturns.size() != requiredReturns) return 0.0f;

        double sumX = 0.0;
        double sumY = 0.0;
        for (std::size_t index = 0;
             index < causalReturnAutocorrelationPairCount; ++index)
        {
            const double x = completedReturns[index];
            const double y = completedReturns[index + 1];
            if (!std::isfinite(x) || !std::isfinite(y)) return 0.0f;
            sumX += x;
            sumY += y;
        }
        const double divisor =
            static_cast<double>(causalReturnAutocorrelationPairCount);
        const double meanX = sumX / divisor;
        const double meanY = sumY / divisor;

        // Two-pass, mean-centered double accumulation avoids the cancellation
        // of E[xy] - E[x]E[y] for small FX log returns.
        double covariance = 0.0;
        double varianceX = 0.0;
        double varianceY = 0.0;
        for (std::size_t index = 0;
             index < causalReturnAutocorrelationPairCount; ++index)
        {
            const double centeredX = completedReturns[index] - meanX;
            const double centeredY = completedReturns[index + 1] - meanY;
            covariance += centeredX * centeredY;
            varianceX += centeredX * centeredX;
            varianceY += centeredY * centeredY;
        }
        if (!std::isfinite(covariance) || !std::isfinite(varianceX) ||
            !std::isfinite(varianceY) ||
            varianceX <= kVarianceEpsilon ||
            varianceY <= kVarianceEpsilon)
        {
            return 0.0f;
        }

        const double denominator = std::sqrt(varianceX * varianceY);
        if (!std::isfinite(denominator) || denominator <= 0.0) return 0.0f;
        const double correlation = covariance / denominator;
        if (!std::isfinite(correlation)) return 0.0f;
        return std::clamp(static_cast<float>(correlation), -1.0f, 1.0f);
    }

    bool hasPreviousClose = false;
    float previousClose = 0.0f;
    std::deque<double> completedReturns;
};

#endif /* CausalReturnAutocorrelationFeatures_hpp */
