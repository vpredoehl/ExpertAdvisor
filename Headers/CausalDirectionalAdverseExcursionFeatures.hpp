#ifndef CausalDirectionalAdverseExcursionFeatures_hpp
#define CausalDirectionalAdverseExcursionFeatures_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>

// Fixed, causal adverse excursion over eight retained completed raw closes.
// The current Tensor close is retained only after its row value is fixed.
class CausalDirectionalAdverseExcursionFeatures
{
public:
    static constexpr std::size_t kCompletedCloseWindow = 8;

    float PriorDirectionalAdverseExcursion() const
    {
        if (completedCloses.size() != kCompletedCloseWindow) return 0.0f;
        for (const float close : completedCloses)
            if (!std::isfinite(close)) return 0.0f;

        const float start = completedCloses.front();
        const float net = completedCloses.back() - start;
        if (!std::isfinite(net) || net == 0.0f) return 0.0f;

        float path = 0.0f;
        float minimum = start;
        float maximum = start;
        for (std::size_t index = 1; index < completedCloses.size(); ++index)
        {
            const float close = completedCloses[index];
            const float increment = close - completedCloses[index - 1];
            if (!std::isfinite(increment)) return 0.0f;
            path += std::fabs(increment);
            minimum = std::min(minimum, close);
            maximum = std::max(maximum, close);
        }
        if (!std::isfinite(path) || path <= 0.0f) return 0.0f;

        const float adverse = net > 0.0f
            ? std::max(0.0f, start - minimum)
            : std::max(0.0f, maximum - start);
        if (!std::isfinite(adverse)) return 0.0f;
        const float result = adverse / path;
        return std::isfinite(result) ? result : 0.0f;
    }

    void RetainCompletedClose(float currentClose)
    {
        completedCloses.push_back(currentClose);
        if (completedCloses.size() > kCompletedCloseWindow)
            completedCloses.pop_front();
    }

private:
    std::deque<float> completedCloses;
};

#endif /* CausalDirectionalAdverseExcursionFeatures_hpp */
