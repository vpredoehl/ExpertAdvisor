#ifndef CausalReturnDirectionImbalanceFeatures_hpp
#define CausalReturnDirectionImbalanceFeatures_hpp

#include <cmath>
#include <cstddef>
#include <deque>

// Fixed, causal direction balance over eight retained completed raw closes.
// The current Tensor close is retained only after its row value is fixed.
class CausalReturnDirectionImbalanceFeatures
{
public:
    static constexpr std::size_t kCompletedCloseWindow = 8;

    float PriorReturnDirectionImbalance() const
    {
        if (completedCloses.size() != kCompletedCloseWindow) return 0.0f;
        for (const float close : completedCloses)
            if (!std::isfinite(close)) return 0.0f;

        int directionSum = 0;
        for (std::size_t index = 1; index < completedCloses.size(); ++index)
        {
            const float increment = completedCloses[index] - completedCloses[index - 1];
            if (!std::isfinite(increment)) return 0.0f;
            if (increment > 0.0f) ++directionSum;
            else if (increment < 0.0f) --directionSum;
        }

        const float imbalance = static_cast<float>(directionSum) / 7.0f;
        return std::isfinite(imbalance) ? imbalance : 0.0f;
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

#endif /* CausalReturnDirectionImbalanceFeatures_hpp */
