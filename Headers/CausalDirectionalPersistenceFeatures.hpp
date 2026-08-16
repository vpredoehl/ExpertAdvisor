#ifndef CausalDirectionalPersistenceFeatures_hpp
#define CausalDirectionalPersistenceFeatures_hpp

#include <cmath>
#include <cstddef>
#include <deque>

// Fixed, causal, signed efficiency over eight retained completed raw closes.
// The current Tensor bar is deliberately retained only after its row value is
// read, so PriorDirectionalEfficiency never includes the current close.
class CausalDirectionalPersistenceFeatures
{
public:
    static constexpr std::size_t kCompletedCloseWindow = 8;

    float PriorDirectionalEfficiency() const
    {
        if (completedCloses.size() != kCompletedCloseWindow) return 0.0f;

        for (const float close : completedCloses)
            if (!std::isfinite(close)) return 0.0f;

        const float net = completedCloses.back() - completedCloses.front();
        if (!std::isfinite(net)) return 0.0f;

        float path = 0.0f;
        for (std::size_t index = 1; index < completedCloses.size(); ++index)
        {
            const float increment = completedCloses[index] - completedCloses[index - 1];
            if (!std::isfinite(increment)) return 0.0f;
            path += std::fabs(increment);
        }
        if (!std::isfinite(path) || path <= 0.0f) return 0.0f;

        const float efficiency = net / path;
        return std::isfinite(efficiency) ? efficiency : 0.0f;
    }

    void RetainCompletedClose(float close)
    {
        completedCloses.push_back(close);
        if (completedCloses.size() > kCompletedCloseWindow)
            completedCloses.pop_front();
    }

private:
    std::deque<float> completedCloses;
};

#endif /* CausalDirectionalPersistenceFeatures_hpp */
