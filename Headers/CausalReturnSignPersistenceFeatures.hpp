#ifndef CausalReturnSignPersistenceFeatures_hpp
#define CausalReturnSignPersistenceFeatures_hpp

#include <cmath>
#include <cstddef>
#include <deque>

// Fixed, causal return-sign persistence over eight retained completed raw
// closes. The current Tensor bar is retained only after its row value is read.
class CausalReturnSignPersistenceFeatures
{
public:
    static constexpr std::size_t kCompletedCloseWindow = 8;

    float PriorReturnSignPersistence() const
    {
        if (completedCloses.size() != kCompletedCloseWindow) return 0.0f;
        for (const float close : completedCloses)
            if (!std::isfinite(close)) return 0.0f;

        float previousIncrement = completedCloses[1] - completedCloses[0];
        if (!std::isfinite(previousIncrement)) return 0.0f;

        int scoreSum = 0;
        for (std::size_t index = 2; index < completedCloses.size(); ++index)
        {
            const float increment = completedCloses[index] - completedCloses[index - 1];
            if (!std::isfinite(increment)) return 0.0f;
            if (previousIncrement != 0.0f && increment != 0.0f)
                scoreSum += (std::signbit(previousIncrement) == std::signbit(increment))
                    ? 1 : -1;
            previousIncrement = increment;
        }

        const float persistence = static_cast<float>(scoreSum) / 6.0f;
        return std::isfinite(persistence) ? persistence : 0.0f;
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

#endif /* CausalReturnSignPersistenceFeatures_hpp */
