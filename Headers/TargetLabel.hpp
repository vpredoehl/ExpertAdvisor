#ifndef TargetLabel_hpp
#define TargetLabel_hpp

#include <cmath>
#include <cstddef>

#include "Tensor.hpp"

struct LookaheadClassInfo
{
    PriceTP dt {};
    PriceTP targetDt {};
    float closeT = 0.0f;
    float targetClose = 0.0f;
    float deltaClose = 0.0f;
    float terminalLogReturn = 0.0f;
    bool upHit = false;
    bool downHit = false;
    size_t upOffset = 0;
    size_t downOffset = 0;
    size_t selectedOffset = prediction_horizon;
    float selectedFutureHigh = 0.0f;
    float selectedFutureLow = 0.0f;
    int terminalCloseClass = 1;
    int assignedClass = 1;
};

inline LookaheadClassInfo BuildLookaheadClassInfo(const Tensor& tensor,
                                                  DataSet::const_iterator startIt)
{
    const auto lastIt = startIt + window_size - 1;
    const auto targetIt = lastIt + prediction_horizon;

    LookaheadClassInfo info;
    info.dt = tensor.RawTimeAtIterator(lastIt);
    info.targetDt = tensor.RawTimeAtIterator(targetIt);
    info.closeT = tensor.RawCloseAtIterator(lastIt);
    info.targetClose = tensor.RawCloseAtIterator(targetIt);
    info.deltaClose = info.targetClose - info.closeT;
    info.terminalLogReturn =
        (std::isfinite(info.closeT) && std::isfinite(info.targetClose) &&
         info.closeT > 0.0f && info.targetClose > 0.0f)
            ? std::log(info.targetClose / info.closeT)
            : 0.0f;
    info.terminalCloseClass =
        (info.terminalLogReturn > c_next_threshold) ? 2 :
        ((info.terminalLogReturn < -c_next_threshold) ? 0 : 1);

    for (size_t lookahead = 1; lookahead <= prediction_horizon; ++lookahead)
    {
        const auto futureIt = lastIt + static_cast<std::ptrdiff_t>(lookahead);
        const float futureHigh = tensor.RawHighAtIterator(futureIt);
        const float futureLow = tensor.RawLowAtIterator(futureIt);

        if (std::isfinite(info.closeT) && info.closeT > 0.0f)
        {
            const float upMove = std::log(futureHigh / info.closeT);
            const float downMove = std::log(futureLow / info.closeT);

            if (!info.upHit && std::isfinite(upMove) && upMove > c_next_threshold)
            {
                info.upHit = true;
                info.upOffset = lookahead;
            }

            if (!info.downHit && std::isfinite(downMove) && downMove < -c_next_threshold)
            {
                info.downHit = true;
                info.downOffset = lookahead;
            }
        }
    }

    if (info.upHit && info.downHit) info.assignedClass = (info.upOffset <= info.downOffset) ? 2 : 0;
    else if (info.upHit) info.assignedClass = 2;
    else if (info.downHit) info.assignedClass = 0;
    else info.assignedClass = 1;

    if (info.assignedClass == 2 && info.upHit) info.selectedOffset = info.upOffset;
    else if (info.assignedClass == 0 && info.downHit) info.selectedOffset = info.downOffset;
    else info.selectedOffset = prediction_horizon;

    const auto selectedIt = lastIt + static_cast<std::ptrdiff_t>(info.selectedOffset);
    info.selectedFutureHigh = tensor.RawHighAtIterator(selectedIt);
    info.selectedFutureLow = tensor.RawLowAtIterator(selectedIt);

    LSTM_ASSERT(info.assignedClass >= 0 && info.assignedClass < static_cast<int>(direction_output_size),
                "BuildLookaheadClassInfo: assigned class out of [0,2]");
    return info;
}

#endif /* TargetLabel_hpp */
