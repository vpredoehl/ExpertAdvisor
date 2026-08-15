#ifndef ReturnFeatureHistory_hpp
#define ReturnFeatureHistory_hpp

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>

namespace EA
{

// Appended return features are anchored to an observation's position in the
// complete Tensor. Batch and inference-window offsets are not history
// coordinates.
inline constexpr std::array<std::size_t, 4> kMultiHorizonReturnLookbacks {
    1, 4, 8, 16
};

template <typename RawCloseAtGlobalPosition>
inline float ComputeLookbackLogReturnAtGlobalPosition(
    std::size_t currentGlobalPosition,
    std::size_t lookbackBars,
    RawCloseAtGlobalPosition&& rawCloseAtGlobalPosition)
{
    if (lookbackBars == 0 || currentGlobalPosition < lookbackBars)
        return 0.0f;

    const float curClose = rawCloseAtGlobalPosition(currentGlobalPosition);
    const float prevClose = rawCloseAtGlobalPosition(currentGlobalPosition - lookbackBars);
    if (!std::isfinite(curClose) || !std::isfinite(prevClose) ||
        curClose <= 0.0f || prevClose <= 0.0f)
    {
        return 0.0f;
    }

    return std::log(curClose / prevClose);
}

template <typename RawCloseAtGlobalPosition>
inline std::size_t AppendMultiHorizonReturnFeaturesAtGlobalPosition(
    std::size_t currentGlobalPosition,
    float* destination,
    std::size_t destinationOffset,
    float featureScale,
    RawCloseAtGlobalPosition&& rawCloseAtGlobalPosition)
{
    std::size_t written = 0;
    for (const std::size_t lookbackBars : kMultiHorizonReturnLookbacks)
    {
        destination[destinationOffset + written] =
            ComputeLookbackLogReturnAtGlobalPosition(
                currentGlobalPosition,
                lookbackBars,
                rawCloseAtGlobalPosition) * featureScale;
        ++written;
    }
    return written;
}

// The production per-row LSTM input assembly used by both training and
// inference after each path translates its local row to a Tensor-global index.
template <typename RawCloseAtGlobalPosition>
inline std::size_t AssembleLstmModelInputRowAtGlobalPosition(
    float* destination,
    const float* sourceTensorFeatures,
    std::size_t tensorFeatureCount,
    std::size_t currentGlobalPosition,
    float featureScale,
    RawCloseAtGlobalPosition&& rawCloseAtGlobalPosition)
{
    std::memcpy(destination,
                sourceTensorFeatures,
                tensorFeatureCount * sizeof(float));
    return AppendMultiHorizonReturnFeaturesAtGlobalPosition(
        currentGlobalPosition,
        destination,
        tensorFeatureCount,
        featureScale,
        std::forward<RawCloseAtGlobalPosition>(rawCloseAtGlobalPosition));
}

} // namespace EA

#endif /* ReturnFeatureHistory_hpp */
