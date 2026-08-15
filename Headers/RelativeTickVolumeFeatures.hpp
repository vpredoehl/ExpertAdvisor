#ifndef RelativeTickVolumeFeatures_hpp
#define RelativeTickVolumeFeatures_hpp

#include <cmath>
#include <cstddef>
#include <deque>

// This is an observation-count lookback, deliberately independent of elapsed
// wall-clock time. It is a fixed feature definition, not an experiment setting.
inline constexpr std::size_t relativeTickVolumeLookback = 32;

// A completed bar's volume is usable only when it is finite and nonnegative.
// The production cursor rejects invalid source rows; this fallback gives the
// feature deterministic, finite behavior for direct callers and tests too.
inline float CanonicalTickVolume(float volume)
{
    return std::isfinite(volume) && volume >= 0.0f ? volume : 0.0f;
}

// Produces one causal feature value, then retains the completed current bar
// for later observations. The reference mean contains at most the preceding
// 32 completed bars, never the current bar.
class CausalRelativeTickVolume32
{
public:
    float AddCompletedBar(float currentVolume)
    {
        const float canonicalCurrent = CanonicalTickVolume(currentVolume);
        float result = 0.0f;
        if (!priorVolumes.empty())
        {
            const double reference = priorVolumeSum /
                static_cast<double>(priorVolumes.size());
            const double numerator = static_cast<double>(canonicalCurrent) + 1.0;
            const double denominator = reference + 1.0;
            const double relative =
                std::isfinite(reference) && reference >= 0.0 &&
                std::isfinite(numerator) && numerator > 0.0 &&
                std::isfinite(denominator) && denominator > 0.0
                    ? std::log(numerator / denominator)
                    : 0.0;
            if (std::isfinite(relative))
                result = static_cast<float>(relative);
            if (!std::isfinite(result))
                result = 0.0f;
        }

        if (priorVolumes.size() == relativeTickVolumeLookback)
        {
            priorVolumeSum -= static_cast<double>(priorVolumes.front());
            priorVolumes.pop_front();
        }
        priorVolumes.push_back(canonicalCurrent);
        priorVolumeSum += static_cast<double>(canonicalCurrent);
        return result;
    }

    std::size_t PriorBarCount() const { return priorVolumes.size(); }

private:
    std::deque<float> priorVolumes;
    double priorVolumeSum = 0.0;
};

#endif /* RelativeTickVolumeFeatures_hpp */
