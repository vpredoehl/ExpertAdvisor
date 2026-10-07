#pragma once

#include <cstddef>

#include "Donchian20Mode.hpp"
#include "FeatureWarmupScope.hpp"
#include "TrainingObjective.hpp"

namespace EA::Training
{

// Immutable snapshot of the already-resolved settings used by one managed
// training run.  It intentionally has no defaults or resolution logic: the
// legacy application remains the sole owner of launch/resume resolution.
struct RuntimeConfig
{
    bool inferenceMode;
    std::size_t predictionHorizon;
    float thresholdLogret;
    std::size_t windowSize;
    std::size_t hiddenSize;
    std::size_t outputSize;
    std::size_t numLayers;
    int normalizationVersion;
    int epochCount;
    float coreLrMultiplier;
    float headWeightLrMultiplier;
    float headBiasLrMultiplier;
    Donchian20Mode donchian20Mode;
    FeatureWarmupScope featureWarmupScope;
    std::size_t donchianLookback;
    TrainingObjective::Configuration trainingObjective;
};

} // namespace EA::Training
