#include <cassert>

#include "TrainingRuntimeConfig.hpp"

int main()
{
    const EA::Training::RuntimeConfig config{
        false,
        4,
        0.001f,
        64,
        32,
        32,
        1,
        1,
        20,
        0.75f,
        1.25f,
        1.5f,
        Donchian20Mode::Enabled,
        EA::FeatureWarmupScope::FullHistoryWarmup,
        20,
        EA::TrainingObjective::Legacy()};

    assert(!config.inferenceMode);
    assert(config.predictionHorizon == 4);
    assert(config.thresholdLogret == 0.001f);
    assert(config.windowSize == 64);
    assert(config.hiddenSize == config.outputSize);
    assert(config.numLayers == 1);
    assert(config.epochCount == 20);
    assert(config.donchianLookback == 20);
    return 0;
}
