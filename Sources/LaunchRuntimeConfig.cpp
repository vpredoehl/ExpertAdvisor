#include "LaunchRuntimeConfig.hpp"

#include "LaunchArguments.hpp"
#include "LstmRuntimeLogging.hpp"
#include "BuildConfig.hpp"
#include "Params.hpp"

#include <stdexcept>

namespace EA::LaunchRuntimeConfig
{
void Apply(const LaunchArgs& launchArgs, bool& runtimeInferenceMode)
{
    runtimeInferenceMode = launchArgs.inferenceMode.value_or(default_runtime_inference_mode);
    if (launchArgs.predictionHorizon) prediction_horizon = *launchArgs.predictionHorizon;
    if (launchArgs.thresholdLogret) c_next_threshold = static_cast<float>(*launchArgs.thresholdLogret);
    if (launchArgs.logLevel) SetRuntimeLogLevel(*launchArgs.logLevel);
    if (launchArgs.windowSize) window_size = *launchArgs.windowSize;
    if (launchArgs.hiddenSize) { hidden_size = *launchArgs.hiddenSize; n_out = hidden_size; }
    if (launchArgs.numLayers)
    {
        if (*launchArgs.numLayers != 1)
            throw std::invalid_argument("--num-layers currently supports only 1; increasing layers would change the LSTM architecture");
        num_layers = *launchArgs.numLayers;
    }
    if (launchArgs.epochs) epoch_count = *launchArgs.epochs;
    if (launchArgs.coreLrMult) core_lr_mult = static_cast<float>(*launchArgs.coreLrMult);
    if (launchArgs.headWeightLrMult) head_weight_lr_mult = static_cast<float>(*launchArgs.headWeightLrMult);
    if (launchArgs.headBiasLrMult) head_bias_lr_mult = static_cast<float>(*launchArgs.headBiasLrMult);
}
}
