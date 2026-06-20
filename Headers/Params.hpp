//
//  Params.hpp
//  LSTM
//
//  Created by Vincent Predoehl on 2/2/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#ifndef Params_h
#define Params_h

#include <cstddef>

enum class CandleTF { m15 = 0, Hour = 1, Hr4 = 2, Day = 3, Week = 4 };
inline constexpr int time_cycle_seconds(CandleTF tc)
{
    switch (tc) {
        case CandleTF::m15:      return 15 * 60;      // 900 seconds
        case CandleTF::Hour:     return 60 * 60;      // 3600 seconds
        case CandleTF::Hr4:      return 4  * 60 * 60; // 14400 seconds
        case CandleTF::Day:      return 24 * 60 * 60; // 86400 seconds
        case CandleTF::Week:     return 7  * 24 * 60 * 60; // 604800 seconds
    }
    return 0;
}


inline constexpr float c_next_threshold_for(CandleTF tc)
{
    switch (tc)
    {
        case CandleTF::m15:      return 0.005f; // ~0.5% for 15m
        case CandleTF::Hour:     return 0.010f; // ~1.0% for 1h
        case CandleTF::Hr4:      return 0.020f; // ~2.0% for 4h
        case CandleTF::Day:      return 0.030f; // ~3.0% for 1d
        case CandleTF::Week:     return 0.050f; // ~5.0% for 1w
    }
    return 0.02f;
}

inline constexpr int window_size_for(CandleTF tf)
{
    // Keep a constant temporal span equivalent to 64 steps at 4 hours
    constexpr int base_window_steps_hr4 = 64;
    constexpr int base_span_seconds = base_window_steps_hr4 * time_cycle_seconds(CandleTF::Hr4);
    const int denom = time_cycle_seconds(tf);
    // round to nearest integer number of steps
    return (base_span_seconds + denom / 2) / denom;
}

inline constexpr int prediction_horizon_for(CandleTF tf)
{
    // Keep a constant prediction span equivalent to 64 hours
    constexpr int base_horizon_seconds = 64 * 60 * 60; // 64 hours
    const int denom = time_cycle_seconds(tf);
    int steps = (base_horizon_seconds + denom / 2) / denom; // round to nearest
    return (steps < 1) ? 1 : steps;
}


inline constexpr std::size_t default_hidden_size = 64;
inline constexpr std::size_t default_window_size = 64;
inline constexpr int default_epoch_count = 60;
inline constexpr std::size_t default_prediction_horizon = 8;
inline constexpr float default_c_next_threshold = .0008f;
inline constexpr std::size_t default_num_layers = 1;
inline constexpr int default_normalization_version = 1;

inline std::size_t hidden_size = default_hidden_size;
constexpr auto feature_size = 32;
//constexpr auto n_in = feature_size + hidden_size;
inline std::size_t n_out = default_hidden_size;
constexpr size_t direction_output_size = 3;

// sequence of features
constexpr CandleTF candle_duration = CandleTF::m15; // default cycle
inline std::size_t window_size = default_window_size;//window_size_for(candle_duration);
constexpr auto batch_size = 256;
inline int epoch_count = default_epoch_count;
inline std::size_t prediction_horizon = default_prediction_horizon;//prediction_horizon_for(candle_duration);
inline std::size_t num_layers = default_num_layers;
inline int normalization_version = default_normalization_version;

constexpr float kFeatureScale = 1000.0f;
inline float c_next_threshold = default_c_next_threshold;  //c_next_threshold_for(candle_duration);

constexpr size_t rolling_vol_lookback = 32;
constexpr size_t rolling_ret_lookback = 32;
// Class weights for 3-class direction loss: order (Down, Neutral, Up)
// Adjust these to rebalance the contribution of each class to the total loss.
constexpr float kClassWeightDown    = 1.0f;
constexpr float kClassWeightNeutral = 1.0f;
constexpr float kClassWeightUp      = 1.0f;

// Utility to combine per-class losses using the weights above.
// loss_total = w_up * loss_up + w_down * loss_down + w_neutral * loss_neutral
inline constexpr float weighted_direction_loss(float lossDown, float lossNeutral, float lossUp)
{
    return kClassWeightDown * lossDown
         + kClassWeightNeutral * lossNeutral
         + kClassWeightUp * lossUp;
}
// Column index for close feature: c_t = log(close_t / close_{t-1})
constexpr size_t closeCol = 1;


#include <vector>
#include "MetaNN/meta_nn.h"

using FeatureMatrix = MetaNN::Matrix<float, MetaNN::DeviceTags::Metal>;
using DataSet = std::vector<FeatureMatrix>;
using Window = std::ranges::subrange<DataSet::const_iterator, DataSet::const_iterator>;
using Batch = Window;


#endif /* Params_h */
