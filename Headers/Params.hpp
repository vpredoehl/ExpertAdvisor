//
//  Params.hpp
//  LSTM
//
//  Created by Vincent Predoehl on 2/2/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#ifndef Params_h
#define Params_h


constexpr auto hidden_size = 64;
constexpr auto feature_size = 10;
constexpr auto n_in = feature_size + hidden_size;
constexpr auto n_out = hidden_size;

// sequence of features
constexpr auto window_size = 64;
constexpr auto batch_size = 256;
constexpr auto epoch_count = 100;
constexpr auto prediction_horizon = 16;

enum class TimeCycle { Minute15 = 0, Hour = 1, Hour4 = 2, Day = 3, Week = 4 };
constexpr TimeCycle kTimeCycle = TimeCycle::Hour4; // default cycle
constexpr float kFeatureScale = 1000.0f;
constexpr float c_next_threshold = 0.02f;   // skip window threshold .02-.03

constexpr size_t rolling_vol_lookback = 32;
constexpr size_t rolling_ret_lookback = 32;


inline constexpr int time_cycle_seconds(TimeCycle tc)
{
    switch (tc) {
        case TimeCycle::Minute15: return 15 * 60;      // 900 seconds
        case TimeCycle::Hour:     return 60 * 60;      // 3600 seconds
        case TimeCycle::Hour4:    return 4  * 60 * 60; // 14400 seconds
        case TimeCycle::Day:      return 24 * 60 * 60; // 86400 seconds
        case TimeCycle::Week:     return 7  * 24 * 60 * 60; // 604800 seconds
    }
    return 0;
}

// Column index for close feature: c_t = log(close_t / close_{t-1})
constexpr size_t closeCol = 1;


#include <vector>
#include "MetaNN/meta_nn.h"

using FeatureMatrix = MetaNN::Matrix<float, MetaNN::DeviceTags::CPU>;
using DataSet = std::vector<FeatureMatrix>;
using Window = std::ranges::subrange<DataSet::const_iterator, DataSet::const_iterator>;
using Batch = Window;


#endif /* Params_h */

