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
constexpr auto feature_size = 4;
constexpr auto n_in = feature_size + hidden_size;
constexpr auto n_out = hidden_size;

// sequence of features
constexpr auto window_size = 64;
constexpr auto batch_size = 128;

constexpr float kFeatureScale = 1000.0f;

#include <vector>
#include "MetaNN/meta_nn.h"

using FeatureMatrix = MetaNN::Matrix<float, MetaNN::DeviceTags::CPU>;
using DataSet = std::vector<FeatureMatrix>;
using Window = std::ranges::subrange<DataSet::const_iterator, DataSet::const_iterator>;
using Batch = Window;


#endif /* Params_h */
