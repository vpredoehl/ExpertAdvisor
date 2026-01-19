//
//  LSTM.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/18/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#include "LSTM.hpp"
#include <random>
#include <cmath>

// Random helpers: uniform real in [low, high] and symmetric [-limit, limit]
static inline float uniform_between(float low, float high) {
    thread_local std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(low, high);
    return dist(rng);
}

static inline float uniform_symmetric(float limit) {
    return uniform_between(-limit, limit);
}

using namespace EA;

LSTM::LSTM(float lt, float st)
    : param{ W{ n_out, n_in }, W{ n_out, n_in }, W{ n_out, n_in }, W{ n_out, n_in } },
      long_term(lt),
      short_term(st)
{
    // Xavier/Glorot uniform initialization limit
    float limit = std::sqrt(6.0f / (static_cast<float>(n_in) + static_cast<float>(n_out)));

    // Initialize each parameter matrix with values in [-limit, limit]
    for (auto& mat : param) {
        // Assuming W is a matrix type with size n_out x n_in and provides
        // row/column access via operator()(r, c). Adjust if your API differs.
        for (int r = 0; r < n_out; ++r) {
            for (int c = 0; c < n_in; ++c) {
                mat.SetValue(r, c, uniform_symmetric(limit));            }
        }
    }
}
