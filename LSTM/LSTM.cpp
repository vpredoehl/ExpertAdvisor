//
//  LSTM.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/18/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#include <random>
#include <cmath>

#include "LSTM.hpp"
#include "Tensor.hpp"

#ifndef LSTM_DEBUG_PRINTS
#define LSTM_DEBUG_PRINTS 1
#endif


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

LSTM::LSTM(const Tensor& tt, float lt, float st)
: t { tt }
{
#if 1
    // Deterministic constant initialization for verification
    const float weightInit = 0.1f;
    const float biasInit   = 0.0f; // used below when initializing bias
    for (int r = 0; r < n_in; ++r)
        for (int c = 0; c < 4 * n_out; ++c)
            param.SetValue(r, c, weightInit);
    
        // initialize bias, previous hidden and previous cell state
    for (size_t j = 0; j < 4 * n_out; ++j) bias.SetValue(0, j, biasInit);
    for (size_t j = 0; j < hidden_size; ++j)
    {
        prevHiddenState.SetValue(0, j, 0.0f);
        prevCellState.SetValue(0, j, 0.0f);
    }

    long_term = lt; short_term = st;
#else
    // Xavier/Glorot uniform initialization limit
    float limit = std::sqrt(6.0f / (static_cast<float>(n_in) + static_cast<float>(n_out)));

    for (int r = 0; r < n_in; ++r)
        for (int c = 0; c < 4 * n_out; ++c)
            param.SetValue(r, c, uniform_symmetric(limit));
    
        // initialize bias, previous hidden and previous cell state
    for (size_t j = 0; j < 4 * n_out; ++j) bias.SetValue(0, j, 0.0f);
    ResetPreviousState();
    
    long_term = lt; short_term = st;
#endif
}

void LSTM::CalculateBatch(short idx)
{
    auto batch = t.GetBatchClamped(idx);  // safe: handles trailing partial batches
    // Slide a 5-step window across this 15-step batch
    for (auto window_start = batch.begin(); window_start + window_size <= batch.end(); ++window_start)
    {
        // Reset states for an independent window
        ResetPreviousState();
        // Build a 5-step window starting at 'start' using the const iterator overload
        Window w = t.GetWindow(window_start);

        for(const auto& f_sample : w)
        {
#if LSTM_DEBUG_PRINTS
                printMatrix("Feature", f_sample);
#endif
            const size_t featWidth = f_sample.Shape()[1];
            // Concatenate [x_t, h_{t-1}] into a contiguous row without element-wise loops
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> InputAndPrevHidden(1, n_in);
            {
                auto lowConcat = MetaNN::LowerAccess(InputAndPrevHidden);
                float* concat_mem = lowConcat.MutableRawMemory();

                auto low_f = MetaNN::LowerAccess(f_sample);
                const float* f_mem = low_f.RawMemory();
                std::copy(f_mem, f_mem + featWidth, concat_mem);

                auto low_h = MetaNN::LowerAccess(prevHiddenState);
                const float* h_mem = low_h.RawMemory();
                std::copy(h_mem, h_mem + hidden_size, concat_mem + featWidth);
            }
#if LSTM_DEBUG_PRINTS
            printMatrix("Concatenated [x_t|h_{t-1}]", InputAndPrevHidden);
#endif
            auto yExpr = MetaNN::Dot(InputAndPrevHidden,  param) + bias;

            // Split gates as a (4 x gateWidth) view
            const size_t gateWidth = yExpr.Shape()[1] / 4;
            auto gates2D = MetaNN::Reshape(yExpr, MetaNN::Shape(4, gateWidth));

            // Gate activations directly on 1D slices
            auto i = MetaNN::Sigmoid(gates2D[0]);
            auto f = MetaNN::Sigmoid(gates2D[1]);
            auto g = MetaNN::Tanh   (gates2D[2]);
            auto o = MetaNN::Sigmoid(gates2D[3]);

            // Debug: evaluate and print gate activations
#if LSTM_DEBUG_PRINTS
            {
                auto inputGateHandle = i.EvalRegister();
                auto forgetGateHandle = f.EvalRegister();
                auto cellCandidateHandle = g.EvalRegister();
                auto outputGateHandle = o.EvalRegister();
                MetaNN::EvalPlan::Inst().Eval();
                auto inputGateMatrix = MetaNN::Reshape(inputGateHandle.Data(), MetaNN::Shape(1, gateWidth));
                auto forgetGateMatrix = MetaNN::Reshape(forgetGateHandle.Data(), MetaNN::Shape(1, gateWidth));
                auto cellCandidateMatrix = MetaNN::Reshape(cellCandidateHandle.Data(), MetaNN::Shape(1, gateWidth));
                auto outputGateMatrix = MetaNN::Reshape(outputGateHandle.Data(), MetaNN::Shape(1, gateWidth));
                printMatrix("Input gate (i)", inputGateMatrix);
                printMatrix("Forget gate (f)", forgetGateMatrix);
                printMatrix("Cell candidate (g)", cellCandidateMatrix);
                printMatrix("Output gate (o)", outputGateMatrix);
            }
            printMatrix("Previous hidden state", prevHiddenState);
            printMatrix("Previous cell state", prevCellState);
#endif

            // View previousCellState as 1D
            auto previousCellState1D = MetaNN::Reshape(prevCellState, MetaNN::Shape(gateWidth));

            // Vectorized LSTM updates
            auto cellStateExpr = f * previousCellState1D + i * g;                 // 1D
            auto hiddenStateExpr = o * MetaNN::Tanh(cellStateExpr);               // 1D

            // Reshape results to (1 x gateWidth) and evaluate once
            auto cellState2DExpr = MetaNN::Reshape(cellStateExpr, MetaNN::Shape(1, gateWidth));
            auto hiddenState2DExpr = MetaNN::Reshape(hiddenStateExpr, MetaNN::Shape(1, gateWidth));
            auto cellStateHandle = cellState2DExpr.EvalRegister();
            auto hiddenStateHandle = hiddenState2DExpr.EvalRegister();
            MetaNN::EvalPlan::Inst().Eval();

#if LSTM_DEBUG_PRINTS
            {
                auto cellStateMatrix = MetaNN::Reshape(cellStateHandle.Data(), MetaNN::Shape(1, gateWidth));
                auto hiddenStateMatrix = MetaNN::Reshape(hiddenStateHandle.Data(), MetaNN::Shape(1, gateWidth));
                printMatrix("Cell state c_t", cellStateMatrix);
                printMatrix("Hidden state h_t", hiddenStateMatrix);
            }
#endif

            // Assign back to persistent states
            prevCellState = cellStateHandle.Data();
            prevHiddenState = hiddenStateHandle.Data();
        }
    }
}


