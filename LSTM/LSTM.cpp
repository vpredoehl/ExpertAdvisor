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
#define LSTM_DEBUG_PRINTS 0
#endif

#ifndef LSTM_TRAINING_PROGRESS
#define LSTM_TRAINING_PROGRESS 1
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
#if 0
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
    
    // Initialize output head (small weights, zero bias)
    for (size_t i = 0; i < hidden_size; ++i) returnHeadWeight.SetValue(i, 0, 0.01f);
    returnHeadBias.SetValue(0, 0, 0.0f);

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

    for (size_t i = 0; i < hidden_size; ++i) returnHeadWeight.SetValue(i, 0, 0.01f);
    returnHeadBias.SetValue(0, 0, 0.0f);

    long_term = lt; short_term = st;
#endif
}

float LSTM::CalculateBatch(std::ranges::subrange<DataSet::const_iterator> batch)
{
    float predicted_close = 0;
#if LSTM_TRAINING_PROGRESS
    double runningLoss = 0.0;
    size_t windowCount = 0;
#endif

    // Slide a 5-step window across this 15-step batch
    for (auto window_start = batch.begin(); window_start + window_size < batch.end(); ++window_start)
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
            auto i_input = MetaNN::Sigmoid(gates2D[0]);
            auto f_forget = MetaNN::Sigmoid(gates2D[1]);
            auto g_cell_candidate = MetaNN::Tanh   (gates2D[2]);
            auto o_output = MetaNN::Sigmoid(gates2D[3]);

            // Debug: evaluate and print gate activations
#if LSTM_DEBUG_PRINTS
            {
                auto inputGateHandle = i_input.EvalRegister();
                auto forgetGateHandle = f_forget.EvalRegister();
                auto cellCandidateHandle = g_cell_candidate.EvalRegister();
                auto outputGateHandle = o_output.EvalRegister();
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
            auto cellStateExpr = f_forget * previousCellState1D + i_input * g_cell_candidate;                 // 1D
            auto hiddenStateExpr = o_output * MetaNN::Tanh(cellStateExpr);               // 1D

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

        // ---- BPTT for next-step return (regression) ----
        // We need the last sample in the window and the next sample after the window
        auto lastIt = window_start + (window_size - 1);
        auto nextIt = window_start + window_size; // safe due to '<' in loop condition

        // Evaluate to access element values
        auto lastFeat  = MetaNN::Evaluate(*lastIt);
        auto nextFeat  = MetaNN::Evaluate(*nextIt);
        // Column index for close price in FeatureMatrix: open(0), close(1), high(2), low(3)
        constexpr size_t closeCol = 1;
        const float close_T    = lastFeat(0, closeCol);
        const float close_next = nextFeat(0, closeCol);

        // At this point, prevHiddenState/prevCellState hold the final h_T and c_T after processing the window.
        // Predict next-step log return from last hidden state h_T
        auto pred = MetaNN::Evaluate(MetaNN::Dot(prevHiddenState, returnHeadWeight) + returnHeadBias);
        const float next_step_prediction = pred(0, 0);  // y_hat
        const float actual_next_step_return   = std::log(close_next) - std::log(close_T);
        predicted_close = std::exp(next_step_prediction) * close_T;

        // Error and loss
        const float err = next_step_prediction - actual_next_step_return; // dL/dyhat for MSE with factor 1
#if LSTM_TRAINING_PROGRESS
        runningLoss += 0.5 * static_cast<double>(err) * static_cast<double>(err);
        ++windowCount;
#endif

        // Gradient into head using MetaNN
        auto last_hidden_state_eval = MetaNN::Evaluate(prevHiddenState);
        for (size_t i = 0; i < hidden_size; ++i)
        {
            const float grad_w = last_hidden_state_eval(0, i) * err;
            const float cur  = returnHeadWeight(i, 0);
            returnHeadWeight.SetValue(i, 0, cur - learningRate * grad_w);
        }
        const float bcur = returnHeadBias(0, 0);
        returnHeadBias.SetValue(0, 0, bcur - learningRate * err);

        // Backprop signal into h_T and c_T using MetaNN expressions
        auto d_h = MetaNN::Evaluate(prevHiddenState * 0.0f + err * MetaNN::Transpose(returnHeadWeight));
        auto d_c = MetaNN::Evaluate(prevHiddenState * 0.0f); // zeros like h

        // Define a lightweight cache that stores only expressions and concatenated input
        struct StepCacheExpr {
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> x;      // (1, input_size)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> h_prev; // (1, hidden_size)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> c_prev; // (1, hidden_size)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> i;      // (1, hidden_size)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> f;      // (1, hidden_size)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> g;      // (1, hidden_size)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> o;      // (1, hidden_size)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> c;      // (1, hidden_size)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> h;      // (1, hidden_size)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> concat; // (1, n_in)
        };

        std::vector<StepCacheExpr> cache;
        cache.reserve(window_size);

        // Recompute forward pass for caching with deferred evaluation
        ResetPreviousState();
        {
            Window w2 = t.GetWindow(window_start);
            for (const auto& f_sample2 : w2)
            {
                const size_t featWidth2 = f_sample2.Shape()[1];
                MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> concat(1, n_in);
                {
                    auto lowConcat = MetaNN::LowerAccess(concat);
                    float* concat_mem = lowConcat.MutableRawMemory();
                    auto low_f = MetaNN::LowerAccess(f_sample2);
                    const float* f_mem = low_f.RawMemory();
                    std::copy(f_mem, f_mem + featWidth2, concat_mem);
                    auto low_h = MetaNN::LowerAccess(prevHiddenState);
                    const float* h_mem = low_h.RawMemory();
                    std::copy(h_mem, h_mem + hidden_size, concat_mem + featWidth2);
                }

                auto yExpr2 = MetaNN::Dot(concat,  param) + bias;                 // (1, 4H)
                const size_t gateWidth2 = yExpr2.Shape()[1] / 4;                   // H
                auto gates2D_2 = MetaNN::Reshape(yExpr2, MetaNN::Shape(4, gateWidth2));

                auto i_input2 = MetaNN::Sigmoid(gates2D_2[0]);
                auto f_forget2 = MetaNN::Sigmoid(gates2D_2[1]);
                auto g_cell_candidate2 = MetaNN::Tanh(gates2D_2[2]);
                auto o_output2 = MetaNN::Sigmoid(gates2D_2[3]);

                auto previousCellState1D_2 = MetaNN::Reshape(prevCellState, MetaNN::Shape(gateWidth2));
                auto cellStateExpr2 = f_forget2 * previousCellState1D_2 + i_input2 * g_cell_candidate2;   // 1D
                auto hiddenStateExpr2 = o_output2 * MetaNN::Tanh(cellStateExpr2);                          // 1D

                auto c2 = MetaNN::Reshape(cellStateExpr2, MetaNN::Shape(1, gateWidth2));
                auto h2 = MetaNN::Reshape(hiddenStateExpr2, MetaNN::Shape(1, gateWidth2));

                // Register evaluation minimally to advance state
                auto cHandle = c2.EvalRegister();
                auto hHandle = h2.EvalRegister();
                auto iHandle = i_input2.EvalRegister();
                auto fHandle = f_forget2.EvalRegister();
                auto gHandle = g_cell_candidate2.EvalRegister();
                auto oHandle = o_output2.EvalRegister();
                MetaNN::EvalPlan::Inst().Eval();

                StepCacheExpr sc;
                sc.x = MetaNN::Evaluate(f_sample2);
                sc.h_prev = MetaNN::Evaluate(prevHiddenState);
                sc.c_prev = MetaNN::Evaluate(prevCellState);
                sc.i = MetaNN::Evaluate(MetaNN::Reshape(iHandle.Data(), MetaNN::Shape(1, gateWidth2)));
                sc.f = MetaNN::Evaluate(MetaNN::Reshape(fHandle.Data(), MetaNN::Shape(1, gateWidth2)));
                sc.g = MetaNN::Evaluate(MetaNN::Reshape(gHandle.Data(), MetaNN::Shape(1, gateWidth2)));
                sc.o = MetaNN::Evaluate(MetaNN::Reshape(oHandle.Data(), MetaNN::Shape(1, gateWidth2)));
                sc.c = MetaNN::Evaluate(cHandle.Data());
                sc.h = MetaNN::Evaluate(hHandle.Data());
                sc.concat = concat;

                prevCellState = sc.c;
                prevHiddenState = sc.h;

                cache.push_back(std::move(sc));
            }
        }

        // Prepare accumulators for gradients as concrete zero matrices
        MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_param_expr(param.Shape()[0], param.Shape()[1]);
        {
            auto low = MetaNN::LowerAccess(d_param_expr);
            std::fill(low.MutableRawMemory(), low.MutableRawMemory() + param.Shape()[0]*param.Shape()[1], 0.0f);
        }
        MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_bias_expr(bias.Shape()[0], bias.Shape()[1]);
        {
            auto low = MetaNN::LowerAccess(d_bias_expr);
            std::fill(low.MutableRawMemory(), low.MutableRawMemory() + bias.Shape()[0]*bias.Shape()[1], 0.0f);
        }

        // Backward through time using MetaNN elementwise ops
        for (int tstep = static_cast<int>(cache.size()) - 1; tstep >= 0; --tstep)
        {
            const auto& sc = cache[static_cast<size_t>(tstep)];

            // tanh(c_t)
            auto tanh_c = MetaNN::Tanh(sc.c);

            // d_o = d_h * tanh(c_t) * o * (1 - o)
            auto d_o = d_h * tanh_c * sc.o * (1.0f - sc.o);

            // d_c_from_h = d_h * o * (1 - tanh(c_t)^2)
            auto d_c_from_h = d_h * sc.o * (1.0f - tanh_c * tanh_c);

            // total d_c contribution at this step will be added later
            auto dct = d_c + d_c_from_h;

            // d_i, d_g, d_f (pre-activation gradients)
            auto d_i = dct * sc.g * sc.i * (1.0f - sc.i);
            auto d_g = dct * sc.i * (1.0f - sc.g * sc.g);
            auto d_f = dct * sc.c_prev * sc.f * (1.0f - sc.f);

            // Reshape 1D gate gradients to (1, hidden_size) for packing
            auto di2D = MetaNN::Reshape(d_i, MetaNN::Shape(1, hidden_size));
            auto df2D = MetaNN::Reshape(d_f, MetaNN::Shape(1, hidden_size));
            auto dg2D = MetaNN::Reshape(d_g, MetaNN::Shape(1, hidden_size));
            auto do2D = MetaNN::Reshape(d_o, MetaNN::Shape(1, hidden_size));

            // Pack [di df dg do] into (1, 4H)
            // Concatenate along feature dimension without MetaNN::Concat (not available)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_pre(1, 4 * hidden_size);
            {
                // Evaluate expressions to concrete matrices before raw access
                auto di_eval = MetaNN::Evaluate(di2D);
                auto df_eval = MetaNN::Evaluate(df2D);
                auto dg_eval = MetaNN::Evaluate(dg2D);
                auto do_eval = MetaNN::Evaluate(do2D);

                auto low_pre = MetaNN::LowerAccess(d_pre);
                float* pre_mem = low_pre.MutableRawMemory();

                auto low_di = MetaNN::LowerAccess(di_eval);
                auto low_df = MetaNN::LowerAccess(df_eval);
                auto low_dg = MetaNN::LowerAccess(dg_eval);
                auto low_do = MetaNN::LowerAccess(do_eval);

                const float* di_mem = low_di.RawMemory();
                const float* df_mem = low_df.RawMemory();
                const float* dg_mem = low_dg.RawMemory();
                const float* do_mem = low_do.RawMemory();

                // Each is shape (1, hidden_size) in row-major contiguous storage
                std::copy(di_mem, di_mem + hidden_size, pre_mem + 0 * hidden_size);
                std::copy(df_mem, df_mem + hidden_size, pre_mem + 1 * hidden_size);
                std::copy(dg_mem, dg_mem + hidden_size, pre_mem + 2 * hidden_size);
                std::copy(do_mem, do_mem + hidden_size, pre_mem + 3 * hidden_size);
            }

            // Accumulate bias gradient
            {
                auto low_dst = MetaNN::LowerAccess(d_bias_expr);
                float* dst = low_dst.MutableRawMemory();
                auto low_src = MetaNN::LowerAccess(d_pre);
                const float* src = low_src.RawMemory();
                const size_t len = 4 * hidden_size;
                for (size_t j = 0; j < len; ++j) { dst[j] += src[j]; }
            }

            // Accumulate weight gradient: concat^T * d_pre
            auto dW_t = MetaNN::Dot(MetaNN::Transpose(sc.concat), d_pre);
            {
                auto dW_eval = MetaNN::Evaluate(dW_t);
                auto low_dst = MetaNN::LowerAccess(d_param_expr);
                float* dst = low_dst.MutableRawMemory();
                auto low_src = MetaNN::LowerAccess(dW_eval);
                const float* src = low_src.RawMemory();
                const size_t rows = param.Shape()[0];
                const size_t cols = param.Shape()[1];
                for (size_t r = 0; r < rows; ++r) {
                    for (size_t c = 0; c < cols; ++c) {
                        dst[r * cols + c] += src[r * cols + c];
                    }
                }
            }

            // Propagate to previous hidden: use the part of W corresponding to h_prev (last hidden_size rows of concat)
            // Extract W_h (the block of param that maps previous hidden state) without SubMatrix
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> W_h(hidden_size, 4 * hidden_size);
            {
                const size_t rows = param.Shape()[0];
                const size_t cols = param.Shape()[1];
                const size_t rowOffset = n_in - hidden_size; // starting row index for h_prev block
                auto low_src = MetaNN::LowerAccess(param);
                const float* src = low_src.RawMemory();
                auto low_dst = MetaNN::LowerAccess(W_h);
                float* dst = low_dst.MutableRawMemory();
                // Copy last hidden_size rows from param into W_h
                for (size_t r = 0; r < hidden_size; ++r) {
                    const size_t srcRow = rowOffset + r;
                    // source row begins at srcRow * cols, copy first 4*hidden_size columns
                    const float* srcRowPtr = src + srcRow * cols;
                    float* dstRowPtr = dst + r * (4 * hidden_size);
                    std::copy(srcRowPtr, srcRowPtr + (4 * hidden_size), dstRowPtr);
                }
            }
            // Now compute d_h_prev = d_pre (1 x 4H) * W_h^T (4H x H) -> (1 x H)
            auto d_h_prev = MetaNN::Dot(d_pre, MetaNN::Transpose(W_h));
            d_h = MetaNN::Evaluate(MetaNN::Reshape(d_h_prev, MetaNN::Shape(1, hidden_size)));
            // Propagate to previous cell
            d_c = MetaNN::Evaluate(MetaNN::Reshape((dct * sc.f), MetaNN::Shape(1, hidden_size)));
        }

        // Removed:
        // auto d_param_eval = d_param_expr.EvalRegister();
        // auto d_bias_eval  = d_bias_expr.EvalRegister();
        // MetaNN::EvalPlan::Inst().Eval();

        // Apply SGD to param and bias using concrete matrices directly
        for (size_t r = 0; r < n_in; ++r)
        {
            for (size_t c = 0; c < 4 * hidden_size; ++c)
            {
                float cur = param(r, c);
                float grad = d_param_expr(r, c);
                param.SetValue(r, c, cur - learningRate * grad);
            }
        }
        for (size_t j = 0; j < 4 * hidden_size; ++j)
        {
            float cur = bias(0, j);
            float grad = d_bias_expr(0, j);
            bias.SetValue(0, j, cur - learningRate * grad);
        }
    }

#if LSTM_TRAINING_PROGRESS
    if (windowCount > 0) {
        std::cout << "Batch MSE: " << (runningLoss / static_cast<double>(windowCount))
                  << " (" << windowCount << " windows)" << std::endl;
    }
    // Print the current returnHeadWeight vector (hidden_size x 1)
    std::cout << "returnHeadWeight: [";
    for (size_t i = 0; i < hidden_size; ++i) {
        std::cout << returnHeadWeight(i, 0);
        if (i + 1 < hidden_size) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
#endif
    return predicted_close;
}


