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


// Debug helpers: compile out debug code and prints when disabled
#if LSTM_DEBUG_PRINTS
#define LSTM_DEBUG(code) do { code } while(0)
#define LSTM_DPRINT(label, mat) printMatrix(label, mat)
#else
#define LSTM_DEBUG(code) do {} while(0)
#define LSTM_DPRINT(label, mat) do {} while(0)
#endif

#ifndef LSTM_MIXED_PRECISION
#define LSTM_MIXED_PRECISION 0
#endif

#if LSTM_MIXED_PRECISION
using AccumScalar = double;  // higher-precision accumulation
#else
using AccumScalar = float;   // default accumulation precision
#endif

#ifndef LSTM_USE_GRAD_CLIP
#define LSTM_USE_GRAD_CLIP 0
#endif
#ifndef LSTM_GRAD_CLIP_THRESHOLD
#define LSTM_GRAD_CLIP_THRESHOLD 1.0f
#endif

#ifndef LSTM_OPTIMIZER_SGD
#define LSTM_OPTIMIZER_SGD 0
#define LSTM_OPTIMIZER_MOMENTUM 1
#define LSTM_OPTIMIZER_ADAM 2
#endif
#ifndef LSTM_OPTIMIZER
#define LSTM_OPTIMIZER LSTM_OPTIMIZER_SGD
#endif
#ifndef LSTM_MOMENTUM
#define LSTM_MOMENTUM 0.9f
#endif
#ifndef LSTM_ADAM_BETA1
#define LSTM_ADAM_BETA1 0.9f
#endif
#ifndef LSTM_ADAM_BETA2
#define LSTM_ADAM_BETA2 0.999f
#endif
#ifndef LSTM_ADAM_EPS
#define LSTM_ADAM_EPS 1e-8f
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

    // Count windows in this batch for gradient averaging
    size_t windowsInBatch = 0;

    // Accumulate head gradients across all windows in the batch
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> d_headW_accum(hidden_size, 1);
    {
        auto low = MetaNN::LowerAccess(d_headW_accum);
        std::fill(low.MutableRawMemory(), low.MutableRawMemory() + hidden_size * 1, 0.0f);
    }
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> d_headB_accum(1, 1);
    {
        auto low = MetaNN::LowerAccess(d_headB_accum);
        std::fill(low.MutableRawMemory(), low.MutableRawMemory() + 1, 0.0f);
    }

    // Accumulate LSTM core gradients across all windows in the batch
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> d_param_accum(param.Shape()[0], param.Shape()[1]);
    {
        auto low = MetaNN::LowerAccess(d_param_accum);
        std::fill(low.MutableRawMemory(), low.MutableRawMemory() + param.Shape()[0] * param.Shape()[1], 0.0f);
    }
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> d_bias_accum(bias.Shape()[0], bias.Shape()[1]);
    {
        auto low = MetaNN::LowerAccess(d_bias_accum);
        std::fill(low.MutableRawMemory(), low.MutableRawMemory() + bias.Shape()[0] * bias.Shape()[1], 0.0f);
    }

    // Slide a 5-step window across this 15-step batch
    for (auto window_start = batch.begin(); window_start + window_size < batch.end(); ++window_start)
    {
        // Reset states for an independent window
        ResetPreviousState();

        struct StepCache
        {
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

        std::vector<StepCache> cache;
        cache.reserve(window_size);

        // Build a 5-step window starting at 'start' using the const iterator overload
        Window w = t.GetWindow(window_start);

        for(const auto& f_sample : w)
        {
            LSTM_DPRINT("Feature", f_sample);

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
            LSTM_DPRINT("Concatenated [x_t|h_{t-1}]", InputAndPrevHidden);

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
            LSTM_DEBUG({
                auto inputGateHandle = i_input.EvalRegister();
                auto forgetGateHandle = f_forget.EvalRegister();
                auto cellCandidateHandle = g_cell_candidate.EvalRegister();
                auto outputGateHandle = o_output.EvalRegister();
                MetaNN::EvalPlan::Inst().Eval();
                auto inputGateMatrix = MetaNN::Reshape(inputGateHandle.Data(), MetaNN::Shape(1, gateWidth));
                auto forgetGateMatrix = MetaNN::Reshape(forgetGateHandle.Data(), MetaNN::Shape(1, gateWidth));
                auto cellCandidateMatrix = MetaNN::Reshape(cellCandidateHandle.Data(), MetaNN::Shape(1, gateWidth));
                auto outputGateMatrix = MetaNN::Reshape(outputGateHandle.Data(), MetaNN::Shape(1, gateWidth));
                LSTM_DPRINT("Input gate (i)", inputGateMatrix);
                LSTM_DPRINT("Forget gate (f)", forgetGateMatrix);
                LSTM_DPRINT("Cell candidate (g)", cellCandidateMatrix);
                LSTM_DPRINT("Output gate (o)", outputGateMatrix);
                LSTM_DPRINT("Previous hidden state", prevHiddenState);
                LSTM_DPRINT("Previous cell state", prevCellState);
            });

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

            LSTM_DEBUG({
                auto cellStateMatrix = MetaNN::Reshape(cellStateHandle.Data(), MetaNN::Shape(1, gateWidth));
                auto hiddenStateMatrix = MetaNN::Reshape(hiddenStateHandle.Data(), MetaNN::Shape(1, gateWidth));
                LSTM_DPRINT("Cell state c_t", cellStateMatrix);
                LSTM_DPRINT("Hidden state h_t", hiddenStateMatrix);
            });

            // Cache this step's intermediates BEFORE updating persistent states
            StepCache sc;
            sc.x = MetaNN::Evaluate(f_sample);
            sc.h_prev = MetaNN::Evaluate(prevHiddenState);
            sc.c_prev = MetaNN::Evaluate(MetaNN::Reshape(previousCellState1D, MetaNN::Shape(1, hidden_size)));
            sc.i = MetaNN::Evaluate(MetaNN::Reshape(i_input, MetaNN::Shape(1, hidden_size)));
            sc.f = MetaNN::Evaluate(MetaNN::Reshape(f_forget, MetaNN::Shape(1, hidden_size)));
            sc.g = MetaNN::Evaluate(MetaNN::Reshape(g_cell_candidate, MetaNN::Shape(1, hidden_size)));
            sc.o = MetaNN::Evaluate(MetaNN::Reshape(o_output, MetaNN::Shape(1, hidden_size)));
            sc.c = MetaNN::Evaluate(cellStateHandle.Data());
            sc.h = MetaNN::Evaluate(hiddenStateHandle.Data());
            sc.concat = InputAndPrevHidden;

            // Now update persistent states
            prevCellState = cellStateHandle.Data();
            prevHiddenState = hiddenStateHandle.Data();

            cache.push_back(std::move(sc));
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
        ++windowsInBatch;

        // Backprop signal into h_T uses a snapshot of head weights (and bias for completeness)
        auto headW_snapshot = MetaNN::Evaluate(returnHeadWeight);
        auto headB_snapshot = MetaNN::Evaluate(returnHeadBias);

        // Gradient into head using MetaNN (lazy expressions) - accumulate for batch update
        auto d_returnHeadWeight = MetaNN::Transpose(prevHiddenState) * err; // (H x 1)
        auto d_returnHeadWeight_eval = MetaNN::Evaluate(d_returnHeadWeight);
        {
            auto low_acc = MetaNN::LowerAccess(d_headW_accum);
            AccumScalar* acc = low_acc.MutableRawMemory();
            auto low_src = MetaNN::LowerAccess(d_returnHeadWeight_eval);
            const float* src = low_src.RawMemory();
            for (size_t i = 0; i < hidden_size; ++i) { acc[i] += static_cast<AccumScalar>(src[i]); }
        }
        // Bias gradient as a lazy expression with proper shape (1 x 1) and accumulate
        auto d_returnHeadBias_expr = headB_snapshot * 0.0f + err;
        auto d_returnHeadBias_eval = MetaNN::Evaluate(d_returnHeadBias_expr);
        {
            auto low_b_acc = MetaNN::LowerAccess(d_headB_accum);
            AccumScalar* bacc = low_b_acc.MutableRawMemory();
            auto low_b_src = MetaNN::LowerAccess(d_returnHeadBias_eval);
            const float* bsrc = low_b_src.RawMemory();
            bacc[0] += static_cast<AccumScalar>(bsrc[0]);
        }

        auto d_h = MetaNN::Evaluate(err * MetaNN::Transpose(headW_snapshot));
        auto d_c = MetaNN::Evaluate(prevHiddenState * 0.0f); // zeros like h

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

        // Accumulate LSTM core gradients for batch update
        {
            auto low_dst = MetaNN::LowerAccess(d_param_accum);
            AccumScalar* dst = low_dst.MutableRawMemory();
            auto low_src = MetaNN::LowerAccess(d_param_expr);
            const float* src = low_src.RawMemory();
            const size_t rows = param.Shape()[0];
            const size_t cols = param.Shape()[1];
            for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < cols; ++c) {
                    dst[r * cols + c] += static_cast<AccumScalar>(src[r * cols + c]);
                }
            }
        }
        {
            auto low_dst = MetaNN::LowerAccess(d_bias_accum);
            AccumScalar* dst = low_dst.MutableRawMemory();
            auto low_src = MetaNN::LowerAccess(d_bias_expr);
            const float* src = low_src.RawMemory();
            const size_t len = bias.Shape()[0] * bias.Shape()[1];
            for (size_t j = 0; j < len; ++j) { dst[j] += static_cast<AccumScalar>(src[j]); }
        }
    }

    // Average gradients by number of windows
    const float lrScale = (windowsInBatch > 0) ? (learningRate / static_cast<float>(windowsInBatch)) : learningRate;

#if LSTM_USE_GRAD_CLIP
    // Global norm gradient clipping across all accumulated gradients
    double sumsq = 0.0;
    {
        auto low = MetaNN::LowerAccess(d_param_accum);
        const AccumScalar* p = low.RawMemory();
        const size_t len = param.Shape()[0] * param.Shape()[1];
        for (size_t i = 0; i < len; ++i) { double v = static_cast<double>(p[i]); sumsq += v * v; }
    }
    {
        auto low = MetaNN::LowerAccess(d_bias_accum);
        const AccumScalar* p = low.RawMemory();
        const size_t len = bias.Shape()[0] * bias.Shape()[1];
        for (size_t i = 0; i < len; ++i) { double v = static_cast<double>(p[i]); sumsq += v * v; }
    }
    {
        auto low = MetaNN::LowerAccess(d_headW_accum);
        const AccumScalar* p = low.RawMemory();
        const size_t len = hidden_size;
        for (size_t i = 0; i < len; ++i) { double v = static_cast<double>(p[i]); sumsq += v * v; }
    }
    {
        auto low = MetaNN::LowerAccess(d_headB_accum);
        const AccumScalar* p = low.RawMemory();
        const size_t len = 1;
        for (size_t i = 0; i < len; ++i) { double v = static_cast<double>(p[i]); sumsq += v * v; }
    }
    const double global_norm = std::sqrt(sumsq);
    if (global_norm > static_cast<double>(LSTM_GRAD_CLIP_THRESHOLD))
    {
        const double scale = static_cast<double>(LSTM_GRAD_CLIP_THRESHOLD) / global_norm;
        {
            auto low = MetaNN::LowerAccess(d_param_accum);
            AccumScalar* p = low.MutableRawMemory();
            const size_t len = param.Shape()[0] * param.Shape()[1];
            for (size_t i = 0; i < len; ++i) { p[i] = static_cast<AccumScalar>(static_cast<double>(p[i]) * scale); }
        }
        {
            auto low = MetaNN::LowerAccess(d_bias_accum);
            AccumScalar* p = low.MutableRawMemory();
            const size_t len = bias.Shape()[0] * bias.Shape()[1];
            for (size_t i = 0; i < len; ++i) { p[i] = static_cast<AccumScalar>(static_cast<double>(p[i]) * scale); }
        }
        {
            auto low = MetaNN::LowerAccess(d_headW_accum);
            AccumScalar* p = low.MutableRawMemory();
            const size_t len = hidden_size;
            for (size_t i = 0; i < len; ++i) { p[i] = static_cast<AccumScalar>(static_cast<double>(p[i]) * scale); }
        }
        {
            auto low = MetaNN::LowerAccess(d_headB_accum);
            AccumScalar* p = low.MutableRawMemory();
            const size_t len = 1;
            for (size_t i = 0; i < len; ++i) { p[i] = static_cast<AccumScalar>(static_cast<double>(p[i]) * scale); }
        }
    }
#endif

    // Initialize optimizer state (Momentum/Adam) once with correct shapes
#if (LSTM_OPTIMIZER == LSTM_OPTIMIZER_MOMENTUM) || (LSTM_OPTIMIZER == LSTM_OPTIMIZER_ADAM)
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> v_param(param.Shape()[0], param.Shape()[1]);
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> v_bias(bias.Shape()[0], bias.Shape()[1]);
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> v_headW(hidden_size, 1);
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> v_headB(1, 1);
    static bool opt_state_initialized = false;
#if LSTM_OPTIMIZER == LSTM_OPTIMIZER_ADAM
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> m_param(param.Shape()[0], param.Shape()[1]);
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> m_bias(bias.Shape()[0], bias.Shape()[1]);
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> m_headW(hidden_size, 1);
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> m_headB(1, 1);
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> s_param(param.Shape()[0], param.Shape()[1]);
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> s_bias(bias.Shape()[0], bias.Shape()[1]);
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> s_headW(hidden_size, 1);
    static MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> s_headB(1, 1);
    static AccumScalar beta1_pow = static_cast<AccumScalar>(1);
    static AccumScalar beta2_pow = static_cast<AccumScalar>(1);
    static size_t adam_step = 0;
#endif
    if (!opt_state_initialized)
    {
        auto zeroFill = [](auto& mat)
        {
            auto low = MetaNN::LowerAccess(mat);
            using ElemT = typename std::remove_reference<decltype(*low.MutableRawMemory())>::type;
            std::fill(low.MutableRawMemory(), low.MutableRawMemory() + mat.Shape()[0] * mat.Shape()[1], static_cast<ElemT>(0));
        };
        zeroFill(v_param); zeroFill(v_bias); zeroFill(v_headW); zeroFill(v_headB);
#if LSTM_OPTIMIZER == LSTM_OPTIMIZER_ADAM
        zeroFill(m_param); zeroFill(m_bias); zeroFill(m_headW); zeroFill(m_headB);
        zeroFill(s_param); zeroFill(s_bias); zeroFill(s_headW); zeroFill(s_headB);
        beta1_pow = static_cast<AccumScalar>(1);
        beta2_pow = static_cast<AccumScalar>(1);
        adam_step = 0;
#endif
        opt_state_initialized = true;
    }
#if LSTM_OPTIMIZER == LSTM_OPTIMIZER_ADAM
    // Advance Adam step and running powers for bias correction
    adam_step += 1;
    beta1_pow = beta1_pow * static_cast<AccumScalar>(LSTM_ADAM_BETA1);
    beta2_pow = beta2_pow * static_cast<AccumScalar>(LSTM_ADAM_BETA2);
#endif
#endif

    // Apply accumulated LSTM core gradients once per batch
#if LSTM_OPTIMIZER == LSTM_OPTIMIZER_MOMENTUM
    {
        const AccumScalar mu = static_cast<AccumScalar>(LSTM_MOMENTUM);
        for (size_t r = 0; r < n_in; ++r)
        {
            for (size_t c = 0; c < 4 * hidden_size; ++c)
            {
                AccumScalar g = d_param_accum(r, c);
                AccumScalar v = mu * v_param(r, c) + g;
                v_param.SetValue(r, c, v);
                float cur = param(r, c);
                float upd = lrScale * static_cast<float>(v);
                param.SetValue(r, c, cur - upd);
            }
        }
        for (size_t j = 0; j < 4 * hidden_size; ++j)
        {
            AccumScalar g = d_bias_accum(0, j);
            AccumScalar v = mu * v_bias(0, j) + g;
            v_bias.SetValue(0, j, v);
            float cur = bias(0, j);
            float upd = lrScale * static_cast<float>(v);
            bias.SetValue(0, j, cur - upd);
        }
    }
#elif LSTM_OPTIMIZER == LSTM_OPTIMIZER_ADAM
    {
        const AccumScalar beta1 = static_cast<AccumScalar>(LSTM_ADAM_BETA1);
        const AccumScalar beta2 = static_cast<AccumScalar>(LSTM_ADAM_BETA2);
        const AccumScalar eps = static_cast<AccumScalar>(LSTM_ADAM_EPS);
        for (size_t r = 0; r < n_in; ++r)
        {
            for (size_t c = 0; c < 4 * hidden_size; ++c)
            {
                AccumScalar g = d_param_accum(r, c);
                AccumScalar m = beta1 * m_param(r, c) + (static_cast<AccumScalar>(1) - beta1) * g;
                AccumScalar s = beta2 * s_param(r, c) + (static_cast<AccumScalar>(1) - beta2) * g * g;
                m_param.SetValue(r, c, m);
                s_param.SetValue(r, c, s);
                AccumScalar mhat = m / (static_cast<AccumScalar>(1) - beta1_pow);
                AccumScalar vhat = s / (static_cast<AccumScalar>(1) - beta2_pow);
                float upd = lrScale * static_cast<float>( mhat / (static_cast<AccumScalar>(std::sqrt(static_cast<double>(vhat))) + eps) );
                float cur = param(r, c);
                param.SetValue(r, c, cur - upd);
            }
        }
        for (size_t j = 0; j < 4 * hidden_size; ++j)
        {
            AccumScalar g = d_bias_accum(0, j);
            AccumScalar m = beta1 * m_bias(0, j) + (static_cast<AccumScalar>(1) - beta1) * g;
            AccumScalar s = beta2 * s_bias(0, j) + (static_cast<AccumScalar>(1) - beta2) * g * g;
            m_bias.SetValue(0, j, m);
            s_bias.SetValue(0, j, s);
            AccumScalar mhat = m / (static_cast<AccumScalar>(1) - beta1_pow);
            AccumScalar vhat = s / (static_cast<AccumScalar>(1) - beta2_pow);
            float upd = lrScale * static_cast<float>( mhat / (static_cast<AccumScalar>(std::sqrt(static_cast<double>(vhat))) + eps) );
            float cur = bias(0, j);
            bias.SetValue(0, j, cur - upd);
        }
    }
#else
    {
        for (size_t r = 0; r < n_in; ++r)
        {
            for (size_t c = 0; c < 4 * hidden_size; ++c)
            {
                float cur = param(r, c);
                float grad = static_cast<float>(d_param_accum(r, c));
                param.SetValue(r, c, cur - lrScale * grad);
            }
        }
        for (size_t j = 0; j < 4 * hidden_size; ++j)
        {
            float cur = bias(0, j);
            float grad = static_cast<float>(d_bias_accum(0, j));
            bias.SetValue(0, j, cur - lrScale * grad);
        }
    }
#endif

    // Apply accumulated head gradients once per batch
#if LSTM_OPTIMIZER == LSTM_OPTIMIZER_MOMENTUM
    {
        const AccumScalar mu = static_cast<AccumScalar>(LSTM_MOMENTUM);
        for (size_t i = 0; i < hidden_size; ++i)
        {
            AccumScalar g = d_headW_accum(i, 0);
            AccumScalar v = mu * v_headW(i, 0) + g;
            v_headW.SetValue(i, 0, v);
            float cur = returnHeadWeight(i, 0);
            float upd = lrScale * static_cast<float>(v);
            returnHeadWeight.SetValue(i, 0, cur - upd);
        }
        {
            AccumScalar g = d_headB_accum(0, 0);
            AccumScalar v = mu * v_headB(0, 0) + g;
            v_headB.SetValue(0, 0, v);
            float bcur = returnHeadBias(0, 0);
            float upd = lrScale * static_cast<float>(v);
            returnHeadBias.SetValue(0, 0, bcur - upd);
        }
    }
#elif LSTM_OPTIMIZER == LSTM_OPTIMIZER_ADAM
    {
        const AccumScalar beta1 = static_cast<AccumScalar>(LSTM_ADAM_BETA1);
        const AccumScalar beta2 = static_cast<AccumScalar>(LSTM_ADAM_BETA2);
        const AccumScalar eps = static_cast<AccumScalar>(LSTM_ADAM_EPS);
        for (size_t i = 0; i < hidden_size; ++i)
        {
            AccumScalar g = d_headW_accum(i, 0);
            AccumScalar m = beta1 * m_headW(i, 0) + (static_cast<AccumScalar>(1) - beta1) * g;
            AccumScalar s = beta2 * s_headW(i, 0) + (static_cast<AccumScalar>(1) - beta2) * g * g;
            m_headW.SetValue(i, 0, m);
            s_headW.SetValue(i, 0, s);
            AccumScalar mhat = m / (static_cast<AccumScalar>(1) - beta1_pow);
            AccumScalar vhat = s / (static_cast<AccumScalar>(1) - beta2_pow);
            float upd = lrScale * static_cast<float>( mhat / (static_cast<AccumScalar>(std::sqrt(static_cast<double>(vhat))) + eps) );
            float cur = returnHeadWeight(i, 0);
            returnHeadWeight.SetValue(i, 0, cur - upd);
        }
        {
            AccumScalar g = d_headB_accum(0, 0);
            AccumScalar m = beta1 * m_headB(0, 0) + (static_cast<AccumScalar>(1) - beta1) * g;
            AccumScalar s = beta2 * s_headB(0, 0) + (static_cast<AccumScalar>(1) - beta2) * g * g;
            m_headB.SetValue(0, 0, m);
            s_headB.SetValue(0, 0, s);
            AccumScalar mhat = m / (static_cast<AccumScalar>(1) - beta1_pow);
            AccumScalar vhat = s / (static_cast<AccumScalar>(1) - beta2_pow);
            float upd = lrScale * static_cast<float>( mhat / (static_cast<AccumScalar>(std::sqrt(static_cast<double>(vhat))) + eps) );
            float bcur = returnHeadBias(0, 0);
            returnHeadBias.SetValue(0, 0, bcur - upd);
        }
    }
#else
    {
        for (size_t i = 0; i < hidden_size; ++i)
        {
            float cur = returnHeadWeight(i, 0);
            float grad = static_cast<float>(d_headW_accum(i, 0));
            returnHeadWeight.SetValue(i, 0, cur - lrScale * grad);
        }
        {
            float bcur = returnHeadBias(0, 0);
            float bgrad = static_cast<float>(d_headB_accum(0, 0));
            returnHeadBias.SetValue(0, 0, bcur - lrScale * bgrad);
        }
    }
#endif

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


