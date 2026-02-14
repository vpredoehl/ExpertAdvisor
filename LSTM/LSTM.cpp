//
//  LSTM.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/18/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#include <random>
#include <cmath>
#include <algorithm>

#include "LSTM.hpp"
#include "Tensor.hpp"
#include "MatrixUtils.hpp"

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

    // Lazy head gradient accumulators (expressions) across all windows in the batch
//    auto d_headW_acc_expr = returnHeadWeight * 0.0f; // (H x 1), lazy
//    auto d_headB_acc_expr = returnHeadBias * 0.0f;   // (1 x 1), lazy
    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_headW_accum_f(hidden_size, 1);
    { auto low = MetaNN::LowerAccess(d_headW_accum_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory() + hidden_size * 1, 0.0f); }
    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_headB_accum_f(1, 1);
    { auto low = MetaNN::LowerAccess(d_headB_accum_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory() + 1, 0.0f); }

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
            // Removed concat: no longer used
        };

        std::vector<StepCache> cache;
        cache.reserve(window_size);

        // Hoist forward weight views per window
        auto W_x_win = NNUtils::ViewTopRows<float, MetaNN::DeviceTags::CPU>(param, param.Shape()[0] - hidden_size);
        auto W_h_win = NNUtils::ViewBottomRows<float, MetaNN::DeviceTags::CPU>(param, hidden_size);

        // Build a 5-step window starting at 'start' using the const iterator overload
        Window w = t.GetWindow(window_start);

        for(const auto& f_sample : w)
        {
            LSTM_DPRINT("Feature", f_sample);

            // Replace manual concatenation with separate dot products
            auto yExpr = MetaNN::Dot(f_sample,  W_x_win) + MetaNN::Dot(prevHiddenState, W_h_win) + bias;

            // Debug prints for inputs instead of concatenated
            LSTM_DPRINT("x_t", f_sample);
            LSTM_DPRINT("h_{t-1}", prevHiddenState);

            // Split gates without copies using zero-copy views, then compute activations on 1D views
            const size_t gateWidth = yExpr.Shape()[1] / 4;
            auto [i2D, f2D, g2D, o2D] = NNUtils::SplitGatesRowExpr(yExpr);
            auto i_input = MetaNN::Sigmoid(MetaNN::Reshape(i2D, MetaNN::Shape(gateWidth)));
            auto f_forget = MetaNN::Sigmoid(MetaNN::Reshape(f2D, MetaNN::Shape(gateWidth)));
            auto g_cell_candidate = MetaNN::Tanh(MetaNN::Reshape(g2D, MetaNN::Shape(gateWidth)));
            auto o_output = MetaNN::Sigmoid(MetaNN::Reshape(o2D, MetaNN::Shape(gateWidth)));

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
            sc.x = f_sample;
            sc.h_prev = prevHiddenState;
            sc.c_prev = MetaNN::Evaluate(MetaNN::Reshape(previousCellState1D, MetaNN::Shape(1, hidden_size)));
            sc.i = MetaNN::Evaluate(MetaNN::Reshape(i_input, MetaNN::Shape(1, hidden_size)));
            sc.f = MetaNN::Evaluate(MetaNN::Reshape(f_forget, MetaNN::Shape(1, hidden_size)));
            sc.g = MetaNN::Evaluate(MetaNN::Reshape(g_cell_candidate, MetaNN::Shape(1, hidden_size)));
            sc.o = MetaNN::Evaluate(MetaNN::Reshape(o_output, MetaNN::Shape(1, hidden_size)));
            sc.c = cellStateHandle.Data();
            sc.h = hiddenStateHandle.Data();
            // Removed sc.concat because concatenation not used anymore

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
        auto lastFeat  = *lastIt;
        auto nextFeat  = *nextIt;
        // Column index for close price in FeatureMatrix: open(0), close(1), high(2), low(3)
        constexpr size_t closeCol = 1;
        const float close_T    = lastFeat(0, closeCol);
        const float close_next = nextFeat(0, closeCol);

        // At this point, prevHiddenState/prevCellState hold the final h_T and c_T after processing the window.
        // Predict next-step log return from last hidden state h_T
        auto pred = MetaNN::Dot(prevHiddenState, returnHeadWeight) + returnHeadBias;
        auto predEval = MetaNN::Evaluate(pred);
        const float next_step_prediction = predEval(0, 0);  // y_hat
        const float actual_next_step_return   = std::log(close_next) - std::log(close_T);
        predicted_close = std::exp(next_step_prediction) * close_T;

        // Error and loss
        const float err = next_step_prediction - actual_next_step_return; // dL/dyhat for MSE with factor 1
#if LSTM_TRAINING_PROGRESS
        runningLoss += 0.5 * static_cast<double>(err) * static_cast<double>(err);
        ++windowCount;
#endif
        ++windowsInBatch;

        // Accumulate head gradients across windows (in-place, no snapshots)
        {
            for (size_t i = 0; i < hidden_size; ++i) {
                d_headW_accum_f.SetValue(i, 0, d_headW_accum_f(i, 0) + prevHiddenState(0, i) * err);
            }
            d_headB_accum_f.SetValue(0, 0, d_headB_accum_f(0, 0) + err);
        }

        // Initialize d_h directly from head weights (no Evaluate) and d_c as zeros
        MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_h(1, hidden_size);
        {
            for (size_t i = 0; i < hidden_size; ++i) {
                d_h.SetValue(0, i, err * returnHeadWeight(i, 0));
            }
        }
        MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_c(1, hidden_size);
        {
            auto low = MetaNN::LowerAccess(d_c);
            std::fill(low.MutableRawMemory(), low.MutableRawMemory() + hidden_size, 0.0f);
        }

        // Gate-wise accumulators (n_in x H) and (1 x H)
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> dW_i(param.Shape()[0], hidden_size);
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> dW_f(param.Shape()[0], hidden_size);
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> dW_g(param.Shape()[0], hidden_size);
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> dW_o(param.Shape()[0], hidden_size);
        {
            auto low=MetaNN::LowerAccess(dW_i); std::fill(low.MutableRawMemory(), low.MutableRawMemory()+param.Shape()[0]*hidden_size, AccumScalar(0));
        }
        {
            auto low=MetaNN::LowerAccess(dW_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory()+param.Shape()[0]*hidden_size, AccumScalar(0));
        }
        {
            auto low=MetaNN::LowerAccess(dW_g); std::fill(low.MutableRawMemory(), low.MutableRawMemory()+param.Shape()[0]*hidden_size, AccumScalar(0));
        }
        {
            auto low=MetaNN::LowerAccess(dW_o); std::fill(low.MutableRawMemory(), low.MutableRawMemory()+param.Shape()[0]*hidden_size, AccumScalar(0));
        }
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU> db_i(1, hidden_size), db_f(1, hidden_size), db_g(1, hidden_size), db_o(1, hidden_size);
        {
            auto low=MetaNN::LowerAccess(db_i); std::fill(low.MutableRawMemory(), low.MutableRawMemory()+hidden_size, AccumScalar(0));
        }
        {
            auto low=MetaNN::LowerAccess(db_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory()+hidden_size, AccumScalar(0));
        }
        {
            auto low=MetaNN::LowerAccess(db_g); std::fill(low.MutableRawMemory(), low.MutableRawMemory()+hidden_size, AccumScalar(0));
        }
        {
            auto low=MetaNN::LowerAccess(db_o); std::fill(low.MutableRawMemory(), low.MutableRawMemory()+hidden_size, AccumScalar(0));
        }

        // Hoist gate column blocks lazily for this window
        auto W_i_block = NNUtils::ViewCols<float, MetaNN::DeviceTags::CPU>(W_h_win, 0 * hidden_size, hidden_size);
        auto W_f_block = NNUtils::ViewCols<float, MetaNN::DeviceTags::CPU>(W_h_win, 1 * hidden_size, hidden_size);
        auto W_g_block = NNUtils::ViewCols<float, MetaNN::DeviceTags::CPU>(W_h_win, 2 * hidden_size, hidden_size);
        auto W_o_block = NNUtils::ViewCols<float, MetaNN::DeviceTags::CPU>(W_h_win, 3 * hidden_size, hidden_size);

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

            // Accumulate parameter and bias grads per gate
            {
                // Outer products: [x_t | h_{t-1}] replaced by two blocks
                // Top block from x_t
                auto dW_i_top = MetaNN::Dot(MetaNN::Transpose(sc.x), di2D);
                auto dW_f_top = MetaNN::Dot(MetaNN::Transpose(sc.x), df2D);
                auto dW_g_top = MetaNN::Dot(MetaNN::Transpose(sc.x), dg2D);
                auto dW_o_top = MetaNN::Dot(MetaNN::Transpose(sc.x), do2D);
                // Bottom block from h_prev
                auto dW_i_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), di2D);
                auto dW_f_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), df2D);
                auto dW_g_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), dg2D);
                auto dW_o_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), do2D);
                // Write into accumulators (convert to float and add)
                auto addBlock = [&](auto& dst, const auto& top, const auto& bot){
                    auto topE = MetaNN::Evaluate(top);
                    auto botE = MetaNN::Evaluate(bot);
                    // copy into dst: first input_size rows from topE, then hidden_size rows from botE
                    auto lowD = MetaNN::LowerAccess(dst);
                    auto lowT = MetaNN::LowerAccess(topE);
                    auto lowB = MetaNN::LowerAccess(botE);
                    auto* dptr = lowD.MutableRawMemory();
                    const auto* tptr = lowT.RawMemory();
                    const auto* bptr = lowB.RawMemory();
                    // dst shape: (n_in x H)
                    const size_t H = hidden_size;
                    const size_t I = param.Shape()[0] - hidden_size;
                    for (size_t r=0; r<I; ++r) {
                        for (size_t c=0; c<H; ++c) dptr[r*H + c] += static_cast<AccumScalar>(tptr[r*H + c]);
                    }
                    for (size_t r=0; r<hidden_size; ++r) {
                        for (size_t c=0; c<H; ++c) dptr[(I+r)*H + c] += static_cast<AccumScalar>(bptr[r*H + c]);
                    }
                };
                addBlock(dW_i, dW_i_top, dW_i_bot);
                addBlock(dW_f, dW_f_top, dW_f_bot);
                addBlock(dW_g, dW_g_top, dW_g_bot);
                addBlock(dW_o, dW_o_top, dW_o_bot);
                // Bias accumulators
                auto accBias = [&](auto& db, const auto& g2D){
                    auto e = MetaNN::Evaluate(g2D);
                    auto lowD = MetaNN::LowerAccess(db); auto* dptr = lowD.MutableRawMemory();
                    auto lowS = MetaNN::LowerAccess(e); const auto* sptr = lowS.RawMemory();
                    for (size_t c=0; c<hidden_size; ++c) dptr[c] += static_cast<AccumScalar>(sptr[c]);
                };
                accBias(db_i, di2D); accBias(db_f, df2D); accBias(db_g, dg2D); accBias(db_o, do2D);
            }

            // Propagate to previous hidden: compute d_h_prev via 4 block products, no concat
            auto d_h_prev = MetaNN::Dot(di2D, MetaNN::Transpose(W_i_block)) + MetaNN::Dot(df2D, MetaNN::Transpose(W_f_block)) + MetaNN::Dot(dg2D, MetaNN::Transpose(W_g_block)) + MetaNN::Dot(do2D, MetaNN::Transpose(W_o_block));
            d_h = MetaNN::Evaluate(MetaNN::Reshape(d_h_prev, MetaNN::Shape(1, hidden_size)));
            d_c = MetaNN::Evaluate(MetaNN::Reshape((dct * sc.f), MetaNN::Shape(1, hidden_size)));
        }

        // Merge gate-wise accumulators into combined d_param_accum (n_in x 4H) and d_bias_accum (1 x 4H)
        auto writeCols = [&](auto& dst, size_t colOffset, const auto& src){
            auto lowD = MetaNN::LowerAccess(dst); auto* dptr = lowD.MutableRawMemory();
            auto lowS = MetaNN::LowerAccess(src); const auto* sptr = lowS.RawMemory();
            const size_t rows = dst.Shape()[0]; const size_t dstCols = dst.Shape()[1]; const size_t H = hidden_size;
            for (size_t r=0; r<rows; ++r) {
                for (size_t c=0; c<H; ++c) {
                    dptr[r*dstCols + (colOffset + c)] += static_cast<AccumScalar>(sptr[r*H + c]);
                }
            }
        };
        writeCols(d_param_accum, 0*hidden_size, dW_i);
        writeCols(d_param_accum, 1*hidden_size, dW_f);
        writeCols(d_param_accum, 2*hidden_size, dW_g);
        writeCols(d_param_accum, 3*hidden_size, dW_o);
        auto writeBias = [&](auto& dst, size_t colOffset, const auto& src){
            auto lowD = MetaNN::LowerAccess(dst); auto* dptr = lowD.MutableRawMemory();
            auto lowS = MetaNN::LowerAccess(src); const auto* sptr = lowS.RawMemory();
            const size_t dstCols = dst.Shape()[1]; const size_t H = hidden_size;
            for (size_t c = 0; c < H; ++c) dptr[colOffset + c] += static_cast<AccumScalar>(sptr[c]);
        };
        writeBias(d_bias_accum, 0*hidden_size, db_i);
        writeBias(d_bias_accum, 1*hidden_size, db_f);
        writeBias(d_bias_accum, 2*hidden_size, db_g);
        writeBias(d_bias_accum, 3*hidden_size, db_o);
    }

    // Head gradients already accumulated in concrete matrices: d_headW_accum_f and d_headB_accum_f

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
        auto low = MetaNN::LowerAccess(d_headW_accum_f);
        const float* p = low.RawMemory();
        const size_t len = hidden_size;
        for (size_t i = 0; i < len; ++i) { double v = static_cast<double>(p[i]); sumsq += v * v; }
    }
    {
        auto low = MetaNN::LowerAccess(d_headB_accum_f);
        const float* p = low.RawMemory();
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
        d_headW_accum_f = MetaNN::Evaluate(static_cast<float>(scale) * d_headW_accum_f);
        d_headB_accum_f = MetaNN::Evaluate(static_cast<float>(scale) * d_headB_accum_f);
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

        // Update velocities lazily in AccumScalar precision
        v_param = MetaNN::Evaluate(mu * v_param + d_param_accum);
        v_bias = MetaNN::Evaluate(mu * v_bias + d_bias_accum);

        // Convert velocities to float and apply in a single lazy expression
        auto v_param_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(v_param);
        auto v_bias_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(v_bias);
        param = MetaNN::Evaluate(param - lrScale * v_param_f);
        bias = MetaNN::Evaluate(bias - lrScale * v_bias_f);

        // Update velocities lazily in AccumScalar precision for head weights
        v_headW = MetaNN::Evaluate(mu * v_headW + NNUtils::CastMatrix<AccumScalar, MetaNN::DeviceTags::CPU>(d_headW_accum_f));
        v_headB = MetaNN::Evaluate(mu * v_headB + NNUtils::CastMatrix<AccumScalar, MetaNN::DeviceTags::CPU>(d_headB_accum_f));

        // Convert velocities to float and apply in a single lazy expression
        auto v_headW_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(v_headW);
        auto v_headB_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(v_headB);
        returnHeadWeight = MetaNN::Evaluate(returnHeadWeight - lrScale * v_headW_f);
        returnHeadBias   = MetaNN::Evaluate(returnHeadBias   - lrScale * v_headB_f);
    }
#elif LSTM_OPTIMIZER == LSTM_OPTIMIZER_ADAM
    {
        const AccumScalar beta1 = static_cast<AccumScalar>(LSTM_ADAM_BETA1);
        const AccumScalar beta2 = static_cast<AccumScalar>(LSTM_ADAM_BETA2);
        const AccumScalar eps = static_cast<AccumScalar>(LSTM_ADAM_EPS);

        // Convert accumulated gradients to AccumScalar and update first/second moments lazily for LSTM core
        auto g_param_acc = d_param_accum;
        auto g_bias_acc = d_bias_accum;
        m_param = MetaNN::Evaluate(beta1 * m_param + (static_cast<AccumScalar>(1) - beta1) * g_param_acc);
        s_param = MetaNN::Evaluate(beta2 * s_param + (static_cast<AccumScalar>(1) - beta2) * (g_param_acc * g_param_acc));
        m_bias = MetaNN::Evaluate(beta1 * m_bias + (static_cast<AccumScalar>(1) - beta1) * g_bias_acc);
        s_bias = MetaNN::Evaluate(beta2 * s_bias + (static_cast<AccumScalar>(1) - beta2) * (g_bias_acc * g_bias_acc));

        // Bias correction lazily for LSTM core
        auto mhat_param = m_param / (static_cast<AccumScalar>(1) - beta1_pow);
        auto vhat_param = s_param / (static_cast<AccumScalar>(1) - beta2_pow);
        auto mhat_bias = m_bias / (static_cast<AccumScalar>(1) - beta1_pow);
        auto vhat_bias = s_bias / (static_cast<AccumScalar>(1) - beta2_pow);

        // Convert to float to match parameter types and compute updates lazily for LSTM core
        auto mhat_param_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhat_param);
        auto vhat_param_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhat_param);
        auto mhat_bias_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhat_bias);
        auto vhat_bias_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhat_bias);

        auto upd_param_f = lrScale * mhat_param_f / (MetaNN::Sqrt(vhat_param_f) + static_cast<float>(eps));
        auto upd_bias_f = lrScale * mhat_bias_f / (MetaNN::Sqrt(vhat_bias_f) + static_cast<float>(eps));

        // Apply updates lazily for LSTM core
        param = MetaNN::Evaluate(param - upd_param_f);
        bias = MetaNN::Evaluate(bias - upd_bias_f);

        // Convert head gradients to AccumScalar and update first/second moments lazily
        auto gW_acc = NNUtils::CastMatrix<AccumScalar, MetaNN::DeviceTags::CPU>(d_headW_accum_f);
        auto gB_acc = NNUtils::CastMatrix<AccumScalar, MetaNN::DeviceTags::CPU>(d_headB_accum_f);
        m_headW = MetaNN::Evaluate(beta1 * m_headW + (static_cast<AccumScalar>(1) - beta1) * gW_acc);
        s_headW = MetaNN::Evaluate(beta2 * s_headW + (static_cast<AccumScalar>(1) - beta2) * (gW_acc * gW_acc));
        m_headB = MetaNN::Evaluate(beta1 * m_headB + (static_cast<AccumScalar>(1) - beta1) * gB_acc);
        s_headB = MetaNN::Evaluate(beta2 * s_headB + (static_cast<AccumScalar>(1) - beta2) * (gB_acc * gB_acc));

        // Bias correction lazily
        auto mhatW = m_headW / (static_cast<AccumScalar>(1) - beta1_pow);
        auto vhatW = s_headW / (static_cast<AccumScalar>(1) - beta2_pow);
        auto mhatB = m_headB / (static_cast<AccumScalar>(1) - beta1_pow);
        auto vhatB = s_headB / (static_cast<AccumScalar>(1) - beta2_pow);

        // Convert to float to match parameter types and compute updates lazily
        auto mhatW_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhatW);
        auto vhatW_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhatW);
        auto mhatB_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhatB);
        auto vhatB_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhatB);
        auto updW = lrScale * mhatW_f / (MetaNN::Sqrt(vhatW_f) + static_cast<float>(eps));
        auto updB = lrScale * mhatB_f / (MetaNN::Sqrt(vhatB_f) + static_cast<float>(eps));

        // Apply in a single lazy expression
        returnHeadWeight = MetaNN::Evaluate(returnHeadWeight - updW);
        returnHeadBias   = MetaNN::Evaluate(returnHeadBias   - updB);
    }
#else
    {
        // Convert accumulated gradients to float
        auto d_param_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(d_param_accum);
        auto d_bias_f  = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(d_bias_accum);

        // Apply in a single lazy expression
        param = MetaNN::Evaluate(param - lrScale * d_param_f);
        bias  = MetaNN::Evaluate(bias  - lrScale * d_bias_f);
    }
#endif

    // Apply accumulated head gradients once per batch
#if LSTM_OPTIMIZER == LSTM_OPTIMIZER_MOMENTUM
    {
        const AccumScalar mu = static_cast<AccumScalar>(LSTM_MOMENTUM);

        // Update velocities lazily in AccumScalar precision
        v_headW = MetaNN::Evaluate(mu * v_headW + NNUtils::CastMatrix<AccumScalar, MetaNN::DeviceTags::CPU>(d_headW_accum_f));
        v_headB = MetaNN::Evaluate(mu * v_headB + NNUtils::CastMatrix<AccumScalar, MetaNN::DeviceTags::CPU>(d_headB_accum_f));

        // Convert velocities to float and apply in a single lazy expression
        auto v_headW_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(v_headW);
        auto v_headB_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(v_headB);
        returnHeadWeight = MetaNN::Evaluate(returnHeadWeight - lrScale * v_headW_f);
        returnHeadBias   = MetaNN::Evaluate(returnHeadBias   - lrScale * v_headB_f);
    }
#elif LSTM_OPTIMIZER == LSTM_OPTIMIZER_ADAM
    {
        const AccumScalar beta1 = static_cast<AccumScalar>(LSTM_ADAM_BETA1);
        const AccumScalar beta2 = static_cast<AccumScalar>(LSTM_ADAM_BETA2);
        const AccumScalar eps = static_cast<AccumScalar>(LSTM_ADAM_EPS);

        // Convert head gradients to AccumScalar and update first/second moments lazily
        auto gW_acc = NNUtils::CastMatrix<AccumScalar, MetaNN::DeviceTags::CPU>(d_headW_accum_f);
        auto gB_acc = NNUtils::CastMatrix<AccumScalar, MetaNN::DeviceTags::CPU>(d_headB_accum_f);
        m_headW = MetaNN::Evaluate(beta1 * m_headW + (static_cast<AccumScalar>(1) - beta1) * gW_acc);
        s_headW = MetaNN::Evaluate(beta2 * s_headW + (static_cast<AccumScalar>(1) - beta2) * (gW_acc * gW_acc));
        m_headB = MetaNN::Evaluate(beta1 * m_headB + (static_cast<AccumScalar>(1) - beta1) * gB_acc);
        s_headB = MetaNN::Evaluate(beta2 * s_headB + (static_cast<AccumScalar>(1) - beta2) * (gB_acc * gB_acc));

        // Bias correction lazily
        auto mhatW = m_headW / (static_cast<AccumScalar>(1) - beta1_pow);
        auto vhatW = s_headW / (static_cast<AccumScalar>(1) - beta2_pow);
        auto mhatB = m_headB / (static_cast<AccumScalar>(1) - beta1_pow);
        auto vhatB = s_headB / (static_cast<AccumScalar>(1) - beta2_pow);

        // Convert to float to match parameter types and compute updates lazily
        auto mhatW_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhatW);
        auto vhatW_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhatW);
        auto mhatB_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhatB);
        auto vhatB_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhatB);
        auto updW = lrScale * mhatW_f / (MetaNN::Sqrt(vhatW_f) + static_cast<float>(eps));
        auto updB = lrScale * mhatB_f / (MetaNN::Sqrt(vhatB_f) + static_cast<float>(eps));

        // Apply in a single lazy expression
        returnHeadWeight = MetaNN::Evaluate(returnHeadWeight - updW);
        returnHeadBias   = MetaNN::Evaluate(returnHeadBias   - updB);
    }
#else
    {
        // Fully lazy SGD update for head
        returnHeadWeight = MetaNN::Evaluate(returnHeadWeight - lrScale * d_headW_accum_f);
        returnHeadBias   = MetaNN::Evaluate(returnHeadBias   - lrScale * d_headB_accum_f);
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


