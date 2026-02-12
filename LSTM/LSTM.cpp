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
    auto d_headW_acc_expr = MetaNN::Evaluate(returnHeadWeight * 0.0f); // (H x 1)
    auto d_headB_acc_expr = MetaNN::Evaluate(returnHeadBias * 0.0f);   // (1 x 1)

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

            // Replace manual concatenation with NNUtils::ConcatCols
            auto InputAndPrevHidden = NNUtils::ConcatCols<float, MetaNN::DeviceTags::CPU>({ f_sample, prevHiddenState });
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

        // Lazy head gradient accumulation across windows (expressions)
        d_headW_acc_expr = MetaNN::Evaluate(d_headW_acc_expr + MetaNN::Transpose(prevHiddenState) * err); // (H x 1)
        d_headB_acc_expr = MetaNN::Evaluate(d_headB_acc_expr + (returnHeadBias * 0.0f + err));            // (1 x 1)

        auto d_h = MetaNN::Evaluate(err * MetaNN::Transpose(headW_snapshot));
        auto d_c = MetaNN::Evaluate(prevHiddenState * 0.0f); // zeros like h

        // Lazy per-window accumulators for LSTM core gradients (expressions)
        MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_param_expr_acc(n_in, 4 * hidden_size);
        {
            auto low = MetaNN::LowerAccess(d_param_expr_acc);
            std::fill(low.MutableRawMemory(), low.MutableRawMemory() + n_in * 4 * hidden_size, 0.0f);
        }
        MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_bias_expr_acc(1, 4 * hidden_size);
        {
            auto low = MetaNN::LowerAccess(d_bias_expr_acc);
            std::fill(low.MutableRawMemory(), low.MutableRawMemory() + 1 * 4 * hidden_size, 0.0f);
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

            // Evaluate reshaped expressions to concrete matrices so we can take raw pointers
            auto di2Dm = MetaNN::Evaluate(di2D);
            auto df2Dm = MetaNN::Evaluate(df2D);
            auto dg2Dm = MetaNN::Evaluate(dg2D);
            auto do2Dm = MetaNN::Evaluate(do2D);

            // Replace manual packing with NNUtils::ConcatCols
            auto d_pre = NNUtils::ConcatCols<float, MetaNN::DeviceTags::CPU>({ di2Dm, df2Dm, dg2Dm, do2Dm });

            // Lazy accumulation into per-window gradient expressions
            {
                auto bias_sum = MetaNN::Evaluate(d_bias_expr_acc + d_pre);
                d_bias_expr_acc = bias_sum;
            }
            auto dW_t = MetaNN::Dot(MetaNN::Transpose(sc.concat), d_pre); // (n_in x 4H)
            {
                auto param_sum = MetaNN::Evaluate(d_param_expr_acc + dW_t);
                d_param_expr_acc = param_sum;
            }

            // Propagate to previous hidden: use the part of W corresponding to h_prev (last hidden_size rows of concat)
            // Build W_h as a concrete (hidden_size x 4*hidden_size) block copied from the bottom of param
            auto W_h = NNUtils::ViewBottomRows<float, MetaNN::DeviceTags::CPU>(param, hidden_size);

            // Now compute d_h_prev = d_pre (1 x 4H) * W_h^T (4H x H) -> (1 x H)
            auto d_h_prev = MetaNN::Dot(d_pre, MetaNN::Transpose(W_h));
            d_h = MetaNN::Evaluate(MetaNN::Reshape(d_h_prev, MetaNN::Shape(1, hidden_size)));
            // Propagate to previous cell
            d_c = MetaNN::Evaluate(MetaNN::Reshape((dct * sc.f), MetaNN::Shape(1, hidden_size)));
        }

        // Evaluate per-window gradient accumulators once
        auto d_param_expr = d_param_expr_acc;
        auto d_bias_expr  = d_bias_expr_acc;

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

    // Evaluate lazy head accumulators once per batch
    auto d_headW_accum_f = MetaNN::Evaluate(d_headW_acc_expr); // (H x 1), float
    auto d_headB_accum_f = MetaNN::Evaluate(d_headB_acc_expr); // (1 x 1), float

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
        auto mhat_param = MetaNN::Evaluate(m_param / (static_cast<AccumScalar>(1) - beta1_pow));
        auto vhat_param = MetaNN::Evaluate(s_param / (static_cast<AccumScalar>(1) - beta2_pow));
        auto mhat_bias = MetaNN::Evaluate(m_bias / (static_cast<AccumScalar>(1) - beta1_pow));
        auto vhat_bias = MetaNN::Evaluate(s_bias / (static_cast<AccumScalar>(1) - beta2_pow));

        // Convert to float to match parameter types and compute updates lazily for LSTM core
        auto mhat_param_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhat_param);
        auto vhat_param_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhat_param);
        auto mhat_bias_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhat_bias);
        auto vhat_bias_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhat_bias);

        auto upd_param_f = MetaNN::Evaluate(lrScale * mhat_param_f / (MetaNN::Sqrt(vhat_param_f) + static_cast<float>(eps)));
        auto upd_bias_f = MetaNN::Evaluate(lrScale * mhat_bias_f / (MetaNN::Sqrt(vhat_bias_f) + static_cast<float>(eps)));

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
        auto mhatW = MetaNN::Evaluate(m_headW / (static_cast<AccumScalar>(1) - beta1_pow));
        auto vhatW = MetaNN::Evaluate(s_headW / (static_cast<AccumScalar>(1) - beta2_pow));
        auto mhatB = MetaNN::Evaluate(m_headB / (static_cast<AccumScalar>(1) - beta1_pow));
        auto vhatB = MetaNN::Evaluate(s_headB / (static_cast<AccumScalar>(1) - beta2_pow));

        // Convert to float to match parameter types and compute updates lazily
        auto mhatW_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhatW);
        auto vhatW_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhatW);
        auto mhatB_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhatB);
        auto vhatB_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhatB);
        auto updW = MetaNN::Evaluate(lrScale * mhatW_f / (MetaNN::Sqrt(vhatW_f) + static_cast<float>(eps)));
        auto updB = MetaNN::Evaluate(lrScale * mhatB_f / (MetaNN::Sqrt(vhatB_f) + static_cast<float>(eps)));

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
        auto mhatW = MetaNN::Evaluate(m_headW / (static_cast<AccumScalar>(1) - beta1_pow));
        auto vhatW = MetaNN::Evaluate(s_headW / (static_cast<AccumScalar>(1) - beta2_pow));
        auto mhatB = MetaNN::Evaluate(m_headB / (static_cast<AccumScalar>(1) - beta1_pow));
        auto vhatB = MetaNN::Evaluate(s_headB / (static_cast<AccumScalar>(1) - beta2_pow));

        // Convert to float to match parameter types and compute updates lazily
        auto mhatW_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhatW);
        auto vhatW_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhatW);
        auto mhatB_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(mhatB);
        auto vhatB_f = NNUtils::CastMatrix<float, MetaNN::DeviceTags::CPU>(vhatB);
        auto updW = MetaNN::Evaluate(lrScale * mhatW_f / (MetaNN::Sqrt(vhatW_f) + static_cast<float>(eps)));
        auto updB = MetaNN::Evaluate(lrScale * mhatB_f / (MetaNN::Sqrt(vhatB_f) + static_cast<float>(eps)));

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


