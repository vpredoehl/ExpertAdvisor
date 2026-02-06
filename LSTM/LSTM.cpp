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

        // Gradient into head: y_hat = h_T * W_y + b_y
        // dL/dW_y = h_T^T * err, dL/db_y = err, dL/dh_T = err * W_y^T
        auto last_hidden_state_eval = MetaNN::Evaluate(prevHiddenState);
        // Keep existing head update (SGD)
        for (size_t i = 0; i < hidden_size; ++i)
        {
            const float grad_w = last_hidden_state_eval(0, i) * err;
            const float cur  = returnHeadWeight(i, 0);
            returnHeadWeight.SetValue(i, 0, cur - learningRate * grad_w);
        }
        const float bcur = returnHeadBias(0, 0);
        returnHeadBias.SetValue(0, 0, bcur - learningRate * err);

        // Backprop signal into h_T
        float d_h_T_vec_capacity = static_cast<float>(hidden_size);
        (void)d_h_T_vec_capacity; // silence unused warning
        MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_h(1, hidden_size);
        {
            auto low = MetaNN::LowerAccess(d_h);
            float* mem = low.MutableRawMemory();
            for (size_t i = 0; i < hidden_size; ++i)
            {
                mem[i] = err * returnHeadWeight(i, 0);
            }
        }
        // Initialize d_c_T = 0
        MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_c(1, hidden_size);
        {
            auto low = MetaNN::LowerAccess(d_c);
            std::fill(low.MutableRawMemory(), low.MutableRawMemory() + hidden_size, 0.0f);
        }

        // To run BPTT we need the cached per-timestep forward values. Re-run the window to cache them.
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

        // Recompute forward pass for caching
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
                auto yExpr2 = MetaNN::Dot(concat,  param) + bias;
                const size_t gateWidth2 = yExpr2.Shape()[1] / 4;
                auto gates2D_2 = MetaNN::Reshape(yExpr2, MetaNN::Shape(4, gateWidth2));
                auto i_input2 = MetaNN::Sigmoid(gates2D_2[0]);
                auto f_forget2 = MetaNN::Sigmoid(gates2D_2[1]);
                auto g_cell_candidate2 = MetaNN::Tanh(gates2D_2[2]);
                auto o_output2 = MetaNN::Sigmoid(gates2D_2[3]);

                auto previousCellState1D_2 = MetaNN::Reshape(prevCellState, MetaNN::Shape(gateWidth2));
                auto cellStateExpr2 = f_forget2 * previousCellState1D_2 + i_input2 * g_cell_candidate2;
                auto hiddenStateExpr2 = o_output2 * MetaNN::Tanh(cellStateExpr2);
                auto cellState2DExpr2 = MetaNN::Reshape(cellStateExpr2, MetaNN::Shape(1, gateWidth2));
                auto hiddenState2DExpr2 = MetaNN::Reshape(hiddenStateExpr2, MetaNN::Shape(1, gateWidth2));
                auto cHandle = cellState2DExpr2.EvalRegister();
                auto hHandle = hiddenState2DExpr2.EvalRegister();
                auto iHandle = i_input2.EvalRegister();
                auto fHandle = f_forget2.EvalRegister();
                auto gHandle = g_cell_candidate2.EvalRegister();
                auto oHandle = o_output2.EvalRegister();
                MetaNN::EvalPlan::Inst().Eval();

                StepCache sc;
                sc.x = MetaNN::Evaluate(f_sample2); // store x_t (1,input_size)
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

        // Gradients for parameters over the window
        MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_param(n_in, 4 * hidden_size);
        MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_bias(1, 4 * hidden_size);
        {
            auto lowW = MetaNN::LowerAccess(d_param);
            std::fill(lowW.MutableRawMemory(), lowW.MutableRawMemory() + n_in * 4 * hidden_size, 0.0f);
            auto lowB = MetaNN::LowerAccess(d_bias);
            std::fill(lowB.MutableRawMemory(), lowB.MutableRawMemory() + 4 * hidden_size, 0.0f);
        }

        // Backward through time
        for (int tstep = static_cast<int>(cache.size()) - 1; tstep >= 0; --tstep)
        {
            const auto& sc = cache[static_cast<size_t>(tstep)];

            // h_t = o_t * tanh(c_t)
            // dL/dc_t accumulates from two paths: from h_t and from future c_{t+1}
            // From h_t path: dL/dc_t += d_h * o_t * (1 - tanh(c_t)^2)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> tanh_c(1, hidden_size);
            {
                auto low = MetaNN::LowerAccess(tanh_c);
                auto lowc = MetaNN::LowerAccess(sc.c);
                float* m = low.MutableRawMemory();
                const float* cm = lowc.RawMemory();
                for (size_t j = 0; j < hidden_size; ++j)
                {
                    float th = std::tanh(cm[j]);
                    m[j] = th;
                }
            }

            // dL/do = d_h * tanh(c_t) * o_t * (1 - o_t)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_o(1, hidden_size);
            // dL/dc from h path
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_c_from_h(1, hidden_size);
            {
                auto low_do = MetaNN::LowerAccess(d_o);
                auto low_dc_h = MetaNN::LowerAccess(d_c_from_h);
                auto low_dh = MetaNN::LowerAccess(d_h);
                auto low_o = MetaNN::LowerAccess(sc.o);
                auto low_tc = MetaNN::LowerAccess(tanh_c);
                float* p_do = low_do.MutableRawMemory();
                float* p_dc_h = low_dc_h.MutableRawMemory();
                const float* p_dh = low_dh.RawMemory();
                const float* p_o = low_o.RawMemory();
                const float* p_tc = low_tc.RawMemory();
                for (size_t j = 0; j < hidden_size; ++j)
                {
                    p_do[j] = p_dh[j] * p_tc[j] * p_o[j] * (1.0f - p_o[j]);
                    p_dc_h[j] = p_dh[j] * p_o[j] * (1.0f - p_tc[j] * p_tc[j]);
                }
            }

            // c_t = f_t * c_{t-1} + i_t * g_t
            // dL/di = d_c * g; dL/dg = d_c * i; dL/df = d_c * c_{t-1}
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_i(1, hidden_size);
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_f(1, hidden_size);
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_g(1, hidden_size);
            {
                auto low_di = MetaNN::LowerAccess(d_i);
                auto low_df = MetaNN::LowerAccess(d_f);
                auto low_dg = MetaNN::LowerAccess(d_g);
                auto low_dc = MetaNN::LowerAccess(d_c);
                auto low_dc_h = MetaNN::LowerAccess(d_c_from_h);
                auto low_i = MetaNN::LowerAccess(sc.i);
                auto low_f = MetaNN::LowerAccess(sc.f);
                auto low_g = MetaNN::LowerAccess(sc.g);
                auto low_cprev = MetaNN::LowerAccess(sc.c_prev);
                float* p_di = low_di.MutableRawMemory();
                float* p_df = low_df.MutableRawMemory();
                float* p_dg = low_dg.MutableRawMemory();
                const float* p_dc = low_dc.RawMemory();
                const float* p_dc_h = low_dc_h.RawMemory();
                const float* p_i = low_i.RawMemory();
                const float* p_f = low_f.RawMemory();
                const float* p_g = low_g.RawMemory();
                const float* p_cprev = low_cprev.RawMemory();
                for (size_t j = 0; j < hidden_size; ++j)
                {
                    float dct = p_dc[j] + p_dc_h[j];
                    p_di[j] = dct * p_g[j] * p_i[j] * (1.0f - p_i[j]);
                    p_dg[j] = dct * p_i[j] * (1.0f - p_g[j] * p_g[j]);
                    p_df[j] = dct * p_cprev[j] * p_f[j] * (1.0f - p_f[j]);
                }
            }

            // Accumulate gradient wrt bias: concat of [di, df, dg, do]
            {
                auto lowB = MetaNN::LowerAccess(d_bias);
                float* pb = lowB.MutableRawMemory();
                auto low_di = MetaNN::LowerAccess(d_i);
                auto low_df = MetaNN::LowerAccess(d_f);
                auto low_dg = MetaNN::LowerAccess(d_g);
                auto low_do = MetaNN::LowerAccess(d_o);
                const float* pdi = low_di.RawMemory();
                const float* pdf = low_df.RawMemory();
                const float* pdg = low_dg.RawMemory();
                const float* pdo = low_do.RawMemory();
                for (size_t j = 0; j < hidden_size; ++j)
                {
                    pb[0 * hidden_size + j] += pdi[j];
                    pb[1 * hidden_size + j] += pdf[j];
                    pb[2 * hidden_size + j] += pdg[j];
                    pb[3 * hidden_size + j] += pdo[j];
                }
            }

            // Gradient wrt param: dW += concat^T * d_preact, where d_preact = [di, df, dg, do]
            {
                auto lowW = MetaNN::LowerAccess(d_param);
                float* pW = lowW.MutableRawMemory();
                auto lowConcat = MetaNN::LowerAccess(sc.concat);
                const float* px = lowConcat.RawMemory(); // length n_in
                auto low_di = MetaNN::LowerAccess(d_i);
                auto low_df = MetaNN::LowerAccess(d_f);
                auto low_dg = MetaNN::LowerAccess(d_g);
                auto low_do = MetaNN::LowerAccess(d_o);
                const float* pdi = low_di.RawMemory();
                const float* pdf = low_df.RawMemory();
                const float* pdg = low_dg.RawMemory();
                const float* pdo = low_do.RawMemory();
                for (size_t r = 0; r < n_in; ++r)
                {
                    for (size_t j = 0; j < hidden_size; ++j)
                    {
                        pW[r * (4 * hidden_size) + (0 * hidden_size + j)] += px[r] * pdi[j];
                        pW[r * (4 * hidden_size) + (1 * hidden_size + j)] += px[r] * pdf[j];
                        pW[r * (4 * hidden_size) + (2 * hidden_size + j)] += px[r] * pdg[j];
                        pW[r * (4 * hidden_size) + (3 * hidden_size + j)] += px[r] * pdo[j];
                    }
                }
            }

            // Propagate to previous hidden and cell for next iteration
            // dL/dc_{t-1} = (d_c + d_c_from_h) * f_t
            // dL/dh_{t-1} = d_preact * W_h (the part of param corresponding to previous hidden in concat)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_pre(1, 4 * hidden_size);
            {
                auto low = MetaNN::LowerAccess(d_pre);
                float* p = low.MutableRawMemory();
                auto low_di = MetaNN::LowerAccess(d_i);
                auto low_df = MetaNN::LowerAccess(d_f);
                auto low_dg = MetaNN::LowerAccess(d_g);
                auto low_do = MetaNN::LowerAccess(d_o);
                const float* pdi = low_di.RawMemory();
                const float* pdf = low_df.RawMemory();
                const float* pdg = low_dg.RawMemory();
                const float* pdo = low_do.RawMemory();
                for (size_t j = 0; j < hidden_size; ++j)
                {
                    p[0 * hidden_size + j] = pdi[j];
                    p[1 * hidden_size + j] = pdf[j];
                    p[2 * hidden_size + j] = pdg[j];
                    p[3 * hidden_size + j] = pdo[j];
                }
            }

            // d_h_prev = d_pre * W_h^T, where W_h are the last hidden_size rows of param (within concat)
            MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> d_h_prev(1, hidden_size);
            {
                auto low_dhp = MetaNN::LowerAccess(d_h_prev);
                float* pdhp = low_dhp.MutableRawMemory();
                std::fill(pdhp, pdhp + hidden_size, 0.0f);
                // param shape: (n_in, 4*hidden_size) where n_in = input_size + hidden_size
                // The last hidden_size entries in the concat correspond to h_{t-1}
                auto low_pre = MetaNN::LowerAccess(d_pre);
                const float* ppre = low_pre.RawMemory();
                for (size_t j = 0; j < hidden_size; ++j)
                {
                    float s = 0.0f;
                    for (size_t g = 0; g < 4; ++g)
                    {
                        size_t col_base = g * hidden_size;
                        size_t row = (n_in - hidden_size) + j;
                        for (size_t k = 0; k < hidden_size; ++k)
                        {
                            s += ppre[col_base + k] * param(row, col_base + k);
                        }
                    }
                    pdhp[j] = s;
                }
            }

            // d_c_prev
            {
                auto low_dc = MetaNN::LowerAccess(d_c);
                auto low_dc_h = MetaNN::LowerAccess(d_c_from_h);
                auto low_f = MetaNN::LowerAccess(sc.f);
                float* pdc = low_dc.MutableRawMemory();
                const float* pdc_h = low_dc_h.RawMemory();
                const float* pf = low_f.RawMemory();
                for (size_t j = 0; j < hidden_size; ++j)
                {
                    pdc[j] = (pdc[j] + pdc_h[j]) * pf[j];
                }
            }

            // Set d_h for next step
            d_h = d_h_prev;
        }

        // Apply SGD to param and bias
        {
            // param = param - lr * d_param
            for (size_t r = 0; r < n_in; ++r)
            {
                for (size_t c = 0; c < 4 * hidden_size; ++c)
                {
                    float cur = param(r, c);
                    float grad = d_param(r, c);
                    param.SetValue(r, c, cur - learningRate * grad);
                }
            }
            // bias update
            for (size_t j = 0; j < 4 * hidden_size; ++j)
            {
                float cur = bias(0, j);
                float grad = d_bias(0, j);
                bias.SetValue(0, j, cur - learningRate * grad);
            }
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


