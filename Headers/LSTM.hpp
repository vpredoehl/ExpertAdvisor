//
//  LSTM.hpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/18/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#ifndef LSTM_hpp
#define LSTM_hpp

#ifndef LSTM_TRAINING_ASSERTS
#define LSTM_TRAINING_ASSERTS 1
#endif

#if LSTM_TRAINING_ASSERTS
#include <cassert>
#include <iostream>
#define LSTM_ASSERT(cond, msg) do { if(!(cond)) { std::cerr << "[LSTM_ASSERT] " << __FILE__ << ":" << __LINE__ << ": " << (msg) << std::endl; assert(cond); } } while(0)
#else
#define LSTM_ASSERT(cond, msg) do {} while(0)
#endif

#include <MetaNN/meta_nn.h>
#include <array>
#include <vector>

#include "Params.hpp"

using std::array;

class Tensor;
namespace EA
{
    class LSTM
    {
        using FloatMatrixCPU = MetaNN::Matrix<float, MetaNN::DeviceTags::CPU>;

        struct GateMatrixView {
            FloatMatrixCPU& m;
            size_t colOffset;
            inline size_t rows() const { return static_cast<size_t>(n_out); }
            inline size_t cols() const { return static_cast<size_t>(n_in); }
            inline MetaNN::Shape<2> Shape() const { return MetaNN::Shape<2>(rows(), cols()); }
            inline float operator()(size_t r, size_t c) const { return m(c, colOffset + r); }
            inline void SetValue(size_t r, size_t c, float v) { m.SetValue(c, colOffset + r, v); }
        };

        struct ConstGateMatrixView {
            const FloatMatrixCPU& m;
            size_t colOffset;
            inline size_t rows() const { return static_cast<size_t>(n_out); }
            inline size_t cols() const { return static_cast<size_t>(n_in); }
            inline MetaNN::Shape<2> Shape() const { return MetaNN::Shape<2>(rows(), cols()); }
            inline float operator()(size_t r, size_t c) const { return m(c, colOffset + r); }
        };


        float Forget()  // calculate forget module
        {
            return 0;
        }
        
        const ::Tensor& t;
    public:
        inline GateMatrixView gateMatrix(size_t gateIndex) {
            LSTM_ASSERT(gateIndex < 4, "gateMatrix: gateIndex must be < 4");
            return GateMatrixView{ param, gateIndex * static_cast<size_t>(n_out) };
        }
        inline ConstGateMatrixView gateMatrix(size_t gateIndex) const {
            LSTM_ASSERT(gateIndex < 4, "gateMatrix const: gateIndex must be < 4");
            return ConstGateMatrixView{ param, gateIndex * static_cast<size_t>(n_out) };
        }
        inline void ResetPreviousState()
        {
            LSTM_ASSERT(prevHiddenState.Shape()[0] == 1 && prevHiddenState.Shape()[1] == hidden_size, "prevHiddenState shape mismatch");
            LSTM_ASSERT(prevCellState.Shape()[0] == 1 && prevCellState.Shape()[1] == hidden_size, "prevCellState shape mismatch");
            for (size_t j = 0; j < hidden_size; ++j)
            {
                prevHiddenState.SetValue(0, j, 0.0f);
                prevCellState.SetValue(0, j, 0.0f);
            }

        }

        // Target mapping metadata (persisted via PgModelIO)
        enum class TargetType : int { LogReturn = 0, PercentReturn = 1 };

        // How the head's scalar output maps to the target used for training/inference
        // y_hat approximates (optionally normalized) of:  t = raw * targetScale + targetBias
        // where raw is either log-return or percent-return depending on targetType
        // If targetUseZScore == true, training target was normalized as (t - targetMean)/targetStd
        TargetType targetType = TargetType::LogReturn; // default to percent return
        float      targetScale = 1.0f;                   // default to 100x pct
        float      targetBias  = 0.0f;                     // default no bias
        bool       targetUseZScore = false;                // default: not normalized
        float      targetMean = 0.0f;                      // z-score mean (if used)
        float      targetStd  = 1.0f;                      // z-score std  (if used)
        // Feature scaling factor applied in Tensor::Add (e.g., log-return * 1000)
        inline static constexpr float kFeatScale = 1000.0f;

        float long_term, short_term, in;
        FloatMatrixCPU param { static_cast<size_t>(n_in), 4 * n_out }; // Combined gate weights matrix with shape [(n_in + hidden_size) x 4*n_out]
        FloatMatrixCPU prevHiddenState { 1, hidden_size }, prevCellState { 1, hidden_size };
        FloatMatrixCPU bias { 1, 4 * n_out };

        // Output head for next-step return regression: y_hat = h_T · returnHeadWeight + returnHeadBias
        FloatMatrixCPU returnHeadWeight { hidden_size, 1 };
        FloatMatrixCPU returnHeadBias { 1, 1 };

        // Simple SGD learning rate for head-only training
        float learningRate = 1e-3f;

        LSTM(const ::Tensor&, float initial_long_term = 1, float initial_short_term = 0);
        LSTM() = delete;

        void SetLearningRate(float lr) { learningRate = lr; }
        
        float CalculateBatch(const Window);
        // Inference-only helpers (forward pass, no training)
        float PredictNextLogReturn(const Window& w, bool resetState = true);
        float PredictNextClose(const Window& w, bool resetState = true);

        std::vector<float> RollingPredictNextLogReturn(const Window& batch, bool resetAtStart = true);
        std::vector<float> RollingPredictNextClose(const Window& batch, bool resetAtStart = true);

    private:
        using AccumScalar = float;

        struct WindowWeights;
        struct StepCache;
        struct HeadLoss;
        struct GateBlocks;
        struct GateAccumulators;

        WindowWeights hoistWindowWeights() const;
        StepCache forwardStep(const FloatMatrixCPU& x_t, const WindowWeights& ww, const FloatMatrixCPU& bias, FloatMatrixCPU& prevHiddenState, FloatMatrixCPU& prevCellState, FloatMatrixCPU& xh_concat) const;
        HeadLoss predictAndLoss(const FloatMatrixCPU& h_T, const FloatMatrixCPU& W, const FloatMatrixCPU& b, float target) const;
        float predictOnly(const FloatMatrixCPU& h_T, const FloatMatrixCPU& W, const FloatMatrixCPU& b) const;
        void accumulateHeadGrads(FloatMatrixCPU& dW_accum, FloatMatrixCPU& dB_accum, const FloatMatrixCPU& h_T, float err) const;
        GateBlocks hoistGateBlocks(const FloatMatrixCPU& W_h_win, size_t H) const;
        void zeroGateAccumulators(GateAccumulators& A, size_t rows, size_t H) const;
        void backwardStep(const StepCache& sc, const GateBlocks& gb, FloatMatrixCPU& d_h, FloatMatrixCPU& d_c, GateAccumulators& A) const;
        void mergeGateAccumulators(const GateAccumulators& A, MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU>& d_param_accum, MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU>& d_bias_accum, size_t H) const;
    };
}

#endif /* LSTM_hpp */

