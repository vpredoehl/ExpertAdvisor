//
//  LSTM.hpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/18/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#ifndef LSTM_hpp
#define LSTM_hpp

#include <MetaNN/meta_nn.h>
#include <array>
#include <cassert>

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
            assert(gateIndex < 4);
            return GateMatrixView{ param, gateIndex * static_cast<size_t>(n_out) };
        }
        inline ConstGateMatrixView gateMatrix(size_t gateIndex) const {
            assert(gateIndex < 4);
            return ConstGateMatrixView{ param, gateIndex * static_cast<size_t>(n_out) };
        }
        inline void ResetPreviousState()
        {
            for (size_t j = 0; j < hidden_size; ++j)
            {
                prevHiddenState.SetValue(0, j, 0.0f);
                prevCellState.SetValue(0, j, 0.0f);
            }

        }

        float long_term, short_term, in;
        FloatMatrixCPU param { n_in, 4 * n_out }; // Combined gate weights matrix with shape [n_in x 4*n_out]
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
        
        float CalculateBatch(const std::ranges::subrange<DataSet::const_iterator>);

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
        void accumulateHeadGrads(FloatMatrixCPU& dW_accum, FloatMatrixCPU& dB_accum, const FloatMatrixCPU& h_T, float err) const;
        GateBlocks hoistGateBlocks(const FloatMatrixCPU& W_h_win, size_t H) const;
        void zeroGateAccumulators(GateAccumulators& A, size_t rows, size_t H) const;
        void backwardStep(const StepCache& sc, const GateBlocks& gb, FloatMatrixCPU& d_h, FloatMatrixCPU& d_c, GateAccumulators& A) const;
        void mergeGateAccumulators(const GateAccumulators& A, MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU>& d_param_accum, MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::CPU>& d_bias_accum, size_t H) const;
    };
}

#endif /* LSTM_hpp */


