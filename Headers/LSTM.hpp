//
//  LSTM.hpp
//  LSTM
//
//  Created by Vincent Predoehl on 1/18/26.
//  Copyright © 2026 Vincent Predoehl. All rights reserved.
//

#ifndef LSTM_hpp
#define LSTM_hpp

// LSTM.hpp
#ifndef LSTM_TRAINING_ASSERTS
#define LSTM_TRAINING_ASSERTS 1
#endif

#if LSTM_TRAINING_ASSERTS
#include <cstdlib>
#include <iostream>

#define LSTM_ASSERT(cond, msg) do {                                      \
    if (!(cond)) {                                                       \
        std::cerr << "[LSTM_ASSERT] " << __FILE__ << ":" << __LINE__     \
                  << ": " << (msg) << std::endl;                         \
        std::abort(); /* or __builtin_trap(); */                         \
    }                                                                    \
} while (0)
#else
#define LSTM_ASSERT(cond, msg) do {} while(0)
#endif
#include <MetaNN/meta_nn.h>
#include <array>
#include <vector>
#include <tuple>

#include "Params.hpp"

using std::array;

class Tensor;
namespace EA
{
    class LSTM
{
    using EAMatrix = MetaNN::Matrix<float, MetaNN::DeviceTags::Metal>;
    const ::Tensor& t;
    
    struct GateMatrixView
    {
        EAMatrix& m;
        size_t colOffset;
        inline size_t rows() const { return static_cast<size_t>(n_out); }
        inline size_t cols() const { return static_cast<size_t>(n_in); }
        inline MetaNN::Shape<2> Shape() const { return MetaNN::Shape<2>(rows(), cols()); }
        inline float operator()(size_t r, size_t c) const { return m(c, colOffset + r); }
        inline void SetValue(size_t r, size_t c, float v) { m.SetValue(c, colOffset + r, v); }
    };
    
    struct ConstGateMatrixView
    {
        const EAMatrix& m;
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
    float PredictLogReturnFromH(const EAMatrix& h)
    {
        // y = h^T W + b
        // MetaNN expressions are lazy; evaluate and extract scalar
        auto expr = Dot(h, returnHeadWeight) + returnHeadBias; // 1x1 tensor
        auto yMat = Evaluate(expr);
        LSTM_ASSERT(yMat.Shape()[0] == 1 && yMat.Shape()[1] == 1, "PredictLogReturnFromH: expected 1x1 result");
        return yMat(0, 0);
    }
    float PredictDirLogitFromH(const EAMatrix& h)
    {
        auto z = Dot(h, returnHeadDirWeight) + returnHeadDirBias;   // logit
        auto zMat = Evaluate(z);
        LSTM_ASSERT(zMat.Shape()[0] == 1 && zMat.Shape()[1] == 1, "PredictLogReturnFromH: expected 1x1 result");
        return zMat(0,0);
    }
    float DirLossAndGrad(float z, int t01, float& dL_dz)
    {
        // stable softplus
        auto softplus = [](float x)
        {
            if (x > 20.f) return x;         // avoid overflow
            if (x < -20.f) return std::exp(x);
            return std::log1p(std::exp(x));
        };
        
        float sp = softplus(z);
        float L  = sp - float(t01) * z;
        float p  = 1.0f / (1.0f + std::exp(-z));
        dL_dz    = p - float(t01);
        return L;
    }
    
    static float Sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }
public:
    inline GateMatrixView gateMatrix(size_t gateIndex)
    {
        LSTM_ASSERT(gateIndex < 4, "gateMatrix: gateIndex must be < 4");
        return GateMatrixView{ param, gateIndex * static_cast<size_t>(n_out) };
    }
    inline ConstGateMatrixView gateMatrix(size_t gateIndex) const
    {
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
    enum class TargetType : int { LogReturn = 0, PercentReturn = 1, RelativeMove = PercentReturn, BinaryReturn };
    
    // How the head's scalar output maps to the target used for training/inference
    // y_hat approximates (optionally normalized) of:  t = raw * targetScale + targetBias
    // where raw is either log-return or percent-return depending on targetType
    // If targetUseZScore == true, training target was normalized as (t - targetMean)/targetStd
    TargetType targetType = TargetType::BinaryReturn; // default to logreturn
    float      targetScale = 1.0f;                   // default to 100x pct
    float      targetBias  = 0.0f;                     // default no bias
    bool       targetUseZScore = false;                // default: not normalized
    float      targetMean = 0.0f;                      // z-score mean (if used)
    float      targetStd  = 1.0f;                      // z-score std  (if used)
    // Feature scaling factor applied in Tensor::Add (e.g., log-return * 1000)
    inline static constexpr float kFeatScale = 1000.0f;
    
    float long_term, short_term, in;
    EAMatrix param { static_cast<size_t>(n_in), 4 * n_out }; // Combined gate weights matrix with shape [(n_in + hidden_size) x 4*n_out]
    EAMatrix prevHiddenState { 1, hidden_size }, prevCellState { 1, hidden_size };
    EAMatrix bias { 1, 4 * n_out };
    
    // Output head for next-step return regression: y_hat = h_T · returnHeadWeight + returnHeadBias
    EAMatrix returnHeadWeight { hidden_size, 1 };
    EAMatrix returnHeadBias { 1, 1 };
    // Binary classification head (direction): p = sigmoid(h_T · returnHeadDirWeight + returnHeadDirBias)
    EAMatrix returnHeadDirWeight { hidden_size, 1 };
    EAMatrix returnHeadDirBias { 1, 1 };
    
    // Simple SGD learning rate for head-only training
    float learningRate = 1e-4f;
    
    LSTM(const ::Tensor&, float initial_long_term = 1, float initial_short_term = 0);
    LSTM() = delete;
    
    void SetLearningRate(float lr) { learningRate = lr; }
    
    std::tuple<float, size_t, size_t> CalculateBatch(const Window);
    // Inference-only helpers (forward pass, no training)
    float PredictNextReturn(const Window& w, bool resetState = true);
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
    struct BatchStepCache;
    struct WindowBatch;
    
    WindowWeights hoistWindowWeights() const;
    StepCache forwardStep(const EAMatrix& x_t, const WindowWeights& ww, const EAMatrix& bias, EAMatrix& prevHiddenState, EAMatrix& prevCellState, EAMatrix& xh_concat) const;
    BatchStepCache forwardStepBatch(const EAMatrix& x_t,
                                    const WindowWeights& ww,
                                    const EAMatrix& bias,
                                    EAMatrix& prevHiddenState,
                                    EAMatrix& prevCellState,
                                    EAMatrix& xh_concat) const;
    HeadLoss predictAndLoss(const EAMatrix& h_T, const EAMatrix& W, const EAMatrix& b, float target) const;
    float predictOnly(const EAMatrix& h_T, const EAMatrix& W, const EAMatrix& b) const;
    void accumulateHeadGrads(EAMatrix& dW_accum, EAMatrix& dB_accum, const EAMatrix& h_T, float err) const;
    GateBlocks hoistGateBlocks(const EAMatrix& W_h_win, size_t H) const;
    void zeroGateAccumulators(GateAccumulators& A, size_t rows, size_t H) const;
    void backwardStep(const StepCache& sc, const GateBlocks& gb, EAMatrix& d_h, EAMatrix& d_c, GateAccumulators& A) const;
    inline void backwardStepBatch(const BatchStepCache& sc,const GateBlocks& gb,EAMatrix& d_h,EAMatrix& d_c,GateAccumulators& A) const;
    void mergeGateAccumulators(const GateAccumulators& A, MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>& d_param_accum, MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>& d_bias_accum, size_t H) const;
    
    auto BuildHeadDhBatch(const std::vector<float>& errs,const EAMatrix& headW,float scale) const -> EAMatrix;
    void AccumulateHeadGradsBatch(EAMatrix& dW_accum,EAMatrix& dB_accum,const EAMatrix& h_batch,const std::vector<float>& errs) const;
    
    static EAMatrix GatherRows(const std::vector<EAMatrix>& rows);
    static EAMatrix SliceRows(const EAMatrix& src, size_t row0, size_t rowCount);
    static void ScatterRows(EAMatrix& dst, const EAMatrix& src, size_t row0);
    static auto RepeatRows(const EAMatrix& row, size_t B) -> EAMatrix;
};
}

#endif /* LSTM_hpp */


