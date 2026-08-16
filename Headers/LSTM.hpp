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
#include <chrono>
#include <optional>
#include "FeatureAblation.hpp"
#include <string>
#include <vector>
#include <tuple>

#include "Params.hpp"

using std::array;

class Tensor;
namespace EA
{
class LSTM
{
    const ::Tensor& t;
    int n_in = 0;

    static inline int ClassFromLogReturn(float r, float threshold)
    {
        if (!std::isfinite(r)) return 1; // neutral
        if (r > threshold)  return 2;    // up
        if (r < -threshold) return 0;    // down
        return 1;                        // neutral
    }

    static inline void Softmax3(const float* z, float* p)
    {
        const float m = std::max(z[0], std::max(z[1], z[2]));
        const float e0 = std::exp(z[0] - m);
        const float e1 = std::exp(z[1] - m);
        const float e2 = std::exp(z[2] - m);
        const float s = e0 + e1 + e2;
        p[0] = e0 / s;
        p[1] = e1 / s;
        p[2] = e2 / s;
    }
    
    static float Sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }
    inline void ApplyForgetGateBiasOffset(float offset)
    {
        // Gate column order is i, f, g, o.  The forget gate occupies
        // bias columns [n_out, 2*n_out).  Use this immediately after the
        // bias matrix is zero-filled or otherwise initialized.
        LSTM_ASSERT(bias.Shape()[0] == 1 && bias.Shape()[1] == 4 * n_out,
                    "ApplyForgetGateBiasOffset: bias shape mismatch");
        for (size_t j = 0; j < static_cast<size_t>(n_out); ++j)
        {
            const size_t forgetCol = static_cast<size_t>(n_out) + j;
            bias.SetValue(0, forgetCol, bias(0, forgetCol) + offset);
        }
    }
public:
    using EAMatrix = MetaNN::Matrix<float, MetaNN::DeviceTags::Metal>;

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
    enum class TargetType : int {
        LogReturn = 0,
        PercentReturn = 1,
        RelativeMove = PercentReturn,
        UpNeutralDownReturn = 2
    };

    static inline float CoreLrMultForTarget(TargetType targetType)
    {
        return (targetType == TargetType::UpNeutralDownReturn) ? ::core_lr_mult : 1.0f;
    }
    
    // How the head's scalar output maps to the target used for training/inference
    // y_hat approximates (optionally normalized) of:  t = raw * targetScale + targetBias
    // where raw is either log-return or percent-return depending on targetType
    // If targetUseZScore == true, training target was normalized as (t - targetMean)/targetStd
    TargetType targetType = TargetType::UpNeutralDownReturn; // default to up-neutral-down return
    float      targetScale = 1.0f;                   // default to 100x pct
    float      targetBias  = 0.0f;                     // default no bias
    bool       targetUseZScore = false;                // default: not normalized
    float      targetMean = 0.0f;                      // z-score mean (if used)
    float      targetStd  = 1.0f;                      // z-score std  (if used)
    // Feature scaling factor applied in Tensor::Add (e.g., log-return * 1000)
    inline static constexpr float kFeatScale = 1000.0f;
    
    float long_term, short_term, in;
    EAMatrix param;
    EAMatrix prevHiddenState { 1, hidden_size }, prevCellState { 1, hidden_size };
    EAMatrix bias { 1, 4 * n_out };
    
    // Output head for next-step return regression: y_hat = h_T · returnHeadWeight + returnHeadBias
    EAMatrix returnHeadWeight { hidden_size, 1 };
    EAMatrix returnHeadBias { 1, 1 };
    // 3-class classification head (down / neutral / up)
    EAMatrix returnHeadDirWeight { hidden_size, direction_output_size };
    EAMatrix returnHeadDirBias { 1, direction_output_size };
    
    // Simple SGD learning rate for head-only training
    float learning_rate = 1e-3f / 3; // or /2 or /4
    size_t optimizerUpdateCount = 0;
    size_t completedEpochs = 0;
    EA::FeatureAblationMask featureAblationMask;
    static bool suppressPhase3HiddenGeometryDiagnostics;
    
    LSTM(const ::Tensor&, float initial_long_term = 1, float initial_short_term = 0,
         TargetType explicitTargetType = TargetType::UpNeutralDownReturn,
         std::optional<std::size_t> modelInputWidth = std::nullopt,
         EA::FeatureAblationMask ablationMask = {});

    inline void InitializeBiasWithForgetGateOffset(float forgetBiasOffset)
    {
        // Keep the base gate biases at zero, then bias only the forget gate.
        // Gate column order is i, f, g, o; ApplyForgetGateBiasOffset targets f.
        LSTM_ASSERT(bias.Shape()[0] == 1 && bias.Shape()[1] == 4 * n_out,
                    "InitializeBiasWithForgetGateOffset: bias shape mismatch");
        for (size_t j = 0; j < static_cast<size_t>(4 * n_out); ++j) bias.SetValue(0, j, 0.0f);
        ApplyForgetGateBiasOffset(forgetBiasOffset);
    }

    void PrintOutputHeadShapes() const;
    LSTM() = delete;
    
    void SetLearningRate(float lr) { learning_rate = lr; }
    int InputFeatureCount() const { return n_in; }
    const ::Tensor* BoundTensorAddress() const { return &t; }
    
    std::tuple<float, size_t, size_t> CalculateBatch(const Window, unsigned short);
    // Inference-only helpers (forward pass, no training)
    float PredictNextReturn(const Window& w, bool resetState = true);
    float PredictNextRelativeMove(const Window& w, bool resetState = true);
    int PredictNextDirectionClass(const Window& w, bool resetState = true);
    std::array<float, direction_output_size> PredictNextDirectionProbs(const Window& w, bool resetState = true);
    
    std::vector<float> RollingPredictNextLogReturn(const Window& batch, bool resetAtStart = true);
    std::vector<float> RollingPredictNextClose(const Window& batch, bool resetAtStart = true);
    
    static void PrintAndResetEpochBuckets();
    static void ConfigureHotspotProfiler(bool enabled,
                                         std::optional<std::string> outputPath = std::nullopt);
    static bool HotspotProfilingEnabled();
    static void RecordHotspot(const char* name, double elapsedUs);
    static void PrintHotspotProfileSummary();
    static bool WriteHotspotProfileReport(const std::string& outputPath);

    class HotspotScope
    {
    public:
        explicit HotspotScope(const char* name);
        ~HotspotScope();
        HotspotScope(const HotspotScope&) = delete;
        HotspotScope& operator=(const HotspotScope&) = delete;

    private:
        const char* name_ = nullptr;
        bool enabled_ = false;
        std::chrono::steady_clock::time_point start_;
    };
private:
    using AccumScalar = float;
    
    struct WindowWeights;
    struct HeadLoss;
    struct HeadLoss3Class { float loss; EAMatrix d_logits; float p_down; float p_neutral; float p_up; int predicted_class; };
    struct GateBlocks;
    struct GateAccumulators;
    struct BatchStepCache;
    struct WindowBatch;
    struct ForwardBatchScratch;
    struct LSTMBatchProfile;
    
    struct GateMatrixView
    {
        EAMatrix& m;
        size_t colOffset;
        size_t inputCols;
        inline size_t rows() const { return static_cast<size_t>(n_out); }
        inline size_t cols() const { return inputCols; }
        inline MetaNN::Shape<2> Shape() const { return MetaNN::Shape<2>(rows(), cols()); }
        inline float operator()(size_t r, size_t c) const { return m(c, colOffset + r); }
        inline void SetValue(size_t r, size_t c, float v) { m.SetValue(c, colOffset + r, v); }
    };
    
    struct ConstGateMatrixView
    {
        const EAMatrix& m;
        size_t colOffset;
        size_t inputCols;
        inline size_t rows() const { return static_cast<size_t>(n_out); }
        inline size_t cols() const { return inputCols; }
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
    std::array<float, direction_output_size> PredictDirLogits3ClassFromH(const EAMatrix& h)
    {
        auto z = Dot(h, returnHeadDirWeight) + returnHeadDirBias;   // 1x3 logits
        auto zMat = Evaluate(z);
        LSTM_ASSERT(zMat.Shape()[0] == 1 && zMat.Shape()[1] == direction_output_size, "PredictDirLogits3ClassFromH: expected 1x3 result");
        return { zMat(0,0), zMat(0,1), zMat(0,2) };
    }
    inline GateMatrixView gateMatrix(size_t gateIndex)
    {
        LSTM_ASSERT(gateIndex < 4, "gateMatrix: gateIndex must be < 4");
        return GateMatrixView{ param, gateIndex * static_cast<size_t>(n_out), static_cast<size_t>(n_in) };
    }
    inline ConstGateMatrixView gateMatrix(size_t gateIndex) const
    {
        LSTM_ASSERT(gateIndex < 4, "gateMatrix const: gateIndex must be < 4");
        return ConstGateMatrixView{ param, gateIndex * static_cast<size_t>(n_out), static_cast<size_t>(n_in) };
    }

    WindowWeights hoistWindowWeights() const;
    void forwardStep(const EAMatrix& x_t, const WindowWeights& ww, const EAMatrix& bias, EAMatrix& prevHiddenState, EAMatrix& prevCellState, EAMatrix& xh_concat) const;
    BatchStepCache forwardStepBatch(const EAMatrix& x_t,
                                    const WindowWeights& ww,
                                    const EAMatrix& bias,
                                    EAMatrix& prevHiddenState,
                                    EAMatrix& prevCellState,
                                    EAMatrix& xh_concat,
                                    ForwardBatchScratch& scratch, LSTMBatchProfile*) const;
    void ComputeGateStateBatchFromContiguous(const EAMatrix& gates_batch,
                                                              const EAMatrix& prevCellState,
                                                              EAMatrix& gate_i_batch,
                                                              EAMatrix& gate_f_batch,
                                                              EAMatrix& gate_g_batch,
                                                              EAMatrix& gate_o_batch,
                                                              EAMatrix& c_batch,
                                                              EAMatrix& h_batch) const;
    HeadLoss predictAndLoss(const EAMatrix& h_T, const EAMatrix& W, const EAMatrix& b, float target) const;
    HeadLoss3Class predictAndLoss3Class(const EAMatrix& h_T, const EAMatrix& W, const EAMatrix& b, int targetClass) const;
    float predictOnly(const EAMatrix& h_T, const EAMatrix& W, const EAMatrix& b) const;
    void accumulateHeadGrads(EAMatrix& dW_accum, EAMatrix& dB_accum, const EAMatrix& h_T, float err) const;
    GateBlocks hoistGateBlocks(const EAMatrix& W_h_win, size_t H) const;
    void zeroGateAccumulators(GateAccumulators& A, size_t rows, size_t H) const;
    void RepeatRowsInto(EAMatrix& out, const EAMatrix& row, size_t B) const;
    auto BuildBatchAtTimestepDirect(const WindowBatch& wb, size_t tstep, LSTMBatchProfile*) -> EAMatrix;
    float ComputeLookbackLogReturn(size_t currentGlobalPosition, size_t lookbackBars) const;
    size_t AppendMultiHorizonReturnFeatures(size_t currentGlobalPosition,
                                            float* dst,
                                            size_t dstOffset) const;
    
    // Batched helpers
    EAMatrix RepeatRows(const EAMatrix& row, size_t B) const;
    EAMatrix BuildHeadDhBatch(const std::vector<float>& errs, const EAMatrix& headW, float scale) const;
    EAMatrix BuildHeadDhBatch3Class(const EAMatrix& d_logits_batch, const EAMatrix& headW, float scale) const;
    void AccumulateHeadGradsBatch(EAMatrix& dW_accum, EAMatrix& dB_accum, const EAMatrix& h_batch, const std::vector<float>& errs) const;
    void AccumulateHeadGradsBatch3Class(EAMatrix& dW_accum, EAMatrix& dB_accum, const EAMatrix& h_batch, const EAMatrix& d_logits_batch) const;
    EAMatrix SliceRows(const EAMatrix& src, size_t row0, size_t rowCount);

    // Batched backward through time
    void backwardStepBatch(const BatchStepCache& sc, const GateBlocks& gb, EAMatrix& d_h, EAMatrix& d_c, GateAccumulators& A) const;
    
    void mergeGateAccumulators(const GateAccumulators& A, MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>& d_param_accum, MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>& d_bias_accum, size_t H) const;

    static EAMatrix GatherRows(const std::vector<EAMatrix>& rows);
    static void ScatterRows(EAMatrix& dst, const EAMatrix& src, size_t row0);
    
    static void PrintMatrixSummary(const char* label,
                                          const EA::LSTM::EAMatrix& m,
                                          size_t maxPrint);
    static void PrintHeadGradNormDiag(
        size_t tag,
        const EAMatrix& gradW, const EAMatrix& gradB,
        const EAMatrix& paramW, const EAMatrix& paramB,
        float learningRateW, float learningRateB);
};
}

#endif /* LSTM_hpp */
