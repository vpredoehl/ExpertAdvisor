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
#include <limits>
#include <iostream>

#include "LSTM.hpp"
#include "Tensor.hpp"
#include "MatrixUtils.hpp"
#include "BuildConfig.hpp"

#ifndef LSTM_TRAINING_PROGRESS
#define LSTM_TRAINING_PROGRESS 1
#endif

#ifndef LSTM_SAT_DEBUG
#define LSTM_SAT_DEBUG 0
#endif

#ifndef LSTM_RESET_STATE_PER_WINDOW
#define LSTM_RESET_STATE_PER_WINDOW 1
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
#define LSTM_MIXED_PRECISION 1
#endif

#if LSTM_MIXED_PRECISION
using AccumScalar = double;  // higher-precision accumulation
#else
using AccumScalar = float;   // default accumulation precision
#endif

#ifndef LSTM_USE_GRAD_CLIP
#define LSTM_USE_GRAD_CLIP 1
#endif
#ifndef LSTM_GRAD_CLIP_THRESHOLD
#define LSTM_GRAD_CLIP_THRESHOLD 10.0f
#endif

#ifndef LSTM_OPTIMIZER_SGD
#define LSTM_OPTIMIZER_SGD 0
#define LSTM_OPTIMIZER_MOMENTUM 1
#define LSTM_OPTIMIZER_ADAM 2
#endif
#ifndef LSTM_OPTIMIZER
#define LSTM_OPTIMIZER LSTM_OPTIMIZER_ADAM
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

#ifndef LSTM_HEAD_LR_MULT
#define LSTM_HEAD_LR_MULT 15.0f
#endif
#ifndef LSTM_CORE_GRAD_SCALE
#define LSTM_CORE_GRAD_SCALE 5.0f
#endif
#ifndef LSTM_WEIGHT_DECAY
#define LSTM_WEIGHT_DECAY 0.0f
#endif

#ifndef LSTM_HEAD_WEIGHT_DECAY
#define LSTM_HEAD_WEIGHT_DECAY 1e-4f
#endif

constexpr size_t mini_batch_windows = 128;


// Random helpers: uniform real in [low, high] and symmetric [-limit, limit]
static inline float uniform_between(float low, float high) {
    thread_local std::mt19937 rng{ 42 };// std::random_device{}() };
    std::uniform_real_distribution<float> dist(low, high);
    return dist(rng);
}

static inline float uniform_symmetric(float limit) {
    return uniform_between(-limit, limit);
}

struct EA::LSTM::HeadLoss { float y_hat; float err; };
struct EA::LSTM::GateBlocks {   EA::LSTM::EAMatrix W_i, W_f, W_g, W_o;  }; // (H x H)
struct EA::LSTM::GateAccumulators
{
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal> dW_i;
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal> dW_f;
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal> dW_g;
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal> dW_o;
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal> db_i;
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal> db_f;
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal> db_g;
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal> db_o;
};

template <typename Mat>
void zeroFill(Mat& m)
{
    auto low = MetaNN::LowerAccess(m);
    using ElemT = std::remove_reference_t<decltype(*low.MutableRawMemory())>;
    std::fill(low.MutableRawMemory(), low.MutableRawMemory() + m.Shape()[0] * m.Shape()[1], static_cast<ElemT>(0));
}

template<typename MatP, typename MatG>
void SGDUpdate(MatP& P, const MatG& G, float lr)
{
    auto pAcc = MetaNN::LowerAccess(P);
    auto gAcc = MetaNN::LowerAccess(G);

    const size_t rows = P.Shape()[0];
    const size_t cols = P.Shape()[1];

    auto* p = pAcc.MutableRawMemory();
    const auto* g = gAcc.RawMemory();

    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            p[r * cols + c] -= lr * g[r * cols + c];
}

// Moving helper structs and functions into EA::LSTM scope
// Struct definitions inside EA::LSTM

struct EA::LSTM::WindowWeights
{
    EA::LSTM::EAMatrix W_cat;   // (n_in + H) x (4H)
    EA::LSTM::EAMatrix W_x_win; // top block (n_in x 4H)
    EA::LSTM::EAMatrix W_h_win; // bottom block (H x 4H)
};

struct EA::LSTM::StepCache
{
    EA::LSTM::EAMatrix x;      // (1, input_size)
    EA::LSTM::EAMatrix h_prev; // (1, hidden_size)
    EA::LSTM::EAMatrix c_prev; // (1, hidden_size)
    EA::LSTM::EAMatrix i;      // (1, hidden_size)
    EA::LSTM::EAMatrix f;      // (1, hidden_size)
    EA::LSTM::EAMatrix g;      // (1, hidden_size)
    EA::LSTM::EAMatrix o;      // (1, hidden_size)
    EA::LSTM::EAMatrix c;      // (1, hidden_size)
    EA::LSTM::EAMatrix h;      // (1, hidden_size)
};

struct EA::LSTM::BatchStepCache
{
    EA::LSTM::EAMatrix x;      // (B, input_size)
    EA::LSTM::EAMatrix h_prev; // (B, hidden_size)
    EA::LSTM::EAMatrix c_prev; // (B, hidden_size)
    EA::LSTM::EAMatrix i;      // (B, hidden_size)
    EA::LSTM::EAMatrix f;      // (B, hidden_size)
    EA::LSTM::EAMatrix g;      // (B, hidden_size)
    EA::LSTM::EAMatrix o;      // (B, hidden_size)
    EA::LSTM::EAMatrix c;      // (B, hidden_size)
    EA::LSTM::EAMatrix h;      // (B, hidden_size)
};

struct EA::LSTM::WindowBatch
{
    std::vector<Window> windows;
    std::vector<float> targets;
    std::vector<float> close_t;
    std::vector<float> close_target;
};

struct EA::LSTM::ForwardBatchScratch
{
    EA::LSTM::EAMatrix affine;     // (B, 4H)
    EA::LSTM::EAMatrix bias_batch; // (B, 4H)
    EA::LSTM::EAMatrix y;          // (B, 4H)
    EA::LSTM::EAMatrix i;          // (B, H)
    EA::LSTM::EAMatrix f;          // (B, H)
    EA::LSTM::EAMatrix g;          // (B, H)
    EA::LSTM::EAMatrix o;          // (B, H)
    EA::LSTM::EAMatrix c;          // (B, H)
    EA::LSTM::EAMatrix h;          // (B, H)

    ForwardBatchScratch() : affine(1,1), bias_batch(1,1), y(1,1), i(1,1), f(1,1), g(1,1), o(1,1), c(1,1), h(1,1)  {}
};

// Member function definitions moved to EA::LSTM

inline auto EA::LSTM::BuildHeadDhBatch(const std::vector<float>& errs,
                                       const EAMatrix& headW,
                                       float scale) const -> EAMatrix
{
    const size_t B = errs.size();
    const size_t H = headW.Shape()[0];

    EAMatrix d_h(B, H);
    auto lowDh = MetaNN::LowerAccess(d_h);
    float* dhp = lowDh.MutableRawMemory();

    auto lowW = MetaNN::LowerAccess(headW);
    const float* wp = lowW.RawMemory();

    for (size_t b = 0; b < B; ++b)
    {
        const float s = errs[b] * scale;
        for (size_t i = 0; i < H; ++i)
            dhp[b * H + i] = s * wp[i];
    }
    return d_h;
}

inline void EA::LSTM::AccumulateHeadGradsBatch(EAMatrix& dW_accum,
                                               EAMatrix& dB_accum,
                                               const EAMatrix& h_batch,
                                               const std::vector<float>& errs) const
{
    const size_t B = h_batch.Shape()[0];
    const size_t H = h_batch.Shape()[1];

    auto h_eval = MetaNN::Evaluate(h_batch);
    auto lowH = MetaNN::LowerAccess(h_eval);
    const float* hptr = lowH.RawMemory();

    auto lowW = MetaNN::LowerAccess(dW_accum);
    float* wptr = lowW.MutableRawMemory();

    auto lowB = MetaNN::LowerAccess(dB_accum);
    float* bptr = lowB.MutableRawMemory();

    for (size_t b = 0; b < B; ++b)
    {
        const float err = errs[b];
        const float* hrow = hptr + b * H;
        for (size_t i = 0; i < H; ++i)
            wptr[i] += hrow[i] * err;
        bptr[0] += err;
    }
}

inline auto EA::LSTM::hoistWindowWeights() const -> WindowWeights
{
    const auto W_x_win = NNUtils::ViewTopRows<float, MetaNN::DeviceTags::Metal>(param, param.Shape()[0] - hidden_size);
    const auto W_h_win = NNUtils::ViewBottomRows<float, MetaNN::DeviceTags::Metal>(param, hidden_size);
    const auto W_cat   = param; // full (n_in + H) x (4H) matrix; dynamic row views taken later
    return WindowWeights{ W_cat, W_x_win, W_h_win };
}

inline auto EA::LSTM::GatherRows(const std::vector<EAMatrix>& rows) -> EAMatrix
{
    if (rows.empty()) return EAMatrix(0, 0);
    const size_t B = rows.size();
    const size_t F = rows.front().Shape()[1];
    EAMatrix out(B, F);
    auto lowOut = MetaNN::LowerAccess(out);
    float* dst = lowOut.MutableRawMemory();
    for (size_t b = 0; b < B; ++b)
    {
        auto rowEval = MetaNN::Evaluate(rows[b]);
        auto lowRow = MetaNN::LowerAccess(rowEval);
        const float* src = lowRow.RawMemory();
        std::copy(src, src + F, dst + b * F);
    }
    return out;
}

inline auto EA::LSTM::RepeatRows(const EAMatrix& row, size_t B) const -> EAMatrix
{
    if (B == 0) return EAMatrix(0, row.Shape()[1]);

    const size_t cols = row.Shape()[1];
    EAMatrix out(B, cols);

    auto rowEval = MetaNN::Evaluate(row);
    auto lowRow = MetaNN::LowerAccess(rowEval);
    const float* src = lowRow.RawMemory();

    auto lowOut = MetaNN::LowerAccess(out);
    float* dst = lowOut.MutableRawMemory();

    for (size_t b = 0; b < B; ++b)
        std::copy(src, src + cols, dst + b * cols);

    return out;
}

// NOTE: SliceRows is kept only for debugging/compatibility. Do NOT use it in the training hot path.
// Prefer keeping tensors in batched (B, *) form and using GatherRows/RepeatRows/ViewRows with
// forwardStepBatch/backwardStepBatch and batched heads.
[[deprecated("Avoid SliceRows in training hot path; use batched views/ops instead")]] inline auto EA::LSTM::SliceRows(const EAMatrix& src, size_t row0, size_t rowCount) -> EAMatrix
{
#if !LSTM_INFERENCE_ONLY
    // SliceRows is poison for throughput in training hot path. Use batched (B, *) ops instead.
    // This assert helps catch accidental use during training builds.
    LSTM_ASSERT(false, "SliceRows() should not be used in training path; refactor to batched ops.");
#endif
    const size_t cols = src.Shape()[1];
    EAMatrix out(rowCount, cols);
    auto srcEval = MetaNN::Evaluate(src);
    auto lowSrc = MetaNN::LowerAccess(srcEval);
    auto lowOut = MetaNN::LowerAccess(out);
    const float* sptr = lowSrc.RawMemory();
    float* dptr = lowOut.MutableRawMemory();
    std::copy(sptr + row0 * cols, sptr + (row0 + rowCount) * cols, dptr);
    return out;
}

inline void EA::LSTM::ScatterRows(EAMatrix& dst, const EAMatrix& src, size_t row0)
{
    const size_t cols = dst.Shape()[1];
    const size_t rowCount = src.Shape()[0];
    auto srcEval = MetaNN::Evaluate(src);
    auto lowSrc = MetaNN::LowerAccess(srcEval);
    auto lowDst = MetaNN::LowerAccess(dst);
    const float* sptr = lowSrc.RawMemory();
    float* dptr = lowDst.MutableRawMemory();
    std::copy(sptr, sptr + rowCount * cols, dptr + row0 * cols);
}

inline auto EA::LSTM::forwardStepBatch(const EAMatrix& x_t,
                                       const WindowWeights& ww,
                                       const EAMatrix& bias,
                                       EAMatrix& prevHiddenState,
                                       EAMatrix& prevCellState,
                                       EAMatrix& xh_concat,
                                       ForwardBatchScratch& scratch) const -> BatchStepCache
{
    const size_t B = x_t.Shape()[0];
    const size_t H = prevHiddenState.Shape()[1];
    const size_t expectedCols = x_t.Shape()[1] + H;
    const size_t gateCols = 4 * H;

    if (xh_concat.Shape()[0] != B || xh_concat.Shape()[1] != expectedCols)
        xh_concat = EAMatrix(B, expectedCols);

    if (scratch.affine.Shape()[0] != B || scratch.affine.Shape()[1] != gateCols)
        scratch.affine = EAMatrix(B, gateCols);
    if (scratch.bias_batch.Shape()[0] != B || scratch.bias_batch.Shape()[1] != gateCols)
        scratch.bias_batch = EAMatrix(B, gateCols);
        scratch.bias_batch = RepeatRows(bias, B);
    if (scratch.y.Shape()[0] != B || scratch.y.Shape()[1] != gateCols)
        scratch.y = EAMatrix(B, gateCols);
    if (scratch.i.Shape()[0] != B || scratch.i.Shape()[1] != H)
        scratch.i = EAMatrix(B, H);
    if (scratch.f.Shape()[0] != B || scratch.f.Shape()[1] != H)
        scratch.f = EAMatrix(B, H);
    if (scratch.g.Shape()[0] != B || scratch.g.Shape()[1] != H)
        scratch.g = EAMatrix(B, H);
    if (scratch.o.Shape()[0] != B || scratch.o.Shape()[1] != H)
        scratch.o = EAMatrix(B, H);
    if (scratch.c.Shape()[0] != B || scratch.c.Shape()[1] != H)
        scratch.c = EAMatrix(B, H);
    if (scratch.h.Shape()[0] != B || scratch.h.Shape()[1] != H)
        scratch.h = EAMatrix(B, H);

    NNUtils::ConcatColsInto(xh_concat, x_t, prevHiddenState);

    const size_t K = xh_concat.Shape()[1];
    const size_t rowsW = ww.W_cat.Shape()[0];
    EAMatrix xh_used = (K == rowsW)
        ? xh_concat
        : NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(xh_concat, 0, rowsW);

    {
        auto affineExpr = MetaNN::Dot(xh_used, ww.W_cat);
        auto affineH = affineExpr.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        scratch.affine = affineH.Data();
    }

    // Ensure bias is added: y = affine + bias (broadcasted as B x 4H)
    {
        auto yExpr = scratch.affine + scratch.bias_batch;
        auto yH = yExpr.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        scratch.y = yH.Data();
    }

    auto i2D = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(scratch.y, 0 * H, H);
    auto f2D = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(scratch.y, 1 * H, H);
    auto g2D = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(scratch.y, 2 * H, H);
    auto o2D = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(scratch.y, 3 * H, H);

    {
        auto iH = MetaNN::Sigmoid(i2D).EvalRegister();
        auto fH = MetaNN::Sigmoid(f2D).EvalRegister();
        auto gH = MetaNN::Tanh(g2D).EvalRegister();
        auto oH = MetaNN::Sigmoid(o2D).EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        scratch.i = iH.Data();
        scratch.f = fH.Data();
        scratch.g = gH.Data();
        scratch.o = oH.Data();
    }

    {
        auto cExpr = scratch.f * prevCellState + scratch.i * scratch.g;
        auto cH = cExpr.EvalRegister();
        auto hExpr = scratch.o * MetaNN::Tanh(cExpr);
        auto hH = hExpr.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        scratch.c = cH.Data();
        scratch.h = hH.Data();
    }

    BatchStepCache sc{
        x_t,
        prevHiddenState,
        prevCellState,
        scratch.i,
        scratch.f,
        scratch.g,
        scratch.o,
        scratch.c,
        scratch.h
    };

    prevCellState = sc.c;
    prevHiddenState = sc.h;
    return sc;
}

inline auto EA::LSTM::forwardStep(const EAMatrix& x_t,
                           const WindowWeights& ww,
                           const EAMatrix& bias,
                           EAMatrix& prevHiddenState,
                           EAMatrix& prevCellState,
                           EAMatrix& xh_concat) const -> StepCache
{
    // Build [x_t | h_{t-1}] and compute all gates in one GEMM
#if LSTM_DEBUG_PRINTS
    printMatrix("x_t", x_t);
#endif
    {
        const size_t expectedCols = x_t.Shape()[1] + prevHiddenState.Shape()[1];
        if (xh_concat.Shape()[0] != 1 || xh_concat.Shape()[1] != expectedCols) {
            xh_concat = EAMatrix(1, expectedCols);
        }
        NNUtils::ConcatColsInto(xh_concat, x_t, prevHiddenState);
#if LSTM_DEBUG_PRINTS
        auto X = MetaNN::Evaluate(x_t);
        auto H = MetaNN::Evaluate(prevHiddenState);
        auto C = MetaNN::Evaluate(xh_concat);
        const size_t Ix = X.Shape()[1];
        const size_t Ih = H.Shape()[1];
        
        for (size_t j = 0; j < Ix; ++j)
            LSTM_ASSERT(std::fabs(C(0, j) - X(0, j)) < 1e-6f, "Concat: X mismatch");
        
        for (size_t j = 0; j < Ih; ++j)
            LSTM_ASSERT(std::fabs(C(0, Ix + j) - H(0, j)) < 1e-6f, "Concat: H mismatch");
#endif
    }


    const size_t K = xh_concat.Shape()[1];
    const size_t rowsW = ww.W_cat.Shape()[0];
    EAMatrix xh_used = (K == rowsW)
        ? xh_concat
        : NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(xh_concat, 0, rowsW);
    auto yExpr = MetaNN::Dot(xh_used, ww.W_cat) + bias;
#if LSTM_DEBUG_PRINTS
    auto yHandle = yExpr.EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();
    std::cout << "bias(0,64): " << bias(0,64) << std::endl
        << "yExpr(0,64) : " << yHandle.Data()(0,64) << std::endl
        << "yExpr(0,0) : " << yHandle.Data()(0,0) << std::endl
        << "yExpr(0,128) : " << yHandle.Data()(0,128) << std::endl
        << "yExpr(0,192) : " << yHandle.Data()(0,192) << std::endl;
#endif

    const size_t H = prevHiddenState.Shape()[1];
    auto [i2D, f2D, g2D, o2D] = NNUtils::SplitGatesRowExpr(yExpr);
#if LSTM_DEBUG_PRINTS
{
    auto yHandle = yExpr.EvalRegister();

    auto iHandle = i2D.EvalRegister();
    auto fHandle = f2D.EvalRegister();
    auto gHandle = g2D.EvalRegister();
    auto oHandle = o2D.EvalRegister();

    MetaNN::EvalPlan::Inst().Eval();

    std::cout
        << "y(0,0)="   << yHandle.Data()(0,0)   << "  i(0,0)=" << iHandle.Data()(0,0) << "\n"
        << "y(0,64)="  << yHandle.Data()(0,64)  << "  f(0,0)=" << fHandle.Data()(0,0) << "\n"
        << "y(0,128)=" << yHandle.Data()(0,128) << "  g(0,0)=" << gHandle.Data()(0,0) << "\n"
        << "y(0,192)=" << yHandle.Data()(0,192) << "  o(0,0)=" << oHandle.Data()(0,0) << "\n";
}
#endif
    auto i_1d = MetaNN::Sigmoid(MetaNN::Reshape(i2D, MetaNN::Shape(H)));
    auto f_1d = MetaNN::Sigmoid(MetaNN::Reshape(f2D, MetaNN::Shape(H)));
    auto g_1d = MetaNN::Tanh   (MetaNN::Reshape(g2D, MetaNN::Shape(H)));
    auto o_1d = MetaNN::Sigmoid(MetaNN::Reshape(o2D, MetaNN::Shape(H)));

    auto c_prev_1d = MetaNN::Reshape(prevCellState, MetaNN::Shape(H));
    auto c_1d = f_1d * c_prev_1d + i_1d * g_1d;
    auto h_1d = o_1d * MetaNN::Tanh(c_1d);



    auto cprev_2d_handle = MetaNN::Reshape(c_prev_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto i_2d_handle = MetaNN::Reshape(i_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto f_2d_handle = MetaNN::Reshape(f_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto g_2d_handle = MetaNN::Reshape(g_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto o_2d_handle = MetaNN::Reshape(o_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto c_2d_handle = MetaNN::Reshape(c_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto h_2d_handle = MetaNN::Reshape(h_1d, MetaNN::Shape(1, H)).EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();
    
#if LSTM_DEBUG_PRINTS
    std::cout << "i_2d_handle.Data()(0,0): " << i_2d_handle.Data()(0,0) << std::endl;
    std::cout << "f_2d_handle.Data()(0,0): " << f_2d_handle.Data()(0,0) << std::endl;
    std::cout << "g_2d_handle.Data()(0,0): " << g_2d_handle.Data()(0,0) << std::endl;
    std::cout << "o_2d_handle.Data()(0,0): " << o_2d_handle.Data()(0,0) << std::endl;
    std::cout << "c_2d_handle.Data()(0,0): " << c_2d_handle.Data()(0,0) << std::endl;
    std::cout << "h_2d_handle.Data()(0,0): " << h_2d_handle.Data()(0,0) << std::endl;
#endif

    StepCache sc{
        x_t,
        prevHiddenState,
        cprev_2d_handle.Data(),
        i_2d_handle.Data(),
        f_2d_handle.Data(),
        g_2d_handle.Data(),
        o_2d_handle.Data(),
        c_2d_handle.Data(),
        h_2d_handle.Data()
    };

    prevCellState   = sc.c;
    prevHiddenState = sc.h;
    return sc;
}

inline auto EA::LSTM::predictAndLoss(const EAMatrix& h_T,
                             const EAMatrix& W,
                             const EAMatrix& b,
                             float target) const -> HeadLoss
{
    auto logits = MetaNN::Dot(h_T, W) + b;
    if (targetType == TargetType::BinaryReturn)
    {
        auto prob = MetaNN::Sigmoid(logits);
        auto pH = prob.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        float p = pH.Data()(0, 0);
        float err = p - target; // BCE gradient wrt logit
        return { p, err };
    }
    else
    {
        auto predH = logits.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        float y_hat = predH.Data()(0, 0);
        float err   = y_hat - target;
        return { y_hat, err };
    }
}

float EA::LSTM::predictOnly(const EAMatrix& h_T,
                            const EAMatrix& W,
                            const EAMatrix& b) const
{
    auto logits = MetaNN::Dot(h_T, W) + b;
    if (targetType == TargetType::BinaryReturn)
    {
        auto prob = MetaNN::Sigmoid(logits);
        auto pH = prob.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        return pH.Data()(0, 0);
    }
    else
    {
        auto predH = logits.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        return predH.Data()(0, 0);
    }
}

inline void EA::LSTM::accumulateHeadGrads(EAMatrix& dW_accum,
                                   EAMatrix& dB_accum,
                                   const EAMatrix& h_T,
                                   float err) const
{
#if LSTM_DEBUG_PRINTS
    std::cout << "err: " << err << std::endl;
    std::cout << "h_T(0,0): " << h_T(0,0) << std::endl;
    std::cout << "dW_accum(0,0): " << dW_accum(0,0) << std::endl;
    std::cout << "dB_accum(0,0): " << dB_accum(0,0) << std::endl;
#endif
    const size_t H = h_T.Shape()[1];
    // Evaluate h_T once and then perform contiguous updates to avoid repeated buffer operations
    auto h_eval = MetaNN::Evaluate(h_T);
    auto lowH = MetaNN::LowerAccess(h_eval);
    const float* hptr = lowH.RawMemory();

    auto lowW = MetaNN::LowerAccess(dW_accum);
    float* wptr = lowW.MutableRawMemory();
    for (size_t i = 0; i < H; ++i)  wptr[i] += hptr[i] * err;

    auto lowB = MetaNN::LowerAccess(dB_accum);
    float* bptr = lowB.MutableRawMemory();
    bptr[0] += err;
    
#if LSTM_DEBUG_PRINTS
    std::cout << "AFTER dW_accum(0,0): " << dW_accum(0,0) << std::endl;
    std::cout << "AFTER dB_accum(0,0): " << dB_accum(0,0) << std::endl;
#endif
}

inline auto EA::LSTM::hoistGateBlocks(const EAMatrix& W_h_win, size_t H) const -> GateBlocks
{
    const auto W_i = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(W_h_win, 0 * H, H);
    const auto W_f = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(W_h_win, 1 * H, H);
    const auto W_g = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(W_h_win, 2 * H, H);
    const auto W_o = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(W_h_win, 3 * H, H);
    return GateBlocks{ W_i, W_f, W_g, W_o };
}

inline void EA::LSTM::zeroGateAccumulators(GateAccumulators& A, size_t rows, size_t H) const
{
    auto zfill = [&](auto& m){
        auto low = MetaNN::LowerAccess(m);
        std::fill(low.MutableRawMemory(), low.MutableRawMemory() + m.Shape()[0] * m.Shape()[1], typename std::remove_reference_t<decltype(*low.MutableRawMemory())>(0));
    };
    zfill(A.dW_i); zfill(A.dW_f); zfill(A.dW_g); zfill(A.dW_o);
    zfill(A.db_i); zfill(A.db_f); zfill(A.db_g); zfill(A.db_o);
}

inline void EA::LSTM::backwardStep(const StepCache& sc,
                            const GateBlocks& gb,
                            EAMatrix& d_h,
                            EAMatrix& d_c,
                            GateAccumulators& A) const
{
    const size_t H = d_h.Shape()[1];



    auto tanh_c = MetaNN::Tanh(sc.c);
    auto d_o = d_h * tanh_c * sc.o * (1.0f - sc.o);
    auto d_c_from_h = d_h * sc.o * (1.0f - tanh_c * tanh_c);
    auto dct = d_c + d_c_from_h;

    auto d_i = dct * sc.g * sc.i * (1.0f - sc.i);
    auto d_g = dct * sc.i * (1.0f - sc.g * sc.g);
    auto d_f = dct * sc.c_prev * sc.f * (1.0f - sc.f);

    auto di2D = MetaNN::Reshape(d_i, MetaNN::Shape(1, H));
    auto df2D = MetaNN::Reshape(d_f, MetaNN::Shape(1, H));
    auto dg2D = MetaNN::Reshape(d_g, MetaNN::Shape(1, H));
    auto do2D = MetaNN::Reshape(d_o, MetaNN::Shape(1, H));

    auto dW_i_top = MetaNN::Dot(MetaNN::Transpose(sc.x), di2D);
    auto dW_f_top = MetaNN::Dot(MetaNN::Transpose(sc.x), df2D);
    auto dW_g_top = MetaNN::Dot(MetaNN::Transpose(sc.x), dg2D);
    auto dW_o_top = MetaNN::Dot(MetaNN::Transpose(sc.x), do2D);
    auto dW_i_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), di2D);
    auto dW_f_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), df2D);
    auto dW_g_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), dg2D);
    auto dW_o_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), do2D);

    auto addBlock = [&](auto& dst, const auto& top, const auto& bot)
    {
        auto topH = top.EvalRegister();
        auto botH = bot.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        auto lowD = MetaNN::LowerAccess(dst);
        auto lowT = MetaNN::LowerAccess(topH.Data());
        auto lowB = MetaNN::LowerAccess(botH.Data());
        auto* dptr = lowD.MutableRawMemory();
        const auto* tptr = lowT.RawMemory();
        const auto* bptr = lowB.RawMemory();
        const size_t I = dst.Shape()[0] - H;
        for (size_t r=0; r<I; ++r) for (size_t c=0; c<H; ++c) dptr[r*H + c] += static_cast<AccumScalar>(tptr[r*H + c]);
        for (size_t r=0; r<H; ++r) for (size_t c=0; c<H; ++c) dptr[(I+r)*H + c] += static_cast<AccumScalar>(bptr[r*H + c]);
    };
    addBlock(A.dW_i, dW_i_top, dW_i_bot);
    addBlock(A.dW_f, dW_f_top, dW_f_bot);
    addBlock(A.dW_g, dW_g_top, dW_g_bot);
    addBlock(A.dW_o, dW_o_top, dW_o_bot);

    auto accBias = [&](auto& db, const auto& g2D){
        auto gH = g2D.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        auto lowD = MetaNN::LowerAccess(db); auto* dptr = lowD.MutableRawMemory();
        auto lowS = MetaNN::LowerAccess(gH.Data()); const auto* sptr = lowS.RawMemory();
        for (size_t c=0; c<H; ++c) dptr[c] += static_cast<AccumScalar>(sptr[c]);
    };
    accBias(A.db_i, di2D); accBias(A.db_f, df2D); accBias(A.db_g, dg2D); accBias(A.db_o, do2D);

    auto d_h_prev = MetaNN::Dot(di2D, MetaNN::Transpose(gb.W_i))
                  + MetaNN::Dot(df2D, MetaNN::Transpose(gb.W_f))
                  + MetaNN::Dot(dg2D, MetaNN::Transpose(gb.W_g))
                  + MetaNN::Dot(do2D, MetaNN::Transpose(gb.W_o));
    auto dh_handle = MetaNN::Reshape(d_h_prev, MetaNN::Shape(1, H)).EvalRegister();
    auto dc_handle = MetaNN::Reshape((dct * sc.f), MetaNN::Shape(1, H)).EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();
    d_h = dh_handle.Data();
    d_c = dc_handle.Data();

}

inline void EA::LSTM::backwardStepBatch(const BatchStepCache& sc,
                                        const GateBlocks& gb,
                                        EAMatrix& d_h,
                                        EAMatrix& d_c,
                                        GateAccumulators& A) const
{
    const size_t B = d_h.Shape()[0];
    const size_t H = d_h.Shape()[1];

    auto tanh_c = MetaNN::Tanh(sc.c);

    auto d_o = d_h * tanh_c * sc.o * (1.0f - sc.o);
    auto d_c_from_h = d_h * sc.o * (1.0f - tanh_c * tanh_c);
    auto dct = d_c + d_c_from_h;

    auto d_i = dct * sc.g * sc.i * (1.0f - sc.i);
    auto d_g = dct * sc.i * (1.0f - sc.g * sc.g);
    auto d_f = dct * sc.c_prev * sc.f * (1.0f - sc.f);

    auto diH = d_i.EvalRegister();
    auto dfH = d_f.EvalRegister();
    auto dgH = d_g.EvalRegister();
    auto doH = d_o.EvalRegister();
    auto dctH = dct.EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();

    const EAMatrix di = diH.Data();
    const EAMatrix df = dfH.Data();
    const EAMatrix dg = dgH.Data();
    const EAMatrix dO = doH.Data();
    const EAMatrix dctM = dctH.Data();

    auto dW_i_top = MetaNN::Dot(MetaNN::Transpose(sc.x), di);
    auto dW_f_top = MetaNN::Dot(MetaNN::Transpose(sc.x), df);
    auto dW_g_top = MetaNN::Dot(MetaNN::Transpose(sc.x), dg);
    auto dW_o_top = MetaNN::Dot(MetaNN::Transpose(sc.x), dO);

    auto dW_i_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), di);
    auto dW_f_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), df);
    auto dW_g_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), dg);
    auto dW_o_bot = MetaNN::Dot(MetaNN::Transpose(sc.h_prev), dO);

    auto dW_i_topH = dW_i_top.EvalRegister();
    auto dW_f_topH = dW_f_top.EvalRegister();
    auto dW_g_topH = dW_g_top.EvalRegister();
    auto dW_o_topH = dW_o_top.EvalRegister();

    auto dW_i_botH = dW_i_bot.EvalRegister();
    auto dW_f_botH = dW_f_bot.EvalRegister();
    auto dW_g_botH = dW_g_bot.EvalRegister();
    auto dW_o_botH = dW_o_bot.EvalRegister();

    auto dh_prev_expr = MetaNN::Dot(di, MetaNN::Transpose(gb.W_i))
                      + MetaNN::Dot(df, MetaNN::Transpose(gb.W_f))
                      + MetaNN::Dot(dg, MetaNN::Transpose(gb.W_g))
                      + MetaNN::Dot(dO, MetaNN::Transpose(gb.W_o));
    auto dh_prevH = dh_prev_expr.EvalRegister();

    auto dc_prev_expr = dctM * sc.f;
    auto dc_prevH = dc_prev_expr.EvalRegister();

    MetaNN::EvalPlan::Inst().Eval();

    auto addBlock = [&](auto& dst, const EAMatrix& top, const EAMatrix& bot)
    {
        auto lowD = MetaNN::LowerAccess(dst);
        auto* dptr = lowD.MutableRawMemory();

        auto lowT = MetaNN::LowerAccess(top);
        const auto* tptr = lowT.RawMemory();

        auto lowB = MetaNN::LowerAccess(bot);
        const auto* bptr = lowB.RawMemory();

        const size_t rows = dst.Shape()[0];
        const size_t cols = dst.Shape()[1];
        const size_t I = rows - H;

        for (size_t r = 0; r < I; ++r)
            for (size_t c = 0; c < H; ++c)
                dptr[r * cols + c] += static_cast<AccumScalar>(tptr[r * H + c]);

        for (size_t r = 0; r < H; ++r)
            for (size_t c = 0; c < H; ++c)
                dptr[(I + r) * cols + c] += static_cast<AccumScalar>(bptr[r * H + c]);
    };

    addBlock(A.dW_i, dW_i_topH.Data(), dW_i_botH.Data());
    addBlock(A.dW_f, dW_f_topH.Data(), dW_f_botH.Data());
    addBlock(A.dW_g, dW_g_topH.Data(), dW_g_botH.Data());
    addBlock(A.dW_o, dW_o_topH.Data(), dW_o_botH.Data());

    auto addBias = [&](auto& db, const EAMatrix& g2D)
    {
        auto lowD = MetaNN::LowerAccess(db);
        auto* dptr = lowD.MutableRawMemory();

        auto lowG = MetaNN::LowerAccess(g2D);
        const auto* gptr = lowG.RawMemory();

        for (size_t c = 0; c < H; ++c)
        {
            AccumScalar s = 0;
            for (size_t b = 0; b < B; ++b)
                s += static_cast<AccumScalar>(gptr[b * H + c]);
            dptr[c] += s;
        }
    };

    addBias(A.db_i, di);
    addBias(A.db_f, df);
    addBias(A.db_g, dg);
    addBias(A.db_o, dO);

    d_h = dh_prevH.Data();
    d_c = dc_prevH.Data();
}

inline void EA::LSTM::mergeGateAccumulators(const GateAccumulators& A,
                                     MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>& d_param_accum,
                                     MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>& d_bias_accum,
                                     size_t H) const
{
    auto writeCols = [&](auto& dst, size_t colOffset, const auto& src){
        auto lowD = MetaNN::LowerAccess(dst); auto* dptr = lowD.MutableRawMemory();
        auto lowS = MetaNN::LowerAccess(src); const auto* sptr = lowS.RawMemory();
        const size_t rows = dst.Shape()[0]; const size_t dstCols = dst.Shape()[1];
        for (size_t r=0; r<rows; ++r) for (size_t c=0; c<H; ++c) dptr[r*dstCols + (colOffset + c)] += static_cast<AccumScalar>(sptr[r*H + c]);
    };
    writeCols(d_param_accum, 0*H, A.dW_i);
    writeCols(d_param_accum, 1*H, A.dW_f);
    writeCols(d_param_accum, 2*H, A.dW_g);
    writeCols(d_param_accum, 3*H, A.dW_o);

    auto writeBias = [&](auto& dst, size_t colOffset, const auto& src){
        auto lowD = MetaNN::LowerAccess(dst); auto* dptr = lowD.MutableRawMemory();
        auto lowS = MetaNN::LowerAccess(src); const auto* sptr = lowS.RawMemory();
        for (size_t c=0; c<H; ++c) dptr[colOffset + c] += static_cast<AccumScalar>(sptr[c]);
    };
    writeBias(d_bias_accum, 0*H, A.db_i);
    writeBias(d_bias_accum, 1*H, A.db_f);
    writeBias(d_bias_accum, 2*H, A.db_g);
    writeBias(d_bias_accum, 3*H, A.db_o);
}

EA::LSTM::LSTM(const Tensor& tt, float lt, float st)
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

    {
        auto low = MetaNN::LowerAccess(param);
        float* p = low.MutableRawMemory();
        const size_t cols = static_cast<size_t>(4 * n_out);
        for (int r = 0; r < n_in; ++r)
        {
            const size_t rowOff = static_cast<size_t>(r) * cols;
            for (size_t c = 0; c < cols; ++c) p[rowOff + c] = uniform_symmetric(0.01f);
        }
    }
    
    {
        auto low = MetaNN::LowerAccess(bias);
        float* bp = low.MutableRawMemory();
        std::fill(bp, bp + static_cast<size_t>(4 * n_out), 0.0f);
    }
    {
        // Add positive bias to forget gate block [H .. 2H)
        const size_t H = hidden_size;
        auto low = MetaNN::LowerAccess(bias);
        float* bp = low.MutableRawMemory();
        for (size_t j = H; j < 2 * H; ++j) bp[j] += 1.0f;
    }
    ResetPreviousState();

    switch(targetType)
    {
        case TargetType::PercentReturn: case TargetType::LogReturn:
            {
                auto lowW = MetaNN::LowerAccess(returnHeadWeight);
                float* wp = lowW.MutableRawMemory();
                std::fill(wp, wp + hidden_size, 0.01f);
            }
            {
                auto lowB = MetaNN::LowerAccess(returnHeadBias);
                float* bp = lowB.MutableRawMemory();
                bp[0] = 0.0f;
            }
            break;
        case TargetType::BinaryReturn:
            {
                auto lowW = MetaNN::LowerAccess(returnHeadDirWeight);
                float* wp = lowW.MutableRawMemory();
                std::fill(wp, wp + hidden_size, 0.01f);
            }
            {
                auto lowB = MetaNN::LowerAccess(returnHeadDirBias);
                float* bp = lowB.MutableRawMemory();
                bp[0] = 0.0f;
            }
    }
    
#if LSTM_DEBUG_PRINTS
    std::cout << "returnHeadWeight [ rows, cols ] = [ " << returnHeadWeight.Shape()[0] << "," << returnHeadWeight.Shape()[1] << " ]" << std::endl
        << "returnHeadWeight(0,0): " << returnHeadWeight(0,0) << std::endl
    << "returnHeadBias(0,0): " << returnHeadBias(0,0) << std::endl
        << "   returnHeadWeight(1,0): " << returnHeadWeight(1,0) << std::endl
        << "   returnHeadWeight(hidden_size-1,0)" << returnHeadWeight(hidden_size-1,0) << std::endl;
#endif
    long_term = lt; short_term = st;
#endif
}

std::tuple<float, size_t, size_t> EA::LSTM::CalculateBatch(Window batch)
{
    double sse = 0.0;
    size_t mseCount = 0;
    float predicted_close = 0;
    double runningLoss = 0.0;
    size_t windowCount = 0;
    size_t windowsInBatch = 0;
    size_t skippedWindows = 0;

    // Class counts for BinaryReturn targets
    size_t up_count = 0;
    size_t down_count = 0;
    size_t zero_count = 0;

    // Lazy head gradient accumulators (expressions) across all windows in the batch
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> xh_concat(mini_batch_windows, static_cast<size_t>(n_in + hidden_size));
    EAMatrix h_batch(mini_batch_windows, hidden_size);
    EAMatrix c_batch(mini_batch_windows, hidden_size);
    EAMatrix d_h_batch(mini_batch_windows, hidden_size);
    EAMatrix d_c_batch(mini_batch_windows, hidden_size);
    ForwardBatchScratch forward_scratch;

    GateAccumulators G_bin {
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size)
    };
    GateAccumulators G_reg {
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
        MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size)
    };

#if !LSTM_INFERENCE_ONLY
    // Head gradient accumulators across all windows in the batch
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_headW_accum_f(hidden_size, 1);
    { auto low = MetaNN::LowerAccess(d_headW_accum_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory() + hidden_size * 1, 0.0f); }
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_headB_accum_f(1, 1);
    { auto low = MetaNN::LowerAccess(d_headB_accum_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory() + 1, 0.0f); }

    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_headDirW_accum_f(hidden_size, 1);
    { auto low = MetaNN::LowerAccess(d_headDirW_accum_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory() + hidden_size * 1, 0.0f); }
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_headDirB_accum_f(1, 1);
    { auto low = MetaNN::LowerAccess(d_headDirB_accum_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory() + 1, 0.0f); }

    // LSTM core gradient accumulators across all windows in the batch
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal> d_param_accum(param.Shape()[0], param.Shape()[1]);
    {
        auto low = MetaNN::LowerAccess(d_param_accum);
        std::fill(low.MutableRawMemory(), low.MutableRawMemory() + param.Shape()[0] * param.Shape()[1], static_cast<AccumScalar>(0));
    }
    MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal> d_bias_accum(bias.Shape()[0], bias.Shape()[1]);
    {
        auto low = MetaNN::LowerAccess(d_bias_accum);
        std::fill(low.MutableRawMemory(), low.MutableRawMemory() + bias.Shape()[0] * bias.Shape()[1], static_cast<AccumScalar>(0));
    }
#endif

#if !LSTM_INFERENCE_ONLY
    // Debug stats for training targets (log returns) and sample predictions
    double y_sum = 0.0, y_sumsq = 0.0;
    float y_min = std::numeric_limits<float>::infinity();
    float y_max = -std::numeric_limits<float>::infinity();
    size_t y_count = 0;
    std::vector<float> yhat_samples;
    std::vector<float> ydenorm_samples; // predicted price delta (predicted_close - close_T)
    double max_abs_y_true = 0.0;
    size_t count_abs_gt_0p01 = 0;
    size_t count_abs_gt_threshold = 0;

    // Per-batch predicted/actual log-return stats
#endif

    ResetPreviousState();

    auto buildWindowBatch = [&](auto first, auto last) -> WindowBatch
    {
        WindowBatch wb;
        for (auto window_start = first;
             window_start != last && window_start + window_size + prediction_horizon - 1 < batch.end();
             ++window_start)
        {
            wb.windows.push_back(t.GetWindow(window_start));

            auto lastIt   = window_start + (window_size - 1);
            auto targetIt = lastIt + prediction_horizon;
            const float close_t_local      = t.RawCloseAtIterator(lastIt);
            const float close_target_local = t.RawCloseAtIterator(targetIt);
            const auto nextFeat = *targetIt;
            const float y_true_scaled = nextFeat(0, closeCol);
            const float y_true_logret = y_true_scaled / EA::LSTM::kFeatScale;

            wb.close_t.push_back(close_t_local);
            wb.close_target.push_back(close_target_local);

            if (targetType == TargetType::BinaryReturn)
            {
                wb.targets.push_back((close_target_local > close_t_local) ? 1.0f : 0.0f);
            }
            else
            {
                float y_true_logret_used = y_true_logret;
                if (std::isfinite(y_true_logret_used) && std::abs(y_true_logret_used) > c_next_threshold)
                    y_true_logret_used = std::copysign(c_next_threshold, y_true_logret_used);

                const float raw = (targetType == TargetType::LogReturn) ? y_true_logret_used
                                                                        : (std::exp(y_true_logret_used) - 1.0f);
                float tval = raw * targetScale + targetBias;
                if (targetUseZScore) tval = (tval - targetMean) / std::max(targetStd, 1e-12f);
                wb.targets.push_back(std::clamp(tval, -10.0f, 10.0f));
            }
        }
        return wb;
    };

    // Materialize overlapping windows, then process them in minibatches.
    std::vector<decltype(batch.begin())> allStarts;
    for (auto window_start = batch.begin();
         window_start + window_size + prediction_horizon - 1 < batch.end();
         ++window_start)
    {
        allStarts.push_back(window_start);
    }

    for (size_t batchBase = 0; batchBase < allStarts.size(); batchBase += mini_batch_windows)
    {
        const size_t batchEnd = std::min(batchBase + mini_batch_windows, allStarts.size());
        WindowBatch wb = buildWindowBatch(allStarts[batchBase], allStarts[batchEnd - 1] + 1);
        const size_t B = wb.windows.size();
        if (B == 0) continue;

        if (h_batch.Shape()[0] != B || h_batch.Shape()[1] != hidden_size)
            h_batch = EAMatrix(B, hidden_size);
        if (c_batch.Shape()[0] != B || c_batch.Shape()[1] != hidden_size)
            c_batch = EAMatrix(B, hidden_size);
        {
            auto lowH = MetaNN::LowerAccess(h_batch);
            auto lowC = MetaNN::LowerAccess(c_batch);
            std::fill(lowH.MutableRawMemory(), lowH.MutableRawMemory() + B * hidden_size, 0.0f);
            std::fill(lowC.MutableRawMemory(), lowC.MutableRawMemory() + B * hidden_size, 0.0f);
        }

        auto ww = hoistWindowWeights();
        std::vector<BatchStepCache> cache;
        cache.reserve(window_size);

        // Pre-materialize minibatch inputs once in time-major layout: (window_size * B, n_in)
        for (size_t tstep = 0; tstep < window_size; ++tstep)
        {
            std::vector<EAMatrix> rows;
            rows.reserve(B);
            for (size_t b = 0; b < B; ++b)
                rows.push_back(wb.windows[b][tstep]);

            auto x_t_batch = GatherRows(rows);
            cache.push_back(forwardStepBatch(x_t_batch, ww, bias, h_batch, c_batch, xh_concat, forward_scratch));
        }

        std::vector<float> errs(B, 0.0f);

        // Batched head forward and loss on (B, H)
        if (targetType == TargetType::BinaryReturn)
        {
            // z = h_batch · W + b (broadcast)
            auto z0 = MetaNN::Dot(h_batch, returnHeadDirWeight);
            auto z0H = z0.EvalRegister();
            MetaNN::EvalPlan::Inst().Eval();

            auto pH = MetaNN::Sigmoid(z0H.Data() + RepeatRows(returnHeadDirBias, B)).EvalRegister();
            MetaNN::EvalPlan::Inst().Eval();
            EAMatrix pMat = pH.Data();

            for (size_t b = 0; b < B; ++b)
            {
                const float p = pMat(b, 0);
                const float target = wb.targets[b];
                errs[b] = p - target; // dL/dz for BCE with sigmoid

                const double p_clamped = std::clamp(static_cast<double>(p), 1e-12, 1.0 - 1e-12);
                const double ce = -(static_cast<double>(target) * std::log(p_clamped)
                                  + (1.0 - static_cast<double>(target)) * std::log(1.0 - p_clamped));
                sse += ce;
                ++mseCount;

                // Class counts for diagnostics
                const float close_t = wb.close_t[b];
                const float close_target = wb.close_target[b];
                if (close_target > close_t)      ++up_count;
                else if (close_target < close_t) ++down_count;
                else                              ++zero_count;

    #if !LSTM_INFERENCE_ONLY
                y_sum   += static_cast<double>(target);
                y_sumsq += static_cast<double>(target) * static_cast<double>(target);
                y_min = std::min(y_min, target);
                y_max = std::max(y_max, target);
                ++y_count;
                if (yhat_samples.size() < 10) yhat_samples.push_back(p);
        #if LSTM_TRAINING_PROGRESS
                runningLoss += ce;
        #endif
    #endif
            }
    #if !LSTM_INFERENCE_ONLY
            windowCount += B;
            windowsInBatch += B;
    #endif
        }
        else
        {
            auto y0 = MetaNN::Dot(h_batch, returnHeadWeight);
            auto y0H = y0.EvalRegister();
            MetaNN::EvalPlan::Inst().Eval();

            auto yH = (y0H.Data() + RepeatRows(returnHeadBias, B)).EvalRegister();
            MetaNN::EvalPlan::Inst().Eval();
            EAMatrix yMat = yH.Data();

            for (size_t b = 0; b < B; ++b)
            {
                const float y_hat = yMat(b, 0);
                const float target = wb.targets[b];
                const float err = y_hat - target;
                errs[b] = err;

                sse += static_cast<double>(err) * static_cast<double>(err);
                ++mseCount;
    #if !LSTM_INFERENCE_ONLY
                y_sum += static_cast<double>(target);
                y_sumsq += static_cast<double>(target) * static_cast<double>(target);
                y_min = std::min(y_min, target);
                y_max = std::max(y_max, target);
                ++y_count;
                if (yhat_samples.size() < 10) yhat_samples.push_back(y_hat);
        #if LSTM_TRAINING_PROGRESS
                runningLoss += 0.5 * static_cast<double>(err) * static_cast<double>(err);
        #endif
    #endif
            }
    #if !LSTM_INFERENCE_ONLY
            windowCount += B;
            windowsInBatch += B;
    #endif
        }

#if !LSTM_INFERENCE_ONLY
        if (targetType == TargetType::BinaryReturn)
        {
            AccumulateHeadGradsBatch(d_headDirW_accum_f, d_headDirB_accum_f, h_batch, errs);

            d_h_batch = BuildHeadDhBatch(errs, returnHeadDirWeight, LSTM_CORE_GRAD_SCALE);
            if (d_c_batch.Shape()[0] != B || d_c_batch.Shape()[1] != hidden_size)
                d_c_batch = EAMatrix(B, hidden_size);
            zeroFill(d_c_batch);

            zeroGateAccumulators(G_bin, param.Shape()[0], hidden_size);

            auto gb = hoistGateBlocks(ww.W_h_win, hidden_size);
            for (int tstep = static_cast<int>(cache.size()) - 1; tstep >= 0; --tstep)
                backwardStepBatch(cache[static_cast<size_t>(tstep)], gb, d_h_batch, d_c_batch, G_bin);

            mergeGateAccumulators(G_bin, d_param_accum, d_bias_accum, hidden_size);
        }
        else
        {
            AccumulateHeadGradsBatch(d_headW_accum_f, d_headB_accum_f, h_batch, errs);

            d_h_batch = BuildHeadDhBatch(errs, returnHeadWeight, LSTM_CORE_GRAD_SCALE);
            if (d_c_batch.Shape()[0] != B || d_c_batch.Shape()[1] != hidden_size)
                d_c_batch = EAMatrix(B, hidden_size);
            zeroFill(d_c_batch);

            zeroGateAccumulators(G_reg, param.Shape()[0], hidden_size);

            auto gb = hoistGateBlocks(ww.W_h_win, hidden_size);
            for (int tstep = static_cast<int>(cache.size()) - 1; tstep >= 0; --tstep)
                backwardStepBatch(cache[static_cast<size_t>(tstep)], gb, d_h_batch, d_c_batch, G_reg);

            mergeGateAccumulators(G_reg, d_param_accum, d_bias_accum, hidden_size);
        }
#endif
    }

#if !LSTM_INFERENCE_ONLY
    // Per-batch diagnostics
    std::cout << "batch_count=" << windowCount << "\n";
    double loss_value = 0.0;
#if LSTM_TRAINING_PROGRESS
    loss_value = (windowCount > 0) ? (runningLoss / static_cast<double>(windowCount)) : 0.0;
#else
    loss_value = (windowCount > 0) ? (sse / static_cast<double>(windowCount)) : 0.0;
#endif
    std::cout << "loss_value=" << loss_value << "\n";
    if (targetType == TargetType::BinaryReturn) std::cout << "up_count=" << up_count
                  << " down_count=" << down_count
                  << " zero_count=" << zero_count << "\n";
#endif

#if !LSTM_INFERENCE_ONLY && LSTM_DEBUG_PRINTS
    if (y_count > 0)
    {
        double y_mean = y_sum / static_cast<double>(y_count);
        double y_var  = std::max(0.0, y_sumsq / static_cast<double>(y_count) - y_mean * y_mean);
        double y_std  = std::sqrt(y_var);
        std::cout << "train: target_log_return stats n=" << y_count
                  << " mean=" << y_mean
                  << " std="  << y_std
                  << " min="  << y_min
                  << " max="  << y_max << std::endl;
        std::cout << "train: pred_raw (log-return) samples:";
        for (float v : yhat_samples) std::cout << ' ' << v;
        std::cout << std::endl;
        std::cout << "train: pred_pct (relative move) samples:";
        for (float v : ydenorm_samples) std::cout << ' ' << v;
        std::cout << std::endl;
        std::cout << "train: skipped_windows=" << skippedWindows << std::endl;
    }
    if (windowCount > 0) std::cout << "Batch MSE: " << (runningLoss / static_cast<double>(windowCount)) << " (" << windowCount << " windows)" << std::endl;
    std::cout << "batch: max_abs_y_true=" << max_abs_y_true
              << " count_abs_gt_0p01=" << count_abs_gt_0p01
              << " count_abs_gt_threshold=" << count_abs_gt_threshold << std::endl;

    if (windowsInBatch > 0)
    {
        double pred_sum = 0.0, pred_sumsq = 0.0;
        double act_sum = 0.0, act_sumsq = 0.0;
        double max_abs_pred_logret = 0.0;
        size_t count_pred_abs_gt_thresh = 0;
        const float pred_action_threshold = 1e-3f; // threshold for |predLogRet|

        const double pred_mean = pred_sum / static_cast<double>(windowsInBatch);
        const double pred_var  = std::max(0.0, pred_sumsq / static_cast<double>(windowsInBatch) - pred_mean * pred_mean);
        const double pred_std  = std::sqrt(pred_var);
        const double act_mean  = act_sum / static_cast<double>(windowsInBatch);
        const double act_var   = std::max(0.0, act_sumsq / static_cast<double>(windowsInBatch) - act_mean * act_mean);
        const double act_std   = std::sqrt(act_var);

        std::cout << "predLogRet: mean=" << pred_mean
                  << " std=" << pred_std
                  << " max_abs=" << max_abs_pred_logret
                  << " count(|pred|>" << pred_action_threshold << ")=" << count_pred_abs_gt_thresh
                  << std::endl;
        std::cout << "actLogRet:  mean=" << act_mean
                  << " std=" << act_std
                  << std::endl;
    }

    // Print the current returnHeadWeight vector (hidden_size x 1)
    std::cout << "returnHeadBias: [" << returnHeadBias(0, 0) << "]" << std::endl;
    std::cout << "returnHeadWeight(0,0): " << returnHeadWeight(0,0)
              << " (1,0): " << returnHeadWeight(1,0)
              << " (63,0): " << returnHeadWeight(63,0) << "\n";
#endif
#if !LSTM_INFERENCE_ONLY
    if (windowCount > 0)
    {
        // Convert accumulators to concrete matrices (ensures RawMemory is valid)
        const auto d_param_f = MetaNN::Evaluate(d_param_accum);
        const auto d_bias_f  = MetaNN::Evaluate(d_bias_accum);
        const auto d_headW_f = MetaNN::Evaluate(d_headW_accum_f);
        const auto d_headB_f = MetaNN::Evaluate(d_headB_accum_f);

        const auto d_headDirW_f = MetaNN::Evaluate(d_headDirW_accum_f);
        const auto d_headDirB_f = MetaNN::Evaluate(d_headDirB_accum_f);

        // Scale learning rate by number of windows so batch size doesn't change step size
        const float invN = 1.0f / static_cast<float>(windowCount);

        const float lrCore = learningRate * invN;
        const float lrHead = learningRate * invN; // or a separate head LR if you want

        SGDUpdate(param, d_param_f, lrCore);
        SGDUpdate(bias,  d_bias_f,  lrCore);
        if (targetType == TargetType::BinaryReturn) {
            SGDUpdate(returnHeadDirWeight, d_headDirW_f, lrHead);
            SGDUpdate(returnHeadDirBias,   d_headDirB_f, lrHead);
        } else {
            SGDUpdate(returnHeadWeight, d_headW_f, lrHead);
            SGDUpdate(returnHeadBias,   d_headB_f, lrHead);
        }
    }
#endif
    double mse = sse / static_cast<double>(std::max<size_t>(mseCount, 1));
    return { mse, windowCount, skippedWindows };
}

std::vector<float> EA::LSTM::RollingPredictNextLogReturn(const Window& batch, bool resetAtStart)
{
    if (resetAtStart) ResetPreviousState();
    std::vector<float> preds;
    preds.reserve(batch.end() - batch.begin());
    for (auto it = batch.begin(); it + window_size < batch.end(); ++it)
    {
        auto w = t.GetWindow(it);
        preds.push_back(PredictNextReturn(w, reset_state_per_window));
    }
    return preds;
}

std::vector<float> EA::LSTM::RollingPredictNextClose(const Window& batch, bool resetAtStart)
{
    if (resetAtStart) ResetPreviousState();
    std::vector<float> preds;
    preds.reserve(batch.end() - batch.begin());
    for (auto it = batch.begin(); it + window_size < batch.end(); ++it)
    {
        auto w = t.GetWindow(it);
        preds.push_back(PredictNextClose(w, reset_state_per_window));
    }
    return preds;
}



inline float EA::LSTM::PredictNextReturn(const Window& w, bool resetState)
{
    if (resetState) {
        ResetPreviousState();
    }

    // Prepare views and concat buffer
    auto ww = hoistWindowWeights();
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> xh_concat(1, static_cast<size_t>(n_in + hidden_size));
    // Forward through the window
    for (const auto& f_sample : w)  forwardStep(f_sample, ww, bias, prevHiddenState, prevCellState, xh_concat);

    // Predict next-step log return from the last hidden state
    float y_hat = (targetType == TargetType::BinaryReturn)
        ? predictOnly(prevHiddenState, returnHeadDirWeight, returnHeadDirBias)
        : predictOnly(prevHiddenState, returnHeadWeight, returnHeadBias);

    if (targetType == TargetType::BinaryReturn) return (y_hat >= 0.5f) ? 1.0f : 0.0f;

    // Invert normalization if used: t = y_hat * std + mean
    float t = y_hat;
    if (targetUseZScore)
        t = y_hat * targetStd + targetMean;
    else
        t = y_hat;
    // Invert affine to raw return
    float raw = (t - targetBias) / std::max(targetScale, 1e-12f);

    // REPLACED HERE: explicit switch return for BinaryReturn type
    switch (targetType)
    {
        case TargetType::PercentReturn: return std::log(1.0f + raw); // convert percent return to log-return
        case TargetType::LogReturn: default:     return raw; // already log-return
    }
}

inline float EA::LSTM::PredictNextClose(const Window& w, bool resetState)
{
    // Minimal inference fix: return predicted relative move (fraction), not a price.
    if (resetState) ResetPreviousState();

    // Prepare views and concat buffer
    auto ww = hoistWindowWeights();
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> xh_concat(1, static_cast<size_t>(n_in + hidden_size));

    // Forward through the window
    for (const auto& f_sample : w)  forwardStep(f_sample, ww, bias, prevHiddenState, prevCellState, xh_concat);

    // Predict next-step return from the last hidden state
    float y_hat = (targetType == TargetType::BinaryReturn)
        ? predictOnly(prevHiddenState, returnHeadDirWeight, returnHeadDirBias)
        : predictOnly(prevHiddenState, returnHeadWeight, returnHeadBias);

    if (targetType == TargetType::BinaryReturn) return (y_hat >= 0.5f) ? 1.0f : 0.0f;

    // Invert optional z-score normalization
    float t = targetUseZScore ? (y_hat * targetStd + targetMean) : y_hat;
    // Invert affine
    float raw = (t - targetBias) / std::max(targetScale, 1e-12f);

    // REPLACED HERE: explicit switch return for BinaryReturn type
    switch (targetType)
    {
        case TargetType::LogReturn: return std::exp(raw) - 1.0f; // convert log-return to percent move
        case TargetType::PercentReturn: default: return raw; // already percent move
    }
}











