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
struct EA::LSTM::GateBlocks {   FloatMatrixGPU W_i, W_f, W_g, W_o;  }; // (H x H)
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
    FloatMatrixGPU W_cat;   // (n_in + H) x (4H)
    FloatMatrixGPU W_x_win; // top block (n_in x 4H)
    FloatMatrixGPU W_h_win; // bottom block (H x 4H)
};

struct EA::LSTM::StepCache
{
    FloatMatrixGPU x;      // (1, input_size)
    FloatMatrixGPU h_prev; // (1, hidden_size)
    FloatMatrixGPU c_prev; // (1, hidden_size)
    FloatMatrixGPU i;      // (1, hidden_size)
    FloatMatrixGPU f;      // (1, hidden_size)
    FloatMatrixGPU g;      // (1, hidden_size)
    FloatMatrixGPU o;      // (1, hidden_size)
    FloatMatrixGPU c;      // (1, hidden_size)
    FloatMatrixGPU h;      // (1, hidden_size)
};

// Member function definitions moved to EA::LSTM

inline auto EA::LSTM::hoistWindowWeights() const -> WindowWeights
{
    const auto W_x_win = NNUtils::ViewTopRows<float, MetaNN::DeviceTags::Metal>(param, param.Shape()[0] - hidden_size);
    const auto W_h_win = NNUtils::ViewBottomRows<float, MetaNN::DeviceTags::Metal>(param, hidden_size);
    const auto W_cat   = param; // full (n_in + H) x (4H) matrix; dynamic row views taken later
    return WindowWeights{ W_cat, W_x_win, W_h_win };
}

inline auto EA::LSTM::forwardStep(const FloatMatrixGPU& x_t,
                           const WindowWeights& ww,
                           const FloatMatrixGPU& bias,
                           FloatMatrixGPU& prevHiddenState,
                           FloatMatrixGPU& prevCellState,
                           FloatMatrixGPU& xh_concat) const -> StepCache
{
    // Build [x_t | h_{t-1}] and compute all gates in one GEMM
#if LSTM_DEBUG_PRINTS
    printMatrix("x_t", x_t);
#endif
    {
        const size_t expectedCols = x_t.Shape()[1] + prevHiddenState.Shape()[1];
        if (xh_concat.Shape()[0] != 1 || xh_concat.Shape()[1] != expectedCols) {
            xh_concat = FloatMatrixGPU(1, expectedCols);
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
    auto W_cat_dyn = NNUtils::ViewRows<float, MetaNN::DeviceTags::Metal>(ww.W_cat, 0, K);
    auto yExpr = MetaNN::Dot(xh_concat, W_cat_dyn) + bias;
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

inline auto EA::LSTM::predictAndLoss(const FloatMatrixGPU& h_T,
                             const FloatMatrixGPU& W,
                             const FloatMatrixGPU& b,
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

float EA::LSTM::predictOnly(const FloatMatrixGPU& h_T,
                            const FloatMatrixGPU& W,
                            const FloatMatrixGPU& b) const
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

inline void EA::LSTM::accumulateHeadGrads(FloatMatrixGPU& dW_accum,
                                   FloatMatrixGPU& dB_accum,
                                   const FloatMatrixGPU& h_T,
                                   float err) const
{
#if LSTM_DEBUG_PRINTS
    std::cout << "err: " << err << std::endl;
    std::cout << "h_T(0,0): " << h_T(0,0) << std::endl;
    std::cout << "dW_accum(0,0): " << dW_accum(0,0) << std::endl;
    std::cout << "dB_accum(0,0): " << dB_accum(0,0) << std::endl;
#endif
    const size_t H = h_T.Shape()[1];
    for (size_t i = 0; i < H; ++i) {
        dW_accum.SetValue(i, 0, dW_accum(i, 0) + h_T(0, i) * err);
    }
    dB_accum.SetValue(0, 0, dB_accum(0, 0) + err);
    
#if LSTM_DEBUG_PRINTS
    std::cout << "AFTER dW_accum(0,0): " << dW_accum(0,0) << std::endl;
    std::cout << "AFTER dB_accum(0,0): " << dB_accum(0,0) << std::endl;
#endif
}

inline auto EA::LSTM::hoistGateBlocks(const FloatMatrixGPU& W_h_win, size_t H) const -> GateBlocks
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
                            FloatMatrixGPU& d_h,
                            FloatMatrixGPU& d_c,
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

    for (int r = 0; r < n_in; ++r)
        for (int c = 0; c < 4 * n_out; ++c)
            param.SetValue(r, c, uniform_symmetric(.01));//(limit));
    
        // initialize bias, previous hidden and previous cell state
    for (size_t j = 0; j < 4 * n_out; ++j) bias.SetValue(0, j, 0.0f);
    {
        // Add positive bias to forget gate block [H .. 2H)
        const size_t H = hidden_size;
        for (size_t j = H; j < 2 * H; ++j) {
            bias.SetValue(0, j, bias(0, j) + 1.0f);
        }
    }
    ResetPreviousState();

    switch(targetType)
    {
        case TargetType::PercentReturn: case TargetType::LogReturn:
            for (size_t i = 0; i < hidden_size; ++i) returnHeadWeight.SetValue(i, 0, 0.01f);
            returnHeadBias.SetValue(0, 0, 0.0f);
            break;
        case TargetType::BinaryReturn:
            for (size_t i = 0; i < hidden_size; ++i) returnHeadDirWeight.SetValue(i, 0, 0.01f);
            returnHeadDirBias.SetValue(0, 0, 0.0f);
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

    // Lazy head gradient accumulators (expressions) across all windows in the batch
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> xh_concat(1, static_cast<size_t>(n_in + hidden_size));

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
    const float pred_action_threshold = 1e-3f; // threshold for |predLogRet|
#endif

    ResetPreviousState();

    // Slide a 5-step window across this 15-step batch
    for (auto window_start = batch.begin(); window_start + window_size < batch.end(); ++window_start)
    {
        // Reset states for an independent window when enabled
        if constexpr (reset_state_per_window) ResetPreviousState();

        std::vector<StepCache> cache;
        cache.reserve(window_size);

        // Hoist forward weight views per window (surgically extracted)
        auto ww = hoistWindowWeights();

        // Build a 5-step window starting at 'start' using the const iterator overload
        Window w = t.GetWindow(window_start);

        for(const auto& f_sample : w)
        {
#if LSTM_DEBUG_PRINTS
            static int t = 0;
            int this_t;
            std::cout << "t: " << (this_t = t++) << std::endl;
#endif
            cache.push_back(forwardStep(f_sample, ww, bias, prevHiddenState, prevCellState, xh_concat));
        }
#if LSTM_SAT_DEBUG
        {
            auto l2norm = [](const auto& m){
                auto low = MetaNN::LowerAccess(m);
                const float* p = low.RawMemory();
                const size_t len = m.Shape()[0] * m.Shape()[1];
                double s = 0.0;
                for (size_t i = 0; i < len; ++i) { double v = static_cast<double>(p[i]); s += v * v; }
                return std::sqrt(s);
            };
            auto meanAll = [](const auto& m){
                auto low = MetaNN::LowerAccess(m);
                const float* p = low.RawMemory();
                const size_t len = m.Shape()[0] * m.Shape()[1];
                double s = 0.0;
                for (size_t i = 0; i < len; ++i) s += static_cast<double>(p[i]);
                return (len > 0) ? (s / static_cast<double>(len)) : 0.0;
            };
            const auto& last = cache.back();
            std::cout << "sat: ||h_T||=" << l2norm(prevHiddenState)
                      << " ||c_T||=" << l2norm(prevCellState)
                      << " gate_means i=" << meanAll(last.i)
                      << " f=" << meanAll(last.f)
                      << " g=" << meanAll(last.g)
                      << " o=" << meanAll(last.o) << std::endl;
        }
#endif

        // ---- BPTT for next-step return (regression) ----
        // We need the last sample in the window and the next sample after the window
        auto lastIt = window_start + (window_size - 1);
        auto nextIt = window_start + window_size; // safe due to '<' in loop condition

        // Removed unused lastFeat assignment:
        // auto lastFeat  = *lastIt;

        auto nextFeat  = *nextIt;
#if LSTM_DEBUG_PRINTS
        // Ground-truth next-step log-return is simply the next sample's close feature
        std::cout << "nextFeat rows=" << nextFeat.Shape()[0]
                  << " cols=" << nextFeat.Shape()[1] << "\n";

        for (size_t r = 0; r < nextFeat.Shape()[0]; ++r)
            std::cout << "nextFeat(" << r << ",0)=" << nextFeat(r,0) << "\n";
#endif
        
        const float y_true_scaled = nextFeat(0, closeCol);
        const float y_true_logret = y_true_scaled / EA::LSTM::kFeatScale;
#if !LSTM_INFERENCE_ONLY
        if (std::isfinite(y_true_logret))
        {
            float abs_y = std::abs(y_true_logret);
            if (abs_y > max_abs_y_true) max_abs_y_true = abs_y;
            if (abs_y > 0.01f) ++count_abs_gt_0p01;
            if (abs_y > c_next_threshold) ++count_abs_gt_threshold;
        }
#endif

        if (std::isfinite(y_true_logret) && std::abs(y_true_logret) > 0.01f)
        {
            float close_t   = t.RawCloseAtIterator(lastIt);
            float close_tp1 = t.RawCloseAtIterator(nextIt);
            std::cout << "ALERT: |c_next|>0.01 raw_close_t=" << close_t
                      << " raw_close_t+1=" << close_tp1
                      << " c_next=" << y_true_logret << std::endl;
        }

        // Validate y_true; skip non-finite, clamp extreme outliers
        if (!std::isfinite(y_true_logret))
        {
#if !LSTM_INFERENCE_ONLY
            ++skippedWindows;
#endif
            float close_t   = t.RawCloseAtIterator(lastIt);
            float close_tp1 = t.RawCloseAtIterator(nextIt);
            std::cout << "BAD c_next(logret)=" << y_true_logret << " raw_close_t=" << close_t << " raw_close_t+1=" << close_tp1 << std::endl;
            continue;
        }

        // clamp instead of skipping (regression only); for BinaryReturn we keep sign
        float y_true_logret_used = y_true_logret;
        if (targetType != TargetType::BinaryReturn)
        {
            if (std::abs(y_true_logret_used) > c_next_threshold)
                y_true_logret_used = std::copysign(c_next_threshold, y_true_logret_used);
        }

        if (targetType == TargetType::BinaryReturn)
        {
            // Binary target: up (>=0) -> 1, down (<0) -> 0
            const float y_bin = (y_true_logret >= 0.0f) ? 1.0f : 0.0f;

            auto head = predictAndLoss(prevHiddenState, returnHeadDirWeight, returnHeadDirBias, y_bin);

            const float p = head.y_hat;           // probability
            const float err = head.err;           // (p - y)
            const double p_clamped = std::clamp(static_cast<double>(p), 1e-12, 1.0 - 1e-12);
            const double ce = -(static_cast<double>(y_bin) * std::log(p_clamped) + (1.0 - static_cast<double>(y_bin)) * std::log(1.0 - p_clamped));
            sse += ce;
            ++mseCount;
#if !LSTM_INFERENCE_ONLY
            // minimal stats
            y_sum   += static_cast<double>(y_bin);
            y_sumsq += static_cast<double>(y_bin) * static_cast<double>(y_bin);
            y_min = std::min(y_min, y_bin);
            y_max = std::max(y_max, y_bin);
            ++y_count;
            if (yhat_samples.size() < 10) yhat_samples.push_back(p);
    #if LSTM_TRAINING_PROGRESS
            runningLoss += ce;
            ++windowCount;
    #endif
            ++windowsInBatch;
#endif
#if !LSTM_INFERENCE_ONLY
            accumulateHeadGrads(d_headDirW_accum_f, d_headDirB_accum_f, prevHiddenState, err);
#endif

            // Backprop into core: use the binary head weights
            MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_h(1, hidden_size);
            for (size_t i = 0; i < hidden_size; ++i)
                d_h.SetValue(0, i, (err * returnHeadDirWeight(i, 0)) * LSTM_CORE_GRAD_SCALE);

            MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_c(1, hidden_size);
            {
                auto low = MetaNN::LowerAccess(d_c);
                std::fill(low.MutableRawMemory(), low.MutableRawMemory() + hidden_size, 0.0f);
            }

            GateAccumulators G {
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size)
            };
            zeroGateAccumulators(G, param.Shape()[0], hidden_size);

            auto gb = hoistGateBlocks(ww.W_h_win, hidden_size);
            for (int tstep = static_cast<int>(cache.size()) - 1; tstep >= 0; --tstep)
                backwardStep(cache[static_cast<size_t>(tstep)], gb, d_h, d_c, G);

#if !LSTM_INFERENCE_ONLY
            mergeGateAccumulators(G, d_param_accum, d_bias_accum, hidden_size);
#endif
        }
        else
        {
            // Regression path (original)
            const float raw = (targetType == TargetType::LogReturn) ? y_true_logret_used
                                                                    : (std::exp(y_true_logret_used) - 1.0f);
            float t = raw * targetScale + targetBias;
            if (targetUseZScore) t = (t - targetMean) / std::max(targetStd, 1e-12f);

            if (!std::isfinite(raw)) { ++skippedWindows; continue; }
            if (!std::isfinite(t))   { ++skippedWindows; continue; }
            t = std::clamp(t, -10.0f, 10.0f);

            auto head = predictAndLoss(prevHiddenState, returnHeadWeight, returnHeadBias, t);

            float t_inv = head.y_hat;
            if (targetUseZScore) t_inv = head.y_hat * targetStd + targetMean;
            float raw_inv = (t_inv - targetBias) / std::max(targetScale, 1e-12f);
            float pred_logret = (targetType == TargetType::LogReturn) ? raw_inv : std::log(1.0f + raw_inv);
            float pred_pct    = std::exp(pred_logret) - 1.0f;
            predicted_close = pred_logret;

            const float err = head.err;
            sse += static_cast<double>(err) * static_cast<double>(err);
            ++mseCount;
#if !LSTM_INFERENCE_ONLY
            y_sum += static_cast<double>(t);
            y_sumsq += static_cast<double>(t) * static_cast<double>(t);
            y_min = std::min(y_min, t);
            y_max = std::max(y_max, t);
            ++y_count;
            if (yhat_samples.size() < 10)
            {
                yhat_samples.push_back(pred_logret);
                ydenorm_samples.push_back(pred_pct);
            }
    #if LSTM_TRAINING_PROGRESS
            runningLoss += 0.5 * static_cast<double>(err) * static_cast<double>(err);
            ++windowCount;
    #endif
            ++windowsInBatch;
#endif
#if !LSTM_INFERENCE_ONLY
            accumulateHeadGrads(d_headW_accum_f, d_headB_accum_f, prevHiddenState, err);
#endif

            MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_h(1, hidden_size);
            for (size_t i = 0; i < hidden_size; ++i)
                d_h.SetValue(0, i, (err * returnHeadWeight(i, 0)) * LSTM_CORE_GRAD_SCALE);

            MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_c(1, hidden_size);
            {
                auto low = MetaNN::LowerAccess(d_c);
                std::fill(low.MutableRawMemory(), low.MutableRawMemory() + hidden_size, 0.0f);
            }

            GateAccumulators G {
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(param.Shape()[0], hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size),
                MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>(1, hidden_size)
            };
            zeroGateAccumulators(G, param.Shape()[0], hidden_size);
            auto gb = hoistGateBlocks(ww.W_h_win, hidden_size);
            for (int tstep = static_cast<int>(cache.size()) - 1; tstep >= 0; --tstep)
                backwardStep(cache[static_cast<size_t>(tstep)], gb, d_h, d_c, G);
#if !LSTM_INFERENCE_ONLY
            mergeGateAccumulators(G, d_param_accum, d_bias_accum, hidden_size);
#endif
        }
    }

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

    if (targetType == TargetType::BinaryReturn) return (y_hat >= 0.5f) ? 1.0f : -1.0f;

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

    if (targetType == TargetType::BinaryReturn) return (y_hat >= 0.5f) ? 1.0f : -1.0f;

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







