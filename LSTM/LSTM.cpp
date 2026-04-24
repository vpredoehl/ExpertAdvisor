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
#include <chrono>
#include <iomanip>
#include <numeric>

#include "LSTM.hpp"
#include "Tensor.hpp"
#include "MatrixUtils.hpp"
#include "BuildConfig.hpp"
#include <MetaNN/metal/metal_matmul.h>

#ifndef LSTM_BATCH_PROFILE
#define LSTM_BATCH_PROFILE 1
#endif

#ifndef LSTM_DIAG
#define LSTM_DIAG 1
#endif

#ifndef LSTM_DIAG_ONLY_FIRST_BATCH
#define LSTM_DIAG_ONLY_FIRST_BATCH 1
#endif

#include <cmath>
#include <cstdio>   // make sure this exists

// ============================
// Distribution Logging (3-class)
// ============================

static size_t epoch_actual[direction_output_size] = {0,0,0};
static size_t epoch_pred[direction_output_size]   = {0,0,0};
static size_t epoch_conf[direction_output_size][direction_output_size] = {{0}};
static size_t epoch_total = 0;
static size_t epoch_correct = 0;

void EA::LSTM::PrintMatrixSummary(const char* label,
                               const EA::LSTM::EAMatrix& m,
                               size_t maxPrint = 16)
{
    MetaNN::NSMetalMatMul::WaitForAll();
    auto ev = MetaNN::Evaluate(m);
    auto low = MetaNN::LowerAccess(ev);

    const float* p = low.RawMemory();
    const size_t rows = ev.Shape()[0];
    const size_t cols = ev.Shape()[1];
    const size_t n = rows * cols;

    if (n == 0)
    {
        std::cout << label << ": empty\n";
        return;
    }

    double sum = 0.0;
    double sumsq = 0.0;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    size_t nz = 0;

    for (size_t i = 0; i < n; ++i)
    {
        const float v = p[i];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
        sumsq += static_cast<double>(v) * static_cast<double>(v);
        if (std::fabs(v) > 1e-12f)
            ++nz;
    }

    const double mean = sum / static_cast<double>(n);
    const double var = std::max(0.0, sumsq / static_cast<double>(n) - mean * mean);
    const double stdv = std::sqrt(var);

    std::cout << label
              << " shape=(" << rows << "," << cols << ")"
              << " min=" << mn
              << " max=" << mx
              << " mean=" << mean
              << " std=" << stdv
              << " nz=" << nz << "/" << n
              << " first=";

    const size_t k = std::min(n, maxPrint);
    for (size_t i = 0; i < k; ++i)
    {
        if (i)
            std::cout << ",";
        std::cout << p[i];
    }
    std::cout << "\n";
}
static void Log3ClassSample(int actual, int predicted)
{
    if (actual >=0 && actual < static_cast<int>(direction_output_size)) epoch_actual[actual]++;
    if (predicted >=0 && predicted < static_cast<int>(direction_output_size)) epoch_pred[predicted]++;

    if (actual >=0 && actual < static_cast<int>(direction_output_size) &&
        predicted >=0 && predicted < static_cast<int>(direction_output_size))
        epoch_conf[actual][predicted]++;

    epoch_total++;
    if (actual == predicted) epoch_correct++;
}

void PrintAndResetDistribution()
{
    if (epoch_total == 0) return;

    auto frac = [](size_t x, size_t t){ return t ? (double)x / (double)t : 0.0; };

    printf("EPOCH_3CLASS_ACTUAL_DISTRIBUTION total=%zu down=%.4f neutral=%.4f up=%.4f\n",
           epoch_total,
           frac(epoch_actual[0], epoch_total),
           frac(epoch_actual[1], epoch_total),
           frac(epoch_actual[2], epoch_total));

    printf("EPOCH_3CLASS_PRED_DISTRIBUTION total=%zu down=%.4f neutral=%.4f up=%.4f\n",
           epoch_total,
           frac(epoch_pred[0], epoch_total),
           frac(epoch_pred[1], epoch_total),
           frac(epoch_pred[2], epoch_total));

    printf("EPOCH_3CLASS_CONFUSION_MATRIX rows=actual cols=predicted\n");
    for (size_t i = 0; i < direction_output_size; ++i)
    {
        printf("row%d: %zu %zu %zu\n",
               i,
               epoch_conf[i][0],
               epoch_conf[i][1],
               epoch_conf[i][2]);
    }

    printf("EPOCH_3CLASS_ACCURACY correct=%zu total=%zu acc=%.4f\n",
           epoch_correct,
           epoch_total,
           frac(epoch_correct, epoch_total));

    // reset
    for (size_t i = 0; i < direction_output_size; ++i)
    {
        epoch_actual[i]=0;
        epoch_pred[i]=0;
        for (size_t j = 0; j < direction_output_size; ++j) epoch_conf[i][j]=0;
    }
    epoch_total = 0;
    epoch_correct = 0;
}

// Frobenius norm of the difference between two matrices, evaluated on host
template <typename Mat>
static double FroNormDeltaHost(const Mat& a, const Mat& b)
{
    // Ensure any queued GPU work is finished before host reads
    MetaNN::NSMetalMatMul::WaitForAll();

    auto ea = MetaNN::Evaluate(a);
    auto eb = MetaNN::Evaluate(b);
    auto la = MetaNN::LowerAccess(ea);
    auto lb = MetaNN::LowerAccess(eb);

    const auto* pa = la.RawMemory();
    const auto* pb = lb.RawMemory();

    const size_t n = static_cast<size_t>(a.Shape()[0]) * static_cast<size_t>(a.Shape()[1]);
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(a.Shape()[0] == b.Shape()[0] && a.Shape()[1] == b.Shape()[1], "FroNormDeltaHost: shape mismatch");
#endif
    long double acc = 0.0L;
    for (size_t i = 0; i < n; ++i)
    {
        const long double dv = static_cast<long double>(pa[i]) - static_cast<long double>(pb[i]);
        acc += dv * dv;
    }
    return std::sqrt(static_cast<double>(acc));
}
template <typename Mat>
static double FroNormEvalHost(const Mat& m)
{
    // Make sure any queued GPU work is finished before we read host-visible memory.
    MetaNN::NSMetalMatMul::WaitForAll();
    auto ev = MetaNN::Evaluate(m);
    auto low = MetaNN::LowerAccess(ev);
    const auto* p = low.RawMemory();
    const size_t n = ev.Shape()[0] * ev.Shape()[1];
    long double acc = 0.0L;
    for (size_t i = 0; i < n; ++i)
    {
        const long double v = static_cast<long double>(p[i]);
        acc += v * v;
    }
    return std::sqrt(static_cast<double>(acc));
}


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
#define LSTM_HEAD_LR_MULT 1.0f
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

#ifndef LSTM_RET_HORIZON_1
#define LSTM_RET_HORIZON_1 1
#endif
#ifndef LSTM_RET_HORIZON_4
#define LSTM_RET_HORIZON_4 1
#endif
#ifndef LSTM_RET_HORIZON_8
#define LSTM_RET_HORIZON_8 1
#endif
#ifndef LSTM_RET_HORIZON_16
#define LSTM_RET_HORIZON_16 1
#endif

constexpr size_t kReturnFeatureCount =
    static_cast<size_t>(LSTM_RET_HORIZON_1) +
    static_cast<size_t>(LSTM_RET_HORIZON_4) +
    static_cast<size_t>(LSTM_RET_HORIZON_8) +
    static_cast<size_t>(LSTM_RET_HORIZON_16);

#ifndef MINI_BATCH_WINDOWS
#define MINI_BATCH_WINDOWS 512
#endif
#ifndef LSTM_MAX_MINI_BATCH_WINDOWS
#define LSTM_MAX_MINI_BATCH_WINDOWS 128
#endif

constexpr size_t mini_batch_windows = MINI_BATCH_WINDOWS;

const size_t effectiveMiniBatchWindows = std::max<size_t>(1, std::min<size_t>(mini_batch_windows, static_cast<size_t>(LSTM_MAX_MINI_BATCH_WINDOWS)));
struct LSTMScopedProfileTimer
{
    std::chrono::steady_clock::time_point t0;
    double& accum_us;

    explicit LSTMScopedProfileTimer(double& dst)
        : t0(std::chrono::steady_clock::now()), accum_us(dst)
    {}

    ~LSTMScopedProfileTimer()
    {
        const auto t1 = std::chrono::steady_clock::now();
        accum_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
};


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
struct EA::LSTM::GateBlocks
{
    EA::LSTM::EAMatrix W_i, W_f, W_g, W_o; // individual recurrent gate blocks (H x H)
    EA::LSTM::EAMatrix W_h_cat;            // full recurrent block (H x 4H)
};
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

struct EA::LSTM::LSTMBatchProfile
{
    double build_window_batch_us = 0.0;
    double forward_step_batches_us = 0.0;
    double concat_cols_us = 0.0;
    double dot_plus_bias_us = 0.0;
    double head_affine_us = 0.0;
    size_t mini_batches = 0;
};

#ifndef LSTM_EPOCH_BUCKETS
#define LSTM_EPOCH_BUCKETS 1
#endif

namespace {
#if LSTM_EPOCH_BUCKETS
// 3-class epoch aggregation counters

static size_t g_epoch_3class_down_count = 0;
static size_t g_epoch_3class_neutral_count = 0;
static size_t g_epoch_3class_up_count = 0;
static size_t g_epoch_3class_bucket_33_35_total = 0;
static size_t g_epoch_3class_bucket_33_35_correct = 0;
static size_t g_epoch_3class_bucket_35_40_total = 0;
static size_t g_epoch_3class_bucket_35_40_correct = 0;

static size_t g_epoch_3class_bucket_40_45_total = 0;
static size_t g_epoch_3class_bucket_40_45_correct = 0;
static size_t g_epoch_3class_bucket_45_50_total = 0;
static size_t g_epoch_3class_bucket_45_50_correct = 0;
static size_t g_epoch_3class_bucket_50_55_total = 0;
static size_t g_epoch_3class_bucket_50_55_correct = 0;
static size_t g_epoch_3class_bucket_55_60_total = 0;
static size_t g_epoch_3class_bucket_55_60_correct = 0;
static size_t g_epoch_3class_bucket_60_70_total = 0;
static size_t g_epoch_3class_bucket_60_70_correct = 0;
static size_t g_epoch_3class_bucket_70p_total = 0;
static size_t g_epoch_3class_bucket_70p_correct = 0;


inline void AccumulateEpochBuckets3Class(size_t down_count, size_t neutral_count, size_t up_count,
                                         size_t bucket_33_35_total, size_t bucket_33_35_correct,
                                         size_t bucket_35_40_total, size_t bucket_35_40_correct,
                                         size_t b40t, size_t b40c, size_t b45t, size_t b45c,
                                         size_t b50t, size_t b50c, size_t b55t, size_t b55c,
                                         size_t b60t, size_t b60c, size_t b70t, size_t b70c)
{
    g_epoch_3class_down_count += down_count;
    g_epoch_3class_neutral_count += neutral_count;
    g_epoch_3class_up_count += up_count;
    g_epoch_3class_bucket_33_35_total += bucket_33_35_total;
    g_epoch_3class_bucket_33_35_correct += bucket_33_35_correct;
    g_epoch_3class_bucket_35_40_total += bucket_35_40_total;
    g_epoch_3class_bucket_35_40_correct += bucket_35_40_correct;
    g_epoch_3class_bucket_40_45_total += b40t;
    g_epoch_3class_bucket_40_45_correct += b40c;
    g_epoch_3class_bucket_45_50_total += b45t;
    g_epoch_3class_bucket_45_50_correct += b45c;
    g_epoch_3class_bucket_50_55_total += b50t;
    g_epoch_3class_bucket_50_55_correct += b50c;
    g_epoch_3class_bucket_55_60_total += b55t;
    g_epoch_3class_bucket_55_60_correct += b55c;
    g_epoch_3class_bucket_60_70_total += b60t;
    g_epoch_3class_bucket_60_70_correct += b60c;
    g_epoch_3class_bucket_70p_total += b70t;
    g_epoch_3class_bucket_70p_correct += b70c;
}
#endif
} // end anonymous namespace


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
    EA::LSTM::EAMatrix W_xh; // full (n_in + H) x (4H)
    EA::LSTM::EAMatrix W_x;  // top block (n_in x 4H)
    EA::LSTM::EAMatrix W_h;  // bottom block (H x 4H)
};

inline double L2Norm(const MetaNN::Matrix<float, MetaNN::DeviceTags::Metal>& M)
{
    auto low = MetaNN::LowerAccess(M);
    const float* p = low.RawMemory();
    const size_t n = M.Shape()[0] * M.Shape()[1];
    double s = 0.0;
    for (size_t i = 0; i < n; ++i)
        s += static_cast<double>(p[i]) * static_cast<double>(p[i]);
    return std::sqrt(s);
}

inline double L2NormAccum(const MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>& M)
{
    auto low = MetaNN::LowerAccess(M);
    const AccumScalar* p = low.RawMemory();
    const size_t n = M.Shape()[0] * M.Shape()[1];
    double s = 0.0;
    for (size_t i = 0; i < n; ++i)
        s += static_cast<double>(p[i]) * static_cast<double>(p[i]);
    return std::sqrt(s);
}

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
    std::vector<EAMatrix> packed_steps; // size = window_size, each matrix is (B, F)
    std::vector<float> targets;
    std::vector<int> classTargets;
    std::vector<float> close_t;
    std::vector<float> close_target;
};

struct EA::LSTM::ForwardBatchScratch
{
    EA::LSTM::EAMatrix gates_batch;          // (B, 4H)
    EA::LSTM::EAMatrix gate_i_batch;          // (B, H)
    EA::LSTM::EAMatrix gate_f_batch;          // (B, H)
    EA::LSTM::EAMatrix gate_g_batch;          // (B, H)
    EA::LSTM::EAMatrix gate_o_batch;          // (B, H)
    EA::LSTM::EAMatrix c;          // (B, H)
    EA::LSTM::EAMatrix h;          // (B, H)

    ForwardBatchScratch() : gates_batch(1,1), gate_i_batch(1,1), gate_f_batch(1,1), gate_g_batch(1,1), gate_o_batch(1,1), c(1,1), h(1,1)  {}
};

// Member function definitions moved to EA::LSTM

void EA::LSTM::PrintOutputHeadShapes() const
{
    std::cout << "[LSTM] returnHeadWeight shape: ("
              << returnHeadWeight.Shape()[0] << ", "
              << returnHeadWeight.Shape()[1] << ")" << std::endl;

    std::cout << "[LSTM] returnHeadBias shape: ("
              << returnHeadBias.Shape()[0] << ", "
              << returnHeadBias.Shape()[1] << ")" << std::endl;

    std::cout << "[LSTM] returnHeadDirWeight shape: ("
              << returnHeadDirWeight.Shape()[0] << ", "
              << returnHeadDirWeight.Shape()[1] << ")" << std::endl;

    std::cout << "[LSTM] returnHeadDirBias shape: ("
              << returnHeadDirBias.Shape()[0] << ", "
              << returnHeadDirBias.Shape()[1] << ")" << std::endl;
}

std::array<float, direction_output_size> EA::LSTM::PredictNextDirectionProbs(const Window& w, bool resetState)
{
    if (resetState)
        ResetPreviousState();

    auto ww = hoistWindowWeights();
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> xh_concat_row(1, static_cast<size_t>(n_in + hidden_size));
    const size_t baseFeatureCount = (w.begin() != w.end()) ? static_cast<size_t>((*w.begin()).Shape()[1]) : 0;
    const size_t modelFeatureCount = static_cast<size_t>(n_in);
    const bool useReturnFeatures = (kReturnFeatureCount > 0);
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(modelFeatureCount == baseFeatureCount + kReturnFeatureCount,
                "PredictNextDirectionProbs: model input width must equal base features + enabled return features");
#endif
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> model_row(1, modelFeatureCount);

    size_t rowIdx = 0;
    for (const auto& f_sample : w)
    {
        auto lowSrc = MetaNN::LowerAccess(f_sample);
        const float* src = lowSrc.RawMemory();

        auto lowDst = MetaNN::LowerAccess(model_row);
        float* dst = lowDst.MutableRawMemory();

        std::memcpy(dst, src, baseFeatureCount * sizeof(float));
        if (useReturnFeatures)
        {
            const size_t appended = AppendMultiHorizonReturnFeatures(w, rowIdx, dst, baseFeatureCount);
#if LSTM_TRAINING_ASSERTS
            LSTM_ASSERT(appended == kReturnFeatureCount,
                        "PredictNextDirectionProbs: appended return feature count mismatch");
#endif
        }

        forwardStep(model_row, ww, bias, prevHiddenState, prevCellState, xh_concat_row);
        ++rowIdx;
    }

    auto logits = MetaNN::Dot(prevHiddenState, returnHeadDirWeight) + returnHeadDirBias;
    static bool printed_head_shapes = false;
    if (!printed_head_shapes)
    {
        PrintOutputHeadShapes();
        printed_head_shapes = true;
    }
    auto predH = logits.EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();

    const auto& z = predH.Data();
    float zz[direction_output_size] = { z(0, 0), z(0, 1), z(0, 2) };
    float p[direction_output_size];
    Softmax3(zz, p);

    return { p[0], p[1], p[2] };
}

int EA::LSTM::PredictNextDirectionClass(const Window& w, bool resetState)
{
    const auto p = PredictNextDirectionProbs(w, resetState);
    return (p[0] > p[1] && p[0] > p[2]) ? 0 : ((p[2] > p[1] && p[2] > p[0]) ? 2 : 1);
}

void EA::LSTM::PrintAndResetEpochBuckets()
{
    auto bucketRate = [](size_t up, size_t total) -> double
    {
        return (total > 0) ? (static_cast<double>(up) / static_cast<double>(total)) : 0.0;
    };

    std::cout << "EPOCH_3CLASS_COUNTS\n";
    std::cout << "down_count=" << g_epoch_3class_down_count
              << " neutral_count=" << g_epoch_3class_neutral_count
              << " up_count=" << g_epoch_3class_up_count << "\n";
    std::cout << "3class_prob_bucket[0.33,0.35): total=" << g_epoch_3class_bucket_33_35_total
              << " correct=" << g_epoch_3class_bucket_33_35_correct
              << " acc=" << bucketRate(g_epoch_3class_bucket_33_35_correct, g_epoch_3class_bucket_33_35_total)
              << "\n";
    std::cout << "3class_prob_bucket[0.35,0.40): total=" << g_epoch_3class_bucket_35_40_total
              << " correct=" << g_epoch_3class_bucket_35_40_correct
              << " acc=" << bucketRate(g_epoch_3class_bucket_35_40_correct, g_epoch_3class_bucket_35_40_total)
                  << "\n";
    std::cout << "3class_prob_bucket[0.40,0.45): total=" << g_epoch_3class_bucket_40_45_total
              << " correct=" << g_epoch_3class_bucket_40_45_correct
              << " acc=" << bucketRate(g_epoch_3class_bucket_40_45_correct, g_epoch_3class_bucket_40_45_total)
              << "\n";
    std::cout << "3class_prob_bucket[0.45,0.50): total=" << g_epoch_3class_bucket_45_50_total
              << " correct=" << g_epoch_3class_bucket_45_50_correct
              << " acc=" << bucketRate(g_epoch_3class_bucket_45_50_correct, g_epoch_3class_bucket_45_50_total)
              << "\n";
    std::cout << "3class_prob_bucket[0.50,0.55): total=" << g_epoch_3class_bucket_50_55_total
              << " correct=" << g_epoch_3class_bucket_50_55_correct
              << " acc=" << bucketRate(g_epoch_3class_bucket_50_55_correct, g_epoch_3class_bucket_50_55_total)
              << "\n";
    std::cout << "3class_prob_bucket[0.55,0.60): total=" << g_epoch_3class_bucket_55_60_total
              << " correct=" << g_epoch_3class_bucket_55_60_correct
              << " acc=" << bucketRate(g_epoch_3class_bucket_55_60_correct, g_epoch_3class_bucket_55_60_total)
              << "\n";
    std::cout << "3class_prob_bucket[0.60,0.70): total=" << g_epoch_3class_bucket_60_70_total
              << " correct=" << g_epoch_3class_bucket_60_70_correct
              << " acc=" << bucketRate(g_epoch_3class_bucket_60_70_correct, g_epoch_3class_bucket_60_70_total)
              << "\n";
    std::cout << "3class_prob_bucket[0.70,1.00]: total=" << g_epoch_3class_bucket_70p_total
              << " correct=" << g_epoch_3class_bucket_70p_correct
              << " acc=" << bucketRate(g_epoch_3class_bucket_70p_correct, g_epoch_3class_bucket_70p_total)
              << "\n";

    // Reset counters
    g_epoch_3class_down_count = 0;
    g_epoch_3class_neutral_count = 0;
    g_epoch_3class_up_count = 0;
    g_epoch_3class_bucket_33_35_total = 0;
    g_epoch_3class_bucket_33_35_correct = 0;
    g_epoch_3class_bucket_35_40_total = 0;
    g_epoch_3class_bucket_35_40_correct = 0;
    g_epoch_3class_bucket_40_45_total = 0;
    g_epoch_3class_bucket_40_45_correct = 0;
    g_epoch_3class_bucket_45_50_total = 0;
    g_epoch_3class_bucket_45_50_correct = 0;
    g_epoch_3class_bucket_50_55_total = 0;
    g_epoch_3class_bucket_50_55_correct = 0;
    g_epoch_3class_bucket_55_60_total = 0;
    g_epoch_3class_bucket_55_60_correct = 0;
    g_epoch_3class_bucket_60_70_total = 0;
    g_epoch_3class_bucket_60_70_correct = 0;
    g_epoch_3class_bucket_70p_total = 0;
    g_epoch_3class_bucket_70p_correct = 0;
}

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

inline auto EA::LSTM::BuildHeadDhBatch3Class(const EAMatrix& d_logits_batch,
                                             const EAMatrix& headW,
                                             float scale) const -> EAMatrix
{
    auto expr = MetaNN::Dot(d_logits_batch, MetaNN::Transpose(headW));
    auto h = expr.EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();

    EAMatrix d_h = NNUtils::DeepCopyMatrix(h.Data());
    auto lowDh = MetaNN::LowerAccess(d_h);
    float* dhp = lowDh.MutableRawMemory();
    const size_t n = d_h.Shape()[0] * d_h.Shape()[1];
    for (size_t i = 0; i < n; ++i)
        dhp[i] *= scale;
    return d_h;
}

inline void EA::LSTM::AccumulateHeadGradsBatch3Class(EAMatrix& dW_accum,
                                                     EAMatrix& dB_accum,
                                                     const EAMatrix& h_batch,
                                                     const EAMatrix& d_logits_batch) const
{
    const size_t B = h_batch.Shape()[0];

    EAMatrix ones_col(B, 1);
    {
        auto lowOnes = MetaNN::LowerAccess(ones_col);
        std::fill(lowOnes.MutableRawMemory(), lowOnes.MutableRawMemory() + B, 1.0f);
    }

    auto dW_expr = MetaNN::Dot(MetaNN::Transpose(h_batch), d_logits_batch);
    auto dB_expr = MetaNN::Dot(MetaNN::Transpose(ones_col), d_logits_batch);

    auto dWH = dW_expr.EvalRegister();
    auto dBH = dB_expr.EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();

    {
        auto lowDst = MetaNN::LowerAccess(dW_accum);
        auto lowSrc = MetaNN::LowerAccess(dWH.Data());
        float* dst = lowDst.MutableRawMemory();
        const float* src = lowSrc.RawMemory();
        const size_t n = dW_accum.Shape()[0] * dW_accum.Shape()[1];
        for (size_t i = 0; i < n; ++i)
            dst[i] += src[i];
    }
    {
        auto lowDst = MetaNN::LowerAccess(dB_accum);
        auto lowSrc = MetaNN::LowerAccess(dBH.Data());
        float* dst = lowDst.MutableRawMemory();
        const float* src = lowSrc.RawMemory();
        const size_t n = dB_accum.Shape()[0] * dB_accum.Shape()[1];
        for (size_t i = 0; i < n; ++i)
            dst[i] += src[i];
    }
}

inline void EA::LSTM::AccumulateHeadGradsBatch(EAMatrix& dW_accum,
                                               EAMatrix& dB_accum,
                                               const EAMatrix& h_batch,
                                               const std::vector<float>& errs) const
{
    const size_t B = h_batch.Shape()[0];
    const size_t H = h_batch.Shape()[1];

    auto lowH = MetaNN::LowerAccess(h_batch);
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
    const auto W_x = NNUtils::ViewTopRows<float, MetaNN::DeviceTags::Metal>(param, param.Shape()[0] - hidden_size);
    const auto W_h = NNUtils::ViewBottomRows<float, MetaNN::DeviceTags::Metal>(param, hidden_size);
    const auto W_xh = param; // full (n_in + H) x (4H) matrix; dynamic row views taken later
    return WindowWeights{ W_xh, W_x, W_h };
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

inline void EA::LSTM::RepeatRowsInto(EAMatrix& out, const EAMatrix& row, size_t B) const
{
    const size_t cols = row.Shape()[1];
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(out.Shape()[0] == B, "RepeatRowsInto: output row count mismatch");
    LSTM_ASSERT(out.Shape()[1] == cols, "RepeatRowsInto: output col count mismatch");
#endif

    auto rowEval = MetaNN::Evaluate(row);
    auto lowRow = MetaNN::LowerAccess(rowEval);
    const float* src = lowRow.RawMemory();

    auto lowOut = MetaNN::LowerAccess(out);
    float* dst = lowOut.MutableRawMemory();

    for (size_t b = 0; b < B; ++b)
        std::copy(src, src + cols, dst + b * cols);
}

inline float EA::LSTM::ComputeLookbackLogReturn(const Window& batch,
                                                size_t rowIdx,
                                                size_t lookbackBars) const
{
    if (lookbackBars == 0 || rowIdx < lookbackBars)
        return 0.0f;

    const auto curIt  = batch.begin() + static_cast<std::ptrdiff_t>(rowIdx);
    const auto prevIt = batch.begin() + static_cast<std::ptrdiff_t>(rowIdx - lookbackBars);

    const float curClose  = t.RawCloseAtIterator(curIt);
    const float prevClose = t.RawCloseAtIterator(prevIt);

    if (!std::isfinite(curClose) || !std::isfinite(prevClose) || curClose <= 0.0f || prevClose <= 0.0f)
        return 0.0f;

    return std::log(curClose / prevClose);
}

inline size_t EA::LSTM::AppendMultiHorizonReturnFeatures(const Window& batch,
                                                         size_t rowIdx,
                                                         float* dst,
                                                         size_t dstOffset) const
{
    size_t written = 0;
#if LSTM_RET_HORIZON_1
    dst[dstOffset + written] = ComputeLookbackLogReturn(batch, rowIdx, 1) * EA::LSTM::kFeatScale;
    ++written;
#endif
#if LSTM_RET_HORIZON_4
    dst[dstOffset + written] = ComputeLookbackLogReturn(batch, rowIdx, 4) * EA::LSTM::kFeatScale;
    ++written;
#endif
#if LSTM_RET_HORIZON_8
    dst[dstOffset + written] = ComputeLookbackLogReturn(batch, rowIdx, 8) * EA::LSTM::kFeatScale;
    ++written;
#endif
#if LSTM_RET_HORIZON_16
    dst[dstOffset + written] = ComputeLookbackLogReturn(batch, rowIdx, 16) * EA::LSTM::kFeatScale;
    ++written;
#endif
    return written;
}

// Run gate activations and recurrent state updates through a single fused Metal
// kernel instead of building multiple MetaNN expressions. The previous version
// still required separate sigmoid/tanh/update expressions plus an EvalPlan sync,
// which can fan out into multiple kernel launches and temporary tensors.
//
// The fused path must do all of the following in one pass over each (b, h):
//   i = sigmoid(gates[b, h + 0*H])
//   f = sigmoid(gates[b, h + 1*H])
//   g = tanh   (gates[b, h + 2*H])
//   o = sigmoid(gates[b, h + 3*H])
//   c = f * prev_c + i * g
//   h = o * tanh(c)
// and write i/f/g/o/c/h directly to the output buffers needed by the next
// timestep and backward pass.
inline void EA::LSTM::ComputeGateStateBatchFromContiguous(const EAMatrix& gates_batch,
                                                          const EAMatrix& prevCellState,
                                                          EAMatrix& gate_i_batch,
                                                          EAMatrix& gate_f_batch,
                                                          EAMatrix& gate_g_batch,
                                                          EAMatrix& gate_o_batch,
                                                          EAMatrix& c_batch,
                                                          EAMatrix& h_batch) const
{
    const size_t B = gates_batch.Shape()[0];
    const size_t W = gates_batch.Shape()[1];
    const size_t H = W / 4;

#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(W % 4 == 0, "ComputeGateStateBatchFromContiguous: gates width must be divisible by 4");
    LSTM_ASSERT(prevCellState.Shape()[0] == B && prevCellState.Shape()[1] == H,
                "ComputeGateStateBatchFromContiguous: prevCellState shape mismatch");
    LSTM_ASSERT(gate_i_batch.Shape()[0] == B && gate_i_batch.Shape()[1] == H,
                "ComputeGateStateBatchFromContiguous: gate_i_batch shape mismatch");
    LSTM_ASSERT(gate_f_batch.Shape()[0] == B && gate_f_batch.Shape()[1] == H,
                "ComputeGateStateBatchFromContiguous: gate_f_batch shape mismatch");
    LSTM_ASSERT(gate_g_batch.Shape()[0] == B && gate_g_batch.Shape()[1] == H,
                "ComputeGateStateBatchFromContiguous: gate_g_batch shape mismatch");
    LSTM_ASSERT(gate_o_batch.Shape()[0] == B && gate_o_batch.Shape()[1] == H,
                "ComputeGateStateBatchFromContiguous: gate_o_batch shape mismatch");
    LSTM_ASSERT(c_batch.Shape()[0] == B && c_batch.Shape()[1] == H,
                "ComputeGateStateBatchFromContiguous: c_batch shape mismatch");
    LSTM_ASSERT(h_batch.Shape()[0] == B && h_batch.Shape()[1] == H,
                "ComputeGateStateBatchFromContiguous: h_batch shape mismatch");
#endif
    auto run_cpu_reference = [&](EAMatrix& out_i,
                                 EAMatrix& out_f,
                                 EAMatrix& out_g,
                                 EAMatrix& out_o,
                                 EAMatrix& out_c,
                                 EAMatrix& out_h)
    {
        auto lowGatesRef = MetaNN::LowerAccess(gates_batch);
        auto lowPrevCRef = MetaNN::LowerAccess(prevCellState);
        auto lowIRef = MetaNN::LowerAccess(out_i);
        auto lowFRef = MetaNN::LowerAccess(out_f);
        auto lowGRef = MetaNN::LowerAccess(out_g);
        auto lowORef = MetaNN::LowerAccess(out_o);
        auto lowCRef = MetaNN::LowerAccess(out_c);
        auto lowHRef = MetaNN::LowerAccess(out_h);

        const float* gatesPtr = lowGatesRef.RawMemory();
        const float* prevCPtr = lowPrevCRef.RawMemory();
        float* iPtr = lowIRef.MutableRawMemory();
        float* fPtr = lowFRef.MutableRawMemory();
        float* gPtr = lowGRef.MutableRawMemory();
        float* oPtr = lowORef.MutableRawMemory();
        float* cPtr = lowCRef.MutableRawMemory();
        float* hPtr = lowHRef.MutableRawMemory();

        auto sigmoid = [](float x) -> float
        {
            return 1.0f / (1.0f + std::exp(-x));
        };

        for (size_t b = 0; b < B; ++b)
        {
            const size_t gateBase = b * (4 * H);
            const size_t stateBase = b * H;
            for (size_t h = 0; h < H; ++h)
            {
                const float i_val = sigmoid(gatesPtr[gateBase + 0 * H + h]);
                const float f_val = sigmoid(gatesPtr[gateBase + 1 * H + h]);
                const float g_val = std::tanh(gatesPtr[gateBase + 2 * H + h]);
                const float o_val = sigmoid(gatesPtr[gateBase + 3 * H + h]);
                const float c_prev = prevCPtr[stateBase + h];
                const float c_val = f_val * c_prev + i_val * g_val;
                const float h_val = o_val * std::tanh(c_val);

                iPtr[stateBase + h] = i_val;
                fPtr[stateBase + h] = f_val;
                gPtr[stateBase + h] = g_val;
                oPtr[stateBase + h] = o_val;
                cPtr[stateBase + h] = c_val;
                hPtr[stateBase + h] = h_val;
            }
        }
    };

#if LSTM_GATESTATE_MODE == 0
    run_cpu_reference(gate_i_batch,
                      gate_f_batch,
                      gate_g_batch,
                      gate_o_batch,
                      c_batch,
                      h_batch);
#elif LSTM_GATESTATE_MODE == 1
    EAMatrix gate_i_ref(B, H);
    EAMatrix gate_f_ref(B, H);
    EAMatrix gate_g_ref(B, H);
    EAMatrix gate_o_ref(B, H);
    EAMatrix c_ref(B, H);
    EAMatrix h_ref(B, H);

    run_cpu_reference(gate_i_ref,
                      gate_f_ref,
                      gate_g_ref,
                      gate_o_ref,
                      c_ref,
                      h_ref);

    {
        auto lowGates = MetaNN::LowerAccess(gates_batch);
        auto lowPrevC = MetaNN::LowerAccess(prevCellState);
        auto lowI = MetaNN::LowerAccess(gate_i_batch);
        auto lowF = MetaNN::LowerAccess(gate_f_batch);
        auto lowG = MetaNN::LowerAccess(gate_g_batch);
        auto lowO = MetaNN::LowerAccess(gate_o_batch);
        auto lowC = MetaNN::LowerAccess(c_batch);
        auto lowH = MetaNN::LowerAccess(h_batch);

        auto gatesMem = lowGates.SharedMemory();
        auto prevCMem = lowPrevC.SharedMemory();
        auto iMem = lowI.SharedMemory();
        auto fMem = lowF.SharedMemory();
        auto gMem = lowG.SharedMemory();
        auto oMem = lowO.SharedMemory();
        auto cMem = lowC.SharedMemory();
        auto hMem = lowH.SharedMemory();

        MetaNN::NSMetalMatMul::GateStateFused(
            gatesMem,
            prevCMem,
            iMem,
            fMem,
            gMem,
            oMem,
            cMem,
            hMem,
            B,
            H);
    }
    MetaNN::NSMetalMatMul::WaitForAll();

    auto validate_same = [&](const char* name, const EAMatrix& actual, const EAMatrix& expected)
    {
        auto lowActual = MetaNN::LowerAccess(actual);
        auto lowExpected = MetaNN::LowerAccess(expected);
        const float* aptr = lowActual.RawMemory();
        const float* eptr = lowExpected.RawMemory();
        const size_t n = actual.Shape()[0] * actual.Shape()[1];
        constexpr float absTol = 1e-5f;
        constexpr float relTol = 1e-4f;

        for (size_t idx = 0; idx < n; ++idx)
        {
            const float a = aptr[idx];
            const float e = eptr[idx];
            const float absDiff = std::fabs(a - e);
            const float relDiff = absDiff / std::max(1.0f, std::fabs(e));
            if (absDiff > absTol && relDiff > relTol)
            {
                std::ostringstream oss;
                oss << "ComputeGateStateBatchFromContiguous: " << name
                    << " mismatch at linear idx=" << idx
                    << " actual=" << a
                    << " expected=" << e
                    << " absDiff=" << absDiff
                    << " relDiff=" << relDiff;
                throw std::runtime_error(oss.str());
            }
        }
    };

    validate_same("gate_i", gate_i_batch, gate_i_ref);
    validate_same("gate_f", gate_f_batch, gate_f_ref);
    validate_same("gate_g", gate_g_batch, gate_g_ref);
    validate_same("gate_o", gate_o_batch, gate_o_ref);
    validate_same("c", c_batch, c_ref);
    validate_same("h", h_batch, h_ref);
#else
    {
        auto lowGates = MetaNN::LowerAccess(gates_batch);
        auto lowPrevC = MetaNN::LowerAccess(prevCellState);
        auto lowI = MetaNN::LowerAccess(gate_i_batch);
        auto lowF = MetaNN::LowerAccess(gate_f_batch);
        auto lowG = MetaNN::LowerAccess(gate_g_batch);
        auto lowO = MetaNN::LowerAccess(gate_o_batch);
        auto lowC = MetaNN::LowerAccess(c_batch);
        auto lowH = MetaNN::LowerAccess(h_batch);

        auto gatesMem = lowGates.SharedMemory();
        auto prevCMem = lowPrevC.SharedMemory();
        auto iMem = lowI.SharedMemory();
        auto fMem = lowF.SharedMemory();
        auto gMem = lowG.SharedMemory();
        auto oMem = lowO.SharedMemory();
        auto cMem = lowC.SharedMemory();
        auto hMem = lowH.SharedMemory();

        MetaNN::NSMetalMatMul::GateStateFused(
            gatesMem,
            prevCMem,
            iMem,
            fMem,
            gMem,
            oMem,
            cMem,
            hMem,
            B,
            H);
    }
#endif
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
                                       EAMatrix& xh_concat_batch,
                                       ForwardBatchScratch& scratch,
                                       LSTMBatchProfile* profile) const -> BatchStepCache
{
    const size_t B = x_t.Shape()[0];
    const size_t H = prevHiddenState.Shape()[1];
    const size_t expectedCols = x_t.Shape()[1] + H;
    const size_t gateCols = 4 * H;

    if (xh_concat_batch.Shape()[0] != B || xh_concat_batch.Shape()[1] != expectedCols)
        xh_concat_batch = EAMatrix(B, expectedCols);

    if (scratch.gates_batch.Shape()[0] != B || scratch.gates_batch.Shape()[1] != gateCols)
        scratch.gates_batch = EAMatrix(B, gateCols);
    if (scratch.gate_i_batch.Shape()[0] != B || scratch.gate_i_batch.Shape()[1] != H)
        scratch.gate_i_batch = EAMatrix(B, H);
    if (scratch.gate_f_batch.Shape()[0] != B || scratch.gate_f_batch.Shape()[1] != H)
        scratch.gate_f_batch = EAMatrix(B, H);
    if (scratch.gate_g_batch.Shape()[0] != B || scratch.gate_g_batch.Shape()[1] != H)
        scratch.gate_g_batch = EAMatrix(B, H);
    if (scratch.gate_o_batch.Shape()[0] != B || scratch.gate_o_batch.Shape()[1] != H)
        scratch.gate_o_batch = EAMatrix(B, H);
    if (scratch.c.Shape()[0] != B || scratch.c.Shape()[1] != H)
        scratch.c = EAMatrix(B, H);
    if (scratch.h.Shape()[0] != B || scratch.h.Shape()[1] != H)
        scratch.h = EAMatrix(B, H);
    // All gate/state outputs are kept as persistent batch buffers so the fused
    // Metal kernel can write directly into them without intermediate expression
    // materialization or host-visible staging.

    #if LSTM_BATCH_PROFILE
    if (profile)
    {
        LSTMScopedProfileTimer timer(profile->concat_cols_us);
        NNUtils::ConcatColsInto(xh_concat_batch, x_t, prevHiddenState);
    }
    else
    #endif
    {
        NNUtils::ConcatColsInto(xh_concat_batch, x_t, prevHiddenState);
    }

    const size_t K = xh_concat_batch.Shape()[1];
    auto W_cat_dyn = NNUtils::ViewRows<float, MetaNN::DeviceTags::Metal>(ww.W_xh, 0, K);

    {
#if LSTM_BATCH_PROFILE
        if (profile)
        {
            LSTMScopedProfileTimer timer(profile->dot_plus_bias_us);
            auto lowA = MetaNN::LowerAccess(xh_concat_batch);
            auto lowB = MetaNN::LowerAccess(W_cat_dyn);
            auto lowBias = MetaNN::LowerAccess(bias);
            auto lowY = MetaNN::LowerAccess(scratch.gates_batch);

            // Store shared memory views in local variables to avoid binding temporaries
            auto aMem = lowA.SharedMemory();
            auto bMem = lowB.SharedMemory();
            auto biasMem = lowBias.SharedMemory();
            auto yMem = lowY.SharedMemory();

            MetaNN::NSMetalMatMul::MatMulBias(
                aMem,
                bMem,
                biasMem,
                yMem,
                B, K, gateCols);
        }
        else
#endif
        {
            auto lowA = MetaNN::LowerAccess(xh_concat_batch);
            auto lowB = MetaNN::LowerAccess(W_cat_dyn);
            auto lowBias = MetaNN::LowerAccess(bias);
            auto lowY = MetaNN::LowerAccess(scratch.gates_batch);

            // Store shared memory views in local variables to avoid binding temporaries
            auto aMem = lowA.SharedMemory();
            auto bMem = lowB.SharedMemory();
            auto biasMem = lowBias.SharedMemory();
            auto yMem = lowY.SharedMemory();

            MetaNN::NSMetalMatMul::MatMulBias(
                aMem,
                bMem,
                biasMem,
                yMem,
                B, K, gateCols);
        }
    }

#if LSTM_DIAG
    static size_t s_forwardStepBatchCalls = 0;
    const bool diag_cond = (!LSTM_DIAG_ONLY_FIRST_BATCH || s_forwardStepBatchCalls++ == 0);
    if (diag_cond &&
        prevHiddenState.Shape()[0] > 0 &&
        prevHiddenState.Shape()[1] > 0)
    {
        static bool s_printed_fused_gate_diag = false;
        if (!s_printed_fused_gate_diag)
        {
            PrintMatrixSummary("DIAG_GATES_BATCH_PRE_FUSED", scratch.gates_batch);
            PrintMatrixSummary("DIAG_PREV_C_PRE_FUSED", prevCellState);
            PrintMatrixSummary("DIAG_PREV_H_PRE_FUSED", prevHiddenState);
            {
                const size_t Hdiag = scratch.gates_batch.Shape()[1] / 4;
                auto gate_i_logits = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(scratch.gates_batch, 0 * Hdiag, Hdiag);
                auto gate_f_logits = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(scratch.gates_batch, 1 * Hdiag, Hdiag);
                auto gate_g_logits = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(scratch.gates_batch, 2 * Hdiag, Hdiag);
                auto gate_o_logits = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(scratch.gates_batch, 3 * Hdiag, Hdiag);
                PrintMatrixSummary("DIAG_GATE_I_LOGITS_PRE_FUSED", gate_i_logits);
                PrintMatrixSummary("DIAG_GATE_F_LOGITS_PRE_FUSED", gate_f_logits);
                PrintMatrixSummary("DIAG_GATE_G_LOGITS_PRE_FUSED", gate_g_logits);
                PrintMatrixSummary("DIAG_GATE_O_LOGITS_PRE_FUSED", gate_o_logits);
            }
            s_printed_fused_gate_diag = true;
        }
    }
#endif
    ComputeGateStateBatchFromContiguous(scratch.gates_batch,prevCellState,scratch.gate_i_batch,scratch.gate_f_batch,scratch.gate_g_batch,scratch.gate_o_batch,scratch.c,scratch.h);
#if LSTM_GATESTATE_MODE == 2
    MetaNN::NSMetalMatMul::WaitForAll();    // Ensure all Metal writes are completed before deep copies
#endif
#if LSTM_DIAG
    if (diag_cond &&
        scratch.h.Shape()[0] > 0 &&
        scratch.h.Shape()[1] > 0)
    {
        static bool s_printed_fused_gate_post_diag = false;
        if (!s_printed_fused_gate_post_diag)
        {
            PrintMatrixSummary("DIAG_GATE_I_POST_FUSED", scratch.gate_i_batch);
            PrintMatrixSummary("DIAG_GATE_F_POST_FUSED", scratch.gate_f_batch);
            PrintMatrixSummary("DIAG_GATE_G_POST_FUSED", scratch.gate_g_batch);
            PrintMatrixSummary("DIAG_GATE_O_POST_FUSED", scratch.gate_o_batch);
            PrintMatrixSummary("DIAG_C_POST_FUSED", scratch.c);
            PrintMatrixSummary("DIAG_H_POST_FUSED", scratch.h);
            s_printed_fused_gate_post_diag = true;
        }
    }
#endif
    BatchStepCache sc
    {
        x_t,    // SHOULD THIS ALSO BE CLONED??
        NNUtils::DeepCopyMatrix(prevHiddenState),
        NNUtils::DeepCopyMatrix(prevCellState),
        NNUtils::DeepCopyMatrix(scratch.gate_i_batch),
        NNUtils::DeepCopyMatrix(scratch.gate_f_batch),
        NNUtils::DeepCopyMatrix(scratch.gate_g_batch),
        NNUtils::DeepCopyMatrix(scratch.gate_o_batch),
        NNUtils::DeepCopyMatrix(scratch.c),
        NNUtils::DeepCopyMatrix(scratch.h)
    };

    prevCellState = sc.c;
    prevHiddenState = sc.h;
    return sc;
}

inline void EA::LSTM::forwardStep(const EAMatrix& x_t,
                           const WindowWeights& ww,
                           const EAMatrix& bias,
                           EAMatrix& prevHiddenState,
                           EAMatrix& prevCellState,
                           EAMatrix& xh_concat_row) const
{
    // Build [x_t | h_{t-1}] and compute all gates in one GEMM
    {
        const size_t expectedCols = x_t.Shape()[1] + prevHiddenState.Shape()[1];
        if (xh_concat_row.Shape()[0] != 1 || xh_concat_row.Shape()[1] != expectedCols) {
            xh_concat_row = EAMatrix(1, expectedCols);
        }
        NNUtils::ConcatColsInto(xh_concat_row, x_t, prevHiddenState);
#if LSTM_DEBUG_PRINTS
        auto X = MetaNN::Evaluate(x_t);
        auto H = MetaNN::Evaluate(prevHiddenState);
        auto C = MetaNN::Evaluate(xh_concat_row);
        const size_t Ix = X.Shape()[1];
        const size_t Ih = H.Shape()[1];
        
        for (size_t j = 0; j < Ix; ++j)
            LSTM_ASSERT(std::fabs(C(0, j) - X(0, j)) < 1e-6f, "Concat: X mismatch");
        
        for (size_t j = 0; j < Ih; ++j)
            LSTM_ASSERT(std::fabs(C(0, Ix + j) - H(0, j)) < 1e-6f, "Concat: H mismatch");
#endif
    }

    const size_t K = xh_concat_row.Shape()[1];
    auto W_cat_dyn = NNUtils::ViewRows<float, MetaNN::DeviceTags::Metal>(ww.W_xh, 0, K);
    const size_t H = prevHiddenState.Shape()[1];

    EAMatrix yMat(1, 4 * H);
    {
        auto lowA = MetaNN::LowerAccess(xh_concat_row);
        auto lowB = MetaNN::LowerAccess(W_cat_dyn);
        auto lowBias = MetaNN::LowerAccess(bias);
        auto lowY = MetaNN::LowerAccess(yMat);

        auto aMem = lowA.SharedMemory();
        auto bMem = lowB.SharedMemory();
        auto biasMem = lowBias.SharedMemory();
        auto yMem = lowY.SharedMemory();

        MetaNN::NSMetalMatMul::MatMulBias(
            aMem,
            bMem,
            biasMem,
            yMem,
            1, K, 4 * H);
    }
    MetaNN::NSMetalMatMul::WaitForAll();
#if LSTM_DEBUG_INTERNAL_PRINTS
    std::cout << "bias(0,64): " << bias(0,64) << std::endl
        << "yMat(0,64) : " << yMat(0,64) << std::endl
        << "yMat(0,0) : " << yMat(0,0) << std::endl
        << "yMat(0,128) : " << yMat(0,128) << std::endl
        << "yMat(0,192) : " << yMat(0,192) << std::endl;
#endif

    auto [i2D, f2D, g2D, o2D] = NNUtils::SplitGatesRowExpr(yMat);
#if LSTM_DEBUG_INTERNAL_PRINTS
{
    auto iHandle = i2D.EvalRegister();
    auto fHandle = f2D.EvalRegister();
    auto gHandle = g2D.EvalRegister();
    auto oHandle = o2D.EvalRegister();

    MetaNN::EvalPlan::Inst().Eval();

    std::cout
        << "y(0,0)="   << yMat(0,0)   << "  i(0,0)=" << iHandle.Data()(0,0) << "\n"
        << "y(0,64)="  << yMat(0,64)  << "  f(0,0)=" << fHandle.Data()(0,0) << "\n"
        << "y(0,128)=" << yMat(0,128) << "  g(0,0)=" << gHandle.Data()(0,0) << "\n"
        << "y(0,192)=" << yMat(0,192) << "  o(0,0)=" << oHandle.Data()(0,0) << "\n";
}
#endif
    auto i_1d = MetaNN::Sigmoid(MetaNN::Reshape(i2D, MetaNN::Shape(H)));
    auto f_1d = MetaNN::Sigmoid(MetaNN::Reshape(f2D, MetaNN::Shape(H)));
    auto g_1d = MetaNN::Tanh   (MetaNN::Reshape(g2D, MetaNN::Shape(H)));
    auto o_1d = MetaNN::Sigmoid(MetaNN::Reshape(o2D, MetaNN::Shape(H)));

    auto c_prev_1d = MetaNN::Reshape(prevCellState, MetaNN::Shape(H));
    auto c_1d = f_1d * c_prev_1d + i_1d * g_1d;
    auto h_1d = o_1d * MetaNN::Tanh(c_1d);



    auto c_2d_handle = MetaNN::Reshape(c_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto h_2d_handle = MetaNN::Reshape(h_1d, MetaNN::Shape(1, H)).EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();
    
#if LSTM_DEBUG_INTERNAL_PRINTS
    auto cprev_2d_handle = MetaNN::Reshape(c_prev_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto i_2d_handle = MetaNN::Reshape(i_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto f_2d_handle = MetaNN::Reshape(f_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto g_2d_handle = MetaNN::Reshape(g_1d, MetaNN::Shape(1, H)).EvalRegister();
    auto o_2d_handle = MetaNN::Reshape(o_1d, MetaNN::Shape(1, H)).EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();
    std::cout << "i_2d_handle.Data()(0,0): " << i_2d_handle.Data()(0,0) << std::endl;
    std::cout << "f_2d_handle.Data()(0,0): " << f_2d_handle.Data()(0,0) << std::endl;
    std::cout << "g_2d_handle.Data()(0,0): " << g_2d_handle.Data()(0,0) << std::endl;
    std::cout << "o_2d_handle.Data()(0,0): " << o_2d_handle.Data()(0,0) << std::endl;
    std::cout << "c_2d_handle.Data()(0,0): " << c_2d_handle.Data()(0,0) << std::endl;
    std::cout << "h_2d_handle.Data()(0,0): " << h_2d_handle.Data()(0,0) << std::endl;
#endif

    MetaNN::NSMetalMatMul::WaitForAll();
    prevCellState   = NNUtils::DeepCopyMatrix(c_2d_handle.Data());   // sc.c;
    prevHiddenState = NNUtils::DeepCopyMatrix(h_2d_handle.Data());   //sc.h;
}

inline auto EA::LSTM::predictAndLoss(const EAMatrix& h_T,
                             const EAMatrix& W,
                             const EAMatrix& b,
                             float target) const -> HeadLoss
{
    {
        auto logits = MetaNN::Dot(h_T, W) + b;
        auto predH = logits.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        float y_hat = predH.Data()(0, 0);
        float err   = y_hat - target;
        return { y_hat, err };
    }
}

auto EA::LSTM::predictAndLoss3Class(const EAMatrix& h_T, const EAMatrix& W, const EAMatrix& b, int targetClass) const -> HeadLoss3Class
{
    // Compute logits z = h_T · W + b (1x3)
    auto z = Dot(h_T, W) + b;
    auto zMat = Evaluate(z);
    LSTM_ASSERT(zMat.Shape()[0] == 1 && zMat.Shape()[1] == direction_output_size, "predictAndLoss3Class: expected 1x3 logits");
    const float z0 = zMat(0,0), z1 = zMat(0,1), z2 = zMat(0,2);

    // Softmax probabilities
    float p[direction_output_size];
    float zArr[direction_output_size] = { z0, z1, z2 };
    Softmax3(zArr, p);

    // One-hot target
    float y[direction_output_size] = {0.f, 0.f, 0.f};
    if (targetClass >= 0 && targetClass < static_cast<int>(direction_output_size)) y[targetClass] = 1.f;

    // Per-class cross-entropy components (unweighted)
    // loss_k = - y_k * log(max(p_k, eps))
    constexpr float eps = 1e-6f;
    float lossDown    = - y[0] * std::log(std::max(p[0], eps));
    float lossNeutral = - y[1] * std::log(std::max(p[1], eps));
    float lossUp      = - y[2] * std::log(std::max(p[2], eps));

    // Apply class weighting from Params.hpp
    float L = weighted_direction_loss(lossDown, lossNeutral, lossUp);

    // Gradient of loss w.r.t logits: w_k * (p_k - y_k)
    float wDown = kClassWeightDown;
    float wNeutral = kClassWeightNeutral;
    float wUp = kClassWeightUp;
    float dL_dz0 = wDown    * (p[0] - y[0]);
    float dL_dz1 = wNeutral * (p[1] - y[1]);
    float dL_dz2 = wUp      * (p[2] - y[2]);

    // Package gradients into a 1x3 matrix for downstream accumulation
    EAMatrix d_logits(1, direction_output_size);
    d_logits.SetValue(0, 0, dL_dz0);
    d_logits.SetValue(0, 1, dL_dz1);
    d_logits.SetValue(0, 2, dL_dz2);

    int predicted_class = (p[0] > p[1] && p[0] > p[2]) ? 0 : ((p[2] > p[1] && p[2] > p[0]) ? 2 : 1);
    return HeadLoss3Class{ L, std::move(d_logits), p[0], p[1], p[2], predicted_class };
}

float EA::LSTM::predictOnly(const EAMatrix& h_T,
                            const EAMatrix& W,
                            const EAMatrix& b) const
{
    auto logits = MetaNN::Dot(h_T, W) + b;
    if (targetType == TargetType::UpNeutralDownReturn)
    {
        auto predH = logits.EvalRegister();
        MetaNN::EvalPlan::Inst().Eval();
        const auto& z = predH.Data();
        float zz[direction_output_size] = { z(0, 0), z(0, 1), z(0, 2) };
        float p[direction_output_size];
        Softmax3(zz, p);
        const int predClass = (p[0] > p[1] && p[0] > p[2]) ? 0 : ((p[2] > p[1] && p[2] > p[0]) ? 2 : 1);
        return (predClass == 0) ? -1.0f : ((predClass == 1) ? 0.0f : 1.0f);
    }
    auto predH = logits.EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();
    return predH.Data()(0, 0);
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
    const auto W_h_cat = W_h_win;
    return GateBlocks{ W_i, W_f, W_g, W_o, W_h_cat };
}
// Keep more of the batched backward path on Metal by fusing the recurrent-gate
// gradient blocks into a single contiguous (B, 4H) matrix before the major GEMMs.
// This reduces the number of separate gate-specific Dot() launches in the hot path:
//   - one fused weight-gradient Dot for [x | h_prev]^T * d_gates
//   - one fused bias-gradient Dot for 1^T * d_gates
//   - one fused recurrent backprop Dot for d_gates * W_h^T
// We still split/accumulate into the per-gate accumulator buffers afterward so the
// rest of the optimizer path can remain unchanged.

inline void EA::LSTM::zeroGateAccumulators(GateAccumulators& A, size_t rows, size_t H) const
{
    auto zfill = [&](auto& m){
        auto low = MetaNN::LowerAccess(m);
        std::fill(low.MutableRawMemory(), low.MutableRawMemory() + m.Shape()[0] * m.Shape()[1], typename std::remove_reference_t<decltype(*low.MutableRawMemory())>(0));
    };
    zfill(A.dW_i); zfill(A.dW_f); zfill(A.dW_g); zfill(A.dW_o);
    zfill(A.db_i); zfill(A.db_f); zfill(A.db_g); zfill(A.db_o);
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

    // Gate gradients (expressions)
    auto d_o_expr = d_h * tanh_c * sc.o * (1.0f - sc.o);
    auto d_c_from_h_expr = d_h * sc.o * (1.0f - tanh_c * tanh_c);
    auto dct_expr = d_c + d_c_from_h_expr;

    auto d_i_expr = dct_expr * sc.g * sc.i * (1.0f - sc.i);
    auto d_g_expr = dct_expr * sc.i * (1.0f - sc.g * sc.g);
    auto d_f_expr = dct_expr * sc.c_prev * sc.f * (1.0f - sc.f);

    auto diH = d_i_expr.EvalRegister();
    auto dfH = d_f_expr.EvalRegister();
    auto dgH = d_g_expr.EvalRegister();
    auto doH = d_o_expr.EvalRegister();
    auto dc_prevH = (dct_expr * sc.f).EvalRegister();

    MetaNN::EvalPlan::Inst().Eval();

    const EAMatrix& d_i_mat = diH.Data();
    const EAMatrix& d_f_mat = dfH.Data();
    const EAMatrix& d_g_mat = dgH.Data();
    const EAMatrix& d_o_mat = doH.Data();

    // Pack all gate gradients into one contiguous (B, 4H) matrix so the major
    // backward GEMMs can stay fused on the Metal path.
    EAMatrix d_gates_batch(B, 4 * H);
    {
        auto lowDst = MetaNN::LowerAccess(d_gates_batch);
        float* dst = lowDst.MutableRawMemory();

        auto lowI = MetaNN::LowerAccess(d_i_mat);
        auto lowF = MetaNN::LowerAccess(d_f_mat);
        auto lowG = MetaNN::LowerAccess(d_g_mat);
        auto lowO = MetaNN::LowerAccess(d_o_mat);

        const float* iptr = lowI.RawMemory();
        const float* fptr = lowF.RawMemory();
        const float* gptr = lowG.RawMemory();
        const float* optr = lowO.RawMemory();

        for (size_t b = 0; b < B; ++b)
        {
            float* rowDst = dst + b * (4 * H);
            std::memcpy(rowDst + 0 * H, iptr + b * H, H * sizeof(float));
            std::memcpy(rowDst + 1 * H, fptr + b * H, H * sizeof(float));
            std::memcpy(rowDst + 2 * H, gptr + b * H, H * sizeof(float));
            std::memcpy(rowDst + 3 * H, optr + b * H, H * sizeof(float));
        }
    }

    EAMatrix xh_concat_batch(B, sc.x.Shape()[1] + sc.h_prev.Shape()[1]);
    NNUtils::ConcatColsInto(xh_concat_batch, sc.x, sc.h_prev);

    EAMatrix ones_col(B, 1);
    {
        auto lowOnes = MetaNN::LowerAccess(ones_col);
        std::fill(lowOnes.MutableRawMemory(), lowOnes.MutableRawMemory() + B, 1.0f);
    }

    auto dW_cat_expr = MetaNN::Dot(MetaNN::Transpose(xh_concat_batch), d_gates_batch);
    auto db_cat_expr = MetaNN::Dot(MetaNN::Transpose(ones_col), d_gates_batch);
    auto dh_prev_expr = MetaNN::Dot(d_gates_batch, MetaNN::Transpose(gb.W_h_cat));

    auto dW_catH = dW_cat_expr.EvalRegister();
    auto db_catH = db_cat_expr.EvalRegister();
    auto dh_prevH = dh_prev_expr.EvalRegister();

    MetaNN::EvalPlan::Inst().Eval();

    auto addColsToGateAccum = [&](auto& dst, const EAMatrix& src, size_t colOffset)
    {
        auto lowD = MetaNN::LowerAccess(dst);
        auto* dptr = lowD.MutableRawMemory();

        auto lowS = MetaNN::LowerAccess(src);
        const auto* sptr = lowS.RawMemory();

        const size_t rows = dst.Shape()[0];
        const size_t dstCols = dst.Shape()[1];
        const size_t srcCols = src.Shape()[1];

        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < H; ++c)
                dptr[r * dstCols + c] += static_cast<AccumScalar>(sptr[r * srcCols + (colOffset + c)]);
    };

    addColsToGateAccum(A.dW_i, dW_catH.Data(), 0 * H);
    addColsToGateAccum(A.dW_f, dW_catH.Data(), 1 * H);
    addColsToGateAccum(A.dW_g, dW_catH.Data(), 2 * H);
    addColsToGateAccum(A.dW_o, dW_catH.Data(), 3 * H);

    auto addBiasColsToGateAccum = [&](auto& dst, const EAMatrix& src, size_t colOffset)
    {
        auto lowD = MetaNN::LowerAccess(dst);
        auto* dptr = lowD.MutableRawMemory();

        auto lowS = MetaNN::LowerAccess(src);
        const auto* sptr = lowS.RawMemory();

        for (size_t c = 0; c < H; ++c)
            dptr[c] += static_cast<AccumScalar>(sptr[colOffset + c]);
    };

    addBiasColsToGateAccum(A.db_i, db_catH.Data(), 0 * H);
    addBiasColsToGateAccum(A.db_f, db_catH.Data(), 1 * H);
    addBiasColsToGateAccum(A.db_g, db_catH.Data(), 2 * H);
    addBiasColsToGateAccum(A.db_o, db_catH.Data(), 3 * H);

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

EA::LSTM::LSTM(const Tensor& tt, float lt, float st, TargetType explicitTargetType)
  : t{ tt },
    n_in { (tt.begin() != tt.end()) ? static_cast<int>((*tt.begin()).Shape()[1] + kReturnFeatureCount) : static_cast<int>(kReturnFeatureCount) },
    targetType { explicitTargetType },
    param  { static_cast<size_t>(n_in), 4 * n_out } // Combined gate weights matrix with shape [(n_in + hidden_size) x 4*n_out];
{
    const size_t baseFeatureCount = (tt.begin() != tt.end())
        ? static_cast<size_t>((*tt.begin()).Shape()[1])
        : 0;

#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(baseFeatureCount > 0, "LSTM ctor: input tensor must contain at least one feature column");
#endif

    // Model input width must match the actual per-timestep feature width.
    // We append enabled multi-horizon return features in CalculateBatch/PredictNext*,
    // so reserve kReturnFeatureCount additional input columns here.
    // n_in = static_cast<int>(baseFeatureCount + kReturnFeatureCount);

    // Rebuild all size-dependent tensors from the resolved runtime input width.
    param = EAMatrix(static_cast<size_t>(n_in + hidden_size), static_cast<size_t>(4 * n_out));
    bias = EAMatrix(1, static_cast<size_t>(4 * n_out));
    prevHiddenState = EAMatrix(1, hidden_size);
    prevCellState = EAMatrix(1, hidden_size);
    returnHeadWeight = EAMatrix(hidden_size, 1);
    returnHeadBias = EAMatrix(1, 1);
    returnHeadDirWeight = EAMatrix(hidden_size, direction_output_size);
    returnHeadDirBias = EAMatrix(1, direction_output_size);
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

#if LSTM_DEBUG_PRINTS
    std::cout << "LSTM ctor: baseFeatureCount=" << baseFeatureCount
              << " resolved_n_in=" << n_in
              << " hidden_size=" << hidden_size
              << std::endl;
#endif
    long_term = lt; short_term = st;
#else
    // Xavier/Glorot uniform initialization limit
    float limit = std::sqrt(6.0f / (static_cast<float>(n_in) + static_cast<float>(n_out)));

    {
        auto low = MetaNN::LowerAccess(param);
        float* p = low.MutableRawMemory();
        const size_t cols = static_cast<size_t>(4 * n_out);
        for (int r = 0; r < n_in + hidden_size; ++r)
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
        case TargetType::PercentReturn:
        case TargetType::LogReturn:
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
        case TargetType::UpNeutralDownReturn:
            {
                auto lowW = MetaNN::LowerAccess(returnHeadDirWeight);
                float* wp = lowW.MutableRawMemory();
                std::fill(wp, wp + hidden_size * returnHeadDirWeight.Shape()[1], 0.01f);
            }
            {
                auto lowB = MetaNN::LowerAccess(returnHeadDirBias);
                float* bp = lowB.MutableRawMemory();
                std::fill(bp, bp + returnHeadDirBias.Shape()[1], 0.0f);
            }
            break;
        default:    throw std::runtime_error("Invalid targetType in LSTM constructor");
    }
    
#if LSTM_DEBUG_PRINTS
    PrintOutputHeadShapes();
#endif
    long_term = lt; short_term = st;
#endif
}

std::tuple<float, size_t, size_t> EA::LSTM::CalculateBatch(Window batch)
{
    static size_t s_calcBatchCalls = 0;
    const size_t calcBatchCallIdx = s_calcBatchCalls++;
    EAMatrix head_logits_batch(effectiveMiniBatchWindows, direction_output_size);

    double sse = 0.0;
    size_t mseCount = 0;
    size_t windowCount = 0;
    size_t windowsInBatch = 0;
    size_t skippedWindows = 0;
    LSTMBatchProfile profile;

    // Class counts for 3-class targets
    size_t up_count = 0;
    size_t down_count = 0;
    size_t neutral_count = 0;

    size_t bucket3_33_35_total = 0;
    size_t bucket3_33_35_correct = 0;
    size_t bucket3_35_40_total = 0;
    size_t bucket3_35_40_correct = 0;
    size_t bucket3_40_45_total = 0;
    size_t bucket3_40_45_correct = 0;
    size_t bucket3_45_50_total = 0;
    size_t bucket3_45_50_correct = 0;
    size_t bucket3_50_55_total = 0;
    size_t bucket3_50_55_correct = 0;
    size_t bucket3_55_60_total = 0;
    size_t bucket3_55_60_correct = 0;
    size_t bucket3_60_70_total = 0;
    size_t bucket3_60_70_correct = 0;
    size_t bucket3_70p_total = 0;
    size_t bucket3_70p_correct = 0;

    // Lazy head gradient accumulators (expressions) across all windows in the batch
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> xh_concat_batch(effectiveMiniBatchWindows, static_cast<size_t>(n_in + hidden_size));
    EAMatrix h_batch(effectiveMiniBatchWindows, hidden_size);
    EAMatrix c_batch(effectiveMiniBatchWindows, hidden_size);
    EAMatrix d_h_batch(effectiveMiniBatchWindows, hidden_size);
    EAMatrix d_c_batch(effectiveMiniBatchWindows, hidden_size);
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

    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_headDirW_accum_f(hidden_size, returnHeadDirWeight.Shape()[1]);
#if !LSTM_INFERENCE_ONLY
    // Head gradient accumulators across all windows in the batch
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_headW_accum_f(hidden_size, 1);
    { auto low = MetaNN::LowerAccess(d_headW_accum_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory() + hidden_size, 0.0f); }
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_headB_accum_f(1, 1);
    { auto low = MetaNN::LowerAccess(d_headB_accum_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory() + 1, 0.0f); }

    { auto low = MetaNN::LowerAccess(d_headDirW_accum_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory() + hidden_size * returnHeadDirWeight.Shape()[1], 0.0f); }
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> d_headDirB_accum_f(1, returnHeadDirBias.Shape()[1]);
    { auto low = MetaNN::LowerAccess(d_headDirB_accum_f); std::fill(low.MutableRawMemory(), low.MutableRawMemory() + returnHeadDirBias.Shape()[1], 0.0f); }

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

    // Prebuild the batch rows into one contiguous (num_rows, F) tensor once so
    // minibatch window assembly can memcpy directly from a contiguous source
    // instead of repeatedly materializing overlapping windows via GetWindow().
    const size_t batchRows = static_cast<size_t>(batch.end() - batch.begin());
    const size_t baseFeatureCount = (batchRows > 0) ? static_cast<size_t>((*batch.begin()).Shape()[1]) : 0;
    const size_t modelFeatureCount = static_cast<size_t>(n_in);
    const bool useReturnFeatures = (kReturnFeatureCount > 0);
#if LSTM_TRAINING_ASSERTS
    if (modelFeatureCount != baseFeatureCount + kReturnFeatureCount)
    {
        std::cout << "DEBUG n_in=" << modelFeatureCount
                  << " baseFeatureCount=" << baseFeatureCount
                  << " kReturnFeatureCount=" << kReturnFeatureCount
                  << " closeCol=" << closeCol
                  << " batchRows=" << batchRows
                  << std::endl;
        LSTM_ASSERT(false, "CalculateBatch: model input width must equal base features + enabled return features");
    }
#endif
    const size_t featureCount = modelFeatureCount;
    EAMatrix prebuilt_rows(batchRows, featureCount);
    {
        auto lowPrebuilt = MetaNN::LowerAccess(prebuilt_rows);
        float* dst = lowPrebuilt.MutableRawMemory();
        size_t r = 0;
        for (auto it = batch.begin(); it != batch.end(); ++it, ++r)
        {
            auto lowRow = MetaNN::LowerAccess(*it);
            const float* src = lowRow.RawMemory();
            float* dstRow = dst + r * featureCount;

            std::memcpy(dstRow, src, baseFeatureCount * sizeof(float));

            if (useReturnFeatures)
            {
                const size_t appended = AppendMultiHorizonReturnFeatures(batch, r, dstRow, baseFeatureCount);
#if LSTM_TRAINING_ASSERTS
                LSTM_ASSERT(appended == kReturnFeatureCount,
                            "CalculateBatch: appended return feature count mismatch");
#endif
            }
        }
    }

    auto buildWindowBatch = [&](auto first, auto last) -> WindowBatch
    {
#if LSTM_BATCH_PROFILE
        LSTMScopedProfileTimer timer(profile.build_window_batch_us);
#endif
        WindowBatch wb;

        const size_t B_est = static_cast<size_t>(last - first);
        if (B_est == 0) return wb;

        const size_t F = featureCount;
        wb.packed_steps.reserve(window_size);
        std::vector<float*> packed_step_ptrs;
        packed_step_ptrs.reserve(window_size);
        for (size_t tstep = 0; tstep < window_size; ++tstep)
        {
            wb.packed_steps.emplace_back(B_est, F);
            auto lowPacked = MetaNN::LowerAccess(wb.packed_steps.back());
            packed_step_ptrs.push_back(lowPacked.MutableRawMemory());
        }

        wb.targets.reserve(B_est);
        wb.classTargets.reserve(B_est);
        wb.close_t.reserve(B_est);
        wb.close_target.reserve(B_est);

#warning "Insert prediction_horizon comment before targets loop"
        // NOTE: For faster learning and less noise, consider reducing prediction_horizon
        // to a shorter range (e.g., 1–4 timesteps) instead of larger horizons.
        auto lowPrebuilt = MetaNN::LowerAccess(prebuilt_rows);
        const float* prebuilt_ptr = lowPrebuilt.RawMemory();

        for (auto it = first; it != last; ++it)
        {
            const size_t start = *it;
            const size_t lastIdx   = start + (window_size - 1);
            const size_t targetIdx = lastIdx + prediction_horizon;
            const auto lastIt      = batch.begin() + static_cast<std::ptrdiff_t>(lastIdx);
            const auto targetIt    = batch.begin() + static_cast<std::ptrdiff_t>(targetIdx);
            const float close_t_local      = t.RawCloseAtIterator(lastIt);
            const float close_target_local = t.RawCloseAtIterator(targetIt);
            const float y_true_scaled      = prebuilt_ptr[targetIdx * F + closeCol];
            const float y_true_logret      = y_true_scaled / EA::LSTM::kFeatScale;

            float regressionTarget = 0.0f;
            int classTarget = 1;

            if (targetType == TargetType::UpNeutralDownReturn)
            {
                classTarget = ClassFromLogReturn(y_true_logret, c_next_threshold);
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
                regressionTarget = std::clamp(tval, -10.0f, 10.0f);
            }

            if (targetType == TargetType::UpNeutralDownReturn)  wb.classTargets.push_back(classTarget);
            else    wb.targets.push_back(regressionTarget);
        }

        const size_t B_final = (targetType == TargetType::UpNeutralDownReturn)
            ? wb.classTargets.size()
            : wb.targets.size();
        if (B_final > 0 && B_final != B_est)
        {
            for (size_t tstep = 0; tstep < window_size; ++tstep)
            {
                EAMatrix compact(B_final, F);
                auto lowSrc = MetaNN::LowerAccess(wb.packed_steps[tstep]);
                auto lowDst = MetaNN::LowerAccess(compact);
                std::memcpy(lowDst.MutableRawMemory(),
                            lowSrc.RawMemory(),
                            B_final * F * sizeof(float));
                wb.packed_steps[tstep] = std::move(compact);
            }
        }
        else if (B_final == 0)
        {
            wb.packed_steps.clear();
        }

        return wb;
    };

    // Materialize overlapping windows, then process them in minibatches.
    std::vector<size_t> allStarts;
    for (size_t start = 0;
         start + window_size + prediction_horizon - 1 < batchRows;
         ++start)
    {
        allStarts.push_back(start);
    }

#if LSTM_DIAG
        const bool diag_capture = (!LSTM_DIAG_ONLY_FIRST_BATCH || calcBatchCallIdx == 0);
        // Pre-update snapshots for true delta norms (compute unconditionally for simplicity)
        EAMatrix param_before_snap = NNUtils::DeepCopyMatrix(param);
        EAMatrix bias_before_snap  = NNUtils::DeepCopyMatrix(bias);
        EAMatrix headW_before_snap = (targetType == TargetType::UpNeutralDownReturn)
            ? NNUtils::DeepCopyMatrix(returnHeadDirWeight)
            : NNUtils::DeepCopyMatrix(returnHeadWeight);
        EAMatrix headB_before_snap = (targetType == TargetType::UpNeutralDownReturn)
            ? NNUtils::DeepCopyMatrix(returnHeadDirBias)
            : NNUtils::DeepCopyMatrix(returnHeadBias);
#endif

    for (size_t batchBase = 0; batchBase < allStarts.size(); batchBase += effectiveMiniBatchWindows)
    {
        const size_t batchEnd = std::min(batchBase + effectiveMiniBatchWindows, allStarts.size());
        WindowBatch wb = buildWindowBatch(allStarts.begin() + static_cast<std::ptrdiff_t>(batchBase),
                                          allStarts.begin() + static_cast<std::ptrdiff_t>(batchEnd));
        const size_t B = (targetType == TargetType::UpNeutralDownReturn)
            ? wb.classTargets.size()
            : wb.targets.size();
        if (B == 0)
        {
            continue;
        }
#if LSTM_BATCH_PROFILE
        ++profile.mini_batches;
#endif

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

        for (size_t tstep = 0; tstep < window_size; ++tstep)
        {
#if LSTM_BATCH_PROFILE
            {
                LSTMScopedProfileTimer timer(profile.forward_step_batches_us);
                const EAMatrix& x_t_batch = wb.packed_steps[tstep];
                cache.push_back(forwardStepBatch(x_t_batch, ww, bias, h_batch, c_batch, xh_concat_batch, forward_scratch, &profile));
            }
#else
            const EAMatrix& x_t_batch = wb.packed_steps[tstep];
            cache.push_back(forwardStepBatch(x_t_batch, ww, bias, h_batch, c_batch, xh_concat_batch, forward_scratch, nullptr));
#endif
        }
        if (calcBatchCallIdx == 0 && batchBase == 0)
        {
            PrintMatrixSummary("DIAG_DIRHEAD_INPUT_h_batch", h_batch);
            PrintMatrixSummary("DIAG_DIRHEAD_CELL_c_batch", c_batch);
            PrintMatrixSummary("DIAG_DIRHEAD_WEIGHT", returnHeadDirWeight);
            PrintMatrixSummary("DIAG_DIRHEAD_BIAS", returnHeadDirBias);
        }

        std::vector<float> errs(B, 0.0f);
        EAMatrix d_logits_batch(1, direction_output_size); // for 3-class path (declared once for scope)

        // Batched head forward and loss on (B, H)
        if (targetType == TargetType::UpNeutralDownReturn)
        {
            if (head_logits_batch.Shape()[0] != B || head_logits_batch.Shape()[1] != direction_output_size)
                head_logits_batch = EAMatrix(B, direction_output_size);
            if (calcBatchCallIdx == 0 && batchBase == 0)    PrintMatrixSummary("DIAG_PRE_HEAD_h_batch", h_batch);
#if LSTM_BATCH_PROFILE
            {
                LSTMScopedProfileTimer timer(profile.head_affine_us);
                auto lowA = MetaNN::LowerAccess(h_batch);
                auto lowB = MetaNN::LowerAccess(returnHeadDirWeight);
                auto lowBias = MetaNN::LowerAccess(returnHeadDirBias);
                auto lowY = MetaNN::LowerAccess(head_logits_batch);

                auto aMem = lowA.SharedMemory();
                auto bMem = lowB.SharedMemory();
                auto biasMem = lowBias.SharedMemory();
                auto yMem = lowY.SharedMemory();

                MetaNN::NSMetalMatMul::MatMulBias(aMem, bMem, biasMem, yMem, B, hidden_size, direction_output_size);
            }
#else
            {
                auto lowA = MetaNN::LowerAccess(h_batch);
                auto lowB = MetaNN::LowerAccess(returnHeadDirWeight);
                auto lowBias = MetaNN::LowerAccess(returnHeadDirBias);
                auto lowY = MetaNN::LowerAccess(head_logits_batch);

                auto aMem = lowA.SharedMemory();
                auto bMem = lowB.SharedMemory();
                auto biasMem = lowBias.SharedMemory();
                auto yMem = lowY.SharedMemory();

                MetaNN::NSMetalMatMul::MatMulBias(aMem, bMem, biasMem, yMem, B, hidden_size, direction_output_size);
            }
#endif
            MetaNN::NSMetalMatMul::WaitForAll();

            d_logits_batch = EAMatrix(B, direction_output_size);
            auto lowLogits = MetaNN::LowerAccess(head_logits_batch);
            const float* lptr = lowLogits.RawMemory();
            auto lowD = MetaNN::LowerAccess(d_logits_batch);
            float* dptr = lowD.MutableRawMemory();

            for (size_t b = 0; b < B; ++b)
            {
                const float* z = lptr + b * direction_output_size;
                float p[direction_output_size];
                Softmax3(z, p);
                const int cls = wb.classTargets[b];
                const float classWeight = (cls == 0) ? kClassWeightDown
                                         : (cls == 1) ? kClassWeightNeutral
                                                      : kClassWeightUp;

                dptr[b * direction_output_size + 0] = kClassWeightDown    * (p[0] - ((cls == 0) ? 1.0f : 0.0f));
                dptr[b * direction_output_size + 1] = kClassWeightNeutral * (p[1] - ((cls == 1) ? 1.0f : 0.0f));
                dptr[b * direction_output_size + 2] = kClassWeightUp      * (p[2] - ((cls == 2) ? 1.0f : 0.0f));

                sse += static_cast<double>(classWeight) *
                       (-std::log(std::max(1e-12f, p[cls])));
                ++mseCount;

                if (cls == 0) ++down_count;
                else if (cls == 1) ++neutral_count;
                else ++up_count;

                // 3-class probability bucket logic
                const float max_prob = std::max(p[0], std::max(p[1], p[2]));
                const int predClass = (p[0] > p[1] && p[0] > p[2]) ? 0 : ((p[2] > p[1] && p[2] > p[0]) ? 2 : 1);
                const bool correct = (predClass == cls);

                Log3ClassSample(cls, predClass);
                if (max_prob >= 0.33f && max_prob < 0.35f)
                {
                    ++bucket3_33_35_total;
                    if (correct) ++bucket3_33_35_correct;
                }
                else if (max_prob >= 0.35f && max_prob < 0.40f)
                {
                    ++bucket3_35_40_total;
                    if (correct) ++bucket3_35_40_correct;
                }
                else if (max_prob >= 0.40f && max_prob < 0.45f)
                {
                    ++bucket3_40_45_total;
                    if (correct) ++bucket3_40_45_correct;
                }
                else if (max_prob >= 0.45f && max_prob < 0.50f)
                {
                    ++bucket3_45_50_total;
                    if (correct) ++bucket3_45_50_correct;
                }
                else if (max_prob >= 0.50f && max_prob < 0.55f)
                {
                    ++bucket3_50_55_total;
                    if (correct) ++bucket3_50_55_correct;
                }
                else if (max_prob >= 0.55f && max_prob < 0.60f)
                {
                    ++bucket3_55_60_total;
                    if (correct) ++bucket3_55_60_correct;
                }
                else if (max_prob >= 0.60f && max_prob < 0.70f)
                {
                    ++bucket3_60_70_total;
                    if (correct) ++bucket3_60_70_correct;
                }
                else if (max_prob >= 0.70f)
                {
                    ++bucket3_70p_total;
                    if (correct) ++bucket3_70p_correct;
                }

#if !LSTM_INFERENCE_ONLY
                y_sum += static_cast<double>(cls);
                y_sumsq += static_cast<double>(cls) * static_cast<double>(cls);
                y_min = std::min(y_min, static_cast<float>(cls));
                y_max = std::max(y_max, static_cast<float>(cls));
                ++y_count;
                if (yhat_samples.size() < 10)
                    yhat_samples.push_back(static_cast<float>(predClass));
#endif
            }
    #if !LSTM_INFERENCE_ONLY
            windowCount += B;
            windowsInBatch += B;
    #endif
        }
        else
        {
            // combine into single evaluation barrier for efficiency
            if (head_logits_batch.Shape()[0] != B || head_logits_batch.Shape()[1] != 1)
                head_logits_batch = EAMatrix(B, 1);
#if LSTM_BATCH_PROFILE
            {
                LSTMScopedProfileTimer timer(profile.head_affine_us);
                auto lowA = MetaNN::LowerAccess(h_batch);
                auto lowB = MetaNN::LowerAccess(returnHeadWeight);
                auto lowBias = MetaNN::LowerAccess(returnHeadBias);
                auto lowY = MetaNN::LowerAccess(head_logits_batch);

                auto aMem = lowA.SharedMemory();
                auto bMem = lowB.SharedMemory();
                auto biasMem = lowBias.SharedMemory();
                auto yMem = lowY.SharedMemory();

                MetaNN::NSMetalMatMul::MatMulBias(
                    aMem,
                    bMem,
                    biasMem,
                    yMem,
                    B, hidden_size, 1);
            }
#else
            {
                auto lowA = MetaNN::LowerAccess(h_batch);
                auto lowB = MetaNN::LowerAccess(returnHeadWeight);
                auto lowBias = MetaNN::LowerAccess(returnHeadBias);
                auto lowY = MetaNN::LowerAccess(head_logits_batch);

                auto aMem = lowA.SharedMemory();
                auto bMem = lowB.SharedMemory();
                auto biasMem = lowBias.SharedMemory();
                auto yMem = lowY.SharedMemory();

                MetaNN::NSMetalMatMul::MatMulBias(
                    aMem,
                    bMem,
                    biasMem,
                    yMem,
                    B, hidden_size, 1);
            }
#endif
            MetaNN::NSMetalMatMul::WaitForAll();
            auto lowYLogits = MetaNN::LowerAccess(head_logits_batch);
            const float* yptr = lowYLogits.RawMemory();

            for (size_t b = 0; b < B; ++b)
            {
                const float y_hat = yptr[b];
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
    #endif
            }
    #if !LSTM_INFERENCE_ONLY
            windowCount += B;
            windowsInBatch += B;
    #endif
        }

#if !LSTM_INFERENCE_ONLY
        if (targetType == TargetType::UpNeutralDownReturn)
        {
            AccumulateHeadGradsBatch3Class(d_headDirW_accum_f, d_headDirB_accum_f, h_batch, d_logits_batch);

            d_h_batch = BuildHeadDhBatch3Class(d_logits_batch, returnHeadDirWeight, LSTM_CORE_GRAD_SCALE);
            if (d_c_batch.Shape()[0] != B || d_c_batch.Shape()[1] != hidden_size)
                d_c_batch = EAMatrix(B, hidden_size);
            zeroFill(d_c_batch);

            zeroGateAccumulators(G_bin, param.Shape()[0], hidden_size);

            auto gb = hoistGateBlocks(ww.W_h, hidden_size);
            for (int tstep = static_cast<int>(cache.size()) - 1; tstep >= 0; --tstep)
                backwardStepBatch(cache[static_cast<size_t>(tstep)], gb, d_h_batch, d_c_batch, G_bin);

            mergeGateAccumulators(G_bin, d_param_accum, d_bias_accum, hidden_size);
            #if LSTM_DIAG
                        if (!LSTM_DIAG_ONLY_FIRST_BATCH || calcBatchCallIdx == 0)
                        {
                            const size_t mbIdx = batchBase / effectiveMiniBatchWindows;
                            const double n_dparam = FroNormEvalHost(d_param_accum);
                            const double n_dbias  = FroNormEvalHost(d_bias_accum);
                            const double n_dheadW = FroNormEvalHost(d_headDirW_accum_f);
                            const double n_dheadB = FroNormEvalHost(d_headDirB_accum_f);
                            const double n_headW  = FroNormEvalHost(returnHeadDirWeight);
                            const double n_headB  = FroNormEvalHost(returnHeadDirBias);

                            const double n_mb_dparam = std::sqrt(
                                std::pow(FroNormEvalHost(G_bin.dW_i), 2.0) +
                                std::pow(FroNormEvalHost(G_bin.dW_f), 2.0) +
                                std::pow(FroNormEvalHost(G_bin.dW_g), 2.0) +
                                std::pow(FroNormEvalHost(G_bin.dW_o), 2.0));
                            const double n_mb_dbias = std::sqrt(
                                std::pow(FroNormEvalHost(G_bin.db_i), 2.0) +
                                std::pow(FroNormEvalHost(G_bin.db_f), 2.0) +
                                std::pow(FroNormEvalHost(G_bin.db_g), 2.0) +
                                std::pow(FroNormEvalHost(G_bin.db_o), 2.0));

                            EAMatrix d_headDirW_mb(hidden_size, returnHeadDirWeight.Shape()[1]);
                            EAMatrix d_headDirB_mb(1, returnHeadDirBias.Shape()[1]);
                            zeroFill(d_headDirW_mb);
                            zeroFill(d_headDirB_mb);
                            AccumulateHeadGradsBatch3Class(d_headDirW_mb, d_headDirB_mb, h_batch, d_logits_batch);
                            const double n_mb_dheadW = FroNormEvalHost(d_headDirW_mb);
                            const double n_mb_dheadB = FroNormEvalHost(d_headDirB_mb);

                            std::cout
                                << "DIAG_MB"
                                << ",calcBatchCall=" << calcBatchCallIdx
                                << ",mb=" << mbIdx
                                << ",B=" << B
                                << ",mb_dparam=" << n_mb_dparam
                                << ",mb_dbias="  << n_mb_dbias
                                << ",mb_dHeadW=" << n_mb_dheadW
                                << ",mb_dHeadB=" << n_mb_dheadB
                                << ",dparam=" << n_dparam
                                << ",dbias="  << n_dbias
                                << ",dHeadW=" << n_dheadW
                                << ",dHeadB=" << n_dheadB
                                << ",headW="  << n_headW
                                << ",headB="  << n_headB
                                << "\n";
                        }
            #endif
        }
        else
        {
            AccumulateHeadGradsBatch(d_headW_accum_f, d_headB_accum_f, h_batch, errs);

            d_h_batch = BuildHeadDhBatch(errs, returnHeadWeight, LSTM_CORE_GRAD_SCALE);
            if (d_c_batch.Shape()[0] != B || d_c_batch.Shape()[1] != hidden_size)
                d_c_batch = EAMatrix(B, hidden_size);
            zeroFill(d_c_batch);

            zeroGateAccumulators(G_reg, param.Shape()[0], hidden_size);

            auto gb = hoistGateBlocks(ww.W_h, hidden_size);
            for (int tstep = static_cast<int>(cache.size()) - 1; tstep >= 0; --tstep)
                backwardStepBatch(cache[static_cast<size_t>(tstep)], gb, d_h_batch, d_c_batch, G_reg);

            mergeGateAccumulators(G_reg, d_param_accum, d_bias_accum, hidden_size);
            #if LSTM_DIAG
                        if (!LSTM_DIAG_ONLY_FIRST_BATCH || calcBatchCallIdx == 0)
                        {
                            const size_t mbIdx = batchBase / effectiveMiniBatchWindows;
                            const double n_dparam = FroNormEvalHost(d_param_accum);
                            const double n_dbias  = FroNormEvalHost(d_bias_accum);
                            const double n_dheadW = FroNormEvalHost(d_headW_accum_f);
                            const double n_dheadB = FroNormEvalHost(d_headB_accum_f);
                            const double n_headW  = FroNormEvalHost(returnHeadWeight);
                            const double n_headB  = FroNormEvalHost(returnHeadBias);

                            const double n_mb_dparam = std::sqrt(
                                std::pow(FroNormEvalHost(G_reg.dW_i), 2.0) +
                                std::pow(FroNormEvalHost(G_reg.dW_f), 2.0) +
                                std::pow(FroNormEvalHost(G_reg.dW_g), 2.0) +
                                std::pow(FroNormEvalHost(G_reg.dW_o), 2.0));
                            const double n_mb_dbias = std::sqrt(
                                std::pow(FroNormEvalHost(G_reg.db_i), 2.0) +
                                std::pow(FroNormEvalHost(G_reg.db_f), 2.0) +
                                std::pow(FroNormEvalHost(G_reg.db_g), 2.0) +
                                std::pow(FroNormEvalHost(G_reg.db_o), 2.0));

                            EAMatrix d_headW_mb(hidden_size, 1);
                            EAMatrix d_headB_mb(1, 1);
                            zeroFill(d_headW_mb);
                            zeroFill(d_headB_mb);
                            AccumulateHeadGradsBatch(d_headW_mb, d_headB_mb, h_batch, errs);
                            const double n_mb_dheadW = FroNormEvalHost(d_headW_mb);
                            const double n_mb_dheadB = FroNormEvalHost(d_headB_mb);

                            std::cout
                                << "DIAG_MB"
                                << ",calcBatchCall=" << calcBatchCallIdx
                                << ",mb=" << mbIdx
                                << ",B=" << B
                                << ",mb_dparam=" << n_mb_dparam
                                << ",mb_dbias="  << n_mb_dbias
                                << ",mb_dHeadW=" << n_mb_dheadW
                                << ",mb_dHeadB=" << n_mb_dheadB
                                << ",dparam=" << n_dparam
                                << ",dbias="  << n_dbias
                                << ",dHeadW=" << n_dheadW
                                << ",dHeadB=" << n_dheadB
                                << ",headW="  << n_headW
                                << ",headB="  << n_headB
                                << "\n";
                        }
            #endif
        }
#endif
    }

#if !LSTM_INFERENCE_ONLY
    // Per-batch diagnostics
    #if LSTM_BATCH_PROFILE
    {
        const double assembly_us = profile.build_window_batch_us + profile.forward_step_batches_us + profile.head_affine_us;
        const double forward_pct = (assembly_us > 0.0) ? (100.0 * profile.forward_step_batches_us / assembly_us) : 0.0;
        const double build_pct = (assembly_us > 0.0) ? (100.0 * profile.build_window_batch_us / assembly_us) : 0.0;
        const double head_affine_pct = (assembly_us > 0.0) ? (100.0 * profile.head_affine_us / assembly_us) : 0.0;
        const double concat_cols_pct_of_forward = (profile.forward_step_batches_us > 0.0)
        ? (100.0 * profile.concat_cols_us / profile.forward_step_batches_us)
        : 0.0;
        const double dot_plus_bias_pct_of_forward = (profile.forward_step_batches_us > 0.0)
        ? (100.0 * profile.dot_plus_bias_us / profile.forward_step_batches_us)
        : 0.0;
        // const double gate_only_pct_of_forward = (profile.forward_step_batches_us > 0.0)
        // ? (100.0 * profile.gate_state_us / profile.forward_step_batches_us)
        // : 0.0;
        
        std::cout << std::fixed << std::setprecision(3)
                  << "assembly_us=" << assembly_us
                  << " build_window_batch_us=" << profile.build_window_batch_us
                  << " forward_step_batches_us=" << profile.forward_step_batches_us
                  << " concat_cols_us=" << profile.concat_cols_us
                  << " dot_plus_bias_us=" << profile.dot_plus_bias_us
                  << " head_affine_us=" << profile.head_affine_us
                  << " forward_pct=" << forward_pct
                  << " build_pct=" << build_pct
                  << " head_affine_pct=" << head_affine_pct
                  << " concat_cols_pct_of_forward=" << concat_cols_pct_of_forward
                  << " dot_plus_bias_pct_of_forward=" << dot_plus_bias_pct_of_forward
                  << " minibatches=" << profile.mini_batches
                  << std::defaultfloat << std::setprecision(15)
                  << "\n";
    }
    #endif
    std::cout << "batch_count=" << windowCount << "\n";
    double loss_value = 0.0;
    loss_value = (windowCount > 0) ? (sse / static_cast<double>(windowCount)) : 0.0;
    std::cout << "loss_value=" << loss_value << "\n";

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
        const float lrHead = learningRate * LSTM_HEAD_LR_MULT * invN; // or a separate head LR if you want
        
        #if LSTM_DIAG
                if (!LSTM_DIAG_ONLY_FIRST_BATCH || calcBatchCallIdx == 0)
                {
                    const double n_param = FroNormEvalHost(param);
                    const double n_bias  = FroNormEvalHost(bias);
                    const double n_headW = (targetType == TargetType::UpNeutralDownReturn) ? FroNormEvalHost(returnHeadDirWeight) : FroNormEvalHost(returnHeadWeight);
                    const double n_headB = (targetType == TargetType::UpNeutralDownReturn) ? FroNormEvalHost(returnHeadDirBias) : FroNormEvalHost(returnHeadBias);
        
                    // Gradient norms (after Evaluate already below, but safe to compute here too)
                    const double n_gparam = FroNormEvalHost(d_param_accum);
                    const double n_gbias  = FroNormEvalHost(d_bias_accum);
                    const double n_gheadW = (targetType == TargetType::UpNeutralDownReturn) ? FroNormEvalHost(d_headDirW_accum_f) : FroNormEvalHost(d_headW_accum_f);
                    const double n_gheadB = (targetType == TargetType::UpNeutralDownReturn) ? FroNormEvalHost(d_headDirB_accum_f) : FroNormEvalHost(d_headB_accum_f);

                    std::cout
                        << "DIAG_PREUPD"
                        << ",calcBatchCall=" << calcBatchCallIdx
                        << ",param=" << n_param
                        << ",bias=" << n_bias
                        << ",headW=" << n_headW
                        << ",headB=" << n_headB
                        << ",gParam=" << n_gparam
                        << ",gBias=" << n_gbias
                        << ",gHeadW=" << n_gheadW
                        << ",gHeadB=" << n_gheadB
                        << "\n";
                }
        #endif
        #if !LSTM_DISABLE_UPDATES
            SGDUpdate(param, d_param_f, lrCore);
            SGDUpdate(bias,  d_bias_f,  lrCore);
            if (targetType == TargetType::UpNeutralDownReturn) {
                SGDUpdate(returnHeadDirWeight, d_headDirW_f, lrHead);
                SGDUpdate(returnHeadDirBias,   d_headDirB_f, lrHead);
            } else {
                SGDUpdate(returnHeadWeight, d_headW_f, lrHead);
                SGDUpdate(returnHeadBias,   d_headB_f, lrHead);
            }
        #endif
#if LSTM_DIAG
                if (!LSTM_DIAG_ONLY_FIRST_BATCH || calcBatchCallIdx == 0)
                {
                    const double n_param2 = FroNormEvalHost(param);
                    const double n_bias2  = FroNormEvalHost(bias);
                    const double n_headW2 = (targetType == TargetType::UpNeutralDownReturn) ? FroNormEvalHost(returnHeadDirWeight) : FroNormEvalHost(returnHeadWeight);
                    const double n_headB2 = (targetType == TargetType::UpNeutralDownReturn) ? FroNormEvalHost(returnHeadDirBias) : FroNormEvalHost(returnHeadBias);

                    // True update magnitudes (Frobenius norms of parameter deltas)
                    double d_param_delta = FroNormDeltaHost(param, param_before_snap);
                    double d_bias_delta  = FroNormDeltaHost(bias,  bias_before_snap);
                    double d_headW_delta = 0.0;
                    double d_headB_delta = 0.0;
                    if (targetType == TargetType::UpNeutralDownReturn)
                    {
                        d_headW_delta = FroNormDeltaHost(returnHeadDirWeight, headW_before_snap);
                        d_headB_delta = FroNormDeltaHost(returnHeadDirBias,   headB_before_snap);
                    }
                    else
                    {
                        d_headW_delta = FroNormDeltaHost(returnHeadWeight, headW_before_snap);
                        d_headB_delta = FroNormDeltaHost(returnHeadBias,   headB_before_snap);
                    }
        
                    std::cout
                        << "DIAG_POSTUPD"
                        << ",calcBatchCall=" << calcBatchCallIdx
                        << ",param=" << n_param2
                        << ",bias=" << n_bias2
                        << ",headW=" << n_headW2
                        << ",headB=" << n_headB2
                        << ",dParam=" << d_param_delta
                        << ",dBias="  << d_bias_delta
                        << ",dHeadW=" << d_headW_delta
                        << ",dHeadB=" << d_headB_delta
                        << "\n";

                    // Combined CSV-friendly line with pre/post norms and deltas
                    const double n_param_pre = FroNormEvalHost(param_before_snap);
                    const double n_bias_pre  = FroNormEvalHost(bias_before_snap);
                    const double n_headW_pre = FroNormEvalHost(headW_before_snap);
                    const double n_headB_pre = FroNormEvalHost(headB_before_snap);

                    // Gradient norms (recomputed here for a single-line summary)
                    const double n_gparam2 = FroNormEvalHost(d_param_accum);
                    const double n_gbias2  = FroNormEvalHost(d_bias_accum);
                    const double n_gheadW2 = (targetType == TargetType::UpNeutralDownReturn) ? FroNormEvalHost(d_headDirW_accum_f) : FroNormEvalHost(d_headW_accum_f);
                    const double n_gheadB2 = (targetType == TargetType::UpNeutralDownReturn) ? FroNormEvalHost(d_headDirB_accum_f) : FroNormEvalHost(d_headB_accum_f);

                    std::cout
                        << "DIAG_COMBINED"
                        << ",calcBatchCall=" << calcBatchCallIdx
                        << ",param_pre=" << n_param_pre
                        << ",bias_pre="  << n_bias_pre
                        << ",headW_pre=" << n_headW_pre
                        << ",headB_pre=" << n_headB_pre
                        << ",gParam="    << n_gparam2
                        << ",gBias="     << n_gbias2
                        << ",gHeadW="    << n_gheadW2
                        << ",gHeadB="    << n_gheadB2
                        << ",param_post=" << n_param2
                        << ",bias_post="  << n_bias2
                        << ",headW_post=" << n_headW2
                        << ",headB_post=" << n_headB2
                        << ",dParam="     << d_param_delta
                        << ",dBias="      << d_bias_delta
                        << ",dHeadW="     << d_headW_delta
                        << ",dHeadB="     << d_headB_delta
                        << "\n";
                }
        #endif
    }
#endif

#if LSTM_EPOCH_BUCKETS
    // Insert epoch-bucket aggregation here (moved from after PredictNextRelativeMove)
    if (targetType == TargetType::UpNeutralDownReturn)
        AccumulateEpochBuckets3Class(down_count, neutral_count, up_count,
                                     bucket3_33_35_total, bucket3_33_35_correct,
                                     bucket3_35_40_total, bucket3_35_40_correct,
                                     bucket3_40_45_total, bucket3_40_45_correct,
                                     bucket3_45_50_total, bucket3_45_50_correct,
                                     bucket3_50_55_total, bucket3_50_55_correct,
                                     bucket3_55_60_total, bucket3_55_60_correct,
                                     bucket3_60_70_total, bucket3_60_70_correct,
                                     bucket3_70p_total, bucket3_70p_correct);
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
        preds.push_back(PredictNextRelativeMove(w, reset_state_per_window));
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
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> xh_concat_row(1, static_cast<size_t>(n_in + hidden_size));
    const size_t baseFeatureCount = (w.begin() != w.end()) ? static_cast<size_t>((*w.begin()).Shape()[1]) : 0;
    const size_t modelFeatureCount = static_cast<size_t>(n_in);
    const bool useReturnFeatures = (kReturnFeatureCount > 0);
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(modelFeatureCount == baseFeatureCount + kReturnFeatureCount,
                "PredictNextReturn: model input width must equal base features + enabled return features");
#endif
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> model_row(1, modelFeatureCount);

    // Forward through the window
    size_t rowIdx = 0;
    for (const auto& f_sample : w)
    {
        auto lowSrc = MetaNN::LowerAccess(f_sample);
        const float* src = lowSrc.RawMemory();

        auto lowDst = MetaNN::LowerAccess(model_row);
        float* dst = lowDst.MutableRawMemory();

        std::memcpy(dst, src, baseFeatureCount * sizeof(float));
        if (useReturnFeatures)
        {
            const size_t appended = AppendMultiHorizonReturnFeatures(w, rowIdx, dst, baseFeatureCount);
#if LSTM_TRAINING_ASSERTS
            LSTM_ASSERT(appended == kReturnFeatureCount,
                        "PredictNextReturn: appended return feature count mismatch");
#endif
        }

        forwardStep(model_row, ww, bias, prevHiddenState, prevCellState, xh_concat_row);
        ++rowIdx;
    }

    // Predict from the last hidden state.
    // For UpNeutralDownReturn, the inference API returns the most likely class encoded as:
    //   -1.0f = down, 0.0f = neutral, 1.0f = up.
    // Continuous return-valued outputs are only available for LogReturn/PercentReturn.
    float y_hat = (targetType == TargetType::UpNeutralDownReturn)
        ? predictOnly(prevHiddenState, returnHeadDirWeight, returnHeadDirBias)
        : predictOnly(prevHiddenState, returnHeadWeight, returnHeadBias);

    if (targetType == TargetType::UpNeutralDownReturn)
    {
        LSTM_ASSERT(false, "PredictNextReturn() is not valid for UpNeutralDownReturn; use PredictNextDirectionClass() or PredictNextDirectionProbs().");
        return 0.0f;
    }
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
        case TargetType::PercentReturn: return std::log(1.0f + std::clamp(raw, -1.0f + 1e-6f, std::numeric_limits<float>::infinity())); // convert percent return to log-return
        case TargetType::LogReturn: default:     return raw; // already log-return
    }
}

inline float EA::LSTM::PredictNextRelativeMove(const Window& w, bool resetState)
{
    // Minimal inference fix: return predicted relative move (fraction), not a price.
    if (resetState) ResetPreviousState();

    // Prepare views and concat buffer
    auto ww = hoistWindowWeights();
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> xh_concat_row(1, static_cast<size_t>(n_in + hidden_size));
    const size_t baseFeatureCount = (w.begin() != w.end()) ? static_cast<size_t>((*w.begin()).Shape()[1]) : 0;
    const size_t modelFeatureCount = static_cast<size_t>(n_in);
    const bool useReturnFeatures = (kReturnFeatureCount > 0);
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(modelFeatureCount == baseFeatureCount + kReturnFeatureCount,
                "PredictNextClose: model input width must equal base features + enabled return features");
#endif
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> model_row(1, modelFeatureCount);

    // Forward through the window
    size_t rowIdx = 0;
    for (const auto& f_sample : w)
    {
        auto lowSrc = MetaNN::LowerAccess(f_sample);
        const float* src = lowSrc.RawMemory();

        auto lowDst = MetaNN::LowerAccess(model_row);
        float* dst = lowDst.MutableRawMemory();

        std::memcpy(dst, src, baseFeatureCount * sizeof(float));
        if (useReturnFeatures)
        {
            const size_t appended = AppendMultiHorizonReturnFeatures(w, rowIdx, dst, baseFeatureCount);
#if LSTM_TRAINING_ASSERTS
            LSTM_ASSERT(appended == kReturnFeatureCount,
                        "PredictNextClose: appended return feature count mismatch");
#endif
        }

        forwardStep(model_row, ww, bias, prevHiddenState, prevCellState, xh_concat_row);
        ++rowIdx;
    }

    // Predict from the last hidden state.
    // For UpNeutralDownReturn, the inference API returns the most likely class encoded as:
    //   -1.0f = down, 0.0f = neutral, 1.0f = up.
    // Relative-move outputs are only available for LogReturn/PercentReturn.
    float y_hat = (targetType == TargetType::UpNeutralDownReturn)
        ? predictOnly(prevHiddenState, returnHeadDirWeight, returnHeadDirBias)
        : predictOnly(prevHiddenState, returnHeadWeight, returnHeadBias);

    if (targetType == TargetType::UpNeutralDownReturn)
    {
        LSTM_ASSERT(false, "PredictNextRelativeMove() is not valid for UpNeutralDownReturn; use PredictNextDirectionClass() or PredictNextDirectionProbs().");
        return 0.0f;
    }
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
