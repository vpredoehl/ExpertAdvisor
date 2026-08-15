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
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <array>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cstring>

#include "LSTM.hpp"
#include "Tensor.hpp"
#include "MatrixUtils.hpp"
#include "BuildConfig.hpp"
#include "TargetLabel.hpp"
#include "ReturnFeatureHistory.hpp"
#include <MetaNN/data_copy/data_copy.h>
#include <MetaNN/metal/metal_matmul.h>

#ifndef LSTM_BATCH_PROFILE
#define LSTM_BATCH_PROFILE 0
#endif

#ifndef LSTM_DIAG
#define LSTM_DIAG 0
#endif

#ifndef LSTM_HEAVY_DIAG
#define LSTM_HEAVY_DIAG 0
#endif

#ifndef LSTM_SHAPE_DIAG
#define LSTM_SHAPE_DIAG 0
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
    HotspotScope hotspot("host_diagnostics_data_copy");
    MetaNN::NSMetalMatMul::WaitForAll();
    auto ev = MetaNN::Evaluate(m);
    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> host(ev.Shape()[0], ev.Shape()[1]);
    MetaNN::DataCopy(ev, host);
    auto low = MetaNN::LowerAccess(host);

    const float* p = low.RawMemory();
    const size_t rows = host.Shape()[0];
    const size_t cols = host.Shape()[1];
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

namespace
{
struct Phase2MatrixStats
{
    size_t finiteCount = 0;
    size_t nanCount = 0;
    size_t infCount = 0;
    double sum = 0.0;
    double sumSq = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double absmax = 0.0;

    void add(double v)
    {
        if (std::isnan(v))
        {
            ++nanCount;
            return;
        }
        if (!std::isfinite(v))
        {
            ++infCount;
            return;
        }
        ++finiteCount;
        min = std::min(min, v);
        max = std::max(max, v);
        absmax = std::max(absmax, std::fabs(v));
        sum += v;
        sumSq += v * v;
    }

    double mean() const
    {
        return finiteCount ? (sum / static_cast<double>(finiteCount)) : 0.0;
    }

    double stddev() const
    {
        if (!finiteCount) return 0.0;
        const double m = mean();
        return std::sqrt(std::max(0.0, sumSq / static_cast<double>(finiteCount) - m * m));
    }
};

template <typename Mat>
void PrintPhase2MatrixDiagnostics(const char* stem,
                                  const char* warnLabel,
                                  const Mat& m,
                                  double absmaxWarnThreshold,
                                  double nearZeroStdThreshold,
                                  size_t colsToPrint = 8,
                                  size_t windowsToPrint = 3)
{
    MetaNN::NSMetalMatMul::WaitForAll();
    auto ev = MetaNN::Evaluate(m);
    MetaNN::Matrix<float, MetaNN::DeviceTags::CPU> host(ev.Shape()[0], ev.Shape()[1]);
    MetaNN::DataCopy(ev, host);
    auto low = MetaNN::LowerAccess(host);

    const float* p = low.RawMemory();
    const size_t rows = host.Shape()[0];
    const size_t cols = host.Shape()[1];
    Phase2MatrixStats globalStats;
    std::vector<Phase2MatrixStats> colStats(cols);

    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            const double v = static_cast<double>(p[r * cols + c]);
            globalStats.add(v);
            colStats[c].add(v);
        }
    }

    size_t zeroVarFeatureCount = 0;
    for (const auto& s : colStats)
    {
        if (s.finiteCount > 0 && s.stddev() <= nearZeroStdThreshold)
            ++zeroVarFeatureCount;
    }

    std::cout << stem
              << "_GLOBAL"
              << ",rows=" << rows
              << ",cols=" << cols
              << ",finite=" << globalStats.finiteCount
              << ",nan=" << globalStats.nanCount
              << ",inf=" << globalStats.infCount
              << ",min=" << (globalStats.finiteCount ? globalStats.min : 0.0)
              << ",max=" << (globalStats.finiteCount ? globalStats.max : 0.0)
              << ",mean=" << globalStats.mean()
              << ",std=" << globalStats.stddev()
              << ",absmax=" << globalStats.absmax
              << ",zero_var_features=" << zeroVarFeatureCount
              << std::endl;

    for (size_t c = 0; c < std::min(cols, colsToPrint); ++c)
    {
        const auto& s = colStats[c];
        std::cout << stem
                  << "_COL"
                  << ",idx=" << c
                  << ",finite=" << s.finiteCount
                  << ",nan=" << s.nanCount
                  << ",inf=" << s.infCount
                  << ",min=" << (s.finiteCount ? s.min : 0.0)
                  << ",max=" << (s.finiteCount ? s.max : 0.0)
                  << ",mean=" << s.mean()
                  << ",std=" << s.stddev()
                  << ",absmax=" << s.absmax
                  << std::endl;
    }

    for (size_t w = 0; w < windowsToPrint && rows >= window_size && w + window_size <= rows; ++w)
    {
        Phase2MatrixStats windowStats;
        for (size_t r = w; r < w + window_size; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
                windowStats.add(static_cast<double>(p[r * cols + c]));
        }

        std::cout << stem
                  << "_WINDOW"
                  << ",window_idx=" << w
                  << ",start_row=" << w
                  << ",rows=" << window_size
                  << ",cols=" << cols
                  << ",finite=" << windowStats.finiteCount
                  << ",nan=" << windowStats.nanCount
                  << ",inf=" << windowStats.infCount
                  << ",min=" << (windowStats.finiteCount ? windowStats.min : 0.0)
                  << ",max=" << (windowStats.finiteCount ? windowStats.max : 0.0)
                  << ",mean=" << windowStats.mean()
                  << ",std=" << windowStats.stddev()
                  << ",absmax=" << windowStats.absmax
                  << std::endl;
    }

    if (globalStats.absmax > absmaxWarnThreshold)
    {
        std::cout << warnLabel
                  << ",kind=absmax_exceeds_sane_threshold"
                  << ",stem=" << stem
                  << ",threshold=" << absmaxWarnThreshold
                  << ",absmax=" << globalStats.absmax
                  << std::endl;
    }

    if (zeroVarFeatureCount > 0)
    {
        std::cout << warnLabel
                  << ",kind=near_zero_std_features"
                  << ",stem=" << stem
                  << ",threshold=" << nearZeroStdThreshold
                  << ",count=" << zeroVarFeatureCount
                  << std::endl;
    }
}
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
    EA::LSTM::HotspotScope hotspot("host_diagnostics_data_copy");
    // Make sure any queued GPU work is finished before we read host-visible memory.
    MetaNN::NSMetalMatMul::WaitForAll();
    auto ev = MetaNN::Evaluate(m);
    MetaNN::NSMetalMatMul::WaitForAll();
    auto low = MetaNN::LowerAccess(ev);
    const auto* p = low.RawMemory();
    const size_t n = ev.Shape()[0] * ev.Shape()[1];
    std::vector<float> host(p, p + n);
    long double acc = 0.0L;
    for (size_t i = 0; i < n; ++i)
    {
        const long double v = static_cast<long double>(host[i]);
        acc += v * v;
    }
    return std::sqrt(static_cast<double>(acc));
}

template <typename Mat>
static bool MatrixAllFiniteHost(const char* name, const Mat& m)
{
    EA::LSTM::HotspotScope hotspot("host_diagnostics_data_copy");
    MetaNN::NSMetalMatMul::WaitForAll();
    auto ev = MetaNN::Evaluate(m);
    auto low = MetaNN::LowerAccess(ev);
    const auto* p = low.RawMemory();
    const size_t n = ev.Shape()[0] * ev.Shape()[1];
    for (size_t i = 0; i < n; ++i)
    {
        const double v = static_cast<double>(p[i]);
        if (!std::isfinite(v))
        {
            std::cout << "DIAG_NONFINITE_MATRIX"
                      << ",name=" << name
                      << ",idx=" << i
                      << ",value=" << v
                      << "\n";
            return false;
        }
    }
    return true;
}

template <typename Mat>
static void ClipMatrixInPlace(Mat& m, float threshold, const char* name)
{
    auto low = MetaNN::LowerAccess(m);
    auto* p = low.MutableRawMemory();
    const size_t n = m.Shape()[0] * m.Shape()[1];
    size_t clipped = 0;
    double maxAbsBefore = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const float v = p[i];
        const double a = std::fabs(static_cast<double>(v));
        maxAbsBefore = std::max(maxAbsBefore, a);
        if (v > threshold)
        {
            p[i] = threshold;
            ++clipped;
        }
        else if (v < -threshold)
        {
            p[i] = -threshold;
            ++clipped;
        }
    }
    if (clipped > 0)
    {
        std::cout << "DIAG_GRAD_CLIP"
                  << ",name=" << name
                  << ",threshold=" << threshold
                  << ",clipped=" << clipped
                  << ",total=" << n
                  << ",max_abs_before=" << maxAbsBefore
                  << "\n";
    }
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
#define LSTM_HEAD_LR_MULT 100.0f
#endif
#ifndef LSTM_CORE_GRAD_SCALE
#define LSTM_CORE_GRAD_SCALE 4.0f
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

static_assert(kReturnFeatureCount == EA::kMultiHorizonReturnLookbacks.size(),
              "the active LSTM input contract requires return lookbacks 1, 4, 8, and 16");
static_assert(feature_size + kReturnFeatureCount == 36,
              "the active LSTM model input width must remain 36");

#ifndef MINI_BATCH_WINDOWS
#define MINI_BATCH_WINDOWS 512
#endif
#ifndef LSTM_MAX_MINI_BATCH_WINDOWS
#define LSTM_MAX_MINI_BATCH_WINDOWS 128
#endif

constexpr size_t mini_batch_windows = MINI_BATCH_WINDOWS;

const size_t effectiveMiniBatchWindows = std::max<size_t>(1, std::min<size_t>(mini_batch_windows, static_cast<size_t>(LSTM_MAX_MINI_BATCH_WINDOWS)));

namespace {
struct LSTMHotspotCounter
{
    size_t calls = 0;
    double totalUs = 0.0;
};

bool g_lstmHotspotProfilingEnabled = false;
std::map<std::string, LSTMHotspotCounter> g_lstmHotspotCounters;

const char* LSTMHotspotMetalCandidate(const std::string& name)
{
    if (name == "window_batch_build" ||
        name == "appended_return_features" ||
        name == "concat_cols" ||
        name == "d_gates_batch_packing" ||
        name == "gate_accumulator_split_merge" ||
        name == "host_diagnostics_data_copy")
        return "high";
    if (name == "matmul_bias_gate_preactivation" ||
        name == "gate_state_fused" ||
        name == "backward_gemms" ||
        name == "optimizer_update" ||
        name == "gradient_clipping")
        return "medium";
    return "low";
}

std::vector<std::pair<std::string, LSTMHotspotCounter>> LSTMHotspotRowsSorted()
{
    std::vector<std::pair<std::string, LSTMHotspotCounter>> rows(
        g_lstmHotspotCounters.begin(),
        g_lstmHotspotCounters.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b)
              {
                  if (a.second.totalUs != b.second.totalUs)
                      return a.second.totalUs > b.second.totalUs;
                  return a.first < b.first;
              });
    return rows;
}

double LSTMHotspotTotalUs()
{
    double total = 0.0;
    for (const auto& item : g_lstmHotspotCounters)
        total += item.second.totalUs;
    return total;
}
}

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

EA::LSTM::HotspotScope::HotspotScope(const char* name)
    : name_(name),
      enabled_(EA::LSTM::HotspotProfilingEnabled()),
      start_(enabled_ ? std::chrono::steady_clock::now()
                      : std::chrono::steady_clock::time_point{})
{}

EA::LSTM::HotspotScope::~HotspotScope()
{
    if (!enabled_ || name_ == nullptr)
        return;
    const auto end = std::chrono::steady_clock::now();
    const double elapsedUs = std::chrono::duration<double, std::micro>(end - start_).count();
    EA::LSTM::RecordHotspot(name_, elapsedUs);
}

void EA::LSTM::ConfigureHotspotProfiler(bool enabled, std::optional<std::string> outputPath)
{
    (void)outputPath;
    g_lstmHotspotProfilingEnabled = enabled;
    g_lstmHotspotCounters.clear();
}

bool EA::LSTM::HotspotProfilingEnabled()
{
    return g_lstmHotspotProfilingEnabled;
}

void EA::LSTM::RecordHotspot(const char* name, double elapsedUs)
{
    if (!g_lstmHotspotProfilingEnabled || name == nullptr)
        return;
    auto& counter = g_lstmHotspotCounters[std::string{name}];
    ++counter.calls;
    counter.totalUs += elapsedUs;
}

void EA::LSTM::PrintHotspotProfileSummary()
{
    if (!g_lstmHotspotProfilingEnabled)
        return;

    const double totalUs = LSTMHotspotTotalUs();
    for (const auto& row : LSTMHotspotRowsSorted())
    {
        const double totalMs = row.second.totalUs / 1000.0;
        const double avgUs = row.second.calls
            ? (row.second.totalUs / static_cast<double>(row.second.calls))
            : 0.0;
        const double pct = totalUs > 0.0 ? (100.0 * row.second.totalUs / totalUs) : 0.0;
        std::cout << "LSTM_PROFILE_HOTSPOT"
                  << ",name=" << row.first
                  << ",calls=" << row.second.calls
                  << ",total_ms=" << totalMs
                  << ",avg_us=" << avgUs
                  << ",percent=" << pct
                  << ",metal_candidate=" << LSTMHotspotMetalCandidate(row.first)
                  << std::endl;
    }
}

bool EA::LSTM::WriteHotspotProfileReport(const std::string& outputPath)
{
    if (!g_lstmHotspotProfilingEnabled || outputPath.empty())
        return false;

    const std::filesystem::path path{outputPath};
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());

    std::ofstream out(path);
    if (!out)
        return false;

    const double totalUs = LSTMHotspotTotalUs();
    out << "# LSTM Hotspot Profile\n\n";
    out << "Profiling is runtime opt-in and aggregates elapsed wall-clock time only. "
        << "It does not alter LSTM math, optimizer behavior, model serialization, "
        << "scheduler behavior, or database state.\n\n";
    out << "| rank | name | calls | total_ms | avg_us | percent | metal_candidate |\n";
    out << "|---|---|---|---|---|---|---|\n";

    size_t rank = 1;
    for (const auto& row : LSTMHotspotRowsSorted())
    {
        const double totalMs = row.second.totalUs / 1000.0;
        const double avgUs = row.second.calls
            ? (row.second.totalUs / static_cast<double>(row.second.calls))
            : 0.0;
        const double pct = totalUs > 0.0 ? (100.0 * row.second.totalUs / totalUs) : 0.0;
        out << "| " << rank++
            << " | " << row.first
            << " | " << row.second.calls
            << " | " << totalMs
            << " | " << avgUs
            << " | " << pct
            << " | " << LSTMHotspotMetalCandidate(row.first)
            << " |\n";
    }
    return true;
}


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
    EA::LSTM::EAMatrix W_hi, W_hf, W_hg, W_ho; // individual recurrent gate blocks (H x H)
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
#ifndef LSTM_PHASE3_GATE_DIAG_LIMIT
#define LSTM_PHASE3_GATE_DIAG_LIMIT 8
#endif
#ifndef LSTM_PHASE3_HEAD_DIAG_LIMIT
#define LSTM_PHASE3_HEAD_DIAG_LIMIT 64
#endif

struct Phase3MatrixStats
{
    size_t count = 0;
    size_t finite = 0;
    size_t nan = 0;
    size_t inf = 0;
    size_t satLow = 0;
    size_t satHigh = 0;
    double sum = 0.0;
    double sumSq = 0.0;
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();
    double absmax = 0.0;
};

struct Phase3HostMatrix
{
    size_t rows = 0;
    size_t cols = 0;
    std::vector<float> data;
};

struct Phase3HiddenReplayCapture
{
    bool valid = false;
    size_t batchBase = 0;
    size_t rows = 0;
    size_t hiddenCols = 0;
    size_t effectiveMiniBatchWindows = 0;
    size_t windowCountAtCapture = 0;
    std::vector<size_t> startIndices;
    std::vector<int> actualClasses;
    size_t replayTimeSteps = 0;
    size_t replayInputCols = 0;
    std::vector<float> replayInputs;
    Phase3HostMatrix hBefore;
    std::array<size_t, 3> actualHist {0, 0, 0};
};

static Phase3HiddenReplayCapture s_phase3HiddenReplayCapture;

static Phase3HostMatrix s_phase3LastHGeometryHostBeforeUpdate;
static std::array<size_t, direction_output_size> s_phase3LastHGeometryActualHistBeforeUpdate {0, 0, 0};
static bool s_phase3LastHGeometryValidBeforeUpdate = false;

// === Helper for DIAG_H_RECOMPUTE_DELTA_ ===
static double Phase3HostMatrixDeltaNorm(const Phase3HostMatrix& a, const Phase3HostMatrix& b)
{
    if (a.rows != b.rows || a.cols != b.cols || a.data.size() != b.data.size())
        return std::numeric_limits<double>::infinity();

    long double ss = 0.0L;
    for (size_t i = 0; i < a.data.size(); ++i)
    {
        const long double d = static_cast<long double>(a.data[i]) - static_cast<long double>(b.data[i]);
        ss += d * d;
    }
    return std::sqrt(static_cast<double>(ss));
}

static double Phase3HostMatrixNorm(const Phase3HostMatrix& m)
{
    long double ss = 0.0L;
    for (float v : m.data)
    {
        const long double d = static_cast<long double>(v);
        ss += d * d;
    }
    return std::sqrt(static_cast<double>(ss));
}
static bool Phase3HostMatrixSameShape(const Phase3HostMatrix& a,
                                      const Phase3HostMatrix& b)
{
    return a.rows == b.rows &&
           a.cols == b.cols &&
           a.data.size() == b.data.size();
}

template <typename Mat>
Phase3HostMatrix Phase3MaterializeHost(const Mat& m)
{
    MetaNN::NSMetalMatMul::WaitForAll();
    auto ev = MetaNN::Evaluate(m);
    MetaNN::NSMetalMatMul::WaitForAll();
    auto low = MetaNN::LowerAccess(ev);
    const float* p = low.RawMemory();
    const size_t rows = ev.Shape()[0];
    const size_t cols = ev.Shape()[1];
    Phase3HostMatrix out;
    out.rows = rows;
    out.cols = cols;
    out.data.assign(p, p + rows * cols);
    return out;
}

inline void Phase3AddStat(Phase3MatrixStats& s, float v, double satLowThreshold, double satHighThreshold)
{
    ++s.count;
    if (std::isnan(v))
    {
        ++s.nan;
        return;
    }
    if (!std::isfinite(v))
    {
        ++s.inf;
        return;
    }

    const double d = static_cast<double>(v);
    ++s.finite;
    s.sum += d;
    s.sumSq += d * d;
    if (d < s.minVal) s.minVal = d;
    if (d > s.maxVal) s.maxVal = d;
    s.absmax = std::max(s.absmax, std::fabs(d));
    if (d <= satLowThreshold) ++s.satLow;
    if (d >= satHighThreshold) ++s.satHigh;
}

inline bool Phase3StatsSelfConsistent(const Phase3MatrixStats& s)
{
    if (!s.finite)
        return true;
    const double endpointAbsmax = std::max(std::fabs(s.minVal), std::fabs(s.maxVal));
    return s.absmax + 1.0e-6 >= endpointAbsmax;
}

template <typename Mat>
Phase3MatrixStats Phase3StatsWhole(const Mat& m, double satLowThreshold, double satHighThreshold)
{
    const auto host = Phase3MaterializeHost(m);
    Phase3MatrixStats s;
    for (const float v : host.data)
        Phase3AddStat(s, v, satLowThreshold, satHighThreshold);
    return s;
}

template <typename Mat>
Phase3MatrixStats Phase3StatsCols(const Mat& m, size_t col0, size_t colCount, double satLowThreshold, double satHighThreshold)
{
    const auto host = Phase3MaterializeHost(m);
    Phase3MatrixStats s;
    const size_t rows = host.rows;
    const size_t cols = host.cols;
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(col0 + colCount <= cols, "Phase3StatsCols: requested columns exceed matrix width");
#endif
    if (col0 + colCount > cols)
        return s;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < colCount; ++c)
            Phase3AddStat(s, host.data[r * cols + col0 + c], satLowThreshold, satHighThreshold);
    return s;
}

inline void PrintPhase3Stats(const char* prefix, size_t callIdx, const char* name, const Phase3MatrixStats& s)
{
    const double denom = s.finite ? static_cast<double>(s.finite) : 1.0;
    const double mean = s.sum / denom;
    const double var = std::max(0.0, s.sumSq / denom - mean * mean);
    std::cout << prefix
              << ",call=" << callIdx
              << ",name=" << name
              << ",count=" << s.count
              << ",finite=" << s.finite
              << ",nan=" << s.nan
              << ",inf=" << s.inf
              << ",min=" << (s.finite ? s.minVal : 0.0)
              << ",max=" << (s.finite ? s.maxVal : 0.0)
              << ",mean=" << mean
              << ",std=" << std::sqrt(var)
              << ",absmax=" << s.absmax
              << ",self_consistent=" << (Phase3StatsSelfConsistent(s) ? 1 : 0)
              << ",sat_low=" << s.satLow
              << ",sat_high=" << s.satHigh
              << std::endl;
}

template <typename Mat>
void PrintPhase3NormStats(const char* prefix, size_t callIdx, const char* name, const Mat& m)
{
    const auto stats = Phase3StatsWhole(m, -20.0, 20.0);
    std::cout << prefix
              << ",call=" << callIdx
              << ",name=" << name
              << ",count=" << stats.count
              << ",finite=" << stats.finite
              << ",nan=" << stats.nan
              << ",inf=" << stats.inf
              << ",min=" << (stats.finite ? stats.minVal : 0.0)
              << ",max=" << (stats.finite ? stats.maxVal : 0.0)
              << ",mean=" << (stats.finite ? stats.sum / static_cast<double>(stats.finite) : 0.0)
              << ",std=";
    if (stats.finite)
    {
        const double mean = stats.sum / static_cast<double>(stats.finite);
        const double var = std::max(0.0, stats.sumSq / static_cast<double>(stats.finite) - mean * mean);
        std::cout << std::sqrt(var);
    }
    else
    {
        std::cout << 0.0;
    }
    std::cout << ",absmax=" << stats.absmax
              << ",self_consistent=" << (Phase3StatsSelfConsistent(stats) ? 1 : 0)
              << ",fro_norm=" << FroNormEvalHost(m)
              << std::endl;
}

template <typename Mat>
size_t Phase3CountAbsGreaterThan(const Mat& m, float threshold)
{
    const auto host = Phase3MaterializeHost(m);
    size_t count = 0;
    for (const float v : host.data)
        if (std::isfinite(v) && std::fabs(v) > static_cast<double>(threshold))
            ++count;
    return count;
}

template <typename Mat>
void PrintPhase3ClipMatrixStats(const char* prefix, size_t callIdx, const char* name, const Mat& m, float threshold)
{
    const auto stats = Phase3StatsWhole(m, -20.0, 20.0);
    const size_t clipped = Phase3CountAbsGreaterThan(m, threshold);
    const double clippedPct = stats.count ? (100.0 * static_cast<double>(clipped) / static_cast<double>(stats.count)) : 0.0;
    const double mean = stats.finite ? stats.sum / static_cast<double>(stats.finite) : 0.0;
    const double var = stats.finite ? std::max(0.0, stats.sumSq / static_cast<double>(stats.finite) - mean * mean) : 0.0;
    std::cout << prefix
              << ",call=" << callIdx
              << ",name=" << name
              << ",count=" << stats.count
              << ",finite=" << stats.finite
              << ",nan=" << stats.nan
              << ",inf=" << stats.inf
              << ",min=" << (stats.finite ? stats.minVal : 0.0)
              << ",max=" << (stats.finite ? stats.maxVal : 0.0)
              << ",std=" << std::sqrt(var)
              << ",absmax=" << stats.absmax
              << ",self_consistent=" << (Phase3StatsSelfConsistent(stats) ? 1 : 0)
              << ",fro_norm=" << FroNormEvalHost(m)
              << ",clip_threshold=" << threshold
              << ",clipped_count=" << clipped
              << ",clipped_pct=" << clippedPct
              << std::endl;
}

template <typename MatPre, typename MatPost>
void PrintPhase3ClipEffect(size_t callIdx, const char* name, const MatPre& pre, const MatPost& post, float threshold)
{
    const auto preStats = Phase3StatsWhole(pre, -20.0, 20.0);
    const auto postStats = Phase3StatsWhole(post, -20.0, 20.0);
    const size_t clipped = Phase3CountAbsGreaterThan(pre, threshold);
    const double clippedPct = preStats.count ? (100.0 * static_cast<double>(clipped) / static_cast<double>(preStats.count)) : 0.0;
    const double preNorm = FroNormEvalHost(pre);
    const double postNorm = FroNormEvalHost(post);
    std::cout << "DIAG_GRAD_CLIP_EFFECT_"
              << ",call=" << callIdx
              << ",name=" << name
              << ",clip_threshold=" << threshold
              << ",count=" << preStats.count
              << ",clipped_count=" << clipped
              << ",clipped_pct=" << clippedPct
              << ",pre_fro_norm=" << preNorm
              << ",post_fro_norm=" << postNorm
              << ",norm_ratio=" << (preNorm > 0.0 ? postNorm / preNorm : 0.0)
              << ",pre_absmax=" << preStats.absmax
              << ",post_absmax=" << postStats.absmax
              << ",pre_self_consistent=" << (Phase3StatsSelfConsistent(preStats) ? 1 : 0)
              << ",post_self_consistent=" << (Phase3StatsSelfConsistent(postStats) ? 1 : 0)
              << std::endl;
}

template <typename MatParam, typename MatGrad>
void PrintPhase3UpdateScale(size_t callIdx, const char* name, const MatParam& paramMat, const MatGrad& gradMat, float effectiveLr)
{
    const double paramNorm = FroNormEvalHost(paramMat);
    const double gradNorm = FroNormEvalHost(gradMat);
    const double updateNorm = static_cast<double>(effectiveLr) * gradNorm;
    std::cout << "DIAG_UPDATE_SCALE_"
              << ",call=" << callIdx
              << ",name=" << name
              << ",param_norm=" << paramNorm
              << ",grad_norm_after_clip=" << gradNorm
              << ",effective_lr=" << effectiveLr
              << ",update_norm=" << updateNorm
              << ",update_to_param_ratio=" << (paramNorm > 0.0 ? updateNorm / paramNorm : 0.0)
              << std::endl;
}

template <typename MatA, typename MatB>
double Phase3MaxAbsDelta(const MatA& after, const MatB& before)
{
    const auto a = Phase3MaterializeHost(after);
    const auto b = Phase3MaterializeHost(before);
    if (a.rows != b.rows || a.cols != b.cols || a.data.size() != b.data.size())
        return std::numeric_limits<double>::infinity();
    double maxAbs = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i)
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(a.data[i]) - static_cast<double>(b.data[i])));
    return maxAbs;
}

template <typename Mat>
double Phase3AbsMaxHost(const Mat& m)
{
    return Phase3StatsWhole(m, -20.0, 20.0).absmax;
}

template <typename MatAfter, typename MatBefore>
void PrintPhase3HeadDelta(size_t callIdx, const char* name, const MatAfter& after, const MatBefore& before)
{
    const double beforeNorm = FroNormEvalHost(before);
    const double afterNorm = FroNormEvalHost(after);
    const double updateNorm = FroNormDeltaHost(after, before);
    std::cout << "DIAG_HEAD_DELTA_"
              << ",call=" << callIdx
              << ",name=" << name
              << ",param_norm_before=" << beforeNorm
              << ",param_norm_after=" << afterNorm
              << ",update_norm=" << updateNorm
              << ",update_to_param_ratio=" << (beforeNorm > 0.0 ? updateNorm / beforeNorm : 0.0)
              << ",max_abs_update=" << Phase3MaxAbsDelta(after, before)
              << ",max_abs_param_before=" << Phase3AbsMaxHost(before)
              << ",max_abs_param_after=" << Phase3AbsMaxHost(after)
              << std::endl;
}

inline EA::LSTM::EAMatrix Phase3DirHeadLogitsCpu(const EA::LSTM::EAMatrix& h,
                                                 const EA::LSTM::EAMatrix& w,
                                                 const EA::LSTM::EAMatrix& b)
{
    const auto hHost = Phase3MaterializeHost(h);
    const auto wHost = Phase3MaterializeHost(w);
    const auto bHost = Phase3MaterializeHost(b);
    EA::LSTM::EAMatrix logits(hHost.rows, wHost.cols);
    auto lowLogits = MetaNN::LowerAccess(logits);
    float* out = lowLogits.MutableRawMemory();
    for (size_t row = 0; row < hHost.rows; ++row)
    {
        for (size_t cls = 0; cls < wHost.cols; ++cls)
        {
            double acc = (bHost.data.size() > cls) ? bHost.data[cls] : 0.0;
            for (size_t hcol = 0; hcol < hHost.cols; ++hcol)
                acc += static_cast<double>(hHost.data[row * hHost.cols + hcol]) *
                       static_cast<double>(wHost.data[hcol * wHost.cols + cls]);
            out[row * wHost.cols + cls] = static_cast<float>(acc);
        }
    }
    return logits;
}

inline EA::LSTM::EAMatrix Phase3SoftmaxProbsCpu(const EA::LSTM::EAMatrix& logits)
{
    const auto logitHost = Phase3MaterializeHost(logits);
    EA::LSTM::EAMatrix probs(logitHost.rows, logitHost.cols);
    auto lowProbs = MetaNN::LowerAccess(probs);
    float* out = lowProbs.MutableRawMemory();
    for (size_t row = 0; row < logitHost.rows; ++row)
    {
        float rowMax = logitHost.data[row * logitHost.cols];
        for (size_t cls = 1; cls < logitHost.cols; ++cls)
            rowMax = std::max(rowMax, logitHost.data[row * logitHost.cols + cls]);
        double sumExp = 0.0;
        for (size_t cls = 0; cls < logitHost.cols; ++cls)
            sumExp += std::exp(static_cast<double>(logitHost.data[row * logitHost.cols + cls] - rowMax));
        for (size_t cls = 0; cls < logitHost.cols; ++cls)
        {
            const double e = std::exp(static_cast<double>(logitHost.data[row * logitHost.cols + cls] - rowMax));
            out[row * logitHost.cols + cls] = static_cast<float>(e / std::max(sumExp, 1.0e-30));
        }
    }
    return probs;
}

template <typename MatBefore, typename MatAfter>
void PrintPhase3LogitDelta(size_t callIdx, const MatBefore& logitsBefore, const MatAfter& logitsAfter,
                           const EA::LSTM::EAMatrix& probsBefore, const EA::LSTM::EAMatrix& probsAfter)
{
    const double logitDeltaNorm = FroNormDeltaHost(logitsAfter, logitsBefore);
    const double probDeltaNorm = FroNormDeltaHost(probsAfter, probsBefore);
    std::cout << "DIAG_LOGIT_DELTA_"
              << ",call=" << callIdx
              << ",logit_absmax_before=" << Phase3AbsMaxHost(logitsBefore)
              << ",logit_absmax_after=" << Phase3AbsMaxHost(logitsAfter)
              << ",logit_delta_norm=" << logitDeltaNorm
              << ",prob_delta_norm=" << probDeltaNorm
              << ",max_abs_logit_delta=" << Phase3MaxAbsDelta(logitsAfter, logitsBefore)
              << ",max_abs_prob_delta=" << Phase3MaxAbsDelta(probsAfter, probsBefore)
              << std::endl;
}
inline bool Phase3ShouldPrintProgressSample(size_t updateCall)
{
    return updateCall == 0 || updateCall == 1 || updateCall == 2 || updateCall == 3 ||
           updateCall == 10 || updateCall == 25 || updateCall == 50 || updateCall == 100;
}

inline double Phase3StatsStddev(const Phase3MatrixStats& s)
{
    if (!s.finite) return 0.0;
    const double mean = s.sum / static_cast<double>(s.finite);
    return std::sqrt(std::max(0.0, s.sumSq / static_cast<double>(s.finite) - mean * mean));
}

template <typename Mat>
void PrintPhase3PreactStats(size_t callIdx, const Mat& gatesBatch)
{
    const size_t H = gatesBatch.Shape()[1] / 4;
    constexpr double kPreactLow = -20.0;
    constexpr double kPreactHigh = 20.0;
    PrintPhase3Stats("DIAG_LSTM_PREACT_", callIdx, "i", Phase3StatsCols(gatesBatch, 0 * H, H, kPreactLow, kPreactHigh));
    PrintPhase3Stats("DIAG_LSTM_PREACT_", callIdx, "f", Phase3StatsCols(gatesBatch, 1 * H, H, kPreactLow, kPreactHigh));
    PrintPhase3Stats("DIAG_LSTM_PREACT_", callIdx, "g", Phase3StatsCols(gatesBatch, 2 * H, H, kPreactLow, kPreactHigh));
    PrintPhase3Stats("DIAG_LSTM_PREACT_", callIdx, "o", Phase3StatsCols(gatesBatch, 3 * H, H, kPreactLow, kPreactHigh));
}

inline const char* Phase3ClassName(size_t cls)
{
    switch (cls)
    {
        case 0: return "down";
        case 1: return "neutral";
        case 2: return "up";
        default: return "unknown";
    }
}

struct Phase3CompareStats
{
    size_t count = 0;
    size_t firstMismatch = static_cast<size_t>(-1);
    double sumAbsDiff = 0.0;
    double maxAbsDiff = 0.0;
};

template <typename Mat>
Phase3CompareStats Phase3CompareMatrices(const Mat& actual, const Mat& expected, float absTol, float relTol)
{
    const auto actualHost = Phase3MaterializeHost(actual);
    const auto expectedHost = Phase3MaterializeHost(expected);
    const size_t n = actualHost.data.size();
    Phase3CompareStats s;
    s.count = n;
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(actualHost.rows == expectedHost.rows && actualHost.cols == expectedHost.cols,
                "Phase3CompareMatrices: shape mismatch");
#endif
    if (actualHost.rows != expectedHost.rows || actualHost.cols != expectedHost.cols)
    {
        s.firstMismatch = 0;
        s.maxAbsDiff = std::numeric_limits<double>::infinity();
        return s;
    }
    for (size_t idx = 0; idx < n; ++idx)
    {
        const float a = actualHost.data[idx];
        const float e = expectedHost.data[idx];
        const double absDiff = std::fabs(static_cast<double>(a) - static_cast<double>(e));
        const double relDiff = absDiff / std::max(1.0, std::fabs(static_cast<double>(e)));
        s.sumAbsDiff += absDiff;
        s.maxAbsDiff = std::max(s.maxAbsDiff, absDiff);
        if (s.firstMismatch == static_cast<size_t>(-1) && absDiff > absTol && relDiff > relTol)
            s.firstMismatch = idx;
    }
    return s;
}

inline void PrintPhase3CompareStats(size_t callIdx, const char* name, const Phase3CompareStats& s)
{
    const double meanAbsDiff = s.count ? s.sumAbsDiff / static_cast<double>(s.count) : 0.0;
    std::cout << "DIAG_LSTM_COMPARE_"
              << ",call=" << callIdx
              << ",name=" << name
              << ",count=" << s.count
              << ",max_abs_diff=" << s.maxAbsDiff
              << ",mean_abs_diff=" << meanAbsDiff
              << ",first_mismatch_idx=";
    if (s.firstMismatch == static_cast<size_t>(-1)) std::cout << -1;
    else std::cout << s.firstMismatch;
    std::cout << std::endl;
}

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

template <typename Mat>
std::array<double, 3> DirectionBiasValues3(const Mat& bias)
{
    std::array<double, 3> out {0.0, 0.0, 0.0};
    const size_t rows = bias.Shape()[0];
    const size_t cols = bias.Shape()[1];
    if (rows >= 1 && cols >= 3)
    {
        auto acc = MetaNN::LowerAccess(bias);
        const auto* p = acc.RawMemory();
        out[0] = static_cast<double>(p[0]);
        out[1] = static_cast<double>(p[1]);
        out[2] = static_cast<double>(p[2]);
    }
    return out;
}

inline void PrintDirectionBiasByClassDiag(size_t call,
                                          const char* stage,
                                          const std::array<double, 3>& b)
{
    std::cout << "DIAG_HEAD_BIAS_BY_CLASS"
              << ",call=" << call
              << ",stage=" << stage
              << ",B_down=" << b[0]
              << ",B_neutral=" << b[1]
              << ",B_up=" << b[2]
              << ",B_down_minus_neutral=" << (b[0] - b[1])
              << ",B_up_minus_neutral=" << (b[2] - b[1])
              << ",B_neutral_minus_mean_edges=" << (b[1] - 0.5 * (b[0] + b[2]))
              << std::endl;
}

inline void PrintDirectionBiasDeltaByClassDiag(size_t call,
                                               const std::array<double, 3>& before,
                                               const std::array<double, 3>& after)
{
    const double deltaDown = after[0] - before[0];
    const double deltaNeutral = after[1] - before[1];
    const double deltaUp = after[2] - before[2];
    std::cout << "DIAG_HEAD_BIAS_DELTA_BY_CLASS"
              << ",call=" << call
              << ",before_down=" << before[0]
              << ",before_neutral=" << before[1]
              << ",before_up=" << before[2]
              << ",after_down=" << after[0]
              << ",after_neutral=" << after[1]
              << ",after_up=" << after[2]
              << ",delta_down=" << deltaDown
              << ",delta_neutral=" << deltaNeutral
              << ",delta_up=" << deltaUp
              << ",delta_down_minus_neutral=" << (deltaDown - deltaNeutral)
              << ",delta_up_minus_neutral=" << (deltaUp - deltaNeutral)
              << ",after_down_minus_neutral=" << (after[0] - after[1])
              << ",after_up_minus_neutral=" << (after[2] - after[1])
              << ",after_neutral_minus_mean_edges=" << (after[1] - 0.5 * (after[0] + after[2]))
              << std::endl;
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
    EA::LSTM::EAMatrix z_f;      // (B, hidden_size)
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
bool EA::LSTM::suppressPhase3HiddenGeometryDiagnostics = false;

void EA::LSTM::PrintOutputHeadShapes() const
{
    if (targetType == TargetType::UpNeutralDownReturn)
    {
        static_assert(direction_output_size == 3, "UpNeutralDownReturn expects exactly 3 output classes");
        LSTM_ASSERT(returnHeadDirWeight.Shape()[1] == direction_output_size,
                    "PrintOutputHeadShapes: classification head weight width mismatch");
        LSTM_ASSERT(returnHeadDirBias.Shape()[1] == direction_output_size,
                    "PrintOutputHeadShapes: classification head bias width mismatch");
        LSTM_ASSERT(direction_output_size == 3,
                    "PrintOutputHeadShapes: classification mode requires output dimension 3");
        std::cout << "DIAG_CLASS_HEAD"
                  << ",target_type=UpNeutralDownReturn"
                  << ",output_dim=" << direction_output_size
                  << ",weight_shape=(" << returnHeadDirWeight.Shape()[0] << "," << returnHeadDirWeight.Shape()[1] << ")"
                  << ",bias_shape=(" << returnHeadDirBias.Shape()[0] << "," << returnHeadDirBias.Shape()[1] << ")"
                  << std::endl;
    }

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
    const size_t windowGlobalStartIdx = static_cast<size_t>(w.begin() - t.begin());
#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(modelFeatureCount == baseFeatureCount + kReturnFeatureCount,
                "PredictNextDirectionProbs: model input width must equal base features + enabled return features");
#endif
    MetaNN::Matrix<float, MetaNN::DeviceTags::Metal> model_row(1, modelFeatureCount);
    static bool s_printed_phase2_infer_direction_diag = false;
    const bool capturePhase2Window = !s_printed_phase2_infer_direction_diag;
    std::optional<EAMatrix> phase2WindowRows;

    if (capturePhase2Window)    phase2WindowRows.emplace(static_cast<size_t>(w.end() - w.begin()), modelFeatureCount);
    size_t rowIdx = 0;
    for (const auto& f_sample : w)
    {
        auto lowSrc = MetaNN::LowerAccess(f_sample);
        const float* src = lowSrc.RawMemory();

        auto lowDst = MetaNN::LowerAccess(model_row);
        float* dst = lowDst.MutableRawMemory();

        if (useReturnFeatures)
        {
            HotspotScope hotspot("appended_return_features");
            const size_t appended = EA::AssembleLstmModelInputRowAtGlobalPosition(
                dst, src, baseFeatureCount, windowGlobalStartIdx + rowIdx, EA::LSTM::kFeatScale,
                [this](size_t globalPosition)
                {
                    return t.RawCloseAtIterator(
                        t.begin() + static_cast<std::ptrdiff_t>(globalPosition));
                });
#if LSTM_TRAINING_ASSERTS
            LSTM_ASSERT(appended == kReturnFeatureCount,
                        "PredictNextDirectionProbs: appended return feature count mismatch");
#endif
        }
        else
        {
            std::memcpy(dst, src, baseFeatureCount * sizeof(float));
        }

        for (size_t c = 0; c < modelFeatureCount; ++c)
        {
            const float v = dst[c];
            if (!std::isfinite(v))
            {
                std::cout << "DIAG_NORM_FAIL"
                          << ",phase=inference_direction"
                          << ",row=" << rowIdx
                          << ",col=" << c
                          << ",value=" << v
                          << std::endl;
                LSTM_ASSERT(false, "PredictNextDirectionProbs: non-finite feature detected before LSTM");
            }

            dst[c] = std::clamp(v, -10.0f, 10.0f);
        }

        if (capturePhase2Window)
        {
            auto lowPhase2 = MetaNN::LowerAccess(*phase2WindowRows);
            float* phase2Ptr = lowPhase2.MutableRawMemory();
            std::memcpy(phase2Ptr + rowIdx * modelFeatureCount,
                        dst,
                        modelFeatureCount * sizeof(float));
        }

        forwardStep(model_row, ww, bias, prevHiddenState, prevCellState, xh_concat_row);
        ++rowIdx;
    }

    if (capturePhase2Window)
    {
        std::cout << "DIAG_FEATURE_CONFIG"
                  << ",phase=inference_direction"
                  << ",base_feature_cols=" << baseFeatureCount
                  << ",appended_return_feature_cols=" << kReturnFeatureCount
                  << ",model_feature_cols=" << modelFeatureCount
                  << ",feature_uses_future_values=0"
                  << std::endl;
        std::cout << "DIAG_NORM_PRELSTM_PARITY"
                  << ",kind=classification_inference_prelstm_clamped"
                  << ",training_clamp=10"
                  << ",classification_inference_clamp=10"
                  << std::endl;
        PrintPhase2MatrixDiagnostics("DIAG_FEATURE_INFER_DIRECTION_PRELSTM",
                                     "DIAG_FEATURE_WARN",
                                     *phase2WindowRows,
                                     50.0,
                                     1e-6);
        s_printed_phase2_infer_direction_diag = true;
    }

    auto logits = MetaNN::Dot(prevHiddenState, returnHeadDirWeight) + returnHeadDirBias;
#if LSTM_HEAVY_DIAG
    static bool s_printed_dir_logits = false;
    if (!LSTM_DIAG_ONLY_FIRST_BATCH || !s_printed_dir_logits)
    {
        auto logits_eval = MetaNN::Evaluate(logits);
        PrintMatrixSummary("DIAG_DIRHEAD_LOGITS", logits_eval);
        s_printed_dir_logits = true;
    }
#endif
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
    static size_t s_headDhDiagCount = 0;
    const size_t headDhDiagIdx = s_headDhDiagCount++;
    const bool headDhDiagEnabled = (headDhDiagIdx < LSTM_PHASE3_HEAD_DIAG_LIMIT);
    const size_t B = d_logits_batch.Shape()[0];
    const size_t C = d_logits_batch.Shape()[1];
    const size_t H = headW.Shape()[0];
    const size_t headClasses = headW.Shape()[1];

#if LSTM_TRAINING_ASSERTS
    LSTM_ASSERT(C == direction_output_size, "BuildHeadDhBatch3Class: d_logits width must equal class count");
    LSTM_ASSERT(headClasses == direction_output_size, "BuildHeadDhBatch3Class: headW width must equal class count");
#endif

    EAMatrix cpu_ref(B, H);
    {
        auto lowD = MetaNN::LowerAccess(d_logits_batch);
        auto lowW = MetaNN::LowerAccess(headW);
        auto lowRef = MetaNN::LowerAccess(cpu_ref);
        const float* dptr = lowD.RawMemory();
        const float* wptr = lowW.RawMemory();
        float* rptr = lowRef.MutableRawMemory();
        for (size_t b = 0; b < B; ++b)
        {
            for (size_t h = 0; h < H; ++h)
            {
                float acc = 0.0f;
                for (size_t c = 0; c < C; ++c)
                    acc += dptr[b * C + c] * wptr[h * headClasses + c];
                rptr[b * H + h] = acc * scale;
            }
        }
    }

    auto expr = MetaNN::Dot(d_logits_batch, MetaNN::Transpose(headW));
    auto h = expr.EvalRegister();
    MetaNN::EvalPlan::Inst().Eval();

    EAMatrix d_h = NNUtils::DeepCopyMatrix(h.Data());
    auto lowDh = MetaNN::LowerAccess(d_h);
    float* dhp = lowDh.MutableRawMemory();
    const size_t n = d_h.Shape()[0] * d_h.Shape()[1];
    for (size_t i = 0; i < n; ++i)
        dhp[i] *= scale;

    if (headDhDiagEnabled)
    {
        std::cout << "DIAG_HEAD_DH_SHAPE"
                  << ",call=" << headDhDiagIdx
                  << ",d_logits_rows=" << B
                  << ",d_logits_cols=" << C
                  << ",headW_rows=" << H
                  << ",headW_cols=" << headClasses
                  << ",d_h_rows=" << d_h.Shape()[0]
                  << ",d_h_cols=" << d_h.Shape()[1]
                  << ",scale=" << scale
                  << std::endl;
        PrintPhase3Stats("DIAG_HEAD_DH_INPUT", headDhDiagIdx, "d_logits",
                         Phase3StatsWhole(d_logits_batch, -20.0, 20.0));
        PrintPhase3Stats("DIAG_HEAD_DH_WEIGHT", headDhDiagIdx, "headW",
                         Phase3StatsWhole(headW, -20.0, 20.0));
        PrintPhase3Stats("DIAG_HEAD_DH_OUTPUT", headDhDiagIdx, "d_h",
                         Phase3StatsWhole(d_h, -20.0, 20.0));
        PrintPhase3Stats("DIAG_HEAD_DH_CPU_REF", headDhDiagIdx, "d_h_ref",
                         Phase3StatsWhole(cpu_ref, -20.0, 20.0));
        const auto cmp = Phase3CompareMatrices(d_h, cpu_ref, 1e-5f, 1e-4f);
        const double meanAbsDiff = cmp.count ? cmp.sumAbsDiff / static_cast<double>(cmp.count) : 0.0;
        std::cout << "DIAG_HEAD_DH_COMPARE"
                  << ",call=" << headDhDiagIdx
                  << ",count=" << cmp.count
                  << ",max_abs_diff=" << cmp.maxAbsDiff
                  << ",mean_abs_diff=" << meanAbsDiff
                  << ",first_mismatch_idx=";
        if (cmp.firstMismatch == static_cast<size_t>(-1)) std::cout << -1;
        else std::cout << cmp.firstMismatch;
        std::cout << std::endl;
    }
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
    static size_t s_gateStateDiagCallCount = 0;
    const size_t gateStateCallIdx = s_gateStateDiagCallCount++;
    const bool phase3DiagEnabled = (gateStateCallIdx < LSTM_PHASE3_GATE_DIAG_LIMIT);

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
    if (phase3DiagEnabled)
    {
        constexpr double kSigmoidLow = 1.0e-4;
        constexpr double kSigmoidHigh = 1.0 - 1.0e-4;
        constexpr double kTanhLow = -1.0 + 1.0e-4;
        constexpr double kTanhHigh = 1.0 - 1.0e-4;
        constexpr double kStateLow = -20.0;
        constexpr double kStateHigh = 20.0;
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "i", Phase3StatsWhole(gate_i_batch, kSigmoidLow, kSigmoidHigh));
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "f", Phase3StatsWhole(gate_f_batch, kSigmoidLow, kSigmoidHigh));
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "g", Phase3StatsWhole(gate_g_batch, kTanhLow, kTanhHigh));
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "o", Phase3StatsWhole(gate_o_batch, kSigmoidLow, kSigmoidHigh));
        PrintPhase3Stats("DIAG_LSTM_STATE_", gateStateCallIdx, "c", Phase3StatsWhole(c_batch, kStateLow, kStateHigh));
        PrintPhase3Stats("DIAG_LSTM_STATE_", gateStateCallIdx, "h", Phase3StatsWhole(h_batch, -1.0 + 1.0e-4, 1.0 - 1.0e-4));
    }
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
        HotspotScope hotspot("gate_state_fused");
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

    if (phase3DiagEnabled)
    {
        constexpr double kSigmoidLow = 1.0e-4;
        constexpr double kSigmoidHigh = 1.0 - 1.0e-4;
        constexpr double kTanhLow = -1.0 + 1.0e-4;
        constexpr double kTanhHigh = 1.0 - 1.0e-4;
        constexpr double kStateLow = -20.0;
        constexpr double kStateHigh = 20.0;
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "i", Phase3StatsWhole(gate_i_batch, kSigmoidLow, kSigmoidHigh));
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "f", Phase3StatsWhole(gate_f_batch, kSigmoidLow, kSigmoidHigh));
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "g", Phase3StatsWhole(gate_g_batch, kTanhLow, kTanhHigh));
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "o", Phase3StatsWhole(gate_o_batch, kSigmoidLow, kSigmoidHigh));
        PrintPhase3Stats("DIAG_LSTM_STATE_", gateStateCallIdx, "c", Phase3StatsWhole(c_batch, kStateLow, kStateHigh));
        PrintPhase3Stats("DIAG_LSTM_STATE_", gateStateCallIdx, "h", Phase3StatsWhole(h_batch, -1.0 + 1.0e-4, 1.0 - 1.0e-4));
        constexpr float absTol = 1e-5f;
        constexpr float relTol = 1e-4f;
        PrintPhase3CompareStats(gateStateCallIdx, "gate_i", Phase3CompareMatrices(gate_i_batch, gate_i_ref, absTol, relTol));
        PrintPhase3CompareStats(gateStateCallIdx, "gate_f", Phase3CompareMatrices(gate_f_batch, gate_f_ref, absTol, relTol));
        PrintPhase3CompareStats(gateStateCallIdx, "gate_g", Phase3CompareMatrices(gate_g_batch, gate_g_ref, absTol, relTol));
        PrintPhase3CompareStats(gateStateCallIdx, "gate_o", Phase3CompareMatrices(gate_o_batch, gate_o_ref, absTol, relTol));
        PrintPhase3CompareStats(gateStateCallIdx, "c", Phase3CompareMatrices(c_batch, c_ref, absTol, relTol));
        PrintPhase3CompareStats(gateStateCallIdx, "h", Phase3CompareMatrices(h_batch, h_ref, absTol, relTol));
    }

    validate_same("gate_i", gate_i_batch, gate_i_ref);
    validate_same("gate_f", gate_f_batch, gate_f_ref);
    validate_same("gate_g", gate_g_batch, gate_g_ref);
    validate_same("gate_o", gate_o_batch, gate_o_ref);
    validate_same("c", c_batch, c_ref);
    validate_same("h", h_batch, h_ref);
#else
    {
        HotspotScope hotspot("gate_state_fused");
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
    if (phase3DiagEnabled)
    {
        constexpr double kSigmoidLow = 1.0e-4;
        constexpr double kSigmoidHigh = 1.0 - 1.0e-4;
        constexpr double kTanhLow = -1.0 + 1.0e-4;
        constexpr double kTanhHigh = 1.0 - 1.0e-4;
        constexpr double kStateLow = -20.0;
        constexpr double kStateHigh = 20.0;
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "i", Phase3StatsWhole(gate_i_batch, kSigmoidLow, kSigmoidHigh));
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "f", Phase3StatsWhole(gate_f_batch, kSigmoidLow, kSigmoidHigh));
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "g", Phase3StatsWhole(gate_g_batch, kTanhLow, kTanhHigh));
        PrintPhase3Stats("DIAG_LSTM_GATE_", gateStateCallIdx, "o", Phase3StatsWhole(gate_o_batch, kSigmoidLow, kSigmoidHigh));
        PrintPhase3Stats("DIAG_LSTM_STATE_", gateStateCallIdx, "c", Phase3StatsWhole(c_batch, kStateLow, kStateHigh));
        PrintPhase3Stats("DIAG_LSTM_STATE_", gateStateCallIdx, "h", Phase3StatsWhole(h_batch, -1.0 + 1.0e-4, 1.0 - 1.0e-4));
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
    HotspotScope forwardHotspot("forward_step_batch");
    const size_t B = x_t.Shape()[0];
    const size_t H = prevHiddenState.Shape()[1];
    const size_t expectedCols = x_t.Shape()[1] + H;
    const size_t gateCols = 4 * H;
    static size_t s_phase3ForwardStepBatchCallCount = 0;
    const size_t phase3ForwardCallIdx = s_phase3ForwardStepBatchCallCount++;

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
            HotspotScope hotspot("matmul_bias_gate_preactivation");
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
            HotspotScope hotspot("matmul_bias_gate_preactivation");
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
#if LSTM_SHAPE_DIAG
        static bool s_printed_batch_matmul_shapes = false;
        if (!s_printed_batch_matmul_shapes)
        {
            std::cout << "DIAG_BATCH_MATMUL_SHAPE"
                      << ",m=" << B
                      << ",k=" << K
                      << ",n=" << gateCols
                      << std::endl;
            s_printed_batch_matmul_shapes = true;
        }
#endif
    }

    if (phase3ForwardCallIdx < LSTM_PHASE3_GATE_DIAG_LIMIT)
        PrintPhase3PreactStats(phase3ForwardCallIdx, scratch.gates_batch);

#if LSTM_HEAVY_DIAG
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
#if LSTM_HEAVY_DIAG
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
    auto z_f_view = NNUtils::ViewCols<float, MetaNN::DeviceTags::Metal>(scratch.gates_batch, 1 * H, H);

    BatchStepCache sc
    {
        NNUtils::DeepCopyMatrix(x_t),
        NNUtils::DeepCopyMatrix(prevHiddenState),
        NNUtils::DeepCopyMatrix(prevCellState),
        NNUtils::DeepCopyMatrix(scratch.gate_i_batch),
        NNUtils::DeepCopyMatrix(scratch.gate_f_batch),
        NNUtils::DeepCopyMatrix(scratch.gate_g_batch),
        NNUtils::DeepCopyMatrix(scratch.gate_o_batch),
        NNUtils::DeepCopyMatrix(scratch.c),
        NNUtils::DeepCopyMatrix(scratch.h),
        NNUtils::DeepCopyMatrix(z_f_view)
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
#if LSTM_SHAPE_DIAG
        static bool s_printed_matmul_shapes = false;
        if (!s_printed_matmul_shapes)
        {
            std::cout << "DIAG_MATMUL_SHAPE"
                      << ",m=" << 1
                      << ",k=" << K
                      << ",n=" << (4 * H)
                      << std::endl;
            s_printed_matmul_shapes = true;
        }
#endif
    }
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

    // Gradient of loss w.r.t logits: scale * w_k * (p_k - y_k).
    // Next diagnostic test: reduce the direction-class loss signal instead of
    // trying another forget-gate bias value.
    constexpr float kDirectionClassGradScale = 0.1f;
    float wDown = kClassWeightDown;
    float wNeutral = kClassWeightNeutral;
    float wUp = kClassWeightUp;
    float dL_dz0 = kDirectionClassGradScale * wDown    * (p[0] - y[0]);
    float dL_dz1 = kDirectionClassGradScale * wNeutral * (p[1] - y[1]);
    float dL_dz2 = kDirectionClassGradScale * wUp      * (p[2] - y[2]);

    // Package gradients into a 1x3 matrix for downstream accumulation
    EAMatrix d_logits(1, direction_output_size);
    d_logits.SetValue(0, 0, dL_dz0);
    d_logits.SetValue(0, 1, dL_dz1);
    d_logits.SetValue(0, 2, dL_dz2);

    int predicted_class = (p[0] > p[1] && p[0] > p[2]) ? 0 : ((p[2] > p[1] && p[2] > p[0]) ? 2 : 1);
    return HeadLoss3Class{ L, std::move(d_logits), p[0], p[1], p[2], predicted_class };
}

void EA::LSTM::PrintHeadGradNormDiag(
    size_t tag,
    const EAMatrix& gradW, const EAMatrix& gradB,
    const EAMatrix& paramW, const EAMatrix& paramB,
    float learningRateW, float learningRateB)
{
    const double gradWNorm  = FroNormEvalHost(gradW);
    const double gradBNorm  = FroNormEvalHost(gradB);
    const double paramWNorm = FroNormEvalHost(paramW);
    const double paramBNorm = FroNormEvalHost(paramB);

    const double updateWNorm = std::fabs(static_cast<double>(learningRateW)) * gradWNorm;
    const double updateBNorm = std::fabs(static_cast<double>(learningRateB)) * gradBNorm;
    std::cout
        << "DIAG_HEAD_GRAD_NORM"
        << ",tag=" << tag
        << ",d_headDirW_norm=" << gradWNorm
        << ",d_headDirB_norm=" << gradBNorm
        << ",headDirW_norm=" << paramWNorm
        << ",headDirB_norm=" << paramBNorm
        << ",learningRateW=" << learningRateW
        << ",learningRateB=" << learningRateB
        << ",updateW_norm=" << updateWNorm
        << ",updateB_norm=" << updateBNorm
        << ",updateW_to_paramW="
        << (paramWNorm > 0.0 ? updateWNorm / paramWNorm : 0.0)
        << ",updateB_to_paramB="
        << (paramBNorm > 0.0 ? updateBNorm / paramBNorm : 0.0)
        << std::endl;
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
    HotspotScope backwardHotspot("backward_step_batch");
    const size_t B = d_h.Shape()[0];
    const size_t H = d_h.Shape()[1];

    {
        static size_t s_phase3GateDerivDiagCount = 0;
        if (s_phase3GateDerivDiagCount < LSTM_PHASE3_HEAD_DIAG_LIMIT)
        {
            auto phase3GateDerivStats = [](const EAMatrix& gate,
                                           bool sigmoidGate,
                                           double& derivMean,
                                           double& derivMax) -> void
            {
                auto low = MetaNN::LowerAccess(gate);
                const float* ptr = low.RawMemory();
                const size_t rows = gate.Shape()[0];
                const size_t cols = gate.Shape()[1];
                const size_t count = rows * cols;

                long double sum = 0.0L;
                double maxVal = 0.0;
                for (size_t idx = 0; idx < count; ++idx)
                {
                    const double v = static_cast<double>(ptr[idx]);
                    const double rawDeriv = sigmoidGate ? (v * (1.0 - v)) : (1.0 - v * v);
                    const double deriv = rawDeriv > 0.0 ? rawDeriv : 0.0;
                    sum += static_cast<long double>(deriv);
                    if (deriv > maxVal)
                        maxVal = deriv;
                }

                derivMean = count ? static_cast<double>(sum / static_cast<long double>(count)) : 0.0;
                derivMax = maxVal;
            };

            double iDerivMean = 0.0;
            double iDerivMax = 0.0;
            double fDerivMean = 0.0;
            double fDerivMax = 0.0;
            double oDerivMean = 0.0;
            double oDerivMax = 0.0;
            double gDerivMean = 0.0;
            double gDerivMax = 0.0;

            phase3GateDerivStats(sc.i, true, iDerivMean, iDerivMax);
            phase3GateDerivStats(sc.f, true, fDerivMean, fDerivMax);
            phase3GateDerivStats(sc.o, true, oDerivMean, oDerivMax);
            phase3GateDerivStats(sc.g, false, gDerivMean, gDerivMax);

            std::cout << "DIAG_BPTT_GATE_DERIV_"
                      << ",call=" << s_phase3GateDerivDiagCount
                      << ",rows=" << B
                      << ",hidden_size=" << H
                      << ",i_sigmoid_deriv_mean=" << iDerivMean
                      << ",i_sigmoid_deriv_max=" << iDerivMax
                      << ",f_sigmoid_deriv_mean=" << fDerivMean
                      << ",f_sigmoid_deriv_max=" << fDerivMax
                      << ",o_sigmoid_deriv_mean=" << oDerivMean
                      << ",o_sigmoid_deriv_max=" << oDerivMax
                      << ",g_tanh_deriv_mean=" << gDerivMean
                      << ",g_tanh_deriv_max=" << gDerivMax
                      << std::endl;

            ++s_phase3GateDerivDiagCount;
        }
    }

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
        HotspotScope hotspot("d_gates_batch_packing");
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

    {
        HotspotScope hotspot("backward_gemms");
        MetaNN::EvalPlan::Inst().Eval();
    }

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

    {
        HotspotScope hotspot("gate_accumulator_split_merge");
        addColsToGateAccum(A.dW_i, dW_catH.Data(), 0 * H);
        addColsToGateAccum(A.dW_f, dW_catH.Data(), 1 * H);
        addColsToGateAccum(A.dW_g, dW_catH.Data(), 2 * H);
        addColsToGateAccum(A.dW_o, dW_catH.Data(), 3 * H);
    }

    auto addBiasColsToGateAccum = [&](auto& dst, const EAMatrix& src, size_t colOffset)
    {
        auto lowD = MetaNN::LowerAccess(dst);
        auto* dptr = lowD.MutableRawMemory();

        auto lowS = MetaNN::LowerAccess(src);
        const auto* sptr = lowS.RawMemory();

        for (size_t c = 0; c < H; ++c)
            dptr[c] += static_cast<AccumScalar>(sptr[colOffset + c]);
    };

    {
        HotspotScope hotspot("gate_accumulator_split_merge");
        addBiasColsToGateAccum(A.db_i, db_catH.Data(), 0 * H);
        addBiasColsToGateAccum(A.db_f, db_catH.Data(), 1 * H);
        addBiasColsToGateAccum(A.db_g, db_catH.Data(), 2 * H);
        addBiasColsToGateAccum(A.db_o, db_catH.Data(), 3 * H);
    }

    {
        static size_t s_phase3RecurrentBackflowDiagCount = 0;
        if (s_phase3RecurrentBackflowDiagCount < LSTM_PHASE3_HEAD_DIAG_LIMIT)
        {
            auto matrix_l2_norm = [](const EAMatrix& m) -> double
            {
                auto low = MetaNN::LowerAccess(m);
                const float* ptr = low.RawMemory();
                const size_t count = m.Shape()[0] * m.Shape()[1];

                long double sumSq = 0.0L;

                for (size_t i = 0; i < count; ++i)
                {
                    const double v = static_cast<double>(ptr[i]);
                    sumSq += v * v;
                }

                return std::sqrt(static_cast<double>(sumSq));
            };

            auto dh_prev_i_expr =
                MetaNN::Dot(d_i_mat, MetaNN::Transpose(gb.W_hi));

            auto dh_prev_f_expr =
                MetaNN::Dot(d_f_mat, MetaNN::Transpose(gb.W_hf));

            auto dh_prev_g_expr =
                MetaNN::Dot(d_g_mat, MetaNN::Transpose(gb.W_hg));

            auto dh_prev_o_expr =
                MetaNN::Dot(d_o_mat, MetaNN::Transpose(gb.W_ho));

            auto dhiH = dh_prev_i_expr.EvalRegister();
            auto dhfH = dh_prev_f_expr.EvalRegister();
            auto dhgH = dh_prev_g_expr.EvalRegister();
            auto dhoH = dh_prev_o_expr.EvalRegister();

            MetaNN::EvalPlan::Inst().Eval();

            const double dhPrevINorm =
                matrix_l2_norm(dhiH.Data());

            const double dhPrevFNorm =
                matrix_l2_norm(dhfH.Data());

            const double dhPrevGNorm =
                matrix_l2_norm(dhgH.Data());

            const double dhPrevONorm =
                matrix_l2_norm(dhoH.Data());

            const double dhPrevTotalNorm =
                matrix_l2_norm(dh_prevH.Data());

            std::cout
                << "DIAG_BPTT_RECURRENT_BACKFLOW_"
                << ",call=" << s_phase3RecurrentBackflowDiagCount
                << ",rows=" << B
                << ",hidden_size=" << H
                << ",dh_prev_i_norm=" << dhPrevINorm
                << ",dh_prev_f_norm=" << dhPrevFNorm
                << ",dh_prev_g_norm=" << dhPrevGNorm
                << ",dh_prev_o_norm=" << dhPrevONorm
                << ",dh_prev_total_norm=" << dhPrevTotalNorm
                << std::endl;

            ++s_phase3RecurrentBackflowDiagCount;
        }
    }
    d_h = dh_prevH.Data();
    d_c = dc_prevH.Data();
}

inline void EA::LSTM::mergeGateAccumulators(const GateAccumulators& A,
                                     MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>& d_param_accum,
                                     MetaNN::Matrix<AccumScalar, MetaNN::DeviceTags::Metal>& d_bias_accum,
                                     size_t H) const
{
    HotspotScope hotspot("gate_accumulator_split_merge");
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
    InitializeBiasWithForgetGateOffset(1.5f);
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
                const size_t cols = returnHeadDirWeight.Shape()[1];
                const float scale = 0.05f; // small random init to break symmetry
                for (size_t i = 0; i < hidden_size; ++i)
                {
                    for (size_t j = 0; j < cols; ++j)
                    {
                        wp[i * cols + j] = uniform_symmetric(scale);
                    }
                }
            }
            {
                auto lowB = MetaNN::LowerAccess(returnHeadDirBias);
                float* bp = lowB.MutableRawMemory();
                for (size_t j = 0; j < returnHeadDirBias.Shape()[1]; ++j)
                {
                    bp[j] = 0.0f;
                }
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

std::tuple<float, size_t, size_t> EA::LSTM::CalculateBatch(Window batch, unsigned short epochIdx)
{
    static size_t s_calcBatchCalls = 0;
    const size_t calcBatchCallIdx = s_calcBatchCalls++;
    const bool isFirstBatchCall = (calcBatchCallIdx == 0);
    EAMatrix head_logits_batch(effectiveMiniBatchWindows, direction_output_size);
    EAMatrix phase3HeadDeltaH(1, hidden_size);
    EAMatrix phase3HeadDeltaLogitsBefore(1, direction_output_size);
    EAMatrix phase3HeadDeltaProbsBefore(1, direction_output_size);
    std::vector<int> phase3HeadDeltaActualClasses;
    bool phase3HeadDeltaCaptured = false;
    double sse = 0.0;
    size_t mseCount = 0;
    size_t windowCount = 0;
    double phase3ClassWeightSum = 0.0;
    size_t windowsInBatch = 0;
    size_t skippedWindows = 0;
    LSTMBatchProfile profile;

    // Class counts for 3-class targets
    size_t up_count = 0;
    size_t down_count = 0;
    size_t neutral_count = 0;
    size_t pred_down_count = 0;
    size_t pred_neutral_count = 0;
    size_t pred_up_count = 0;

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

    // Debug stats for training targets (log returns) and sample predictions
    double y_sum = 0.0, y_sumsq = 0.0;
    float y_min = std::numeric_limits<float>::infinity();
    float y_max = -std::numeric_limits<float>::infinity();
    size_t y_count = 0;
    std::vector<float> yhat_samples;
    std::vector<float> ydenorm_samples; // predicted price delta (predicted_close - close_T)

    // Per-batch predicted/actual log-return stats

    ResetPreviousState();

    // Prebuild the batch rows into one contiguous (num_rows, F) tensor once so
    // minibatch window assembly can memcpy directly from a contiguous source
    // instead of repeatedly materializing overlapping windows via GetWindow().
    const size_t batchRows = static_cast<size_t>(batch.end() - batch.begin());
    const size_t batchGlobalStartIdx = static_cast<size_t>(batch.begin() - t.begin());
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
    if (isFirstBatchCall)
    {
        std::cout << "DIAG_FEATURE_CONFIG"
                  << ",phase=train"
                  << ",base_feature_cols=" << baseFeatureCount
                  << ",appended_return_feature_cols=" << kReturnFeatureCount
                  << ",model_feature_cols=" << modelFeatureCount
                  << ",feature_uses_future_values=0"
                  << std::endl;
    }
    {
        auto lowPrebuilt = MetaNN::LowerAccess(prebuilt_rows);
        float* dst = lowPrebuilt.MutableRawMemory();
        size_t r = 0;
        for (auto it = batch.begin(); it != batch.end(); ++it, ++r)
        {
            auto lowRow = MetaNN::LowerAccess(*it);
            const float* src = lowRow.RawMemory();
            float* dstRow = dst + r * featureCount;

            if (useReturnFeatures)
            {
                HotspotScope hotspot("appended_return_features");
                const size_t appended = EA::AssembleLstmModelInputRowAtGlobalPosition(
                    dstRow, src, baseFeatureCount, batchGlobalStartIdx + r, EA::LSTM::kFeatScale,
                    [this](size_t globalPosition)
                    {
                        return t.RawCloseAtIterator(
                            t.begin() + static_cast<std::ptrdiff_t>(globalPosition));
                    });
#if LSTM_TRAINING_ASSERTS
                LSTM_ASSERT(appended == kReturnFeatureCount,
                            "CalculateBatch: appended return feature count mismatch");
#endif
            }
            else
            {
                std::memcpy(dstRow, src, baseFeatureCount * sizeof(float));
            }

            for (size_t c = 0; c < featureCount; ++c)
            {
                const float v = dstRow[c];
                if (!std::isfinite(v))
                {
                    std::cout << "DIAG_NORM_FAIL"
                              << ",phase=train"
                              << ",row=" << r
                              << ",col=" << c
                              << ",value=" << v
                              << std::endl;
                    LSTM_ASSERT(false, "CalculateBatch: non-finite feature detected before LSTM");
                }
            }
        }
    }

    if (isFirstBatchCall)
        PrintPhase2MatrixDiagnostics("DIAG_FEATURE_PRELSTM", "DIAG_FEATURE_WARN", prebuilt_rows, 50.0, 1e-6);

    {
        auto lowPrebuilt = MetaNN::LowerAccess(prebuilt_rows);
        float* dst = lowPrebuilt.MutableRawMemory();
        for (size_t i = 0; i < batchRows * featureCount; ++i)
            dst[i] = std::clamp(dst[i], -10.0f, 10.0f);
    }

    if (isFirstBatchCall)
        PrintPhase2MatrixDiagnostics("DIAG_NORM_PRELSTM", "DIAG_NORM_WARN", prebuilt_rows, 9.99, 1e-6);

    struct Phase3LabelReturnSeparationStats
    {
        size_t count = 0;
        size_t horizonCount = 0;
        double terminalLogReturnSum = 0.0;
        double maxFutureHighLogReturnSum = 0.0;
        double minFutureLowLogReturnSum = 0.0;
        double rangeToThresholdRatioSum = 0.0;
    };
    std::array<Phase3LabelReturnSeparationStats, direction_output_size> phase3LabelReturnStats {};
    auto buildWindowBatch = [&](auto first, auto last) -> WindowBatch
    {
        HotspotScope hotspot("window_batch_build");
#if LSTM_BATCH_PROFILE
        LSTMScopedProfileTimer timer(profile.build_window_batch_us);
#endif
        static size_t s_targetIndexDiagCount = 0;
        constexpr size_t kTargetIndexDiagLimit = 50;
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

        // NOTE: For faster learning and less noise, consider reducing prediction_horizon
        // to a shorter range (e.g., 1–4 timesteps) instead of larger horizons.
        for (auto it = first; it != last; ++it)
        {
            const size_t start = *it;
            const size_t lastIdx   = start + (window_size - 1);
            const size_t targetIdx = lastIdx + prediction_horizon;
            const size_t globalStartIdx = batchGlobalStartIdx + start;
            const size_t globalLastIdx = batchGlobalStartIdx + lastIdx;
            const size_t globalTargetIdx = batchGlobalStartIdx + targetIdx;
            const auto lastIt      = t.begin() + static_cast<std::ptrdiff_t>(globalLastIdx);
            const auto targetIt    = t.begin() + static_cast<std::ptrdiff_t>(globalTargetIdx);
            const float close_t_local      = t.RawCloseAtIterator(lastIt);
            const float close_target_local = t.RawCloseAtIterator(targetIt);
            const float y_true_logret =
                (std::isfinite(close_t_local) && std::isfinite(close_target_local) &&
                 close_t_local > 0.0f && close_target_local > 0.0f)
                    ? std::log(close_target_local / close_t_local)
                    : 0.0f;

            float regressionTarget = 0.0f;
            int classTarget = 1;

            if (targetType == TargetType::UpNeutralDownReturn)
            {
                const auto labelInfo = BuildLookaheadClassInfo(t, t.begin() + static_cast<std::ptrdiff_t>(globalStartIdx));
                classTarget = labelInfo.assignedClass;
                LSTM_ASSERT(classTarget >= 0 && classTarget < static_cast<int>(direction_output_size),
                            "CalculateBatch: classTarget out of [0,2]");
                double maxFutureHighLogReturn = -std::numeric_limits<double>::infinity();
                double minFutureLowLogReturn = std::numeric_limits<double>::infinity();
                size_t validHorizonReturnCount = 0;

                for (size_t lookahead = 1; lookahead <= prediction_horizon; ++lookahead)
                {
                    const size_t globalFutureIdx = globalLastIdx + lookahead;
                    const auto futureIt = t.begin() + static_cast<std::ptrdiff_t>(globalFutureIdx);
                    const float futureHigh = t.RawHighAtIterator(futureIt);
                    const float futureLow = t.RawLowAtIterator(futureIt);

                    if (std::isfinite(close_t_local) && close_t_local > 0.0f &&
                        std::isfinite(futureHigh) && futureHigh > 0.0f)
                    {
                        const double highLogReturn =
                            static_cast<double>(std::log(futureHigh / close_t_local));
                        maxFutureHighLogReturn =
                            std::max(maxFutureHighLogReturn, highLogReturn);
                        ++validHorizonReturnCount;
                    }

                    if (std::isfinite(close_t_local) && close_t_local > 0.0f &&
                        std::isfinite(futureLow) && futureLow > 0.0f)
                    {
                        const double lowLogReturn =
                            static_cast<double>(std::log(futureLow / close_t_local));
                        minFutureLowLogReturn =
                            std::min(minFutureLowLogReturn, lowLogReturn);
                    }
                }
                

                auto& s = phase3LabelReturnStats[static_cast<size_t>(classTarget)];
                ++s.count;  s.horizonCount += validHorizonReturnCount;
                
                const double terminalLogReturn =
                    (std::isfinite(close_t_local) && close_t_local > 0.0f &&
                     std::isfinite(close_target_local) && close_target_local > 0.0f)
                        ? static_cast<double>(std::log(close_target_local / close_t_local))
                        : 0.0;
                
                const double highLogReturn =
                    (std::isfinite(close_t_local) && close_t_local > 0.0f &&
                     std::isfinite(labelInfo.selectedFutureHigh) && labelInfo.selectedFutureHigh > 0.0f)
                        ? static_cast<double>(std::log(labelInfo.selectedFutureHigh / close_t_local))
                        : 0.0;

                const double lowLogReturn =
                    (std::isfinite(close_t_local) && close_t_local > 0.0f &&
                     std::isfinite(labelInfo.selectedFutureLow) && labelInfo.selectedFutureLow > 0.0f)
                        ? static_cast<double>(std::log(labelInfo.selectedFutureLow / close_t_local))
                        : 0.0;
                
                const double thresholdDenom =
                    std::max(static_cast<double>(std::fabs(c_next_threshold)), 1.0e-12);

                s.terminalLogReturnSum += terminalLogReturn;
                s.maxFutureHighLogReturnSum += highLogReturn;
                s.minFutureLowLogReturnSum += lowLogReturn;
                s.rangeToThresholdRatioSum += (highLogReturn - lowLogReturn) / thresholdDenom;
                
                if (s_targetIndexDiagCount < kTargetIndexDiagLimit)
                {
                    const size_t globalFutureIdx = globalLastIdx + labelInfo.selectedOffset;
                    const size_t miniBatchRow = static_cast<size_t>(it - first);

                    std::cout << "DIAG_TARGET_INDEX"
                              << ",batch_row_idx=" << start
                              << ",mini_batch_row_idx=" << miniBatchRow
                              << ",batch_start_global_idx=" << batchGlobalStartIdx
                              << ",global_tensor_row_idx=" << globalStartIdx
                              << ",last_row_idx=" << globalLastIdx
                              << ",target_row_idx=" << globalTargetIdx
                              << ",future_row_idx=" << globalFutureIdx
                              << ",close_t=" << close_t_local
                              << ",close_target=" << close_target_local
                              << ",futureHigh=" << labelInfo.selectedFutureHigh
                              << ",futureLow=" << labelInfo.selectedFutureLow
                              << ",assigned_class=" << classTarget
                              << std::endl;
                    ++s_targetIndexDiagCount;
                }
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

            wb.close_t.push_back(close_t_local);
            wb.close_target.push_back(close_target_local);
            if (targetType == TargetType::UpNeutralDownReturn) wb.classTargets.push_back(classTarget);
            else wb.targets.push_back(regressionTarget);

            auto lowPrebuilt = MetaNN::LowerAccess(prebuilt_rows);
            const float* prebuiltPtr = lowPrebuilt.RawMemory();
            const size_t batchRow = static_cast<size_t>(it - first);
            for (size_t tstep = 0; tstep < window_size; ++tstep)
            {
                float* dstRow = packed_step_ptrs[tstep] + batchRow * F;
                const float* srcRow = prebuiltPtr + (start + tstep) * F;
                std::memcpy(dstRow, srcRow, F * sizeof(float));
            }
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

    for (size_t batchBase = 0; batchBase < allStarts.size(); batchBase += effectiveMiniBatchWindows)
    {
        const bool isFirstMiniBatch = (batchBase == 0);
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
#if LSTM_HEAVY_DIAG
            if (isFirstBatchCall && isFirstMiniBatch && tstep == 0) PrintMatrixSummary("DIAG_X_T_BATCH_T0", wb.packed_steps[tstep]);
#endif
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
#if LSTM_HEAVY_DIAG
        if (isFirstBatchCall && isFirstMiniBatch)
        {
            PrintMatrixSummary("DIAG_DIRHEAD_INPUT_h_batch", h_batch);
            PrintMatrixSummary("DIAG_DIRHEAD_CELL_c_batch", c_batch);
            PrintMatrixSummary("DIAG_DIRHEAD_WEIGHT", returnHeadDirWeight);
            PrintMatrixSummary("DIAG_DIRHEAD_BIAS", returnHeadDirBias);
        }
#endif

        std::vector<float> errs(B, 0.0f);
        EAMatrix d_logits_batch(1, direction_output_size); // for 3-class path (declared once for scope)

        // Batched head forward and loss on (B, H)
        if (targetType == TargetType::UpNeutralDownReturn)
        {
            static size_t s_phase3HeadDiagCount = 0;
            const size_t phase3HeadDiagIdx = s_phase3HeadDiagCount++;
            const bool phase3HeadDiagEnabled = (phase3HeadDiagIdx < LSTM_PHASE3_HEAD_DIAG_LIMIT);
            if (head_logits_batch.Shape()[0] != B || head_logits_batch.Shape()[1] != direction_output_size)
                head_logits_batch = EAMatrix(B, direction_output_size);
            if (phase3HeadDiagEnabled)
            {
                PrintPhase3Stats("DIAG_HEAD_INPUT_", phase3HeadDiagIdx, "h_batch",
                                 Phase3StatsWhole(h_batch, -20.0, 20.0));
                // --- BEGIN PATCH: DIAG HEAD INPUT BY CLASS SEPARATION ---
                {
                    const auto hHostByClass = Phase3MaterializeHost(h_batch);

                    std::array<size_t, direction_output_size> clsCount {0, 0, 0};
                    std::array<long double, direction_output_size> clsSum {0.0L, 0.0L, 0.0L};
                    std::array<long double, direction_output_size> clsSqSum {0.0L, 0.0L, 0.0L};
                    std::array<double, direction_output_size> clsAbsMax {0.0, 0.0, 0.0};
                    std::array<std::vector<long double>, direction_output_size> centroidSum;

                    const size_t rows = hHostByClass.rows;
                    const size_t cols = hHostByClass.cols;

                    for (size_t cls = 0; cls < direction_output_size; ++cls)
                        centroidSum[cls].assign(cols, 0.0L);

                    size_t invalidClassRows = 0;
                    size_t nonFiniteRows = 0;

                    for (size_t r = 0; r < rows && r < wb.classTargets.size(); ++r)
                    {
                        const int clsInt = wb.classTargets[r];
                        if (clsInt < 0 || clsInt >= static_cast<int>(direction_output_size))
                        {
                            ++invalidClassRows;
                            continue;
                        }

                        const size_t cls = static_cast<size_t>(clsInt);
                        bool rowFinite = true;
                        for (size_t h = 0; h < cols; ++h)
                        {
                            const double v = static_cast<double>(hHostByClass.data[r * cols + h]);
                            if (!std::isfinite(v))
                            {
                                rowFinite = false;
                                break;
                            }
                        }
                        if (!rowFinite)
                        {
                            ++nonFiniteRows;
                            continue;
                        }

                        ++clsCount[cls];
                        for (size_t h = 0; h < cols; ++h)
                        {
                            const double v = static_cast<double>(hHostByClass.data[r * cols + h]);
                            clsSum[cls] += static_cast<long double>(v);
                            clsSqSum[cls] += static_cast<long double>(v) * static_cast<long double>(v);
                            clsAbsMax[cls] = std::max(clsAbsMax[cls], std::fabs(v));
                            centroidSum[cls][h] += static_cast<long double>(v);
                        }
                    }

                    std::array<std::vector<double>, direction_output_size> centroid;
                    std::array<double, direction_output_size> centroidNorm {0.0, 0.0, 0.0};
                    for (size_t cls = 0; cls < direction_output_size; ++cls)
                    {
                        centroid[cls].assign(cols, 0.0);
                        long double centroidSq = 0.0L;
                        if (clsCount[cls] > 0)
                        {
                            const long double denom = static_cast<long double>(clsCount[cls]);
                            for (size_t h = 0; h < cols; ++h)
                            {
                                const double meanH = static_cast<double>(centroidSum[cls][h] / denom);
                                centroid[cls][h] = meanH;
                                centroidSq += static_cast<long double>(meanH) * static_cast<long double>(meanH);
                            }
                        }
                        centroidNorm[cls] = std::sqrt(static_cast<double>(centroidSq));
                    }

                    auto centroidDistance = [&](size_t a, size_t b) -> double
                    {
                        if (clsCount[a] == 0 || clsCount[b] == 0)
                            return 0.0;
                        long double ss = 0.0L;
                        for (size_t h = 0; h < cols; ++h)
                        {
                            const long double d = static_cast<long double>(centroid[a][h]) -
                                                  static_cast<long double>(centroid[b][h]);
                            ss += d * d;
                        }
                        return std::sqrt(static_cast<double>(ss));
                    };

                    for (size_t cls = 0; cls < direction_output_size; ++cls)
                    {
                        const size_t elemCount = clsCount[cls] * cols;
                        const double mean = elemCount > 0
                            ? static_cast<double>(clsSum[cls] / static_cast<long double>(elemCount))
                            : 0.0;
                        const double meanSq = elemCount > 0
                            ? static_cast<double>(clsSqSum[cls] / static_cast<long double>(elemCount))
                            : 0.0;
                        const double var = std::max(0.0, meanSq - mean * mean);

                        std::cout << "DIAG_HEAD_INPUT_BY_CLASS_"
                                  << ",call=" << phase3HeadDiagIdx
                                  << ",class=" << Phase3ClassName(cls)
                                  << ",count=" << clsCount[cls]
                                  << ",h_mean=" << mean
                                  << ",h_std=" << std::sqrt(var)
                                  << ",h_absmax=" << clsAbsMax[cls]
                                  << ",centroid_norm=" << centroidNorm[cls]
                                  << std::endl;
                    }

                    std::cout << "DIAG_HEAD_CLASS_SEPARATION_"
                              << ",call=" << phase3HeadDiagIdx
                              << ",rows=" << rows
                              << ",hidden_size=" << cols
                              << ",actual_down=" << clsCount[0]
                              << ",actual_neutral=" << clsCount[1]
                              << ",actual_up=" << clsCount[2]
                              << ",down_neutral_dist=" << centroidDistance(0, 1)
                              << ",down_up_dist=" << centroidDistance(0, 2)
                              << ",neutral_up_dist=" << centroidDistance(1, 2)
                              << ",invalid_class_rows=" << invalidClassRows
                              << ",nonfinite_rows=" << nonFiniteRows
                              << std::endl;
                }
                // --- END PATCH ---
                for (size_t cls = 0; cls < direction_output_size; ++cls)
                    PrintPhase3Stats("DIAG_HEAD_WEIGHT_", phase3HeadDiagIdx, Phase3ClassName(cls),
                                     Phase3StatsCols(returnHeadDirWeight, cls, 1, -20.0, 20.0));
                PrintPhase3Stats("DIAG_HEAD_WEIGHT_", phase3HeadDiagIdx, "bias",
                                 Phase3StatsWhole(returnHeadDirBias, -20.0, 20.0));
            }
#if LSTM_HEAVY_DIAG
            if (isFirstBatchCall && isFirstMiniBatch)    PrintMatrixSummary("DIAG_PRE_HEAD_h_batch", h_batch);
#endif
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
#if LSTM_SHAPE_DIAG
                static bool s_printed_head_matmul_shapes = false;
                if (!s_printed_head_matmul_shapes)
                {
                    std::cout << "DIAG_HEAD_MATMUL_SHAPE"
                              << ",m=" << B
                              << ",k=" << hidden_size
                              << ",n=" << direction_output_size
                              << std::endl;
                    s_printed_head_matmul_shapes = true;
                }
#endif
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
#if LSTM_SHAPE_DIAG
                static bool s_printed_head_matmul_shapes = false;
                if (!s_printed_head_matmul_shapes)
                {
                    std::cout << "DIAG_HEAD_MATMUL_SHAPE"
                              << ",m=" << B
                              << ",k=" << hidden_size
                              << ",n=" << direction_output_size
                              << std::endl;
                    s_printed_head_matmul_shapes = true;
                }
#endif
            }
// --- BEGIN PATCH: DIAG RAW LOGITS BEFORE SOFTMAX ---
            if (phase3HeadDiagEnabled)
            {
                const auto phase3LogitsHost = Phase3MaterializeHost(head_logits_batch);
                const auto phase3BiasHost = Phase3MaterializeHost(returnHeadDirBias);

                const size_t phase3Rows = phase3LogitsHost.rows;
                const size_t phase3Cols = phase3LogitsHost.cols;

                std::array<size_t, direction_output_size> phase3WinHist {0, 0, 0};
                std::array<size_t, direction_output_size> phase3ActualHist {0, 0, 0};
                std::array<long double, direction_output_size> phase3ChSum {0.0L, 0.0L, 0.0L};
                std::array<long double, direction_output_size> phase3ChSqSum {0.0L, 0.0L, 0.0L};
                std::array<double, direction_output_size> phase3ChMin {0.0, 0.0, 0.0};
                std::array<double, direction_output_size> phase3ChMax {0.0, 0.0, 0.0};
                std::array<std::array<long double, direction_output_size>, direction_output_size> phase3ActualLogitSum {{
                    {0.0L, 0.0L, 0.0L},
                    {0.0L, 0.0L, 0.0L},
                    {0.0L, 0.0L, 0.0L}
                }};

                size_t phase3FiniteRows = 0;
                size_t phase3NonFiniteRows = 0;
                size_t phase3InvalidActualRows = 0;

                if (phase3Cols == direction_output_size)
                {
                    bool phase3MinMaxInitialized = false;
                    for (size_t r = 0; r < phase3Rows; ++r)
                    {
                        bool rowFinite = true;
                        for (size_t c = 0; c < direction_output_size; ++c)
                        {
                            const double v = static_cast<double>(phase3LogitsHost.data[r * phase3Cols + c]);
                            if (!std::isfinite(v))
                            {
                                rowFinite = false;
                                break;
                            }
                        }

                        if (!rowFinite)
                        {
                            ++phase3NonFiniteRows;
                            continue;
                        }

                        ++phase3FiniteRows;

                        size_t winClass = 0;
                        double bestLogit = static_cast<double>(phase3LogitsHost.data[r * phase3Cols + 0]);

                        for (size_t c = 0; c < direction_output_size; ++c)
                        {
                            const double v = static_cast<double>(phase3LogitsHost.data[r * phase3Cols + c]);
                            if (!phase3MinMaxInitialized)
                            {
                                phase3ChMin[c] = v;
                                phase3ChMax[c] = v;
                            }
                            else
                            {
                                phase3ChMin[c] = std::min(phase3ChMin[c], v);
                                phase3ChMax[c] = std::max(phase3ChMax[c], v);
                            }

                            phase3ChSum[c] += static_cast<long double>(v);
                            phase3ChSqSum[c] += static_cast<long double>(v) * static_cast<long double>(v);

                            if (c == 0 || v > bestLogit)
                            {
                                bestLogit = v;
                                winClass = c;
                            }
                        }
                        phase3MinMaxInitialized = true;
                        ++phase3WinHist[winClass];

                        if (r < wb.classTargets.size())
                        {
                            const int clsInt = wb.classTargets[r];
                            if (clsInt < 0 || clsInt >= static_cast<int>(direction_output_size))
                            {
                                ++phase3InvalidActualRows;
                            }
                            else
                            {
                                const size_t actual = static_cast<size_t>(clsInt);
                                ++phase3ActualHist[actual];
                                for (size_t c = 0; c < direction_output_size; ++c)
                                {
                                    phase3ActualLogitSum[actual][c] += static_cast<long double>(
                                        phase3LogitsHost.data[r * phase3Cols + c]);
                                }
                            }
                        }
                        else
                        {
                            ++phase3InvalidActualRows;
                        }
                    }
                }
                else
                {
                    phase3NonFiniteRows = phase3Rows;
                }

                std::array<double, direction_output_size> phase3ChMean {0.0, 0.0, 0.0};
                std::array<double, direction_output_size> phase3ChStd {0.0, 0.0, 0.0};
                for (size_t c = 0; c < direction_output_size; ++c)
                {
                    if (phase3FiniteRows > 0)
                    {
                        phase3ChMean[c] = static_cast<double>(
                            phase3ChSum[c] / static_cast<long double>(phase3FiniteRows));
                        const double meanSq = static_cast<double>(
                            phase3ChSqSum[c] / static_cast<long double>(phase3FiniteRows));
                        phase3ChStd[c] = std::sqrt(std::max(0.0, meanSq - phase3ChMean[c] * phase3ChMean[c]));
                    }
                }

                auto phase3AvgActualLogit = [&](size_t actual, size_t channel) -> double
                {
                    return phase3ActualHist[actual] > 0
                        ? static_cast<double>(phase3ActualLogitSum[actual][channel] /
                                              static_cast<long double>(phase3ActualHist[actual]))
                        : 0.0;
                };

                auto phase3SafeBias = [&](size_t idx) -> double
                {
                    return idx < phase3BiasHost.data.size()
                        ? static_cast<double>(phase3BiasHost.data[idx])
                        : 0.0;
                };

                std::cout << "DIAG_LOGITS_"
                          << ",call=" << phase3HeadDiagIdx
                          << ",rows=" << phase3Rows
                          << ",cols=" << phase3Cols
                          << ",finite_rows=" << phase3FiniteRows
                          << ",nonfinite_rows=" << phase3NonFiniteRows
                          << ",invalid_actual_rows=" << phase3InvalidActualRows
                          << ",down_min=" << phase3ChMin[0]
                          << ",down_max=" << phase3ChMax[0]
                          << ",down_mean=" << phase3ChMean[0]
                          << ",down_std=" << phase3ChStd[0]
                          << ",neutral_min=" << phase3ChMin[1]
                          << ",neutral_max=" << phase3ChMax[1]
                          << ",neutral_mean=" << phase3ChMean[1]
                          << ",neutral_std=" << phase3ChStd[1]
                          << ",up_min=" << phase3ChMin[2]
                          << ",up_max=" << phase3ChMax[2]
                          << ",up_mean=" << phase3ChMean[2]
                          << ",up_std=" << phase3ChStd[2]
                          << ",actual_down=" << phase3ActualHist[0]
                          << ",actual_neutral=" << phase3ActualHist[1]
                          << ",actual_up=" << phase3ActualHist[2]
                          << ",win_down=" << phase3WinHist[0]
                          << ",win_neutral=" << phase3WinHist[1]
                          << ",win_up=" << phase3WinHist[2]
                          << ",bias_down=" << phase3SafeBias(0)
                          << ",bias_neutral=" << phase3SafeBias(1)
                          << ",bias_up=" << phase3SafeBias(2)
                          << ",actual_down_logit_down_mean=" << phase3AvgActualLogit(0, 0)
                          << ",actual_down_logit_neutral_mean=" << phase3AvgActualLogit(0, 1)
                          << ",actual_down_logit_up_mean=" << phase3AvgActualLogit(0, 2)
                          << ",actual_neutral_logit_down_mean=" << phase3AvgActualLogit(1, 0)
                          << ",actual_neutral_logit_neutral_mean=" << phase3AvgActualLogit(1, 1)
                          << ",actual_neutral_logit_up_mean=" << phase3AvgActualLogit(1, 2)
                          << ",actual_up_logit_down_mean=" << phase3AvgActualLogit(2, 0)
                          << ",actual_up_logit_neutral_mean=" << phase3AvgActualLogit(2, 1)
                          << ",actual_up_logit_up_mean=" << phase3AvgActualLogit(2, 2)
                          << std::endl;
            }
            // --- END PATCH ---
#endif
#if LSTM_HEAVY_DIAG
            static bool s_printed_batch_dir_logits = false;
            if (!s_printed_batch_dir_logits)
            {
                MetaNN::NSMetalMatMul::WaitForAll();
                PrintMatrixSummary("DIAG_BATCH_DIRHEAD_LOGITS", head_logits_batch);
                s_printed_batch_dir_logits = true;
            }
#endif
            if (!phase3HeadDeltaCaptured && phase3HeadDiagEnabled)
            {
                phase3HeadDeltaH = NNUtils::DeepCopyMatrix(h_batch);
                phase3HeadDeltaLogitsBefore = NNUtils::DeepCopyMatrix(head_logits_batch);
                phase3HeadDeltaProbsBefore = Phase3SoftmaxProbsCpu(phase3HeadDeltaLogitsBefore);
                phase3HeadDeltaActualClasses.assign(wb.classTargets.begin(),
                                                     wb.classTargets.begin() + static_cast<std::ptrdiff_t>(std::min(B, wb.classTargets.size())));
                phase3HeadDeltaCaptured = true;
            }
            d_logits_batch = EAMatrix(B, direction_output_size);
            auto lowLogits = MetaNN::LowerAccess(head_logits_batch);
            const float* lptr = lowLogits.RawMemory();
            auto lowD = MetaNN::LowerAccess(d_logits_batch);
            float* dptr = lowD.MutableRawMemory();
            double totalSampleWeight = 0.0;
            std::array<size_t, direction_output_size> actualHist {0, 0, 0};
            std::array<size_t, direction_output_size> predHist {0, 0, 0};
            std::array<size_t, direction_output_size> correctHist {0, 0, 0};
            std::array<std::array<size_t, direction_output_size>, direction_output_size> confusionByActual {};
            std::array<double, direction_output_size> weightedLossByClass {0.0, 0.0, 0.0};
            std::array<Phase3MatrixStats, direction_output_size> probStats {};
            std::array<std::array<double, direction_output_size>, direction_output_size> logitSumByActual {};
            std::array<std::array<double, direction_output_size>, direction_output_size> probSumByActual {};
            std::array<double, direction_output_size> hiddenHeadProjectionMean {0.0, 0.0, 0.0};
            std::array<double, direction_output_size> hiddenHeadProjectionAbsMean {0.0, 0.0, 0.0};
            const auto hHostForSeparation = Phase3MaterializeHost(h_batch);
            std::array<std::vector<double>, direction_output_size> hSumByActual;
            std::array<double, direction_output_size> hSumSqByActual {0.0, 0.0, 0.0};
            for (size_t cls = 0; cls < direction_output_size; ++cls)
                hSumByActual[cls].assign(hidden_size, 0.0);

            for (size_t b = 0; b < B; ++b)
            {
                const float* z = lptr + b * direction_output_size;
                float p[direction_output_size];
                Softmax3(z, p);
                const int cls = wb.classTargets[b];
                LSTM_ASSERT(cls >= 0 && cls < static_cast<int>(direction_output_size),
                            "CalculateBatch: wb.classTargets contains class id out of [0,2]");
                const float classWeight = (cls == 0) ? kClassWeightDown
                                         : (cls == 1) ? kClassWeightNeutral
                                                      : kClassWeightUp;
                constexpr float kDirectionClassGradScale = 0.1f;
                
                totalSampleWeight += static_cast<double>(classWeight);
                dptr[b * direction_output_size + 0] = kDirectionClassGradScale * classWeight * (p[0] - ((cls == 0) ? 1.0f : 0.0f));
                dptr[b * direction_output_size + 1] = kDirectionClassGradScale * classWeight * (p[1] - ((cls == 1) ? 1.0f : 0.0f));
                dptr[b * direction_output_size + 2] = kDirectionClassGradScale * classWeight * (p[2] - ((cls == 2) ? 1.0f : 0.0f));
                
                const double weightedLoss = static_cast<double>(classWeight) *
                    (-std::log(std::max(1e-12f, p[cls])));
                sse += weightedLoss;
                phase3ClassWeightSum += static_cast<double>(classWeight);
                weightedLossByClass[static_cast<size_t>(cls)] += weightedLoss;
                ++mseCount;

                if (cls == 0) ++down_count;
                else if (cls == 1) ++neutral_count;
                else ++up_count;
                ++actualHist[static_cast<size_t>(cls)];
                for (size_t pc = 0; pc < direction_output_size; ++pc)
                {
                    Phase3AddStat(probStats[pc], p[pc], 1.0e-4, 1.0 - 1.0e-4);
                    logitSumByActual[static_cast<size_t>(cls)][pc] += static_cast<double>(z[pc]);
                    probSumByActual[static_cast<size_t>(cls)][pc] += static_cast<double>(p[pc]);
                }
                if (hHostForSeparation.cols == hidden_size && b < hHostForSeparation.rows)
                {
                    const size_t clsIdx = static_cast<size_t>(cls);
                    for (size_t h = 0; h < hidden_size; ++h)
                    {
                        const double hv = static_cast<double>(hHostForSeparation.data[b * hHostForSeparation.cols + h]);
                        hSumByActual[clsIdx][h] += hv;
                        hSumSqByActual[clsIdx] += hv * hv;
                    }
                }

                {
                    const size_t clsIdx = static_cast<size_t>(cls);
                    double projection = 0.0;

                    if (hHostForSeparation.cols == hidden_size && b < hHostForSeparation.rows)
                    {
                        auto lowHeadW = MetaNN::LowerAccess(returnHeadDirWeight);
                        const float* wptr = lowHeadW.RawMemory();

                        for (size_t h = 0; h < hidden_size; ++h)
                        {
                            const double hv = static_cast<double>(
                                hHostForSeparation.data[b * hHostForSeparation.cols + h]);
                            const double wv = static_cast<double>(
                                wptr[h * direction_output_size + clsIdx]);
                            projection += hv * wv;
                        }
                    }

                    hiddenHeadProjectionMean[clsIdx] += projection;
                    hiddenHeadProjectionAbsMean[clsIdx] += std::abs(projection);
                }

                // 3-class probability bucket logic
                const float max_prob = std::max(p[0], std::max(p[1], p[2]));
                const int predClass = (p[0] > p[1] && p[0] > p[2]) ? 0 : ((p[2] > p[1] && p[2] > p[0]) ? 2 : 1);
                const bool correct = (predClass == cls);
                ++predHist[static_cast<size_t>(predClass)];
                ++confusionByActual[static_cast<size_t>(cls)][static_cast<size_t>(predClass)];
                if (correct)
                    ++correctHist[static_cast<size_t>(cls)];

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

                if (cls >= 0 && cls < static_cast<int>(direction_output_size))
                {
                    y_sum += static_cast<double>(cls);
                    y_sumsq += static_cast<double>(cls) * static_cast<double>(cls);
                    y_min = std::min(y_min, static_cast<float>(cls));
                    y_max = std::max(y_max, static_cast<float>(cls));
                    ++y_count;
                }
                else
                {
                    std::cout << "DIAG_BAD_CLASS_TARGET"
                              << ",calcBatchCall=" << calcBatchCallIdx
                              << ",miniBatchBase=" << batchBase
                              << ",row=" << b
                              << ",class=" << cls
                              << std::endl;
                }
                if (yhat_samples.size() < 10)
                    yhat_samples.push_back(static_cast<float>(predClass));
            }
            const double weightedDenom = std::max(totalSampleWeight, 1.0e-12);
            constexpr bool kDirectionLogitsUseWeightedDenom = false;

            std::cout
                << "DIAG_CLASS_WEIGHT_DENOM"
                << ",samples=" << B
                << ",weight_sum=" << totalSampleWeight
                << ",weighted_denom_for_loss=" << weightedDenom
                << ",avg_weight="
                << (B ? totalSampleWeight / static_cast<double>(B) : 0.0)
                << ",normalizes_gradients="
                << (kDirectionLogitsUseWeightedDenom ? 1 : 0)
                << ",gradient_normalization_stage=update_invN"
                << std::endl;
            
            if (phase3HeadDiagEnabled)
            {
                const size_t phase3ValidRows = actualHist[0] + actualHist[1] + actualHist[2];
                const size_t phase3PredTotal = predHist[0] + predHist[1] + predHist[2];
                const size_t phase3CorrectTotal = correctHist[0] + correctHist[1] + correctHist[2];
                auto phase3Recall = [&](size_t cls) -> double
                {
                    return actualHist[cls] > 0
                        ? static_cast<double>(correctHist[cls]) / static_cast<double>(actualHist[cls])
                        : 0.0;
                };
                const double phase3RecallDown = phase3Recall(0);
                const double phase3RecallNeutral = phase3Recall(1);
                const double phase3RecallUp = phase3Recall(2);
                const double phase3BalancedAccuracy =
                    (phase3RecallDown + phase3RecallNeutral + phase3RecallUp) / 3.0;

                std::cout << "DIAG_PRED_CLASS_COUNTS_"
                          << ",call=" << phase3HeadDiagIdx
                          << ",rows=" << B
                          << ",cols=3"
                          << ",valid_rows=" << phase3ValidRows
                          << ",actual_down=" << actualHist[0]
                          << ",actual_neutral=" << actualHist[1]
                          << ",actual_up=" << actualHist[2]
                          << ",pred_down=" << predHist[0]
                          << ",pred_neutral=" << predHist[1]
                          << ",pred_up=" << predHist[2]
                          << ",correct_down=" << correctHist[0]
                          << ",correct_neutral=" << correctHist[1]
                          << ",correct_up=" << correctHist[2]
                          << ",correct_total=" << phase3CorrectTotal
                          << ",accuracy=" << (phase3ValidRows > 0
                                  ? static_cast<double>(phase3CorrectTotal) / static_cast<double>(phase3ValidRows)
                                  : 0.0)
                          << ",recall_down=" << phase3RecallDown
                          << ",recall_neutral=" << phase3RecallNeutral
                          << ",recall_up=" << phase3RecallUp
                          << ",macro_recall=" << phase3BalancedAccuracy
                          << ",balanced_accuracy=" << phase3BalancedAccuracy
                          << ",conf_down_down=" << confusionByActual[0][0]
                          << ",conf_down_neutral=" << confusionByActual[0][1]
                          << ",conf_down_up=" << confusionByActual[0][2]
                          << ",conf_neutral_down=" << confusionByActual[1][0]
                          << ",conf_neutral_neutral=" << confusionByActual[1][1]
                          << ",conf_neutral_up=" << confusionByActual[1][2]
                          << ",conf_up_down=" << confusionByActual[2][0]
                          << ",conf_up_neutral=" << confusionByActual[2][1]
                          << ",conf_up_up=" << confusionByActual[2][2]
                          << ",pred_total=" << phase3PredTotal
                          << std::endl;
            }
            
            if (phase3HeadDiagEnabled)
            {
                for (size_t cls = 0; cls < direction_output_size; ++cls)
                {
                    PrintPhase3Stats("DIAG_HEAD_LOGITS_", phase3HeadDiagIdx, Phase3ClassName(cls),
                                     Phase3StatsCols(head_logits_batch, cls, 1, -20.0, 20.0));
                    PrintPhase3Stats("DIAG_HEAD_PROBS_", phase3HeadDiagIdx, Phase3ClassName(cls), probStats[cls]);
                    PrintPhase3Stats("DIAG_GRAD_PRECLIP_", phase3HeadDiagIdx, Phase3ClassName(cls),
                                     Phase3StatsCols(d_logits_batch, cls, 1, -20.0, 20.0));
                }
                std::cout << "DIAG_HEAD_PRED_"
                          << ",call=" << phase3HeadDiagIdx
                          << ",B=" << B
                          << ",pred_down=" << predHist[0]
                          << ",pred_neutral=" << predHist[1]
                          << ",pred_up=" << predHist[2]
                          << ",actual_down=" << actualHist[0]
                          << ",actual_neutral=" << actualHist[1]
                          << ",actual_up=" << actualHist[2]
                          << std::endl;
                std::cout << "DIAG_LOSS_"
                          << ",call=" << phase3HeadDiagIdx
                          << ",B=" << B
                          << ",class_weight_down=" << kClassWeightDown
                          << ",class_weight_neutral=" << kClassWeightNeutral
                          << ",class_weight_up=" << kClassWeightUp
                          << ",weighted_sample_sum_running=" << phase3ClassWeightSum
                          << ",weighted_loss_down=" << weightedLossByClass[0]
                          << ",weighted_loss_neutral=" << weightedLossByClass[1]
                          << ",weighted_loss_up=" << weightedLossByClass[2]
                          << ",weighted_loss_total=" << (weightedLossByClass[0] + weightedLossByClass[1] + weightedLossByClass[2])
                          << std::endl;
                std::cout << "DIAG_CLASS_WEIGHT_"
                          << ",target=UpNeutralDownReturn"
                          << ",down=" << kClassWeightDown
                          << ",neutral=" << kClassWeightNeutral
                          << ",up=" << kClassWeightUp
                          << ",active_path=1"
                          << std::endl;
                const double phase3WeightedLossTotal =
                    weightedLossByClass[0] + weightedLossByClass[1] + weightedLossByClass[2];

                if (Phase3ShouldPrintProgressSample(phase3HeadDiagIdx))
                {
                    const auto logitsProgress = Phase3DirHeadLogitsCpu(h_batch, returnHeadDirWeight, returnHeadDirBias);
                    const auto probsProgress = Phase3SoftmaxProbsCpu(logitsProgress);

                    const auto headWStats = Phase3StatsWhole(returnHeadDirWeight, -20.0, 20.0);
                    const auto headBStats = Phase3StatsWhole(returnHeadDirBias, -20.0, 20.0);
                    const auto logitStats = Phase3StatsWhole(logitsProgress, -20.0, 20.0);
                    const auto probStats = Phase3StatsWhole(probsProgress, 1.0e-4, 1.0 - 1.0e-4);

                    std::cout << "DIAG_HEAD_PROGRESS_"
                              << ",update=" << phase3HeadDiagIdx
                              << ",head_weight_norm=" << FroNormEvalHost(returnHeadDirWeight)
                              << ",head_weight_absmax=" << headWStats.absmax
                              << ",head_bias_norm=" << FroNormEvalHost(returnHeadDirBias)
                              << ",head_bias_absmax=" << headBStats.absmax
                              << std::endl;

                    std::cout << "DIAG_LOGIT_PROGRESS_"
                              << ",update=" << phase3HeadDiagIdx
                              << ",logit_absmax=" << logitStats.absmax
                              << ",logit_std=" << Phase3StatsStddev(logitStats)
                              << std::endl;

                    std::cout << "DIAG_PROB_PROGRESS_"
                              << ",update=" << phase3HeadDiagIdx
                              << ",prob_absmax=" << probStats.absmax
                              << ",prob_std=" << Phase3StatsStddev(probStats)
                              << std::endl;

                    std::cout << "DIAG_PRED_PROGRESS_"
                              << ",update=" << phase3HeadDiagIdx
                              << ",B=" << B
                              << ",pred_down=" << predHist[0]
                              << ",pred_neutral=" << predHist[1]
                              << ",pred_up=" << predHist[2]
                              << ",actual_down=" << actualHist[0]
                              << ",actual_neutral=" << actualHist[1]
                              << ",actual_up=" << actualHist[2]
                              << ",weighted_loss_total=" << phase3WeightedLossTotal
                              << std::endl;
                }
            }

            // === BEGIN CLASS SEPARATION DIAGNOSTIC ===
            if (Phase3ShouldPrintProgressSample(phase3HeadDiagIdx))
            {
                for (size_t actualCls = 0; actualCls < direction_output_size; ++actualCls)
                {
                    const double denom = actualHist[actualCls]
                        ? static_cast<double>(actualHist[actualCls])
                        : 1.0;
                    std::cout << "DIAG_CLASS_SEPARATION_"
                              << ",update=" << phase3HeadDiagIdx
                              << ",actual_class=" << Phase3ClassName(actualCls)
                              << ",count=" << actualHist[actualCls]
                              << ",mean_logit_down=" << (logitSumByActual[actualCls][0] / denom)
                              << ",mean_logit_neutral=" << (logitSumByActual[actualCls][1] / denom)
                              << ",mean_logit_up=" << (logitSumByActual[actualCls][2] / denom)
                              << ",mean_prob_down=" << (probSumByActual[actualCls][0] / denom)
                              << ",mean_prob_neutral=" << (probSumByActual[actualCls][1] / denom)
                              << ",mean_prob_up=" << (probSumByActual[actualCls][2] / denom)
                              << std::endl;
                }
            }
            if (Phase3ShouldPrintProgressSample(phase3HeadDiagIdx))
            {
                std::array<double, direction_output_size> hMeanNormByActual {0.0, 0.0, 0.0};
                std::array<double, direction_output_size> hStdByActual {0.0, 0.0, 0.0};

                for (size_t actualCls = 0; actualCls < direction_output_size; ++actualCls)
                {
                    const double count = actualHist[actualCls]
                        ? static_cast<double>(actualHist[actualCls])
                        : 1.0;
                    double meanNormSq = 0.0;
                    double meanSq = 0.0;
                    for (size_t h = 0; h < hidden_size; ++h)
                    {
                        const double mean = hSumByActual[actualCls][h] / count;
                        meanNormSq += mean * mean;
                        meanSq += mean * mean;
                    }
                    const double elemMeanSq = meanSq / static_cast<double>(hidden_size);
                    const double elemSecondMoment = hSumSqByActual[actualCls] / (count * static_cast<double>(hidden_size));
                    hMeanNormByActual[actualCls] = std::sqrt(meanNormSq);
                    hStdByActual[actualCls] = std::sqrt(std::max(0.0, elemSecondMoment - elemMeanSq));
                }

                auto classMeanDistance = [&](size_t a, size_t b) -> double
                {
                    const double countA = actualHist[a] ? static_cast<double>(actualHist[a]) : 1.0;
                    const double countB = actualHist[b] ? static_cast<double>(actualHist[b]) : 1.0;
                    double distSq = 0.0;
                    for (size_t h = 0; h < hidden_size; ++h)
                    {
                        const double ma = hSumByActual[a][h] / countA;
                        const double mb = hSumByActual[b][h] / countB;
                        const double d = ma - mb;
                        distSq += d * d;
                    }
                    return std::sqrt(distSq);
                };

                // === BEGIN LEAVE-ONE-OUT CENTROID-ACCURACY DIAGNOSTIC ===
                size_t centroidPredHist[direction_output_size] = {0, 0, 0};
                size_t centroidActualHist[direction_output_size] = {0, 0, 0};
                size_t centroidCorrect = 0;
                size_t centroidTotal = 0;
                size_t centroidSkippedSingletonActual = 0;
                size_t centroidSkippedInvalidActual = 0;
                size_t centroidSkippedNonfiniteRow = 0;
                long double centroidRowNormSum = 0.0L;
                long double centroidActualDistSum = 0.0L;
                long double centroidNearestOtherDistSum = 0.0L;
                long double centroidMarginSum = 0.0L;
                const double centroidLargeDistance = std::numeric_limits<double>::max();
                double centroidActualDistMin = centroidLargeDistance;
                double centroidActualDistMax = 0.0;
                double centroidNearestOtherDistMin = centroidLargeDistance;
                double centroidNearestOtherDistMax = 0.0;
                double centroidMarginMin = centroidLargeDistance;
                double centroidMarginMax = -centroidLargeDistance;

                if (hHostForSeparation.cols == hidden_size)
                {
                    for (size_t row = 0; row < B && row < hHostForSeparation.rows; ++row)
                    {
                        const int actual = wb.classTargets[row];
                        if (actual < 0 || actual >= static_cast<int>(direction_output_size))
                        {
                            ++centroidSkippedInvalidActual;
                            continue;
                        }
                        const size_t actualClass = static_cast<size_t>(actual);
                        if (actualHist[actualClass] <= 1)
                        {
                            ++centroidSkippedSingletonActual;
                            continue;
                        }

                        bool rowFinite = true;
                        long double rowNormSq = 0.0L;
                        for (size_t h = 0; h < hidden_size; ++h)
                        {
                            const double hv = static_cast<double>(
                                hHostForSeparation.data[row * hHostForSeparation.cols + h]);
                            if (!std::isfinite(hv))
                            {
                                rowFinite = false;
                                break;
                            }
                            rowNormSq += static_cast<long double>(hv) * static_cast<long double>(hv);
                        }
                        if (!rowFinite)
                        {
                            ++centroidSkippedNonfiniteRow;
                            continue;
                        }

                        double bestDistSq = centroidLargeDistance;
                        size_t bestClass = 1;
                        double actualDistSq = centroidLargeDistance;
                        double nearestOtherDistSq = centroidLargeDistance;
                        for (size_t candidate = 0; candidate < direction_output_size; ++candidate)
                        {
                            const bool excludeRow = (candidate == actualClass);
                            const size_t candidateCount = actualHist[candidate] - (excludeRow ? 1 : 0);
                            if (candidateCount == 0)
                                continue;

                            const double count = static_cast<double>(candidateCount);
                            double distSq = 0.0;
                            for (size_t h = 0; h < hidden_size; ++h)
                            {
                                const double hv = static_cast<double>(
                                    hHostForSeparation.data[row * hHostForSeparation.cols + h]);
                                const double centroidSum = hSumByActual[candidate][h] - (excludeRow ? hv : 0.0);
                                const double mean = centroidSum / count;
                                const double d = hv - mean;
                                distSq += d * d;
                            }

                            if (candidate == actualClass)
                                actualDistSq = distSq;
                            else if (distSq < nearestOtherDistSq)
                                nearestOtherDistSq = distSq;

                            if (distSq < bestDistSq)
                            {
                                bestDistSq = distSq;
                                bestClass = candidate;
                            }
                        }

                        if (bestDistSq == centroidLargeDistance || actualDistSq == centroidLargeDistance)
                            continue;

                        const double actualDist = std::sqrt(actualDistSq);
                        const double nearestOtherDist = (nearestOtherDistSq != centroidLargeDistance)
                            ? std::sqrt(nearestOtherDistSq)
                            : 0.0;
                        const double margin = nearestOtherDist - actualDist;

                        ++centroidTotal;
                        ++centroidPredHist[bestClass];
                        ++centroidActualHist[actualClass];
                        if (bestClass == actualClass)
                            ++centroidCorrect;
                        centroidRowNormSum += std::sqrt(static_cast<double>(rowNormSq));
                        centroidActualDistSum += actualDist;
                        centroidNearestOtherDistSum += nearestOtherDist;
                        centroidMarginSum += margin;
                        centroidActualDistMin = std::min(centroidActualDistMin, actualDist);
                        centroidActualDistMax = std::max(centroidActualDistMax, actualDist);
                        centroidNearestOtherDistMin = std::min(centroidNearestOtherDistMin, nearestOtherDist);
                        centroidNearestOtherDistMax = std::max(centroidNearestOtherDistMax, nearestOtherDist);
                        centroidMarginMin = std::min(centroidMarginMin, margin);
                        centroidMarginMax = std::max(centroidMarginMax, margin);
                    }
                }
                if (centroidTotal == 0)
                {
                    centroidActualDistMin = 0.0;
                    centroidNearestOtherDistMin = 0.0;
                    centroidMarginMin = 0.0;
                    centroidMarginMax = 0.0;
                }

                std::cout << "DIAG_H_SEPARATION_"
                          << ",update=" << phase3HeadDiagIdx
                          << ",hidden_size=" << hidden_size
                          << ",count_down=" << actualHist[0]
                          << ",count_neutral=" << actualHist[1]
                          << ",count_up=" << actualHist[2]
                          << ",h_mean_norm_down=" << hMeanNormByActual[0]
                          << ",h_mean_norm_neutral=" << hMeanNormByActual[1]
                          << ",h_mean_norm_up=" << hMeanNormByActual[2]
                          << ",h_std_down=" << hStdByActual[0]
                          << ",h_std_neutral=" << hStdByActual[1]
                          << ",h_std_up=" << hStdByActual[2]
                          << ",mean_dist_down_neutral=" << classMeanDistance(0, 1)
                          << ",mean_dist_down_up=" << classMeanDistance(0, 2)
                          << ",mean_dist_neutral_up=" << classMeanDistance(1, 2)
                          << std::endl;

                std::cout << "DIAG_H_CENTROID_ACC_"
                          << ",update=" << phase3HeadDiagIdx
                          << ",mode=leave_one_out"
                          << ",hidden_size=" << hidden_size
                          << ",total=" << centroidTotal
                          << ",correct=" << centroidCorrect
                          << ",accuracy=" << (centroidTotal ? static_cast<double>(centroidCorrect) / static_cast<double>(centroidTotal) : 0.0)
                          << ",pred_down=" << centroidPredHist[0]
                          << ",pred_neutral=" << centroidPredHist[1]
                          << ",pred_up=" << centroidPredHist[2]
                          << ",actual_down=" << centroidActualHist[0]
                          << ",actual_neutral=" << centroidActualHist[1]
                          << ",actual_up=" << centroidActualHist[2]
                          << ",skipped_singleton_actual=" << centroidSkippedSingletonActual
                          << ",skipped_invalid_actual=" << centroidSkippedInvalidActual
                          << ",skipped_nonfinite_row=" << centroidSkippedNonfiniteRow
                          << ",row_norm_mean=" << (centroidTotal ? static_cast<double>(centroidRowNormSum / static_cast<long double>(centroidTotal)) : 0.0)
                          << ",actual_centroid_dist_mean=" << (centroidTotal ? static_cast<double>(centroidActualDistSum / static_cast<long double>(centroidTotal)) : 0.0)
                          << ",actual_centroid_dist_min=" << centroidActualDistMin
                          << ",actual_centroid_dist_max=" << centroidActualDistMax
                          << ",nearest_other_centroid_dist_mean=" << (centroidTotal ? static_cast<double>(centroidNearestOtherDistSum / static_cast<long double>(centroidTotal)) : 0.0)
                          << ",nearest_other_centroid_dist_min=" << centroidNearestOtherDistMin
                          << ",nearest_other_centroid_dist_max=" << centroidNearestOtherDistMax
                          << ",nearest_other_minus_actual_dist_mean=" << (centroidTotal ? static_cast<double>(centroidMarginSum / static_cast<long double>(centroidTotal)) : 0.0)
                          << ",nearest_other_minus_actual_dist_min=" << centroidMarginMin
                          << ",nearest_other_minus_actual_dist_max=" << centroidMarginMax
                          << std::endl;

                s_phase3LastHGeometryHostBeforeUpdate = hHostForSeparation;
                s_phase3LastHGeometryActualHistBeforeUpdate[0] = actualHist[0];
                s_phase3LastHGeometryActualHistBeforeUpdate[1] = actualHist[1];
                s_phase3LastHGeometryActualHistBeforeUpdate[2] = actualHist[2];
                s_phase3LastHGeometryValidBeforeUpdate = true;
                s_phase3HiddenReplayCapture.valid = true;
                s_phase3HiddenReplayCapture.batchBase = batchBase;
                s_phase3HiddenReplayCapture.rows = B;
                s_phase3HiddenReplayCapture.hiddenCols = hidden_size;
                s_phase3HiddenReplayCapture.effectiveMiniBatchWindows = effectiveMiniBatchWindows;
                s_phase3HiddenReplayCapture.windowCountAtCapture = windowCount;
                s_phase3HiddenReplayCapture.startIndices.clear();
                s_phase3HiddenReplayCapture.startIndices.reserve(B);

                for (size_t replayRow = 0;
                     replayRow < B && (batchBase + replayRow) < allStarts.size();
                     ++replayRow)
                {
                    s_phase3HiddenReplayCapture.startIndices.push_back(allStarts[batchBase + replayRow]);
                }
                s_phase3HiddenReplayCapture.actualClasses.clear();
                s_phase3HiddenReplayCapture.actualClasses.reserve(B);
                for (size_t replayRow = 0; replayRow < B; ++replayRow)
                {
                    s_phase3HiddenReplayCapture.actualClasses.push_back(wb.classTargets[replayRow]);
                }
                
                s_phase3HiddenReplayCapture.replayTimeSteps = cache.size();
                s_phase3HiddenReplayCapture.replayInputCols = static_cast<size_t>(n_in);
                s_phase3HiddenReplayCapture.replayInputs.clear();
                s_phase3HiddenReplayCapture.replayInputs.reserve(
                    cache.size() * B * static_cast<size_t>(n_in));

                for (const auto& replayStepCache : cache)
                {
                    auto replayLowX = MetaNN::LowerAccess(replayStepCache.x);
                    const float* replayX = replayLowX.RawMemory();

                    const size_t replayRows = replayStepCache.x.Shape()[0];
                    const size_t replayCols = replayStepCache.x.Shape()[1];

                    if (replayRows == B && replayCols == static_cast<size_t>(n_in))
                    {
                        s_phase3HiddenReplayCapture.replayInputs.insert(
                            s_phase3HiddenReplayCapture.replayInputs.end(),
                            replayX,
                            replayX + replayRows * replayCols);
                    }
                }
                s_phase3HiddenReplayCapture.hBefore = hHostForSeparation;
                s_phase3HiddenReplayCapture.actualHist[0] = actualHist[0];
                s_phase3HiddenReplayCapture.actualHist[1] = actualHist[1];
                s_phase3HiddenReplayCapture.actualHist[2] = actualHist[2];
                // === HEAD PROJECTION DIAGNOSTIC ===
                std::cout << "DIAG_HEAD_ALIGNMENT_"
                          << ",update=" << phase3HeadDiagIdx
                          << ",count_down=" << actualHist[0]
                          << ",count_neutral=" << actualHist[1]
                          << ",count_up=" << actualHist[2]
                          << ",mean_proj_down="
                          << (hiddenHeadProjectionMean[0] /
                              std::max<size_t>(actualHist[0], 1))
                          << ",mean_proj_neutral="
                          << (hiddenHeadProjectionMean[1] /
                              std::max<size_t>(actualHist[1], 1))
                          << ",mean_proj_up="
                          << (hiddenHeadProjectionMean[2] /
                              std::max<size_t>(actualHist[2], 1))
                          << ",mean_abs_proj_down="
                          << (hiddenHeadProjectionAbsMean[0] /
                              std::max<size_t>(actualHist[0], 1))
                          << ",mean_abs_proj_neutral="
                          << (hiddenHeadProjectionAbsMean[1] /
                              std::max<size_t>(actualHist[1], 1))
                          << ",mean_abs_proj_up="
                          << (hiddenHeadProjectionAbsMean[2] /
                              std::max<size_t>(actualHist[2], 1))
                          << std::endl;
            }
            windowCount += B;
            windowsInBatch += B;
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
#if LSTM_SHAPE_DIAG
                static bool s_printed_head_reg_matmul_shapes = false;
                if (!s_printed_head_reg_matmul_shapes)
                {
                    std::cout << "DIAG_HEAD_REG_MATMUL_SHAPE"
                              << ",m=" << B
                              << ",k=" << hidden_size
                              << ",n=1"
                              << std::endl;
                    s_printed_head_reg_matmul_shapes = true;
                }
#endif
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
#if LSTM_SHAPE_DIAG
                static bool s_printed_head_reg_matmul_shapes = false;
                if (!s_printed_head_reg_matmul_shapes)
                {
                    std::cout << "DIAG_HEAD_REG_MATMUL_SHAPE"
                              << ",m=" << B
                              << ",k=" << hidden_size
                              << ",n=1"
                              << std::endl;
                    s_printed_head_reg_matmul_shapes = true;
                }
#endif
            }
#endif
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
                y_sum += static_cast<double>(target);
                y_sumsq += static_cast<double>(target) * static_cast<double>(target);
                y_min = std::min(y_min, target);
                y_max = std::max(y_max, target);
                ++y_count;
                if (yhat_samples.size() < 10) yhat_samples.push_back(y_hat);
            }
            windowCount += B;
            windowsInBatch += B;
        }

        if (targetType == TargetType::UpNeutralDownReturn)
        {
            AccumulateHeadGradsBatch3Class(d_headDirW_accum_f, d_headDirB_accum_f, h_batch, d_logits_batch);

            d_h_batch = BuildHeadDhBatch3Class(d_logits_batch, returnHeadDirWeight, LSTM_CORE_GRAD_SCALE);
            static size_t s_phase3DhFromHeadDiagCount = 0;
            const size_t phase3DhFromHeadDiagIdx = s_phase3DhFromHeadDiagCount++;
            const bool phase3DhDiagEnabled = (phase3DhFromHeadDiagIdx < LSTM_PHASE3_HEAD_DIAG_LIMIT);
            if (phase3DhDiagEnabled)
                PrintPhase3NormStats("DIAG_DH_FROM_HEAD_", phase3DhFromHeadDiagIdx, "d_h_batch", d_h_batch);
            if (phase3DhDiagEnabled)
            {
                auto lowDhHead = MetaNN::LowerAccess(d_h_batch);
                const auto* dhPtr = lowDhHead.RawMemory();
                const size_t dhRows = d_h_batch.Shape()[0];
                const size_t dhCols = d_h_batch.Shape()[1];

                std::array<size_t, direction_output_size> dhActualHist {0, 0, 0};
                std::array<std::vector<double>, direction_output_size> dhMeanByActual;
                for (size_t cls = 0; cls < direction_output_size; ++cls)
                    dhMeanByActual[cls].assign(dhCols, 0.0);
                std::vector<double> dhGlobalMean(dhCols, 0.0);

                double dhTotalSq = 0.0;
                for (size_t row = 0; row < dhRows; ++row)
                {
                    const int actual = wb.classTargets[row];
                    if (actual >= 0 && actual < static_cast<int>(direction_output_size))
                    {
                        ++dhActualHist[static_cast<size_t>(actual)];
                        for (size_t h = 0; h < dhCols; ++h)
                        {
                            const double v = static_cast<double>(dhPtr[row * dhCols + h]);
                            dhMeanByActual[static_cast<size_t>(actual)][h] += v;
                            dhGlobalMean[h] += v;
                            dhTotalSq += v * v;
                        }
                    }
                }

                for (size_t h = 0; h < dhCols; ++h)
                    dhGlobalMean[h] /= static_cast<double>(std::max<size_t>(dhRows, 1));
                for (size_t cls = 0; cls < direction_output_size; ++cls)
                {
                    const double denom = static_cast<double>(std::max<size_t>(dhActualHist[cls], 1));
                    for (size_t h = 0; h < dhCols; ++h)
                        dhMeanByActual[cls][h] /= denom;
                }

                auto phase3VecNorm = [](const std::vector<double>& v) -> double
                {
                    long double ss = 0.0L;
                    for (double x : v)
                        ss += static_cast<long double>(x) * static_cast<long double>(x);
                    return std::sqrt(static_cast<double>(ss));
                };

                auto phase3VecDistance = [](const std::vector<double>& a, const std::vector<double>& b) -> double
                {
                    const size_t n = std::min(a.size(), b.size());
                    long double ss = 0.0L;
                    for (size_t i = 0; i < n; ++i)
                    {
                        const long double d = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
                        ss += d * d;
                    }
                    return std::sqrt(static_cast<double>(ss));
                };

                auto phase3Cosine = [](const std::vector<double>& a, const std::vector<double>& b) -> double
                {
                    const size_t n = std::min(a.size(), b.size());
                    long double dot = 0.0L;
                    long double aa = 0.0L;
                    long double bb = 0.0L;
                    for (size_t i = 0; i < n; ++i)
                    {
                        const long double av = static_cast<long double>(a[i]);
                        const long double bv = static_cast<long double>(b[i]);
                        dot += av * bv;
                        aa += av * av;
                        bb += bv * bv;
                    }
                    const double denom = std::sqrt(static_cast<double>(aa)) * std::sqrt(static_cast<double>(bb));
                    return denom > 1.0e-12 ? static_cast<double>(dot) / denom : 0.0;
                };

                const double dhTotalNorm = std::sqrt(dhTotalSq);
                const double dhGlobalMeanNorm = phase3VecNorm(dhGlobalMean);
                const double dhDownMeanNorm = phase3VecNorm(dhMeanByActual[0]);
                const double dhNeutralMeanNorm = phase3VecNorm(dhMeanByActual[1]);
                const double dhUpMeanNorm = phase3VecNorm(dhMeanByActual[2]);
                const double dhDownNeutralDist = phase3VecDistance(dhMeanByActual[0], dhMeanByActual[1]);
                const double dhDownUpDist = phase3VecDistance(dhMeanByActual[0], dhMeanByActual[2]);
                const double dhNeutralUpDist = phase3VecDistance(dhMeanByActual[1], dhMeanByActual[2]);
                const double dhMeanClassDistAvg = (dhDownNeutralDist + dhDownUpDist + dhNeutralUpDist) / 3.0;
                const double dhCommonModeRatio = dhTotalNorm > 1.0e-12
                    ? (dhGlobalMeanNorm * std::sqrt(static_cast<double>(std::max<size_t>(dhRows, 1))) / dhTotalNorm)
                    : 0.0;

                std::cout << "DIAG_DH_CLASS_DIRECTION_"
                          << ",call=" << phase3DhFromHeadDiagIdx
                          << ",B=" << dhRows
                          << ",hidden_size=" << dhCols
                          << ",actual_down=" << dhActualHist[0]
                          << ",actual_neutral=" << dhActualHist[1]
                          << ",actual_up=" << dhActualHist[2]
                          << ",dh_total_norm=" << dhTotalNorm
                          << ",dh_global_mean_norm=" << dhGlobalMeanNorm
                          << ",dh_common_mode_ratio=" << dhCommonModeRatio
                          << ",mean_norm_down=" << dhDownMeanNorm
                          << ",mean_norm_neutral=" << dhNeutralMeanNorm
                          << ",mean_norm_up=" << dhUpMeanNorm
                          << ",mean_dist_down_neutral=" << dhDownNeutralDist
                          << ",mean_dist_down_up=" << dhDownUpDist
                          << ",mean_dist_neutral_up=" << dhNeutralUpDist
                          << ",mean_class_dist_avg=" << dhMeanClassDistAvg
                          << ",class_dist_to_global_mean_ratio=" << (dhGlobalMeanNorm > 1.0e-12 ? dhMeanClassDistAvg / dhGlobalMeanNorm : 0.0)
                          << ",cos_down_neutral=" << phase3Cosine(dhMeanByActual[0], dhMeanByActual[1])
                          << ",cos_down_up=" << phase3Cosine(dhMeanByActual[0], dhMeanByActual[2])
                          << ",cos_neutral_up=" << phase3Cosine(dhMeanByActual[1], dhMeanByActual[2])
                          << std::endl;
            }
            if (d_c_batch.Shape()[0] != B || d_c_batch.Shape()[1] != hidden_size)
                d_c_batch = EAMatrix(B, hidden_size);
            zeroFill(d_c_batch);

            zeroGateAccumulators(G_bin, param.Shape()[0], hidden_size);

            auto gb = hoistGateBlocks(ww.W_h, hidden_size);
            if (phase3DhDiagEnabled)
            {
                const double whINorm = FroNormEvalHost(gb.W_hi);
                const double whFNorm = FroNormEvalHost(gb.W_hf);
                const double whGNorm = FroNormEvalHost(gb.W_hg);
                const double whONorm = FroNormEvalHost(gb.W_ho);

                auto phase3MaxAbsHost = [](const EAMatrix& m) -> double
                {
                    auto low = MetaNN::LowerAccess(m);
                    const float* ptr = low.RawMemory();
                    const size_t count = m.Shape()[0] * m.Shape()[1];
                    double maxAbs = 0.0;
                    for (size_t idx = 0; idx < count; ++idx)
                    {
                        const double v = static_cast<double>(ptr[idx]);
                        if (std::isfinite(v))
                            maxAbs = std::max(maxAbs, std::abs(v));
                    }
                    return maxAbs;
                };

                const double whIMaxAbs = phase3MaxAbsHost(gb.W_hi);
                const double whFMaxAbs = phase3MaxAbsHost(gb.W_hf);
                const double whGMaxAbs = phase3MaxAbsHost(gb.W_hg);
                const double whOMaxAbs = phase3MaxAbsHost(gb.W_ho);

                const double whIAvgColNorm = hidden_size > 0 ? whINorm / std::sqrt(static_cast<double>(hidden_size)) : 0.0;
                const double whFAvgColNorm = hidden_size > 0 ? whFNorm / std::sqrt(static_cast<double>(hidden_size)) : 0.0;
                const double whGAvgColNorm = hidden_size > 0 ? whGNorm / std::sqrt(static_cast<double>(hidden_size)) : 0.0;
                const double whOAvgColNorm = hidden_size > 0 ? whONorm / std::sqrt(static_cast<double>(hidden_size)) : 0.0;

                std::cout << "DIAG_BPTT_RECURRENT_BLOCK_SCALE_"
                          << ",call=" << phase3DhFromHeadDiagIdx
                          << ",hidden_size=" << hidden_size
                          << ",W_hi_fro_norm=" << whINorm
                          << ",W_hf_fro_norm=" << whFNorm
                          << ",W_hg_fro_norm=" << whGNorm
                          << ",W_ho_fro_norm=" << whONorm
                          << ",W_hi_avg_col_norm=" << whIAvgColNorm
                          << ",W_hf_avg_col_norm=" << whFAvgColNorm
                          << ",W_hg_avg_col_norm=" << whGAvgColNorm
                          << ",W_ho_avg_col_norm=" << whOAvgColNorm
                          << ",W_hi_max_abs=" << whIMaxAbs
                          << ",W_hf_max_abs=" << whFMaxAbs
                          << ",W_hg_max_abs=" << whGMaxAbs
                          << ",W_ho_max_abs=" << whOMaxAbs
                          << ",g_to_i_fro_ratio=" << (whINorm > 1.0e-12 ? whGNorm / whINorm : 0.0)
                          << ",g_to_f_fro_ratio=" << (whFNorm > 1.0e-12 ? whGNorm / whFNorm : 0.0)
                          << ",g_to_o_fro_ratio=" << (whONorm > 1.0e-12 ? whGNorm / whONorm : 0.0)
                          << std::endl;

                PrintPhase3NormStats("DIAG_BPTT_DH_IN_", phase3DhFromHeadDiagIdx, "d_h_before_bptt", d_h_batch);
                PrintPhase3NormStats("DIAG_BPTT_DC_IN_", phase3DhFromHeadDiagIdx, "d_c_before_bptt", d_c_batch);
            }

            for (int tstep = static_cast<int>(cache.size()) - 1; tstep >= 0; --tstep)
            {
                if (phase3DhDiagEnabled && tstep == static_cast<int>(cache.size()) - 1)
                {
                    PrintPhase3NormStats("DIAG_BPTT_LAST_STEP_IN_", phase3DhFromHeadDiagIdx, "d_h_before_last_step", d_h_batch);
                    PrintPhase3NormStats("DIAG_BPTT_LAST_STEP_IN_", phase3DhFromHeadDiagIdx, "d_c_before_last_step", d_c_batch);
                }

                const bool phase3IsLastBackwardStep =
                    (phase3DhDiagEnabled && tstep == static_cast<int>(cache.size()) - 1);

                double phase3GateWiBefore = 0.0;
                double phase3GateWfBefore = 0.0;
                double phase3GateWgBefore = 0.0;
                double phase3GateWoBefore = 0.0;
                double phase3GateBiBefore = 0.0;
                double phase3GateBfBefore = 0.0;
                double phase3GateBgBefore = 0.0;
                double phase3GateBoBefore = 0.0;
                if (phase3IsLastBackwardStep)
                {
                    phase3GateWiBefore = FroNormEvalHost(G_bin.dW_i);
                    phase3GateWfBefore = FroNormEvalHost(G_bin.dW_f);
                    phase3GateWgBefore = FroNormEvalHost(G_bin.dW_g);
                    phase3GateWoBefore = FroNormEvalHost(G_bin.dW_o);
                    phase3GateBiBefore = FroNormEvalHost(G_bin.db_i);
                    phase3GateBfBefore = FroNormEvalHost(G_bin.db_f);
                    phase3GateBgBefore = FroNormEvalHost(G_bin.db_g);
                    phase3GateBoBefore = FroNormEvalHost(G_bin.db_o);
                }

                // --- BEGIN PATCH: DIAG BPTT STEP ATTENUATION ---
                const double phase3StepDhInNorm = phase3DhDiagEnabled ? FroNormEvalHost(d_h_batch) : 0.0;
                const double phase3StepDcInNorm = phase3DhDiagEnabled ? FroNormEvalHost(d_c_batch) : 0.0;
                // --- END PATCH ---

                if (phase3DhDiagEnabled)
                {
                    static size_t s_phase3CellCarryDiagCount = 0;
                    if (s_phase3CellCarryDiagCount < LSTM_PHASE3_HEAD_DIAG_LIMIT)
                    {
                        const auto& phase3CellCarrySc = cache[static_cast<size_t>(tstep)];

                        auto lowF = MetaNN::LowerAccess(phase3CellCarrySc.f);
                        auto lowO = MetaNN::LowerAccess(phase3CellCarrySc.o);
                        auto lowC = MetaNN::LowerAccess(phase3CellCarrySc.c);
                        auto lowZf = MetaNN::LowerAccess(phase3CellCarrySc.z_f);
                        auto lowDh = MetaNN::LowerAccess(d_h_batch);
                        auto lowDc = MetaNN::LowerAccess(d_c_batch);

                        const float* fptr = lowF.RawMemory();
                        const float* optr = lowO.RawMemory();
                        const float* cptr = lowC.RawMemory();
                        const float* zfptr = lowZf.RawMemory();
                        const float* dhptr = lowDh.RawMemory();
                        const float* dcptr = lowDc.RawMemory();

                        const size_t rows = d_h_batch.Shape()[0];
                        const size_t cols = d_h_batch.Shape()[1];
                        const size_t count = rows * cols;

                        double fSum = 0.0;
                        double fSq = 0.0;
                        double fMin = count ? static_cast<double>(fptr[0]) : 0.0;
                        double fMax = fMin;
                        double zfSum = 0.0;
                        double zfMin = count ? static_cast<double>(zfptr[0]) : 0.0;
                        double zfMax = zfMin;
                        double fBiasPlus05Sum = 0.0, fBiasPlus10Sum = 0.0;
                        double fBiasPlus05Sq = 0.0, fBiasPlus10Sq = 0.0;
                        long double dcPrevFromDcSq = 0.0L;
                        long double dcPrevFromDhSq = 0.0L;
                        long double dcPrevTotalSq = 0.0L;

                        for (size_t idx = 0; idx < count; ++idx)
                        {
                            const double f = static_cast<double>(fptr[idx]);
                            const double o = static_cast<double>(optr[idx]);
                            const double c = static_cast<double>(cptr[idx]);
                            const double zf = static_cast<double>(zfptr[idx]);
                            const double dh = static_cast<double>(dhptr[idx]);
                            const double dc = static_cast<double>(dcptr[idx]);

                            const double tanhC = std::tanh(c);
                            const double tanhDeriv = 1.0 - tanhC * tanhC;
                            const double dcPrevFromDc = dc * f;
                            const double dcPrevFromDh = dh * o * tanhDeriv * f;
                            const double dcPrevTotal = dcPrevFromDc + dcPrevFromDh;

                            fSum += f;
                            fSq += f * f;
                            fMin = std::min(fMin, f);
                            fMax = std::max(fMax, f);
                            zfSum += zf;
                            zfMin = std::min(zfMin, zf);
                            zfMax = std::max(zfMax, zf);
                            const double fBiasPlus05 = 1.0 / (1.0 + std::exp(-(zf + 0.5)));
                            const double fBiasPlus10 = 1.0 / (1.0 + std::exp(-(zf + 1.0)));

                            fBiasPlus05Sum += fBiasPlus05;
                            fBiasPlus10Sum += fBiasPlus10;
                            fBiasPlus05Sq += fBiasPlus05 * fBiasPlus05;
                            fBiasPlus10Sq += fBiasPlus10 * fBiasPlus10;
                            dcPrevFromDcSq += static_cast<long double>(dcPrevFromDc) * static_cast<long double>(dcPrevFromDc);
                            dcPrevFromDhSq += static_cast<long double>(dcPrevFromDh) * static_cast<long double>(dcPrevFromDh);
                            dcPrevTotalSq += static_cast<long double>(dcPrevTotal) * static_cast<long double>(dcPrevTotal);
                        }

                        const int phase3StepsFromLast =
                            static_cast<int>(cache.size()) - 1 - tstep;
                        const double phase3FMean =
                            count ? fSum / static_cast<double>(count) : 0.0;
                        const double phase3CarryPowerEst =
                            std::pow(phase3FMean, static_cast<double>(phase3StepsFromLast));
                        const double phase3ZfMean =
                            count ? zfSum / static_cast<double>(count) : 0.0;
                        const double phase3SigmoidZfMean =
                            1.0 / (1.0 + std::exp(-phase3ZfMean));
                        const double phase3FBiasPlus05Mean = count ? fBiasPlus05Sum / double(count) : 0.0;
                        const double phase3FBiasPlus10Mean = count ? fBiasPlus10Sum / double(count) : 0.0;

                        const double phase3CarryPowerEstBiasPlus05 =
                            std::pow(phase3FBiasPlus05Mean, double(phase3StepsFromLast));
                        const double phase3CarryPowerEstBiasPlus10 =
                            std::pow(phase3FBiasPlus10Mean, double(phase3StepsFromLast));
                        std::cout << "DIAG_BPTT_FORGET_OPERATING_POINT_"
                                  << ",call=" << s_phase3CellCarryDiagCount
                                  << ",source_call=" << phase3DhFromHeadDiagIdx
                                  << ",tstep=" << tstep
                                  << ",steps_from_last=" << phase3StepsFromLast
                                  << ",rows=" << rows
                                  << ",hidden_size=" << cols
                                  << ",z_f_mean=" << phase3ZfMean
                                  << ",z_f_min=" << zfMin
                                  << ",z_f_max=" << zfMax
                                  << ",sigmoid_z_f_mean=" << phase3SigmoidZfMean
                                  << ",f_mean=" << phase3FMean
                                  << ",f_mean_minus_sigmoid_z_f_mean=" << (phase3FMean - phase3SigmoidZfMean)
                        << ",f_bias_plus_0p5_mean=" << phase3FBiasPlus05Mean
                        << ",carry_power_est_bias_plus_0p5=" << phase3CarryPowerEstBiasPlus05
                        << ",f_bias_plus_1p0_mean=" << phase3FBiasPlus10Mean
                        << ",carry_power_est_bias_plus_1p0=" << phase3CarryPowerEstBiasPlus10
                        << std::endl;

                        std::cout << "DIAG_BPTT_CELL_CARRY_"
                                  << ",call=" << s_phase3CellCarryDiagCount
                                  << ",source_call=" << phase3DhFromHeadDiagIdx
                                  << ",tstep=" << tstep
                                  << ",steps_from_last=" << phase3StepsFromLast
                                  << ",carry_power_est=" << phase3CarryPowerEst
                                  << ",rows=" << rows
                                  << ",hidden_size=" << cols
                                  << ",f_mean=" << phase3FMean
                                  << ",f_min=" << fMin
                                  << ",f_max=" << fMax
                                  << ",f_rms=" << (count ? std::sqrt(fSq / static_cast<double>(count)) : 0.0)
                                  << ",dc_prev_from_dc_norm=" << std::sqrt(static_cast<double>(dcPrevFromDcSq))
                                  << ",dc_prev_from_dh_norm=" << std::sqrt(static_cast<double>(dcPrevFromDhSq))
                                  << ",dc_prev_total_norm=" << std::sqrt(static_cast<double>(dcPrevTotalSq))
                                  << std::endl;

                        ++s_phase3CellCarryDiagCount;
                    }
                }

                backwardStepBatch(cache[static_cast<size_t>(tstep)], gb, d_h_batch, d_c_batch, G_bin);

                // --- BEGIN PATCH: DIAG BPTT STEP ATTENUATION ---
                if (phase3DhDiagEnabled)
                {
                    const double phase3StepDhOutNorm = FroNormEvalHost(d_h_batch);
                    const double phase3StepDcOutNorm = FroNormEvalHost(d_c_batch);

                    std::cout << "DIAG_BPTT_STEP_ATTENUATION_"
                              << ",call=" << phase3DhFromHeadDiagIdx
                              << ",tstep=" << tstep
                              << ",dh_in_norm=" << phase3StepDhInNorm
                              << ",dc_in_norm=" << phase3StepDcInNorm
                              << ",dh_out_norm=" << phase3StepDhOutNorm
                              << ",dc_out_norm=" << phase3StepDcOutNorm
                              << ",dh_out_to_in_ratio=" << (phase3StepDhInNorm > 1.0e-12 ? phase3StepDhOutNorm / phase3StepDhInNorm : 0.0)
                              << ",dc_out_to_in_ratio=" << (phase3StepDcInNorm > 1.0e-12 ? phase3StepDcOutNorm / phase3StepDcInNorm : 0.0)
                              << std::endl;
                }
                // --- END PATCH ---

                if (phase3IsLastBackwardStep)
                {
                    PrintPhase3NormStats("DIAG_BPTT_LAST_STEP_OUT_", phase3DhFromHeadDiagIdx, "d_h_after_last_step", d_h_batch);
                    PrintPhase3NormStats("DIAG_BPTT_LAST_STEP_OUT_", phase3DhFromHeadDiagIdx, "d_c_after_last_step", d_c_batch);

                    const double phase3GateWiAfter = FroNormEvalHost(G_bin.dW_i);
                    const double phase3GateWfAfter = FroNormEvalHost(G_bin.dW_f);
                    const double phase3GateWgAfter = FroNormEvalHost(G_bin.dW_g);
                    const double phase3GateWoAfter = FroNormEvalHost(G_bin.dW_o);
                    const double phase3GateBiAfter = FroNormEvalHost(G_bin.db_i);
                    const double phase3GateBfAfter = FroNormEvalHost(G_bin.db_f);
                    const double phase3GateBgAfter = FroNormEvalHost(G_bin.db_g);
                    const double phase3GateBoAfter = FroNormEvalHost(G_bin.db_o);

                    std::cout << "DIAG_BPTT_LAST_STEP_GATE_ACCUM_"
                              << ",call=" << phase3DhFromHeadDiagIdx
                              << ",tstep=" << tstep
                              << ",dW_i_delta_norm=" << std::max(0.0, phase3GateWiAfter - phase3GateWiBefore)
                              << ",dW_f_delta_norm=" << std::max(0.0, phase3GateWfAfter - phase3GateWfBefore)
                              << ",dW_g_delta_norm=" << std::max(0.0, phase3GateWgAfter - phase3GateWgBefore)
                              << ",dW_o_delta_norm=" << std::max(0.0, phase3GateWoAfter - phase3GateWoBefore)
                              << ",db_i_delta_norm=" << std::max(0.0, phase3GateBiAfter - phase3GateBiBefore)
                              << ",db_f_delta_norm=" << std::max(0.0, phase3GateBfAfter - phase3GateBfBefore)
                              << ",db_g_delta_norm=" << std::max(0.0, phase3GateBgAfter - phase3GateBgBefore)
                              << ",db_o_delta_norm=" << std::max(0.0, phase3GateBoAfter - phase3GateBoBefore)
                              << ",dW_i_after_norm=" << phase3GateWiAfter
                              << ",dW_f_after_norm=" << phase3GateWfAfter
                              << ",dW_g_after_norm=" << phase3GateWgAfter
                              << ",dW_o_after_norm=" << phase3GateWoAfter
                              << ",db_i_after_norm=" << phase3GateBiAfter
                              << ",db_f_after_norm=" << phase3GateBfAfter
                              << ",db_g_after_norm=" << phase3GateBgAfter
                              << ",db_o_after_norm=" << phase3GateBoAfter
                              << std::endl;
                }
            }
            if (phase3DhDiagEnabled)
            {
                PrintPhase3NormStats("DIAG_DH_AFTER_BPTT_", phase3DhFromHeadDiagIdx, "d_h_batch", d_h_batch);
                PrintPhase3NormStats("DIAG_DH_AFTER_BPTT_", phase3DhFromHeadDiagIdx, "d_c_batch", d_c_batch);
            }

            mergeGateAccumulators(G_bin, d_param_accum, d_bias_accum, hidden_size);
            {
                static size_t s_phase3GradDiagCount = 0;
                const size_t phase3GradDiagIdx = s_phase3GradDiagCount++;
                if (phase3GradDiagIdx < LSTM_PHASE3_HEAD_DIAG_LIMIT)
                {
                    const double mbCoreWNorm = std::sqrt(
                        std::pow(FroNormEvalHost(G_bin.dW_i), 2.0) +
                        std::pow(FroNormEvalHost(G_bin.dW_f), 2.0) +
                        std::pow(FroNormEvalHost(G_bin.dW_g), 2.0) +
                        std::pow(FroNormEvalHost(G_bin.dW_o), 2.0));
                    const double mbCoreBNorm = std::sqrt(
                        std::pow(FroNormEvalHost(G_bin.db_i), 2.0) +
                        std::pow(FroNormEvalHost(G_bin.db_f), 2.0) +
                        std::pow(FroNormEvalHost(G_bin.db_g), 2.0) +
                        std::pow(FroNormEvalHost(G_bin.db_o), 2.0));
                    std::cout << "DIAG_GRAD_PRECLIP_"
                              << ",call=" << phase3GradDiagIdx
                              << ",name=norms"
                              << ",B=" << B
                              << ",d_logits_norm=" << FroNormEvalHost(d_logits_batch)
                              << ",d_h_norm=" << FroNormEvalHost(d_h_batch)
                              << ",mb_core_weight_norm=" << mbCoreWNorm
                              << ",mb_core_bias_norm=" << mbCoreBNorm
                              << ",accum_core_weight_norm=" << FroNormEvalHost(d_param_accum)
                              << ",accum_core_bias_norm=" << FroNormEvalHost(d_bias_accum)
                              << ",accum_head_weight_norm=" << FroNormEvalHost(d_headDirW_accum_f)
                              << ",accum_head_bias_norm=" << FroNormEvalHost(d_headDirB_accum_f)
                              << std::endl;
                    // === CORE VS HEAD GRADIENT SCALE DIAGNOSTIC ===
                    const double headWNorm =
                        FroNormEvalHost(d_headDirW_accum_f);
                    const double headBNorm =
                        FroNormEvalHost(d_headDirB_accum_f);

                    const double coreWNorm =
                        FroNormEvalHost(d_param_accum);

                    const double coreBNorm =
                        FroNormEvalHost(d_bias_accum);

                    const double recurrentSliceNorm =
                    [&]() -> double
                    {
                        const size_t recurrentRowsBegin = static_cast<size_t>(n_in);
                        const size_t recurrentRowsCount = hidden_size;

                        auto low = MetaNN::LowerAccess(d_param_accum);
                        const AccumScalar* ptr = low.RawMemory();

                        const size_t cols = d_param_accum.Shape()[1];

                        double ss = 0.0;

                        for (size_t r = 0; r < recurrentRowsCount; ++r)
                        {
                            const size_t rowIdx = recurrentRowsBegin + r;

                            for (size_t c = 0; c < cols; ++c)
                            {
                                const double v =
                                    static_cast<double>(
                                        ptr[rowIdx * cols + c]);

                                ss += v * v;
                            }
                        }

                        return std::sqrt(ss);
                    }();

                    const double inputSliceNorm =
                    [&]() -> double
                    {
                        auto low = MetaNN::LowerAccess(d_param_accum);
                        const AccumScalar* ptr = low.RawMemory();

                        const size_t cols = d_param_accum.Shape()[1];

                        double ss = 0.0;

                        for (size_t rowIdx = 0;
                             rowIdx < static_cast<size_t>(n_in);
                             ++rowIdx)
                        {
                            for (size_t c = 0; c < cols; ++c)
                            {
                                const double v =
                                    static_cast<double>(
                                        ptr[rowIdx * cols + c]);

                                ss += v * v;
                            }
                        }

                        return std::sqrt(ss);
                    }();

                    std::cout << "DIAG_CORE_HEAD_GRAD_RATIO_"
                              << ",call=" << phase3GradDiagIdx
                              << ",B=" << B
                              << ",head_weight_grad_norm=" << headWNorm
                              << ",head_bias_grad_norm=" << headBNorm
                              << ",core_weight_grad_norm=" << coreWNorm
                              << ",core_bias_grad_norm=" << coreBNorm
                              << ",recurrent_grad_norm=" << recurrentSliceNorm
                              << ",input_grad_norm=" << inputSliceNorm
                              << ",core_to_head_ratio="
                              << (headWNorm > 1.0e-12
                                      ? (coreWNorm / headWNorm)
                                      : 0.0)
                              << ",recurrent_to_head_ratio="
                              << (headWNorm > 1.0e-12
                                      ? (recurrentSliceNorm / headWNorm)
                                      : 0.0)
                              << ",input_to_head_ratio="
                              << (headWNorm > 1.0e-12
                                      ? (inputSliceNorm / headWNorm)
                                      : 0.0)
                              << std::endl;                }
            }
            #if LSTM_HEAVY_DIAG
                        if ((!LSTM_DIAG_ONLY_FIRST_BATCH || isFirstBatchCall) && isFirstMiniBatch)
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
            #if LSTM_HEAVY_DIAG
                        if ((!LSTM_DIAG_ONLY_FIRST_BATCH || isFirstBatchCall) && isFirstMiniBatch)
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
    }

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
    if (targetType == TargetType::UpNeutralDownReturn)
    {
        static size_t s_phase3LabelReturnSeparationDiagCount = 0;
        if (s_phase3LabelReturnSeparationDiagCount < LSTM_PHASE3_HEAD_DIAG_LIMIT)
        {
            for (size_t cls = 0; cls < direction_output_size; ++cls)
            {
                const auto& s = phase3LabelReturnStats[cls];
                const double denom = s.count ? static_cast<double>(s.count) : 1.0;

                std::cout << "DIAG_LABEL_RETURN_SEPARATION_"
                          << ",call=" << s_phase3LabelReturnSeparationDiagCount
                          << ",class=" << Phase3ClassName(cls)
                          << ",count=" << s.count
                          << ",horizon_count=" << s.horizonCount
                          << ",mean_terminal_logret=" << (s.terminalLogReturnSum / denom)
                          << ",mean_max_future_high_logret=" << (s.maxFutureHighLogReturnSum / denom)
                          << ",mean_min_future_low_logret=" << (s.minFutureLowLogReturnSum / denom)
                          << ",mean_range_to_threshold_ratio=" << (s.rangeToThresholdRatioSum / denom)
                          << std::endl;
            }

            ++s_phase3LabelReturnSeparationDiagCount;
        }
    }
    std::cout << "batch_count=" << windowCount << "\n";
    double loss_value = 0.0;
    const double lossDenominator = (targetType == TargetType::UpNeutralDownReturn && phase3ClassWeightSum > 0.0)
        ? phase3ClassWeightSum
        : static_cast<double>(windowCount);
    loss_value = (lossDenominator > 0.0) ? (sse / lossDenominator) : 0.0;
    std::cout << "loss_value=" << loss_value << "\n";


#if LSTM_DEBUG_PRINTS
    if (y_count > 0)
    {
        double y_mean = y_sum / static_cast<double>(y_count);
        double y_var  = std::max(0.0, y_sumsq / static_cast<double>(y_count) - y_mean * y_mean);
        double y_std  = std::sqrt(y_var);
        if (targetType == TargetType::UpNeutralDownReturn)
        {
            const size_t class_total = down_count + neutral_count + up_count;
            const double down_frac = (class_total > 0) ? static_cast<double>(down_count) / static_cast<double>(class_total) : 0.0;
            const double neutral_frac = (class_total > 0) ? static_cast<double>(neutral_count) / static_cast<double>(class_total) : 0.0;
            const double up_frac = (class_total > 0) ? static_cast<double>(up_count) / static_cast<double>(class_total) : 0.0;

            std::cout << "train: class_target counts n=" << class_total
                      << " down=" << down_count
                      << " neutral=" << neutral_count
                      << " up=" << up_count
                      << " down_frac=" << down_frac
                      << " neutral_frac=" << neutral_frac
                      << " up_frac=" << up_frac
                      << std::endl;
            std::cout << "train: pred_class samples:";
            for (float v : yhat_samples) std::cout << ' ' << v;
            std::cout << std::endl;
        }
        else
        {
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
        }
        std::cout << "train: skipped_windows=" << skippedWindows << std::endl;
    }

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
    if (windowCount > 0)
    {
        // Convert accumulators to concrete matrices (ensures RawMemory is valid)
        auto d_param_f = MetaNN::Evaluate(d_param_accum);
        auto d_bias_f  = MetaNN::Evaluate(d_bias_accum);
        auto d_headW_f = MetaNN::Evaluate(d_headW_accum_f);
        auto d_headB_f = MetaNN::Evaluate(d_headB_accum_f);

        auto d_headDirW_f = MetaNN::Evaluate(d_headDirW_accum_f);
        auto d_headDirB_f = MetaNN::Evaluate(d_headDirB_accum_f);

        // Convert accumulated gradients to mean gradients before clipping.
        // Classification keeps class weights in d_logits_batch and uses this
        // shared invN path so head and recurrent gradients have one denominator.
        const double gradDenominator = (targetType == TargetType::UpNeutralDownReturn && phase3ClassWeightSum > 0.0)
            ? phase3ClassWeightSum
            : static_cast<double>(windowCount);
        const char* gradDenominatorName = (targetType == TargetType::UpNeutralDownReturn)
            ? "weighted_sample_sum"
            : "windowCount";
        const float invN = 1.0f / static_cast<float>(gradDenominator);
        auto phase3ScaleMatrixInPlace = [](auto& m, float scale)
        {
            auto low = MetaNN::LowerAccess(m);
            auto* ptr = low.MutableRawMemory();
            const size_t count = m.Shape()[0] * m.Shape()[1];
            for (size_t idx = 0; idx < count; ++idx)
                ptr[idx] = ptr[idx] * scale;
        };

        phase3ScaleMatrixInPlace(d_param_f, invN);
        phase3ScaleMatrixInPlace(d_bias_f, invN);
        phase3ScaleMatrixInPlace(d_headW_f, invN);
        phase3ScaleMatrixInPlace(d_headB_f, invN);
        phase3ScaleMatrixInPlace(d_headDirW_f, invN);
        phase3ScaleMatrixInPlace(d_headDirB_f, invN);

        auto d_param_preclip = NNUtils::DeepCopyMatrix(d_param_f);
        auto d_bias_preclip = NNUtils::DeepCopyMatrix(d_bias_f);
        auto d_headW_preclip = NNUtils::DeepCopyMatrix(d_headW_f);
        auto d_headB_preclip = NNUtils::DeepCopyMatrix(d_headB_f);
        auto d_headDirW_preclip = NNUtils::DeepCopyMatrix(d_headDirW_f);
        auto d_headDirB_preclip = NNUtils::DeepCopyMatrix(d_headDirB_f);

        // For UpNeutralDownReturn, allow the recurrent core to learn faster
        // so hidden-state geometry can keep up with the direction head.
        const float core_lr_mult = CoreLrMultForTarget(targetType);

        const float lrCore = learning_rate * core_lr_mult;
        const float lrHeadBase = learning_rate * LSTM_HEAD_LR_MULT;
        const float lrHeadDirActiveMult =
            (targetType == TargetType::UpNeutralDownReturn) ? 0.5f : 1.0f;
        const float lrHead = lrHeadBase * lrHeadDirActiveMult;
        const float lrHeadBias = lrHead * 0.01f; // intentionally slower bias adaptation

        const float directionHeadWeightLr = learning_rate * head_weight_lr_mult;
        const float directionHeadBiasLr   = learning_rate * head_bias_lr_mult;

        static size_t s_phase3LrScaleDiagCount = 0;
        const size_t phase3LrScaleDiagIdx = s_phase3LrScaleDiagCount++;
        const bool phase3LrScaleDiagEnabled = (phase3LrScaleDiagIdx < LSTM_PHASE3_HEAD_DIAG_LIMIT);
        if (phase3LrScaleDiagEnabled)
        {
            const size_t plannedMiniBatches = effectiveMiniBatchWindows
                ? ((allStarts.size() + effectiveMiniBatchWindows - 1) / effectiveMiniBatchWindows)
                : 0;
            std::cout << "DIAG_UPDATE_DENOM_"
                      << ",call=" << phase3LrScaleDiagIdx
                      << ",denominator_name=" << gradDenominatorName
                      << ",denominator_value=" << gradDenominator
                      << ",raw_windowCount=" << windowCount
                      << ",weighted_sample_sum=" << phase3ClassWeightSum
                      << ",uses_weighted_sample_sum=" << (targetType == TargetType::UpNeutralDownReturn ? 1 : 0)
                      << ",mseCount=" << mseCount
                      << ",allStarts_count=" << allStarts.size()
                      << ",effectiveMiniBatchWindows=" << effectiveMiniBatchWindows
                      << ",plannedMiniBatches=" << plannedMiniBatches
                      << ",profileMiniBatches=" << profile.mini_batches
                      << ",denominator_is_full_calculate_batch_windows=" << (targetType == TargetType::UpNeutralDownReturn ? 0 : 1)
                      << ",denominator_is_weighted_sample_sum=" << (targetType == TargetType::UpNeutralDownReturn ? 1 : 0)
                      << ",gradients_preclip_are_mean_gradients=1"
                      << ",denominator_is_minibatch_size=0"
                      << std::endl;
            std::cout << "DIAG_LR_SCALE_"
                      << ",call=" << phase3LrScaleDiagIdx
                      << ",raw_learningRate=" << learning_rate
                      << ",core_lr_mult=" << core_lr_mult
                      << ",head_lr_mult=" << LSTM_HEAD_LR_MULT
                      << ",head_lr_base=" << lrHeadBase
                      << ",direction_head_active_lr_mult=" << lrHeadDirActiveMult
                      << ",direction_head_active_path=" << (targetType == TargetType::UpNeutralDownReturn ? 1 : 0)
                      << ",invN=" << invN
                      << ",lrCore=" << lrCore
                      << ",lrHead=" << lrHead
                      << ",lrHeadBias=" << lrHeadBias
                      << ",directionHeadWeightLr=" << directionHeadWeightLr
                      << ",directionHeadBiasLr=" << directionHeadBiasLr
                      << ",head_weight_lr_mult=" << head_weight_lr_mult
                      << ",head_bias_lr_mult=" << head_bias_lr_mult
                      << ",core_formula=learningRate_after_mean_gradient_scaling"
                      << ",head_formula=learningRate*LSTM_HEAD_LR_MULT*direction_head_active_lr_mult_after_mean_gradient_scaling"
                      << ",direction_head_override_formula=weight:learningRate*head_weight_lr_mult,bias:learningRate*head_bias_lr_mult_after_mean_gradient_scaling"
                      << std::endl;
        }

        bool gradsFinite =
            MatrixAllFiniteHost("d_param", d_param_f) &&
            MatrixAllFiniteHost("d_bias", d_bias_f) &&
            MatrixAllFiniteHost("d_headW", d_headW_f) &&
            MatrixAllFiniteHost("d_headB", d_headB_f) &&
            MatrixAllFiniteHost("d_headDirW", d_headDirW_f) &&
            MatrixAllFiniteHost("d_headDirB", d_headDirB_f);

        static size_t s_phase3ClipFullDiagCount = 0;
        const size_t phase3ClipFullDiagIdx = s_phase3ClipFullDiagCount++;
        const bool phase3ClipFullDiagEnabled = (phase3ClipFullDiagIdx < LSTM_PHASE3_HEAD_DIAG_LIMIT);
        if (phase3ClipFullDiagEnabled)
        {
            PrintPhase3ClipMatrixStats("DIAG_GRAD_PRECLIP_FULL_", phase3ClipFullDiagIdx, "d_param", d_param_preclip, LSTM_GRAD_CLIP_THRESHOLD);
            PrintPhase3ClipMatrixStats("DIAG_GRAD_PRECLIP_FULL_", phase3ClipFullDiagIdx, "d_bias", d_bias_preclip, LSTM_GRAD_CLIP_THRESHOLD);
            if (targetType == TargetType::UpNeutralDownReturn)
            {
                PrintPhase3ClipMatrixStats("DIAG_GRAD_PRECLIP_FULL_", phase3ClipFullDiagIdx, "d_headDirW", d_headDirW_preclip, LSTM_GRAD_CLIP_THRESHOLD);
                PrintPhase3ClipMatrixStats("DIAG_GRAD_PRECLIP_FULL_", phase3ClipFullDiagIdx, "d_headDirB", d_headDirB_preclip, LSTM_GRAD_CLIP_THRESHOLD);
            }
            else
            {
                PrintPhase3ClipMatrixStats("DIAG_GRAD_PRECLIP_FULL_", phase3ClipFullDiagIdx, "d_headW", d_headW_preclip, LSTM_GRAD_CLIP_THRESHOLD);
                PrintPhase3ClipMatrixStats("DIAG_GRAD_PRECLIP_FULL_", phase3ClipFullDiagIdx, "d_headB", d_headB_preclip, LSTM_GRAD_CLIP_THRESHOLD);
            }
        }

#if LSTM_USE_GRAD_CLIP
        if (gradsFinite)
        {
            HotspotScope hotspot("gradient_clipping");
            ClipMatrixInPlace(d_param_f, LSTM_GRAD_CLIP_THRESHOLD, "d_param");
            ClipMatrixInPlace(d_bias_f, LSTM_GRAD_CLIP_THRESHOLD, "d_bias");
            ClipMatrixInPlace(d_headW_f, LSTM_GRAD_CLIP_THRESHOLD, "d_headW");
            ClipMatrixInPlace(d_headB_f, LSTM_GRAD_CLIP_THRESHOLD, "d_headB");
            ClipMatrixInPlace(d_headDirW_f, LSTM_GRAD_CLIP_THRESHOLD, "d_headDirW");
            ClipMatrixInPlace(d_headDirB_f, LSTM_GRAD_CLIP_THRESHOLD, "d_headDirB");
        }
#endif
        if (phase3ClipFullDiagEnabled)
        {
            PrintPhase3ClipMatrixStats("DIAG_GRAD_POSTCLIP_FULL_", phase3ClipFullDiagIdx, "d_param", d_param_f, LSTM_GRAD_CLIP_THRESHOLD);
            PrintPhase3ClipMatrixStats("DIAG_GRAD_POSTCLIP_FULL_", phase3ClipFullDiagIdx, "d_bias", d_bias_f, LSTM_GRAD_CLIP_THRESHOLD);
            PrintPhase3ClipEffect(phase3ClipFullDiagIdx, "d_param", d_param_preclip, d_param_f, LSTM_GRAD_CLIP_THRESHOLD);
            PrintPhase3ClipEffect(phase3ClipFullDiagIdx, "d_bias", d_bias_preclip, d_bias_f, LSTM_GRAD_CLIP_THRESHOLD);
            if (targetType == TargetType::UpNeutralDownReturn)
            {
                PrintPhase3ClipMatrixStats("DIAG_GRAD_POSTCLIP_FULL_", phase3ClipFullDiagIdx, "d_headDirW", d_headDirW_f, LSTM_GRAD_CLIP_THRESHOLD);
                PrintPhase3ClipMatrixStats("DIAG_GRAD_POSTCLIP_FULL_", phase3ClipFullDiagIdx, "d_headDirB", d_headDirB_f, LSTM_GRAD_CLIP_THRESHOLD);
                PrintPhase3ClipEffect(phase3ClipFullDiagIdx, "d_headDirW", d_headDirW_preclip, d_headDirW_f, LSTM_GRAD_CLIP_THRESHOLD);
                PrintPhase3ClipEffect(phase3ClipFullDiagIdx, "d_headDirB", d_headDirB_preclip, d_headDirB_f, LSTM_GRAD_CLIP_THRESHOLD);
            }
            else
            {
                PrintPhase3ClipMatrixStats("DIAG_GRAD_POSTCLIP_FULL_", phase3ClipFullDiagIdx, "d_headW", d_headW_f, LSTM_GRAD_CLIP_THRESHOLD);
                PrintPhase3ClipMatrixStats("DIAG_GRAD_POSTCLIP_FULL_", phase3ClipFullDiagIdx, "d_headB", d_headB_f, LSTM_GRAD_CLIP_THRESHOLD);
                PrintPhase3ClipEffect(phase3ClipFullDiagIdx, "d_headW", d_headW_preclip, d_headW_f, LSTM_GRAD_CLIP_THRESHOLD);
                PrintPhase3ClipEffect(phase3ClipFullDiagIdx, "d_headB", d_headB_preclip, d_headB_f, LSTM_GRAD_CLIP_THRESHOLD);
            }
            PrintPhase3UpdateScale(phase3ClipFullDiagIdx, "param", param, d_param_f, lrCore);
            PrintPhase3UpdateScale(phase3ClipFullDiagIdx, "bias", bias, d_bias_f, lrCore);
            std::cout << "DIAG_UPDATE_EFFECTIVE_"
                      << ",call=" << phase3ClipFullDiagIdx
                      << ",name=param"
                      << ",effective_lr=" << lrCore
                      << ",denominator_name=" << gradDenominatorName
                      << ",denominator_value=" << gradDenominator
                      << ",grad_norm_after_clip=" << FroNormEvalHost(d_param_f)
                      << ",param_norm=" << FroNormEvalHost(param)
                      << ",update_norm=" << (static_cast<double>(lrCore) * FroNormEvalHost(d_param_f))
                      << std::endl;
            std::cout << "DIAG_UPDATE_EFFECTIVE_"
                      << ",call=" << phase3ClipFullDiagIdx
                      << ",name=bias"
                      << ",effective_lr=" << lrCore
                      << ",denominator_name=" << gradDenominatorName
                      << ",denominator_value=" << gradDenominator
                      << ",grad_norm_after_clip=" << FroNormEvalHost(d_bias_f)
                      << ",bias_norm=" << FroNormEvalHost(bias)
                      << ",update_norm=" << (static_cast<double>(lrCore) * FroNormEvalHost(d_bias_f))
                      << std::endl;
            if (targetType == TargetType::UpNeutralDownReturn)
            {
                PrintPhase3UpdateScale(phase3ClipFullDiagIdx, "returnHeadDirWeight", returnHeadDirWeight, d_headDirW_f, directionHeadWeightLr);
                PrintPhase3UpdateScale(phase3ClipFullDiagIdx, "returnHeadDirBias", returnHeadDirBias, d_headDirB_f, directionHeadBiasLr);
                PrintHeadGradNormDiag(
                    phase3ClipFullDiagIdx,
                    d_headDirW_f, d_headDirB_f,
                    returnHeadDirWeight, returnHeadDirBias,
                    directionHeadWeightLr, directionHeadBiasLr);
                {
                    auto phase3HeadGradLow = MetaNN::LowerAccess(d_headDirW_f);
                    const float* phase3HeadGradPtr = phase3HeadGradLow.RawMemory();

                    const size_t phase3HeadGradRows = d_headDirW_f.Shape()[0];
                    const size_t phase3HeadGradCols = d_headDirW_f.Shape()[1];

                    std::array<long double, direction_output_size> phase3HeadGradColSum {0.0L, 0.0L, 0.0L};
                    std::array<long double, direction_output_size> phase3HeadGradColSqSum {0.0L, 0.0L, 0.0L};
                    std::array<double, direction_output_size> phase3HeadGradColAbsMax {0.0, 0.0, 0.0};
                    std::array<size_t, direction_output_size> phase3HeadGradFiniteCount {0, 0, 0};
                    std::array<size_t, direction_output_size> phase3HeadGradNonFiniteCount {0, 0, 0};

                    if (phase3HeadGradCols >= direction_output_size)
                    {
                        for (size_t r = 0; r < phase3HeadGradRows; ++r)
                        {
                            for (size_t cls = 0; cls < direction_output_size; ++cls)
                            {
                                const double v = static_cast<double>(
                                    phase3HeadGradPtr[r * phase3HeadGradCols + cls]);

                                if (!std::isfinite(v))
                                {
                                    ++phase3HeadGradNonFiniteCount[cls];
                                    continue;
                                }

                                ++phase3HeadGradFiniteCount[cls];
                                phase3HeadGradColSum[cls] += static_cast<long double>(v);
                                phase3HeadGradColSqSum[cls] += static_cast<long double>(v) * static_cast<long double>(v);
                                phase3HeadGradColAbsMax[cls] = std::max(phase3HeadGradColAbsMax[cls], std::fabs(v));
                            }
                        }
                    }

                    auto phase3HeadGradColNorm = [&](size_t cls) -> double
                    {
                        return std::sqrt(static_cast<double>(phase3HeadGradColSqSum[cls]));
                    };

                    auto phase3HeadGradColMean = [&](size_t cls) -> double
                    {
                        return phase3HeadGradFiniteCount[cls] > 0
                            ? static_cast<double>(phase3HeadGradColSum[cls] /
                                                  static_cast<long double>(phase3HeadGradFiniteCount[cls]))
                            : 0.0;
                    };

                    std::cout << "DIAG_HEAD_CLASS_GRAD_"
                              << ",call=" << phase3ClipFullDiagIdx
                              << ",rows=" << phase3HeadGradRows
                              << ",cols=" << phase3HeadGradCols
                              << ",dW_down_norm=" << phase3HeadGradColNorm(0)
                              << ",dW_neutral_norm=" << phase3HeadGradColNorm(1)
                              << ",dW_up_norm=" << phase3HeadGradColNorm(2)
                              << ",dW_down_mean=" << phase3HeadGradColMean(0)
                              << ",dW_neutral_mean=" << phase3HeadGradColMean(1)
                              << ",dW_up_mean=" << phase3HeadGradColMean(2)
                              << ",dW_down_absmax=" << phase3HeadGradColAbsMax[0]
                              << ",dW_neutral_absmax=" << phase3HeadGradColAbsMax[1]
                              << ",dW_up_absmax=" << phase3HeadGradColAbsMax[2]
                              << ",dW_down_finite=" << phase3HeadGradFiniteCount[0]
                              << ",dW_neutral_finite=" << phase3HeadGradFiniteCount[1]
                              << ",dW_up_finite=" << phase3HeadGradFiniteCount[2]
                              << ",dW_down_nonfinite=" << phase3HeadGradNonFiniteCount[0]
                              << ",dW_neutral_nonfinite=" << phase3HeadGradNonFiniteCount[1]
                              << ",dW_up_nonfinite=" << phase3HeadGradNonFiniteCount[2]
                              << std::endl;
                }
#if LSTM_HEAVY_DIAG
                // --- BEGIN PATCH: DIAG HEAD WEIGHT BY CLASS ---
                {
                    auto phase3HeadWLow = MetaNN::LowerAccess(returnHeadDirWeight);
                    const float* phase3HeadWPtr = phase3HeadWLow.RawMemory();

                    const size_t phase3HeadWRows = returnHeadDirWeight.Shape()[0];
                    const size_t phase3HeadWCols = returnHeadDirWeight.Shape()[1];

                    std::array<long double, direction_output_size> phase3HeadWColSum {0.0L, 0.0L, 0.0L};
                    std::array<long double, direction_output_size> phase3HeadWColSqSum {0.0L, 0.0L, 0.0L};
                    std::array<double, direction_output_size> phase3HeadWColAbsMax {0.0, 0.0, 0.0};
                    std::array<size_t, direction_output_size> phase3HeadWFiniteCount {0, 0, 0};
                    std::array<size_t, direction_output_size> phase3HeadWNonFiniteCount {0, 0, 0};

                    if (phase3HeadWCols >= direction_output_size)
                    {
                        for (size_t r = 0; r < phase3HeadWRows; ++r)
                        {
                            for (size_t cls = 0; cls < direction_output_size; ++cls)
                            {
                                const double v = static_cast<double>(
                                    phase3HeadWPtr[r * phase3HeadWCols + cls]);

                                if (!std::isfinite(v))
                                {
                                    ++phase3HeadWNonFiniteCount[cls];
                                    continue;
                                }

                                ++phase3HeadWFiniteCount[cls];
                                phase3HeadWColSum[cls] += static_cast<long double>(v);
                                phase3HeadWColSqSum[cls] += static_cast<long double>(v) * static_cast<long double>(v);
                                phase3HeadWColAbsMax[cls] = std::max(phase3HeadWColAbsMax[cls], std::fabs(v));
                            }
                        }
                    }

                    auto phase3HeadWColNorm = [&](size_t cls) -> double
                    {
                        return std::sqrt(static_cast<double>(phase3HeadWColSqSum[cls]));
                    };

                    auto phase3HeadWColMean = [&](size_t cls) -> double
                    {
                        return phase3HeadWFiniteCount[cls] > 0
                            ? static_cast<double>(phase3HeadWColSum[cls] /
                                                  static_cast<long double>(phase3HeadWFiniteCount[cls]))
                            : 0.0;
                    };

                    std::cout << "DIAG_HEAD_WEIGHT_BY_CLASS_"
                              << ",call=" << phase3ClipFullDiagIdx
                              << ",rows=" << phase3HeadWRows
                              << ",cols=" << phase3HeadWCols
                              << ",W_down_norm=" << phase3HeadWColNorm(0)
                              << ",W_neutral_norm=" << phase3HeadWColNorm(1)
                              << ",W_up_norm=" << phase3HeadWColNorm(2)
                              << ",W_down_mean=" << phase3HeadWColMean(0)
                              << ",W_neutral_mean=" << phase3HeadWColMean(1)
                              << ",W_up_mean=" << phase3HeadWColMean(2)
                              << ",W_down_absmax=" << phase3HeadWColAbsMax[0]
                              << ",W_neutral_absmax=" << phase3HeadWColAbsMax[1]
                              << ",W_up_absmax=" << phase3HeadWColAbsMax[2]
                              << ",W_down_finite=" << phase3HeadWFiniteCount[0]
                              << ",W_neutral_finite=" << phase3HeadWFiniteCount[1]
                              << ",W_up_finite=" << phase3HeadWFiniteCount[2]
                              << ",W_down_nonfinite=" << phase3HeadWNonFiniteCount[0]
                              << ",W_neutral_nonfinite=" << phase3HeadWNonFiniteCount[1]
                              << ",W_up_nonfinite=" << phase3HeadWNonFiniteCount[2]
                              << std::endl;
                }
                // --- END PATCH ---
                // --- BEGIN PATCH: DIAG HEAD BIAS BY CLASS PRE-UPDATE ---
                {
                    auto phase3HeadBLow = MetaNN::LowerAccess(returnHeadDirBias);
                    const float* phase3HeadBPtr = phase3HeadBLow.RawMemory();

                    const size_t phase3HeadBRows = returnHeadDirBias.Shape()[0];
                    const size_t phase3HeadBCols = returnHeadDirBias.Shape()[1];

                    double bDown = 0.0;
                    double bNeutral = 0.0;
                    double bUp = 0.0;

                    if (phase3HeadBRows >= 1 && phase3HeadBCols >= direction_output_size)
                    {
                        bDown = static_cast<double>(phase3HeadBPtr[0]);
                        bNeutral = static_cast<double>(phase3HeadBPtr[1]);
                        bUp = static_cast<double>(phase3HeadBPtr[2]);
                    }

                    std::cout << "DIAG_HEAD_BIAS_BY_CLASS_"
                              << ",call=" << phase3ClipFullDiagIdx
                              << ",stage=pre_update"
                              << ",rows=" << phase3HeadBRows
                              << ",cols=" << phase3HeadBCols
                              << ",B_down=" << bDown
                              << ",B_neutral=" << bNeutral
                              << ",B_up=" << bUp
                              << ",B_down_minus_neutral=" << (bDown - bNeutral)
                              << ",B_up_minus_neutral=" << (bUp - bNeutral)
                              << ",B_neutral_minus_mean_edges=" << (bNeutral - 0.5 * (bDown + bUp))
                              << std::endl;
                }
                // --- END PATCH ---
#endif

                std::cout << "DIAG_UPDATE_EFFECTIVE_"
                          << ",call=" << phase3ClipFullDiagIdx
                          << ",name=returnHeadDirWeight"
                          << ",effective_lr=" << directionHeadWeightLr
                          << ",denominator_name=" << gradDenominatorName
                          << ",denominator_value=" << gradDenominator
                          << ",grad_norm_after_clip=" << FroNormEvalHost(d_headDirW_f)
                          << ",update_norm=" << (static_cast<double>(directionHeadWeightLr) * FroNormEvalHost(d_headDirW_f))
                          << std::endl;
                std::cout << "DIAG_UPDATE_EFFECTIVE_"
                          << ",call=" << phase3ClipFullDiagIdx
                          << ",name=returnHeadDirBias"
                          << ",effective_lr=" << directionHeadBiasLr
                          << ",denominator_name=" << gradDenominatorName
                          << ",denominator_value=" << gradDenominator
                          << ",grad_norm_after_clip=" << FroNormEvalHost(d_headDirB_f)
                          << ",update_norm=" << (static_cast<double>(directionHeadBiasLr) * FroNormEvalHost(d_headDirB_f))
                          << std::endl;
            }
            else
            {
                PrintPhase3UpdateScale(phase3ClipFullDiagIdx, "returnHeadWeight", returnHeadWeight, d_headW_f, lrHead);
                PrintPhase3UpdateScale(phase3ClipFullDiagIdx, "returnHeadBias", returnHeadBias, d_headB_f, lrHead);
                std::cout << "DIAG_UPDATE_EFFECTIVE_"
                          << ",call=" << phase3ClipFullDiagIdx
                          << ",name=returnHeadWeight"
                          << ",effective_lr=" << lrHead
                          << ",denominator_name=" << gradDenominatorName
                          << ",denominator_value=" << gradDenominator
                          << ",grad_norm_after_clip=" << FroNormEvalHost(d_headW_f)
                          << ",update_norm=" << (static_cast<double>(lrHead) * FroNormEvalHost(d_headW_f))
                          << std::endl;
                std::cout << "DIAG_UPDATE_EFFECTIVE_"
                          << ",call=" << phase3ClipFullDiagIdx
                          << ",name=returnHeadBias"
                          << ",effective_lr=" << lrHead
                          << ",denominator_name=" << gradDenominatorName
                          << ",denominator_value=" << gradDenominator
                          << ",grad_norm_after_clip=" << FroNormEvalHost(d_headB_f)
                          << ",update_norm=" << (static_cast<double>(lrHead) * FroNormEvalHost(d_headB_f))
                          << std::endl;
            }
{
    static size_t s_phase3CoreUpdateScaleCount = 0;
    if (s_phase3CoreUpdateScaleCount < LSTM_PHASE3_HEAD_DIAG_LIMIT)
    {
        const bool directionHeadPath = (targetType == TargetType::UpNeutralDownReturn);

        const double coreParamNorm = FroNormEvalHost(param);
        const double coreBiasNorm = FroNormEvalHost(bias);
        const double headWNorm = directionHeadPath
            ? FroNormEvalHost(returnHeadDirWeight)
            : FroNormEvalHost(returnHeadWeight);
        const double headBNorm = directionHeadPath
            ? FroNormEvalHost(returnHeadDirBias)
            : FroNormEvalHost(returnHeadBias);

        const double coreParamGradNorm = FroNormEvalHost(d_param_f);
        const double coreBiasGradNorm = FroNormEvalHost(d_bias_f);
        const double headWGradNorm = directionHeadPath
            ? FroNormEvalHost(d_headDirW_f)
            : FroNormEvalHost(d_headW_f);
        const double headBGradNorm = directionHeadPath
            ? FroNormEvalHost(d_headDirB_f)
            : FroNormEvalHost(d_headB_f);

        const double activeHeadWlr = directionHeadPath ? directionHeadWeightLr : lrHead;
        const double activeHeadBlr = directionHeadPath ? directionHeadBiasLr : lrHead;

        const double coreParamUpdateNorm = static_cast<double>(lrCore) * coreParamGradNorm;
        const double coreBiasUpdateNorm = static_cast<double>(lrCore) * coreBiasGradNorm;
        const double headWUpdateNorm = activeHeadWlr * headWGradNorm;
        const double headBUpdateNorm = activeHeadBlr * headBGradNorm;

        std::cout << "DIAG_CORE_UPDATE_SCALE_"
                  << ",call=" << s_phase3CoreUpdateScaleCount
                  << ",phase3_call=" << phase3ClipFullDiagIdx
                  << ",target=" << (directionHeadPath ? "UpNeutralDownReturn" : "RegressionReturn")
                  << ",windowCount=" << windowCount
                  << ",effectiveMiniBatchWindows=" << effectiveMiniBatchWindows
                  << ",learningRate=" << learning_rate
                  << ",core_lr=" << lrCore
                  << ",head_lr_base=" << lrHeadBase
                  << ",head_weight_lr=" << activeHeadWlr
                  << ",head_bias_lr=" << activeHeadBlr
                  << ",head_lr_mult=" << LSTM_HEAD_LR_MULT
                  << ",direction_head_active_lr_mult=" << lrHeadDirActiveMult
                  << ",direction_head_active_path=" << (directionHeadPath ? 1 : 0)
                  << ",core_param_norm=" << coreParamNorm
                  << ",core_param_grad_norm=" << coreParamGradNorm
                  << ",core_param_update_norm=" << coreParamUpdateNorm
                  << ",core_param_update_ratio=" << (coreParamNorm > 0.0 ? coreParamUpdateNorm / coreParamNorm : 0.0)
                  << ",core_bias_norm=" << coreBiasNorm
                  << ",core_bias_grad_norm=" << coreBiasGradNorm
                  << ",core_bias_update_norm=" << coreBiasUpdateNorm
                  << ",core_bias_update_ratio=" << (coreBiasNorm > 0.0 ? coreBiasUpdateNorm / coreBiasNorm : 0.0)
                  << ",head_weight_norm=" << headWNorm
                  << ",head_weight_grad_norm=" << headWGradNorm
                  << ",head_weight_update_norm=" << headWUpdateNorm
                  << ",head_weight_update_ratio=" << (headWNorm > 0.0 ? headWUpdateNorm / headWNorm : 0.0)
                  << ",head_bias_norm=" << headBNorm
                  << ",head_bias_grad_norm=" << headBGradNorm
                  << ",head_bias_update_norm=" << headBUpdateNorm
                  << ",head_bias_update_ratio=" << (headBNorm > 0.0 ? headBUpdateNorm / headBNorm : 0.0)
                  << ",core_to_head_weight_update_norm_ratio=" << (headWUpdateNorm > 0.0 ? coreParamUpdateNorm / headWUpdateNorm : 0.0)
                  << ",head_to_core_weight_update_norm_ratio=" << (coreParamUpdateNorm > 0.0 ? headWUpdateNorm / coreParamUpdateNorm : 0.0)
                  << ",core_to_head_bias_update_norm_ratio=" << (headBUpdateNorm > 0.0 ? coreBiasUpdateNorm / headBUpdateNorm : 0.0)
                  << ",head_to_core_bias_update_norm_ratio=" << (coreBiasUpdateNorm > 0.0 ? headBUpdateNorm / coreBiasUpdateNorm : 0.0)
                  << std::endl;

        ++s_phase3CoreUpdateScaleCount;
    }
}
        }
        
        #if !LSTM_DISABLE_UPDATES
            EAMatrix phase3HeadWUpdateBefore = (targetType == TargetType::UpNeutralDownReturn)
                ? NNUtils::DeepCopyMatrix(returnHeadDirWeight)
                : NNUtils::DeepCopyMatrix(returnHeadWeight);
            EAMatrix phase3HeadBUpdateBefore = (targetType == TargetType::UpNeutralDownReturn)
                ? NNUtils::DeepCopyMatrix(returnHeadDirBias)
                : NNUtils::DeepCopyMatrix(returnHeadBias);
            if (!gradsFinite)
            {
                std::cout << "DIAG_SKIP_UPDATE_NONFINITE_GRAD"
                          << ",calcBatchCall=" << calcBatchCallIdx
                          << ",windowCount=" << windowCount
                          << "\n";
            }
            else
            {
                {
                    HotspotScope hotspot("optimizer_update");
                    SGDUpdate(param, d_param_f, lrCore);
                    SGDUpdate(bias,  d_bias_f,  lrCore);
                }
                if (targetType == TargetType::UpNeutralDownReturn) {
                    const double directionHeadWeightGradNorm = FroNormEvalHost(d_headDirW_f);
                    const double directionHeadBiasGradNorm = FroNormEvalHost(d_headDirB_f);
                    const double directionHeadWeightEffectiveScale =
                        static_cast<double>(directionHeadWeightLr) * static_cast<double>(invN);
                    const double directionHeadBiasEffectiveScale =
                        static_cast<double>(directionHeadBiasLr) * static_cast<double>(invN);

                    std::cout << "DIAG_CLASS_GRAD_NORMALIZATION"
                              << ",denom=" << gradDenominator
                              << ",windowCount=" << windowCount
                              << ",weighted_sample_sum=" << phase3ClassWeightSum
                              << ",denominator_name=" << gradDenominatorName
                              << ",uses_weighted_denom=0"
                              << ",applies_later_invN=1"
                              << ",invN=" << invN
                              << std::endl;
                    std::cout << "DIAG_DIRHEAD_UPDATE_SCALE"
                              << ",effective_scale=" << directionHeadWeightEffectiveScale
                              << ",effective_lr=" << directionHeadWeightLr
                              << ",grad_norm_after_scaling=" << directionHeadWeightGradNorm
                              << ",update_norm=" << (static_cast<double>(directionHeadWeightLr) * directionHeadWeightGradNorm)
                              << ",bias_effective_scale=" << directionHeadBiasEffectiveScale
                              << ",bias_effective_lr=" << directionHeadBiasLr
                              << ",bias_grad_norm_after_scaling=" << directionHeadBiasGradNorm
                              << ",bias_update_norm=" << (static_cast<double>(directionHeadBiasLr) * directionHeadBiasGradNorm)
                              << ",denom=" << gradDenominator
                              << ",windowCount=" << windowCount
                              << ",weighted_sample_sum=" << phase3ClassWeightSum
                              << ",uses_weighted_denom=0"
                              << ",applies_later_invN=1"
                              << ",invN=" << invN
                              << std::endl;

                    std::cout << "DIAG_ACTIVE_UPDATE_PATH_"
                    << ",target=UpNeutralDownReturn"
                    << ",name=returnHeadDirWeight"
                    << ",effective_lr=" << directionHeadWeightLr
                    << ",base_head_lr=" << lrHeadBase
                    << ",direction_head_active_lr_mult=" << lrHeadDirActiveMult
                    << ",grad_norm=" << directionHeadWeightGradNorm
                    << ",expected_update_norm=" << (static_cast<double>(directionHeadWeightLr) * directionHeadWeightGradNorm)
                    << std::endl;

                    std::cout << "DIAG_ACTIVE_UPDATE_PATH_"
                    << ",target=UpNeutralDownReturn"
                    << ",name=returnHeadDirBias"
                    << ",effective_lr=" << directionHeadBiasLr
                    << ",base_head_lr=" << lrHeadBase
                    << ",direction_head_active_lr_mult=" << lrHeadDirActiveMult
                    << ",grad_norm=" << directionHeadBiasGradNorm
                    << ",expected_update_norm=" << (static_cast<double>(directionHeadBiasLr) * directionHeadBiasGradNorm)
                    << std::endl;

                    const auto dirBiasBeforeUpdate = DirectionBiasValues3(returnHeadDirBias);
                    PrintDirectionBiasByClassDiag(phase3ClipFullDiagIdx, "pre_update", dirBiasBeforeUpdate);

                    {
                        HotspotScope hotspot("optimizer_update");
                        SGDUpdate(returnHeadDirWeight, d_headDirW_f, directionHeadWeightLr);
                        SGDUpdate(returnHeadDirBias, d_headDirB_f, directionHeadBiasLr);
                    }

                    const auto dirBiasAfterUpdate = DirectionBiasValues3(returnHeadDirBias);
                    PrintDirectionBiasByClassDiag(phase3ClipFullDiagIdx, "post_update", dirBiasAfterUpdate);
                    PrintDirectionBiasDeltaByClassDiag(phase3ClipFullDiagIdx, dirBiasBeforeUpdate, dirBiasAfterUpdate);
                } else {
                    HotspotScope hotspot("optimizer_update");
                    SGDUpdate(returnHeadWeight, d_headW_f, lrHead);
                    SGDUpdate(returnHeadBias,   d_headB_f, lrHead);
                }
                ++optimizerUpdateCount;
            }
            const bool phase3HiddenGeometryCheckpointDiag =
                !EA::LSTM::suppressPhase3HiddenGeometryDiagnostics &&
                targetType == TargetType::UpNeutralDownReturn &&
                ((epochIdx + 1) == 1 ||
                 ((epochIdx + 1) % 5) == 0 ||
                 (epochIdx + 1) == epoch_count);

            const bool phase3PostUpdateDiagEnabled =
                phase3ClipFullDiagEnabled || phase3HiddenGeometryCheckpointDiag;

            if (phase3PostUpdateDiagEnabled)
            {
                if (targetType == TargetType::UpNeutralDownReturn)
                {
                    if (phase3ClipFullDiagEnabled)
                    {
                        PrintPhase3HeadDelta(phase3ClipFullDiagIdx, "returnHeadDirWeight", returnHeadDirWeight, phase3HeadWUpdateBefore);
                        PrintPhase3HeadDelta(phase3ClipFullDiagIdx, "returnHeadDirBias", returnHeadDirBias, phase3HeadBUpdateBefore);
                    // --- BEGIN PATCH: DIAG HEAD WEIGHT DELTA BY CLASS ---
                    if (phase3ClipFullDiagEnabled)
                    {
                        {
                            auto phase3WeightBeforeLow = MetaNN::LowerAccess(phase3HeadWUpdateBefore);
                            auto phase3WeightAfterLow = MetaNN::LowerAccess(returnHeadDirWeight);
                            const float* phase3WeightBeforePtr = phase3WeightBeforeLow.RawMemory();
                            const float* phase3WeightAfterPtr = phase3WeightAfterLow.RawMemory();

                            const size_t phase3WeightRows = returnHeadDirWeight.Shape()[0];
                            const size_t phase3WeightCols = returnHeadDirWeight.Shape()[1];

                            std::array<long double, direction_output_size> beforeSqSum {0.0L, 0.0L, 0.0L};
                            std::array<long double, direction_output_size> afterSqSum {0.0L, 0.0L, 0.0L};
                            std::array<long double, direction_output_size> deltaSqSum {0.0L, 0.0L, 0.0L};
                            std::array<long double, direction_output_size> deltaSum {0.0L, 0.0L, 0.0L};
                            std::array<double, direction_output_size> deltaAbsMax {0.0, 0.0, 0.0};
                            std::array<size_t, direction_output_size> finiteCount {0, 0, 0};
                            std::array<size_t, direction_output_size> nonFiniteCount {0, 0, 0};

                            if (phase3WeightCols >= direction_output_size)
                            {
                                for (size_t r = 0; r < phase3WeightRows; ++r)
                                {
                                    for (size_t cls = 0; cls < direction_output_size; ++cls)
                                    {
                                        const size_t idx = r * phase3WeightCols + cls;
                                        const double before = static_cast<double>(phase3WeightBeforePtr[idx]);
                                        const double after = static_cast<double>(phase3WeightAfterPtr[idx]);

                                        if (!std::isfinite(before) || !std::isfinite(after))
                                        {
                                            ++nonFiniteCount[cls];
                                            continue;
                                        }

                                        const double delta = after - before;
                                        ++finiteCount[cls];
                                        beforeSqSum[cls] += static_cast<long double>(before) * static_cast<long double>(before);
                                        afterSqSum[cls] += static_cast<long double>(after) * static_cast<long double>(after);
                                        deltaSqSum[cls] += static_cast<long double>(delta) * static_cast<long double>(delta);
                                        deltaSum[cls] += static_cast<long double>(delta);
                                        deltaAbsMax[cls] = std::max(deltaAbsMax[cls], std::fabs(delta));
                                    }
                                }
                            }

                            auto colNorm = [](long double ss) -> double
                            {
                                return std::sqrt(static_cast<double>(ss));
                            };

                            auto deltaMean = [&](size_t cls) -> double
                            {
                                return finiteCount[cls] > 0
                                    ? static_cast<double>(deltaSum[cls] / static_cast<long double>(finiteCount[cls]))
                                    : 0.0;
                            };

                            const double downDeltaNorm = colNorm(deltaSqSum[0]);
                            const double neutralDeltaNorm = colNorm(deltaSqSum[1]);
                            const double upDeltaNorm = colNorm(deltaSqSum[2]);
                            const double edgeDeltaMeanNorm = 0.5 * (downDeltaNorm + upDeltaNorm);

                            std::cout << "DIAG_HEAD_WEIGHT_DELTA_BY_CLASS_"
                                      << ",call=" << phase3ClipFullDiagIdx
                                      << ",rows=" << phase3WeightRows
                                      << ",cols=" << phase3WeightCols
                                      << ",before_down_norm=" << colNorm(beforeSqSum[0])
                                      << ",before_neutral_norm=" << colNorm(beforeSqSum[1])
                                      << ",before_up_norm=" << colNorm(beforeSqSum[2])
                                      << ",after_down_norm=" << colNorm(afterSqSum[0])
                                      << ",after_neutral_norm=" << colNorm(afterSqSum[1])
                                      << ",after_up_norm=" << colNorm(afterSqSum[2])
                                      << ",delta_down_norm=" << downDeltaNorm
                                      << ",delta_neutral_norm=" << neutralDeltaNorm
                                      << ",delta_up_norm=" << upDeltaNorm
                                      << ",delta_neutral_over_down=" << (downDeltaNorm > 1.0e-12 ? neutralDeltaNorm / downDeltaNorm : 0.0)
                                      << ",delta_neutral_over_up=" << (upDeltaNorm > 1.0e-12 ? neutralDeltaNorm / upDeltaNorm : 0.0)
                                      << ",delta_neutral_over_mean_edges=" << (edgeDeltaMeanNorm > 1.0e-12 ? neutralDeltaNorm / edgeDeltaMeanNorm : 0.0)
                                      << ",delta_down_mean=" << deltaMean(0)
                                      << ",delta_neutral_mean=" << deltaMean(1)
                                      << ",delta_up_mean=" << deltaMean(2)
                                      << ",delta_down_absmax=" << deltaAbsMax[0]
                                      << ",delta_neutral_absmax=" << deltaAbsMax[1]
                                      << ",delta_up_absmax=" << deltaAbsMax[2]
                                      << ",down_finite=" << finiteCount[0]
                                      << ",neutral_finite=" << finiteCount[1]
                                      << ",up_finite=" << finiteCount[2]
                                      << ",down_nonfinite=" << nonFiniteCount[0]
                                      << ",neutral_nonfinite=" << nonFiniteCount[1]
                                      << ",up_nonfinite=" << nonFiniteCount[2]
                                      << std::endl;
                        }
                        // --- END PATCH ---
#if LSTM_HEAVY_DIAG
                        // --- BEGIN PATCH: DIAG HEAD BIAS DELTA BY CLASS ---
                        {
                            auto phase3BiasBeforeLow = MetaNN::LowerAccess(phase3HeadBUpdateBefore);
                            auto phase3BiasAfterLow = MetaNN::LowerAccess(returnHeadDirBias);
                            const float* phase3BiasBeforePtr = phase3BiasBeforeLow.RawMemory();
                            const float* phase3BiasAfterPtr = phase3BiasAfterLow.RawMemory();

                            const size_t phase3BiasRows = returnHeadDirBias.Shape()[0];
                            const size_t phase3BiasCols = returnHeadDirBias.Shape()[1];

                            double beforeDown = 0.0;
                            double beforeNeutral = 0.0;
                            double beforeUp = 0.0;
                            double afterDown = 0.0;
                            double afterNeutral = 0.0;
                            double afterUp = 0.0;

                            if (phase3BiasRows >= 1 && phase3BiasCols >= direction_output_size)
                            {
                                beforeDown = static_cast<double>(phase3BiasBeforePtr[0]);
                                beforeNeutral = static_cast<double>(phase3BiasBeforePtr[1]);
                                beforeUp = static_cast<double>(phase3BiasBeforePtr[2]);
                                afterDown = static_cast<double>(phase3BiasAfterPtr[0]);
                                afterNeutral = static_cast<double>(phase3BiasAfterPtr[1]);
                                afterUp = static_cast<double>(phase3BiasAfterPtr[2]);
                            }

                            const double deltaDown = afterDown - beforeDown;
                            const double deltaNeutral = afterNeutral - beforeNeutral;
                            const double deltaUp = afterUp - beforeUp;

                            std::cout << "DIAG_HEAD_BIAS_DELTA_BY_CLASS_"
                                      << ",call=" << phase3ClipFullDiagIdx
                                      << ",rows=" << phase3BiasRows
                                      << ",cols=" << phase3BiasCols
                                      << ",before_down=" << beforeDown
                                      << ",before_neutral=" << beforeNeutral
                                      << ",before_up=" << beforeUp
                                      << ",after_down=" << afterDown
                                      << ",after_neutral=" << afterNeutral
                                      << ",after_up=" << afterUp
                                      << ",delta_down=" << deltaDown
                                      << ",delta_neutral=" << deltaNeutral
                                      << ",delta_up=" << deltaUp
                                      << ",delta_down_minus_neutral=" << (deltaDown - deltaNeutral)
                                      << ",delta_up_minus_neutral=" << (deltaUp - deltaNeutral)
                                      << ",after_down_minus_neutral=" << (afterDown - afterNeutral)
                                      << ",after_up_minus_neutral=" << (afterUp - afterNeutral)
                                      << ",after_neutral_minus_mean_edges=" << (afterNeutral - 0.5 * (afterDown + afterUp))
                                      << std::endl;
                        }
                        // --- END PATCH ---
#endif
                    }
                    }
                    if (phase3PostUpdateDiagEnabled && s_phase3LastHGeometryValidBeforeUpdate)
                    {
                        // --- BEGIN PATCH: EPOCH CHECKPOINT HIDDEN GEOMETRY RECOMPUTE ---
                        const bool phase3EpochCheckpointOnly =
                            phase3HiddenGeometryCheckpointDiag && !phase3ClipFullDiagEnabled;

                        const bool phase3CanReplayHiddenBatch =
                            s_phase3HiddenReplayCapture.valid &&
                            s_phase3HiddenReplayCapture.actualClasses.size() == s_phase3HiddenReplayCapture.rows &&
                            s_phase3HiddenReplayCapture.hBefore.rows == s_phase3HiddenReplayCapture.rows &&
                            s_phase3HiddenReplayCapture.hBefore.cols > 0 &&
                            s_phase3HiddenReplayCapture.hBefore.cols == hidden_size &&
                            s_phase3HiddenReplayCapture.startIndices.size() == s_phase3HiddenReplayCapture.rows &&
                            s_phase3HiddenReplayCapture.replayTimeSteps > 0 &&
                            s_phase3HiddenReplayCapture.replayInputCols == static_cast<size_t>(n_in) &&
                            s_phase3HiddenReplayCapture.replayInputs.size() ==
                                s_phase3HiddenReplayCapture.replayTimeSteps *
                                s_phase3HiddenReplayCapture.rows *
                                s_phase3HiddenReplayCapture.replayInputCols;
                        
                        if (phase3EpochCheckpointOnly)
                        {
                            if (phase3CanReplayHiddenBatch)
                            {
                                EAMatrix replayH(s_phase3HiddenReplayCapture.rows, hidden_size);
                                EAMatrix replayC(s_phase3HiddenReplayCapture.rows, hidden_size);
                                EAMatrix replayConcat(
                                    s_phase3HiddenReplayCapture.rows,
                                    static_cast<size_t>(n_in + hidden_size));

                                zeroFill(replayH);
                                zeroFill(replayC);

                                auto wwReplay = hoistWindowWeights();

                                for (size_t replayT = 0;
                                     replayT < s_phase3HiddenReplayCapture.replayTimeSteps;
                                     ++replayT)
                                {
                                    EAMatrix replayX(
                                        s_phase3HiddenReplayCapture.rows,
                                        s_phase3HiddenReplayCapture.replayInputCols);

                                    auto replayLowX = MetaNN::LowerAccess(replayX);
                                    float* replayDst = replayLowX.MutableRawMemory();

                                    const size_t replayOffset =
                                        replayT *
                                        s_phase3HiddenReplayCapture.rows *
                                        s_phase3HiddenReplayCapture.replayInputCols;

                                    const size_t replayCount =
                                        s_phase3HiddenReplayCapture.rows *
                                        s_phase3HiddenReplayCapture.replayInputCols;

                                    std::memcpy(
                                        replayDst,
                                        s_phase3HiddenReplayCapture.replayInputs.data() + replayOffset,
                                        replayCount * sizeof(float));

                                    ForwardBatchScratch replayScratch;
                                    forwardStepBatch(
                                        replayX,
                                        wwReplay,
                                        bias,
                                        replayH,
                                        replayC,
                                        replayConcat,
                                        replayScratch,
                                        &profile);
                                }

                                s_phase3LastHGeometryHostBeforeUpdate =
                                    Phase3MaterializeHost(replayH);
                            }
                        }
                        // --- END PATCH ---
                        const double hBeforeNorm = [&]() -> double
                        {
                            long double ss = 0.0L;
                            for (float v : s_phase3LastHGeometryHostBeforeUpdate.data)
                            {
                                const long double d = static_cast<long double>(v);
                                ss += d * d;
                            }
                            return std::sqrt(static_cast<double>(ss));
                        }();

                        // --- BEGIN PATCH: DIAG HIDDEN REPRESENTATION COLLAPSE ---
                        {
                            const size_t hRows = s_phase3LastHGeometryHostBeforeUpdate.rows;
                            const size_t hCols = s_phase3LastHGeometryHostBeforeUpdate.cols;
                            const bool hasClassIds =
                                s_phase3HiddenReplayCapture.valid &&
                                s_phase3HiddenReplayCapture.actualClasses.size() == hRows;

                            std::array<size_t, direction_output_size> classCount {0, 0, 0};
                            std::array<std::vector<long double>, direction_output_size> classSum;
                            std::array<long double, direction_output_size> classSqSum {0.0L, 0.0L, 0.0L};
                            for (size_t cls = 0; cls < direction_output_size; ++cls)
                                classSum[cls].assign(hCols, 0.0L);

                            long double totalSq = 0.0L;
                            long double rowNormSum = 0.0L;
                            long double rowNormSqSum = 0.0L;
                            double rowNormMin = std::numeric_limits<double>::max();                            double rowNormMax = 0.0;
                            size_t rowNormFinite = 0;
                            size_t rowNormTiny = 0;
                            size_t nonFiniteH = 0;

                            for (size_t r = 0; r < hRows; ++r)
                            {
                                long double rowSq = 0.0L;
                                bool rowFinite = true;
                                for (size_t c = 0; c < hCols; ++c)
                                {
                                    const float hv = s_phase3LastHGeometryHostBeforeUpdate.data[r * hCols + c];
                                    if (!std::isfinite(static_cast<double>(hv)))
                                    {
                                        rowFinite = false;
                                        ++nonFiniteH;
                                        continue;
                                    }

                                    const long double d = static_cast<long double>(hv);
                                    rowSq += d * d;
                                    totalSq += d * d;
                                }

                                if (!rowFinite)
                                    continue;

                                const double rowNorm = std::sqrt(static_cast<double>(rowSq));
                                rowNormMin = std::min(rowNormMin, rowNorm);
                                rowNormMax = std::max(rowNormMax, rowNorm);
                                rowNormSum += static_cast<long double>(rowNorm);
                                rowNormSqSum += static_cast<long double>(rowNorm) * static_cast<long double>(rowNorm);
                                ++rowNormFinite;
                                if (rowNorm < 1.0e-6)
                                    ++rowNormTiny;

                                if (hasClassIds)
                                {
                                    const size_t cls = static_cast<size_t>(s_phase3HiddenReplayCapture.actualClasses[r]);
                                    if (cls < direction_output_size)
                                    {
                                        ++classCount[cls];
                                        classSqSum[cls] += rowSq;
                                        for (size_t c = 0; c < hCols; ++c)
                                            classSum[cls][c] += static_cast<long double>(s_phase3LastHGeometryHostBeforeUpdate.data[r * hCols + c]);
                                    }
                                }
                            }

                            auto classMeanNorm = [&](size_t cls) -> double
                            {
                                if (!hasClassIds || classCount[cls] == 0)
                                    return 0.0;

                                long double ss = 0.0L;
                                for (size_t c = 0; c < hCols; ++c)
                                {
                                    const long double m = classSum[cls][c] / static_cast<long double>(classCount[cls]);
                                    ss += m * m;
                                }
                                return std::sqrt(static_cast<double>(ss));
                            };

                            auto classWithinRms = [&](size_t cls) -> double
                            {
                                if (!hasClassIds || classCount[cls] == 0)
                                    return 0.0;

                                const double meanNorm = classMeanNorm(cls);
                                const long double meanSq = static_cast<long double>(meanNorm) * static_cast<long double>(meanNorm);
                                const long double avgSq = classSqSum[cls] / static_cast<long double>(classCount[cls]);
                                return std::sqrt(static_cast<double>(std::max(0.0L, avgSq - meanSq)));
                            };

                            auto centroidDistance = [&](size_t a, size_t b) -> double
                            {
                                if (!hasClassIds || classCount[a] == 0 || classCount[b] == 0)
                                    return 0.0;

                                long double ss = 0.0L;
                                for (size_t c = 0; c < hCols; ++c)
                                {
                                    const long double ma = classSum[a][c] / static_cast<long double>(classCount[a]);
                                    const long double mb = classSum[b][c] / static_cast<long double>(classCount[b]);
                                    const long double d = ma - mb;
                                    ss += d * d;
                                }
                                return std::sqrt(static_cast<double>(ss));
                            };

                            auto centroidCosine = [&](size_t a, size_t b) -> double
                            {
                                if (!hasClassIds || classCount[a] == 0 || classCount[b] == 0)
                                    return 0.0;

                                long double dot = 0.0L;
                                long double aa = 0.0L;
                                long double bb = 0.0L;
                                for (size_t c = 0; c < hCols; ++c)
                                {
                                    const long double ma = classSum[a][c] / static_cast<long double>(classCount[a]);
                                    const long double mb = classSum[b][c] / static_cast<long double>(classCount[b]);
                                    dot += ma * mb;
                                    aa += ma * ma;
                                    bb += mb * mb;
                                }

                                const long double denom = std::sqrt(aa) * std::sqrt(bb);
                                return denom > 1.0e-18L ? static_cast<double>(dot / denom) : 0.0;
                            };

                            const double totalRms = (hRows * hCols) > 0
                                ? std::sqrt(static_cast<double>(totalSq / static_cast<long double>(hRows * hCols)))
                                : 0.0;
                            const double rowNormMean = rowNormFinite > 0
                                ? static_cast<double>(rowNormSum / static_cast<long double>(rowNormFinite))
                                : 0.0;
                            const double rowNormStd = rowNormFinite > 0
                                ? std::sqrt(std::max(0.0,
                                      static_cast<double>(rowNormSqSum / static_cast<long double>(rowNormFinite)) -
                                      rowNormMean * rowNormMean))
                                : 0.0;
                            if (rowNormFinite == 0) rowNormMin = 0.0;
                            
                            const double downNeutralDist = centroidDistance(0, 1);
                            const double neutralUpDist = centroidDistance(1, 2);
                            const double downUpDist = centroidDistance(0, 2);
                            const double meanCentroidDist = (downNeutralDist + neutralUpDist + downUpDist) / 3.0;
                            const double downWithin = classWithinRms(0);
                            const double neutralWithin = classWithinRms(1);
                            const double upWithin = classWithinRms(2);
                            const double meanWithin = (downWithin + neutralWithin + upWithin) / 3.0;
                            auto phase3HiddenClassName = [](size_t cls) -> const char*  {   return cls == 0 ? "down" : cls == 1 ? "neutral" : "up"; };

                            for (size_t cls = 0; cls < direction_output_size; ++cls)
                                std::cout << "DIAG_HIDDEN_CLASS_STATS"
                                          << ",call=" << phase3ClipFullDiagIdx
                                          << ",epoch=" << (epochIdx + 1)
                                          << ",epoch_checkpoint=" << (phase3ClipFullDiagEnabled ? 0 : 1)
                                          << ",actual_class=" << phase3HiddenClassName(cls)
                                          << ",class_id=" << cls
                                          << ",count=" << classCount[cls]
                                          << ",hidden_size=" << hCols
                                          << ",centroid_norm=" << classMeanNorm(cls)
                                          << ",within_rms=" << classWithinRms(cls)
                                          << std::endl;

                            std::cout << "DIAG_HIDDEN_CLASS_COSINE"
                                      << ",call=" << phase3ClipFullDiagIdx
                                        << ",epoch=" << (epochIdx + 1)
                                      << ",epoch_checkpoint=" << (phase3ClipFullDiagEnabled ? 0 : 1)
                                      << ",hidden_size=" << hCols
                                      << ",count_down=" << classCount[0]
                                      << ",count_neutral=" << classCount[1]
                                      << ",count_up=" << classCount[2]
                                      << ",down_neutral=" << centroidCosine(0, 1)
                                      << ",neutral_up=" << centroidCosine(1, 2)
                                      << ",down_up=" << centroidCosine(0, 2)
                                      << std::endl;
                            
                            std::cout << "DIAG_HIDDEN_REP_GEOMETRY_"
                                      << ",call=" << phase3ClipFullDiagIdx
                                        << ",epoch=" << (epochIdx + 1)
                                      << ",epoch_checkpoint=" << (phase3ClipFullDiagEnabled ? 0 : 1)
                                      << ",rows=" << hRows
                                      << ",cols=" << hCols
                                      << ",has_class_ids=" << (hasClassIds ? 1 : 0)
                                      << ",nonfinite_h=" << nonFiniteH
                                      << ",h_fro_norm=" << hBeforeNorm
                                      << ",h_total_rms=" << totalRms
                                      << ",row_norm_mean=" << rowNormMean
                                      << ",row_norm_std=" << rowNormStd
                                      << ",row_norm_min=" << rowNormMin
                                      << ",row_norm_max=" << rowNormMax
                                      << ",row_norm_tiny_lt_1e_minus_6=" << rowNormTiny
                                      << ",count_down=" << classCount[0]
                                      << ",count_neutral=" << classCount[1]
                                      << ",count_up=" << classCount[2]
                                      << ",centroid_norm_down=" << classMeanNorm(0)
                                      << ",centroid_norm_neutral=" << classMeanNorm(1)
                                      << ",centroid_norm_up=" << classMeanNorm(2)
                                      << ",within_rms_down=" << downWithin
                                      << ",within_rms_neutral=" << neutralWithin
                                      << ",within_rms_up=" << upWithin
                                      << ",centroid_dist_down_neutral=" << downNeutralDist
                                      << ",centroid_dist_neutral_up=" << neutralUpDist
                                      << ",centroid_dist_down_up=" << downUpDist
                                      << ",centroid_cos_down_neutral=" << centroidCosine(0, 1)
                                      << ",centroid_cos_neutral_up=" << centroidCosine(1, 2)
                                      << ",centroid_cos_down_up=" << centroidCosine(0, 2)
                                      << ",mean_centroid_dist=" << meanCentroidDist
                                      << ",mean_within_rms=" << meanWithin
                                      << ",separation_to_within_ratio=" << (meanWithin > 1.0e-12 ? meanCentroidDist / meanWithin : 0.0)
                                      << std::endl;
                        }
                        // --- END PATCH ---

                        std::cout << "DIAG_H_UPDATE_DELTA_"
                                  << ",call=" << phase3ClipFullDiagIdx
                                  << ",mode=captured_pre_update_hidden_geometry"
                                  << ",hidden_rows=" << s_phase3LastHGeometryHostBeforeUpdate.rows
                                  << ",hidden_cols=" << s_phase3LastHGeometryHostBeforeUpdate.cols
                                  << ",actual_down=" << s_phase3LastHGeometryActualHistBeforeUpdate[0]
                                  << ",actual_neutral=" << s_phase3LastHGeometryActualHistBeforeUpdate[1]
                                  << ",actual_up=" << s_phase3LastHGeometryActualHistBeforeUpdate[2]
                                  << ",h_before_norm=" << hBeforeNorm
                                  << ",core_param_update_norm=" << (static_cast<double>(lrCore) * FroNormEvalHost(d_param_f))
                                  << ",core_bias_update_norm=" << (static_cast<double>(lrCore) * FroNormEvalHost(d_bias_f))
                                  << ",head_weight_update_norm=" << FroNormDeltaHost(returnHeadDirWeight, phase3HeadWUpdateBefore)
                                  << ",head_bias_update_norm=" << FroNormDeltaHost(returnHeadDirBias, phase3HeadBUpdateBefore)
                                  << ",requires_forward_recompute_for_true_h_after=1"
                                  << std::endl;
                        const double hBeforeNormForRecompute =
                            Phase3HostMatrixNorm(s_phase3LastHGeometryHostBeforeUpdate);

                        if (!phase3CanReplayHiddenBatch)
                        {
                            std::cout << "DIAG_H_RECOMPUTE_DELTA_"
                                      << ",call=" << phase3ClipFullDiagIdx
                                      << ",status=replay_unavailable"
                                      << ",reason=must_capture_same_window_batch_inputs_and_initial_state_before_update"
                                      << ",hidden_rows=" << s_phase3LastHGeometryHostBeforeUpdate.rows
                                      << ",hidden_cols=" << s_phase3LastHGeometryHostBeforeUpdate.cols
                                      << ",actual_down=" << s_phase3LastHGeometryActualHistBeforeUpdate[0]
                                      << ",actual_neutral=" << s_phase3LastHGeometryActualHistBeforeUpdate[1]
                                      << ",actual_up=" << s_phase3LastHGeometryActualHistBeforeUpdate[2]
                                      << ",h_before_norm=" << hBeforeNormForRecompute
                                      << ",h_after_norm=0"
                                      << ",h_delta_norm=0"
                                      << ",h_delta_ratio=0"
                                      << ",needs_captured_window_replay=1"
                                      << std::endl;
                        }
                        else
                        {
                            EAMatrix replayH(s_phase3HiddenReplayCapture.rows, hidden_size);
                            EAMatrix replayC(s_phase3HiddenReplayCapture.rows, hidden_size);
                            EAMatrix replayConcat(
                                s_phase3HiddenReplayCapture.rows,
                                static_cast<size_t>(n_in + hidden_size));

                            zeroFill(replayH);
                            zeroFill(replayC);

                            auto wwReplay = hoistWindowWeights();
                            bool replayOk = true;

                            for (size_t replayT = 0;
                                 replayT < s_phase3HiddenReplayCapture.replayTimeSteps;
                                 ++replayT)
                            {
                                EAMatrix replayX(
                                    s_phase3HiddenReplayCapture.rows,
                                    s_phase3HiddenReplayCapture.replayInputCols);

                                auto replayLowX = MetaNN::LowerAccess(replayX);
                                float* replayDst = replayLowX.MutableRawMemory();

                                const size_t replayOffset =
                                    replayT *
                                    s_phase3HiddenReplayCapture.rows *
                                    s_phase3HiddenReplayCapture.replayInputCols;

                                const size_t replayCount =
                                    s_phase3HiddenReplayCapture.rows *
                                    s_phase3HiddenReplayCapture.replayInputCols;

                                if (replayOffset + replayCount >
                                    s_phase3HiddenReplayCapture.replayInputs.size())
                                {
                                    replayOk = false;
                                    break;
                                }

                                std::memcpy(
                                    replayDst,
                                    s_phase3HiddenReplayCapture.replayInputs.data() + replayOffset,
                                    replayCount * sizeof(float));

                                ForwardBatchScratch replayScratch;
                                forwardStepBatch(
                                    replayX,
                                    wwReplay,
                                    bias,
                                    replayH,
                                    replayC,
                                    replayConcat,
                                    replayScratch,
                                    &profile);
                            }

                            Phase3HostMatrix replayAfterHost;
                            if (replayOk)
                            {
                                replayAfterHost = Phase3MaterializeHost(replayH);
                            }

                            const double hAfterNorm =
                                replayOk ? Phase3HostMatrixNorm(replayAfterHost) : 0.0;
                                                        
                            const double hDeltaNorm =
                                (replayOk &&
                                 Phase3HostMatrixSameShape(
                                     s_phase3HiddenReplayCapture.hBefore,
                                     replayAfterHost))
                                    ? Phase3HostMatrixDeltaNorm(
                                          s_phase3HiddenReplayCapture.hBefore,
                                          replayAfterHost)
                                    : 0.0;

                            const double hDeltaRatio =
                                hBeforeNormForRecompute > 1.0e-12
                                    ? hDeltaNorm / hBeforeNormForRecompute
                                    : 0.0;

                            auto phase3ReplayCentroidDistance =
                            [&](const Phase3HostMatrix& hmat, size_t classA, size_t classB) -> double
                            {
                                if (hmat.rows == 0 || hmat.cols != hidden_size)
                                    return 0.0;
                                if (s_phase3HiddenReplayCapture.actualClasses.size() != hmat.rows)
                                    return 0.0;
                                if (s_phase3HiddenReplayCapture.actualHist[classA] == 0 ||
                                    s_phase3HiddenReplayCapture.actualHist[classB] == 0)
                                    return 0.0;

                                std::vector<double> meanA(hidden_size, 0.0);
                                std::vector<double> meanB(hidden_size, 0.0);

                                for (size_t row = 0; row < hmat.rows; ++row)
                                {
                                    const int cls = s_phase3HiddenReplayCapture.actualClasses[row];
                                    if (cls != static_cast<int>(classA) && cls != static_cast<int>(classB))
                                        continue;

                                    std::vector<double>& mean =
                                        (cls == static_cast<int>(classA)) ? meanA : meanB;
                                    for (size_t h = 0; h < hidden_size; ++h)
                                    {
                                        mean[h] += static_cast<double>(hmat.data[row * hmat.cols + h]);
                                    }
                                }

                                const double denomA = static_cast<double>(s_phase3HiddenReplayCapture.actualHist[classA]);
                                const double denomB = static_cast<double>(s_phase3HiddenReplayCapture.actualHist[classB]);

                                double distSq = 0.0;
                                for (size_t h = 0; h < hidden_size; ++h)
                                {
                                    const double a = meanA[h] / denomA;
                                    const double b = meanB[h] / denomB;
                                    const double d = a - b;
                                    distSq += d * d;
                                }

                                return std::sqrt(distSq);
                            };

                            const double beforeDownNeutral = replayOk
                                ? phase3ReplayCentroidDistance(s_phase3HiddenReplayCapture.hBefore, 0, 1)
                                : 0.0;
                            const double beforeDownUp = replayOk
                                ? phase3ReplayCentroidDistance(s_phase3HiddenReplayCapture.hBefore, 0, 2)
                                : 0.0;
                            const double beforeNeutralUp = replayOk
                                ? phase3ReplayCentroidDistance(s_phase3HiddenReplayCapture.hBefore, 1, 2)
                                : 0.0;
                            const double afterDownNeutral = replayOk
                                ? phase3ReplayCentroidDistance(replayAfterHost, 0, 1)
                                : 0.0;
                            const double afterDownUp = replayOk
                                ? phase3ReplayCentroidDistance(replayAfterHost, 0, 2)
                                : 0.0;
                            const double afterNeutralUp = replayOk
                                ? phase3ReplayCentroidDistance(replayAfterHost, 1, 2)
                                : 0.0;

                            std::cout << "DIAG_H_RECOMPUTE_DELTA_"
                                      << ",call=" << phase3ClipFullDiagIdx
                                    << ",status=" << (replayOk ? "recomputed" : "replay_helper_compile_disabled")
                            << ",reason=" << (replayOk
                                  ? "same_batch_forward_replay_after_update"
                                  : "forward_replay_helper_disabled_pending_signature_match")
                            << ",hidden_rows=" << s_phase3HiddenReplayCapture.hBefore.rows
                                      << ",hidden_cols=" << s_phase3HiddenReplayCapture.hBefore.cols
                                      << ",actual_down=" << s_phase3HiddenReplayCapture.actualHist[0]
                                      << ",actual_neutral=" << s_phase3HiddenReplayCapture.actualHist[1]
                                      << ",actual_up=" << s_phase3HiddenReplayCapture.actualHist[2]
                                      << ",h_before_norm=" << hBeforeNormForRecompute
                                      << ",h_after_norm=" << hAfterNorm
                                      << ",h_delta_norm=" << hDeltaNorm
                                      << ",h_delta_ratio=" << hDeltaRatio
                                      << ",capture_valid=1"
                                      << ",capture_rows=" << s_phase3HiddenReplayCapture.rows
                                      << ",capture_start_count=" << s_phase3HiddenReplayCapture.startIndices.size()
                                      << ",capture_batchBase=" << s_phase3HiddenReplayCapture.batchBase
                                      << ",capture_windowCount=" << s_phase3HiddenReplayCapture.windowCountAtCapture
                                      << ",replay_time_steps=" << s_phase3HiddenReplayCapture.replayTimeSteps
                                      << ",replay_input_cols=" << s_phase3HiddenReplayCapture.replayInputCols
                                      << ",replay_input_count=" << s_phase3HiddenReplayCapture.replayInputs.size()
                                      << std::endl;
                            if (replayOk)
                            {
                                std::cout << "DIAG_H_RECOMPUTE_SEPARATION_DELTA_"
                                          << ",call=" << phase3ClipFullDiagIdx
                                          << ",hidden_rows=" << s_phase3HiddenReplayCapture.hBefore.rows
                                          << ",hidden_cols=" << s_phase3HiddenReplayCapture.hBefore.cols
                                          << ",actual_down=" << s_phase3HiddenReplayCapture.actualHist[0]
                                          << ",actual_neutral=" << s_phase3HiddenReplayCapture.actualHist[1]
                                          << ",actual_up=" << s_phase3HiddenReplayCapture.actualHist[2]
                                          << ",h_delta_ratio=" << hDeltaRatio
                                          << ",down_neutral_before=" << beforeDownNeutral
                                          << ",down_neutral_after=" << afterDownNeutral
                                          << ",down_neutral_delta=" << (afterDownNeutral - beforeDownNeutral)
                                          << ",down_up_before=" << beforeDownUp
                                          << ",down_up_after=" << afterDownUp
                                          << ",down_up_delta=" << (afterDownUp - beforeDownUp)
                                          << ",neutral_up_before=" << beforeNeutralUp
                                          << ",neutral_up_after=" << afterNeutralUp
                                          << ",neutral_up_delta=" << (afterNeutralUp - beforeNeutralUp)
                                          << std::endl;
                            }
                        }
                    }
                    if (phase3HeadDeltaCaptured)
                    {
                        const EAMatrix logitsAfter = Phase3DirHeadLogitsCpu(phase3HeadDeltaH, returnHeadDirWeight, returnHeadDirBias);
                        const EAMatrix probsAfter = Phase3SoftmaxProbsCpu(logitsAfter);
                        PrintPhase3LogitDelta(phase3ClipFullDiagIdx, phase3HeadDeltaLogitsBefore, logitsAfter,
                                              phase3HeadDeltaProbsBefore, probsAfter);

                        // --- BEGIN PATCH: DIAG CLASS TARGET PROB/ERROR BY ACTUAL CLASS ---
                        {
                            auto probsLowTargetDiag = MetaNN::LowerAccess(probsAfter);
                            const float* probsTargetDiagPtr = probsLowTargetDiag.RawMemory();
                            const size_t targetDiagRows = probsAfter.Shape()[0];
                            const size_t targetDiagCols = probsAfter.Shape()[1];

                            const std::vector<int>* targetActualClasses = nullptr;
                            const char* targetActualSource = "none";

                            if (phase3HeadDeltaActualClasses.size() == targetDiagRows)
                            {
                                targetActualClasses = &phase3HeadDeltaActualClasses;
                                targetActualSource = "phase3HeadDeltaActualClasses";
                            }
                            else if (s_phase3HiddenReplayCapture.actualClasses.size() == targetDiagRows)
                            {
                                targetActualClasses = &s_phase3HiddenReplayCapture.actualClasses;
                                targetActualSource = "s_phase3HiddenReplayCapture.actualClasses";
                            }

                            if (targetDiagCols >= direction_output_size && targetActualClasses != nullptr)
                            {
                                std::array<long double, direction_output_size> trueProbSum {0.0L, 0.0L, 0.0L};
                                std::array<long double, direction_output_size> trueErrSum {0.0L, 0.0L, 0.0L};
                                std::array<size_t, direction_output_size> trueCount {0, 0, 0};
                                long double allTrueProbSum = 0.0L;
                                long double allTrueErrSum = 0.0L;
                                size_t allTrueCount = 0;

                                for (size_t r = 0; r < targetDiagRows; ++r)
                                {
                                    const int actualClass = (*targetActualClasses)[r];
                                    if (actualClass < 0 || actualClass >= static_cast<int>(direction_output_size))
                                        continue;

                                    const size_t cls = static_cast<size_t>(actualClass);
                                    const double pTrue = static_cast<double>(probsTargetDiagPtr[r * targetDiagCols + cls]);
                                    const double errTrue = pTrue - 1.0;

                                    trueProbSum[cls] += static_cast<long double>(pTrue);
                                    trueErrSum[cls] += static_cast<long double>(errTrue);
                                    ++trueCount[cls];
                                    allTrueProbSum += static_cast<long double>(pTrue);
                                    allTrueErrSum += static_cast<long double>(errTrue);
                                    ++allTrueCount;
                                }

                                auto targetMean = [&](const std::array<long double, direction_output_size>& values, size_t cls) -> double
                                {
                                    return trueCount[cls] > 0
                                        ? static_cast<double>(values[cls] / static_cast<long double>(trueCount[cls]))
                                        : 0.0;
                                };

                                const double trueClassProbMean = allTrueCount > 0
                                    ? static_cast<double>(allTrueProbSum / static_cast<long double>(allTrueCount))
                                    : 0.0;
                                const double trueClassErrMean = allTrueCount > 0
                                    ? static_cast<double>(allTrueErrSum / static_cast<long double>(allTrueCount))
                                    : 0.0;

                                std::cout << "DIAG_CLASS_TARGET_PROB_"
                                          << ",call=" << phase3ClipFullDiagIdx
                                          << ",rows=" << targetDiagRows
                                          << ",cols=" << targetDiagCols
                                          << ",actual_source=" << targetActualSource
                                          << ",actual_down_count=" << trueCount[0]
                                          << ",actual_neutral_count=" << trueCount[1]
                                          << ",actual_up_count=" << trueCount[2]
                                          << ",actual_down_prob_mean=" << targetMean(trueProbSum, 0)
                                          << ",actual_neutral_prob_mean=" << targetMean(trueProbSum, 1)
                                          << ",actual_up_prob_mean=" << targetMean(trueProbSum, 2)
                                          << ",true_class_prob_mean=" << trueClassProbMean
                                          << std::endl;

                                std::cout << "DIAG_CLASS_ERROR_BY_TARGET_"
                                          << ",call=" << phase3ClipFullDiagIdx
                                          << ",rows=" << targetDiagRows
                                          << ",cols=" << targetDiagCols
                                          << ",actual_source=" << targetActualSource
                                          << ",actual_down_count=" << trueCount[0]
                                          << ",actual_neutral_count=" << trueCount[1]
                                          << ",actual_up_count=" << trueCount[2]
                                          << ",err_true_class_mean=" << trueClassErrMean
                                          << ",err_down_when_actual_down=" << targetMean(trueErrSum, 0)
                                          << ",err_neutral_when_actual_neutral=" << targetMean(trueErrSum, 1)
                                          << ",err_up_when_actual_up=" << targetMean(trueErrSum, 2)
                                          << std::endl;
                            }
                            else
                            {
                                std::cout << "DIAG_CLASS_TARGET_PROB_"
                                          << ",call=" << phase3ClipFullDiagIdx
                                          << ",status=skipped"
                                          << ",reason=missing_or_mismatched_actual_classes"
                                          << ",rows=" << targetDiagRows
                                          << ",cols=" << targetDiagCols
                                          << ",phase3_head_delta_actual_class_count=" << phase3HeadDeltaActualClasses.size()
                                          << ",replay_actual_class_count=" << s_phase3HiddenReplayCapture.actualClasses.size()
                                          << std::endl;

                                std::cout << "DIAG_CLASS_ERROR_BY_TARGET_"
                                          << ",call=" << phase3ClipFullDiagIdx
                                          << ",status=skipped"
                                          << ",reason=missing_or_mismatched_actual_classes"
                                          << ",rows=" << targetDiagRows
                                          << ",cols=" << targetDiagCols
                                          << ",phase3_head_delta_actual_class_count=" << phase3HeadDeltaActualClasses.size()
                                          << ",replay_actual_class_count=" << s_phase3HiddenReplayCapture.actualClasses.size()
                                          << std::endl;
                            }
                        }
                        // --- END PATCH ---
                        // --- BEGIN PATCH: DIAG CLASS LOGIT/PROB/MARGIN STATS ---
                        {
                            // Logit stats
                            auto logitsLow = MetaNN::LowerAccess(logitsAfter);
                            const float* logitsPtr = logitsLow.RawMemory();
                            const size_t logitRows = logitsAfter.Shape()[0];
                            const size_t logitCols = logitsAfter.Shape()[1];
                            if (logitCols >= direction_output_size) {
                                std::array<long double, direction_output_size> sum {0.0L, 0.0L, 0.0L};
                                std::array<long double, direction_output_size> sqsum {0.0L, 0.0L, 0.0L};
                                std::array<double, direction_output_size> minv {
                                    std::numeric_limits<double>::max(),
                                    std::numeric_limits<double>::max(),
                                    std::numeric_limits<double>::max()};
                                std::array<double, direction_output_size> maxv {
                                    std::numeric_limits<double>::lowest(),
                                    std::numeric_limits<double>::lowest(),
                                    std::numeric_limits<double>::lowest()};
                                for (size_t r = 0; r < logitRows; ++r) {
                                    for (size_t cls = 0; cls < direction_output_size; ++cls) {
                                        double v = static_cast<double>(logitsPtr[r * logitCols + cls]);
                                        sum[cls] += static_cast<long double>(v);
                                        sqsum[cls] += static_cast<long double>(v) * static_cast<long double>(v);
                                        minv[cls] = std::min(minv[cls], v);
                                        maxv[cls] = std::max(maxv[cls], v);
                                    }
                                }
                                auto mean = [&](size_t cls) -> double {
                                    return logitRows > 0 ? static_cast<double>(sum[cls] / static_cast<long double>(logitRows)) : 0.0;
                                };
                                auto stddev = [&](size_t cls) -> double {
                                    if (logitRows == 0) return 0.0;
                                    double m = mean(cls);
                                    double var = static_cast<double>(sqsum[cls] / static_cast<long double>(logitRows)) - m * m;
                                    return std::sqrt(std::max(0.0, var));
                                };
                                std::cout << "DIAG_CLASS_LOGIT_STATS_"
                                          << ",call=" << phase3ClipFullDiagIdx
                                          << ",rows=" << logitRows
                                          << ",cols=" << logitCols
                                          << ",logit_down_mean=" << mean(0)
                                          << ",logit_neutral_mean=" << mean(1)
                                          << ",logit_up_mean=" << mean(2)
                                          << ",logit_down_std=" << stddev(0)
                                          << ",logit_neutral_std=" << stddev(1)
                                          << ",logit_up_std=" << stddev(2)
                                          << ",logit_down_min=" << minv[0]
                                          << ",logit_neutral_min=" << minv[1]
                                          << ",logit_up_min=" << minv[2]
                                          << ",logit_down_max=" << maxv[0]
                                          << ",logit_neutral_max=" << maxv[1]
                                          << ",logit_up_max=" << maxv[2]
                                          << std::endl;
                            }
                            // Prob stats
                            auto probsLow = MetaNN::LowerAccess(probsAfter);
                            const float* probsPtr = probsLow.RawMemory();
                            const size_t probRows = probsAfter.Shape()[0];
                            const size_t probCols = probsAfter.Shape()[1];
                            if (probCols >= direction_output_size) {
                                std::array<long double, direction_output_size> sum {0.0L, 0.0L, 0.0L};
                                std::array<long double, direction_output_size> sqsum {0.0L, 0.0L, 0.0L};
                                std::array<double, direction_output_size> minv {
                                    std::numeric_limits<double>::max(),
                                    std::numeric_limits<double>::max(),
                                    std::numeric_limits<double>::max()};
                                std::array<double, direction_output_size> maxv {
                                    std::numeric_limits<double>::lowest(),
                                    std::numeric_limits<double>::lowest(),
                                    std::numeric_limits<double>::lowest()};
                                for (size_t r = 0; r < probRows; ++r) {
                                    for (size_t cls = 0; cls < direction_output_size; ++cls) {
                                        double v = static_cast<double>(probsPtr[r * probCols + cls]);
                                        sum[cls] += static_cast<long double>(v);
                                        sqsum[cls] += static_cast<long double>(v) * static_cast<long double>(v);
                                        minv[cls] = std::min(minv[cls], v);
                                        maxv[cls] = std::max(maxv[cls], v);
                                    }
                                }
                                auto mean = [&](size_t cls) -> double {
                                    return probRows > 0 ? static_cast<double>(sum[cls] / static_cast<long double>(probRows)) : 0.0;
                                };
                                auto stddev = [&](size_t cls) -> double {
                                    if (probRows == 0) return 0.0;
                                    double m = mean(cls);
                                    double var = static_cast<double>(sqsum[cls] / static_cast<long double>(probRows)) - m * m;
                                    return std::sqrt(std::max(0.0, var));
                                };
                                std::cout << "DIAG_CLASS_PROB_STATS_"
                                          << ",call=" << phase3ClipFullDiagIdx
                                          << ",rows=" << probRows
                                          << ",cols=" << probCols
                                          << ",prob_down_mean=" << mean(0)
                                          << ",prob_neutral_mean=" << mean(1)
                                          << ",prob_up_mean=" << mean(2)
                                          << ",prob_down_std=" << stddev(0)
                                          << ",prob_neutral_std=" << stddev(1)
                                          << ",prob_up_std=" << stddev(2)
                                          << ",prob_down_min=" << minv[0]
                                          << ",prob_neutral_min=" << minv[1]
                                          << ",prob_up_min=" << minv[2]
                                          << ",prob_down_max=" << maxv[0]
                                          << ",prob_neutral_max=" << maxv[1]
                                          << ",prob_up_max=" << maxv[2]
                                          << std::endl;
                            }
                            // Margin stats
                            if (probCols >= direction_output_size) {
                                long double marginSum = 0.0L;
                                long double marginSqSum = 0.0L;
                                double marginMin = std::numeric_limits<double>::max();
                                double marginMax = std::numeric_limits<double>::lowest();
                                size_t predDown = 0, predNeutral = 0, predUp = 0;
                                for (size_t r = 0; r < probRows; ++r) {
                                    // Find top two probs and argmax
                                    double best = -1.0, second = -1.0;
                                    size_t bestIdx = 0;
                                    for (size_t cls = 0; cls < direction_output_size; ++cls) {
                                        double v = static_cast<double>(probsPtr[r * probCols + cls]);
                                        if (v > best) {
                                            second = best;
                                            best = v;
                                            bestIdx = cls;
                                        } else if (v > second) {
                                            second = v;
                                        }
                                    }
                                    double margin = best - second;
                                    marginSum += static_cast<long double>(margin);
                                    marginSqSum += static_cast<long double>(margin) * static_cast<long double>(margin);
                                    marginMin = std::min(marginMin, margin);
                                    marginMax = std::max(marginMax, margin);
                                    if (bestIdx == 0) ++predDown;
                                    else if (bestIdx == 1) ++predNeutral;
                                    else if (bestIdx == 2) ++predUp;
                                }
                                double marginMean = probRows > 0 ? static_cast<double>(marginSum / static_cast<long double>(probRows)) : 0.0;
                                double marginStd = probRows > 0
                                    ? std::sqrt(std::max(0.0, static_cast<double>(marginSqSum / static_cast<long double>(probRows)) - marginMean * marginMean))
                                    : 0.0;
                                if (probRows == 0) { marginMin = 0.0; marginMax = 0.0; }
                                std::cout << "DIAG_CLASS_MARGIN_"
                                          << ",call=" << phase3ClipFullDiagIdx
                                          << ",rows=" << probRows
                                          << ",margin_mean=" << marginMean
                                          << ",margin_std=" << marginStd
                                          << ",margin_min=" << marginMin
                                          << ",margin_max=" << marginMax
                                          << ",pred_down=" << predDown
                                          << ",pred_neutral=" << predNeutral
                                          << ",pred_up=" << predUp
                                          << std::endl;
                            }
                        }
                        // --- END PATCH ---

                        auto phase3HDeltaLow = MetaNN::LowerAccess(phase3HeadDeltaH);
                        auto phase3WBeforeLow = MetaNN::LowerAccess(phase3HeadWUpdateBefore);
                        auto phase3WAfterLow = MetaNN::LowerAccess(returnHeadDirWeight);
                        auto phase3BBeforeLow = MetaNN::LowerAccess(phase3HeadBUpdateBefore);
                        auto phase3BAfterLow = MetaNN::LowerAccess(returnHeadDirBias);

                        const float* hPtr = phase3HDeltaLow.RawMemory();
                        const float* wBeforePtr = phase3WBeforeLow.RawMemory();
                        const float* wAfterPtr = phase3WAfterLow.RawMemory();
                        const float* bBeforePtr = phase3BBeforeLow.RawMemory();
                        const float* bAfterPtr = phase3BAfterLow.RawMemory();

                        const size_t hRows = phase3HeadDeltaH.Shape()[0];
                        const size_t hCols = phase3HeadDeltaH.Shape()[1];
                        const size_t wRows = returnHeadDirWeight.Shape()[0];
                        const size_t wCols = returnHeadDirWeight.Shape()[1];
                        const size_t bCols = returnHeadDirBias.Shape()[1];

                        std::array<long double, direction_output_size> meanWeightBefore {0.0L, 0.0L, 0.0L};
                        std::array<long double, direction_output_size> meanWeightAfter {0.0L, 0.0L, 0.0L};
                        std::array<long double, direction_output_size> meanWeightDelta {0.0L, 0.0L, 0.0L};
                        std::array<long double, direction_output_size> meanBiasDelta {0.0L, 0.0L, 0.0L};
                        std::array<long double, direction_output_size> meanTotalDelta {0.0L, 0.0L, 0.0L};
                        std::array<long double, direction_output_size> absWeightDeltaSum {0.0L, 0.0L, 0.0L};
                        std::array<long double, direction_output_size> absBiasDeltaSum {0.0L, 0.0L, 0.0L};
                        size_t rowCount = 0;

                        if (hCols == wRows && wCols >= direction_output_size && bCols >= direction_output_size)
                        {
                            for (size_t row = 0; row < hRows; ++row)
                            {
                                ++rowCount;
                                for (size_t cls = 0; cls < direction_output_size; ++cls)
                                {
                                    long double beforeNoBias = 0.0L;
                                    long double afterNoBias = 0.0L;
                                    for (size_t h = 0; h < hCols; ++h)
                                    {
                                        const long double hv = static_cast<long double>(hPtr[row * hCols + h]);
                                        beforeNoBias += hv * static_cast<long double>(wBeforePtr[h * wCols + cls]);
                                        afterNoBias += hv * static_cast<long double>(wAfterPtr[h * wCols + cls]);
                                    }

                                    const long double biasBefore = static_cast<long double>(bBeforePtr[cls]);
                                    const long double biasAfter = static_cast<long double>(bAfterPtr[cls]);
                                    const long double weightDelta = afterNoBias - beforeNoBias;
                                    const long double biasDelta = biasAfter - biasBefore;

                                    meanWeightBefore[cls] += beforeNoBias;
                                    meanWeightAfter[cls] += afterNoBias;
                                    meanWeightDelta[cls] += weightDelta;
                                    meanBiasDelta[cls] += biasDelta;
                                    meanTotalDelta[cls] += weightDelta + biasDelta;
                                    absWeightDeltaSum[cls] += std::fabs(static_cast<double>(weightDelta));
                                    absBiasDeltaSum[cls] += std::fabs(static_cast<double>(biasDelta));
                                }
                            }
                        }

                        auto meanContribution = [&](const std::array<long double, direction_output_size>& values, size_t cls) -> double
                        {
                            return rowCount > 0
                                ? static_cast<double>(values[cls] / static_cast<long double>(rowCount))
                                : 0.0;
                        };

                        const double neutralWeightDeltaMean = meanContribution(meanWeightDelta, 1);
                        const double neutralBiasDeltaMean = meanContribution(meanBiasDelta, 1);
                        const double neutralTotalDeltaMean = meanContribution(meanTotalDelta, 1);
                        const double neutralAbsWeightDeltaMean = meanContribution(absWeightDeltaSum, 1);
                        const double neutralAbsBiasDeltaMean = meanContribution(absBiasDeltaSum, 1);

                        std::cout << "DIAG_LOGIT_COMPONENT_DELTA_"
                                  << ",call=" << phase3ClipFullDiagIdx
                                  << ",rows=" << hRows
                                  << ",hidden_cols=" << hCols
                                  << ",weight_rows=" << wRows
                                  << ",weight_cols=" << wCols
                                  << ",valid_rows=" << rowCount
                                  << ",down_weight_before_mean=" << meanContribution(meanWeightBefore, 0)
                                  << ",neutral_weight_before_mean=" << meanContribution(meanWeightBefore, 1)
                                  << ",up_weight_before_mean=" << meanContribution(meanWeightBefore, 2)
                                  << ",down_weight_after_mean=" << meanContribution(meanWeightAfter, 0)
                                  << ",neutral_weight_after_mean=" << meanContribution(meanWeightAfter, 1)
                                  << ",up_weight_after_mean=" << meanContribution(meanWeightAfter, 2)
                                  << ",down_weight_delta_mean=" << meanContribution(meanWeightDelta, 0)
                                  << ",neutral_weight_delta_mean=" << neutralWeightDeltaMean
                                  << ",up_weight_delta_mean=" << meanContribution(meanWeightDelta, 2)
                                  << ",down_bias_delta_mean=" << meanContribution(meanBiasDelta, 0)
                                  << ",neutral_bias_delta_mean=" << neutralBiasDeltaMean
                                  << ",up_bias_delta_mean=" << meanContribution(meanBiasDelta, 2)
                                  << ",down_total_delta_mean=" << meanContribution(meanTotalDelta, 0)
                                  << ",neutral_total_delta_mean=" << neutralTotalDeltaMean
                                  << ",up_total_delta_mean=" << meanContribution(meanTotalDelta, 2)
                                  << ",neutral_abs_weight_delta_mean=" << neutralAbsWeightDeltaMean
                                  << ",neutral_abs_bias_delta_mean=" << neutralAbsBiasDeltaMean
                                  << ",neutral_abs_weight_over_bias=" << (neutralAbsBiasDeltaMean > 1.0e-12 ? neutralAbsWeightDeltaMean / neutralAbsBiasDeltaMean : 0.0)
                                  << std::endl;
                    }
                }
                else
                {
                    PrintPhase3HeadDelta(phase3ClipFullDiagIdx, "returnHeadWeight", returnHeadWeight, phase3HeadWUpdateBefore);
                    PrintPhase3HeadDelta(phase3ClipFullDiagIdx, "returnHeadBias", returnHeadBias, phase3HeadBUpdateBefore);
                }
            }
        #endif
    }

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
    const size_t windowGlobalStartIdx = static_cast<size_t>(w.begin() - t.begin());
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

        if (useReturnFeatures)
        {
            HotspotScope hotspot("appended_return_features");
            const size_t appended = EA::AssembleLstmModelInputRowAtGlobalPosition(
                dst, src, baseFeatureCount, windowGlobalStartIdx + rowIdx, EA::LSTM::kFeatScale,
                [this](size_t globalPosition)
                {
                    return t.RawCloseAtIterator(
                        t.begin() + static_cast<std::ptrdiff_t>(globalPosition));
                });
#if LSTM_TRAINING_ASSERTS
            LSTM_ASSERT(appended == kReturnFeatureCount,
                        "PredictNextReturn: appended return feature count mismatch");
#endif
        }
        else
        {
            std::memcpy(dst, src, baseFeatureCount * sizeof(float));
        }

        for (size_t c = 0; c < modelFeatureCount; ++c)
        {
            float& v = dst[c];
            if (!std::isfinite(v))
            {
                std::cout << "DIAG_NORM_FAIL"
                          << ",phase=inference_regression_logret"
                          << ",row=" << rowIdx
                          << ",col=" << c
                          << ",value=" << v
                          << std::endl;
                LSTM_ASSERT(false, "PredictNextReturn: non-finite feature detected before LSTM");
            }
            v = std::clamp(v, -10.0f, 10.0f);
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
    const size_t windowGlobalStartIdx = static_cast<size_t>(w.begin() - t.begin());
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

        if (useReturnFeatures)
        {
            HotspotScope hotspot("appended_return_features");
            const size_t appended = EA::AssembleLstmModelInputRowAtGlobalPosition(
                dst, src, baseFeatureCount, windowGlobalStartIdx + rowIdx, EA::LSTM::kFeatScale,
                [this](size_t globalPosition)
                {
                    return t.RawCloseAtIterator(
                        t.begin() + static_cast<std::ptrdiff_t>(globalPosition));
                });
#if LSTM_TRAINING_ASSERTS
            LSTM_ASSERT(appended == kReturnFeatureCount,
                        "PredictNextClose: appended return feature count mismatch");
#endif
        }
        else
        {
            std::memcpy(dst, src, baseFeatureCount * sizeof(float));
        }

        for (size_t c = 0; c < modelFeatureCount; ++c)
        {
            float& v = dst[c];
            if (!std::isfinite(v))
            {
                std::cout << "DIAG_NORM_FAIL"
                          << ",phase=inference_regression_relmove"
                          << ",row=" << rowIdx
                          << ",col=" << c
                          << ",value=" << v
                          << std::endl;
                LSTM_ASSERT(false, "PredictNextRelativeMove: non-finite feature detected before LSTM");
            }
            v = std::clamp(v, -10.0f, 10.0f);
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
