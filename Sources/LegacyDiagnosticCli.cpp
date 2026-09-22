#include "LegacyDiagnosticCli.hpp"

#include "MarketDataCore.hpp"
#include "ModelInputContract.hpp"
#include "ReturnFeatureHistory.hpp"
#include "TargetLabel.hpp"
#include "Tensor.hpp"
#include "LSTM.hpp"

#include <MetaNN/operation/math/sigmoid.h>
#include <MetaNN/operation/math/tanh.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <pqxx/pqxx>

namespace EA::LegacyDiagnosticCli
{
namespace
{

constexpr size_t BaselineReturnFeatureCount = EA::kModelReturnFeatureCount;

std::string ForexDbConnectionString()
{
    const char* host = std::getenv("FOREX_DB_HOST");
    const char* database = std::getenv("FOREX_DB_NAME");
    const std::string resolvedHost = (host && *host) ? host : "127.0.0.1";
    const std::string resolvedDatabase = (database && *database) ? database : "forex";
    return "hostaddr=" + resolvedHost + " gssencmode=disable user=pqxx dbname=" + resolvedDatabase;
}

struct BaselineExample
{
    size_t batchStart = 0;
    size_t batchRows = 0;
    size_t localStart = 0;
    size_t globalStart = 0;
    int label = 1;
};

struct BaselineMetrics
{
    std::array<size_t, direction_output_size> hist {0, 0, 0};
    size_t confusion[direction_output_size][direction_output_size] = {};
    size_t total = 0;
    size_t correct = 0;
};

double ClassWeightForBaseline(int cls)
{
    switch (cls)
    {
        case 0: return kClassWeightDown;
        case 1: return kClassWeightNeutral;
        case 2: return kClassWeightUp;
        default: return 1.0;
    }
}

float BaselineLookbackLogReturn(const Tensor& tensor,
                                size_t currentGlobalPosition,
                                size_t lookbackBars)
{
    return EA::ComputeLookbackLogReturnAtGlobalPosition(
        currentGlobalPosition,
        lookbackBars,
        [&tensor](size_t globalPosition)
        {
            return tensor.RawCloseAtIterator(
                tensor.begin() + static_cast<std::ptrdiff_t>(globalPosition));
        });
}

float BaselineFeatureAt(const Tensor& tensor,
                        const BaselineExample& ex,
                        size_t windowRow,
                        size_t col,
                        size_t baseFeatureCount)
{
    const auto batchBegin = tensor.begin() + static_cast<std::ptrdiff_t>(ex.batchStart);
    const size_t localRow = ex.localStart + windowRow;
    const size_t currentGlobalPosition = ex.globalStart + windowRow;
    LSTM_ASSERT(localRow < ex.batchRows, "BaselineFeatureAt: local row out of batch bounds");

    float v = 0.0f;
    if (col < baseFeatureCount)
    {
        const auto rowIt = batchBegin + static_cast<std::ptrdiff_t>(localRow);
        auto low = MetaNN::LowerAccess(*rowIt);
        v = low.RawMemory()[col];
    }
    else
    {
        const size_t retCol = col - baseFeatureCount;
        size_t written = 0;
#if LSTM_RET_HORIZON_1
        if (retCol == written) v = BaselineLookbackLogReturn(tensor, currentGlobalPosition, 1) * EA::LSTM::kFeatScale;
        ++written;
#endif
#if LSTM_RET_HORIZON_4
        if (retCol == written) v = BaselineLookbackLogReturn(tensor, currentGlobalPosition, 4) * EA::LSTM::kFeatScale;
        ++written;
#endif
#if LSTM_RET_HORIZON_8
        if (retCol == written) v = BaselineLookbackLogReturn(tensor, currentGlobalPosition, 8) * EA::LSTM::kFeatScale;
        ++written;
#endif
#if LSTM_RET_HORIZON_16
        if (retCol == written) v = BaselineLookbackLogReturn(tensor, currentGlobalPosition, 16) * EA::LSTM::kFeatScale;
        ++written;
#endif
        (void)written;
    }

    LSTM_ASSERT(std::isfinite(v), "BaselineFeatureAt: non-finite feature before clamp");
    return std::clamp(v, -10.0f, 10.0f);
}

void FillBaselineFeatureVector(const Tensor& tensor,
                               const BaselineExample& ex,
                               size_t modelFeatureCount,
                               size_t baseFeatureCount,
                               std::vector<float>& out)
{
    const size_t dims = window_size * modelFeatureCount;
    out.resize(dims);
    size_t dst = 0;
    for (size_t wr = 0; wr < window_size; ++wr)
    {
        for (size_t c = 0; c < modelFeatureCount; ++c)
            out[dst++] = BaselineFeatureAt(tensor, ex, wr, c, baseFeatureCount);
    }
}

std::array<double, direction_output_size> BaselineSoftmax(const std::array<double, direction_output_size>& logits)
{
    const double m = std::max(logits[0], std::max(logits[1], logits[2]));
    std::array<double, direction_output_size> p {
        std::exp(logits[0] - m),
        std::exp(logits[1] - m),
        std::exp(logits[2] - m)
    };
    const double s = p[0] + p[1] + p[2];
    p[0] /= s;
    p[1] /= s;
    p[2] /= s;
    return p;
}

int BaselinePredict(const std::vector<double>& weights,
                    const std::array<double, direction_output_size>& bias,
                    const std::vector<float>& x)
{
    const size_t dims = x.size();
    std::array<double, direction_output_size> logits = bias;
    for (size_t cls = 0; cls < direction_output_size; ++cls)
    {
        const double* w = weights.data() + cls * dims;
        double z = logits[cls];
        for (size_t j = 0; j < dims; ++j)
            z += w[j] * static_cast<double>(x[j]);
        logits[cls] = z;
    }
    if (logits[0] > logits[1] && logits[0] > logits[2]) return 0;
    if (logits[2] > logits[1] && logits[2] > logits[0]) return 2;
    return 1;
}

BaselineMetrics EvaluateBaseline(const Tensor& tensor,
                                 const std::vector<BaselineExample>& examples,
                                 size_t beginIdx,
                                 size_t endIdx,
                                 size_t modelFeatureCount,
                                 size_t baseFeatureCount,
                                 const std::vector<double>& weights,
                                 const std::array<double, direction_output_size>& bias)
{
    BaselineMetrics m;
    std::vector<float> x;
    for (size_t i = beginIdx; i < endIdx; ++i)
    {
        const auto& ex = examples[i];
        FillBaselineFeatureVector(tensor, ex, modelFeatureCount, baseFeatureCount, x);
        const int pred = BaselinePredict(weights, bias, x);
        const int actual = ex.label;
        ++m.hist[static_cast<size_t>(actual)];
        ++m.confusion[static_cast<size_t>(actual)][static_cast<size_t>(pred)];
        ++m.total;
        if (pred == actual) ++m.correct;
    }
    return m;
}

void PrintBaselineMetrics(const char* prefix, const BaselineMetrics& m)
{
    const double total = m.total ? static_cast<double>(m.total) : 1.0;
    std::cout << prefix
              << "_HIST,total=" << m.total
              << ",down=" << m.hist[0]
              << ",neutral=" << m.hist[1]
              << ",up=" << m.hist[2]
              << ",down_frac=" << (static_cast<double>(m.hist[0]) / total)
              << ",neutral_frac=" << (static_cast<double>(m.hist[1]) / total)
              << ",up_frac=" << (static_cast<double>(m.hist[2]) / total)
              << std::endl;

    const double acc = m.total ? static_cast<double>(m.correct) / static_cast<double>(m.total) : 0.0;
    const double neutralOnly = m.total ? static_cast<double>(m.hist[1]) / static_cast<double>(m.total) : 0.0;
    std::cout << prefix
              << "_ACCURACY,correct=" << m.correct
              << ",total=" << m.total
              << ",accuracy=" << acc
              << ",neutral_only_accuracy=" << neutralOnly
              << std::endl;

    std::cout << prefix << "_CONFUSION_MATRIX,rows=actual,cols=pred"
              << ",row0=" << m.confusion[0][0] << ":" << m.confusion[0][1] << ":" << m.confusion[0][2]
              << ",row1=" << m.confusion[1][0] << ":" << m.confusion[1][1] << ":" << m.confusion[1][2]
              << ",row2=" << m.confusion[2][0] << ":" << m.confusion[2][1] << ":" << m.confusion[2][2]
              << std::endl;

    double macroF1 = 0.0;
    double balancedAccuracy = 0.0;
    for (size_t cls = 0; cls < direction_output_size; ++cls)
    {
        const double tp = static_cast<double>(m.confusion[cls][cls]);
        double fp = 0.0;
        double fn = 0.0;
        for (size_t other = 0; other < direction_output_size; ++other)
        {
            if (other != cls)
            {
                fp += static_cast<double>(m.confusion[other][cls]);
                fn += static_cast<double>(m.confusion[cls][other]);
            }
        }
        const double precision = (tp + fp) > 0.0 ? tp / (tp + fp) : 0.0;
        const double recall = (tp + fn) > 0.0 ? tp / (tp + fn) : 0.0;
        const double f1 = (precision + recall) > 0.0 ? (2.0 * precision * recall / (precision + recall)) : 0.0;
        macroF1 += f1;
        balancedAccuracy += recall;
        std::cout << prefix
                  << "_CLASS_METRIC,class=" << cls
                  << ",precision=" << precision
                  << ",recall=" << recall
                  << ",f1=" << f1
                  << std::endl;
    }
    macroF1 /= static_cast<double>(direction_output_size);
    balancedAccuracy /= static_cast<double>(direction_output_size);
    std::cout << prefix
              << "_SUMMARY,macro_f1=" << macroF1
              << ",balanced_accuracy=" << balancedAccuracy
              << std::endl;
}

struct FeatureSeparabilityStats
{
    std::array<size_t, direction_output_size> hist {0, 0, 0};
    std::array<double, direction_output_size> centroidNorm {0.0, 0.0, 0.0};
    std::array<double, direction_output_size> withinRms {0.0, 0.0, 0.0};
    double downNeutralDist = 0.0;
    double downUpDist = 0.0;
    double neutralUpDist = 0.0;
    double meanCentroidDist = 0.0;
    double meanWithinRms = 0.0;
    double separationToWithinRatio = 0.0;
    double nearestCentroidAccuracy = 0.0;
    size_t nearestCentroidCorrect = 0;
    size_t nearestCentroidTotal = 0;
};

std::vector<float> BuildLastStepFeatureMatrix(const Tensor& tensor,
                                              const std::vector<BaselineExample>& examples,
                                              size_t modelFeatureCount,
                                              size_t baseFeatureCount)
{
    std::vector<float> features(examples.size() * modelFeatureCount);
    for (size_t i = 0; i < examples.size(); ++i)
    {
        float* dst = features.data() + i * modelFeatureCount;
        for (size_t c = 0; c < modelFeatureCount; ++c)
            dst[c] = BaselineFeatureAt(tensor, examples[i], window_size - 1, c, baseFeatureCount);
    }
    return features;
}

std::vector<float> BuildFlatWindowFeatureMatrix(const Tensor& tensor,
                                                const std::vector<BaselineExample>& examples,
                                                size_t modelFeatureCount,
                                                size_t baseFeatureCount)
{
    const size_t dims = window_size * modelFeatureCount;
    std::vector<float> features(examples.size() * dims);
    std::vector<float> x;
    for (size_t i = 0; i < examples.size(); ++i)
    {
        FillBaselineFeatureVector(tensor, examples[i], modelFeatureCount, baseFeatureCount, x);
        std::copy(x.begin(), x.end(), features.begin() + static_cast<std::ptrdiff_t>(i * dims));
    }
    return features;
}

FeatureSeparabilityStats ComputeSeparability(const std::vector<float>& features,
                                             const std::vector<int>& labels,
                                             size_t beginIdx,
                                             size_t endIdx,
                                             size_t dims)
{
    FeatureSeparabilityStats s;
    std::array<std::vector<long double>, direction_output_size> sums;
    for (auto& v : sums) v.assign(dims, 0.0L);

    for (size_t i = beginIdx; i < endIdx; ++i)
    {
        const int cls = labels[i];
        if (cls < 0 || cls >= static_cast<int>(direction_output_size)) continue;
        ++s.hist[static_cast<size_t>(cls)];
        const float* x = features.data() + i * dims;
        for (size_t j = 0; j < dims; ++j)
            sums[static_cast<size_t>(cls)][j] += static_cast<long double>(x[j]);
    }

    std::array<std::vector<double>, direction_output_size> centroids;
    for (size_t cls = 0; cls < direction_output_size; ++cls)
    {
        centroids[cls].assign(dims, 0.0);
        long double normSq = 0.0L;
        const long double denom = s.hist[cls] ? static_cast<long double>(s.hist[cls]) : 1.0L;
        for (size_t j = 0; j < dims; ++j)
        {
            const double mean = static_cast<double>(sums[cls][j] / denom);
            centroids[cls][j] = mean;
            normSq += static_cast<long double>(mean) * static_cast<long double>(mean);
        }
        s.centroidNorm[cls] = std::sqrt(static_cast<double>(normSq));
    }

    std::array<long double, direction_output_size> withinSq {0.0L, 0.0L, 0.0L};
    for (size_t i = beginIdx; i < endIdx; ++i)
    {
        const int clsInt = labels[i];
        if (clsInt < 0 || clsInt >= static_cast<int>(direction_output_size)) continue;
        const size_t cls = static_cast<size_t>(clsInt);
        const float* x = features.data() + i * dims;
        for (size_t j = 0; j < dims; ++j)
        {
            const long double d = static_cast<long double>(x[j]) - static_cast<long double>(centroids[cls][j]);
            withinSq[cls] += d * d;
        }
    }
    for (size_t cls = 0; cls < direction_output_size; ++cls)
        s.withinRms[cls] = s.hist[cls] ? std::sqrt(static_cast<double>(withinSq[cls] / static_cast<long double>(s.hist[cls]))) : 0.0;

    auto centroidDistance = [&](size_t a, size_t b)
    {
        long double distSq = 0.0L;
        for (size_t j = 0; j < dims; ++j)
        {
            const long double d = static_cast<long double>(centroids[a][j]) - static_cast<long double>(centroids[b][j]);
            distSq += d * d;
        }
        return std::sqrt(static_cast<double>(distSq));
    };
    s.downNeutralDist = centroidDistance(0, 1);
    s.downUpDist = centroidDistance(0, 2);
    s.neutralUpDist = centroidDistance(1, 2);
    s.meanCentroidDist = (s.downNeutralDist + s.downUpDist + s.neutralUpDist) / 3.0;
    s.meanWithinRms = (s.withinRms[0] + s.withinRms[1] + s.withinRms[2]) / 3.0;
    s.separationToWithinRatio = s.meanWithinRms > 0.0 ? s.meanCentroidDist / s.meanWithinRms : 0.0;

    for (size_t i = beginIdx; i < endIdx; ++i)
    {
        const int actual = labels[i];
        if (actual < 0 || actual >= static_cast<int>(direction_output_size)) continue;
        const float* x = features.data() + i * dims;
        double bestDist = std::numeric_limits<double>::infinity();
        int bestClass = 1;
        for (size_t cls = 0; cls < direction_output_size; ++cls)
        {
            if (s.hist[cls] == 0) continue;
            long double distSq = 0.0L;
            const bool excludeSelf = (static_cast<int>(cls) == actual && s.hist[cls] > 1);
            const long double denom = excludeSelf ? static_cast<long double>(s.hist[cls] - 1) : static_cast<long double>(s.hist[cls]);
            for (size_t j = 0; j < dims; ++j)
            {
                long double mean = sums[cls][j];
                if (excludeSelf) mean -= static_cast<long double>(x[j]);
                mean /= denom;
                const long double d = static_cast<long double>(x[j]) - mean;
                distSq += d * d;
            }
            const double dist = std::sqrt(static_cast<double>(distSq));
            if (dist < bestDist)
            {
                bestDist = dist;
                bestClass = static_cast<int>(cls);
            }
        }
        ++s.nearestCentroidTotal;
        if (bestClass == actual) ++s.nearestCentroidCorrect;
    }
    s.nearestCentroidAccuracy = s.nearestCentroidTotal
        ? static_cast<double>(s.nearestCentroidCorrect) / static_cast<double>(s.nearestCentroidTotal)
        : 0.0;
    return s;
}

void PrintSeparabilityStats(const char* prefix,
                            const char* view,
                            const char* split,
                            size_t dims,
                            const FeatureSeparabilityStats& s)
{
    const size_t total = s.hist[0] + s.hist[1] + s.hist[2];
    const double denom = total ? static_cast<double>(total) : 1.0;
    std::cout << prefix
              << ",view=" << view
              << ",split=" << split
              << ",total=" << total
              << ",dims=" << dims
              << ",down=" << s.hist[0]
              << ",neutral=" << s.hist[1]
              << ",up=" << s.hist[2]
              << ",down_frac=" << static_cast<double>(s.hist[0]) / denom
              << ",neutral_frac=" << static_cast<double>(s.hist[1]) / denom
              << ",up_frac=" << static_cast<double>(s.hist[2]) / denom
              << ",centroid_norm_down=" << s.centroidNorm[0]
              << ",centroid_norm_neutral=" << s.centroidNorm[1]
              << ",centroid_norm_up=" << s.centroidNorm[2]
              << ",within_rms_down=" << s.withinRms[0]
              << ",within_rms_neutral=" << s.withinRms[1]
              << ",within_rms_up=" << s.withinRms[2]
              << ",centroid_dist_down_neutral=" << s.downNeutralDist
              << ",centroid_dist_down_up=" << s.downUpDist
              << ",centroid_dist_neutral_up=" << s.neutralUpDist
              << ",mean_centroid_dist=" << s.meanCentroidDist
              << ",mean_within_rms=" << s.meanWithinRms
              << ",separation_to_within_ratio=" << s.separationToWithinRatio
              << ",nearest_centroid_accuracy=" << s.nearestCentroidAccuracy
              << std::endl;
}

BaselineMetrics EvaluatePredictions(const std::vector<int>& labels,
                                    size_t beginIdx,
                                    size_t endIdx,
                                    const std::vector<int>& preds)
{
    BaselineMetrics m;
    for (size_t i = beginIdx; i < endIdx; ++i)
    {
        const int actual = labels[i];
        const int pred = preds[i - beginIdx];
        ++m.hist[static_cast<size_t>(actual)];
        ++m.confusion[static_cast<size_t>(actual)][static_cast<size_t>(pred)];
        ++m.total;
        if (pred == actual) ++m.correct;
    }
    return m;
}

void TrainFeatureLogReg(const std::vector<float>& features,
                        const std::vector<int>& labels,
                        size_t trainCount,
                        size_t dims,
                        size_t epochs,
                        double lr,
                        std::vector<double>& weights,
                        std::array<double, direction_output_size>& bias)
{
    weights.assign(direction_output_size * dims, 0.0);
    bias = {0.0, 0.0, 0.0};
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < trainCount; ++i)
        {
            const float* x = features.data() + i * dims;
            std::array<double, direction_output_size> logits = bias;
            for (size_t cls = 0; cls < direction_output_size; ++cls)
            {
                const double* w = weights.data() + cls * dims;
                for (size_t j = 0; j < dims; ++j)
                    logits[cls] += w[j] * static_cast<double>(x[j]);
            }
            const auto probs = BaselineSoftmax(logits);
            for (size_t cls = 0; cls < direction_output_size; ++cls)
            {
                const double target = (static_cast<int>(cls) == labels[i]) ? 1.0 : 0.0;
                const double g = probs[cls] - target;
                double* w = weights.data() + cls * dims;
                for (size_t j = 0; j < dims; ++j)
                    w[j] -= lr * g * static_cast<double>(x[j]);
                bias[cls] -= lr * g;
            }
        }
    }
}

std::vector<int> PredictFeatureLogReg(const std::vector<float>& features,
                                      size_t beginIdx,
                                      size_t endIdx,
                                      size_t dims,
                                      const std::vector<double>& weights,
                                      const std::array<double, direction_output_size>& bias)
{
    std::vector<int> preds;
    preds.reserve(endIdx - beginIdx);
    for (size_t i = beginIdx; i < endIdx; ++i)
    {
        const float* x = features.data() + i * dims;
        std::array<double, direction_output_size> logits = bias;
        for (size_t cls = 0; cls < direction_output_size; ++cls)
        {
            const double* w = weights.data() + cls * dims;
            for (size_t j = 0; j < dims; ++j)
                logits[cls] += w[j] * static_cast<double>(x[j]);
        }
        preds.push_back((logits[0] > logits[1] && logits[0] > logits[2]) ? 0 :
                        ((logits[2] > logits[1] && logits[2] > logits[0]) ? 2 : 1));
    }
    return preds;
}

struct ShallowMlp
{
    size_t dims = 0;
    size_t hidden = 0;
    std::vector<double> w1;
    std::vector<double> b1;
    std::vector<double> w2;
    std::array<double, direction_output_size> b2 {0.0, 0.0, 0.0};
};

void TrainShallowMlp(const std::vector<float>& features,
                     const std::vector<int>& labels,
                     size_t trainCount,
                     size_t dims,
                     ShallowMlp& mlp)
{
    constexpr size_t kHidden = 32;
    constexpr size_t kEpochs = 5;
    constexpr double kLr = 5.0e-5;
    mlp.dims = dims;
    mlp.hidden = kHidden;
    mlp.w1.assign(dims * kHidden, 0.0);
    mlp.b1.assign(kHidden, 0.0);
    mlp.w2.assign(kHidden * direction_output_size, 0.0);
    mlp.b2 = {0.0, 0.0, 0.0};
    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> init(-0.02, 0.02);
    for (double& v : mlp.w1) v = init(rng);
    for (double& v : mlp.w2) v = init(rng);

    std::vector<double> h(kHidden);
    std::array<double, direction_output_size> logits;
    for (size_t epoch = 0; epoch < kEpochs; ++epoch)
    {
        for (size_t i = 0; i < trainCount; ++i)
        {
            const float* x = features.data() + i * dims;
            for (size_t k = 0; k < kHidden; ++k)
            {
                double z = mlp.b1[k];
                for (size_t j = 0; j < dims; ++j)
                    z += static_cast<double>(x[j]) * mlp.w1[j * kHidden + k];
                h[k] = std::tanh(z);
            }
            logits = mlp.b2;
            for (size_t cls = 0; cls < direction_output_size; ++cls)
                for (size_t k = 0; k < kHidden; ++k)
                    logits[cls] += h[k] * mlp.w2[k * direction_output_size + cls];

            const auto probs = BaselineSoftmax(logits);
            std::array<double, direction_output_size> dz2;
            for (size_t cls = 0; cls < direction_output_size; ++cls)
                dz2[cls] = probs[cls] - ((static_cast<int>(cls) == labels[i]) ? 1.0 : 0.0);

            std::vector<double> dh(kHidden, 0.0);
            for (size_t k = 0; k < kHidden; ++k)
            {
                for (size_t cls = 0; cls < direction_output_size; ++cls)
                {
                    dh[k] += dz2[cls] * mlp.w2[k * direction_output_size + cls];
                    mlp.w2[k * direction_output_size + cls] -= kLr * dz2[cls] * h[k];
                }
            }
            for (size_t cls = 0; cls < direction_output_size; ++cls)
                mlp.b2[cls] -= kLr * dz2[cls];

            for (size_t k = 0; k < kHidden; ++k)
            {
                const double dz1 = dh[k] * (1.0 - h[k] * h[k]);
                for (size_t j = 0; j < dims; ++j)
                    mlp.w1[j * kHidden + k] -= kLr * dz1 * static_cast<double>(x[j]);
                mlp.b1[k] -= kLr * dz1;
            }
        }
    }
}

std::vector<int> PredictShallowMlp(const std::vector<float>& features,
                                   size_t beginIdx,
                                   size_t endIdx,
                                   const ShallowMlp& mlp)
{
    std::vector<int> preds;
    preds.reserve(endIdx - beginIdx);
    std::vector<double> h(mlp.hidden);
    for (size_t i = beginIdx; i < endIdx; ++i)
    {
        const float* x = features.data() + i * mlp.dims;
        for (size_t k = 0; k < mlp.hidden; ++k)
        {
            double z = mlp.b1[k];
            for (size_t j = 0; j < mlp.dims; ++j)
                z += static_cast<double>(x[j]) * mlp.w1[j * mlp.hidden + k];
            h[k] = std::tanh(z);
        }
        std::array<double, direction_output_size> logits = mlp.b2;
        for (size_t cls = 0; cls < direction_output_size; ++cls)
            for (size_t k = 0; k < mlp.hidden; ++k)
                logits[cls] += h[k] * mlp.w2[k * direction_output_size + cls];
        preds.push_back((logits[0] > logits[1] && logits[0] > logits[2]) ? 0 :
                        ((logits[2] > logits[1] && logits[2] > logits[0]) ? 2 : 1));
    }
    return preds;
}

struct RandomStump
{
    size_t feature = 0;
    float threshold = 0.0f;
    std::array<size_t, direction_output_size> leftHist {0, 0, 0};
    std::array<size_t, direction_output_size> rightHist {0, 0, 0};
};

int MajorityClass(const std::array<size_t, direction_output_size>& hist)
{
    if (hist[0] > hist[1] && hist[0] > hist[2]) return 0;
    if (hist[2] > hist[1] && hist[2] > hist[0]) return 2;
    return 1;
}

double Gini(const std::array<size_t, direction_output_size>& hist)
{
    const double total = static_cast<double>(hist[0] + hist[1] + hist[2]);
    if (total <= 0.0) return 1.0;
    double sumSq = 0.0;
    for (size_t cls = 0; cls < direction_output_size; ++cls)
    {
        const double p = static_cast<double>(hist[cls]) / total;
        sumSq += p * p;
    }
    return 1.0 - sumSq;
}

std::vector<RandomStump> TrainRandomStumpForest(const std::vector<float>& features,
                                                const std::vector<int>& labels,
                                                size_t trainCount,
                                                size_t dims)
{
    constexpr size_t kTrees = 128;
    constexpr size_t kFeatureTries = 12;
    constexpr size_t kThresholdSamples = 1024;
    constexpr size_t kEvalSamples = 4096;
    std::mt19937 rng(20260611);
    std::uniform_int_distribution<size_t> featureDist(0, dims - 1);
    std::uniform_int_distribution<size_t> rowDist(0, trainCount - 1);
    std::vector<RandomStump> forest;
    forest.reserve(kTrees);

    for (size_t tree = 0; tree < kTrees; ++tree)
    {
        double bestScore = std::numeric_limits<double>::infinity();
        RandomStump best;
        for (size_t ft = 0; ft < kFeatureTries; ++ft)
        {
            const size_t feature = featureDist(rng);
            std::vector<float> thresholds;
            thresholds.reserve(8);
            for (size_t sIdx = 0; sIdx < kThresholdSamples; ++sIdx)
                thresholds.push_back(features[rowDist(rng) * dims + feature]);
            std::sort(thresholds.begin(), thresholds.end());
            for (double q : {0.2, 0.35, 0.5, 0.65, 0.8})
            {
                const size_t qi = std::min(thresholds.size() - 1, static_cast<size_t>(q * static_cast<double>(thresholds.size() - 1)));
                const float thr = thresholds[qi];
                RandomStump candidate;
                candidate.feature = feature;
                candidate.threshold = thr;
                for (size_t sIdx = 0; sIdx < kEvalSamples; ++sIdx)
                {
                    const size_t row = rowDist(rng);
                    auto& hist = (features[row * dims + feature] <= thr) ? candidate.leftHist : candidate.rightHist;
                    ++hist[static_cast<size_t>(labels[row])];
                }
                const double leftN = static_cast<double>(candidate.leftHist[0] + candidate.leftHist[1] + candidate.leftHist[2]);
                const double rightN = static_cast<double>(candidate.rightHist[0] + candidate.rightHist[1] + candidate.rightHist[2]);
                const double score = leftN * Gini(candidate.leftHist) + rightN * Gini(candidate.rightHist);
                if (score < bestScore)
                {
                    bestScore = score;
                    best = candidate;
                }
            }
        }

        best.leftHist = {0, 0, 0};
        best.rightHist = {0, 0, 0};
        for (size_t i = 0; i < trainCount; ++i)
        {
            auto& hist = (features[i * dims + best.feature] <= best.threshold) ? best.leftHist : best.rightHist;
            ++hist[static_cast<size_t>(labels[i])];
        }
        forest.push_back(best);
    }
    return forest;
}

std::vector<int> PredictRandomStumpForest(const std::vector<float>& features,
                                          size_t beginIdx,
                                          size_t endIdx,
                                          size_t dims,
                                          const std::vector<RandomStump>& forest)
{
    std::vector<int> preds;
    preds.reserve(endIdx - beginIdx);
    for (size_t i = beginIdx; i < endIdx; ++i)
    {
        std::array<size_t, direction_output_size> votes {0, 0, 0};
        for (const auto& stump : forest)
        {
            const auto& hist = (features[i * dims + stump.feature] <= stump.threshold) ? stump.leftHist : stump.rightHist;
            ++votes[static_cast<size_t>(MajorityClass(hist))];
        }
        preds.push_back(MajorityClass(votes));
    }
    return preds;
}

struct GridLabelInfo
{
    int assignedClass = 1;
    bool upHit = false;
    bool downHit = false;
    size_t upOffset = 0;
    size_t downOffset = 0;
};

struct GridHitStats
{
    size_t total = 0;
    size_t bothHit = 0;
    size_t upHit = 0;
    size_t downHit = 0;
    double upOffsetSum = 0.0;
    double downOffsetSum = 0.0;
};

struct GridResultSummary
{
    size_t horizon = 0;
    float threshold = 0.0f;
    double validationAccuracy = 0.0;
    double validationNeutralOnly = 0.0;
    double deltaVsNeutral = 0.0;
    double macroF1 = 0.0;
    double balancedAccuracy = 0.0;
};

GridLabelInfo BuildGridLookaheadClassInfo(const Tensor& tensor,
                                          const BaselineExample& ex,
                                          size_t horizon,
                                          float threshold)
{
    const auto lastIt = tensor.begin() + static_cast<std::ptrdiff_t>(ex.globalStart + window_size - 1);
    const float closeT = tensor.RawCloseAtIterator(lastIt);

    GridLabelInfo info;
    for (size_t lookahead = 1; lookahead <= horizon; ++lookahead)
    {
        const auto futureIt = lastIt + static_cast<std::ptrdiff_t>(lookahead);
        const float futureHigh = tensor.RawHighAtIterator(futureIt);
        const float futureLow = tensor.RawLowAtIterator(futureIt);

        if (std::isfinite(closeT) && closeT > 0.0f)
        {
            const float upMove = std::log(futureHigh / closeT);
            const float downMove = std::log(futureLow / closeT);

            if (!info.upHit && std::isfinite(upMove) && upMove > threshold)
            {
                info.upHit = true;
                info.upOffset = lookahead;
            }

            if (!info.downHit && std::isfinite(downMove) && downMove < -threshold)
            {
                info.downHit = true;
                info.downOffset = lookahead;
            }
        }
    }

    if (info.upHit && info.downHit) info.assignedClass = (info.upOffset <= info.downOffset) ? 2 : 0;
    else if (info.upHit) info.assignedClass = 2;
    else if (info.downHit) info.assignedClass = 0;
    else info.assignedClass = 1;

    return info;
}

std::vector<BaselineExample> BuildGridExamples(const Tensor& tensor, size_t maxHorizon)
{
    std::vector<BaselineExample> examples;
    tensor.ForEachBatch([&](auto b)
    {
        const size_t batchStart = static_cast<size_t>(b.begin() - tensor.begin());
        const size_t batchRows = static_cast<size_t>(b.end() - b.begin());
        for (size_t localStart = 0;
             localStart + window_size + maxHorizon - 1 < batchRows;
             ++localStart)
        {
            examples.push_back(BaselineExample{
                batchStart,
                batchRows,
                localStart,
                batchStart + localStart,
                1
            });
        }
    });
    return examples;
}

std::vector<float> BuildGridLastStepFeatureMatrix(const Tensor& tensor,
                                                  const std::vector<BaselineExample>& examples,
                                                  size_t modelFeatureCount,
                                                  size_t baseFeatureCount)
{
    std::vector<float> features(examples.size() * modelFeatureCount);
    for (size_t i = 0; i < examples.size(); ++i)
    {
        float* dst = features.data() + i * modelFeatureCount;
        for (size_t c = 0; c < modelFeatureCount; ++c)
            dst[c] = BaselineFeatureAt(tensor, examples[i], window_size - 1, c, baseFeatureCount);
    }
    return features;
}

std::vector<int> BuildGridLabelsAndStats(const Tensor& tensor,
                                         const std::vector<BaselineExample>& examples,
                                         size_t horizon,
                                         float threshold,
                                         GridHitStats& hitStats)
{
    std::vector<int> labels(examples.size(), 1);
    for (size_t i = 0; i < examples.size(); ++i)
    {
        const auto info = BuildGridLookaheadClassInfo(tensor, examples[i], horizon, threshold);
        labels[i] = info.assignedClass;
        ++hitStats.total;
        if (info.upHit && info.downHit) ++hitStats.bothHit;
        if (info.upHit)
        {
            ++hitStats.upHit;
            hitStats.upOffsetSum += static_cast<double>(info.upOffset);
        }
        if (info.downHit)
        {
            ++hitStats.downHit;
            hitStats.downOffsetSum += static_cast<double>(info.downOffset);
        }
    }
    return labels;
}

int PredictGridLogReg(const std::vector<double>& weights,
                      const std::array<double, direction_output_size>& bias,
                      const float* x,
                      size_t dims)
{
    std::array<double, direction_output_size> logits = bias;
    for (size_t cls = 0; cls < direction_output_size; ++cls)
    {
        const double* w = weights.data() + cls * dims;
        double z = logits[cls];
        for (size_t j = 0; j < dims; ++j)
            z += w[j] * static_cast<double>(x[j]);
        logits[cls] = z;
    }
    if (logits[0] > logits[1] && logits[0] > logits[2]) return 0;
    if (logits[2] > logits[1] && logits[2] > logits[0]) return 2;
    return 1;
}

BaselineMetrics EvaluateGridLogReg(const std::vector<float>& features,
                                   const std::vector<int>& labels,
                                   size_t beginIdx,
                                   size_t endIdx,
                                   size_t dims,
                                   const std::vector<double>& weights,
                                   const std::array<double, direction_output_size>& bias)
{
    BaselineMetrics m;
    for (size_t i = beginIdx; i < endIdx; ++i)
    {
        const int actual = labels[i];
        const int pred = PredictGridLogReg(weights, bias, features.data() + i * dims, dims);
        ++m.hist[static_cast<size_t>(actual)];
        ++m.confusion[static_cast<size_t>(actual)][static_cast<size_t>(pred)];
        ++m.total;
        if (pred == actual) ++m.correct;
    }
    return m;
}

double BaselineAccuracy(const BaselineMetrics& m)
{
    return m.total ? static_cast<double>(m.correct) / static_cast<double>(m.total) : 0.0;
}

double BaselineNeutralOnlyAccuracy(const BaselineMetrics& m)
{
    return m.total ? static_cast<double>(m.hist[1]) / static_cast<double>(m.total) : 0.0;
}

double BaselineClassPrecision(const BaselineMetrics& m, size_t cls)
{
    const double tp = static_cast<double>(m.confusion[cls][cls]);
    double fp = 0.0;
    for (size_t other = 0; other < direction_output_size; ++other)
    {
        if (other != cls)
            fp += static_cast<double>(m.confusion[other][cls]);
    }
    return (tp + fp) > 0.0 ? tp / (tp + fp) : 0.0;
}

double BaselineClassRecall(const BaselineMetrics& m, size_t cls)
{
    const double tp = static_cast<double>(m.confusion[cls][cls]);
    double fn = 0.0;
    for (size_t other = 0; other < direction_output_size; ++other)
    {
        if (other != cls)
            fn += static_cast<double>(m.confusion[cls][other]);
    }
    return (tp + fn) > 0.0 ? tp / (tp + fn) : 0.0;
}

double BaselineClassF1(const BaselineMetrics& m, size_t cls)
{
    const double precision = BaselineClassPrecision(m, cls);
    const double recall = BaselineClassRecall(m, cls);
    return (precision + recall) > 0.0 ? (2.0 * precision * recall / (precision + recall)) : 0.0;
}

double BaselineMacroF1(const BaselineMetrics& m)
{
    double macroF1 = 0.0;
    for (size_t cls = 0; cls < direction_output_size; ++cls)
        macroF1 += BaselineClassF1(m, cls);
    return macroF1 / static_cast<double>(direction_output_size);
}

double BaselineBalancedAccuracy(const BaselineMetrics& m)
{
    double balancedAccuracy = 0.0;
    for (size_t cls = 0; cls < direction_output_size; ++cls)
        balancedAccuracy += BaselineClassRecall(m, cls);
    return balancedAccuracy / static_cast<double>(direction_output_size);
}

void PrintGridHist(const char* prefix, size_t horizon, float threshold, const BaselineMetrics& m)
{
    const double total = m.total ? static_cast<double>(m.total) : 1.0;
    std::cout << prefix
              << ",horizon=" << horizon
              << ",threshold_logret=" << threshold
              << ",total=" << m.total
              << ",down=" << m.hist[0]
              << ",neutral=" << m.hist[1]
              << ",up=" << m.hist[2]
              << ",down_frac=" << (static_cast<double>(m.hist[0]) / total)
              << ",neutral_frac=" << (static_cast<double>(m.hist[1]) / total)
              << ",up_frac=" << (static_cast<double>(m.hist[2]) / total)
              << ",neutral_only_accuracy=" << BaselineNeutralOnlyAccuracy(m)
              << std::endl;
}

void PrintGridMetrics(const char* prefix, size_t horizon, float threshold, const BaselineMetrics& m)
{
    std::cout << prefix << "_BASELINE"
              << ",horizon=" << horizon
              << ",threshold_logret=" << threshold
              << ",accuracy=" << BaselineAccuracy(m)
              << ",neutral_only_accuracy=" << BaselineNeutralOnlyAccuracy(m)
              << ",delta_vs_neutral=" << (BaselineAccuracy(m) - BaselineNeutralOnlyAccuracy(m))
              << ",macro_f1=" << BaselineMacroF1(m)
              << ",balanced_accuracy=" << BaselineBalancedAccuracy(m)
              << std::endl;

    std::cout << prefix << "_CONFUSION_MATRIX"
              << ",horizon=" << horizon
              << ",threshold_logret=" << threshold
              << ",rows=actual,cols=pred"
              << ",row0=" << m.confusion[0][0] << ":" << m.confusion[0][1] << ":" << m.confusion[0][2]
              << ",row1=" << m.confusion[1][0] << ":" << m.confusion[1][1] << ":" << m.confusion[1][2]
              << ",row2=" << m.confusion[2][0] << ":" << m.confusion[2][1] << ":" << m.confusion[2][2]
              << std::endl;

    for (size_t cls = 0; cls < direction_output_size; ++cls)
    {
        std::cout << prefix << "_CLASS_METRIC"
                  << ",horizon=" << horizon
                  << ",threshold_logret=" << threshold
                  << ",class=" << cls
                  << ",precision=" << BaselineClassPrecision(m, cls)
                  << ",recall=" << BaselineClassRecall(m, cls)
                  << ",f1=" << BaselineClassF1(m, cls)
                  << std::endl;
    }
}

void TrainGridLogReg(const std::vector<float>& features,
                     const std::vector<int>& labels,
                     size_t trainCount,
                     size_t dims,
                     std::vector<double>& weights,
                     std::array<double, direction_output_size>& bias)
{
    constexpr size_t kGridBaselineEpochs = 3;
    constexpr double kGridBaselineLearningRate = 1.0e-4;
    for (size_t epoch = 0; epoch < kGridBaselineEpochs; ++epoch)
    {
        for (size_t i = 0; i < trainCount; ++i)
        {
            const float* x = features.data() + i * dims;
            std::array<double, direction_output_size> logits = bias;
            for (size_t cls = 0; cls < direction_output_size; ++cls)
            {
                const double* w = weights.data() + cls * dims;
                double z = logits[cls];
                for (size_t j = 0; j < dims; ++j)
                    z += w[j] * static_cast<double>(x[j]);
                logits[cls] = z;
            }

            const auto probs = BaselineSoftmax(logits);
            const double sampleWeight = ClassWeightForBaseline(labels[i]);
            for (size_t cls = 0; cls < direction_output_size; ++cls)
            {
                const double target = (static_cast<int>(cls) == labels[i]) ? 1.0 : 0.0;
                const double g = sampleWeight * (probs[cls] - target);
                double* w = weights.data() + cls * dims;
                for (size_t j = 0; j < dims; ++j)
                    w[j] -= kGridBaselineLearningRate * g * static_cast<double>(x[j]);
                bias[cls] -= kGridBaselineLearningRate * g;
            }
        }
    }
}

int RunLabelGridDiagnostic3Class(const std::string& fromDate, const std::string& toDate)
{
    const std::array<size_t, 7> horizons {1, 2, 4, 8, 12, 16, 24};
    const std::array<float, 7> thresholds {0.0004f, 0.0006f, 0.0008f, 0.0010f, 0.0012f, 0.0015f, 0.0020f};
    const size_t maxHorizon = *std::max_element(horizons.begin(), horizons.end());

    pqxx::connection c_forex { ForexDbConnectionString() };
    pqxx::work w_forex { c_forex };
    const auto tables = EA::MarketData::DiscoverRawPriceTables(w_forex);
    if (tables.empty())
    {
        std::cerr << "LABEL_GRID_ERROR,no_rmp_tables=1" << std::endl;
        return 1;
    }

    const std::string rawPriceTableName{ tables.front() };
    const auto marketData = EA::MarketData::LoadCandlesticks(
        w_forex, {rawPriceTableName, fromDate, fromDate, toDate,
                  rawPriceTableName + "_label_grid_candlestick_stream", false});
    const std::string& query = marketData.query;
    Tensor tensor{ rawPriceTableName };
    for (const auto& row : marketData.rows) tensor.Add(row);

    const size_t rows = tensor.RowCount();
    const size_t baseFeatureCount = rows ? static_cast<size_t>((*tensor.begin()).Shape()[1]) : 0;
    const size_t modelFeatureCount = baseFeatureCount + BaselineReturnFeatureCount;
    auto examples = BuildGridExamples(tensor, maxHorizon);
    if (examples.size() < 10)
    {
        std::cerr << "LABEL_GRID_ERROR,too_few_examples=" << examples.size() << std::endl;
        return 1;
    }

    const size_t trainCount = std::max<size_t>(1, static_cast<size_t>(std::floor(static_cast<double>(examples.size()) * 0.8)));
    const size_t valCount = examples.size() - trainCount;
    if (valCount == 0)
    {
        std::cerr << "LABEL_GRID_ERROR,no_validation_examples=1" << std::endl;
        return 1;
    }

    auto features = BuildGridLastStepFeatureMatrix(tensor, examples, modelFeatureCount, baseFeatureCount);

    std::cout << std::setprecision(15);
    std::cout << "LABEL_GRID_CONFIG"
              << ",model=multinomial_logistic_regression_sgd"
              << ",feature_view=last_window_row_model_features"
              << ",source=postgresql_candlestick_query"
              << ",table=" << rawPriceTableName
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",query=\"" << query << "\""
              << ",rows=" << rows
              << ",examples=" << examples.size()
              << ",split=chronological_80_20_common_max_horizon_eligible_lstm_batch_window_order"
              << ",train_examples=" << trainCount
              << ",validation_examples=" << valCount
              << ",base_feature_cols=" << baseFeatureCount
              << ",return_feature_cols=" << BaselineReturnFeatureCount
              << ",model_feature_cols=" << modelFeatureCount
              << ",window_size=" << window_size
              << ",max_prediction_horizon=" << maxHorizon
              << ",baseline_epochs=3"
              << ",baseline_learning_rate=0.0001"
              << ",feature_clamp_min=-10"
              << ",feature_clamp_max=10"
              << ",class_weight_down=" << kClassWeightDown
              << ",class_weight_neutral=" << kClassWeightNeutral
              << ",class_weight_up=" << kClassWeightUp
              << std::endl;

    std::vector<GridResultSummary> summaries;
    summaries.reserve(horizons.size() * thresholds.size());

    for (size_t horizon : horizons)
    {
        for (float threshold : thresholds)
        {
            GridHitStats hitStats;
            auto labels = BuildGridLabelsAndStats(tensor, examples, horizon, threshold, hitStats);
            std::vector<double> weights(direction_output_size * modelFeatureCount, 0.0);
            std::array<double, direction_output_size> bias {0.0, 0.0, 0.0};
            TrainGridLogReg(features, labels, trainCount, modelFeatureCount, weights, bias);

            const auto trainMetrics = EvaluateGridLogReg(features, labels, 0, trainCount, modelFeatureCount, weights, bias);
            const auto valMetrics = EvaluateGridLogReg(features, labels, trainCount, examples.size(), modelFeatureCount, weights, bias);
            const double valAcc = BaselineAccuracy(valMetrics);
            const double valNeutral = BaselineNeutralOnlyAccuracy(valMetrics);
            const double macroF1 = BaselineMacroF1(valMetrics);
            const double balancedAccuracy = BaselineBalancedAccuracy(valMetrics);

            std::cout << "LABEL_GRID_COMBO"
                      << ",horizon=" << horizon
                      << ",threshold_logret=" << threshold
                      << std::endl;
            PrintGridHist("LABEL_GRID_TRAIN_HIST", horizon, threshold, trainMetrics);
            PrintGridHist("LABEL_GRID_VALIDATION_HIST", horizon, threshold, valMetrics);
            std::cout << "LABEL_GRID_HIT_STATS"
                      << ",horizon=" << horizon
                      << ",threshold_logret=" << threshold
                      << ",scope=all"
                      << ",total=" << hitStats.total
                      << ",both_hit=" << hitStats.bothHit
                      << ",both_hit_rate=" << (hitStats.total ? static_cast<double>(hitStats.bothHit) / static_cast<double>(hitStats.total) : 0.0)
                      << ",up_hit_count=" << hitStats.upHit
                      << ",avg_bars_to_up_hit=" << (hitStats.upHit ? hitStats.upOffsetSum / static_cast<double>(hitStats.upHit) : 0.0)
                      << ",down_hit_count=" << hitStats.downHit
                      << ",avg_bars_to_down_hit=" << (hitStats.downHit ? hitStats.downOffsetSum / static_cast<double>(hitStats.downHit) : 0.0)
                      << std::endl;
            PrintGridMetrics("LABEL_GRID_TRAIN", horizon, threshold, trainMetrics);
            PrintGridMetrics("LABEL_GRID_VALIDATION", horizon, threshold, valMetrics);

            summaries.push_back(GridResultSummary{
                horizon,
                threshold,
                valAcc,
                valNeutral,
                valAcc - valNeutral,
                macroF1,
                balancedAccuracy
            });
        }
    }

    auto byMacroF1 = summaries;
    std::sort(byMacroF1.begin(), byMacroF1.end(), [](const auto& a, const auto& b)
    {
        if (a.macroF1 != b.macroF1) return a.macroF1 > b.macroF1;
        return a.balancedAccuracy > b.balancedAccuracy;
    });
    auto byBalancedAccuracy = summaries;
    std::sort(byBalancedAccuracy.begin(), byBalancedAccuracy.end(), [](const auto& a, const auto& b)
    {
        if (a.balancedAccuracy != b.balancedAccuracy) return a.balancedAccuracy > b.balancedAccuracy;
        return a.macroF1 > b.macroF1;
    });

    const size_t topN = std::min<size_t>(5, summaries.size());
    for (size_t i = 0; i < topN; ++i)
    {
        const auto& s = byMacroF1[i];
        std::cout << "LABEL_GRID_TOP_MACRO_F1"
                  << ",rank=" << (i + 1)
                  << ",horizon=" << s.horizon
                  << ",threshold_logret=" << s.threshold
                  << ",validation_accuracy=" << s.validationAccuracy
                  << ",neutral_only_accuracy=" << s.validationNeutralOnly
                  << ",delta_vs_neutral=" << s.deltaVsNeutral
                  << ",macro_f1=" << s.macroF1
                  << ",balanced_accuracy=" << s.balancedAccuracy
                  << std::endl;
    }
    for (size_t i = 0; i < topN; ++i)
    {
        const auto& s = byBalancedAccuracy[i];
        std::cout << "LABEL_GRID_TOP_BALANCED_ACCURACY"
                  << ",rank=" << (i + 1)
                  << ",horizon=" << s.horizon
                  << ",threshold_logret=" << s.threshold
                  << ",validation_accuracy=" << s.validationAccuracy
                  << ",neutral_only_accuracy=" << s.validationNeutralOnly
                  << ",delta_vs_neutral=" << s.deltaVsNeutral
                  << ",macro_f1=" << s.macroF1
                  << ",balanced_accuracy=" << s.balancedAccuracy
                  << std::endl;
    }

    return 0;
}

std::vector<BaselineExample> BuildBaselineExamples(const Tensor& tensor)
{
    std::vector<BaselineExample> examples;
    tensor.ForEachBatch([&](auto b)
    {
        const size_t batchStart = static_cast<size_t>(b.begin() - tensor.begin());
        const size_t batchRows = static_cast<size_t>(b.end() - b.begin());
        for (size_t localStart = 0;
             localStart + window_size + prediction_horizon - 1 < batchRows;
             ++localStart)
        {
            const size_t globalStart = batchStart + localStart;
            const auto label = BuildLookaheadClassInfo(
                tensor,
                tensor.begin() + static_cast<std::ptrdiff_t>(globalStart)).assignedClass;
            examples.push_back(BaselineExample{
                batchStart,
                batchRows,
                localStart,
                globalStart,
                label
            });
        }
    });
    return examples;
}

int RunBaseline3Class(const std::string& fromDate, const std::string& toDate)
{
    pqxx::connection c_forex { ForexDbConnectionString() };
    pqxx::work w_forex { c_forex };
    const auto tables = EA::MarketData::DiscoverRawPriceTables(w_forex);
    if (tables.empty())
    {
        std::cerr << "BASELINE_3CLASS_ERROR,no_rmp_tables=1" << std::endl;
        return 1;
    }

    const std::string rawPriceTableName{ tables.front() };
    const auto marketData = EA::MarketData::LoadCandlesticks(
        w_forex, {rawPriceTableName, fromDate, fromDate, toDate,
                  rawPriceTableName + "_baseline_candlestick_stream", false});
    const std::string& query = marketData.query;
    Tensor tensor{ rawPriceTableName };
    for (const auto& row : marketData.rows) tensor.Add(row);

    const size_t rows = tensor.RowCount();
    const size_t baseFeatureCount = rows ? static_cast<size_t>((*tensor.begin()).Shape()[1]) : 0;
    const size_t modelFeatureCount = baseFeatureCount + BaselineReturnFeatureCount;
    const size_t dims = window_size * modelFeatureCount;
    auto examples = BuildBaselineExamples(tensor);
    if (examples.size() < 10)
    {
        std::cerr << "BASELINE_3CLASS_ERROR,too_few_examples=" << examples.size() << std::endl;
        return 1;
    }

    const size_t trainCount = std::max<size_t>(1, static_cast<size_t>(std::floor(static_cast<double>(examples.size()) * 0.8)));
    const size_t valCount = examples.size() - trainCount;
    if (valCount == 0)
    {
        std::cerr << "BASELINE_3CLASS_ERROR,no_validation_examples=1" << std::endl;
        return 1;
    }

    std::cout << std::setprecision(15);
    std::cout << "BASELINE_3CLASS_CONFIG"
              << ",model=multinomial_logistic_regression_sgd"
              << ",source=postgresql_candlestick_query"
              << ",table=" << rawPriceTableName
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",query=\"" << query << "\""
              << ",rows=" << rows
              << ",examples=" << examples.size()
              << ",split=chronological_80_20_lstm_batch_window_order"
              << ",train_examples=" << trainCount
              << ",validation_examples=" << valCount
              << ",base_feature_cols=" << baseFeatureCount
              << ",return_feature_cols=" << BaselineReturnFeatureCount
              << ",model_feature_cols=" << modelFeatureCount
              << ",flattened_window_dims=" << dims
              << ",window_size=" << window_size
              << ",prediction_horizon=" << prediction_horizon
              << ",threshold_logret=" << c_next_threshold
              << ",feature_clamp_min=-10"
              << ",feature_clamp_max=10"
              << ",class_weight_down=" << kClassWeightDown
              << ",class_weight_neutral=" << kClassWeightNeutral
              << ",class_weight_up=" << kClassWeightUp
              << std::endl;

    constexpr size_t kBaselineEpochs = 3;
    constexpr double kBaselineLearningRate = 1.0e-4;
    std::vector<double> weights(direction_output_size * dims, 0.0);
    std::array<double, direction_output_size> bias {0.0, 0.0, 0.0};
    std::vector<float> x;

    for (size_t epoch = 0; epoch < kBaselineEpochs; ++epoch)
    {
        double weightedLossSum = 0.0;
        double weightSum = 0.0;
        size_t correct = 0;
        for (size_t i = 0; i < trainCount; ++i)
        {
            const auto& ex = examples[i];
            FillBaselineFeatureVector(tensor, ex, modelFeatureCount, baseFeatureCount, x);

            std::array<double, direction_output_size> logits = bias;
            for (size_t cls = 0; cls < direction_output_size; ++cls)
            {
                const double* w = weights.data() + cls * dims;
                double z = logits[cls];
                for (size_t j = 0; j < dims; ++j)
                    z += w[j] * static_cast<double>(x[j]);
                logits[cls] = z;
            }
            const auto probs = BaselineSoftmax(logits);
            const int pred = (probs[0] > probs[1] && probs[0] > probs[2]) ? 0 :
                             ((probs[2] > probs[1] && probs[2] > probs[0]) ? 2 : 1);
            if (pred == ex.label) ++correct;

            const double sampleWeight = ClassWeightForBaseline(ex.label);
            weightedLossSum += sampleWeight * (-std::log(std::max(1.0e-12, probs[static_cast<size_t>(ex.label)])));
            weightSum += sampleWeight;

            for (size_t cls = 0; cls < direction_output_size; ++cls)
            {
                const double target = (static_cast<int>(cls) == ex.label) ? 1.0 : 0.0;
                const double g = sampleWeight * (probs[cls] - target);
                double* w = weights.data() + cls * dims;
                for (size_t j = 0; j < dims; ++j)
                    w[j] -= kBaselineLearningRate * g * static_cast<double>(x[j]);
                bias[cls] -= kBaselineLearningRate * g;
            }
        }

        std::cout << "BASELINE_3CLASS_TRAIN_EPOCH"
                  << ",epoch=" << (epoch + 1)
                  << ",weighted_loss=" << (weightSum > 0.0 ? weightedLossSum / weightSum : 0.0)
                  << ",stream_train_accuracy=" << (trainCount ? static_cast<double>(correct) / static_cast<double>(trainCount) : 0.0)
                  << std::endl;
    }

    auto trainMetrics = EvaluateBaseline(tensor, examples, 0, trainCount, modelFeatureCount, baseFeatureCount, weights, bias);
    auto valMetrics = EvaluateBaseline(tensor, examples, trainCount, examples.size(), modelFeatureCount, baseFeatureCount, weights, bias);
    PrintBaselineMetrics("BASELINE_3CLASS_TRAIN", trainMetrics);
    PrintBaselineMetrics("BASELINE_3CLASS_VALIDATION", valMetrics);

    const double valAccuracy = valMetrics.total ? static_cast<double>(valMetrics.correct) / static_cast<double>(valMetrics.total) : 0.0;
    const double valNeutralOnly = valMetrics.total ? static_cast<double>(valMetrics.hist[1]) / static_cast<double>(valMetrics.total) : 0.0;
    std::cout << "BASELINE_3CLASS_COMPARE_NEUTRAL"
              << ",validation_accuracy=" << valAccuracy
              << ",neutral_only_accuracy=" << valNeutralOnly
              << ",delta_accuracy=" << (valAccuracy - valNeutralOnly)
              << std::endl;
    return 0;
}

std::vector<float> BuildHiddenStateFeatureMatrix(EA::LSTM& lstm,
                                                 const Tensor& tensor,
                                                 const std::vector<BaselineExample>& examples,
                                                 const std::vector<size_t>& indices,
                                                 size_t modelFeatureCount,
                                                 size_t baseFeatureCount)
{
    std::vector<float> features(indices.size() * hidden_size);
    auto paramHost = MetaNN::Evaluate(lstm.param);
    auto biasHost = MetaNN::Evaluate(lstm.bias);
    auto sigmoid = [](double z) -> double
    {
        if (z >= 0.0)
        {
            const double e = std::exp(-z);
            return 1.0 / (1.0 + e);
        }
        const double e = std::exp(z);
        return e / (1.0 + e);
    };

    std::vector<double> h(hidden_size, 0.0);
    std::vector<double> c(hidden_size, 0.0);
    std::vector<double> hNext(hidden_size, 0.0);
    std::vector<double> cNext(hidden_size, 0.0);
    std::vector<float> x(modelFeatureCount, 0.0f);

    for (size_t outIdx = 0; outIdx < indices.size(); ++outIdx)
    {
        const size_t i = indices[outIdx];
        const auto& ex = examples[i];
        std::fill(h.begin(), h.end(), 0.0);
        std::fill(c.begin(), c.end(), 0.0);

        for (size_t wr = 0; wr < window_size; ++wr)
        {
            for (size_t col = 0; col < modelFeatureCount; ++col)
                x[col] = BaselineFeatureAt(tensor, ex, wr, col, baseFeatureCount);

            for (size_t unit = 0; unit < hidden_size; ++unit)
            {
                double zi = biasHost(0, unit);
                double zf = biasHost(0, hidden_size + unit);
                double zg = biasHost(0, 2 * hidden_size + unit);
                double zo = biasHost(0, 3 * hidden_size + unit);

                for (size_t col = 0; col < modelFeatureCount; ++col)
                {
                    const double xv = static_cast<double>(x[col]);
                    zi += xv * static_cast<double>(paramHost(col, unit));
                    zf += xv * static_cast<double>(paramHost(col, hidden_size + unit));
                    zg += xv * static_cast<double>(paramHost(col, 2 * hidden_size + unit));
                    zo += xv * static_cast<double>(paramHost(col, 3 * hidden_size + unit));
                }
                for (size_t prev = 0; prev < hidden_size; ++prev)
                {
                    const size_t row = modelFeatureCount + prev;
                    const double hv = h[prev];
                    zi += hv * static_cast<double>(paramHost(row, unit));
                    zf += hv * static_cast<double>(paramHost(row, hidden_size + unit));
                    zg += hv * static_cast<double>(paramHost(row, 2 * hidden_size + unit));
                    zo += hv * static_cast<double>(paramHost(row, 3 * hidden_size + unit));
                }

                const double gi = sigmoid(zi);
                const double gf = sigmoid(zf);
                const double gg = std::tanh(zg);
                const double go = sigmoid(zo);
                cNext[unit] = gf * c[unit] + gi * gg;
                hNext[unit] = go * std::tanh(cNext[unit]);
            }

            h.swap(hNext);
            c.swap(cNext);
        }

        for (size_t j = 0; j < hidden_size; ++j)
            features[outIdx * hidden_size + j] = static_cast<float>(h[j]);
    }
    return features;
}

std::vector<size_t> BuildHiddenDiagnosticIndices(const std::vector<int>& labels,
                                                 size_t beginIdx,
                                                 size_t endIdx,
                                                 size_t perClassLimit)
{
    std::array<size_t, direction_output_size> used {0, 0, 0};
    std::vector<size_t> indices;
    indices.reserve(perClassLimit * direction_output_size);
    for (size_t i = beginIdx; i < endIdx; ++i)
    {
        const int cls = labels[i];
        if (cls < 0 || cls >= static_cast<int>(direction_output_size)) continue;
        const size_t c = static_cast<size_t>(cls);
        if (used[c] >= perClassLimit) continue;
        ++used[c];
        indices.push_back(i);
        if (used[0] >= perClassLimit && used[1] >= perClassLimit && used[2] >= perClassLimit)
            break;
    }
    return indices;
}

int RunFeatureTrainability3Class(const std::string& fromDate, const std::string& toDate)
{
    pqxx::connection c_forex { ForexDbConnectionString() };
    pqxx::work w_forex { c_forex };
    const auto tables = EA::MarketData::DiscoverRawPriceTables(w_forex);
    if (tables.empty())
    {
        std::cerr << "FEATURE_TRAINABILITY_ERROR,no_rmp_tables=1" << std::endl;
        return 1;
    }

    const std::string rawPriceTableName{ tables.front() };
    const auto marketData = EA::MarketData::LoadCandlesticks(
        w_forex, {rawPriceTableName, fromDate, fromDate, toDate,
                  rawPriceTableName + "_feature_trainability_candlestick_stream", false});
    const std::string& query = marketData.query;
    Tensor tensor{ rawPriceTableName };
    for (const auto& row : marketData.rows) tensor.Add(row);

    const size_t rows = tensor.RowCount();
    const size_t baseFeatureCount = rows ? static_cast<size_t>((*tensor.begin()).Shape()[1]) : 0;
    const size_t modelFeatureCount = baseFeatureCount + BaselineReturnFeatureCount;
    const size_t flatDims = window_size * modelFeatureCount;
    auto examples = BuildBaselineExamples(tensor);
    if (examples.size() < 10)
    {
        std::cerr << "FEATURE_TRAINABILITY_ERROR,too_few_examples=" << examples.size() << std::endl;
        return 1;
    }

    const size_t trainCount = std::max<size_t>(1, static_cast<size_t>(std::floor(static_cast<double>(examples.size()) * 0.8)));
    const size_t valCount = examples.size() - trainCount;
    if (valCount == 0)
    {
        std::cerr << "FEATURE_TRAINABILITY_ERROR,no_validation_examples=1" << std::endl;
        return 1;
    }

    std::vector<int> labels;
    labels.reserve(examples.size());
    for (const auto& ex : examples) labels.push_back(ex.label);
    constexpr size_t kHiddenDiagPerClassLimit = 256;
    const auto hiddenDiagIndices = BuildHiddenDiagnosticIndices(labels, trainCount, examples.size(), kHiddenDiagPerClassLimit);
    std::vector<int> hiddenDiagLabels;
    hiddenDiagLabels.reserve(hiddenDiagIndices.size());
    for (size_t idx : hiddenDiagIndices)
        hiddenDiagLabels.push_back(labels[idx]);

    std::cout << std::setprecision(15);
    std::cout << "FEATURE_TRAINABILITY_CONFIG"
              << ",source=postgresql_candlestick_query"
              << ",table=" << rawPriceTableName
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",query=\"" << query << "\""
              << ",rows=" << rows
              << ",examples=" << examples.size()
              << ",split=chronological_80_20_lstm_batch_window_order"
              << ",train_examples=" << trainCount
              << ",validation_examples=" << valCount
              << ",base_feature_cols=" << baseFeatureCount
              << ",return_feature_cols=" << BaselineReturnFeatureCount
              << ",model_feature_cols=" << modelFeatureCount
              << ",flat_window_dims=" << flatDims
              << ",window_size=" << window_size
              << ",prediction_horizon=" << prediction_horizon
              << ",threshold_logret=" << c_next_threshold
              << ",class_weight_down=" << kClassWeightDown
              << ",class_weight_neutral=" << kClassWeightNeutral
              << ",class_weight_up=" << kClassWeightUp
              << ",baseline_sample_weighting=unweighted"
              << ",hidden_diag_per_class_limit=" << kHiddenDiagPerClassLimit
              << ",hidden_diag_examples=" << hiddenDiagIndices.size()
              << std::endl;
    std::cout << "FEATURE_TRAINABILITY_TREE_BACKEND"
              << ",xgboost_linked=0"
              << ",lightgbm_linked=0"
              << ",sklearn_available_in_binary=0"
              << ",fallback=random_stump_forest"
              << std::endl;

    auto lastFeatures = BuildLastStepFeatureMatrix(tensor, examples, modelFeatureCount, baseFeatureCount);
    auto flatFeatures = BuildFlatWindowFeatureMatrix(tensor, examples, modelFeatureCount, baseFeatureCount);

    PrintSeparabilityStats("FEATURE_SEPARABILITY",
                           "raw_last_step_model_features",
                           "train",
                           modelFeatureCount,
                           ComputeSeparability(lastFeatures, labels, 0, trainCount, modelFeatureCount));
    PrintSeparabilityStats("FEATURE_SEPARABILITY",
                           "raw_last_step_model_features",
                           "validation",
                           modelFeatureCount,
                           ComputeSeparability(lastFeatures, labels, trainCount, examples.size(), modelFeatureCount));
    PrintSeparabilityStats("FEATURE_SEPARABILITY",
                           "raw_flat_window_model_features",
                           "validation",
                           flatDims,
                           ComputeSeparability(flatFeatures, labels, trainCount, examples.size(), flatDims));

    std::vector<double> logRegWeights;
    std::array<double, direction_output_size> logRegBias {0.0, 0.0, 0.0};
    TrainFeatureLogReg(flatFeatures, labels, trainCount, flatDims, 3, 1.0e-4, logRegWeights, logRegBias);
    PrintBaselineMetrics("FEATURE_BASELINE_LOGREG_FLAT_TRAIN",
                         EvaluatePredictions(labels, 0, trainCount,
                                             PredictFeatureLogReg(flatFeatures, 0, trainCount, flatDims, logRegWeights, logRegBias)));
    PrintBaselineMetrics("FEATURE_BASELINE_LOGREG_FLAT_VALIDATION",
                         EvaluatePredictions(labels, trainCount, examples.size(),
                                             PredictFeatureLogReg(flatFeatures, trainCount, examples.size(), flatDims, logRegWeights, logRegBias)));

    std::vector<double> lastLogRegWeights;
    std::array<double, direction_output_size> lastLogRegBias {0.0, 0.0, 0.0};
    TrainFeatureLogReg(lastFeatures, labels, trainCount, modelFeatureCount, 5, 1.0e-4, lastLogRegWeights, lastLogRegBias);
    PrintBaselineMetrics("FEATURE_BASELINE_LOGREG_LAST_TRAIN",
                         EvaluatePredictions(labels, 0, trainCount,
                                             PredictFeatureLogReg(lastFeatures, 0, trainCount, modelFeatureCount, lastLogRegWeights, lastLogRegBias)));
    PrintBaselineMetrics("FEATURE_BASELINE_LOGREG_LAST_VALIDATION",
                         EvaluatePredictions(labels, trainCount, examples.size(),
                                             PredictFeatureLogReg(lastFeatures, trainCount, examples.size(), modelFeatureCount, lastLogRegWeights, lastLogRegBias)));

    ShallowMlp mlp;
    TrainShallowMlp(lastFeatures, labels, trainCount, modelFeatureCount, mlp);
    PrintBaselineMetrics("FEATURE_BASELINE_MLP_LAST_TRAIN",
                         EvaluatePredictions(labels, 0, trainCount,
                                             PredictShallowMlp(lastFeatures, 0, trainCount, mlp)));
    PrintBaselineMetrics("FEATURE_BASELINE_MLP_LAST_VALIDATION",
                         EvaluatePredictions(labels, trainCount, examples.size(),
                                             PredictShallowMlp(lastFeatures, trainCount, examples.size(), mlp)));

    const auto forest = TrainRandomStumpForest(lastFeatures, labels, trainCount, modelFeatureCount);
    PrintBaselineMetrics("FEATURE_BASELINE_RANDOM_STUMP_FOREST_LAST_TRAIN",
                         EvaluatePredictions(labels, 0, trainCount,
                                             PredictRandomStumpForest(lastFeatures, 0, trainCount, modelFeatureCount, forest)));
    PrintBaselineMetrics("FEATURE_BASELINE_RANDOM_STUMP_FOREST_LAST_VALIDATION",
                         EvaluatePredictions(labels, trainCount, examples.size(),
                                             PredictRandomStumpForest(lastFeatures, trainCount, examples.size(), modelFeatureCount, forest)));

    EA::LSTM lstm { tensor, hidden_size, 1, 0,
                    EA::LSTM::TargetType::UpNeutralDownReturn };
    auto hiddenBefore = BuildHiddenStateFeatureMatrix(lstm, tensor, examples, hiddenDiagIndices, modelFeatureCount, baseFeatureCount);
    PrintSeparabilityStats("FEATURE_HIDDEN_SEPARABILITY",
                           "lstm_last_hidden_state",
                           "validation_before_training",
                           hidden_size,
                           ComputeSeparability(hiddenBefore, hiddenDiagLabels, 0, hiddenDiagLabels.size(), hidden_size));

    const bool previousHiddenDiagSuppression = EA::LSTM::suppressPhase3HiddenGeometryDiagnostics;
    EA::LSTM::suppressPhase3HiddenGeometryDiagnostics = true;
    std::ofstream diagnosticNullStream("/dev/null");
    auto* previousCoutBuffer = std::cout.rdbuf(diagnosticNullStream.rdbuf());
    tensor.ForEachBatch([&](auto b)
    {
        lstm.CalculateBatch(b, 0);
    });
    std::cout.rdbuf(previousCoutBuffer);
    EA::LSTM::suppressPhase3HiddenGeometryDiagnostics = previousHiddenDiagSuppression;
    EA::LSTM::PrintAndResetEpochBuckets();

    auto hiddenAfter = BuildHiddenStateFeatureMatrix(lstm, tensor, examples, hiddenDiagIndices, modelFeatureCount, baseFeatureCount);
    PrintSeparabilityStats("FEATURE_HIDDEN_SEPARABILITY",
                           "lstm_last_hidden_state",
                           "validation_after_epoch1_training",
                           hidden_size,
                           ComputeSeparability(hiddenAfter, hiddenDiagLabels, 0, hiddenDiagLabels.size(), hidden_size));

    return 0;
}

} // namespace

std::optional<int> TryRun(int argc, const char* argv[])
{
    if (argc >= 4 && std::string(argv[1]) == "--baseline-3class")
        return RunBaseline3Class(argv[2], argv[3]);

    if (argc >= 4 && std::string(argv[1]) == "--label-grid-3class")
        return RunLabelGridDiagnostic3Class(argv[2], argv[3]);

    if (argc >= 4 && std::string(argv[1]) == "--feature-trainability-3class")
        return RunFeatureTrainability3Class(argv[2], argv[3]);

    return std::nullopt;
}

} // namespace EA::LegacyDiagnosticCli
