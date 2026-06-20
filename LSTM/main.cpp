//
//  main.cpp
//  LSTM
//
//  Created by Vincent Predoehl on 11/27/25.
//  Copyright © 2025 Vincent Predoehl. All rights reserved.
//

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <random>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <pqxx/pqxx>

#include <device_tags.h>
#include <tensor.h>
#include <permute.h>
#include <conv.h>

#include "db_cursor_iterator.hpp"
#include "Tensor.hpp"
#include "LSTM.hpp"
#include "PgModelIO.hpp"
#include "BuildConfig.hpp"
#include "TargetLabel.hpp"

#ifndef EARLY_STOP_PATIENCE
#define EARLY_STOP_PATIENCE 10
#endif

#include <MetaNN/operation/math/sigmoid.h>
#include <MetaNN/operation/math/tanh.h>
#include <MetaNN/operation/tensor/reshape.h>
#include <MetaNN/operation/tensor/slice.h>
#include "scalable_tensor.h"

namespace
{
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

constexpr size_t BaselineReturnFeatureCount =
    static_cast<size_t>(LSTM_RET_HORIZON_1) +
    static_cast<size_t>(LSTM_RET_HORIZON_4) +
    static_cast<size_t>(LSTM_RET_HORIZON_8) +
    static_cast<size_t>(LSTM_RET_HORIZON_16);

const char* GateStateModeLabel()
{
    switch (LSTM_GATESTATE_MODE)
    {
        case 0: return "cpu_reference";
        case 1: return "cpu_validate_metal";
        case 2: return "metal_fused";
        default: return "unknown";
    }
}

bool gRuntimeInferenceMode = inference_only;

const char* CurrentRangeKindLabel()
{
    return gRuntimeInferenceMode ? "inference" : "train";
}

struct TrainConfigMeta
{
    int schemaVersion = DBIO::PgModelIO::kTrainConfigMetaSchemaVersion;
    size_t predictionHorizon = static_cast<size_t>(prediction_horizon);
    float thresholdLogret = c_next_threshold;
    size_t windowSize = static_cast<size_t>(window_size);
    int labelRuleId = DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId;
    float classWeightDown = kClassWeightDown;
    float classWeightNeutral = kClassWeightNeutral;
    float classWeightUp = kClassWeightUp;
    size_t numLayers = static_cast<size_t>(num_layers);
    int normalizationVersion = normalization_version;
    std::optional<size_t> epochsTrained;
};

struct ModelConfigValidationResult
{
    std::optional<TrainConfigMeta> trainConfigMeta;
};

struct EvalLabelConfig
{
    size_t predictionHorizon = static_cast<size_t>(prediction_horizon);
    float thresholdLogret = c_next_threshold;
    size_t windowSize = static_cast<size_t>(window_size);
    int labelRuleId = DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId;
    const char* source = "runtime_defaults";
};

EvalLabelConfig RuntimeDefaultEvalLabelConfig()
{
    return {
        static_cast<size_t>(prediction_horizon),
        c_next_threshold,
        static_cast<size_t>(window_size),
        DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId,
        "runtime_defaults"
    };
}

EvalLabelConfig ModelTrainEvalLabelConfig(const TrainConfigMeta& meta)
{
    return {
        meta.predictionHorizon,
        meta.thresholdLogret,
        meta.windowSize,
        meta.labelRuleId,
        "model_train_config"
    };
}

EvalLabelConfig& ActiveEvalLabelConfig()
{
    static EvalLabelConfig config = RuntimeDefaultEvalLabelConfig();
    return config;
}

void UseRuntimeDefaultEvalLabelConfig()
{
    ActiveEvalLabelConfig() = RuntimeDefaultEvalLabelConfig();
}

void UseModelTrainEvalLabelConfig(const TrainConfigMeta& meta)
{
    ActiveEvalLabelConfig() = ModelTrainEvalLabelConfig(meta);
}

void PrintClassificationProofDiagnostics(const EA::LSTM& l,
                                         const Tensor& tensor,
                                         const std::string& fromDate,
                                         const std::string& toDate)
{
    if (l.targetType != EA::LSTM::TargetType::UpNeutralDownReturn)
        return;

    static bool printed = false;
    if (printed)
        return;
    printed = true;

    static_assert(direction_output_size == 3, "UpNeutralDownReturn expects exactly 3 output classes");
    LSTM_ASSERT(direction_output_size == 3,
                "PrintClassificationProofDiagnostics: output dimension must be 3 in classification mode");
    LSTM_ASSERT(l.returnHeadDirWeight.Shape()[1] == direction_output_size,
                "PrintClassificationProofDiagnostics: returnHeadDirWeight width mismatch");
    LSTM_ASSERT(l.returnHeadDirBias.Shape()[1] == direction_output_size,
                "PrintClassificationProofDiagnostics: returnHeadDirBias width mismatch");

    const auto& evalConfig = ActiveEvalLabelConfig();
    std::cout << "DIAG_CLASS_THRESHOLDS"
              << ",range_kind=" << CurrentRangeKindLabel()
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",threshold_logret=" << evalConfig.thresholdLogret
              << ",prediction_horizon=" << evalConfig.predictionHorizon
              << ",window_size=" << evalConfig.windowSize
              << ",label_rule_id=" << evalConfig.labelRuleId
              << ",config_source=" << evalConfig.source
              << ",neutral_rule=no_threshold_hit_within_horizon"
              << ",output_dim=" << direction_output_size
              << std::endl;
    std::cout << "DIAG_CLASS_LABEL_RULE"
              << ",training_label=lookahead_high_low_first_hit"
              << ",inference_eval_label=lookahead_high_low_first_hit"
              << std::endl;

    std::array<size_t, direction_output_size> hist {0, 0, 0};
    size_t windowCount = 0;
    size_t printedRows = 0;

    for (auto it = tensor.begin();
         it + static_cast<std::ptrdiff_t>(evalConfig.windowSize - 1 + evalConfig.predictionHorizon) < tensor.end();
         ++it)
    {
        const auto info = BuildLookaheadClassInfo(tensor,
                                                  it,
                                                  evalConfig.windowSize,
                                                  evalConfig.predictionHorizon,
                                                  evalConfig.thresholdLogret);
        ++hist[static_cast<size_t>(info.assignedClass)];
        ++windowCount;

        if (printedRows < 20)
        {
            std::cout << "DIAG_CLASS_ROW"
                      << ",idx=" << printedRows
                      << ",dt=" << info.dt
                      << ",target_dt=" << info.targetDt
                      << ",close_t=" << info.closeT
                      << ",target_close=" << info.targetClose
                      << ",delta_close=" << info.deltaClose
                      << ",terminal_logret=" << info.terminalLogReturn
                      << ",up_hit=" << static_cast<int>(info.upHit)
                      << ",down_hit=" << static_cast<int>(info.downHit)
                      << ",up_offset=" << info.upOffset
                      << ",down_offset=" << info.downOffset
                      << ",assigned_class=" << info.assignedClass
                      << std::endl;
            ++printedRows;
        }
    }

    const double denom = (windowCount > 0) ? static_cast<double>(windowCount) : 1.0;
    std::cout << "DIAG_CLASS_HIST"
              << ",range_kind=" << CurrentRangeKindLabel()
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",windows=" << windowCount
              << ",down=" << hist[0]
              << ",neutral=" << hist[1]
              << ",up=" << hist[2]
              << ",down_frac=" << (static_cast<double>(hist[0]) / denom)
              << ",neutral_frac=" << (static_cast<double>(hist[1]) / denom)
              << ",up_frac=" << (static_cast<double>(hist[2]) / denom)
              << std::endl;
}

struct Phase2ScalarStats
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

bool IsSaneFxPrice(double v)
{
    return std::isfinite(v) && v > 0.0 && v >= 0.2 && v <= 2.0;
}

void PrintPhase2ScalarLine(const char* label,
                           const std::string& rangeKind,
                           const std::string& fromDate,
                           const std::string& toDate,
                           const char* name,
                           const Phase2ScalarStats& s)
{
    std::cout << label
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",name=" << name
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

void PrintPhase2TensorDiagnostics(const EA::LSTM& l,
                                  const Tensor& tensor,
                                  const std::string& fromDate,
                                  const std::string& toDate)
{
    static bool printed = false;
    if (printed)
        return;
    printed = true;

    const std::string rangeKind = CurrentRangeKindLabel();
    const size_t rows = static_cast<size_t>(tensor.end() - tensor.begin());
    const size_t cols = (rows > 0) ? static_cast<size_t>((*tensor.begin()).Shape()[1]) : 0;
    constexpr size_t kDiagFeatureCols = 8;
    constexpr size_t kDiagWindowCount = 3;
    constexpr size_t kDiagBadRowLimit = 50;
    constexpr double kNearZeroStdThreshold = 1e-6;
    constexpr double kFeatureAbsMaxWarnThreshold = 50.0;
    constexpr double kRawFxLowerBound = 0.2;
    constexpr double kRawFxUpperBound = 2.0;

    std::cout << "DIAG_DATA_CONFIG"
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",source=postgresql_candlestick_query"
              << ",rows=" << rows
              << ",raw_open=1"
              << ",raw_close=1"
              << ",raw_high=1"
              << ",raw_low=1"
              << ",raw_volume=0"
              << ",raw_target=0"
              << std::endl;
    std::cout << "DIAG_FEATURE_CONFIG"
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",base_feature_cols=" << cols
              << ",feature_size_constant=" << feature_size
              << ",kFeatureScale=" << kFeatureScale
              << ",feature_uses_future_values=0"
              << ",feature_uses_target_column=0"
              << std::endl;
    std::cout << "DIAG_NORM_CONFIG"
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",target_use_zscore=" << (l.targetUseZScore ? 1 : 0)
              << ",target_mean=" << l.targetMean
              << ",target_std=" << l.targetStd
              << ",train_pre_lstm_nonfinite_fail=1"
              << ",train_pre_lstm_clamp=10"
              << ",classification_inference_pre_lstm_nonfinite_fail=1"
              << ",classification_inference_pre_lstm_clamp=0"
              << ",regression_inference_pre_lstm_nonfinite_fail=1"
              << ",regression_inference_pre_lstm_clamp=10"
              << std::endl;
    std::cout << "DIAG_DATA_WARN"
              << ",range_kind=" << rangeKind
              << ",kind=volume_not_ingested_in_active_postgresql_path"
              << std::endl;
    std::cout << "DIAG_DATA_WARN"
              << ",range_kind=" << rangeKind
              << ",kind=target_column_not_used_in_active_postgresql_path"
              << std::endl;

    Phase2ScalarStats rawOpenStats;
    Phase2ScalarStats rawCloseStats;
    Phase2ScalarStats rawHighStats;
    Phase2ScalarStats rawLowStats;
    Phase2ScalarStats featureGlobalStats;
    std::vector<Phase2ScalarStats> featureColStats(cols);
    std::vector<double> featureColAbsmax(cols, 0.0);
    std::vector<double> featureColAbsmaxValue(cols, 0.0);
    std::vector<size_t> featureColAbsmaxRow(cols, 0);
    size_t badRawRowCount = 0;
    size_t badRawRowPrinted = 0;
    size_t featureSpikeCount = 0;
    size_t featureSpikePrinted = 0;
    double globalFeatureAbsmax = 0.0;
    double globalFeatureAbsmaxValue = 0.0;
    size_t globalFeatureAbsmaxCol = 0;
    size_t globalFeatureAbsmaxRow = 0;

    size_t rowIdx = 0;
    for (auto it = tensor.begin(); it != tensor.end(); ++it, ++rowIdx)
    {
        const double rawOpen = static_cast<double>(tensor.RawOpenAtIterator(it));
        const double rawClose = static_cast<double>(tensor.RawCloseAtIterator(it));
        const double rawHigh = static_cast<double>(tensor.RawHighAtIterator(it));
        const double rawLow = static_cast<double>(tensor.RawLowAtIterator(it));
        const auto dt = tensor.RawTimeAtIterator(it);

        rawOpenStats.add(rawOpen);
        rawCloseStats.add(rawClose);
        rawHighStats.add(rawHigh);
        rawLowStats.add(rawLow);

        const bool badOpen = !IsSaneFxPrice(rawOpen);
        const bool badClose = !IsSaneFxPrice(rawClose);
        const bool badHigh = !IsSaneFxPrice(rawHigh);
        const bool badLow = !IsSaneFxPrice(rawLow);
        const bool highLtLow = std::isfinite(rawHigh) && std::isfinite(rawLow) && rawHigh < rawLow;
        if (badOpen || badClose || badHigh || badLow || highLtLow)
        {
            ++badRawRowCount;
            if (badRawRowPrinted < kDiagBadRowLimit)
            {
                std::cout << "DIAG_DATA_BAD_ROW"
                          << ",range_kind=" << rangeKind
                          << ",row=" << rowIdx
                          << ",dt=" << dt
                          << ",open=" << rawOpen
                          << ",close=" << rawClose
                          << ",high=" << rawHigh
                          << ",low=" << rawLow
                          << ",bad_open=" << static_cast<int>(badOpen)
                          << ",bad_close=" << static_cast<int>(badClose)
                          << ",bad_high=" << static_cast<int>(badHigh)
                          << ",bad_low=" << static_cast<int>(badLow)
                          << ",high_lt_low=" << static_cast<int>(highLtLow)
                          << ",sane_lower_bound=" << kRawFxLowerBound
                          << ",sane_upper_bound=" << kRawFxUpperBound
                          << std::endl;
                ++badRawRowPrinted;
            }
        }

        auto low = MetaNN::LowerAccess(*it);
        const float* p = low.RawMemory();
        for (size_t c = 0; c < cols; ++c)
        {
            const double v = static_cast<double>(p[c]);
            const double absV = std::fabs(v);
            featureGlobalStats.add(v);
            featureColStats[c].add(v);
            if (absV > featureColAbsmax[c])
            {
                featureColAbsmax[c] = absV;
                featureColAbsmaxValue[c] = v;
                featureColAbsmaxRow[c] = rowIdx;
            }
            if (absV > globalFeatureAbsmax)
            {
                globalFeatureAbsmax = absV;
                globalFeatureAbsmaxValue = v;
                globalFeatureAbsmaxCol = c;
                globalFeatureAbsmaxRow = rowIdx;
            }
            if (absV > kFeatureAbsMaxWarnThreshold)
            {
                ++featureSpikeCount;
                if (featureSpikePrinted < kDiagBadRowLimit)
                {
                    std::cout << "DIAG_FEATURE_SPIKE"
                              << ",range_kind=" << rangeKind
                              << ",row=" << rowIdx
                              << ",dt=" << dt
                              << ",col=" << c
                              << ",value=" << v
                              << ",abs_value=" << absV
                              << ",open=" << rawOpen
                              << ",close=" << rawClose
                              << ",high=" << rawHigh
                              << ",low=" << rawLow
                              << std::endl;
                    ++featureSpikePrinted;
                }
            }
        }
    }

    PrintPhase2ScalarLine("DIAG_DATA_RAW_COL", rangeKind, fromDate, toDate, "open", rawOpenStats);
    PrintPhase2ScalarLine("DIAG_DATA_RAW_COL", rangeKind, fromDate, toDate, "close", rawCloseStats);
    PrintPhase2ScalarLine("DIAG_DATA_RAW_COL", rangeKind, fromDate, toDate, "high", rawHighStats);
    PrintPhase2ScalarLine("DIAG_DATA_RAW_COL", rangeKind, fromDate, toDate, "low", rawLowStats);
    std::cout << "DIAG_DATA_BAD_ROW_SUMMARY"
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",count=" << badRawRowCount
              << ",printed=" << badRawRowPrinted
              << ",limit=" << kDiagBadRowLimit
              << std::endl;

    size_t zeroVarFeatureCount = 0;
    for (const auto& colStat : featureColStats)
    {
        if (colStat.finiteCount > 0 && colStat.stddev() <= kNearZeroStdThreshold)
            ++zeroVarFeatureCount;
    }

    std::cout << "DIAG_FEATURE_BASE_GLOBAL"
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",rows=" << rows
              << ",cols=" << cols
              << ",finite=" << featureGlobalStats.finiteCount
              << ",nan=" << featureGlobalStats.nanCount
              << ",inf=" << featureGlobalStats.infCount
              << ",min=" << (featureGlobalStats.finiteCount ? featureGlobalStats.min : 0.0)
              << ",max=" << (featureGlobalStats.finiteCount ? featureGlobalStats.max : 0.0)
              << ",mean=" << featureGlobalStats.mean()
              << ",std=" << featureGlobalStats.stddev()
              << ",absmax=" << featureGlobalStats.absmax
              << ",zero_var_features=" << zeroVarFeatureCount
              << std::endl;

    for (size_t c = 0; c < std::min(cols, kDiagFeatureCols); ++c)
    {
        const auto& s = featureColStats[c];
        std::cout << "DIAG_FEATURE_BASE_COL"
                  << ",range_kind=" << rangeKind
                  << ",from=" << fromDate
                  << ",to=" << toDate
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

    std::cout << "DIAG_FEATURE_SPIKE_GLOBAL"
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",col=" << globalFeatureAbsmaxCol
              << ",row=" << globalFeatureAbsmaxRow
              << ",value=" << globalFeatureAbsmaxValue
              << ",absmax=" << globalFeatureAbsmax
              << ",dt=" << tensor.RawTimeAtIterator(tensor.begin() + static_cast<std::ptrdiff_t>(globalFeatureAbsmaxRow))
              << ",open=" << tensor.RawOpenAtIterator(tensor.begin() + static_cast<std::ptrdiff_t>(globalFeatureAbsmaxRow))
              << ",close=" << tensor.RawCloseAtIterator(tensor.begin() + static_cast<std::ptrdiff_t>(globalFeatureAbsmaxRow))
              << ",high=" << tensor.RawHighAtIterator(tensor.begin() + static_cast<std::ptrdiff_t>(globalFeatureAbsmaxRow))
              << ",low=" << tensor.RawLowAtIterator(tensor.begin() + static_cast<std::ptrdiff_t>(globalFeatureAbsmaxRow))
              << std::endl;

    for (size_t c = 0; c < cols; ++c)
    {
        if (featureColAbsmax[c] <= kFeatureAbsMaxWarnThreshold)
            continue;

        const auto it = tensor.begin() + static_cast<std::ptrdiff_t>(featureColAbsmaxRow[c]);
        std::cout << "DIAG_FEATURE_SPIKE_COL"
                  << ",range_kind=" << rangeKind
                  << ",from=" << fromDate
                  << ",to=" << toDate
                  << ",col=" << c
                  << ",row=" << featureColAbsmaxRow[c]
                  << ",value=" << featureColAbsmaxValue[c]
                  << ",absmax=" << featureColAbsmax[c]
                  << ",dt=" << tensor.RawTimeAtIterator(it)
                  << ",open=" << tensor.RawOpenAtIterator(it)
                  << ",close=" << tensor.RawCloseAtIterator(it)
                  << ",high=" << tensor.RawHighAtIterator(it)
                  << ",low=" << tensor.RawLowAtIterator(it)
                  << std::endl;
    }

    std::cout << "DIAG_FEATURE_SPIKE_SUMMARY"
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",count=" << featureSpikeCount
              << ",printed=" << featureSpikePrinted
              << ",limit=" << kDiagBadRowLimit
              << ",threshold=" << kFeatureAbsMaxWarnThreshold
              << std::endl;

    for (size_t w = 0; w < kDiagWindowCount && rows >= window_size && w + window_size <= rows; ++w)
    {
        Phase2ScalarStats windowStats;
        auto it = tensor.begin() + static_cast<std::ptrdiff_t>(w);
        for (size_t r = 0; r < window_size; ++r, ++it)
        {
            auto low = MetaNN::LowerAccess(*it);
            const float* p = low.RawMemory();
            for (size_t c = 0; c < cols; ++c)
                windowStats.add(static_cast<double>(p[c]));
        }

        std::cout << "DIAG_FEATURE_BASE_WINDOW"
                  << ",range_kind=" << rangeKind
                  << ",from=" << fromDate
                  << ",to=" << toDate
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

    size_t badLookaheadWindows = 0;
    size_t badLookaheadRows = 0;
    size_t badLookaheadPrinted = 0;
    for (size_t startRow = 0;
         startRow + window_size + prediction_horizon - 1 < rows;
         ++startRow)
    {
        const auto startIt = tensor.begin() + static_cast<std::ptrdiff_t>(startRow);
        const size_t lastRow = startRow + window_size - 1;
        const auto lastIt = tensor.begin() + static_cast<std::ptrdiff_t>(lastRow);
        bool windowHasBadLookahead = false;

        for (size_t lookahead = 1; lookahead <= prediction_horizon; ++lookahead)
        {
            const size_t futureRow = lastRow + lookahead;
            const auto futureIt = tensor.begin() + static_cast<std::ptrdiff_t>(futureRow);
            const double futureHigh = static_cast<double>(tensor.RawHighAtIterator(futureIt));
            const double futureLow = static_cast<double>(tensor.RawLowAtIterator(futureIt));
            const bool badFutureHigh = !IsSaneFxPrice(futureHigh);
            const bool badFutureLow = !IsSaneFxPrice(futureLow);
            const bool futureHighLtLow = std::isfinite(futureHigh) && std::isfinite(futureLow) && futureHigh < futureLow;
            if (!(badFutureHigh || badFutureLow || futureHighLtLow))
                continue;

            windowHasBadLookahead = true;
            ++badLookaheadRows;
            if (badLookaheadPrinted < kDiagBadRowLimit)
            {
                const auto info = BuildLookaheadClassInfo(tensor, startIt);
                std::cout << "DIAG_DATA_BAD_LOOKAHEAD"
                          << ",range_kind=" << rangeKind
                          << ",start_row=" << startRow
                          << ",start_dt=" << tensor.RawTimeAtIterator(lastIt)
                          << ",close_t=" << tensor.RawCloseAtIterator(lastIt)
                          << ",future_row=" << futureRow
                          << ",future_dt=" << tensor.RawTimeAtIterator(futureIt)
                          << ",future_high=" << futureHigh
                          << ",future_low=" << futureLow
                          << ",bad_high=" << static_cast<int>(badFutureHigh)
                          << ",bad_low=" << static_cast<int>(badFutureLow)
                          << ",high_lt_low=" << static_cast<int>(futureHighLtLow)
                          << ",assigned_class=" << info.assignedClass
                          << ",terminal_logret=" << info.terminalLogReturn
                          << ",up_hit=" << static_cast<int>(info.upHit)
                          << ",down_hit=" << static_cast<int>(info.downHit)
                          << ",up_offset=" << info.upOffset
                          << ",down_offset=" << info.downOffset
                          << std::endl;
                ++badLookaheadPrinted;
            }
        }

        if (windowHasBadLookahead)
            ++badLookaheadWindows;
    }

    std::cout << "DIAG_DATA_BAD_LOOKAHEAD_SUMMARY"
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",windows=" << badLookaheadWindows
              << ",future_rows=" << badLookaheadRows
              << ",printed=" << badLookaheadPrinted
              << ",limit=" << kDiagBadRowLimit
              << std::endl;

    if (featureGlobalStats.nanCount > 0 || featureGlobalStats.infCount > 0 ||
        rawOpenStats.nanCount > 0 || rawOpenStats.infCount > 0 ||
        rawCloseStats.nanCount > 0 || rawCloseStats.infCount > 0 ||
        rawHighStats.nanCount > 0 || rawHighStats.infCount > 0 ||
        rawLowStats.nanCount > 0 || rawLowStats.infCount > 0)
    {
        LSTM_ASSERT(false, "PrintPhase2TensorDiagnostics: non-finite raw data or base features detected before LSTM");
    }

    if (featureGlobalStats.absmax > kFeatureAbsMaxWarnThreshold)
    {
        std::cout << "DIAG_FEATURE_WARN"
                  << ",range_kind=" << rangeKind
                  << ",kind=absmax_exceeds_sane_threshold"
                  << ",threshold=" << kFeatureAbsMaxWarnThreshold
                  << ",absmax=" << featureGlobalStats.absmax
                  << std::endl;
    }

    if (zeroVarFeatureCount > 0)
    {
        std::cout << "DIAG_FEATURE_WARN"
                  << ",range_kind=" << rangeKind
                  << ",kind=near_zero_std_features"
                  << ",threshold=" << kNearZeroStdThreshold
                  << ",count=" << zeroVarFeatureCount
                  << std::endl;
    }
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
                                DataSet::const_iterator batchBegin,
                                size_t localRow,
                                size_t lookbackBars)
{
    if (localRow < lookbackBars)
        return 0.0f;

    const auto curIt = batchBegin + static_cast<std::ptrdiff_t>(localRow);
    const auto prevIt = batchBegin + static_cast<std::ptrdiff_t>(localRow - lookbackBars);
    const float curClose = tensor.RawCloseAtIterator(curIt);
    const float prevClose = tensor.RawCloseAtIterator(prevIt);
    if (!std::isfinite(curClose) || !std::isfinite(prevClose) || curClose <= 0.0f || prevClose <= 0.0f)
        return 0.0f;
    return std::log(curClose / prevClose);
}

float BaselineFeatureAt(const Tensor& tensor,
                        const BaselineExample& ex,
                        size_t windowRow,
                        size_t col,
                        size_t baseFeatureCount)
{
    const auto batchBegin = tensor.begin() + static_cast<std::ptrdiff_t>(ex.batchStart);
    const size_t localRow = ex.localStart + windowRow;
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
        if (retCol == written) v = BaselineLookbackLogReturn(tensor, batchBegin, localRow, 1) * EA::LSTM::kFeatScale;
        ++written;
#endif
#if LSTM_RET_HORIZON_4
        if (retCol == written) v = BaselineLookbackLogReturn(tensor, batchBegin, localRow, 4) * EA::LSTM::kFeatScale;
        ++written;
#endif
#if LSTM_RET_HORIZON_8
        if (retCol == written) v = BaselineLookbackLogReturn(tensor, batchBegin, localRow, 8) * EA::LSTM::kFeatScale;
        ++written;
#endif
#if LSTM_RET_HORIZON_16
        if (retCol == written) v = BaselineLookbackLogReturn(tensor, batchBegin, localRow, 16) * EA::LSTM::kFeatScale;
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

    pqxx::connection c_forex { "hostaddr=127.0.0.1  user=pqxx dbname=forex" };
    pqxx::work w_forex { c_forex };
    pqxx::result tables = w_forex.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    if (tables.empty())
    {
        std::cerr << "LABEL_GRID_ERROR,no_rmp_tables=1" << std::endl;
        return 1;
    }

    const std::string rawPriceTableName{ tables[0][0].c_str() };
    const std::string query = "select * from candlestick('" + rawPriceTableName + "', 15, 'minute', '" + fromDate + "', '" + toDate + "') order by dt;";
    db_cursor_stream<Feature> cs_cur{ w_forex, query, rawPriceTableName + "_label_grid_candlestick_stream" };
    db_input_iterator csb = cs_cur.begin(), cse = cs_cur.end();
    Tensor tensor{ rawPriceTableName };
    while (csb != cse) tensor.Add(*csb++);

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
    pqxx::connection c_forex { "hostaddr=127.0.0.1  user=pqxx dbname=forex" };
    pqxx::work w_forex { c_forex };
    pqxx::result tables = w_forex.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    if (tables.empty())
    {
        std::cerr << "BASELINE_3CLASS_ERROR,no_rmp_tables=1" << std::endl;
        return 1;
    }

    const std::string rawPriceTableName{ tables[0][0].c_str() };
    const std::string query = "select * from candlestick('" + rawPriceTableName + "', 15, 'minute', '" + fromDate + "', '" + toDate + "') order by dt;";
    db_cursor_stream<Feature> cs_cur{ w_forex, query, rawPriceTableName + "_baseline_candlestick_stream" };
    db_input_iterator csb = cs_cur.begin(), cse = cs_cur.end();
    Tensor tensor{ rawPriceTableName };
    while (csb != cse) tensor.Add(*csb++);

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
    pqxx::connection c_forex { "hostaddr=127.0.0.1  user=pqxx dbname=forex" };
    pqxx::work w_forex { c_forex };
    pqxx::result tables = w_forex.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    if (tables.empty())
    {
        std::cerr << "FEATURE_TRAINABILITY_ERROR,no_rmp_tables=1" << std::endl;
        return 1;
    }

    const std::string rawPriceTableName{ tables[0][0].c_str() };
    const std::string query = "select * from candlestick('" + rawPriceTableName + "', 15, 'minute', '" + fromDate + "', '" + toDate + "') order by dt;";
    db_cursor_stream<Feature> cs_cur{ w_forex, query, rawPriceTableName + "_feature_trainability_candlestick_stream" };
    db_input_iterator csb = cs_cur.begin(), cse = cs_cur.end();
    Tensor tensor{ rawPriceTableName };
    while (csb != cse) tensor.Add(*csb++);

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

    EA::LSTM lstm { tensor, 1, 0, EA::LSTM::TargetType::UpNeutralDownReturn };
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
}

struct PredictionStats
{
    size_t correctLog = 0;
    size_t actedLog = 0;
    size_t windows = 0;
    double absErrMove = 0.0;
    size_t correctDir = 0;
    size_t actedDir = 0;
    size_t confusion[direction_output_size][direction_output_size] = {};
};

static PredictionStats ProcessBatchPredict(EA::LSTM& l, const Tensor& tensor, const Window& b)
{
    auto stats = [](const auto& v)
    {
        struct S { double min, max, mean, std, uniq; } s{};
        if (v.empty()) return s;
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        double sum = 0.0, sumsq = 0.0;
        std::unordered_map<long long, int> buckets;
        buckets.reserve(v.size());
        for (float x : v)
        {
            mn = std::min(mn, static_cast<double>(x));
            mx = std::max(mx, static_cast<double>(x));
            sum += x; sumsq += static_cast<double>(x) * static_cast<double>(x);
            long long key = static_cast<long long>(std::llround(static_cast<double>(x) * 1e6));
            ++buckets[key];
        }
        double n = static_cast<double>(v.size());
        double mean = sum / n;
        double var = std::max(0.0, sumsq / n - mean * mean);
        s.min = mn; s.max = mx; s.mean = mean; s.std = std::sqrt(var); s.uniq = static_cast<double>(buckets.size());
        return s;
    };

    if (l.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
    {
        const auto& evalConfig = ActiveEvalLabelConfig();
        static bool s_inferLabelRulePrinted = false;
        static size_t s_inferLabelRowDiagCount = 0;
        constexpr size_t kInferLabelRowDiagLimit = 50;
        if (!s_inferLabelRulePrinted)
        {
            std::cout << "DIAG_INFER_LABEL_RULE"
                      << ",eval_label=lookahead_high_low_first_hit"
                      << ",old_terminal_close_class=reported_for_comparison"
                      << ",threshold_logret=" << evalConfig.thresholdLogret
                      << ",prediction_horizon=" << evalConfig.predictionHorizon
                      << ",window_size=" << evalConfig.windowSize
                      << ",label_rule_id=" << evalConfig.labelRuleId
                      << ",config_source=" << evalConfig.source
                      << std::endl;
            s_inferLabelRulePrinted = true;
        }

        std::vector<int> predClass;
        std::vector<int> actClass;
        std::vector<float> predMaxProb;

        const auto nWindows = static_cast<size_t>(
            std::max<std::ptrdiff_t>(0,
                                     (b.end() - b.begin()) -
                                     static_cast<std::ptrdiff_t>(evalConfig.windowSize + evalConfig.predictionHorizon) +
                                     1));
        predClass.reserve(nWindows);
        actClass.reserve(nWindows);
        predMaxProb.reserve(nWindows);

        PredictionStats result;

        for (auto it = b.begin();
             it + static_cast<std::ptrdiff_t>(evalConfig.windowSize - 1 + evalConfig.predictionHorizon) < b.end();
             ++it)
        {
            auto w = Window{it, it + static_cast<std::ptrdiff_t>(evalConfig.windowSize)};

            const auto probs = l.PredictNextDirectionProbs(w, /*resetState=*/true);
            const int pred = (probs[0] > probs[1] && probs[0] > probs[2]) ? 0
                           : ((probs[2] > probs[1] && probs[2] > probs[0]) ? 2 : 1);
            const float maxProb = std::max(probs[0], std::max(probs[1], probs[2]));

            const auto labelInfo = BuildLookaheadClassInfo(tensor,
                                                           it,
                                                           evalConfig.windowSize,
                                                           evalConfig.predictionHorizon,
                                                           evalConfig.thresholdLogret);
            const int actual = labelInfo.assignedClass;

            if (s_inferLabelRowDiagCount < kInferLabelRowDiagLimit)
            {
                const size_t globalStartIdx = static_cast<size_t>(it - tensor.begin());
                const size_t globalLastIdx = globalStartIdx + evalConfig.windowSize - 1;
                const size_t globalTargetIdx = globalLastIdx + evalConfig.predictionHorizon;
                const size_t globalFutureIdx = globalLastIdx + labelInfo.selectedOffset;
                std::cout << "DIAG_INFER_LABEL_ROW"
                          << ",global_tensor_row_idx=" << globalStartIdx
                          << ",last_row_idx=" << globalLastIdx
                          << ",target_row_idx=" << globalTargetIdx
                          << ",future_row_idx=" << globalFutureIdx
                          << ",close_t=" << labelInfo.closeT
                          << ",target_close=" << labelInfo.targetClose
                          << ",terminal_logret=" << labelInfo.terminalLogReturn
                          << ",old_terminal_close_class=" << labelInfo.terminalCloseClass
                          << ",new_lookahead_first_hit_class=" << labelInfo.assignedClass
                          << ",up_hit=" << static_cast<int>(labelInfo.upHit)
                          << ",down_hit=" << static_cast<int>(labelInfo.downHit)
                          << ",up_offset=" << labelInfo.upOffset
                          << ",down_offset=" << labelInfo.downOffset
                          << ",pred_class=" << pred
                          << std::endl;
                ++s_inferLabelRowDiagCount;
            }

            predClass.push_back(pred);
            actClass.push_back(actual);
            predMaxProb.push_back(maxProb);
            ++result.confusion[actual][pred];
        }

        const size_t N = std::min(predClass.size(), actClass.size());
        if (N == 0)
            return result;

        size_t correct = 0;
        for (size_t i = 0; i < N; ++i)
            if (predClass[i] == actClass[i]) ++correct;

        std::vector<float> predClassF, actClassF;
        predClassF.reserve(N);
        actClassF.reserve(N);
        for (size_t i = 0; i < N; ++i)
        {
            predClassF.push_back(static_cast<float>(predClass[i]));
            actClassF.push_back(static_cast<float>(actClass[i]));
        }

        const auto s_pred = stats(predClassF);
        const auto s_act  = stats(actClassF);
        const auto s_prob = stats(predMaxProb);

        std::cout << "pred_class stats: min=" << s_pred.min << " max=" << s_pred.max
                  << " mean=" << s_pred.mean << " std=" << s_pred.std
                  << " uniq~=" << s_pred.uniq << std::endl;
        std::cout << "act_class stats: min=" << s_act.min << " max=" << s_act.max
                  << " mean=" << s_act.mean << " std=" << s_act.std
                  << " uniq~=" << s_act.uniq << std::endl;
        std::cout << "pred_max_prob stats: min=" << s_prob.min << " max=" << s_prob.max
                  << " mean=" << s_prob.mean << " std=" << s_prob.std
                  << " uniq~=" << s_prob.uniq << std::endl;

        const double acc = static_cast<double>(correct) / static_cast<double>(N) * 100.0;
        std::cout << "3-class accuracy: " << acc << "% over " << N << " windows" << std::endl;
        std::cout << "3-class confusion matrix (rows=actual [down,neutral,up], cols=pred [down,neutral,up]): "
                  << "[[" << result.confusion[0][0] << ", " << result.confusion[0][1] << ", " << result.confusion[0][2] << "], "
                  << "[" << result.confusion[1][0] << ", " << result.confusion[1][1] << ", " << result.confusion[1][2] << "], "
                  << "[" << result.confusion[2][0] << ", " << result.confusion[2][1] << ", " << result.confusion[2][2] << "]]"
                  << std::endl;

#if LSTM_DEBUG_INTERNAL_PRINTS
        const size_t toPrint = std::min<size_t>(N, 10);
        for (size_t i = 0; i < toPrint; ++i)
        {
            std::cout << "i=" << i
                      << " predClass=" << predClass[i]
                      << " actClass=" << actClass[i]
                      << " maxProb=" << predMaxProb[i]
                      << " match=" << (predClass[i] == actClass[i])
                      << "\n";
        }
#endif

        result.correctLog = correct;
        result.actedLog = N;
        result.windows = N;
        result.correctDir = correct;
        result.actedDir = N;
        return result;
    }

    auto predLogRet = l.RollingPredictNextLogReturn(b, /*resetAtStart=*/true);

#if LSTM_DEBUG_PRINTS
    std::cout << "predLogRet samples: ";
    for (size_t i = 0; i < std::min<size_t>(10, predLogRet.size()); ++i) std::cout << predLogRet[i] << " ";
    std::cout << "\n";
#endif

    std::vector<float> predRel; predRel.reserve(predLogRet.size());
    for (float logret : predLogRet) predRel.push_back(std::exp(logret) - 1.0f);

    size_t gtOutliers = 0;
    std::vector<float> actRel, actLogRet;
    actRel.reserve(predRel.size()); actLogRet.reserve(predRel.size());
    for (auto it = b.begin(); it + window_size - 1 + prediction_horizon < b.end(); ++it)
    {
        const auto lastIt = it + window_size - 1;
        const auto targetIt = it + window_size - 1 + prediction_horizon;
        const float close_t = tensor.RawCloseAtIterator(lastIt);
        const float close_target = tensor.RawCloseAtIterator(targetIt);
        const float v_unscaled =
            (std::isfinite(close_t) && std::isfinite(close_target) &&
             close_t > 0.0f && close_target > 0.0f)
                ? std::log(close_target / close_t)
                : 0.0f;

        if(std::fabs(v_unscaled) > 0.02f) gtOutliers++;
        actLogRet.push_back(v_unscaled);
        actRel.push_back(std::exp(v_unscaled) - 1.0f);
    }
    auto s_gt = stats(actLogRet);
#if LSTM_DEBUG_INTERNAL_PRINTS
    std::cout << "actLogRet stats: min=" << s_gt.min << " max=" << s_gt.max
              << " mean=" << s_gt.mean << " std=" << s_gt.std
              << " uniq~=" << s_gt.uniq << std::endl << "GT outliers |v|>0.02: " << gtOutliers
              << " of " << actLogRet.size() << std::endl;
#endif
    assert(predRel.size() == actRel.size());
    if (predRel.empty() || actRel.empty())
    {
        return {};
    }

    auto s_raw = stats(predLogRet);
    auto s_rel = stats(predRel);
    std::cout << "pred_raw stats: min=" << s_raw.min << " max=" << s_raw.max
              << " mean=" << s_raw.mean << " std=" << s_raw.std
              << " uniq~=" << s_raw.uniq << std::endl;
    std::cout << "std ratio (pred_raw/actLogRet): " << (s_gt.std > 0.0 ? (s_raw.std / s_gt.std) : 0.0) << std::endl;
    std::cout << "means: pred_raw=" << s_raw.mean << " actLogRet=" << s_gt.mean << std::endl;

    std::cout << "pred_rel stats: min=" << s_rel.min << " max=" << s_rel.max
              << " mean=" << s_rel.mean << " std=" << s_rel.std
              << " uniq~=" << s_rel.uniq << std::endl;

    std::vector<float> predMove = predRel;
    std::vector<float> actualMove = actRel;
    size_t N = std::min(predMove.size(), actualMove.size());
    const size_t toPrint = std::min<size_t>(N, 10);

    size_t actedLog = 0;
    size_t correctLog = 0;
    const float actedThrLog = 2e-4f;
    for (size_t i = 0; i < N && i < predLogRet.size() && i < actLogRet.size(); ++i)
    {
        if (std::fabs(predLogRet[i]) < actedThrLog) continue;
        ++actedLog;
        bool predUp = predLogRet[i] >= 0.0f;
        bool actUp  = actLogRet[i]  >= 0.0f;
        if (predUp == actUp) ++correctLog;
    }
    double accLog = actedLog ? (static_cast<double>(correctLog) / static_cast<double>(actedLog) * 100.0) : 0.0;
    double covLog = N ? (static_cast<double>(actedLog) / static_cast<double>(N) * 100.0) : 0.0;
    std::cout << "Direction accuracy (log-return): " << accLog
              << "% over " << actedLog << " acted (of " << N << ")"
              << " thr=" << actedThrLog
              << " coverage=" << covLog << "%" << std::endl;

#if LSTM_DEBUG_INTERNAL_PRINTS
    const size_t M = std::min<size_t>(5, std::min(predLogRet.size(), actLogRet.size()));
    for (size_t i = 0; i < M; ++i)  std::cout << "align i=" << i << " predLogRet=" << predLogRet[i]  << " actLogRet(unscaled)=" << actLogRet[i] << " actRel=" << (std::exp(actLogRet[i]) - 1.0f) << "\n";
    for (size_t i = 0; i < toPrint; ++i)
    {
        auto sgn = [](float x){ return (x > 0) - (x < 0); };
        int sp = sgn(predMove[i]);
        int sa = sgn(actualMove[i]);
        std::cout << "i=" << i
            << " pred_rel=" << predMove[i]
            << " act_rel=" << actualMove[i]
            << " sp=" << sp
            << " sa=" << sa
            << " match=" << (sp == sa)
            << " |pred|=" << std::abs(predMove[i])
            << " |act|=" << std::abs(actualMove[i])
            << "\n";
    }
#endif
    double maeMove = 0.0;
    size_t correctDir = 0;
    size_t acted = 0;

    const float predThr = 1e-4f;
    const float actThr  = 1e-4f;

    for (size_t i = 0; i < N; ++i)
    {
        maeMove += std::abs(static_cast<double>(predMove[i]) -
                            static_cast<double>(actualMove[i]));

        if (std::fabs(predMove[i]) < predThr) continue;
        if (std::fabs(actualMove[i]) < actThr) continue;

        ++acted;

        bool predUp = predMove[i] >= 0.0f;
        bool actUp  = actualMove[i] >= 0.0f;
        if (predUp == actUp) ++correctDir;
    }

    if (N > 0)  std::cout << "Batch MAE (relative move fraction): "
                  << (maeMove / static_cast<double>(N))
                  << " | Direction accuracy (relative move, thresholded): "
                  << (acted ? (static_cast<double>(correctDir) /
                               static_cast<double>(acted) * 100.0)
                            : 0.0)
                  << "% over " << acted << " acted (of " << N << ")"
                  << " predThr=" << predThr
                  << " actThr=" << actThr
                  << " coverage=" << (static_cast<double>(acted) / static_cast<double>(N) * 100.0) << "%"
                  << std::endl;

    PredictionStats result;
    result.correctLog = correctLog;
    result.actedLog = actedLog;
    result.windows = N;
    result.absErrMove = maeMove;
    result.correctDir = correctDir;
    result.actedDir = acted;
    return result;
}

const std::string dbName = "forex";
const std::string dbModelName = "LSTM";

namespace
{
struct LaunchArgs
{
    std::string fromDate;
    std::string toDate;
    std::optional<long long> modelId;
    std::optional<bool> inferenceMode;
    std::optional<size_t> predictionHorizon;
    std::optional<double> thresholdLogret;
    std::optional<size_t> windowSize;
    std::optional<size_t> hiddenSize;
    std::optional<size_t> numLayers;
    std::optional<int> epochs;
};

long long ParseModelIdArg(const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument("--model requires a non-empty model_id");

    size_t consumed = 0;
    long long modelId = 0;
    try
    {
        modelId = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid --model value '" + value + "'; expected a positive integer model_id");
    }

    if (consumed != value.size() || modelId <= 0)
        throw std::invalid_argument("invalid --model value '" + value + "'; expected a positive integer model_id");

    return modelId;
}

size_t ParsePositiveSizeArg(const std::string& optionName, const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument(optionName + " requires a non-empty integer value");

    size_t consumed = 0;
    unsigned long long parsed = 0;
    try
    {
        parsed = std::stoull(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a positive integer");
    }

    if (consumed != value.size() || parsed == 0)
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a positive integer");

    return static_cast<size_t>(parsed);
}

int ParsePositiveIntArg(const std::string& optionName, const std::string& value)
{
    const size_t parsed = ParsePositiveSizeArg(optionName, value);
    if (parsed > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; exceeds int range");
    return static_cast<int>(parsed);
}

double ParsePositiveDoubleArg(const std::string& optionName, const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument(optionName + " requires a non-empty numeric value");

    size_t consumed = 0;
    double parsed = 0.0;
    try
    {
        parsed = std::stod(value, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a positive number");
    }

    if (consumed != value.size() || !std::isfinite(parsed) || parsed <= 0.0)
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a positive number");

    return parsed;
}

bool SplitOptionWithValue(const std::string& arg,
                          const char* optionName,
                          std::string& value)
{
    const std::string prefix = std::string(optionName) + "=";
    if (arg.rfind(prefix, 0) != 0)
        return false;

    value = arg.substr(prefix.size());
    return true;
}

LaunchArgs ParseLaunchArgs(int argc, const char* argv[])
{
    constexpr const char* kModelPrefix = "--model=";
    constexpr size_t kModelPrefixLen = 8;

    LaunchArgs parsed;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg{ argv[i] };

        if (arg.rfind(kModelPrefix, 0) == 0)
        {
            if (parsed.modelId.has_value())
                throw std::invalid_argument("--model specified more than once");

            parsed.modelId = ParseModelIdArg(arg.substr(kModelPrefixLen));
        }
        else if (arg == "--model")
        {
            if (parsed.modelId.has_value())
                throw std::invalid_argument("--model specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--model requires a model_id value");

            parsed.modelId = ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--train")
        {
            if (parsed.inferenceMode.has_value())
                throw std::invalid_argument("--train and --infer are mutually exclusive");
            parsed.inferenceMode = false;
        }
        else if (arg == "--infer")
        {
            if (parsed.inferenceMode.has_value())
                throw std::invalid_argument("--train and --infer are mutually exclusive");
            parsed.inferenceMode = true;
        }
        else
        {
            std::string value;
            if (SplitOptionWithValue(arg, "--prediction-horizon", value))
            {
                parsed.predictionHorizon = ParsePositiveSizeArg("--prediction-horizon", value);
            }
            else if (SplitOptionWithValue(arg, "--threshold", value))
            {
                parsed.thresholdLogret = ParsePositiveDoubleArg("--threshold", value);
            }
            else if (SplitOptionWithValue(arg, "--window-size", value))
            {
                parsed.windowSize = ParsePositiveSizeArg("--window-size", value);
            }
            else if (SplitOptionWithValue(arg, "--hidden-size", value))
            {
                parsed.hiddenSize = ParsePositiveSizeArg("--hidden-size", value);
            }
            else if (SplitOptionWithValue(arg, "--num-layers", value))
            {
                parsed.numLayers = ParsePositiveSizeArg("--num-layers", value);
            }
            else if (SplitOptionWithValue(arg, "--epochs", value))
            {
                parsed.epochs = ParsePositiveIntArg("--epochs", value);
            }
            else if (arg.rfind("--", 0) == 0)
            {
                throw std::invalid_argument("unknown option '" + arg + "'");
            }
            else
            {
                positional.push_back(arg);
            }
        }
    }

    if (positional.size() != 2)
        throw std::invalid_argument("expected arguments: [--train|--infer] [--model=<model_id>] [--prediction-horizon=<int>] [--threshold=<double>] [--window-size=<int>] [--hidden-size=<int>] [--num-layers=<int>] [--epochs=<int>] <fromDate> <toDate>");

    parsed.fromDate = positional[0];
    parsed.toDate = positional[1];
    return parsed;
}

void ApplyLaunchRuntimeConfig(const LaunchArgs& launchArgs)
{
    gRuntimeInferenceMode = launchArgs.inferenceMode.value_or(inference_only);

    if (launchArgs.predictionHorizon.has_value())
        prediction_horizon = *launchArgs.predictionHorizon;
    if (launchArgs.thresholdLogret.has_value())
        c_next_threshold = static_cast<float>(*launchArgs.thresholdLogret);
    if (launchArgs.windowSize.has_value())
        window_size = *launchArgs.windowSize;
    if (launchArgs.hiddenSize.has_value())
    {
        hidden_size = *launchArgs.hiddenSize;
        n_out = hidden_size;
    }
    if (launchArgs.numLayers.has_value())
    {
        if (*launchArgs.numLayers != 1)
            throw std::invalid_argument("--num-layers currently supports only 1; increasing layers would change the LSTM architecture");
        num_layers = *launchArgs.numLayers;
    }
    if (launchArgs.epochs.has_value())
        epoch_count = *launchArgs.epochs;
}

void PrintRuntimeConfig()
{
    std::cout << "RUNTIME_CONFIG"
              << ",train=" << (gRuntimeInferenceMode ? "false" : "true")
              << ",infer=" << (gRuntimeInferenceMode ? "true" : "false")
              << ",prediction_horizon=" << prediction_horizon
              << ",threshold=" << c_next_threshold
              << ",window_size=" << window_size
              << ",hidden_size=" << hidden_size
              << ",num_layers=" << num_layers
              << ",epochs=" << epoch_count
              << std::endl;
}

const char* TargetTypeName(EA::LSTM::TargetType targetType)
{
    switch (targetType)
    {
        case EA::LSTM::TargetType::LogReturn: return "LogReturn";
        case EA::LSTM::TargetType::PercentReturn: return "PercentReturn";
        case EA::LSTM::TargetType::UpNeutralDownReturn: return "UpNeutralDownReturn";
    }
    return "Unknown";
}

const char* DirectionLabelRuleName()
{
    return "lookahead_high_low_first_hit";
}

int DirectionLabelRuleId()
{
    return DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId;
}

const char* TrainConfigMetaFieldMapping()
{
    return "schema_version,prediction_horizon,threshold_logret,window_size,label_rule_id,class_weight_down,class_weight_neutral,class_weight_up,num_layers,normalization_version,epochs_trained";
}

size_t RuntimeModelInputWidth(const Tensor& tensor)
{
    const size_t baseFeatureCount = (tensor.begin() != tensor.end())
        ? static_cast<size_t>((*tensor.begin()).Shape()[1])
        : 0;
    return baseFeatureCount + BaselineReturnFeatureCount;
}

std::string JoinStrings(const std::vector<std::string>& values, const char* separator)
{
    if (values.empty())
        return "none";

    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i) oss << separator;
        oss << values[i];
    }
    return oss.str();
}

void PrintEvalLabelConfig()
{
    const auto& evalConfig = ActiveEvalLabelConfig();
    std::cout << "EVAL_LABEL_CONFIG"
              << ",label_rule=" << DirectionLabelRuleName()
              << ",prediction_horizon=" << evalConfig.predictionHorizon
              << ",threshold_logret=" << evalConfig.thresholdLogret
              << ",window_size=" << evalConfig.windowSize
              << ",label_rule_id=" << evalConfig.labelRuleId
              << ",config_source=" << evalConfig.source
              << std::endl;
}

void PrintInferenceConfig(const std::optional<long long>& loadedModelId,
                          const std::string& loadSource,
                          const EA::LSTM& lstm,
                          const std::string& fromDate,
                          const std::string& toDate)
{
    const auto& evalConfig = ActiveEvalLabelConfig();
    std::cout << "INFERENCE_CONFIG_SOURCE=" << evalConfig.source << std::endl;
    std::cout << "INFERENCE_CONFIG"
              << ",model_id=";
    if (loadedModelId.has_value())
        std::cout << *loadedModelId;
    else
        std::cout << -1;

    std::cout << ",load_source=" << loadSource
              << ",prediction_horizon=" << evalConfig.predictionHorizon
              << ",threshold_logret=" << evalConfig.thresholdLogret
              << ",target_type=" << TargetTypeName(lstm.targetType)
              << ",window_size=" << evalConfig.windowSize
              << ",label_rule_id=" << evalConfig.labelRuleId
              << ",from=" << fromDate
              << ",to=" << toDate
              << std::endl;
}

ModelConfigValidationResult PrintModelConfigValidation(pqxx::work& w,
                                                       long long modelId,
                                                       EA::LSTM::TargetType requestedTargetType,
                                                       const Tensor& tensor)
{
    ModelConfigValidationResult result;
    std::vector<std::string> persistedParams;
    std::vector<std::string> persistedMetadata;
    bool hasTargetMeta = false;
    bool hasModelMeta = false;
    bool hasTrainConfigMeta = false;
    bool hasTrainConfigNumLayers = false;
    bool hasTrainConfigNormalizationVersion = false;

    std::cout << "MODEL_TRAIN_CONFIG_META_FIELDS,"
              << TrainConfigMetaFieldMapping()
              << std::endl;

    try
    {
        pqxx::result params = w.exec_params(
            "SELECT DISTINCT param_name FROM matrix WHERE model_id = $1 ORDER BY param_name;",
            modelId);
        for (const auto& row : params)
        {
            const std::string paramName = row[0].as<std::string>();
            persistedParams.push_back(paramName);
            if (paramName == "target_meta")
            {
                hasTargetMeta = true;
                persistedMetadata.push_back("target_meta(type;scale;bias;use_zscore;mean;std)");
            }
            else if (paramName == "model_meta")
            {
                hasModelMeta = true;
                persistedMetadata.push_back("model_meta(schemaVersion;n_in;hidden_size)");
            }
            else if (paramName == "train_config_meta")
            {
                hasTrainConfigMeta = true;
                persistedMetadata.push_back("train_config_meta(schema_version;prediction_horizon;threshold_logret;window_size;label_rule_id;class_weight_down;class_weight_neutral;class_weight_up;num_layers;normalization_version;epochs_trained)");
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "MODEL_METADATA_READ_FAIL"
                  << ",model_id=" << modelId
                  << ",error=" << e.what()
                  << std::endl;
    }

    std::cout << "MODEL_METADATA_PERSISTED"
              << ",model_id=" << modelId
              << ",param_names=" << JoinStrings(persistedParams, ";")
              << ",metadata=" << JoinStrings(persistedMetadata, ";")
              << std::endl;

    bool targetMetaMatches = false;
    bool modelMetaMatches = false;
    bool trainConfigMetaMatches = false;
    bool mismatch = false;

    auto printMismatch = [&](const char* field, const auto& modelValue, const auto& runtimeValue)
    {
        mismatch = true;
        std::cout << "MODEL_CONFIG_MISMATCH"
                  << ",field=" << field
                  << ",model=" << modelValue
                  << ",runtime=" << runtimeValue
                  << std::endl;
    };

    auto compareIntField = [&](const char* field, double modelValue, long long runtimeValue) -> bool
    {
        const long long roundedModelValue = static_cast<long long>(std::llround(modelValue));
        if (roundedModelValue != runtimeValue)
        {
            printMismatch(field, roundedModelValue, runtimeValue);
            return false;
        }
        return true;
    };

    auto compareFloatField = [&](const char* field, double modelValue, double runtimeValue) -> bool
    {
        constexpr double kFloatCompareTolerance = 1e-7;
        if (std::fabs(modelValue - runtimeValue) > kFloatCompareTolerance)
        {
            printMismatch(field, modelValue, runtimeValue);
            return false;
        }
        return true;
    };

    try
    {
        auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "target_meta");
        auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "target_meta");
        if (dims.n_rows != 1 || dims.n_cols != 6 || vals.size() != 6)
        {
            printMismatch("target_meta_shape", std::to_string(dims.n_rows) + "x" + std::to_string(dims.n_cols), "1x6");
        }
        else
        {
            bool sectionMatches = true;
            const auto modelTargetType = static_cast<EA::LSTM::TargetType>(static_cast<int>(vals[0]));
            if (static_cast<int>(modelTargetType) != static_cast<int>(requestedTargetType))
            {
                printMismatch("target_type", TargetTypeName(modelTargetType), TargetTypeName(requestedTargetType));
                sectionMatches = false;
            }
            targetMetaMatches = sectionMatches;
        }
    }
    catch (const std::exception&)
    {
        // Older models may not have target_meta.
    }

    try
    {
        auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "model_meta");
        auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "model_meta");
        if (dims.n_rows != 1 || dims.n_cols != 3 || vals.size() != 3)
        {
            printMismatch("model_meta_shape", std::to_string(dims.n_rows) + "x" + std::to_string(dims.n_cols), "1x3");
        }
        else
        {
            bool sectionMatches = true;
            const int schemaVersion = static_cast<int>(vals[0]);
            const int modelInputWidth = static_cast<int>(vals[1]);
            const int modelHiddenSize = static_cast<int>(vals[2]);
            const int runtimeInputWidth = static_cast<int>(RuntimeModelInputWidth(tensor));
            const int runtimeHiddenSize = static_cast<int>(hidden_size);

            if (schemaVersion != 1)
            {
                printMismatch("model_meta_schema_version", schemaVersion, 1);
                sectionMatches = false;
            }
            if (modelInputWidth != runtimeInputWidth)
            {
                printMismatch("feature_count", modelInputWidth, runtimeInputWidth);
                sectionMatches = false;
            }
            if (modelHiddenSize != runtimeHiddenSize)
            {
                printMismatch("hidden_size", modelHiddenSize, runtimeHiddenSize);
                sectionMatches = false;
            }
            modelMetaMatches = sectionMatches;
        }
    }
    catch (const std::exception&)
    {
        // Older models may not have model_meta.
    }

    try
    {
        auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "train_config_meta");
        auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "train_config_meta");
        if (dims.n_rows != 1 ||
            dims.n_cols < DBIO::PgModelIO::kTrainConfigMetaFieldCount ||
            vals.size() < static_cast<size_t>(DBIO::PgModelIO::kTrainConfigMetaFieldCount))
        {
            printMismatch("train_config_meta_shape",
                          std::to_string(dims.n_rows) + "x" + std::to_string(dims.n_cols),
                          "1x>=8");
        }
        else
        {
            TrainConfigMeta trainConfigMeta;
            trainConfigMeta.schemaVersion = static_cast<int>(std::llround(vals[0]));
            trainConfigMeta.predictionHorizon = static_cast<size_t>(std::llround(vals[1]));
            trainConfigMeta.thresholdLogret = static_cast<float>(vals[2]);
            trainConfigMeta.windowSize = static_cast<size_t>(std::llround(vals[3]));
            trainConfigMeta.labelRuleId = static_cast<int>(std::llround(vals[4]));
            trainConfigMeta.classWeightDown = static_cast<float>(vals[5]);
            trainConfigMeta.classWeightNeutral = static_cast<float>(vals[6]);
            trainConfigMeta.classWeightUp = static_cast<float>(vals[7]);
            if (vals.size() >= 9)
            {
                trainConfigMeta.numLayers = static_cast<size_t>(std::llround(vals[8]));
                hasTrainConfigNumLayers = true;
            }
            if (vals.size() >= 10)
            {
                trainConfigMeta.normalizationVersion = static_cast<int>(std::llround(vals[9]));
                hasTrainConfigNormalizationVersion = true;
            }
            if (vals.size() >= 11)
            {
                trainConfigMeta.epochsTrained = static_cast<size_t>(std::llround(vals[10]));
            }
            result.trainConfigMeta = trainConfigMeta;

            std::cout << "MODEL_TRAIN_CONFIG_META"
                      << ",model_id=" << modelId
                      << ",schema_version=" << trainConfigMeta.schemaVersion
                      << ",prediction_horizon=" << trainConfigMeta.predictionHorizon
                      << ",threshold_logret=" << trainConfigMeta.thresholdLogret
                      << ",window_size=" << trainConfigMeta.windowSize
                      << ",label_rule_id=" << trainConfigMeta.labelRuleId
                      << ",class_weight_down=" << trainConfigMeta.classWeightDown
                      << ",class_weight_neutral=" << trainConfigMeta.classWeightNeutral
                      << ",class_weight_up=" << trainConfigMeta.classWeightUp
                      << ",num_layers=" << trainConfigMeta.numLayers
                      << ",normalization_version=" << trainConfigMeta.normalizationVersion
                      << ",epochs_trained=";
            if (trainConfigMeta.epochsTrained.has_value())
                std::cout << *trainConfigMeta.epochsTrained;
            else
                std::cout << "missing";
            std::cout
                      << std::endl;

            bool sectionMatches = true;
            sectionMatches = compareIntField("schema_version",
                                             vals[0],
                                             DBIO::PgModelIO::kTrainConfigMetaSchemaVersion) && sectionMatches;
            sectionMatches = compareIntField("prediction_horizon",
                                             vals[1],
                                             prediction_horizon) && sectionMatches;
            sectionMatches = compareFloatField("threshold_logret",
                                               vals[2],
                                               c_next_threshold) && sectionMatches;
            sectionMatches = compareIntField("window_size",
                                             vals[3],
                                             window_size) && sectionMatches;
            sectionMatches = compareIntField("label_rule_id",
                                             vals[4],
                                             DirectionLabelRuleId()) && sectionMatches;
            sectionMatches = compareFloatField("class_weight_down",
                                               vals[5],
                                               kClassWeightDown) && sectionMatches;
            sectionMatches = compareFloatField("class_weight_neutral",
                                               vals[6],
                                               kClassWeightNeutral) && sectionMatches;
            sectionMatches = compareFloatField("class_weight_up",
                                               vals[7],
                                               kClassWeightUp) && sectionMatches;
            if (vals.size() >= 9)
                sectionMatches = compareIntField("num_layers",
                                                 vals[8],
                                                 num_layers) && sectionMatches;
            if (vals.size() >= 10)
                sectionMatches = compareIntField("normalization_version",
                                                 vals[9],
                                                 normalization_version) && sectionMatches;
            trainConfigMetaMatches = sectionMatches;
        }
    }
    catch (const std::exception&)
    {
        // Older models may not have train_config_meta.
    }

    std::vector<std::string> missingMinimum;

    if (!hasTrainConfigMeta)
    {
        missingMinimum.push_back("prediction_horizon");
        missingMinimum.push_back("threshold_logret");
        missingMinimum.push_back("window_size");
        missingMinimum.push_back("label_rule");
        missingMinimum.push_back("class_weight_down");
        missingMinimum.push_back("class_weight_neutral");
        missingMinimum.push_back("class_weight_up");
        missingMinimum.push_back("num_layers");
        missingMinimum.push_back("normalization_version");
    }
    else
    {
        if (!hasTrainConfigNumLayers)
            missingMinimum.push_back("num_layers");
        if (!hasTrainConfigNormalizationVersion)
            missingMinimum.push_back("normalization_version");
    }
    if (!hasTargetMeta)
        missingMinimum.push_back("target_type");
    if (!hasModelMeta)
    {
        missingMinimum.push_back("feature_count");
        missingMinimum.push_back("hidden_size");
    }

    if (targetMetaMatches &&
        modelMetaMatches &&
        trainConfigMetaMatches &&
        !mismatch &&
        missingMinimum.empty())
    {
        std::cout << "MODEL_CONFIG_MATCH=1" << std::endl;
    }

    if (!missingMinimum.empty())
    {
        std::cout << "MODEL_CONFIG_METADATA_GAP"
                  << ",model_id=" << modelId
                  << ",missing=" << JoinStrings(missingMinimum, ";")
                  << ",recommend_minimum_additions=" << JoinStrings(missingMinimum, ";")
                  << std::endl;
    }

    return result;
}
}


int main(int argc, const char * argv[])
{
    if (argc >= 4 && std::string(argv[1]) == "--baseline-3class")
        return RunBaseline3Class(argv[2], argv[3]);
    if (argc >= 4 && std::string(argv[1]) == "--label-grid-3class")
        return RunLabelGridDiagnostic3Class(argv[2], argv[3]);
    if (argc >= 4 && std::string(argv[1]) == "--feature-trainability-3class")
        return RunFeatureTrainability3Class(argv[2], argv[3]);

    LaunchArgs launchArgs;
    try
    {
        launchArgs = ParseLaunchArgs(argc, argv);
        ApplyLaunchRuntimeConfig(launchArgs);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n"
                  << "Usage: " << argv[0] << " [--train|--infer] [--model=<model_id>] [--prediction-horizon=<int>] [--threshold=<double>] [--window-size=<int>] [--hidden-size=<int>] [--num-layers=<int>] [--epochs=<int>] <fromDate> <toDate>\n";
        return 1;
    }

    pqxx::connection c_forex { "hostaddr=127.0.0.1  user=pqxx dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::connection c_LSTM { "hostaddr=127.0.0.1  user=pqxx dbname=" + dbModelName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w_forex { c_forex }, w_LSTM { c_LSTM };
    pqxx::result tables = w_forex.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    std::string fromDate  { launchArgs.fromDate }, toDate { launchArgs.toDate };
    
    std::cout << "candle_duration=" << static_cast<int>(candle_duration) << '\n';
    std::cout << "window_size=" << window_size << '\n';
    std::cout << "prediction_horizon=" << prediction_horizon << '\n';
    std::cout << "c_next_threshold=" << c_next_threshold << '\n';
    PrintRuntimeConfig();
    std::cout << "DIAG_GATESTATE_MODE=" << LSTM_GATESTATE_MODE
              << " (" << GateStateModeLabel() << ")\n";

    w_LSTM.exec("SET TRANSACTION READ WRITE;");
    try
    {
        pqxx::result tables = w_forex.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
        for (auto tbl : tables)
        {
            std::string rawPriceTableName{ tbl[0].c_str() };
            std::string query = "select * from candlestick('" + rawPriceTableName + "', 15, 'minute', '" + fromDate + "', '" + toDate + "') order by dt;";
            db_cursor_stream<Feature> cs_cur{ w_forex, query, rawPriceTableName + "_candlestick_stream" };
            db_input_iterator csb = cs_cur.begin(), cse = cs_cur.end();
            Tensor t{ rawPriceTableName };
            
            std::cout << "Candlestick query: " << query << "\n";
            std::cout << "Building tensor for table: " << rawPriceTableName << std::endl;
            while (csb != cse) t.Add(*csb++);
  
            constexpr auto requestedTargetType = EA::LSTM::TargetType::UpNeutralDownReturn;
            EA::LSTM l { t, 1, 0, requestedTargetType };
            static size_t s_lstmBindingDiagCount = 0;
            constexpr size_t kLstmBindingDiagLimit = 50;
            if (s_lstmBindingDiagCount < kLstmBindingDiagLimit)
            {
                std::cout << "DIAG_LSTM_BINDING"
                          << ",table=" << rawPriceTableName
                          << ",tensor_rows=" << t.RowCount()
                          << ",tensor_addr=" << static_cast<const void*>(&t)
                          << ",lstm_tensor_ref_addr=" << static_cast<const void*>(l.BoundTensorAddress())
                          << ",newly_constructed=1"
                          << ",reused=0"
                          << std::endl;
                ++s_lstmBindingDiagCount;
            }

            // Track whether we started from scratch (no model loaded)
            std::optional<long long> loadedModelId;
            bool startedFromScratch = true;
            const std::string loadSource = launchArgs.modelId.has_value() ? "--model" : "latest";

            // Load an explicitly requested model, or default to the latest stored model.
            try
            {
                long long modelIdToLoad = -1;
                const bool requestedModel = launchArgs.modelId.has_value();

                if (requestedModel) modelIdToLoad = *launchArgs.modelId;
                else if (load_latest || gRuntimeInferenceMode)
                {
                    pqxx::result r = w_LSTM.exec("SELECT max(model_id) FROM model;");
                    if (!r.empty() && !r[0][0].is_null()) modelIdToLoad = r[0][0].as<long long>();
                    else std::cout << "No models found; using default-initialized parameters" << std::endl;
                }
                else std::cout << "load_latest=false; using default-initialized parameters" << std::endl;

                if (modelIdToLoad > 0)
                {
                    DBIO::PgModelIO::loadAll(w_LSTM, modelIdToLoad, l);
                    loadedModelId = modelIdToLoad;
                    startedFromScratch = false;
                    std::cout << "Loaded model_id=" << *loadedModelId
                              << " source=" << loadSource
                              << std::endl;
                }
            }
            catch (const std::exception& e)
            {
                if (launchArgs.modelId.has_value())
                {
                    std::cerr << "Load requested model_id=" << *launchArgs.modelId
                              << " failed: " << e.what() << std::endl;
                    return 1;
                }

                std::cout << "Load latest failed: " << e.what()
                          << "; using default params" << std::endl;
            }

            UseRuntimeDefaultEvalLabelConfig();
            ModelConfigValidationResult modelConfigValidation;
            if (loadedModelId.has_value())
                modelConfigValidation = PrintModelConfigValidation(w_LSTM, *loadedModelId, requestedTargetType, t);
            if (gRuntimeInferenceMode)
            {
                if (modelConfigValidation.trainConfigMeta.has_value())
                    UseModelTrainEvalLabelConfig(*modelConfigValidation.trainConfigMeta);
                else
                    UseRuntimeDefaultEvalLabelConfig();
            }
            PrintEvalLabelConfig();
            if (gRuntimeInferenceMode)
                PrintInferenceConfig(loadedModelId, loadSource, l, fromDate, toDate);

            PrintClassificationProofDiagnostics(l, t, fromDate, toDate);
            PrintPhase2TensorDiagnostics(l, t, fromDate, toDate);

            
            size_t totalCorrectLog = 0;
            size_t totalActedLog = 0;
            size_t totalWindows = 0;
            double totalAbsErrMove = 0.0;
            size_t totalCorrectDir = 0;
            size_t totalActedDir = 0;
            size_t totalConfusion[direction_output_size][direction_output_size] = {};
            // Iterate all batches; inference evaluates once, training runs configured epochs.
            std::cout << std::setprecision(15);
                const int evalPassCount = gRuntimeInferenceMode ? 1 : epoch_count;
                for(auto e = 0; e < evalPassCount; e++)
                {
                    t.ForEachBatch( [&](auto b)
                                   {
                        if (gRuntimeInferenceMode)
                        {
                            const auto predictionStats = ProcessBatchPredict(l, t, b);
                            totalCorrectLog += predictionStats.correctLog;
                            totalActedLog += predictionStats.actedLog;
                            totalWindows += predictionStats.windows;
                            totalAbsErrMove += predictionStats.absErrMove;
                            totalCorrectDir += predictionStats.correctDir;
                            totalActedDir += predictionStats.actedDir;

                            for (size_t actual = 0; actual < direction_output_size; ++actual)
                                for (size_t pred = 0; pred < direction_output_size; ++pred)
                                    totalConfusion[actual][pred] += predictionStats.confusion[actual][pred];
                        }
                        else
                        {
                            auto l2 = [](const auto& m){
                                // Ensure we operate on a concrete, materialized matrix to avoid stale/lazy views
                                auto cm = MetaNN::Evaluate(m);
                                auto low = MetaNN::LowerAccess(cm);
                                const float* p = low.RawMemory();
                                size_t len = cm.Shape()[0] * cm.Shape()[1];
                                double s = 0;
                                for (size_t i = 0; i < len; ++i) { double v = p[i]; s += v * v; }
                                return std::sqrt(s);
                            };
                            
                            double p0 = l2(l.param);
                            double b0 = l2(l.bias);
                            double hw0 = l2(l.returnHeadWeight);
                            double hb0 = l2(l.returnHeadBias);
                            double dhw0 = l2(l.returnHeadDirWeight);
                            double dhb0 = l2(l.returnHeadDirBias);
                            
                            auto [loss, _unused1, _unused2] = l.CalculateBatch(b,e);
                            (void)_unused1; (void)_unused2;
                            
                            double p1 = l2(l.param);
                            double b1 = l2(l.bias);
                            double hw1 = l2(l.returnHeadWeight);
                            double hb1 = l2(l.returnHeadBias);
                            double dhw1 = l2(l.returnHeadDirWeight);
                            double dhb1 = l2(l.returnHeadDirBias);
                            
                            std::cout << "epoch " << (e+1)
                            << " loss=" << loss
                            << " ||param|| " << p0  << " -> " << p1
                            << " ||bias|| "  << b0  << " -> " << b1;
                            if (l.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
                                std::cout << " ||dirHeadW|| " << dhw0 << " -> " << dhw1 << " ||dirHeadB|| " << dhb0 << " -> " << dhb1 << std::endl;
                            else
                                std::cout << " ||headW|| " << hw0 << " -> " << hw1 << " ||headB|| " << hb0 << " -> " << hb1 << std::endl;
                        }
                    } );
                    if (l.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
                    {
                        void PrintAndResetDistribution();
                        
                        EA::LSTM::PrintAndResetEpochBuckets();
                        PrintAndResetDistribution();
                    }
                }
            if (gRuntimeInferenceMode)
            {
                if (l.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
                {
                    const double overallAcc3Class = totalWindows
                        ? (static_cast<double>(totalCorrectDir) / static_cast<double>(totalWindows) * 100.0)
                        : 0.0;

                    std::cout << "Overall 3-class accuracy: " << overallAcc3Class
                              << "% over " << totalWindows << " windows" << std::endl;
                    std::cout << "Overall 3-class confusion matrix (rows=actual [down,neutral,up], cols=pred [down,neutral,up]): "
                              << "[[" << totalConfusion[0][0] << ", " << totalConfusion[0][1] << ", " << totalConfusion[0][2] << "], "
                              << "[" << totalConfusion[1][0] << ", " << totalConfusion[1][1] << ", " << totalConfusion[1][2] << "], "
                              << "[" << totalConfusion[2][0] << ", " << totalConfusion[2][1] << ", " << totalConfusion[2][2] << "]]"
                              << std::endl;
                }
                else
                {
                    const double overallAccLog = totalActedLog
                        ? (static_cast<double>(totalCorrectLog) / static_cast<double>(totalActedLog) * 100.0)
                        : 0.0;
                    const double overallCovLog = totalWindows
                        ? (static_cast<double>(totalActedLog) / static_cast<double>(totalWindows) * 100.0)
                        : 0.0;
                    const double overallMaeMove = totalWindows
                        ? (totalAbsErrMove / static_cast<double>(totalWindows))
                        : 0.0;
                    const double overallAccDir = totalActedDir
                        ? (static_cast<double>(totalCorrectDir) / static_cast<double>(totalActedDir) * 100.0)
                        : 0.0;
                    const double overallCovDir = totalWindows
                        ? (static_cast<double>(totalActedDir) / static_cast<double>(totalWindows) * 100.0)
                        : 0.0;

                    std::cout << "Overall direction accuracy (log-return): " << overallAccLog
                              << "% over " << totalActedLog << " acted (of " << totalWindows << ")"
                              << " coverage=" << overallCovLog << "%" << std::endl;
                    std::cout << "Overall MAE (relative move fraction): " << overallMaeMove
                              << " | Overall direction accuracy (relative move, thresholded): " << overallAccDir
                              << "% over " << totalActedDir << " acted (of " << totalWindows << ")"
                              << " coverage=" << overallCovDir << "%" << std::endl;
                }
            }

            // Persist trained model parameters to DB
            try
            {
                if (save_enable && !gRuntimeInferenceMode)
                {
                    // Ensure this transaction is read-write for saving
                    w_LSTM.exec("SET TRANSACTION READ WRITE;");
                    
                    long long modelId = -1;
                    if (startedFromScratch)
                        if constexpr (save_overwrite)
                            // Overwrite the latest model if one exists; otherwise create a new snapshot
                            try {
                                pqxx::result rLatest = w_LSTM.exec("SELECT max(model_id) FROM model;");
                                if (!rLatest.empty() && !rLatest[0][0].is_null()) {
                                    modelId = rLatest[0][0].as<long long>();
                                    std::cout << "Overwriting latest model_id=" << modelId << " (started from scratch, overwrite enabled)" << std::endl;
                                } else {
                                    modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                                    std::cout << "Created new model_id=" << modelId << " (no existing model to overwrite)" << std::endl;
                                }
                            } catch (const std::exception& e)
                            {
                                std::cout << "Fetch latest model_id failed (" << e.what() << "); creating new snapshot" << std::endl;
                                modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                            }
                        else
                        {
                            // Create a new snapshot when saving (do not overwrite existing)
                            modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                            std::cout << "Created new model_id=" << modelId << " (started from scratch)" << std::endl;
                        }
                    else
                        if constexpr (save_overwrite)
                            if (loadedModelId.has_value())
                            {
                                modelId = *loadedModelId;
                                std::cout << "Overwriting existing model_id=" << modelId << std::endl;
                            }
                            else
                            {
                                modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                                std::cout << "Created new model_id=" << modelId << " (no prior model to overwrite)" << std::endl;
                            }
                        else
                        {
                            // Create a new snapshot when saving (do not overwrite existing)
                            modelId = DBIO::PgModelIO::createModel(w_LSTM, rawPriceTableName + "-model", "trained parameters");
                            std::cout << "Created new model_id=" << modelId << std::endl;
                        }

                    DBIO::PgModelIO::saveAll(w_LSTM, modelId, l);
                    w_LSTM.commit();
                    std::cout << "Saved model with model_id=" << modelId << std::endl;
                }
                else
                    if (gRuntimeInferenceMode)
                        std::cout << "inference_only=true; skipping model save" << std::endl;
                    else if constexpr (!save_enable)
                        std::cout << "save_enable=false; skipping model save" << std::endl;
                    else
                        std::cout << "skipping model save (unknown reason)" << std::endl;
            }
            catch (const std::exception& e) { std::cerr << "Model save/load error: " << e.what() << std::endl;    }

            break;
        }

    }
    catch (const pqxx::broken_connection& e)
    {
        std::cerr << "Broken connection: " << e.what() << "\n";
        return 1;
    }
    catch (const pqxx::failure& e)
    {
        std::cerr << "pqxx::failure: " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "std::exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
