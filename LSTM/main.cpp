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
#include <iomanip>
#include <cassert>
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

const char* CurrentRangeKindLabel()
{
    return inference_only ? "inference" : "train";
}

struct LookaheadClassInfo
{
    PriceTP dt {};
    PriceTP targetDt {};
    float closeT = 0.0f;
    float targetClose = 0.0f;
    float deltaClose = 0.0f;
    float terminalLogReturn = 0.0f;
    bool upHit = false;
    bool downHit = false;
    size_t upOffset = 0;
    size_t downOffset = 0;
    int assignedClass = 1;
};

LookaheadClassInfo BuildLookaheadClassInfo(const Tensor& tensor,
                                          DataSet::const_iterator startIt)
{
    const auto lastIt = startIt + window_size - 1;
    const auto targetIt = lastIt + prediction_horizon;

    LookaheadClassInfo info;
    info.dt = tensor.RawTimeAtIterator(lastIt);
    info.targetDt = tensor.RawTimeAtIterator(targetIt);
    info.closeT = tensor.RawCloseAtIterator(lastIt);
    info.targetClose = tensor.RawCloseAtIterator(targetIt);
    info.deltaClose = info.targetClose - info.closeT;
    info.terminalLogReturn =
        (std::isfinite(info.closeT) && std::isfinite(info.targetClose) &&
         info.closeT > 0.0f && info.targetClose > 0.0f)
            ? std::log(info.targetClose / info.closeT)
            : 0.0f;

    for (size_t lookahead = 1; lookahead <= prediction_horizon; ++lookahead)
    {
        const auto futureIt = lastIt + static_cast<std::ptrdiff_t>(lookahead);
        const float futureHigh = tensor.RawHighAtIterator(futureIt);
        const float futureLow = tensor.RawLowAtIterator(futureIt);

        if (std::isfinite(info.closeT) && info.closeT > 0.0f)
        {
            const float upMove = std::log(futureHigh / info.closeT);
            const float downMove = std::log(futureLow / info.closeT);

            if (!info.upHit && std::isfinite(upMove) && upMove > c_next_threshold)
            {
                info.upHit = true;
                info.upOffset = lookahead;
            }

            if (!info.downHit && std::isfinite(downMove) && downMove < -c_next_threshold)
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

    LSTM_ASSERT(info.assignedClass >= 0 && info.assignedClass < static_cast<int>(direction_output_size),
                "BuildLookaheadClassInfo: assigned class out of [0,2]");
    return info;
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

    std::cout << "DIAG_CLASS_THRESHOLDS"
              << ",range_kind=" << CurrentRangeKindLabel()
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",threshold_logret=" << c_next_threshold
              << ",prediction_horizon=" << prediction_horizon
              << ",neutral_rule=no_threshold_hit_within_horizon"
              << ",output_dim=" << direction_output_size
              << std::endl;
    std::cout << "DIAG_CLASS_LABEL_RULE"
              << ",training_label=lookahead_high_low_first_hit"
              << ",inference_eval_label=terminal_close_logret"
              << std::endl;

    std::array<size_t, direction_output_size> hist {0, 0, 0};
    size_t windowCount = 0;
    size_t printedRows = 0;

    for (auto it = tensor.begin(); it + window_size - 1 + prediction_horizon < tensor.end(); ++it)
    {
        const auto info = BuildLookaheadClassInfo(tensor, it);
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
}

static auto ProcessBatchPredict(EA::LSTM& l, const Tensor& tensor, const Window& b) -> std::tuple<size_t, size_t, size_t, double, size_t, size_t>
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
        std::vector<int> predClass;
        std::vector<int> actClass;
        std::vector<float> predMaxProb;

        const auto nWindows = static_cast<size_t>(
            std::max<std::ptrdiff_t>(0, (b.end() - b.begin()) - static_cast<std::ptrdiff_t>(window_size + prediction_horizon) + 1));
        predClass.reserve(nWindows);
        actClass.reserve(nWindows);
        predMaxProb.reserve(nWindows);

        size_t confusion[direction_output_size][direction_output_size] = {};

        for (auto it = b.begin(); it + window_size - 1 + prediction_horizon < b.end(); ++it)
        {
            auto w = Window{it, it + window_size};

            const auto probs = l.PredictNextDirectionProbs(w, /*resetState=*/true);
            const int pred = (probs[0] > probs[1] && probs[0] > probs[2]) ? 0
                           : ((probs[2] > probs[1] && probs[2] > probs[0]) ? 2 : 1);
            const float maxProb = std::max(probs[0], std::max(probs[1], probs[2]));

            const auto lastIt = it + window_size - 1;
            const auto targetIt = it + window_size - 1 + prediction_horizon;
            const float close_t = tensor.RawCloseAtIterator(lastIt);
            const float close_target = tensor.RawCloseAtIterator(targetIt);
            const float v_unscaled =
                (std::isfinite(close_t) && std::isfinite(close_target) &&
                 close_t > 0.0f && close_target > 0.0f)
                    ? std::log(close_target / close_t)
                    : 0.0f;
            const int actual = (v_unscaled > c_next_threshold) ? 2
                             : ((v_unscaled < -c_next_threshold) ? 0 : 1);

            predClass.push_back(pred);
            actClass.push_back(actual);
            predMaxProb.push_back(maxProb);
            ++confusion[actual][pred];
        }

        const size_t N = std::min(predClass.size(), actClass.size());
        if (N == 0)
            return {0, 0, 0, 0.0, 0, 0};

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
                  << "[[" << confusion[0][0] << ", " << confusion[0][1] << ", " << confusion[0][2] << "], "
                  << "[" << confusion[1][0] << ", " << confusion[1][1] << ", " << confusion[1][2] << "], "
                  << "[" << confusion[2][0] << ", " << confusion[2][1] << ", " << confusion[2][2] << "]]"
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

        return {correct, N, N, 0.0, correct, N};
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
        return {0, 0, 0, 0.0, 0, 0};
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

    return {correctLog, actedLog, N, maeMove, correctDir, acted};
}

const std::string dbName = "forex";
const std::string dbModelName = "LSTM";


int main(int argc, const char * argv[])
{
    pqxx::connection c_forex { "hostaddr=127.0.0.1  user=pqxx dbname=" + dbName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::connection c_LSTM { "hostaddr=127.0.0.1  user=pqxx dbname=" + dbModelName }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::work w_forex { c_forex }, w_LSTM { c_LSTM };
    pqxx::result tables = w_forex.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp';");
    std::string fromDate  { argv[1] }, toDate { argv[2] };
    
    std::cout << "candle_duration=" << static_cast<int>(candle_duration) << '\n';
    std::cout << "window_size=" << window_size << '\n';
    std::cout << "prediction_horizon=" << prediction_horizon << '\n';
    std::cout << "c_next_threshold=" << c_next_threshold << '\n';
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
  
            thread_local EA::LSTM l { t, 1, 0, EA::LSTM::TargetType::UpNeutralDownReturn };

            // Track whether we started from scratch (no model loaded)
            std::optional<long long> loadedModelId;
            bool startedFromScratch = true;

            // Optionally load the latest model parameters from the LSTM DB
            if constexpr (load_latest || inference_only)
                try
                {
                    pqxx::result r = w_LSTM.exec("SELECT max(model_id) FROM model;");
                    if (!r.empty() && !r[0][0].is_null())
                    {
                        loadedModelId = r[0][0].as<long long>();
                        DBIO::PgModelIO::loadAll(w_LSTM, *loadedModelId, l);
                        startedFromScratch = false;
                        std::cout << "Loaded model_id=" << *loadedModelId << std::endl;
                    }
                    else {  std::cout << "No models found; using default-initialized parameters" << std::endl;  }
                }
                catch (const std::exception& e) {   std::cout << "Load latest failed: " << e.what() << "; using default params" << std::endl;   }
            else    std::cout << "load_latest=false; using default-initialized parameters" << std::endl;

            PrintClassificationProofDiagnostics(l, t, fromDate, toDate);
            PrintPhase2TensorDiagnostics(l, t, fromDate, toDate);

            
            size_t totalCorrectLog = 0;
            size_t totalActedLog = 0;
            size_t totalWindows = 0;
            double totalAbsErrMove = 0.0;
            size_t totalCorrectDir = 0;
            size_t totalActedDir = 0;
            size_t totalConfusion[direction_output_size][direction_output_size] = {};
            // Iterate all batches (including trailing partial batch) and process each via CalculateBatch
            std::cout << std::setprecision(15);
                for(auto e = 0; e < epoch_count; e++)
                {
                    t.ForEachBatch( [&](auto b)
                                   {
                        if constexpr (inference_only)
                        {
                            auto [correctLog, actedLog, windows, absErrMove, correctDir, actedDir] = ProcessBatchPredict(l, t, b);
                            totalCorrectLog += correctLog;
                            totalActedLog += actedLog;
                            totalWindows += windows;
                            totalAbsErrMove += absErrMove;
                            totalCorrectDir += correctDir;
                            totalActedDir += actedDir;

                            if (l.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
                            {
                                for (auto it = b.begin(); it + window_size - 1 + prediction_horizon < b.end(); ++it)
                                {
                                    auto w = Window{it, it + window_size};
                                    const auto probs = l.PredictNextDirectionProbs(w, /*resetState=*/true);
                                    const int pred = (probs[0] > probs[1] && probs[0] > probs[2]) ? 0
                                                   : ((probs[2] > probs[1] && probs[2] > probs[0]) ? 2 : 1);

                                    const auto lastIt = it + window_size - 1;
                                    const auto targetIt = it + window_size - 1 + prediction_horizon;
                                    const float close_t = t.RawCloseAtIterator(lastIt);
                                    const float close_target = t.RawCloseAtIterator(targetIt);
                                    const float v_unscaled =
                                        (std::isfinite(close_t) && std::isfinite(close_target) &&
                                         close_t > 0.0f && close_target > 0.0f)
                                            ? std::log(close_target / close_t)
                                            : 0.0f;
                                    const int actual = (v_unscaled > c_next_threshold) ? 2
                                                     : ((v_unscaled < -c_next_threshold) ? 0 : 1);

                                    ++totalConfusion[actual][pred];
                                }
                            }
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
                            
                            auto [loss, _unused1, _unused2] = l.CalculateBatch(b);
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
            if constexpr (inference_only)
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
                if constexpr ( save_enable && !inference_only )
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
                    if constexpr (!(save_enable && !inference_only))
                        std::cout << "save_enable=false; skipping model save" << std::endl;
                    else if constexpr (inference_only)
                        std::cout << "inference_only=true; skipping model save" << std::endl;
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
