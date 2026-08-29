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
#include <signal.h>
#include <ctime>
#include <unistd.h>
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
#include "ExperimentScheduler.hpp"
#include "ExperimentMetaAnalyzer.hpp"
#include "GlobalExperimentControl.hpp"
#include "CanonicalSymbol.hpp"
#include "ExperimentCurrentOperation.hpp"
#include "FxPriceSanity.hpp"
#include "WorkerLifecycleDiagnostics.hpp"
#include "Donchian20Mode.hpp"
#include "DonchianLookback.hpp"
#include "FeatureWarmupScope.hpp"
#include "ModelInputContract.hpp"
#include "ModelInputExpansion.hpp"
#include "ReturnFeatureHistory.hpp"
#include "InferenceProfitabilityRepository.hpp"
#include "ProfitabilityVerificationRepository.hpp"
#include "EconomicEventImportService.hpp"
#include "EconomicEventConsensusImport.hpp"
#include "EconomicEventRepository.hpp"

#ifndef EARLY_STOP_PATIENCE
#define EARLY_STOP_PATIENCE 10
#endif

#include <MetaNN/operation/math/sigmoid.h>
#include <MetaNN/operation/math/tanh.h>
#include <MetaNN/operation/tensor/reshape.h>
#include <MetaNN/operation/tensor/slice.h>
#include "scalable_tensor.h"

void PrintAndResetDistribution();

namespace
{
std::string ForexDbConnectionString();

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

static_assert(BaselineReturnFeatureCount == EA::kModelReturnFeatureCount,
              "runtime return-feature contract must contain four columns");

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

enum class RuntimeLogLevel
{
    Quiet = 0,
    Summary = 1,
    Diagnostic = 2
};

RuntimeLogLevel gRuntimeLogLevel = RuntimeLogLevel::Summary;
bool gRuntimeInferenceMode = default_runtime_inference_mode;

bool LogSummary()
{
    return static_cast<int>(gRuntimeLogLevel) >= static_cast<int>(RuntimeLogLevel::Summary);
}

bool LogDiagnostic()
{
    return gRuntimeLogLevel == RuntimeLogLevel::Diagnostic;
}

RuntimeLogLevel ParseRuntimeLogLevel(const std::string& value)
{
    if (value == "quiet")
        return RuntimeLogLevel::Quiet;
    if (value == "summary")
        return RuntimeLogLevel::Summary;
    if (value == "diagnostic")
        return RuntimeLogLevel::Diagnostic;
    throw std::invalid_argument("invalid --log-level value '" + value + "'; expected quiet, summary, or diagnostic");
}

const char* RuntimeLogLevelName(RuntimeLogLevel level)
{
    switch (level)
    {
        case RuntimeLogLevel::Quiet: return "quiet";
        case RuntimeLogLevel::Summary: return "summary";
        case RuntimeLogLevel::Diagnostic: return "diagnostic";
    }
    return "unknown";
}

class NullLogBuffer : public std::streambuf
{
public:
    int overflow(int c) override
    {
        return c;
    }
};

std::ostream& DiagnosticOut()
{
    static NullLogBuffer nullBuffer;
    static std::ostream nullStream(&nullBuffer);
    return LogDiagnostic() ? std::cout : nullStream;
}

class ScopedDiagnosticCoutSilencer
{
public:
    ScopedDiagnosticCoutSilencer()
    {
        if (!LogDiagnostic())
            previousBuffer = std::cout.rdbuf(nullStream.rdbuf());
    }

    ~ScopedDiagnosticCoutSilencer()
    {
        if (previousBuffer != nullptr)
            std::cout.rdbuf(previousBuffer);
    }

    ScopedDiagnosticCoutSilencer(const ScopedDiagnosticCoutSilencer&) = delete;
    ScopedDiagnosticCoutSilencer& operator=(const ScopedDiagnosticCoutSilencer&) = delete;

private:
    NullLogBuffer nullBuffer;
    std::ostream nullStream { &nullBuffer };
    std::streambuf* previousBuffer = nullptr;
};

const char* CurrentRangeKindLabel()
{
    return gRuntimeInferenceMode ? "inference" : "train";
}

struct TrainConfigMeta
{
    std::optional<std::string> symbol;
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
    std::optional<float> coreLrMult;
    std::optional<float> headWeightLrMult;
    std::optional<float> headBiasLrMult;
};

struct ModelConfigValidationResult
{
    std::optional<TrainConfigMeta> trainConfigMeta;
    bool configMatch = false;
    bool hasMismatch = false;
    bool metadataGap = false;
};

void PrintDatabaseModelSymbol(long long modelId, const std::string& symbol);
void PrintLegacyModelSymbol(long long modelId, const std::string& symbol);
void PrintMissingModelSymbol(long long modelId);
void ValidateRuntimeSymbolMatchesModel(const std::optional<std::string>& runtimeSymbol,
                                       const std::string& modelSymbol);

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
    DiagnosticOut() << "DIAG_CLASS_THRESHOLDS"
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
    DiagnosticOut() << "DIAG_CLASS_LABEL_RULE"
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
            DiagnosticOut() << "DIAG_CLASS_ROW"
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
    DiagnosticOut() << "DIAG_CLASS_HIST"
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

void PrintPhase2ScalarLine(const char* label,
                           const std::string& rangeKind,
                           const std::string& fromDate,
                           const std::string& toDate,
                           const char* name,
                           const Phase2ScalarStats& s)
{
    DiagnosticOut() << label
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
    const std::string tensorSymbol = tensor.TableName();
    const auto rawFxBounds = EA::FxPriceSanity::BoundsForSymbol(tensorSymbol);

    DiagnosticOut() << "DIAG_DATA_CONFIG"
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",source=postgresql_candlestick_query"
              << ",rows=" << rows
              << ",raw_open=1"
              << ",raw_close=1"
              << ",raw_high=1"
              << ",raw_low=1"
              << ",raw_volume=1"
              << ",raw_target=0"
              << std::endl;
    DiagnosticOut() << "DIAG_FEATURE_CONFIG"
              << ",range_kind=" << rangeKind
              << ",from=" << fromDate
              << ",to=" << toDate
              << ",base_feature_cols=" << cols
              << ",feature_size_constant=" << feature_size
              << ",kFeatureScale=" << kFeatureScale
              << ",feature_uses_future_values=0"
              << ",feature_uses_target_column=0"
              << std::endl;
    DiagnosticOut() << "DIAG_NORM_CONFIG"
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
    DiagnosticOut() << "DIAG_DATA_WARN"
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

        const bool badOpen = !EA::FxPriceSanity::IsSanePrice(rawOpen, rawFxBounds);
        const bool badClose = !EA::FxPriceSanity::IsSanePrice(rawClose, rawFxBounds);
        const bool badHigh = !EA::FxPriceSanity::IsSanePrice(rawHigh, rawFxBounds);
        const bool badLow = !EA::FxPriceSanity::IsSanePrice(rawLow, rawFxBounds);
        const bool highLtLow = std::isfinite(rawHigh) && std::isfinite(rawLow) && rawHigh < rawLow;
        if (badOpen || badClose || badHigh || badLow || highLtLow)
        {
            ++badRawRowCount;
            if (badRawRowPrinted < kDiagBadRowLimit)
            {
                DiagnosticOut() << "DIAG_DATA_BAD_ROW"
                          << ",range_kind=" << rangeKind
                          << ",row=" << rowIdx
                          << ",dt=" << dt
                          << ",open=" << rawOpen
                          << ",close=" << rawClose
                          << ",high=" << rawHigh
                          << ",low=" << rawLow
                          << ",symbol=" << tensorSymbol
                          << ",bad_open=" << static_cast<int>(badOpen)
                          << ",bad_close=" << static_cast<int>(badClose)
                          << ",bad_high=" << static_cast<int>(badHigh)
                          << ",bad_low=" << static_cast<int>(badLow)
                          << ",high_lt_low=" << static_cast<int>(highLtLow)
                          << ",sane_lower_bound=" << rawFxBounds.lower
                          << ",sane_upper_bound=" << rawFxBounds.upper
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
                    DiagnosticOut() << "DIAG_FEATURE_SPIKE"
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
    DiagnosticOut() << "DIAG_DATA_BAD_ROW_SUMMARY"
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

    DiagnosticOut() << "DIAG_FEATURE_BASE_GLOBAL"
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
        DiagnosticOut() << "DIAG_FEATURE_BASE_COL"
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

    DiagnosticOut() << "DIAG_FEATURE_SPIKE_GLOBAL"
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
        DiagnosticOut() << "DIAG_FEATURE_SPIKE_COL"
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

    DiagnosticOut() << "DIAG_FEATURE_SPIKE_SUMMARY"
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

        DiagnosticOut() << "DIAG_FEATURE_BASE_WINDOW"
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
            const bool badFutureHigh = !EA::FxPriceSanity::IsSanePrice(futureHigh, rawFxBounds);
            const bool badFutureLow = !EA::FxPriceSanity::IsSanePrice(futureLow, rawFxBounds);
            const bool futureHighLtLow = std::isfinite(futureHigh) && std::isfinite(futureLow) && futureHigh < futureLow;
            if (!(badFutureHigh || badFutureLow || futureHighLtLow))
                continue;

            windowHasBadLookahead = true;
            ++badLookaheadRows;
            if (badLookaheadPrinted < kDiagBadRowLimit)
            {
                const auto info = BuildLookaheadClassInfo(tensor, startIt);
                DiagnosticOut() << "DIAG_DATA_BAD_LOOKAHEAD"
                          << ",range_kind=" << rangeKind
                          << ",start_row=" << startRow
                          << ",start_dt=" << tensor.RawTimeAtIterator(lastIt)
                          << ",close_t=" << tensor.RawCloseAtIterator(lastIt)
                          << ",future_row=" << futureRow
                          << ",future_dt=" << tensor.RawTimeAtIterator(futureIt)
                          << ",future_high=" << futureHigh
                          << ",future_low=" << futureLow
                          << ",symbol=" << tensorSymbol
                          << ",bad_high=" << static_cast<int>(badFutureHigh)
                          << ",bad_low=" << static_cast<int>(badFutureLow)
                          << ",high_lt_low=" << static_cast<int>(futureHighLtLow)
                          << ",sane_lower_bound=" << rawFxBounds.lower
                          << ",sane_upper_bound=" << rawFxBounds.upper
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

    DiagnosticOut() << "DIAG_DATA_BAD_LOOKAHEAD_SUMMARY"
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
        DiagnosticOut() << "DIAG_FEATURE_WARN"
                  << ",range_kind=" << rangeKind
                  << ",kind=absmax_exceeds_sane_threshold"
                  << ",threshold=" << kFeatureAbsMaxWarnThreshold
                  << ",absmax=" << featureGlobalStats.absmax
                  << std::endl;
    }

    if (zeroVarFeatureCount > 0)
    {
        DiagnosticOut() << "DIAG_FEATURE_WARN"
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
    pqxx::connection c_forex { ForexDbConnectionString() };
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
    pqxx::connection c_forex { ForexDbConnectionString() };
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

const char* ClassName3(size_t cls)
{
    switch (cls)
    {
        case 0: return "down";
        case 1: return "neutral";
        case 2: return "up";
        default: return "unknown";
    }
}

double SafeRatio(double numerator, double denominator)
{
    return denominator != 0.0 ? numerator / denominator : 0.0;
}

double ProfitFactor(double grossPositiveLogReturn, double grossNegativeLogReturn)
{
    return grossNegativeLogReturn < 0.0
        ? grossPositiveLogReturn / std::fabs(grossNegativeLogReturn)
        : 0.0;
}

void PrintEvalTradingMetrics(
    const EA::InferenceProfitability::Statistics& stats,
    const size_t confusion[direction_output_size][direction_output_size])
{
    size_t total = 0;
    size_t correct = 0;
    double recallSum = 0.0;
    double f1Sum = 0.0;

    for (size_t cls = 0; cls < direction_output_size; ++cls)
    {
        const size_t tp = confusion[cls][cls];
        size_t predicted = 0;
        size_t support = 0;
        for (size_t i = 0; i < direction_output_size; ++i)
        {
            support += confusion[cls][i];
            predicted += confusion[i][cls];
        }

        total += support;
        correct += tp;

        const double precision = SafeRatio(static_cast<double>(tp), static_cast<double>(predicted));
        const double recall = SafeRatio(static_cast<double>(tp), static_cast<double>(support));
        const double f1 = (precision + recall) > 0.0
            ? (2.0 * precision * recall / (precision + recall))
            : 0.0;

        recallSum += recall;
        f1Sum += f1;

        std::cout << "PER_CLASS_METRICS"
                  << ",class=" << ClassName3(cls)
                  << ",precision=" << precision
                  << ",recall=" << recall
                  << ",f1=" << f1
                  << ",support=" << support
                  << std::endl;
    }

    const double accuracy = SafeRatio(static_cast<double>(correct), static_cast<double>(total));
    const double macroF1 = f1Sum / static_cast<double>(direction_output_size);
    const double balancedAccuracy = recallSum / static_cast<double>(direction_output_size);

    std::cout << "SUMMARY_CLASS_METRICS"
              << ",accuracy=" << accuracy
              << ",macro_f1=" << macroF1
              << ",balanced_accuracy=" << balancedAccuracy
              << std::endl;

    const size_t directionalSamples =
        confusion[0][0] + confusion[0][1] + confusion[0][2] +
        confusion[2][0] + confusion[2][1] + confusion[2][2];
    const size_t directionalCorrect = confusion[0][0] + confusion[2][2];
    const double directionalAccuracy =
        SafeRatio(static_cast<double>(directionalCorrect), static_cast<double>(directionalSamples));

    std::cout << "DIRECTIONAL_ONLY_METRICS"
              << ",samples=" << directionalSamples
              << ",correct=" << directionalCorrect
              << ",accuracy=" << directionalAccuracy
              << std::endl;

    const size_t flatCount = confusion[0][1] + confusion[1][1] + confusion[2][1];
    const double winRate = SafeRatio(
        static_cast<double>(stats.winningActionableCount),
        static_cast<double>(stats.actionableCount));
    const double avgLogReturn =
        stats.AverageTerminalHorizonLogReturnPerActionablePrediction()
            .value_or(0.0);
    const double profitFactor = ProfitFactor(
        stats.grossPositiveTerminalHorizonLogReturnSum,
        stats.grossNegativeTerminalHorizonLogReturnSum);
    const double aggregateTotalLogReturn =
        stats.aggregateTerminalHorizonLogReturnSum;

    std::cout << "TRADING_SIGNAL_METRICS"
              << ",trade_count=" << stats.actionableCount
              << ",long_count=" << stats.upActionableCount
              << ",short_count=" << stats.downActionableCount
              << ",flat_count=" << flatCount
              << ",win_count=" << stats.winningActionableCount
              << ",loss_count=" << stats.losingActionableCount
              << ",win_rate=" << winRate
              << ",avg_log_return=" << avgLogReturn
              << ",gross_positive_log_return="
              << stats.grossPositiveTerminalHorizonLogReturnSum
              << ",gross_negative_log_return="
              << stats.grossNegativeTerminalHorizonLogReturnSum
              << ",profit_factor=" << profitFactor
              << std::endl;

    const double longWinRate = SafeRatio(
        static_cast<double>(stats.upWinningActionableCount),
        static_cast<double>(stats.upActionableCount));
    const double shortWinRate = SafeRatio(
        static_cast<double>(stats.downWinningActionableCount),
        static_cast<double>(stats.downActionableCount));
    const double longAvgLogReturn = SafeRatio(
        stats.upTerminalHorizonLogReturnSum,
        static_cast<double>(stats.upActionableCount));
    const double shortAvgLogReturn = SafeRatio(
        stats.downTerminalHorizonLogReturnSum,
        static_cast<double>(stats.downActionableCount));
    const double longProfitFactor = ProfitFactor(
        stats.upGrossPositiveTerminalHorizonLogReturnSum,
        stats.upGrossNegativeTerminalHorizonLogReturnSum);
    const double shortProfitFactor = ProfitFactor(
        stats.downGrossPositiveTerminalHorizonLogReturnSum,
        stats.downGrossNegativeTerminalHorizonLogReturnSum);

    std::cout << "TRADING_SIDE_METRICS"
              << ",long_count=" << stats.upActionableCount
              << ",long_win_count=" << stats.upWinningActionableCount
              << ",long_loss_count=" << stats.upLosingActionableCount
              << ",long_win_rate=" << longWinRate
              << ",long_total_log_return=" << stats.upTerminalHorizonLogReturnSum
              << ",long_gross_positive_log_return="
              << stats.upGrossPositiveTerminalHorizonLogReturnSum
              << ",long_gross_negative_log_return="
              << stats.upGrossNegativeTerminalHorizonLogReturnSum
              << ",long_profit_factor=" << longProfitFactor
              << ",long_avg_log_return=" << longAvgLogReturn
              << ",short_count=" << stats.downActionableCount
              << ",short_win_count=" << stats.downWinningActionableCount
              << ",short_loss_count=" << stats.downLosingActionableCount
              << ",short_win_rate=" << shortWinRate
              << ",short_total_log_return=" << stats.downTerminalHorizonLogReturnSum
              << ",short_gross_positive_log_return="
              << stats.downGrossPositiveTerminalHorizonLogReturnSum
              << ",short_gross_negative_log_return="
              << stats.downGrossNegativeTerminalHorizonLogReturnSum
              << ",short_profit_factor=" << shortProfitFactor
              << ",short_avg_log_return=" << shortAvgLogReturn
              << std::endl;

    const size_t sideTradeCount =
        stats.upActionableCount + stats.downActionableCount;
    const double sideTotalLogReturn =
        stats.upTerminalHorizonLogReturnSum +
        stats.downTerminalHorizonLogReturnSum;
    constexpr double kTradingAttributionTolerance = 1e-9;
    if (sideTradeCount != stats.actionableCount ||
        std::fabs(sideTotalLogReturn - aggregateTotalLogReturn) > kTradingAttributionTolerance)
    {
        std::cout << "TRADING_SIDE_VALIDATION_WARNING"
                  << ",trade_count=" << stats.actionableCount
                  << ",long_plus_short_count=" << sideTradeCount
                  << ",aggregate_total_log_return=" << aggregateTotalLogReturn
                  << ",long_plus_short_total_log_return=" << sideTotalLogReturn
                  << ",abs_log_return_diff=" << std::fabs(sideTotalLogReturn - aggregateTotalLogReturn)
                  << std::endl;
    }

    std::string dominantSide = "balanced";
    const double totalPositiveSidePnl =
        std::max(0.0, stats.upTerminalHorizonLogReturnSum) +
        std::max(0.0, stats.downTerminalHorizonLogReturnSum);
    if (totalPositiveSidePnl > 0.0)
    {
        const double longFractionOfTotalPnl =
            std::max(0.0, stats.upTerminalHorizonLogReturnSum) /
            totalPositiveSidePnl;
        const double shortFractionOfTotalPnl =
            std::max(0.0, stats.downTerminalHorizonLogReturnSum) /
            totalPositiveSidePnl;
        if (longFractionOfTotalPnl > 0.60)
            dominantSide = "long";
        else if (shortFractionOfTotalPnl > 0.60)
            dominantSide = "short";

        std::cout << "TRADING_SIDE_SUMMARY"
                  << ",dominant_side=" << dominantSide
                  << ",long_fraction_of_total_pnl=" << longFractionOfTotalPnl
                  << ",short_fraction_of_total_pnl=" << shortFractionOfTotalPnl
                  << std::endl;
    }
    else
    {
        std::cout << "TRADING_SIDE_SUMMARY"
                  << ",dominant_side=" << dominantSide
                  << ",long_fraction_of_total_pnl=0"
                  << ",short_fraction_of_total_pnl=0"
                  << std::endl;
    }
}

struct ModelAcceptanceSummary
{
    bool acceptModel = false;
    std::string rejectReason = "none";
    std::array<double, direction_output_size> predFrac {0.0, 0.0, 0.0};
    std::array<double, direction_output_size> actualFrac {0.0, 0.0, 0.0};
    std::array<double, direction_output_size> precision {0.0, 0.0, 0.0};
    std::array<double, direction_output_size> recall {0.0, 0.0, 0.0};
};

constexpr double kAcceptNeutralMax = 0.60;
constexpr double kAcceptMinDown = 0.15;
constexpr double kAcceptMinUp = 0.15;

ModelAcceptanceSummary ComputeModelAcceptanceSummary(const size_t confusion[direction_output_size][direction_output_size])
{
    ModelAcceptanceSummary summary;
    std::array<size_t, direction_output_size> actualCounts {0, 0, 0};
    std::array<size_t, direction_output_size> predCounts {0, 0, 0};
    size_t total = 0;

    for (size_t actual = 0; actual < direction_output_size; ++actual)
    {
        for (size_t pred = 0; pred < direction_output_size; ++pred)
        {
            actualCounts[actual] += confusion[actual][pred];
            predCounts[pred] += confusion[actual][pred];
            total += confusion[actual][pred];
        }
    }

    for (size_t cls = 0; cls < direction_output_size; ++cls)
    {
        summary.predFrac[cls] = SafeRatio(static_cast<double>(predCounts[cls]), static_cast<double>(total));
        summary.actualFrac[cls] = SafeRatio(static_cast<double>(actualCounts[cls]), static_cast<double>(total));
        summary.precision[cls] = SafeRatio(static_cast<double>(confusion[cls][cls]), static_cast<double>(predCounts[cls]));
        summary.recall[cls] = SafeRatio(static_cast<double>(confusion[cls][cls]), static_cast<double>(actualCounts[cls]));
    }

    std::vector<std::string> rejectReasons;
    if (summary.predFrac[1] > kAcceptNeutralMax)
        rejectReasons.push_back("pred_neutral_gt_0.60");
    if (summary.predFrac[0] < kAcceptMinDown)
        rejectReasons.push_back("pred_down_lt_0.15");
    if (summary.predFrac[2] < kAcceptMinUp)
        rejectReasons.push_back("pred_up_lt_0.15");

    if (!rejectReasons.empty())
    {
        std::ostringstream oss;
        for (size_t i = 0; i < rejectReasons.size(); ++i)
        {
            if (i) oss << ";";
            oss << rejectReasons[i];
        }
        summary.rejectReason = oss.str();
    }
    summary.acceptModel = rejectReasons.empty();
    return summary;
}

void PrintModelAcceptanceDiagnostic(const size_t confusion[direction_output_size][direction_output_size])
{
    const ModelAcceptanceSummary summary = ComputeModelAcceptanceSummary(confusion);

    std::cout << "MODEL_ACCEPTANCE"
              << ",ACCEPT_MODEL=" << (summary.acceptModel ? "true" : "false")
              << ",REJECT_REASON=" << summary.rejectReason
              << ",threshold_neutral_max=" << kAcceptNeutralMax
              << ",threshold_min_down=" << kAcceptMinDown
              << ",threshold_min_up=" << kAcceptMinUp
              << ",pred_down=" << summary.predFrac[0]
              << ",pred_neutral=" << summary.predFrac[1]
              << ",pred_up=" << summary.predFrac[2]
              << ",actual_down=" << summary.actualFrac[0]
              << ",actual_neutral=" << summary.actualFrac[1]
              << ",actual_up=" << summary.actualFrac[2]
              << ",precision_down=" << summary.precision[0]
              << ",precision_neutral=" << summary.precision[1]
              << ",precision_up=" << summary.precision[2]
              << ",recall_down=" << summary.recall[0]
              << ",recall_neutral=" << summary.recall[1]
              << ",recall_up=" << summary.recall[2]
              << std::endl;
}

static PredictionStats ProcessBatchPredict(
    EA::LSTM& l,
    const Tensor& tensor,
    const Window& b,
    EA::InferenceProfitability::Accumulator& profitability)
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
            DiagnosticOut() << "DIAG_INFER_LABEL_RULE"
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
                DiagnosticOut() << "DIAG_INFER_LABEL_ROW"
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

            profitability.Observe(pred, labelInfo.closeT, labelInfo.targetClose);
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

        DiagnosticOut() << "pred_class stats: min=" << s_pred.min << " max=" << s_pred.max
                        << " mean=" << s_pred.mean << " std=" << s_pred.std
                        << " uniq~=" << s_pred.uniq << std::endl;
        DiagnosticOut() << "act_class stats: min=" << s_act.min << " max=" << s_act.max
                        << " mean=" << s_act.mean << " std=" << s_act.std
                        << " uniq~=" << s_act.uniq << std::endl;
        DiagnosticOut() << "pred_max_prob stats: min=" << s_prob.min << " max=" << s_prob.max
                        << " mean=" << s_prob.mean << " std=" << s_prob.std
                        << " uniq~=" << s_prob.uniq << std::endl;

        const double acc = static_cast<double>(correct) / static_cast<double>(N) * 100.0;
        DiagnosticOut() << "3-class accuracy: " << acc << "% over " << N << " windows" << std::endl;
        DiagnosticOut() << "3-class confusion matrix (rows=actual [down,neutral,up], cols=pred [down,neutral,up]): "
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
    DiagnosticOut() << "pred_raw stats: min=" << s_raw.min << " max=" << s_raw.max
                    << " mean=" << s_raw.mean << " std=" << s_raw.std
                    << " uniq~=" << s_raw.uniq << std::endl;
    DiagnosticOut() << "std ratio (pred_raw/actLogRet): " << (s_gt.std > 0.0 ? (s_raw.std / s_gt.std) : 0.0) << std::endl;
    DiagnosticOut() << "means: pred_raw=" << s_raw.mean << " actLogRet=" << s_gt.mean << std::endl;

    DiagnosticOut() << "pred_rel stats: min=" << s_rel.min << " max=" << s_rel.max
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
    DiagnosticOut() << "Direction accuracy (log-return): " << accLog
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

    if (N > 0)  DiagnosticOut() << "Batch MAE (relative move fraction): "
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
std::string ForexDbConnectionString()
{
    const char* host = std::getenv("FOREX_DB_HOST");
    const char* database = std::getenv("FOREX_DB_NAME");
    return "hostaddr=" +
           std::string{
               host != nullptr && *host != '\0'
                   ? host
                   : "127.0.0.1"} +
           " gssencmode=disable user=pqxx dbname=" +
           std::string{
               database != nullptr && *database != '\0'
                   ? database
                   : dbName};
}

std::string LstmDbConnectionString()
{
    const char* host = std::getenv("LSTM_DB_HOST");
    const char* database = std::getenv("LSTM_DB_NAME");
    return "hostaddr=" +
           std::string{
               host != nullptr && *host != '\0'
                   ? host
                   : "127.0.0.1"} +
           " gssencmode=disable user=pqxx dbname=" +
           std::string{
               database != nullptr && *database != '\0'
                   ? database
                   : dbModelName};
}

std::string CurrentUtcDate()
{
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%d");
    return output.str();
}

std::optional<long long> ResolveSchedulerExperimentIdForCurrentProcess()
{
    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        pqxx::result rows = w.exec_params(
            "SELECT experiment_id "
            "FROM experiment "
            "WHERE worker_pid = $1 "
            "AND status = 'running' "
            "AND phase = 'train' "
            "ORDER BY updated_at DESC "
            "LIMIT 2;",
            static_cast<int>(::getpid()));
        w.commit();
        if (rows.size() == 1)
            return rows[0][0].as<long long>();
    }
    catch (const std::exception& e)
    {
        std::cerr << "SCHEDULER_PROGRESS_EXPERIMENT_LOOKUP_FAILED"
                  << ",pid=" << static_cast<int>(::getpid())
                  << ",error=" << e.what()
                  << std::endl;
    }
    return std::nullopt;
}

void UpdateSchedulerExperimentProgress(const std::optional<long long>& experimentId,
                                       int completedEpoch)
{
    const std::optional<long long> effectiveExperimentId =
        experimentId.has_value() ? experimentId : ResolveSchedulerExperimentIdForCurrentProcess();
    if (!effectiveExperimentId.has_value())
        return;
    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        w.exec("SET TRANSACTION READ WRITE;");
        w.exec_params(
            "UPDATE experiment "
            "SET current_epoch = $1, worker_pid = $2, "
            "current_operation = $3, updated_at = now() "
            "WHERE experiment_id = $4;",
            completedEpoch,
            static_cast<int>(::getpid()),
            std::string{EA::ExperimentLifecycle::kTrainOperation},
            *effectiveExperimentId);
        w.commit();
    }
    catch (const std::exception& e)
    {
        std::cerr << "SCHEDULER_PROGRESS_UPDATE_FAILED"
                  << ",experiment_id=" << *effectiveExperimentId
                  << ",epoch=" << completedEpoch
                  << ",error=" << e.what()
                  << std::endl;
    }
}

bool SchedulerExperimentColumnExists(pqxx::work& w, const std::string& columnName)
{
    pqxx::result rows = w.exec_params(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'experiment' AND column_name = $1 LIMIT 1;",
        columnName);
    return !rows.empty();
}

bool SchedulerTableExists(pqxx::work& w, const std::string& tableName)
{
    pqxx::result rows = w.exec_params(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = $1 LIMIT 1;",
        tableName);
    return !rows.empty();
}

bool SchedulerColumnExists(pqxx::work& w,
                           const std::string& tableName,
                           const std::string& columnName)
{
    pqxx::result rows = w.exec_params(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2 LIMIT 1;",
        tableName,
        columnName);
    return !rows.empty();
}

struct CheckpointInferConfig
{
    bool enabled = false;
    std::string symbol;
    int predictionHorizon = 0;
    std::optional<int> minEpoch;
    std::optional<int> interval;
};

std::optional<CheckpointInferConfig> LoadCheckpointInferConfig(const std::optional<long long>& experimentId,
                                                               int checkpointEpoch,
                                                               const std::optional<int>& launchCheckpointEvery)
{
    if (!experimentId.has_value())
        return std::nullopt;

    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        if (!SchedulerTableExists(w, "experiment") ||
            !SchedulerTableExists(w, "experiment_checkpoint_eval"))
        {
            w.commit();
            return std::nullopt;
        }

        const bool hasCheckpointInferEnabled =
            SchedulerExperimentColumnExists(w, "checkpoint_infer_enabled");
        const bool hasOpportunisticCheckpointInfer =
            SchedulerExperimentColumnExists(w, "opportunistic_checkpoint_infer");
        if (!hasCheckpointInferEnabled && !hasOpportunisticCheckpointInfer)
        {
            w.commit();
            return std::nullopt;
        }

        std::ostringstream sql;
        sql << "SELECT ";
        if (hasCheckpointInferEnabled && hasOpportunisticCheckpointInfer)
            sql << "(checkpoint_infer_enabled OR opportunistic_checkpoint_infer)";
        else if (hasCheckpointInferEnabled)
            sql << "checkpoint_infer_enabled";
        else
            sql << "opportunistic_checkpoint_infer";
        sql << ", symbol, prediction_horizon, checkpoint_infer_min_epoch, "
            << "COALESCE(checkpoint_infer_interval, NULLIF(checkpoint_interval, 0)) "
            << "FROM experiment WHERE experiment_id = $1;";

        pqxx::result rows = w.exec_params(sql.str(), *experimentId);
        w.commit();
        if (rows.empty())
            return std::nullopt;

        CheckpointInferConfig config;
        config.enabled = !rows[0][0].is_null() && rows[0][0].as<bool>();
        config.symbol = rows[0][1].as<std::string>();
        config.predictionHorizon = rows[0][2].as<int>();
        if (!rows[0][3].is_null())
            config.minEpoch = rows[0][3].as<int>();
        if (!rows[0][4].is_null())
            config.interval = rows[0][4].as<int>();
        else if (launchCheckpointEvery.has_value() && *launchCheckpointEvery > 0)
            config.interval = *launchCheckpointEvery;

        if (!config.enabled)
        {
            std::cout << "CHECKPOINT_INFER_SKIPPED"
                      << " experiment_id=" << *experimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " reason=disabled"
                      << " symbol=" << config.symbol
                      << " horizon=" << config.predictionHorizon
                      << std::endl;
            return std::nullopt;
        }
        if (config.minEpoch.has_value() && checkpointEpoch < *config.minEpoch)
        {
            std::cout << "CHECKPOINT_INFER_SKIPPED"
                      << " experiment_id=" << *experimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " reason=below_min_epoch"
                      << " symbol=" << config.symbol
                      << " horizon=" << config.predictionHorizon
                      << std::endl;
            return std::nullopt;
        }
        if (!config.interval.has_value() || *config.interval <= 0)
        {
            std::cout << "CHECKPOINT_INFER_SKIPPED"
                      << " experiment_id=" << *experimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " reason=missing_interval"
                      << " symbol=" << config.symbol
                      << " horizon=" << config.predictionHorizon
                      << std::endl;
            return std::nullopt;
        }
        if (checkpointEpoch % *config.interval != 0)
        {
            std::cout << "CHECKPOINT_INFER_SKIPPED"
                      << " experiment_id=" << *experimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " reason=interval_mismatch"
                      << " interval=" << *config.interval
                      << " symbol=" << config.symbol
                      << " horizon=" << config.predictionHorizon
                      << std::endl;
            return std::nullopt;
        }

        return config;
    }
    catch (const std::exception& e)
    {
        std::cerr << "CHECKPOINT_INFER_SKIPPED"
                  << " experiment_id=" << *experimentId
                  << " checkpoint_epoch=" << checkpointEpoch
                  << " reason=config_load_failed"
                  << " error=" << e.what()
                  << std::endl;
        return std::nullopt;
    }
}

void QueueCheckpointInferenceIfEligible(const std::optional<long long>& schedulerExperimentId,
                                        const std::optional<int>& checkpointEvery,
                                        int checkpointEpoch,
                                        long long checkpointModelId)
{
    const std::optional<CheckpointInferConfig> config =
        LoadCheckpointInferConfig(schedulerExperimentId,
                                  checkpointEpoch,
                                  checkpointEvery);
    if (!config.has_value() || !schedulerExperimentId.has_value())
        return;

    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        w.exec("SET TRANSACTION READ WRITE;");
        EA::GlobalExperimentControl::AcquireCoordinationLock(w);
        if (!SchedulerTableExists(w, "experiment_checkpoint_eval"))
        {
            w.commit();
            return;
        }
        if (SchedulerExperimentColumnExists(w, "cancellation_request_id"))
        {
            const pqxx::result cancellation = w.exec(
                "SELECT cancellation_request_id FROM experiment "
                "WHERE experiment_id=$1;",
                pqxx::params{*schedulerExperimentId});
            if (!cancellation.empty() && !cancellation[0][0].is_null())
            {
                w.commit();
                std::cout << "CHECKPOINT_INFER_SKIPPED"
                          << " experiment_id=" << *schedulerExperimentId
                          << " checkpoint_epoch=" << checkpointEpoch
                          << " checkpoint_model_id=" << checkpointModelId
                          << " reason=global_cancellation"
                          << " symbol=" << config->symbol
                          << " horizon=" << config->predictionHorizon
                          << std::endl;
                return;
            }
        }

        pqxx::result modelRows = w.exec_params(
            "SELECT 1 FROM model WHERE model_id = $1 LIMIT 1;",
            checkpointModelId);
        if (modelRows.empty())
        {
            w.commit();
            std::cout << "CHECKPOINT_INFER_SKIPPED"
                      << " experiment_id=" << *schedulerExperimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " checkpoint_model_id=" << checkpointModelId
                      << " reason=model_not_found"
                      << " symbol=" << config->symbol
                      << " horizon=" << config->predictionHorizon
                      << std::endl;
            return;
        }

        std::cout << "CHECKPOINT_INFER_ELIGIBLE"
                  << " experiment_id=" << *schedulerExperimentId
                  << " checkpoint_epoch=" << checkpointEpoch
                  << " checkpoint_model_id=" << checkpointModelId
                  << " symbol=" << config->symbol
                  << " horizon=" << config->predictionHorizon
                  << std::endl;

        const bool hasParentExperimentId =
            SchedulerColumnExists(w, "experiment_checkpoint_eval", "parent_experiment_id");
        const bool hasSymbol =
            SchedulerColumnExists(w, "experiment_checkpoint_eval", "symbol");
        const bool hasPredictionHorizon =
            SchedulerColumnExists(w, "experiment_checkpoint_eval", "prediction_horizon");

        std::ostringstream sql;
        sql << "INSERT INTO experiment_checkpoint_eval (experiment_id";
        if (hasParentExperimentId)
            sql << ", parent_experiment_id";
        sql << ", checkpoint_epoch, checkpoint_model_id";
        if (hasSymbol)
            sql << ", symbol";
        if (hasPredictionHorizon)
            sql << ", prediction_horizon";
        sql << ") VALUES ($1";
        int param = 2;
        if (hasParentExperimentId)
            sql << ", $" << param++;
        sql << ", $" << param++ << ", $" << param++;
        if (hasSymbol)
            sql << ", $" << param++;
        if (hasPredictionHorizon)
            sql << ", $" << param++;
        sql << ") ON CONFLICT ";
        if (hasParentExperimentId)
            sql << "(parent_experiment_id, checkpoint_model_id, checkpoint_epoch) ";
        else
            sql << "(experiment_id, checkpoint_epoch, checkpoint_model_id) ";
        sql << "DO NOTHING RETURNING checkpoint_eval_id;";

        pqxx::result inserted;
        if (hasParentExperimentId && hasSymbol && hasPredictionHorizon)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->symbol,
                                     config->predictionHorizon);
        }
        else if (hasParentExperimentId && hasSymbol)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->symbol);
        }
        else if (hasParentExperimentId && hasPredictionHorizon)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->predictionHorizon);
        }
        else if (hasParentExperimentId)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId);
        }
        else if (hasSymbol && hasPredictionHorizon)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->symbol,
                                     config->predictionHorizon);
        }
        else if (hasSymbol)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->symbol);
        }
        else if (hasPredictionHorizon)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->predictionHorizon);
        }
        else
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId);
        }
        w.commit();

        if (inserted.empty())
        {
            std::cout << "CHECKPOINT_INFER_DUPLICATE"
                      << " experiment_id=" << *schedulerExperimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " checkpoint_model_id=" << checkpointModelId
                      << " symbol=" << config->symbol
                      << " horizon=" << config->predictionHorizon
                      << std::endl;
            return;
        }

        std::cout << "CHECKPOINT_INFER_QUEUED"
                  << " experiment_id=" << *schedulerExperimentId
                  << " checkpoint_epoch=" << checkpointEpoch
                  << " checkpoint_model_id=" << checkpointModelId
                  << " symbol=" << config->symbol
                  << " horizon=" << config->predictionHorizon
                  << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "CHECKPOINT_INFER_SKIPPED"
                  << " experiment_id=" << *schedulerExperimentId
                  << " checkpoint_epoch=" << checkpointEpoch
                  << " checkpoint_model_id=" << checkpointModelId
                  << " reason=queue_failed"
                  << " error=" << e.what()
                  << std::endl;
    }
}

struct CheckpointStopConfig
{
    int requestedEpoch = 0;
    int effectiveEpoch = 0;
    bool cancellationRequested = false;
};

std::optional<CheckpointStopConfig> LoadCheckpointStopConfig(const std::optional<long long>& experimentId,
                                                             int checkpointEpoch,
                                                             int targetEpochs,
                                                             const std::optional<int>& checkpointEvery)
{
    if (!experimentId.has_value() || !checkpointEvery.has_value() || *checkpointEvery <= 0)
        return std::nullopt;

    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        EA::GlobalExperimentControl::AcquireCoordinationLock(w);
        if (!SchedulerExperimentColumnExists(w, "stop_after_checkpoint_epoch"))
        {
            w.commit();
            return std::nullopt;
        }

        const bool hasCancellationControl =
            SchedulerExperimentColumnExists(w, "cancellation_request_id") &&
            SchedulerExperimentColumnExists(
                w, "last_checkpoint_stop_decision_epoch");
        pqxx::result rows = hasCancellationControl
            ? w.exec_params(
                  "SELECT stop_after_checkpoint_epoch,cancellation_request_id "
                  "FROM experiment WHERE experiment_id = $1 FOR UPDATE;",
                  *experimentId)
            : w.exec_params(
                  "SELECT stop_after_checkpoint_epoch,NULL::bigint "
                  "FROM experiment WHERE experiment_id = $1 FOR UPDATE;",
                  *experimentId);
        if (hasCancellationControl)
            w.exec_params(
                "UPDATE experiment "
                "SET last_checkpoint_stop_decision_epoch=$1,updated_at=now() "
                "WHERE experiment_id=$2;",
                checkpointEpoch,
                *experimentId);
        w.commit();
        if (rows.empty() || rows[0][0].is_null())
            return std::nullopt;

        const int requestedEpoch = rows[0][0].as<int>();
        if (requestedEpoch <= 0)
        {
            std::cout << "CHECKPOINT_STOP_IGNORED"
                      << " reason=invalid_requested_epoch"
                      << " experiment_id=" << *experimentId
                      << " requested_epoch=" << requestedEpoch
                      << std::endl;
            return std::nullopt;
        }
        if (requestedEpoch >= targetEpochs)
        {
            std::cout << "CHECKPOINT_STOP_IGNORED"
                      << " reason=requested_epoch_not_before_target"
                      << " experiment_id=" << *experimentId
                      << " requested_epoch=" << requestedEpoch
                      << " target_epochs=" << targetEpochs
                      << std::endl;
            return std::nullopt;
        }

        const int interval = *checkpointEvery;
        const int effectiveEpoch = ((requestedEpoch + interval - 1) / interval) * interval;
        if (effectiveEpoch >= targetEpochs)
        {
            std::cout << "CHECKPOINT_STOP_IGNORED"
                      << " reason=effective_epoch_not_before_target"
                      << " experiment_id=" << *experimentId
                      << " requested_epoch=" << requestedEpoch
                      << " effective_epoch=" << effectiveEpoch
                      << " target_epochs=" << targetEpochs
                      << std::endl;
            return std::nullopt;
        }
        if (checkpointEpoch < effectiveEpoch)
            return std::nullopt;

        std::cout << "CHECKPOINT_STOP_REQUESTED"
                  << " experiment_id=" << *experimentId
                  << " requested_epoch=" << requestedEpoch
                  << " effective_epoch=" << effectiveEpoch
                  << " checkpoint_epoch=" << checkpointEpoch
                  << std::endl;
        std::cout << "CHECKPOINT_STOP_LIVE_REQUEST_DETECTED"
                  << " experiment_id=" << *experimentId
                  << " requested_epoch=" << requestedEpoch
                  << " effective_epoch=" << effectiveEpoch
                  << " checkpoint_epoch=" << checkpointEpoch
                  << std::endl;
        return CheckpointStopConfig{
            requestedEpoch,
            effectiveEpoch,
            !rows[0][1].is_null()};
    }
    catch (const std::exception& e)
    {
        std::cerr << "CHECKPOINT_STOP_IGNORED"
                  << " reason=config_load_failed"
                  << " experiment_id=" << *experimentId
                  << " error=" << e.what()
                  << std::endl;
        return std::nullopt;
    }
}

bool RecordCheckpointStopReached(const std::optional<long long>& experimentId,
                                 const std::optional<long long>& workerAttemptId,
                                 int epoch,
                                 long long modelId)
{
    if (!experimentId.has_value() || !workerAttemptId.has_value())
        return false;

    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        w.exec("SET TRANSACTION READ WRITE;");
        EA::GlobalExperimentControl::AcquireCoordinationLock(w);
        if (!SchedulerExperimentColumnExists(w, "stopped_at_checkpoint_epoch") ||
            !SchedulerExperimentColumnExists(w, "stopped_at_checkpoint_model_id") ||
            !SchedulerExperimentColumnExists(w, "cancellation_request_id"))
        {
            w.commit();
            std::cout << "CHECKPOINT_STOP_IGNORED"
                      << " reason=migration_required"
                      << " experiment_id=" << *experimentId
                      << std::endl;
            return false;
        }

        const EA::GlobalExperimentControl::CheckpointStopRecordResult result =
            EA::GlobalExperimentControl::RecordCheckpointStopReached(
                w, experimentId, *workerAttemptId, epoch, modelId);
        w.commit();
        if (!result.recorded)
        {
            std::cout << "CHECKPOINT_STOP_IGNORED"
                      << " reason=" << result.detail
                      << " experiment_id=" << *experimentId
                      << std::endl;
            return false;
        }

        std::cout << "CHECKPOINT_STOP_REACHED"
                  << " experiment_id=" << *experimentId
                  << " epoch=" << epoch
                  << " model_id=" << modelId
                  << std::endl;
        if (result.cancellationRequested)
            std::cout << "GLOBAL_CANCELLATION_CHECKPOINT_REACHED"
                      << " request_id=" << *result.cancellationRequestId
                      << " experiment_id=" << *experimentId
                      << " epoch=" << epoch
                      << " model_id=" << modelId
                      << " infer_before_cancel="
                      << (result.inferenceRequested ? "1" : "0")
                      << " detail=" << result.detail
                      << std::endl;
        else
            std::cout << "CHECKPOINT_STOP_ADVANCE_TO_INFER"
                      << " experiment_id=" << *experimentId
                      << " model_id=" << modelId
                      << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "CHECKPOINT_STOP_IGNORED"
                  << " reason=record_failed"
                  << " experiment_id=" << *experimentId
                  << " epoch=" << epoch
                  << " model_id=" << modelId
                  << " error=" << e.what()
                  << std::endl;
        return false;
    }
}

struct LaunchArgs
{
    std::string fromDate;
    std::string toDate;
    bool positionalDateRangeSupplied = false;
    std::optional<std::string> symbol;
    std::optional<long long> modelId;
    std::optional<long long> resumeModelId;
    std::optional<int> targetEpochs;
    std::optional<std::string> newModelName;
    std::optional<bool> inferenceMode;
    std::optional<size_t> predictionHorizon;
    std::optional<double> thresholdLogret;
    std::optional<size_t> windowSize;
    std::optional<size_t> hiddenSize;
    std::optional<size_t> numLayers;
    std::optional<int> epochs;
    std::optional<double> coreLrMult;
    std::optional<double> headWeightLrMult;
    std::optional<double> headBiasLrMult;
    std::optional<int> checkpointEvery;
    std::optional<long long> schedulerExperimentId;
    std::optional<long long> schedulerCheckpointEvalId;
    std::optional<long long> schedulerWorkerAttemptId;
    std::optional<EA::TrainingObjective::Configuration> trainingObjective;
    std::optional<long long> inferStartAfterModelId;
    std::optional<EA::FeatureWarmupScope> featureWarmupScope;
    std::optional<Donchian20Mode> donchian20Mode;
    std::optional<std::size_t> donchianLookback;
    std::optional<RuntimeLogLevel> logLevel;
    std::optional<std::string> lstmProfileOutputPath;
    bool evalTrading = false;
    bool inferAll = false;
    bool forceInfer = false;
    bool lstmProfileHotspots = false;
    bool resumeExpandInputWidth = false;
    struct FrozenOutcomeSpec
    {
        std::string cohortHash;
        long long sourceExperimentId = -1;
        long long sourceModelId = -1;
        std::string outcomeStart;
        std::string outcomeEnd;
        std::string jobHash;
    };
    std::optional<FrozenOutcomeSpec> frozenOutcome;
};

struct LSTMHotspotProfileFinalizer
{
    bool enabled = false;
    std::optional<std::string> outputPath;

    ~LSTMHotspotProfileFinalizer()
    {
        if (!enabled)
            return;

        EA::LSTM::PrintHotspotProfileSummary();
        if (outputPath.has_value())
        {
            const bool wrote = EA::LSTM::WriteHotspotProfileReport(*outputPath);
            std::cout << "LSTM_PROFILE_REPORT"
                      << ",path=" << *outputPath
                      << ",written=" << (wrote ? 1 : 0)
                      << std::endl;
        }
    }
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

LaunchArgs::FrozenOutcomeSpec ParseFrozenOutcomeSpec(const std::string& value)
{
    std::vector<std::string> fields;
    std::size_t begin = 0;
    for (;;)
    {
        const std::size_t comma = value.find(',', begin);
        fields.push_back(value.substr(begin, comma - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    if (fields.size() != 6 ||
        std::any_of(fields.begin(), fields.end(),
                    [](const std::string& field) { return field.empty(); }))
        throw std::invalid_argument(
            "--run-frozen-model-outcome-inference requires "
            "COHORT_HASH,SOURCE_EXPERIMENT_ID,SOURCE_MODEL_ID,FROM_DATE,"
            "TO_DATE,JOB_HASH");
    LaunchArgs::FrozenOutcomeSpec spec;
    spec.cohortHash = fields[0];
    spec.sourceExperimentId = ParseModelIdArg(fields[1]);
    spec.sourceModelId = ParseModelIdArg(fields[2]);
    spec.outcomeStart = fields[3];
    spec.outcomeEnd = fields[4];
    spec.jobHash = fields[5];
    return spec;
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

int ParseNonNegativeIntArg(const std::string& optionName, const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument(optionName + " requires a non-empty integer value");

    size_t consumed = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a non-negative integer");
    }

    if (consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'; expected a non-negative integer");

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
    constexpr const char* kResumeModelPrefix = "--resume-model-id=";
    constexpr size_t kResumeModelPrefixLen = 18;

    LaunchArgs parsed;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg{ argv[i] };

        if (arg.rfind("--run-frozen-model-outcome-inference=", 0) == 0)
        {
            if (parsed.frozenOutcome)
                throw std::invalid_argument(
                    "--run-frozen-model-outcome-inference specified more than once");
            parsed.frozenOutcome = ParseFrozenOutcomeSpec(arg.substr(
                std::string{"--run-frozen-model-outcome-inference="}.size()));
        }
        else if (arg.rfind(kModelPrefix, 0) == 0)
        {
            if (parsed.modelId.has_value())
                throw std::invalid_argument("--model specified more than once");

            parsed.modelId = ParseModelIdArg(arg.substr(kModelPrefixLen));
        }
        else if (arg.rfind(kResumeModelPrefix, 0) == 0)
        {
            if (parsed.resumeModelId.has_value())
                throw std::invalid_argument("--resume-model-id specified more than once");
            parsed.resumeModelId = ParseModelIdArg(arg.substr(kResumeModelPrefixLen));
        }
        else if (arg == "--resume-model-id")
        {
            if (parsed.resumeModelId.has_value())
                throw std::invalid_argument("--resume-model-id specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--resume-model-id requires a model_id value");
            parsed.resumeModelId = ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--resume-expand-input-width")
        {
            if (parsed.resumeExpandInputWidth)
                throw std::invalid_argument(
                    "--resume-expand-input-width specified more than once");
            parsed.resumeExpandInputWidth = true;
        }
        else if (arg == "--target-epochs")
        {
            if (parsed.targetEpochs.has_value())
                throw std::invalid_argument("--target-epochs specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--target-epochs requires a value");
            parsed.targetEpochs = ParsePositiveIntArg("--target-epochs", argv[++i]);
        }
        else if (arg == "--new-model-name")
        {
            if (parsed.newModelName.has_value())
                throw std::invalid_argument("--new-model-name specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--new-model-name requires a value");
            std::string value{ argv[++i] };
            if (value.empty())
                throw std::invalid_argument("--new-model-name requires a non-empty value");
            parsed.newModelName = value;
        }
        else if (arg == "--checkpoint-every")
        {
            if (parsed.checkpointEvery.has_value())
                throw std::invalid_argument("--checkpoint-every specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--checkpoint-every requires a value");
            parsed.checkpointEvery = ParseNonNegativeIntArg("--checkpoint-every", argv[++i]);
        }
        else if (arg == "--donchian20-mode")
        {
            if (parsed.donchian20Mode.has_value())
                throw std::invalid_argument("--donchian20-mode specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--donchian20-mode requires enabled or zero_ablation");
            parsed.donchian20Mode = ParseDonchian20Mode(argv[++i]);
        }
        else if (arg == "--feature-warmup-scope")
        {
            if (parsed.featureWarmupScope.has_value())
                throw std::invalid_argument("--feature-warmup-scope specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--feature-warmup-scope requires a value");
            parsed.featureWarmupScope = EA::ParseFeatureWarmupScope(argv[++i]);
        }
        else if (arg == "--donchian-lookback")
        {
            if (parsed.donchianLookback.has_value())
                throw std::invalid_argument("--donchian-lookback specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--donchian-lookback requires a positive integer");
            parsed.donchianLookback = ParseDonchianLookback(argv[++i]);
        }
        else if (arg == "--infer-start-after-model-id")
        {
            if (parsed.inferStartAfterModelId.has_value())
                throw std::invalid_argument("--infer-start-after-model-id specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--infer-start-after-model-id requires a model_id value");
            parsed.inferStartAfterModelId = ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--scheduler-experiment-id")
        {
            if (parsed.schedulerExperimentId.has_value())
                throw std::invalid_argument("--scheduler-experiment-id specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--scheduler-experiment-id requires an experiment_id value");
            parsed.schedulerExperimentId = ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--scheduler-checkpoint-eval-id")
        {
            if (parsed.schedulerCheckpointEvalId.has_value())
                throw std::invalid_argument("--scheduler-checkpoint-eval-id specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--scheduler-checkpoint-eval-id requires a checkpoint_eval_id value");
            parsed.schedulerCheckpointEvalId = ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--scheduler-worker-attempt-id")
        {
            if (parsed.schedulerWorkerAttemptId.has_value())
                throw std::invalid_argument(
                    "--scheduler-worker-attempt-id specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument(
                    "--scheduler-worker-attempt-id requires an attempt_id value");
            parsed.schedulerWorkerAttemptId =
                ParseModelIdArg(argv[++i]);
        }
        else if (arg == "--training-objective")
        {
            if (parsed.trainingObjective.has_value())
                throw std::invalid_argument(
                    "--training-objective specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument(
                    "--training-objective requires a value");
            parsed.trainingObjective =
                EA::TrainingObjective::ParseCliSelection(argv[++i]);
        }
        else if (arg == "--log-level")
        {
            if (parsed.logLevel.has_value())
                throw std::invalid_argument("--log-level specified more than once");
            if (i + 1 >= argc)
                throw std::invalid_argument("--log-level requires quiet, summary, or diagnostic");
            parsed.logLevel = ParseRuntimeLogLevel(argv[++i]);
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
        else if (arg == "--eval-trading")
        {
            parsed.evalTrading = true;
        }
        else if (arg == "--infer-all")
        {
            parsed.inferAll = true;
        }
        else if (arg == "--force-infer")
        {
            parsed.forceInfer = true;
        }
        else if (arg == "--lstm-profile-hotspots")
        {
            parsed.lstmProfileHotspots = true;
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
            else if (SplitOptionWithValue(arg, "--core-lr-mult", value))
            {
                parsed.coreLrMult = ParsePositiveDoubleArg("--core-lr-mult", value);
            }
            else if (SplitOptionWithValue(arg, "--head-weight-lr-mult", value))
            {
                parsed.headWeightLrMult = ParsePositiveDoubleArg("--head-weight-lr-mult", value);
            }
            else if (SplitOptionWithValue(arg, "--head-bias-lr-mult", value))
            {
                parsed.headBiasLrMult = ParsePositiveDoubleArg("--head-bias-lr-mult", value);
            }
            else if (SplitOptionWithValue(arg, "--checkpoint-every", value))
            {
                if (parsed.checkpointEvery.has_value())
                    throw std::invalid_argument("--checkpoint-every specified more than once");
                parsed.checkpointEvery = ParseNonNegativeIntArg("--checkpoint-every", value);
            }
            else if (SplitOptionWithValue(arg, "--donchian20-mode", value))
            {
                if (parsed.donchian20Mode.has_value())
                    throw std::invalid_argument("--donchian20-mode specified more than once");
                parsed.donchian20Mode = ParseDonchian20Mode(value);
            }
            else if (SplitOptionWithValue(arg, "--feature-warmup-scope", value))
            {
                if (parsed.featureWarmupScope.has_value())
                    throw std::invalid_argument("--feature-warmup-scope specified more than once");
                parsed.featureWarmupScope = EA::ParseFeatureWarmupScope(value);
            }
            else if (SplitOptionWithValue(arg, "--donchian-lookback", value))
            {
                if (parsed.donchianLookback.has_value())
                    throw std::invalid_argument("--donchian-lookback specified more than once");
                parsed.donchianLookback = ParseDonchianLookback(value);
            }
            else if (SplitOptionWithValue(arg, "--infer-start-after-model-id", value))
            {
                if (parsed.inferStartAfterModelId.has_value())
                    throw std::invalid_argument("--infer-start-after-model-id specified more than once");
                parsed.inferStartAfterModelId = ParseModelIdArg(value);
            }
            else if (SplitOptionWithValue(arg, "--scheduler-experiment-id", value))
            {
                if (parsed.schedulerExperimentId.has_value())
                    throw std::invalid_argument("--scheduler-experiment-id specified more than once");
                parsed.schedulerExperimentId = ParseModelIdArg(value);
            }
            else if (SplitOptionWithValue(arg, "--scheduler-checkpoint-eval-id", value))
            {
                if (parsed.schedulerCheckpointEvalId.has_value())
                    throw std::invalid_argument("--scheduler-checkpoint-eval-id specified more than once");
                parsed.schedulerCheckpointEvalId = ParseModelIdArg(value);
            }
            else if (SplitOptionWithValue(
                         arg, "--scheduler-worker-attempt-id", value))
            {
                if (parsed.schedulerWorkerAttemptId.has_value())
                    throw std::invalid_argument(
                        "--scheduler-worker-attempt-id specified more than once");
                parsed.schedulerWorkerAttemptId =
                    ParseModelIdArg(value);
            }
            else if (SplitOptionWithValue(
                         arg, "--training-objective", value))
            {
                if (parsed.trainingObjective.has_value())
                    throw std::invalid_argument(
                        "--training-objective specified more than once");
                parsed.trainingObjective =
                    EA::TrainingObjective::ParseCliSelection(value);
            }
            else if (SplitOptionWithValue(arg, "--log-level", value))
            {
                if (parsed.logLevel.has_value())
                    throw std::invalid_argument("--log-level specified more than once");
                parsed.logLevel = ParseRuntimeLogLevel(value);
            }
            else if (SplitOptionWithValue(arg, "--lstm-profile-output", value))
            {
                if (parsed.lstmProfileOutputPath.has_value())
                    throw std::invalid_argument("--lstm-profile-output specified more than once");
                if (value.empty())
                    throw std::invalid_argument("--lstm-profile-output requires a non-empty path");
                parsed.lstmProfileOutputPath = value;
            }
            else if (SplitOptionWithValue(arg, "--symbol", value))
            {
                if (parsed.symbol.has_value())
                    throw std::invalid_argument("--symbol specified more than once");
                if (value.empty())
                    throw std::invalid_argument("--symbol requires a non-empty table_name");
                parsed.symbol = EA::CanonicalSymbol::Normalize(value);
            }
            else if (SplitOptionWithValue(arg, "--target-epochs", value))
            {
                if (parsed.targetEpochs.has_value())
                    throw std::invalid_argument("--target-epochs specified more than once");
                parsed.targetEpochs = ParsePositiveIntArg("--target-epochs", value);
            }
            else if (SplitOptionWithValue(arg, "--new-model-name", value))
            {
                if (parsed.newModelName.has_value())
                    throw std::invalid_argument("--new-model-name specified more than once");
                if (value.empty())
                    throw std::invalid_argument("--new-model-name requires a non-empty value");
                parsed.newModelName = value;
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

    if (parsed.frozenOutcome)
    {
        const bool hasOverride = parsed.inferenceMode.has_value() ||
            parsed.symbol.has_value() || parsed.modelId.has_value() ||
            parsed.resumeModelId.has_value() || parsed.targetEpochs.has_value() ||
            parsed.newModelName.has_value() || parsed.predictionHorizon.has_value() ||
            parsed.thresholdLogret.has_value() || parsed.windowSize.has_value() ||
            parsed.hiddenSize.has_value() || parsed.numLayers.has_value() ||
            parsed.epochs.has_value() || parsed.coreLrMult.has_value() ||
            parsed.headWeightLrMult.has_value() ||
            parsed.headBiasLrMult.has_value() || parsed.checkpointEvery.has_value() ||
            parsed.schedulerExperimentId.has_value() ||
            parsed.schedulerCheckpointEvalId.has_value() ||
            parsed.schedulerWorkerAttemptId.has_value() ||
            parsed.trainingObjective.has_value() || parsed.inferAll ||
            parsed.forceInfer || parsed.evalTrading ||
            parsed.resumeExpandInputWidth || parsed.featureWarmupScope.has_value() ||
            parsed.donchian20Mode.has_value() ||
            parsed.donchianLookback.has_value() || !positional.empty();
        if (hasOverride)
            throw std::invalid_argument(
                "--run-frozen-model-outcome-inference rejects training, model, "
                "feature, scheduler, and inference semantic overrides");
        parsed.inferenceMode = true;
        parsed.modelId = parsed.frozenOutcome->sourceModelId;
        parsed.fromDate = parsed.frozenOutcome->outcomeStart;
        parsed.toDate = parsed.frozenOutcome->outcomeEnd;
        return parsed;
    }

    if (parsed.trainingObjective.has_value())
    {
        if (!parsed.schedulerExperimentId.has_value())
            throw std::invalid_argument(
                "--training-objective requires --scheduler-experiment-id");
        if (!parsed.inferenceMode.has_value() || *parsed.inferenceMode)
            throw std::invalid_argument(
                "--training-objective is valid only for scheduler-managed training");
    }

    if (parsed.resumeModelId.has_value())
    {
        if (parsed.schedulerCheckpointEvalId.has_value())
            throw std::invalid_argument("--scheduler-checkpoint-eval-id cannot be combined with --resume-model-id");
        if (parsed.inferAll)
            throw std::invalid_argument("--infer-all cannot be combined with --resume-model-id");
        if (parsed.inferStartAfterModelId.has_value())
            throw std::invalid_argument("--infer-start-after-model-id cannot be combined with --resume-model-id");
        if (positional.size() == 2)
        {
            parsed.fromDate = positional[0];
            parsed.toDate = positional[1];
            parsed.positionalDateRangeSupplied = true;
        }
        else if (!positional.empty())
        {
            throw std::invalid_argument("resume mode accepts no positional date arguments");
        }
        return parsed;
    }

    if (parsed.resumeExpandInputWidth)
        throw std::invalid_argument(
            "--resume-expand-input-width requires --resume-model-id");

    if (parsed.inferStartAfterModelId.has_value() && !parsed.inferAll)
        throw std::invalid_argument("--infer-start-after-model-id requires --infer-all");
    if (parsed.forceInfer && !parsed.inferAll)
        throw std::invalid_argument("--force-infer requires --infer-all");
    if (parsed.schedulerCheckpointEvalId.has_value())
    {
        if (!parsed.inferenceMode.has_value() || !*parsed.inferenceMode)
            throw std::invalid_argument("--scheduler-checkpoint-eval-id requires explicit --infer");
        if (!parsed.modelId.has_value())
            throw std::invalid_argument("--scheduler-checkpoint-eval-id requires --model=<checkpoint_model_id>");
        if (parsed.inferAll)
            throw std::invalid_argument("--scheduler-checkpoint-eval-id cannot be combined with --infer-all");
        if (parsed.schedulerExperimentId.has_value())
            throw std::invalid_argument("--scheduler-checkpoint-eval-id cannot be combined with --scheduler-experiment-id");
    }
    if (parsed.schedulerExperimentId.has_value() &&
        parsed.inferenceMode.has_value() && *parsed.inferenceMode &&
        (!parsed.modelId.has_value() || parsed.inferAll))
    {
        throw std::invalid_argument("scheduler-managed final inference requires one explicit --model");
    }
    if (parsed.inferAll)
    {
        if (!parsed.inferenceMode.has_value() || !*parsed.inferenceMode)
            throw std::invalid_argument("--infer-all requires explicit --infer");
        if (!parsed.symbol.has_value() && !parsed.modelId.has_value())
            throw std::invalid_argument("--infer-all requires --model=<anchor_model_id> or --symbol=<table_name>");
        if (parsed.modelId.has_value() && parsed.inferStartAfterModelId.has_value())
            throw std::invalid_argument("--infer-all cannot combine --model anchor with --infer-start-after-model-id");
    }

    if (positional.size() != 2)
        throw std::invalid_argument("expected arguments: [--train|--infer] [--infer-all] [--force-infer] [--infer-start-after-model-id <model_id>] [--eval-trading] [--log-level quiet|summary|diagnostic] [--lstm-profile-hotspots] [--lstm-profile-output=<path>] [--resume-model-id=<model_id>] [--resume-expand-input-width] [--target-epochs=<absolute_final_epoch>] [--new-model-name=<name>] [--checkpoint-every <N>] [--symbol=<table_name>] [--model=<model_id>] [--prediction-horizon=<int>] [--threshold=<double>] [--window-size=<int>] [--hidden-size=<int>] [--num-layers=<int>] [--epochs=<int>] [--core-lr-mult=<float>] [--head-weight-lr-mult=<float>] [--head-bias-lr-mult=<float>] <fromDate> <toDate>; preferred inference: --infer --model=<model_id> <fromDate> <toDate>; preferred infer-all: --infer --infer-all --model=<anchor_model_id> <fromDate> <toDate>");

    parsed.fromDate = positional[0];
    parsed.toDate = positional[1];
    return parsed;
}

void ApplyLaunchRuntimeConfig(const LaunchArgs& launchArgs)
{
    gRuntimeInferenceMode = launchArgs.inferenceMode.value_or(default_runtime_inference_mode);

    if (launchArgs.predictionHorizon.has_value())
        prediction_horizon = *launchArgs.predictionHorizon;
    if (launchArgs.thresholdLogret.has_value())
        c_next_threshold = static_cast<float>(*launchArgs.thresholdLogret);
    if (launchArgs.logLevel.has_value())
        gRuntimeLogLevel = *launchArgs.logLevel;
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
    if (launchArgs.coreLrMult.has_value())
        core_lr_mult = static_cast<float>(*launchArgs.coreLrMult);
    if (launchArgs.headWeightLrMult.has_value())
        head_weight_lr_mult = static_cast<float>(*launchArgs.headWeightLrMult);
    if (launchArgs.headBiasLrMult.has_value())
        head_bias_lr_mult = static_cast<float>(*launchArgs.headBiasLrMult);
}

bool PrintResumeOverrideRejected(const char* param)
{
    std::cerr << "RESUME_CONFIG_OVERRIDE_REJECTED"
              << ",param=" << param
              << ",reason=resume_uses_database_config_only"
              << std::endl;
    return false;
}

bool ValidateResumeLaunchArgs(const LaunchArgs& launchArgs)
{
    if (!launchArgs.resumeModelId.has_value())
        return true;
    if (launchArgs.inferenceMode.has_value() && *launchArgs.inferenceMode)
        return PrintResumeOverrideRejected("--infer");
    if (launchArgs.modelId.has_value())
        return PrintResumeOverrideRejected("--model");
    if (launchArgs.symbol.has_value())
        return PrintResumeOverrideRejected("--symbol");
    if (launchArgs.positionalDateRangeSupplied)
        return PrintResumeOverrideRejected("date_range");
    if (launchArgs.evalTrading)
        return PrintResumeOverrideRejected("--eval-trading");
    if (launchArgs.predictionHorizon.has_value())
        return PrintResumeOverrideRejected("--prediction-horizon");
    if (launchArgs.thresholdLogret.has_value())
        return PrintResumeOverrideRejected("--threshold");
    if (launchArgs.windowSize.has_value())
        return PrintResumeOverrideRejected("--window-size");
    if (launchArgs.hiddenSize.has_value())
        return PrintResumeOverrideRejected("--hidden-size");
    if (launchArgs.numLayers.has_value())
        return PrintResumeOverrideRejected("--num-layers");
    if (launchArgs.epochs.has_value())
        return PrintResumeOverrideRejected("--epochs");
    if (launchArgs.coreLrMult.has_value())
        return PrintResumeOverrideRejected("--core-lr-mult");
    if (launchArgs.headWeightLrMult.has_value())
        return PrintResumeOverrideRejected("--head-weight-lr-mult");
    if (launchArgs.headBiasLrMult.has_value())
        return PrintResumeOverrideRejected("--head-bias-lr-mult");
    if (!launchArgs.targetEpochs.has_value())
        return PrintResumeOverrideRejected("--target-epochs");
    return true;
}

struct ResumeCheckpointConfig
{
    long long sourceModelId = -1;
    std::string symbol;
    std::string fromDate;
    std::string toDate;
    TrainConfigMeta trainConfig;
    int modelInputWidth = 0;
    size_t modelHiddenSize = 0;
    EA::LSTM::TargetType targetType = EA::LSTM::TargetType::UpNeutralDownReturn;
    size_t completedEpoch = 0;
    size_t optimizerUpdateCount = 0;
    EA::FeatureWarmupScope featureWarmupScope =
        EA::FeatureWarmupScope::LegacyColdBoundary;
    Donchian20Mode donchian20Mode = kDefaultDonchian20Mode;
    std::size_t donchianLookback = kDefaultDonchianLookback;
    EA::FeatureAblationMask featureAblationMask;
    bool expandInputWidthRequested = false;
    bool parameterExpansionRequired = false;
    std::optional<EA::InputWidthExpansionProvenance>
        inputWidthExpansionProvenance;
    EA::TrainingObjective::Configuration trainingObjective =
        EA::TrainingObjective::Legacy();
};

EA::TrainingObjective::Configuration LoadExperimentTrainingObjective(
    pqxx::work& w,
    long long experimentId)
{
    const pqxx::result rows = w.exec(
        "SELECT training_objective_canonical,training_objective_hash "
        "FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    if (rows.empty())
        throw std::runtime_error(
            "training_objective_experiment_not_found");
    return EA::TrainingObjective::ResolvePersisted(
        rows[0][0].as<std::string>(), rows[0][1].as<std::string>());
}

void ConfigureInputWidthExpansionForResume(pqxx::work& w,
                                           ResumeCheckpointConfig& cfg,
                                           bool requested,
                                           std::optional<long long>
                                               schedulerExperimentId)
{
    cfg.expandInputWidthRequested = requested;
    if (!requested) return;

    DBIO::PgModelIO::validateModelInputSemanticsForExpansion(
        w, cfg.sourceModelId);
    const std::size_t sourceWidth =
        static_cast<std::size_t>(cfg.modelInputWidth);
    if (sourceWidth < EA::kCurrentModelInputWidth)
    {
        const EA::InputWidthExpansionPlan plan =
            EA::BuildInputWidthExpansionPlan(sourceWidth);
        cfg.parameterExpansionRequired = true;
        cfg.inputWidthExpansionProvenance =
            EA::MakeInputWidthExpansionProvenance(cfg.sourceModelId, plan);
        return;
    }
    if (sourceWidth > EA::kCurrentModelInputWidth)
    {
        (void)EA::BuildInputWidthExpansionPlan(sourceWidth);
        return;
    }

    // Scheduler retry may resume from a checkpoint already widened by this
    // same experiment.  Accept the no-op only when durable expansion
    // provenance proves that fact; arbitrary current-width sources still fail.
    if (!schedulerExperimentId.has_value())
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_NOT_REQUIRED,source_n_in=" +
            std::to_string(sourceWidth) + ",target_n_in=" +
            std::to_string(EA::kCurrentModelInputWidth));
    }
    const pqxx::result owningExperiment = w.exec(
        "SELECT 1 FROM model WHERE model_id=$1 AND experiment_id=$2;",
        pqxx::params{cfg.sourceModelId, *schedulerExperimentId});
    if (owningExperiment.empty())
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_RETRY_EXPERIMENT_LINEAGE_MISMATCH");
    }
    cfg.inputWidthExpansionProvenance =
        DBIO::PgModelIO::loadRequiredInputWidthExpansionMeta(
            w, cfg.sourceModelId);
    if (cfg.inputWidthExpansionProvenance->expandedInputWidth !=
        EA::kCurrentModelInputWidth)
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_RETRY_PROVENANCE_TARGET_MISMATCH");
    }
    const pqxx::result lineage = w.exec(
        "WITH RECURSIVE ancestry(model_id,parent_model_id) AS ("
        " SELECT model_id,parent_model_id FROM model WHERE model_id=$1"
        " UNION"
        " SELECT m.model_id,m.parent_model_id FROM model m"
        " JOIN ancestry a ON m.model_id=a.parent_model_id"
        ") SELECT 1 FROM ancestry WHERE model_id=$2 LIMIT 1;",
        pqxx::params{
            cfg.sourceModelId,
            cfg.inputWidthExpansionProvenance->sourceModelId});
    if (lineage.empty())
    {
        throw std::runtime_error(
            "MODEL_INPUT_EXPANSION_RETRY_SOURCE_LINEAGE_MISMATCH");
    }
}

void ValidateExpandedResumeAblationComposition(
    const EA::FeatureAblationMask& sourceMask,
    const EA::FeatureAblationMask& requestedMask,
    std::size_t sourceInputWidth)
{
    const auto sourceContract = EA::ContractForModelInputWidth(sourceInputWidth);
    const auto contains = [](const std::vector<std::size_t>& values,
                             std::size_t value)
    {
        return std::find(values.begin(), values.end(), value) != values.end();
    };
    for (const std::size_t sourceColumn : sourceMask.tensorColumns())
    {
        if (!contains(requestedMask.tensorColumns(), sourceColumn))
            throw std::runtime_error(
                "FEATURE_ABLATION_MASK_EXPANSION_REMOVES_SOURCE_ABLATION");
    }
    for (const std::size_t requestedColumn : requestedMask.tensorColumns())
    {
        if (!contains(sourceMask.tensorColumns(), requestedColumn) &&
            requestedColumn < sourceContract.tensorFeatureCount)
        {
            throw std::runtime_error(
                "FEATURE_ABLATION_MASK_EXPANSION_CHANGES_HISTORICAL_FEATURE");
        }
    }
    requestedMask.ValidateForTensorFeatureCount(feature_size);
}

EA::FeatureAblationMask LoadModelFeatureAblationMask(pqxx::work& w,
                                                     long long modelId)
{
    const pqxx::result rows = w.exec_params(
        "SELECT m.experiment_id, e.feature_ablation_mask FROM model m "
        "LEFT JOIN experiment e ON e.experiment_id=m.experiment_id "
        "WHERE m.model_id=$1;", modelId);
    if (rows.empty())
        throw std::runtime_error("model_not_found_for_feature_ablation_mask");
    // Models predating experiment_id provenance have no ablation contract and
    // therefore retain the durable compatibility interpretation: no mask.
    if (rows[0][0].is_null()) return {};
    if (rows[0][1].is_null())
        throw std::runtime_error("model_experiment_lineage_missing_feature_ablation_mask");
    return EA::FeatureAblationMask::Parse(rows[0][1].as<std::string>());
}

EA::FeatureAblationMask LoadSchedulerFeatureAblationMask(
    pqxx::work& w, const LaunchArgs& launchArgs)
{
    std::optional<long long> experimentId = launchArgs.schedulerExperimentId;
    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        const pqxx::result rows = w.exec_params(
            "SELECT COALESCE(parent_experiment_id, experiment_id) FROM "
            "experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            *launchArgs.schedulerCheckpointEvalId);
        if (rows.empty()) throw std::runtime_error("checkpoint_eval_not_found_for_feature_ablation_mask");
        experimentId = rows[0][0].as<long long>();
    }
    if (!experimentId.has_value()) return {};
    const pqxx::result rows = w.exec_params(
        "SELECT feature_ablation_mask FROM experiment WHERE experiment_id=$1;",
        *experimentId);
    if (rows.empty()) throw std::runtime_error("experiment_not_found_for_feature_ablation_mask");
    return EA::FeatureAblationMask::Parse(rows[0][0].as<std::string>());
}

void ValidateSchedulerModelFeatureAblationMask(
    const EA::FeatureAblationMask& modelMask,
    const EA::FeatureAblationMask& schedulerMask,
    long long modelId)
{
    if (modelMask.CanonicalText() == schedulerMask.CanonicalText())
        return;
    throw std::runtime_error(
        "FEATURE_ABLATION_MASK_LINEAGE_MISMATCH:model_id=" +
        std::to_string(modelId) + ",model=" + modelMask.CanonicalText() +
        ",scheduler=" + schedulerMask.CanonicalText());
}

void ValidateSchedulerResumeFeatureAblationMask(
    const ResumeCheckpointConfig& resume,
    const EA::FeatureAblationMask& schedulerMask)
{
    if (!resume.expandInputWidthRequested)
    {
        ValidateSchedulerModelFeatureAblationMask(
            resume.featureAblationMask, schedulerMask, resume.sourceModelId);
        return;
    }
    ValidateExpandedResumeAblationComposition(
        resume.featureAblationMask,
        schedulerMask,
        static_cast<std::size_t>(resume.modelInputWidth));
}

TrainConfigMeta LoadRequiredTrainConfigMetaForResume(pqxx::work& w, long long modelId)
{
    auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "train_config_meta");
    auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "train_config_meta");
    if (dims.n_rows != 1 ||
        dims.n_cols < DBIO::PgModelIO::kTrainConfigMetaExtendedFieldCount ||
        vals.size() < static_cast<size_t>(DBIO::PgModelIO::kTrainConfigMetaExtendedFieldCount))
        throw std::runtime_error("resume requires complete train_config_meta with 14 fields");

    TrainConfigMeta meta;
    meta.schemaVersion = static_cast<int>(std::llround(vals[0]));
    meta.predictionHorizon = static_cast<size_t>(std::llround(vals[1]));
    meta.thresholdLogret = static_cast<float>(vals[2]);
    meta.windowSize = static_cast<size_t>(std::llround(vals[3]));
    meta.labelRuleId = static_cast<int>(std::llround(vals[4]));
    meta.classWeightDown = static_cast<float>(vals[5]);
    meta.classWeightNeutral = static_cast<float>(vals[6]);
    meta.classWeightUp = static_cast<float>(vals[7]);
    meta.numLayers = static_cast<size_t>(std::llround(vals[8]));
    meta.normalizationVersion = static_cast<int>(std::llround(vals[9]));
    meta.epochsTrained = static_cast<size_t>(std::llround(vals[10]));
    meta.coreLrMult = static_cast<float>(vals[11]);
    meta.headWeightLrMult = static_cast<float>(vals[12]);
    meta.headBiasLrMult = static_cast<float>(vals[13]);
    return meta;
}

ResumeCheckpointConfig LoadResumeCheckpointConfig(
    pqxx::work& w,
    long long modelId,
    const EA::TrainingObjective::Configuration& requestedObjective)
{
    DBIO::PgModelIO::validateTrainingResumeState(w, modelId);

    ResumeCheckpointConfig cfg;
    cfg.sourceModelId = modelId;
    cfg.trainingObjective =
        DBIO::PgModelIO::loadTrainingObjectiveMeta(w, modelId);
    EA::TrainingObjective::RequireResumeCompatible(
        cfg.trainingObjective, requestedObjective);
    cfg.featureWarmupScope = DBIO::PgModelIO::loadFeatureWarmupScopeMeta(w, modelId);
    cfg.donchian20Mode = DBIO::PgModelIO::loadDonchian20ModeMeta(w, modelId);
    cfg.donchianLookback = DBIO::PgModelIO::loadDonchianLookbackMeta(w, modelId);
    cfg.featureAblationMask = LoadModelFeatureAblationMask(w, modelId);
    cfg.trainConfig = LoadRequiredTrainConfigMetaForResume(w, modelId);
    cfg.completedEpoch = cfg.trainConfig.epochsTrained.value_or(0);
    cfg.symbol = DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
    PrintDatabaseModelSymbol(modelId, cfg.symbol);
    const auto range = DBIO::PgModelIO::decodeTrainRangeMeta(w, modelId);
    cfg.fromDate = range.first;
    cfg.toDate = range.second;

    {
        const auto modelMeta =
            DBIO::PgModelIO::loadRequiredModelMeta(w, modelId);
        cfg.modelInputWidth = static_cast<int>(modelMeta.inputWidth);
        cfg.modelHiddenSize = modelMeta.hiddenSize;
    }

    {
        cfg.targetType =
            DBIO::PgModelIO::loadRequiredTargetMeta(w, modelId).targetType;
    }

    auto optimizerDims = DBIO::PgModelIO::loadParameterDims(w, modelId, "optimizer_meta");
    auto optimizerVals = DBIO::PgModelIO::loadParameterValues(w, modelId, "optimizer_meta");
    if (optimizerDims.n_rows != 1 ||
        optimizerDims.n_cols < DBIO::PgModelIO::kOptimizerMetaFieldCount ||
        optimizerVals.size() < static_cast<size_t>(DBIO::PgModelIO::kOptimizerMetaFieldCount))
        throw std::runtime_error("resume requires valid optimizer_meta");
    const int optimizerSchema = static_cast<int>(std::llround(optimizerVals[0]));
    const int optimizerType = static_cast<int>(std::llround(optimizerVals[1]));
    if (optimizerSchema != DBIO::PgModelIO::kOptimizerMetaSchemaVersion ||
        optimizerType != DBIO::PgModelIO::kOptimizerTypeSgd)
        throw std::runtime_error("resume optimizer_meta is not supported by this binary");
    cfg.optimizerUpdateCount = static_cast<size_t>(std::llround(optimizerVals[2]));
    return cfg;
}

void ApplyResumeRuntimeConfig(const ResumeCheckpointConfig& cfg, int targetEpochs)
{
    (void)EA::TrainingObjective::ParseSupportedCanonicalText(
        EA::TrainingObjective::CanonicalText(cfg.trainingObjective));
    if (cfg.trainConfig.schemaVersion != DBIO::PgModelIO::kTrainConfigMetaSchemaVersion)
        throw std::runtime_error("resume train_config_meta schema_version unsupported");
    if (cfg.trainConfig.labelRuleId != DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId)
        throw std::runtime_error("resume label_rule_id unsupported by this binary");
    if (cfg.trainConfig.numLayers != 1)
        throw std::runtime_error("resume num_layers unsupported by this binary");
    if (std::fabs(cfg.trainConfig.classWeightDown - kClassWeightDown) > 1e-7f ||
        std::fabs(cfg.trainConfig.classWeightNeutral - kClassWeightNeutral) > 1e-7f ||
        std::fabs(cfg.trainConfig.classWeightUp - kClassWeightUp) > 1e-7f)
        throw std::runtime_error("resume class weights differ from this binary");
    if (static_cast<size_t>(targetEpochs) <= cfg.completedEpoch)
        throw std::runtime_error("target epochs is an absolute final epoch and must be greater than checkpoint completed epoch");

    gRuntimeInferenceMode = false;
    prediction_horizon = cfg.trainConfig.predictionHorizon;
    c_next_threshold = cfg.trainConfig.thresholdLogret;
    window_size = cfg.trainConfig.windowSize;
    hidden_size = cfg.modelHiddenSize;
    n_out = hidden_size;
    num_layers = cfg.trainConfig.numLayers;
    normalization_version = cfg.trainConfig.normalizationVersion;
    core_lr_mult = cfg.trainConfig.coreLrMult.value();
    head_weight_lr_mult = cfg.trainConfig.headWeightLrMult.value();
    head_bias_lr_mult = cfg.trainConfig.headBiasLrMult.value();
    epoch_count = targetEpochs;
}

void PrintRuntimeConfig()
{
    if (!LogSummary())
        return;
    std::cout << "RUNTIME_CONFIG"
              << ",train=" << (gRuntimeInferenceMode ? "false" : "true")
              << ",infer=" << (gRuntimeInferenceMode ? "true" : "false")
              << ",log_level=" << RuntimeLogLevelName(gRuntimeLogLevel)
              << ",prediction_horizon=" << prediction_horizon
              << ",threshold=" << c_next_threshold
              << ",window_size=" << window_size
              << ",hidden_size=" << hidden_size
              << ",num_layers=" << num_layers
              << ",epochs=" << epoch_count
              << ",core_lr_mult=" << core_lr_mult
              << ",head_weight_lr_mult=" << head_weight_lr_mult
              << ",head_bias_lr_mult=" << head_bias_lr_mult
              << std::endl;
}

Donchian20Mode LoadExperimentDonchian20Mode(pqxx::work& w,
                                            long long experimentId)
{
    const pqxx::result rows = w.exec_params(
        "SELECT donchian20_mode FROM experiment WHERE experiment_id=$1;",
        experimentId);
    if (rows.empty())
        throw std::runtime_error("experiment_not_found_for_donchian20_mode");
    return ParseDonchian20Mode(rows[0][0].as<std::string>());
}

EA::FeatureWarmupScope LoadExperimentFeatureWarmupScope(
    pqxx::work& w,
    long long experimentId)
{
    const pqxx::result rows = w.exec_params(
        "SELECT feature_warmup_scope FROM experiment WHERE experiment_id=$1;",
        experimentId);
    if (rows.empty())
        throw std::runtime_error("experiment_not_found_for_feature_warmup_scope");
    return EA::ParseFeatureWarmupScope(rows[0][0].as<std::string>());
}

std::size_t LoadExperimentDonchianLookback(pqxx::work& w,
                                           long long experimentId)
{
    const pqxx::result rows = w.exec_params(
        "SELECT donchian_lookback FROM experiment WHERE experiment_id=$1;",
        experimentId);
    if (rows.empty())
        throw std::runtime_error("experiment_not_found_for_donchian_lookback");
    return ParseDonchianLookback(rows[0][0].as<std::string>());
}

void ValidateSchedulerDonchian20Mode(pqxx::work& w,
                                     const LaunchArgs& launchArgs,
                                     Donchian20Mode runtimeMode)
{
    std::optional<long long> experimentId = launchArgs.schedulerExperimentId;
    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        const pqxx::result rows = w.exec_params(
            "SELECT COALESCE(parent_experiment_id, experiment_id) "
            "FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            *launchArgs.schedulerCheckpointEvalId);
        if (rows.empty())
            throw std::runtime_error("checkpoint_eval_not_found_for_donchian20_mode");
        experimentId = rows[0][0].as<long long>();
    }
    if (!experimentId.has_value())
        return;

    const Donchian20Mode persistedMode =
        LoadExperimentDonchian20Mode(w, *experimentId);
    if (persistedMode != runtimeMode)
        throw std::runtime_error(
            std::string{"scheduler experiment Donchian-20 mode mismatch: persisted="} +
            Donchian20ModeText(persistedMode) + ", runtime=" +
            Donchian20ModeText(runtimeMode));
}

void ValidateSchedulerFeatureWarmupScope(
    pqxx::work& w,
    const LaunchArgs& launchArgs,
    EA::FeatureWarmupScope runtimeScope)
{
    std::optional<long long> experimentId = launchArgs.schedulerExperimentId;
    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        const pqxx::result rows = w.exec_params(
            "SELECT COALESCE(parent_experiment_id, experiment_id) "
            "FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            *launchArgs.schedulerCheckpointEvalId);
        if (rows.empty())
            throw std::runtime_error("checkpoint_eval_not_found_for_feature_warmup_scope");
        experimentId = rows[0][0].as<long long>();
    }
    if (experimentId.has_value() &&
        LoadExperimentFeatureWarmupScope(w, *experimentId) != runtimeScope)
        throw std::runtime_error("scheduler experiment feature warmup scope mismatch");
}

void ValidateSchedulerDonchianLookback(pqxx::work& w,
                                       const LaunchArgs& launchArgs,
                                       std::size_t runtimeLookback)
{
    std::optional<long long> experimentId = launchArgs.schedulerExperimentId;
    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        const pqxx::result rows = w.exec_params(
            "SELECT COALESCE(parent_experiment_id, experiment_id) "
            "FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            *launchArgs.schedulerCheckpointEvalId);
        if (rows.empty())
            throw std::runtime_error("checkpoint_eval_not_found_for_donchian_lookback");
        experimentId = rows[0][0].as<long long>();
    }
    if (experimentId.has_value() &&
        LoadExperimentDonchianLookback(w, *experimentId) != runtimeLookback)
        throw std::runtime_error("scheduler experiment Donchian lookback mismatch");
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

void PrintRuntimeLrConfig(const EA::LSTM& lstm)
{
    if (!LogDiagnostic())
        return;
    const float coreLrMult = EA::LSTM::CoreLrMultForTarget(lstm.targetType);
    const bool directionHeadPath =
        (lstm.targetType == EA::LSTM::TargetType::UpNeutralDownReturn);
    const float effectiveHeadWeightLr = lstm.learning_rate * head_weight_lr_mult;
    const float effectiveHeadBiasLr = lstm.learning_rate * head_bias_lr_mult;
    std::cout << "RUNTIME_LR_CONFIG"
              << ",target_type=" << TargetTypeName(lstm.targetType)
              << ",raw_learning_rate=" << lstm.learning_rate
              << ",core_lr_mult=" << coreLrMult
              << ",effective_core_lr=" << (lstm.learning_rate * coreLrMult)
              << ",head_weight_lr_mult=" << head_weight_lr_mult
              << ",effective_head_weight_lr=";
    if (directionHeadPath)
        std::cout << effectiveHeadWeightLr;
    else
        std::cout << "not_applicable";
    std::cout << ",head_bias_lr_mult=" << head_bias_lr_mult
              << ",effective_head_bias_lr=";
    if (directionHeadPath)
        std::cout << effectiveHeadBiasLr;
    else
        std::cout << "not_applicable";
    std::cout << std::endl;
}

std::string CheckpointBaseModelName(const LaunchArgs& launchArgs,
                                    const std::optional<ResumeCheckpointConfig>& resumeConfig,
                                    const std::string& rawPriceTableName)
{
    if (resumeConfig.has_value())
        return launchArgs.newModelName.value_or(rawPriceTableName + "-resume-model");
    return launchArgs.newModelName.value_or(rawPriceTableName + "-model");
}

std::string EpochCheckpointModelName(const std::string& baseModelName, size_t completedEpoch)
{
    std::ostringstream oss;
    oss << baseModelName
        << "_epoch"
        << std::setw(3)
        << std::setfill('0')
        << completedEpoch;
    return oss.str();
}

bool ModelNameExists(pqxx::work& w, const std::string& modelName)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM model WHERE name = $1 LIMIT 1;",
        modelName);
    return !r.empty();
}

std::string UniqueModelName(pqxx::work& w, const std::string& desiredName)
{
    if (!ModelNameExists(w, desiredName))
        return desiredName;

    for (int suffix = 1; suffix <= 9999; ++suffix)
    {
        std::ostringstream candidate;
        candidate << desiredName << "_dup" << std::setw(3) << std::setfill('0') << suffix;
        if (!ModelNameExists(w, candidate.str()))
            return candidate.str();
    }

    throw std::runtime_error("unable to allocate unique model name for checkpoint '" + desiredName + "'");
}

void LinkModelToSchedulerExperimentIfPresent(pqxx::work& w,
                                             long long modelId,
                                             const std::optional<long long>& schedulerExperimentId)
{
    if (!schedulerExperimentId.has_value())
        return;

    pqxx::result current = w.exec_params(
        "SELECT experiment_id FROM model WHERE model_id = $1 FOR UPDATE;",
        modelId);
    if (current.empty())
        throw std::runtime_error("MODEL_EXPERIMENT_LINK_FAILED model_not_found model_id=" + std::to_string(modelId));

    if (!current[0][0].is_null())
    {
        const long long existingExperimentId = current[0][0].as<long long>();
        if (existingExperimentId == *schedulerExperimentId)
            return;

        std::cerr << "MODEL_EXPERIMENT_LINK_CONFLICT"
                  << ",model_id=" << modelId
                  << ",existing_experiment_id=" << existingExperimentId
                  << ",requested_experiment_id=" << *schedulerExperimentId
                  << std::endl;
        throw std::runtime_error("MODEL_EXPERIMENT_LINK_CONFLICT model_id=" + std::to_string(modelId));
    }

    w.exec_params(
        "UPDATE model SET experiment_id = $1 WHERE model_id = $2 AND experiment_id IS NULL;",
        *schedulerExperimentId,
        modelId);
}

void LinkModelParentIfPresent(pqxx::work& w,
                              long long modelId,
                              const std::optional<long long>& parentModelId)
{
    if (!parentModelId.has_value())
        return;

    pqxx::result current = w.exec_params(
        "SELECT parent_model_id FROM model WHERE model_id = $1 FOR UPDATE;",
        modelId);
    if (current.empty())
        throw std::runtime_error("MODEL_PARENT_LINK_FAILED model_not_found model_id=" + std::to_string(modelId));

    if (!current[0][0].is_null())
    {
        const long long existingParentModelId = current[0][0].as<long long>();
        if (existingParentModelId == *parentModelId)
            return;

        std::cerr << "MODEL_PARENT_LINK_CONFLICT"
                  << ",model_id=" << modelId
                  << ",existing_parent_model_id=" << existingParentModelId
                  << ",requested_parent_model_id=" << *parentModelId
                  << std::endl;
        throw std::runtime_error("MODEL_PARENT_LINK_CONFLICT model_id=" + std::to_string(modelId));
    }

    w.exec_params(
        "UPDATE model SET parent_model_id = $1 WHERE model_id = $2 AND parent_model_id IS NULL;",
        *parentModelId,
        modelId);
}

std::optional<long long> ParentModelIdForResume(const std::optional<ResumeCheckpointConfig>& resumeConfig)
{
    if (!resumeConfig.has_value())
        return std::nullopt;
    return resumeConfig->sourceModelId;
}

std::optional<long long> SavePeriodicCheckpointIfDue(const LaunchArgs& launchArgs,
                                                     const std::optional<ResumeCheckpointConfig>& resumeConfig,
                                                     const std::string& rawPriceTableName,
                                                     const std::string& fromDate,
                                                     const std::string& toDate,
                                                     EA::LSTM& lstm,
                                                     Donchian20Mode donchian20Mode,
                                                     EA::FeatureWarmupScope featureWarmupScope,
                                                     std::size_t donchianLookback,
                                                     const EA::TrainingObjective::Configuration&
                                                         trainingObjective)
{
    if (!launchArgs.checkpointEvery.has_value() || *launchArgs.checkpointEvery <= 0)
        return std::nullopt;
    if (gRuntimeInferenceMode)
    {
        DiagnosticOut() << "CHECKPOINT_SAVE_SKIPPED reason=inference_mode" << std::endl;
        return std::nullopt;
    }
    if constexpr (!save_enable)
    {
        DiagnosticOut() << "CHECKPOINT_SAVE_SKIPPED reason=save_disabled" << std::endl;
        return std::nullopt;
    }

    const size_t completedEpoch = lstm.completedEpochs;
    const size_t checkpointEvery = static_cast<size_t>(*launchArgs.checkpointEvery);
    if (completedEpoch == 0 || completedEpoch % checkpointEvery != 0)
        return std::nullopt;

    const std::string baseName = CheckpointBaseModelName(launchArgs, resumeConfig, rawPriceTableName);
    const std::string requestedName = EpochCheckpointModelName(baseName, completedEpoch);
    pqxx::connection cCheckpoint { LstmDbConnectionString() };
    pqxx::work wCheckpoint { cCheckpoint };
    wCheckpoint.exec("SET TRANSACTION READ WRITE;");
    const std::string checkpointName = UniqueModelName(wCheckpoint, requestedName);

    std::cout << "CHECKPOINT_SAVE_BEGIN"
              << " epoch=" << completedEpoch
              << " name=" << checkpointName
              << std::endl;
    if (checkpointName != requestedName)
        std::cout << "CHECKPOINT_SAVE_RENAMED"
                  << " reason=name_exists"
                  << " requested_name=" << requestedName
                  << " using_name=" << checkpointName
                  << std::endl;

    const long long checkpointModelId =
        DBIO::PgModelIO::createModel(wCheckpoint,
                                     checkpointName,
                                     "periodic training checkpoint",
                                     launchArgs.schedulerExperimentId,
                                     ParentModelIdForResume(resumeConfig));
    DBIO::PgModelIO::saveAll(wCheckpoint, checkpointModelId, lstm, rawPriceTableName, fromDate, toDate,
                             donchian20Mode, featureWarmupScope,
                             donchianLookback,
                             resumeConfig.has_value()
                                 ? resumeConfig->inputWidthExpansionProvenance
                                 : std::nullopt,
                             trainingObjective);
    wCheckpoint.commit();
    std::cout << "CHECKPOINT_SAVE_DONE"
              << " epoch=" << completedEpoch
              << " model_id=" << checkpointModelId
              << " name=" << checkpointName
              << std::endl;
    return checkpointModelId;
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
    return "schema_version,prediction_horizon,threshold_logret,window_size,label_rule_id,class_weight_down,class_weight_neutral,class_weight_up,num_layers,normalization_version,epochs_trained,core_lr_mult,head_weight_lr_mult,head_bias_lr_mult";
}

size_t RuntimeTensorFeatureWidth(const Tensor& tensor)
{
    return (tensor.begin() != tensor.end())
        ? static_cast<size_t>((*tensor.begin()).Shape()[1])
        : 0;
}

size_t RuntimeModelInputWidth(
    const Tensor& tensor,
    std::optional<std::size_t> persistedModelInputWidth = std::nullopt)
{
    const size_t baseFeatureCount = RuntimeTensorFeatureWidth(tensor);
    if (persistedModelInputWidth.has_value())
        return EA::ResolveModelInputContract(*persistedModelInputWidth,
                                             baseFeatureCount).modelInputWidth;
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
    if (!LogDiagnostic())
        return;
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
    if (!LogSummary())
        return;
    const auto& evalConfig = ActiveEvalLabelConfig();
    DiagnosticOut() << "INFERENCE_CONFIG_SOURCE=" << evalConfig.source << std::endl;
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
                                                       const Tensor& tensor,
                                                       const std::string& runtimeSymbol)
{
    ModelConfigValidationResult result;
    std::vector<std::string> persistedParams;
    std::vector<std::string> persistedMetadata;
    bool hasTargetMeta = false;
    bool hasModelMeta = false;
    bool hasTrainSymbolMeta = false;
    bool hasTrainConfigMeta = false;
    bool hasTrainConfigNumLayers = false;
    bool hasTrainConfigNormalizationVersion = false;
    bool hasTrainConfigCoreLrMult = false;
    bool hasTrainConfigHeadWeightLrMult = false;
    bool hasTrainConfigHeadBiasLrMult = false;

    DiagnosticOut() << "MODEL_TRAIN_CONFIG_META_FIELDS,"
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
                persistedMetadata.push_back("train_config_meta(schema_version;prediction_horizon;threshold_logret;window_size;label_rule_id;class_weight_down;class_weight_neutral;class_weight_up;num_layers;normalization_version;epochs_trained;core_lr_mult;head_weight_lr_mult;head_bias_lr_mult)");
            }
            else if (paramName == "train_symbol_meta")
            {
                hasTrainSymbolMeta = true;
                persistedMetadata.push_back("train_symbol_meta(ascii_table_name)");
            }
            else if (paramName == "train_range_meta")
            {
                persistedMetadata.push_back("train_range_meta(fromDate;toDate)");
            }
            else if (paramName == "donchian20_mode_meta")
            {
                persistedMetadata.push_back("donchian20_mode_meta(ascii_mode)");
            }
            else if (paramName == "optimizer_meta")
            {
                persistedMetadata.push_back("optimizer_meta(schema_version;optimizer_type;update_count;first_moment_buffer_count;second_moment_buffer_count)");
            }
            else if (paramName == "training_objective_canonical_meta")
            {
                persistedMetadata.push_back(
                    "training_objective_canonical_meta(ascii_canonical_contract)");
            }
            else if (paramName == "training_objective_hash_meta")
            {
                persistedMetadata.push_back(
                    "training_objective_hash_meta(ascii_fnv1a64_identity)");
            }
        }
    }
    catch (const std::exception& e)
    {
        DiagnosticOut() << "MODEL_METADATA_READ_FAIL"
                        << ",model_id=" << modelId
                        << ",error=" << e.what()
                        << std::endl;
    }

    DiagnosticOut() << "MODEL_METADATA_PERSISTED"
                    << ",model_id=" << modelId
                    << ",param_names=" << JoinStrings(persistedParams, ";")
                    << ",metadata=" << JoinStrings(persistedMetadata, ";")
                    << std::endl;

    bool targetMetaMatches = false;
    bool modelMetaMatches = false;
    bool trainSymbolMetaMatches = true;
    bool trainConfigMetaMatches = false;
    bool mismatch = false;
    std::optional<std::string> decodedModelSymbol;

    auto printMismatch = [&](const char* field, const auto& modelValue, const auto& runtimeValue)
    {
        mismatch = true;
        if (LogSummary())
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

    auto compareStringField = [&](const char* field,
                                  const std::string& modelValue,
                                  const std::string& runtimeValue) -> bool
    {
        if (modelValue != runtimeValue)
        {
            printMismatch(field, modelValue, runtimeValue);
            return false;
        }
        return true;
    };

    try
    {
        const std::string modelSymbol = DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
        decodedModelSymbol = modelSymbol;
        PrintDatabaseModelSymbol(modelId, modelSymbol);
        if (LogSummary())
            std::cout << "MODEL_TRAIN_SYMBOL_META"
                      << ",model_id=" << modelId
                      << ",symbol=" << modelSymbol
                      << std::endl;
        if (result.trainConfigMeta.has_value())
            result.trainConfigMeta->symbol = modelSymbol;
        trainSymbolMetaMatches = compareStringField("symbol",
                                                    EA::CanonicalSymbol::Normalize(modelSymbol),
                                                    EA::CanonicalSymbol::Normalize(runtimeSymbol));
    }
    catch (const std::exception& e)
    {
        if (hasTrainSymbolMeta)
        {
            printMismatch("train_symbol_meta", e.what(), runtimeSymbol);
            trainSymbolMetaMatches = false;
        }
        else
        {
            PrintMissingModelSymbol(modelId);
            DiagnosticOut() << "MODEL_CONFIG_WARN"
                            << ",model_id=" << modelId
                            << ",field=symbol"
                            << ",model=missing_legacy_train_symbol_meta"
                            << ",runtime=" << runtimeSymbol
                            << std::endl;
        }
    }
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
        const auto modelMeta =
            DBIO::PgModelIO::loadRequiredModelMeta(w, modelId);
        const int modelInputWidth = static_cast<int>(modelMeta.inputWidth);
        const int modelHiddenSize = static_cast<int>(modelMeta.hiddenSize);
        const int runtimeInputWidth = static_cast<int>(RuntimeModelInputWidth(
            tensor, modelMeta.inputWidth));
        const int runtimeHiddenSize = static_cast<int>(hidden_size);
        bool sectionMatches = true;
        if (modelMeta.schemaVersion != 1)
        {
            printMismatch("model_meta_schema_version", modelMeta.schemaVersion, 1);
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
    catch (const std::exception& error)
    {
        if (hasModelMeta)
            printMismatch("model_meta", error.what(), "supported persisted model");
    }

    try
    {
        const Donchian20Mode modelMode =
            DBIO::PgModelIO::loadDonchian20ModeMeta(w, modelId);
        if (modelMode != tensor.GetDonchian20Mode())
            printMismatch("donchian20_mode",
                          Donchian20ModeText(modelMode),
                          Donchian20ModeText(tensor.GetDonchian20Mode()));
    }
    catch (const std::exception& error)
    {
        printMismatch("donchian20_mode", error.what(),
                      Donchian20ModeText(tensor.GetDonchian20Mode()));
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
            if (vals.size() >= 12)
            {
                trainConfigMeta.coreLrMult = static_cast<float>(vals[11]);
                hasTrainConfigCoreLrMult = true;
            }
            if (vals.size() >= 13)
            {
                trainConfigMeta.headWeightLrMult = static_cast<float>(vals[12]);
                hasTrainConfigHeadWeightLrMult = true;
            }
            if (vals.size() >= 14)
            {
                trainConfigMeta.headBiasLrMult = static_cast<float>(vals[13]);
                hasTrainConfigHeadBiasLrMult = true;
            }
            if (decodedModelSymbol.has_value())
                trainConfigMeta.symbol = *decodedModelSymbol;
            result.trainConfigMeta = trainConfigMeta;

            if (LogSummary())
            {
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
                std::cout << ",core_lr_mult=";
                if (trainConfigMeta.coreLrMult.has_value())
                    std::cout << *trainConfigMeta.coreLrMult;
                else
                    std::cout << "missing";
                std::cout << ",head_weight_lr_mult=";
                if (trainConfigMeta.headWeightLrMult.has_value())
                    std::cout << *trainConfigMeta.headWeightLrMult;
                else
                    std::cout << "missing";
                std::cout << ",head_bias_lr_mult=";
                if (trainConfigMeta.headBiasLrMult.has_value())
                    std::cout << *trainConfigMeta.headBiasLrMult;
                else
                    std::cout << "missing";
                std::cout << std::endl;
            }

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
            if (vals.size() >= 12)
                sectionMatches = compareFloatField("core_lr_mult",
                                                   vals[11],
                                                   EA::LSTM::CoreLrMultForTarget(requestedTargetType)) && sectionMatches;
            if (vals.size() >= 13)
                sectionMatches = compareFloatField("head_weight_lr_mult",
                                                   vals[12],
                                                   head_weight_lr_mult) && sectionMatches;
            if (vals.size() >= 14)
                sectionMatches = compareFloatField("head_bias_lr_mult",
                                                   vals[13],
                                                   head_bias_lr_mult) && sectionMatches;
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
        missingMinimum.push_back("core_lr_mult");
        missingMinimum.push_back("head_weight_lr_mult");
        missingMinimum.push_back("head_bias_lr_mult");
    }
    else
    {
        if (!hasTrainConfigNumLayers)
            missingMinimum.push_back("num_layers");
        if (!hasTrainConfigNormalizationVersion)
            missingMinimum.push_back("normalization_version");
        if (!hasTrainConfigCoreLrMult)
            missingMinimum.push_back("core_lr_mult");
        if (!hasTrainConfigHeadWeightLrMult)
            missingMinimum.push_back("head_weight_lr_mult");
        if (!hasTrainConfigHeadBiasLrMult)
            missingMinimum.push_back("head_bias_lr_mult");
    }
    if (!hasTargetMeta)
        missingMinimum.push_back("target_type");
    if (!hasTrainSymbolMeta)
        missingMinimum.push_back("symbol");
    if (!hasModelMeta)
    {
        missingMinimum.push_back("feature_count");
        missingMinimum.push_back("hidden_size");
    }

    result.hasMismatch = mismatch;
    result.metadataGap = !missingMinimum.empty();
    result.configMatch =
        targetMetaMatches &&
        modelMetaMatches &&
        trainSymbolMetaMatches &&
        trainConfigMetaMatches &&
        !mismatch &&
        missingMinimum.empty();

    if (result.configMatch)
    {
        if (LogSummary())
            std::cout << "MODEL_CONFIG_MATCH=1" << std::endl;
    }

    if (!missingMinimum.empty())
    {
        if (LogSummary())
            std::cout << "MODEL_CONFIG_METADATA_GAP"
                      << ",model_id=" << modelId
                      << ",missing=" << JoinStrings(missingMinimum, ";")
                      << ",recommend_minimum_additions=" << JoinStrings(missingMinimum, ";")
                      << std::endl;
    }

    return result;
}

struct PersistedInferenceConfig
{
    long long modelId = -1;
    std::string modelName;
    std::string symbol;
    std::string symbolSource = "unknown";
    TrainConfigMeta trainConfig;
    bool hasTrainConfigMeta = false;
    bool hasTargetMeta = false;
    bool hasModelMeta = false;
    bool hasCompleteTrainConfigMeta = false;
    int modelInputWidth = 0;
    size_t modelHiddenSize = 0;
    EA::LSTM::TargetType targetType = EA::LSTM::TargetType::UpNeutralDownReturn;
    EA::FeatureWarmupScope featureWarmupScope =
        EA::FeatureWarmupScope::LegacyColdBoundary;
    Donchian20Mode donchian20Mode = kDefaultDonchian20Mode;
    std::size_t donchianLookback = kDefaultDonchianLookback;
    EA::FeatureAblationMask featureAblationMask;
};

template <typename T>
std::string ValueToString(const T& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

std::string ModelNameForId(pqxx::work& w, long long modelId)
{
    pqxx::result r = w.exec_params(
        "SELECT COALESCE(name, '') FROM model WHERE model_id = $1;",
        modelId);
    if (r.empty())
        throw std::runtime_error("model_id not found");
    return r[0][0].as<std::string>();
}

long long LatestModelId(pqxx::work& w)
{
    pqxx::result r = w.exec("SELECT max(model_id) FROM model;");
    if (r.empty() || r[0][0].is_null())
        return -1;
    return r[0][0].as<long long>();
}

void PrintDatabaseModelSymbol(long long modelId, const std::string& symbol)
{
    if (!LogSummary())
        return;
    std::cout << "MODEL_SYMBOL"
              << ",source=database"
              << ",model_id=" << modelId
              << ",symbol=" << symbol
              << std::endl;
}

void PrintLegacyModelSymbol(long long modelId, const std::string& symbol)
{
    if (!LogSummary())
        return;
    std::cout << "MODEL_SYMBOL"
              << ",source=legacy"
              << ",model_id=" << modelId
              << ",symbol=" << symbol
              << ",warning=missing_metadata"
              << std::endl;
}

void PrintMissingModelSymbol(long long modelId)
{
    if (!LogSummary())
        return;
    std::cout << "MODEL_SYMBOL_MISSING"
              << ",model_id=" << modelId
              << std::endl;
}

void ValidateRuntimeSymbolMatchesModel(const std::optional<std::string>& runtimeSymbol,
                                       const std::string& modelSymbol)
{
    if (!runtimeSymbol.has_value())
        return;

    const std::string canonicalRuntimeSymbol = EA::CanonicalSymbol::Normalize(*runtimeSymbol);
    const std::string canonicalModelSymbol = EA::CanonicalSymbol::Normalize(modelSymbol);
    if (canonicalRuntimeSymbol != canonicalModelSymbol)
    {
        std::cerr << "MODEL_SYMBOL_MISMATCH"
                  << ",runtime=" << canonicalRuntimeSymbol
                  << ",model=" << canonicalModelSymbol
                  << std::endl;
        throw std::runtime_error("MODEL_SYMBOL_MISMATCH");
    }

    DiagnosticOut() << "INFERENCE_CLI_ARG_REDUNDANT"
                    << ",param=--symbol"
                    << ",value=" << canonicalRuntimeSymbol
                    << ",source=persisted_model"
                    << std::endl;
}

std::optional<std::string> ResolveLegacySymbolFromModelName(const std::string& modelName,
                                                            const std::vector<std::string>& availableSymbols)
{
    std::optional<std::string> bestMatch;
    for (const auto& symbol : availableSymbols)
    {
        const bool exact = (modelName == symbol);
        const bool prefixWithDash = modelName.rfind(symbol + "-", 0) == 0;
        const bool prefixWithUnderscore = modelName.rfind(symbol + "_", 0) == 0;
        if (exact || prefixWithDash || prefixWithUnderscore)
        {
            if (!bestMatch.has_value() || symbol.size() > bestMatch->size())
                bestMatch = symbol;
        }
    }
    return bestMatch;
}

std::string ResolveLegacyModelSymbol(long long modelId,
                                     const std::string& modelName,
                                     const std::optional<std::string>& runtimeSymbol,
                                     const std::vector<std::string>& availableSymbols)
{
    PrintMissingModelSymbol(modelId);
    const auto modelNameSymbol = ResolveLegacySymbolFromModelName(modelName, availableSymbols);
    if (modelNameSymbol.has_value())
    {
        const std::string legacySymbol = EA::CanonicalSymbol::Normalize(*modelNameSymbol);
        if (runtimeSymbol.has_value())
            ValidateRuntimeSymbolMatchesModel(runtimeSymbol, legacySymbol);
        PrintLegacyModelSymbol(modelId, legacySymbol);
        return legacySymbol;
    }

    if (runtimeSymbol.has_value())
    {
        const std::string legacySymbol = EA::CanonicalSymbol::Normalize(*runtimeSymbol);
        PrintLegacyModelSymbol(modelId, legacySymbol);
        return legacySymbol;
    }

    throw std::runtime_error("unable to resolve legacy model symbol; train_symbol_meta is missing");
}

TrainConfigMeta LoadTrainConfigMetaForInference(pqxx::work& w,
                                                long long modelId,
                                                bool& hasCompleteExtendedFields)
{
    auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "train_config_meta");
    auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "train_config_meta");
    if (dims.n_rows != 1 ||
        dims.n_cols < DBIO::PgModelIO::kTrainConfigMetaFieldCount ||
        vals.size() < static_cast<size_t>(DBIO::PgModelIO::kTrainConfigMetaFieldCount))
        throw std::runtime_error("train_config_meta missing required 1x8 inference fields");

    TrainConfigMeta meta;
    meta.schemaVersion = static_cast<int>(std::llround(vals[0]));
    meta.predictionHorizon = static_cast<size_t>(std::llround(vals[1]));
    meta.thresholdLogret = static_cast<float>(vals[2]);
    meta.windowSize = static_cast<size_t>(std::llround(vals[3]));
    meta.labelRuleId = static_cast<int>(std::llround(vals[4]));
    meta.classWeightDown = static_cast<float>(vals[5]);
    meta.classWeightNeutral = static_cast<float>(vals[6]);
    meta.classWeightUp = static_cast<float>(vals[7]);
    if (vals.size() >= 9)
        meta.numLayers = static_cast<size_t>(std::llround(vals[8]));
    if (vals.size() >= 10)
        meta.normalizationVersion = static_cast<int>(std::llround(vals[9]));
    if (vals.size() >= 11)
        meta.epochsTrained = static_cast<size_t>(std::llround(vals[10]));
    if (vals.size() >= 12)
        meta.coreLrMult = static_cast<float>(vals[11]);
    if (vals.size() >= 13)
        meta.headWeightLrMult = static_cast<float>(vals[12]);
    if (vals.size() >= 14)
        meta.headBiasLrMult = static_cast<float>(vals[13]);

    hasCompleteExtendedFields =
        vals.size() >= static_cast<size_t>(DBIO::PgModelIO::kTrainConfigMetaExtendedFieldCount);
    return meta;
}

template <typename T>
void ValidateRedundantCliInteger(const char* param,
                                 const std::optional<T>& cliValue,
                                 size_t modelValue)
{
    if (!cliValue.has_value())
        return;
    if (static_cast<size_t>(*cliValue) != modelValue)
    {
        std::cerr << "CONFIG_MISMATCH"
                  << ",param=" << param
                  << ",model=" << modelValue
                  << ",cli=" << *cliValue
                  << std::endl;
        throw std::runtime_error(std::string("CONFIG_MISMATCH for ") + param);
    }
    DiagnosticOut() << "INFERENCE_CLI_ARG_REDUNDANT"
                    << ",param=" << param
                    << ",value=" << *cliValue
                    << ",source=persisted_model"
                    << std::endl;
}

void ValidateRedundantCliFloat(const char* param,
                               const std::optional<double>& cliValue,
                               double modelValue)
{
    if (!cliValue.has_value())
        return;
    constexpr double kTolerance = 1e-7;
    if (std::fabs(*cliValue - modelValue) > kTolerance)
    {
        std::cerr << "CONFIG_MISMATCH"
                  << ",param=" << param
                  << ",model=" << modelValue
                  << ",cli=" << *cliValue
                  << std::endl;
        throw std::runtime_error(std::string("CONFIG_MISMATCH for ") + param);
    }
    DiagnosticOut() << "INFERENCE_CLI_ARG_REDUNDANT"
                    << ",param=" << param
                    << ",value=" << *cliValue
                    << ",source=persisted_model"
                    << std::endl;
}

PersistedInferenceConfig LoadPersistedInferenceConfig(pqxx::work& w,
                                                      long long modelId,
                                                      const LaunchArgs& launchArgs,
                                                      const std::vector<std::string>& availableSymbols)
{
    PersistedInferenceConfig cfg;
    cfg.modelId = modelId;
    cfg.modelName = ModelNameForId(w, modelId);
    cfg.featureWarmupScope = DBIO::PgModelIO::loadFeatureWarmupScopeMeta(w, modelId);
    if (launchArgs.featureWarmupScope.has_value() &&
        *launchArgs.featureWarmupScope != cfg.featureWarmupScope)
        throw std::runtime_error("feature warmup scope mismatch: persisted/runtime");

    std::optional<std::string> databaseSymbol;
    try
    {
        databaseSymbol = DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
    }
    catch (const std::exception&)
    {
    }

    if (databaseSymbol.has_value())
    {
        cfg.symbol = *databaseSymbol;
        cfg.symbolSource = "database";
        ValidateRuntimeSymbolMatchesModel(launchArgs.symbol, cfg.symbol);
        PrintDatabaseModelSymbol(modelId, cfg.symbol);
    }
    else
    {
        cfg.symbol = ResolveLegacyModelSymbol(modelId,
                                              cfg.modelName,
                                              launchArgs.symbol,
                                              availableSymbols);
        cfg.symbolSource = "legacy";
    }

    cfg.trainConfig = LoadTrainConfigMetaForInference(w,
                                                      modelId,
                                                      cfg.hasCompleteTrainConfigMeta);
    cfg.hasTrainConfigMeta = true;
    cfg.trainConfig.symbol = cfg.symbol;
    if (cfg.trainConfig.schemaVersion != DBIO::PgModelIO::kTrainConfigMetaSchemaVersion)
        throw std::runtime_error("unsupported train_config_meta schema_version");
    if (cfg.trainConfig.labelRuleId != DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId)
        throw std::runtime_error("unsupported train_config_meta label_rule_id");
    if (cfg.trainConfig.numLayers != 1)
        throw std::runtime_error("unsupported persisted num_layers; this binary supports only 1");

    {
        const auto modelMeta =
            DBIO::PgModelIO::loadRequiredModelMeta(w, modelId);
        cfg.modelInputWidth = static_cast<int>(modelMeta.inputWidth);
        cfg.modelHiddenSize = modelMeta.hiddenSize;
        cfg.hasModelMeta = true;
    }

    cfg.donchian20Mode = DBIO::PgModelIO::loadDonchian20ModeMeta(w, modelId);
    cfg.donchianLookback = DBIO::PgModelIO::loadDonchianLookbackMeta(w, modelId);
    cfg.featureAblationMask = LoadModelFeatureAblationMask(w, modelId);
    if (launchArgs.donchianLookback.has_value() &&
        *launchArgs.donchianLookback != cfg.donchianLookback)
        throw std::runtime_error("Donchian lookback mismatch: persisted/runtime");

    try
    {
        auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "target_meta");
        auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "target_meta");
        if (dims.n_rows != 1 || dims.n_cols != 6 || vals.size() != 6)
            throw std::runtime_error("target_meta has invalid shape");
        cfg.targetType = static_cast<EA::LSTM::TargetType>(static_cast<int>(std::llround(vals[0])));
        cfg.hasTargetMeta = true;
    }
    catch (const std::exception&)
    {
        cfg.targetType = EA::LSTM::TargetType::UpNeutralDownReturn;
    }

    ValidateRedundantCliInteger("--prediction-horizon", launchArgs.predictionHorizon, cfg.trainConfig.predictionHorizon);
    ValidateRedundantCliFloat("--threshold", launchArgs.thresholdLogret, cfg.trainConfig.thresholdLogret);
    ValidateRedundantCliInteger("--window-size", launchArgs.windowSize, cfg.trainConfig.windowSize);
    ValidateRedundantCliInteger("--hidden-size", launchArgs.hiddenSize, cfg.modelHiddenSize);
    ValidateRedundantCliInteger("--num-layers", launchArgs.numLayers, cfg.trainConfig.numLayers);
    if (cfg.trainConfig.coreLrMult.has_value())
        ValidateRedundantCliFloat("--core-lr-mult", launchArgs.coreLrMult, *cfg.trainConfig.coreLrMult);
    if (cfg.trainConfig.headWeightLrMult.has_value())
        ValidateRedundantCliFloat("--head-weight-lr-mult", launchArgs.headWeightLrMult, *cfg.trainConfig.headWeightLrMult);
    if (cfg.trainConfig.headBiasLrMult.has_value())
        ValidateRedundantCliFloat("--head-bias-lr-mult", launchArgs.headBiasLrMult, *cfg.trainConfig.headBiasLrMult);

    return cfg;
}

void ApplyPersistedInferenceRuntimeConfig(const PersistedInferenceConfig& cfg)
{
    prediction_horizon = cfg.trainConfig.predictionHorizon;
    c_next_threshold = cfg.trainConfig.thresholdLogret;
    window_size = cfg.trainConfig.windowSize;
    hidden_size = cfg.modelHiddenSize;
    n_out = hidden_size;
    num_layers = cfg.trainConfig.numLayers;
    normalization_version = cfg.trainConfig.normalizationVersion;
    if (cfg.trainConfig.coreLrMult.has_value())
        core_lr_mult = *cfg.trainConfig.coreLrMult;
    if (cfg.trainConfig.headWeightLrMult.has_value())
        head_weight_lr_mult = *cfg.trainConfig.headWeightLrMult;
    if (cfg.trainConfig.headBiasLrMult.has_value())
        head_bias_lr_mult = *cfg.trainConfig.headBiasLrMult;
}

void PrintResolvedInferenceConfig(const PersistedInferenceConfig& cfg)
{
    if (!LogSummary())
        return;
    DiagnosticOut() << "INFER_SYMBOL_RESOLVED"
                    << ",source=" << cfg.symbolSource
                    << ",symbol=" << cfg.symbol
                    << std::endl;
    std::cout << "INFERENCE_CONFIG_RESOLVED"
              << ",model_id=" << cfg.modelId
              << ",symbol=" << cfg.symbol
              << ",prediction_horizon=" << cfg.trainConfig.predictionHorizon
              << ",threshold=" << cfg.trainConfig.thresholdLogret
              << ",window_size=" << cfg.trainConfig.windowSize
              << ",target_type=" << TargetTypeName(cfg.targetType)
              << ",donchian20_mode=" << Donchian20ModeText(cfg.donchian20Mode)
              << ",donchian_lookback=" << cfg.donchianLookback
              << ",feature_warmup_scope=" << EA::FeatureWarmupScopeText(cfg.featureWarmupScope)
              << ",label_rule_id=" << cfg.trainConfig.labelRuleId
              << ",source=persisted_model"
              << std::endl;
}

void ValidateLoadedModelSymbolForSelectedTable(pqxx::work& w,
                                               long long modelId,
                                               const std::string& selectedSymbol)
{
    std::optional<std::string> databaseSymbol;
    try
    {
        databaseSymbol = DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
    }
    catch (const std::exception&)
    {
    }

    if (databaseSymbol.has_value())
    {
        PrintDatabaseModelSymbol(modelId, *databaseSymbol);
        ValidateRuntimeSymbolMatchesModel(EA::CanonicalSymbol::Normalize(selectedSymbol), *databaseSymbol);
        return;
    }

    PrintMissingModelSymbol(modelId);
    PrintLegacyModelSymbol(modelId, EA::CanonicalSymbol::Normalize(selectedSymbol));
}

std::optional<long long> InferenceAnchorModelId(pqxx::work& w, const LaunchArgs& launchArgs)
{
    if (!gRuntimeInferenceMode)
        return std::nullopt;
    if (launchArgs.modelId.has_value())
        return *launchArgs.modelId;
    if (launchArgs.inferAll)
        return std::nullopt;
    const long long latest = LatestModelId(w);
    if (latest > 0)
        return latest;
    return std::nullopt;
}

struct InferAllCandidate
{
    long long modelId = -1;
    std::string name;
    std::size_t modelInputWidth = 0;
    std::optional<size_t> completedEpochs;
    bool legacyMissingSymbol = false;
    bool metadataGap = false;
};

struct InferAllSummaryRow
{
    long long modelId = -1;
    std::string name;
    std::optional<size_t> completedEpochs;
    double accuracy = 0.0;
    ModelAcceptanceSummary acceptance;
    std::optional<EA::InferenceProfitability::Statistics> profitability;
    std::string profitabilitySourceContentHash;
};

struct InferenceIdentity
{
    long long modelId = -1;
    std::string symbol;
    size_t predictionHorizon = 0;
    double thresholdLogret = 0.0;
    size_t windowSize = 0;
    int labelRuleId = DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId;
    int targetType = 0;
    std::string fromDate;
    std::string toDate;
};

struct SchedulerInferencePersistenceContext
{
    std::string inferenceScope = "final";
    long long modelId = -1;
    std::optional<long long> schedulerExperimentId;
    std::optional<long long> checkpointEvalId;
    std::optional<long long> parentExperimentId;
    std::optional<long long> checkpointModelId;
    std::optional<int> checkpointEpoch;
    std::string evalStatus = "unknown";
};

bool InferenceEvalResultScopeColumnsExist(pqxx::work& w)
{
    return SchedulerTableExists(w, "inference_eval_result") &&
           SchedulerColumnExists(w, "inference_eval_result", "inference_scope") &&
           SchedulerColumnExists(w, "inference_eval_result", "checkpoint_eval_id") &&
           SchedulerColumnExists(w, "inference_eval_result", "parent_experiment_id") &&
           SchedulerColumnExists(w, "inference_eval_result", "checkpoint_epoch");
}

std::string DatePrefix(const std::string& value)
{
    return value.substr(0, std::min<size_t>(10, value.size()));
}

void LogCheckpointInferencePersistenceFailure(
    const SchedulerInferencePersistenceContext& context,
    const std::string& error)
{
    std::cerr << "CHECKPOINT_INFER_RESULT_PERSIST_FAILED"
              << ",checkpoint_eval_id=" << context.checkpointEvalId.value_or(-1)
              << ",parent_experiment_id=" << context.parentExperimentId.value_or(-1)
              << ",checkpoint_model_id=" << context.checkpointModelId.value_or(context.modelId)
              << ",checkpoint_epoch=" << context.checkpointEpoch.value_or(-1)
              << ",model_id=" << context.modelId
              << ",inference_scope=checkpoint"
              << ",status=" << context.evalStatus
              << ",error=" << error
              << std::endl;
}

SchedulerInferencePersistenceContext ResolveSchedulerInferencePersistenceContext(
    pqxx::work& w,
    const LaunchArgs& launchArgs,
    const std::string& resolvedSymbol,
    const std::string& fromDate,
    const std::string& toDate)
{
    SchedulerInferencePersistenceContext context;
    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        context.inferenceScope = "checkpoint";
        context.checkpointEvalId = *launchArgs.schedulerCheckpointEvalId;
    }
    if (launchArgs.modelId.has_value())
        context.modelId = *launchArgs.modelId;

    if (!launchArgs.modelId.has_value())
        throw std::runtime_error("scheduler inference persistence requires an explicit model_id");
    if (!InferenceEvalResultScopeColumnsExist(w))
    {
        if (context.inferenceScope == "checkpoint")
            LogCheckpointInferencePersistenceFailure(context, "migration_018_required");
        throw std::runtime_error("migration 018 is required for scheduler inference persistence");
    }

    if (launchArgs.schedulerCheckpointEvalId.has_value())
    {
        auto reject = [&](const std::string& error) -> void
        {
            LogCheckpointInferencePersistenceFailure(context, error);
            throw std::runtime_error(error);
        };

        pqxx::result rows = w.exec_params(
            "SELECT ce.checkpoint_eval_id, "
            "COALESCE(ce.parent_experiment_id, ce.experiment_id), ce.experiment_id, "
            "ce.checkpoint_model_id, ce.checkpoint_epoch, ce.status, ce.phase, "
            "ce.symbol, ce.prediction_horizon, e.symbol, e.prediction_horizon, "
            "e.c_next_threshold, e.infer_start::date::text, e.infer_end::date::text, "
            "m.experiment_id, "
            "(SELECT MAX(round(mx.value)::int) FROM matrix mx "
            " WHERE mx.model_id = ce.checkpoint_model_id "
            " AND mx.param_name = 'train_config_meta' "
            " AND mx.row_idx = 0 AND mx.col_idx = 10) "
            "FROM experiment_checkpoint_eval ce "
            "JOIN experiment e ON e.experiment_id = COALESCE(ce.parent_experiment_id, ce.experiment_id) "
            "JOIN model m ON m.model_id = ce.checkpoint_model_id "
            "WHERE ce.checkpoint_eval_id = $1;",
            *launchArgs.schedulerCheckpointEvalId);
        if (rows.empty())
            reject("checkpoint_eval_not_found");

        const pqxx::row row = rows[0];
        context.parentExperimentId = row[1].as<long long>();
        const long long legacyExperimentId = row[2].as<long long>();
        const long long checkpointModelId = row[3].as<long long>();
        context.checkpointModelId = checkpointModelId;
        context.checkpointEpoch = row[4].as<int>();
        context.evalStatus = row[5].as<std::string>();
        const std::string evalPhase = row[6].as<std::string>();
        const std::optional<std::string> evalSymbol =
            row[7].is_null() ? std::optional<std::string>{} :
                               std::optional<std::string>{EA::CanonicalSymbol::Normalize(row[7].as<std::string>())};
        const std::optional<int> evalHorizon =
            row[8].is_null() ? std::optional<int>{} : std::optional<int>{row[8].as<int>()};
        const std::string parentSymbol = EA::CanonicalSymbol::Normalize(row[9].as<std::string>());
        const int parentHorizon = row[10].as<int>();
        const double parentThreshold = row[11].as<double>();
        const std::optional<std::string> parentInferStart =
            row[12].is_null() ? std::optional<std::string>{} : std::optional<std::string>{row[12].as<std::string>()};
        const std::optional<std::string> parentInferEnd =
            row[13].is_null() ? std::optional<std::string>{} : std::optional<std::string>{row[13].as<std::string>()};
        const std::optional<long long> modelExperimentId =
            row[14].is_null() ? std::optional<long long>{} : std::optional<long long>{row[14].as<long long>()};
        const std::optional<int> modelCompletedEpoch =
            row[15].is_null() ? std::optional<int>{} : std::optional<int>{row[15].as<int>()};

        if (checkpointModelId != context.modelId)
            reject("checkpoint_model_id_mismatch");
        if (legacyExperimentId != *context.parentExperimentId)
            reject("checkpoint_parent_experiment_id_mismatch");
        if (!modelExperimentId.has_value() || *modelExperimentId != *context.parentExperimentId)
            reject("checkpoint_model_parent_experiment_id_mismatch");
        if (evalPhase != "infer")
            reject("checkpoint_eval_not_in_infer_phase");
        if (context.evalStatus != "pending" && context.evalStatus != "running")
            reject("checkpoint_eval_not_pending_or_running");
        if (evalSymbol.has_value() && *evalSymbol != parentSymbol)
            reject("checkpoint_eval_symbol_parent_mismatch");
        if (evalHorizon.has_value() && *evalHorizon != parentHorizon)
            reject("checkpoint_eval_horizon_parent_mismatch");
        if (parentSymbol != EA::CanonicalSymbol::Normalize(resolvedSymbol))
            reject("checkpoint_runtime_symbol_parent_mismatch");
        if (parentHorizon != static_cast<int>(prediction_horizon))
            reject("checkpoint_runtime_horizon_parent_mismatch");
        if (std::fabs(parentThreshold - static_cast<double>(c_next_threshold)) > 1e-7)
            reject("checkpoint_runtime_threshold_parent_mismatch");
        if (!parentInferStart.has_value() || !parentInferEnd.has_value())
            reject("checkpoint_parent_missing_infer_range");
        if (*parentInferStart != DatePrefix(fromDate) || *parentInferEnd != DatePrefix(toDate))
            reject("checkpoint_runtime_infer_range_parent_mismatch");
        if (modelCompletedEpoch.has_value() && *modelCompletedEpoch != *context.checkpointEpoch)
            reject("checkpoint_model_completed_epoch_mismatch");

        return context;
    }

    if (!launchArgs.schedulerExperimentId.has_value())
        throw std::runtime_error("missing scheduler inference identity");

    context.schedulerExperimentId = *launchArgs.schedulerExperimentId;
    pqxx::result rows = w.exec_params(
        "SELECT e.experiment_id, e.last_model_id, e.status, e.phase, e.symbol, "
        "e.prediction_horizon, e.c_next_threshold, e.infer_start::date::text, "
        "e.infer_end::date::text, m.experiment_id "
        "FROM experiment e "
        "JOIN model m ON m.model_id = $2 "
        "WHERE e.experiment_id = $1;",
        *launchArgs.schedulerExperimentId,
        *launchArgs.modelId);
    if (rows.empty())
        throw std::runtime_error("scheduler_final_experiment_not_found");
    const pqxx::row row = rows[0];
    if (row[1].is_null() || row[1].as<long long>() != context.modelId)
        throw std::runtime_error("scheduler_final_model_id_mismatch");
    context.evalStatus = row[2].as<std::string>();
    if (context.evalStatus != "pending" && context.evalStatus != "running")
        throw std::runtime_error("scheduler_final_experiment_not_pending_or_running");
    if (row[3].as<std::string>() != "infer")
        throw std::runtime_error("scheduler_final_experiment_not_in_infer_phase");
    if (EA::CanonicalSymbol::Normalize(row[4].as<std::string>()) !=
        EA::CanonicalSymbol::Normalize(resolvedSymbol))
        throw std::runtime_error("scheduler_final_symbol_mismatch");
    if (row[5].as<int>() != static_cast<int>(prediction_horizon))
        throw std::runtime_error("scheduler_final_horizon_mismatch");
    if (std::fabs(row[6].as<double>() - static_cast<double>(c_next_threshold)) > 1e-7)
        throw std::runtime_error("scheduler_final_threshold_mismatch");
    if (row[7].is_null() || row[8].is_null() ||
        row[7].as<std::string>() != DatePrefix(fromDate) ||
        row[8].as<std::string>() != DatePrefix(toDate))
        throw std::runtime_error("scheduler_final_infer_range_mismatch");
    if (row[9].is_null() || row[9].as<long long>() != *launchArgs.schedulerExperimentId)
        throw std::runtime_error("scheduler_final_model_experiment_id_mismatch");
    return context;
}

struct InferAllSkipDetail
{
    std::string reason = "CONFIG_MISMATCH";
    std::string field;
    std::string anchor;
    std::string candidate;
    std::string detail;
};

struct InferenceEvaluationResult
{
    double accuracy = 0.0;
    std::optional<size_t> completedEpochs;
    ModelAcceptanceSummary acceptance;
    std::optional<EA::InferenceProfitability::Statistics> profitability;
    std::string profitabilitySourceContentHash;
};

EA::LSTM CreateLstmForRuntimeLogLevel(const Tensor& tensor,
                                      float initialLongTerm,
                                      float initialShortTerm,
                                      EA::LSTM::TargetType targetType,
                                      std::optional<std::size_t> modelInputWidth = std::nullopt,
                                      EA::FeatureAblationMask ablationMask = {})
{
    ScopedDiagnosticCoutSilencer silence;
    return EA::LSTM { tensor, initialLongTerm, initialShortTerm, targetType,
                      modelInputWidth, std::move(ablationMask) };
}

InferenceIdentity BuildInferenceIdentity(long long modelId,
                                         const std::string& symbol,
                                         EA::LSTM::TargetType targetType,
                                         const std::string& fromDate,
                                         const std::string& toDate)
{
    InferenceIdentity identity;
    identity.modelId = modelId;
    identity.symbol = symbol;
    identity.predictionHorizon = prediction_horizon;
    identity.thresholdLogret = c_next_threshold;
    identity.windowSize = window_size;
    identity.labelRuleId = DirectionLabelRuleId();
    identity.targetType = static_cast<int>(targetType);
    identity.fromDate = fromDate;
    identity.toDate = toDate;
    return identity;
}

bool RequireInferenceEvalResultTable(pqxx::work& w)
{
    if (InferenceEvalResultScopeColumnsExist(w) &&
        EA::InferenceProfitability::SchemaExists(w))
        return true;

    std::cerr << "DATABASE_MIGRATION_REQUIRED"
              << ",missing=inference_profitability_observation"
              << ",command=./migrate_lstm_db.sh"
              << std::endl;
    return false;
}

std::optional<InferAllSummaryRow> LoadCompletedInferenceResult(pqxx::work& w,
                                                               const InferenceIdentity& identity,
                                                               const InferAllCandidate& candidate)
{
    pqxx::result rows = w.exec_params(
        "SELECT completed_epochs, accuracy, accept_model, COALESCE(reject_reason, ''), "
        "COALESCE(pred_down, 0), COALESCE(pred_neutral, 0), COALESCE(pred_up, 0) "
        "FROM inference_eval_result "
        "WHERE model_id = $1 AND symbol = $2 AND prediction_horizon = $3 "
        "AND threshold_logret = $4 AND window_size = $5 AND label_rule_id = $6 "
        "AND target_type = $7 AND from_date = $8 AND to_date = $9 "
        "AND status = 'completed' AND inference_scope = 'final' "
        "AND checkpoint_eval_id IS NULL "
        "ORDER BY completed_at DESC LIMIT 1;",
        identity.modelId,
        identity.symbol,
        static_cast<long long>(identity.predictionHorizon),
        identity.thresholdLogret,
        static_cast<long long>(identity.windowSize),
        identity.labelRuleId,
        identity.targetType,
        identity.fromDate,
        identity.toDate);

    if (rows.empty())
        return std::nullopt;

    InferAllSummaryRow row;
    row.modelId = identity.modelId;
    row.name = candidate.name;
    if (!rows[0][0].is_null())
        row.completedEpochs = rows[0][0].as<size_t>();
    else
        row.completedEpochs = candidate.completedEpochs;
    row.accuracy = rows[0][1].is_null() ? 0.0 : rows[0][1].as<double>();
    row.acceptance.acceptModel = !rows[0][2].is_null() && rows[0][2].as<bool>();
    row.acceptance.rejectReason = rows[0][3].as<std::string>();
    if (row.acceptance.rejectReason.empty())
        row.acceptance.rejectReason = "none";
    row.acceptance.predFrac[0] = rows[0][4].as<double>();
    row.acceptance.predFrac[1] = rows[0][5].as<double>();
    row.acceptance.predFrac[2] = rows[0][6].as<double>();
    return row;
}

long long PersistCompletedInferenceResult(pqxx::work& w,
                                          const InferenceIdentity& identity,
                                          const InferAllSummaryRow& row)
{
    const long long completedEpochs = row.completedEpochs.has_value()
        ? static_cast<long long>(*row.completedEpochs)
        : -1;
    const pqxx::result persisted = w.exec_params(
        "INSERT INTO inference_eval_result ("
        "model_id, symbol, prediction_horizon, threshold_logret, window_size, label_rule_id, target_type, "
        "from_date, to_date, completed_epochs, accuracy, accept_model, reject_reason, "
        "pred_down, pred_neutral, pred_up, status, completed_at, inference_scope, "
        "checkpoint_eval_id, parent_experiment_id, checkpoint_epoch"
        ") VALUES ("
        "$1,$2,$3,$4,$5,$6,$7,$8,$9,NULLIF($10::bigint, -1),$11,$12,$13,$14,$15,$16,"
        "'completed',now(),'final',NULL,NULL,NULL"
        ") ON CONFLICT ("
        "model_id, symbol, prediction_horizon, threshold_logret, window_size, label_rule_id, "
        "target_type, from_date, to_date"
        ") WHERE status = 'completed' AND inference_scope = 'final' DO UPDATE SET "
        "completed_epochs = EXCLUDED.completed_epochs, accuracy = EXCLUDED.accuracy, "
        "accept_model = EXCLUDED.accept_model, reject_reason = EXCLUDED.reject_reason, "
        "pred_down = EXCLUDED.pred_down, pred_neutral = EXCLUDED.pred_neutral, "
        "pred_up = EXCLUDED.pred_up, completed_at = now() RETURNING id;",
        identity.modelId,
        identity.symbol,
        static_cast<long long>(identity.predictionHorizon),
        identity.thresholdLogret,
        static_cast<long long>(identity.windowSize),
        identity.labelRuleId,
        identity.targetType,
        identity.fromDate,
        identity.toDate,
        completedEpochs,
        row.accuracy,
        row.acceptance.acceptModel,
        row.acceptance.rejectReason,
        row.acceptance.predFrac[0],
        row.acceptance.predFrac[1],
        row.acceptance.predFrac[2]);
    if (persisted.size() != 1)
        throw std::runtime_error("completed_final_inference_result_not_returned");
    return persisted.one_row()[0].as<long long>();
}

void PersistFailedInferenceResult(pqxx::work& w,
                                  const InferenceIdentity& identity,
                                  const InferAllCandidate& candidate,
                                  const std::string& error)
{
    const long long completedEpochs = candidate.completedEpochs.has_value()
        ? static_cast<long long>(*candidate.completedEpochs)
        : -1;
    w.exec_params(
        "INSERT INTO inference_eval_result ("
        "model_id, symbol, prediction_horizon, threshold_logret, window_size, label_rule_id, target_type, "
        "from_date, to_date, completed_epochs, reject_reason, status, completed_at, inference_scope, "
        "checkpoint_eval_id, parent_experiment_id, checkpoint_epoch"
        ") VALUES ("
        "$1,$2,$3,$4,$5,$6,$7,$8,$9,NULLIF($10::bigint, -1),$11,'failed',now(),"
        "'final',NULL,NULL,NULL"
        ");",
        identity.modelId,
        identity.symbol,
        static_cast<long long>(identity.predictionHorizon),
        identity.thresholdLogret,
        static_cast<long long>(identity.windowSize),
        identity.labelRuleId,
        identity.targetType,
        identity.fromDate,
        identity.toDate,
        completedEpochs,
        error);
}

long long PersistCompletedCheckpointInferenceResult(
    pqxx::work& w,
    const InferenceIdentity& identity,
    const InferAllSummaryRow& row,
    const SchedulerInferencePersistenceContext& context)
{
    if (context.inferenceScope != "checkpoint" ||
        !context.checkpointEvalId.has_value() ||
        !context.parentExperimentId.has_value() ||
        !context.checkpointEpoch.has_value())
    {
        throw std::runtime_error("incomplete_checkpoint_inference_persistence_context");
    }
    if (identity.modelId != context.modelId)
        throw std::runtime_error("checkpoint_persistence_model_id_mismatch");

    pqxx::result existing = w.exec_params(
        "SELECT id, model_id, parent_experiment_id, checkpoint_epoch, status "
        "FROM inference_eval_result "
        "WHERE checkpoint_eval_id = $1 AND inference_scope = 'checkpoint' "
        "FOR UPDATE;",
        *context.checkpointEvalId);
    if (!existing.empty())
    {
        const bool identityMatches =
            existing[0][1].as<long long>() == identity.modelId &&
            !existing[0][2].is_null() &&
            existing[0][2].as<long long>() == *context.parentExperimentId &&
            !existing[0][3].is_null() &&
            existing[0][3].as<int>() == *context.checkpointEpoch;
        if (!identityMatches)
            throw std::runtime_error("existing_checkpoint_inference_result_identity_mismatch");
        if (existing[0][4].as<std::string>() == "completed")
        {
            std::cout << "CHECKPOINT_INFER_RESULT_DUPLICATE"
                      << ",checkpoint_eval_id=" << *context.checkpointEvalId
                      << ",parent_experiment_id=" << *context.parentExperimentId
                      << ",checkpoint_model_id=" << context.checkpointModelId.value_or(context.modelId)
                      << ",checkpoint_epoch=" << *context.checkpointEpoch
                      << ",model_id=" << identity.modelId
                      << ",inference_scope=checkpoint"
                      << ",status=completed"
                      << std::endl;
            return existing[0][0].as<long long>();
        }
    }

    const long long completedEpochs = row.completedEpochs.has_value()
        ? static_cast<long long>(*row.completedEpochs)
        : static_cast<long long>(*context.checkpointEpoch);
    const pqxx::result persisted = w.exec_params(
        "INSERT INTO inference_eval_result ("
        "model_id, symbol, prediction_horizon, threshold_logret, window_size, label_rule_id, target_type, "
        "from_date, to_date, completed_epochs, accuracy, accept_model, reject_reason, "
        "pred_down, pred_neutral, pred_up, status, completed_at, inference_scope, "
        "checkpoint_eval_id, parent_experiment_id, checkpoint_epoch"
        ") VALUES ("
        "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,"
        "'completed',now(),'checkpoint',$17,$18,$19"
        ") ON CONFLICT (checkpoint_eval_id) WHERE inference_scope = 'checkpoint' DO UPDATE SET "
        "model_id = EXCLUDED.model_id, symbol = EXCLUDED.symbol, "
        "prediction_horizon = EXCLUDED.prediction_horizon, threshold_logret = EXCLUDED.threshold_logret, "
        "window_size = EXCLUDED.window_size, label_rule_id = EXCLUDED.label_rule_id, "
        "target_type = EXCLUDED.target_type, from_date = EXCLUDED.from_date, to_date = EXCLUDED.to_date, "
        "completed_epochs = EXCLUDED.completed_epochs, accuracy = EXCLUDED.accuracy, "
        "accept_model = EXCLUDED.accept_model, reject_reason = EXCLUDED.reject_reason, "
        "pred_down = EXCLUDED.pred_down, pred_neutral = EXCLUDED.pred_neutral, "
        "pred_up = EXCLUDED.pred_up, status = 'completed', completed_at = now(), "
        "parent_experiment_id = EXCLUDED.parent_experiment_id, "
        "checkpoint_epoch = EXCLUDED.checkpoint_epoch RETURNING id;",
        identity.modelId,
        identity.symbol,
        static_cast<long long>(identity.predictionHorizon),
        identity.thresholdLogret,
        static_cast<long long>(identity.windowSize),
        identity.labelRuleId,
        identity.targetType,
        identity.fromDate,
        identity.toDate,
        completedEpochs,
        row.accuracy,
        row.acceptance.acceptModel,
        row.acceptance.rejectReason,
        row.acceptance.predFrac[0],
        row.acceptance.predFrac[1],
        row.acceptance.predFrac[2],
        *context.checkpointEvalId,
        *context.parentExperimentId,
        *context.checkpointEpoch);

    std::cout << "CHECKPOINT_INFER_RESULT_PERSISTED"
              << ",checkpoint_eval_id=" << *context.checkpointEvalId
              << ",parent_experiment_id=" << *context.parentExperimentId
              << ",checkpoint_model_id=" << context.checkpointModelId.value_or(context.modelId)
              << ",checkpoint_epoch=" << *context.checkpointEpoch
              << ",model_id=" << identity.modelId
              << ",inference_scope=checkpoint"
              << ",status=completed"
              << std::endl;
    if (persisted.size() != 1)
        throw std::runtime_error("completed_checkpoint_inference_result_not_returned");
    return persisted.one_row()[0].as<long long>();
}

std::optional<long long> ResolveProfitabilityExperimentId(
    pqxx::work& w,
    const InferenceIdentity& identity,
    const std::optional<long long>& explicitExperimentId)
{
    if (explicitExperimentId.has_value())
        return *explicitExperimentId;
    const pqxx::result rows = w.exec_params(
        "SELECT experiment_id FROM model WHERE model_id=$1;",
        identity.modelId);
    if (rows.size() != 1 || rows[0][0].is_null())
        return std::nullopt;
    return rows[0][0].as<long long>();
}

void PersistInferenceProfitabilityObservation(
    pqxx::work& w,
    long long inferenceEvalResultId,
    const InferenceIdentity& identity,
    const InferAllSummaryRow& row,
    EA::InferenceProfitability::Scope scope,
    const std::optional<long long>& checkpointEvalId,
    const std::optional<long long>& explicitExperimentId)
{
    if (!row.profitability.has_value())
        return;

    EA::InferenceProfitability::ObservationRequest request;
    request.provenance.experimentId =
        ResolveProfitabilityExperimentId(w, identity, explicitExperimentId);
    request.provenance.modelId = identity.modelId;
    request.provenance.inferenceEvalResultId = inferenceEvalResultId;
    request.provenance.scope = scope;
    request.provenance.checkpointEvalId = checkpointEvalId;
    request.provenance.inferenceStart = identity.fromDate;
    request.provenance.inferenceEnd = identity.toDate;
    request.statistics = *row.profitability;
    request.sourceContentHash = row.profitabilitySourceContentHash;

    const auto persisted =
        EA::InferenceProfitability::PersistObservationIdempotently(w, request);
    std::cout << "INFERENCE_PROFITABILITY_OBSERVATION_PERSISTED"
              << ",observation_id=" << persisted.observation.observationId
              << ",observation_identity="
              << persisted.observation.observationIdentityHash
              << ",idempotent_existing="
              << (persisted.created ? "false" : "true")
              << ",inference_scope="
              << EA::InferenceProfitability::ScopeText(scope)
              << ",experiment_id="
              << request.provenance.experimentId.value_or(-1)
              << ",model_id=" << identity.modelId
              << ",inference_eval_result_id=" << inferenceEvalResultId
              << ",checkpoint_eval_id=" << checkpointEvalId.value_or(-1)
              << ",actionable_count=" << request.statistics.actionableCount
              << ",aggregate_terminal_horizon_log_return_sum="
              << request.statistics.aggregateTerminalHorizonLogReturnSum
              << ",metric_definition_identity="
              << persisted.observation.metricDefinitionHash
              << std::endl;
}

InferenceEvaluationResult RunInferenceEvaluation(pqxx::work& w,
                                                 const LaunchArgs& launchArgs,
                                                 EA::LSTM& lstm,
                                                 const Tensor& tensor,
                                                 const std::optional<long long>& loadedModelId,
                                                 const std::string& loadSource,
                                                 EA::LSTM::TargetType requestedTargetType,
                                                 const std::string& rawPriceTableName,
                                                 const std::string& fromDate,
                                                 const std::string& toDate,
                                                 size_t logicalOutputStartIndex,
                                                 bool failOnConfigMismatch)
{
    UseRuntimeDefaultEvalLabelConfig();
    ModelConfigValidationResult modelConfigValidation;
    if (loadedModelId.has_value())
    {
        modelConfigValidation =
            PrintModelConfigValidation(w, *loadedModelId, requestedTargetType, tensor, rawPriceTableName);
        if (failOnConfigMismatch && modelConfigValidation.hasMismatch)
            throw std::runtime_error("candidate became incompatible during validation");
    }

    if (modelConfigValidation.trainConfigMeta.has_value())
        UseModelTrainEvalLabelConfig(*modelConfigValidation.trainConfigMeta);
    else
        UseRuntimeDefaultEvalLabelConfig();

    PrintEvalLabelConfig();
    PrintInferenceConfig(loadedModelId, loadSource, lstm, fromDate, toDate);
    PrintClassificationProofDiagnostics(lstm, tensor, fromDate, toDate);
    PrintPhase2TensorDiagnostics(lstm, tensor, fromDate, toDate);

    size_t totalCorrectLog = 0;
    size_t totalActedLog = 0;
    size_t totalWindows = 0;
    double totalAbsErrMove = 0.0;
    size_t totalCorrectDir = 0;
    size_t totalActedDir = 0;
    size_t totalConfusion[direction_output_size][direction_output_size] = {};
    EA::InferenceProfitability::Accumulator profitability;

    {
        ScopedDiagnosticCoutSilencer silence;
        tensor.ForEachBatchFrom(logicalOutputStartIndex, [&](auto b)
        {
            const auto predictionStats =
                ProcessBatchPredict(lstm, tensor, b, profitability);
            totalCorrectLog += predictionStats.correctLog;
            totalActedLog += predictionStats.actedLog;
            totalWindows += predictionStats.windows;
            totalAbsErrMove += predictionStats.absErrMove;
            totalCorrectDir += predictionStats.correctDir;
            totalActedDir += predictionStats.actedDir;

            for (size_t actual = 0; actual < direction_output_size; ++actual)
                for (size_t pred = 0; pred < direction_output_size; ++pred)
                    totalConfusion[actual][pred] += predictionStats.confusion[actual][pred];

        });
    }

    if (lstm.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
    {
        ScopedDiagnosticCoutSilencer silence;
        EA::LSTM::PrintAndResetEpochBuckets();
        ::PrintAndResetDistribution();
    }

    InferenceEvaluationResult result;
    if (modelConfigValidation.trainConfigMeta.has_value() &&
        modelConfigValidation.trainConfigMeta->epochsTrained.has_value())
        result.completedEpochs = *modelConfigValidation.trainConfigMeta->epochsTrained;

    if (lstm.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
    {
        if (profitability.statistics().predictionCount != totalWindows)
            throw std::runtime_error(
                "inference_profitability_prediction_count_mismatch");
        result.accuracy = totalWindows
            ? (static_cast<double>(totalCorrectDir) / static_cast<double>(totalWindows))
            : 0.0;
        if (LogSummary())
        {
            std::cout << "Overall 3-class accuracy: " << (result.accuracy * 100.0)
                      << "% over " << totalWindows << " windows" << std::endl;
            std::cout << "Overall 3-class confusion matrix (rows=actual [down,neutral,up], cols=pred [down,neutral,up]): "
                      << "[[" << totalConfusion[0][0] << ", " << totalConfusion[0][1] << ", " << totalConfusion[0][2] << "], "
                      << "[" << totalConfusion[1][0] << ", " << totalConfusion[1][1] << ", " << totalConfusion[1][2] << "], "
                      << "[" << totalConfusion[2][0] << ", " << totalConfusion[2][1] << ", " << totalConfusion[2][2] << "]]"
                      << std::endl;
        }
        PrintModelAcceptanceDiagnostic(totalConfusion);
        result.acceptance = ComputeModelAcceptanceSummary(totalConfusion);
        result.profitability = profitability.statistics();
        result.profitabilitySourceContentHash =
            profitability.SourceContentHash();

        if (launchArgs.evalTrading)
            PrintEvalTradingMetrics(profitability.statistics(), totalConfusion);
    }
    else
    {
        result.accuracy = totalActedLog
            ? (static_cast<double>(totalCorrectLog) / static_cast<double>(totalActedLog))
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

        DiagnosticOut() << "Overall direction accuracy (log-return): " << (result.accuracy * 100.0)
                        << "% over " << totalActedLog << " acted (of " << totalWindows << ")"
                        << " coverage=" << overallCovLog << "%" << std::endl;
        DiagnosticOut() << "Overall MAE (relative move fraction): " << overallMaeMove
                        << " | Overall direction accuracy (relative move, thresholded): " << overallAccDir
                        << "% over " << totalActedDir << " acted (of " << totalWindows << ")"
                        << " coverage=" << overallCovDir << "%" << std::endl;
    }

    return result;
}

bool MatrixParamExists(pqxx::work& w, long long modelId, const std::string& paramName)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM matrix WHERE model_id = $1 AND param_name = $2 LIMIT 1;",
        modelId, paramName);
    return !r.empty();
}

bool NearlyEqualDouble(double lhs, double rhs, double tolerance = 1e-7)
{
    return std::fabs(lhs - rhs) <= tolerance;
}

bool InferAllCandidateCompatible(pqxx::work& w,
                                 long long modelId,
                                 const std::string& runtimeSymbol,
                                 EA::LSTM::TargetType requestedTargetType,
                                 const Tensor& tensor,
                                 const TrainConfigMeta* anchorTrainConfig,
                                 InferAllCandidate& candidate,
                                 InferAllSkipDetail& skipDetail)
{
    candidate.legacyMissingSymbol = false;
    candidate.metadataGap = false;

    auto configMismatch = [&](const char* field, const auto& anchorValue, const auto& candidateValue) -> bool
    {
        skipDetail.reason = "CONFIG_MISMATCH";
        skipDetail.field = field;
        skipDetail.anchor = ValueToString(anchorValue);
        skipDetail.candidate = ValueToString(candidateValue);
        return false;
    };

    auto invalidMetadata = [&](const char* field, const std::string& detail) -> bool
    {
        skipDetail.reason = "INVALID_METADATA";
        skipDetail.field = field;
        skipDetail.detail = detail;
        return false;
    };

    if (MatrixParamExists(w, modelId, "train_symbol_meta"))
    {
        try
        {
            const std::string modelSymbol = DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
            PrintDatabaseModelSymbol(modelId, modelSymbol);
            if (modelSymbol != runtimeSymbol)
                return configMismatch("symbol", runtimeSymbol, modelSymbol);
        }
        catch (const std::exception& e)
        {
            return invalidMetadata("train_symbol_meta", e.what());
        }
    }
    else
    {
        PrintMissingModelSymbol(modelId);
        const auto legacySymbol = ResolveLegacySymbolFromModelName(candidate.name, { runtimeSymbol });
        if (!legacySymbol.has_value())
            return configMismatch("symbol", runtimeSymbol, "missing_train_symbol_meta");
        candidate.legacyMissingSymbol = true;
        candidate.metadataGap = true;
        PrintLegacyModelSymbol(modelId, *legacySymbol);
    }

    if (MatrixParamExists(w, modelId, "target_meta"))
    {
        try
        {
            auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "target_meta");
            auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "target_meta");
            if (dims.n_rows != 1 || dims.n_cols != 6 || vals.size() != 6)
            {
                return invalidMetadata("target_meta", "invalid_shape");
            }
            const auto modelTargetType =
                static_cast<EA::LSTM::TargetType>(static_cast<int>(std::llround(vals[0])));
            if (static_cast<int>(modelTargetType) != static_cast<int>(requestedTargetType))
                return configMismatch("target_type", TargetTypeName(requestedTargetType), TargetTypeName(modelTargetType));
        }
        catch (const std::exception& e)
        {
            return invalidMetadata("target_meta", e.what());
        }
    }
    else
        candidate.metadataGap = true;

    if (MatrixParamExists(w, modelId, "model_meta"))
    {
        try
        {
            const auto modelMeta =
                DBIO::PgModelIO::loadRequiredModelMeta(w, modelId);
            const auto inputContract = EA::ResolveModelInputContract(
                modelMeta.inputWidth,
                RuntimeTensorFeatureWidth(tensor));
            candidate.modelInputWidth = inputContract.modelInputWidth;
            if (modelMeta.hiddenSize != static_cast<std::size_t>(hidden_size))
                return configMismatch("hidden_size", hidden_size, modelMeta.hiddenSize);
        }
        catch (const std::exception& e)
        {
            return invalidMetadata("model_meta", e.what());
        }
    }
    else
    {
        // Legacy infer-all candidates may predate model_meta.  Preserve the
        // existing metadata-gap inclusion only when the parameter matrix
        // itself yields one of the supported structural widths.
        try
        {
            const auto paramDims =
                DBIO::PgModelIO::loadParameterDims(w, modelId, "param");
            if (paramDims.n_rows <= 0 || paramDims.n_cols <= 0 ||
                paramDims.n_cols % 4 != 0 ||
                paramDims.n_rows <= paramDims.n_cols / 4)
                return invalidMetadata("param", "invalid_lstm_gate_matrix_shape");
            candidate.modelInputWidth = static_cast<std::size_t>(
                paramDims.n_rows - paramDims.n_cols / 4);
            (void)EA::ResolveModelInputContract(
                candidate.modelInputWidth,
                RuntimeTensorFeatureWidth(tensor));
        }
        catch (const std::exception& e)
        {
            return invalidMetadata("model_meta", e.what());
        }
        candidate.metadataGap = true;
    }

    try
    {
        const Donchian20Mode modelMode =
            DBIO::PgModelIO::loadDonchian20ModeMeta(w, modelId);
        if (modelMode != tensor.GetDonchian20Mode())
            return configMismatch("donchian20_mode",
                                  Donchian20ModeText(tensor.GetDonchian20Mode()),
                                  Donchian20ModeText(modelMode));
    }
    catch (const std::exception& e)
    {
        return invalidMetadata("donchian20_mode_meta", e.what());
    }

    if (MatrixParamExists(w, modelId, "train_config_meta"))
    {
        try
        {
            auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "train_config_meta");
            auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "train_config_meta");
            if (dims.n_rows != 1 ||
                dims.n_cols < DBIO::PgModelIO::kTrainConfigMetaFieldCount ||
                vals.size() < static_cast<size_t>(DBIO::PgModelIO::kTrainConfigMetaFieldCount))
            {
                return invalidMetadata("train_config_meta", "invalid_shape");
            }

            auto checkInt = [&](const char* field, double modelValue, long long runtimeValue) -> bool
            {
                const long long candidateValue = static_cast<long long>(std::llround(modelValue));
                if (candidateValue != runtimeValue)
                    return configMismatch(field, runtimeValue, candidateValue);
                return true;
            };
            auto checkFloat = [&](const char* field, double modelValue, double runtimeValue) -> bool
            {
                if (!NearlyEqualDouble(modelValue, runtimeValue))
                    return configMismatch(field, runtimeValue, modelValue);
                return true;
            };

            const double anchorClassWeightDown = anchorTrainConfig ? anchorTrainConfig->classWeightDown : kClassWeightDown;
            const double anchorClassWeightNeutral = anchorTrainConfig ? anchorTrainConfig->classWeightNeutral : kClassWeightNeutral;
            const double anchorClassWeightUp = anchorTrainConfig ? anchorTrainConfig->classWeightUp : kClassWeightUp;
            const double anchorCoreLrMult = (anchorTrainConfig && anchorTrainConfig->coreLrMult.has_value())
                ? *anchorTrainConfig->coreLrMult
                : EA::LSTM::CoreLrMultForTarget(requestedTargetType);
            const double anchorHeadWeightLrMult = (anchorTrainConfig && anchorTrainConfig->headWeightLrMult.has_value())
                ? *anchorTrainConfig->headWeightLrMult
                : head_weight_lr_mult;
            const double anchorHeadBiasLrMult = (anchorTrainConfig && anchorTrainConfig->headBiasLrMult.has_value())
                ? *anchorTrainConfig->headBiasLrMult
                : head_bias_lr_mult;

            if (!checkInt("schema_version", vals[0], DBIO::PgModelIO::kTrainConfigMetaSchemaVersion) ||
                !checkInt("prediction_horizon", vals[1], prediction_horizon) ||
                !checkFloat("threshold_logret", vals[2], c_next_threshold) ||
                !checkInt("window_size", vals[3], window_size) ||
                !checkInt("label_rule_id", vals[4], DirectionLabelRuleId()) ||
                !checkFloat("class_weight_down", vals[5], anchorClassWeightDown) ||
                !checkFloat("class_weight_neutral", vals[6], anchorClassWeightNeutral) ||
                !checkFloat("class_weight_up", vals[7], anchorClassWeightUp))
                return false;

            if (vals.size() >= 9)
            {
                if (!checkInt("num_layers", vals[8], num_layers))
                    return false;
            }
            else
                candidate.metadataGap = true;

            if (vals.size() >= 10)
            {
                if (!checkInt("normalization_version", vals[9], normalization_version))
                    return false;
            }
            else
                candidate.metadataGap = true;

            if (vals.size() >= 11)
                candidate.completedEpochs = static_cast<size_t>(std::llround(vals[10]));
            else
                candidate.metadataGap = true;

            if (vals.size() >= 12)
            {
                if (!checkFloat("core_lr_mult", vals[11], anchorCoreLrMult))
                    return false;
            }
            else
                candidate.metadataGap = true;

            if (vals.size() >= 13)
            {
                if (!checkFloat("head_weight_lr_mult", vals[12], anchorHeadWeightLrMult))
                    return false;
            }
            else
                candidate.metadataGap = true;

            if (vals.size() >= 14)
            {
                if (!checkFloat("head_bias_lr_mult", vals[13], anchorHeadBiasLrMult))
                    return false;
            }
            else
                candidate.metadataGap = true;
        }
        catch (const std::exception& e)
        {
            return invalidMetadata("train_config_meta", e.what());
        }
    }
    else
        candidate.metadataGap = true;

    return true;
}

std::vector<InferAllCandidate> LoadInferAllCandidates(pqxx::work& w,
                                                      const LaunchArgs& launchArgs,
                                                      const std::string& runtimeSymbol,
                                                      EA::LSTM::TargetType requestedTargetType,
                                                      const Tensor& tensor,
                                                      const TrainConfigMeta* anchorTrainConfig,
                                                      size_t& skippedDueToResume,
                                                      size_t& skippedIncompatible,
                                                      std::vector<std::string>& skippedModelLogs)
{
    skippedDueToResume = 0;
    skippedIncompatible = 0;
    const long long startAfter = launchArgs.modelId.value_or(launchArgs.inferStartAfterModelId.value_or(0));
    if (startAfter > 0)
    {
        pqxx::result skipped = w.exec_params(
            "SELECT count(*) FROM model WHERE model_id <= $1;",
            startAfter);
        if (!skipped.empty())
            skippedDueToResume = skipped[0][0].as<size_t>();
    }

    pqxx::result rows = w.exec_params(
        "SELECT model_id, COALESCE(name, '') FROM model WHERE model_id > $1 ORDER BY model_id ASC;",
        startAfter);

    std::vector<InferAllCandidate> candidates;
    candidates.reserve(rows.size());
    for (const auto& row : rows)
    {
        InferAllCandidate candidate;
        candidate.modelId = row[0].as<long long>();
        candidate.name = row[1].as<std::string>();
        InferAllSkipDetail skipDetail;
        if (InferAllCandidateCompatible(w,
                                        candidate.modelId,
                                        runtimeSymbol,
                                        requestedTargetType,
                                        tensor,
                                        anchorTrainConfig,
                                        candidate,
                                        skipDetail))
        {
            candidates.push_back(candidate);
        }
        else
        {
            ++skippedIncompatible;
            std::ostringstream oss;
            oss << "INFER_ALL_SKIP_MODEL"
                << ",model_id=" << candidate.modelId
                << ",reason=" << skipDetail.reason;
            if (!skipDetail.field.empty())
                oss << ",field=" << skipDetail.field;
            if (!skipDetail.anchor.empty())
                oss << ",anchor=" << skipDetail.anchor;
            if (!skipDetail.candidate.empty())
                oss << ",candidate=" << skipDetail.candidate;
            if (!skipDetail.detail.empty())
                oss << ",detail=" << skipDetail.detail;
            skippedModelLogs.push_back(oss.str());
        }
    }
    return candidates;
}

InferAllSummaryRow RunInferAllModel(pqxx::work& w,
                                    const LaunchArgs& launchArgs,
                                    const InferAllCandidate& candidate,
                                    const std::string& rawPriceTableName,
                                    const std::string& fromDate,
                                    const std::string& toDate,
                                    const Tensor& tensor,
                                    size_t logicalOutputStartIndex,
                                    EA::LSTM::TargetType requestedTargetType)
{
    if (LogSummary())
        std::cout << "INFER_ALL_MODEL_BEGIN"
                  << " model_id=" << candidate.modelId
                  << " name=" << candidate.name
                  << std::endl;
    if (candidate.legacyMissingSymbol)
        DiagnosticOut() << "INFER_ALL_MODEL_WARN"
                        << " model_id=" << candidate.modelId
                        << " name=" << candidate.name
                        << " reason=missing_legacy_train_symbol_meta_included"
                        << std::endl;
    if (candidate.metadataGap)
        DiagnosticOut() << "INFER_ALL_MODEL_WARN"
                        << " model_id=" << candidate.modelId
                        << " name=" << candidate.name
                        << " reason=metadata_gap_included"
                        << std::endl;

    EA::LSTM lstm = CreateLstmForRuntimeLogLevel(
        tensor, 1, 0, requestedTargetType, candidate.modelInputWidth,
        LoadModelFeatureAblationMask(w, candidate.modelId));
    PrintRuntimeLrConfig(lstm);
    DiagnosticOut() << "DIAG_LSTM_BINDING"
                    << ",table=" << rawPriceTableName
                    << ",tensor_rows=" << tensor.RowCount()
                    << ",tensor_addr=" << static_cast<const void*>(&tensor)
                    << ",lstm_tensor_ref_addr=" << static_cast<const void*>(lstm.BoundTensorAddress())
                    << ",newly_constructed=1"
                    << ",reused=0"
                    << std::endl;

    {
        ScopedDiagnosticCoutSilencer silence;
        DBIO::PgModelIO::loadAll(w, candidate.modelId, lstm);
    }
    DiagnosticOut() << "Loaded model_id=" << candidate.modelId
                    << " source=--infer-all"
                    << std::endl;

    const InferenceEvaluationResult evaluation =
        RunInferenceEvaluation(w,
                               launchArgs,
                               lstm,
                               tensor,
                               std::optional<long long>{candidate.modelId},
                               "--infer-all",
                               requestedTargetType,
                               rawPriceTableName,
                               fromDate,
                               toDate,
                               logicalOutputStartIndex,
                               true);

    InferAllSummaryRow row;
    row.modelId = candidate.modelId;
    row.name = candidate.name;
    row.completedEpochs = evaluation.completedEpochs.has_value()
        ? evaluation.completedEpochs
        : candidate.completedEpochs;
    row.accuracy = evaluation.accuracy;
    row.acceptance = evaluation.acceptance;
    row.profitability = evaluation.profitability;
    row.profitabilitySourceContentHash =
        evaluation.profitabilitySourceContentHash;

    if (LogSummary())
        std::cout << "INFER_ALL_MODEL_DONE"
                  << " model_id=" << row.modelId
                  << " accuracy=" << row.accuracy
                  << " accept_model=" << (row.acceptance.acceptModel ? "true" : "false")
                  << " reject_reason=" << row.acceptance.rejectReason
                  << std::endl;
    return row;
}

int RunInferAllForSymbol(pqxx::work& w,
                         const LaunchArgs& launchArgs,
                         const std::string& rawPriceTableName,
                         const std::string& fromDate,
                         const std::string& toDate,
                         const Tensor& tensor,
                         size_t logicalOutputStartIndex,
                         EA::LSTM::TargetType requestedTargetType,
                         const std::optional<PersistedInferenceConfig>& inferenceConfig)
{
    size_t skippedDueToResume = 0;
    size_t skippedIncompatible = 0;
    std::vector<std::string> skippedModelLogs;
    const long long startAfter = launchArgs.modelId.value_or(launchArgs.inferStartAfterModelId.value_or(0));
    if (!RequireInferenceEvalResultTable(w))
        return 1;
    std::vector<InferAllCandidate> candidates =
        LoadInferAllCandidates(w,
                               launchArgs,
                               rawPriceTableName,
                               requestedTargetType,
                               tensor,
                               inferenceConfig.has_value() ? &inferenceConfig->trainConfig : nullptr,
                               skippedDueToResume,
                               skippedIncompatible,
                               skippedModelLogs);

    if (startAfter > 0)
    {
        DiagnosticOut() << "INFER_ALL_RESUME"
                        << " start_after_model_id=" << startAfter
                        << " skipped_due_to_resume=" << skippedDueToResume
                        << std::endl;
    }

    if (LogSummary())
    {
        std::cout << "INFER_ALL_BEGIN"
                  << " symbol=" << rawPriceTableName
                  << " model_count=" << candidates.size()
                  << " start_after_model_id=";
        if (startAfter > 0)
            std::cout << startAfter;
        else
            std::cout << "none";
        std::cout << std::endl;
    }

    for (const auto& skippedModelLog : skippedModelLogs)
        DiagnosticOut() << skippedModelLog << std::endl;

    if (candidates.empty())
    {
        if (LogSummary())
            std::cout << "INFER_ALL_NO_MODELS"
                      << " symbol=" << rawPriceTableName
                      << std::endl;
        std::cout << "INFER_ALL_DONE"
                  << " evaluated=0"
                  << " skipped_existing=0"
                  << " skipped_config_mismatch=" << skippedIncompatible
                  << " skipped_failed=0"
                  << " skipped_resume=" << skippedDueToResume
                  << " best_model_id=-1"
                  << " best_accuracy=0"
                  << std::endl;
        std::cout << "INFER_ALL_SUMMARY" << std::endl;
        std::cout << "model_id,name,completed_epochs,accuracy,accept_model,reject_reason,pred_down,pred_neutral,pred_up" << std::endl;
        DiagnosticOut() << "runtime_infer=true; skipping model save" << std::endl;
        w.commit();
        return 0;
    }

    std::vector<InferAllSummaryRow> summaries;
    summaries.reserve(candidates.size());
    size_t skippedDuringEvaluation = 0;
    size_t skippedExisting = 0;
    size_t newlyEvaluated = 0;
    for (const auto& candidate : candidates)
    {
        const InferenceIdentity identity =
            BuildInferenceIdentity(candidate.modelId,
                                   rawPriceTableName,
                                   requestedTargetType,
                                   fromDate,
                                   toDate);
        if (!launchArgs.forceInfer)
        {
            const auto existing = LoadCompletedInferenceResult(w, identity, candidate);
            if (existing.has_value())
            {
                ++skippedExisting;
                if (LogDiagnostic())
                {
                    std::cout << "INFER_ALL_SKIP_EXISTING"
                              << " model_id=" << candidate.modelId
                              << " completed_epochs=";
                    if (existing->completedEpochs.has_value())
                        std::cout << *existing->completedEpochs;
                    else
                        std::cout << "missing";
                    std::cout << " from=" << fromDate
                              << " to=" << toDate
                              << std::endl;
                    std::cout << "INFER_ALL_MODEL_EXISTING"
                              << " model_id=" << existing->modelId
                              << " accuracy=" << existing->accuracy
                              << " accept_model=" << (existing->acceptance.acceptModel ? "true" : "false")
                              << " reject_reason=" << existing->acceptance.rejectReason
                              << std::endl;
                }
                summaries.push_back(*existing);
                continue;
            }
        }

        try
        {
            InferAllSummaryRow row = RunInferAllModel(w,
                                                      launchArgs,
                                                      candidate,
                                                      rawPriceTableName,
                                                      fromDate,
                                                      toDate,
                                                      tensor,
                                                      logicalOutputStartIndex,
                                                      requestedTargetType);
            const long long inferenceEvalResultId =
                PersistCompletedInferenceResult(w, identity, row);
            PersistInferenceProfitabilityObservation(
                w,
                inferenceEvalResultId,
                identity,
                row,
                EA::InferenceProfitability::Scope::finalInference,
                std::nullopt,
                std::nullopt);
            summaries.push_back(row);
            ++newlyEvaluated;
        }
        catch (const std::exception& e)
        {
            ++skippedDuringEvaluation;
            try
            {
                PersistFailedInferenceResult(w, identity, candidate, e.what());
            }
            catch (const std::exception& persistError)
            {
                DiagnosticOut() << "INFER_ALL_RESULT_WRITE_FAILED"
                                << ",model_id=" << candidate.modelId
                                << ",status=failed"
                                << ",error=" << persistError.what()
                                << std::endl;
            }
            DiagnosticOut() << "INFER_ALL_SKIP_MODEL"
                            << ",model_id=" << candidate.modelId
                            << ",reason=EVALUATION_FAILED"
                            << ",detail=" << e.what()
                            << std::endl;
        }
    }

    long long bestModelId = -1;
    double bestAccuracy = -1.0;
    for (const auto& row : summaries)
    {
        if (row.accuracy > bestAccuracy)
        {
            bestAccuracy = row.accuracy;
            bestModelId = row.modelId;
        }
    }
    if (bestAccuracy < 0.0)
        bestAccuracy = 0.0;

    std::cout << "INFER_ALL_DONE"
              << " evaluated=" << newlyEvaluated
              << " skipped_existing=" << skippedExisting
              << " skipped_config_mismatch=" << skippedIncompatible
              << " skipped_failed=" << skippedDuringEvaluation
              << " skipped_resume=" << skippedDueToResume
              << " best_model_id=" << bestModelId
              << " best_accuracy=" << bestAccuracy
              << std::endl;

    std::cout << "INFER_ALL_SUMMARY" << std::endl;
    std::cout << "model_id,name,completed_epochs,accuracy,accept_model,reject_reason,pred_down,pred_neutral,pred_up" << std::endl;
    for (const auto& row : summaries)
    {
        std::cout << row.modelId
                  << "," << row.name
                  << ",";
        if (row.completedEpochs.has_value())
            std::cout << *row.completedEpochs;
        else
            std::cout << "missing";
        std::cout << "," << row.accuracy
                  << "," << (row.acceptance.acceptModel ? "true" : "false")
                  << "," << row.acceptance.rejectReason
                  << "," << row.acceptance.predFrac[0]
                  << "," << row.acceptance.predFrac[1]
                  << "," << row.acceptance.predFrac[2]
                  << std::endl;
    }

    DiagnosticOut() << "runtime_infer=true; skipping model save" << std::endl;
    w.commit();
    return 0;
}
}

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled()
{
    return LogDiagnostic();
}

std::optional<int> RunCheckpointStopOwnershipTestBoundary(
    const LaunchArgs& launchArgs)
{
    const char* enabled =
        std::getenv("EA_SCHEDULER_OWNERSHIP_TEST_ENABLE");
    const char* boundary =
        std::getenv("EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY");
    const char* database = std::getenv("LSTM_DB_NAME");
    if (!enabled || std::string{enabled} != "1" ||
        !boundary ||
        std::string{boundary} !=
            "checkpoint_stop_after_transition_before_exit" ||
        !database ||
        std::string{database}.rfind(
            "ea_scheduler_process_test_", 0) != 0)
    {
        return std::nullopt;
    }
    if (!launchArgs.schedulerExperimentId ||
        !launchArgs.schedulerWorkerAttemptId ||
        !launchArgs.inferenceMode)
    {
        return 91;
    }

    if (*launchArgs.inferenceMode)
    {
        std::cout << "SCHEDULER_TEST_CHECKPOINT_NEXT_PHASE_HELD"
                  << ",experiment_id="
                  << *launchArgs.schedulerExperimentId
                  << ",worker_attempt_id="
                  << *launchArgs.schedulerWorkerAttemptId
                  << ",phase=infer"
                  << std::endl;
        std::cout.flush();
        if (::kill(::getpid(), SIGSTOP) != 0)
            return 92;
        return 0;
    }

    const char* modelText =
        std::getenv(
            "EA_SCHEDULER_OWNERSHIP_TEST_CHECKPOINT_MODEL_ID");
    if (!modelText || !*modelText ||
        !launchArgs.checkpointEvery ||
        !launchArgs.epochs)
    {
        return 93;
    }
    const long long modelId = ParseModelIdArg(modelText);
    const int checkpointEpoch = *launchArgs.checkpointEvery;
    const std::optional<CheckpointStopConfig> stop =
        LoadCheckpointStopConfig(
            launchArgs.schedulerExperimentId,
            checkpointEpoch,
            *launchArgs.epochs,
            launchArgs.checkpointEvery);
    if (!stop ||
        !RecordCheckpointStopReached(
            launchArgs.schedulerExperimentId,
            launchArgs.schedulerWorkerAttemptId,
            checkpointEpoch,
            modelId))
    {
        return 94;
    }
    std::cout << "SCHEDULER_TEST_CHECKPOINT_TRAIN_ATTEMPT_TERMINAL"
              << ",experiment_id="
              << *launchArgs.schedulerExperimentId
              << ",worker_attempt_id="
              << *launchArgs.schedulerWorkerAttemptId
              << ",checkpoint_epoch=" << checkpointEpoch
              << ",checkpoint_model_id=" << modelId
              << std::endl;
    std::cout.flush();
    if (::kill(::getpid(), SIGSTOP) != 0)
        return 95;
    return 0;
}

int main(int argc, const char * argv[])
{
    if (argc >= 4 && std::string(argv[1]) == "--baseline-3class")
        return RunBaseline3Class(argv[2], argv[3]);
    if (argc >= 4 && std::string(argv[1]) == "--label-grid-3class")
        return RunLabelGridDiagnostic3Class(argv[2], argv[3]);
    if (argc >= 4 && std::string(argv[1]) == "--feature-trainability-3class")
        return RunFeatureTrainability3Class(argv[2], argv[3]);
    if (EA::ExperimentMetaAnalyzer::IsMetaAnalysisCommand(argc, argv))
        return EA::ExperimentMetaAnalyzer::RunMetaAnalysisCli(argc, argv);
    if (EA::EconomicCalendar::IsEconomicEventImportCommand(argc, argv))
        return EA::EconomicCalendar::RunEconomicEventImportCli(argc, argv);
    if (EA::EconomicCalendar::IsEconomicEventConsensusImportCommand(argc, argv))
        return EA::EconomicCalendar::RunEconomicEventConsensusImportCli(
            argc, argv);
    if (EA::ExperimentScheduler::IsExperimentSchedulerCommand(argc, argv))
        return EA::ExperimentScheduler::RunExperimentSchedulerCli(argc, argv);

    LaunchArgs launchArgs;
    try
    {
        launchArgs = ParseLaunchArgs(argc, argv);
        if ((launchArgs.schedulerExperimentId.has_value() ||
             launchArgs.schedulerCheckpointEvalId.has_value()) &&
            !launchArgs.schedulerWorkerAttemptId.has_value())
        {
            throw std::invalid_argument(
                "direct CLI execution of scheduler-managed work is "
                "prohibited; an exact --scheduler-worker-attempt-id "
                "is required");
        }
        if (launchArgs.schedulerWorkerAttemptId.has_value() &&
            !EA::ExperimentScheduler::RegisterSchedulerWorkerAttempt(
                *launchArgs.schedulerWorkerAttemptId,
                launchArgs.schedulerExperimentId,
                launchArgs.schedulerCheckpointEvalId,
                launchArgs.schedulerCheckpointEvalId.has_value()
                    ? "checkpoint_infer"
                    : "experiment",
                launchArgs.schedulerCheckpointEvalId.has_value()
                    ? "infer"
                    : ((launchArgs.inferenceMode.has_value() &&
                        *launchArgs.inferenceMode)
                           ? "infer"
                           : "train")))
        {
            return 125;
        }
        if (const std::optional<int> testBoundary =
                RunCheckpointStopOwnershipTestBoundary(launchArgs))
        {
            return *testBoundary;
        }
        if (launchArgs.logLevel.has_value())
            gRuntimeLogLevel = *launchArgs.logLevel;
        if (launchArgs.schedulerCheckpointEvalId.has_value())
        {
            EA::ExperimentScheduler::LogWorkerStarted(
                "CHECKPOINT_INFER_WORKER_STARTED",
                "checkpoint_infer",
                std::nullopt,
                launchArgs.modelId,
                launchArgs.schedulerCheckpointEvalId);
        }
        else if (launchArgs.schedulerExperimentId.has_value())
        {
            const bool inferenceWorker =
                launchArgs.inferenceMode.has_value() &&
                *launchArgs.inferenceMode;
            EA::ExperimentScheduler::LogWorkerStarted(
                inferenceWorker
                    ? "INFERENCE_WORKER_STARTED"
                    : "TRAINING_WORKER_STARTED",
                inferenceWorker ? "infer" : "train",
                launchArgs.schedulerExperimentId,
                inferenceWorker ? launchArgs.modelId : launchArgs.resumeModelId);
        }
        if (!ValidateResumeLaunchArgs(launchArgs))
            return 1;
        if (!launchArgs.resumeModelId.has_value())
            ApplyLaunchRuntimeConfig(launchArgs);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n"
                  << "Usage: " << argv[0] << " [--train|--infer] [--infer-all] [--force-infer] [--donchian20-mode=enabled|zero_ablation] [--infer-start-after-model-id <model_id>] [--eval-trading] [--log-level quiet|summary|diagnostic] [--lstm-profile-hotspots] [--lstm-profile-output=<path>] [--resume-model-id=<model_id>] [--resume-expand-input-width] [--target-epochs=<absolute_final_epoch>] [--new-model-name=<name>] [--checkpoint-every <N>] [--symbol=<table_name>] [--model=<model_id>] [--prediction-horizon=<int>] [--threshold=<double>] [--window-size=<int>] [--hidden-size=<int>] [--num-layers=<int>] [--epochs=<int>] [--core-lr-mult=<float>] [--head-weight-lr-mult=<float>] [--head-bias-lr-mult=<float>] [--training-objective=<scheduler-persisted-objective>] <fromDate> <toDate>\n"
                  << "Preferred inference: " << argv[0] << " --infer --model=<model_id> <fromDate> <toDate>\n"
                  << "Preferred infer-all: " << argv[0] << " --infer --infer-all --model=<anchor_model_id> <fromDate> <toDate>\n";
        return 1;
    }

    EA::LSTM::ConfigureHotspotProfiler(launchArgs.lstmProfileHotspots,
                                       launchArgs.lstmProfileOutputPath);
    LSTMHotspotProfileFinalizer hotspotProfileFinalizer{
        launchArgs.lstmProfileHotspots,
        launchArgs.lstmProfileOutputPath
    };

    pqxx::connection c_forex { ForexDbConnectionString() }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    pqxx::connection c_LSTM { LstmDbConnectionString() }; // "user = postgres password=pass123 hostaddr=127.0.0.1 port=5432." };
    std::string fromDate  { launchArgs.fromDate }, toDate { launchArgs.toDate };
    std::optional<EA::ProfitabilityVerification::
        CampaignProfitabilityOutcomeJob> frozenOutcomeJob;
    if (launchArgs.frozenOutcome)
    {
        try
        {
            pqxx::read_transaction frozenRead{c_LSTM};
            frozenRead.exec(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
            const auto& spec = *launchArgs.frozenOutcome;
            frozenOutcomeJob = EA::ProfitabilityVerification::
                LoadCampaignProfitabilityOutcomeExecutionJob(
                    frozenRead, spec.cohortHash, spec.sourceExperimentId,
                    spec.sourceModelId, spec.outcomeStart, spec.outcomeEnd,
                    spec.jobHash, CurrentUtcDate());
            std::cout
                << "CAMPAIGN_PROFITABILITY_OUTCOME_EXECUTION_START"
                << ",validation_cohort_identity_hash="
                << frozenOutcomeJob->validationCohortIdentityHash
                << ",ranking_snapshot_id=" << frozenOutcomeJob->rankingSnapshotId
                << ",source_evaluation_run_id="
                << frozenOutcomeJob->sourceEvaluationRunId
                << ",source_experiment_id="
                << frozenOutcomeJob->sourceExperimentId
                << ",source_model_id=" << frozenOutcomeJob->sourceModelId
                << ",symbol=" << frozenOutcomeJob->symbol
                << ",horizon=" << frozenOutcomeJob->horizon
                << ",outcome_start=" << frozenOutcomeJob->outcomeStart
                << ",outcome_end=" << frozenOutcomeJob->outcomeEnd
                << ",metric_hash=" << frozenOutcomeJob->metricDefinitionHash
                << ",source_content_hash=PENDING,outcome_identity_hash=PENDING"
                << ",job_hash=" << frozenOutcomeJob->hash
                << ",inference_scope=prospective_outcome"
                << ",activation=false,live_profitability_weight=0"
                << ",production_ranking_modified=false"
                << ",recommendation_modified=false"
                << ",ranking_snapshot_modified=false,training_started=false"
                << ",experiment_created=false,experiment_queued=false"
                << ",scheduler_modified=false" << std::endl;
        }
        catch (const std::exception& error)
        {
            std::cerr << "CAMPAIGN_PROFITABILITY_OUTCOME_EXECUTION_REJECTED"
                      << ",error=" << error.what() << std::endl;
            return 1;
        }
    }
    EA::TrainingObjective::Configuration runtimeTrainingObjective =
        EA::TrainingObjective::Legacy();
    if (!gRuntimeInferenceMode && launchArgs.schedulerExperimentId.has_value())
    {
        try
        {
            pqxx::work objectiveRead { c_LSTM };
            objectiveRead.exec("SET TRANSACTION READ ONLY;");
            runtimeTrainingObjective = LoadExperimentTrainingObjective(
                objectiveRead, *launchArgs.schedulerExperimentId);
            objectiveRead.commit();
            (void)EA::TrainingObjective::ParseSupportedCanonicalText(
                EA::TrainingObjective::CanonicalText(
                    runtimeTrainingObjective));
            if (launchArgs.trainingObjective.has_value() &&
                !EA::TrainingObjective::ResumeCompatible(
                    runtimeTrainingObjective,
                    *launchArgs.trainingObjective))
            {
                throw std::invalid_argument(
                    "scheduler_child_training_objective_mismatch");
            }
            std::cout << "TRAINING_OBJECTIVE_ACTIVE"
                      << ",experiment_id="
                      << *launchArgs.schedulerExperimentId
                      << ",objective_id="
                      << runtimeTrainingObjective.objectiveIdentifier
                      << ",objective_hash="
                      << EA::TrainingObjective::Identity(
                             runtimeTrainingObjective)
                      << ",auxiliary_enabled="
                      << (EA::TrainingObjective::AuxiliaryEnabled(
                              runtimeTrainingObjective)
                              ? "1"
                              : "0")
                      << std::endl;
        }
        catch (const std::exception& error)
        {
            std::cerr << "TRAINING_OBJECTIVE_RESOLVE_FAILED"
                      << ",experiment_id="
                      << *launchArgs.schedulerExperimentId
                      << ",error=" << error.what() << std::endl;
            return 1;
        }
    }
    EA::FeatureWarmupScope featureWarmupScope =
        launchArgs.featureWarmupScope.value_or(EA::kDefaultFeatureWarmupScope);
    std::optional<ResumeCheckpointConfig> resumeConfig;
    std::optional<SchedulerInferencePersistenceContext> schedulerInferenceContext;

    if (launchArgs.resumeModelId.has_value())
    {
        try
        {
            pqxx::work resumeRead { c_LSTM };
            resumeRead.exec("SET TRANSACTION READ ONLY;");
            resumeConfig = LoadResumeCheckpointConfig(
                resumeRead, *launchArgs.resumeModelId,
                runtimeTrainingObjective);
            ConfigureInputWidthExpansionForResume(
                resumeRead, *resumeConfig,
                launchArgs.resumeExpandInputWidth,
                launchArgs.schedulerExperimentId);
            resumeRead.commit();
            if (launchArgs.featureWarmupScope.has_value() &&
                *launchArgs.featureWarmupScope != resumeConfig->featureWarmupScope)
                throw std::runtime_error("feature warmup scope mismatch: persisted/runtime");
            featureWarmupScope = resumeConfig->featureWarmupScope;
            ApplyResumeRuntimeConfig(*resumeConfig, *launchArgs.targetEpochs);
            fromDate = resumeConfig->fromDate;
            toDate = resumeConfig->toDate;
            std::cout << "RESUME_LOAD_MODEL_ID=" << *launchArgs.resumeModelId << std::endl;
            std::cout << "RESUME_COMPLETED_EPOCH=" << resumeConfig->completedEpoch << std::endl;
            std::cout << "RESUME_TARGET_EPOCH=" << *launchArgs.targetEpochs << std::endl;
            std::cout << "RESUME_EPOCHS_TO_RUN="
                      << (static_cast<size_t>(*launchArgs.targetEpochs) - resumeConfig->completedEpoch)
                      << std::endl;
            std::cout << "RESUME_USING_DB_CONFIG_ONLY=1" << std::endl;
            if (resumeConfig->expandInputWidthRequested)
            {
                std::cout << "RESUME_INPUT_WIDTH_EXPANSION"
                          << ",source_model_id="
                          << resumeConfig->inputWidthExpansionProvenance->sourceModelId
                          << ",source_n_in="
                          << resumeConfig->inputWidthExpansionProvenance->sourceInputWidth
                          << ",expanded_n_in="
                          << resumeConfig->inputWidthExpansionProvenance->expandedInputWidth
                          << ",initialization="
                          << resumeConfig->inputWidthExpansionProvenance->initializationPolicy
                          << ",parameter_expansion_required="
                          << (resumeConfig->parameterExpansionRequired ? 1 : 0)
                          << std::endl;
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "RESUME_CONFIG_LOAD_FAILED"
                      << ",model_id=" << *launchArgs.resumeModelId
                      << ",error=" << e.what()
                      << std::endl;
            return 1;
        }
    }
    try
    {
        std::vector<std::string> availableSymbols;
        {
            pqxx::work forexMetadataRead { c_forex };
            forexMetadataRead.exec("SET TRANSACTION READ ONLY;");
            pqxx::result tables = forexMetadataRead.exec("select table_name from information_schema.tables where table_schema = 'public' and table_name like '%rmp' order by table_name;");
            availableSymbols.reserve(tables.size());
            for (auto tbl : tables)
                availableSymbols.emplace_back(EA::CanonicalSymbol::Normalize(tbl[0].c_str()));
            forexMetadataRead.commit();
        }

        pqxx::work configurationRead { c_LSTM };
        configurationRead.exec("SET TRANSACTION READ ONLY;");
        std::optional<PersistedInferenceConfig> inferenceConfig;
        if (!resumeConfig.has_value() && gRuntimeInferenceMode)
        {
            const auto anchorModelId = InferenceAnchorModelId(
                configurationRead, launchArgs);
            if (anchorModelId.has_value())
            {
                try
                {
                    inferenceConfig = LoadPersistedInferenceConfig(configurationRead,
                                                                   *anchorModelId,
                                                                   launchArgs,
                                                                   availableSymbols);
                    ApplyPersistedInferenceRuntimeConfig(*inferenceConfig);
                    featureWarmupScope = inferenceConfig->featureWarmupScope;
                    PrintResolvedInferenceConfig(*inferenceConfig);
                }
                catch (const std::exception& e)
                {
                    std::cerr << "INFERENCE_CONFIG_RESOLVE_FAILED"
                              << ",model_id=" << *anchorModelId
                              << ",error=" << e.what()
                              << std::endl;
                    return 1;
                }
            }
            else if (launchArgs.symbol.has_value())
            {
                DiagnosticOut() << "INFER_SYMBOL_RESOLVED"
                                << ",source=cli"
                                << ",symbol=" << *launchArgs.symbol
                                << std::endl;
            }
        }

        if (gRuntimeInferenceMode &&
            (launchArgs.schedulerExperimentId.has_value() ||
             launchArgs.schedulerCheckpointEvalId.has_value()))
        {
            if (!inferenceConfig.has_value())
                throw std::runtime_error("scheduler inference could not resolve persisted model configuration");
            schedulerInferenceContext = ResolveSchedulerInferencePersistenceContext(
                configurationRead,
                launchArgs,
                inferenceConfig->symbol,
                fromDate,
                toDate);
        }

        Donchian20Mode runtimeDonchian20Mode = kDefaultDonchian20Mode;
        std::size_t runtimeDonchianLookback = kDefaultDonchianLookback;
        if (resumeConfig.has_value())
        {
            runtimeDonchian20Mode = resumeConfig->donchian20Mode;
            runtimeDonchianLookback = resumeConfig->donchianLookback;
        }
        else if (inferenceConfig.has_value())
        {
            runtimeDonchian20Mode = inferenceConfig->donchian20Mode;
            runtimeDonchianLookback = inferenceConfig->donchianLookback;
        }
        else if (gRuntimeInferenceMode && launchArgs.modelId.has_value())
        {
            runtimeDonchian20Mode = DBIO::PgModelIO::loadDonchian20ModeMeta(
                configurationRead, *launchArgs.modelId);
            runtimeDonchianLookback = DBIO::PgModelIO::loadDonchianLookbackMeta(
                configurationRead, *launchArgs.modelId);
            featureWarmupScope = DBIO::PgModelIO::loadFeatureWarmupScopeMeta(
                configurationRead, *launchArgs.modelId);
        }
        else if (launchArgs.donchian20Mode.has_value())
            runtimeDonchian20Mode = *launchArgs.donchian20Mode;
        if (!resumeConfig.has_value() && !inferenceConfig.has_value() &&
            !(gRuntimeInferenceMode && launchArgs.modelId.has_value()) &&
            launchArgs.donchianLookback.has_value())
            runtimeDonchianLookback = *launchArgs.donchianLookback;

        if (launchArgs.donchian20Mode.has_value() &&
            *launchArgs.donchian20Mode != runtimeDonchian20Mode)
            throw std::runtime_error(
                std::string{"Donchian-20 mode mismatch: persisted="} +
                Donchian20ModeText(runtimeDonchian20Mode) + ", runtime=" +
                Donchian20ModeText(*launchArgs.donchian20Mode));
        ValidateSchedulerDonchian20Mode(
            configurationRead, launchArgs, runtimeDonchian20Mode);
        ValidateSchedulerFeatureWarmupScope(
            configurationRead, launchArgs, featureWarmupScope);
        ValidateSchedulerDonchianLookback(
            configurationRead, launchArgs, runtimeDonchianLookback);

        std::optional<EA::FeatureAblationMask> schedulerFeatureAblationMask;
        if (launchArgs.schedulerExperimentId.has_value() ||
            launchArgs.schedulerCheckpointEvalId.has_value())
        {
            schedulerFeatureAblationMask =
                LoadSchedulerFeatureAblationMask(configurationRead, launchArgs);
            if (resumeConfig.has_value())
            {
                ValidateSchedulerResumeFeatureAblationMask(
                    *resumeConfig, *schedulerFeatureAblationMask);
            }
            if (inferenceConfig.has_value())
            {
                ValidateSchedulerModelFeatureAblationMask(
                    inferenceConfig->featureAblationMask,
                    *schedulerFeatureAblationMask,
                    inferenceConfig->modelId);
            }
        }
        configurationRead.commit();

        DiagnosticOut() << "candle_duration=" << static_cast<int>(candle_duration) << '\n';
        DiagnosticOut() << "window_size=" << window_size << '\n';
        DiagnosticOut() << "prediction_horizon=" << prediction_horizon << '\n';
        DiagnosticOut() << "c_next_threshold=" << c_next_threshold << '\n';
        PrintRuntimeConfig();
        DiagnosticOut() << "DIAG_GATESTATE_MODE=" << LSTM_GATESTATE_MODE
                        << " (" << GateStateModeLabel() << ")\n";

        std::vector<std::string> selectedSymbols;
        if (resumeConfig.has_value())
        {
            const auto it = std::find(availableSymbols.begin(),
                                      availableSymbols.end(),
                                      resumeConfig->symbol);
            if (it == availableSymbols.end())
            {
                std::cerr << "Requested resume symbol/table not found: " << resumeConfig->symbol << std::endl;
                std::cerr << "AVAILABLE_SYMBOLS";
                for (const auto& symbol : availableSymbols)
                    std::cerr << "," << symbol;
                std::cerr << std::endl;
                return 1;
            }
            selectedSymbols.push_back(*it);
        }
        else if (inferenceConfig.has_value())
        {
            const auto it = std::find(availableSymbols.begin(),
                                      availableSymbols.end(),
                                      inferenceConfig->symbol);
            if (it == availableSymbols.end())
            {
                std::cerr << "Resolved inference symbol/table not found: " << inferenceConfig->symbol << std::endl;
                std::cerr << "AVAILABLE_SYMBOLS";
                for (const auto& symbol : availableSymbols)
                    std::cerr << "," << symbol;
                std::cerr << std::endl;
                return 1;
            }
            selectedSymbols.push_back(*it);
        }
        else if (launchArgs.symbol.has_value())
        {
            const auto it = std::find(availableSymbols.begin(),
                                      availableSymbols.end(),
                                      *launchArgs.symbol);
            if (it == availableSymbols.end())
            {
                std::cerr << "Requested symbol/table not found: " << *launchArgs.symbol << std::endl;
                std::cerr << "AVAILABLE_SYMBOLS";
                for (const auto& symbol : availableSymbols)
                    std::cerr << "," << symbol;
                std::cerr << std::endl;
                return 1;
            }
            selectedSymbols.push_back(*it);
        }
        else
        {
            selectedSymbols = availableSymbols;
        }

        for (const auto& rawPriceTableName : selectedSymbols)
        {
            pqxx::work forexDataRead { c_forex };
            forexDataRead.exec("SET TRANSACTION READ ONLY;");
            DiagnosticOut() << "SYMBOL_SELECTION"
                            << ",requested=" << (resumeConfig.has_value() ? resumeConfig->symbol : (inferenceConfig.has_value() ? inferenceConfig->symbol : (launchArgs.symbol.has_value() ? *launchArgs.symbol : "none")))
                            << ",selected=" << rawPriceTableName
                            << ",available_count=" << availableSymbols.size()
                            << std::endl;

            const bool fullHistoryWarmup =
                featureWarmupScope == EA::FeatureWarmupScope::FullHistoryWarmup;
            const std::string queryStart = fullHistoryWarmup
                ? EA::kTensorFeatureHistoryQueryStart : fromDate;
            const std::string query =
                "select * from candlestick(" + forexDataRead.quote(rawPriceTableName) +
                ", 15, 'minute', " + forexDataRead.quote(queryStart) + ", " +
                forexDataRead.quote(toDate) + ") order by dt;";
            if (frozenOutcomeJob)
            {
                const std::string coverageSql =
                    "WITH bars AS (SELECT dt FROM candlestick(" +
                    forexDataRead.quote(rawPriceTableName) +
                    ",15,'minute'," + forexDataRead.quote(fromDate) + "," +
                    forexDataRead.quote(toDate) + ")) "
                    "SELECT min(dt)::text,max(dt)::text,count(*),"
                    "COALESCE(min(dt) <= " + forexDataRead.quote(fromDate) +
                    "::timestamp,false),COALESCE(max(dt) >= (" +
                    forexDataRead.quote(toDate) +
                    "::timestamp - interval '15 minutes'),false) FROM bars";
                const pqxx::row coverage =
                    forexDataRead.exec(coverageSql).one_row();
                const long long barCount = coverage[2].as<long long>();
                const bool coversStart = coverage[3].as<bool>();
                const bool coversEnd = coverage[4].as<bool>();
                const long long minimumBars = static_cast<long long>(
                    window_size + prediction_horizon + 1);
                const bool complete = coversStart && coversEnd &&
                    barCount >= minimumBars;
                std::cout
                    << "CAMPAIGN_PROFITABILITY_OUTCOME_JOB_READINESS"
                    << ",validation_cohort_identity_hash="
                    << frozenOutcomeJob->validationCohortIdentityHash
                    << ",ranking_snapshot_id="
                    << frozenOutcomeJob->rankingSnapshotId
                    << ",source_evaluation_run_id="
                    << frozenOutcomeJob->sourceEvaluationRunId
                    << ",source_experiment_id="
                    << frozenOutcomeJob->sourceExperimentId
                    << ",source_model_id=" << frozenOutcomeJob->sourceModelId
                    << ",symbol=" << frozenOutcomeJob->symbol
                    << ",horizon=" << frozenOutcomeJob->horizon
                    << ",outcome_start=" << frozenOutcomeJob->outcomeStart
                    << ",outcome_end=" << frozenOutcomeJob->outcomeEnd
                    << ",metric_hash="
                    << frozenOutcomeJob->metricDefinitionHash
                    << ",source_content_hash=PENDING"
                    << ",outcome_identity_hash=PENDING,job_hash="
                    << frozenOutcomeJob->hash
                    << ",readiness="
                    << (complete ? "ready_to_execute" : "partially_available")
                    << ",market_data_coverage_checked=true"
                    << ",market_data_first="
                    << (coverage[0].is_null() ? "NULL" :
                        coverage[0].as<std::string>())
                    << ",market_data_last="
                    << (coverage[1].is_null() ? "NULL" :
                        coverage[1].as<std::string>())
                    << ",market_bar_count=" << barCount
                    << ",minimum_required_bar_count=" << minimumBars
                    << ",activation=false,live_profitability_weight=0"
                    << ",production_ranking_modified=false"
                    << ",recommendation_modified=false"
                    << ",ranking_snapshot_modified=false"
                    << ",training_started=false,experiment_created=false"
                    << ",experiment_queued=false,scheduler_modified=false"
                    << std::endl;
                if (!complete)
                    throw std::runtime_error(
                        "campaign_profitability_outcome_market_data_incomplete");
            }
            const std::string warmupCountQuery = fullHistoryWarmup
                ? "select count(*) from candlestick(" +
                    forexDataRead.quote(rawPriceTableName) + ", 15, 'minute', " +
                    forexDataRead.quote(EA::kTensorFeatureHistoryQueryStart) + ", " +
                    forexDataRead.quote(toDate) + ") where dt < " +
                    forexDataRead.quote(fromDate) + ";"
                : "SELECT 0;";
            const size_t logicalOutputStartIndex =
                forexDataRead.exec1(warmupCountQuery)[0].as<size_t>();
            std::vector<EA::EconomicCalendar::EconomicEvent> economicEvents;
            {
                pqxx::work economicEventRead { c_LSTM };
                economicEventRead.exec("SET TRANSACTION READ ONLY;");
                economicEvents =
                    EA::EconomicCalendar::LoadEconomicEventsForFeatureRange(
                        economicEventRead,
                        std::string{EA::EconomicCalendar::
                            kEconomicEventFeatureCurrency},
                        queryStart,
                        toDate);
                economicEventRead.commit();
            }
            Tensor t{ rawPriceTableName, runtimeDonchian20Mode,
                      runtimeDonchianLookback, std::move(economicEvents) };
            
            DiagnosticOut() << "Candlestick query: " << query << "\n";
            DiagnosticOut() << "FEATURE_WARMUP_SCOPE"
                            << ",mode=" << EA::FeatureWarmupScopeText(featureWarmupScope)
                            << ",source_start=" << queryStart
                            << ",output_start=" << fromDate
                            << ",output_end=" << toDate
                            << ",warmup_rows=" << logicalOutputStartIndex
                            << std::endl;
            DiagnosticOut() << "Building tensor for table: " << rawPriceTableName << std::endl;
            {
                db_cursor_stream<Feature> cs_cur{
                    forexDataRead,
                    query,
                    rawPriceTableName + "_candlestick_stream"};
                db_input_iterator csb = cs_cur.begin(), cse = cs_cur.end();
                while (csb != cse)
                    t.Add(*csb++);
            }
            forexDataRead.commit();
            if (logicalOutputStartIndex > t.RowCount())
                throw std::runtime_error("feature warmup query returned more rows than the source tensor");
            const std::optional<std::size_t> persistedModelInputWidth =
                resumeConfig.has_value()
                    ? std::optional<std::size_t>{
                          resumeConfig->expandInputWidthRequested
                              ? EA::kCurrentModelInputWidth
                              : static_cast<std::size_t>(
                                    resumeConfig->modelInputWidth)}
                    : (inferenceConfig.has_value()
                           ? std::optional<std::size_t>{static_cast<std::size_t>(inferenceConfig->modelInputWidth)}
                           : std::nullopt);
            if (persistedModelInputWidth.has_value())
            {
                try
                {
                    (void)RuntimeModelInputWidth(t, persistedModelInputWidth);
                }
                catch (const std::exception& error)
                {
                    std::cout << "MODEL_CONFIG_MISMATCH"
                              << ",field=feature_count"
                              << ",model=" << *persistedModelInputWidth
                              << ",runtime=" << RuntimeModelInputWidth(t)
                              << ",diagnostic=" << error.what()
                              << std::endl;
                    return 1;
                }
            }

            pqxx::work runtimeDatabaseWork { c_LSTM };
            runtimeDatabaseWork.exec("SET TRANSACTION READ WRITE;");
  
            const auto requestedTargetType = resumeConfig.has_value()
                ? resumeConfig->targetType
                : (inferenceConfig.has_value()
                   ? inferenceConfig->targetType
                   : EA::LSTM::TargetType::UpNeutralDownReturn);
            if (launchArgs.inferAll)
                return RunInferAllForSymbol(runtimeDatabaseWork,
                                            launchArgs,
                                            rawPriceTableName,
                                            fromDate,
                                            toDate,
                                            t,
                                            logicalOutputStartIndex,
                                            requestedTargetType,
                                            inferenceConfig);
            const EA::FeatureAblationMask runtimeFeatureAblationMask =
                schedulerFeatureAblationMask.has_value()
                    ? *schedulerFeatureAblationMask
                    : (resumeConfig.has_value() ? resumeConfig->featureAblationMask :
                       (inferenceConfig.has_value() ? inferenceConfig->featureAblationMask :
                        EA::FeatureAblationMask{}));
            EA::LSTM l = CreateLstmForRuntimeLogLevel(
                t, 1, 0, requestedTargetType, persistedModelInputWidth,
                runtimeFeatureAblationMask);
            if (!gRuntimeInferenceMode)
                l.SetTrainingObjective(runtimeTrainingObjective);
            PrintRuntimeLrConfig(l);
            static size_t s_lstmBindingDiagCount = 0;
            constexpr size_t kLstmBindingDiagLimit = 50;
            if (s_lstmBindingDiagCount < kLstmBindingDiagLimit)
            {
                DiagnosticOut() << "DIAG_LSTM_BINDING"
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
            const std::string loadSource = resumeConfig.has_value()
                ? "--resume-model-id"
                : (launchArgs.modelId.has_value() ? "--model" : "latest");

            // Load an explicitly requested model, or default to the latest stored model.
            try
            {
                long long modelIdToLoad = -1;
                const bool requestedModel = launchArgs.modelId.has_value();

                if (resumeConfig.has_value()) modelIdToLoad = resumeConfig->sourceModelId;
                else if (requestedModel) modelIdToLoad = *launchArgs.modelId;
                else if (load_latest || gRuntimeInferenceMode)
                {
                    pqxx::result r = runtimeDatabaseWork.exec("SELECT max(model_id) FROM model;");
                    if (!r.empty() && !r[0][0].is_null()) modelIdToLoad = r[0][0].as<long long>();
                    else DiagnosticOut() << "No models found; using default-initialized parameters" << std::endl;
                }
                else DiagnosticOut() << "load_latest=false; using default-initialized parameters" << std::endl;

                if (modelIdToLoad > 0)
                {
                    if (!gRuntimeInferenceMode)
                    {
                        const auto persistedObjective =
                            DBIO::PgModelIO::loadTrainingObjectiveMeta(
                                runtimeDatabaseWork, modelIdToLoad);
                        EA::TrainingObjective::RequireResumeCompatible(
                            persistedObjective, runtimeTrainingObjective);
                    }
                    {
                        ScopedDiagnosticCoutSilencer silence;
                        DBIO::PgModelIO::loadAll(
                            runtimeDatabaseWork, modelIdToLoad, l,
                            resumeConfig.has_value() &&
                                resumeConfig->parameterExpansionRequired);
                    }
                    if (resumeConfig.has_value())
                    {
                        ScopedDiagnosticCoutSilencer silence;
                        DBIO::PgModelIO::loadOptimizerMeta(runtimeDatabaseWork, modelIdToLoad, l);
                        l.completedEpochs = resumeConfig->completedEpoch;
                        std::cout << "RESUME_OPTIMIZER_STATE_RESTORED=1" << std::endl;
                    }
                    const Donchian20Mode modelMode =
                        DBIO::PgModelIO::loadDonchian20ModeMeta(runtimeDatabaseWork, modelIdToLoad);
                    if (modelMode != runtimeDonchian20Mode)
                        throw std::runtime_error(
                            std::string{"model/runtime Donchian-20 mode mismatch: model="} +
                            Donchian20ModeText(modelMode) + ", runtime=" +
                            Donchian20ModeText(runtimeDonchian20Mode));
                    loadedModelId = modelIdToLoad;
                    startedFromScratch = false;
                    DiagnosticOut() << "Loaded model_id=" << *loadedModelId
                                    << " source=" << loadSource
                                    << std::endl;
                }
            }
            catch (const std::exception& e)
            {
                if (std::string{e.what()}.find("Donchian-20 mode mismatch") !=
                    std::string::npos ||
                    std::string{e.what()}.find("model/runtime Donchian-20 mode mismatch") !=
                    std::string::npos)
                {
                    std::cerr << "DONCHIAN20_MODE_COMPATIBILITY_FAILURE"
                              << ",error=" << e.what() << std::endl;
                    return 1;
                }
                if (resumeConfig.has_value())
                {
                    std::cerr << "Resume load model_id=" << resumeConfig->sourceModelId
                              << " failed: " << e.what() << std::endl;
                    return 1;
                }
                if (launchArgs.modelId.has_value())
                {
                    std::cerr << "Load requested model_id=" << *launchArgs.modelId
                              << " failed: " << e.what() << std::endl;
                    return 1;
                }

                DiagnosticOut() << "Load latest failed: " << e.what()
                                << "; using default params" << std::endl;
            }

            if (loadedModelId.has_value())
                ValidateLoadedModelSymbolForSelectedTable(
                    runtimeDatabaseWork, *loadedModelId, rawPriceTableName);

            if (gRuntimeInferenceMode)
            {
                const InferenceEvaluationResult evaluation =
                    RunInferenceEvaluation(runtimeDatabaseWork,
                                           launchArgs,
                                           l,
                                           t,
                                           loadedModelId,
                                           loadSource,
                                           requestedTargetType,
                                           rawPriceTableName,
                                           fromDate,
                                           toDate,
                                           logicalOutputStartIndex,
                                           false);
                if (frozenOutcomeJob)
                {
                    try
                    {
                        if (!loadedModelId ||
                            *loadedModelId != frozenOutcomeJob->sourceModelId)
                            throw std::runtime_error(
                                "frozen_outcome_loaded_model_identity_mismatch");
                        if (!evaluation.profitability)
                            throw std::runtime_error(
                                "frozen_outcome_profitability_statistics_missing");
                        const auto& statistics = *evaluation.profitability;
                        EA::ProfitabilityVerification::
                            CampaignProfitabilityOutcomePersistRequest request;
                        request.job = *frozenOutcomeJob;
                        request.inferenceAccuracy = evaluation.accuracy;
                        request.predictionCount = statistics.predictionCount;
                        request.actionableCount = statistics.actionableCount;
                        request.winningActionableCount =
                            statistics.winningActionableCount;
                        request.losingActionableCount =
                            statistics.losingActionableCount;
                        request.grossPositiveReturn =
                            statistics.grossPositiveTerminalHorizonLogReturnSum;
                        request.grossNegativeReturn =
                            statistics.grossNegativeTerminalHorizonLogReturnSum;
                        request.aggregateReturn =
                            statistics.aggregateTerminalHorizonLogReturnSum;
                        request.averageReturn = statistics.
                            AverageTerminalHorizonLogReturnPerActionablePrediction();
                        request.sourceContentHash =
                            evaluation.profitabilitySourceContentHash;
                        const auto persisted = EA::ProfitabilityVerification::
                            PersistCampaignProfitabilityOutcomeIdempotently(
                                runtimeDatabaseWork, request);
                        const std::string identityPrefix =
                            ",validation_cohort_identity_hash=" +
                            frozenOutcomeJob->validationCohortIdentityHash +
                            ",ranking_snapshot_id=" +
                            std::to_string(frozenOutcomeJob->rankingSnapshotId) +
                            ",source_evaluation_run_id=" +
                            std::to_string(frozenOutcomeJob->sourceEvaluationRunId) +
                            ",source_experiment_id=" +
                            std::to_string(frozenOutcomeJob->sourceExperimentId) +
                            ",source_model_id=" +
                            std::to_string(frozenOutcomeJob->sourceModelId) +
                            ",symbol=" + frozenOutcomeJob->symbol +
                            ",horizon=" +
                            std::to_string(frozenOutcomeJob->horizon) +
                            ",outcome_start=" + frozenOutcomeJob->outcomeStart +
                            ",outcome_end=" + frozenOutcomeJob->outcomeEnd +
                            ",metric_hash=" +
                            frozenOutcomeJob->metricDefinitionHash +
                            ",source_content_hash=" +
                            request.sourceContentHash +
                            ",outcome_identity_hash=" +
                            persisted.outcomeIdentityHash + ",job_hash=" +
                            frozenOutcomeJob->hash;
                        const char* safety =
                            ",activation=false,live_profitability_weight=0"
                            ",production_ranking_modified=false"
                            ",recommendation_modified=false"
                            ",ranking_snapshot_modified=false"
                            ",training_started=false,experiment_created=false"
                            ",experiment_queued=false,scheduler_modified=false";
                        std::cout
                            << "CAMPAIGN_PROFITABILITY_OUTCOME_INFERENCE"
                            << identityPrefix
                            << ",inference_scope=prospective_outcome"
                            << ",inference_accuracy=" << evaluation.accuracy
                            << ",prediction_count=" << statistics.predictionCount
                            << safety << std::endl;
                        std::cout
                            << "CAMPAIGN_PROFITABILITY_OUTCOME_PROFITABILITY"
                            << identityPrefix
                            << ",aggregate_terminal_horizon_log_return_sum="
                            << statistics.aggregateTerminalHorizonLogReturnSum
                            << ",average_terminal_horizon_log_return_per_"
                               "actionable_prediction="
                            << (request.averageReturn
                                    ? std::to_string(*request.averageReturn)
                                    : "NULL")
                            << ",actionable_count=" << statistics.actionableCount
                            << ",winning_actionable_count="
                            << statistics.winningActionableCount
                            << ",losing_actionable_count="
                            << statistics.losingActionableCount
                            << safety << std::endl;
                        std::cout
                            << "CAMPAIGN_PROFITABILITY_OUTCOME_IDENTITY"
                            << identityPrefix
                            << ",prospective_outcome_result_id="
                            << persisted.resultId
                            << ",idempotent_existing="
                            << (persisted.created ? "false" : "true")
                            << ",final_evidence_modified=false"
                            << ",checkpoint_evidence_used=false"
                            << safety << std::endl;
                        std::cout
                            << "CAMPAIGN_PROFITABILITY_OUTCOME_SUMMARY"
                            << identityPrefix
                            << ",completed_outcome_count=1"
                            << ",unique_model_outcome_count=1"
                            << ",recommendation_count="
                            << frozenOutcomeJob->recommendationIds.size()
                            << safety << std::endl;
                        runtimeDatabaseWork.commit();
                    }
                    catch (const std::exception& error)
                    {
                        std::cerr
                            << "CAMPAIGN_PROFITABILITY_OUTCOME_PERSIST_FAILED"
                            << ",source_experiment_id="
                            << frozenOutcomeJob->sourceExperimentId
                            << ",source_model_id="
                            << frozenOutcomeJob->sourceModelId
                            << ",error=" << error.what() << std::endl;
                        return 1;
                    }
                    DiagnosticOut()
                        << "prospective_outcome_infer=true; skipping FINAL "
                           "inference persistence and model save"
                        << std::endl;
                    break;
                }
                if (schedulerInferenceContext.has_value())
                {
                    try
                    {
                        if (!loadedModelId.has_value() ||
                            *loadedModelId != schedulerInferenceContext->modelId)
                        {
                            throw std::runtime_error("loaded_model_id_does_not_match_scheduler_inference_identity");
                        }

                        const InferenceIdentity identity =
                            BuildInferenceIdentity(*loadedModelId,
                                                   rawPriceTableName,
                                                   requestedTargetType,
                                                   fromDate,
                                                   toDate);
                        InferAllSummaryRow row;
                        row.modelId = *loadedModelId;
                        row.name = schedulerInferenceContext->inferenceScope + "-scheduler-inference";
                        row.completedEpochs = evaluation.completedEpochs;
                        row.accuracy = evaluation.accuracy;
                        row.acceptance = evaluation.acceptance;
                        row.profitability = evaluation.profitability;
                        row.profitabilitySourceContentHash =
                            evaluation.profitabilitySourceContentHash;

                        long long inferenceEvalResultId = -1;
                        if (schedulerInferenceContext->inferenceScope == "checkpoint")
                        {
                            inferenceEvalResultId =
                                PersistCompletedCheckpointInferenceResult(
                                    runtimeDatabaseWork,
                                    identity,
                                    row,
                                    *schedulerInferenceContext);
                            PersistInferenceProfitabilityObservation(
                                runtimeDatabaseWork,
                                inferenceEvalResultId,
                                identity,
                                row,
                                EA::InferenceProfitability::Scope::checkpointInference,
                                schedulerInferenceContext->checkpointEvalId,
                                schedulerInferenceContext->parentExperimentId);
                        }
                        else
                        {
                            inferenceEvalResultId =
                                PersistCompletedInferenceResult(
                                    runtimeDatabaseWork, identity, row);
                            PersistInferenceProfitabilityObservation(
                                runtimeDatabaseWork,
                                inferenceEvalResultId,
                                identity,
                                row,
                                EA::InferenceProfitability::Scope::finalInference,
                                std::nullopt,
                                schedulerInferenceContext->schedulerExperimentId);
                            std::cout << "SCHEDULER_INFER_RESULT_PERSISTED"
                                      << ",experiment_id="
                                      << schedulerInferenceContext->schedulerExperimentId.value_or(-1)
                                      << ",model_id=" << identity.modelId
                                      << ",inference_scope=final"
                                      << ",status=completed"
                                      << std::endl;
                        }
                        runtimeDatabaseWork.commit();
                    }
                    catch (const std::exception& e)
                    {
                        if (schedulerInferenceContext->inferenceScope == "checkpoint")
                            LogCheckpointInferencePersistenceFailure(*schedulerInferenceContext, e.what());
                        else
                            std::cerr << "SCHEDULER_INFER_RESULT_PERSIST_FAILED"
                                      << ",experiment_id="
                                      << schedulerInferenceContext->schedulerExperimentId.value_or(-1)
                                      << ",model_id=" << schedulerInferenceContext->modelId
                                      << ",inference_scope=final"
                                      << ",status=failed"
                                      << ",error=" << e.what()
                                      << std::endl;
                        return 1;
                    }
                }
                DiagnosticOut() << "runtime_infer=true; skipping model save" << std::endl;
                break;
            }

            UseRuntimeDefaultEvalLabelConfig();
            ModelConfigValidationResult modelConfigValidation;
            if (loadedModelId.has_value())
                modelConfigValidation = PrintModelConfigValidation(
                    runtimeDatabaseWork,
                    *loadedModelId,
                    requestedTargetType,
                    t,
                    rawPriceTableName);
            PrintEvalLabelConfig();

            // All startup/configuration/model reads from the LSTM database are
            // complete before the long-running training loop begins.  End this
            // transaction now so an idle training worker does not retain an
            // AccessShareLock (and MVCC snapshot) on experiment for hours.
            // Checkpoint persistence already uses its own short transaction;
            // final model persistence opens another short transaction below.
            runtimeDatabaseWork.commit();

            PrintClassificationProofDiagnostics(l, t, fromDate, toDate);
            PrintPhase2TensorDiagnostics(l, t, fromDate, toDate);

            // Iterate all training batches; inference exits through RunInferenceEvaluation above.
            std::cout << std::setprecision(15);
                const int startEpoch = resumeConfig.has_value()
                    ? static_cast<int>(resumeConfig->completedEpoch)
                    : 0;
                bool checkpointStopReached = false;
                for(auto e = startEpoch; e < epoch_count; e++)
                {
                    t.ForEachBatchFrom(logicalOutputStartIndex, [&](auto b)
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

                        float loss = 0.0f;
                        size_t _unused1 = 0;
                        size_t _unused2 = 0;
                        {
                            ScopedDiagnosticCoutSilencer silence;
                            auto result = l.CalculateBatch(b,e);
                            loss = std::get<0>(result);
                            _unused1 = std::get<1>(result);
                            _unused2 = std::get<2>(result);
                        }
                        (void)_unused1; (void)_unused2;

                        double p1 = l2(l.param);
                        double b1 = l2(l.bias);
                        double hw1 = l2(l.returnHeadWeight);
                        double hb1 = l2(l.returnHeadBias);
                        double dhw1 = l2(l.returnHeadDirWeight);
                        double dhb1 = l2(l.returnHeadDirBias);

                        DiagnosticOut() << "epoch " << (e+1)
                        << " loss=" << loss
                        << " ||param|| " << p0  << " -> " << p1
                        << " ||bias|| "  << b0  << " -> " << b1;
                        if (l.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
                            DiagnosticOut() << " ||dirHeadW|| " << dhw0 << " -> " << dhw1 << " ||dirHeadB|| " << dhb0 << " -> " << dhb1 << std::endl;
                        else
                            DiagnosticOut() << " ||headW|| " << hw0 << " -> " << hw1 << " ||headB|| " << hb0 << " -> " << hb1 << std::endl;
                    } );
                    if (l.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
                    {
                        ScopedDiagnosticCoutSilencer silence;
                        EA::LSTM::PrintAndResetEpochBuckets();
                        PrintAndResetDistribution();
                    }
                    l.completedEpochs = static_cast<size_t>(e + 1);
                    UpdateSchedulerExperimentProgress(launchArgs.schedulerExperimentId,
                                                      static_cast<int>(e + 1));
                    const std::optional<long long> checkpointModelId =
                        SavePeriodicCheckpointIfDue(launchArgs,
                                                    resumeConfig,
                                                    rawPriceTableName,
                                                    fromDate,
                                                    toDate,
                                                    l,
                                                    runtimeDonchian20Mode,
                                                    featureWarmupScope,
                                                    runtimeDonchianLookback,
                                                    runtimeTrainingObjective);
                    if (checkpointModelId.has_value())
                    {
                        QueueCheckpointInferenceIfEligible(launchArgs.schedulerExperimentId,
                                                           launchArgs.checkpointEvery,
                                                           static_cast<int>(e + 1),
                                                           *checkpointModelId);
                        const std::optional<CheckpointStopConfig> checkpointStopConfig =
                            LoadCheckpointStopConfig(launchArgs.schedulerExperimentId,
                                                     static_cast<int>(e + 1),
                                                     epoch_count,
                                                     launchArgs.checkpointEvery);
                        if (checkpointStopConfig.has_value() &&
                            RecordCheckpointStopReached(
                                launchArgs.schedulerExperimentId,
                                launchArgs.schedulerWorkerAttemptId,
                                static_cast<int>(e + 1),
                                *checkpointModelId))
                        {
                            checkpointStopReached = true;
                            break;
                        }
                        if (checkpointStopConfig.has_value() &&
                            checkpointStopConfig->cancellationRequested)
                        {
                            std::cerr
                                << "GLOBAL_CANCELLATION_CHECKPOINT_RECORD_FAILED"
                                << " experiment_id="
                                << *launchArgs.schedulerExperimentId
                                << " epoch=" << (e + 1)
                                << " model_id=" << *checkpointModelId
                                << std::endl;
                            return 1;
                        }
                    }
                }
                if (checkpointStopReached)
                    return 0;
            // Persist trained model parameters to DB
            try
            {
                if (save_enable)
                {
                    // Startup reads were committed before training.  Use a
                    // fresh, short transaction for final model persistence.
                    pqxx::work wSave { c_LSTM };
                    wSave.exec("SET TRANSACTION READ WRITE;");
                    
                    long long modelId = -1;
                    if (resumeConfig.has_value())
                    {
                        const std::string resumeModelName = launchArgs.newModelName.value_or(rawPriceTableName + "-resume-model");
                        modelId = DBIO::PgModelIO::createModel(wSave,
                                                               resumeModelName,
                                                               "resumed trained parameters",
                                                               launchArgs.schedulerExperimentId,
                                                               ParentModelIdForResume(resumeConfig));
                        std::cout << "Created new model_id=" << modelId
                                  << " (resumed from model_id=" << resumeConfig->sourceModelId << ")"
                                  << std::endl;
                    }
                    else if (startedFromScratch)
                        if constexpr (save_overwrite)
                            // Overwrite the latest model if one exists; otherwise create a new snapshot
                            try {
                                pqxx::result rLatest = wSave.exec("SELECT max(model_id) FROM model;");
                                if (!rLatest.empty() && !rLatest[0][0].is_null()) {
                                    modelId = rLatest[0][0].as<long long>();
                                    LinkModelToSchedulerExperimentIfPresent(wSave, modelId, launchArgs.schedulerExperimentId);
                                    LinkModelParentIfPresent(wSave, modelId, ParentModelIdForResume(resumeConfig));
                                    std::cout << "Overwriting latest model_id=" << modelId << " (started from scratch, overwrite enabled)" << std::endl;
                                } else {
                                    modelId = DBIO::PgModelIO::createModel(wSave,
                                                                           rawPriceTableName + "-model",
                                                                           "trained parameters",
                                                                           launchArgs.schedulerExperimentId);
                                    std::cout << "Created new model_id=" << modelId << " (no existing model to overwrite)" << std::endl;
                                }
                            } catch (const std::exception& e)
                            {
                                std::cout << "Fetch latest model_id failed (" << e.what() << "); creating new snapshot" << std::endl;
                                modelId = DBIO::PgModelIO::createModel(wSave,
                                                                       rawPriceTableName + "-model",
                                                                       "trained parameters",
                                                                       launchArgs.schedulerExperimentId);
                            }
                        else
                        {
                            // Create a new snapshot when saving (do not overwrite existing)
                            modelId = DBIO::PgModelIO::createModel(wSave,
                                                                   rawPriceTableName + "-model",
                                                                   "trained parameters",
                                                                   launchArgs.schedulerExperimentId);
                            std::cout << "Created new model_id=" << modelId << " (started from scratch)" << std::endl;
                        }
                    else
                        if constexpr (save_overwrite)
                            if (loadedModelId.has_value())
                            {
                                modelId = *loadedModelId;
                                LinkModelToSchedulerExperimentIfPresent(wSave, modelId, launchArgs.schedulerExperimentId);
                                LinkModelParentIfPresent(wSave, modelId, ParentModelIdForResume(resumeConfig));
                                std::cout << "Overwriting existing model_id=" << modelId << std::endl;
                            }
                            else
                            {
                                modelId = DBIO::PgModelIO::createModel(wSave,
                                                                       rawPriceTableName + "-model",
                                                                       "trained parameters",
                                                                       launchArgs.schedulerExperimentId);
                                std::cout << "Created new model_id=" << modelId << " (no prior model to overwrite)" << std::endl;
                            }
                        else
                        {
                            // Create a new snapshot when saving (do not overwrite existing)
                            modelId = DBIO::PgModelIO::createModel(wSave,
                                                                   rawPriceTableName + "-model",
                                                                   "trained parameters",
                                                                   launchArgs.schedulerExperimentId);
                            std::cout << "Created new model_id=" << modelId << std::endl;
                        }

                    DBIO::PgModelIO::saveAll(wSave, modelId, l, rawPriceTableName, fromDate, toDate,
                                             runtimeDonchian20Mode, featureWarmupScope,
                                             runtimeDonchianLookback,
                                             resumeConfig.has_value()
                                                 ? resumeConfig->inputWidthExpansionProvenance
                                                 : std::nullopt,
                                             runtimeTrainingObjective);
                    wSave.commit();
                    std::cout << "Saved model with model_id=" << modelId << std::endl;
                    if (resumeConfig.has_value())
                        std::cout << "RESUME_SAVED_NEW_MODEL_ID=" << modelId << std::endl;
                }
                else
                    if constexpr (!save_enable)
                        DiagnosticOut() << "save_enable=false; skipping model save" << std::endl;
                    else
                        DiagnosticOut() << "skipping model save (unknown reason)" << std::endl;
            }
            catch (const std::exception& e) { std::cerr << "Model save/load error: " << e.what() << std::endl;    }

            break;
        }

    }
    catch (const pqxx::broken_connection& e)
    {
        std::cerr << "Broken connection: " << e.what() << "\n";
        std::cout.flush();
        std::cerr.flush();
        return 1;
    }
    catch (const pqxx::failure& e)
    {
        std::cerr << "pqxx::failure: " << e.what() << "\n";
        std::cout.flush();
        std::cerr.flush();
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "std::exception: " << e.what() << "\n";
        std::cout.flush();
        std::cerr.flush();
        return 1;
    }
    catch (...)
    {
        std::cerr << "unknown exception\n";
        std::cout.flush();
        std::cerr.flush();
        return 1;
    }

    std::cout.flush();
    std::cerr.flush();
    return 0;
}
