#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <pqxx/pqxx>

#include "MarketDataCore.hpp"
#include "ModelInputPreparation.hpp"
#include "Tensor.hpp"
#include "LSTM.hpp"
#include "../Sources/LaunchArguments.hpp"
#include "../Sources/TrainingWorkerApplication.hpp"
#include "../Sources/TrainingWorkerFeatureAblation.hpp"
#include "../Sources/TrainingRuntimeConfig.hpp"
#include "../Sources/PersistedModelRuntimeConfig.hpp"
#include "../Sources/ModelRuntimeValidation.hpp"
#include "../Sources/RuntimeLogging.hpp"
#include "../Sources/SchedulerRuntimeConfigValidation.hpp"
#include "../Sources/LstmRuntimeConstruction.hpp"
#include "../Sources/RuntimeDatabaseConnection.hpp"
#include "../Sources/LaunchRuntimeConfig.hpp"
#include "../Sources/RuntimeEconomicCalendarIdentity.hpp"
#include "../Sources/SchedulerWorkerOwnershipTestBoundary.hpp"
#include "../Sources/LstmHotspotProfileFinalizer.hpp"
#include "../Sources/CheckpointTrainingControl.hpp"
#include "PgModelIO.hpp"
#include "BuildConfig.hpp"
#include "TargetLabel.hpp"
#include "ExperimentScheduler.hpp"
#include "../Sources/SchedulerCore/SchedulerWorkerRegistration.hpp"
#include "WorkerLifecycleDiagnostics.hpp"
#include "CanonicalSymbol.hpp"
#include "FxPriceSanity.hpp"
#include "LstmRuntimeLogging.hpp"
#include "Donchian20Mode.hpp"
#include "DonchianLookback.hpp"
#include "FeatureWarmupScope.hpp"
#include "ModelInputContract.hpp"

void PrintAndResetDistribution();

namespace
{

using EA::PersistedModelRuntimeConfig::ApplyResumeRuntimeConfig;
using EA::PersistedModelRuntimeConfig::ConfigureInputWidthExpansionForResume;
using EA::PersistedModelRuntimeConfig::LoadExperimentTrainingObjective;
using EA::PersistedModelRuntimeConfig::LoadModelFeatureAblationMask;
using EA::PersistedModelRuntimeConfig::LoadResumeCheckpointConfig;
using EA::PersistedModelRuntimeConfig::LoadSchedulerFeatureAblationMask;
using EA::PersistedModelRuntimeConfig::LoadSchedulerModelInputIdentity;
using EA::PersistedModelRuntimeConfig::ResumeCheckpointConfig;
using EA::PersistedModelRuntimeConfig::SchedulerModelInputIdentity;
using EA::PersistedModelRuntimeConfig::ValidateResumeLaunchArgs;
using EA::PersistedModelRuntimeConfig::ValidateSchedulerModelFeatureAblationMask;
using EA::PersistedModelRuntimeConfig::ValidateSchedulerResumeFeatureAblationMask;
using EA::ModelRuntimeValidation::ModelConfigValidationResult;
using EA::ModelRuntimeValidation::PrintDatabaseModelSymbol;
using EA::ModelRuntimeValidation::PrintLegacyModelSymbol;
using EA::ModelRuntimeValidation::PrintMissingModelSymbol;
using EA::ModelRuntimeValidation::PrintModelConfigValidation;
using EA::ModelRuntimeValidation::PrintMaterializedModelConfigValidation;
using EA::ModelRuntimeValidation::RuntimeModelInputWidth;
using EA::ModelRuntimeValidation::RuntimeTensorFeatureWidth;
using EA::ModelRuntimeValidation::ValidateLoadedModelSymbolForSelectedTable;
using EA::ModelRuntimeValidation::ValidateRuntimeSymbolMatchesModel;
using EA::ModelRuntimeValidation::ResolveLegacyModelSymbol;
using EA::ModelRuntimeValidation::ResolveLegacySymbolFromModelName;
using EA::ModelRuntimeValidation::TargetTypeName;
using EA::ModelRuntimeValidation::DirectionLabelRuleName;
using EA::ModelRuntimeValidation::DirectionLabelRuleId;
using EA::RuntimeLogging::LogSummary;
using EA::RuntimeLogging::LogDiagnostic;
using EA::RuntimeLogging::DiagnosticOut;
using EA::RuntimeLogging::ScopedDiagnosticCoutSilencer;
using EA::SchedulerRuntimeConfigValidation::ValidateSchedulerDonchian20Mode;
using EA::SchedulerRuntimeConfigValidation::ValidateSchedulerFeatureWarmupScope;
using EA::SchedulerRuntimeConfigValidation::ValidateSchedulerDonchianLookback;

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


bool gRuntimeInferenceMode = false;

const char* RuntimeLogLevelName(EA::RuntimeLogLevel level)
{
    switch (level)
    {
        case EA::RuntimeLogLevel::Quiet: return "quiet";
        case EA::RuntimeLogLevel::Summary: return "summary";
        case EA::RuntimeLogLevel::Diagnostic: return "diagnostic";
    }
    return "unknown";
}

const char* CurrentRangeKindLabel()
{
    return "train";
}


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

EvalLabelConfig& ActiveEvalLabelConfig()
{
    static EvalLabelConfig config = RuntimeDefaultEvalLabelConfig();
    return config;
}

void UseRuntimeDefaultEvalLabelConfig()
{
    ActiveEvalLabelConfig() = RuntimeDefaultEvalLabelConfig();
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

void PrintRuntimeConfig()
{
    if (!LogSummary())
        return;
    std::cout << "RUNTIME_CONFIG"
              << ",train=" << (gRuntimeInferenceMode ? "false" : "true")
              << ",infer=" << (gRuntimeInferenceMode ? "true" : "false")
              << ",log_level="
              << RuntimeLogLevelName(EA::RuntimeLogLevelValue())
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

std::string CheckpointBaseModelName(const EA::LaunchArgs& launchArgs,
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

std::optional<long long> SavePeriodicCheckpointIfDue(const EA::LaunchArgs& launchArgs,
                                                     const std::optional<ResumeCheckpointConfig>& resumeConfig,
                                                     const std::string& rawPriceTableName,
                                                     const std::string& fromDate,
                                                     const std::string& toDate,
                                                     EA::LSTM& lstm,
                                                     const EA::Training::RuntimeConfig& runtimeConfig)
{
    if (!launchArgs.checkpointEvery.has_value() || *launchArgs.checkpointEvery <= 0)
        return std::nullopt;
    if (runtimeConfig.inferenceMode)
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
    pqxx::connection cCheckpoint { EA::RuntimeDatabaseConnection::LstmConnectionString() };
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
                             runtimeConfig.donchian20Mode,
                             runtimeConfig.featureWarmupScope,
                             runtimeConfig.donchianLookback,
                             resumeConfig.has_value()
                                 ? resumeConfig->inputWidthExpansionProvenance
                                 : std::nullopt,
                             runtimeConfig.trainingObjective);
    wCheckpoint.commit();
    std::cout << "CHECKPOINT_SAVE_DONE"
              << " epoch=" << completedEpoch
              << " model_id=" << checkpointModelId
              << " name=" << checkpointName
              << std::endl;
    return checkpointModelId;
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


} // namespace

int EA::Training::RunTrainingWorkerApplication(int argc, const char* argv[])
{
    EA::RuntimeLogging::InstallModelRuntimeValidationDiagnostics();
    EA::LaunchArgs launchArgs;
    try
    {
        launchArgs = EA::ParseLaunchArgs(argc, argv);
        if (!launchArgs.schedulerExperimentId.has_value() ||
            launchArgs.schedulerCheckpointEvalId.has_value() ||
            !launchArgs.schedulerWorkerAttemptId.has_value() ||
            launchArgs.inferenceMode.value_or(true))
        {
            throw std::invalid_argument(
                "lstm-train-worker requires scheduler-managed --train work "
                "with an exact --scheduler-experiment-id and "
                "--scheduler-worker-attempt-id");
        }
        if (!EA::SchedulerCore::RegisterSchedulerWorker({
                *launchArgs.schedulerWorkerAttemptId,
                launchArgs.schedulerExperimentId,
                std::nullopt,
                "experiment",
                "train"}))
        {
            return 125;
        }
        if (const std::optional<int> testBoundary =
                EA::SchedulerWorkerOwnershipTestBoundary::Run(launchArgs))
        {
            return *testBoundary;
        }
        if (launchArgs.logLevel.has_value())
            EA::SetRuntimeLogLevel(*launchArgs.logLevel);
        EA::ExperimentScheduler::LogWorkerStarted(
            "TRAINING_WORKER_STARTED",
            "train",
            launchArgs.schedulerExperimentId,
            launchArgs.resumeModelId);
        if (!ValidateResumeLaunchArgs(launchArgs))
            return 1;
        if (!launchArgs.resumeModelId.has_value())
            EA::LaunchRuntimeConfig::Apply(launchArgs, gRuntimeInferenceMode);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n"
                  << "Usage: " << argv[0]
                  << " --train --scheduler-experiment-id=<id> "
                     "--scheduler-worker-attempt-id=<id> [training options] "
                     "<fromDate> <toDate>\n";
        return 1;
    }

    EA::LSTM::ConfigureHotspotProfiler(
        launchArgs.lstmProfileHotspots,
        launchArgs.lstmProfileOutputPath);
    EA::LstmHotspotProfileFinalizer hotspotProfileFinalizer{
        launchArgs.lstmProfileHotspots,
        launchArgs.lstmProfileOutputPath};

    pqxx::connection c_forex{
        EA::RuntimeDatabaseConnection::ForexConnectionString()};
    pqxx::connection c_LSTM{
        EA::RuntimeDatabaseConnection::LstmConnectionString()};
    std::string fromDate{launchArgs.fromDate};
    std::string toDate{launchArgs.toDate};


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
    std::optional<DBIO::PgModelIO::PersistedModelMaterialization>
        selectedModelMaterialization;

    if (launchArgs.resumeModelId.has_value())
    {
        try
        {
            pqxx::work resumeRead { c_LSTM };
            resumeRead.exec(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
            selectedModelMaterialization =
                DBIO::PgModelIO::ReadPersistedModelMaterialization(
                    resumeRead, *launchArgs.resumeModelId);
            resumeConfig = LoadResumeCheckpointConfig(
                *selectedModelMaterialization,
                runtimeTrainingObjective,
                PrintDatabaseModelSymbol);
            ConfigureInputWidthExpansionForResume(
                *selectedModelMaterialization, *resumeConfig,
                launchArgs.resumeExpandInputWidth,
                launchArgs.schedulerExperimentId);
            resumeRead.commit();
            if (launchArgs.featureWarmupScope.has_value() &&
                *launchArgs.featureWarmupScope != resumeConfig->featureWarmupScope)
                throw std::runtime_error("feature warmup scope mismatch: persisted/runtime");
            featureWarmupScope = resumeConfig->featureWarmupScope;
            ApplyResumeRuntimeConfig(*resumeConfig, *launchArgs.targetEpochs,
                                     gRuntimeInferenceMode);
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
            availableSymbols = EA::MarketData::DiscoverRawPriceTables(
                forexMetadataRead);
            forexMetadataRead.commit();
        }

        std::optional<EA::FeatureAblationMask> schedulerFeatureAblationMask;
        std::optional<SchedulerModelInputIdentity> schedulerModelInputIdentity;
        pqxx::work configurationRead { c_LSTM };
        configurationRead.exec("SET TRANSACTION READ ONLY;");

        Donchian20Mode runtimeDonchian20Mode = kDefaultDonchian20Mode;
        std::size_t runtimeDonchianLookback = kDefaultDonchianLookback;
        if (resumeConfig.has_value())
        {
            runtimeDonchian20Mode = resumeConfig->donchian20Mode;
            runtimeDonchianLookback = resumeConfig->donchianLookback;
        }
        else if (launchArgs.donchian20Mode.has_value())
            runtimeDonchian20Mode = *launchArgs.donchian20Mode;
        if (!resumeConfig.has_value() && launchArgs.donchianLookback.has_value())
            runtimeDonchianLookback = *launchArgs.donchianLookback;

        if (launchArgs.donchian20Mode.has_value() &&
            *launchArgs.donchian20Mode != runtimeDonchian20Mode)
        {
            throw std::runtime_error(
                std::string{"Donchian-20 mode mismatch: persisted="} +
                Donchian20ModeText(runtimeDonchian20Mode) + ", runtime=" +
                Donchian20ModeText(*launchArgs.donchian20Mode));
        }
        ValidateSchedulerDonchian20Mode(
            configurationRead, launchArgs, runtimeDonchian20Mode);
        ValidateSchedulerFeatureWarmupScope(
            configurationRead, launchArgs, featureWarmupScope);
        ValidateSchedulerDonchianLookback(
            configurationRead, launchArgs, runtimeDonchianLookback);

        schedulerModelInputIdentity =
            LoadSchedulerModelInputIdentity(configurationRead, launchArgs);
        schedulerFeatureAblationMask =
            LoadSchedulerFeatureAblationMask(configurationRead, launchArgs);
        if (resumeConfig.has_value())
        {
            ValidateSchedulerResumeFeatureAblationMask(
                *resumeConfig, *schedulerFeatureAblationMask);
        }
        configurationRead.commit();

        const EA::Training::RuntimeConfig trainingRuntimeConfig{
            gRuntimeInferenceMode,
            prediction_horizon,
            c_next_threshold,
            window_size,
            hidden_size,
            n_out,
            num_layers,
            normalization_version,
            epoch_count,
            core_lr_mult,
            head_weight_lr_mult,
            head_bias_lr_mult,
            runtimeDonchian20Mode,
            featureWarmupScope,
            runtimeDonchianLookback,
            runtimeTrainingObjective};


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
                std::cerr << "Requested resume symbol/table not found: "
                          << resumeConfig->symbol << std::endl;
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
                std::cerr << "Requested symbol/table not found: "
                          << *launchArgs.symbol << std::endl;
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
            DiagnosticOut() << "SYMBOL_SELECTION"
                            << ",requested="
                            << (resumeConfig.has_value()
                                    ? resumeConfig->symbol
                                    : (launchArgs.symbol.has_value()
                                           ? *launchArgs.symbol
                                           : "none"))
                            << ",selected=" << rawPriceTableName
                            << ",available_count=" << availableSymbols.size()
                            << std::endl;


            std::optional<EA::EconomicCalendar::
                EconomicCalendarSnapshotIdentity> economicCalendarSnapshot;
            if (selectedModelMaterialization.has_value())
            {
                economicCalendarSnapshot =
                    EA::RuntimeEconomicCalendarIdentity::FromMaterialization(
                        *selectedModelMaterialization);
            }
            else
            {
                pqxx::work economicEventRead { c_LSTM };
                economicEventRead.exec("SET TRANSACTION READ ONLY;");
                economicCalendarSnapshot =
                    EA::RuntimeEconomicCalendarIdentity::Resolve(
                        economicEventRead, launchArgs);
                economicEventRead.commit();
            }
            const auto preparedInput = EA::ModelInputPreparation::Prepare(
                {rawPriceTableName, fromDate, toDate, featureWarmupScope,
                 runtimeDonchian20Mode, runtimeDonchianLookback,
                 economicCalendarSnapshot},
                {EA::RuntimeDatabaseConnection::ForexConnectionString(),
                 EA::RuntimeDatabaseConnection::LstmConnectionString()});
            Tensor t = std::move(preparedInput.tensor);
            const size_t logicalOutputStartIndex =
                preparedInput.logicalOutputStartIndex;
            economicCalendarSnapshot = preparedInput.calendarSnapshot;


            std::optional<std::size_t> persistedModelInputWidth =
                resumeConfig.has_value()
                    ? std::optional<std::size_t>{
                          resumeConfig->expandInputWidthRequested
                              ? EA::kCurrentModelInputWidth
                              : static_cast<std::size_t>(
                                    resumeConfig->modelInputWidth)}
                    : std::nullopt;


            if (schedulerModelInputIdentity.has_value())
            {
                if (persistedModelInputWidth.has_value() &&
                    *persistedModelInputWidth !=
                        schedulerModelInputIdentity->width)
                {
                    throw std::runtime_error(
                        "scheduler_model_input_width_mismatch:experiment=" +
                        std::to_string(
                            schedulerModelInputIdentity->experimentId) +
                        ",expected=" + std::to_string(
                            schedulerModelInputIdentity->width) +
                        ",model=" +
                        std::to_string(*persistedModelInputWidth));
                }
                persistedModelInputWidth = schedulerModelInputIdentity->width;
            }
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


            const auto requestedTargetType = resumeConfig.has_value()
                ? resumeConfig->targetType
                : EA::LSTM::TargetType::UpNeutralDownReturn;
            const std::size_t runtimeLstmHiddenSize = resumeConfig.has_value()
                ? resumeConfig->modelHiddenSize
                : trainingRuntimeConfig.hiddenSize;

            std::optional<pqxx::work> runtimeDatabaseWork;
            runtimeDatabaseWork.emplace(c_LSTM);
            runtimeDatabaseWork->exec("SET TRANSACTION READ WRITE;");

            const EA::FeatureAblationMask runtimeFeatureAblationMask =
                schedulerFeatureAblationMask.has_value()
                    ? *schedulerFeatureAblationMask
                    : (resumeConfig.has_value()
                           ? resumeConfig->featureAblationMask
                           : EA::FeatureAblationMask{});
            std::optional<long long> loadedModelId;
            bool startedFromScratch = true;
            const std::string loadSource = resumeConfig.has_value()
                ? "--resume-model-id"
                : (launchArgs.modelId.has_value() ? "--model" : "latest");
            std::unique_ptr<EA::LSTM> runtimeModel =
                std::make_unique<EA::LSTM>(
                    EA::LstmRuntimeConstruction::CreateLstmForRuntimeLogLevel(
                        t, runtimeLstmHiddenSize, 1, 0, requestedTargetType,
                        persistedModelInputWidth, runtimeFeatureAblationMask,
                        launchArgs.freshInitializationSeed.value_or(42U)));
            runtimeModel->SetTrainingObjective(
                trainingRuntimeConfig.trainingObjective);
            EA::LSTM& l = *runtimeModel;
            PrintRuntimeLrConfig(l);
            static size_t s_lstmBindingDiagCount = 0;
            constexpr size_t kLstmBindingDiagLimit = 50;
            if (s_lstmBindingDiagCount < kLstmBindingDiagLimit)
            {
                DiagnosticOut() << "DIAG_LSTM_BINDING"
                                << ",table=" << rawPriceTableName
                                << ",tensor_rows=" << t.RowCount()
                                << ",tensor_addr="
                                << static_cast<const void*>(&t)
                                << ",lstm_tensor_ref_addr="
                                << static_cast<const void*>(
                                       l.BoundTensorAddress())
                                << ",newly_constructed=1"
                                << ",reused=0" << std::endl;
                ++s_lstmBindingDiagCount;
            }

            try
            {
                long long modelIdToLoad = -1;
                const bool requestedModel = launchArgs.modelId.has_value();

                if (selectedModelMaterialization.has_value())
                    modelIdToLoad =
                        selectedModelMaterialization->identity.modelId;
                else if (requestedModel)
                    modelIdToLoad = *launchArgs.modelId;
                else if (load_latest)
                {
                    pqxx::result r = runtimeDatabaseWork->exec(
                        "SELECT max(model_id) FROM model;");
                    if (!r.empty() && !r[0][0].is_null())
                        modelIdToLoad = r[0][0].as<long long>();
                    else
                        DiagnosticOut()
                            << "No models found; using default-initialized parameters"
                            << std::endl;
                }
                else
                {
                    DiagnosticOut()
                        << "load_latest=false; using default-initialized parameters"
                        << std::endl;
                }

                if (modelIdToLoad > 0)
                {
                    if (selectedModelMaterialization.has_value())
                    {
                        EA::TrainingObjective::RequireResumeCompatible(
                            selectedModelMaterialization->trainingObjective,
                            runtimeTrainingObjective);
                        ScopedDiagnosticCoutSilencer silence;
                        DBIO::PgModelIO::ApplyPersistedModelMaterialization(
                            *selectedModelMaterialization,
                            l,
                            resumeConfig.has_value() &&
                                resumeConfig->parameterExpansionRequired);
                    }
                    else
                    {
                        const auto persistedObjective =
                            DBIO::PgModelIO::loadTrainingObjectiveMeta(
                                *runtimeDatabaseWork, modelIdToLoad);
                        EA::TrainingObjective::RequireResumeCompatible(
                            persistedObjective, runtimeTrainingObjective);
                        ScopedDiagnosticCoutSilencer silence;
                        DBIO::PgModelIO::loadAll(
                            *runtimeDatabaseWork,
                            modelIdToLoad,
                            l,
                            resumeConfig.has_value() &&
                                resumeConfig->parameterExpansionRequired);
                    }
                    if (resumeConfig.has_value() &&
                        !selectedModelMaterialization.has_value())
                    {
                        ScopedDiagnosticCoutSilencer silence;
                        DBIO::PgModelIO::loadOptimizerMeta(
                            *runtimeDatabaseWork, modelIdToLoad, l);
                        l.completedEpochs = resumeConfig->completedEpoch;
                        std::cout << "RESUME_OPTIMIZER_STATE_RESTORED=1"
                                  << std::endl;
                    }
                    const Donchian20Mode modelMode =
                        selectedModelMaterialization.has_value()
                            ? selectedModelMaterialization->donchian20Mode
                            : DBIO::PgModelIO::loadDonchian20ModeMeta(
                                  *runtimeDatabaseWork, modelIdToLoad);
                    if (modelMode != runtimeDonchian20Mode)
                    {
                        throw std::runtime_error(
                            std::string{
                                "model/runtime Donchian-20 mode mismatch: model="} +
                            Donchian20ModeText(modelMode) + ", runtime=" +
                            Donchian20ModeText(runtimeDonchian20Mode));
                    }
                    loadedModelId = modelIdToLoad;
                    startedFromScratch = false;
                    DiagnosticOut() << "Loaded model_id=" << *loadedModelId
                                    << " source=" << loadSource << std::endl;
                }
            }
            catch (const std::exception& e)
            {
                if (std::string{e.what()}.find(
                        "Donchian-20 mode mismatch") != std::string::npos ||
                    std::string{e.what()}.find(
                        "model/runtime Donchian-20 mode mismatch") !=
                        std::string::npos)
                {
                    std::cerr << "DONCHIAN20_MODE_COMPATIBILITY_FAILURE"
                              << ",error=" << e.what() << std::endl;
                    return 1;
                }
                if (resumeConfig.has_value())
                {
                    std::cerr << "Resume load model_id="
                              << resumeConfig->sourceModelId
                              << " failed: " << e.what() << std::endl;
                    return 1;
                }
                if (launchArgs.modelId.has_value())
                {
                    std::cerr << "Load requested model_id="
                              << *launchArgs.modelId
                              << " failed: " << e.what() << std::endl;
                    return 1;
                }
                DiagnosticOut() << "Load latest failed: " << e.what()
                                << "; using default params" << std::endl;
            }

            if (loadedModelId.has_value())
            {
                if (selectedModelMaterialization.has_value())
                {
                    if (selectedModelMaterialization->trainSymbol.has_value())
                    {
                        PrintDatabaseModelSymbol(
                            *loadedModelId,
                            *selectedModelMaterialization->trainSymbol);
                        ValidateRuntimeSymbolMatchesModel(
                            EA::CanonicalSymbol::Normalize(rawPriceTableName),
                            *selectedModelMaterialization->trainSymbol);
                    }
                    else
                    {
                        PrintMissingModelSymbol(*loadedModelId);
                        PrintLegacyModelSymbol(
                            *loadedModelId,
                            EA::CanonicalSymbol::Normalize(rawPriceTableName));
                    }
                }
                else
                {
                    ValidateLoadedModelSymbolForSelectedTable(
                        *runtimeDatabaseWork,
                        *loadedModelId,
                        rawPriceTableName);
                }
            }

            UseRuntimeDefaultEvalLabelConfig();
            const EA::Training::RuntimeConfig& trainingConfig =
                trainingRuntimeConfig;
            ModelConfigValidationResult modelConfigValidation;
            if (loadedModelId.has_value())
            {
                modelConfigValidation = selectedModelMaterialization.has_value()
                    ? PrintMaterializedModelConfigValidation(
                          *selectedModelMaterialization,
                          requestedTargetType,
                          t,
                          rawPriceTableName)
                    : PrintModelConfigValidation(
                          *runtimeDatabaseWork,
                          *loadedModelId,
                          requestedTargetType,
                          t,
                          rawPriceTableName);
            }
            PrintEvalLabelConfig();

            runtimeDatabaseWork->commit();

            PrintClassificationProofDiagnostics(
                l, t, fromDate, toDate);
            PrintPhase2TensorDiagnostics(l, t, fromDate, toDate);

            std::cout << std::setprecision(15);


                const int startEpoch = resumeConfig.has_value()
                    ? static_cast<int>(resumeConfig->completedEpoch)
                    : 0;
                bool checkpointStopReached = false;
                for(auto e = startEpoch; e < trainingConfig.epochCount; e++)
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

                        const bool batchDiagnosticsEnabled = LogDiagnostic();
                        const bool directionHead =
                            l.targetType == EA::LSTM::TargetType::UpNeutralDownReturn;
                        double p0 = 0.0, b0 = 0.0, hw0 = 0.0, hb0 = 0.0;
                        if (batchDiagnosticsEnabled)
                        {
                            p0 = l2(l.param);
                            b0 = l2(l.bias);
                            hw0 = l2(directionHead ? l.returnHeadDirWeight : l.returnHeadWeight);
                            hb0 = l2(directionHead ? l.returnHeadDirBias : l.returnHeadBias);
                        }

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

                        if (batchDiagnosticsEnabled)
                        {
                            const double p1 = l2(l.param);
                            const double b1 = l2(l.bias);
                            const double hw1 = l2(directionHead ? l.returnHeadDirWeight : l.returnHeadWeight);
                            const double hb1 = l2(directionHead ? l.returnHeadDirBias : l.returnHeadBias);

                            DiagnosticOut() << "epoch " << (e+1)
                            << " loss=" << loss
                            << " ||param|| " << p0  << " -> " << p1
                            << " ||bias|| "  << b0  << " -> " << b1;
                            if (directionHead)
                                DiagnosticOut() << " ||dirHeadW|| " << hw0 << " -> " << hw1 << " ||dirHeadB|| " << hb0 << " -> " << hb1 << std::endl;
                            else
                                DiagnosticOut() << " ||headW|| " << hw0 << " -> " << hw1 << " ||headB|| " << hb0 << " -> " << hb1 << std::endl;
                        }
                    } );
                    if (l.targetType == EA::LSTM::TargetType::UpNeutralDownReturn)
                    {
                        ScopedDiagnosticCoutSilencer silence;
                        EA::LSTM::PrintAndResetEpochBuckets();
                        PrintAndResetDistribution();
                    }
                    l.completedEpochs = static_cast<size_t>(e + 1);
                    EA::CheckpointTrainingControl::UpdateSchedulerExperimentProgress(launchArgs.schedulerExperimentId,
                                                      static_cast<int>(e + 1));
                    const std::optional<long long> checkpointModelId =
                        SavePeriodicCheckpointIfDue(launchArgs,
                                                    resumeConfig,
                                                    rawPriceTableName,
                                                    fromDate,
                                                    toDate,
                                                    l,
                                                    trainingConfig);
                    if (checkpointModelId.has_value())
                    {
                        EA::CheckpointTrainingControl::QueueCheckpointInferenceIfEligible(launchArgs.schedulerExperimentId,
                                                           launchArgs.checkpointEvery,
                                                           static_cast<int>(e + 1),
                                                           *checkpointModelId);
                        const std::optional<EA::CheckpointTrainingControl::CheckpointStopConfig> checkpointStopConfig =
                            EA::CheckpointTrainingControl::LoadCheckpointStopConfig(launchArgs.schedulerExperimentId,
                                                     static_cast<int>(e + 1),
                                                     trainingConfig.epochCount,
                                                     launchArgs.checkpointEvery);
                        if (checkpointStopConfig.has_value() &&
                            EA::CheckpointTrainingControl::RecordCheckpointStopReached(
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
                                             trainingConfig.donchian20Mode,
                                             trainingConfig.featureWarmupScope,
                                             trainingConfig.donchianLookback,
                                             resumeConfig.has_value()
                                                 ? resumeConfig->inputWidthExpansionProvenance
                                                 : std::nullopt,
                                             trainingConfig.trainingObjective);
                    DBIO::PgModelIO::bindProducerWorkerAttempt(
                        wSave, modelId, launchArgs.schedulerWorkerAttemptId);
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
