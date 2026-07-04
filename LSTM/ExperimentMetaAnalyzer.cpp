#include "../Headers/ExperimentMetaAnalyzer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pqxx/pqxx>

#include "CanonicalSymbol.hpp"
#include "SupportedSymbols.hpp"

namespace EA::ExperimentMetaAnalyzer
{

namespace
{
constexpr int kMetaRecommendationCheckpointInterval = 20;
constexpr const char* kMetaRecommendationTrainStart = "2010-01-01";
constexpr const char* kMetaRecommendationTrainEnd = "2025-01-01";
constexpr const char* kMetaRecommendationInferStart = "2025-01-01";
constexpr const char* kMetaRecommendationInferEnd = "2026-01-01";

volatile std::sig_atomic_t gMetaAnalysisInterrupted = 0;

void HandleMetaAnalysisSignal(int)
{
    gMetaAnalysisInterrupted = 1;
}
} // namespace

std::string CurrentUtcTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

bool IsMetaAnalysisCommand(int argc, const char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg{argv[i]};
        if (arg == "--meta-analyze" ||
            arg == "--meta-analyze-once" ||
            arg == "--meta-analysis-report" ||
            arg == "--queue-meta-recommendations")
            return true;
    }
    return false;
}

std::string GetEnvOrDefault(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return (value && *value) ? std::string{value} : std::string{fallback};
}

std::string LstmDbConnectionString()
{
    return "hostaddr=" + GetEnvOrDefault("LSTM_DB_HOST", "127.0.0.1") +
           " user=pqxx dbname=" + GetEnvOrDefault("LSTM_DB_NAME", "LSTM");
}

bool SplitOptionWithValue(const std::string& arg,
                                 const std::string& optionName,
                                 std::string& value)
{
    const std::string prefix = optionName + "=";
    if (arg.rfind(prefix, 0) != 0)
        return false;
    value = arg.substr(prefix.size());
    return true;
}

std::string RequireNextArg(int argc, const char* argv[], int& i, const std::string& optionName)
{
    if (i + 1 >= argc)
        throw std::invalid_argument(optionName + " requires a value");
    return argv[++i];
}

int ParsePositiveInt(const std::string& optionName, const std::string& value)
{
    size_t consumed = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size() || parsed <= 0 || parsed > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return static_cast<int>(parsed);
}

std::string RecommendationEpochPolicyName(RecommendationEpochPolicy policy)
{
    switch (policy)
    {
        case RecommendationEpochPolicy::Leader:
            return "leader";
        case RecommendationEpochPolicy::Highest:
            return "highest";
    }
    return "highest";
}

RecommendationEpochPolicy ParseRecommendationEpochPolicy(const std::string& value)
{
    if (value == "leader")
        return RecommendationEpochPolicy::Leader;
    if (value == "highest")
        return RecommendationEpochPolicy::Highest;
    throw std::invalid_argument("--recommendation-epoch-policy must be leader or highest");
}

MetaAnalysisOptions ParseMetaAnalysisArgs(int argc, const char* argv[])
{
    MetaAnalysisOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg{argv[i]};
        std::string value;
        if (arg == "--meta-analyze")
            options.metaAnalyze = true;
        else if (arg == "--meta-analyze-once")
            options.metaAnalyzeOnce = true;
        else if (arg == "--meta-analysis-report")
            options.metaAnalysisReport = true;
        else if (arg == "--queue-meta-recommendations")
            options.queueMetaRecommendations = true;
        else if (arg == "--meta-analysis-json")
            options.outputJson = true;
        else if (arg == "--meta-analysis-markdown")
            options.outputMarkdown = true;
        else if (arg == "--dry-run")
            options.dryRun = true;
        else if (arg == "--meta-analysis-limit")
            options.limit = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--meta-analysis-symbol")
            options.symbol = EA::CanonicalSymbol::Normalize(RequireNextArg(argc, argv, i, arg));
        else if (arg == "--meta-analysis-horizon")
            options.horizon = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--meta-analysis-output")
            options.outputFile = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--meta-analysis-interval")
            options.intervalSeconds = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--recommendation-epoch-policy")
            options.recommendationEpochPolicy =
                ParseRecommendationEpochPolicy(RequireNextArg(argc, argv, i, arg));
        else if (SplitOptionWithValue(arg, "--meta-analysis-limit", value))
            options.limit = ParsePositiveInt("--meta-analysis-limit", value);
        else if (SplitOptionWithValue(arg, "--meta-analysis-symbol", value))
            options.symbol = EA::CanonicalSymbol::Normalize(value);
        else if (SplitOptionWithValue(arg, "--meta-analysis-horizon", value))
            options.horizon = ParsePositiveInt("--meta-analysis-horizon", value);
        else if (SplitOptionWithValue(arg, "--meta-analysis-output", value))
            options.outputFile = value;
        else if (SplitOptionWithValue(arg, "--meta-analysis-interval", value))
            options.intervalSeconds = ParsePositiveInt("--meta-analysis-interval", value);
        else if (SplitOptionWithValue(arg, "--recommendation-epoch-policy", value))
            options.recommendationEpochPolicy = ParseRecommendationEpochPolicy(value);
        else if (arg.rfind("--", 0) == 0)
            throw std::invalid_argument("unknown meta-analysis option '" + arg + "'");
        else
            throw std::invalid_argument("unexpected positional meta-analysis argument '" + arg + "'");
    }

    const int commandCount = (options.metaAnalyze ? 1 : 0) +
                             (options.metaAnalyzeOnce ? 1 : 0) +
                             (options.metaAnalysisReport ? 1 : 0) +
                             (options.queueMetaRecommendations ? 1 : 0);
    if (commandCount != 1)
        throw std::invalid_argument("expected exactly one meta-analysis command");
    if (options.limit <= 0)
        throw std::invalid_argument("--meta-analysis-limit must be positive");
    if (options.intervalSeconds <= 0)
        throw std::invalid_argument("--meta-analysis-interval must be positive");
    return options;
}

bool TableExists(pqxx::work& w, const std::string& tableName)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = $1 LIMIT 1;",
        tableName);
    return !r.empty();
}

bool RequireMetaAnalysisTables(pqxx::work& w)
{
    const std::vector<std::string> required = {
        "experiment",
        "experiment_analysis_result",
        "inference_eval_result",
        "model",
        "experiment_meta_analysis"
    };
    for (const auto& table : required)
    {
        if (!TableExists(w, table))
        {
            std::cerr << "DATABASE_MIGRATION_REQUIRED"
                      << ",table=" << table
                      << ",command=./migrate_lstm_db.sh"
                      << std::endl;
            return false;
        }
    }
    return true;
}

std::string FormatDouble(double value)
{
    if (!std::isfinite(value))
        return "null";
    std::ostringstream oss;
    oss << std::setprecision(10) << value;
    return oss.str();
}

std::string FormatDoubleFull(double value)
{
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

std::string FormatOptionalDouble(const std::optional<double>& value)
{
    return value.has_value() ? FormatDouble(*value) : "null";
}

std::string FormatMetricJson(const ScalarStats& stats, double value)
{
    return stats.count > 0 ? FormatDouble(value) : "null";
}

std::string FormatMetricMarkdown(const ScalarStats& stats, double value)
{
    return stats.count > 0 ? FormatDouble(value) : "n/a";
}

std::string FormatOptionalInt(const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "null";
}

std::string FormatOptionalLongLong(const std::optional<long long>& value)
{
    return value.has_value() ? std::to_string(*value) : "null";
}

std::string JsonEscape(const std::string& input)
{
    std::ostringstream out;
    for (const char ch : input)
    {
        switch (ch)
        {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20)
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setfill(' ');
                else
                    out << ch;
                break;
        }
    }
    return out.str();
}

std::string JsonString(const std::string& value)
{
    return "\"" + JsonEscape(value) + "\"";
}

std::optional<double> OptionalDoubleCell(const pqxx::row& row, size_t idx)
{
    if (row[idx].is_null())
        return std::nullopt;
    return row[idx].as<double>();
}

std::optional<int> OptionalIntCell(const pqxx::row& row, size_t idx)
{
    if (row[idx].is_null())
        return std::nullopt;
    return row[idx].as<int>();
}

std::optional<long long> OptionalLongLongCell(const pqxx::row& row, size_t idx)
{
    if (row[idx].is_null())
        return std::nullopt;
    return row[idx].as<long long>();
}

std::optional<std::string> OptionalStringCell(const pqxx::row& row, size_t idx)
{
    if (row[idx].is_null())
        return std::nullopt;
    return row[idx].as<std::string>();
}

ScalarStats ComputeStats(std::vector<double> values, bool higherIsBetter = true)
{
    ScalarStats stats;
    values.erase(std::remove_if(values.begin(), values.end(), [](double v) {
        return !std::isfinite(v);
    }), values.end());
    if (values.empty())
        return stats;

    stats.count = values.size();
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    stats.mean = total / static_cast<double>(values.size());
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    stats.median = (values.size() % 2 == 0) ? (values[mid - 1] + values[mid]) * 0.5 : values[mid];
    stats.worst = higherIsBetter ? values.front() : values.back();
    stats.best = higherIsBetter ? values.back() : values.front();

    double variance = 0.0;
    for (const double value : values)
        variance += (value - stats.mean) * (value - stats.mean);
    stats.stddev = std::sqrt(variance / static_cast<double>(values.size()));
    return stats;
}

std::string ConfidenceForSampleSize(size_t n)
{
    if (n >= 30)
        return "Very High";
    if (n >= 15)
        return "High";
    if (n >= 8)
        return "Moderate";
    if (n >= 3)
        return "Low";
    return "Insufficient Evidence";
}

std::string ScopeForOptions(const MetaAnalysisOptions& options)
{
    std::ostringstream scope;
    scope << "all";
    if (options.symbol.has_value())
        scope << ";symbol=" << *options.symbol;
    if (options.horizon.has_value())
        scope << ";horizon=" << *options.horizon;
    return scope.str();
}

std::string GroupKeyDouble(const std::optional<double>& value)
{
    return value.has_value() ? FormatDouble(*value) : "missing";
}

std::string GroupKeyInt(const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "missing";
}

std::vector<ExperimentRecord> LoadExperimentRecords(pqxx::work& w,
                                                           const MetaAnalysisOptions& options)
{
    std::ostringstream sql;
    sql << "SELECT e.experiment_id, e.symbol, e.prediction_horizon, e.c_next_threshold, "
        << "e.core_lr_mult, e.head_lr_mult, e.target_epochs, e.checkpoint_interval, "
        << "e.status, e.phase, e.resume_model_id, e.last_model_id, "
        << "a.model_id, a.completed_epochs, a.train_accuracy, a.validation_accuracy, "
        << "a.infer_accuracy, a.accept_rate, a.accept_accuracy, a.leader_score, m.name, "
        << "e.created_at::text, e.completed_at::text "
        << "FROM experiment e "
        << "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
        << "LEFT JOIN model m ON m.model_id = COALESCE(a.model_id, e.last_model_id) "
        << "WHERE 1=1 ";
    if (options.symbol.has_value())
        sql << "AND e.symbol = " << w.quote(*options.symbol) << " ";
    if (options.horizon.has_value())
        sql << "AND e.prediction_horizon = " << *options.horizon << " ";
    sql << "ORDER BY e.experiment_id, a.model_id NULLS LAST;";

    pqxx::result rows = w.exec(sql.str());
    std::vector<ExperimentRecord> records;
    records.reserve(rows.size());
    for (const auto& row : rows)
    {
        ExperimentRecord record;
        record.experimentId = row[0].as<long long>();
        record.symbol = row[1].as<std::string>();
        record.horizon = row[2].as<int>();
        record.threshold = row[3].as<double>();
        record.coreLr = OptionalDoubleCell(row, 4);
        record.headLr = OptionalDoubleCell(row, 5);
        record.targetEpochs = row[6].as<int>();
        record.checkpointInterval = row[7].as<int>();
        record.status = row[8].as<std::string>();
        record.phase = row[9].as<std::string>();
        record.resumed = !row[10].is_null();
        record.modelId = OptionalLongLongCell(row, 12);
        if (!record.modelId.has_value())
            record.modelId = OptionalLongLongCell(row, 11);
        record.completedEpochs = OptionalIntCell(row, 13);
        record.trainAccuracy = OptionalDoubleCell(row, 14);
        record.validationAccuracy = OptionalDoubleCell(row, 15);
        record.inferAccuracy = OptionalDoubleCell(row, 16);
        record.acceptRate = OptionalDoubleCell(row, 17);
        record.acceptAccuracy = OptionalDoubleCell(row, 18);
        record.leaderScore = OptionalDoubleCell(row, 19);
        record.modelName = OptionalStringCell(row, 20).value_or("");
        record.createdAt = OptionalStringCell(row, 21);
        record.completedAt = OptionalStringCell(row, 22);
        record.dataSource = "scheduler";
        records.push_back(std::move(record));
    }
    return records;
}

double PredictionImbalancePenaltyFromFractions(const std::optional<double>& predDown,
                                               const std::optional<double>& predNeutral,
                                               const std::optional<double>& predUp)
{
    if (!predDown.has_value() || !predNeutral.has_value() || !predUp.has_value())
        return 1.0;
    const double maxPredFrac = std::max({*predDown, *predNeutral, *predUp});
    if (maxPredFrac <= 0.60)
        return 1.0;
    return std::max(0.25, 1.0 - ((maxPredFrac - 0.60) / 0.40));
}

std::optional<double> ComputeLegacyLeaderScore(const std::optional<double>& inferAccuracy,
                                               const std::optional<double>& acceptAccuracy,
                                               const std::optional<double>& predDown,
                                               const std::optional<double>& predNeutral,
                                               const std::optional<double>& predUp)
{
    if (!inferAccuracy.has_value())
        return std::nullopt;
    const double acceptedAccuracy = acceptAccuracy.value_or(*inferAccuracy);
    const double penalty = PredictionImbalancePenaltyFromFractions(predDown, predNeutral, predUp);
    return (*inferAccuracy) * (0.75 + 0.25 * acceptedAccuracy) * penalty;
}

std::vector<ExperimentRecord> LoadLegacyExperimentRecords(pqxx::work& w,
                                                          const MetaAnalysisOptions& options,
                                                          const std::set<long long>& schedulerModelIds,
                                                          long long& legacyModelCount,
                                                          long long& duplicateModelsSkipped)
{
    std::ostringstream sql;
    sql << "WITH cfg AS ("
        << "  SELECT model_id,"
        << "         max(value) FILTER (WHERE col_idx = 1) AS prediction_horizon,"
        << "         max(value) FILTER (WHERE col_idx = 2) AS threshold_logret,"
        << "         max(value) FILTER (WHERE col_idx = 10) AS completed_epochs,"
        << "         max(value) FILTER (WHERE col_idx = 11) AS core_lr_mult,"
        << "         max(value) FILTER (WHERE col_idx = 12) AS head_weight_lr_mult "
        << "  FROM matrix "
        << "  WHERE param_name = 'train_config_meta' AND row_idx = 0 "
        << "  GROUP BY model_id"
        << "), sym AS ("
        << "  SELECT model_id, string_agg(chr(round(value)::int), '' ORDER BY col_idx) AS symbol "
        << "  FROM matrix "
        << "  WHERE param_name = 'train_symbol_meta' AND row_idx = 0 "
        << "  GROUP BY model_id"
        << "), latest_eval AS ("
        << "  SELECT DISTINCT ON (model_id) model_id, completed_epochs AS eval_completed_epochs, accuracy, "
        << "         accept_model, pred_down, pred_neutral, pred_up "
        << "  FROM inference_eval_result "
        << "  WHERE status = 'completed' "
        << "  ORDER BY model_id, completed_at DESC, id DESC"
        << ") "
        << "SELECT m.model_id, m.name, m.comment, sym.symbol, "
        << "       cfg.prediction_horizon, cfg.threshold_logret, cfg.completed_epochs, "
        << "       cfg.core_lr_mult, cfg.head_weight_lr_mult, "
        << "       latest_eval.eval_completed_epochs, latest_eval.accuracy, latest_eval.accept_model, "
        << "       latest_eval.pred_down, latest_eval.pred_neutral, latest_eval.pred_up "
        << "FROM model m "
        << "LEFT JOIN cfg ON cfg.model_id = m.model_id "
        << "LEFT JOIN sym ON sym.model_id = m.model_id "
        << "LEFT JOIN latest_eval ON latest_eval.model_id = m.model_id "
        << "WHERE cfg.model_id IS NOT NULL ";
    if (options.symbol.has_value())
        sql << "AND sym.symbol = " << w.quote(*options.symbol) << " ";
    if (options.horizon.has_value())
        sql << "AND cfg.prediction_horizon = " << *options.horizon << " ";
    sql << "ORDER BY m.model_id;";

    pqxx::result rows = w.exec(sql.str());
    legacyModelCount = static_cast<long long>(rows.size());
    std::vector<ExperimentRecord> records;
    records.reserve(rows.size());
    for (const auto& row : rows)
    {
        const long long modelId = row[0].as<long long>();
        if (schedulerModelIds.count(modelId))
        {
            ++duplicateModelsSkipped;
            continue;
        }

        ExperimentRecord record;
        record.experimentId = -modelId;
        record.modelId = modelId;
        record.modelName = OptionalStringCell(row, 1).value_or("");
        const std::string comment = OptionalStringCell(row, 2).value_or("");
        record.symbol = OptionalStringCell(row, 3).value_or("unknown");
        record.horizon = row[4].is_null() ? 0 : static_cast<int>(std::llround(row[4].as<double>()));
        record.threshold = OptionalDoubleCell(row, 5).value_or(0.0);
        record.completedEpochs = OptionalDoubleCell(row, 9).has_value()
            ? std::optional<int>{static_cast<int>(std::llround(*OptionalDoubleCell(row, 9)))}
            : (OptionalDoubleCell(row, 6).has_value()
                ? std::optional<int>{static_cast<int>(std::llround(*OptionalDoubleCell(row, 6)))}
                : std::nullopt);
        record.targetEpochs = record.completedEpochs.value_or(0);
        record.coreLr = OptionalDoubleCell(row, 7);
        record.headLr = OptionalDoubleCell(row, 8);
        record.status = "completed";
        record.phase = "done";
        record.resumed = (comment.find("resumed") != std::string::npos);
        record.inferAccuracy = OptionalDoubleCell(row, 10);
        if (!row[11].is_null())
            record.acceptRate = row[11].as<bool>() ? 1.0 : 0.0;
        record.acceptAccuracy = record.inferAccuracy;
        const std::optional<double> predDown = OptionalDoubleCell(row, 12);
        const std::optional<double> predNeutral = OptionalDoubleCell(row, 13);
        const std::optional<double> predUp = OptionalDoubleCell(row, 14);
        record.leaderScore = ComputeLegacyLeaderScore(record.inferAccuracy,
                                                      record.acceptAccuracy,
                                                      predDown,
                                                      predNeutral,
                                                      predUp);
        record.dataSource = "legacy_model";
        records.push_back(std::move(record));
    }
    return records;
}

long long CountDistinctCompletedModels(pqxx::work& w, const MetaAnalysisOptions& options)
{
    std::ostringstream sql;
    sql << "SELECT count(DISTINCT COALESCE(a.model_id, e.last_model_id)) "
        << "FROM experiment e "
        << "LEFT JOIN experiment_analysis_result a ON a.experiment_id = e.experiment_id "
        << "WHERE e.status = 'completed' "
        << "AND COALESCE(a.model_id, e.last_model_id) IS NOT NULL ";
    if (options.symbol.has_value())
        sql << "AND e.symbol = " << w.quote(*options.symbol) << " ";
    if (options.horizon.has_value())
        sql << "AND e.prediction_horizon = " << *options.horizon << " ";
    return w.exec(sql.str())[0][0].as<long long>();
}

void LoadExperimentCounts(pqxx::work& w,
                                 const MetaAnalysisOptions& options,
                                 MetaAnalysisResult& result)
{
    std::ostringstream sql;
    sql << "SELECT count(*), "
        << "count(*) FILTER (WHERE status = 'completed'), "
        << "count(*) FILTER (WHERE status = 'failed'), "
        << "count(*) FILTER (WHERE status = 'running'), "
        << "count(*) FILTER (WHERE status = 'pending') "
        << "FROM experiment WHERE 1=1 ";
    if (options.symbol.has_value())
        sql << "AND symbol = " << w.quote(*options.symbol) << " ";
    if (options.horizon.has_value())
        sql << "AND prediction_horizon = " << *options.horizon << " ";

    pqxx::row row = w.exec(sql.str())[0];
    result.totalExperiments = row[0].as<long long>();
    result.completedExperiments = row[1].as<long long>();
    result.failedExperiments = row[2].as<long long>();
    result.runningExperiments = row[3].as<long long>();
    result.pendingExperiments = row[4].as<long long>();
    result.completedModels = CountDistinctCompletedModels(w, options);
    const long long terminal = result.completedExperiments + result.failedExperiments;
    result.successRate = terminal > 0
        ? static_cast<double>(result.completedExperiments) / static_cast<double>(terminal)
        : 0.0;
}

std::set<long long> SchedulerModelIds(const std::vector<ExperimentRecord>& records)
{
    std::set<long long> modelIds;
    for (const auto& record : records)
    {
        if (record.modelId.has_value())
            modelIds.insert(*record.modelId);
    }
    return modelIds;
}

long long CountCompletedRecords(const std::vector<ExperimentRecord>& records)
{
    return static_cast<long long>(std::count_if(records.begin(), records.end(), [](const ExperimentRecord& record) {
        return record.status == "completed";
    }));
}

long long CountDistinctCompletedRecordModels(const std::vector<ExperimentRecord>& records)
{
    std::set<long long> modelIds;
    for (const auto& record : records)
    {
        if (record.status == "completed" && record.modelId.has_value())
            modelIds.insert(*record.modelId);
    }
    return static_cast<long long>(modelIds.size());
}

bool HasRankingMetric(const ExperimentRecord& record)
{
    return (record.leaderScore.has_value() && std::isfinite(*record.leaderScore)) ||
           (record.inferAccuracy.has_value() && std::isfinite(*record.inferAccuracy));
}

long long CountRecordsWithMetrics(const std::vector<ExperimentRecord>& records,
                                  const std::string& dataSource)
{
    return static_cast<long long>(std::count_if(records.begin(), records.end(), [&](const ExperimentRecord& record) {
        return record.dataSource == dataSource && HasRankingMetric(record);
    }));
}

long long CountRecordsWithMetrics(const std::vector<ExperimentRecord>& records)
{
    return static_cast<long long>(std::count_if(records.begin(), records.end(), [](const ExperimentRecord& record) {
        return HasRankingMetric(record);
    }));
}

void ApplyMergedCounts(MetaAnalysisResult& result,
                       const std::vector<ExperimentRecord>& schedulerRecords,
                       long long legacyModelCount,
                       long long duplicateModelsSkipped)
{
    result.dataSources.schedulerExperiments = static_cast<long long>(schedulerRecords.size());
    result.dataSources.legacyModels = legacyModelCount;
    result.dataSources.duplicateModelsSkipped = duplicateModelsSkipped;
    result.dataSources.mergedRecords = static_cast<long long>(result.records.size());
    const long long possibleRecords = result.dataSources.schedulerExperiments + result.dataSources.legacyModels;
    result.dataSources.coveragePercentage = possibleRecords > 0
        ? 100.0 * static_cast<double>(result.dataSources.mergedRecords) / static_cast<double>(possibleRecords)
        : 0.0;
    result.dataSources.legacyModelsWithInferenceMetrics = CountRecordsWithMetrics(result.records, "legacy_model");
    result.dataSources.legacyModelsMissingInferenceMetrics =
        std::max<long long>(0, result.dataSources.legacyModels - result.dataSources.legacyModelsWithInferenceMetrics);
    result.dataSources.schedulerRecordsWithMetrics = CountRecordsWithMetrics(result.records, "scheduler");
    result.dataSources.mergedRecordsWithMetrics = CountRecordsWithMetrics(result.records);
    result.dataSources.metricCoveragePercentage = result.dataSources.mergedRecords > 0
        ? 100.0 * static_cast<double>(result.dataSources.mergedRecordsWithMetrics) /
          static_cast<double>(result.dataSources.mergedRecords)
        : 0.0;

    result.completedExperiments = CountCompletedRecords(result.records);
    result.completedModels = CountDistinctCompletedRecordModels(result.records);
    result.totalExperiments = result.completedExperiments +
                              result.failedExperiments +
                              result.runningExperiments +
                              result.pendingExperiments;
    const long long terminal = result.completedExperiments + result.failedExperiments;
    result.successRate = terminal > 0
        ? static_cast<double>(result.completedExperiments) / static_cast<double>(terminal)
        : 0.0;
}

void AddMetricValue(std::vector<double>& values, const std::optional<double>& value)
{
    if (value.has_value() && std::isfinite(*value))
        values.push_back(*value);
}

GroupStats ComputeGroupStats(const std::string& dimension,
                                    const std::string& key,
                                    const std::vector<const ExperimentRecord*>& rows)
{
    std::vector<double> leaderScore;
    std::vector<double> inferAccuracy;
    std::vector<double> acceptRate;
    std::vector<double> acceptAccuracy;
    for (const ExperimentRecord* row : rows)
    {
        AddMetricValue(leaderScore, row->leaderScore);
        AddMetricValue(inferAccuracy, row->inferAccuracy);
        AddMetricValue(acceptRate, row->acceptRate);
        AddMetricValue(acceptAccuracy, row->acceptAccuracy);
    }

    GroupStats stats;
    stats.dimension = dimension;
    stats.key = key;
    stats.sampleCount = rows.size();
    stats.leaderScore = ComputeStats(std::move(leaderScore));
    stats.inferAccuracy = ComputeStats(std::move(inferAccuracy));
    stats.acceptRate = ComputeStats(std::move(acceptRate));
    stats.acceptAccuracy = ComputeStats(std::move(acceptAccuracy));
    return stats;
}

template <typename KeyFn>
void BuildGrouping(const std::vector<ExperimentRecord>& records,
                          const std::string& dimension,
                          KeyFn keyFn,
                          std::vector<GroupStats>& out)
{
    std::map<std::string, std::vector<const ExperimentRecord*>> groups;
    for (const auto& record : records)
    {
        if (record.status != "completed")
            continue;
        groups[keyFn(record)].push_back(&record);
    }
    for (const auto& [key, rows] : groups)
        out.push_back(ComputeGroupStats(dimension, key, rows));
}

std::vector<GroupStats> BuildAllGroupStats(const std::vector<ExperimentRecord>& records)
{
    std::vector<GroupStats> stats;
    BuildGrouping(records, "symbol", [](const ExperimentRecord& r) { return r.symbol; }, stats);
    BuildGrouping(records, "prediction_horizon", [](const ExperimentRecord& r) { return "H" + std::to_string(r.horizon); }, stats);
    BuildGrouping(records, "core_lr_mult", [](const ExperimentRecord& r) { return GroupKeyDouble(r.coreLr); }, stats);
    BuildGrouping(records, "head_lr_mult", [](const ExperimentRecord& r) { return GroupKeyDouble(r.headLr); }, stats);
    BuildGrouping(records, "target_epochs", [](const ExperimentRecord& r) { return std::to_string(r.targetEpochs); }, stats);
    BuildGrouping(records, "threshold", [](const ExperimentRecord& r) { return FormatDouble(r.threshold); }, stats);
    BuildGrouping(records, "checkpoint_interval", [](const ExperimentRecord& r) { return std::to_string(r.checkpointInterval); }, stats);
    BuildGrouping(records, "resume_mode", [](const ExperimentRecord& r) { return r.resumed ? "resume" : "fresh"; }, stats);
    return stats;
}

std::vector<LeaderRow> BuildLeaders(const std::vector<ExperimentRecord>& records, int limit)
{
    std::vector<LeaderRow> leaders;
    for (const auto& record : records)
    {
        if (record.status != "completed")
            continue;
        if (!HasRankingMetric(record))
            continue;
        const double rankingScore = record.leaderScore.value_or(record.inferAccuracy.value_or(-1.0));
        if (rankingScore <= 0.0)
            continue;
        LeaderRow row;
        row.experimentId = record.experimentId;
        row.modelId = record.modelId;
        row.symbol = record.symbol;
        row.horizon = record.horizon;
        row.targetEpochs = record.completedEpochs.value_or(record.targetEpochs);
        row.inferAccuracy = record.inferAccuracy;
        row.acceptAccuracy = record.acceptAccuracy;
        row.acceptRate = record.acceptRate;
        row.leaderScore = record.leaderScore;
        row.modelName = record.modelName;
        leaders.push_back(std::move(row));
    }
    std::sort(leaders.begin(), leaders.end(), [](const LeaderRow& a, const LeaderRow& b) {
        const double as = a.leaderScore.value_or(a.inferAccuracy.value_or(-1.0));
        const double bs = b.leaderScore.value_or(b.inferAccuracy.value_or(-1.0));
        if (as != bs)
            return as > bs;
        return a.experimentId < b.experimentId;
    });
    if (static_cast<int>(leaders.size()) > limit)
        leaders.resize(static_cast<size_t>(limit));
    return leaders;
}

std::string PlateauKey(const ExperimentRecord& record)
{
    std::ostringstream key;
    key << record.symbol
        << "|H" << record.horizon
        << "|threshold=" << FormatDouble(record.threshold)
        << "|core=" << GroupKeyDouble(record.coreLr)
        << "|head=" << GroupKeyDouble(record.headLr);
    return key.str();
}

std::vector<PlateauSignal> DetectPlateaus(const std::vector<ExperimentRecord>& records)
{
    std::map<std::string, std::map<int, std::vector<double>>> series;
    std::map<std::string, size_t> sampleCounts;
    for (const auto& record : records)
    {
        if (record.status != "completed")
            continue;
        const auto score = record.leaderScore.has_value() ? record.leaderScore : record.inferAccuracy;
        if (!score.has_value() || !std::isfinite(*score))
            continue;
        if (*score <= 0.0)
            continue;
        const int epoch = record.completedEpochs.value_or(record.targetEpochs);
        const std::string key = PlateauKey(record);
        series[key][epoch].push_back(*score);
        ++sampleCounts[key];
    }

    std::vector<PlateauSignal> signals;
    for (const auto& [key, epochMap] : series)
    {
        if (epochMap.size() < 2)
            continue;
        std::vector<std::pair<int, double>> points;
        for (const auto& [epoch, values] : epochMap)
        {
            if (values.empty())
                continue;
            const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                                static_cast<double>(values.size());
            points.emplace_back(epoch, mean);
        }
        if (points.size() < 2)
            continue;
        std::sort(points.begin(), points.end());
        const auto& earlier = points[points.size() - 2];
        const auto& later = points[points.size() - 1];
        PlateauSignal signal;
        signal.key = key;
        signal.earlierEpoch = earlier.first;
        signal.laterEpoch = later.first;
        signal.earlierScore = earlier.second;
        signal.laterScore = later.second;
        signal.delta = later.second - earlier.second;
        if (signal.delta < -0.005)
            signal.classification = "regression";
        else if (std::abs(signal.delta) <= 0.005)
            signal.classification = "plateau";
        else if (signal.delta >= 0.015)
            signal.classification = "consistent_improvement";
        else
            signal.classification = "small_improvement";
        signal.confidence = ConfidenceForSampleSize(sampleCounts[key]);
        signals.push_back(std::move(signal));
    }
    std::sort(signals.begin(), signals.end(), [](const PlateauSignal& a, const PlateauSignal& b) {
        return std::abs(a.delta) < std::abs(b.delta);
    });
    return signals;
}

const GroupStats* BestGroup(const std::vector<GroupStats>& groups, const std::string& dimension)
{
    const GroupStats* best = nullptr;
    for (const auto& group : groups)
    {
        const ScalarStats& rankingStats = group.leaderScore.count > 0 ? group.leaderScore : group.inferAccuracy;
        if (group.dimension != dimension || rankingStats.count == 0)
            continue;
        const ScalarStats& bestStats = best
            ? (best->leaderScore.count > 0 ? best->leaderScore : best->inferAccuracy)
            : group.leaderScore;
        if (!best || rankingStats.best > bestStats.best)
            best = &group;
    }
    return best;
}

std::vector<std::string> KnownSymbols()
{
    return EA::SupportedSymbols::TrainingSymbols();
}

std::optional<std::string> SymbolFromScope(const std::string& scope)
{
    const std::string marker = "symbol=";
    const size_t start = scope.find(marker);
    if (start == std::string::npos)
        return std::nullopt;
    const size_t valueStart = start + marker.size();
    const size_t valueEnd = scope.find(';', valueStart);
    return scope.substr(valueStart, valueEnd == std::string::npos
        ? std::string::npos
        : valueEnd - valueStart);
}

std::vector<Recommendation> BuildRecommendations(const MetaAnalysisResult& result)
{
    std::vector<Recommendation> recs;
    int priority = 1;
    const std::string globalConfidence = ConfidenceForSampleSize(static_cast<size_t>(result.completedExperiments));

    if (result.completedExperiments < 5)
    {
        recs.push_back(Recommendation{
            priority++,
            "Collect more completed experiments before making strong parameter conclusions.",
            "The completed experiment sample is too small for stable cross-experiment inference.",
            "completed_experiments=" + std::to_string(result.completedExperiments),
            "High information gain from filling the initial evidence base.",
            "Insufficient Evidence"
        });
    }

    if (result.dataSources.mergedRecords > 0 &&
        result.dataSources.mergedRecordsWithMetrics < result.dataSources.mergedRecords)
    {
        recs.push_back(Recommendation{
            priority++,
            "Run inference analysis for completed models missing metrics.",
            "Historical model coverage is high, but performance coverage is limited; missing inference rows are excluded from rankings and averages.",
            "merged_records=" + std::to_string(result.dataSources.mergedRecords) +
                ", records_with_metrics=" + std::to_string(result.dataSources.mergedRecordsWithMetrics) +
                ", legacy_missing_inference_metrics=" +
                std::to_string(result.dataSources.legacyModelsMissingInferenceMetrics),
            "High: converts existing historical training runs into comparable performance evidence.",
            ConfidenceForSampleSize(static_cast<size_t>(result.dataSources.mergedRecordsWithMetrics))
        });
    }

    for (const auto& group : result.groupStats)
    {
        if (group.dimension == "prediction_horizon" &&
            group.sampleCount >= 5 &&
            group.inferAccuracy.count == 0)
        {
            recs.push_back(Recommendation{
                priority++,
                "Populate inference_eval_result for " + group.key + " completed models.",
                "This horizon has completed historical models but no inference metrics, so it cannot be compared honestly yet.",
                "samples=" + std::to_string(group.sampleCount) + ", infer_metric_count=0",
                "High: enables horizon comparisons without treating missing metrics as zero performance.",
                "Insufficient Evidence"
            });
            break;
        }
    }

    if (!result.leaders.empty())
    {
        const LeaderRow& leader = result.leaders.front();
        recs.push_back(Recommendation{
            priority++,
            "Repeat or extend the current top configuration: " + leader.symbol +
                " H" + std::to_string(leader.horizon) +
                " epoch " + std::to_string(leader.targetEpochs) + ".",
            "It is the current leader by stored leader_score without changing the ranking formula.",
            "leader_score=" + FormatOptionalDouble(leader.leaderScore) +
                ", infer_accuracy=" + FormatOptionalDouble(leader.inferAccuracy),
            "Moderate: validates whether the result is repeatable or benefits from more epochs.",
            globalConfidence
        });
    }

    if (const GroupStats* bestHorizon = BestGroup(result.groupStats, "prediction_horizon"))
    {
        const ScalarStats& rankingStats = bestHorizon->leaderScore.count > 0
            ? bestHorizon->leaderScore
            : bestHorizon->inferAccuracy;
        recs.push_back(Recommendation{
            priority++,
            "Prioritize additional experiments for " + bestHorizon->key + ".",
            "This horizon currently has the best observed metric among horizon groups with inference evidence.",
            "best_metric=" + FormatDouble(rankingStats.best) +
                ", samples=" + std::to_string(bestHorizon->sampleCount) +
                ", metric_count=" + std::to_string(rankingStats.count),
            "High if paired with under-sampled symbols.",
            ConfidenceForSampleSize(rankingStats.count)
        });
    }

    if (const GroupStats* bestSymbol = BestGroup(result.groupStats, "symbol"))
    {
        const ScalarStats& rankingStats = bestSymbol->leaderScore.count > 0
            ? bestSymbol->leaderScore
            : bestSymbol->inferAccuracy;
        recs.push_back(Recommendation{
            priority++,
            "Use " + bestSymbol->key + " as the near-term benchmark symbol.",
            "It has the strongest current symbol-level result among groups with inference evidence.",
            "best_metric=" + FormatDouble(rankingStats.best) +
                ", samples=" + std::to_string(bestSymbol->sampleCount) +
                ", metric_count=" + std::to_string(rankingStats.count),
            "Moderate: benchmark stability improves comparisons across horizons and LR settings.",
            ConfidenceForSampleSize(rankingStats.count)
        });
    }

    for (const auto& signal : result.plateauSignals)
    {
        if (signal.classification == "plateau")
        {
            recs.push_back(Recommendation{
                priority++,
                "Do not extend this configuration until replicated: " + signal.key + ".",
                "Latest epoch interval shows a plateau.",
                "epoch_" + std::to_string(signal.earlierEpoch) + "_score=" + FormatDouble(signal.earlierScore) +
                    ", epoch_" + std::to_string(signal.laterEpoch) + "_score=" + FormatDouble(signal.laterScore) +
                    ", delta=" + FormatDouble(signal.delta),
                "Low-to-moderate: run a replicate or redirect epochs to another horizon/symbol.",
                signal.confidence
            });
            break;
        }
    }

    std::set<std::string> exploredSymbols;
    std::set<std::string> exploredSymbolHorizon;
    for (const auto& record : result.records)
    {
        if (record.status != "completed")
            continue;
        exploredSymbols.insert(record.symbol);
        exploredSymbolHorizon.insert(record.symbol + "|H" + std::to_string(record.horizon));
    }

    if (exploredSymbols.empty())
    {
        if (const auto scopedSymbol = SymbolFromScope(result.scope))
        {
            recs.push_back(Recommendation{
                priority++,
                "Run initial completed experiments for " + *scopedSymbol + ".",
                "No completed result exists for the requested symbol scope.",
                "symbol=" + *scopedSymbol + ", completed_samples=0",
                "High: fills the requested symbol-specific evidence gap.",
                "Insufficient Evidence"
            });
            return recs;
        }
    }

    for (const std::string& symbol : KnownSymbols())
    {
        if (!exploredSymbols.count(symbol))
        {
            recs.push_back(Recommendation{
                priority++,
                "Run initial completed experiments for " + symbol + ".",
                "No completed result exists for this symbol in the current analysis scope.",
                "symbol=" + symbol + ", completed_samples=0",
                "High: fills a coverage gap in symbol-level comparisons.",
                "Insufficient Evidence"
            });
            break;
        }
    }

    const std::vector<int> candidateHorizons = {4, 6, 8, 12};
    for (const auto& record : result.records)
    {
        if (record.status != "completed")
            continue;
        for (int horizon : candidateHorizons)
        {
            const std::string key = record.symbol + "|H" + std::to_string(horizon);
            if (!exploredSymbolHorizon.count(key))
            {
                recs.push_back(Recommendation{
                    priority++,
                    "Explore " + record.symbol + " H" + std::to_string(horizon) + ".",
                    "This symbol/horizon combination has no completed evidence yet.",
                    "missing_combination=" + key,
                    "Moderate: expands horizon comparison coverage.",
                    "Insufficient Evidence"
                });
                return recs;
            }
        }
    }

    return recs;
}

std::string ConfigDoubleKey(double value)
{
    std::ostringstream oss;
    oss << std::setprecision(12) << value;
    return oss.str();
}

int RecommendationTargetEpochs(const ExperimentRecord& record)
{
    return std::max(record.targetEpochs, record.completedEpochs.value_or(0));
}

std::string ExperimentConfigKey(const std::string& symbol,
                                int horizon,
                                int targetEpochs,
                                double threshold,
                                double coreLr,
                                double headLr)
{
    std::ostringstream key;
    key << symbol
        << "|H" << horizon
        << "|epochs=" << targetEpochs
        << "|threshold=" << ConfigDoubleKey(threshold)
        << "|core=" << ConfigDoubleKey(coreLr)
        << "|head=" << ConfigDoubleKey(headLr);
    return key.str();
}

std::optional<std::string> ExperimentConfigKey(const ExperimentRecord& record)
{
    if (record.symbol.empty() ||
        record.symbol == "unknown" ||
        record.horizon <= 0 ||
        RecommendationTargetEpochs(record) <= 0 ||
        record.threshold <= 0.0 ||
        !record.coreLr.has_value() ||
        !record.headLr.has_value())
    {
        return std::nullopt;
    }

    return ExperimentConfigKey(record.symbol,
                               record.horizon,
                               RecommendationTargetEpochs(record),
                               record.threshold,
                               *record.coreLr,
                               *record.headLr);
}

struct ExistingExperimentConfig
{
    std::string symbol;
    int horizon = 0;
    int targetEpochs = 0;
    double threshold = 0.0;
    double coreLr = 0.0;
    double headLr = 0.0;
};

bool NearlyEqual(double a, double b, double epsilon)
{
    return std::abs(a - b) <= epsilon;
}

bool MatchesExistingExperimentConfig(const ExistingExperimentConfig& existing,
                                     const std::string& symbol,
                                     int horizon,
                                     int targetEpochs,
                                     double threshold,
                                     double coreLr,
                                     double headLr)
{
    return existing.symbol == symbol &&
           existing.horizon == horizon &&
           existing.targetEpochs == targetEpochs &&
           NearlyEqual(existing.threshold, threshold, 1e-12) &&
           NearlyEqual(existing.coreLr, coreLr, 1e-9) &&
           NearlyEqual(existing.headLr, headLr, 1e-9);
}

std::vector<ExistingExperimentConfig> LoadExistingExperimentConfigs(pqxx::work& w)
{
    pqxx::result rows = w.exec(
        "SELECT symbol, prediction_horizon, target_epochs, "
        "c_next_threshold, core_lr_mult, head_lr_mult "
        "FROM experiment "
        "WHERE core_lr_mult IS NOT NULL "
        "AND head_lr_mult IS NOT NULL;");

    std::vector<ExistingExperimentConfig> configs;
    configs.reserve(rows.size());
    for (const auto& row : rows)
    {
        ExistingExperimentConfig config;
        config.symbol = row[0].as<std::string>();
        config.horizon = row[1].as<int>();
        config.targetEpochs = row[2].as<int>();
        config.threshold = row[3].as<double>();
        config.coreLr = row[4].as<double>();
        config.headLr = row[5].as<double>();
        configs.push_back(std::move(config));
    }
    return configs;
}

double RecommendationRankingScore(const ExperimentRecord& record)
{
    if (record.leaderScore.has_value() && std::isfinite(*record.leaderScore))
        return *record.leaderScore;
    if (record.inferAccuracy.has_value() && std::isfinite(*record.inferAccuracy))
        return *record.inferAccuracy;
    return -1.0;
}

std::vector<const ExperimentRecord*> RankedRecommendationSources(const std::vector<ExperimentRecord>& records)
{
    std::vector<const ExperimentRecord*> sources;
    for (const auto& record : records)
    {
        if (record.status != "completed")
            continue;
        if (!HasRankingMetric(record))
            continue;
        if (RecommendationRankingScore(record) <= 0.0)
            continue;
        sources.push_back(&record);
    }

    std::sort(sources.begin(), sources.end(), [](const ExperimentRecord* a, const ExperimentRecord* b) {
        const double as = RecommendationRankingScore(*a);
        const double bs = RecommendationRankingScore(*b);
        if (as != bs)
            return as > bs;
        return a->experimentId < b->experimentId;
    });
    return sources;
}

std::vector<const ExperimentRecord*> RankedHighestEpochRecommendationSources(const std::vector<ExperimentRecord>& records)
{
    std::map<std::string, std::vector<const ExperimentRecord*>> groups;
    for (const ExperimentRecord* source : RankedRecommendationSources(records))
    {
        const std::string key = source->symbol + "|H" + std::to_string(source->horizon);
        groups[key].push_back(source);
    }

    std::vector<const ExperimentRecord*> matureSources;
    for (const auto& [key, rows] : groups)
    {
        int highestEpoch = 0;
        for (const ExperimentRecord* row : rows)
            highestEpoch = std::max(highestEpoch, RecommendationTargetEpochs(*row));
        for (const ExperimentRecord* row : rows)
        {
            if (RecommendationTargetEpochs(*row) == highestEpoch)
            {
                matureSources.push_back(row);
                break;
            }
        }
    }

    std::sort(matureSources.begin(), matureSources.end(), [](const ExperimentRecord* a, const ExperimentRecord* b) {
        const double as = RecommendationRankingScore(*a);
        const double bs = RecommendationRankingScore(*b);
        if (as != bs)
            return as > bs;
        if (a->symbol != b->symbol)
            return a->symbol < b->symbol;
        if (a->horizon != b->horizon)
            return a->horizon < b->horizon;
        return a->experimentId < b->experimentId;
    });
    return matureSources;
}

std::vector<const ExperimentRecord*> RecommendationSourcesForPolicy(const std::vector<ExperimentRecord>& records,
                                                                    RecommendationEpochPolicy policy)
{
    if (policy == RecommendationEpochPolicy::Highest)
    {
        std::vector<const ExperimentRecord*> sources = RankedHighestEpochRecommendationSources(records);
        if (!sources.empty())
            return sources;
    }
    return RankedRecommendationSources(records);
}

void AddNextExperimentCandidate(std::vector<NextExperimentRecommendation>& out,
                                std::set<std::string>& emittedKeys,
                                const std::vector<ExistingExperimentConfig>& existingConfigs,
                                const ExperimentRecord& source,
                                double coreLr,
                                double headLr,
                                const std::string& reason,
                                int limit)
{
    if (static_cast<int>(out.size()) >= limit)
        return;
    if (coreLr <= 0.0 || headLr <= 0.0)
        return;

    const int targetEpochs = RecommendationTargetEpochs(source);
    const std::string key = ExperimentConfigKey(source.symbol,
                                                source.horizon,
                                                targetEpochs,
                                                source.threshold,
                                                coreLr,
                                                headLr);
    for (const auto& existing : existingConfigs)
    {
        if (MatchesExistingExperimentConfig(existing,
                                            source.symbol,
                                            source.horizon,
                                            targetEpochs,
                                            source.threshold,
                                            coreLr,
                                            headLr))
        {
            return;
        }
    }
    if (!emittedKeys.insert(key).second)
        return;

    NextExperimentRecommendation rec;
    rec.rank = static_cast<int>(out.size()) + 1;
    rec.symbol = source.symbol;
    rec.horizon = source.horizon;
    rec.targetEpochs = targetEpochs;
    rec.threshold = source.threshold;
    rec.coreLr = coreLr;
    rec.headLr = headLr;
    rec.reason = reason;
    rec.sourceLeaderExperimentId = source.experimentId;
    rec.sourceModelId = source.modelId;
    out.push_back(std::move(rec));
}

bool CanBuildNeighborhoodRecommendation(const ExperimentRecord& source)
{
    return source.symbol != "unknown" &&
           source.horizon > 0 &&
           RecommendationTargetEpochs(source) > 0 &&
           source.threshold > 0.0 &&
           source.coreLr.has_value() &&
           source.headLr.has_value();
}

void AddNeighborhoodRecommendations(std::vector<NextExperimentRecommendation>& out,
                                    std::set<std::string>& emittedKeys,
                                    const std::vector<ExistingExperimentConfig>& existingConfigs,
                                    const ExperimentRecord& source,
                                    const std::string& sourceReason,
                                    int limit)
{
    if (static_cast<int>(out.size()) >= limit)
        return;
    if (!CanBuildNeighborhoodRecommendation(source))
        return;

    const double core = *source.coreLr;
    const double head = *source.headLr;
    AddNextExperimentCandidate(out,
                               emittedKeys,
                               existingConfigs,
                               source,
                               core - 20.0,
                               head,
                               sourceReason + " / fills missing core_lr neighborhood",
                               limit);
    AddNextExperimentCandidate(out,
                               emittedKeys,
                               existingConfigs,
                               source,
                               core + 20.0,
                               head,
                               sourceReason + " / fills missing core_lr neighborhood",
                               limit);
    AddNextExperimentCandidate(out,
                               emittedKeys,
                               existingConfigs,
                               source,
                               core,
                               head - 10.0,
                               sourceReason + " / tests head_lr sensitivity near leader",
                               limit);
    AddNextExperimentCandidate(out,
                               emittedKeys,
                               existingConfigs,
                               source,
                               core,
                               head + 10.0,
                               sourceReason + " / tests head_lr sensitivity near leader",
                               limit);
}

std::vector<NextExperimentRecommendation> BuildNextExperimentRecommendations(const MetaAnalysisResult& result,
                                                                             const std::vector<ExistingExperimentConfig>& existingConfigs,
                                                                             RecommendationEpochPolicy epochPolicy,
                                                                             int limit,
                                                                             std::vector<std::string>& notes)
{
    std::vector<NextExperimentRecommendation> recs;
    std::set<std::string> emittedKeys;
    const std::vector<const ExperimentRecord*> sources =
        RecommendationSourcesForPolicy(result.records, epochPolicy);
    if (sources.empty())
    {
        notes.push_back("No completed analyzed records with ranking evidence are available for next-experiment recommendations.");
        return recs;
    }

    int skippedMissingConfig = 0;
    auto addSource = [&](const ExperimentRecord& source, const std::string& reason) {
        if (!CanBuildNeighborhoodRecommendation(source))
        {
            ++skippedMissingConfig;
            return;
        }
        AddNeighborhoodRecommendations(recs, emittedKeys, existingConfigs, source, reason, limit);
    };

    addSource(*sources.front(),
              epochPolicy == RecommendationEpochPolicy::Highest
                  ? "highest-epoch neighborhood around current leader group"
                  : "neighbor sweep around current global leader");

    std::set<std::string> symbolSeen;
    for (const ExperimentRecord* source : sources)
    {
        if (static_cast<int>(recs.size()) >= limit)
            break;
        if (!symbolSeen.insert(source->symbol).second)
            continue;
        addSource(*source,
                  epochPolicy == RecommendationEpochPolicy::Highest
                      ? "best mature result for symbol " + source->symbol +
                            " horizon " + std::to_string(source->horizon)
                      : "best result for symbol " + source->symbol);
    }

    std::set<int> horizonSeen;
    for (const ExperimentRecord* source : sources)
    {
        if (static_cast<int>(recs.size()) >= limit)
            break;
        if (!horizonSeen.insert(source->horizon).second)
            continue;
        addSource(*source,
                  epochPolicy == RecommendationEpochPolicy::Highest
                      ? "preferred highest completed epoch count for symbol/horizon"
                      : "best result for horizon " + std::to_string(source->horizon));
    }

    for (const ExperimentRecord* source : sources)
    {
        if (static_cast<int>(recs.size()) >= limit)
            break;
        addSource(*source, "promising completed configuration");
    }

    if (skippedMissingConfig > 0)
    {
        notes.push_back("Skipped " + std::to_string(skippedMissingConfig) +
                        " completed leader records because core_lr, head_lr, threshold, or target_epochs metadata was unavailable.");
    }
    if (recs.empty())
    {
        notes.push_back("No queueable nearby sweep candidates remained after duplicate filtering against existing experiment configurations.");
    }
    return recs;
}

std::optional<long long> FindExistingExperimentForRecommendation(pqxx::work& w,
                                                                 const NextExperimentRecommendation& rec)
{
    pqxx::result rows = w.exec(
        "SELECT experiment_id "
        "FROM experiment "
        "WHERE symbol = " + w.quote(rec.symbol) + " "
        "AND prediction_horizon = " + std::to_string(rec.horizon) + " "
        "AND target_epochs = " + std::to_string(rec.targetEpochs) + " "
        "AND abs(c_next_threshold - " + FormatDoubleFull(rec.threshold) + ") <= 1e-12 "
        "AND core_lr_mult IS NOT NULL "
        "AND abs(core_lr_mult - " + FormatDoubleFull(rec.coreLr) + ") <= 1e-9 "
        "AND head_lr_mult IS NOT NULL "
        "AND abs(head_lr_mult - " + FormatDoubleFull(rec.headLr) + ") <= 1e-9 "
        "ORDER BY experiment_id ASC LIMIT 1;");
    if (rows.empty())
        return std::nullopt;
    return rows[0][0].as<long long>();
}

long long InsertMetaRecommendationExperiment(pqxx::work& w,
                                             const NextExperimentRecommendation& rec)
{
    std::ostringstream sql;
    sql << "INSERT INTO experiment ("
        << "symbol, prediction_horizon, c_next_threshold, core_lr_mult, head_lr_mult, "
        << "target_epochs, checkpoint_interval, train_start, train_end, infer_start, infer_end, "
        << "resume_model_id, duplicate_nonce, status, phase, updated_at"
        << ") VALUES ("
        << w.quote(rec.symbol) << ","
        << rec.horizon << ","
        << FormatDoubleFull(rec.threshold) << ","
        << FormatDoubleFull(rec.coreLr) << ","
        << FormatDoubleFull(rec.headLr) << ","
        << rec.targetEpochs << ","
        << kMetaRecommendationCheckpointInterval << ","
        << w.quote(kMetaRecommendationTrainStart) << "::timestamptz,"
        << w.quote(kMetaRecommendationTrainEnd) << "::timestamptz,"
        << w.quote(kMetaRecommendationInferStart) << "::timestamptz,"
        << w.quote(kMetaRecommendationInferEnd) << "::timestamptz,"
        << "NULL,"
        << "0,"
        << "'pending','train',now()) RETURNING experiment_id;";
    return w.exec(sql.str())[0][0].as<long long>();
}

void PrintMetaRecommendationQueueFields(const NextExperimentRecommendation& rec,
                                        bool includeRecommendationReason = true)
{
    std::cout << ",rank=" << rec.rank
              << ",symbol=" << rec.symbol
              << ",prediction_horizon=" << rec.horizon
              << ",target_epochs=" << rec.targetEpochs
              << ",threshold=" << FormatDouble(rec.threshold)
              << ",core_lr=" << FormatDouble(rec.coreLr)
              << ",head_lr=" << FormatDouble(rec.headLr);
    if (includeRecommendationReason)
        std::cout << ",reason=" << rec.reason;
}

std::string GroupStatsJson(const std::vector<GroupStats>& groups)
{
    std::ostringstream json;
    json << "[";
    for (size_t i = 0; i < groups.size(); ++i)
    {
        const auto& g = groups[i];
        if (i)
            json << ",";
        json << "{"
             << "\"dimension\":" << JsonString(g.dimension)
             << ",\"key\":" << JsonString(g.key)
             << ",\"sample_count\":" << g.sampleCount
             << ",\"confidence\":" << JsonString(ConfidenceForSampleSize(g.sampleCount))
             << ",\"leader_score\":{\"count\":" << g.leaderScore.count
             << ",\"mean\":" << FormatMetricJson(g.leaderScore, g.leaderScore.mean)
             << ",\"median\":" << FormatMetricJson(g.leaderScore, g.leaderScore.median)
             << ",\"best\":" << FormatMetricJson(g.leaderScore, g.leaderScore.best)
             << ",\"worst\":" << FormatMetricJson(g.leaderScore, g.leaderScore.worst)
             << ",\"stddev\":" << FormatMetricJson(g.leaderScore, g.leaderScore.stddev) << "}"
             << ",\"infer_accuracy\":{\"count\":" << g.inferAccuracy.count
             << ",\"mean\":" << FormatMetricJson(g.inferAccuracy, g.inferAccuracy.mean)
             << ",\"median\":" << FormatMetricJson(g.inferAccuracy, g.inferAccuracy.median)
             << ",\"best\":" << FormatMetricJson(g.inferAccuracy, g.inferAccuracy.best)
             << ",\"worst\":" << FormatMetricJson(g.inferAccuracy, g.inferAccuracy.worst)
             << ",\"stddev\":" << FormatMetricJson(g.inferAccuracy, g.inferAccuracy.stddev) << "}"
             << ",\"accept_rate\":{\"count\":" << g.acceptRate.count
             << ",\"mean\":" << FormatMetricJson(g.acceptRate, g.acceptRate.mean)
             << ",\"median\":" << FormatMetricJson(g.acceptRate, g.acceptRate.median)
             << ",\"best\":" << FormatMetricJson(g.acceptRate, g.acceptRate.best)
             << ",\"worst\":" << FormatMetricJson(g.acceptRate, g.acceptRate.worst)
             << ",\"stddev\":" << FormatMetricJson(g.acceptRate, g.acceptRate.stddev) << "}"
             << ",\"accept_accuracy\":{\"count\":" << g.acceptAccuracy.count
             << ",\"mean\":" << FormatMetricJson(g.acceptAccuracy, g.acceptAccuracy.mean)
             << ",\"median\":" << FormatMetricJson(g.acceptAccuracy, g.acceptAccuracy.median)
             << ",\"best\":" << FormatMetricJson(g.acceptAccuracy, g.acceptAccuracy.best)
             << ",\"worst\":" << FormatMetricJson(g.acceptAccuracy, g.acceptAccuracy.worst)
             << ",\"stddev\":" << FormatMetricJson(g.acceptAccuracy, g.acceptAccuracy.stddev) << "}"
             << "}";
    }
    json << "]";
    return json.str();
}

std::string LeadersJson(const std::vector<LeaderRow>& leaders)
{
    std::ostringstream json;
    json << "[";
    for (size_t i = 0; i < leaders.size(); ++i)
    {
        const auto& l = leaders[i];
        if (i)
            json << ",";
        json << "{"
             << "\"experiment_id\":" << l.experimentId
             << ",\"model_id\":" << FormatOptionalLongLong(l.modelId)
             << ",\"model_name\":" << JsonString(l.modelName)
             << ",\"symbol\":" << JsonString(l.symbol)
             << ",\"prediction_horizon\":" << l.horizon
             << ",\"target_epochs\":" << l.targetEpochs
             << ",\"infer_accuracy\":" << FormatOptionalDouble(l.inferAccuracy)
             << ",\"accept_accuracy\":" << FormatOptionalDouble(l.acceptAccuracy)
             << ",\"accept_rate\":" << FormatOptionalDouble(l.acceptRate)
             << ",\"leader_score\":" << FormatOptionalDouble(l.leaderScore)
             << "}";
    }
    json << "]";
    return json.str();
}

std::string RecommendationsJson(const std::vector<Recommendation>& recs)
{
    std::ostringstream json;
    json << "[";
    for (size_t i = 0; i < recs.size(); ++i)
    {
        const auto& r = recs[i];
        if (i)
            json << ",";
        json << "{"
             << "\"priority\":" << r.priority
             << ",\"action\":" << JsonString(r.action)
             << ",\"reason\":" << JsonString(r.reason)
             << ",\"supporting_evidence\":" << JsonString(r.evidence)
             << ",\"expected_information_gain\":" << JsonString(r.expectedInformationGain)
             << ",\"confidence\":" << JsonString(r.confidence)
             << "}";
    }
    json << "]";
    return json.str();
}

std::string NextExperimentRecommendationsJson(const std::vector<NextExperimentRecommendation>& recs)
{
    std::ostringstream json;
    json << "[";
    for (size_t i = 0; i < recs.size(); ++i)
    {
        const auto& r = recs[i];
        if (i)
            json << ",";
        json << "{"
             << "\"rank\":" << r.rank
             << ",\"symbol\":" << JsonString(r.symbol)
             << ",\"prediction_horizon\":" << r.horizon
             << ",\"target_epochs\":" << r.targetEpochs
             << ",\"threshold\":" << FormatDouble(r.threshold)
             << ",\"core_lr\":" << FormatDouble(r.coreLr)
             << ",\"head_lr\":" << FormatDouble(r.headLr)
             << ",\"reason\":" << JsonString(r.reason)
             << ",\"source_leader_experiment_id\":" << r.sourceLeaderExperimentId
             << ",\"source_model_id\":" << FormatOptionalLongLong(r.sourceModelId)
             << "}";
    }
    json << "]";
    return json.str();
}

std::string StringVectorJson(const std::vector<std::string>& values)
{
    std::ostringstream json;
    json << "[";
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i)
            json << ",";
        json << JsonString(values[i]);
    }
    json << "]";
    return json.str();
}

std::string PlateauJson(const std::vector<PlateauSignal>& signals)
{
    std::ostringstream json;
    json << "[";
    for (size_t i = 0; i < signals.size(); ++i)
    {
        const auto& s = signals[i];
        if (i)
            json << ",";
        json << "{"
             << "\"key\":" << JsonString(s.key)
             << ",\"earlier_epoch\":" << s.earlierEpoch
             << ",\"later_epoch\":" << s.laterEpoch
             << ",\"earlier_score\":" << FormatDouble(s.earlierScore)
             << ",\"later_score\":" << FormatDouble(s.laterScore)
             << ",\"delta\":" << FormatDouble(s.delta)
             << ",\"classification\":" << JsonString(s.classification)
             << ",\"confidence\":" << JsonString(s.confidence)
             << "}";
    }
    json << "]";
    return json.str();
}

std::string BuildStatisticsJson(const MetaAnalysisResult& result)
{
    std::ostringstream json;
    json << "{"
         << "\"scope\":" << JsonString(result.scope)
         << ",\"summary\":{"
         << "\"total_experiments\":" << result.totalExperiments
         << ",\"completed_experiments\":" << result.completedExperiments
         << ",\"failed_experiments\":" << result.failedExperiments
         << ",\"running_experiments\":" << result.runningExperiments
         << ",\"pending_experiments\":" << result.pendingExperiments
         << ",\"completed_models\":" << result.completedModels
         << ",\"success_rate\":" << FormatDouble(result.successRate)
         << ",\"confidence\":" << JsonString(ConfidenceForSampleSize(static_cast<size_t>(result.completedExperiments)))
         << "}"
         << ",\"data_sources\":{"
         << "\"scheduler_experiments\":" << result.dataSources.schedulerExperiments
         << ",\"legacy_model_count\":" << result.dataSources.legacyModels
         << ",\"legacy_models\":" << result.dataSources.legacyModels
         << ",\"merged_record_count\":" << result.dataSources.mergedRecords
         << ",\"duplicate_model_count\":" << result.dataSources.duplicateModelsSkipped
         << ",\"duplicate_models_skipped\":" << result.dataSources.duplicateModelsSkipped
         << ",\"legacy_models_with_inference_metrics\":" << result.dataSources.legacyModelsWithInferenceMetrics
         << ",\"legacy_models_missing_inference_metrics\":" << result.dataSources.legacyModelsMissingInferenceMetrics
         << ",\"scheduler_records_with_metrics\":" << result.dataSources.schedulerRecordsWithMetrics
         << ",\"merged_records_with_metrics\":" << result.dataSources.mergedRecordsWithMetrics
         << ",\"metric_coverage_percentage\":" << FormatDouble(result.dataSources.metricCoveragePercentage)
         << ",\"coverage_percentage\":" << FormatDouble(result.dataSources.coveragePercentage)
         << "}"
         << ",\"group_statistics\":" << GroupStatsJson(result.groupStats)
         << ",\"plateau_signals\":" << PlateauJson(result.plateauSignals)
         << "}";
    return json.str();
}

std::string BuildCombinedJson(const MetaAnalysisResult& result)
{
    std::ostringstream json;
    json << "{"
         << "\"summary\":{"
         << "\"meta_analysis_id\":" << result.metaAnalysisId
         << ",\"scope\":" << JsonString(result.scope)
         << ",\"generated_at\":" << JsonString(result.generatedAt)
         << ",\"recommendation_epoch_policy\":"
         << JsonString(RecommendationEpochPolicyName(result.recommendationEpochPolicy))
         << ",\"completed_experiments\":" << result.completedExperiments
         << ",\"completed_models\":" << result.completedModels
         << ",\"legacy_model_count\":" << result.dataSources.legacyModels
         << ",\"scheduler_experiment_count\":" << result.dataSources.schedulerExperiments
         << ",\"merged_record_count\":" << result.dataSources.mergedRecords
         << ",\"duplicate_model_count\":" << result.dataSources.duplicateModelsSkipped
         << ",\"merged_records_with_metrics\":" << result.dataSources.mergedRecordsWithMetrics
         << ",\"metric_coverage_percentage\":" << FormatDouble(result.dataSources.metricCoveragePercentage)
         << "}"
         << ",\"data_sources\":{"
         << "\"scheduler_experiments\":" << result.dataSources.schedulerExperiments
         << ",\"legacy_model_count\":" << result.dataSources.legacyModels
         << ",\"legacy_models\":" << result.dataSources.legacyModels
         << ",\"merged_record_count\":" << result.dataSources.mergedRecords
         << ",\"duplicate_model_count\":" << result.dataSources.duplicateModelsSkipped
         << ",\"duplicate_models_skipped\":" << result.dataSources.duplicateModelsSkipped
         << ",\"legacy_models_with_inference_metrics\":" << result.dataSources.legacyModelsWithInferenceMetrics
         << ",\"legacy_models_missing_inference_metrics\":" << result.dataSources.legacyModelsMissingInferenceMetrics
         << ",\"scheduler_records_with_metrics\":" << result.dataSources.schedulerRecordsWithMetrics
         << ",\"merged_records_with_metrics\":" << result.dataSources.mergedRecordsWithMetrics
         << ",\"metric_coverage_percentage\":" << FormatDouble(result.dataSources.metricCoveragePercentage)
         << ",\"coverage_percentage\":" << FormatDouble(result.dataSources.coveragePercentage)
         << "}"
         << ",\"statistics\":" << result.statisticsJson
         << ",\"leaderboards\":" << result.leaderboardJson
         << ",\"recommendations\":" << result.recommendationsJson
         << ",\"next_experiment_recommendations\":"
         << NextExperimentRecommendationsJson(result.nextExperimentRecommendations)
         << ",\"next_experiment_notes\":" << StringVectorJson(result.nextExperimentNotes)
         << "}";
    return json.str();
}

void AppendMetricTable(std::ostringstream& report,
                              const std::vector<GroupStats>& groups,
                              const std::string& title,
                              const std::string& dimension,
                              int limit)
{
    std::vector<GroupStats> filtered;
    for (const auto& group : groups)
    {
        if (group.dimension == dimension)
            filtered.push_back(group);
    }
    std::sort(filtered.begin(), filtered.end(), [](const GroupStats& a, const GroupStats& b) {
        const bool aHasMetric = a.leaderScore.count > 0 || a.inferAccuracy.count > 0;
        const bool bHasMetric = b.leaderScore.count > 0 || b.inferAccuracy.count > 0;
        if (aHasMetric != bHasMetric)
            return aHasMetric;
        const ScalarStats& aStats = a.leaderScore.count > 0 ? a.leaderScore : a.inferAccuracy;
        const ScalarStats& bStats = b.leaderScore.count > 0 ? b.leaderScore : b.inferAccuracy;
        if (aHasMetric && bHasMetric && aStats.best != bStats.best)
            return aStats.best > bStats.best;
        return a.sampleCount > b.sampleCount;
    });

    report << "\n## " << title << "\n\n";
    report << "| Group | Samples | Best Leader | Mean Leader | Mean Infer Acc | Mean Accept Rate | Confidence |\n";
    report << "|---|---:|---:|---:|---:|---:|---|\n";
    int shown = 0;
    for (const auto& group : filtered)
    {
        if (shown++ >= limit)
            break;
        report << "| " << group.key
               << " | " << group.sampleCount
               << " | " << FormatMetricMarkdown(group.leaderScore, group.leaderScore.best)
               << " | " << FormatMetricMarkdown(group.leaderScore, group.leaderScore.mean)
               << " | " << FormatMetricMarkdown(group.inferAccuracy, group.inferAccuracy.mean)
               << " | " << FormatMetricMarkdown(group.acceptRate, group.acceptRate.mean)
               << " | " << ConfidenceForSampleSize(group.sampleCount)
               << " |\n";
    }
}

void AppendLeaderRowsTable(std::ostringstream& report,
                           const std::string& title,
                           const std::vector<LeaderRow>& rows,
                           int limit)
{
    report << "\n## " << title << "\n\n";
    if (rows.empty())
    {
        report << "No completed experiments with inference metrics are available.\n";
        return;
    }

    report << "| Rank | Experiment | Model | Symbol | Horizon | Epochs | Infer Acc | Accept Acc | Accept Rate | Leader Score |\n";
    report << "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|\n";
    for (size_t i = 0; i < rows.size() && static_cast<int>(i) < limit; ++i)
    {
        const auto& leader = rows[i];
        report << "| " << (i + 1)
               << " | " << leader.experimentId
               << " | " << FormatOptionalLongLong(leader.modelId)
               << " | " << leader.symbol
               << " | " << leader.horizon
               << " | " << leader.targetEpochs
               << " | " << FormatOptionalDouble(leader.inferAccuracy)
               << " | " << FormatOptionalDouble(leader.acceptAccuracy)
               << " | " << FormatOptionalDouble(leader.acceptRate)
               << " | " << FormatOptionalDouble(leader.leaderScore)
               << " |\n";
    }
}

std::vector<LeaderRow> BestLeadersBySymbol(const std::vector<ExperimentRecord>& records)
{
    std::vector<LeaderRow> all = BuildLeaders(records, std::numeric_limits<int>::max());
    std::set<std::string> seen;
    std::vector<LeaderRow> best;
    for (const auto& row : all)
    {
        if (seen.insert(row.symbol).second)
            best.push_back(row);
    }
    return best;
}

std::vector<LeaderRow> BestLeadersByHorizon(const std::vector<ExperimentRecord>& records)
{
    std::vector<LeaderRow> all = BuildLeaders(records, std::numeric_limits<int>::max());
    std::set<int> seen;
    std::vector<LeaderRow> best;
    for (const auto& row : all)
    {
        if (seen.insert(row.horizon).second)
            best.push_back(row);
    }
    return best;
}

void AppendCountMapTable(std::ostringstream& report,
                         const std::string& title,
                         const std::string& keyColumn,
                         const std::map<std::string, long long>& counts,
                         int limit)
{
    report << "\n### " << title << "\n\n";
    if (counts.empty())
    {
        report << "No records available.\n";
        return;
    }

    std::vector<std::pair<std::string, long long>> rows(counts.begin(), counts.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second)
            return a.second > b.second;
        return a.first < b.first;
    });

    report << "| " << keyColumn << " | Count |\n";
    report << "|---|---:|\n";
    for (size_t i = 0; i < rows.size() && static_cast<int>(i) < limit; ++i)
        report << "| " << rows[i].first << " | " << rows[i].second << " |\n";
}

void AppendExperimentCoverageTables(std::ostringstream& report,
                                    const MetaAnalysisResult& result,
                                    int limit)
{
    std::map<std::string, long long> statusPhaseCounts;
    std::map<std::string, long long> symbolHorizonCounts;
    for (const auto& record : result.records)
    {
        ++statusPhaseCounts[record.status + "/" + record.phase];
        ++symbolHorizonCounts[record.symbol + "/H" + std::to_string(record.horizon)];
    }

    report << "\n## Experiment Coverage\n\n";
    report << "- Scheduler experiments analyzed: " << result.dataSources.schedulerExperiments << "\n";
    report << "- Legacy models reconstructed: " << result.dataSources.legacyModels << "\n";
    report << "- Merged experiment records: " << result.dataSources.mergedRecords << "\n";
    report << "- Duplicate models skipped: " << result.dataSources.duplicateModelsSkipped << "\n";
    report << "- Legacy models with inference metrics: " << result.dataSources.legacyModelsWithInferenceMetrics << "\n";
    report << "- Legacy models missing inference metrics: " << result.dataSources.legacyModelsMissingInferenceMetrics << "\n";
    report << "- Scheduler records with metrics: " << result.dataSources.schedulerRecordsWithMetrics << "\n";
    report << "- Merged records with metrics: " << result.dataSources.mergedRecordsWithMetrics << "\n";
    report << "- Metric coverage percentage: " << FormatDouble(result.dataSources.metricCoveragePercentage) << "\n";
    report << "- Data source coverage percentage: " << FormatDouble(result.dataSources.coveragePercentage) << "\n";

    AppendCountMapTable(report,
                        "Experiment Count By Status/Phase",
                        "Status / Phase",
                        statusPhaseCounts,
                        limit);
    AppendCountMapTable(report,
                        "Experiment Count By Symbol/Horizon",
                        "Symbol / Horizon",
                        symbolHorizonCounts,
                        limit);
}

void AppendRecentCompletedExperiments(std::ostringstream& report,
                                      const std::vector<ExperimentRecord>& records,
                                      int limit)
{
    std::vector<const ExperimentRecord*> completed;
    for (const auto& record : records)
    {
        if (record.status == "completed")
            completed.push_back(&record);
    }

    std::sort(completed.begin(), completed.end(), [](const ExperimentRecord* a, const ExperimentRecord* b) {
        const std::string at = a->completedAt.value_or("");
        const std::string bt = b->completedAt.value_or("");
        if (at != bt)
            return at > bt;
        return a->experimentId > b->experimentId;
    });

    report << "\n## Recent Completed Experiments\n\n";
    if (completed.empty())
    {
        report << "No completed experiments are available.\n";
        return;
    }

    report << "| Experiment | Model | Symbol | Horizon | Epochs | Completed At | Infer Acc | Leader Score | Source |\n";
    report << "|---:|---:|---|---:|---:|---|---:|---:|---|\n";
    for (size_t i = 0; i < completed.size() && static_cast<int>(i) < limit; ++i)
    {
        const auto& row = *completed[i];
        report << "| " << row.experimentId
               << " | " << FormatOptionalLongLong(row.modelId)
               << " | " << row.symbol
               << " | " << row.horizon
               << " | " << row.completedEpochs.value_or(row.targetEpochs)
               << " | " << row.completedAt.value_or("unknown")
               << " | " << FormatOptionalDouble(row.inferAccuracy)
               << " | " << FormatOptionalDouble(row.leaderScore)
               << " | " << row.dataSource
               << " |\n";
    }
}

std::string BuildMarkdownReport(const MetaAnalysisResult& result, int limit)
{
    std::ostringstream report;
    report << "# LSTM Experiment Meta-Analysis\n\n";
    report << "Generated: `" << result.generatedAt << "`\n\n";

    report << "## Report Metadata\n\n";
    report << "| Field | Value |\n";
    report << "|---|---:|\n";
    report << "| Scope | `" << result.scope << "` |\n";
    report << "| Recommendation epoch policy | " << RecommendationEpochPolicyName(result.recommendationEpochPolicy) << " |\n";
    report << "| Total experiments | " << result.totalExperiments << " |\n";
    report << "| Completed experiments | " << result.completedExperiments << " |\n";
    report << "| Failed experiments | " << result.failedExperiments << " |\n";
    report << "| Completed models | " << result.completedModels << " |\n";
    report << "| Merged records | " << result.dataSources.mergedRecords << " |\n";
    report << "| Records with metrics | " << result.dataSources.mergedRecordsWithMetrics << " |\n";
    report << "| Metric coverage % | " << FormatDouble(result.dataSources.metricCoveragePercentage) << " |\n";

    report << "## Executive Summary\n\n";
    report << "- Total experiments: " << result.totalExperiments << "\n";
    report << "- Completed experiments: " << result.completedExperiments << "\n";
    report << "- Failed experiments: " << result.failedExperiments << "\n";
    report << "- Completed models: " << result.completedModels << "\n";
    report << "- Success rate: " << FormatDouble(result.successRate) << "\n";
    report << "- Overall confidence: "
           << ConfidenceForSampleSize(static_cast<size_t>(result.completedExperiments)) << "\n";

    AppendLeaderRowsTable(report, "Current Leaders", result.leaders, limit);
    AppendLeaderRowsTable(report, "Leaders by Symbol", BestLeadersBySymbol(result.records), limit);
    AppendLeaderRowsTable(report, "Leaders by Horizon", BestLeadersByHorizon(result.records), limit);
    AppendExperimentCoverageTables(report, result, limit);
    AppendRecentCompletedExperiments(report, result.records, limit);

    AppendMetricTable(report, result.groupStats, "Symbol Performance Summary", "symbol", limit);
    AppendMetricTable(report, result.groupStats, "Horizon Performance Summary", "prediction_horizon", limit);
    AppendMetricTable(report, result.groupStats, "Learning Rate Analysis: Core LR", "core_lr_mult", limit);
    AppendMetricTable(report, result.groupStats, "Learning Rate Analysis: Head LR", "head_lr_mult", limit);
    AppendMetricTable(report, result.groupStats, "Epoch Analysis", "target_epochs", limit);
    AppendMetricTable(report, result.groupStats, "Threshold Analysis", "threshold", limit);
    AppendMetricTable(report, result.groupStats, "Checkpoint Interval Analysis", "checkpoint_interval", limit);
    AppendMetricTable(report, result.groupStats, "Resume Vs Fresh Training", "resume_mode", limit);

    report << "\n## Plateau Analysis\n\n";
    if (result.plateauSignals.empty())
    {
        report << "No statistically useful epoch-series plateau signal is available yet.\n";
    }
    else
    {
        report << "| Configuration | Earlier Epoch | Later Epoch | Delta | Classification | Confidence |\n";
        report << "|---|---:|---:|---:|---|---|\n";
        for (size_t i = 0; i < result.plateauSignals.size() && static_cast<int>(i) < limit; ++i)
        {
            const auto& signal = result.plateauSignals[i];
            report << "| `" << signal.key << "`"
                   << " | " << signal.earlierEpoch
                   << " | " << signal.laterEpoch
                   << " | " << FormatDouble(signal.delta)
                   << " | " << signal.classification
                   << " | " << signal.confidence
                   << " |\n";
        }
    }

    report << "\n## Failure Analysis\n\n";
    report << "- Failed experiments: " << result.failedExperiments << "\n";
    report << "- Running experiments: " << result.runningExperiments << "\n";
    report << "- Pending experiments: " << result.pendingExperiments << "\n";
    report << "- Failure rate among terminal experiments: "
           << FormatDouble((result.completedExperiments + result.failedExperiments) > 0
               ? static_cast<double>(result.failedExperiments) /
                 static_cast<double>(result.completedExperiments + result.failedExperiments)
               : 0.0)
           << "\n";

    report << "\n## Top Recommendations\n\n";
    for (const auto& rec : result.recommendations)
    {
        report << "### Priority " << rec.priority << ": " << rec.action << "\n\n";
        report << "- Reason: " << rec.reason << "\n";
        report << "- Supporting evidence: " << rec.evidence << "\n";
        report << "- Expected information gain: " << rec.expectedInformationGain << "\n";
        report << "- Confidence: " << rec.confidence << "\n\n";
    }

    report << "## Recommended Next Experiments\n\n";
    report << "Recommendation epoch policy: "
           << RecommendationEpochPolicyName(result.recommendationEpochPolicy)
           << "\n\n";
    if (!result.nextExperimentNotes.empty())
    {
        for (const auto& note : result.nextExperimentNotes)
            report << "- Note: " << note << "\n";
        report << "\n";
    }
    if (result.nextExperimentRecommendations.empty())
    {
        report << "No queueable next-experiment candidates were generated from completed analyzed results.\n\n";
    }
    else
    {
        report << "| Rank | Symbol | Horizon | Target Epochs | Threshold | Core LR | Head LR | Reason | Source Leader Experiment | Source Model |\n";
        report << "|---:|---|---:|---:|---:|---:|---:|---|---:|---:|\n";
        for (const auto& rec : result.nextExperimentRecommendations)
        {
            report << "| " << rec.rank
                   << " | " << rec.symbol
                   << " | " << rec.horizon
                   << " | " << rec.targetEpochs
                   << " | " << FormatDouble(rec.threshold)
                   << " | " << FormatDouble(rec.coreLr)
                   << " | " << FormatDouble(rec.headLr)
                   << " | " << rec.reason
                   << " | " << rec.sourceLeaderExperimentId
                   << " | " << FormatOptionalLongLong(rec.sourceModelId)
                   << " |\n";
        }
        report << "\n";
    }

    report << "## Open Questions / Gaps\n\n";
    report << "- Are current leaders repeatable across independent runs?\n";
    report << "- Which symbol/horizon pairs still lack enough completed samples?\n";
    report << "- Do longer epoch runs improve leader_score after the detected plateau points?\n";
    report << "- Are acceptance-rate gains aligned with inference accuracy gains?\n";

    report << "\n## Suggested Next Experiments\n\n";
    for (const auto& rec : result.recommendations)
        report << "- " << rec.action << "\n";

    return report.str();
}

MetaAnalysisResult BuildMetaAnalysis(pqxx::work& w, const MetaAnalysisOptions& options)
{
    MetaAnalysisResult result;
    result.scope = ScopeForOptions(options);
    result.generatedAt = CurrentUtcTimestamp();
    result.recommendationEpochPolicy = options.recommendationEpochPolicy;
    LoadExperimentCounts(w, options, result);
    std::vector<ExperimentRecord> schedulerRecords = LoadExperimentRecords(w, options);
    const std::set<long long> schedulerModelIds = SchedulerModelIds(schedulerRecords);
    long long legacyModelCount = 0;
    long long duplicateModelsSkipped = 0;
    std::vector<ExperimentRecord> legacyRecords = LoadLegacyExperimentRecords(w,
                                                                               options,
                                                                               schedulerModelIds,
                                                                               legacyModelCount,
                                                                               duplicateModelsSkipped);
    result.records = schedulerRecords;
    result.records.insert(result.records.end(), legacyRecords.begin(), legacyRecords.end());
    ApplyMergedCounts(result, schedulerRecords, legacyModelCount, duplicateModelsSkipped);
    result.groupStats = BuildAllGroupStats(result.records);
    result.leaders = BuildLeaders(result.records, options.limit);
    result.plateauSignals = DetectPlateaus(result.records);
    result.recommendations = BuildRecommendations(result);
    const std::vector<ExistingExperimentConfig> existingExperimentConfigs = LoadExistingExperimentConfigs(w);
    result.nextExperimentRecommendations = BuildNextExperimentRecommendations(result,
                                                                              existingExperimentConfigs,
                                                                              options.recommendationEpochPolicy,
                                                                              options.limit,
                                                                              result.nextExperimentNotes);
    result.statisticsJson = BuildStatisticsJson(result);
    result.recommendationsJson = RecommendationsJson(result.recommendations);
    result.leaderboardJson = LeadersJson(result.leaders);
    result.markdown = BuildMarkdownReport(result, options.limit);
    return result;
}

void PrintMarkers(const MetaAnalysisResult& result)
{
    std::cout << "META_ANALYSIS_RECOMMENDATION_POLICY"
              << ",epoch_policy=" << RecommendationEpochPolicyName(result.recommendationEpochPolicy)
              << std::endl;

    std::cout << "META_ANALYSIS_STATISTIC"
              << ",scope=" << result.scope
              << ",total_experiments=" << result.totalExperiments
              << ",completed_experiments=" << result.completedExperiments
              << ",failed_experiments=" << result.failedExperiments
              << ",completed_models=" << result.completedModels
              << ",success_rate=" << FormatDouble(result.successRate)
              << std::endl;
    std::cout << "META_ANALYSIS_DATA_SOURCE"
              << ",scheduler_experiments=" << result.dataSources.schedulerExperiments
              << ",legacy_models=" << result.dataSources.legacyModels
              << ",merged_records=" << result.dataSources.mergedRecords
              << ",duplicate_models_skipped=" << result.dataSources.duplicateModelsSkipped
              << ",coverage_percentage=" << FormatDouble(result.dataSources.coveragePercentage)
              << std::endl;
    std::cout << "META_ANALYSIS_METRIC_COVERAGE"
              << ",legacy_models=" << result.dataSources.legacyModels
              << ",legacy_models_with_inference_metrics=" << result.dataSources.legacyModelsWithInferenceMetrics
              << ",legacy_models_missing_inference_metrics=" << result.dataSources.legacyModelsMissingInferenceMetrics
              << ",scheduler_records_with_metrics=" << result.dataSources.schedulerRecordsWithMetrics
              << ",merged_records=" << result.dataSources.mergedRecords
              << ",merged_records_with_metrics=" << result.dataSources.mergedRecordsWithMetrics
              << ",metric_coverage_percentage=" << FormatDouble(result.dataSources.metricCoveragePercentage)
              << std::endl;
    std::cout << "META_ANALYSIS_CONFIDENCE"
              << ",scope=" << result.scope
              << ",confidence=" << ConfidenceForSampleSize(static_cast<size_t>(result.completedExperiments))
              << ",sample_count=" << result.completedExperiments
              << std::endl;

    for (size_t i = 0; i < result.leaders.size(); ++i)
    {
        const auto& leader = result.leaders[i];
        std::cout << "META_ANALYSIS_LEADER"
                  << ",rank=" << (i + 1)
                  << ",experiment_id=" << leader.experimentId
                  << ",model_id=" << FormatOptionalLongLong(leader.modelId)
                  << ",symbol=" << leader.symbol
                  << ",prediction_horizon=" << leader.horizon
                  << ",target_epochs=" << leader.targetEpochs
                  << ",leader_score=" << FormatOptionalDouble(leader.leaderScore)
                  << ",infer_accuracy=" << FormatOptionalDouble(leader.inferAccuracy)
                  << std::endl;
    }

    for (const auto& signal : result.plateauSignals)
    {
        std::cout << "META_ANALYSIS_PLATEAU"
                  << ",key=" << signal.key
                  << ",earlier_epoch=" << signal.earlierEpoch
                  << ",later_epoch=" << signal.laterEpoch
                  << ",delta=" << FormatDouble(signal.delta)
                  << ",classification=" << signal.classification
                  << ",confidence=" << signal.confidence
                  << std::endl;
    }

    for (const auto& rec : result.recommendations)
    {
        std::cout << "META_ANALYSIS_RECOMMENDATION"
                  << ",priority=" << rec.priority
                  << ",confidence=" << rec.confidence
                  << ",action=" << rec.action
                  << std::endl;
        std::cout << "META_ANALYSIS_NEXT_EXPERIMENT"
                  << ",priority=" << rec.priority
                  << ",action=" << rec.action
                  << std::endl;
    }

    for (const auto& note : result.nextExperimentNotes)
    {
        std::cout << "META_ANALYSIS_RECOMMENDATION_NOTE"
                  << ",note=" << note
                  << std::endl;
    }

    for (const auto& rec : result.nextExperimentRecommendations)
    {
        std::cout << "META_ANALYSIS_RECOMMENDATION"
                  << ",rank=" << rec.rank
                  << ",symbol=" << rec.symbol
                  << ",prediction_horizon=" << rec.horizon
                  << ",target_epochs=" << rec.targetEpochs
                  << ",core_lr=" << FormatDouble(rec.coreLr)
                  << ",head_lr=" << FormatDouble(rec.headLr)
                  << ",threshold=" << FormatDouble(rec.threshold)
                  << ",reason=" << rec.reason
                  << ",source_leader_experiment_id=" << rec.sourceLeaderExperimentId
                  << ",source_model_id=" << FormatOptionalLongLong(rec.sourceModelId)
                  << std::endl;
        std::cout << "META_ANALYSIS_NEXT_EXPERIMENT"
                  << ",rank=" << rec.rank
                  << ",symbol=" << rec.symbol
                  << ",prediction_horizon=" << rec.horizon
                  << ",target_epochs=" << rec.targetEpochs
                  << ",core_lr=" << FormatDouble(rec.coreLr)
                  << ",head_lr=" << FormatDouble(rec.headLr)
                  << ",threshold=" << FormatDouble(rec.threshold)
                  << ",source_leader_experiment_id=" << rec.sourceLeaderExperimentId
                  << ",source_model_id=" << FormatOptionalLongLong(rec.sourceModelId)
                  << std::endl;
    }
}

long long StoreMetaAnalysis(pqxx::work& w, const MetaAnalysisResult& result)
{
    std::ostringstream sql;
    sql << "INSERT INTO experiment_meta_analysis ("
        << "analysis_scope, completed_experiments, completed_models, summary_markdown, "
        << "recommendations_json, statistics_json, leaderboard_snapshot"
        << ") VALUES ("
        << w.quote(result.scope) << ","
        << result.completedExperiments << ","
        << result.completedModels << ","
        << w.quote(result.markdown) << ","
        << w.quote(result.recommendationsJson) << "::jsonb,"
        << w.quote(result.statisticsJson) << "::jsonb,"
        << w.quote(result.leaderboardJson) << "::jsonb"
        << ") RETURNING meta_analysis_id;";
    return w.exec(sql.str())[0][0].as<long long>();
}

void WriteOutputFile(const std::string& path, const std::string& content)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("failed to open output file '" + path + "'");
    out << content;
}

int RunMetaAnalysisOnce(const MetaAnalysisOptions& options)
{
    std::cout << "META_ANALYSIS_STARTED"
              << ",scope=" << ScopeForOptions(options)
              << std::endl;

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    w.exec("SET TRANSACTION READ WRITE;");
    if (!RequireMetaAnalysisTables(w))
        return 1;

    MetaAnalysisResult result = BuildMetaAnalysis(w, options);
    result.metaAnalysisId = StoreMetaAnalysis(w, result);
    w.commit();

    PrintMarkers(result);

    const std::string combinedJson = BuildCombinedJson(result);
    if (options.outputJson)
        std::cout << combinedJson << std::endl;
    if (options.outputMarkdown || (!options.outputJson && !options.outputFile.has_value()))
        std::cout << result.markdown << std::endl;
    if (options.outputFile.has_value())
    {
        const std::string content = options.outputJson && !options.outputMarkdown
            ? combinedJson
            : result.markdown;
        WriteOutputFile(*options.outputFile, content);
    }

    std::cout << "META_ANALYSIS_COMPLETED"
              << ",meta_analysis_id=" << result.metaAnalysisId
              << ",scope=" << result.scope
              << ",completed_experiments=" << result.completedExperiments
              << ",completed_models=" << result.completedModels
              << std::endl;
    return 0;
}

int QueueMetaRecommendations(const MetaAnalysisOptions& options)
{
    std::cout << "META_ANALYSIS_RECOMMENDATION_POLICY"
              << ",epoch_policy=" << RecommendationEpochPolicyName(options.recommendationEpochPolicy)
              << std::endl;

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    w.exec("SET TRANSACTION READ WRITE;");
    if (!RequireMetaAnalysisTables(w))
        return 2;

    MetaAnalysisResult result = BuildMetaAnalysis(w, options);

    // Queueing intentionally reuses the same leader-neighborhood candidate generator,
    // then performs a fresh database duplicate check immediately before insert.
    result.nextExperimentNotes.clear();
    result.nextExperimentRecommendations = BuildNextExperimentRecommendations(result,
                                                                              {},
                                                                              options.recommendationEpochPolicy,
                                                                              options.limit,
                                                                              result.nextExperimentNotes);

    int queued = 0;
    int skipped = 0;
    int candidates = 0;
    for (const auto& rec : result.nextExperimentRecommendations)
    {
        ++candidates;
        const std::optional<long long> duplicate = FindExistingExperimentForRecommendation(w, rec);
        if (duplicate.has_value())
        {
            ++skipped;
            std::cout << "META_RECOMMENDATION_QUEUE_SKIPPED";
            PrintMetaRecommendationQueueFields(rec, false);
            std::cout << ",reason=duplicate_existing_experiment"
                      << ",existing_experiment_id=" << *duplicate
                      << ",recommendation_reason=" << rec.reason
                      << std::endl;
            continue;
        }

        if (options.dryRun)
        {
            std::cout << "META_RECOMMENDATION_QUEUE_DRY_RUN";
            PrintMetaRecommendationQueueFields(rec);
            std::cout << std::endl;
            continue;
        }

        const long long experimentId = InsertMetaRecommendationExperiment(w, rec);
        ++queued;
        std::cout << "META_RECOMMENDATION_QUEUED"
                  << ",experiment_id=" << experimentId;
        PrintMetaRecommendationQueueFields(rec);
        std::cout << std::endl;
    }

    if (options.dryRun)
        w.abort();
    else
        w.commit();

    std::cout << "META_RECOMMENDATION_QUEUE_SUMMARY"
              << ",total_recommendations=" << candidates
              << ",queued=" << queued
              << ",skipped=" << skipped
              << ",dry_run=" << (options.dryRun ? 1 : 0)
              << std::endl;
    return 0;
}

int RunContinuousMetaAnalysis(MetaAnalysisOptions options)
{
    gMetaAnalysisInterrupted = 0;
    auto previousInt = std::signal(SIGINT, HandleMetaAnalysisSignal);
    auto previousTerm = std::signal(SIGTERM, HandleMetaAnalysisSignal);

    std::cout << "META_ANALYSIS_LOOP_STARTED"
              << ",scope=" << ScopeForOptions(options)
              << ",interval_seconds=" << options.intervalSeconds
              << std::endl;

    int lastRc = 0;
    while (!gMetaAnalysisInterrupted)
    {
        std::cout << "META_ANALYSIS_WAKE"
                  << ",scope=" << ScopeForOptions(options)
                  << std::endl;
        lastRc = RunMetaAnalysisOnce(options);

        if (gMetaAnalysisInterrupted)
            break;

        std::cout << "META_ANALYSIS_SLEEP"
                  << ",seconds=" << options.intervalSeconds
                  << std::endl;
        for (int second = 0; second < options.intervalSeconds && !gMetaAnalysisInterrupted; ++second)
            std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "META_ANALYSIS_INTERRUPTED"
              << ",scope=" << ScopeForOptions(options)
              << std::endl;

    std::signal(SIGINT, previousInt);
    std::signal(SIGTERM, previousTerm);
    return lastRc;
}

int PrintLatestReport(const MetaAnalysisOptions& options)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!RequireMetaAnalysisTables(w))
        return 1;

    std::ostringstream sql;
    sql << "SELECT meta_analysis_id, analysis_scope, summary_markdown, "
        << "recommendations_json::text, statistics_json::text, leaderboard_snapshot::text "
        << "FROM experiment_meta_analysis WHERE 1=1 ";
    if (options.symbol.has_value() || options.horizon.has_value())
        sql << "AND analysis_scope = " << w.quote(ScopeForOptions(options)) << " ";
    sql << "ORDER BY generated_at DESC, meta_analysis_id DESC LIMIT 1;";

    pqxx::result rows = w.exec(sql.str());
    if (rows.empty())
    {
        std::cerr << "META_ANALYSIS_FAILED"
                  << ",reason=no_stored_report"
                  << ",scope=" << ScopeForOptions(options)
                  << std::endl;
        return 1;
    }

    const long long id = rows[0][0].as<long long>();
    const std::string scope = rows[0][1].as<std::string>();
    const std::string markdown = rows[0][2].as<std::string>();
    const std::string recommendations = rows[0][3].as<std::string>();
    const std::string statistics = rows[0][4].as<std::string>();
    const std::string leaders = rows[0][5].as<std::string>();
    const std::string json = "{\"meta_analysis_id\":" + std::to_string(id) +
        ",\"scope\":" + JsonString(scope) +
        ",\"statistics\":" + statistics +
        ",\"leaderboards\":" + leaders +
        ",\"recommendations\":" + recommendations + "}";

    const bool printJson = options.outputJson;
    const bool printMarkdown = options.outputMarkdown || !options.outputJson;
    if (printJson)
        std::cout << json << std::endl;
    if (printMarkdown)
        std::cout << markdown << std::endl;
    if (options.outputFile.has_value())
        WriteOutputFile(*options.outputFile, printJson && !printMarkdown ? json : markdown);
    return 0;
}

int RunMetaAnalysisCli(int argc, const char* argv[])
{
    MetaAnalysisOptions options;
    try
    {
        options = ParseMetaAnalysisArgs(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n"
                  << "Usage: " << (argc > 0 ? argv[0] : "LSTM_Release")
                  << " --meta-analyze [--meta-analysis-interval SECONDS] "
                  << "[--meta-analysis-json|--meta-analysis-markdown] "
                  << "[--meta-analysis-limit N] [--meta-analysis-symbol SYMBOL] "
                  << "[--meta-analysis-horizon N] [--meta-analysis-output FILE] "
                  << "[--recommendation-epoch-policy=leader|highest]\n"
                  << "Usage: " << (argc > 0 ? argv[0] : "LSTM_Release")
                  << " --meta-analyze-once [--meta-analysis-json|--meta-analysis-markdown] "
                  << "[--meta-analysis-limit N] [--meta-analysis-symbol SYMBOL] "
                  << "[--meta-analysis-horizon N] [--meta-analysis-output FILE] "
                  << "[--recommendation-epoch-policy=leader|highest]\n"
                  << "Usage: " << (argc > 0 ? argv[0] : "LSTM_Release")
                  << " --meta-analysis-report [--meta-analysis-json|--meta-analysis-markdown] "
                  << "[--meta-analysis-symbol SYMBOL] [--meta-analysis-horizon N] "
                  << "[--meta-analysis-output FILE] "
                  << "[--recommendation-epoch-policy=leader|highest]\n"
                  << "Usage: " << (argc > 0 ? argv[0] : "LSTM_Release")
                  << " --queue-meta-recommendations [--meta-analysis-limit N] "
                  << "[--recommendation-epoch-policy=leader|highest] [--dry-run]\n"
                  << "Meta-analysis reports include advisory Recommended Next Experiments; "
                  << "they do not queue or launch experiments.\n";
        return 1;
    }

    try
    {
        if (options.metaAnalyzeOnce)
            return RunMetaAnalysisOnce(options);
        if (options.queueMetaRecommendations)
            return QueueMetaRecommendations(options);
        if (options.metaAnalyze)
            return RunContinuousMetaAnalysis(options);
        if (options.metaAnalysisReport)
            return PrintLatestReport(options);
    }
    catch (const std::exception& e)
    {
        std::cerr << "META_ANALYSIS_FAILED"
                  << ",error=" << e.what()
                  << std::endl;
        return 1;
    }
    return 1;
}

} // namespace EA::ExperimentMetaAnalyzer
