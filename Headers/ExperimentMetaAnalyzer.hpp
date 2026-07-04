#pragma once

#include <algorithm>
#include <cmath>
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
#include <vector>

#include <pqxx/pqxx>

#include "CanonicalSymbol.hpp"

namespace EA::ExperimentMetaAnalyzer
{

struct MetaAnalysisOptions
{
    bool metaAnalyze = false;
    bool metaAnalyzeOnce = false;
    bool metaAnalysisReport = false;
    bool outputJson = false;
    bool outputMarkdown = false;
    int limit = 20;
    std::optional<std::string> symbol;
    std::optional<int> horizon;
    std::optional<std::string> outputFile;
};

struct ExperimentRecord
{
    long long experimentId = -1;
    std::optional<long long> modelId;
    std::string modelName;
    std::string symbol;
    int horizon = 0;
    double threshold = 0.0;
    std::optional<double> coreLr;
    std::optional<double> headLr;
    int targetEpochs = 0;
    int checkpointInterval = 0;
    std::string status;
    std::string phase;
    bool resumed = false;
    std::optional<int> completedEpochs;
    std::optional<double> trainAccuracy;
    std::optional<double> validationAccuracy;
    std::optional<double> inferAccuracy;
    std::optional<double> acceptRate;
    std::optional<double> acceptAccuracy;
    std::optional<double> leaderScore;
};

struct ScalarStats
{
    size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double best = 0.0;
    double worst = 0.0;
    double stddev = 0.0;
};

struct GroupStats
{
    std::string dimension;
    std::string key;
    size_t sampleCount = 0;
    ScalarStats leaderScore;
    ScalarStats inferAccuracy;
    ScalarStats acceptRate;
    ScalarStats acceptAccuracy;
};

struct LeaderRow
{
    long long experimentId = -1;
    std::optional<long long> modelId;
    std::string symbol;
    int horizon = 0;
    int targetEpochs = 0;
    std::optional<double> inferAccuracy;
    std::optional<double> acceptAccuracy;
    std::optional<double> acceptRate;
    std::optional<double> leaderScore;
    std::string modelName;
};

struct Recommendation
{
    int priority = 0;
    std::string action;
    std::string reason;
    std::string evidence;
    std::string expectedInformationGain;
    std::string confidence;
};

struct PlateauSignal
{
    std::string key;
    int earlierEpoch = 0;
    int laterEpoch = 0;
    double earlierScore = 0.0;
    double laterScore = 0.0;
    double delta = 0.0;
    std::string classification;
    std::string confidence;
};

struct MetaAnalysisResult
{
    std::string scope;
    long long totalExperiments = 0;
    long long completedExperiments = 0;
    long long failedExperiments = 0;
    long long runningExperiments = 0;
    long long pendingExperiments = 0;
    long long completedModels = 0;
    double successRate = 0.0;
    std::vector<ExperimentRecord> records;
    std::vector<GroupStats> groupStats;
    std::vector<LeaderRow> leaders;
    std::vector<PlateauSignal> plateauSignals;
    std::vector<Recommendation> recommendations;
    std::string statisticsJson;
    std::string recommendationsJson;
    std::string leaderboardJson;
    std::string markdown;
    long long metaAnalysisId = -1;
};

inline bool IsMetaAnalysisCommand(int argc, const char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg{argv[i]};
        if (arg == "--meta-analyze" ||
            arg == "--meta-analyze-once" ||
            arg == "--meta-analysis-report")
            return true;
    }
    return false;
}

inline std::string GetEnvOrDefault(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return (value && *value) ? std::string{value} : std::string{fallback};
}

inline std::string LstmDbConnectionString()
{
    return "hostaddr=" + GetEnvOrDefault("LSTM_DB_HOST", "127.0.0.1") +
           " user=pqxx dbname=" + GetEnvOrDefault("LSTM_DB_NAME", "LSTM");
}

inline bool SplitOptionWithValue(const std::string& arg,
                                 const std::string& optionName,
                                 std::string& value)
{
    const std::string prefix = optionName + "=";
    if (arg.rfind(prefix, 0) != 0)
        return false;
    value = arg.substr(prefix.size());
    return true;
}

inline std::string RequireNextArg(int argc, const char* argv[], int& i, const std::string& optionName)
{
    if (i + 1 >= argc)
        throw std::invalid_argument(optionName + " requires a value");
    return argv[++i];
}

inline int ParsePositiveInt(const std::string& optionName, const std::string& value)
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

inline MetaAnalysisOptions ParseMetaAnalysisArgs(int argc, const char* argv[])
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
        else if (arg == "--meta-analysis-json")
            options.outputJson = true;
        else if (arg == "--meta-analysis-markdown")
            options.outputMarkdown = true;
        else if (arg == "--meta-analysis-limit")
            options.limit = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--meta-analysis-symbol")
            options.symbol = EA::CanonicalSymbol::Normalize(RequireNextArg(argc, argv, i, arg));
        else if (arg == "--meta-analysis-horizon")
            options.horizon = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--meta-analysis-output")
            options.outputFile = RequireNextArg(argc, argv, i, arg);
        else if (SplitOptionWithValue(arg, "--meta-analysis-limit", value))
            options.limit = ParsePositiveInt("--meta-analysis-limit", value);
        else if (SplitOptionWithValue(arg, "--meta-analysis-symbol", value))
            options.symbol = EA::CanonicalSymbol::Normalize(value);
        else if (SplitOptionWithValue(arg, "--meta-analysis-horizon", value))
            options.horizon = ParsePositiveInt("--meta-analysis-horizon", value);
        else if (SplitOptionWithValue(arg, "--meta-analysis-output", value))
            options.outputFile = value;
        else if (arg.rfind("--", 0) == 0)
            throw std::invalid_argument("unknown meta-analysis option '" + arg + "'");
        else
            throw std::invalid_argument("unexpected positional meta-analysis argument '" + arg + "'");
    }

    const int commandCount = (options.metaAnalyze ? 1 : 0) +
                             (options.metaAnalyzeOnce ? 1 : 0) +
                             (options.metaAnalysisReport ? 1 : 0);
    if (commandCount != 1)
        throw std::invalid_argument("expected exactly one meta-analysis command");
    if (options.limit <= 0)
        throw std::invalid_argument("--meta-analysis-limit must be positive");
    return options;
}

inline bool TableExists(pqxx::work& w, const std::string& tableName)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = $1 LIMIT 1;",
        tableName);
    return !r.empty();
}

inline bool RequireMetaAnalysisTables(pqxx::work& w)
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

inline std::string FormatDouble(double value)
{
    if (!std::isfinite(value))
        return "null";
    std::ostringstream oss;
    oss << std::setprecision(10) << value;
    return oss.str();
}

inline std::string FormatOptionalDouble(const std::optional<double>& value)
{
    return value.has_value() ? FormatDouble(*value) : "null";
}

inline std::string FormatOptionalInt(const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "null";
}

inline std::string FormatOptionalLongLong(const std::optional<long long>& value)
{
    return value.has_value() ? std::to_string(*value) : "null";
}

inline std::string JsonEscape(const std::string& input)
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

inline std::string JsonString(const std::string& value)
{
    return "\"" + JsonEscape(value) + "\"";
}

inline std::optional<double> OptionalDoubleCell(const pqxx::row& row, size_t idx)
{
    if (row[idx].is_null())
        return std::nullopt;
    return row[idx].as<double>();
}

inline std::optional<int> OptionalIntCell(const pqxx::row& row, size_t idx)
{
    if (row[idx].is_null())
        return std::nullopt;
    return row[idx].as<int>();
}

inline std::optional<long long> OptionalLongLongCell(const pqxx::row& row, size_t idx)
{
    if (row[idx].is_null())
        return std::nullopt;
    return row[idx].as<long long>();
}

inline std::optional<std::string> OptionalStringCell(const pqxx::row& row, size_t idx)
{
    if (row[idx].is_null())
        return std::nullopt;
    return row[idx].as<std::string>();
}

inline ScalarStats ComputeStats(std::vector<double> values, bool higherIsBetter = true)
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

inline std::string ConfidenceForSampleSize(size_t n)
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

inline std::string ScopeForOptions(const MetaAnalysisOptions& options)
{
    std::ostringstream scope;
    scope << "all";
    if (options.symbol.has_value())
        scope << ";symbol=" << *options.symbol;
    if (options.horizon.has_value())
        scope << ";horizon=" << *options.horizon;
    return scope.str();
}

inline std::string GroupKeyDouble(const std::optional<double>& value)
{
    return value.has_value() ? FormatDouble(*value) : "missing";
}

inline std::string GroupKeyInt(const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "missing";
}

inline std::vector<ExperimentRecord> LoadExperimentRecords(pqxx::work& w,
                                                           const MetaAnalysisOptions& options)
{
    std::ostringstream sql;
    sql << "SELECT e.experiment_id, e.symbol, e.prediction_horizon, e.c_next_threshold, "
        << "e.core_lr_mult, e.head_lr_mult, e.target_epochs, e.checkpoint_interval, "
        << "e.status, e.phase, e.resume_model_id, e.last_model_id, "
        << "a.model_id, a.completed_epochs, a.train_accuracy, a.validation_accuracy, "
        << "a.infer_accuracy, a.accept_rate, a.accept_accuracy, a.leader_score, m.name "
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
        records.push_back(std::move(record));
    }
    return records;
}

inline long long CountDistinctCompletedModels(pqxx::work& w, const MetaAnalysisOptions& options)
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

inline void LoadExperimentCounts(pqxx::work& w,
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

inline void AddMetricValue(std::vector<double>& values, const std::optional<double>& value)
{
    if (value.has_value() && std::isfinite(*value))
        values.push_back(*value);
}

inline GroupStats ComputeGroupStats(const std::string& dimension,
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
inline void BuildGrouping(const std::vector<ExperimentRecord>& records,
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

inline std::vector<GroupStats> BuildAllGroupStats(const std::vector<ExperimentRecord>& records)
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

inline std::vector<LeaderRow> BuildLeaders(const std::vector<ExperimentRecord>& records, int limit)
{
    std::vector<LeaderRow> leaders;
    for (const auto& record : records)
    {
        if (record.status != "completed")
            continue;
        if (!record.leaderScore.has_value() && !record.inferAccuracy.has_value())
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

inline std::string PlateauKey(const ExperimentRecord& record)
{
    std::ostringstream key;
    key << record.symbol
        << "|H" << record.horizon
        << "|threshold=" << FormatDouble(record.threshold)
        << "|core=" << GroupKeyDouble(record.coreLr)
        << "|head=" << GroupKeyDouble(record.headLr);
    return key.str();
}

inline std::vector<PlateauSignal> DetectPlateaus(const std::vector<ExperimentRecord>& records)
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

inline const GroupStats* BestGroup(const std::vector<GroupStats>& groups, const std::string& dimension)
{
    const GroupStats* best = nullptr;
    for (const auto& group : groups)
    {
        if (group.dimension != dimension || group.leaderScore.count == 0)
            continue;
        if (!best || group.leaderScore.best > best->leaderScore.best)
            best = &group;
    }
    return best;
}

inline std::vector<std::string> KnownSymbols()
{
    return {
        "audcadrmp",
        "audusdrmp",
        "eurusdrmp",
        "gbpusdrmp",
        "usdcadrmp",
        "usdpjyrmp"
    };
}

inline std::optional<std::string> SymbolFromScope(const std::string& scope)
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

inline std::vector<Recommendation> BuildRecommendations(const MetaAnalysisResult& result)
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
        recs.push_back(Recommendation{
            priority++,
            "Prioritize additional experiments for " + bestHorizon->key + ".",
            "This horizon currently has the best observed leader_score among horizon groups.",
            "best_leader_score=" + FormatDouble(bestHorizon->leaderScore.best) +
                ", samples=" + std::to_string(bestHorizon->sampleCount),
            "High if paired with under-sampled symbols.",
            ConfidenceForSampleSize(bestHorizon->sampleCount)
        });
    }

    if (const GroupStats* bestSymbol = BestGroup(result.groupStats, "symbol"))
    {
        recs.push_back(Recommendation{
            priority++,
            "Use " + bestSymbol->key + " as the near-term benchmark symbol.",
            "It has the strongest current symbol-level result.",
            "best_leader_score=" + FormatDouble(bestSymbol->leaderScore.best) +
                ", samples=" + std::to_string(bestSymbol->sampleCount),
            "Moderate: benchmark stability improves comparisons across horizons and LR settings.",
            ConfidenceForSampleSize(bestSymbol->sampleCount)
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

inline std::string GroupStatsJson(const std::vector<GroupStats>& groups)
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
             << ",\"mean\":" << FormatDouble(g.leaderScore.mean)
             << ",\"median\":" << FormatDouble(g.leaderScore.median)
             << ",\"best\":" << FormatDouble(g.leaderScore.best)
             << ",\"worst\":" << FormatDouble(g.leaderScore.worst)
             << ",\"stddev\":" << FormatDouble(g.leaderScore.stddev) << "}"
             << ",\"infer_accuracy\":{\"count\":" << g.inferAccuracy.count
             << ",\"mean\":" << FormatDouble(g.inferAccuracy.mean)
             << ",\"median\":" << FormatDouble(g.inferAccuracy.median)
             << ",\"best\":" << FormatDouble(g.inferAccuracy.best)
             << ",\"worst\":" << FormatDouble(g.inferAccuracy.worst)
             << ",\"stddev\":" << FormatDouble(g.inferAccuracy.stddev) << "}"
             << ",\"accept_rate\":{\"count\":" << g.acceptRate.count
             << ",\"mean\":" << FormatDouble(g.acceptRate.mean)
             << ",\"median\":" << FormatDouble(g.acceptRate.median)
             << ",\"best\":" << FormatDouble(g.acceptRate.best)
             << ",\"worst\":" << FormatDouble(g.acceptRate.worst)
             << ",\"stddev\":" << FormatDouble(g.acceptRate.stddev) << "}"
             << ",\"accept_accuracy\":{\"count\":" << g.acceptAccuracy.count
             << ",\"mean\":" << FormatDouble(g.acceptAccuracy.mean)
             << ",\"median\":" << FormatDouble(g.acceptAccuracy.median)
             << ",\"best\":" << FormatDouble(g.acceptAccuracy.best)
             << ",\"worst\":" << FormatDouble(g.acceptAccuracy.worst)
             << ",\"stddev\":" << FormatDouble(g.acceptAccuracy.stddev) << "}"
             << "}";
    }
    json << "]";
    return json.str();
}

inline std::string LeadersJson(const std::vector<LeaderRow>& leaders)
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

inline std::string RecommendationsJson(const std::vector<Recommendation>& recs)
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

inline std::string PlateauJson(const std::vector<PlateauSignal>& signals)
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

inline std::string BuildStatisticsJson(const MetaAnalysisResult& result)
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
         << ",\"group_statistics\":" << GroupStatsJson(result.groupStats)
         << ",\"plateau_signals\":" << PlateauJson(result.plateauSignals)
         << "}";
    return json.str();
}

inline std::string BuildCombinedJson(const MetaAnalysisResult& result)
{
    std::ostringstream json;
    json << "{"
         << "\"summary\":{"
         << "\"meta_analysis_id\":" << result.metaAnalysisId
         << ",\"scope\":" << JsonString(result.scope)
         << ",\"completed_experiments\":" << result.completedExperiments
         << ",\"completed_models\":" << result.completedModels
         << "}"
         << ",\"statistics\":" << result.statisticsJson
         << ",\"leaderboards\":" << result.leaderboardJson
         << ",\"recommendations\":" << result.recommendationsJson
         << "}";
    return json.str();
}

inline void AppendMetricTable(std::ostringstream& report,
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
        if (a.leaderScore.best != b.leaderScore.best)
            return a.leaderScore.best > b.leaderScore.best;
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
               << " | " << FormatDouble(group.leaderScore.best)
               << " | " << FormatDouble(group.leaderScore.mean)
               << " | " << FormatDouble(group.inferAccuracy.mean)
               << " | " << FormatDouble(group.acceptRate.mean)
               << " | " << ConfidenceForSampleSize(group.sampleCount)
               << " |\n";
    }
}

inline std::string BuildMarkdownReport(const MetaAnalysisResult& result, int limit)
{
    std::ostringstream report;
    report << "# LSTM Experiment Meta-Analysis\n\n";
    report << "Scope: `" << result.scope << "`\n\n";
    report << "## Executive Summary\n\n";
    report << "- Total experiments: " << result.totalExperiments << "\n";
    report << "- Completed experiments: " << result.completedExperiments << "\n";
    report << "- Failed experiments: " << result.failedExperiments << "\n";
    report << "- Completed models: " << result.completedModels << "\n";
    report << "- Success rate: " << FormatDouble(result.successRate) << "\n";
    report << "- Overall confidence: "
           << ConfidenceForSampleSize(static_cast<size_t>(result.completedExperiments)) << "\n";

    report << "\n## Current Leaders\n\n";
    report << "| Rank | Experiment | Model | Symbol | Horizon | Epochs | Infer Acc | Accept Acc | Accept Rate | Leader Score |\n";
    report << "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|\n";
    for (size_t i = 0; i < result.leaders.size() && static_cast<int>(i) < limit; ++i)
    {
        const auto& leader = result.leaders[i];
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

    AppendMetricTable(report, result.groupStats, "Best Models By Symbol", "symbol", limit);
    AppendMetricTable(report, result.groupStats, "Best Models By Horizon", "prediction_horizon", limit);
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

    report << "## Open Questions\n\n";
    report << "- Are current leaders repeatable across independent runs?\n";
    report << "- Which symbol/horizon pairs still lack enough completed samples?\n";
    report << "- Do longer epoch runs improve leader_score after the detected plateau points?\n";
    report << "- Are acceptance-rate gains aligned with inference accuracy gains?\n";

    report << "\n## Suggested Next Experiments\n\n";
    for (const auto& rec : result.recommendations)
        report << "- " << rec.action << "\n";

    return report.str();
}

inline MetaAnalysisResult BuildMetaAnalysis(pqxx::work& w, const MetaAnalysisOptions& options)
{
    MetaAnalysisResult result;
    result.scope = ScopeForOptions(options);
    LoadExperimentCounts(w, options, result);
    result.records = LoadExperimentRecords(w, options);
    result.groupStats = BuildAllGroupStats(result.records);
    result.leaders = BuildLeaders(result.records, options.limit);
    result.plateauSignals = DetectPlateaus(result.records);
    result.recommendations = BuildRecommendations(result);
    result.statisticsJson = BuildStatisticsJson(result);
    result.recommendationsJson = RecommendationsJson(result.recommendations);
    result.leaderboardJson = LeadersJson(result.leaders);
    result.markdown = BuildMarkdownReport(result, options.limit);
    return result;
}

inline void PrintMarkers(const MetaAnalysisResult& result)
{
    std::cout << "META_ANALYSIS_STATISTIC"
              << ",scope=" << result.scope
              << ",total_experiments=" << result.totalExperiments
              << ",completed_experiments=" << result.completedExperiments
              << ",failed_experiments=" << result.failedExperiments
              << ",completed_models=" << result.completedModels
              << ",success_rate=" << FormatDouble(result.successRate)
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
}

inline long long StoreMetaAnalysis(pqxx::work& w, const MetaAnalysisResult& result)
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

inline void WriteOutputFile(const std::string& path, const std::string& content)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("failed to open output file '" + path + "'");
    out << content;
}

inline int RunMetaAnalyze(const MetaAnalysisOptions& options)
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

inline int PrintLatestReport(const MetaAnalysisOptions& options)
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

inline int RunMetaAnalysisCli(int argc, const char* argv[])
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
                  << " --meta-analyze-once [--meta-analysis-json|--meta-analysis-markdown] "
                  << "[--meta-analysis-limit N] [--meta-analysis-symbol SYMBOL] "
                  << "[--meta-analysis-horizon N] [--meta-analysis-output FILE]\n"
                  << "Usage: " << (argc > 0 ? argv[0] : "LSTM_Release")
                  << " --meta-analysis-report [--meta-analysis-json|--meta-analysis-markdown] "
                  << "[--meta-analysis-symbol SYMBOL] [--meta-analysis-horizon N] "
                  << "[--meta-analysis-output FILE]\n";
        return 1;
    }

    try
    {
        if (options.metaAnalyze || options.metaAnalyzeOnce)
            return RunMetaAnalyze(options);
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
