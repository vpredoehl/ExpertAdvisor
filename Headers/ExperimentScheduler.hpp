#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

#include <pqxx/pqxx>

#include "CanonicalSymbol.hpp"
#include "PgModelIO.hpp"

namespace EA::ExperimentScheduler
{

struct SchedulerOptions
{
    bool scheduleExperiments = false;
    bool enqueueExperiment = false;
    bool analyzeCompletedExperiments = false;
    std::optional<long long> analyzeExperimentId;
    bool printLeaderboard = false;
    bool dryRun = false;
    bool schedulerOnce = false;
    int maxTrainProcs = 1;
    int maxInferProcs = 1;
    int maxAnalyzeProcs = 1;
    int schedulerPollSeconds = 30;
    std::string schedulerLogDir = "experiment_logs";
    std::string selfPath;

    std::optional<std::string> symbol;
    std::optional<int> predictionHorizon;
    std::optional<double> cNextThreshold;
    std::optional<double> coreLrMult;
    std::optional<double> headLrMult;
    std::optional<int> targetEpochs;
    int checkpointInterval = 20;
    std::optional<std::string> trainStart;
    std::optional<std::string> trainEnd;
    std::optional<std::string> inferStart;
    std::optional<std::string> inferEnd;
    std::optional<long long> resumeModelId;
    bool allowDuplicateExperiment = false;

    std::optional<std::string> leaderboardSymbol;
    std::optional<int> leaderboardHorizon;
    int leaderboardLimit = 20;
};

struct ExperimentRow
{
    long long experimentId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    double cNextThreshold = 0.0;
    std::optional<double> coreLrMult;
    std::optional<double> headLrMult;
    int targetEpochs = 0;
    int checkpointInterval = 20;
    std::string trainStart;
    std::string trainEnd;
    std::optional<std::string> inferStart;
    std::optional<std::string> inferEnd;
    std::optional<long long> lastModelId;
    std::optional<long long> resumeModelId;
    std::optional<std::string> trainLogPath;
    std::optional<std::string> inferLogPath;
    std::optional<std::string> analysisLogPath;
};

struct ChildResult
{
    int exitCode = 1;
    bool launched = false;
};

struct RunningExperimentChild
{
    ExperimentRow experiment;
    pid_t pid = -1;
    std::string logPath;
};

struct ParsedMetrics
{
    std::optional<long long> modelId;
    std::optional<int> completedEpochs;
    std::optional<double> trainAccuracy;
    std::optional<double> validationAccuracy;
    std::optional<double> inferAccuracy;
    std::optional<double> acceptAccuracy;
    std::optional<double> acceptRate;
    std::optional<double> lossLast;
    std::optional<bool> acceptModel;
    std::optional<std::string> rejectReason;
    long long confusion[3][3] = {};
    bool hasConfusion = false;
};

inline bool IsExperimentSchedulerCommand(int argc, const char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg{argv[i]};
        if (arg == "--schedule-experiments" ||
            arg == "--enqueue-experiment" ||
            arg == "--analyze-completed-experiments" ||
            arg == "--print-experiment-leaderboard" ||
            arg == "--analyze-experiment" ||
            arg.rfind("--analyze-experiment=", 0) == 0)
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

inline std::string SqlNullable(pqxx::work& w, const std::optional<std::string>& value)
{
    return value.has_value() ? w.quote(*value) : "NULL";
}

inline std::string SqlNullable(pqxx::work& w, const std::optional<double>& value)
{
    return value.has_value() ? w.quote(*value) : "NULL";
}

inline std::string SqlNullable(pqxx::work& w, const std::optional<long long>& value)
{
    return value.has_value() ? w.quote(*value) : "NULL";
}

inline std::string FormatDouble(double value)
{
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
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

inline long long ParsePositiveLongLong(const std::string& optionName, const std::string& value)
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
    if (consumed != value.size() || parsed <= 0)
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return parsed;
}

inline double ParsePositiveDouble(const std::string& optionName, const std::string& value)
{
    size_t consumed = 0;
    double parsed = 0.0;
    try
    {
        parsed = std::stod(value, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size() || parsed <= 0.0)
        throw std::invalid_argument("invalid " + optionName + " value '" + value + "'");
    return parsed;
}

inline SchedulerOptions ParseSchedulerArgs(int argc, const char* argv[])
{
    SchedulerOptions options;
    options.selfPath = argc > 0 ? argv[0] : "./LSTM_Release";

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg{argv[i]};
        std::string value;

        if (arg == "--schedule-experiments")
            options.scheduleExperiments = true;
        else if (arg == "--enqueue-experiment")
            options.enqueueExperiment = true;
        else if (arg == "--analyze-completed-experiments")
            options.analyzeCompletedExperiments = true;
        else if (arg == "--print-experiment-leaderboard")
            options.printLeaderboard = true;
        else if (arg == "--dry-run")
            options.dryRun = true;
        else if (arg == "--scheduler-once")
            options.schedulerOnce = true;
        else if (arg == "--allow-duplicate-experiment")
            options.allowDuplicateExperiment = true;
        else if (arg == "--analyze-experiment")
            options.analyzeExperimentId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--symbol")
            options.symbol = EA::CanonicalSymbol::Normalize(RequireNextArg(argc, argv, i, arg));
        else if (arg == "--prediction-horizon")
            options.predictionHorizon = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--c-next-threshold")
            options.cNextThreshold = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--core-lr-mult")
            options.coreLrMult = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--head-lr-mult")
            options.headLrMult = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--target-epochs")
            options.targetEpochs = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--checkpoint-interval")
            options.checkpointInterval = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--train-start")
            options.trainStart = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--train-end")
            options.trainEnd = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--infer-start")
            options.inferStart = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--infer-end")
            options.inferEnd = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--resume-model-id")
            options.resumeModelId = ParsePositiveLongLong(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--max-train-procs")
            options.maxTrainProcs = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--max-infer-procs")
            options.maxInferProcs = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--max-analyze-procs")
            options.maxAnalyzeProcs = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--scheduler-poll-seconds")
            options.schedulerPollSeconds = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--scheduler-log-dir")
            options.schedulerLogDir = RequireNextArg(argc, argv, i, arg);
        else if (arg == "--leaderboard-symbol")
            options.leaderboardSymbol = EA::CanonicalSymbol::Normalize(RequireNextArg(argc, argv, i, arg));
        else if (arg == "--leaderboard-horizon")
            options.leaderboardHorizon = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--leaderboard-limit")
            options.leaderboardLimit = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (SplitOptionWithValue(arg, "--symbol", value))
            options.symbol = EA::CanonicalSymbol::Normalize(value);
        else if (SplitOptionWithValue(arg, "--prediction-horizon", value))
            options.predictionHorizon = ParsePositiveInt("--prediction-horizon", value);
        else if (SplitOptionWithValue(arg, "--c-next-threshold", value))
            options.cNextThreshold = ParsePositiveDouble("--c-next-threshold", value);
        else if (SplitOptionWithValue(arg, "--core-lr-mult", value))
            options.coreLrMult = ParsePositiveDouble("--core-lr-mult", value);
        else if (SplitOptionWithValue(arg, "--head-lr-mult", value))
            options.headLrMult = ParsePositiveDouble("--head-lr-mult", value);
        else if (SplitOptionWithValue(arg, "--target-epochs", value))
            options.targetEpochs = ParsePositiveInt("--target-epochs", value);
        else if (SplitOptionWithValue(arg, "--checkpoint-interval", value))
            options.checkpointInterval = ParsePositiveInt("--checkpoint-interval", value);
        else if (SplitOptionWithValue(arg, "--train-start", value))
            options.trainStart = value;
        else if (SplitOptionWithValue(arg, "--train-end", value))
            options.trainEnd = value;
        else if (SplitOptionWithValue(arg, "--infer-start", value))
            options.inferStart = value;
        else if (SplitOptionWithValue(arg, "--infer-end", value))
            options.inferEnd = value;
        else if (SplitOptionWithValue(arg, "--resume-model-id", value))
            options.resumeModelId = ParsePositiveLongLong("--resume-model-id", value);
        else if (SplitOptionWithValue(arg, "--max-train-procs", value))
            options.maxTrainProcs = ParsePositiveInt("--max-train-procs", value);
        else if (SplitOptionWithValue(arg, "--max-infer-procs", value))
            options.maxInferProcs = ParsePositiveInt("--max-infer-procs", value);
        else if (SplitOptionWithValue(arg, "--max-analyze-procs", value))
            options.maxAnalyzeProcs = ParsePositiveInt("--max-analyze-procs", value);
        else if (SplitOptionWithValue(arg, "--scheduler-poll-seconds", value))
            options.schedulerPollSeconds = ParsePositiveInt("--scheduler-poll-seconds", value);
        else if (SplitOptionWithValue(arg, "--scheduler-log-dir", value))
            options.schedulerLogDir = value;
        else if (SplitOptionWithValue(arg, "--analyze-experiment", value))
            options.analyzeExperimentId = ParsePositiveLongLong("--analyze-experiment", value);
        else if (SplitOptionWithValue(arg, "--leaderboard-symbol", value))
            options.leaderboardSymbol = EA::CanonicalSymbol::Normalize(value);
        else if (SplitOptionWithValue(arg, "--leaderboard-horizon", value))
            options.leaderboardHorizon = ParsePositiveInt("--leaderboard-horizon", value);
        else if (SplitOptionWithValue(arg, "--leaderboard-limit", value))
            options.leaderboardLimit = ParsePositiveInt("--leaderboard-limit", value);
        else if (arg.rfind("--", 0) == 0)
            throw std::invalid_argument("unknown scheduler option '" + arg + "'");
        else
            throw std::invalid_argument("unexpected positional scheduler argument '" + arg + "'");
    }

    const int commandCount =
        (options.scheduleExperiments ? 1 : 0) +
        (options.enqueueExperiment ? 1 : 0) +
        (options.analyzeCompletedExperiments ? 1 : 0) +
        (options.analyzeExperimentId.has_value() ? 1 : 0) +
        (options.printLeaderboard ? 1 : 0);
    if (commandCount != 1)
        throw std::invalid_argument("expected exactly one experiment scheduler command");

    return options;
}

inline void EnsureRequiredEnqueueOptions(const SchedulerOptions& options)
{
    if ((!options.symbol.has_value() && !options.resumeModelId.has_value()) ||
        !options.predictionHorizon.has_value() ||
        !options.cNextThreshold.has_value() ||
        !options.targetEpochs.has_value() ||
        !options.trainStart.has_value() ||
        !options.trainEnd.has_value())
    {
        throw std::invalid_argument("--enqueue-experiment requires --symbol unless --resume-model-id is supplied, plus --prediction-horizon, --c-next-threshold, --target-epochs, --train-start, and --train-end");
    }
    if (options.inferStart.has_value() != options.inferEnd.has_value())
        throw std::invalid_argument("--infer-start and --infer-end must be supplied together");
}

inline bool TableExists(pqxx::work& w, const std::string& tableName)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = $1 LIMIT 1;",
        tableName);
    return !r.empty();
}

inline bool RequireSchedulerTables(pqxx::work& w)
{
    std::vector<std::string> missing;
    if (!TableExists(w, "experiment"))
        missing.push_back("experiment");
    if (!TableExists(w, "experiment_analysis_result"))
        missing.push_back("experiment_analysis_result");

    if (missing.empty())
        return true;

    std::ostringstream oss;
    for (size_t i = 0; i < missing.size(); ++i)
    {
        if (i)
            oss << "|";
        oss << missing[i];
    }
    std::cerr << "DATABASE_MIGRATION_REQUIRED"
              << ",missing=" << oss.str()
              << ",command=./migrate_lstm_db.sh"
              << std::endl;
    return false;
}

inline void SetTransactionReadWrite(pqxx::work& w)
{
    w.exec("SET TRANSACTION READ WRITE;");
}

inline void PrintModelSymbolMismatch(const std::string& runtimeSymbol,
                                     const std::string& modelSymbol)
{
    std::cerr << "MODEL_SYMBOL_MISMATCH"
              << ",runtime=" << runtimeSymbol
              << ",model=" << modelSymbol
              << std::endl;
}

inline void PrintModelSymbolMissing(long long modelId)
{
    std::cout << "MODEL_SYMBOL_MISSING"
              << ",model_id=" << modelId
              << std::endl;
}

inline std::optional<std::string> TryLoadPersistedCanonicalSymbol(pqxx::work& w,
                                                                  long long modelId)
{
    try
    {
        return DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

inline std::string ResolveExperimentCanonicalSymbol(pqxx::work& w,
                                                    const SchedulerOptions& options)
{
    if (!options.resumeModelId.has_value())
        return EA::CanonicalSymbol::Normalize(*options.symbol);

    const std::optional<std::string> persistedSymbol =
        TryLoadPersistedCanonicalSymbol(w, *options.resumeModelId);
    if (persistedSymbol.has_value())
    {
        if (options.symbol.has_value())
        {
            const std::string runtimeSymbol = EA::CanonicalSymbol::Normalize(*options.symbol);
            if (runtimeSymbol != *persistedSymbol)
            {
                PrintModelSymbolMismatch(runtimeSymbol, *persistedSymbol);
                throw std::runtime_error("MODEL_SYMBOL_MISMATCH");
            }
        }
        std::cout << "MODEL_SYMBOL"
                  << ",source=database"
                  << ",model_id=" << *options.resumeModelId
                  << ",symbol=" << *persistedSymbol
                  << std::endl;
        return *persistedSymbol;
    }

    PrintModelSymbolMissing(*options.resumeModelId);
    if (options.symbol.has_value())
    {
        const std::string legacySymbol = EA::CanonicalSymbol::Normalize(*options.symbol);
        std::cout << "MODEL_SYMBOL"
                  << ",source=legacy"
                  << ",model_id=" << *options.resumeModelId
                  << ",symbol=" << legacySymbol
                  << ",warning=missing_metadata"
                  << std::endl;
        return legacySymbol;
    }

    throw std::runtime_error("resume-model-id is missing train_symbol_meta and --symbol was not supplied for legacy fallback");
}

inline std::string DuplicateWhereClause(pqxx::work& w,
                                        const SchedulerOptions& options,
                                        const std::string& canonicalSymbol)
{
    std::ostringstream sql;
    sql << "symbol = " << w.quote(canonicalSymbol)
        << " AND prediction_horizon = " << *options.predictionHorizon
        << " AND c_next_threshold = " << FormatDouble(*options.cNextThreshold)
        << " AND core_lr_mult IS NOT DISTINCT FROM " << SqlNullable(w, options.coreLrMult)
        << " AND head_lr_mult IS NOT DISTINCT FROM " << SqlNullable(w, options.headLrMult)
        << " AND target_epochs = " << *options.targetEpochs
        << " AND checkpoint_interval = " << options.checkpointInterval
        << " AND train_start = " << w.quote(*options.trainStart) << "::timestamptz"
        << " AND train_end = " << w.quote(*options.trainEnd) << "::timestamptz"
        << " AND infer_start IS NOT DISTINCT FROM " << SqlNullable(w, options.inferStart)
        << "::timestamptz"
        << " AND infer_end IS NOT DISTINCT FROM " << SqlNullable(w, options.inferEnd)
        << "::timestamptz"
        << " AND resume_model_id IS NOT DISTINCT FROM " << SqlNullable(w, options.resumeModelId)
        << " AND status <> 'cancelled'";
    return sql.str();
}

inline long long CurrentDuplicateNonce()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

inline int EnqueueExperiment(const SchedulerOptions& options)
{
    EnsureRequiredEnqueueOptions(options);

    std::optional<std::string> dryRunSymbol;
    if (options.symbol.has_value())
        dryRunSymbol = EA::CanonicalSymbol::Normalize(*options.symbol);

    if (options.dryRun)
    {
        std::cout << "SCHEDULER_DRY_RUN=1" << std::endl;
        std::cout << "EXPERIMENT_ENQUEUE_DRY_RUN"
                  << ",symbol=" << (dryRunSymbol.has_value() ? *dryRunSymbol : "database_model")
                  << ",prediction_horizon=" << *options.predictionHorizon
                  << ",c_next_threshold=" << FormatDouble(*options.cNextThreshold)
                  << ",core_lr_mult=" << (options.coreLrMult.has_value() ? FormatDouble(*options.coreLrMult) : "NULL")
                  << ",head_lr_mult=" << (options.headLrMult.has_value() ? FormatDouble(*options.headLrMult) : "NULL")
                  << ",target_epochs=" << *options.targetEpochs
                  << ",checkpoint_interval=" << options.checkpointInterval
                  << ",train_start=" << *options.trainStart
                  << ",train_end=" << *options.trainEnd
                  << ",infer_start=" << (options.inferStart.has_value() ? *options.inferStart : "NULL")
                  << ",infer_end=" << (options.inferEnd.has_value() ? *options.inferEnd : "NULL")
                  << std::endl;
        return 0;
    }

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
        return 1;

    const std::string canonicalSymbol = ResolveExperimentCanonicalSymbol(w, options);

    if (!options.allowDuplicateExperiment)
    {
        pqxx::result duplicate = w.exec(
            "SELECT experiment_id FROM experiment WHERE " +
            DuplicateWhereClause(w, options, canonicalSymbol) +
            " LIMIT 1;");
        if (!duplicate.empty())
        {
            std::cout << "SCHEDULER_DUPLICATE_REJECTED"
                      << ",experiment_id=" << duplicate[0][0].as<long long>()
                      << ",symbol=" << canonicalSymbol
                      << ",prediction_horizon=" << *options.predictionHorizon
                      << std::endl;
            return 1;
        }
    }

    const long long duplicateNonce = options.allowDuplicateExperiment ? CurrentDuplicateNonce() : 0;
    std::ostringstream sql;
    sql << "INSERT INTO experiment ("
        << "symbol, prediction_horizon, c_next_threshold, core_lr_mult, head_lr_mult, "
        << "target_epochs, checkpoint_interval, train_start, train_end, infer_start, infer_end, "
        << "resume_model_id, duplicate_nonce, status, phase, updated_at"
        << ") VALUES ("
        << w.quote(canonicalSymbol) << ","
        << *options.predictionHorizon << ","
        << FormatDouble(*options.cNextThreshold) << ","
        << SqlNullable(w, options.coreLrMult) << ","
        << SqlNullable(w, options.headLrMult) << ","
        << *options.targetEpochs << ","
        << options.checkpointInterval << ","
        << w.quote(*options.trainStart) << "::timestamptz,"
        << w.quote(*options.trainEnd) << "::timestamptz,"
        << SqlNullable(w, options.inferStart) << "::timestamptz,"
        << SqlNullable(w, options.inferEnd) << "::timestamptz,"
        << SqlNullable(w, options.resumeModelId) << ","
        << duplicateNonce << ","
        << "'pending','train',now()) RETURNING experiment_id;";

    pqxx::result inserted = w.exec(sql.str());
    const long long experimentId = inserted[0][0].as<long long>();
    w.commit();

    std::cout << "EXPERIMENT_ENQUEUED"
              << ",experiment_id=" << experimentId
              << ",symbol=" << canonicalSymbol
              << ",prediction_horizon=" << *options.predictionHorizon
              << ",target_epochs=" << *options.targetEpochs
              << std::endl;
    return 0;
}

inline std::optional<double> OptionalDoubleCell(const pqxx::row& row, int index)
{
    if (row[index].is_null())
        return std::nullopt;
    return row[index].as<double>();
}

inline std::optional<long long> OptionalLongLongCell(const pqxx::row& row, int index)
{
    if (row[index].is_null())
        return std::nullopt;
    return row[index].as<long long>();
}

inline std::optional<std::string> OptionalStringCell(const pqxx::row& row, int index)
{
    if (row[index].is_null())
        return std::nullopt;
    return row[index].as<std::string>();
}

inline ExperimentRow RowToExperiment(const pqxx::row& row)
{
    ExperimentRow experiment;
    experiment.experimentId = row[0].as<long long>();
    experiment.symbol = EA::CanonicalSymbol::Normalize(row[1].as<std::string>());
    experiment.predictionHorizon = row[2].as<int>();
    experiment.cNextThreshold = row[3].as<double>();
    experiment.coreLrMult = OptionalDoubleCell(row, 4);
    experiment.headLrMult = OptionalDoubleCell(row, 5);
    experiment.targetEpochs = row[6].as<int>();
    experiment.checkpointInterval = row[7].as<int>();
    experiment.trainStart = row[8].as<std::string>();
    experiment.trainEnd = row[9].as<std::string>();
    experiment.inferStart = OptionalStringCell(row, 10);
    experiment.inferEnd = OptionalStringCell(row, 11);
    experiment.lastModelId = OptionalLongLongCell(row, 12);
    experiment.resumeModelId = OptionalLongLongCell(row, 13);
    experiment.trainLogPath = OptionalStringCell(row, 14);
    experiment.inferLogPath = OptionalStringCell(row, 15);
    experiment.analysisLogPath = OptionalStringCell(row, 16);
    return experiment;
}

inline std::vector<ExperimentRow> LoadPendingExperiments(pqxx::work& w,
                                                        const std::string& phase,
                                                        int limit)
{
    pqxx::result rows = w.exec_params(
        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
        "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
        "train_start::text, train_end::text, infer_start::text, infer_end::text, "
        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path "
        "FROM experiment "
        "WHERE status = 'pending' AND phase = $1 "
        "ORDER BY updated_at ASC, experiment_id ASC "
        "LIMIT $2;",
        phase,
        limit);

    std::vector<ExperimentRow> experiments;
    experiments.reserve(rows.size());
    for (const auto& row : rows)
        experiments.push_back(RowToExperiment(row));
    return experiments;
}

inline void RecoverOrphanedRunningExperiments(pqxx::work& w)
{
    pqxx::result rows = w.exec(
        "UPDATE experiment "
        "SET status = 'failed', "
        "exit_code = -1, "
        "error_message = 'SCHEDULER_RECOVERED_ORPHANED_RUNNING_EXPERIMENT', "
        "completed_at = now(), "
        "updated_at = now() "
        "WHERE status = 'running' "
        "RETURNING experiment_id;");
    for (const auto& row : rows)
    {
        std::cout << "SCHEDULER_RECOVERED_ORPHANED_RUNNING_EXPERIMENT"
                  << ",experiment_id=" << row[0].as<long long>()
                  << std::endl;
    }
}

inline void EnsureLogDir(const std::string& logDir)
{
    std::filesystem::create_directories(logDir);
}

inline std::string LogPathFor(const SchedulerOptions& options,
                              const ExperimentRow& experiment,
                              const std::string& phase)
{
    std::ostringstream oss;
    oss << options.schedulerLogDir
        << "/experiment_" << experiment.experimentId
        << "_" << EA::CanonicalSymbol::Normalize(experiment.symbol)
        << "_" << phase << ".log";
    return oss.str();
}

inline std::string BaseModelName(const ExperimentRow& experiment)
{
    std::ostringstream oss;
    oss << EA::CanonicalSymbol::Normalize(experiment.symbol)
        << "-experiment" << experiment.experimentId
        << "_h" << experiment.predictionHorizon
        << "_e" << experiment.targetEpochs;
    return oss.str();
}

inline std::string ShellDisplayQuote(const std::string& value)
{
    if (value.find_first_of(" \t\n\"'\\$`") == std::string::npos)
        return value;
    std::string quoted = "'";
    for (char c : value)
    {
        if (c == '\'')
            quoted += "'\\''";
        else
            quoted.push_back(c);
    }
    quoted.push_back('\'');
    return quoted;
}

inline std::string CommandForDisplay(const std::vector<std::string>& argv)
{
    std::ostringstream oss;
    for (size_t i = 0; i < argv.size(); ++i)
    {
        if (i)
            oss << ' ';
        oss << ShellDisplayQuote(argv[i]);
    }
    return oss.str();
}

inline void AddCliFlag(std::vector<std::string>& argv, const std::string& optionName)
{
    argv.push_back(optionName);
}

inline void AddCliOption(std::vector<std::string>& argv,
                         const std::string& optionName,
                         const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument(optionName + " requires a non-empty value");
    argv.push_back(optionName + "=" + value);
}

inline void AddCliPositional(std::vector<std::string>& argv, const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument("positional argument must not be empty");
    argv.push_back(value);
}

inline void PrintSchedulerExec(const std::vector<std::string>& argv)
{
    std::cout << "SCHEDULER_EXEC: "
              << CommandForDisplay(argv)
              << std::endl;
}

inline pid_t LaunchChildProcess(const std::vector<std::string>& argv, const std::string& logPath)
{
    if (argv.empty())
        throw std::runtime_error("empty child argv");

    const int fd = ::open(logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        throw std::runtime_error("failed to open log file '" + logPath + "'");

    pid_t pid = ::fork();
    if (pid < 0)
    {
        ::close(fd);
        throw std::runtime_error("fork failed");
    }

    if (pid == 0)
    {
        ::dup2(fd, STDOUT_FILENO);
        ::dup2(fd, STDERR_FILENO);
        ::close(fd);

        std::vector<char*> childArgv;
        childArgv.reserve(argv.size() + 1);
        for (const auto& arg : argv)
            childArgv.push_back(const_cast<char*>(arg.c_str()));
        childArgv.push_back(nullptr);

        if (argv[0].find('/') != std::string::npos)
            ::execv(argv[0].c_str(), childArgv.data());
        else
            ::execvp(argv[0].c_str(), childArgv.data());
        _exit(127);
    }

    ::close(fd);
    return pid;
}

inline int WaitForChildProcess(pid_t pid)
{
    if (pid <= 0)
        return 1;

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0)
    {
        return 1;
    }

    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    else if (WIFSIGNALED(status))
        return 128 + WTERMSIG(status);
    return 1;
}

inline ChildResult RunChild(const std::vector<std::string>& argv, const std::string& logPath)
{
    ChildResult result;
    const pid_t pid = LaunchChildProcess(argv, logPath);
    result.launched = true;
    result.exitCode = WaitForChildProcess(pid);
    return result;
}

inline std::vector<std::string> BuildTrainCommand(const SchedulerOptions& options,
                                                  const ExperimentRow& experiment)
{
    std::vector<std::string> argv;
    argv.push_back(options.selfPath);
    AddCliFlag(argv, "--train");
    AddCliOption(argv, "--log-level", "summary");
    AddCliOption(argv, "--checkpoint-every", std::to_string(experiment.checkpointInterval));
    AddCliOption(argv, "--new-model-name", BaseModelName(experiment));

    const std::optional<long long> resumeFrom =
        experiment.resumeModelId.has_value() ? experiment.resumeModelId : experiment.lastModelId;
    if (resumeFrom.has_value())
    {
        AddCliOption(argv, "--resume-model-id", std::to_string(*resumeFrom));
        AddCliOption(argv, "--target-epochs", std::to_string(experiment.targetEpochs));
        return argv;
    }

    AddCliOption(argv, "--symbol", experiment.symbol);
    AddCliOption(argv, "--prediction-horizon", std::to_string(experiment.predictionHorizon));
    AddCliOption(argv, "--threshold", FormatDouble(experiment.cNextThreshold));
    AddCliOption(argv, "--epochs", std::to_string(experiment.targetEpochs));
    if (experiment.coreLrMult.has_value())
        AddCliOption(argv, "--core-lr-mult", FormatDouble(*experiment.coreLrMult));
    if (experiment.headLrMult.has_value())
        AddCliOption(argv, "--head-weight-lr-mult", FormatDouble(*experiment.headLrMult));
    AddCliPositional(argv, experiment.trainStart.substr(0, 10));
    AddCliPositional(argv, experiment.trainEnd.substr(0, 10));
    return argv;
}

inline std::vector<std::string> BuildInferCommand(const SchedulerOptions& options,
                                                  const ExperimentRow& experiment)
{
    if (!experiment.lastModelId.has_value())
        throw std::runtime_error("infer phase has no last_model_id");
    if (!experiment.inferStart.has_value() || !experiment.inferEnd.has_value())
        throw std::runtime_error("infer phase has no infer date range");

    std::vector<std::string> argv;
    argv.push_back(options.selfPath);
    AddCliFlag(argv, "--infer");
    AddCliOption(argv, "--model", std::to_string(*experiment.lastModelId));
    AddCliOption(argv, "--log-level", "summary");
    AddCliPositional(argv, experiment.inferStart->substr(0, 10));
    AddCliPositional(argv, experiment.inferEnd->substr(0, 10));
    return argv;
}

inline std::vector<std::string> BuildAnalyzeCommand(const SchedulerOptions& options,
                                                    const ExperimentRow& experiment)
{
    std::vector<std::string> argv;
    argv.push_back(options.selfPath);
    AddCliOption(argv, "--analyze-experiment", std::to_string(experiment.experimentId));
    return argv;
}

inline std::string ReadFileIfExists(const std::optional<std::string>& path)
{
    if (!path.has_value())
        return {};
    std::ifstream in{*path};
    if (!in)
        return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline void WriteTextFile(const std::string& path, const std::string& text)
{
    std::ofstream out{path};
    out << text;
}

inline std::optional<long long> ExtractLastModelId(const std::string& text)
{
    std::regex idRegex{
        "(Saved model with model_id=|Created new model_id=|RESUME_SAVED_NEW_MODEL_ID=|CHECKPOINT_SAVE_DONE[^\\n]*model_id=)([0-9]+)"};
    std::optional<long long> last;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), idRegex);
         it != std::sregex_iterator();
         ++it)
    {
        last = std::stoll((*it)[2].str());
    }
    return last;
}

inline std::string ModelIdSearchPatternDescription()
{
    return "Saved model with model_id=;Created new model_id=;RESUME_SAVED_NEW_MODEL_ID=;CHECKPOINT_SAVE_DONE model_id=";
}

inline std::string LastLines(const std::string& text, size_t lineCount)
{
    std::vector<std::string> lines;
    std::istringstream input{text};
    std::string line;
    while (std::getline(input, line))
    {
        lines.push_back(line);
        if (lines.size() > lineCount)
            lines.erase(lines.begin());
    }

    std::ostringstream out;
    for (const auto& item : lines)
        out << item << "\n";
    return out.str();
}

inline void PrintModelIdDetectionWarning(long long experimentId,
                                         const std::string& logPath,
                                         const std::string& logText)
{
    std::cout << "SCHEDULER_MODEL_ID_PARSE_WARNING"
              << ",experiment_id=" << experimentId
              << ",log_path=" << logPath
              << ",patterns=" << ShellDisplayQuote(ModelIdSearchPatternDescription())
              << std::endl;
    std::cout << "SCHEDULER_TRAIN_LOG_TAIL_BEGIN"
              << ",experiment_id=" << experimentId
              << ",lines=50"
              << std::endl;
    std::istringstream tail{LastLines(logText, 50)};
    std::string line;
    while (std::getline(tail, line))
    {
        std::cout << "SCHEDULER_TRAIN_LOG_TAIL"
                  << ",experiment_id=" << experimentId
                  << ",line=" << line
                  << std::endl;
    }
    std::cout << "SCHEDULER_TRAIN_LOG_TAIL_END"
              << ",experiment_id=" << experimentId
              << std::endl;
}

inline std::optional<double> ExtractLastDouble(const std::string& text, const std::regex& regex)
{
    std::optional<double> value;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), regex);
         it != std::sregex_iterator();
         ++it)
    {
        value = std::stod((*it)[1].str());
    }
    return value;
}

inline ParsedMetrics ParseMetricsFromLogs(const ExperimentRow& experiment)
{
    ParsedMetrics metrics;
    const std::string trainLog = ReadFileIfExists(experiment.trainLogPath);
    const std::string inferLog = ReadFileIfExists(experiment.inferLogPath);
    const std::string combined = trainLog + "\n" + inferLog;

    metrics.modelId = ExtractLastModelId(combined);
    if (!metrics.modelId.has_value())
        metrics.modelId = experiment.lastModelId;

    metrics.inferAccuracy = ExtractLastDouble(
        inferLog,
        std::regex{"Overall 3-class accuracy:\\s*([0-9]+(?:\\.[0-9]+)?)%"});
    if (metrics.inferAccuracy.has_value())
        metrics.inferAccuracy = *metrics.inferAccuracy / 100.0;

    metrics.validationAccuracy = ExtractLastDouble(
        combined,
        std::regex{"EPOCH_3CLASS_ACCURACY[^\\n]*(?:validation_accuracy|val_accuracy|accuracy)=([0-9]+(?:\\.[0-9]+)?)"});
    metrics.trainAccuracy = ExtractLastDouble(
        combined,
        std::regex{"EPOCH_3CLASS_ACCURACY[^\\n]*(?:train_accuracy)=([0-9]+(?:\\.[0-9]+)?)"});
    metrics.lossLast = ExtractLastDouble(
        combined,
        std::regex{"(?:loss|LOSS)[= :]([0-9]+(?:\\.[0-9]+)?)"});

    const std::optional<double> parsedCompletedEpochs = ExtractLastDouble(
        combined,
        std::regex{"(?:completed_epochs|epochs_trained|RESUME_TARGET_EPOCH)[= ]([0-9]+)"});
    if (parsedCompletedEpochs.has_value())
        metrics.completedEpochs = static_cast<int>(*parsedCompletedEpochs);

    std::regex confusionRegex{
        "Overall 3-class confusion matrix[^\\n]*\\[\\[([0-9]+),\\s*([0-9]+),\\s*([0-9]+)\\],\\s*\\[([0-9]+),\\s*([0-9]+),\\s*([0-9]+)\\],\\s*\\[([0-9]+),\\s*([0-9]+),\\s*([0-9]+)\\]\\]"};
    std::smatch match;
    std::string::const_iterator searchStart = inferLog.cbegin();
    while (std::regex_search(searchStart, inferLog.cend(), match, confusionRegex))
    {
        for (int i = 0; i < 9; ++i)
            metrics.confusion[i / 3][i % 3] = std::stoll(match[i + 1].str());
        metrics.hasConfusion = true;
        searchStart = match.suffix().first;
    }

    std::regex acceptRegex{"ACCEPT_MODEL=(true|false)"};
    for (auto it = std::sregex_iterator(combined.begin(), combined.end(), acceptRegex);
         it != std::sregex_iterator();
         ++it)
    {
        metrics.acceptModel = ((*it)[1].str() == "true");
    }

    std::regex rejectRegex{"REJECT_REASON=([^\\n,]+)"};
    for (auto it = std::sregex_iterator(combined.begin(), combined.end(), rejectRegex);
         it != std::sregex_iterator();
         ++it)
    {
        metrics.rejectReason = (*it)[1].str();
    }

    if (metrics.hasConfusion)
    {
        long long total = 0;
        long long correct = 0;
        for (int actual = 0; actual < 3; ++actual)
        {
            for (int pred = 0; pred < 3; ++pred)
            {
                total += metrics.confusion[actual][pred];
                if (actual == pred)
                    correct += metrics.confusion[actual][pred];
            }
        }
        if (!metrics.inferAccuracy.has_value() && total > 0)
            metrics.inferAccuracy = static_cast<double>(correct) / static_cast<double>(total);
        metrics.acceptRate = metrics.acceptModel.has_value() && *metrics.acceptModel ? 1.0 : 0.0;
        metrics.acceptAccuracy = metrics.inferAccuracy;
    }

    return metrics;
}

inline void ApplyPersistedSymbolToAnalysisExperiment(pqxx::work& w,
                                                     ExperimentRow& experiment,
                                                     const ParsedMetrics& metrics)
{
    const std::optional<long long> modelId =
        metrics.modelId.has_value() ? metrics.modelId : experiment.lastModelId;
    if (!modelId.has_value())
    {
        experiment.symbol = EA::CanonicalSymbol::Normalize(experiment.symbol);
        return;
    }

    const std::optional<std::string> persistedSymbol =
        TryLoadPersistedCanonicalSymbol(w, *modelId);
    if (persistedSymbol.has_value())
    {
        experiment.symbol = *persistedSymbol;
        std::cout << "MODEL_SYMBOL"
                  << ",source=database"
                  << ",model_id=" << *modelId
                  << ",symbol=" << experiment.symbol
                  << std::endl;
        return;
    }

    PrintModelSymbolMissing(*modelId);
    experiment.symbol = EA::CanonicalSymbol::Normalize(experiment.symbol);
    std::cout << "MODEL_SYMBOL"
              << ",source=legacy"
              << ",model_id=" << *modelId
              << ",symbol=" << experiment.symbol
              << ",warning=missing_metadata"
              << std::endl;
}

inline bool ApplyStructuredInferenceMetrics(pqxx::work& w,
                                            const ExperimentRow& experiment,
                                            ParsedMetrics& metrics)
{
    if (!TableExists(w, "inference_eval_result") ||
        !experiment.lastModelId.has_value() ||
        !experiment.inferStart.has_value() ||
        !experiment.inferEnd.has_value())
    {
        return false;
    }

    pqxx::result rows = w.exec_params(
        "SELECT accuracy, accept_model, COALESCE(reject_reason, ''), completed_epochs "
        "FROM inference_eval_result "
        "WHERE model_id = $1 "
        "AND symbol = $2 "
        "AND prediction_horizon = $3 "
        "AND threshold_logret = $4 "
        "AND from_date = $5 "
        "AND to_date = $6 "
        "AND status = 'completed' "
        "ORDER BY completed_at DESC LIMIT 1;",
        *experiment.lastModelId,
        experiment.symbol,
        experiment.predictionHorizon,
        experiment.cNextThreshold,
        experiment.inferStart->substr(0, 10),
        experiment.inferEnd->substr(0, 10));
    if (rows.empty())
        return false;

    if (!rows[0][0].is_null())
        metrics.inferAccuracy = rows[0][0].as<double>();
    if (!rows[0][1].is_null())
        metrics.acceptModel = rows[0][1].as<bool>();
    const std::string rejectReason = rows[0][2].as<std::string>();
    if (!rejectReason.empty())
        metrics.rejectReason = rejectReason;
    if (!rows[0][3].is_null())
        metrics.completedEpochs = rows[0][3].as<int>();
    metrics.modelId = experiment.lastModelId;
    metrics.acceptAccuracy = metrics.inferAccuracy;
    metrics.acceptRate = metrics.acceptModel.has_value() && *metrics.acceptModel ? 1.0 : 0.0;
    return true;
}

inline double PredictionImbalancePenalty(const ParsedMetrics& metrics)
{
    if (!metrics.hasConfusion)
        return 1.0;

    long long pred[3] = {};
    long long total = 0;
    for (int actual = 0; actual < 3; ++actual)
    {
        for (int cls = 0; cls < 3; ++cls)
        {
            pred[cls] += metrics.confusion[actual][cls];
            total += metrics.confusion[actual][cls];
        }
    }
    if (total == 0)
        return 1.0;

    const double maxPredFrac = static_cast<double>(*std::max_element(pred, pred + 3)) /
                               static_cast<double>(total);
    if (maxPredFrac <= 0.60)
        return 1.0;
    return std::max(0.25, 1.0 - ((maxPredFrac - 0.60) / 0.40));
}

inline std::optional<double> ComputeLeaderScore(const ParsedMetrics& metrics)
{
    if (!metrics.inferAccuracy.has_value())
        return std::nullopt;
    const double acceptAccuracy = metrics.acceptAccuracy.value_or(*metrics.inferAccuracy);
    const double penalty = PredictionImbalancePenalty(metrics);
    return (*metrics.inferAccuracy) * (0.75 + 0.25 * acceptAccuracy) * penalty;
}

inline std::string MetricSql(pqxx::work& w, const std::optional<double>& value)
{
    return value.has_value() ? FormatDouble(*value) : "NULL";
}

inline std::string MetricSql(pqxx::work& w, const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "NULL";
}

inline std::string MetricSql(pqxx::work& w, const std::optional<long long>& value)
{
    return value.has_value() ? std::to_string(*value) : "NULL";
}

inline std::string MetricSql(pqxx::work& w, const std::optional<std::string>& value)
{
    return value.has_value() ? w.quote(*value) : "NULL";
}

inline std::string MetricSqlBool(const std::optional<bool>& value)
{
    return value.has_value() ? (*value ? "TRUE" : "FALSE") : "NULL";
}

inline void UpsertAnalysisResult(pqxx::work& w,
                                 const ExperimentRow& experiment,
                                 const ParsedMetrics& metrics,
                                 const std::optional<double>& leaderScore)
{
    long long actual[3] = {};
    long long pred[3] = {};
    if (metrics.hasConfusion)
    {
        for (int a = 0; a < 3; ++a)
        {
            for (int p = 0; p < 3; ++p)
            {
                actual[a] += metrics.confusion[a][p];
                pred[p] += metrics.confusion[a][p];
            }
        }
    }

    if (!metrics.modelId.has_value())
    {
        w.exec_params(
            "DELETE FROM experiment_analysis_result "
            "WHERE experiment_id = $1 AND model_id IS NULL;",
            experiment.experimentId);
    }

    std::ostringstream sql;
    sql << "INSERT INTO experiment_analysis_result ("
        << "experiment_id, model_id, symbol, prediction_horizon, target_epochs, completed_epochs, "
        << "train_accuracy, validation_accuracy, infer_accuracy, "
        << "actual_down_count, actual_neutral_count, actual_up_count, "
        << "pred_down_count, pred_neutral_count, pred_up_count, "
        << "confusion_down_down, confusion_down_neutral, confusion_down_up, "
        << "confusion_neutral_down, confusion_neutral_neutral, confusion_neutral_up, "
        << "confusion_up_down, confusion_up_neutral, confusion_up_up, "
        << "accept_count, accept_rate, accept_accuracy, reject_count, loss_last, "
        << "best_metric_name, best_metric_value, leader_score, analysis_status, analysis_notes, "
        << "source_train_log_path, source_infer_log_path, updated_at"
        << ") VALUES ("
        << experiment.experimentId << ","
        << MetricSql(w, metrics.modelId) << ","
        << w.quote(experiment.symbol) << ","
        << experiment.predictionHorizon << ","
        << experiment.targetEpochs << ","
        << MetricSql(w, metrics.completedEpochs) << ","
        << MetricSql(w, metrics.trainAccuracy) << ","
        << MetricSql(w, metrics.validationAccuracy) << ","
        << MetricSql(w, metrics.inferAccuracy) << ","
        << (metrics.hasConfusion ? std::to_string(actual[0]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(actual[1]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(actual[2]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(pred[0]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(pred[1]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(pred[2]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[0][0]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[0][1]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[0][2]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[1][0]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[1][1]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[1][2]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[2][0]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[2][1]) : "NULL") << ","
        << (metrics.hasConfusion ? std::to_string(metrics.confusion[2][2]) : "NULL") << ","
        << (metrics.acceptModel.has_value() && *metrics.acceptModel ? "1" : "0") << ","
        << MetricSql(w, metrics.acceptRate) << ","
        << MetricSql(w, metrics.acceptAccuracy) << ","
        << (metrics.acceptModel.has_value() && !*metrics.acceptModel ? "1" : "0") << ","
        << MetricSql(w, metrics.lossLast) << ","
        << w.quote("infer_accuracy") << ","
        << MetricSql(w, metrics.inferAccuracy) << ","
        << MetricSql(w, leaderScore) << ","
        << w.quote("completed") << ","
        << MetricSql(w, metrics.rejectReason) << ","
        << MetricSql(w, experiment.trainLogPath) << ","
        << MetricSql(w, experiment.inferLogPath) << ","
        << "now()) "
        << "ON CONFLICT (experiment_id, model_id) DO UPDATE SET "
        << "symbol = EXCLUDED.symbol, "
        << "prediction_horizon = EXCLUDED.prediction_horizon, "
        << "target_epochs = EXCLUDED.target_epochs, "
        << "completed_epochs = EXCLUDED.completed_epochs, "
        << "train_accuracy = EXCLUDED.train_accuracy, "
        << "validation_accuracy = EXCLUDED.validation_accuracy, "
        << "infer_accuracy = EXCLUDED.infer_accuracy, "
        << "actual_down_count = EXCLUDED.actual_down_count, "
        << "actual_neutral_count = EXCLUDED.actual_neutral_count, "
        << "actual_up_count = EXCLUDED.actual_up_count, "
        << "pred_down_count = EXCLUDED.pred_down_count, "
        << "pred_neutral_count = EXCLUDED.pred_neutral_count, "
        << "pred_up_count = EXCLUDED.pred_up_count, "
        << "confusion_down_down = EXCLUDED.confusion_down_down, "
        << "confusion_down_neutral = EXCLUDED.confusion_down_neutral, "
        << "confusion_down_up = EXCLUDED.confusion_down_up, "
        << "confusion_neutral_down = EXCLUDED.confusion_neutral_down, "
        << "confusion_neutral_neutral = EXCLUDED.confusion_neutral_neutral, "
        << "confusion_neutral_up = EXCLUDED.confusion_neutral_up, "
        << "confusion_up_down = EXCLUDED.confusion_up_down, "
        << "confusion_up_neutral = EXCLUDED.confusion_up_neutral, "
        << "confusion_up_up = EXCLUDED.confusion_up_up, "
        << "accept_count = EXCLUDED.accept_count, "
        << "accept_rate = EXCLUDED.accept_rate, "
        << "accept_accuracy = EXCLUDED.accept_accuracy, "
        << "reject_count = EXCLUDED.reject_count, "
        << "loss_last = EXCLUDED.loss_last, "
        << "best_metric_name = EXCLUDED.best_metric_name, "
        << "best_metric_value = EXCLUDED.best_metric_value, "
        << "leader_score = EXCLUDED.leader_score, "
        << "analysis_status = EXCLUDED.analysis_status, "
        << "analysis_notes = EXCLUDED.analysis_notes, "
        << "source_train_log_path = EXCLUDED.source_train_log_path, "
        << "source_infer_log_path = EXCLUDED.source_infer_log_path, "
        << "updated_at = now();";
    w.exec(sql.str());
}

inline int AnalyzeExperimentById(long long experimentId, const SchedulerOptions& options)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
        return 1;

    pqxx::result rows = w.exec_params(
        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
        "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
        "train_start::text, train_end::text, infer_start::text, infer_end::text, "
        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path "
        "FROM experiment WHERE experiment_id = $1;",
        experimentId);
    if (rows.empty())
    {
        std::cerr << "EXPERIMENT_ANALYSIS_FAILED"
                  << ",experiment_id=" << experimentId
                  << ",reason=not_found"
                  << std::endl;
        return 1;
    }

    ExperimentRow experiment = RowToExperiment(rows[0]);
    std::cout << "EXPERIMENT_ANALYSIS_STARTED"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
              << std::endl;

    ParsedMetrics metrics = ParseMetricsFromLogs(experiment);
    ApplyPersistedSymbolToAnalysisExperiment(w, experiment, metrics);
    const bool usedStructuredInferenceMetrics = ApplyStructuredInferenceMetrics(w, experiment, metrics);
    const std::optional<double> leaderScore = ComputeLeaderScore(metrics);
    UpsertAnalysisResult(w, experiment, metrics, leaderScore);
    w.exec_params(
        "UPDATE experiment "
        "SET status = 'completed', phase = 'done', completed_at = COALESCE(completed_at, now()), updated_at = now() "
        "WHERE experiment_id = $1;",
        experiment.experimentId);
    w.commit();

    std::ostringstream analysisLog;
    analysisLog << "EXPERIMENT_ANALYSIS_METRIC"
                << ",experiment_id=" << experiment.experimentId
                << ",metric=infer_accuracy"
                << ",value=" << (metrics.inferAccuracy.has_value() ? FormatDouble(*metrics.inferAccuracy) : "NULL")
                << "\n";
    analysisLog << "EXPERIMENT_LEADER_SCORE"
                << ",experiment_id=" << experiment.experimentId
                << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
                << ",leader_score=" << (leaderScore.has_value() ? FormatDouble(*leaderScore) : "NULL")
                << "\n";
    if (experiment.analysisLogPath.has_value())
        WriteTextFile(*experiment.analysisLogPath, analysisLog.str());

    std::cout << "EXPERIMENT_ANALYSIS_SOURCE"
              << ",experiment_id=" << experiment.experimentId
              << ",source=" << (usedStructuredInferenceMetrics ? "inference_eval_result" : "logs")
              << ",train_log=" << (experiment.trainLogPath.has_value() ? *experiment.trainLogPath : "none")
              << ",infer_log=" << (experiment.inferLogPath.has_value() ? *experiment.inferLogPath : "none")
              << std::endl;
    std::cout << "EXPERIMENT_ANALYSIS_METRIC"
              << ",experiment_id=" << experiment.experimentId
              << ",metric=infer_accuracy"
              << ",value=" << (metrics.inferAccuracy.has_value() ? FormatDouble(*metrics.inferAccuracy) : "NULL")
              << std::endl;
    std::cout << "EXPERIMENT_LEADER_SCORE"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
              << ",leader_score=" << (leaderScore.has_value() ? FormatDouble(*leaderScore) : "NULL")
              << std::endl;
    std::cout << "EXPERIMENT_LEADER_UPDATED"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
              << std::endl;
    std::cout << "EXPERIMENT_ANALYSIS_COMPLETED"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (metrics.modelId.has_value() ? std::to_string(*metrics.modelId) : "none")
              << std::endl;
    return 0;
}

inline int AnalyzeCompletedExperiments()
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!RequireSchedulerTables(w))
        return 1;

    pqxx::result rows = w.exec(
        "SELECT experiment_id FROM experiment "
        "WHERE phase = 'analyze' AND status = 'pending' "
        "ORDER BY updated_at ASC, experiment_id ASC;");
    std::vector<long long> ids;
    ids.reserve(rows.size());
    for (const auto& row : rows)
        ids.push_back(row[0].as<long long>());
    w.commit();

    int rc = 0;
    SchedulerOptions unused;
    for (long long id : ids)
        rc |= AnalyzeExperimentById(id, unused);
    return rc;
}

inline void MarkExperimentRunning(pqxx::work& w,
                                  const ExperimentRow& experiment,
                                  const std::string& phase,
                                  const std::string& logPath)
{
    const char* logColumn = phase == "train" ? "train_log_path" :
                            phase == "infer" ? "infer_log_path" :
                            "analysis_log_path";
    w.exec(
        "UPDATE experiment SET status = 'running', started_at = COALESCE(started_at, now()), " +
        std::string{logColumn} + " = " + w.quote(logPath) + ", updated_at = now() "
        "WHERE experiment_id = " + std::to_string(experiment.experimentId) + ";");
}

inline bool CompleteTrainPhase(pqxx::work& w,
                               const ExperimentRow& experiment,
                               int exitCode,
                               const std::string& logPath,
                               const std::string& commandDisplay)
{
    const std::string logText = ReadFileIfExists(logPath);
    const std::optional<long long> lastModelId = ExtractLastModelId(logText);
    const bool success = exitCode == 0 && lastModelId.has_value();
    if (success)
    {
        const bool shouldInfer = experiment.inferStart.has_value() && experiment.inferEnd.has_value();
        w.exec_params(
            "UPDATE experiment "
            "SET status = 'pending', phase = $1, last_model_id = $2, exit_code = $3, error_message = NULL, updated_at = now() "
            "WHERE experiment_id = $4;",
            shouldInfer ? "infer" : "analyze",
            *lastModelId,
            exitCode,
            experiment.experimentId);
        std::cout << "EXPERIMENT_LAST_MODEL_ID"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << *lastModelId
                  << std::endl;
        std::cout << "EXPERIMENT_PHASE_CHANGED"
                  << ",experiment_id=" << experiment.experimentId
                  << ",phase=" << (shouldInfer ? "infer" : "analyze")
                  << ",status=pending"
                  << std::endl;
        return true;
    }
    else
    {
        const std::string failureReason = lastModelId.has_value() ? "train_failed" : "train_failed_no_model_id";
        if (exitCode == 0 && !lastModelId.has_value())
            PrintModelIdDetectionWarning(experiment.experimentId, logPath, logText);
        const std::string errorMessage =
            failureReason +
            ";exit_code=" + std::to_string(exitCode) +
            ";command=" + commandDisplay +
            ";model_id_patterns=" + ModelIdSearchPatternDescription();
        w.exec_params(
            "UPDATE experiment "
            "SET status = 'failed', exit_code = $1, error_message = $2, completed_at = now(), updated_at = now() "
            "WHERE experiment_id = $3;",
            exitCode,
            errorMessage,
            experiment.experimentId);
        return false;
    }
}

inline void CompleteInferPhase(pqxx::work& w,
                               const ExperimentRow& experiment,
                               int exitCode,
                               const std::string& commandDisplay)
{
    if (exitCode == 0)
    {
        w.exec_params(
            "UPDATE experiment "
            "SET status = 'pending', phase = 'analyze', exit_code = $1, error_message = NULL, updated_at = now() "
            "WHERE experiment_id = $2;",
            exitCode,
            experiment.experimentId);
        std::cout << "EXPERIMENT_PHASE_CHANGED"
                  << ",experiment_id=" << experiment.experimentId
                  << ",phase=analyze,status=pending"
                  << std::endl;
    }
    else
    {
        const std::string errorMessage =
            "infer_failed;exit_code=" + std::to_string(exitCode) +
            ";command=" + commandDisplay;
        w.exec_params(
            "UPDATE experiment "
            "SET status = 'failed', exit_code = $1, error_message = $2, completed_at = now(), updated_at = now() "
            "WHERE experiment_id = $3;",
            exitCode,
            errorMessage,
            experiment.experimentId);
    }
}

inline int RunTrainJobs(const SchedulerOptions& options)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!RequireSchedulerTables(w))
        return 1;

    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "train", options.maxTrainProcs);
    if (options.dryRun)
    {
        for (const auto& job : jobs)
        {
            const std::vector<std::string> command = BuildTrainCommand(options, job);
            std::cout << "EXPERIMENT_CHILD_COMMAND"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=train"
                      << ",dry_run=1"
                      << ",argv=" << CommandForDisplay(command)
                      << std::endl;
        }
        w.commit();
        return 0;
    }

    EnsureLogDir(options.schedulerLogDir);
    w.commit();

    std::vector<RunningExperimentChild> running;
    running.reserve(jobs.size());
    for (const auto& job : jobs)
    {
        const std::string logPath = LogPathFor(options, job, "train");
        {
            pqxx::connection c2{LstmDbConnectionString()};
            pqxx::work wt{c2};
            SetTransactionReadWrite(wt);
            MarkExperimentRunning(wt, job, "train", logPath);
            wt.commit();
        }

        const std::vector<std::string> command = BuildTrainCommand(options, job);
        const std::string commandDisplay = CommandForDisplay(command);
        if (job.lastModelId.has_value() && !job.resumeModelId.has_value())
        {
            std::cout << "SCHEDULER_RESUME_FROM_LAST_MODEL"
                      << ",experiment_id=" << job.experimentId
                      << ",model_id=" << *job.lastModelId
                      << std::endl;
        }
        std::cout << "EXPERIMENT_STARTED"
                  << ",experiment_id=" << job.experimentId
                  << ",phase=train"
                  << std::endl;
        std::cout << "EXPERIMENT_CHILD_COMMAND"
                  << ",experiment_id=" << job.experimentId
                  << ",phase=train"
                  << ",argv=" << commandDisplay
                  << std::endl;
        PrintSchedulerExec(command);
        const pid_t pid = LaunchChildProcess(command, logPath);
        running.push_back(RunningExperimentChild{job, pid, logPath});
    }

    int rc = 0;
    for (const auto& child : running)
    {
        const int exitCode = WaitForChildProcess(child.pid);
        bool trainSuccess = false;
        {
            pqxx::connection c3{LstmDbConnectionString()};
            pqxx::work wt{c3};
            SetTransactionReadWrite(wt);
            const std::vector<std::string> command = BuildTrainCommand(options, child.experiment);
            trainSuccess = CompleteTrainPhase(wt, child.experiment, exitCode, child.logPath, CommandForDisplay(command));
            wt.commit();
        }
        if (trainSuccess)
        {
            std::cout << "EXPERIMENT_COMPLETED"
                      << ",experiment_id=" << child.experiment.experimentId
                      << ",phase=train"
                      << ",exit_code=0"
                      << std::endl;
        }
        else
        {
            rc = 1;
            std::cout << "EXPERIMENT_FAILED"
                      << ",experiment_id=" << child.experiment.experimentId
                      << ",phase=train"
                      << ",exit_code=" << exitCode
                      << std::endl;
        }
    }
    return rc;
}

inline int RunInferJobs(const SchedulerOptions& options)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!RequireSchedulerTables(w))
        return 1;

    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "infer", options.maxInferProcs);
    if (options.dryRun)
    {
        for (const auto& job : jobs)
        {
            const std::vector<std::string> command = BuildInferCommand(options, job);
            std::cout << "EXPERIMENT_CHILD_COMMAND"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=infer"
                      << ",dry_run=1"
                      << ",argv=" << CommandForDisplay(command)
                      << std::endl;
        }
        w.commit();
        return 0;
    }

    EnsureLogDir(options.schedulerLogDir);
    w.commit();

    std::vector<RunningExperimentChild> running;
    running.reserve(jobs.size());
    for (const auto& job : jobs)
    {
        const std::string logPath = LogPathFor(options, job, "infer");
        {
            pqxx::connection c2{LstmDbConnectionString()};
            pqxx::work wi{c2};
            SetTransactionReadWrite(wi);
            MarkExperimentRunning(wi, job, "infer", logPath);
            wi.commit();
        }

        const std::vector<std::string> command = BuildInferCommand(options, job);
        const std::string commandDisplay = CommandForDisplay(command);
        std::cout << "EXPERIMENT_STARTED"
                  << ",experiment_id=" << job.experimentId
                  << ",phase=infer"
                  << std::endl;
        std::cout << "EXPERIMENT_CHILD_COMMAND"
                  << ",experiment_id=" << job.experimentId
                  << ",phase=infer"
                  << ",argv=" << commandDisplay
                  << std::endl;
        PrintSchedulerExec(command);
        const pid_t pid = LaunchChildProcess(command, logPath);
        running.push_back(RunningExperimentChild{job, pid, logPath});
    }

    int rc = 0;
    for (const auto& child : running)
    {
        const int exitCode = WaitForChildProcess(child.pid);
        {
            pqxx::connection c3{LstmDbConnectionString()};
            pqxx::work wi{c3};
            SetTransactionReadWrite(wi);
            const std::vector<std::string> command = BuildInferCommand(options, child.experiment);
            CompleteInferPhase(wi, child.experiment, exitCode, CommandForDisplay(command));
            wi.commit();
        }
        if (exitCode == 0)
        {
            std::cout << "EXPERIMENT_COMPLETED"
                      << ",experiment_id=" << child.experiment.experimentId
                      << ",phase=infer"
                      << ",exit_code=0"
                      << std::endl;
        }
        else
        {
            rc = 1;
            std::cout << "EXPERIMENT_FAILED"
                      << ",experiment_id=" << child.experiment.experimentId
                      << ",phase=infer"
                      << ",exit_code=" << exitCode
                      << std::endl;
        }
    }
    return rc;
}

inline int RunAnalyzeJobs(const SchedulerOptions& options)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!RequireSchedulerTables(w))
        return 1;

    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "analyze", options.maxAnalyzeProcs);
    if (options.dryRun)
    {
        for (const auto& job : jobs)
        {
            const std::vector<std::string> command = BuildAnalyzeCommand(options, job);
            std::cout << "EXPERIMENT_ANALYSIS_STARTED"
                      << ",experiment_id=" << job.experimentId
                      << ",dry_run=1"
                      << std::endl;
            std::cout << "EXPERIMENT_CHILD_COMMAND"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=analyze"
                      << ",dry_run=1"
                      << ",argv=" << CommandForDisplay(command)
                      << std::endl;
        }
        w.commit();
        return 0;
    }
    w.commit();

    int rc = 0;
    for (const auto& job : jobs)
    {
        {
            pqxx::connection c2{LstmDbConnectionString()};
            pqxx::work wa{c2};
            SetTransactionReadWrite(wa);
            const std::string logPath = LogPathFor(options, job, "analysis");
            MarkExperimentRunning(wa, job, "analyze", logPath);
            wa.commit();
        }
        rc |= AnalyzeExperimentById(job.experimentId, options);
    }
    return rc;
}

inline int RunSchedulerOnce(const SchedulerOptions& options)
{
    int rc = 0;
    rc |= RunTrainJobs(options);
    rc |= RunInferJobs(options);
    rc |= RunAnalyzeJobs(options);
    return rc;
}

inline int RunScheduler(const SchedulerOptions& options)
{
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        SetTransactionReadWrite(w);
        if (!RequireSchedulerTables(w))
            return 1;
        if (!options.dryRun)
            RecoverOrphanedRunningExperiments(w);
        w.commit();
    }

    std::cout << "SCHEDULER_START"
              << ",dry_run=" << (options.dryRun ? "1" : "0")
              << ",max_train_procs=" << options.maxTrainProcs
              << ",max_infer_procs=" << options.maxInferProcs
              << ",max_analyze_procs=" << options.maxAnalyzeProcs
              << std::endl;
    if (options.dryRun)
        std::cout << "SCHEDULER_DRY_RUN=1" << std::endl;

    int rc = 0;
    do
    {
        rc |= RunSchedulerOnce(options);
        if (options.schedulerOnce)
            break;
        ::sleep(static_cast<unsigned int>(options.schedulerPollSeconds));
    } while (true);

    std::cout << "SCHEDULER_STOP"
              << ",exit_code=" << rc
              << std::endl;
    return rc;
}

inline int PrintLeaderboard(const SchedulerOptions& options)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!RequireSchedulerTables(w))
        return 1;

    std::ostringstream sql;
    sql << "SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
        << "a.target_epochs, a.infer_accuracy, a.accept_accuracy, a.accept_rate, a.leader_score "
        << "FROM experiment_analysis_result a "
        << "JOIN experiment e ON e.experiment_id = a.experiment_id "
        << "WHERE 1=1 ";
    if (options.leaderboardSymbol.has_value())
        sql << "AND a.symbol = " << w.quote(*options.leaderboardSymbol) << " ";
    if (options.leaderboardHorizon.has_value())
        sql << "AND a.prediction_horizon = " << *options.leaderboardHorizon << " ";
    sql << "ORDER BY a.leader_score DESC NULLS LAST LIMIT " << options.leaderboardLimit << ";";

    pqxx::result rows = w.exec(sql.str());
    std::cout << "EXPERIMENT_LEADERBOARD_BEGIN"
              << ",rows=" << rows.size()
              << std::endl;
    std::cout << "experiment_id,model_id,symbol,prediction_horizon,target_epochs,infer_accuracy,accept_accuracy,accept_rate,leader_score"
              << std::endl;
    for (const auto& row : rows)
    {
        std::cout << row[0].c_str() << ","
                  << (row[1].is_null() ? "" : row[1].c_str()) << ","
                  << (row[2].is_null() ? "" : row[2].c_str()) << ","
                  << (row[3].is_null() ? "" : row[3].c_str()) << ","
                  << (row[4].is_null() ? "" : row[4].c_str()) << ","
                  << (row[5].is_null() ? "" : row[5].c_str()) << ","
                  << (row[6].is_null() ? "" : row[6].c_str()) << ","
                  << (row[7].is_null() ? "" : row[7].c_str()) << ","
                  << (row[8].is_null() ? "" : row[8].c_str()) << std::endl;
    }
    std::cout << "EXPERIMENT_LEADERBOARD_DONE"
              << ",rows=" << rows.size()
              << std::endl;
    return 0;
}

inline int RunExperimentSchedulerCli(int argc, const char* argv[])
{
    SchedulerOptions options;
    try
    {
        options = ParseSchedulerArgs(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << "\n"
                  << "Usage: " << (argc > 0 ? argv[0] : "LSTM_Release")
                  << " --enqueue-experiment --symbol=SYMBOL --prediction-horizon=N --c-next-threshold=VALUE "
                  << "--core-lr-mult=VALUE --head-lr-mult=VALUE --target-epochs=N --checkpoint-interval=N "
                  << "--train-start=YYYY-MM-DD --train-end=YYYY-MM-DD [--infer-start=YYYY-MM-DD --infer-end=YYYY-MM-DD] "
                  << "[--resume-model-id=MODEL_ID] [--allow-duplicate-experiment]\n"
                  << "Usage: " << (argc > 0 ? argv[0] : "LSTM_Release")
                  << " --schedule-experiments [--max-train-procs=N] [--max-infer-procs=N] "
                  << "[--max-analyze-procs=N] [--scheduler-poll-seconds=N] [--scheduler-once] "
                  << "[--scheduler-log-dir=PATH] [--dry-run]\n"
                  << "Usage: " << (argc > 0 ? argv[0] : "LSTM_Release")
                  << " --analyze-experiment=EXPERIMENT_ID | --analyze-completed-experiments | "
                  << "--print-experiment-leaderboard [--leaderboard-symbol=SYMBOL] "
                  << "[--leaderboard-horizon=N] [--leaderboard-limit=N]\n";
        return 1;
    }

    try
    {
        if (options.enqueueExperiment)
            return EnqueueExperiment(options);
        if (options.scheduleExperiments)
            return RunScheduler(options);
        if (options.analyzeExperimentId.has_value())
            return AnalyzeExperimentById(*options.analyzeExperimentId, options);
        if (options.analyzeCompletedExperiments)
            return AnalyzeCompletedExperiments();
        if (options.printLeaderboard)
            return PrintLeaderboard(options);
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXPERIMENT_FAILED"
                  << ",error=" << e.what()
                  << std::endl;
        return 1;
    }

    return 1;
}

} // namespace EA::ExperimentScheduler
