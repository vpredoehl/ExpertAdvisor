#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <signal.h>
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
    bool recoverOrphansOnly = false;
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

struct RunningExperimentState
{
    ExperimentRow experiment;
    std::string phase;
};

struct QueueSnapshot
{
    int pendingTrain = 0;
    int pendingInfer = 0;
    int pendingAnalyze = 0;
    int runningTrain = 0;
    int runningInfer = 0;
    int runningAnalyze = 0;
};

struct PhaseSchedulingStats
{
    std::string phase;
    int examined = 0;
    int skipped = 0;
    int launched = 0;
    int freeSlots = 0;
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
        else if (arg == "--recover-orphans-only")
            options.recoverOrphansOnly = true;
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
    if (options.recoverOrphansOnly && !options.scheduleExperiments)
        throw std::invalid_argument("--recover-orphans-only requires --schedule-experiments");

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

inline bool ModelExists(pqxx::work& w, long long modelId)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM model WHERE model_id = $1 LIMIT 1;",
        modelId);
    return !r.empty();
}

inline bool ExistingFilePath(const std::optional<std::string>& path)
{
    return path.has_value() && std::filesystem::exists(*path);
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
                                                        const std::string& phase)
{
    pqxx::result rows = w.exec_params(
        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
        "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
        "train_start::text, train_end::text, infer_start::text, infer_end::text, "
        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path "
        "FROM experiment "
        "WHERE status = 'pending' AND phase = $1 "
        "ORDER BY updated_at ASC, experiment_id ASC;",
        phase);

    std::vector<ExperimentRow> experiments;
    experiments.reserve(rows.size());
    for (const auto& row : rows)
        experiments.push_back(RowToExperiment(row));
    return experiments;
}

inline std::vector<RunningExperimentState> LoadRunningExperiments(pqxx::work& w)
{
    pqxx::result rows = w.exec(
        "SELECT experiment_id, symbol, prediction_horizon, c_next_threshold, "
        "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
        "train_start::text, train_end::text, infer_start::text, infer_end::text, "
        "last_model_id, resume_model_id, train_log_path, infer_log_path, analysis_log_path, phase "
        "FROM experiment "
        "WHERE status = 'running' "
        "ORDER BY updated_at ASC, experiment_id ASC;");

    std::vector<RunningExperimentState> experiments;
    experiments.reserve(rows.size());
    for (const auto& row : rows)
        experiments.push_back(RunningExperimentState{RowToExperiment(row), row[17].as<std::string>()});
    return experiments;
}

inline int RecoverOrphanedRunningExperiments(pqxx::work& w);

inline QueueSnapshot LoadQueueSnapshot(pqxx::work& w)
{
    QueueSnapshot snapshot;
    pqxx::result rows = w.exec(
        "SELECT phase, status, count(*) "
        "FROM experiment "
        "WHERE status IN ('pending', 'running') "
        "AND phase IN ('train', 'infer', 'analyze') "
        "GROUP BY phase, status;");
    for (const auto& row : rows)
    {
        const std::string phase = row[0].as<std::string>();
        const std::string status = row[1].as<std::string>();
        const int count = row[2].as<int>();
        if (phase == "train" && status == "pending")
            snapshot.pendingTrain = count;
        else if (phase == "infer" && status == "pending")
            snapshot.pendingInfer = count;
        else if (phase == "analyze" && status == "pending")
            snapshot.pendingAnalyze = count;
        else if (phase == "train" && status == "running")
            snapshot.runningTrain = count;
        else if (phase == "infer" && status == "running")
            snapshot.runningInfer = count;
        else if (phase == "analyze" && status == "running")
            snapshot.runningAnalyze = count;
    }
    return snapshot;
}

inline void PrintQueueSnapshot(const QueueSnapshot& snapshot)
{
    std::cout << "SCHEDULER_QUEUE"
              << ",pending_train=" << snapshot.pendingTrain
              << ",pending_infer=" << snapshot.pendingInfer
              << ",pending_analyze=" << snapshot.pendingAnalyze
              << ",running_train=" << snapshot.runningTrain
              << ",running_infer=" << snapshot.runningInfer
              << ",running_analyze=" << snapshot.runningAnalyze
              << std::endl;
}

inline int RunningCountForPhase(const QueueSnapshot& snapshot, const std::string& phase)
{
    if (phase == "train")
        return snapshot.runningTrain;
    if (phase == "infer")
        return snapshot.runningInfer;
    if (phase == "analyze")
        return snapshot.runningAnalyze;
    return 0;
}

inline void LogSkip(const std::string& phase,
                    long long experimentId,
                    const std::string& reason)
{
    std::cout << "SCHEDULER_SKIP_" << (phase == "train" ? "TRAIN" : phase == "infer" ? "INFER" : "ANALYZE")
              << ",experiment_id=" << experimentId
              << ",reason=" << reason
              << std::endl;
}

inline void PrintPhaseSchedulingStats(const PhaseSchedulingStats& stats)
{
    std::cout << "SCHEDULER_QUEUE_PHASE"
              << ",phase=" << stats.phase
              << ",examined=" << stats.examined
              << ",skipped=" << stats.skipped
              << ",launched=" << stats.launched
              << ",free_slots=" << stats.freeSlots
              << std::endl;
}

inline int FailInvalidSchedulerPhases(pqxx::work& w)
{
    pqxx::result rows = w.exec(
        "SELECT experiment_id, phase FROM experiment "
        "WHERE status IN ('pending', 'running') "
        "AND phase NOT IN ('train', 'infer', 'analyze', 'done') "
        "ORDER BY updated_at ASC, experiment_id ASC;");

    for (const auto& row : rows)
    {
        const long long experimentId = row[0].as<long long>();
        const std::string phase = row[1].as<std::string>();
        w.exec_params(
            "UPDATE experiment "
            "SET status = 'failed', "
            "exit_code = -1, "
            "error_message = $1, "
            "completed_at = now(), "
            "updated_at = now() "
            "WHERE experiment_id = $2;",
            "unknown_scheduler_phase:" + phase,
            experimentId);
        std::cout << "EXPERIMENT_FAILED"
                  << ",experiment_id=" << experimentId
                  << ",phase=" << phase
                  << ",reason=unknown_scheduler_phase"
                  << std::endl;
    }

    return rows.empty() ? 0 : 1;
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

inline bool RunningTrainProcessExistsForExperiment(const ExperimentRow& experiment)
{
    FILE* pipe = ::popen("ps -axo command", "r");
    if (!pipe)
        return false;

    const std::string baseModelName = BaseModelName(experiment);
    const std::string experimentNeedle = "experiment" + std::to_string(experiment.experimentId);
    char buffer[4096];
    bool found = false;
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        const std::string command{buffer};
        if (command.find("LSTM_Release") == std::string::npos ||
            command.find("--train") == std::string::npos)
        {
            continue;
        }
        if (command.find(baseModelName) != std::string::npos ||
            command.find(experimentNeedle) != std::string::npos)
        {
            found = true;
            break;
        }
    }
    ::pclose(pipe);
    return found;
}

inline bool RunningProcessExistsForExperiment(const ExperimentRow& experiment,
                                             const std::string& phase)
{
    if (phase == "train")
        return RunningTrainProcessExistsForExperiment(experiment);

    FILE* pipe = ::popen("ps -axo command", "r");
    if (!pipe)
        return false;

    const std::string modelNeedle =
        experiment.lastModelId.has_value() ? "--model=" + std::to_string(*experiment.lastModelId) : "";
    const std::string analyzeNeedle = "--analyze-experiment=" + std::to_string(experiment.experimentId);
    char buffer[4096];
    bool found = false;
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        const std::string command{buffer};
        if (command.find("LSTM_Release") == std::string::npos)
            continue;

        if (phase == "infer")
        {
            const bool dateRangeMatches =
                !experiment.inferStart.has_value() ||
                !experiment.inferEnd.has_value() ||
                (command.find(experiment.inferStart->substr(0, 10)) != std::string::npos &&
                 command.find(experiment.inferEnd->substr(0, 10)) != std::string::npos);
            if (command.find("--infer") != std::string::npos &&
                command.find("--infer-all") == std::string::npos &&
                !modelNeedle.empty() &&
                command.find(modelNeedle) != std::string::npos &&
                dateRangeMatches)
            {
                found = true;
                break;
            }
        }
        else if (phase == "analyze")
        {
            if (command.find(analyzeNeedle) != std::string::npos)
            {
                found = true;
                break;
            }
        }
    }
    ::pclose(pipe);
    return found;
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

    int pidPipe[2] = {-1, -1};
    if (::pipe(pidPipe) != 0)
    {
        ::close(fd);
        throw std::runtime_error("failed to create child pid pipe");
    }

    pid_t pid = ::fork();
    if (pid < 0)
    {
        ::close(pidPipe[0]);
        ::close(pidPipe[1]);
        ::close(fd);
        throw std::runtime_error("fork failed");
    }

    if (pid == 0)
    {
        ::close(pidPipe[0]);
        if (::setsid() < 0)
            _exit(127);

        const pid_t grandchildPid = ::fork();
        if (grandchildPid < 0)
            _exit(127);
        if (grandchildPid > 0)
        {
            const std::string pidText = std::to_string(static_cast<long long>(grandchildPid));
            (void)::write(pidPipe[1], pidText.c_str(), pidText.size());
            ::close(pidPipe[1]);
            ::close(fd);
            _exit(0);
        }

        ::close(pidPipe[1]);
        ::signal(SIGHUP, SIG_IGN);
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

    ::close(pidPipe[1]);
    ::close(fd);

    char buffer[64] = {};
    const ssize_t bytesRead = ::read(pidPipe[0], buffer, sizeof(buffer) - 1);
    ::close(pidPipe[0]);

    int status = 0;
    (void)::waitpid(pid, &status, 0);
    if (bytesRead <= 0)
        throw std::runtime_error("failed to obtain detached child pid");
    return static_cast<pid_t>(std::stoll(std::string{buffer, static_cast<size_t>(bytesRead)}));
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

inline std::string ReadFileIfExists(const std::string& path)
{
    std::ifstream in{path};
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

inline std::optional<long long> ExtractLastModelIdByRegex(const std::string& text,
                                                          const std::regex& idRegex,
                                                          size_t captureIndex)
{
    std::optional<long long> last;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), idRegex);
         it != std::sregex_iterator();
         ++it)
    {
        last = std::stoll((*it)[captureIndex].str());
    }
    return last;
}

inline std::optional<long long> ExtractCompletedTrainModelId(const std::string& text)
{
    return ExtractLastModelIdByRegex(
        text,
        std::regex{"(Saved model with model_id=|RESUME_SAVED_NEW_MODEL_ID=)([0-9]+)"},
        2);
}

inline std::optional<long long> ExtractCheckpointModelId(const std::string& text)
{
    return ExtractLastModelIdByRegex(
        text,
        std::regex{"CHECKPOINT_SAVE_DONE[^\\n]*model_id=([0-9]+)"},
        1);
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

inline bool IsValidInferenceLogText(const std::string& text)
{
    return text.find("Overall 3-class accuracy") != std::string::npos &&
           text.find("Overall 3-class confusion matrix") != std::string::npos &&
           text.find("MODEL_ACCEPTANCE") != std::string::npos;
}

inline bool HasValidInferenceLogPath(const ExperimentRow& experiment)
{
    if (!experiment.inferLogPath.has_value())
        return false;
    return IsValidInferenceLogText(ReadFileIfExists(experiment.inferLogPath));
}

inline std::optional<std::string> DiscoverValidInferenceLog(const ExperimentRow& experiment)
{
    if (!experiment.lastModelId.has_value())
        return std::nullopt;

    const std::string modelNeedle = std::to_string(*experiment.lastModelId);
    const std::vector<std::filesystem::path> roots = {
        std::filesystem::current_path(),
        std::filesystem::current_path() / "experiment_logs"};

    for (const auto& root : roots)
    {
        std::error_code ec;
        if (!std::filesystem::exists(root, ec) || !std::filesystem::is_directory(root, ec))
            continue;

        for (const auto& entry : std::filesystem::directory_iterator(root, ec))
        {
            if (ec)
                break;
            if (!entry.is_regular_file(ec))
                continue;

            const std::string filename = entry.path().filename().string();
            if (filename.find(modelNeedle) == std::string::npos ||
                filename.find("infer") == std::string::npos)
            {
                continue;
            }

            const std::string path = entry.path().string();
            if (IsValidInferenceLogText(ReadFileIfExists(path)))
                return path;
        }
    }
    return std::nullopt;
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

inline bool HasCompletedInferenceResult(pqxx::work& w,
                                        const ExperimentRow& experiment)
{
    if (!TableExists(w, "inference_eval_result") ||
        !experiment.lastModelId.has_value() ||
        !experiment.inferStart.has_value() ||
        !experiment.inferEnd.has_value())
    {
        return false;
    }

    pqxx::result rows = w.exec_params(
        "SELECT 1 "
        "FROM inference_eval_result "
        "WHERE model_id = $1 "
        "AND symbol = $2 "
        "AND prediction_horizon = $3 "
        "AND threshold_logret = $4 "
        "AND from_date = $5 "
        "AND to_date = $6 "
        "AND status = 'completed' "
        "LIMIT 1;",
        *experiment.lastModelId,
        experiment.symbol,
        experiment.predictionHorizon,
        experiment.cNextThreshold,
        experiment.inferStart->substr(0, 10),
        experiment.inferEnd->substr(0, 10));
    return !rows.empty();
}

inline bool HasCompletedAnalysisResult(pqxx::work& w,
                                       const ExperimentRow& experiment)
{
    if (!experiment.lastModelId.has_value())
        return false;

    pqxx::result rows = w.exec_params(
        "SELECT 1 "
        "FROM experiment_analysis_result "
        "WHERE experiment_id = $1 "
        "AND model_id = $2 "
        "AND analysis_status = 'completed' "
        "LIMIT 1;",
        experiment.experimentId,
        *experiment.lastModelId);
    return !rows.empty();
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

inline void MarkAnalyzeFailed(pqxx::work& w,
                              long long experimentId,
                              const std::string& errorMessage)
{
    w.exec_params(
        "UPDATE experiment "
        "SET status = 'failed', "
        "exit_code = -1, "
        "error_message = $1, "
        "completed_at = now(), "
        "updated_at = now() "
        "WHERE experiment_id = $2;",
        errorMessage,
        experimentId);
}

inline void MarkAnalyzeFailedById(long long experimentId,
                                  const std::optional<long long>& modelId,
                                  const std::string& errorMessage)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadWrite(w);
    MarkAnalyzeFailed(w, experimentId, errorMessage);
    w.commit();
    std::cerr << "SCHEDULER_ANALYZE_FAILED"
              << ",experiment_id=" << experimentId
              << ",model_id=" << (modelId.has_value() ? std::to_string(*modelId) : "none")
              << ",error=" << errorMessage
              << std::endl;
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
    if (!experiment.lastModelId.has_value())
    {
        MarkAnalyzeFailed(w, experiment.experimentId, "analyze_missing_last_model_id");
        w.commit();
        std::cerr << "SCHEDULER_ANALYZE_FAILED"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=none"
                  << ",error=analyze_missing_last_model_id"
                  << std::endl;
        return 1;
    }

    const long long modelId = *experiment.lastModelId;
    std::cout << "SCHEDULER_ANALYZE_STARTED"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << modelId
              << std::endl;
    std::cout << "EXPERIMENT_ANALYSIS_STARTED"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << modelId
              << std::endl;

    ParsedMetrics metrics = ParseMetricsFromLogs(experiment);
    metrics.modelId = modelId;
    ApplyPersistedSymbolToAnalysisExperiment(w, experiment, metrics);
    const bool usedStructuredInferenceMetrics = ApplyStructuredInferenceMetrics(w, experiment, metrics);
    if (usedStructuredInferenceMetrics)
    {
        std::cout << "SCHEDULER_ANALYZE_USING_EXISTING_INFERENCE"
                  << ",model_id=" << modelId
                  << std::endl;
    }
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

    std::cout << "SCHEDULER_PHASE_TRANSITION"
              << ",experiment_id=" << experiment.experimentId
              << ",from_phase=analyze"
              << ",to_phase=done"
              << std::endl;
    std::cout << "SCHEDULER_PIPELINE_DONE"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << modelId
              << std::endl;
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
    std::cout << "SCHEDULER_ANALYZE_COMPLETED"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << modelId
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

inline void LogPhaseTransition(long long experimentId,
                               const std::string& fromPhase,
                               const std::string& toPhase)
{
    std::cout << "SCHEDULER_PHASE_TRANSITION"
              << ",experiment_id=" << experimentId
              << ",from_phase=" << fromPhase
              << ",to_phase=" << toPhase
              << std::endl;
}

inline void MarkExperimentPendingPhase(pqxx::work& w,
                                       const ExperimentRow& experiment,
                                       const std::string& fromPhase,
                                       const std::string& toPhase,
                                       const std::optional<int>& exitCode = std::nullopt)
{
    std::ostringstream sql;
    sql << "UPDATE experiment "
        << "SET status = 'pending', phase = " << w.quote(toPhase)
        << ", exit_code = " << (exitCode.has_value() ? std::to_string(*exitCode) : "NULL")
        << ", error_message = NULL, updated_at = now() "
        << "WHERE experiment_id = " << experiment.experimentId << ";";
    w.exec(sql.str());
    LogPhaseTransition(experiment.experimentId, fromPhase, toPhase);
    std::cout << "EXPERIMENT_PHASE_CHANGED"
              << ",experiment_id=" << experiment.experimentId
              << ",phase=" << toPhase
              << ",status=pending"
              << std::endl;
}

inline void MarkExperimentDone(pqxx::work& w,
                               const ExperimentRow& experiment,
                               const std::string& fromPhase)
{
    w.exec_params(
        "UPDATE experiment "
        "SET status = 'completed', phase = 'done', completed_at = COALESCE(completed_at, now()), updated_at = now() "
        "WHERE experiment_id = $1;",
        experiment.experimentId);
    LogPhaseTransition(experiment.experimentId, fromPhase, "done");
    std::cout << "SCHEDULER_PIPELINE_DONE"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
              << std::endl;
}

inline void MarkExperimentFailed(pqxx::work& w,
                                 const ExperimentRow& experiment,
                                 const std::string& errorMessage,
                                 int exitCode = -1)
{
    w.exec_params(
        "UPDATE experiment "
        "SET status = 'failed', exit_code = $1, error_message = $2, completed_at = now(), updated_at = now() "
        "WHERE experiment_id = $3;",
        exitCode,
        errorMessage,
        experiment.experimentId);
}

inline void UpdateInferLogPath(pqxx::work& w,
                               const ExperimentRow& experiment,
                               const std::string& inferLogPath)
{
    w.exec_params(
        "UPDATE experiment "
        "SET infer_log_path = $1, updated_at = now() "
        "WHERE experiment_id = $2;",
        inferLogPath,
        experiment.experimentId);
}

inline void TransitionRecoveredInferenceToAnalyze(pqxx::work& w,
                                                  const ExperimentRow& experiment,
                                                  const std::string& sourceMarker,
                                                  const std::string& reason,
                                                  const std::optional<std::string>& recoveredLogPath = std::nullopt)
{
    if (recoveredLogPath.has_value())
        UpdateInferLogPath(w, experiment, *recoveredLogPath);

    std::cout << sourceMarker
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
              << ",phase=infer"
              << ",reason=" << reason;
    if (recoveredLogPath.has_value())
        std::cout << ",infer_log_path=" << *recoveredLogPath;
    else if (experiment.inferLogPath.has_value())
        std::cout << ",infer_log_path=" << *experiment.inferLogPath;
    std::cout << std::endl;

    MarkExperimentPendingPhase(w, experiment, "infer", "analyze", 0);
    std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
              << std::endl;
}

inline bool TransitionAfterTrainModelAvailable(pqxx::work& w,
                                               const ExperimentRow& experiment,
                                               long long modelId,
                                               int exitCode,
                                               const std::string& fromPhase)
{
    ExperimentRow updatedExperiment = experiment;
    updatedExperiment.lastModelId = modelId;
    w.exec_params(
        "UPDATE experiment "
        "SET last_model_id = $1, exit_code = $2, error_message = NULL, updated_at = now() "
        "WHERE experiment_id = $3;",
        modelId,
        exitCode,
        experiment.experimentId);
    std::cout << "EXPERIMENT_LAST_MODEL_ID"
              << ",experiment_id=" << experiment.experimentId
              << ",model_id=" << modelId
              << std::endl;

    if (HasCompletedAnalysisResult(w, updatedExperiment))
    {
        std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << modelId
                  << std::endl;
        MarkExperimentDone(w, updatedExperiment, fromPhase);
    }
    else if (HasCompletedInferenceResult(w, updatedExperiment))
    {
        std::cout << "SCHEDULER_SKIP_EXISTING_INFERENCE"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << modelId
                  << std::endl;
        MarkExperimentPendingPhase(w, updatedExperiment, fromPhase, "analyze", exitCode);
        std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << modelId
                  << std::endl;
    }
    else if (updatedExperiment.inferStart.has_value() && updatedExperiment.inferEnd.has_value())
    {
        MarkExperimentPendingPhase(w, updatedExperiment, fromPhase, "infer", exitCode);
        std::cout << "SCHEDULER_ENQUEUE_INFER"
                  << ",experiment_id=" << experiment.experimentId
                  << ",model_id=" << modelId
                  << std::endl;
    }
    else
    {
        MarkExperimentFailed(w, updatedExperiment, "train_completed_missing_inference_range", exitCode);
        return false;
    }
    return true;
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
        return TransitionAfterTrainModelAvailable(w, experiment, *lastModelId, exitCode, "train");
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

inline int RecoverOrphanedRunningExperiments(pqxx::work& w)
{
    int recoveredOrFailed = 0;
    const std::vector<RunningExperimentState> runningExperiments = LoadRunningExperiments(w);
    for (const auto& state : runningExperiments)
    {
        const ExperimentRow& experiment = state.experiment;
        const std::string& phase = state.phase;
        if (RunningProcessExistsForExperiment(experiment, phase))
        {
            if (phase == "infer")
            {
                std::cout << "SCHEDULER_ORPHAN_INFER_STILL_RUNNING"
                          << ",experiment_id=" << experiment.experimentId
                          << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
                          << ",phase=infer"
                          << ",reason=matching_process_exists"
                          << std::endl;
            }
            else
            {
                std::cout << "SCHEDULER_RUNNING_EXPERIMENT_PRESENT"
                          << ",experiment_id=" << experiment.experimentId
                          << ",phase=" << phase
                          << std::endl;
            }
            continue;
        }

        ++recoveredOrFailed;
        std::cout << "SCHEDULER_ORPHAN_DETECTED"
                  << ",experiment_id=" << experiment.experimentId
                  << ",phase=" << phase
                  << std::endl;

        if (phase == "infer")
        {
            if (HasCompletedInferenceResult(w, experiment))
            {
                TransitionRecoveredInferenceToAnalyze(
                    w,
                    experiment,
                    "SCHEDULER_ORPHAN_RECOVERED_INFERENCE_RESULT",
                    "completed_inference_eval_result");
            }
            else if (HasValidInferenceLogPath(experiment))
            {
                TransitionRecoveredInferenceToAnalyze(
                    w,
                    experiment,
                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
                    "valid_infer_log_path");
            }
            else if (const std::optional<std::string> discoveredLog = DiscoverValidInferenceLog(experiment);
                     discoveredLog.has_value())
            {
                TransitionRecoveredInferenceToAnalyze(
                    w,
                    experiment,
                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
                    "discovered_valid_infer_log",
                    discoveredLog);
            }
            else
            {
                const std::string error = "orphaned_running_infer_no_process_no_result";
                MarkExperimentFailed(w, experiment, error);
                std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
                          << ",experiment_id=" << experiment.experimentId
                          << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
                          << ",phase=infer"
                          << ",error=" << error
                          << std::endl;
            }
            continue;
        }

        if (phase == "analyze")
        {
            if (HasCompletedAnalysisResult(w, experiment))
            {
                std::cout << "SCHEDULER_ORPHAN_RECOVERED_ANALYSIS"
                          << ",experiment_id=" << experiment.experimentId
                          << ",model_id=" << (experiment.lastModelId.has_value() ? std::to_string(*experiment.lastModelId) : "none")
                          << std::endl;
                MarkExperimentDone(w, experiment, "analyze");
            }
            else
            {
                const std::string error = "orphaned_running_analyze_no_process_no_result";
                MarkExperimentFailed(w, experiment, error);
                std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
                          << ",experiment_id=" << experiment.experimentId
                          << ",error=" << error
                          << std::endl;
            }
            continue;
        }

        if (phase != "train")
        {
            const std::string error = "orphaned_running_invalid_phase";
            MarkExperimentFailed(w, experiment, error);
            std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
                      << ",experiment_id=" << experiment.experimentId
                      << ",error=" << error
                      << std::endl;
            continue;
        }

        const std::string logText = ReadFileIfExists(experiment.trainLogPath);
        const std::optional<long long> finalModelId = ExtractCompletedTrainModelId(logText);
        if (finalModelId.has_value())
        {
            std::cout << "SCHEDULER_ORPHAN_RECOVERED_MODEL"
                      << ",experiment_id=" << experiment.experimentId
                      << ",model_id=" << *finalModelId
                      << std::endl;
            TransitionAfterTrainModelAvailable(w, experiment, *finalModelId, 0, "train");
            continue;
        }

        const std::optional<long long> checkpointModelId = ExtractCheckpointModelId(logText);
        if (checkpointModelId.has_value())
        {
            const std::string error = "orphaned_running_train_checkpoint_requires_manual_resume";
            std::cout << "SCHEDULER_ORPHAN_RECOVERED_CHECKPOINT"
                      << ",experiment_id=" << experiment.experimentId
                      << ",checkpoint_model_id=" << *checkpointModelId
                      << std::endl;
            MarkExperimentFailed(w, experiment, error + ";checkpoint_model_id=" + std::to_string(*checkpointModelId));
            std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
                      << ",experiment_id=" << experiment.experimentId
                      << ",error=" << error
                      << std::endl;
            continue;
        }

        const std::string error = "orphaned_running_train_no_process_no_model";
        MarkExperimentFailed(w, experiment, error);
        std::cout << "SCHEDULER_ORPHAN_MARKED_FAILED"
                  << ",experiment_id=" << experiment.experimentId
                  << ",error=" << error
                  << std::endl;
    }
    return recoveredOrFailed;
}

inline void CompleteInferPhase(pqxx::work& w,
                               const ExperimentRow& experiment,
                               int exitCode,
                               const std::string& commandDisplay)
{
    if (!experiment.lastModelId.has_value())
    {
        MarkExperimentFailed(w, experiment, "infer_missing_last_model_id", -1);
        return;
    }

    if (exitCode == 0)
    {
        if (HasCompletedAnalysisResult(w, experiment))
        {
            std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
                      << ",experiment_id=" << experiment.experimentId
                      << ",model_id=" << *experiment.lastModelId
                      << std::endl;
            MarkExperimentDone(w, experiment, "infer");
        }
        else
        {
            MarkExperimentPendingPhase(w, experiment, "infer", "analyze", exitCode);
            std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
                      << ",experiment_id=" << experiment.experimentId
                      << ",model_id=" << *experiment.lastModelId
                      << std::endl;
        }
    }
    else
    {
        const std::string errorMessage =
            "infer_failed;exit_code=" + std::to_string(exitCode) +
            ";command=" + commandDisplay;
        MarkExperimentFailed(w, experiment, errorMessage, exitCode);
    }
}

inline int RunTrainJobs(const SchedulerOptions& options, const QueueSnapshot& snapshot)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!options.dryRun)
        SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
        return 1;

    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "train");
    PhaseSchedulingStats stats;
    stats.phase = "train";
    stats.examined = static_cast<int>(jobs.size());
    stats.freeSlots = std::max(0, options.maxTrainProcs - RunningCountForPhase(snapshot, "train"));
    int freeSlots = stats.freeSlots;
    int rc = 0;

    EnsureLogDir(options.schedulerLogDir);
    for (const auto& job : jobs)
    {
        const std::optional<long long> resumeFrom =
            job.resumeModelId.has_value() ? job.resumeModelId : job.lastModelId;
        if (resumeFrom.has_value() && !ModelExists(w, *resumeFrom))
        {
            ++stats.skipped;
            LogSkip("train", job.experimentId, "model_not_found");
            if (!options.dryRun)
                MarkExperimentFailed(w, job, "train_model_not_found");
            continue;
        }
        if (RunningProcessExistsForExperiment(job, "train"))
        {
            ++stats.skipped;
            LogSkip("train", job.experimentId, "already_running");
            if (!options.dryRun)
            {
                MarkExperimentRunning(w, job, "train", LogPathFor(options, job, "train"));
                if (freeSlots > 0)
                    --freeSlots;
            }
            continue;
        }
        if (freeSlots <= 0)
        {
            ++stats.skipped;
            LogSkip("train", job.experimentId, "train_slots_full");
            continue;
        }

        const std::string logPath = LogPathFor(options, job, "train");
        const std::vector<std::string> command = BuildTrainCommand(options, job);
        const std::string commandDisplay = CommandForDisplay(command);
        if (options.dryRun)
        {
            std::cout << "EXPERIMENT_CHILD_COMMAND"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=train"
                      << ",dry_run=1"
                      << ",argv=" << commandDisplay
                      << std::endl;
            --freeSlots;
            ++stats.launched;
            continue;
        }

        MarkExperimentRunning(w, job, "train", logPath);
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
        try
        {
            const pid_t pid = LaunchChildProcess(command, logPath);
            std::cout << "SCHEDULER_CHILD_DETACHED"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=train"
                      << ",pid=" << pid
                      << ",log_path=" << logPath
                      << std::endl;
            --freeSlots;
            ++stats.launched;
        }
        catch (const std::exception& e)
        {
            MarkExperimentFailed(w, job, std::string{"train_launch_failed:"} + e.what());
            std::cout << "EXPERIMENT_FAILED"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=train"
                      << ",error=train_launch_failed"
                      << std::endl;
            rc = 1;
        }
    }
    PrintPhaseSchedulingStats(stats);
    w.commit();
    return rc;
}

inline int RunInferJobs(const SchedulerOptions& options, const QueueSnapshot& snapshot)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!options.dryRun)
        SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
        return 1;

    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "infer");
    PhaseSchedulingStats stats;
    stats.phase = "infer";
    stats.examined = static_cast<int>(jobs.size());
    stats.freeSlots = std::max(0, options.maxInferProcs - RunningCountForPhase(snapshot, "infer"));
    int freeSlots = stats.freeSlots;
    int rc = 0;

    EnsureLogDir(options.schedulerLogDir);
    for (const auto& job : jobs)
    {
        if (!job.lastModelId.has_value())
        {
            ++stats.skipped;
            LogSkip("infer", job.experimentId, "missing_model");
            if (!options.dryRun)
                MarkExperimentFailed(w, job, "infer_missing_last_model_id");
            continue;
        }
        if (!ModelExists(w, *job.lastModelId))
        {
            ++stats.skipped;
            LogSkip("infer", job.experimentId, "model_not_found");
            if (!options.dryRun)
                MarkExperimentFailed(w, job, "infer_model_not_found");
            continue;
        }
        if (HasCompletedInferenceResult(w, job))
        {
            ++stats.skipped;
            LogSkip("infer", job.experimentId, "existing_inference");
            if (!options.dryRun)
            {
                std::cout << "SCHEDULER_SKIP_EXISTING_INFERENCE"
                          << ",experiment_id=" << job.experimentId
                          << ",model_id=" << *job.lastModelId
                          << std::endl;
                if (HasCompletedAnalysisResult(w, job))
                {
                    std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
                              << ",experiment_id=" << job.experimentId
                              << ",model_id=" << *job.lastModelId
                              << std::endl;
                    MarkExperimentDone(w, job, "infer");
                }
                else
                {
                    MarkExperimentPendingPhase(w, job, "infer", "analyze");
                    std::cout << "SCHEDULER_ENQUEUE_ANALYZE"
                              << ",experiment_id=" << job.experimentId
                              << ",model_id=" << *job.lastModelId
                              << std::endl;
                }
            }
            continue;
        }
        if (HasValidInferenceLogPath(job))
        {
            ++stats.skipped;
            LogSkip("infer", job.experimentId, "valid_infer_log_path");
            if (!options.dryRun)
            {
                TransitionRecoveredInferenceToAnalyze(
                    w,
                    job,
                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
                    "valid_infer_log_path");
            }
            continue;
        }
        if (const std::optional<std::string> discoveredLog = DiscoverValidInferenceLog(job);
            discoveredLog.has_value())
        {
            ++stats.skipped;
            LogSkip("infer", job.experimentId, "discovered_valid_infer_log");
            if (!options.dryRun)
            {
                TransitionRecoveredInferenceToAnalyze(
                    w,
                    job,
                    "SCHEDULER_ORPHAN_RECOVERED_INFER_LOG",
                    "discovered_valid_infer_log",
                    discoveredLog);
            }
            continue;
        }
        if (RunningProcessExistsForExperiment(job, "infer"))
        {
            ++stats.skipped;
            LogSkip("infer", job.experimentId, "already_running");
            if (!options.dryRun)
            {
                MarkExperimentRunning(w, job, "infer", LogPathFor(options, job, "infer"));
                if (freeSlots > 0)
                    --freeSlots;
            }
            continue;
        }
        if (freeSlots <= 0)
        {
            ++stats.skipped;
            LogSkip("infer", job.experimentId, "infer_slots_full");
            continue;
        }

        const std::string logPath = LogPathFor(options, job, "infer");
        const std::vector<std::string> command = BuildInferCommand(options, job);
        const std::string commandDisplay = CommandForDisplay(command);
        if (options.dryRun)
        {
            std::cout << "EXPERIMENT_CHILD_COMMAND"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=infer"
                      << ",dry_run=1"
                      << ",argv=" << commandDisplay
                      << std::endl;
            --freeSlots;
            ++stats.launched;
            continue;
        }

        MarkExperimentRunning(w, job, "infer", logPath);
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
        try
        {
            const pid_t pid = LaunchChildProcess(command, logPath);
            std::cout << "SCHEDULER_CHILD_DETACHED"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=infer"
                      << ",pid=" << pid
                      << ",log_path=" << logPath
                      << std::endl;
            --freeSlots;
            ++stats.launched;
        }
        catch (const std::exception& e)
        {
            MarkExperimentFailed(w, job, std::string{"infer_launch_failed:"} + e.what());
            std::cout << "EXPERIMENT_FAILED"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=infer"
                      << ",error=infer_launch_failed"
                      << std::endl;
            rc = 1;
        }
    }
    PrintPhaseSchedulingStats(stats);
    w.commit();
    return rc;
}

inline int RunAnalyzeJobs(const SchedulerOptions& options, const QueueSnapshot& snapshot)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!options.dryRun)
        SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
        return 1;

    const std::vector<ExperimentRow> jobs = LoadPendingExperiments(w, "analyze");
    PhaseSchedulingStats stats;
    stats.phase = "analyze";
    stats.examined = static_cast<int>(jobs.size());
    stats.freeSlots = std::max(0, options.maxAnalyzeProcs - RunningCountForPhase(snapshot, "analyze"));
    int freeSlots = stats.freeSlots;
    int rc = 0;

    EnsureLogDir(options.schedulerLogDir);
    for (const auto& job : jobs)
    {
        if (!job.lastModelId.has_value())
        {
            ++stats.skipped;
            LogSkip("analyze", job.experimentId, "missing_model");
            if (!options.dryRun)
                MarkAnalyzeFailed(w, job.experimentId, "analyze_missing_last_model_id");
            continue;
        }
        if (!ModelExists(w, *job.lastModelId))
        {
            ++stats.skipped;
            LogSkip("analyze", job.experimentId, "model_not_found");
            if (!options.dryRun)
                MarkAnalyzeFailed(w, job.experimentId, "analyze_model_not_found");
            continue;
        }
        if (HasCompletedAnalysisResult(w, job))
        {
            ++stats.skipped;
            LogSkip("analyze", job.experimentId, "existing_analysis");
            if (!options.dryRun)
            {
                std::cout << "SCHEDULER_SKIP_EXISTING_ANALYSIS"
                          << ",experiment_id=" << job.experimentId
                          << ",model_id=" << *job.lastModelId
                          << std::endl;
                MarkExperimentDone(w, job, "analyze");
            }
            continue;
        }
        if (!HasCompletedInferenceResult(w, job) && !HasValidInferenceLogPath(job))
        {
            ++stats.skipped;
            LogSkip("analyze", job.experimentId, "infer_log_missing");
            continue;
        }
        if (RunningProcessExistsForExperiment(job, "analyze"))
        {
            ++stats.skipped;
            LogSkip("analyze", job.experimentId, "already_running");
            if (!options.dryRun)
            {
                MarkExperimentRunning(w, job, "analyze", LogPathFor(options, job, "analysis"));
                if (freeSlots > 0)
                    --freeSlots;
            }
            continue;
        }
        if (freeSlots <= 0)
        {
            ++stats.skipped;
            LogSkip("analyze", job.experimentId, "analyze_slots_full");
            continue;
        }

        const std::string logPath = LogPathFor(options, job, "analysis");
        const std::vector<std::string> command = BuildAnalyzeCommand(options, job);
        const std::string commandDisplay = CommandForDisplay(command);
        if (options.dryRun)
        {
            std::cout << "EXPERIMENT_ANALYSIS_STARTED"
                      << ",experiment_id=" << job.experimentId
                      << ",dry_run=1"
                      << std::endl;
            std::cout << "EXPERIMENT_CHILD_COMMAND"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=analyze"
                      << ",dry_run=1"
                      << ",argv=" << commandDisplay
                      << std::endl;
            --freeSlots;
            ++stats.launched;
            continue;
        }

        MarkExperimentRunning(w, job, "analyze", logPath);
        std::cout << "EXPERIMENT_STARTED"
                  << ",experiment_id=" << job.experimentId
                  << ",phase=analyze"
                  << std::endl;
        std::cout << "EXPERIMENT_CHILD_COMMAND"
                  << ",experiment_id=" << job.experimentId
                  << ",phase=analyze"
                  << ",argv=" << commandDisplay
                  << std::endl;
        PrintSchedulerExec(command);
        try
        {
            const pid_t pid = LaunchChildProcess(command, logPath);
            std::cout << "SCHEDULER_CHILD_DETACHED"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=analyze"
                      << ",pid=" << pid
                      << ",log_path=" << logPath
                      << std::endl;
            --freeSlots;
            ++stats.launched;
        }
        catch (const std::exception& e)
        {
            MarkAnalyzeFailed(w, job.experimentId, std::string{"analyze_launch_failed:"} + e.what());
            std::cout << "EXPERIMENT_FAILED"
                      << ",experiment_id=" << job.experimentId
                      << ",phase=analyze"
                      << ",error=analyze_launch_failed"
                      << std::endl;
            rc = 1;
        }
    }
    PrintPhaseSchedulingStats(stats);
    w.commit();
    return rc;
}

inline int RunSchedulerOnce(const SchedulerOptions& options)
{
    int rc = 0;
    QueueSnapshot snapshot;
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        if (!options.dryRun)
            SetTransactionReadWrite(w);
        if (!RequireSchedulerTables(w))
            return 1;
        if (!options.dryRun)
        {
            RecoverOrphanedRunningExperiments(w);
            rc |= FailInvalidSchedulerPhases(w);
        }
        snapshot = LoadQueueSnapshot(w);
        PrintQueueSnapshot(snapshot);
        w.commit();
    }

    rc |= RunTrainJobs(options, snapshot);
    rc |= RunInferJobs(options, snapshot);
    rc |= RunAnalyzeJobs(options, snapshot);
    return rc;
}

inline int RunScheduler(const SchedulerOptions& options)
{
    int recoveryCount = 0;
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        SetTransactionReadWrite(w);
        if (!RequireSchedulerTables(w))
            return 1;
        if (!options.dryRun)
        {
            recoveryCount = RecoverOrphanedRunningExperiments(w);
            if (FailInvalidSchedulerPhases(w) != 0)
            {
                w.commit();
                return 1;
            }
        }
        w.commit();
    }

    std::cout << "SCHEDULER_START"
              << ",dry_run=" << (options.dryRun ? "1" : "0")
              << ",max_train_procs=" << options.maxTrainProcs
              << ",max_infer_procs=" << options.maxInferProcs
              << ",max_analyze_procs=" << options.maxAnalyzeProcs
              << ",recover_orphans_only=" << (options.recoverOrphansOnly ? "1" : "0")
              << std::endl;
    if (options.dryRun)
        std::cout << "SCHEDULER_DRY_RUN=1" << std::endl;
    if (options.recoverOrphansOnly)
    {
        std::cout << "SCHEDULER_ORPHAN_RECOVERY_DONE"
                  << ",recovered_or_failed=" << recoveryCount
                  << std::endl;
        std::cout << "SCHEDULER_STOP"
                  << ",exit_code=0"
                  << std::endl;
        return 0;
    }

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
                  << "[--scheduler-log-dir=PATH] [--dry-run] [--recover-orphans-only]\n"
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
