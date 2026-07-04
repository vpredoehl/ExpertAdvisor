#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
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
#include "Params.hpp"
#include "SupportedSymbols.hpp"

namespace EA::ExperimentScheduler
{

struct SchedulerOptions
{
    bool scheduleExperiments = false;
    bool enqueueExperiment = false;
    bool queueExperiment = false;
    bool queueSweep = false;
    bool autoResume = false;
    bool analyzeCompletedExperiments = false;
    std::optional<long long> analyzeExperimentId;
    bool printLeaderboard = false;
    bool schedulerStatus = false;
    bool help = false;
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
    std::optional<int> epochs;
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
    std::string logLevel = "summary";
};

struct QueueDefaults
{
    double threshold = default_c_next_threshold;
    double coreLrMult = default_core_lr_mult;
    double headLrMult = default_head_weight_lr_mult;
    int checkpointInterval = 20;
    std::string trainStart = "2010-01-01";
    std::string trainEnd = "2025-01-01";
    std::string inferStart = "2025-01-01";
    std::string inferEnd = "2026-01-01";
};

struct QueueResumeMeta
{
    long long modelId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    double threshold = 0.0;
    std::string trainStart;
    std::string trainEnd;
    int completedEpochs = 0;
    std::optional<double> coreLrMult;
    std::optional<double> headLrMult;
};

struct AutoResumeCandidate
{
    long long modelId = -1;
    std::string name;
    int completedEpochs = 0;
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

struct SchedulerStatusJob
{
    long long experimentId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    std::string phase;
    std::string status;
    int targetEpochs = 0;
    int checkpointInterval = 0;
    std::optional<long long> modelId;
    std::optional<int> completedEpochs;
    std::optional<int> currentEpoch;
    std::optional<int> lastCheckpointEpoch;
    std::optional<long long> lastCheckpointModelId;
    std::optional<int> nextCheckpointEpoch;
    std::optional<double> loss;
    std::optional<double> validationAccuracy;
    std::optional<double> elapsedSeconds;
    std::optional<double> etaSeconds;
    std::optional<int> pid;
    std::optional<double> cpuPercent;
    std::optional<double> memPercent;
    std::optional<double> rssMb;
    std::optional<std::string> recentProgress;
    std::optional<std::string> trainLogPath;
    std::optional<std::string> inferLogPath;
    std::optional<std::string> analysisLogPath;
    std::string startedAt;
    std::string updatedAt;
    std::string completedAt;
    std::string errorMessage;
};

struct SchedulerProcessResource
{
    int pid = -1;
    double cpuPercent = 0.0;
    double memPercent = 0.0;
    double rssMb = 0.0;
};

struct SchedulerResourceAggregate
{
    int workers = 0;
    double cpuPercent = 0.0;
    double memPercent = 0.0;
    double rssMb = 0.0;
};

struct SchedulerStatusCounts
{
    int queued = 0;
    int running = 0;
    int completed = 0;
    int failed = 0;
    int cancelled = 0;
};

struct SchedulerStatusProcessSnapshot
{
    bool processDetectionAvailable = false;
    std::vector<int> schedulerPids;
    std::map<int, SchedulerProcessResource> resourcesByPid;
    std::map<long long, int> trainPidByExperiment;
    std::map<long long, int> inferPidByExperiment;
    std::map<long long, int> analysisPidByExperiment;
    SchedulerResourceAggregate schedulerResources;
    SchedulerResourceAggregate trainResources;
    SchedulerResourceAggregate inferResources;
    SchedulerResourceAggregate analysisResources;
    int trainWorkers = 0;
    int inferWorkers = 0;
    int analysisWorkers = 0;
    std::optional<int> maxTrainProcs;
    std::optional<int> maxInferProcs;
    std::optional<int> maxAnalyzeProcs;
    std::optional<int> schedulerPollSeconds;
    std::optional<double> totalCpuPercent;
    std::optional<double> systemMemoryUsedMb;
    std::optional<double> systemMemoryTotalMb;
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
            arg == "--queue-experiment" ||
            arg == "--queue-sweep" ||
            arg == "--analyze-completed-experiments" ||
            arg == "--print-experiment-leaderboard" ||
            arg == "--scheduler-status" ||
            arg == "--help" ||
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
        else if (arg == "--queue-experiment")
            options.queueExperiment = true;
        else if (arg == "--queue-sweep")
            options.queueSweep = true;
        else if (arg == "--auto-resume")
            options.autoResume = true;
        else if (arg == "--analyze-completed-experiments")
            options.analyzeCompletedExperiments = true;
        else if (arg == "--print-experiment-leaderboard")
            options.printLeaderboard = true;
        else if (arg == "--scheduler-status")
            options.schedulerStatus = true;
        else if (arg == "--help")
            options.help = true;
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
        else if (arg == "--threshold")
            options.cNextThreshold = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--core-lr-mult")
            options.coreLrMult = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--core-lr")
            options.coreLrMult = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--head-lr-mult")
            options.headLrMult = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--head-lr")
            options.headLrMult = ParsePositiveDouble(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--target-epochs")
            options.targetEpochs = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
        else if (arg == "--epochs")
            options.epochs = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));
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
        else if (arg == "--log-level")
            options.logLevel = RequireNextArg(argc, argv, i, arg);
        else if (SplitOptionWithValue(arg, "--symbol", value))
            options.symbol = EA::CanonicalSymbol::Normalize(value);
        else if (SplitOptionWithValue(arg, "--prediction-horizon", value))
            options.predictionHorizon = ParsePositiveInt("--prediction-horizon", value);
        else if (SplitOptionWithValue(arg, "--c-next-threshold", value))
            options.cNextThreshold = ParsePositiveDouble("--c-next-threshold", value);
        else if (SplitOptionWithValue(arg, "--threshold", value))
            options.cNextThreshold = ParsePositiveDouble("--threshold", value);
        else if (SplitOptionWithValue(arg, "--core-lr-mult", value))
            options.coreLrMult = ParsePositiveDouble("--core-lr-mult", value);
        else if (SplitOptionWithValue(arg, "--core-lr", value))
            options.coreLrMult = ParsePositiveDouble("--core-lr", value);
        else if (SplitOptionWithValue(arg, "--head-lr-mult", value))
            options.headLrMult = ParsePositiveDouble("--head-lr-mult", value);
        else if (SplitOptionWithValue(arg, "--head-lr", value))
            options.headLrMult = ParsePositiveDouble("--head-lr", value);
        else if (SplitOptionWithValue(arg, "--target-epochs", value))
            options.targetEpochs = ParsePositiveInt("--target-epochs", value);
        else if (SplitOptionWithValue(arg, "--epochs", value))
            options.epochs = ParsePositiveInt("--epochs", value);
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
        else if (SplitOptionWithValue(arg, "--log-level", value))
            options.logLevel = value;
        else if (arg.rfind("--", 0) == 0)
            throw std::invalid_argument("unknown scheduler option '" + arg + "'");
        else
            throw std::invalid_argument("unexpected positional scheduler argument '" + arg + "'");
    }

    const int commandCount =
        (options.scheduleExperiments ? 1 : 0) +
        (options.enqueueExperiment ? 1 : 0) +
        (options.queueExperiment ? 1 : 0) +
        (options.queueSweep ? 1 : 0) +
        (options.analyzeCompletedExperiments ? 1 : 0) +
        (options.analyzeExperimentId.has_value() ? 1 : 0) +
        (options.printLeaderboard ? 1 : 0) +
        (options.schedulerStatus ? 1 : 0) +
        (options.help ? 1 : 0);
    if (commandCount != 1)
        throw std::invalid_argument("expected exactly one experiment scheduler command");
    if (options.recoverOrphansOnly && !options.scheduleExperiments)
        throw std::invalid_argument("--recover-orphans-only requires --schedule-experiments");
    if (options.logLevel != "quiet" &&
        options.logLevel != "summary" &&
        options.logLevel != "diagnostic")
        throw std::invalid_argument("--log-level must be quiet, summary, or diagnostic");

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

inline std::string DateOnly(const std::string& value)
{
    return value.size() >= 10 ? value.substr(0, 10) : value;
}

inline bool SameDate(const std::string& lhs, const std::string& rhs)
{
    return DateOnly(lhs) == DateOnly(rhs);
}

inline QueueResumeMeta LoadQueueResumeMeta(pqxx::work& w, long long modelId)
{
    if (!ModelExists(w, modelId))
        throw std::runtime_error("resume model_id not found");

    auto dims = DBIO::PgModelIO::loadParameterDims(w, modelId, "train_config_meta");
    auto vals = DBIO::PgModelIO::loadParameterValues(w, modelId, "train_config_meta");
    if (dims.n_rows != 1 ||
        dims.n_cols < DBIO::PgModelIO::kTrainConfigMetaExtendedFieldCount ||
        vals.size() < static_cast<size_t>(DBIO::PgModelIO::kTrainConfigMetaExtendedFieldCount))
        throw std::runtime_error("resume requires complete train_config_meta with 14 fields");

    QueueResumeMeta meta;
    meta.modelId = modelId;
    meta.symbol = DBIO::PgModelIO::decodeTrainSymbolMeta(w, modelId);
    meta.predictionHorizon = static_cast<int>(std::llround(vals[1]));
    meta.threshold = vals[2];
    meta.completedEpochs = static_cast<int>(std::llround(vals[10]));
    meta.coreLrMult = vals[11];
    meta.headLrMult = vals[12];
    const auto range = DBIO::PgModelIO::decodeTrainRangeMeta(w, modelId);
    meta.trainStart = DateOnly(range.first);
    meta.trainEnd = DateOnly(range.second);
    return meta;
}

inline void PrintQueueResumeMeta(const char* marker,
                                 const QueueResumeMeta& meta,
                                 int targetEpochs)
{
    std::cout << marker
              << ",resume_model_id=" << meta.modelId
              << ",symbol=" << meta.symbol
              << ",prediction_horizon=" << meta.predictionHorizon
              << ",threshold=" << FormatDouble(meta.threshold)
              << ",completed_epochs=" << meta.completedEpochs
              << ",target_epochs=" << targetEpochs
              << ",train_start=" << meta.trainStart
              << ",train_end=" << meta.trainEnd
              << std::endl;
}

inline void ThrowQueueResumeInvalid(const std::string& reason,
                                    long long modelId,
                                    const std::string& detail = {})
{
    std::cout << "QUEUE_RESUME_INVALID"
              << ",resume_model_id=" << modelId
              << ",reason=" << reason;
    if (!detail.empty())
        std::cout << ",detail=" << detail;
    std::cout << std::endl;
    throw std::invalid_argument("QUEUE_RESUME_INVALID:" + reason);
}

inline void MergeResumeMetaIntoQueueOptions(SchedulerOptions& options,
                                            const QueueResumeMeta& meta)
{
    if (!options.targetEpochs.has_value())
        ThrowQueueResumeInvalid("missing_target_epochs", meta.modelId);
    if (*options.targetEpochs <= meta.completedEpochs)
    {
        ThrowQueueResumeInvalid("target_epochs_not_greater_than_completed_epoch",
                                meta.modelId,
                                "completed_epochs=" + std::to_string(meta.completedEpochs) +
                                ";target_epochs=" + std::to_string(*options.targetEpochs));
    }

    if (options.symbol.has_value() &&
        EA::CanonicalSymbol::Normalize(*options.symbol) != meta.symbol)
    {
        ThrowQueueResumeInvalid("symbol_mismatch",
                                meta.modelId,
                                "model=" + meta.symbol + ";runtime=" + *options.symbol);
    }
    if (options.predictionHorizon.has_value() &&
        *options.predictionHorizon != meta.predictionHorizon)
    {
        ThrowQueueResumeInvalid("prediction_horizon_mismatch",
                                meta.modelId,
                                "model=" + std::to_string(meta.predictionHorizon) +
                                ";runtime=" + std::to_string(*options.predictionHorizon));
    }
    if (options.cNextThreshold.has_value() &&
        std::fabs(*options.cNextThreshold - meta.threshold) > 1e-7)
    {
        ThrowQueueResumeInvalid("threshold_mismatch",
                                meta.modelId,
                                "model=" + FormatDouble(meta.threshold) +
                                ";runtime=" + FormatDouble(*options.cNextThreshold));
    }
    if (options.trainStart.has_value() && !SameDate(*options.trainStart, meta.trainStart))
    {
        ThrowQueueResumeInvalid("train_start_mismatch",
                                meta.modelId,
                                "model=" + meta.trainStart + ";runtime=" + *options.trainStart);
    }
    if (options.trainEnd.has_value() && !SameDate(*options.trainEnd, meta.trainEnd))
    {
        ThrowQueueResumeInvalid("train_end_mismatch",
                                meta.modelId,
                                "model=" + meta.trainEnd + ";runtime=" + *options.trainEnd);
    }

    options.symbol = meta.symbol;
    options.predictionHorizon = meta.predictionHorizon;
    options.cNextThreshold = meta.threshold;
    options.trainStart = meta.trainStart;
    options.trainEnd = meta.trainEnd;
    options.coreLrMult = meta.coreLrMult;
    options.headLrMult = meta.headLrMult;

    PrintQueueResumeMeta("QUEUE_RESUME_MODEL", meta, *options.targetEpochs);
}

inline std::vector<AutoResumeCandidate> LoadAutoResumeCandidates(pqxx::work& w,
                                                                 const SchedulerOptions& options)
{
    const std::string canonicalSymbol = EA::CanonicalSymbol::Normalize(*options.symbol);
    const std::string trainRange = DateOnly(*options.trainStart) + "|" + DateOnly(*options.trainEnd);

    std::ostringstream sql;
    sql << "WITH cfg AS ("
        << "  SELECT model_id,"
        << "         max(value) FILTER (WHERE col_idx = 1) AS prediction_horizon,"
        << "         max(value) FILTER (WHERE col_idx = 2) AS threshold_logret,"
        << "         max(value) FILTER (WHERE col_idx = 10) AS completed_epochs"
        << "  FROM matrix"
        << "  WHERE param_name = 'train_config_meta' AND row_idx = 0"
        << "  GROUP BY model_id"
        << "), sym AS ("
        << "  SELECT model_id, string_agg(chr(round(value)::int), '' ORDER BY col_idx) AS symbol"
        << "  FROM matrix"
        << "  WHERE param_name = 'train_symbol_meta' AND row_idx = 0"
        << "  GROUP BY model_id"
        << "), rng AS ("
        << "  SELECT model_id, string_agg(chr(round(value)::int), '' ORDER BY col_idx) AS train_range"
        << "  FROM matrix"
        << "  WHERE param_name = 'train_range_meta' AND row_idx = 0"
        << "  GROUP BY model_id"
        << ") "
        << "SELECT m.model_id, COALESCE(m.name, ''), round(cfg.completed_epochs)::int "
        << "FROM model m "
        << "JOIN cfg ON cfg.model_id = m.model_id "
        << "JOIN sym ON sym.model_id = m.model_id "
        << "JOIN rng ON rng.model_id = m.model_id "
        << "WHERE sym.symbol = " << w.quote(canonicalSymbol)
        << " AND round(cfg.prediction_horizon)::int = " << *options.predictionHorizon
        << " AND abs(cfg.threshold_logret - " << FormatDouble(*options.cNextThreshold) << ") <= 1e-7"
        << " AND rng.train_range = " << w.quote(trainRange)
        << " AND round(cfg.completed_epochs)::int < " << *options.targetEpochs
        << " ORDER BY round(cfg.completed_epochs)::int DESC, m.model_id DESC;";

    pqxx::result rows = w.exec(sql.str());
    std::vector<AutoResumeCandidate> candidates;
    candidates.reserve(rows.size());
    for (const auto& row : rows)
    {
        candidates.push_back(AutoResumeCandidate{
            row[0].as<long long>(),
            row[1].as<std::string>(),
            row[2].as<int>()
        });
    }
    return candidates;
}

inline long long ResolveAutoResumeModelId(pqxx::work& w,
                                          const SchedulerOptions& options)
{
    const std::vector<AutoResumeCandidate> candidates = LoadAutoResumeCandidates(w, options);
    if (candidates.empty())
    {
        std::cout << "QUEUE_RESUME_AUTO_NO_MATCH"
                  << ",symbol=" << *options.symbol
                  << ",prediction_horizon=" << *options.predictionHorizon
                  << ",threshold=" << FormatDouble(*options.cNextThreshold)
                  << ",target_epochs=" << *options.targetEpochs
                  << ",train_start=" << *options.trainStart
                  << ",train_end=" << *options.trainEnd
                  << std::endl;
        throw std::invalid_argument("QUEUE_RESUME_AUTO_NO_MATCH");
    }

    const int bestCompletedEpochs = candidates.front().completedEpochs;
    int tiedBestCount = 0;
    for (const auto& candidate : candidates)
    {
        if (candidate.completedEpochs == bestCompletedEpochs)
            ++tiedBestCount;
    }

    for (const auto& candidate : candidates)
    {
        std::cout << "QUEUE_RESUME_AUTO_CANDIDATE"
                  << ",model_id=" << candidate.modelId
                  << ",name=" << candidate.name
                  << ",completed_epochs=" << candidate.completedEpochs
                  << std::endl;
    }

    if (tiedBestCount > 1)
    {
        std::cout << "QUEUE_RESUME_AUTO_AMBIGUOUS"
                  << ",best_completed_epochs=" << bestCompletedEpochs
                  << ",candidate_count=" << candidates.size()
                  << std::endl;
        throw std::invalid_argument("QUEUE_RESUME_AUTO_AMBIGUOUS");
    }

    std::cout << "QUEUE_RESUME_AUTO_SELECTED"
              << ",resume_model_id=" << candidates.front().modelId
              << ",completed_epochs=" << candidates.front().completedEpochs
              << std::endl;
    return candidates.front().modelId;
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

inline std::string QueueDuplicateWhereClause(pqxx::work& w,
                                             const SchedulerOptions& options,
                                             const std::string& canonicalSymbol)
{
    std::ostringstream sql;
    sql << "symbol = " << w.quote(canonicalSymbol)
        << " AND prediction_horizon = " << *options.predictionHorizon
        << " AND target_epochs = " << *options.targetEpochs
        << " AND c_next_threshold = " << FormatDouble(*options.cNextThreshold)
        << " AND train_start = " << w.quote(*options.trainStart) << "::timestamptz"
        << " AND train_end = " << w.quote(*options.trainEnd) << "::timestamptz"
        << " AND status NOT IN ('failed', 'cancelled')";
    return sql.str();
}

inline long long CurrentDuplicateNonce()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

inline long long InsertExperimentRecord(pqxx::work& w,
                                        const SchedulerOptions& options,
                                        const std::string& canonicalSymbol,
                                        long long duplicateNonce)
{
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
    return inserted[0][0].as<long long>();
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
    const long long experimentId = InsertExperimentRecord(w, options, canonicalSymbol, duplicateNonce);
    w.commit();

    std::cout << "EXPERIMENT_ENQUEUED"
              << ",experiment_id=" << experimentId
              << ",symbol=" << canonicalSymbol
              << ",prediction_horizon=" << *options.predictionHorizon
              << ",target_epochs=" << *options.targetEpochs
              << std::endl;
    return 0;
}

inline SchedulerOptions ApplyQueueDefaults(SchedulerOptions options)
{
    const QueueDefaults defaults;
    if (!options.cNextThreshold.has_value())
        options.cNextThreshold = defaults.threshold;
    if (!options.coreLrMult.has_value())
        options.coreLrMult = defaults.coreLrMult;
    if (!options.headLrMult.has_value())
        options.headLrMult = defaults.headLrMult;
    if (options.checkpointInterval <= 0)
        options.checkpointInterval = defaults.checkpointInterval;
    if (!options.trainStart.has_value())
        options.trainStart = defaults.trainStart;
    if (!options.trainEnd.has_value())
        options.trainEnd = defaults.trainEnd;
    if (!options.inferStart.has_value())
        options.inferStart = defaults.inferStart;
    if (!options.inferEnd.has_value())
        options.inferEnd = defaults.inferEnd;
    return options;
}

inline void EnsureRequiredQueueOptions(const SchedulerOptions& options)
{
    if (options.epochs.has_value())
        throw std::invalid_argument("--queue-experiment/--queue-sweep use --target-epochs as the absolute final epoch; --epochs is not supported");
    if (options.autoResume && !options.queueExperiment)
        throw std::invalid_argument("--auto-resume is supported only with --queue-experiment");
    if (options.autoResume && options.resumeModelId.has_value())
        throw std::invalid_argument("--auto-resume cannot be combined with --resume-model-id");
    if (options.queueSweep && options.resumeModelId.has_value())
        throw std::invalid_argument("--queue-sweep does not support --resume-model-id");
    if (options.queueSweep && options.autoResume)
        throw std::invalid_argument("--queue-sweep does not support --auto-resume");
    if (!options.targetEpochs.has_value())
        throw std::invalid_argument("--queue-experiment/--queue-sweep require --target-epochs");
    if (!options.resumeModelId.has_value() && !options.autoResume && !options.predictionHorizon.has_value())
        throw std::invalid_argument("--queue-experiment/--queue-sweep require --prediction-horizon");
    if (options.queueExperiment && !options.symbol.has_value())
    {
        if (!options.resumeModelId.has_value())
            throw std::invalid_argument("--queue-experiment requires --symbol unless --resume-model-id is supplied");
    }
    if (options.queueSweep && options.symbol.has_value())
        throw std::invalid_argument("--queue-sweep queues all supported symbols; do not pass --symbol");
}

inline void PrintQueueConfig(const char* marker,
                             const SchedulerOptions& options,
                             const std::string& canonicalSymbol,
                             const std::optional<long long>& experimentId = std::nullopt)
{
    std::cout << marker;
    if (experimentId.has_value())
        std::cout << ",experiment_id=" << *experimentId;
    std::cout << ",symbol=" << canonicalSymbol
              << ",prediction_horizon=" << *options.predictionHorizon
              << ",target_epochs=" << *options.targetEpochs
              << ",resume_model_id=" << (options.resumeModelId.has_value() ? std::to_string(*options.resumeModelId) : "NULL")
              << ",threshold=" << FormatDouble(*options.cNextThreshold)
              << ",core_lr=" << (options.coreLrMult.has_value() ? FormatDouble(*options.coreLrMult) : "NULL")
              << ",head_lr=" << (options.headLrMult.has_value() ? FormatDouble(*options.headLrMult) : "NULL")
              << ",checkpoint_interval=" << options.checkpointInterval
              << ",train_start=" << *options.trainStart
              << ",train_end=" << *options.trainEnd
              << ",infer_start=" << (options.inferStart.has_value() ? *options.inferStart : "NULL")
              << ",infer_end=" << (options.inferEnd.has_value() ? *options.inferEnd : "NULL")
              << std::endl;
}

inline std::optional<long long> FindQueueDuplicate(pqxx::work& w,
                                                   const SchedulerOptions& options,
                                                   const std::string& canonicalSymbol)
{
    pqxx::result duplicate = w.exec(
        "SELECT experiment_id FROM experiment WHERE " +
        QueueDuplicateWhereClause(w, options, canonicalSymbol) +
        " ORDER BY experiment_id ASC LIMIT 1;");
    if (duplicate.empty())
        return std::nullopt;
    return duplicate[0][0].as<long long>();
}

inline bool QueueOneExperiment(pqxx::work& w,
                               const SchedulerOptions& options,
                               const std::string& canonicalSymbol)
{
    const std::optional<long long> duplicateExperimentId =
        FindQueueDuplicate(w, options, canonicalSymbol);
    if (duplicateExperimentId.has_value())
    {
        PrintQueueConfig("QUEUE_ALREADY_EXISTS", options, canonicalSymbol, duplicateExperimentId);
        return false;
    }

    const long long experimentId = InsertExperimentRecord(w, options, canonicalSymbol, 0);
    PrintQueueConfig("QUEUE_EXPERIMENT_CREATED", options, canonicalSymbol, experimentId);
    return true;
}

inline int QueueExperiments(const SchedulerOptions& rawOptions)
{
    SchedulerOptions options = rawOptions;

    if (options.resumeModelId.has_value() && options.epochs.has_value())
    {
        ThrowQueueResumeInvalid("epochs_conflicts_with_absolute_target_epochs",
                                *options.resumeModelId);
    }

    if (options.dryRun && !options.resumeModelId.has_value() && !options.autoResume)
    {
        options = ApplyQueueDefaults(options);
        EnsureRequiredQueueOptions(options);
        std::cout << "SCHEDULER_DRY_RUN=1" << std::endl;
        const auto& symbols = options.queueSweep
            ? EA::SupportedSymbols::TrainingSymbols()
            : std::vector<std::string>{EA::CanonicalSymbol::Normalize(*options.symbol)};
        for (const auto& symbol : symbols)
            PrintQueueConfig("QUEUE_EXPERIMENT_DRY_RUN", options, EA::CanonicalSymbol::Normalize(symbol));
        return 0;
    }

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadWrite(w);
    if (!RequireSchedulerTables(w))
        return 2;

    if (options.autoResume)
    {
        options = ApplyQueueDefaults(options);
        EnsureRequiredQueueOptions(options);
        options.resumeModelId = ResolveAutoResumeModelId(w, options);
        options.autoResume = false;
    }

    if (options.resumeModelId.has_value())
    {
        if (rawOptions.coreLrMult.has_value())
            ThrowQueueResumeInvalid("core_lr_override_not_allowed_in_resume", *options.resumeModelId);
        if (rawOptions.headLrMult.has_value())
            ThrowQueueResumeInvalid("head_lr_override_not_allowed_in_resume", *options.resumeModelId);
        const QueueResumeMeta meta = LoadQueueResumeMeta(w, *options.resumeModelId);
        MergeResumeMetaIntoQueueOptions(options, meta);
    }

    options = ApplyQueueDefaults(options);
    EnsureRequiredQueueOptions(options);

    if (options.dryRun)
    {
        std::cout << "SCHEDULER_DRY_RUN=1" << std::endl;
        const auto& symbols = options.queueSweep
            ? EA::SupportedSymbols::TrainingSymbols()
            : std::vector<std::string>{EA::CanonicalSymbol::Normalize(*options.symbol)};
        for (const auto& symbol : symbols)
            PrintQueueConfig("QUEUE_EXPERIMENT_DRY_RUN", options, EA::CanonicalSymbol::Normalize(symbol));
        w.commit();
        return 0;
    }

    int created = 0;
    int duplicates = 0;
    if (options.queueSweep)
    {
        for (const auto& symbol : EA::SupportedSymbols::TrainingSymbols())
        {
            SchedulerOptions perSymbol = options;
            perSymbol.symbol = EA::CanonicalSymbol::Normalize(symbol);
            if (QueueOneExperiment(w, perSymbol, *perSymbol.symbol))
                ++created;
            else
                ++duplicates;
        }
    }
    else
    {
        const std::string canonicalSymbol = EA::CanonicalSymbol::Normalize(*options.symbol);
        if (QueueOneExperiment(w, options, canonicalSymbol))
            ++created;
        else
            ++duplicates;
    }

    w.commit();
    std::cout << "QUEUE_DONE"
              << ",created=" << created
              << ",duplicates=" << duplicates
              << std::endl;
    return created > 0 ? 0 : (duplicates > 0 ? 3 : 0);
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

inline void PersistExperimentRunningBeforeLaunch(const ExperimentRow& experiment,
                                                 const std::string& phase,
                                                 const std::string& logPath)
{
    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    SetTransactionReadWrite(w);
    MarkExperimentRunning(w, experiment, phase, logPath);
    w.commit();
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

        PersistExperimentRunningBeforeLaunch(job, "train", logPath);
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

        PersistExperimentRunningBeforeLaunch(job, "infer", logPath);
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

        PersistExperimentRunningBeforeLaunch(job, "analyze", logPath);
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

inline bool SchedulerStatusShouldEmitMachineRecords(const SchedulerOptions& options)
{
    return options.logLevel == "summary" || options.logLevel == "diagnostic";
}

inline bool UseAnsiColors()
{
    const char* term = std::getenv("TERM");
    return ::isatty(STDOUT_FILENO) && term && std::string{term} != "dumb";
}

inline std::string Colorize(const std::string& value,
                            const std::string& ansiCode,
                            bool useColor)
{
    if (!useColor)
        return value;
    return "\033[" + ansiCode + "m" + value + "\033[0m";
}

inline std::string ColorForStatus(const std::string& status, bool useColor)
{
    if (status == "running")
        return Colorize(status, "32", useColor);
    if (status == "pending")
        return Colorize(status, "33", useColor);
    if (status == "completed" || status == "done")
        return Colorize(status, "34", useColor);
    if (status == "failed")
        return Colorize(status, "31", useColor);
    return status;
}

inline std::string OptionalLongLongText(const std::optional<long long>& value)
{
    return value.has_value() ? std::to_string(*value) : "unknown";
}

inline std::string OptionalIntText(const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "unknown";
}

inline std::string OptionalDoubleText(const std::optional<double>& value, int precision = 1)
{
    if (!value.has_value())
        return "unknown";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << *value;
    return oss.str();
}

inline std::string OptionalPercentText(const std::optional<double>& value, int precision = 1)
{
    return value.has_value() ? OptionalDoubleText(value, precision) + "%" : "unknown";
}

inline std::string OptionalMbText(const std::optional<double>& value, int precision = 0)
{
    return value.has_value() ? OptionalDoubleText(value, precision) + " MB" : "unknown";
}

inline std::string FormatPercentComplete(const SchedulerStatusJob& job)
{
    const std::optional<int> epoch = job.currentEpoch.has_value() ? job.currentEpoch : job.completedEpochs;
    if (!epoch.has_value() || job.targetEpochs <= 0)
        return "unknown";
    const double percent = std::min(100.0,
                                    100.0 * static_cast<double>(*epoch) /
                                        static_cast<double>(job.targetEpochs));
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << percent << "%";
    return oss.str();
}

inline std::string FormatDurationSeconds(double seconds)
{
    if (seconds < 0.0 || !std::isfinite(seconds))
        return "unknown";
    long long total = static_cast<long long>(std::llround(seconds));
    const long long days = total / 86400;
    total %= 86400;
    const long long hours = total / 3600;
    total %= 3600;
    const long long minutes = total / 60;
    const long long secs = total % 60;

    std::ostringstream oss;
    if (days > 0)
        oss << days << "d ";
    if (hours > 0 || days > 0)
        oss << hours << "h ";
    if (minutes > 0 || hours > 0 || days > 0)
        oss << minutes << "m ";
    oss << secs << "s";
    return oss.str();
}

inline std::string FormatOptionalDuration(const std::optional<double>& seconds)
{
    return seconds.has_value() ? FormatDurationSeconds(*seconds) : "unknown";
}

inline std::string EstimateEta(const SchedulerStatusJob& job)
{
    if (!job.completedEpochs.has_value() ||
        !job.elapsedSeconds.has_value() ||
        *job.completedEpochs <= 0 ||
        job.targetEpochs <= 0)
    {
        return "unknown";
    }
    if (*job.completedEpochs >= job.targetEpochs)
        return "0s";

    const double secondsPerEpoch = *job.elapsedSeconds / static_cast<double>(*job.completedEpochs);
    return FormatDurationSeconds(secondsPerEpoch * static_cast<double>(job.targetEpochs - *job.completedEpochs));
}

inline std::optional<double> EstimateEtaSeconds(const SchedulerStatusJob& job)
{
    if (!job.currentEpoch.has_value() ||
        !job.elapsedSeconds.has_value() ||
        *job.currentEpoch <= 0 ||
        job.targetEpochs <= 0 ||
        *job.currentEpoch >= job.targetEpochs)
    {
        return std::nullopt;
    }

    const double secondsPerEpoch = *job.elapsedSeconds / static_cast<double>(*job.currentEpoch);
    return secondsPerEpoch * static_cast<double>(job.targetEpochs - *job.currentEpoch);
}

inline std::string FormatProgressBar(const SchedulerStatusJob& job)
{
    constexpr int width = 20;
    if (!job.currentEpoch.has_value() || job.targetEpochs <= 0)
        return "[--------------------] unknown";

    const double clamped = std::clamp(static_cast<double>(*job.currentEpoch) /
                                          static_cast<double>(job.targetEpochs),
                                      0.0,
                                      1.0);
    const int filled = static_cast<int>(std::llround(clamped * width));
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < width; ++i)
        oss << (i < filled ? "#" : "-");
    oss << "] " << std::fixed << std::setprecision(1) << (100.0 * clamped) << "%";
    return oss.str();
}

inline std::string ReadFileTailIfExists(const std::optional<std::string>& path,
                                        std::streamoff maxBytes = 262144)
{
    if (!path.has_value() || path->empty())
        return {};

    std::ifstream in(*path, std::ios::binary);
    if (!in)
        return {};

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size <= 0)
        return {};

    const std::streamoff start = std::max<std::streamoff>(0, size - maxBytes);
    in.seekg(start, std::ios::beg);
    std::string text;
    text.resize(static_cast<size_t>(size - start));
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    return text;
}

inline std::optional<int> ExtractLastIntFromText(const std::string& text,
                                                 const std::regex& regex)
{
    std::optional<int> value;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), regex);
         it != std::sregex_iterator();
         ++it)
    {
        value = std::stoi((*it)[1].str());
    }
    return value;
}

inline std::optional<long long> ExtractLastLongLongFromText(const std::string& text,
                                                            const std::regex& regex)
{
    std::optional<long long> value;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), regex);
         it != std::sregex_iterator();
         ++it)
    {
        value = std::stoll((*it)[1].str());
    }
    return value;
}

inline std::optional<double> ExtractLastDoubleFromText(const std::string& text,
                                                       const std::regex& regex)
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

inline std::optional<std::string> ExtractLastProgressLine(const std::string& text)
{
    std::optional<std::string> value;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.find("CHECKPOINT_SAVE") != std::string::npos ||
            line.find("EPOCH_3CLASS_ACCURACY") != std::string::npos ||
            line.find("Saved model with model_id=") != std::string::npos ||
            line.find("RESUME_") != std::string::npos ||
            line.find("Overall 3-class accuracy") != std::string::npos ||
            line.find("EXPERIMENT_ANALYSIS_") != std::string::npos)
        {
            if (line.size() > 160)
                value = line.substr(0, 157) + "...";
            else
                value = line;
        }
    }
    return value;
}

inline std::optional<long long> ExtractExperimentIdFromCommand(const std::string& command)
{
    std::smatch match;
    if (std::regex_search(command, match, std::regex{R"(experiment([0-9]+))"}))
        return std::stoll(match[1].str());
    if (std::regex_search(command, match, std::regex{R"(--analyze-experiment(?:=|\s+)([0-9]+))"}))
        return std::stoll(match[1].str());
    return std::nullopt;
}

inline std::string ReadCommandOutput(const std::string& command)
{
    std::string output;
    FILE* pipe = ::popen(command.c_str(), "r");
    if (!pipe)
        return output;

    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe))
        output += buffer;
    ::pclose(pipe);
    return output;
}

inline std::optional<int> ExtractCommandIntOption(const std::string& command,
                                                  const std::string& option)
{
    const std::regex regex(option + R"((?:=|\s+)([0-9]+))");
    std::smatch match;
    if (!std::regex_search(command, match, regex))
        return std::nullopt;
    return ParsePositiveInt(option, match[1].str());
}

inline void AddResourceToAggregate(SchedulerResourceAggregate& aggregate,
                                   const SchedulerProcessResource& resource)
{
    ++aggregate.workers;
    aggregate.cpuPercent += resource.cpuPercent;
    aggregate.memPercent += resource.memPercent;
    aggregate.rssMb += resource.rssMb;
}

inline std::optional<double> ParseFirstDoubleFromText(const std::string& text)
{
    std::smatch match;
    if (!std::regex_search(text, match, std::regex{R"(([0-9]+(?:\.[0-9]+)?))"}))
        return std::nullopt;
    return std::stod(match[1].str());
}

inline std::optional<double> LoadSystemMemoryTotalMb()
{
    const std::string output = ReadCommandOutput("sysctl -n hw.memsize 2>/dev/null");
    std::smatch match;
    if (!std::regex_search(output, match, std::regex{R"(([0-9]+))"}))
        return std::nullopt;
    const double bytes = std::stod(match[1].str());
    return bytes / (1024.0 * 1024.0);
}

inline std::optional<double> ExtractVmStatPages(const std::string& text,
                                                const std::string& label)
{
    const std::regex regex(label + R"(:\s+([0-9]+)\.)");
    std::smatch match;
    if (!std::regex_search(text, match, regex))
        return std::nullopt;
    return std::stod(match[1].str());
}

inline std::optional<double> LoadSystemMemoryUsedMb()
{
    const std::string output = ReadCommandOutput("vm_stat 2>/dev/null");
    if (output.empty())
        return std::nullopt;

    double pageSize = 4096.0;
    std::smatch pageMatch;
    if (std::regex_search(output, pageMatch, std::regex{R"(page size of ([0-9]+) bytes)"}))
        pageSize = std::stod(pageMatch[1].str());

    double usedPages = 0.0;
    bool found = false;
    const std::vector<std::string> labels = {
        "Pages active",
        "Pages inactive",
        "Pages speculative",
        "Pages wired down",
        "Pages occupied by compressor"
    };
    for (const auto& label : labels)
    {
        if (const auto pages = ExtractVmStatPages(output, label))
        {
            usedPages += *pages;
            found = true;
        }
    }
    if (!found)
        return std::nullopt;
    return usedPages * pageSize / (1024.0 * 1024.0);
}

inline SchedulerStatusProcessSnapshot LoadSchedulerStatusProcessSnapshot()
{
    SchedulerStatusProcessSnapshot snapshot;
    const std::string psOutput = ReadCommandOutput("ps -axo pid=,pcpu=,pmem=,rss=,command= 2>/dev/null");
    snapshot.systemMemoryTotalMb = LoadSystemMemoryTotalMb();
    snapshot.systemMemoryUsedMb = LoadSystemMemoryUsedMb();
    if (psOutput.empty())
        return snapshot;

    snapshot.processDetectionAvailable = true;
    std::istringstream stream(psOutput);
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
            continue;
        std::istringstream lineStream(line);
        int pid = -1;
        double cpuPercent = 0.0;
        double memPercent = 0.0;
        long long rssKb = 0;
        lineStream >> pid;
        lineStream >> cpuPercent;
        lineStream >> memPercent;
        lineStream >> rssKb;
        std::string command;
        std::getline(lineStream, command);
        if (pid <= 0)
            continue;

        SchedulerProcessResource resource;
        resource.pid = pid;
        resource.cpuPercent = cpuPercent;
        resource.memPercent = memPercent;
        resource.rssMb = static_cast<double>(rssKb) / 1024.0;
        snapshot.resourcesByPid[pid] = resource;

        const bool isLstm = command.find("LSTM_Release") != std::string::npos ||
                            command.find("/LSTM ") != std::string::npos ||
                            command.find(" LSTM ") != std::string::npos;
        if (!isLstm)
            continue;

        if (command.find("--schedule-experiments") != std::string::npos)
        {
            snapshot.schedulerPids.push_back(pid);
            AddResourceToAggregate(snapshot.schedulerResources, resource);
            if (!snapshot.maxTrainProcs.has_value())
                snapshot.maxTrainProcs = ExtractCommandIntOption(command, "--max-train-procs");
            if (!snapshot.maxInferProcs.has_value())
                snapshot.maxInferProcs = ExtractCommandIntOption(command, "--max-infer-procs");
            if (!snapshot.maxAnalyzeProcs.has_value())
                snapshot.maxAnalyzeProcs = ExtractCommandIntOption(command, "--max-analyze-procs");
            if (!snapshot.schedulerPollSeconds.has_value())
                snapshot.schedulerPollSeconds = ExtractCommandIntOption(command, "--scheduler-poll-seconds");
        }
        else if (command.find("--train") != std::string::npos)
        {
            ++snapshot.trainWorkers;
            AddResourceToAggregate(snapshot.trainResources, resource);
            if (const auto experimentId = ExtractExperimentIdFromCommand(command))
                snapshot.trainPidByExperiment[*experimentId] = pid;
        }
        else if (command.find("--infer") != std::string::npos)
        {
            ++snapshot.inferWorkers;
            AddResourceToAggregate(snapshot.inferResources, resource);
            if (const auto experimentId = ExtractExperimentIdFromCommand(command))
                snapshot.inferPidByExperiment[*experimentId] = pid;
        }
        else if (command.find("--analyze-experiment") != std::string::npos ||
                 command.find("--analyze-completed-experiments") != std::string::npos)
        {
            ++snapshot.analysisWorkers;
            AddResourceToAggregate(snapshot.analysisResources, resource);
            if (const auto experimentId = ExtractExperimentIdFromCommand(command))
                snapshot.analysisPidByExperiment[*experimentId] = pid;
        }
    }
    snapshot.totalCpuPercent = snapshot.schedulerResources.cpuPercent +
                               snapshot.trainResources.cpuPercent +
                               snapshot.inferResources.cpuPercent +
                               snapshot.analysisResources.cpuPercent;
    return snapshot;
}

inline SchedulerStatusCounts LoadSchedulerStatusCounts(pqxx::work& w)
{
    SchedulerStatusCounts counts;
    pqxx::result rows = w.exec(
        "SELECT status, count(*) "
        "FROM experiment "
        "GROUP BY status;");
    for (const auto& row : rows)
    {
        const std::string status = row[0].as<std::string>();
        const int count = row[1].as<int>();
        if (status == "pending")
            counts.queued = count;
        else if (status == "running")
            counts.running = count;
        else if (status == "completed")
            counts.completed = count;
        else if (status == "failed")
            counts.failed = count;
        else if (status == "cancelled")
            counts.cancelled = count;
    }
    return counts;
}

inline SchedulerStatusJob RowToSchedulerStatusJob(const pqxx::row& row)
{
    SchedulerStatusJob job;
    job.experimentId = row[0].as<long long>();
    job.symbol = row[1].as<std::string>();
    job.predictionHorizon = row[2].as<int>();
    job.phase = row[3].as<std::string>();
    job.status = row[4].as<std::string>();
    job.targetEpochs = row[5].as<int>();
    job.checkpointInterval = row[6].as<int>();
    job.modelId = OptionalLongLongCell(row, 7);
    if (!row[8].is_null())
        job.completedEpochs = row[8].as<int>();
    if (!row[9].is_null())
        job.elapsedSeconds = row[9].as<double>();
    job.startedAt = row[10].is_null() ? "" : row[10].as<std::string>();
    job.updatedAt = row[11].is_null() ? "" : row[11].as<std::string>();
    job.completedAt = row[12].is_null() ? "" : row[12].as<std::string>();
    job.errorMessage = row[13].is_null() ? "" : row[13].as<std::string>();
    job.trainLogPath = OptionalStringCell(row, 14);
    job.inferLogPath = OptionalStringCell(row, 15);
    job.analysisLogPath = OptionalStringCell(row, 16);
    return job;
}

inline std::vector<SchedulerStatusJob> LoadSchedulerStatusJobs(pqxx::work& w,
                                                               const std::string& status,
                                                               const std::optional<std::string>& phase,
                                                               int limit,
                                                               bool newestFirst)
{
    std::ostringstream sql;
    sql << "WITH latest_analysis AS ("
        << "  SELECT experiment_id, model_id, MAX(completed_epochs) AS completed_epochs "
        << "  FROM experiment_analysis_result "
        << "  WHERE completed_epochs IS NOT NULL "
        << "  GROUP BY experiment_id, model_id"
        << "), latest_infer AS ("
        << "  SELECT model_id, MAX(completed_epochs) AS completed_epochs "
        << "  FROM inference_eval_result "
        << "  WHERE completed_epochs IS NOT NULL AND status = 'completed' "
        << "  GROUP BY model_id"
        << "), train_meta AS ("
        << "  SELECT model_id, MAX(round(value)::int) AS completed_epochs "
        << "  FROM matrix "
        << "  WHERE param_name = 'train_config_meta' AND row_idx = 0 AND col_idx = 10 "
        << "  GROUP BY model_id"
        << ") "
        << "SELECT e.experiment_id, e.symbol, e.prediction_horizon, e.phase, e.status, "
        << "e.target_epochs, e.checkpoint_interval, COALESCE(e.last_model_id, e.resume_model_id) AS model_id, "
        << "COALESCE(la.completed_epochs, li.completed_epochs, tm.completed_epochs) AS completed_epochs, "
        << "CASE WHEN e.started_at IS NULL THEN NULL "
        << "     WHEN e.completed_at IS NULL THEN EXTRACT(EPOCH FROM (now() - e.started_at)) "
        << "     ELSE EXTRACT(EPOCH FROM (e.completed_at - e.started_at)) END AS elapsed_seconds, "
        << "e.started_at::text, e.updated_at::text, e.completed_at::text, e.error_message, "
        << "e.train_log_path, e.infer_log_path, e.analysis_log_path "
        << "FROM experiment e "
        << "LEFT JOIN latest_analysis la ON la.experiment_id = e.experiment_id "
        << "LEFT JOIN latest_infer li ON li.model_id = e.last_model_id "
        << "LEFT JOIN train_meta tm ON tm.model_id = COALESCE(e.last_model_id, e.resume_model_id) "
        << "WHERE e.status = " << w.quote(status) << " ";
    if (phase.has_value())
        sql << "AND e.phase = " << w.quote(*phase) << " ";
    sql << "ORDER BY ";
    if (status == "pending")
        sql << "e.updated_at ASC, e.experiment_id ASC ";
    else if (newestFirst)
        sql << "COALESCE(e.completed_at, e.updated_at) DESC, e.experiment_id DESC ";
    else
        sql << "COALESCE(e.started_at, e.updated_at) ASC, e.experiment_id ASC ";
    sql << "LIMIT " << limit << ";";

    pqxx::result rows = w.exec(sql.str());
    std::vector<SchedulerStatusJob> jobs;
    jobs.reserve(rows.size());
    for (const auto& row : rows)
        jobs.push_back(RowToSchedulerStatusJob(row));
    return jobs;
}

inline void EnrichSchedulerStatusJobFromLogs(SchedulerStatusJob& job,
                                             const SchedulerStatusProcessSnapshot& processes)
{
    const auto assignPid = [&](const std::map<long long, int>& pids) {
        const auto it = pids.find(job.experimentId);
        if (it != pids.end())
            job.pid = it->second;
    };

    if (job.phase == "train")
        assignPid(processes.trainPidByExperiment);
    else if (job.phase == "infer")
        assignPid(processes.inferPidByExperiment);
    else if (job.phase == "analyze")
        assignPid(processes.analysisPidByExperiment);

    if (job.pid.has_value())
    {
        const auto resourceIt = processes.resourcesByPid.find(*job.pid);
        if (resourceIt != processes.resourcesByPid.end())
        {
            job.cpuPercent = resourceIt->second.cpuPercent;
            job.memPercent = resourceIt->second.memPercent;
            job.rssMb = resourceIt->second.rssMb;
        }
    }

    std::optional<std::string> logPath;
    if (job.phase == "train")
        logPath = job.trainLogPath;
    else if (job.phase == "infer")
        logPath = job.inferLogPath;
    else if (job.phase == "analyze")
        logPath = job.analysisLogPath;

    const std::string tail = ReadFileTailIfExists(logPath);
    if (!tail.empty())
    {
        job.recentProgress = ExtractLastProgressLine(tail);

        const auto resumeCompleted = ExtractLastIntFromText(
            tail,
            std::regex{R"(RESUME_COMPLETED_EPOCH=([0-9]+))"});
        const auto resumeTarget = ExtractLastIntFromText(
            tail,
            std::regex{R"(RESUME_TARGET_EPOCH=([0-9]+))"});
        const auto metaEpoch = ExtractLastIntFromText(
            tail,
            std::regex{R"(epochs_trained=([0-9]+))"});
        const auto epochKv = ExtractLastIntFromText(
            tail,
            std::regex{R"((?:^|[,[:space:]])(?:epoch|current_epoch|completed_epoch|completed_epochs)=([0-9]+))"});
        const auto checkpointEpoch = ExtractLastIntFromText(
            tail,
            std::regex{R"(CHECKPOINT_SAVE_DONE[^[:cntrl:]]*epoch=([0-9]+))"});
        const auto checkpointModel = ExtractLastLongLongFromText(
            tail,
            std::regex{R"(CHECKPOINT_SAVE_DONE[^[:cntrl:]]*model_id=([0-9]+))"});
        const auto finalModel = ExtractLastLongLongFromText(
            tail,
            std::regex{R"((?:Saved model with model_id=|RESUME_SAVED_NEW_MODEL_ID=)([0-9]+))"});
        const auto loss = ExtractLastDoubleFromText(
            tail,
            std::regex{R"((?:^|[,[:space:]])(?:loss|loss_last|weighted_loss)=([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?))"});
        const auto validationAccuracy = ExtractLastDoubleFromText(
            tail,
            std::regex{R"((?:validation_accuracy|val_accuracy|acc)=([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?))"});

        if (checkpointEpoch.has_value())
            job.lastCheckpointEpoch = checkpointEpoch;
        if (checkpointModel.has_value())
            job.lastCheckpointModelId = checkpointModel;
        else if (finalModel.has_value())
            job.lastCheckpointModelId = finalModel;
        if (loss.has_value())
            job.loss = loss;
        if (validationAccuracy.has_value())
            job.validationAccuracy = validationAccuracy;

        int currentEpoch = 0;
        if (job.completedEpochs.has_value())
            currentEpoch = std::max(currentEpoch, *job.completedEpochs);
        if (resumeCompleted.has_value())
            currentEpoch = std::max(currentEpoch, *resumeCompleted);
        if (metaEpoch.has_value())
            currentEpoch = std::max(currentEpoch, *metaEpoch);
        if (epochKv.has_value())
            currentEpoch = std::max(currentEpoch, *epochKv);
        if (checkpointEpoch.has_value())
            currentEpoch = std::max(currentEpoch, *checkpointEpoch);
        if (currentEpoch > 0)
            job.currentEpoch = currentEpoch;

        if (resumeTarget.has_value() && job.targetEpochs <= 0)
            job.targetEpochs = *resumeTarget;
    }

    if (!job.currentEpoch.has_value() && job.completedEpochs.has_value())
        job.currentEpoch = job.completedEpochs;

    if (!job.lastCheckpointEpoch.has_value() && job.completedEpochs.has_value())
        job.lastCheckpointEpoch = job.completedEpochs;
    if (!job.lastCheckpointModelId.has_value() && job.modelId.has_value())
        job.lastCheckpointModelId = job.modelId;

    if (job.checkpointInterval > 0 && job.currentEpoch.has_value() && job.targetEpochs > 0)
    {
        const int next = std::min(job.targetEpochs,
                                  ((*job.currentEpoch / job.checkpointInterval) + 1) * job.checkpointInterval);
        if (next > *job.currentEpoch)
            job.nextCheckpointEpoch = next;
    }

    job.etaSeconds = EstimateEtaSeconds(job);
}

inline void EnrichSchedulerStatusJobs(std::vector<SchedulerStatusJob>& jobs,
                                      const SchedulerStatusProcessSnapshot& processes)
{
    for (auto& job : jobs)
        EnrichSchedulerStatusJobFromLogs(job, processes);
}

inline void PrintSchedulerStatusJobMachine(const SchedulerStatusJob& job)
{
    std::cout << "SCHEDULER_STATUS_JOB"
              << ",experiment_id=" << job.experimentId
              << ",phase=" << job.phase
              << ",status=" << job.status
              << ",symbol=" << job.symbol
              << ",prediction_horizon=" << job.predictionHorizon
              << ",pid=" << OptionalIntText(job.pid)
              << ",cpu_percent=" << OptionalDoubleText(job.cpuPercent, 1)
              << ",rss_mb=" << OptionalDoubleText(job.rssMb, 0)
              << ",mem_percent=" << OptionalDoubleText(job.memPercent, 1)
              << ",current_epoch=" << OptionalIntText(job.currentEpoch)
              << ",target_epochs=" << job.targetEpochs
              << ",percent_complete=" << (job.currentEpoch.has_value() && job.targetEpochs > 0
                                               ? OptionalDoubleText(100.0 * static_cast<double>(*job.currentEpoch) /
                                                                        static_cast<double>(job.targetEpochs),
                                                                    1)
                                               : "unknown")
              << ",completed_epochs=" << OptionalIntText(job.completedEpochs)
              << ",model_id=" << OptionalLongLongText(job.modelId)
              << ",last_checkpoint_epoch=" << OptionalIntText(job.lastCheckpointEpoch)
              << ",last_checkpoint_model_id=" << OptionalLongLongText(job.lastCheckpointModelId)
              << ",next_checkpoint_epoch=" << OptionalIntText(job.nextCheckpointEpoch)
              << ",eta_seconds=" << OptionalDoubleText(job.etaSeconds, 0)
              << ",loss=" << OptionalDoubleText(job.loss, 6)
              << ",validation_accuracy=" << OptionalDoubleText(job.validationAccuracy, 4)
              << std::endl;
}

inline void PrintStatusJobTable(const std::string& title,
                                const std::vector<SchedulerStatusJob>& jobs,
                                bool useColor,
                                bool showEta,
                                bool showError)
{
    (void)showEta;
    std::cout << "\n" << title << "\n";
    if (jobs.empty())
    {
        std::cout << "  none\n";
        return;
    }

    for (const auto& job : jobs)
    {
        const bool active = job.status == "running" &&
                            (job.phase == "train" || job.phase == "infer" || job.phase == "analyze");
        if (!active)
        {
            std::cout << "  experiment_id=" << job.experimentId
                      << " symbol=" << job.symbol
                      << " H=" << job.predictionHorizon
                      << " phase=" << job.phase
                      << " status=" << ColorForStatus(job.status, useColor)
                      << " model_id=" << OptionalLongLongText(job.modelId)
                      << " completed_epochs=" << OptionalIntText(job.completedEpochs)
                      << " target_epochs=" << job.targetEpochs
                      << " percent=" << FormatPercentComplete(job)
                      << " elapsed=" << FormatOptionalDuration(job.elapsedSeconds);
            if (showError && !job.errorMessage.empty())
                std::cout << " error=" << job.errorMessage;
            std::cout << "\n";
            continue;
        }

        std::cout << "  experiment_id=" << job.experimentId
                  << " symbol=" << job.symbol
                  << " H=" << job.predictionHorizon
                  << " phase=" << job.phase
                  << " status=" << ColorForStatus(job.status, useColor)
                  << " pid=" << OptionalIntText(job.pid)
                  << "\n";
        std::cout << "    cpu=" << OptionalPercentText(job.cpuPercent)
                  << " rss=" << OptionalMbText(job.rssMb)
                  << " mem=" << OptionalPercentText(job.memPercent)
                  << "\n";
        std::cout << "    model_id=" << OptionalLongLongText(job.modelId)
                  << " current_epoch=" << OptionalIntText(job.currentEpoch)
                  << " target_epochs=" << job.targetEpochs
                  << " progress=" << FormatProgressBar(job)
                  << "\n";
        std::cout << "    elapsed=" << FormatOptionalDuration(job.elapsedSeconds)
                  << " eta=" << (job.etaSeconds.has_value() ? FormatDurationSeconds(*job.etaSeconds) : "unknown")
                  << " last_checkpoint_epoch=" << OptionalIntText(job.lastCheckpointEpoch)
                  << " last_checkpoint_model_id=" << OptionalLongLongText(job.lastCheckpointModelId)
                  << " next_checkpoint_epoch=" << OptionalIntText(job.nextCheckpointEpoch)
                  << "\n";
        std::cout << "    loss=" << OptionalDoubleText(job.loss, 6)
                  << " validation_accuracy=" << OptionalDoubleText(job.validationAccuracy, 4);
        if (job.recentProgress.has_value())
            std::cout << " recent=\"" << *job.recentProgress << "\"";
        std::cout << "\n";
    }
}

inline std::string FormatAggregateResource(const SchedulerResourceAggregate& aggregate)
{
    std::ostringstream oss;
    oss << "workers=" << aggregate.workers
        << " cpu=" << std::fixed << std::setprecision(1) << aggregate.cpuPercent << "%"
        << " rss=" << std::fixed << std::setprecision(0) << aggregate.rssMb << " MB"
        << " mem=" << std::fixed << std::setprecision(1) << aggregate.memPercent << "%";
    return oss.str();
}

inline std::vector<std::string> BuildSchedulerStatusWarnings(const SchedulerStatusProcessSnapshot& processes)
{
    std::vector<std::string> warnings;
    if (!processes.processDetectionAvailable)
        warnings.push_back("process detection unavailable");
    if (processes.schedulerPids.size() > 1)
        warnings.push_back("multiple scheduler processes detected: " + std::to_string(processes.schedulerPids.size()));
    if (processes.maxTrainProcs.has_value() && processes.trainWorkers > *processes.maxTrainProcs)
        warnings.push_back("train worker count exceeds max-train-procs");
    if (processes.maxInferProcs.has_value() && processes.inferWorkers > *processes.maxInferProcs)
        warnings.push_back("infer worker count exceeds max-infer-procs");
    if (processes.maxAnalyzeProcs.has_value() && processes.analysisWorkers > *processes.maxAnalyzeProcs)
        warnings.push_back("analysis worker count exceeds max-analyze-procs");

    if (processes.systemMemoryUsedMb.has_value() &&
        processes.systemMemoryTotalMb.has_value() &&
        *processes.systemMemoryTotalMb > 0.0)
    {
        const double fraction = *processes.systemMemoryUsedMb / *processes.systemMemoryTotalMb;
        if (fraction >= 0.90)
            warnings.push_back("system memory usage is high");
        const double workerRss = processes.trainResources.rssMb +
                                 processes.inferResources.rssMb +
                                 processes.analysisResources.rssMb;
        if (workerRss / *processes.systemMemoryTotalMb >= 0.80)
            warnings.push_back("worker RSS is high relative to system memory");
    }

    return warnings;
}

inline void PrintSchedulerResourceUsage(const SchedulerStatusProcessSnapshot& processes)
{
    std::cout << "\nRESOURCE USAGE\n";
    std::cout << "  total_cpu=" << OptionalPercentText(processes.totalCpuPercent) << "\n";
    std::cout << "  system_memory_used=" << OptionalMbText(processes.systemMemoryUsedMb)
              << " total=" << OptionalMbText(processes.systemMemoryTotalMb);
    if (processes.systemMemoryUsedMb.has_value() &&
        processes.systemMemoryTotalMb.has_value() &&
        *processes.systemMemoryTotalMb > 0.0)
    {
        const double pct = 100.0 * *processes.systemMemoryUsedMb / *processes.systemMemoryTotalMb;
        std::cout << " used_percent=" << OptionalPercentText(pct);
    }
    std::cout << "\n";
    std::cout << "  scheduler " << FormatAggregateResource(processes.schedulerResources) << "\n";
    std::cout << "  train     " << FormatAggregateResource(processes.trainResources) << "\n";
    std::cout << "  infer     " << FormatAggregateResource(processes.inferResources) << "\n";
    std::cout << "  analysis  " << FormatAggregateResource(processes.analysisResources) << "\n";
}

inline void PrintSchedulerWarnings(const std::vector<std::string>& warnings)
{
    std::cout << "\nWARNINGS\n";
    if (warnings.empty())
    {
        std::cout << "  none\n";
        return;
    }
    for (const auto& warning : warnings)
        std::cout << "  " << warning << "\n";
}

inline int PrintSchedulerStatus(const SchedulerOptions& options)
{
    const bool useColor = UseAnsiColors();
    const SchedulerStatusProcessSnapshot processes = LoadSchedulerStatusProcessSnapshot();

    pqxx::connection c{LstmDbConnectionString()};
    pqxx::work w{c};
    if (!RequireSchedulerTables(w))
        return 1;

    const QueueSnapshot queueSnapshot = LoadQueueSnapshot(w);
    const SchedulerStatusCounts counts = LoadSchedulerStatusCounts(w);
    std::vector<SchedulerStatusJob> runningTrain = LoadSchedulerStatusJobs(w, "running", "train", 50, false);
    std::vector<SchedulerStatusJob> runningInfer = LoadSchedulerStatusJobs(w, "running", "infer", 50, false);
    std::vector<SchedulerStatusJob> runningAnalyze = LoadSchedulerStatusJobs(w, "running", "analyze", 50, false);
    std::vector<SchedulerStatusJob> queued = LoadSchedulerStatusJobs(w, "pending", std::nullopt, 50, false);
    std::vector<SchedulerStatusJob> completed = LoadSchedulerStatusJobs(w, "completed", std::nullopt, 10, true);
    std::vector<SchedulerStatusJob> failed = LoadSchedulerStatusJobs(w, "failed", std::nullopt, 20, true);
    w.commit();

    EnrichSchedulerStatusJobs(runningTrain, processes);
    EnrichSchedulerStatusJobs(runningInfer, processes);
    EnrichSchedulerStatusJobs(runningAnalyze, processes);
    EnrichSchedulerStatusJobs(queued, processes);
    EnrichSchedulerStatusJobs(completed, processes);
    EnrichSchedulerStatusJobs(failed, processes);

    const bool schedulerRunning = !processes.schedulerPids.empty();
    const std::string schedulerPid =
        schedulerRunning ? std::to_string(processes.schedulerPids.front()) : "unknown";

    std::cout << "Scheduler Status\n";
    if (schedulerRunning)
    {
        std::cout << "Scheduler process: "
                  << Colorize("running", "32", useColor)
                  << " pid=" << schedulerPid;
        if (processes.schedulerPids.size() > 1)
            std::cout << " additional_pids=" << (processes.schedulerPids.size() - 1);
        std::cout << "\n";
    }
    else if (!processes.processDetectionAvailable)
    {
        std::cout << "Scheduler process: unknown (process detection unavailable)\n";
    }
    else
    {
        std::cout << Colorize("Scheduler process not detected.", "31", useColor) << "\n";
    }

    std::cout << "Poll interval: "
              << (processes.schedulerPollSeconds.has_value()
                      ? std::to_string(*processes.schedulerPollSeconds) + "s"
                      : "unknown")
              << "\n";
    std::cout << "Configured worker limits: train="
              << OptionalIntText(processes.maxTrainProcs)
              << " infer=" << OptionalIntText(processes.maxInferProcs)
              << " analyze=" << OptionalIntText(processes.maxAnalyzeProcs)
              << "\n";
    std::cout << "Detected worker processes: train=" << processes.trainWorkers
              << " infer=" << processes.inferWorkers
              << " analyze=" << processes.analysisWorkers
              << "\n";

    PrintSchedulerResourceUsage(processes);
    const std::vector<std::string> warnings = BuildSchedulerStatusWarnings(processes);
    PrintSchedulerWarnings(warnings);

    std::cout << "\nOverall Counts\n"
              << "  queued=" << counts.queued
              << " running=" << counts.running
              << " completed=" << counts.completed
              << " failed=" << counts.failed
              << " cancelled=" << counts.cancelled
              << "\n";
    std::cout << "  pending_train=" << queueSnapshot.pendingTrain
              << " pending_infer=" << queueSnapshot.pendingInfer
              << " pending_analyze=" << queueSnapshot.pendingAnalyze
              << " running_train=" << queueSnapshot.runningTrain
              << " running_infer=" << queueSnapshot.runningInfer
              << " running_analyze=" << queueSnapshot.runningAnalyze
              << "\n";

    PrintStatusJobTable("Active Training Jobs", runningTrain, useColor, true, false);
    PrintStatusJobTable("Active Inference Jobs", runningInfer, useColor, false, false);
    PrintStatusJobTable("Active Analysis Jobs", runningAnalyze, useColor, false, false);
    PrintStatusJobTable("Queued Jobs", queued, useColor, false, false);
    PrintStatusJobTable("Recent Completed Experiments", completed, useColor, false, false);
    PrintStatusJobTable("Failed Experiments Summary", failed, useColor, false, true);

    if (SchedulerStatusShouldEmitMachineRecords(options))
    {
        std::cout << "\nSCHEDULER_STATUS"
                  << ",running=" << (schedulerRunning ? "1" : "0")
                  << ",pid=" << (schedulerRunning ? schedulerPid : "unknown")
                  << ",train_workers=" << processes.trainWorkers
                  << ",infer_workers=" << processes.inferWorkers
                  << ",analysis_workers=" << processes.analysisWorkers
                  << ",poll_seconds=" << OptionalIntText(processes.schedulerPollSeconds)
                  << ",max_train_procs=" << OptionalIntText(processes.maxTrainProcs)
                  << ",max_infer_procs=" << OptionalIntText(processes.maxInferProcs)
                  << ",max_analyze_procs=" << OptionalIntText(processes.maxAnalyzeProcs)
                  << std::endl;
        std::cout << "SCHEDULER_STATUS_RESOURCE"
                  << ",total_cpu_percent=" << OptionalDoubleText(processes.totalCpuPercent, 1)
                  << ",system_memory_used_mb=" << OptionalDoubleText(processes.systemMemoryUsedMb, 0)
                  << ",system_memory_total_mb=" << OptionalDoubleText(processes.systemMemoryTotalMb, 0)
                  << ",scheduler_cpu_percent=" << OptionalDoubleText(processes.schedulerResources.cpuPercent, 1)
                  << ",scheduler_rss_mb=" << OptionalDoubleText(processes.schedulerResources.rssMb, 0)
                  << ",scheduler_mem_percent=" << OptionalDoubleText(processes.schedulerResources.memPercent, 1)
                  << ",train_cpu_percent=" << OptionalDoubleText(processes.trainResources.cpuPercent, 1)
                  << ",train_rss_mb=" << OptionalDoubleText(processes.trainResources.rssMb, 0)
                  << ",train_mem_percent=" << OptionalDoubleText(processes.trainResources.memPercent, 1)
                  << ",train_workers=" << processes.trainWorkers
                  << ",infer_cpu_percent=" << OptionalDoubleText(processes.inferResources.cpuPercent, 1)
                  << ",infer_rss_mb=" << OptionalDoubleText(processes.inferResources.rssMb, 0)
                  << ",infer_mem_percent=" << OptionalDoubleText(processes.inferResources.memPercent, 1)
                  << ",infer_workers=" << processes.inferWorkers
                  << ",analysis_cpu_percent=" << OptionalDoubleText(processes.analysisResources.cpuPercent, 1)
                  << ",analysis_rss_mb=" << OptionalDoubleText(processes.analysisResources.rssMb, 0)
                  << ",analysis_mem_percent=" << OptionalDoubleText(processes.analysisResources.memPercent, 1)
                  << ",analysis_workers=" << processes.analysisWorkers
                  << std::endl;
        std::cout << "SCHEDULER_STATUS_COUNT"
                  << ",queued=" << counts.queued
                  << ",running=" << counts.running
                  << ",completed=" << counts.completed
                  << ",failed=" << counts.failed
                  << ",cancelled=" << counts.cancelled
                  << ",pending_train=" << queueSnapshot.pendingTrain
                  << ",pending_infer=" << queueSnapshot.pendingInfer
                  << ",pending_analyze=" << queueSnapshot.pendingAnalyze
                  << ",running_train=" << queueSnapshot.runningTrain
                  << ",running_infer=" << queueSnapshot.runningInfer
                  << ",running_analyze=" << queueSnapshot.runningAnalyze
                  << std::endl;
        for (const auto& job : runningTrain)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : runningInfer)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : runningAnalyze)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : queued)
            PrintSchedulerStatusJobMachine(job);
    }

    return 0;
}

inline void PrintExperimentSchedulerHelp(const char* executable)
{
    const std::string exe = executable ? executable : "LSTM_Release";
    std::cout
        << "Usage: " << exe << " --queue-experiment --symbol=SYMBOL --prediction-horizon=N --target-epochs=N "
        << "[--threshold=VALUE] [--core-lr=VALUE] [--head-lr=VALUE] [--checkpoint-interval=N] "
        << "[--train-start=YYYY-MM-DD] [--train-end=YYYY-MM-DD] [--infer-start=YYYY-MM-DD] [--infer-end=YYYY-MM-DD]\n"
        << "Example: " << exe << " --queue-experiment --symbol=eurusdrmp --prediction-horizon=12 --target-epochs=240\n"
        << "Resume: " << exe << " --queue-experiment --resume-model-id=MODEL_ID --target-epochs=240 "
        << "[--checkpoint-interval=N] [--infer-start=YYYY-MM-DD] [--infer-end=YYYY-MM-DD]\n"
        << "Auto-resume: " << exe << " --queue-experiment --auto-resume --symbol=SYMBOL --prediction-horizon=N "
        << "--target-epochs=240 [--threshold=VALUE] [--train-start=YYYY-MM-DD] [--train-end=YYYY-MM-DD]\n"
        << "Usage: " << exe << " --queue-sweep --prediction-horizon=N --target-epochs=N "
        << "[--threshold=VALUE] [--core-lr=VALUE] [--head-lr=VALUE] [--checkpoint-interval=N]\n"
        << "Example: " << exe << " --queue-sweep --prediction-horizon=12 --target-epochs=240\n"
        << "Supported sweep symbols:";
    for (const auto& symbol : EA::SupportedSymbols::TrainingSymbols())
        std::cout << " " << symbol;
    std::cout << "\n"
        << "Usage: " << exe
        << " --enqueue-experiment --symbol=SYMBOL --prediction-horizon=N --c-next-threshold=VALUE "
        << "--core-lr-mult=VALUE --head-lr-mult=VALUE --target-epochs=N --checkpoint-interval=N "
        << "--train-start=YYYY-MM-DD --train-end=YYYY-MM-DD [--infer-start=YYYY-MM-DD --infer-end=YYYY-MM-DD] "
        << "[--resume-model-id=MODEL_ID] [--allow-duplicate-experiment]\n"
        << "Usage: " << exe
        << " --schedule-experiments [--max-train-procs=N] [--max-infer-procs=N] "
        << "[--max-analyze-procs=N] [--scheduler-poll-seconds=N] [--scheduler-once] "
        << "[--scheduler-log-dir=PATH] [--dry-run] [--recover-orphans-only]\n"
        << "Usage: " << exe
        << " --scheduler-status [--log-level=quiet|summary|diagnostic]\n"
        << "Usage: " << exe
        << " --analyze-experiment=EXPERIMENT_ID | --analyze-completed-experiments | "
        << "--print-experiment-leaderboard [--leaderboard-symbol=SYMBOL] "
        << "[--leaderboard-horizon=N] [--leaderboard-limit=N]\n"
        << "Queue exit codes: 0=created, 1=invalid_arguments, 2=database_error, 3=duplicates_only\n";
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
                  << std::endl;
        PrintExperimentSchedulerHelp(argc > 0 ? argv[0] : "LSTM_Release");
        return 1;
    }

    try
    {
        if (options.help)
        {
            PrintExperimentSchedulerHelp(argc > 0 ? argv[0] : "LSTM_Release");
            return 0;
        }
        if (options.queueExperiment || options.queueSweep)
            return QueueExperiments(options);
        if (options.enqueueExperiment)
            return EnqueueExperiment(options);
        if (options.scheduleExperiments)
            return RunScheduler(options);
        if (options.schedulerStatus)
            return PrintSchedulerStatus(options);
        if (options.analyzeExperimentId.has_value())
            return AnalyzeExperimentById(*options.analyzeExperimentId, options);
        if (options.analyzeCompletedExperiments)
            return AnalyzeCompletedExperiments();
        if (options.printLeaderboard)
            return PrintLeaderboard(options);
    }
    catch (const pqxx::failure& e)
    {
        std::cerr << "EXPERIMENT_DATABASE_ERROR"
                  << ",error=" << e.what()
                  << std::endl;
        return 2;
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
