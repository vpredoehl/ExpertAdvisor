#include "SchedulerStatusService.hpp"
#include "SchedulerOperationalReadModel.hpp"
#include "ExperimentScheduler.hpp"
#include "CanonicalSymbol.hpp"
#include "CausalSurpriseObservabilityService.hpp"
#include "CheckpointPolicy.hpp"
#include "ContinuationPolicy.hpp"
#include "ContinuationPolicyInheritance.hpp"
#include "ContinuationPolicyPersistence.hpp"
#include "Donchian20Mode.hpp"
#include "DonchianLookback.hpp"
#include "EconomicEventRepository.hpp"
#include "ExperimentCurrentOperation.hpp"
#include "ExperimentRecommendationCampaignActivationService.hpp"
#include "ExperimentRecommendationCampaignApprovalService.hpp"
#include "ExperimentRecommendationCampaignExecutionService.hpp"
#include "ExperimentRecommendationCampaignHandoffService.hpp"
#include "ExperimentRecommendationCampaignLaunchService.hpp"
#include "ExperimentRecommendationCampaignMaterializationService.hpp"
#include "ExperimentRecommendationCampaignOutcomeAssessmentService.hpp"
#include "ExperimentRecommendationCampaignPlanningService.hpp"
#include "ExperimentRecommendationCampaignProposalReviewService.hpp"
#include "ExperimentRecommendationCampaignReviewService.hpp"
#include "ExperimentRecommendationCampaignStatusService.hpp"
#include "ExperimentRecommendationConversionActivationService.hpp"
#include "ExperimentRecommendationConversionExecutionService.hpp"
#include "ExperimentRecommendationConversionProposalReviewService.hpp"
#include "ExperimentRecommendationConversionWorkflowService.hpp"
#include "ExperimentRecommendationEvaluationService.hpp"
#include "ExperimentRecommendationRankingService.hpp"
#include "ExperimentRecommendationService.hpp"
#include "FeatureAblation.hpp"
#include "FeatureAblationPairEvaluationRepository.hpp"
#include "FeatureAblationPairEvaluationService.hpp"
#include "FeatureAblationReplicationEvaluationService.hpp"
#include "FeatureWarmupScope.hpp"
#include "SchedulerOperationalObservation.hpp"
#include "InferenceProfitabilityRepository.hpp"
#include "PairedTrainingObjectiveEvaluationService.hpp"
#include "Params.hpp"
#include "PgModelIO.hpp"
#include "ProfitabilityVerificationService.hpp"
#include "RunMetadata.hpp"
#include "SchedulerChildStatus.hpp"
#include "SchedulerExecutablePath.hpp"
#include "SchedulerOwnershipRepository.hpp"
#include "SchedulerStatusProcessRecognition.hpp"
#include "SchedulerOwnershipPolicy.hpp"
#include "SupportedSymbols.hpp"
#include "TrainingObjective.hpp"
#include "WorkerLifecycleDiagnostics.hpp"
#include "SchedulerCore/CheckpointAnalysisOrchestrationService.hpp"
#include "SchedulerCore/CheckpointEvaluationService.hpp"
#include "SchedulerCore/ContinuationOrchestrationService.hpp"
#include "SchedulerCore/ExperimentTransitionService.hpp"
#include "SchedulerCore/FinalExperimentDispatchService.hpp"
#include "SchedulerCore/InferenceWorkerSelection.hpp"
#include "SchedulerCore/PostgresSchedulerRepository.hpp"
#include "SchedulerCore/ReconciliationService.hpp"
#include "SchedulerCore/SchedulerAdmissionService.hpp"
#include "SchedulerCore/SchedulerAuthorityService.hpp"
#include "SchedulerCore/SchedulerChildCompletionService.hpp"
#include "SchedulerCore/SchedulerCycleService.hpp"
#include "SchedulerCore/SchedulerDaemonCli.hpp"
#include "SchedulerCore/SchedulerEngine.hpp"
#include "SchedulerCore/SchedulerPolicy.hpp"
#include "SchedulerCore/SchedulerRuntimeContext.hpp"
#include "SchedulerCore/SchedulerSemanticAdmission.hpp"
#include "SchedulerCore/SchedulerWorkerRegistration.hpp"
#include "SchedulerCore/WorkerAttemptLifecycleService.hpp"
#include "SchedulerCore/WorkerControlService.hpp"
#include "SchedulerCore/WorkerProcessController.hpp"
#include "SchedulerCore/ProductionSchedulerRuntimeInternal.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <pqxx/pqxx>

namespace EA::SchedulerCore
{
using namespace EA::SchedulerCore::ProductionRuntimeDetail;

namespace
{
thread_local std::ostream* statusOutput = &std::cout;
thread_local std::ostream* statusError = &std::cerr;

std::ostream& StatusOutput()
{
    return *statusOutput;
}

std::ostream& StatusError()
{
    return *statusError;
}

class ScopedStatusStreams
{
public:
    ScopedStatusStreams(std::ostream& output, std::ostream& error)
        : previousOutput_{statusOutput}, previousError_{statusError}
    {
        statusOutput = &output;
        statusError = &error;
    }

    ~ScopedStatusStreams()
    {
        statusOutput = previousOutput_;
        statusError = previousError_;
    }

private:
    std::ostream* previousOutput_;
    std::ostream* previousError_;
};

bool StatusTableExists(pqxx::transaction_base& transaction,
                       const std::string& tableName)
{
    return transaction.exec(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=" +
        transaction.quote(tableName) + ");").one_row()[0].as<bool>();
}

bool StatusColumnExists(pqxx::transaction_base& transaction,
                        const std::string& tableName,
                        const std::string& columnName)
{
    return transaction.exec(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=" +
        transaction.quote(tableName) + " AND column_name=" +
        transaction.quote(columnName) + ");").one_row()[0].as<bool>();
}

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
    bool currentEpochFromTable = false;
    std::optional<int> lastCheckpointEpoch;
    std::optional<long long> lastCheckpointModelId;
    std::optional<int> nextCheckpointEpoch;
    std::optional<int> stopAfterCheckpointEpoch;
    std::optional<int> stoppedAtCheckpointEpoch;
    std::optional<long long> stoppedAtCheckpointModelId;
    std::optional<bool> opportunisticCheckpointInfer;
    std::optional<int> checkpointInferMinEpoch;
    std::optional<int> checkpointInferInterval;
    std::optional<bool> checkpointPolicyEnabled;
    std::optional<double> checkpointPolicyMinLeaderScore;
    std::optional<double> checkpointPolicyMinInferAccuracy;
    std::optional<int> checkpointPolicyTopN;
    std::optional<std::string> checkpointPolicyScope;
    std::optional<std::string> checkpointPolicyStopMode;
    std::optional<int> checkpointPolicyGraceEvals;
    std::optional<std::string> checkpointPolicyLastDecision;
    std::optional<long long> checkpointPolicyLastEvalId;
    std::optional<std::string> checkpointPolicyLastReason;
    int checkpointEvalPending = 0;
    int checkpointEvalRunning = 0;
    int checkpointEvalCompleted = 0;
    int checkpointEvalFailed = 0;
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
    std::string currentOperation;
    std::string startedAt;
    std::string updatedAt;
    std::string completedAt;
    std::string errorMessage;
    bool operatorForcedFinalInferenceRerunRequested = false;
};

struct SchedulerCheckpointStatusJob
{
    long long checkpointEvalId = -1;
    long long experimentId = -1;
    int checkpointEpoch = 0;
    long long checkpointModelId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    std::string status;
    std::string phase;
    std::optional<int> pid;
    std::optional<double> cpuPercent;
    std::optional<double> memPercent;
    std::optional<double> rssMb;
    std::string workerControlState;
    std::optional<std::string> inferLogPath;
};

struct SchedulerProcessResource
{
    int pid = -1;
    double cpuPercent = 0.0;
    double memPercent = 0.0;
    double rssMb = 0.0;
};

struct SchedulerProcessInfo
{
    int pid = -1;
    std::string command;
    SchedulerProcessResource resource;
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
    int paused = 0;
    int running = 0;
    int completed = 0;
    int failed = 0;
    int cancelled = 0;
};
struct SchedulerDetectedWorker
{
    int pid = -1;
    std::string kind;
    std::string command;
    SchedulerProcessResource resource;
    bool stopped = false;
};

struct SchedulerUnmanagedWorker
{
    int pid = -1;
    std::string kind;
    std::string reason;
    std::string command;
};

struct SchedulerStatusProcessSnapshot
{
    bool processDetectionAvailable = false;
    std::vector<SchedulerProcessInfo> processes;
    std::vector<int> schedulerPids;
    std::map<int, SchedulerProcessResource> resourcesByPid;
    std::map<long long, int> trainPidByExperiment;
    std::map<long long, int> inferPidByExperiment;
    std::map<long long, int> analysisPidByExperiment;
    std::vector<SchedulerDetectedWorker> workerProcesses;
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
    std::optional<bool> autoEvaluateContinuations;
    std::optional<bool> autoQueueContinuations;
    std::optional<bool> continuationDryRun;
    std::optional<int> continuationScanSeconds;
    std::optional<int> continuationMaxQueuesPerScan;
    std::optional<double> totalCpuPercent;
    std::optional<double> systemMemoryUsedMb;
    std::optional<double> systemMemoryTotalMb;
};

struct SchedulerWorkerAccounting
{
    int managedTrain = 0;
    int managedInfer = 0;
    int managedAnalyze = 0;
    int managedRunningTrain = 0;
    int managedRunningInfer = 0;
    int managedRunningAnalyze = 0;
    int managedPausedTrain = 0;
    int managedPausedInfer = 0;
    int managedPausedAnalyze = 0;
    int unmanagedTrain = 0;
    int unmanagedInfer = 0;
    int unmanagedAnalyze = 0;
    int identityMismatchTrain = 0;
    int identityMismatchInfer = 0;
    int identityMismatchAnalyze = 0;
    int expectedMissingTrain = 0;
    int expectedMissingInfer = 0;
    int expectedMissingAnalyze = 0;
    SchedulerResourceAggregate managedTrainResources;
    SchedulerResourceAggregate managedInferResources;
    SchedulerResourceAggregate managedAnalysisResources;
    SchedulerResourceAggregate unmanagedTrainResources;
    SchedulerResourceAggregate unmanagedInferResources;
    SchedulerResourceAggregate unmanagedAnalysisResources;
    std::vector<SchedulerUnmanagedWorker> unmanagedWorkers;
    std::vector<EA::GlobalExperimentControl::SchedulerWorkerClassification>
        workerClassifications;
};

struct SchedulerIntelligenceRecord
{
    long long experimentId = -1;
    std::optional<long long> modelId;
    std::string symbol;
    int predictionHorizon = 0;
    std::optional<double> leaderScore;
    std::optional<double> inferAccuracy;
    std::optional<double> acceptAccuracy;
    int targetEpochs = 0;
    std::optional<int> completedEpochs;
};

struct SchedulerDominatedRecord
{
    SchedulerIntelligenceRecord dominated;
    long long dominatingExperimentId = -1;
    std::optional<double> dominatingLeaderScore;
    std::optional<double> leaderScoreDelta;
};

struct SchedulerIntelligenceSnapshot
{
    std::optional<SchedulerIntelligenceRecord> overallLeader;
    std::optional<SchedulerIntelligenceRecord> recentBest24h;
    std::vector<SchedulerIntelligenceRecord> leadersBySymbol;
    std::vector<SchedulerIntelligenceRecord> leadersByHorizon;
    std::vector<SchedulerIntelligenceRecord> top5;
    std::vector<SchedulerIntelligenceRecord> worst5;
    std::vector<SchedulerDominatedRecord> dominated;
    long long completedToday = 0;
    long long failedToday = 0;
    int waitingTrain = 0;
    int waitingInfer = 0;
    int waitingAnalyze = 0;
};

std::string CurrentOperationForStatusJob(const SchedulerStatusJob& job);
bool SchedulerStatusShouldEmitMachineRecords(const SchedulerOptions& options)
{
    return options.logLevel == "summary" || options.logLevel == "diagnostic";
}

bool UseAnsiColors()
{
    const char* term = std::getenv("TERM");
    return ::isatty(STDOUT_FILENO) && term && std::string{term} != "dumb";
}

std::string Colorize(const std::string& value,
                            const std::string& ansiCode,
                            bool useColor)
{
    if (!useColor)
        return value;
    return "\033[" + ansiCode + "m" + value + "\033[0m";
}

std::string ColorForStatus(const std::string& status, bool useColor)
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

std::string OptionalLongLongText(const std::optional<long long>& value)
{
    return value.has_value() ? std::to_string(*value) : "unknown";
}

std::string OptionalIntText(const std::optional<int>& value)
{
    return value.has_value() ? std::to_string(*value) : "unknown";
}

std::string OptionalDoubleText(const std::optional<double>& value, int precision = 1)
{
    if (!value.has_value())
        return "unknown";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << *value;
    return oss.str();
}

std::string OptionalPercentText(const std::optional<double>& value, int precision = 1)
{
    return value.has_value() ? OptionalDoubleText(value, precision) + "%" : "unknown";
}

std::string OptionalMbText(const std::optional<double>& value, int precision = 0)
{
    return value.has_value() ? OptionalDoubleText(value, precision) + " MB" : "unknown";
}

std::string FormatPercentComplete(const SchedulerStatusJob& job)
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

std::string FormatDurationSeconds(double seconds)
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

std::string FormatOptionalDuration(const std::optional<double>& seconds)
{
    return seconds.has_value() ? FormatDurationSeconds(*seconds) : "unknown";
}


std::optional<double> EstimateEtaSeconds(const SchedulerStatusJob& job)
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

std::string FormatProgressBar(const SchedulerStatusJob& job)
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

std::string ReadFileTailIfExists(const std::optional<std::string>& path,
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

std::optional<int> ExtractLastIntFromText(const std::string& text,
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

std::optional<long long> ExtractLastLongLongFromText(const std::string& text,
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

std::optional<double> ExtractLastDoubleFromText(const std::string& text,
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

std::optional<std::string> ExtractLastProgressLine(const std::string& text)
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

std::optional<long long> ExtractExperimentIdFromCommand(const std::string& command)
{
    std::smatch match;
    if (std::regex_search(command, match, std::regex{R"(experiment([0-9]+))"}))
        return std::stoll(match[1].str());
    if (std::regex_search(command, match, std::regex{R"(--analyze-experiment(?:=|\s+)([0-9]+))"}))
        return std::stoll(match[1].str());
    return std::nullopt;
}

std::string ReadCommandOutput(const std::string& command)
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

std::optional<int> ExtractCommandIntOption(const std::string& command,
                                           const std::string& option)
{
    const std::regex regex(option + R"((?:=|\s+)([0-9]+))");
    std::smatch match;
    if (!std::regex_search(command, match, regex))
        return std::nullopt;
    try
    {
        const int value = std::stoi(match[1].str());
        return value > 0 ? std::optional<int>{value} : std::nullopt;
    }
    catch (...)
    {
        return std::nullopt;
    }
}

void AddResourceToAggregate(SchedulerResourceAggregate& aggregate,
                                   const SchedulerProcessResource& resource)
{
    ++aggregate.workers;
    aggregate.cpuPercent += resource.cpuPercent;
    aggregate.memPercent += resource.memPercent;
    aggregate.rssMb += resource.rssMb;
}


std::optional<double> LoadSystemMemoryTotalMb()
{
    const std::string output = ReadCommandOutput("sysctl -n hw.memsize 2>/dev/null");
    std::smatch match;
    if (!std::regex_search(output, match, std::regex{R"(([0-9]+))"}))
        return std::nullopt;
    const double bytes = std::stod(match[1].str());
    return bytes / (1024.0 * 1024.0);
}

std::optional<double> ExtractVmStatPages(const std::string& text,
                                                const std::string& label)
{
    const std::regex regex(label + R"(:\s+([0-9]+)\.)");
    std::smatch match;
    if (!std::regex_search(text, match, regex))
        return std::nullopt;
    return std::stod(match[1].str());
}

std::optional<double> LoadSystemMemoryUsedMb()
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

SchedulerStatusProcessSnapshot LoadSchedulerStatusProcessSnapshot()
{
    SchedulerStatusProcessSnapshot snapshot;
    const std::string psOutput = ReadCommandOutput(
        "ps -axo pid=,pcpu=,pmem=,rss=,state=,command= 2>/dev/null");
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
        std::string processState;
        lineStream >> pid;
        lineStream >> cpuPercent;
        lineStream >> memPercent;
        lineStream >> rssKb;
        lineStream >> processState;
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
        snapshot.processes.push_back(SchedulerProcessInfo{pid, command, resource});

        const bool isScheduler =
            IsSchedulerStatusSchedulerProcessCommand(command);
        const bool isLegacyLstm =
            SchedulerStatusCommandHasExecutableBasename(
                command, "LSTM_Release") ||
            SchedulerStatusCommandHasExecutableBasename(command, "LSTM");
        if (!isScheduler && !isLegacyLstm)
            continue;

        if (isScheduler)
        {
            snapshot.schedulerPids.push_back(pid);
            AddResourceToAggregate(snapshot.schedulerResources, resource);
            if (!snapshot.maxTrainProcs.has_value())
                snapshot.maxTrainProcs = ExtractSchedulerWorkerLimitFromCommand(
                    command, "--max-train-procs");
            if (!snapshot.maxInferProcs.has_value())
                snapshot.maxInferProcs = ExtractSchedulerWorkerLimitFromCommand(
                    command, "--max-infer-procs");
            if (!snapshot.maxAnalyzeProcs.has_value())
                snapshot.maxAnalyzeProcs = ExtractSchedulerWorkerLimitFromCommand(
                    command, "--max-analyze-procs");
            if (!snapshot.schedulerPollSeconds.has_value())
                snapshot.schedulerPollSeconds = ExtractCommandIntOption(command, "--scheduler-poll-seconds");
            if (!snapshot.autoQueueContinuations.has_value())
            {
                const bool autoQueue =
                    command.find("--auto-queue-continuations") != std::string::npos;
                const bool autoEvaluate = autoQueue ||
                    command.find("--auto-evaluate-continuations") != std::string::npos;
                snapshot.autoQueueContinuations = autoQueue;
                snapshot.autoEvaluateContinuations = autoEvaluate;
                snapshot.continuationDryRun =
                    command.find("--continuation-dry-run") != std::string::npos ||
                    command.find("--dry-run") != std::string::npos;
                snapshot.continuationScanSeconds =
                    ExtractCommandIntOption(command, "--continuation-scan-seconds").value_or(300);
                snapshot.continuationMaxQueuesPerScan =
                    ExtractCommandIntOption(command, "--continuation-max-queues-per-scan").value_or(1);
            }
        }
        else if (command.find("--train") != std::string::npos)
        {
            ++snapshot.trainWorkers;
            AddResourceToAggregate(snapshot.trainResources, resource);
            snapshot.workerProcesses.push_back(SchedulerDetectedWorker{
                pid, "train", command, resource,
                !processState.empty() &&
                    (processState[0] == 'T' || processState[0] == 't')});
            if (const auto experimentId = ExtractExperimentIdFromCommand(command))
                snapshot.trainPidByExperiment[*experimentId] = pid;
        }
        else if (command.find("--infer") != std::string::npos)
        {
            ++snapshot.inferWorkers;
            AddResourceToAggregate(snapshot.inferResources, resource);
            snapshot.workerProcesses.push_back(SchedulerDetectedWorker{
                pid, "infer", command, resource,
                !processState.empty() &&
                    (processState[0] == 'T' || processState[0] == 't')});
            if (const auto experimentId = ExtractExperimentIdFromCommand(command))
                snapshot.inferPidByExperiment[*experimentId] = pid;
        }
        else if (command.find("--analyze-experiment") != std::string::npos ||
                 command.find("--analyze-completed-experiments") != std::string::npos)
        {
            ++snapshot.analysisWorkers;
            AddResourceToAggregate(snapshot.analysisResources, resource);
            snapshot.workerProcesses.push_back(SchedulerDetectedWorker{
                pid, "analyze", command, resource,
                !processState.empty() &&
                    (processState[0] == 'T' || processState[0] == 't')});
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

bool CommandContainsOptionValue(const std::string& command,
                                const std::string& option,
                                long long value)
{
    const std::string valueText = std::to_string(value);
    return command.find(option + "=" + valueText) != std::string::npos ||
           command.find(option + " " + valueText) != std::string::npos;
}
SchedulerStatusCounts LoadSchedulerStatusCounts(pqxx::transaction_base& w)
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
        else if (status == "paused")
            counts.paused = count;
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

std::vector<EA::GlobalExperimentControl::ManagedWorker>
LoadAuthoritativeSchedulerWorkers(pqxx::transaction_base& w)
{
    pqxx::result rows = w.exec(
        "SELECT e.experiment_id,NULL::bigint AS checkpoint_eval_id,"
        "e.status,e.phase,a.worker_pid,a.worker_process_group_id,"
        "a.canonical_executable_path,a.command_line,"
        "a.worker_process_start_identity,a.worker_attempt_id,"
        "a.worker_kind,a.capacity_class,a.ownership_origin,"
        "a.lifecycle_state,"
        "a.launch_attempt_identity,"
        "(e.worker_pid IS NOT DISTINCT FROM a.worker_pid AND "
        " e.worker_process_group_id IS NOT DISTINCT FROM "
        "     a.worker_process_group_id AND "
        " e.worker_process_start_identity IS NOT DISTINCT FROM "
        "     a.worker_process_start_identity AND "
        " e.worker_executable IS NOT DISTINCT FROM "
        "     a.canonical_executable_path AND "
        " e.worker_command_line IS NOT DISTINCT FROM a.command_line) "
        "FROM experiment e "
        "JOIN experiment_scheduler_worker_attempt a "
        "ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id "
        "WHERE e.status IN ('running','paused') "
        "AND e.phase IN ('train','infer','analyze') "
        "UNION ALL "
        "SELECT COALESCE(ce.parent_experiment_id,ce.experiment_id),"
        "ce.checkpoint_eval_id,ce.status,'checkpoint_infer',a.worker_pid,"
        "a.worker_process_group_id,a.canonical_executable_path,"
        "a.command_line,a.worker_process_start_identity,"
        "a.worker_attempt_id,a.worker_kind,a.capacity_class,"
        "a.ownership_origin,a.lifecycle_state,a.launch_attempt_identity,"
        "(ce.worker_pid IS NOT DISTINCT FROM a.worker_pid AND "
        " ce.worker_process_group_id IS NOT DISTINCT FROM "
        "     a.worker_process_group_id AND "
        " ce.worker_process_start_identity IS NOT DISTINCT FROM "
        "     a.worker_process_start_identity AND "
        " ce.worker_executable IS NOT DISTINCT FROM "
        "     a.canonical_executable_path AND "
        " ce.worker_command_line IS NOT DISTINCT FROM a.command_line) "
        "FROM experiment_checkpoint_eval ce "
        "JOIN experiment_scheduler_worker_attempt a "
        "ON a.worker_attempt_id=ce.active_scheduler_worker_attempt_id "
        "WHERE ce.status='running' AND ce.phase='infer' "
        "ORDER BY 1,2 NULLS FIRST;");

    std::vector<EA::GlobalExperimentControl::ManagedWorker> workers;
    workers.reserve(rows.size());
    for (const auto& row : rows)
    {
        EA::GlobalExperimentControl::ManagedWorker worker;
        worker.experimentId = row[0].as<long long>();
        worker.checkpointEvalId = OptionalLongLongCell(row, 1);
        worker.lifecycleStatus = row[2].as<std::string>();
        worker.phase = row[3].as<std::string>();
        if (!row[4].is_null())
            worker.pid = row[4].as<int>();
        if (!row[5].is_null())
            worker.processGroupId = row[5].as<int>();
        worker.executable = OptionalStringCell(row, 6);
        worker.commandLine = OptionalStringCell(row, 7);
        worker.processStartIdentity = OptionalStringCell(row, 8);
        worker.workerAttemptId = OptionalLongLongCell(row, 9);
        worker.workerKind = row[10].as<std::string>();
        worker.capacityClass = row[11].as<std::string>();
        worker.ownershipOrigin = row[12].as<std::string>();
        worker.attemptLifecycleState = row[13].as<std::string>();
        worker.launchAttemptIdentity = row[14].as<std::string>();
        worker.authoritativeBindingMatches = row[15].as<bool>();
        workers.push_back(std::move(worker));
    }
    return workers;
}

std::vector<SchedulerCheckpointStatusJob>
LoadActiveCheckpointStatusJobs(pqxx::transaction_base& w)
{
    pqxx::result rows = w.exec(
        "SELECT ce.checkpoint_eval_id,"
        "COALESCE(ce.parent_experiment_id,ce.experiment_id),"
        "ce.checkpoint_epoch,ce.checkpoint_model_id,"
        "COALESCE(ce.symbol,e.symbol),"
        "COALESCE(ce.prediction_horizon,e.prediction_horizon),"
        "ce.status,ce.phase,ce.worker_pid,ce.worker_control_state,"
        "ce.infer_log_path "
        "FROM experiment_checkpoint_eval ce "
        "LEFT JOIN experiment e "
        "ON e.experiment_id="
        "COALESCE(ce.parent_experiment_id,ce.experiment_id) "
        "WHERE ce.status='running' AND ce.phase='infer' "
        "ORDER BY ce.created_at,ce.checkpoint_eval_id;");

    std::vector<SchedulerCheckpointStatusJob> jobs;
    jobs.reserve(rows.size());
    for (const auto& row : rows)
    {
        SchedulerCheckpointStatusJob job;
        job.checkpointEvalId = row[0].as<long long>();
        job.experimentId = row[1].as<long long>();
        job.checkpointEpoch = row[2].as<int>();
        job.checkpointModelId = row[3].as<long long>();
        job.symbol =
            row[4].is_null() ? "unknown" : row[4].as<std::string>();
        job.predictionHorizon =
            row[5].is_null() ? 0 : row[5].as<int>();
        job.status = row[6].as<std::string>();
        job.phase = row[7].as<std::string>();
        if (!row[8].is_null())
            job.pid = row[8].as<int>();
        job.workerControlState =
            row[9].is_null() ? "unknown" : row[9].as<std::string>();
        job.inferLogPath = OptionalStringCell(row, 10);
        jobs.push_back(std::move(job));
    }
    return jobs;
}

std::string OptionalLongLongText(const std::optional<long long>& value);
std::string OptionalIntText(const std::optional<int>& value);
std::string OptionalDoubleText(const std::optional<double>& value, int precision);
SchedulerStatusJob RowToSchedulerStatusJob(const pqxx::row& row)
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
    if (!row[17].is_null())
    {
        job.currentEpoch = row[17].as<int>();
        job.currentEpochFromTable = true;
    }
    if (!row[18].is_null())
        job.pid = row[18].as<int>();
    if (!row[19].is_null())
    {
        const auto currentOperation =
            EA::ExperimentLifecycle::NormalizePersistedCurrentOperation(
                row[19].as<std::string>());
        job.currentOperation = currentOperation;
    }
    if (!row[20].is_null())
        job.stopAfterCheckpointEpoch = row[20].as<int>();
    if (!row[21].is_null())
        job.stoppedAtCheckpointEpoch = row[21].as<int>();
    if (!row[22].is_null())
        job.stoppedAtCheckpointModelId = row[22].as<long long>();
    if (!row[23].is_null())
        job.opportunisticCheckpointInfer = row[23].as<bool>();
    if (!row[24].is_null())
        job.checkpointInferMinEpoch = row[24].as<int>();
    if (!row[25].is_null())
        job.checkpointInferInterval = row[25].as<int>();
    if (!row[26].is_null())
        job.checkpointPolicyEnabled = row[26].as<bool>();
    job.checkpointPolicyMinLeaderScore = OptionalDoubleCell(row, 27);
    job.checkpointPolicyMinInferAccuracy = OptionalDoubleCell(row, 28);
    if (!row[29].is_null())
        job.checkpointPolicyTopN = row[29].as<int>();
    if (!row[30].is_null())
        job.checkpointPolicyScope = row[30].as<std::string>();
    if (!row[31].is_null())
        job.checkpointPolicyStopMode = row[31].as<std::string>();
    if (!row[32].is_null())
        job.checkpointPolicyGraceEvals = row[32].as<int>();
    if (!row[33].is_null())
        job.checkpointPolicyLastDecision = row[33].as<std::string>();
    if (!row[34].is_null())
        job.checkpointPolicyLastEvalId = row[34].as<long long>();
    if (!row[35].is_null())
        job.checkpointPolicyLastReason = row[35].as<std::string>();
    job.checkpointEvalPending = row[36].as<int>();
    job.checkpointEvalRunning = row[37].as<int>();
    job.checkpointEvalCompleted = row[38].as<int>();
    job.checkpointEvalFailed = row[39].as<int>();
    job.operatorForcedFinalInferenceRerunRequested = row[40].as<bool>();
    return job;
}

std::vector<SchedulerStatusJob> LoadSchedulerStatusJobs(pqxx::transaction_base& w,
                                                               const std::string& status,
                                                               const std::optional<std::string>& phase,
                                                               int limit,
                                                               bool newestFirst)
{
    const bool hasCheckpointColumns = StatusColumnExists(w, "experiment", "stop_after_checkpoint_epoch");
    const bool hasCheckpointInferEnabled = StatusColumnExists(w, "experiment", "checkpoint_infer_enabled");
    const bool hasOpportunisticCheckpointInfer = StatusColumnExists(w, "experiment", "opportunistic_checkpoint_infer");
    const bool hasCheckpointEvalTable = StatusTableExists(w, "experiment_checkpoint_eval");
    const bool hasCheckpointPolicy = StatusColumnExists(w, "experiment", "checkpoint_policy_enabled");
    std::ostringstream sql;
    sql << "WITH latest_analysis AS ("
        << "  SELECT experiment_id, model_id, MAX(completed_epochs) AS completed_epochs "
        << "  FROM experiment_analysis_result "
        << "  WHERE completed_epochs IS NOT NULL "
        << "  AND COALESCE(analysis_scope, 'final') = 'final' "
        << "  GROUP BY experiment_id, model_id"
        << "), latest_infer AS ("
        << "  SELECT model_id, MAX(completed_epochs) AS completed_epochs "
        << "  FROM inference_eval_result "
        << "  WHERE completed_epochs IS NOT NULL AND status = 'completed' "
        << "  AND inference_scope = 'final' AND checkpoint_eval_id IS NULL "
        << "  GROUP BY model_id"
        << "), train_meta AS ("
        << "  SELECT model_id, MAX(round(value)::int) AS completed_epochs "
        << "  FROM matrix "
        << "  WHERE param_name = 'train_config_meta' AND row_idx = 0 AND col_idx = 10 "
        << "  GROUP BY model_id"
        << ")";
    if (hasCheckpointEvalTable)
    {
        sql << ", checkpoint_eval_counts AS ("
            << "  SELECT COALESCE(parent_experiment_id, experiment_id) AS parent_experiment_id, "
            << "         count(*) FILTER (WHERE status = 'pending') AS pending_count, "
            << "         count(*) FILTER (WHERE status = 'running') AS running_count, "
            << "         count(*) FILTER (WHERE status = 'completed') AS completed_count, "
            << "         count(*) FILTER (WHERE status = 'failed') AS failed_count "
            << "  FROM experiment_checkpoint_eval "
            << "  GROUP BY COALESCE(parent_experiment_id, experiment_id)"
            << ")";
    }
    sql
        << " SELECT e.experiment_id, e.symbol, e.prediction_horizon, e.phase, e.status, "
        << "e.target_epochs, e.checkpoint_interval, COALESCE(e.last_model_id, e.resume_model_id) AS model_id, "
        << "COALESCE(la.completed_epochs, li.completed_epochs, tm.completed_epochs) AS completed_epochs, "
        << "CASE WHEN e.started_at IS NULL THEN NULL "
        << "     WHEN e.completed_at IS NULL THEN EXTRACT(EPOCH FROM (now() - e.started_at)) "
        << "     ELSE EXTRACT(EPOCH FROM (e.completed_at - e.started_at)) END AS elapsed_seconds, "
        << "e.started_at::text, e.updated_at::text, e.completed_at::text, e.error_message, "
        << "e.train_log_path, e.infer_log_path, e.analysis_log_path, "
        << "e.current_epoch, e.worker_pid, e.current_operation, ";
    if (hasCheckpointColumns)
    {
        sql << "e.stop_after_checkpoint_epoch, e.stopped_at_checkpoint_epoch, "
            << "e.stopped_at_checkpoint_model_id, ";
        if (hasCheckpointInferEnabled && hasOpportunisticCheckpointInfer)
            sql << "(e.checkpoint_infer_enabled OR e.opportunistic_checkpoint_infer), ";
        else if (hasCheckpointInferEnabled)
            sql << "e.checkpoint_infer_enabled, ";
        else if (hasOpportunisticCheckpointInfer)
            sql << "e.opportunistic_checkpoint_infer, ";
        else
            sql << "NULL::boolean, ";
        sql << "e.checkpoint_infer_min_epoch, e.checkpoint_infer_interval, ";
    }
    else
    {
        sql << "NULL::integer, NULL::integer, NULL::bigint, NULL::boolean, NULL::integer, NULL::integer, ";
    }
    if (hasCheckpointPolicy)
    {
        sql << "e.checkpoint_policy_enabled, e.checkpoint_policy_min_leader_score, "
            << "e.checkpoint_policy_min_infer_accuracy, e.checkpoint_policy_top_n, "
            << "e.checkpoint_policy_scope, e.checkpoint_policy_stop_mode, "
            << "e.checkpoint_policy_grace_evals, e.checkpoint_policy_last_decision, "
            << "e.checkpoint_policy_last_checkpoint_eval_id, e.checkpoint_policy_last_reason, ";
    }
    else
    {
        sql << "NULL::boolean, NULL::double precision, NULL::double precision, NULL::integer, "
            << "NULL::text, NULL::text, NULL::integer, NULL::text, NULL::bigint, NULL::text, ";
    }
    if (hasCheckpointEvalTable)
    {
        sql << "COALESCE(cec.pending_count, 0)::int, "
            << "COALESCE(cec.running_count, 0)::int, "
            << "COALESCE(cec.completed_count, 0)::int, "
            << "COALESCE(cec.failed_count, 0)::int, ";
    }
    else
    {
        sql << "0::int, 0::int, 0::int, 0::int, ";
    }
    sql << "e.operator_forced_final_inference_rerun_requested ";
    sql
        << "FROM experiment e "
        << "LEFT JOIN latest_analysis la ON la.experiment_id = e.experiment_id "
        << "LEFT JOIN latest_infer li ON li.model_id = e.last_model_id "
        << "LEFT JOIN train_meta tm ON tm.model_id = COALESCE(e.last_model_id, e.resume_model_id) ";
    if (hasCheckpointEvalTable)
        sql << "LEFT JOIN checkpoint_eval_counts cec ON cec.parent_experiment_id = e.experiment_id ";
    sql
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

std::optional<SchedulerStatusJob> LoadSchedulerStatusJobById(pqxx::transaction_base& w, long long experimentId)
{
    const bool hasCheckpointColumns = StatusColumnExists(w, "experiment", "stop_after_checkpoint_epoch");
    const bool hasCheckpointInferEnabled = StatusColumnExists(w, "experiment", "checkpoint_infer_enabled");
    const bool hasOpportunisticCheckpointInfer = StatusColumnExists(w, "experiment", "opportunistic_checkpoint_infer");
    const bool hasCheckpointEvalTable = StatusTableExists(w, "experiment_checkpoint_eval");
    const bool hasCheckpointPolicy = StatusColumnExists(w, "experiment", "checkpoint_policy_enabled");
    std::ostringstream sql;
    sql << "WITH latest_analysis AS ("
        << "  SELECT experiment_id, model_id, MAX(completed_epochs) AS completed_epochs "
        << "  FROM experiment_analysis_result "
        << "  WHERE completed_epochs IS NOT NULL "
        << "  AND COALESCE(analysis_scope, 'final') = 'final' "
        << "  GROUP BY experiment_id, model_id"
        << "), latest_infer AS ("
        << "  SELECT model_id, MAX(completed_epochs) AS completed_epochs "
        << "  FROM inference_eval_result "
        << "  WHERE completed_epochs IS NOT NULL AND status = 'completed' "
        << "  AND inference_scope = 'final' AND checkpoint_eval_id IS NULL "
        << "  GROUP BY model_id"
        << "), train_meta AS ("
        << "  SELECT model_id, MAX(round(value)::int) AS completed_epochs "
        << "  FROM matrix "
        << "  WHERE param_name = 'train_config_meta' AND row_idx = 0 AND col_idx = 10 "
        << "  GROUP BY model_id"
        << ")";
    if (hasCheckpointEvalTable)
    {
        sql << ", checkpoint_eval_counts AS ("
            << "  SELECT COALESCE(parent_experiment_id, experiment_id) AS parent_experiment_id, "
            << "         count(*) FILTER (WHERE status = 'pending') AS pending_count, "
            << "         count(*) FILTER (WHERE status = 'running') AS running_count, "
            << "         count(*) FILTER (WHERE status = 'completed') AS completed_count, "
            << "         count(*) FILTER (WHERE status = 'failed') AS failed_count "
            << "  FROM experiment_checkpoint_eval "
            << "  GROUP BY COALESCE(parent_experiment_id, experiment_id)"
            << ")";
    }
    sql
        << " SELECT e.experiment_id, e.symbol, e.prediction_horizon, e.phase, e.status, "
        << "e.target_epochs, e.checkpoint_interval, COALESCE(e.last_model_id, e.resume_model_id) AS model_id, "
        << "COALESCE(la.completed_epochs, li.completed_epochs, tm.completed_epochs) AS completed_epochs, "
        << "CASE WHEN e.started_at IS NULL THEN NULL "
        << "     WHEN e.completed_at IS NULL THEN EXTRACT(EPOCH FROM (now() - e.started_at)) "
        << "     ELSE EXTRACT(EPOCH FROM (e.completed_at - e.started_at)) END AS elapsed_seconds, "
        << "e.started_at::text, e.updated_at::text, e.completed_at::text, e.error_message, "
        << "e.train_log_path, e.infer_log_path, e.analysis_log_path, "
        << "e.current_epoch, e.worker_pid, e.current_operation, ";
    if (hasCheckpointColumns)
    {
        sql << "e.stop_after_checkpoint_epoch, e.stopped_at_checkpoint_epoch, "
            << "e.stopped_at_checkpoint_model_id, ";
        if (hasCheckpointInferEnabled && hasOpportunisticCheckpointInfer)
            sql << "(e.checkpoint_infer_enabled OR e.opportunistic_checkpoint_infer), ";
        else if (hasCheckpointInferEnabled)
            sql << "e.checkpoint_infer_enabled, ";
        else if (hasOpportunisticCheckpointInfer)
            sql << "e.opportunistic_checkpoint_infer, ";
        else
            sql << "NULL::boolean, ";
        sql << "e.checkpoint_infer_min_epoch, e.checkpoint_infer_interval, ";
    }
    else
    {
        sql << "NULL::integer, NULL::integer, NULL::bigint, NULL::boolean, NULL::integer, NULL::integer, ";
    }
    if (hasCheckpointPolicy)
    {
        sql << "e.checkpoint_policy_enabled, e.checkpoint_policy_min_leader_score, "
            << "e.checkpoint_policy_min_infer_accuracy, e.checkpoint_policy_top_n, "
            << "e.checkpoint_policy_scope, e.checkpoint_policy_stop_mode, "
            << "e.checkpoint_policy_grace_evals, e.checkpoint_policy_last_decision, "
            << "e.checkpoint_policy_last_checkpoint_eval_id, e.checkpoint_policy_last_reason, ";
    }
    else
    {
        sql << "NULL::boolean, NULL::double precision, NULL::double precision, NULL::integer, "
            << "NULL::text, NULL::text, NULL::integer, NULL::text, NULL::bigint, NULL::text, ";
    }
    if (hasCheckpointEvalTable)
    {
        sql << "COALESCE(cec.pending_count, 0)::int, "
            << "COALESCE(cec.running_count, 0)::int, "
            << "COALESCE(cec.completed_count, 0)::int, "
            << "COALESCE(cec.failed_count, 0)::int, ";
    }
    else
    {
        sql << "0::int, 0::int, 0::int, 0::int, ";
    }
    sql << "e.operator_forced_final_inference_rerun_requested ";
    sql
        << "FROM experiment e "
        << "LEFT JOIN latest_analysis la ON la.experiment_id = e.experiment_id "
        << "LEFT JOIN latest_infer li ON li.model_id = e.last_model_id "
        << "LEFT JOIN train_meta tm ON tm.model_id = COALESCE(e.last_model_id, e.resume_model_id) ";
    if (hasCheckpointEvalTable)
        sql << "LEFT JOIN checkpoint_eval_counts cec ON cec.parent_experiment_id = e.experiment_id ";
    sql
        << "WHERE e.experiment_id = " << experimentId << " "
        << "LIMIT 1;";

    pqxx::result rows = w.exec(sql.str());
    if (rows.empty())
        return std::nullopt;
    return RowToSchedulerStatusJob(rows[0]);
}

void EnrichSchedulerStatusJobFromLogs(SchedulerStatusJob& job,
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
    {
        assignPid(processes.inferPidByExperiment);
        if (!job.pid.has_value() && job.modelId.has_value())
        {
            for (const auto& process : processes.workerProcesses)
            {
                if (process.kind == "infer" &&
                    process.command.find(
                        "--scheduler-checkpoint-eval-id") ==
                        std::string::npos &&
                    CommandContainsOptionValue(process.command, "--model", *job.modelId))
                {
                    job.pid = process.pid;
                    break;
                }
            }
        }
    }
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

        if (!job.currentEpochFromTable)
        {
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
        }

        if (resumeTarget.has_value() && job.targetEpochs <= 0)
            job.targetEpochs = *resumeTarget;
    }

    if (!job.currentEpochFromTable && !job.currentEpoch.has_value() && job.completedEpochs.has_value())
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

void EnrichSchedulerStatusJobs(std::vector<SchedulerStatusJob>& jobs,
                                      const SchedulerStatusProcessSnapshot& processes)
{
    for (auto& job : jobs)
        EnrichSchedulerStatusJobFromLogs(job, processes);
}

void EnrichCheckpointStatusJobs(
    std::vector<SchedulerCheckpointStatusJob>& jobs,
    const SchedulerStatusProcessSnapshot& processes)
{
    for (auto& job : jobs)
    {
        if (!job.pid)
            continue;
        const auto resource = processes.resourcesByPid.find(*job.pid);
        if (resource == processes.resourcesByPid.end())
            continue;
        job.cpuPercent = resource->second.cpuPercent;
        job.memPercent = resource->second.memPercent;
        job.rssMb = resource->second.rssMb;
    }
}

std::string TruncateCommandForStatus(const std::string& command)
{
    constexpr size_t kMaxCommandLength = 220;
    if (command.size() <= kMaxCommandLength)
        return command;
    return command.substr(0, kMaxCommandLength - 3) + "...";
}

SchedulerWorkerAccounting ComputeSchedulerWorkerAccounting(
    const SchedulerStatusProcessSnapshot& processes,
    const std::vector<EA::GlobalExperimentControl::ManagedWorker>&
        authoritativeWorkers,
    EA::GlobalExperimentControl::ProcessObserver& processObserver)
{
    std::vector<EA::GlobalExperimentControl::SchedulerWorkerCandidate>
        candidates;
    candidates.reserve(processes.workerProcesses.size());
    for (const auto& process : processes.workerProcesses)
    {
        candidates.push_back(
            EA::GlobalExperimentControl::SchedulerWorkerCandidate{
                process.pid,
                process.kind,
                process.command,
                process.resource.cpuPercent,
                process.resource.memPercent,
                process.resource.rssMb,
                process.stopped});
    }
    const auto classifications =
        EA::GlobalExperimentControl::ClassifySchedulerWorkers(
            candidates, authoritativeWorkers, processObserver);
    const auto summary =
        EA::GlobalExperimentControl::SummarizeSchedulerWorkers(
            classifications);

    SchedulerWorkerAccounting accounting;
    const auto copyAggregate = [](
        const EA::GlobalExperimentControl::SchedulerWorkerAggregate& from,
        SchedulerResourceAggregate& to) {
        to.workers = from.workers;
        to.cpuPercent = from.cpuPercent;
        to.memPercent = from.memPercent;
        to.rssMb = from.rssMb;
    };
    accounting.managedTrain = summary.managedTrain.workers;
    accounting.managedInfer = summary.managedInfer.workers;
    accounting.managedAnalyze = summary.managedAnalyze.workers;
    accounting.managedRunningTrain = summary.managedRunningTrain.workers;
    accounting.managedRunningInfer = summary.managedRunningInfer.workers;
    accounting.managedRunningAnalyze = summary.managedRunningAnalyze.workers;
    accounting.managedPausedTrain = summary.managedPausedTrain.workers;
    accounting.managedPausedInfer = summary.managedPausedInfer.workers;
    accounting.managedPausedAnalyze = summary.managedPausedAnalyze.workers;
    accounting.unmanagedTrain = summary.unmanagedTrain.workers;
    accounting.unmanagedInfer = summary.unmanagedInfer.workers;
    accounting.unmanagedAnalyze = summary.unmanagedAnalyze.workers;
    accounting.identityMismatchTrain = summary.identityMismatchTrain.workers;
    accounting.identityMismatchInfer = summary.identityMismatchInfer.workers;
    accounting.identityMismatchAnalyze = summary.identityMismatchAnalyze.workers;
    accounting.expectedMissingTrain = summary.expectedMissingTrain.workers;
    accounting.expectedMissingInfer = summary.expectedMissingInfer.workers;
    accounting.expectedMissingAnalyze = summary.expectedMissingAnalyze.workers;
    copyAggregate(
        summary.managedTrain, accounting.managedTrainResources);
    copyAggregate(
        summary.managedInfer, accounting.managedInferResources);
    copyAggregate(
        summary.managedAnalyze, accounting.managedAnalysisResources);
    copyAggregate(
        summary.unmanagedTrain, accounting.unmanagedTrainResources);
    copyAggregate(
        summary.unmanagedInfer, accounting.unmanagedInferResources);
    copyAggregate(
        summary.unmanagedAnalyze, accounting.unmanagedAnalysisResources);

    accounting.workerClassifications = classifications;
    for (size_t i = 0; i < processes.workerProcesses.size(); ++i)
    {
        const auto& process = processes.workerProcesses[i];
        const auto& classification = classifications[i];
        if (classification.managed || classification.authoritative)
            continue;
        accounting.unmanagedWorkers.push_back(SchedulerUnmanagedWorker{
            process.pid,
            process.kind,
            classification.reason,
            TruncateCommandForStatus(process.command)
        });
    }
    return accounting;
}

void PrintCheckpointStatusJobs(
    const std::vector<SchedulerCheckpointStatusJob>& jobs)
{
    StatusOutput() << "\nActive Checkpoint Inference Jobs\n";
    if (jobs.empty())
    {
        StatusOutput() << "  none\n";
        return;
    }
    for (const auto& job : jobs)
    {
        StatusOutput() << "  checkpoint_eval_id=" << job.checkpointEvalId
                  << " experiment_id=" << job.experimentId
                  << " epoch=" << job.checkpointEpoch
                  << " model_id=" << job.checkpointModelId
                  << " symbol=" << job.symbol
                  << " horizon=" << job.predictionHorizon
                  << " pid=" << OptionalIntText(job.pid)
                  << " control=" << job.workerControlState
                  << " cpu=" << OptionalDoubleText(job.cpuPercent, 1)
                  << "% rss_mb=" << OptionalDoubleText(job.rssMb, 0)
                  << " log=" << job.inferLogPath.value_or("none")
                  << "\n";
    }
}

void PrintCheckpointStatusJobMachine(
    const SchedulerCheckpointStatusJob& job)
{
    StatusOutput() << "SCHEDULER_STATUS_CHECKPOINT_JOB"
              << ",checkpoint_eval_id=" << job.checkpointEvalId
              << ",experiment_id=" << job.experimentId
              << ",checkpoint_epoch=" << job.checkpointEpoch
              << ",checkpoint_model_id=" << job.checkpointModelId
              << ",status=" << job.status
              << ",phase=" << job.phase
              << ",symbol=" << job.symbol
              << ",prediction_horizon=" << job.predictionHorizon
              << ",pid=" << OptionalIntText(job.pid)
              << ",worker_control_state=" << job.workerControlState
              << ",cpu_percent="
              << OptionalDoubleText(job.cpuPercent, 1)
              << ",rss_mb=" << OptionalDoubleText(job.rssMb, 0)
              << ",mem_percent="
              << OptionalDoubleText(job.memPercent, 1)
              << ",infer_log_path="
              << job.inferLogPath.value_or("NULL")
              << std::endl;
}

void PrintSchedulerStatusJobMachine(const SchedulerStatusJob& job)
{
    StatusOutput() << "SCHEDULER_STATUS_JOB"
              << ",experiment_id=" << job.experimentId
              << ",phase=" << job.phase
              << ",status=" << job.status
              << ",current_operation=" << CurrentOperationForStatusJob(job)
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
              << ",stop_after_checkpoint_epoch=" << OptionalIntText(job.stopAfterCheckpointEpoch)
              << ",stopped_at_checkpoint_epoch=" << OptionalIntText(job.stoppedAtCheckpointEpoch)
              << ",stopped_at_checkpoint_model_id=" << OptionalLongLongText(job.stoppedAtCheckpointModelId)
              << ",opportunistic_checkpoint_infer="
              << (job.opportunisticCheckpointInfer.has_value() ? (*job.opportunisticCheckpointInfer ? "1" : "0") : "unknown")
              << ",checkpoint_infer_min_epoch=" << OptionalIntText(job.checkpointInferMinEpoch)
              << ",checkpoint_infer_interval=" << OptionalIntText(job.checkpointInferInterval)
              << ",checkpoint_policy="
              << (job.checkpointPolicyEnabled.has_value() ? (*job.checkpointPolicyEnabled ? "1" : "0") : "unknown")
              << ",checkpoint_policy_min_leader_score=" << OptionalDoubleText(job.checkpointPolicyMinLeaderScore, 6)
              << ",checkpoint_policy_min_infer_accuracy=" << OptionalDoubleText(job.checkpointPolicyMinInferAccuracy, 6)
              << ",checkpoint_policy_top_n=" << OptionalIntText(job.checkpointPolicyTopN)
              << ",checkpoint_policy_scope=" << (job.checkpointPolicyScope.has_value() ? *job.checkpointPolicyScope : "unknown")
              << ",checkpoint_policy_stop_mode=" << (job.checkpointPolicyStopMode.has_value() ? *job.checkpointPolicyStopMode : "unknown")
              << ",checkpoint_policy_grace_evals=" << OptionalIntText(job.checkpointPolicyGraceEvals)
              << ",checkpoint_policy_last_decision=" << (job.checkpointPolicyLastDecision.has_value() ? *job.checkpointPolicyLastDecision : "unknown")
              << ",checkpoint_policy_last_eval_id=" << OptionalLongLongText(job.checkpointPolicyLastEvalId)
              << ",checkpoint_policy_last_reason=" << (job.checkpointPolicyLastReason.has_value() ? *job.checkpointPolicyLastReason : "unknown")
              << ",checkpoint_eval_pending=" << job.checkpointEvalPending
              << ",checkpoint_eval_running=" << job.checkpointEvalRunning
              << ",checkpoint_eval_completed=" << job.checkpointEvalCompleted
              << ",checkpoint_eval_failed=" << job.checkpointEvalFailed
              << ",operator_forced_final_inference_rerun_requested="
              << (job.operatorForcedFinalInferenceRerunRequested ? "1" : "0")
              << ",eta_seconds=" << OptionalDoubleText(job.etaSeconds, 0)
              << ",loss=" << OptionalDoubleText(job.loss, 6)
              << ",validation_accuracy=" << OptionalDoubleText(job.validationAccuracy, 4)
              << std::endl;
}

void PrintStatusJobTable(const std::string& title,
                                const std::vector<SchedulerStatusJob>& jobs,
                                bool useColor,
                                bool showEta,
                                bool showError)
{
    (void)showEta;
    StatusOutput() << "\n" << title << "\n";
    if (jobs.empty())
    {
        StatusOutput() << "  none\n";
        return;
    }

    for (const auto& job : jobs)
    {
        const bool active = job.status == "running" &&
                            (job.phase == "train" || job.phase == "infer" || job.phase == "analyze");
        if (!active)
        {
            StatusOutput() << "  experiment_id=" << job.experimentId
                      << " symbol=" << job.symbol
                      << " H=" << job.predictionHorizon
                      << " phase=" << job.phase
                      << " status=" << ColorForStatus(job.status, useColor)
                      << " model_id=" << OptionalLongLongText(job.modelId)
                      << " completed_epochs=" << OptionalIntText(job.completedEpochs)
                      << " target_epochs=" << job.targetEpochs
                      << " percent=" << FormatPercentComplete(job)
                      << " elapsed=" << FormatOptionalDuration(job.elapsedSeconds)
                      << " forced_final_infer_rerun="
                      << (job.operatorForcedFinalInferenceRerunRequested
                              ? "requested"
                              : "none");
            if (job.stopAfterCheckpointEpoch.has_value())
                StatusOutput() << " stop_after_checkpoint=" << *job.stopAfterCheckpointEpoch;
            if (job.stoppedAtCheckpointEpoch.has_value())
                StatusOutput() << " stopped_checkpoint=" << *job.stoppedAtCheckpointEpoch
                          << "/" << OptionalLongLongText(job.stoppedAtCheckpointModelId);
            if (job.opportunisticCheckpointInfer.has_value() && *job.opportunisticCheckpointInfer)
                StatusOutput() << " checkpoint_eval="
                          << job.checkpointEvalPending << "/"
                          << job.checkpointEvalRunning << "/"
                          << job.checkpointEvalCompleted << "/"
                          << job.checkpointEvalFailed;
            if (job.checkpointPolicyEnabled.has_value() && *job.checkpointPolicyEnabled)
                StatusOutput() << " checkpoint_policy=enabled"
                          << " last_decision="
                          << (job.checkpointPolicyLastDecision.has_value() ? *job.checkpointPolicyLastDecision : "unknown");
            if (showError && !job.errorMessage.empty())
                StatusOutput() << " error=" << job.errorMessage;
            StatusOutput() << "\n";
            continue;
        }

        StatusOutput() << "  experiment_id=" << job.experimentId
                  << " symbol=" << job.symbol
                  << " H=" << job.predictionHorizon
                  << " phase=" << job.phase
                  << " status=" << ColorForStatus(job.status, useColor)
                  << " pid=" << OptionalIntText(job.pid)
                  << " forced_final_infer_rerun="
                  << (job.operatorForcedFinalInferenceRerunRequested
                          ? "requested"
                          : "none")
                  << "\n";
        StatusOutput() << "    cpu=" << OptionalPercentText(job.cpuPercent)
                  << " rss=" << OptionalMbText(job.rssMb)
                  << " mem=" << OptionalPercentText(job.memPercent)
                  << "\n";
        StatusOutput() << "    model_id=" << OptionalLongLongText(job.modelId)
                  << " current_epoch=" << OptionalIntText(job.currentEpoch)
                  << " target_epochs=" << job.targetEpochs
                  << " progress=" << FormatProgressBar(job)
                  << "\n";
        StatusOutput() << "    elapsed=" << FormatOptionalDuration(job.elapsedSeconds)
                  << " eta=" << (job.etaSeconds.has_value() ? FormatDurationSeconds(*job.etaSeconds) : "unknown")
                  << " last_checkpoint_epoch=" << OptionalIntText(job.lastCheckpointEpoch)
                  << " last_checkpoint_model_id=" << OptionalLongLongText(job.lastCheckpointModelId)
                  << " next_checkpoint_epoch=" << OptionalIntText(job.nextCheckpointEpoch)
                  << "\n";
        StatusOutput() << "    loss=" << OptionalDoubleText(job.loss, 6)
                  << " validation_accuracy=" << OptionalDoubleText(job.validationAccuracy, 4);
        if (job.recentProgress.has_value())
            StatusOutput() << " recent=\"" << *job.recentProgress << "\"";
        StatusOutput() << "\n";
        if (job.stopAfterCheckpointEpoch.has_value() ||
            job.stoppedAtCheckpointEpoch.has_value() ||
            (job.opportunisticCheckpointInfer.has_value() && *job.opportunisticCheckpointInfer) ||
            (job.checkpointPolicyEnabled.has_value() && *job.checkpointPolicyEnabled))
        {
            StatusOutput() << "    checkpoint_stop_after=" << OptionalIntText(job.stopAfterCheckpointEpoch)
                      << " stopped_epoch=" << OptionalIntText(job.stoppedAtCheckpointEpoch)
                      << " stopped_model_id=" << OptionalLongLongText(job.stoppedAtCheckpointModelId)
                      << " checkpoint_infer="
                      << (job.opportunisticCheckpointInfer.has_value() ? (*job.opportunisticCheckpointInfer ? "enabled" : "disabled") : "unknown")
                      << " min_epoch=" << OptionalIntText(job.checkpointInferMinEpoch)
                      << " interval=" << OptionalIntText(job.checkpointInferInterval)
                      << " evals=pending:" << job.checkpointEvalPending
                      << ",running:" << job.checkpointEvalRunning
                      << ",completed:" << job.checkpointEvalCompleted
                      << ",failed:" << job.checkpointEvalFailed
                      << "\n";
            StatusOutput() << "    checkpoint_policy="
                      << (job.checkpointPolicyEnabled.has_value() ? (*job.checkpointPolicyEnabled ? "enabled" : "disabled") : "unknown")
                      << " rules=leader>=" << OptionalDoubleText(job.checkpointPolicyMinLeaderScore, 6)
                      << ",infer>=" << OptionalDoubleText(job.checkpointPolicyMinInferAccuracy, 6)
                      << ",top_n=" << OptionalIntText(job.checkpointPolicyTopN)
                      << ",scope=" << (job.checkpointPolicyScope.has_value() ? *job.checkpointPolicyScope : "unknown")
                      << ",stop_mode=" << (job.checkpointPolicyStopMode.has_value() ? *job.checkpointPolicyStopMode : "unknown")
                      << ",grace_evals=" << OptionalIntText(job.checkpointPolicyGraceEvals)
                      << " last=decision=" << (job.checkpointPolicyLastDecision.has_value() ? *job.checkpointPolicyLastDecision : "unknown")
                      << ",checkpoint_eval_id=" << OptionalLongLongText(job.checkpointPolicyLastEvalId)
                      << ",reason=" << (job.checkpointPolicyLastReason.has_value() ? *job.checkpointPolicyLastReason : "unknown")
                      << "\n";
        }
    }
}

void PrintCompactStatusField(const std::string& label, const std::string& value)
{
    StatusOutput() << std::left << std::setw(24) << (label + ":") << value << "\n";
}

std::string CurrentOperationForStatusJob(const SchedulerStatusJob& job)
{
    const std::optional<std::string> currentOperation =
        job.currentOperation.empty()
            ? std::nullopt
            : std::optional<std::string>{job.currentOperation};
    return EA::ExperimentLifecycle::CurrentOperationForStatus(
        currentOperation,
        job.phase);
}

void PrintCompactStatusJob(const SchedulerStatusJob& job)
{
    StatusOutput() << "\nExperiment " << job.experimentId << "\n"
              << "---------------\n";
    PrintCompactStatusField("Status", job.status);
    PrintCompactStatusField("Phase", job.phase);
    PrintCompactStatusField("Symbol", job.symbol);
    PrintCompactStatusField("Horizon", std::to_string(job.predictionHorizon));
    PrintCompactStatusField("Model ID", OptionalLongLongText(job.modelId));
    PrintCompactStatusField("Epoch", OptionalIntText(job.currentEpoch) + " / " + std::to_string(job.targetEpochs));
    PrintCompactStatusField("Progress", FormatPercentComplete(job));
    PrintCompactStatusField("Runtime", FormatOptionalDuration(job.elapsedSeconds));
    PrintCompactStatusField("PID", OptionalIntText(job.pid));
    PrintCompactStatusField("Operation", CurrentOperationForStatusJob(job));
    PrintCompactStatusField(
        "Forced FINAL Infer Rerun",
        job.operatorForcedFinalInferenceRerunRequested
            ? "requested"
            : "none");
    if (job.lastCheckpointModelId.has_value())
        PrintCompactStatusField("Checkpoint Model", OptionalLongLongText(job.lastCheckpointModelId));
    if (job.lastCheckpointEpoch.has_value())
        PrintCompactStatusField("Checkpoint Epoch", OptionalIntText(job.lastCheckpointEpoch));
    if (job.stopAfterCheckpointEpoch.has_value())
        PrintCompactStatusField("Stop After Checkpoint", OptionalIntText(job.stopAfterCheckpointEpoch));
    if (job.stoppedAtCheckpointEpoch.has_value())
    {
        PrintCompactStatusField("Stopped Checkpoint", OptionalIntText(job.stoppedAtCheckpointEpoch));
        PrintCompactStatusField("Stopped Model ID", OptionalLongLongText(job.stoppedAtCheckpointModelId));
    }
    if (job.opportunisticCheckpointInfer.has_value())
    {
        PrintCompactStatusField("Checkpoint Infer",
                                *job.opportunisticCheckpointInfer ? "enabled" : "disabled");
        PrintCompactStatusField("Checkpoint Infer Min", OptionalIntText(job.checkpointInferMinEpoch));
        PrintCompactStatusField("Checkpoint Infer Interval", OptionalIntText(job.checkpointInferInterval));
        PrintCompactStatusField("Checkpoint Evals",
                                "pending=" + std::to_string(job.checkpointEvalPending) +
                                " running=" + std::to_string(job.checkpointEvalRunning) +
                                " completed=" + std::to_string(job.checkpointEvalCompleted) +
                                " failed=" + std::to_string(job.checkpointEvalFailed));
    }
    if (job.checkpointPolicyEnabled.has_value())
    {
        PrintCompactStatusField("Checkpoint Policy",
                                *job.checkpointPolicyEnabled ? "enabled" : "disabled");
        PrintCompactStatusField("Checkpoint Policy Rules",
                                "leader>=" + OptionalDoubleText(job.checkpointPolicyMinLeaderScore, 6) +
                                " infer>=" + OptionalDoubleText(job.checkpointPolicyMinInferAccuracy, 6) +
                                " top_n=" + OptionalIntText(job.checkpointPolicyTopN) +
                                " scope=" + (job.checkpointPolicyScope.has_value() ? *job.checkpointPolicyScope : "unknown"));
        PrintCompactStatusField("Checkpoint Policy Last",
                                "decision=" + (job.checkpointPolicyLastDecision.has_value() ? *job.checkpointPolicyLastDecision : "unknown") +
                                " checkpoint_eval_id=" + OptionalLongLongText(job.checkpointPolicyLastEvalId) +
                                " reason=" + (job.checkpointPolicyLastReason.has_value() ? *job.checkpointPolicyLastReason : "unknown"));
    }
    if (!job.errorMessage.empty())
        PrintCompactStatusField("Error", job.errorMessage);
}

int PrintCompactExperimentStatusFromTransaction(
    const SchedulerOptions& options,
    pqxx::transaction_base& w)
{
    const SchedulerStatusProcessSnapshot processes = LoadSchedulerStatusProcessSnapshot();

    std::vector<SchedulerStatusJob> jobs;
    if (options.statusExperimentId.has_value())
    {
        auto job = LoadSchedulerStatusJobById(w, *options.statusExperimentId);
        if (!job.has_value())
        {
            StatusError() << "ERROR: experiment not found." << std::endl;
            return 1;
        }
        jobs.push_back(*job);
    }
    else
    {
        std::vector<SchedulerStatusJob> train = LoadSchedulerStatusJobs(w, "running", "train", 500, false);
        std::vector<SchedulerStatusJob> infer = LoadSchedulerStatusJobs(w, "running", "infer", 500, false);
        std::vector<SchedulerStatusJob> analyze = LoadSchedulerStatusJobs(w, "running", "analyze", 500, false);
        jobs.reserve(train.size() + infer.size() + analyze.size());
        jobs.insert(jobs.end(), train.begin(), train.end());
        jobs.insert(jobs.end(), infer.begin(), infer.end());
        jobs.insert(jobs.end(), analyze.begin(), analyze.end());
    }

    EnrichSchedulerStatusJobs(jobs, processes);

    if (!options.statusExperimentId.has_value() && jobs.empty())
    {
        StatusOutput() << "No running experiments." << std::endl;
        return 0;
    }

    StatusOutput() << (options.statusExperimentId.has_value() ? "EXPERIMENT STATUS" : "RUNNING EXPERIMENTS") << "\n";
    for (const auto& job : jobs)
        PrintCompactStatusJob(job);
    return 0;
}

int PrintCompactExperimentStatusImpl(const SchedulerOptions& options)
{
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadOnly(transaction);
    if (!RequireSchedulerTables(transaction))
        return 1;

    const int result =
        PrintCompactExperimentStatusFromTransaction(options, transaction);
    transaction.commit();
    return result;
}

std::string FormatAggregateResource(const SchedulerResourceAggregate& aggregate)
{
    std::ostringstream oss;
    oss << "workers=" << aggregate.workers
        << " cpu=" << std::fixed << std::setprecision(1) << aggregate.cpuPercent << "%"
        << " rss=" << std::fixed << std::setprecision(0) << aggregate.rssMb << " MB"
        << " mem=" << std::fixed << std::setprecision(1) << aggregate.memPercent << "%";
    return oss.str();
}

std::vector<std::string> BuildSchedulerStatusWarnings(const SchedulerStatusProcessSnapshot& processes,
                                                      const SchedulerWorkerAccounting& accounting)
{
    std::vector<std::string> warnings;
    if (!processes.processDetectionAvailable)
        warnings.push_back("process detection unavailable");
    if (processes.schedulerPids.size() > 1)
        warnings.push_back("multiple scheduler processes detected: " + std::to_string(processes.schedulerPids.size()));
    if (processes.maxTrainProcs.has_value() &&
        SchedulerWorkerCapacityExceeded(
            *processes.maxTrainProcs, accounting.managedTrain))
        warnings.push_back("train worker count exceeds max-train-procs");
    if (processes.maxInferProcs.has_value() &&
        SchedulerWorkerCapacityExceeded(
            *processes.maxInferProcs, accounting.managedInfer))
        warnings.push_back("infer worker count exceeds max-infer-procs");
    if (processes.maxAnalyzeProcs.has_value() &&
        SchedulerWorkerCapacityExceeded(
            *processes.maxAnalyzeProcs, accounting.managedAnalyze))
        warnings.push_back("analysis worker count exceeds max-analyze-procs");
    if (!accounting.unmanagedWorkers.empty())
        warnings.push_back("unmanaged LSTM_Release worker processes detected: " + std::to_string(accounting.unmanagedWorkers.size()));
    const int identityMismatches = accounting.identityMismatchTrain +
        accounting.identityMismatchInfer + accounting.identityMismatchAnalyze;
    if (identityMismatches > 0)
        warnings.push_back(
            "authoritative worker identity mismatches detected: " +
            std::to_string(identityMismatches));
    const int expectedMissing = accounting.expectedMissingTrain +
        accounting.expectedMissingInfer + accounting.expectedMissingAnalyze;
    if (expectedMissing > 0)
        warnings.push_back(
            "expected worker processes missing: " +
            std::to_string(expectedMissing));

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

void PrintSchedulerResourceUsage(const SchedulerStatusProcessSnapshot& processes,
                                 const SchedulerWorkerAccounting& accounting)
{
    StatusOutput() << "\nRESOURCE USAGE\n";
    StatusOutput() << "  total_cpu=" << OptionalPercentText(processes.totalCpuPercent) << "\n";
    StatusOutput() << "  system_memory_used=" << OptionalMbText(processes.systemMemoryUsedMb)
              << " total=" << OptionalMbText(processes.systemMemoryTotalMb);
    if (processes.systemMemoryUsedMb.has_value() &&
        processes.systemMemoryTotalMb.has_value() &&
        *processes.systemMemoryTotalMb > 0.0)
    {
        const double pct = 100.0 * *processes.systemMemoryUsedMb / *processes.systemMemoryTotalMb;
        StatusOutput() << " used_percent=" << OptionalPercentText(pct);
    }
    StatusOutput() << "\n";
    StatusOutput() << "  scheduler " << FormatAggregateResource(processes.schedulerResources) << "\n";
    StatusOutput() << "  managed train     " << FormatAggregateResource(accounting.managedTrainResources) << "\n";
    StatusOutput() << "  managed infer     " << FormatAggregateResource(accounting.managedInferResources) << "\n";
    StatusOutput() << "  managed analysis  " << FormatAggregateResource(accounting.managedAnalysisResources) << "\n";
    StatusOutput() << "  unmanaged train   " << FormatAggregateResource(accounting.unmanagedTrainResources) << "\n";
    StatusOutput() << "  unmanaged infer   " << FormatAggregateResource(accounting.unmanagedInferResources) << "\n";
    StatusOutput() << "  unmanaged analysis " << FormatAggregateResource(accounting.unmanagedAnalysisResources) << "\n";
}

void PrintUnmanagedWorkers(const SchedulerWorkerAccounting& accounting)
{
    StatusOutput() << "\nUNMANAGED LSTM PROCESSES\n";
    if (accounting.unmanagedWorkers.empty())
    {
        StatusOutput() << "  none\n";
        return;
    }
    for (const auto& worker : accounting.unmanagedWorkers)
    {
        StatusOutput() << "  pid=" << worker.pid
                  << " kind=" << worker.kind
                  << " reason=" << worker.reason
                  << " command=" << worker.command
                  << "\n";
    }
}

void PrintSchedulerWarnings(const std::vector<std::string>& warnings)
{
    StatusOutput() << "\nWARNINGS\n";
    if (warnings.empty())
    {
        StatusOutput() << "  none\n";
        return;
    }
    for (const auto& warning : warnings)
        StatusOutput() << "  " << warning << "\n";
}

SchedulerIntelligenceRecord RowToIntelligenceRecord(const pqxx::row& row)
{
    SchedulerIntelligenceRecord record;
    record.experimentId = row[0].as<long long>();
    record.modelId = OptionalLongLongCell(row, 1);
    record.symbol = row[2].is_null() ? "unknown" : row[2].as<std::string>();
    record.predictionHorizon = row[3].is_null() ? 0 : row[3].as<int>();
    record.leaderScore = OptionalDoubleCell(row, 4);
    record.inferAccuracy = OptionalDoubleCell(row, 5);
    record.acceptAccuracy = OptionalDoubleCell(row, 6);
    record.targetEpochs = row[7].is_null() ? 0 : row[7].as<int>();
    if (!row[8].is_null())
        record.completedEpochs = row[8].as<int>();
    return record;
}

std::vector<SchedulerIntelligenceRecord> RowsToIntelligenceRecords(const pqxx::result& rows)
{
    std::vector<SchedulerIntelligenceRecord> records;
    records.reserve(rows.size());
    for (const auto& row : rows)
        records.push_back(RowToIntelligenceRecord(row));
    return records;
}

std::string IntelligenceBaseSql()
{
    return
        "WITH eligible AS ("
        "  SELECT e.experiment_id, a.model_id, a.symbol, a.prediction_horizon, "
        "         a.leader_score, a.infer_accuracy, a.accept_accuracy, "
        "         a.target_epochs, a.completed_epochs, "
        "         e.completed_at, a.updated_at "
        "  FROM experiment_analysis_result a "
        "  JOIN experiment e ON e.experiment_id = a.experiment_id "
        "  WHERE e.status = 'completed' "
        "    AND a.analysis_status = 'completed' "
        "    AND COALESCE(a.analysis_scope, 'final') = 'final' "
        "    AND a.leader_score IS NOT NULL "
        "    AND a.infer_accuracy IS NOT NULL "
        ")";
}

SchedulerIntelligenceSnapshot LoadSchedulerIntelligenceSnapshot(pqxx::transaction_base& w,
                                                                       const QueueSnapshot& queueSnapshot)
{
    SchedulerIntelligenceSnapshot snapshot;
    snapshot.waitingTrain = queueSnapshot.pendingTrain;
    snapshot.waitingInfer = queueSnapshot.pendingInfer;
    snapshot.waitingAnalyze = queueSnapshot.pendingAnalyze;

    pqxx::result counts = w.exec(
        "SELECT "
        "COUNT(*) FILTER (WHERE status = 'completed' AND completed_at >= date_trunc('day', now())), "
        "COUNT(*) FILTER (WHERE status = 'failed' AND completed_at >= date_trunc('day', now())) "
        "FROM experiment;");
    if (!counts.empty())
    {
        snapshot.completedToday = counts[0][0].as<long long>();
        snapshot.failedToday = counts[0][1].as<long long>();
    }

    const std::string base = IntelligenceBaseSql();

    pqxx::result overall = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM eligible "
        " ORDER BY leader_score DESC NULLS LAST, infer_accuracy DESC NULLS LAST, experiment_id DESC "
        " LIMIT 1;");
    if (!overall.empty())
        snapshot.overallLeader = RowToIntelligenceRecord(overall[0]);

    pqxx::result recent = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM eligible "
        " WHERE completed_at >= now() - interval '24 hours' "
        " ORDER BY leader_score DESC NULLS LAST, infer_accuracy DESC NULLS LAST, experiment_id DESC "
        " LIMIT 1;");
    if (!recent.empty())
        snapshot.recentBest24h = RowToIntelligenceRecord(recent[0]);

    pqxx::result bySymbol = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM ("
        "   SELECT eligible.*, row_number() OVER (PARTITION BY symbol ORDER BY leader_score DESC, infer_accuracy DESC, experiment_id DESC) AS rn "
        "   FROM eligible"
        " ) ranked "
        " WHERE rn = 1 "
        " ORDER BY symbol ASC "
        " LIMIT 20;");
    snapshot.leadersBySymbol = RowsToIntelligenceRecords(bySymbol);

    pqxx::result byHorizon = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM ("
        "   SELECT eligible.*, row_number() OVER (PARTITION BY prediction_horizon ORDER BY leader_score DESC, infer_accuracy DESC, experiment_id DESC) AS rn "
        "   FROM eligible"
        " ) ranked "
        " WHERE rn = 1 "
        " ORDER BY prediction_horizon ASC "
        " LIMIT 20;");
    snapshot.leadersByHorizon = RowsToIntelligenceRecords(byHorizon);

    pqxx::result top = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM eligible "
        " ORDER BY leader_score DESC NULLS LAST, infer_accuracy DESC NULLS LAST, experiment_id DESC "
        " LIMIT 5;");
    snapshot.top5 = RowsToIntelligenceRecords(top);

    pqxx::result worst = w.exec(base +
        " SELECT experiment_id, model_id, symbol, prediction_horizon, leader_score, infer_accuracy, accept_accuracy, "
        "        target_epochs, completed_epochs "
        " FROM eligible "
        " ORDER BY leader_score ASC NULLS LAST, infer_accuracy ASC NULLS LAST, experiment_id ASC "
        " LIMIT 5;");
    snapshot.worst5 = RowsToIntelligenceRecords(worst);

    pqxx::result dominatedRows = w.exec(base +
        " SELECT d.experiment_id, d.model_id, d.symbol, d.prediction_horizon, d.leader_score, d.infer_accuracy, d.accept_accuracy, "
        "        d.target_epochs, d.completed_epochs, x.experiment_id AS dominating_experiment_id, "
        "        x.leader_score AS dominating_leader_score, (x.leader_score - d.leader_score) AS leader_score_delta "
        " FROM eligible d "
        " JOIN LATERAL ("
        "   SELECT e2.experiment_id, e2.leader_score "
        "   FROM eligible e2 "
        "   WHERE e2.symbol = d.symbol "
        "     AND e2.prediction_horizon = d.prediction_horizon "
        "     AND e2.experiment_id <> d.experiment_id "
        "     AND e2.leader_score > d.leader_score "
        "     AND e2.infer_accuracy >= d.infer_accuracy "
        "     AND (d.accept_accuracy IS NULL OR e2.accept_accuracy IS NULL OR e2.accept_accuracy >= d.accept_accuracy) "
        "   ORDER BY e2.leader_score DESC, e2.infer_accuracy DESC, e2.experiment_id DESC "
        "   LIMIT 1 "
        " ) x ON true "
        " ORDER BY leader_score_delta DESC NULLS LAST, d.leader_score ASC "
        " LIMIT 10;");
    snapshot.dominated.reserve(dominatedRows.size());
    for (const auto& row : dominatedRows)
    {
        SchedulerDominatedRecord record;
        record.dominated = RowToIntelligenceRecord(row);
        record.dominatingExperimentId = row[9].as<long long>();
        record.dominatingLeaderScore = OptionalDoubleCell(row, 10);
        record.leaderScoreDelta = OptionalDoubleCell(row, 11);
        snapshot.dominated.push_back(record);
    }

    return snapshot;
}

void PrintIntelligenceRecord(const SchedulerIntelligenceRecord& record,
                                    const std::string& prefix = "  ")
{
    StatusOutput() << prefix
              << "experiment_id=" << record.experimentId
              << " model_id=" << OptionalLongLongText(record.modelId)
              << " symbol=" << record.symbol
              << " H=" << record.predictionHorizon
              << " leader_score=" << OptionalDoubleText(record.leaderScore, 4)
              << " infer_accuracy=" << OptionalDoubleText(record.inferAccuracy, 4)
              << " accept_accuracy=" << OptionalDoubleText(record.acceptAccuracy, 4)
              << "\n";
}

void PrintIntelligenceList(const std::string& title,
                                  const std::vector<SchedulerIntelligenceRecord>& records)
{
    StatusOutput() << "\n" << title << "\n";
    if (records.empty())
    {
        StatusOutput() << "  none\n";
        return;
    }
    for (const auto& record : records)
        PrintIntelligenceRecord(record);
}

void PrintExperimentIntelligence(const SchedulerIntelligenceSnapshot& intelligence)
{
    StatusOutput() << "\nEXPERIMENT INTELLIGENCE\n";
    StatusOutput() << "Completed today: " << intelligence.completedToday
              << " failed today: " << intelligence.failedToday
              << " waiting train=" << intelligence.waitingTrain
              << " infer=" << intelligence.waitingInfer
              << " analyze=" << intelligence.waitingAnalyze
              << "\n";

    StatusOutput() << "\nOverall leader:\n";
    if (intelligence.overallLeader.has_value())
        PrintIntelligenceRecord(*intelligence.overallLeader);
    else
        StatusOutput() << "  none\n";

    StatusOutput() << "\nBest completed in last 24h:\n";
    if (intelligence.recentBest24h.has_value())
        PrintIntelligenceRecord(*intelligence.recentBest24h);
    else
        StatusOutput() << "  none\n";

    PrintIntelligenceList("Leaders by symbol:", intelligence.leadersBySymbol);
    PrintIntelligenceList("Leaders by horizon:", intelligence.leadersByHorizon);
    PrintIntelligenceList("Top 5:", intelligence.top5);
    PrintIntelligenceList("Worst 5:", intelligence.worst5);

    StatusOutput() << "\nDominated candidates:\n";
    if (intelligence.dominated.empty())
    {
        StatusOutput() << "  none\n";
    }
    else
    {
        for (const auto& record : intelligence.dominated)
        {
            StatusOutput() << "  experiment_id=" << record.dominated.experimentId
                      << " symbol=" << record.dominated.symbol
                      << " H=" << record.dominated.predictionHorizon
                      << " leader_score=" << OptionalDoubleText(record.dominated.leaderScore, 4)
                      << " dominated_by=" << record.dominatingExperimentId
                      << " leader_score_delta=" << OptionalDoubleText(record.leaderScoreDelta, 4)
                      << "\n";
        }
    }
}

void PrintSchedulerStatusLeaderMachine(const std::string& scope,
                                              const SchedulerIntelligenceRecord& record,
                                              const std::optional<std::string>& scopeValue = std::nullopt)
{
    StatusOutput() << "SCHEDULER_STATUS_LEADER"
              << ",scope=" << scope;
    if (scopeValue.has_value())
        StatusOutput() << "," << *scopeValue;
    StatusOutput() << ",experiment_id=" << record.experimentId
              << ",model_id=" << OptionalLongLongText(record.modelId)
              << ",symbol=" << record.symbol
              << ",prediction_horizon=" << record.predictionHorizon
              << ",leader_score=" << OptionalDoubleText(record.leaderScore, 4)
              << ",infer_accuracy=" << OptionalDoubleText(record.inferAccuracy, 4)
              << ",accept_accuracy=" << OptionalDoubleText(record.acceptAccuracy, 4)
              << std::endl;
}

void PrintExperimentIntelligenceMachine(const SchedulerIntelligenceSnapshot& intelligence)
{
    StatusOutput() << "SCHEDULER_STATUS_INTELLIGENCE"
              << ",completed_today=" << intelligence.completedToday
              << ",failed_today=" << intelligence.failedToday
              << ",waiting_train=" << intelligence.waitingTrain
              << ",waiting_infer=" << intelligence.waitingInfer
              << ",waiting_analyze=" << intelligence.waitingAnalyze
              << ",dominated_count=" << intelligence.dominated.size()
              << ",overall_leader_experiment_id="
              << (intelligence.overallLeader.has_value()
                      ? std::to_string(intelligence.overallLeader->experimentId)
                      : "unknown")
              << ",overall_leader_score="
              << (intelligence.overallLeader.has_value()
                      ? OptionalDoubleText(intelligence.overallLeader->leaderScore, 4)
                      : "unknown")
              << std::endl;

    if (intelligence.overallLeader.has_value())
        PrintSchedulerStatusLeaderMachine("overall", *intelligence.overallLeader);
    if (intelligence.recentBest24h.has_value())
        PrintSchedulerStatusLeaderMachine("recent_24h", *intelligence.recentBest24h);
    for (const auto& record : intelligence.leadersBySymbol)
        PrintSchedulerStatusLeaderMachine("symbol", record, "symbol=" + record.symbol);
    for (const auto& record : intelligence.leadersByHorizon)
        PrintSchedulerStatusLeaderMachine("horizon", record, "prediction_horizon=" + std::to_string(record.predictionHorizon));
    for (const auto& record : intelligence.dominated)
    {
        StatusOutput() << "SCHEDULER_STATUS_DOMINATED"
                  << ",experiment_id=" << record.dominated.experimentId
                  << ",dominated_by_experiment_id=" << record.dominatingExperimentId
                  << ",symbol=" << record.dominated.symbol
                  << ",prediction_horizon=" << record.dominated.predictionHorizon
                  << ",leader_score=" << OptionalDoubleText(record.dominated.leaderScore, 4)
                  << ",dominating_leader_score=" << OptionalDoubleText(record.dominatingLeaderScore, 4)
                  << std::endl;
    }
}

int CountStatusRows(pqxx::transaction_base& transaction,
                    const std::string& sql)
{
    return transaction.exec(sql).one_row()[0].as<int>();
}

QueueSnapshot LoadQueueSnapshotForStatus(pqxx::transaction_base& transaction)
{
    QueueSnapshot snapshot;
    const pqxx::result rows = transaction.exec(
        "SELECT phase,status,count(*) FROM experiment "
        "WHERE status IN ('pending','running') "
        "AND phase IN ('train','infer','analyze') "
        "GROUP BY phase,status;");
    for (const auto& row : rows)
    {
        const std::string phase = row[0].as<std::string>();
        const std::string status = row[1].as<std::string>();
        const int count = row[2].as<int>();
        if (phase == "train" && status == "pending") snapshot.pendingTrain = count;
        else if (phase == "infer" && status == "pending") snapshot.pendingInfer = count;
        else if (phase == "analyze" && status == "pending") snapshot.pendingAnalyze = count;
        else if (phase == "train" && status == "running") snapshot.runningTrain = count;
        else if (phase == "infer" && status == "running") snapshot.runningInfer = count;
        else if (phase == "analyze" && status == "running") snapshot.runningAnalyze = count;
    }
    return snapshot;
}

int PrintSchedulerStatusFromTransaction(
    const SchedulerOptions& options,
    pqxx::transaction_base& w,
    const EA::GlobalExperimentControl::ControlSnapshot& globalControl)
{
    const bool useColor = UseAnsiColors();
    const SchedulerStatusProcessSnapshot processes = LoadSchedulerStatusProcessSnapshot();

    const int globallyPausedWorkers = CountStatusRows(
        w,
        "SELECT ("
        " (SELECT count(*) FROM experiment WHERE status='running' "
        "  AND worker_control_state='paused') +"
        " (SELECT count(*) FROM experiment_checkpoint_eval "
        "  WHERE status='running' AND phase='infer' "
        "  AND worker_control_state='paused')"
        ")::bigint;");
    const int selectivelyReleasedWorkers = CountStatusRows(
        w,
        "SELECT ("
        " (SELECT count(*) FROM experiment e "
        "  JOIN experiment_global_control c ON c.singleton "
        "  WHERE e.status='running' "
        "  AND e.worker_control_state='running' "
        "  AND e.worker_global_pause_request_id=c.current_pause_request_id) +"
        " (SELECT count(*) FROM experiment_checkpoint_eval ce "
        "  JOIN experiment_global_control c ON c.singleton "
        "  WHERE ce.status='running' AND ce.phase='infer' "
        "  AND ce.worker_control_state='running' "
        "  AND ce.worker_global_pause_request_id=c.current_pause_request_id)"
        ")::bigint;");
    const int pendingCheckpointCancellations = CountStatusRows(
        w,
        "SELECT count(*) FROM experiment "
        "WHERE status IN ('running','pending') "
        "AND cancel_after_checkpoint_epoch IS NOT NULL;");
    pqxx::result latestAdmin = w.exec(
        "SELECT request_id,action,COALESCE(cancellation_mode,'NULL'),"
        "infer_before_cancel,status,requested_at::text,"
        "COALESCE(completed_at::text,'NULL'),successful_count,"
        "missing_count,rejected_count,failed_count "
        "FROM experiment_admin_request "
        "ORDER BY request_id DESC LIMIT 1;");
    const QueueSnapshot queueSnapshot =
        LoadQueueSnapshotForStatus(w);
    const SchedulerStatusCounts counts = LoadSchedulerStatusCounts(w);
    const SchedulerIntelligenceSnapshot intelligence = LoadSchedulerIntelligenceSnapshot(w, queueSnapshot);
    std::vector<SchedulerStatusJob> runningTrain = LoadSchedulerStatusJobs(w, "running", "train", 50, false);
    std::vector<SchedulerStatusJob> runningInfer = LoadSchedulerStatusJobs(w, "running", "infer", 50, false);
    std::vector<SchedulerStatusJob> runningAnalyze = LoadSchedulerStatusJobs(w, "running", "analyze", 50, false);
    std::vector<SchedulerStatusJob> queued = LoadSchedulerStatusJobs(w, "pending", std::nullopt, 50, false);
    std::vector<SchedulerStatusJob> paused = LoadSchedulerStatusJobs(w, "paused", std::nullopt, 50, false);
    std::vector<SchedulerStatusJob> completed = LoadSchedulerStatusJobs(w, "completed", std::nullopt, 10, true);
    std::vector<SchedulerStatusJob> failed = LoadSchedulerStatusJobs(w, "failed", std::nullopt, 20, true);
    const auto authoritativeWorkers = LoadAuthoritativeSchedulerWorkers(w);
    std::vector<SchedulerCheckpointStatusJob> activeCheckpointInfer =
        LoadActiveCheckpointStatusJobs(w);
    pqxx::result schedulerLease = w.exec(
        "SELECT l.authority_state,"
        "COALESCE(l.owner_scheduler_invocation_id,'NULL'),"
        "l.fencing_token,COALESCE(l.acquired_at::text,'NULL'),"
        "COALESCE(l.heartbeat_at::text,'NULL'),"
        "COALESCE(l.expires_at::text,'NULL'),l.transition_reason,"
        "(l.expires_at IS NOT NULL "
        " AND l.expires_at<=clock_timestamp()) AS expired,"
        "COALESCE(i.canonical_executable_path,'NULL'),"
        "i.process_pid,i.process_group_id,i.process_start_identity "
        "FROM experiment_scheduler_lease l "
        "LEFT JOIN experiment_scheduler_invocation i "
        "ON i.scheduler_invocation_id="
        "l.owner_scheduler_invocation_id "
        "WHERE l.singleton=true;");
    pqxx::result schedulerProtocol = w.exec(
        "SELECT required_generation,cutover_state,"
        "COALESCE(cutover_completed_at::text,'NULL'),"
        "COALESCE(cutover_completed_by,'NULL'),"
        "COALESCE(cutover_process_evidence,'NULL'),"
        "legacy_no_pid_grace_seconds,"
        "COALESCE(failure_diagnostic,'NULL') "
        "FROM experiment_scheduler_protocol "
        "WHERE singleton=true;");
    pqxx::result durableWorkerCounts = w.exec(
        "SELECT capacity_class,"
        "count(*) FILTER (WHERE lifecycle_state IN "
        " ('reserved','spawned','running','observed',"
        "  'identity_ambiguous')) AS consuming,"
        "count(*) FILTER (WHERE lifecycle_state='reserved') "
        " AS reservations,"
        "count(*) FILTER (WHERE lifecycle_state='identity_ambiguous') "
        " AS identity_mismatches,"
        "count(*) FILTER (WHERE lifecycle_state='observed') "
        " AS observed,"
        "count(*) FILTER (WHERE worker_kind='checkpoint_infer' "
        " AND lifecycle_state IN "
        " ('reserved','spawned','running','observed',"
        "  'identity_ambiguous')) AS checkpoint_workers "
        "FROM experiment_scheduler_worker_attempt "
        "GROUP BY capacity_class ORDER BY capacity_class;");
    pqxx::result attemptOwnershipCounts = w.exec(
        "SELECT "
        "count(*) FILTER (WHERE a.lifecycle_state IN "
        " ('reserved','spawned','running','observed',"
        "  'stopped','identity_ambiguous') "
        " AND a.scheduler_invocation_id="
        "     l.owner_scheduler_invocation_id) AS current_owner,"
        "count(*) FILTER (WHERE a.lifecycle_state IN "
        " ('reserved','spawned','running','observed',"
        "  'stopped','identity_ambiguous') "
        " AND a.scheduler_invocation_id IS DISTINCT FROM "
        "     l.owner_scheduler_invocation_id) AS prior_or_legacy,"
        "count(*) FILTER (WHERE a.lifecycle_state='reserved') "
        " AS unresolved_launches,"
        "count(*) FILTER (WHERE a.lifecycle_state="
        " 'identity_ambiguous') AS orphan_candidates "
        "FROM experiment_scheduler_worker_attempt a "
        "CROSS JOIN experiment_scheduler_lease l "
        "WHERE l.singleton=true;");
    pqxx::result activeAttemptRows = w.exec(
        "SELECT a.worker_attempt_id,a.worker_kind,"
        "a.lifecycle_phase,a.capacity_class,a.lifecycle_state,"
        "COALESCE(a.scheduler_invocation_id,'legacy'),"
        "a.experiment_id,a.checkpoint_eval_id,a.worker_pid,"
        "COALESCE(a.observed_by_scheduler_invocation_id,'NULL'),"
        "COALESCE(a.reconciliation_result,'NULL'),"
        "COALESCE(a.diagnostic,'NULL') "
        "FROM experiment_scheduler_worker_attempt a "
        "WHERE a.lifecycle_state IN "
        "('reserved','spawned','running','observed',"
        "'stopped','identity_ambiguous') "
        "ORDER BY a.worker_attempt_id;");
    pqxx::result unresolvedLegacyNoPid = w.exec(
        "SELECT capacity_class,count(*) "
        "FROM experiment_scheduler_worker_attempt "
        "WHERE ownership_origin='legacy_unverified' "
        "AND lifecycle_state='identity_ambiguous' "
        "AND worker_pid IS NULL "
        "GROUP BY capacity_class ORDER BY capacity_class;");

    EnrichSchedulerStatusJobs(runningTrain, processes);
    EnrichSchedulerStatusJobs(runningInfer, processes);
    EnrichSchedulerStatusJobs(runningAnalyze, processes);
    EnrichSchedulerStatusJobs(queued, processes);
    EnrichSchedulerStatusJobs(paused, processes);
    EnrichSchedulerStatusJobs(completed, processes);
    EnrichSchedulerStatusJobs(failed, processes);
    EnrichCheckpointStatusJobs(activeCheckpointInfer, processes);
    auto processObserver =
        EA::GlobalExperimentControl::CreateNativeProcessObserver();
    const SchedulerWorkerAccounting workerAccounting =
        ComputeSchedulerWorkerAccounting(
            processes, authoritativeWorkers, *processObserver);

    const bool schedulerRunning = !processes.schedulerPids.empty();
    const std::string schedulerPid =
        schedulerRunning ? std::to_string(processes.schedulerPids.front()) : "unknown";
    const auto automationStateText = [](const std::optional<bool>& enabled) {
        if (!enabled.has_value())
            return std::string{"unknown"};
        return std::string{*enabled ? "enabled" : "disabled"};
    };
    SchedulerOwnerProcessEvidence schedulerExecutableEvidence =
        SchedulerOwnerProcessEvidence::Ambiguous;
    EA::GlobalExperimentControl::ProcessObservation
        schedulerOwnerObservation;
    const std::string schedulerCanonicalExecutable =
        schedulerLease.size() == 1
            ? schedulerLease[0][8].as<std::string>()
            : "NULL";
    if (schedulerLease.size() == 1 &&
        schedulerLease[0][0].as<std::string>() == "active" &&
        schedulerCanonicalExecutable != "NULL" &&
        !schedulerLease[0][9].is_null() &&
        !schedulerLease[0][10].is_null() &&
        !schedulerLease[0][11].is_null())
    {
        schedulerExecutableEvidence = InspectSchedulerOwnerProcess(
            schedulerLease[0][9].as<int>(),
            schedulerLease[0][10].as<int>(),
            schedulerLease[0][11].as<std::string>(),
            schedulerCanonicalExecutable,
            &schedulerOwnerObservation);
    }
    const std::string schedulerObservedExecutable =
        schedulerOwnerObservation.executable.empty()
            ? "NULL" : schedulerOwnerObservation.executable;
    const std::string schedulerExecutableIdentityMatch =
        schedulerExecutableEvidence == SchedulerOwnerProcessEvidence::Valid
            ? "1"
            : (schedulerExecutableEvidence ==
                       SchedulerOwnerProcessEvidence::Ambiguous
                   ? "unknown" : "0");

    StatusOutput() << "Scheduler Status\n";
    if (schedulerLease.size() == 1)
    {
        StatusOutput() << "Scheduler authority: "
                  << schedulerLease[0][0].as<std::string>()
                  << " owner=" << schedulerLease[0][1].as<std::string>()
                  << " fence=" << schedulerLease[0][2].as<long long>()
                  << " acquired=" << schedulerLease[0][3].as<std::string>()
                  << " heartbeat=" << schedulerLease[0][4].as<std::string>()
                  << " expires=" << schedulerLease[0][5].as<std::string>()
                  << " expired="
                  << (schedulerLease[0][7].as<bool>() ? 1 : 0)
                  << "\n";
    }
    if (schedulerProtocol.size() == 1)
    {
        StatusOutput() << "Scheduler protocol: generation="
                  << schedulerProtocol[0][0].as<int>()
                  << " cutover_state="
                  << schedulerProtocol[0][1].as<std::string>()
                  << " completed_at="
                  << schedulerProtocol[0][2].as<std::string>()
                  << " legacy_no_pid_grace_seconds="
                  << schedulerProtocol[0][5].as<int>()
                  << "\n";
    }
    StatusOutput() << "Status reporter executable: " << options.schedulerExecutablePath << "\n"
              << "Scheduler canonical executable: "
              << schedulerCanonicalExecutable << "\n"
              << "Scheduler executable identity: "
              << EA::SchedulerCore::SchedulerOwnerProcessEvidenceText(
                     schedulerExecutableEvidence)
              << " observed=" << schedulerObservedExecutable << "\n";
    if (attemptOwnershipCounts.size() == 1)
    {
        StatusOutput() << "Durable workers: current_owner="
                  << attemptOwnershipCounts[0][0].as<long long>()
                  << " prior_or_legacy="
                  << attemptOwnershipCounts[0][1].as<long long>()
                  << " unresolved_launches="
                  << attemptOwnershipCounts[0][2].as<long long>()
                  << " orphan_candidates="
                  << attemptOwnershipCounts[0][3].as<long long>()
                  << "\n";
    }
    StatusOutput() << "Global experiment execution: "
              << globalControl.desiredState
              << " active_request_id="
              << (globalControl.activeRequestId
                      ? std::to_string(*globalControl.activeRequestId)
                      : "none")
              << " current_pause_request_id="
              << (globalControl.currentPauseRequestId
                      ? std::to_string(*globalControl.currentPauseRequestId)
                      : "none")
              << " paused_workers=" << globallyPausedWorkers
              << " selectively_released_workers="
              << selectivelyReleasedWorkers
              << " cancellation_mode="
              << globalControl.cancellationMode.value_or("none")
              << " infer_before_cancel="
              << (globalControl.inferBeforeCancel ? "yes" : "no")
              << " pending_checkpoint_cancellations="
              << pendingCheckpointCancellations
              << "\n";
    if (!latestAdmin.empty())
    {
        StatusOutput() << "Latest administrative request: id="
                  << latestAdmin[0][0].as<long long>()
                  << " action=" << latestAdmin[0][1].as<std::string>()
                  << " mode=" << latestAdmin[0][2].as<std::string>()
                  << " infer_before_cancel="
                  << (latestAdmin[0][3].as<bool>() ? "yes" : "no")
                  << " status=" << latestAdmin[0][4].as<std::string>()
                  << " successful="
                  << latestAdmin[0][7].as<int>()
                  << " missing=" << latestAdmin[0][8].as<int>()
                  << " rejected=" << latestAdmin[0][9].as<int>()
                  << " failed=" << latestAdmin[0][10].as<int>()
                  << "\n";
    }
    if (schedulerRunning)
    {
        StatusOutput() << "Scheduler process: "
                  << Colorize("running", "32", useColor)
                  << " pid=" << schedulerPid;
        if (processes.schedulerPids.size() > 1)
            StatusOutput() << " additional_pids=" << (processes.schedulerPids.size() - 1);
        StatusOutput() << "\n";
    }
    else if (!processes.processDetectionAvailable)
    {
        StatusOutput() << "Scheduler process: unknown (process detection unavailable)\n";
    }
    else
    {
        StatusOutput() << Colorize("Scheduler process not detected.", "31", useColor) << "\n";
    }

    StatusOutput() << "Poll interval: "
              << (processes.schedulerPollSeconds.has_value()
                      ? std::to_string(*processes.schedulerPollSeconds) + "s"
                      : "unknown")
              << "\n";
    StatusOutput() << "Configured worker limits: train="
              << OptionalIntText(processes.maxTrainProcs)
              << " infer=" << OptionalIntText(processes.maxInferProcs)
              << " analyze=" << OptionalIntText(processes.maxAnalyzeProcs)
              << "\n";
    StatusOutput() << "Continuation Automation:\n"
              << "  automatic evaluation="
              << automationStateText(processes.autoEvaluateContinuations)
              << " automatic queueing="
              << automationStateText(processes.autoQueueContinuations)
              << " dry-run=" << automationStateText(processes.continuationDryRun)
              << "\n"
              << "  scan interval="
              << (processes.continuationScanSeconds.has_value()
                      ? std::to_string(*processes.continuationScanSeconds) + "s"
                      : "unknown")
              << " maximum queues per scan="
              << OptionalIntText(processes.continuationMaxQueuesPerScan)
              << "\n"
              << "  last scan=unavailable next scan=unavailable"
              << " counts=unavailable (scheduler process memory only)\n";
    StatusOutput() << "Detected worker processes:\n"
              << "  managed train=" << workerAccounting.managedTrain
              << " infer=" << workerAccounting.managedInfer
              << " analyze=" << workerAccounting.managedAnalyze
              << "\n"
              << "  managed running train="
              << workerAccounting.managedRunningTrain
              << " infer=" << workerAccounting.managedRunningInfer
              << " analyze=" << workerAccounting.managedRunningAnalyze
              << "\n"
              << "  managed paused train="
              << workerAccounting.managedPausedTrain
              << " infer=" << workerAccounting.managedPausedInfer
              << " analyze=" << workerAccounting.managedPausedAnalyze
              << "\n"
              << "  unmanaged train=" << workerAccounting.unmanagedTrain
              << " infer=" << workerAccounting.unmanagedInfer
              << " analyze=" << workerAccounting.unmanagedAnalyze
              << "\n"
              << "  identity mismatch train="
              << workerAccounting.identityMismatchTrain
              << " infer=" << workerAccounting.identityMismatchInfer
              << " analyze=" << workerAccounting.identityMismatchAnalyze
              << "\n"
              << "  expected missing train="
              << workerAccounting.expectedMissingTrain
              << " infer=" << workerAccounting.expectedMissingInfer
              << " analyze=" << workerAccounting.expectedMissingAnalyze
              << "\n";

    PrintSchedulerResourceUsage(processes, workerAccounting);
    PrintUnmanagedWorkers(workerAccounting);
    const std::vector<std::string> warnings = BuildSchedulerStatusWarnings(processes, workerAccounting);
    PrintSchedulerWarnings(warnings);

    StatusOutput() << "\nOverall Counts\n"
              << "  queued=" << counts.queued
              << " paused=" << counts.paused
              << " running=" << counts.running
              << " completed=" << counts.completed
              << " failed=" << counts.failed
              << " cancelled=" << counts.cancelled
              << "\n";
    StatusOutput() << "  pending_train=" << queueSnapshot.pendingTrain
              << " pending_infer=" << queueSnapshot.pendingInfer
              << " pending_analyze=" << queueSnapshot.pendingAnalyze
              << " running_train=" << queueSnapshot.runningTrain
              << " running_infer=" << queueSnapshot.runningInfer
              << " running_analyze=" << queueSnapshot.runningAnalyze
              << "\n";

    PrintExperimentIntelligence(intelligence);

    PrintStatusJobTable("Active Training Jobs", runningTrain, useColor, true, false);
    PrintStatusJobTable("Active Inference Jobs", runningInfer, useColor, false, false);
    PrintCheckpointStatusJobs(activeCheckpointInfer);
    PrintStatusJobTable("Active Analysis Jobs", runningAnalyze, useColor, false, false);
    PrintStatusJobTable("Queued Jobs", queued, useColor, false, false);
    PrintStatusJobTable("Paused Jobs", paused, useColor, false, false);
    PrintStatusJobTable("Recent Completed Experiments", completed, useColor, false, false);
    PrintStatusJobTable("Failed Experiments Summary", failed, useColor, false, true);

    if (SchedulerStatusShouldEmitMachineRecords(options))
    {
        if (schedulerLease.size() == 1)
        {
            StatusOutput() << "\nSCHEDULER_STATUS_OWNERSHIP"
                      << ",authority_state="
                      << schedulerLease[0][0].as<std::string>()
                      << ",owner_scheduler_invocation_id="
                      << schedulerLease[0][1].as<std::string>()
                      << ",fencing_token="
                      << schedulerLease[0][2].as<long long>()
                      << ",acquired_at="
                      << schedulerLease[0][3].as<std::string>()
                      << ",heartbeat_at="
                      << schedulerLease[0][4].as<std::string>()
                      << ",expires_at="
                      << schedulerLease[0][5].as<std::string>()
                      << ",transition_reason="
                      << schedulerLease[0][6].as<std::string>()
                      << ",expired="
                      << (schedulerLease[0][7].as<bool>() ? 1 : 0)
                      << ",canonical_executable_path="
                      << schedulerLease[0][8].as<std::string>()
                      << ",scheduler_canonical_executable_path="
                      << schedulerCanonicalExecutable
                      << std::endl;
        }
        StatusOutput() << "SCHEDULER_STATUS_EXECUTABLE_IDENTITY"
                  << ",status_reporter_executable_path="
                  << options.schedulerExecutablePath
                  << ",scheduler_canonical_executable_path="
                  << schedulerCanonicalExecutable
                  << ",scheduler_observed_executable_path="
                  << schedulerObservedExecutable
                  << ",scheduler_identity_result="
                  << EA::SchedulerCore::SchedulerOwnerProcessEvidenceText(
                         schedulerExecutableEvidence)
                  << ",scheduler_identity_match="
                  << schedulerExecutableIdentityMatch
                  << std::endl;
        if (schedulerProtocol.size() == 1)
        {
            StatusOutput()
                << "SCHEDULER_STATUS_PROTOCOL"
                << ",required_generation="
                << schedulerProtocol[0][0].as<int>()
                << ",cutover_state="
                << schedulerProtocol[0][1].as<std::string>()
                << ",cutover_completed_at="
                << schedulerProtocol[0][2].as<std::string>()
                << ",cutover_completed_by="
                << schedulerProtocol[0][3].as<std::string>()
                << ",cutover_process_evidence="
                << schedulerProtocol[0][4].as<std::string>()
                << ",legacy_no_pid_grace_seconds="
                << schedulerProtocol[0][5].as<int>()
                << ",failure_diagnostic="
                << schedulerProtocol[0][6].as<std::string>()
                << std::endl;
        }
        for (const pqxx::row& legacy :
             unresolvedLegacyNoPid)
        {
            StatusOutput()
                << "SCHEDULER_STATUS_LEGACY_NO_PID"
                << ",capacity_class="
                << legacy[0].as<std::string>()
                << ",unresolved="
                << legacy[1].as<long long>()
                << ",capacity_consumed="
                << legacy[1].as<long long>()
                << std::endl;
        }
        for (const pqxx::row& capacity : durableWorkerCounts)
        {
            StatusOutput() << "SCHEDULER_STATUS_GLOBAL_CAPACITY"
                      << ",capacity_class="
                      << capacity[0].as<std::string>()
                      << ",consuming=" << capacity[1].as<long long>()
                      << ",reservations=" << capacity[2].as<long long>()
                      << ",identity_mismatches="
                      << capacity[3].as<long long>()
                      << ",observed_prior_workers="
                      << capacity[4].as<long long>()
                      << ",checkpoint_workers="
                      << capacity[5].as<long long>()
                      << std::endl;
        }
        if (attemptOwnershipCounts.size() == 1)
        {
            StatusOutput() << "SCHEDULER_STATUS_WORKER_OWNERSHIP"
                      << ",current_owner="
                      << attemptOwnershipCounts[0][0].as<long long>()
                      << ",prior_or_legacy="
                      << attemptOwnershipCounts[0][1].as<long long>()
                      << ",unresolved_launches="
                      << attemptOwnershipCounts[0][2].as<long long>()
                      << ",orphan_candidates="
                      << attemptOwnershipCounts[0][3].as<long long>()
                      << std::endl;
        }
        for (const pqxx::row& attempt : activeAttemptRows)
        {
            StatusOutput() << "SCHEDULER_STATUS_WORKER_ATTEMPT"
                      << ",worker_attempt_id="
                      << attempt[0].as<long long>()
                      << ",worker_kind="
                      << attempt[1].as<std::string>()
                      << ",phase=" << attempt[2].as<std::string>()
                      << ",capacity_class="
                      << attempt[3].as<std::string>()
                      << ",state=" << attempt[4].as<std::string>()
                      << ",launch_scheduler_invocation_id="
                      << attempt[5].as<std::string>()
                      << ",experiment_id="
                      << attempt[6].as<long long>()
                      << ",checkpoint_eval_id="
                      << (attempt[7].is_null()
                              ? "NULL"
                              : attempt[7].c_str())
                      << ",pid="
                      << (attempt[8].is_null()
                              ? "NULL"
                              : attempt[8].c_str())
                      << ",observed_by="
                      << attempt[9].as<std::string>()
                      << ",reconciliation_result="
                      << attempt[10].as<std::string>()
                      << ",diagnostic="
                      << attempt[11].as<std::string>()
                      << std::endl;
        }
        StatusOutput() << "\nSCHEDULER_STATUS"
                  << ",running=" << (schedulerRunning ? "1" : "0")
                  << ",global_desired_state=" << globalControl.desiredState
                  << ",active_admin_request_id="
                  << (globalControl.activeRequestId
                          ? std::to_string(*globalControl.activeRequestId)
                          : "NULL")
                  << ",current_pause_request_id="
                  << (globalControl.currentPauseRequestId
                          ? std::to_string(
                                *globalControl.currentPauseRequestId)
                          : "NULL")
                  << ",globally_paused_workers=" << globallyPausedWorkers
                  << ",selectively_released_workers="
                  << selectivelyReleasedWorkers
                  << ",active_cancellation_mode="
                  << globalControl.cancellationMode.value_or("NULL")
                  << ",active_infer_before_cancel="
                  << (globalControl.inferBeforeCancel ? "1" : "0")
                  << ",pending_checkpoint_cancellations="
                  << pendingCheckpointCancellations
                  << ",latest_admin_request_id="
                  << (latestAdmin.empty()
                          ? "NULL"
                          : std::to_string(
                                latestAdmin[0][0].as<long long>()))
                  << ",latest_admin_status="
                  << (latestAdmin.empty()
                          ? "NULL"
                          : latestAdmin[0][4].as<std::string>())
                  << ",pid=" << (schedulerRunning ? schedulerPid : "unknown")
                  << ",train_workers=" << processes.trainWorkers
                  << ",infer_workers=" << processes.inferWorkers
                  << ",analysis_workers=" << processes.analysisWorkers
                  << ",managed_train_workers=" << workerAccounting.managedTrain
                  << ",managed_infer_workers=" << workerAccounting.managedInfer
                  << ",managed_analysis_workers=" << workerAccounting.managedAnalyze
                  << ",managed_running_train_workers="
                  << workerAccounting.managedRunningTrain
                  << ",managed_running_infer_workers="
                  << workerAccounting.managedRunningInfer
                  << ",managed_running_analysis_workers="
                  << workerAccounting.managedRunningAnalyze
                  << ",managed_paused_train_workers="
                  << workerAccounting.managedPausedTrain
                  << ",managed_paused_infer_workers="
                  << workerAccounting.managedPausedInfer
                  << ",managed_paused_analysis_workers="
                  << workerAccounting.managedPausedAnalyze
                  << ",unmanaged_train_workers=" << workerAccounting.unmanagedTrain
                  << ",unmanaged_infer_workers=" << workerAccounting.unmanagedInfer
                  << ",unmanaged_analysis_workers=" << workerAccounting.unmanagedAnalyze
                  << ",identity_mismatch_train_workers="
                  << workerAccounting.identityMismatchTrain
                  << ",identity_mismatch_infer_workers="
                  << workerAccounting.identityMismatchInfer
                  << ",identity_mismatch_analysis_workers="
                  << workerAccounting.identityMismatchAnalyze
                  << ",expected_missing_train_workers="
                  << workerAccounting.expectedMissingTrain
                  << ",expected_missing_infer_workers="
                  << workerAccounting.expectedMissingInfer
                  << ",expected_missing_analysis_workers="
                  << workerAccounting.expectedMissingAnalyze
                  << ",poll_seconds=" << OptionalIntText(processes.schedulerPollSeconds)
                  << ",max_train_procs=" << OptionalIntText(processes.maxTrainProcs)
                  << ",max_infer_procs=" << OptionalIntText(processes.maxInferProcs)
                  << ",max_analyze_procs=" << OptionalIntText(processes.maxAnalyzeProcs)
                  << std::endl;
        StatusOutput() << "SCHEDULER_STATUS_RESOURCE"
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
                  << ",managed_train_cpu_percent=" << OptionalDoubleText(workerAccounting.managedTrainResources.cpuPercent, 1)
                  << ",managed_train_rss_mb=" << OptionalDoubleText(workerAccounting.managedTrainResources.rssMb, 0)
                  << ",managed_train_workers=" << workerAccounting.managedTrain
                  << ",unmanaged_train_cpu_percent=" << OptionalDoubleText(workerAccounting.unmanagedTrainResources.cpuPercent, 1)
                  << ",unmanaged_train_rss_mb=" << OptionalDoubleText(workerAccounting.unmanagedTrainResources.rssMb, 0)
                  << ",unmanaged_train_workers=" << workerAccounting.unmanagedTrain
                  << ",infer_cpu_percent=" << OptionalDoubleText(processes.inferResources.cpuPercent, 1)
                  << ",infer_rss_mb=" << OptionalDoubleText(processes.inferResources.rssMb, 0)
                  << ",infer_mem_percent=" << OptionalDoubleText(processes.inferResources.memPercent, 1)
                  << ",infer_workers=" << processes.inferWorkers
                  << ",managed_infer_cpu_percent=" << OptionalDoubleText(workerAccounting.managedInferResources.cpuPercent, 1)
                  << ",managed_infer_rss_mb=" << OptionalDoubleText(workerAccounting.managedInferResources.rssMb, 0)
                  << ",managed_infer_workers=" << workerAccounting.managedInfer
                  << ",unmanaged_infer_cpu_percent=" << OptionalDoubleText(workerAccounting.unmanagedInferResources.cpuPercent, 1)
                  << ",unmanaged_infer_rss_mb=" << OptionalDoubleText(workerAccounting.unmanagedInferResources.rssMb, 0)
                  << ",unmanaged_infer_workers=" << workerAccounting.unmanagedInfer
                  << ",analysis_cpu_percent=" << OptionalDoubleText(processes.analysisResources.cpuPercent, 1)
                  << ",analysis_rss_mb=" << OptionalDoubleText(processes.analysisResources.rssMb, 0)
                  << ",analysis_mem_percent=" << OptionalDoubleText(processes.analysisResources.memPercent, 1)
                  << ",analysis_workers=" << processes.analysisWorkers
                  << ",managed_analysis_cpu_percent=" << OptionalDoubleText(workerAccounting.managedAnalysisResources.cpuPercent, 1)
                  << ",managed_analysis_rss_mb=" << OptionalDoubleText(workerAccounting.managedAnalysisResources.rssMb, 0)
                  << ",managed_analysis_workers=" << workerAccounting.managedAnalyze
                  << ",unmanaged_analysis_cpu_percent=" << OptionalDoubleText(workerAccounting.unmanagedAnalysisResources.cpuPercent, 1)
                  << ",unmanaged_analysis_rss_mb=" << OptionalDoubleText(workerAccounting.unmanagedAnalysisResources.rssMb, 0)
                  << ",unmanaged_analysis_workers=" << workerAccounting.unmanagedAnalyze
                  << std::endl;
        StatusOutput() << "SCHEDULER_STATUS_CONTINUATION"
                  << ",auto_evaluate="
                  << (processes.autoEvaluateContinuations.has_value()
                          ? (*processes.autoEvaluateContinuations ? "1" : "0")
                          : "unknown")
                  << ",auto_queue="
                  << (processes.autoQueueContinuations.has_value()
                          ? (*processes.autoQueueContinuations ? "1" : "0")
                          : "unknown")
                  << ",dry_run="
                  << (processes.continuationDryRun.has_value()
                          ? (*processes.continuationDryRun ? "1" : "0")
                          : "unknown")
                  << ",scan_seconds=" << OptionalIntText(processes.continuationScanSeconds)
                  << ",max_queues_per_scan="
                  << OptionalIntText(processes.continuationMaxQueuesPerScan)
                  << ",last_scan=unavailable"
                  << ",next_scan=unavailable"
                  << ",scan_candidates=unavailable"
                  << ",scan_evaluated=unavailable"
                  << ",scan_already_satisfied=unavailable"
                  << ",scan_eligible=unavailable"
                  << ",scan_queued=unavailable"
                  << ",scan_errors=unavailable"
                  << std::endl;
        for (const auto& worker : workerAccounting.unmanagedWorkers)
        {
            StatusOutput() << "SCHEDULER_STATUS_UNMANAGED_WORKER"
                      << ",pid=" << worker.pid
                      << ",kind=" << worker.kind
                      << ",reason=" << worker.reason
                      << ",command=" << worker.command
                      << std::endl;
        }
        for (const auto& worker :
             workerAccounting.workerClassifications)
        {
            StatusOutput() << "SCHEDULER_STATUS_WORKER"
                      << ",pid=" << worker.pid
                      << ",kind=" << worker.kind
                      << ",managed=" << (worker.managed ? 1 : 0)
                      << ",authoritative="
                      << (worker.authoritative ? 1 : 0)
                      << ",detected=" << (worker.detected ? 1 : 0)
                      << ",execution_state="
                      << EA::GlobalExperimentControl::ToString(
                             worker.executionState)
                      << ",lifecycle_status="
                      << (worker.lifecycleStatus.empty()
                              ? "NULL" : worker.lifecycleStatus)
                      << ",attempt_state="
                      << (worker.attemptLifecycleState.empty()
                              ? "NULL" : worker.attemptLifecycleState)
                      << ",identity_result="
                      << EA::GlobalExperimentControl::ToString(
                             worker.identity)
                      << ",executable_identity_match="
                      << (worker.executableIdentityMatch
                              ? (*worker.executableIdentityMatch ? "1" : "0")
                              : "unknown")
                      << ",canonical_executable_path="
                      << worker.expectedExecutable.value_or("NULL")
                      << ",observed_executable_path="
                      << worker.observedExecutable.value_or("NULL")
                      << ",experiment_id="
                      << (worker.experimentId
                              ? std::to_string(*worker.experimentId)
                              : "NULL")
                      << ",checkpoint_eval_id="
                      << (worker.checkpointEvalId
                              ? std::to_string(*worker.checkpointEvalId)
                              : "NULL")
                      << ",reason=" << worker.reason
                      << std::endl;
        }
        StatusOutput() << "SCHEDULER_STATUS_COUNT"
                  << ",queued=" << counts.queued
                  << ",paused=" << counts.paused
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
        PrintExperimentIntelligenceMachine(intelligence);
        for (const auto& job : runningTrain)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : runningInfer)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : activeCheckpointInfer)
            PrintCheckpointStatusJobMachine(job);
        for (const auto& job : runningAnalyze)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : queued)
            PrintSchedulerStatusJobMachine(job);
        for (const auto& job : paused)
            PrintSchedulerStatusJobMachine(job);
    }

    return 0;
}

int PrintSchedulerStatusImpl(const SchedulerOptions& options)
{
    pqxx::connection connection{LstmDbConnectionString()};
    pqxx::work transaction{connection};
    SetTransactionReadWrite(transaction);
    if (!RequireSchedulerTables(transaction))
        return 1;
    const auto globalControl = LoadLockedGlobalControl(transaction);
    if (!globalControl.has_value())
        return 1;
    const int result = PrintSchedulerStatusFromTransaction(
        options, transaction, *globalControl);
    transaction.commit();
    return result;
}

// This is deliberately a typed capture model, distinct from the legacy text
// renderers above.  JSON is only a transport for these observation values; it
// is never parsed back into scheduler state or authority.
struct OperationalEvidenceLease
{
    std::optional<std::string> authorityState;
    std::optional<std::string> ownerInvocationId;
    std::optional<long long> fencingToken;
    std::optional<bool> expired;
    std::optional<int> ownerPid;
    std::optional<int> ownerProcessGroupId;
    std::optional<std::string> ownerStartIdentity;
    std::optional<std::string> canonicalExecutable;
};

struct OperationalEvidenceProtocol
{
    std::optional<int> requiredGeneration;
    std::optional<std::string> cutoverState;
    std::optional<std::string> cutoverCompletedBy;
    std::optional<std::string> cutoverProcessEvidence;
    std::optional<int> legacyNoPidGraceSeconds;
    std::optional<std::string> failureDiagnostic;
};

struct OperationalEvidenceWorkerCount
{
    std::string capacityClass;
    long long consuming = 0;
    long long reservations = 0;
    long long identityMismatches = 0;
    long long observed = 0;
    long long checkpointWorkers = 0;
};

struct OperationalEvidenceAttempt
{
    long long workerAttemptId = -1;
    std::string workerKind;
    std::string lifecyclePhase;
    std::string capacityClass;
    std::string lifecycleState;
    std::optional<std::string> schedulerInvocationId;
    std::optional<int> pid;
    std::optional<int> processGroupId;
    std::optional<std::string> canonicalExecutable;
    std::optional<std::string> processStartIdentity;
    std::string ownershipOrigin;
};

struct OperationalEvidenceProcess
{
    bool available = false;
    std::vector<int> schedulerPids;
    std::optional<int> maxTrainProcs;
    std::optional<int> maxInferProcs;
    std::optional<int> maxAnalyzeProcs;
    std::optional<int> schedulerPollSeconds;
    std::optional<EA::GlobalExperimentControl::ProcessObservation>
        schedulerOwner;
    std::string schedulerOwnerValidation = "not_observed";
    std::vector<EA::GlobalExperimentControl::SchedulerWorkerClassification>
        workers;
    std::vector<std::string> warnings;
};

struct SchedulerOperationalEvidence
{
    OperationalEvidenceLease lease;
    OperationalEvidenceProtocol protocol;
    EA::GlobalExperimentControl::ControlSnapshot control;
    SchedulerStatusCounts lifecycleCounts;
    QueueSnapshot phaseCounts;
    std::vector<OperationalEvidenceWorkerCount> durableWorkerCounts;
    OperationalEvidenceProcess process;
    std::string reporterExecutable;
};

struct ExperimentOperationalEvidence
{
    SchedulerStatusJob experiment;
    std::vector<OperationalEvidenceAttempt> attempts;
    OperationalEvidenceProcess process;
};

std::string JsonEscape(const std::string& value)
{
    std::ostringstream escaped;
    for (const unsigned char character : value)
    {
        switch (character)
        {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (character < 0x20)
                {
                    escaped << "\\u" << std::hex << std::setw(4)
                            << std::setfill('0') << static_cast<int>(character)
                            << std::dec << std::setfill(' ');
                }
                else
                    escaped << static_cast<char>(character);
        }
    }
    return escaped.str();
}

std::string JsonString(const std::string& value)
{
    return "\"" + JsonEscape(value) + "\"";
}

template <typename T>
void WriteJsonOptional(std::ostream& output, const std::optional<T>& value)
{
    if (value.has_value())
        output << *value;
    else
        output << "null";
}

void WriteJsonOptionalString(std::ostream& output,
                             const std::optional<std::string>& value)
{
    if (value.has_value())
        output << JsonString(*value);
    else
        output << "null";
}

void WriteJsonOptionalBool(std::ostream& output,
                           const std::optional<bool>& value)
{
    if (!value.has_value())
        output << "null";
    else
        output << (*value ? "true" : "false");
}

void WriteJsonProcessObservation(
    std::ostream& output,
    const EA::GlobalExperimentControl::ProcessObservation& observation)
{
    output << "{\"exists\":" << (observation.exists ? "true" : "false")
           << ",\"inspection_succeeded\":"
           << (observation.inspectionSucceeded ? "true" : "false")
           << ",\"permission_denied\":"
           << (observation.permissionDenied ? "true" : "false")
           << ",\"stopped\":" << (observation.stopped ? "true" : "false")
           << ",\"pid\":" << (observation.pid > 0
                                      ? std::to_string(observation.pid)
                                      : "null")
           << ",\"process_group_id\":"
           << (observation.processGroupId > 0
                   ? std::to_string(observation.processGroupId)
                   : "null")
           << ",\"executable\":"
           << (observation.executable.empty() ? "null" : JsonString(observation.executable))
           << ",\"command_line\":"
           << (observation.commandLine.empty() ? "null" : JsonString(observation.commandLine))
           << ",\"process_start_identity\":"
           << (observation.processStartIdentity.empty()
                   ? "null"
                   : JsonString(observation.processStartIdentity))
           << "}";
}

void WriteJsonWorkerClassification(
    std::ostream& output,
    const EA::GlobalExperimentControl::SchedulerWorkerClassification& worker)
{
    output << "{\"pid\":" << worker.pid
           << ",\"kind\":" << JsonString(worker.kind)
           << ",\"managed\":" << (worker.managed ? "true" : "false")
           << ",\"authoritative\":" << (worker.authoritative ? "true" : "false")
           << ",\"identity\":"
           << JsonString(EA::GlobalExperimentControl::ToString(worker.identity))
           << ",\"execution_state\":"
           << JsonString(EA::GlobalExperimentControl::ToString(worker.executionState))
           << ",\"reason\":" << JsonString(worker.reason)
           << ",\"experiment_id\":";
    WriteJsonOptional(output, worker.experimentId);
    output << ",\"checkpoint_eval_id\":";
    WriteJsonOptional(output, worker.checkpointEvalId);
    output << ",\"expected_executable\":";
    WriteJsonOptionalString(output, worker.expectedExecutable);
    output << ",\"observed_executable\":";
    WriteJsonOptionalString(output, worker.observedExecutable);
    output << ",\"executable_identity_match\":";
    WriteJsonOptionalBool(output, worker.executableIdentityMatch);
    output << "}";
}

void WriteJsonProcessEvidence(std::ostream& output,
                              const OperationalEvidenceProcess& process,
                              std::optional<long long> experimentId = std::nullopt)
{
    output << "{\"available\":" << (process.available ? "true" : "false")
           << ",\"scheduler_pids\":[";
    for (size_t index = 0; index < process.schedulerPids.size(); ++index)
    {
        if (index != 0)
            output << ',';
        output << process.schedulerPids[index];
    }
    output << "],\"configured_worker_limits\":{\"train\":";
    WriteJsonOptional(output, process.maxTrainProcs);
    output << ",\"infer\":";
    WriteJsonOptional(output, process.maxInferProcs);
    output << ",\"analyze\":";
    WriteJsonOptional(output, process.maxAnalyzeProcs);
    output << "},\"scheduler_poll_seconds\":";
    WriteJsonOptional(output, process.schedulerPollSeconds);
    output << ",\"scheduler_owner_validation\":"
           << JsonString(process.schedulerOwnerValidation)
           << ",\"scheduler_owner\":";
    if (process.schedulerOwner.has_value())
        WriteJsonProcessObservation(output, *process.schedulerOwner);
    else
        output << "null";
    output << ",\"workers\":[";
    bool first = true;
    for (const auto& worker : process.workers)
    {
        if (experimentId.has_value() && worker.experimentId != experimentId)
            continue;
        if (!first)
            output << ',';
        first = false;
        WriteJsonWorkerClassification(output, worker);
    }
    output << "],\"warnings\":[";
    for (size_t index = 0; index < process.warnings.size(); ++index)
    {
        if (index != 0)
            output << ',';
        output << JsonString(process.warnings[index]);
    }
    output << "]}";
}

OperationalEvidenceLease LoadOperationalEvidenceLease(
    pqxx::transaction_base& transaction)
{
    OperationalEvidenceLease lease;
    const pqxx::result rows = transaction.exec(
        "SELECT l.authority_state,l.owner_scheduler_invocation_id,"
        "l.fencing_token,(l.expires_at IS NOT NULL AND "
        "l.expires_at<=clock_timestamp()),i.process_pid,i.process_group_id,"
        "i.process_start_identity,i.canonical_executable_path "
        "FROM experiment_scheduler_lease l "
        "LEFT JOIN experiment_scheduler_invocation i ON "
        "i.scheduler_invocation_id=l.owner_scheduler_invocation_id "
        "WHERE l.singleton=true;");
    if (rows.size() != 1)
        return lease;
    const auto& row = rows[0];
    lease.authorityState = OptionalStringCell(row, 0);
    lease.ownerInvocationId = OptionalStringCell(row, 1);
    if (!row[2].is_null())
        lease.fencingToken = row[2].as<long long>();
    if (!row[3].is_null())
        lease.expired = row[3].as<bool>();
    if (!row[4].is_null())
        lease.ownerPid = row[4].as<int>();
    if (!row[5].is_null())
        lease.ownerProcessGroupId = row[5].as<int>();
    lease.ownerStartIdentity = OptionalStringCell(row, 6);
    lease.canonicalExecutable = OptionalStringCell(row, 7);
    return lease;
}

OperationalEvidenceProtocol LoadOperationalEvidenceProtocol(
    pqxx::transaction_base& transaction)
{
    OperationalEvidenceProtocol protocol;
    const pqxx::result rows = transaction.exec(
        "SELECT required_generation,cutover_state,cutover_completed_by,"
        "cutover_process_evidence,legacy_no_pid_grace_seconds,failure_diagnostic "
        "FROM experiment_scheduler_protocol WHERE singleton=true;");
    if (rows.size() != 1)
        return protocol;
    const auto& row = rows[0];
    if (!row[0].is_null())
        protocol.requiredGeneration = row[0].as<int>();
    protocol.cutoverState = OptionalStringCell(row, 1);
    protocol.cutoverCompletedBy = OptionalStringCell(row, 2);
    protocol.cutoverProcessEvidence = OptionalStringCell(row, 3);
    if (!row[4].is_null())
        protocol.legacyNoPidGraceSeconds = row[4].as<int>();
    protocol.failureDiagnostic = OptionalStringCell(row, 5);
    return protocol;
}

std::vector<OperationalEvidenceWorkerCount> LoadOperationalEvidenceWorkerCounts(
    pqxx::transaction_base& transaction)
{
    const pqxx::result rows = transaction.exec(
        "SELECT capacity_class,"
        "count(*) FILTER (WHERE lifecycle_state IN "
        "('reserved','spawned','running','observed','identity_ambiguous')),"
        "count(*) FILTER (WHERE lifecycle_state='reserved'),"
        "count(*) FILTER (WHERE lifecycle_state='identity_ambiguous'),"
        "count(*) FILTER (WHERE lifecycle_state='observed'),"
        "count(*) FILTER (WHERE worker_kind='checkpoint_infer' AND "
        "lifecycle_state IN "
        "('reserved','spawned','running','observed','identity_ambiguous')) "
        "FROM experiment_scheduler_worker_attempt "
        "GROUP BY capacity_class ORDER BY capacity_class;");
    std::vector<OperationalEvidenceWorkerCount> counts;
    counts.reserve(rows.size());
    for (const auto& row : rows)
    {
        counts.push_back(OperationalEvidenceWorkerCount{
            row[0].as<std::string>(), row[1].as<long long>(),
            row[2].as<long long>(), row[3].as<long long>(),
            row[4].as<long long>(), row[5].as<long long>()});
    }
    return counts;
}

std::vector<OperationalEvidenceAttempt> LoadOperationalEvidenceAttempts(
    pqxx::transaction_base& transaction, long long experimentId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT worker_attempt_id,worker_kind,lifecycle_phase,capacity_class,"
        "lifecycle_state,scheduler_invocation_id,worker_pid,"
        "worker_process_group_id,canonical_executable_path,"
        "worker_process_start_identity,ownership_origin "
        "FROM experiment_scheduler_worker_attempt WHERE experiment_id=" +
        std::to_string(experimentId) + " AND lifecycle_state IN "
        "('reserved','spawned','running','observed','stopped',"
        "'identity_ambiguous') ORDER BY worker_attempt_id;");
    std::vector<OperationalEvidenceAttempt> attempts;
    attempts.reserve(rows.size());
    for (const auto& row : rows)
    {
        OperationalEvidenceAttempt attempt;
        attempt.workerAttemptId = row[0].as<long long>();
        attempt.workerKind = row[1].as<std::string>();
        attempt.lifecyclePhase = row[2].as<std::string>();
        attempt.capacityClass = row[3].as<std::string>();
        attempt.lifecycleState = row[4].as<std::string>();
        attempt.schedulerInvocationId = OptionalStringCell(row, 5);
        if (!row[6].is_null())
            attempt.pid = row[6].as<int>();
        if (!row[7].is_null())
            attempt.processGroupId = row[7].as<int>();
        attempt.canonicalExecutable = OptionalStringCell(row, 8);
        attempt.processStartIdentity = OptionalStringCell(row, 9);
        attempt.ownershipOrigin = row[10].as<std::string>();
        attempts.push_back(std::move(attempt));
    }
    return attempts;
}

OperationalEvidenceProcess CaptureOperationalEvidenceProcess(
    const SchedulerStatusProcessSnapshot& processes,
    const std::vector<EA::GlobalExperimentControl::ManagedWorker>& workers,
    const OperationalEvidenceLease& lease)
{
    OperationalEvidenceProcess evidence;
    evidence.available = processes.processDetectionAvailable;
    evidence.schedulerPids = processes.schedulerPids;
    std::sort(evidence.schedulerPids.begin(), evidence.schedulerPids.end());
    evidence.maxTrainProcs = processes.maxTrainProcs;
    evidence.maxInferProcs = processes.maxInferProcs;
    evidence.maxAnalyzeProcs = processes.maxAnalyzeProcs;
    evidence.schedulerPollSeconds = processes.schedulerPollSeconds;
    auto observer = EA::GlobalExperimentControl::CreateNativeProcessObserver();
    const SchedulerWorkerAccounting accounting = ComputeSchedulerWorkerAccounting(
        processes, workers, *observer);
    evidence.workers = accounting.workerClassifications;
    std::sort(evidence.workers.begin(), evidence.workers.end(),
              [](const auto& left, const auto& right) {
                  if (left.pid != right.pid)
                      return left.pid < right.pid;
                  return left.kind < right.kind;
              });
    evidence.warnings = BuildSchedulerStatusWarnings(processes, accounting);
    if (lease.authorityState && *lease.authorityState == "active" &&
        lease.ownerPid && lease.ownerProcessGroupId &&
        lease.ownerStartIdentity && lease.canonicalExecutable)
    {
        EA::GlobalExperimentControl::ProcessObservation observation;
        const SchedulerOwnerProcessEvidence validation = InspectSchedulerOwnerProcess(
            *lease.ownerPid, *lease.ownerProcessGroupId,
            *lease.ownerStartIdentity, *lease.canonicalExecutable, &observation);
        evidence.schedulerOwner = std::move(observation);
        evidence.schedulerOwnerValidation = SchedulerOwnerProcessEvidenceText(validation);
    }
    return evidence;
}

SchedulerOperationalEvidence CaptureSchedulerOperationalEvidence(
    pqxx::transaction_base& transaction)
{
    SchedulerOperationalEvidence evidence;
    evidence.reporterExecutable =
        EA::ExperimentScheduler::ResolveCanonicalExecutablePath();
    evidence.lease = LoadOperationalEvidenceLease(transaction);
    evidence.protocol = LoadOperationalEvidenceProtocol(transaction);
    evidence.control = EA::GlobalExperimentControl::LoadControlSnapshot(transaction);
    evidence.lifecycleCounts = LoadSchedulerStatusCounts(transaction);
    evidence.phaseCounts = LoadQueueSnapshotForStatus(transaction);
    evidence.durableWorkerCounts = LoadOperationalEvidenceWorkerCounts(transaction);
    const SchedulerStatusProcessSnapshot processes = LoadSchedulerStatusProcessSnapshot();
    const auto workers = LoadAuthoritativeSchedulerWorkers(transaction);
    evidence.process = CaptureOperationalEvidenceProcess(
        processes, workers, evidence.lease);
    return evidence;
}

std::optional<ExperimentOperationalEvidence> CaptureExperimentOperationalEvidence(
    pqxx::transaction_base& transaction, long long experimentId)
{
    const auto job = LoadSchedulerStatusJobById(transaction, experimentId);
    if (!job.has_value())
        return std::nullopt;
    ExperimentOperationalEvidence evidence;
    evidence.experiment = *job;
    evidence.attempts = LoadOperationalEvidenceAttempts(transaction, experimentId);
    const SchedulerStatusProcessSnapshot processes = LoadSchedulerStatusProcessSnapshot();
    const auto workers = LoadAuthoritativeSchedulerWorkers(transaction);
    const OperationalEvidenceLease lease = LoadOperationalEvidenceLease(transaction);
    evidence.process = CaptureOperationalEvidenceProcess(processes, workers, lease);
    return evidence;
}

void WriteJsonSchedulerEvidence(std::ostream& output,
                                const SchedulerOperationalEvidence& evidence)
{
    output << "{\"schema\":\"expertadvisor-operational-evidence-v1\","
           << "\"kind\":\"scheduler\",\"durable\":{"
           << "\"scheduler_authority\":{\"state\":";
    WriteJsonOptionalString(output, evidence.lease.authorityState);
    output << ",\"owner_scheduler_invocation_id\":";
    WriteJsonOptionalString(output, evidence.lease.ownerInvocationId);
    output << ",\"fencing_token\":";
    WriteJsonOptional(output, evidence.lease.fencingToken);
    output << ",\"expired\":";
    WriteJsonOptionalBool(output, evidence.lease.expired);
    output << "},\"scheduler_protocol\":{\"required_generation\":";
    WriteJsonOptional(output, evidence.protocol.requiredGeneration);
    output << ",\"cutover_state\":";
    WriteJsonOptionalString(output, evidence.protocol.cutoverState);
    output << ",\"cutover_completed_by\":";
    WriteJsonOptionalString(output, evidence.protocol.cutoverCompletedBy);
    output << ",\"cutover_process_evidence\":";
    WriteJsonOptionalString(output, evidence.protocol.cutoverProcessEvidence);
    output << ",\"legacy_no_pid_grace_seconds\":";
    WriteJsonOptional(output, evidence.protocol.legacyNoPidGraceSeconds);
    output << ",\"failure_diagnostic\":";
    WriteJsonOptionalString(output, evidence.protocol.failureDiagnostic);
    output << "},\"reporter_executable\":"
           << JsonString(evidence.reporterExecutable)
           << ",\"canonical_scheduler_executable\":";
    WriteJsonOptionalString(output, evidence.lease.canonicalExecutable);
    output << ",\"global_execution\":{\"desired_state\":"
           << JsonString(evidence.control.desiredState)
           << ",\"active_request_id\":";
    WriteJsonOptional(output, evidence.control.activeRequestId);
    output << ",\"active_action\":";
    WriteJsonOptionalString(output, evidence.control.activeAction);
    output << ",\"cancellation_mode\":";
    WriteJsonOptionalString(output, evidence.control.cancellationMode);
    output << ",\"infer_before_cancel\":"
           << (evidence.control.inferBeforeCancel ? "true" : "false")
           << "},\"experiment_lifecycle_counts\":{\"pending\":"
           << evidence.lifecycleCounts.queued << ",\"paused\":"
           << evidence.lifecycleCounts.paused << ",\"running\":"
           << evidence.lifecycleCounts.running << ",\"completed\":"
           << evidence.lifecycleCounts.completed << ",\"failed\":"
           << evidence.lifecycleCounts.failed << ",\"cancelled\":"
           << evidence.lifecycleCounts.cancelled
           << "},\"phase_counts\":{\"pending\":{\"train\":"
           << evidence.phaseCounts.pendingTrain << ",\"infer\":"
           << evidence.phaseCounts.pendingInfer << ",\"analyze\":"
           << evidence.phaseCounts.pendingAnalyze
           << "},\"running\":{\"train\":"
           << evidence.phaseCounts.runningTrain << ",\"infer\":"
           << evidence.phaseCounts.runningInfer << ",\"analyze\":"
           << evidence.phaseCounts.runningAnalyze << "}},\"worker_attempt_counts\":[";
    for (size_t index = 0; index < evidence.durableWorkerCounts.size(); ++index)
    {
        const auto& count = evidence.durableWorkerCounts[index];
        if (index != 0)
            output << ',';
        output << "{\"capacity_class\":" << JsonString(count.capacityClass)
               << ",\"consuming\":" << count.consuming
               << ",\"reservations\":" << count.reservations
               << ",\"identity_mismatches\":" << count.identityMismatches
               << ",\"observed\":" << count.observed
               << ",\"checkpoint_workers\":" << count.checkpointWorkers << "}";
    }
    output << "}},\"process_observation\":";
    WriteJsonProcessEvidence(output, evidence.process);
    output << "}\n";
}

void WriteJsonExperimentEvidence(std::ostream& output,
                                 const ExperimentOperationalEvidence& evidence)
{
    const auto& job = evidence.experiment;
    output << "{\"schema\":\"expertadvisor-operational-evidence-v1\","
           << "\"kind\":\"experiment\",\"durable\":{\"experiment\":{"
           << "\"experiment_id\":" << job.experimentId
           << ",\"status\":" << JsonString(job.status)
           << ",\"phase\":" << JsonString(job.phase)
           << ",\"symbol\":" << JsonString(job.symbol)
           << ",\"prediction_horizon\":" << job.predictionHorizon
           << ",\"target_epochs\":" << job.targetEpochs
           << ",\"completed_epochs\":";
    WriteJsonOptional(output, job.completedEpochs);
    output << ",\"current_epoch\":";
    WriteJsonOptional(output, job.currentEpoch);
    output << ",\"model_id\":";
    WriteJsonOptional(output, job.modelId);
    output << ",\"operation\":"
           << JsonString(CurrentOperationForStatusJob(job))
           << "},\"checkpoint_state\":{\"interval\":"
           << job.checkpointInterval << ",\"last_checkpoint_epoch\":";
    WriteJsonOptional(output, job.lastCheckpointEpoch);
    output << ",\"last_checkpoint_model_id\":";
    WriteJsonOptional(output, job.lastCheckpointModelId);
    output << ",\"pending_evaluations\":" << job.checkpointEvalPending
           << ",\"running_evaluations\":" << job.checkpointEvalRunning
           << ",\"completed_evaluations\":" << job.checkpointEvalCompleted
           << ",\"failed_evaluations\":" << job.checkpointEvalFailed
           << "},\"worker_attempts\":[";
    for (size_t index = 0; index < evidence.attempts.size(); ++index)
    {
        const auto& attempt = evidence.attempts[index];
        if (index != 0)
            output << ',';
        output << "{\"worker_attempt_id\":" << attempt.workerAttemptId
               << ",\"worker_kind\":" << JsonString(attempt.workerKind)
               << ",\"lifecycle_phase\":" << JsonString(attempt.lifecyclePhase)
               << ",\"capacity_class\":" << JsonString(attempt.capacityClass)
               << ",\"lifecycle_state\":" << JsonString(attempt.lifecycleState)
               << ",\"scheduler_invocation_id\":";
        WriteJsonOptionalString(output, attempt.schedulerInvocationId);
        output << ",\"pid\":";
        WriteJsonOptional(output, attempt.pid);
        output << ",\"process_group_id\":";
        WriteJsonOptional(output, attempt.processGroupId);
        output << ",\"canonical_executable\":";
        WriteJsonOptionalString(output, attempt.canonicalExecutable);
        output << ",\"process_start_identity\":";
        WriteJsonOptionalString(output, attempt.processStartIdentity);
        output << ",\"ownership_origin\":" << JsonString(attempt.ownershipOrigin)
               << "}";
    }
    output << "]}},\"process_observation\":";
    WriteJsonProcessEvidence(output, evidence.process, job.experimentId);
    output << "}\n";
}
} // namespace

int PrintCompactExperimentStatus(const ProductionRuntimeDetail::SchedulerOptions& options)
{
    return PrintCompactExperimentStatusImpl(options);
}

int PrintSchedulerStatus(const ProductionRuntimeDetail::SchedulerOptions& options)
{
    return PrintSchedulerStatusImpl(options);
}

int PrintObserverSchedulerStatus(SchedulerOperationalReadModel& readModel,
                                 std::ostream& output,
                                 std::ostream& error)
{
    ScopedStatusStreams streams{output, error};
    return readModel.withReadOnlySnapshot(
        [](pqxx::read_transaction& transaction) {
            const auto globalControl =
                EA::GlobalExperimentControl::LoadControlSnapshot(transaction);
            SchedulerOptions options;
            options.schedulerExecutablePath =
                EA::ExperimentScheduler::ResolveCanonicalExecutablePath();
            return PrintSchedulerStatusFromTransaction(
                options, transaction, globalControl);
        });
}

int PrintObserverExperimentStatus(SchedulerOperationalReadModel& readModel,
                                  std::optional<long long> experimentId,
                                  std::ostream& output,
                                  std::ostream& error)
{
    ScopedStatusStreams streams{output, error};
    return readModel.withReadOnlySnapshot(
        [experimentId](pqxx::read_transaction& transaction) {
            SchedulerOptions options;
            options.statusExperimentId = experimentId;
            return PrintCompactExperimentStatusFromTransaction(options, transaction);
        });
}

int PrintObserverSchedulerEvidence(SchedulerOperationalReadModel& readModel,
                                   std::ostream& output,
                                   std::ostream& error)
{
    return readModel.withReadOnlySnapshot(
        [&output, &error](pqxx::read_transaction& transaction) {
            try
            {
                WriteJsonSchedulerEvidence(
                    output, CaptureSchedulerOperationalEvidence(transaction));
                return 0;
            }
            catch (const std::exception& exception)
            {
                error << "OBSERVER_EVIDENCE_ERROR,error=" << exception.what() << "\n";
                return 2;
            }
        });
}

int PrintObserverExperimentEvidence(SchedulerOperationalReadModel& readModel,
                                    long long experimentId,
                                    std::ostream& output,
                                    std::ostream& error)
{
    return readModel.withReadOnlySnapshot(
        [experimentId, &output, &error](pqxx::read_transaction& transaction) {
            const auto evidence = CaptureExperimentOperationalEvidence(
                transaction, experimentId);
            if (!evidence.has_value())
            {
                error << "ERROR: experiment not found.\n";
                return 1;
            }
            WriteJsonExperimentEvidence(output, *evidence);
            return 0;
        });
}

} // namespace EA::SchedulerCore
