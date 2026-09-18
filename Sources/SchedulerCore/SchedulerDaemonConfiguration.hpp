#pragma once

#include "SchedulerDaemonDefaults.hpp"

#include <optional>
#include <string>

namespace EA::SchedulerCore
{

// The complete validated input contract for the scheduler daemon.  General
// experiment, operator, status, and recommendation commands deliberately do
// not belong here.
struct SchedulerDaemonConfiguration
{
    bool help = false;
    int maxTrainProcs = kDefaultSchedulerMaxTrainProcs;
    int maxInferProcs = kDefaultSchedulerMaxInferProcs;
    int maxAnalyzeProcs = kDefaultSchedulerMaxAnalyzeProcs;
    int schedulerPollSeconds = 30;
    std::string schedulerLogDir = "experiment_logs";
    std::string experimentReportDir = "experiment_reports";
    bool autoGenerateReports = false;
    bool lstmProfileHotspots = false;
    std::optional<std::string> lstmProfileOutputPath;
    bool schedulerVerbose = false;
    bool schedulerOnce = false;
    bool dryRun = false;
    bool recoverOrphansOnly = false;
    bool autoEvaluateContinuations = false;
    bool autoQueueContinuations = false;
    bool continuationDryRun = false;
    int continuationScanSeconds = 300;
    int continuationMaxQueuesPerScan = 1;
    std::string semanticWorkerRegistryPath;
    std::optional<std::string> legacyLayout6InferWorkerPath;

    // Invocation evidence is captured once by the compatibility adapter and
    // consumed by scheduler authority registration.
    std::string invocationCommandLine;
};

bool IsSchedulerDaemonCommand(int argc, const char* argv[]);
SchedulerDaemonConfiguration ParseSchedulerDaemonConfiguration(
    int argc,
    const char* argv[]);
void PrintSchedulerDaemonHelp(const char* executable);

} // namespace EA::SchedulerCore
