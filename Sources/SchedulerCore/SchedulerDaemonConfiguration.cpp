#include "SchedulerDaemonConfiguration.hpp"

#include "SchedulerWorkerLimits.hpp"
#include "SemanticWorkerRegistry.hpp"

#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace EA::SchedulerCore
{
namespace
{

bool SplitOptionWithValue(
    const std::string& argument,
    const std::string& optionName,
    std::string& value)
{
    const std::string prefix = optionName + "=";
    if (argument.rfind(prefix, 0) != 0)
        return false;
    value = argument.substr(prefix.size());
    return true;
}

std::string RequireNextArgument(
    int argc,
    const char* argv[],
    int& index,
    const std::string& optionName)
{
    if (index + 1 >= argc)
        throw std::invalid_argument(optionName + " requires a value");
    return argv[++index];
}

int ParsePositiveInt(
    const std::string& optionName,
    const std::string& value)
{
    std::size_t consumed = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument(
            "invalid " + optionName + " value '" + value + "'");
    }
    if (consumed != value.size() || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument(
            "invalid " + optionName + " value '" + value + "'");
    }
    return static_cast<int>(parsed);
}

} // namespace

bool IsSchedulerDaemonCommand(int argc, const char* argv[])
{
    for (int index = 1; index < argc; ++index)
    {
        if (argv[index] != nullptr &&
            std::string{argv[index]} == "--schedule-experiments")
        {
            return true;
        }
    }
    return false;
}

SchedulerDaemonConfiguration ParseSchedulerDaemonConfiguration(
    int argc,
    const char* argv[])
{
    SchedulerDaemonConfiguration configuration;
    configuration.semanticWorkerRegistryPath =
        (std::filesystem::current_path() / "Builds" /
         "SemanticWorkers" / "registry.json").string();

    bool commandSeen = false;
    bool semanticWorkerRegistrySpecified = false;
    for (int index = 0; index < argc; ++index)
    {
        if (index != 0)
            configuration.invocationCommandLine.push_back(' ');
        configuration.invocationCommandLine +=
            argv[index] != nullptr ? argv[index] : "";
    }

    for (int index = 1; index < argc; ++index)
    {
        const std::string argument{argv[index]};
        std::string value;
        if (EA::ExperimentScheduler::TryParseSchedulerWorkerLimitArgument(
                argc,
                argv,
                index,
                configuration.maxTrainProcs,
                configuration.maxInferProcs,
                configuration.maxAnalyzeProcs))
        {
            continue;
        }

        if (argument == "--schedule-experiments")
            commandSeen = true;
        else if (argument == "--help")
            configuration.help = true;
        else if (argument == "--scheduler-once")
            configuration.schedulerOnce = true;
        else if (argument == "--scheduler-verbose")
            configuration.schedulerVerbose = true;
        else if (argument == "--dry-run")
            configuration.dryRun = true;
        else if (argument == "--recover-orphans-only")
            configuration.recoverOrphansOnly = true;
        else if (argument == "--auto-generate-reports")
            configuration.autoGenerateReports = true;
        else if (argument == "--lstm-profile-hotspots")
            configuration.lstmProfileHotspots = true;
        else if (argument == "--auto-evaluate-continuations")
            configuration.autoEvaluateContinuations = true;
        else if (argument == "--auto-queue-continuations")
            configuration.autoQueueContinuations = true;
        else if (argument == "--continuation-dry-run")
            configuration.continuationDryRun = true;
        else if (argument == "--scheduler-poll-seconds")
            configuration.schedulerPollSeconds = ParsePositiveInt(
                argument,
                RequireNextArgument(argc, argv, index, argument));
        else if (argument == "--continuation-scan-seconds")
            configuration.continuationScanSeconds = ParsePositiveInt(
                argument,
                RequireNextArgument(argc, argv, index, argument));
        else if (argument == "--continuation-max-queues-per-scan")
            configuration.continuationMaxQueuesPerScan = ParsePositiveInt(
                argument,
                RequireNextArgument(argc, argv, index, argument));
        else if (argument == "--scheduler-log-dir")
            configuration.schedulerLogDir =
                RequireNextArgument(argc, argv, index, argument);
        else if (argument == "--experiment-report-dir")
            configuration.experimentReportDir =
                RequireNextArgument(argc, argv, index, argument);
        else if (argument == "--lstm-profile-output")
            configuration.lstmProfileOutputPath =
                RequireNextArgument(argc, argv, index, argument);
        else if (argument == "--semantic-worker-registry")
        {
            if (semanticWorkerRegistrySpecified)
                throw std::invalid_argument(
                    "--semantic-worker-registry specified more than once");
            configuration.semanticWorkerRegistryPath =
                RequireNextArgument(argc, argv, index, argument);
            semanticWorkerRegistrySpecified = true;
        }
        else if (argument == "--legacy-layout6-infer-worker")
        {
            if (configuration.legacyLayout6InferWorkerPath)
                throw std::invalid_argument(
                    "--legacy-layout6-infer-worker specified more than once");
            configuration.legacyLayout6InferWorkerPath =
                EA::Scheduler::ValidateAndCanonicalizeWorkerExecutable(
                    RequireNextArgument(argc, argv, index, argument),
                    argument);
        }
        else if (SplitOptionWithValue(
                     argument, "--scheduler-poll-seconds", value))
            configuration.schedulerPollSeconds =
                ParsePositiveInt("--scheduler-poll-seconds", value);
        else if (SplitOptionWithValue(
                     argument, "--continuation-scan-seconds", value))
            configuration.continuationScanSeconds =
                ParsePositiveInt("--continuation-scan-seconds", value);
        else if (SplitOptionWithValue(
                     argument,
                     "--continuation-max-queues-per-scan",
                     value))
            configuration.continuationMaxQueuesPerScan = ParsePositiveInt(
                "--continuation-max-queues-per-scan", value);
        else if (SplitOptionWithValue(argument, "--scheduler-log-dir", value))
            configuration.schedulerLogDir = value;
        else if (SplitOptionWithValue(
                     argument, "--experiment-report-dir", value))
            configuration.experimentReportDir = value;
        else if (SplitOptionWithValue(
                     argument, "--lstm-profile-output", value))
            configuration.lstmProfileOutputPath = value;
        else if (SplitOptionWithValue(
                     argument, "--semantic-worker-registry", value))
        {
            if (semanticWorkerRegistrySpecified)
                throw std::invalid_argument(
                    "--semantic-worker-registry specified more than once");
            if (value.empty())
                throw std::invalid_argument(
                    "--semantic-worker-registry requires a path");
            configuration.semanticWorkerRegistryPath = value;
            semanticWorkerRegistrySpecified = true;
        }
        else if (SplitOptionWithValue(
                     argument, "--legacy-layout6-infer-worker", value))
        {
            if (configuration.legacyLayout6InferWorkerPath)
                throw std::invalid_argument(
                    "--legacy-layout6-infer-worker specified more than once");
            configuration.legacyLayout6InferWorkerPath =
                EA::Scheduler::ValidateAndCanonicalizeWorkerExecutable(
                    value, "--legacy-layout6-infer-worker");
        }
        else if (argument.rfind("--", 0) == 0)
            throw std::invalid_argument(
                "unknown scheduler option '" + argument + "'");
        else
            throw std::invalid_argument(
                "unexpected positional scheduler argument '" + argument +
                "'");
    }

    if (!commandSeen)
        throw std::invalid_argument(
            "expected exactly one experiment scheduler command");
    if (configuration.autoQueueContinuations)
        configuration.autoEvaluateContinuations = true;
    if (configuration.continuationDryRun &&
        !configuration.autoEvaluateContinuations)
    {
        throw std::invalid_argument(
            "--continuation-dry-run requires --auto-evaluate-continuations "
            "or --auto-queue-continuations");
    }
    return configuration;
}

void PrintSchedulerDaemonHelp(const char* executable)
{
    const std::string name = executable != nullptr
        ? executable
        : "LSTM_Release";
    std::cout
        << "Usage: " << name
        << " --schedule-experiments [--max-train-procs=N] "
        << "[--max-infer-procs=N] [--max-analyze-procs=N] "
        << "[--scheduler-poll-seconds=N] [--scheduler-once] "
        << "[--semantic-worker-registry=/absolute/path/to/registry.json] "
        << "[--legacy-layout6-infer-worker=/absolute/path/to/LSTM_Release] "
        << "[--scheduler-log-dir=PATH] [--auto-generate-reports] "
        << "[--experiment-report-dir=PATH] [--lstm-profile-hotspots] "
        << "[--lstm-profile-output=PATH] [--scheduler-verbose] [--dry-run] "
        << "[--recover-orphans-only] [--auto-evaluate-continuations] "
        << "[--auto-queue-continuations] [--continuation-scan-seconds=N] "
        << "[--continuation-max-queues-per-scan=N] "
        << "[--continuation-dry-run]\n"
        << "Scheduler worker limits accept non-negative integers. Zero "
        << "prevents new workers in that capacity class without stopping "
        << "the scheduler or existing workers.\n"
        << "Continuation automation is disabled by default. Evaluation-only "
        << "persists/reuses Phase 3A decisions without creating children. "
        << "--auto-queue-continuations implies evaluation; children enter "
        << "the normal pending/train queue and obey ordinary scheduler "
        << "capacity. Defaults: scan interval 300 seconds, maximum 1 queue "
        << "per scan. --continuation-dry-run computes and logs "
        << "evaluation/queue proposals without database writes.\n";
}

} // namespace EA::SchedulerCore
