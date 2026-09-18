#include "../Sources/SchedulerCore/SchedulerDaemonConfiguration.hpp"

#include <cassert>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

EA::SchedulerCore::SchedulerDaemonConfiguration Parse(
    std::initializer_list<const char*> arguments)
{
    std::vector<const char*> argv{arguments};
    return EA::SchedulerCore::ParseSchedulerDaemonConfiguration(
        static_cast<int>(argv.size()), argv.data());
}

void ExpectInvalid(
    std::initializer_list<const char*> arguments,
    const std::string& expected)
{
    bool rejected = false;
    try
    {
        (void)Parse(arguments);
    }
    catch (const std::invalid_argument& error)
    {
        rejected = std::string{error.what()} == expected;
    }
    assert(rejected);
}

} // namespace

int main()
{
    using namespace EA::SchedulerCore;

    const char* daemonArgv[] = {
        "LSTM_Release", "--schedule-experiments"};
    assert(IsSchedulerDaemonCommand(2, daemonArgv));
    const char* statusArgv[] = {"LSTM_Release", "--scheduler-status"};
    assert(!IsSchedulerDaemonCommand(2, statusArgv));

    const auto defaults = Parse({
        "LSTM_Release", "--schedule-experiments"});
    assert(defaults.maxTrainProcs == 1);
    assert(!defaults.help);
    assert(defaults.maxInferProcs == 1);
    assert(defaults.maxAnalyzeProcs == 1);
    assert(defaults.schedulerPollSeconds == 30);
    assert(defaults.schedulerLogDir == "experiment_logs");
    assert(defaults.experimentReportDir == "experiment_reports");
    assert(!defaults.autoGenerateReports);
    assert(!defaults.lstmProfileHotspots);
    assert(!defaults.lstmProfileOutputPath);
    assert(!defaults.schedulerVerbose);
    assert(!defaults.schedulerOnce);
    assert(!defaults.dryRun);
    assert(!defaults.recoverOrphansOnly);
    assert(!defaults.autoEvaluateContinuations);
    assert(!defaults.autoQueueContinuations);
    assert(!defaults.continuationDryRun);
    assert(defaults.continuationScanSeconds == 300);
    assert(defaults.continuationMaxQueuesPerScan == 1);
    assert(
        defaults.semanticWorkerRegistryPath ==
        (std::filesystem::current_path() / "Builds" /
         "SemanticWorkers" / "registry.json").string());
    assert(!defaults.legacyLayout6InferWorkerPath);
    assert(
        defaults.invocationCommandLine ==
        "LSTM_Release --schedule-experiments");

    const auto configured = Parse({
        "LSTM_Release",
        "--schedule-experiments",
        "--max-train-procs", "0",
        "--max-infer-procs=2",
        "--max-analyze-procs", "3",
        "--scheduler-poll-seconds=7",
        "--scheduler-once",
        "--semantic-worker-registry=/tmp/semantic-workers.json",
        "--legacy-layout6-infer-worker=/bin/sh",
        "--scheduler-log-dir", "/tmp/scheduler logs",
        "--auto-generate-reports",
        "--experiment-report-dir=/tmp/reports",
        "--lstm-profile-hotspots",
        "--lstm-profile-output", "/tmp/profile.csv",
        "--scheduler-verbose",
        "--dry-run",
        "--recover-orphans-only",
        "--auto-evaluate-continuations",
        "--auto-queue-continuations",
        "--continuation-scan-seconds", "11",
        "--continuation-max-queues-per-scan=4",
        "--continuation-dry-run"});
    assert(configured.maxTrainProcs == 0);
    assert(configured.maxInferProcs == 2);
    assert(configured.maxAnalyzeProcs == 3);
    assert(configured.schedulerPollSeconds == 7);
    assert(configured.schedulerOnce);
    assert(
        configured.semanticWorkerRegistryPath ==
        "/tmp/semantic-workers.json");
    assert(configured.legacyLayout6InferWorkerPath);
    assert(
        std::filesystem::equivalent(
            *configured.legacyLayout6InferWorkerPath, "/bin/sh"));
    assert(configured.schedulerLogDir == "/tmp/scheduler logs");
    assert(configured.autoGenerateReports);
    assert(configured.experimentReportDir == "/tmp/reports");
    assert(configured.lstmProfileHotspots);
    assert(configured.lstmProfileOutputPath == "/tmp/profile.csv");
    assert(configured.schedulerVerbose);
    assert(configured.dryRun);
    assert(configured.recoverOrphansOnly);
    assert(configured.autoEvaluateContinuations);
    assert(configured.autoQueueContinuations);
    assert(configured.continuationScanSeconds == 11);
    assert(configured.continuationMaxQueuesPerScan == 4);
    assert(configured.continuationDryRun);

    const auto impliedEvaluation = Parse({
        "LSTM_Release",
        "--schedule-experiments",
        "--auto-queue-continuations"});
    assert(impliedEvaluation.autoEvaluateContinuations);

    ExpectInvalid(
        {"LSTM_Release", "--scheduler-once"},
        "expected exactly one experiment scheduler command");
    ExpectInvalid(
        {"LSTM_Release", "--schedule-experiments", "--scheduler-poll-seconds=0"},
        "invalid --scheduler-poll-seconds value '0'");
    ExpectInvalid(
        {"LSTM_Release", "--schedule-experiments", "--max-train-procs=-1"},
        "invalid --max-train-procs value '-1'");
    ExpectInvalid(
        {"LSTM_Release", "--schedule-experiments", "--continuation-dry-run"},
        "--continuation-dry-run requires --auto-evaluate-continuations or "
        "--auto-queue-continuations");
    ExpectInvalid(
        {"LSTM_Release", "--schedule-experiments", "--unknown"},
        "unknown scheduler option '--unknown'");
    ExpectInvalid(
        {"LSTM_Release", "--schedule-experiments", "positional"},
        "unexpected positional scheduler argument 'positional'");
    const auto repeatedCommand = Parse({
        "LSTM_Release", "--schedule-experiments", "--schedule-experiments"});
    assert(!repeatedCommand.help);
    const auto help = Parse({
        "LSTM_Release", "--schedule-experiments", "--help"});
    assert(help.help);
    ExpectInvalid(
        {"LSTM_Release", "--schedule-experiments",
         "--semantic-worker-registry=/tmp/one",
         "--semantic-worker-registry=/tmp/two"},
        "--semantic-worker-registry specified more than once");

    return 0;
}
