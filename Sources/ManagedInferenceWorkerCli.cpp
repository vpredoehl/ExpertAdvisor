#include "ManagedInferenceWorkerCli.hpp"

#include "CanonicalSymbol.hpp"
#include "Donchian20Mode.hpp"
#include "FeatureWarmupScope.hpp"
#include "LstmRuntimeLogging.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace EA::Inference
{
namespace
{
long long ParsePositiveId(const char* option, const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument(std::string{option} + " requires a positive integer value");
    std::size_t consumed = 0;
    long long parsed = 0;
    try
    {
        parsed = std::stoll(value, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument(std::string{"invalid "} + option + " value '" + value + "'");
    }
    if (consumed != value.size() || parsed <= 0)
        throw std::invalid_argument(std::string{"invalid "} + option + " value '" + value + "'");
    return parsed;
}

std::size_t ParsePositiveSize(const char* option, const std::string& value)
{
    const long long parsed = ParsePositiveId(option, value);
    if (static_cast<unsigned long long>(parsed) >
        static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
        throw std::invalid_argument(std::string{"invalid "} + option + " value '" + value + "'");
    return static_cast<std::size_t>(parsed);
}

bool SplitOption(const std::string& argument, const char* option,
                 std::string& value)
{
    const std::string prefix = std::string{option} + "=";
    if (argument.rfind(prefix, 0) != 0) return false;
    value = argument.substr(prefix.size());
    return true;
}

std::string RequireNext(int argc, const char* argv[], int& index,
                        const char* option)
{
    if (++index >= argc)
        throw std::invalid_argument(std::string{option} + " requires a value");
    return argv[index];
}

EA::RuntimeLogLevel ParseSummaryLogLevel(const std::string& value)
{
    if (value != "summary")
        throw std::invalid_argument(
            "managed inference worker requires --log-level=summary");
    return EA::RuntimeLogLevel::Summary;
}

ModelInputPreparation::DatabaseConnectionSettings DefaultDatabaseSettings()
{
    const char* forexHost = std::getenv("FOREX_DB_HOST");
    const char* forexDatabase = std::getenv("FOREX_DB_NAME");
    const char* lstmHost = std::getenv("LSTM_DB_HOST");
    const char* lstmDatabase = std::getenv("LSTM_DB_NAME");
    return {
        "hostaddr=" + std::string{forexHost && *forexHost ? forexHost : "127.0.0.1"} +
            " gssencmode=disable user=pqxx dbname=" +
            std::string{forexDatabase && *forexDatabase ? forexDatabase : "forex"},
        "hostaddr=" + std::string{lstmHost && *lstmHost ? lstmHost : "127.0.0.1"} +
            " gssencmode=disable user=pqxx dbname=" +
            std::string{lstmDatabase && *lstmDatabase ? lstmDatabase : "LSTM"}};
}
} // namespace

ManagedInferenceWorkerLaunch ParseManagedInferenceWorkerArgs(
    int argc,
    const char* argv[],
    ModelInputPreparation::DatabaseConnectionSettings database)
{
    ManagedInferenceWorkerLaunch launch;
    launch.request.database = std::move(database);
    bool infer = false;
    bool model = false;
    bool finalBinding = false;
    bool checkpointBinding = false;
    bool attempt = false;
    bool donchianMode = false;
    bool warmupScope = false;
    bool lookback = false;
    bool logLevel = false;
    bool profileHotspots = false;
    bool profileOutput = false;
    std::vector<std::string> positional;

    for (int index = 1; index < argc; ++index)
    {
        const std::string argument{argv[index]};
        std::string value;
        if (argument == "--infer")
        {
            if (infer) throw std::invalid_argument("--infer specified more than once");
            infer = true;
        }
        else if (argument == "--lstm-profile-hotspots")
        {
            if (profileHotspots) throw std::invalid_argument("--lstm-profile-hotspots specified more than once");
            profileHotspots = true;
        }
        else if (argument == "--model")
        {
            if (model) throw std::invalid_argument("--model specified more than once");
            launch.request.modelId = ParsePositiveId("--model", RequireNext(argc, argv, index, "--model"));
            model = true;
        }
        else if (argument == "--scheduler-experiment-id")
        {
            if (finalBinding) throw std::invalid_argument("--scheduler-experiment-id specified more than once");
            launch.request.finalExperimentId = ParsePositiveId("--scheduler-experiment-id", RequireNext(argc, argv, index, "--scheduler-experiment-id"));
            finalBinding = true;
        }
        else if (argument == "--scheduler-checkpoint-eval-id")
        {
            if (checkpointBinding) throw std::invalid_argument("--scheduler-checkpoint-eval-id specified more than once");
            launch.request.checkpointEvalId = ParsePositiveId("--scheduler-checkpoint-eval-id", RequireNext(argc, argv, index, "--scheduler-checkpoint-eval-id"));
            checkpointBinding = true;
        }
        else if (argument == "--scheduler-worker-attempt-id")
        {
            if (attempt) throw std::invalid_argument("--scheduler-worker-attempt-id specified more than once");
            launch.request.workerAttemptId = ParsePositiveId("--scheduler-worker-attempt-id", RequireNext(argc, argv, index, "--scheduler-worker-attempt-id"));
            attempt = true;
        }
        else if (argument == "--donchian20-mode")
        {
            if (donchianMode) throw std::invalid_argument("--donchian20-mode specified more than once");
            launch.request.requestedDonchian20Mode = ParseDonchian20Mode(RequireNext(argc, argv, index, "--donchian20-mode"));
            donchianMode = true;
        }
        else if (argument == "--feature-warmup-scope")
        {
            if (warmupScope) throw std::invalid_argument("--feature-warmup-scope specified more than once");
            launch.request.requestedFeatureWarmupScope = EA::ParseFeatureWarmupScope(RequireNext(argc, argv, index, "--feature-warmup-scope"));
            warmupScope = true;
        }
        else if (argument == "--donchian-lookback")
        {
            if (lookback) throw std::invalid_argument("--donchian-lookback specified more than once");
            launch.request.requestedDonchianLookback = ParsePositiveSize("--donchian-lookback", RequireNext(argc, argv, index, "--donchian-lookback"));
            lookback = true;
        }
        else if (argument == "--log-level")
        {
            if (logLevel) throw std::invalid_argument("--log-level specified more than once");
            launch.logLevel = ParseSummaryLogLevel(RequireNext(argc, argv, index, "--log-level"));
            logLevel = true;
        }
        else if (SplitOption(argument, "--model", value))
        {
            if (model) throw std::invalid_argument("--model specified more than once");
            launch.request.modelId = ParsePositiveId("--model", value);
            model = true;
        }
        else if (SplitOption(argument, "--scheduler-experiment-id", value))
        {
            if (finalBinding) throw std::invalid_argument("--scheduler-experiment-id specified more than once");
            launch.request.finalExperimentId = ParsePositiveId("--scheduler-experiment-id", value);
            finalBinding = true;
        }
        else if (SplitOption(argument, "--scheduler-checkpoint-eval-id", value))
        {
            if (checkpointBinding) throw std::invalid_argument("--scheduler-checkpoint-eval-id specified more than once");
            launch.request.checkpointEvalId = ParsePositiveId("--scheduler-checkpoint-eval-id", value);
            checkpointBinding = true;
        }
        else if (SplitOption(argument, "--scheduler-worker-attempt-id", value))
        {
            if (attempt) throw std::invalid_argument("--scheduler-worker-attempt-id specified more than once");
            launch.request.workerAttemptId = ParsePositiveId("--scheduler-worker-attempt-id", value);
            attempt = true;
        }
        else if (SplitOption(argument, "--donchian20-mode", value))
        {
            if (donchianMode) throw std::invalid_argument("--donchian20-mode specified more than once");
            launch.request.requestedDonchian20Mode = ParseDonchian20Mode(value);
            donchianMode = true;
        }
        else if (SplitOption(argument, "--feature-warmup-scope", value))
        {
            if (warmupScope) throw std::invalid_argument("--feature-warmup-scope specified more than once");
            launch.request.requestedFeatureWarmupScope = EA::ParseFeatureWarmupScope(value);
            warmupScope = true;
        }
        else if (SplitOption(argument, "--donchian-lookback", value))
        {
            if (lookback) throw std::invalid_argument("--donchian-lookback specified more than once");
            launch.request.requestedDonchianLookback = ParsePositiveSize("--donchian-lookback", value);
            lookback = true;
        }
        else if (SplitOption(argument, "--log-level", value))
        {
            if (logLevel) throw std::invalid_argument("--log-level specified more than once");
            launch.logLevel = ParseSummaryLogLevel(value);
            logLevel = true;
        }
        else if (SplitOption(argument, "--lstm-profile-output", value))
        {
            if (profileOutput || value.empty()) throw std::invalid_argument("invalid or duplicate --lstm-profile-output");
            profileOutput = true;
        }
        else if (argument.rfind("--", 0) == 0)
        {
            throw std::invalid_argument("unsupported managed inference worker option '" + argument + "'");
        }
        else
        {
            positional.push_back(argument);
        }
    }

    if (!infer)
        throw std::invalid_argument("managed inference worker requires --infer");
    if (!model)
        throw std::invalid_argument("managed inference worker requires --model");
    if (!attempt)
        throw std::invalid_argument("direct CLI execution of scheduler-managed work is prohibited; an exact --scheduler-worker-attempt-id is required");
    if (finalBinding == checkpointBinding)
        throw std::invalid_argument("managed inference worker requires exactly one scheduler binding");
    if (!donchianMode || !warmupScope || !lookback || !logLevel)
        throw std::invalid_argument("managed inference worker requires scheduler persisted runtime options");
    if (profileOutput && !profileHotspots)
        throw std::invalid_argument("--lstm-profile-output requires --lstm-profile-hotspots");
    if (positional.size() != 2)
        throw std::invalid_argument("managed inference worker requires <infer_start> <infer_end>");
    launch.request.fromDate = positional[0];
    launch.request.toDate = positional[1];
    return launch;
}

int RunManagedInferenceWorker(const ManagedInferenceWorkerLaunch& launch)
{
    if (launch.logLevel) EA::SetRuntimeLogLevel(*launch.logLevel);
    try
    {
        (void)RunManagedInference(launch.request);
        return 0;
    }
    catch (const std::exception& error)
    {
        if (std::string{error.what()} == "managed_inference_worker_registration_failed")
            return 125;
        std::cerr << "SCHEDULER_INFER_RESULT_PERSIST_FAILED"
                  << ",model_id=" << launch.request.modelId
                  << ",error=" << error.what() << std::endl;
        return 1;
    }
}

int RunStandaloneManagedInferenceWorkerCli(int argc, const char* argv[])
{
    try
    {
        return RunManagedInferenceWorker(ParseManagedInferenceWorkerArgs(
            argc, argv, DefaultDatabaseSettings()));
    }
    catch (const std::exception& error)
    {
        std::cerr << "Argument error: " << error.what() << "\n"
                  << "Usage: " << (argc > 0 ? argv[0] : "lstm-infer-worker")
                  << " --infer --model=MODEL_ID"
                  << " (--scheduler-experiment-id=EXPERIMENT_ID|--scheduler-checkpoint-eval-id=CHECKPOINT_EVAL_ID)"
                  << " --donchian20-mode=MODE --feature-warmup-scope=SCOPE"
                  << " --donchian-lookback=LOOKBACK --log-level=summary"
                  << " INFER_START INFER_END --scheduler-worker-attempt-id=ATTEMPT_ID\n";
        return 1;
    }
}

} // namespace EA::Inference
