#include "../Sources/GlobalExperimentControl.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace
{

std::optional<long long> OptionValue(
    int argc,
    char* argv[],
    const std::string& prefix)
{
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument{argv[index]};
        if (argument.rfind(prefix, 0) == 0)
            return std::stoll(argument.substr(prefix.size()));
    }
    return std::nullopt;
}

bool HasArgument(int argc, char* argv[], const std::string& expected)
{
    for (int index = 1; index < argc; ++index)
    {
        if (argv[index] == expected)
            return true;
    }
    return false;
}

std::string ConnectionString()
{
    const char* database = std::getenv("LSTM_DB_NAME");
    if (database == nullptr || *database == '\0')
        throw std::runtime_error("LSTM_DB_NAME is required");
    return std::string{"dbname="} + database;
}

std::string Invocation(const std::string& action, long long identity)
{
    return "campaign-control-harness:pid:" + std::to_string(::getpid()) +
           ":action:" + action + ":identity:" +
           std::to_string(identity);
}

} // namespace

int main(int argc, char* argv[])
{
    using namespace EA::GlobalExperimentControl;
    try
    {
        const bool dryRun = HasArgument(argc, argv, "--dry-run");
        const bool confirmed = HasArgument(argc, argv, "--yes");
        if (const auto id = OptionValue(
                argc, argv, "--pause-campaign-materialization="))
        {
            CampaignMaterializationControlCommand command;
            command.materializationId = *id;
            command.dryRun = dryRun;
            command.confirmed = confirmed;
            command.invocationIdentity = Invocation("pause", *id);
            command.requesterIdentity = "isolated_test_operator";
            return RunCampaignMaterializationPauseCommand(
                ConnectionString(), command, std::cout, std::cerr);
        }
        if (const auto id = OptionValue(
                argc, argv, "--resume-campaign-materialization="))
        {
            CampaignMaterializationControlCommand command;
            command.materializationId = *id;
            command.dryRun = dryRun;
            command.confirmed = confirmed;
            command.invocationIdentity = Invocation("resume", *id);
            command.requesterIdentity = "isolated_test_operator";
            return RunCampaignMaterializationResumeCommand(
                ConnectionString(), command, std::cout, std::cerr);
        }
        if (const auto id = OptionValue(argc, argv, "--pause-experiment="))
        {
            ExperimentPauseCommand command;
            command.experimentId = *id;
            command.dryRun = dryRun;
            command.confirmed = confirmed;
            command.invocationIdentity = Invocation("individual_pause", *id);
            command.requesterIdentity = "isolated_test_operator";
            return RunExperimentPauseCommand(
                ConnectionString(), command, std::cout, std::cerr);
        }
        if (const auto id = OptionValue(argc, argv, "--resume-experiment="))
        {
            ExperimentResumeCommand command;
            command.experimentId = *id;
            command.dryRun = dryRun;
            command.confirmed = confirmed;
            command.invocationIdentity = Invocation("individual_resume", *id);
            command.requesterIdentity = "isolated_test_operator";
            return RunExperimentResumeCommand(
                ConnectionString(), command, std::cout, std::cerr);
        }
        std::cerr << "one control option is required\n";
        return 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "CAMPAIGN_CONTROL_HARNESS_ERROR,error="
                  << error.what() << "\n";
        return 2;
    }
}
