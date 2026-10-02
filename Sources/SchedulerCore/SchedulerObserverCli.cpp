#include "SchedulerObserverCli.hpp"

#include "SchedulerOperationalReadModel.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

namespace EA::SchedulerCore
{
namespace
{
std::string ConnectionString()
{
    const char* host = std::getenv("LSTM_DB_HOST");
    const char* database = std::getenv("LSTM_DB_NAME");
    return "hostaddr=" + std::string{host && *host ? host : "127.0.0.1"} +
        " gssencmode=disable user=pqxx dbname=" +
        std::string{database && *database ? database : "LSTM"};
}

std::optional<long long> PositiveId(const std::string& value)
{
    try
    {
        std::size_t consumed = 0;
        const long long id = std::stoll(value, &consumed);
        if (consumed != value.size() || id <= 0)
            return std::nullopt;
        return id;
    }
    catch (...)
    {
        return std::nullopt;
    }
}
} // namespace

int RunSchedulerObserverCli(int argc, const char* argv[], std::ostream& output,
                            std::ostream& error)
{
    if (argc < 2 || argc > 3)
    {
        error << "Usage: lstm-observer scheduler | experiment [EXPERIMENT_ID]\n";
        return 1;
    }
    const std::string operation{argv[1]};
    if (operation != "scheduler" && operation != "experiment")
    {
        error << "unsupported observer operation: " << argv[1] << "\n";
        return 1;
    }
    if (operation == "scheduler" && argc != 2)
    {
        error << "scheduler accepts no arguments\n";
        return 1;
    }
    std::optional<long long> experimentId;
    if (argc == 3)
    {
        experimentId = PositiveId(argv[2]);
        if (!experimentId.has_value())
        {
            error << "EXPERIMENT_ID must be a positive integer\n";
            return 1;
        }
    }
    try
    {
        SchedulerOperationalReadModel readModel{ConnectionString()};
        if (operation == "scheduler")
            return PrintObserverSchedulerStatus(readModel, output, error);
        return PrintObserverExperimentStatus(
            readModel, experimentId, output, error);
    }
    catch (const std::exception& exception)
    {
        error << "OBSERVER_DATABASE_ERROR,error=" << exception.what() << "\n";
        return 2;
    }
}

} // namespace EA::SchedulerCore
