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
    if (argc < 2 || argc > 4)
    {
        error << "Usage: lstm-observer scheduler | experiment [EXPERIMENT_ID] | "
                 "evidence scheduler | evidence experiment EXPERIMENT_ID | "
                 "evidence inference EXPERIMENT_ID | "
                 "evidence profitability EXPERIMENT_ID\n";
        return 1;
    }
    const std::string operation{argv[1]};
    if (operation == "evidence")
    {
        if (argc < 3 || argc > 4)
        {
            error << "evidence requires scheduler, experiment, inference, or "
                     "profitability EXPERIMENT_ID\n";
            return 1;
        }
        const std::string evidenceKind{argv[2]};
        if (evidenceKind != "scheduler" && evidenceKind != "experiment" &&
            evidenceKind != "inference" && evidenceKind != "profitability")
        {
            error << "unsupported evidence kind: " << argv[2] << "\n";
            return 1;
        }
        if (evidenceKind == "scheduler" && argc != 3)
        {
            error << "evidence scheduler accepts no arguments\n";
            return 1;
        }
        if ((evidenceKind == "experiment" || evidenceKind == "inference" ||
             evidenceKind == "profitability") && argc != 4)
        {
            error << "evidence " << evidenceKind
                  << " requires EXPERIMENT_ID\n";
            return 1;
        }
        const auto experimentId = evidenceKind != "scheduler"
            ? PositiveId(argv[3])
            : std::optional<long long>{};
        if (evidenceKind != "scheduler" && !experimentId.has_value())
        {
            error << "EXPERIMENT_ID must be a positive integer\n";
            return 1;
        }
        try
        {
            SchedulerOperationalReadModel readModel{ConnectionString()};
            if (evidenceKind == "scheduler")
                return PrintObserverSchedulerEvidence(readModel, output, error);
            if (evidenceKind == "experiment")
                return PrintObserverExperimentEvidence(
                    readModel, *experimentId, output, error);
            if (evidenceKind == "inference")
                return PrintObserverInferenceEvidence(
                    readModel, *experimentId, output, error);
            return PrintObserverProfitabilityEvidence(
                readModel, *experimentId, output, error);
        }
        catch (const std::exception& exception)
        {
            error << "OBSERVER_DATABASE_ERROR,error=" << exception.what() << "\n";
            return 2;
        }
    }
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
