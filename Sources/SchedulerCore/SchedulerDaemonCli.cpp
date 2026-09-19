#include "SchedulerDaemonCli.hpp"

#include "ProductionSchedulerDaemon.hpp"
#include "SchedulerDaemonConfiguration.hpp"

#include <iostream>

#include <pqxx/pqxx>

namespace EA::SchedulerCore
{

int RunSchedulerDaemonCli(int argc, const char* argv[])
{
    SchedulerDaemonConfiguration configuration;
    try
    {
        configuration = ParseSchedulerDaemonConfiguration(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Argument error: " << error.what() << "\n"
                  << std::endl;
        std::cout.flush();
        std::cerr.flush();
        PrintSchedulerDaemonHelp(argc > 0 ? argv[0] : "LSTM_Release");
        return 1;
    }

    if (configuration.help)
    {
        PrintSchedulerDaemonHelp(argc > 0 ? argv[0] : "LSTM_Release");
        return 0;
    }

    try
    {
        return RunProductionSchedulerDaemon(configuration);
    }
    catch (const pqxx::failure& error)
    {
        std::cerr << "EXPERIMENT_DATABASE_ERROR"
                  << ",error=" << error.what()
                  << std::endl;
        std::cout.flush();
        std::cerr.flush();
        return 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "EXPERIMENT_FAILED"
                  << ",error=" << error.what()
                  << std::endl;
        std::cout.flush();
        std::cerr.flush();
        return 1;
    }
    catch (...)
    {
        std::cerr << "EXPERIMENT_FAILED,error=unknown_exception"
                  << std::endl;
        std::cout.flush();
        std::cerr.flush();
        return 1;
    }
}

} // namespace EA::SchedulerCore
