#include "../Sources/SchedulerCore/SchedulerEngine.hpp"

#include <cassert>
#include <string_view>

int main()
{
    EA::SchedulerCore::SchedulerDaemonConfiguration configuration;
    configuration.schedulerOnce = true;

    int cycles = 0;
    EA::SchedulerCore::SchedulerDaemonOperations operations;
    operations.stopRequested = [] { return false; };
    operations.refreshAuthority = [] { return true; };
    operations.runCycle = [&] {
        ++cycles;
        return 0;
    };
    operations.runAutomaticContinuationScan = [] {};
    operations.reportAuthorityLost = [](std::string_view) {};
    operations.sleepSeconds = [](unsigned int) {};
    operations.reportStop = [](int, bool, bool) {};

    const EA::SchedulerCore::SchedulerEngine engine;
    assert(engine.run(configuration, operations) == 0);
    assert(cycles == 1);
    return 0;
}
