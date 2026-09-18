#pragma once

#include "SchedulerDaemonConfiguration.hpp"

#include <functional>
#include <string_view>

namespace EA::SchedulerCore
{

// Concrete scheduler composition supplies the already-extracted services for
// one cycle; the engine owns daemon timing, authority refresh cadence, stop
// handling, continuation scan cadence, and terminal result aggregation.
struct SchedulerDaemonOperations
{
    std::function<bool()> stopRequested;
    std::function<bool()> refreshAuthority;
    std::function<int()> runCycle;
    std::function<void()> runAutomaticContinuationScan;
    std::function<void(std::string_view)> reportAuthorityLost;
    std::function<void(unsigned int)> sleepSeconds;
    std::function<void(int, bool, bool)> reportStop;
};

class SchedulerEngine final
{
public:
    int run(
        const SchedulerDaemonConfiguration& configuration,
        const SchedulerDaemonOperations& operations) const;
};

} // namespace EA::SchedulerCore
