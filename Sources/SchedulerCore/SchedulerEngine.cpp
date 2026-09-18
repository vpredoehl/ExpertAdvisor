#include "SchedulerEngine.hpp"

#include "SchedulerAuthorityService.hpp"

#include <chrono>
#include <stdexcept>

namespace EA::SchedulerCore
{

namespace
{

void RequireOperations(const SchedulerDaemonOperations& operations)
{
    if (!operations.stopRequested ||
        !operations.refreshAuthority ||
        !operations.runCycle ||
        !operations.runAutomaticContinuationScan ||
        !operations.reportAuthorityLost ||
        !operations.sleepSeconds ||
        !operations.reportStop)
    {
        throw std::invalid_argument(
            "scheduler daemon operations are incomplete");
    }
}

} // namespace

int SchedulerEngine::run(
    const SchedulerDaemonConfiguration& configuration,
    const SchedulerDaemonOperations& operations) const
{
    RequireOperations(operations);
    int result = 0;
    bool ownershipLost = false;
    auto nextContinuationScan = std::chrono::steady_clock::time_point::min();

    do
    {
        if (operations.stopRequested())
            break;
        if (!operations.refreshAuthority())
        {
            ownershipLost = true;
            result = 4;
            break;
        }
        try
        {
            result |= operations.runCycle();
        }
        catch (const SchedulerAuthorityLost& error)
        {
            operations.reportAuthorityLost(error.what());
            ownershipLost = true;
            result = 4;
            break;
        }

        if (configuration.autoEvaluateContinuations &&
            std::chrono::steady_clock::now() >= nextContinuationScan)
        {
            try
            {
                operations.runAutomaticContinuationScan();
                nextContinuationScan =
                    std::chrono::steady_clock::now() +
                    std::chrono::seconds(
                        configuration.continuationScanSeconds);
            }
            catch (const SchedulerAuthorityLost& error)
            {
                operations.reportAuthorityLost(error.what());
                ownershipLost = true;
                result = 4;
                break;
            }
        }

        if (configuration.schedulerOnce)
            break;
        for (int elapsed = 0;
             elapsed < configuration.schedulerPollSeconds &&
             !operations.stopRequested();
             ++elapsed)
        {
            operations.sleepSeconds(1);
            if ((elapsed + 1) % (kSchedulerLeaseSeconds / 3) == 0 &&
                !operations.refreshAuthority())
            {
                ownershipLost = true;
                result = 4;
                break;
            }
        }
        if (ownershipLost)
            break;
    } while (true);

    operations.reportStop(
        result, ownershipLost, operations.stopRequested());
    return result;
}

} // namespace EA::SchedulerCore
