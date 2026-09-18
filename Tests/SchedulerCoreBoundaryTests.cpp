#include "../Sources/SchedulerCore/SchedulerEngine.hpp"
#include "../Sources/SchedulerCore/SchedulerPolicy.hpp"
#include "../Sources/SchedulerCore/SchedulerRuntimeContext.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

int main()
{
    using namespace EA::SchedulerCore;

    SchedulerDaemonConfiguration configuration;
    configuration.schedulerOnce = true;
    configuration.autoEvaluateContinuations = true;
    int refreshCount = 0;
    int cycleCount = 0;
    int continuationCount = 0;
    int reportedResult = -1;
    bool reportedOwnershipLost = true;
    bool reportedShutdown = true;
    SchedulerDaemonOperations operations;
    operations.stopRequested = [] { return false; };
    operations.refreshAuthority = [&] {
        ++refreshCount;
        return true;
    };
    operations.runCycle = [&] {
        ++cycleCount;
        return 2;
    };
    operations.runAutomaticContinuationScan = [&] {
        ++continuationCount;
    };
    operations.reportAuthorityLost = [](std::string_view) {
        assert(false);
    };
    operations.sleepSeconds = [](unsigned int) { assert(false); };
    operations.reportStop = [&] (
        int result,
        bool ownershipLost,
        bool shutdownRequested) {
        reportedResult = result;
        reportedOwnershipLost = ownershipLost;
        reportedShutdown = shutdownRequested;
    };
    const SchedulerEngine engine;
    assert(engine.run(configuration, operations) == 2);
    assert(refreshCount == 1);
    assert(cycleCount == 1);
    assert(continuationCount == 1);
    assert(reportedResult == 2);
    assert(!reportedOwnershipLost);
    assert(!reportedShutdown);

    SchedulerRuntimeContext runtime;
    runtime.ownedChildren.emplace(
        77, SchedulerOwnedChild{.pid = 77, .experimentId = 618});
    assert(runtime.ownedChildren.size() == 1);
    assert(runtime.ownedChildren.at(77).experimentId == 618);

    assert(PriorityRank("high") < PriorityRank("normal"));
    assert(PriorityRank("normal") < PriorityRank("low"));
    assert(ResumeOriginRank("operator") < ResumeOriginRank("preemption"));
    assert(ResumeOriginRank("preemption") < ResumeOriginRank("none"));
    assert(CanPreempt("high", "normal"));
    assert(CanPreempt("normal", "low"));
    assert(!CanPreempt("normal", "normal"));
    assert(!CanPreempt("low", "normal"));

    bool rejected = false;
    try
    {
        (void)PriorityRank("urgent");
    }
    catch (const std::runtime_error& error)
    {
        rejected = std::string{error.what()} ==
            "invalid_scheduler_priority:urgent";
    }
    assert(rejected);
    return 0;
}
