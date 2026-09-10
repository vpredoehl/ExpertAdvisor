#include "../Sources/SchedulerCore/SchedulerEngine.hpp"
#include "../Sources/SchedulerCore/SchedulerPolicy.hpp"

#include "../Headers/ExperimentScheduler.hpp"

#include <cassert>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{
int recognizedArgc = -1;
int runArgc = -1;
EA::SchedulerCore::WorkerAttemptRegistration observedRegistration;
}

namespace EA::ExperimentScheduler
{

bool IsExperimentSchedulerCommand(int argc, const char*[])
{
    recognizedArgc = argc;
    return true;
}

int RunExperimentSchedulerCli(int argc, const char*[])
{
    runArgc = argc;
    return 37;
}

bool RegisterSchedulerWorkerAttempt(
    long long workerAttemptId,
    const std::optional<long long>& experimentId,
    const std::optional<long long>& checkpointEvalId,
    const std::string& workerKind,
    const std::string& lifecyclePhase)
{
    observedRegistration = {
        workerAttemptId,
        experimentId,
        checkpointEvalId,
        workerKind,
        lifecyclePhase};
    return true;
}

} // namespace EA::ExperimentScheduler

int main()
{
    using namespace EA::SchedulerCore;

    const char* argv[] = {"LSTM_Release", "--scheduler-status"};
    const CommandInvocation invocation{2, argv};
    const SchedulerEngine engine;
    assert(engine.recognizes(invocation));
    assert(recognizedArgc == 2);
    assert(engine.run(invocation) == 37);
    assert(runArgc == 2);

    const WorkerAttemptRegistration registration{
        991,
        618,
        std::nullopt,
        "experiment",
        "train"};
    assert(engine.registerWorkerAttempt(registration));
    assert(observedRegistration.workerAttemptId == 991);
    assert(observedRegistration.experimentId == 618);
    assert(!observedRegistration.checkpointEvalId);
    assert(observedRegistration.workerKind == "experiment");
    assert(observedRegistration.lifecyclePhase == "train");

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
