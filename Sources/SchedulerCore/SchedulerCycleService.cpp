#include "SchedulerCycleService.hpp"

#include <stdexcept>
#include <utility>

namespace EA::SchedulerCore
{

SchedulerCycleService::SchedulerCycleService(
    SchedulerCycleOperations operations)
    : operations_{std::move(operations)}
{
    if (!operations_.beginPoll || !operations_.prepare ||
        !operations_.runTrain || !operations_.runFinalInference ||
        !operations_.runFinalAnalysis ||
        !operations_.runCheckpointInference ||
        !operations_.runCheckpointAnalysis || !operations_.finishPoll)
    {
        throw std::invalid_argument(
            "complete scheduler cycle operations required");
    }
}

int SchedulerCycleService::runOnce()
{
    operations_.beginPoll();
    const SchedulerCyclePreparation preparation = operations_.prepare();
    if (!preparation.ready)
        return preparation.result;

    int result = preparation.result;
    if (!preparation.normalSchedulingAllowed)
    {
        if (preparation.cancellationCheckpointTrainAllowed)
            result |= operations_.runTrain(true);
        if (preparation.cancellationInferenceAllowed)
            result |= operations_.runCheckpointInference();
        operations_.finishPoll();
        return result;
    }

    result |= operations_.runTrain(false);
    result |= operations_.runFinalInference();
    result |= operations_.runFinalAnalysis();
    result |= operations_.runCheckpointInference();
    result |= operations_.runCheckpointAnalysis();
    operations_.finishPoll();
    return result;
}

} // namespace EA::SchedulerCore
