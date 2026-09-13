#pragma once

#include <functional>

namespace EA::SchedulerCore
{

struct SchedulerCyclePreparation
{
    bool ready = false;
    bool normalSchedulingAllowed = false;
    bool cancellationInferenceAllowed = false;
    bool cancellationCheckpointTrainAllowed = false;
    int result = 0;
};

// Adapter boundary for one scheduler poll. Implementations retain transaction,
// authority, admission, worker-attempt, process, and persistence semantics.
struct SchedulerCycleOperations
{
    std::function<void()> beginPoll;
    std::function<SchedulerCyclePreparation()> prepare;
    std::function<int(bool cancellationOnly)> runTrain;
    std::function<int()> runFinalInference;
    std::function<int()> runFinalAnalysis;
    std::function<int()> runCheckpointInference;
    std::function<int()> runCheckpointAnalysis;
    std::function<void()> finishPoll;
};

class SchedulerCycleService final
{
public:
    explicit SchedulerCycleService(SchedulerCycleOperations operations);

    int runOnce();

private:
    SchedulerCycleOperations operations_;
};

} // namespace EA::SchedulerCore
