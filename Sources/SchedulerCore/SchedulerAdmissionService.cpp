#include "SchedulerAdmissionService.hpp"

#include "SchedulerPolicy.hpp"

#include "SchedulerWorkerLimits.hpp"

namespace EA::SchedulerCore
{

bool PhaseAdmissionSnapshot::hasCapacity() const noexcept
{
    return EA::ExperimentScheduler::SchedulerWorkerCapacityHasSlot(
        capacityLimit, capacityUsed);
}

int PhaseAdmissionSnapshot::availableSlots() const noexcept
{
    return EA::ExperimentScheduler::AvailableWorkerProcessSlots(
        capacityLimit, capacityUsed);
}

SchedulerAdmissionService::SchedulerAdmissionService(
    SchedulerRepository& repository)
    : repository_{repository}
{
}

PhaseAdmissionSnapshot SchedulerAdmissionService::loadPhase(
    std::string_view phase,
    int capacityLimit,
    bool cancellationOnly)
{
    return {
        repository_.loadPendingExperiments(phase, cancellationOnly),
        capacityLimit,
        repository_.countWorkersConsumingCapacity(phase)};
}

std::vector<PendingSchedulerExperimentRecord>
SchedulerAdmissionService::loadCandidates(
    std::string_view phase,
    bool cancellationOnly)
{
    return repository_.loadPendingExperiments(phase, cancellationOnly);
}

SchedulerQueueSnapshot SchedulerAdmissionService::loadQueueSnapshot()
{
    return repository_.loadQueueSnapshot();
}

int SchedulerAdmissionService::capacityUsed(
    std::string_view capacityClass)
{
    return repository_.countWorkersConsumingCapacity(capacityClass);
}

bool SchedulerAdmissionService::hasCapacity(
    std::string_view capacityClass,
    int capacityLimit)
{
    return EA::ExperimentScheduler::SchedulerWorkerCapacityHasSlot(
        capacityLimit, capacityUsed(capacityClass));
}

std::optional<PreemptionVictimRecord>
SchedulerAdmissionService::selectPreemptionVictim(
    std::string_view phase,
    std::string_view candidatePriority)
{
    if (phase != "train" && phase != "infer")
        return std::nullopt;
    return repository_.loadPreemptionVictim(
        phase, PriorityRank(candidatePriority));
}

} // namespace EA::SchedulerCore
