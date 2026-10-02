#pragma once

#include "SchedulerOperationalObservation.hpp"

namespace EA::GlobalExperimentControl::SchedulerObservationDetail
{

bool ContainsExactOptionValue(const std::string& command,
                              const std::string& option,
                              long long value);

ValidatedWorker ValidateResumedStoppedWorkerForCancellation(
    const ManagedWorker& worker,
    ProcessObserver& processes);

} // namespace EA::GlobalExperimentControl::SchedulerObservationDetail
