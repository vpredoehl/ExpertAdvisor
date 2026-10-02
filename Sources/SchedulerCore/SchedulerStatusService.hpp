#pragma once

namespace EA::SchedulerCore::ProductionRuntimeDetail
{
struct SchedulerOptions;
}

namespace EA::SchedulerCore
{

// Legacy status entry points share this implementation.  The service only
// observes durable scheduler state and operating-system evidence; it does not
// acquire scheduler authority or perform lifecycle/control mutations.
int PrintCompactExperimentStatus(
    const ProductionRuntimeDetail::SchedulerOptions& options);
int PrintSchedulerStatus(
    const ProductionRuntimeDetail::SchedulerOptions& options);

} // namespace EA::SchedulerCore
