#pragma once

#include "SchedulerDaemonConfiguration.hpp"

namespace EA::SchedulerCore
{

// Production composition root for the scheduler daemon. It derives all
// runtime state and concrete adapters from the daemon configuration before
// invoking the typed SchedulerEngine boundary.
int RunProductionSchedulerDaemon(
    const SchedulerDaemonConfiguration& configuration);

} // namespace EA::SchedulerCore
