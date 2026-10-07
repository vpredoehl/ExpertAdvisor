#pragma once

#include <optional>

namespace EA { struct LaunchArgs; }

namespace EA::SchedulerWorkerOwnershipTestBoundary
{
std::optional<int> Run(const LaunchArgs& launchArgs);
}
