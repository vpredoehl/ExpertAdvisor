#pragma once

#include <string_view>

namespace EA::SchedulerCore
{

// Persisted values are intentionally kept as strings at the repository seam.
// These helpers centralize the existing ordering without changing that schema.
int PriorityRank(std::string_view priority);
int ResumeOriginRank(std::string_view resumeOrigin) noexcept;
bool CanPreempt(
    std::string_view candidatePriority,
    std::string_view victimPriority);

} // namespace EA::SchedulerCore
