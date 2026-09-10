#include "SchedulerPolicy.hpp"

#include <stdexcept>
#include <string>

namespace EA::SchedulerCore
{

int PriorityRank(std::string_view priority)
{
    if (priority == "high")
        return 0;
    if (priority == "normal")
        return 1;
    if (priority == "low")
        return 2;
    throw std::runtime_error(
        "invalid_scheduler_priority:" + std::string{priority});
}

int ResumeOriginRank(std::string_view resumeOrigin) noexcept
{
    if (resumeOrigin == "operator")
        return 0;
    if (resumeOrigin == "preemption")
        return 1;
    return 2;
}

bool CanPreempt(
    std::string_view candidatePriority,
    std::string_view victimPriority)
{
    return PriorityRank(candidatePriority) < PriorityRank(victimPriority);
}

} // namespace EA::SchedulerCore
