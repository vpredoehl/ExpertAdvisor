#include "SchedulerRepository.hpp"

#include "SchedulerPolicy.hpp"

#include <tuple>

namespace EA::SchedulerCore
{

bool PendingSelectionPrecedes(
    const PendingSelectionMetadata& lhs,
    const PendingSelectionMetadata& rhs)
{
    return std::tuple{
               PriorityRank(lhs.priority),
               ResumeOriginRank(lhs.resumeOrigin),
               lhs.updatedAt,
               lhs.experimentId} <
           std::tuple{
               PriorityRank(rhs.priority),
               ResumeOriginRank(rhs.resumeOrigin),
               rhs.updatedAt,
               rhs.experimentId};
}

bool HasCompleteSpawnedWorkerAttemptUpdate(
    const SpawnedWorkerAttemptUpdate& update) noexcept
{
    return update.workerAttemptId > 0 &&
           !update.schedulerInvocationId.empty() &&
           update.schedulerFencingToken > 0 &&
           update.experimentId > 0 &&
           !update.phase.empty() &&
           update.workerPid > 0 &&
           !update.processStartIdentity.empty() &&
           !update.canonicalExecutablePath.empty() &&
           !update.commandLine.empty();
}

} // namespace EA::SchedulerCore
