#pragma once

#include "SchedulerRepository.hpp"

#include <pqxx/pqxx>

namespace EA::SchedulerCore
{

// Transaction-scoped PostgreSQL adapter.  The caller retains ownership of the
// transaction so existing lock order, authority refreshes, and commit timing
// remain unchanged.
class PostgresSchedulerRepository final : public SchedulerRepository
{
public:
    explicit PostgresSchedulerRepository(pqxx::transaction_base& transaction);

    std::vector<PendingSchedulerExperimentRecord>
    loadPendingExperiments(std::string_view phase,
                           bool cancellationOnly) override;
    std::vector<RunningSchedulerExperimentRecord>
    loadRunningExperiments() override;
    SchedulerQueueSnapshot loadQueueSnapshot() override;
    int countWorkersConsumingCapacity(
        std::string_view capacityClass) override;
    std::optional<PreemptionVictimRecord> loadPreemptionVictim(
        std::string_view phase,
        int candidatePriorityRank) override;

    SpawnPersistenceResult persistSpawnedWorkerAttempt(
        const SpawnedWorkerAttemptUpdate& update) override;
    LaunchFailurePersistenceResult persistWorkerAttemptLaunchFailure(
        const WorkerAttemptLaunchFailureUpdate& update) override;

private:
    pqxx::transaction_base& transaction_;
};

} // namespace EA::SchedulerCore
