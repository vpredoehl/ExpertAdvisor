#pragma once

#include "SchedulerAuthorityRepository.hpp"
#include "SchedulerRepository.hpp"

#include <pqxx/pqxx>

namespace EA::SchedulerCore
{

// Transaction-scoped PostgreSQL adapter.  The caller retains ownership of the
// transaction so existing lock order, authority refreshes, and commit timing
// remain unchanged.
class PostgresSchedulerRepository final
    : public SchedulerRepository,
      public SchedulerAuthorityRepository
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

    void acquireAuthorityCoordinationLock() override;
    std::optional<SchedulerProtocolState>
    loadSchedulerProtocolForUpdate() override;
    void registerSchedulerInvocation(
        const SchedulerInvocationRecord& invocation) override;
    std::optional<SchedulerLeaseState>
    loadSchedulerLeaseForUpdate() override;
    void rejectSchedulerInvocation(
        std::string_view schedulerInvocationId,
        std::string_view terminalReason) override;
    void markSchedulerInvocationCrashed(
        std::string_view schedulerInvocationId,
        std::string_view terminalReason) override;
    bool acquireSchedulerLease(
        const SchedulerLeaseAcquisition& acquisition) override;
    void markSchedulerInvocationOwner(
        std::string_view schedulerInvocationId) override;
    bool renewSchedulerLease(
        const SchedulerAuthorityIdentity& authority,
        int leaseSeconds) override;
    void touchSchedulerInvocation(
        std::string_view schedulerInvocationId) override;
    bool releaseSchedulerLease(
        const SchedulerAuthorityIdentity& authority,
        std::string_view reason) override;
    void markSchedulerInvocationReleased(
        std::string_view schedulerInvocationId,
        std::string_view reason) override;
    bool completeSchedulerProtocolCutover(
        const SchedulerProtocolCutoverUpdate& update) override;
    bool displaceSchedulerAuthorityForTest(
        const SchedulerAuthorityIdentity& authority,
        std::string_view foreignSchedulerInvocationId,
        std::string_view transitionReason,
        int leaseSeconds) override;

private:
    pqxx::transaction_base& transaction_;
};

} // namespace EA::SchedulerCore
