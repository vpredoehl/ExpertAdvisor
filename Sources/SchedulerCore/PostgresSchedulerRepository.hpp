#pragma once

#include "SchedulerAuthorityRepository.hpp"
#include "CheckpointEvaluationService.hpp"
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

    WorkerAttemptReservationResult reserveExperimentWorkerAttempt(
        const ExperimentWorkerAttemptReservation& reservation) override;
    WorkerAttemptReservationResult reserveCheckpointWorkerAttempt(
        const CheckpointWorkerAttemptReservation& reservation) override;
    WorkerAttemptReservationResult reserveCheckpointAnalysisAttempt(
        const CheckpointAnalysisAttemptReservation& reservation) override;

    SpawnPersistenceResult persistSpawnedWorkerAttempt(
        const SpawnedWorkerAttemptUpdate& update) override;
    LaunchFailurePersistenceResult persistWorkerAttemptLaunchFailure(
        const WorkerAttemptLaunchFailureUpdate& update) override;
    bool persistCheckpointAnalysisCompletion(
        const CheckpointAnalysisCompletionUpdate& update) override;
    CheckpointAnalysisPersistenceResult persistCheckpointAnalysisTerminalState(
        const CheckpointAnalysisTerminalUpdate& update) override;

    std::optional<ExperimentTransitionRecord> loadExperimentTransition(
        long long experimentId,
        bool forUpdate) override;
    ExperimentTransitionPersistenceResult applyExperimentTransition(
        const ExperimentTransitionUpdate& update) override;

    bool checkpointPolicySchemaAvailable();
    std::optional<EA::ExperimentScheduler::CheckpointPolicyConfig>
    loadCheckpointPolicyForEvaluation(long long parentExperimentId);
    EA::ExperimentScheduler::CheckpointPolicyConfig
    reconcileCheckpointPolicyIdentity(
        const CheckpointEvaluationRecord& evaluation,
        EA::ExperimentScheduler::CheckpointPolicyConfig config);
    CheckpointPolicyEvidenceLoadResult loadCheckpointPolicyEvidence(
        const CheckpointEvaluationRecord& evaluation);
    CheckpointPolicyPopulation loadCompletedCheckpointPolicyPopulation(
        long long parentExperimentId);
    CheckpointPolicyPopulation loadCheckpointPolicyRankPopulation(
        const CheckpointEvaluationRecord& evaluation,
        const EA::ExperimentScheduler::CheckpointPolicyConfig& config);
    PersistedCheckpointPolicyDecision persistCheckpointPolicyDecision(
        const CheckpointEvaluationRecord& evaluation,
        const EA::ExperimentScheduler::CheckpointPolicyConfig& config,
        const EA::ExperimentScheduler::CheckpointPolicyDecision& decision,
        const ValidatedCheckpointPolicyEvidence& evidence,
        const EA::ExperimentScheduler::CheckpointPolicyEvidenceIdentity&
            evidenceIdentity);
    std::string applyCheckpointPolicyStopRequest(
        const CheckpointEvaluationRecord& evaluation,
        const EA::ExperimentScheduler::CheckpointPolicyConfig& config,
        const EA::ExperimentScheduler::CheckpointPolicyDecision& decision,
        const PersistedCheckpointPolicyDecision& persisted,
        const std::string& expectedEvidenceWatermark);

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
    std::string markCheckpointPolicyDecisionSuperseded(
        long long decisionId,
        std::string_view reason);

    pqxx::transaction_base& transaction_;
};

} // namespace EA::SchedulerCore
