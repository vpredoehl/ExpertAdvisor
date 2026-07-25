#include "CampaignOperationsDispatchService.hpp"
#include "ExperimentRecommendationCampaignExecutionRepository.hpp"

#include <array>
#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace EA::CampaignOperations
{
namespace
{

void Invoke(const DispatchTestHook& hook, DispatchTestInjectionPoint point,
    pqxx::transaction_base& transaction)
{
    if (!hook) return;
    try
    {
        hook(point);
    }
    catch (const DispatchTestSqlState& injected)
    {
        transaction.exec(
            "DO $phase3_test$ BEGIN RAISE EXCEPTION "
            "'Campaign Operations Phase 3 deterministic test fault' "
            "USING ERRCODE = " +
            transaction.quote(injected.sqlState()) +
            "; END $phase3_test$;");
    }
}

void InvokeAfterCommit(
    const DispatchTestHook& hook, DispatchTestInjectionPoint point)
{
    if (hook) hook(point);
}

std::string GenerateOpaqueLeaseToken()
{
    std::array<unsigned char, 32> bytes{};
    std::random_device source;
    for (auto& byte : bytes)
        byte = static_cast<unsigned char>(source());
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const auto byte : bytes)
        result << std::setw(2) << static_cast<unsigned int>(byte);
    return result.str();
}

void ValidateSafetyGate(pqxx::connection& connection,
    const IsolatedDispatchSafetyGate& gate)
{
    if (gate.acknowledgement !=
            kCampaignOperationsPhase3TestAcknowledgement ||
        gate.expectedDatabase.rfind(
            kCampaignOperationsPhase3TestDatabasePrefix, 0U) != 0U)
        throw std::invalid_argument(
            "campaign_operations_phase3_test_gate_required");
    pqxx::read_transaction transaction{connection};
    const auto database = transaction.exec(
        "SELECT current_database();").one_row()[0].as<std::string>();
    if (database != gate.expectedDatabase)
        throw std::invalid_argument(
            "campaign_operations_phase3_test_database_mismatch");
}

void SetRole(pqxx::transaction_base& transaction, const char* role)
{
    transaction.exec("SET LOCAL ROLE " + transaction.quote_name(role) + ";");
}

DispatchServiceResult ExistingResult(
    const PersistedDispatchBinding& existing, int transactionAttempts,
    UncertainCommitRecoveryClassification recovery)
{
    return {DispatchResultClassification::existingIdentical,
        DownstreamEvidenceClassification::
            completeCampaignOperationsBinding,
        ExactReplayDisposition::authoritativeExisting, recovery,
        existing.bindingSet.requestId,
        existing.bindingSet.identity.hash(), transactionAttempts,
        DispatchServiceFailureClassification::none,
        "dispatch_existing_identical"};
}

std::optional<DispatchServiceResult> LookupBeforeRetry(
    pqxx::connection& connection, OperationalRequestId requestId,
    int transactionAttempts,
    UncertainCommitRecoveryClassification recovery)
{
    pqxx::read_transaction transaction{connection};
    SetRole(transaction, kCampaignOperationsPhase5TransactionalRole);
    const auto existing =
        FindAndValidateCompleteDispatchBinding(transaction, requestId);
    if (existing)
        return ExistingResult(*existing, transactionAttempts, recovery);
    const auto outcome = FindLatestDispatchOutcome(transaction, requestId);
    if (!outcome) return std::nullopt;
    if (outcome->result == DispatchResultClassification::createdAndBound ||
        outcome->result ==
            DispatchResultClassification::adoptedExistingPendingAndBound ||
        outcome->result == DispatchResultClassification::existingIdentical)
        throw std::runtime_error(
            "campaign_operations_success_outcome_without_complete_binding");
    return DispatchServiceResult{outcome->result,
        outcome->downstreamEvidence,
        ExactReplayDisposition::authoritativeExisting,
        outcome->recovery, requestId,
        outcome->bindingSetIdentityHash.value_or(std::string{}),
        transactionAttempts, DispatchServiceFailureClassification::none,
        outcome->diagnosticCode};
}

bool Retryable(const pqxx::sql_error& error)
{
    return error.sqlstate() == "40001" || error.sqlstate() == "40P01";
}

void RetryBackoff(int attemptNumber)
{
    std::random_device source;
    std::uniform_int_distribution<int> jitter(1, 5);
    std::this_thread::sleep_for(std::chrono::milliseconds(
        attemptNumber * 2 + jitter(source)));
}

std::optional<DispatchLease> LookupRecoverableLease(
    pqxx::connection& connection, OperationalRequestId requestId)
{
    pqxx::read_transaction transaction{connection};
    SetRole(transaction, kCampaignOperationsDispatcherRole);
    return FindRecoverableDispatchLease(transaction, requestId);
}

DispatchServiceResult RetryExhausted(OperationalRequestId requestId,
    int transactionAttempts, const std::string& sqlState)
{
    return {
        DispatchResultClassification::rejected,
        DownstreamEvidenceClassification::noPhase5Evidence,
        ExactReplayDisposition::provenAbsent,
        UncertainCommitRecoveryClassification::provenNoCommit,
        requestId, {}, transactionAttempts,
        DispatchServiceFailureClassification::
            transientDatabaseRetryExhausted,
        "dispatch_retry_exhausted_" + sqlState};
}

SemanticConflictClassification ConflictFor(
    DownstreamEvidenceClassification classification)
{
    switch (classification)
    {
        case DownstreamEvidenceClassification::partialPhase5Evidence:
            return SemanticConflictClassification::partialDownstreamEvidence;
        case DownstreamEvidenceClassification::pausedOnlyEvidence:
            return SemanticConflictClassification::pausedOnlyEvidence;
        case DownstreamEvidenceClassification::progressedUnboundEvidence:
            return SemanticConflictClassification::progressedUnboundEvidence;
        case DownstreamEvidenceClassification::causallyAmbiguous:
            return SemanticConflictClassification::causalityMismatch;
        default: return SemanticConflictClassification::bindingMismatch;
    }
}

} // namespace

DispatchServiceResult DispatchOneRequestForIsolatedTest(
    const std::string& connectionString, OperationalRequestId requestId,
    int expectedRequestVersion, const ActorIdentity& dispatcher,
    const IsolatedDispatchSafetyGate& safetyGate,
    DispatchTestHook testHook)
{
    pqxx::connection connection{connectionString};
    ValidateSafetyGate(connection, safetyGate);

    if (const auto existing = LookupBeforeRetry(connection, requestId, 0,
            UncertainCommitRecoveryClassification::
                completeAuthoritativeBinding))
        return *existing;

    std::optional<DispatchLease> lease =
        LookupRecoverableLease(connection, requestId);
    const LeaseTokenDigest leaseTokenDigest = lease
        ? lease->acquisition.leaseTokenDigest
        : LeaseTokenDigest::Derive(GenerateOpaqueLeaseToken());
    for (int acquisitionAttempt = 1;
         !lease &&
         acquisitionAttempt <=
             kCampaignOperationsDispatchMaximumTransactionRetries;
         ++acquisitionAttempt)
    {
        pqxx::connection acquisitionConnection{connectionString};
        ValidateSafetyGate(acquisitionConnection, safetyGate);
        if (const auto existing = LookupBeforeRetry(
                acquisitionConnection, requestId, acquisitionAttempt,
                UncertainCommitRecoveryClassification::
                    completeAuthoritativeBinding))
            return *existing;
        if (const auto recovered =
                LookupRecoverableLease(acquisitionConnection, requestId))
        {
            lease.emplace(*recovered);
            break;
        }
        try
        {
            pqxx::work transaction{acquisitionConnection};
            SetRole(transaction, kCampaignOperationsDispatcherRole);
            DispatchLease acquired = AcquireDispatchLeaseInTransaction(
                transaction, requestId, expectedRequestVersion,
                leaseTokenDigest, dispatcher,
                [&](DispatchTestInjectionPoint point)
                {
                    Invoke(testHook, point, transaction);
                });
            Invoke(testHook,
                DispatchTestInjectionPoint::beforeAcquisitionCommit,
                transaction);
            transaction.commit();
            lease.emplace(std::move(acquired));
        }
        catch (const pqxx::sql_error& error)
        {
            if (!Retryable(error)) throw;
            if (acquisitionAttempt >=
                kCampaignOperationsDispatchMaximumTransactionRetries)
                return RetryExhausted(
                    requestId, acquisitionAttempt, error.sqlstate());
            RetryBackoff(acquisitionAttempt);
        }
        catch (const pqxx::broken_connection&)
        {
            pqxx::connection recoveryConnection{connectionString};
            ValidateSafetyGate(recoveryConnection, safetyGate);
            if (const auto existing = LookupBeforeRetry(
                    recoveryConnection, requestId, acquisitionAttempt,
                    UncertainCommitRecoveryClassification::
                        completeAuthoritativeBinding))
                return *existing;
            if (const auto recovered =
                    LookupRecoverableLease(recoveryConnection, requestId))
            {
                lease.emplace(*recovered);
                break;
            }
            if (acquisitionAttempt >=
                kCampaignOperationsDispatchMaximumTransactionRetries)
                return {
                    DispatchResultClassification::reconciliationRequired,
                    DownstreamEvidenceClassification::causallyAmbiguous,
                    ExactReplayDisposition::reconciliationRequired,
                    UncertainCommitRecoveryClassification::ambiguousEvidence,
                    requestId, {}, acquisitionAttempt,
                    DispatchServiceFailureClassification::
                        commitOutcomeUnknown,
                    "dispatch_acquisition_commit_outcome_unknown"};
            RetryBackoff(acquisitionAttempt);
        }
    }
    if (!lease)
        throw std::runtime_error(
            "campaign_operations_dispatch_acquisition_retry_exhausted");

    for (int attemptNumber = 1;
         attemptNumber <= kCampaignOperationsDispatchMaximumTransactionRetries;
         ++attemptNumber)
    {
        pqxx::connection attemptConnection{connectionString};
        ValidateSafetyGate(attemptConnection, safetyGate);
        if (const auto existing = LookupBeforeRetry(
                attemptConnection, requestId, attemptNumber,
                UncertainCommitRecoveryClassification::
                    completeAuthoritativeBinding))
            return *existing;
        try
        {
            pqxx::work transaction{attemptConnection};
            SetRole(transaction,
                kCampaignOperationsPhase5TransactionalRole);
            DispatchLockedAuthority authority =
                LockAndRevalidateDispatchAuthority(transaction, requestId,
                    lease->acquisition.resultingRequestVersion,
                    leaseTokenDigest, false);
            const auto persistedAttempt = FindDispatchAttempt(
                transaction, lease->attemptId);
            if (!persistedAttempt ||
                persistedAttempt->acquisition != lease->acquisition)
                throw std::runtime_error(
                    "campaign_operations_dispatch_attempt_mismatch");

            if (const auto existing =
                    FindAndValidateCompleteDispatchBinding(
                        transaction, requestId))
            {
                transaction.abort();
                return ExistingResult(*existing, attemptNumber,
                    UncertainCommitRecoveryClassification::
                        completeAuthoritativeBinding);
            }

            const auto materialization =
                ExperimentRecommendation::
                    LoadRecommendationCampaignLaunchMaterialization(
                        transaction, authority.materializationId);
            if (!materialization ||
                materialization->selectedMemberCount !=
                    authority.memberCount)
                throw std::runtime_error(
                    "campaign_operations_materialization_mismatch");

            // Freeze the Phase 5 proposal/review/execution domain before
            // classifying evidence.  The transaction-bound launch reacquires
            // these same locks and then continues through the established
            // activation/experiment lock domains.
            ExperimentRecommendation::LockRecommendationCampaignExecutions(
                transaction, *materialization);
            Invoke(testHook, DispatchTestInjectionPoint::
                beforeDownstreamEvidenceClassification, transaction);
            const auto downstream =
                ClassifyDownstreamEvidence(transaction, authority);
            BindingDisposition disposition = BindingDisposition::created;
            if (downstream ==
                DownstreamEvidenceClassification::
                    exactCompletePendingTrain)
            {
                if (HasDownstreamControlOwnerCollision(
                        transaction, authority))
                {
                    const auto outcome = PersistFailedDispatchOutcome(
                        transaction, authority, *persistedAttempt,
                        downstream,
                        SemanticConflictClassification::
                            controlOwnerCollision,
                        "dispatch_control_owner_collision");
                    transaction.commit();
                    return {outcome.result, downstream,
                        ExactReplayDisposition::reconciliationRequired,
                        outcome.recovery, requestId, {}, attemptNumber,
                        DispatchServiceFailureClassification::none,
                        outcome.diagnosticCode};
                }
                if (!authority.adoptionAuthorizationId)
                {
                    const auto outcome = PersistFailedDispatchOutcome(
                        transaction, authority, *persistedAttempt,
                        downstream,
                        SemanticConflictClassification::
                            authorizationInactive,
                        "dispatch_adoption_authorization_required");
                    transaction.commit();
                    return {outcome.result, downstream,
                        ExactReplayDisposition::reconciliationRequired,
                        outcome.recovery, requestId, {}, attemptNumber,
                        DispatchServiceFailureClassification::none,
                        outcome.diagnosticCode};
                }
                disposition =
                    BindingDisposition::adoptedExistingPending;
            }
            else if (downstream !=
                DownstreamEvidenceClassification::noPhase5Evidence)
            {
                const auto outcome = PersistFailedDispatchOutcome(
                    transaction, authority, *persistedAttempt, downstream,
                    ConflictFor(downstream),
                    "dispatch_downstream_reconciliation_required");
                transaction.commit();
                return {outcome.result, downstream,
                    ExactReplayDisposition::reconciliationRequired,
                    outcome.recovery, requestId, {}, attemptNumber,
                    DispatchServiceFailureClassification::none,
                    outcome.diagnosticCode};
            }

            Invoke(testHook,
                DispatchTestInjectionPoint::beforePhase5Invocation,
                transaction);
            const auto launch =
                ExperimentRecommendation::
                    LaunchRecommendationCampaignInTransaction(
                        transaction,
                        ExperimentRecommendation::
                            RecommendationCampaignLaunchRequest{
                                authority.materializationId, false},
                        *materialization,
                        [&](ExperimentRecommendation::
                                RecommendationCampaignLaunchTestPoint point)
                        {
                            switch (point)
                            {
                                case ExperimentRecommendation::
                                    RecommendationCampaignLaunchTestPoint::
                                        afterExecutionMutation:
                                    Invoke(testHook,
                                        DispatchTestInjectionPoint::
                                            afterPhase5ExecutionMutation,
                                        transaction);
                                    break;
                                case ExperimentRecommendation::
                                    RecommendationCampaignLaunchTestPoint::
                                        afterExperimentCreationOrReuseMutation:
                                    Invoke(testHook,
                                        DispatchTestInjectionPoint::
                                            afterExperimentCreationOrReuseMutation,
                                        transaction);
                                    break;
                                case ExperimentRecommendation::
                                    RecommendationCampaignLaunchTestPoint::
                                        afterActivationMutation:
                                    Invoke(testHook,
                                        DispatchTestInjectionPoint::
                                            afterPhase5ActivationMutation,
                                        transaction);
                                    break;
                            }
                        });
            RequestBindingSet bindingSet = PersistCompleteDispatchBinding(
                transaction, authority, *materialization, launch,
                disposition,
                [&](DispatchTestInjectionPoint point)
                {
                    Invoke(testHook, point, transaction);
                });
            PersistControlOwners(transaction, authority, bindingSet,
                [&](DispatchTestInjectionPoint point)
                {
                    Invoke(testHook, point, transaction);
                });
            (void)CommitReservationForBinding(
                transaction, authority, bindingSet,
                [&](DispatchTestInjectionPoint point)
                {
                    Invoke(testHook, point, transaction);
                });
            const auto outcome =
                BindRequestAndPersistSuccessfulOutcome(transaction,
                    authority, *persistedAttempt, bindingSet, downstream,
                    disposition,
                    [&](DispatchTestInjectionPoint point)
                    {
                        Invoke(testHook, point, transaction);
                    });
            Invoke(testHook,
                DispatchTestInjectionPoint::beforeHandoffCommit,
                transaction);
            transaction.commit();
            InvokeAfterCommit(testHook,
                DispatchTestInjectionPoint::
                    afterSuccessfulCommitBeforeResponse);
            return {outcome.result, downstream,
                ExactReplayDisposition::newOperation,
                UncertainCommitRecoveryClassification::provenNoCommit,
                requestId, bindingSet.identity.hash(), attemptNumber,
                DispatchServiceFailureClassification::none,
                outcome.diagnosticCode};
        }
        catch (const pqxx::sql_error& error)
        {
            if (Retryable(error) &&
                attemptNumber <
                    kCampaignOperationsDispatchMaximumTransactionRetries)
            {
                RetryBackoff(attemptNumber);
                continue;
            }
            if (Retryable(error))
                return RetryExhausted(
                    requestId, attemptNumber, error.sqlstate());
            throw;
        }
        catch (const pqxx::broken_connection&)
        {
            pqxx::connection recoveryConnection{connectionString};
            ValidateSafetyGate(recoveryConnection, safetyGate);
            if (const auto existing = LookupBeforeRetry(
                    recoveryConnection, requestId, attemptNumber,
                    UncertainCommitRecoveryClassification::
                        completeAuthoritativeBinding))
                return *existing;
            if (attemptNumber <
                kCampaignOperationsDispatchMaximumTransactionRetries)
            {
                RetryBackoff(attemptNumber);
                continue;
            }
            return {
                DispatchResultClassification::reconciliationRequired,
                DownstreamEvidenceClassification::causallyAmbiguous,
                ExactReplayDisposition::reconciliationRequired,
                UncertainCommitRecoveryClassification::ambiguousEvidence,
                requestId, {}, attemptNumber,
                DispatchServiceFailureClassification::commitOutcomeUnknown,
                "dispatch_commit_outcome_unknown"};
        }
    }
    throw std::runtime_error(
        "campaign_operations_dispatch_retry_exhausted");
}

} // namespace EA::CampaignOperations
