#include "CampaignOperationsDispatchService.hpp"
#include "CampaignOperationsManager.hpp"
#include "CampaignOperationsProductionAdmissionService.hpp"
#include "ExperimentRecommendationCampaignExecutionRepository.hpp"

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
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

bool IsReservedManagerOperationKey(const std::string& operationKey)
{
    return operationKey.rfind("mgr-v1:", 0U) == 0U;
}

bool IsExactGrandfatheredH2ManagerOperation(
    const std::string& connectionString, OperationalRequestId requestId,
    const std::string& operationKey)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    SetRole(transaction, kProductionPhase5TransactionalRole);
    return EA::CampaignOperations::IsExactGrandfatheredH2ManagerOperation(
        transaction, requestId, operationKey);
}

bool HasH2ManagerKeyCompatibilityInventory(const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    SetRole(transaction, kProductionPhase5TransactionalRole);
    return transaction.exec(
        "SELECT to_regclass("
        "'campaign_operations_h2_manager_key_compatibility') IS NOT NULL;")
        .one_row()[0].as<bool>();
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
    UncertainCommitRecoveryClassification recovery, bool production,
    int expectedRequestVersion = 0, const std::string& operationKey = {},
    const std::string& requestingActor = {},
    const std::string& approvedBuildContractCanonical = {},
    const std::optional<std::string>& managerSourceCanonical = std::nullopt)
{
    pqxx::read_transaction transaction{connection};
    SetRole(transaction, production ? kProductionPhase5TransactionalRole
                                    : kCampaignOperationsPhase5TransactionalRole);
    const auto existing =
        FindAndValidateCompleteDispatchBinding(
            transaction, requestId, production);
    if (existing)
    {
        if (production)
        {
            // The successful outcome is the authoritative edge from the
            // complete binding back to the exact Attempt V2 that produced it.
            // Attempt ordinal is only a recovery ordering field; it is never
            // binding ownership or replay identity.
            const auto persisted = FindProductionDispatchAttemptV2(
                transaction, existing->outcome.attemptId);
            if (!persisted)
                throw std::runtime_error(
                    "production_dispatch_replay_evidence_missing");
            if (persisted->attempt.requestId != requestId ||
                persisted->attempt.operationKey != operationKey ||
                persisted->attempt.expectedRequestVersion !=
                    expectedRequestVersion ||
                persisted->attempt.requestingActor.value() != requestingActor ||
                persisted->attempt.approvedBuildContract.identity.canonicalText() !=
                    approvedBuildContractCanonical ||
                existing->outcome.attemptCanonicalText !=
                    persisted->attempt.identity.canonicalText())
                throw std::runtime_error(
                    "production_dispatch_conflicting_replay");
            if (managerSourceCanonical)
            {
                RequireExactManagerOperationSourceEvidence(transaction,
                    persisted->attemptId, requestId, operationKey,
                    persisted->attempt.requestIdentityCanonical,
                    expectedRequestVersion, *managerSourceCanonical);
            }
        }
        return ExistingResult(*existing, transactionAttempts, recovery);
    }
    const auto requestState = transaction.exec(
        "SELECT request_state FROM "
        "campaign_operations_operational_request "
        "WHERE operational_request_id=$1;",
        pqxx::params{requestId.value()});
    if (requestState.empty())
        throw std::runtime_error(
            "campaign_operations_dispatch_request_not_found");
    if (requestState.one_row()[0].as<std::string>() == "ready")
        return std::nullopt;
    const auto outcome = FindLatestDispatchOutcome(
        transaction, requestId, production);
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
    pqxx::connection& connection, OperationalRequestId requestId,
    bool production, const std::string& operationKey,
    const std::optional<std::string>& managerSourceCanonical = std::nullopt)
{
    pqxx::read_transaction transaction{connection};
    SetRole(transaction, production
        ? kProductionPhase5TransactionalRole
        : kCampaignOperationsDispatcherRole);
    return production
        ? FindRecoverableProductionDispatchLease(
              transaction, requestId, operationKey, managerSourceCanonical)
        : FindRecoverableDispatchLease(transaction, requestId);
}

std::optional<DispatchLease> RecoverProductionLeaseAfterConflictingReplay(
    const std::string& connectionString, OperationalRequestId requestId,
    int expectedRequestVersion, const std::string& operationKey,
    const std::string& requestingActor,
    const std::string& approvedBuildContractCanonical,
    const std::optional<std::string>& managerSourceCanonical)
{
    pqxx::connection connection{connectionString};
    {
        pqxx::read_transaction transaction{connection};
        SetRole(transaction, kProductionPhase5TransactionalRole);
        const auto persisted = FindProductionDispatchAttemptV2(
            transaction, requestId, operationKey);
        if (!persisted)
        {
            transaction.commit();
            return std::nullopt;
        }
        if (persisted->attempt.expectedRequestVersion !=
                expectedRequestVersion ||
            persisted->attempt.requestingActor.value() != requestingActor ||
            persisted->attempt.approvedBuildContract.identity.canonicalText() !=
                approvedBuildContractCanonical)
            throw std::runtime_error(
                "production_dispatch_conflicting_replay");
        if (managerSourceCanonical)
            RequireExactManagerOperationSourceEvidence(transaction,
                persisted->attemptId, requestId, operationKey,
                persisted->attempt.requestIdentityCanonical,
                expectedRequestVersion, *managerSourceCanonical);
        transaction.commit();
    }
    return LookupRecoverableLease(connection, requestId, true, operationKey,
        managerSourceCanonical);
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

DispatchServiceResult RunDispatchAdapter(
    const std::string& connectionString, OperationalRequestId requestId,
    int expectedRequestVersion, const ActorIdentity& dispatcher,
    const std::optional<IsolatedDispatchSafetyGate>& safetyGate,
    bool production, const std::string& operationKey,
    const ManagerBuildContract* executingBuild,
    DispatchTestHook testHook,
    const std::optional<std::string>& managerSourceCanonical = std::nullopt,
    bool grandfatheredH2Only = false)
{
    pqxx::connection connection{connectionString};
    if (safetyGate) ValidateSafetyGate(connection, *safetyGate);
    if (production &&
        (!executingBuild || !IsValidProductionOperationKey(operationKey)))
        throw std::invalid_argument(
            "campaign_operations_production_dispatch_identity_invalid");

    const std::string approvedBuildContractCanonical = executingBuild
        ? executingBuild->identity.canonicalText() : std::string{};
    const auto lookupExisting = [&](pqxx::connection& candidate,
        int transactionAttempts,
        UncertainCommitRecoveryClassification recovery)
    {
        return LookupBeforeRetry(candidate, requestId, transactionAttempts,
            recovery, production, expectedRequestVersion, operationKey,
            dispatcher.value(), approvedBuildContractCanonical,
            managerSourceCanonical);
    };

    if (const auto existing = lookupExisting(connection, 0,
            UncertainCommitRecoveryClassification::
                completeAuthoritativeBinding))
        return *existing;

    std::optional<DispatchLease> lease =
        LookupRecoverableLease(connection, requestId, production, operationKey,
            managerSourceCanonical);
    if (grandfatheredH2Only && !lease)
        throw std::runtime_error(
            "campaign_operations_grandfathered_h2_operation_not_recoverable");
    const LeaseTokenDigest leaseTokenDigest = lease
        ? lease->acquisition.leaseTokenDigest
        : LeaseTokenDigest::Derive(GenerateOpaqueLeaseToken());
    for (int acquisitionAttempt = 1;
         !lease &&
         acquisitionAttempt <=
             kCampaignOperationsDispatchMaximumTransactionRetries;
         ++acquisitionAttempt)
    {
        auto acquisitionConnection =
            std::make_unique<pqxx::connection>(connectionString);
        if (safetyGate) ValidateSafetyGate(*acquisitionConnection, *safetyGate);
        if (const auto existing = lookupExisting(
                *acquisitionConnection, acquisitionAttempt,
                UncertainCommitRecoveryClassification::
                    completeAuthoritativeBinding))
            return *existing;
        if (const auto recovered =
                LookupRecoverableLease(*acquisitionConnection, requestId,
                    production, operationKey, managerSourceCanonical))
        {
            lease.emplace(*recovered);
            break;
        }
        try
        {
            pqxx::work transaction{*acquisitionConnection};
            SetRole(transaction, production ? kProductionDispatchServiceRole :
                kCampaignOperationsDispatcherRole);
            DispatchLease acquired = production
                ? AcquireProductionDispatchLeaseInTransaction(
                    transaction, requestId, expectedRequestVersion,
                    leaseTokenDigest, operationKey, dispatcher,
                    executingBuild->identity.canonicalText(),
                    managerSourceCanonical)
                : AcquireDispatchLeaseInTransaction(
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
            // Acquisition is a separate durable transaction.  The connection
            // has committed before the after-commit seam.  If that seam
            // reports uncertainty, the catch path explicitly abandons this
            // connection before opening the recovery connection.
            InvokeAfterCommit(testHook,
                DispatchTestInjectionPoint::afterAcquisitionCommitBeforeHandoff);
            acquisitionConnection.reset();
            lease.emplace(std::move(acquired));
        }
        catch (const pqxx::sql_error& error)
        {
            // Two identical callers can both enter the protected acquisition
            // transition before the winner commits.  The database replay
            // function sees the loser's provisional lease digest and reports
            // 23505; reconcile that exact key against the committed Attempt
            // V2 before classifying it as a conflict.  No other key or input
            // is allowed through this path.
            if (production && error.sqlstate() == "23505")
            {
                if (const auto recovered =
                        RecoverProductionLeaseAfterConflictingReplay(
                            connectionString, requestId, expectedRequestVersion,
                            operationKey, dispatcher.value(),
                            approvedBuildContractCanonical,
                            managerSourceCanonical))
                {
                    lease.emplace(*recovered);
                    break;
                }
            }
            if (!Retryable(error)) throw;
            if (acquisitionAttempt >=
                kCampaignOperationsDispatchMaximumTransactionRetries)
                return RetryExhausted(
                    requestId, acquisitionAttempt, error.sqlstate());
            RetryBackoff(acquisitionAttempt);
        }
        catch (const pqxx::in_doubt_error&)
        {
            // Destroy the original connection before the authoritative
            // recovery read.  pqxx exposes abandonment through destruction.
            acquisitionConnection.reset();
            pqxx::connection recoveryConnection{connectionString};
            InvokeAfterCommit(testHook,
                DispatchTestInjectionPoint::afterAcquisitionRecoveryConnectionOpened);
            if (safetyGate)
                ValidateSafetyGate(recoveryConnection, *safetyGate);
            if (const auto existing = lookupExisting(
                    recoveryConnection, acquisitionAttempt,
                    UncertainCommitRecoveryClassification::
                        completeAuthoritativeBinding))
                return *existing;
            if (const auto recovered = LookupRecoverableLease(
                    recoveryConnection, requestId, production, operationKey,
                    managerSourceCanonical))
            {
                lease.emplace(*recovered);
                break;
            }
            if (acquisitionAttempt <
                kCampaignOperationsDispatchMaximumTransactionRetries)
            {
                RetryBackoff(acquisitionAttempt);
                continue;
            }
            return {DispatchResultClassification::reconciliationRequired,
                DownstreamEvidenceClassification::causallyAmbiguous,
                ExactReplayDisposition::reconciliationRequired,
                UncertainCommitRecoveryClassification::ambiguousEvidence,
                requestId, {}, acquisitionAttempt,
                DispatchServiceFailureClassification::commitOutcomeUnknown,
                "dispatch_commit_outcome_unknown"};
        }
        catch (const pqxx::broken_connection&)
        {
            pqxx::connection recoveryConnection{connectionString};
            if (safetyGate)
                ValidateSafetyGate(recoveryConnection, *safetyGate);
            if (const auto existing = lookupExisting(
                    recoveryConnection, acquisitionAttempt,
                    UncertainCommitRecoveryClassification::
                        completeAuthoritativeBinding))
                return *existing;
            if (const auto recovered =
                    LookupRecoverableLease(recoveryConnection, requestId,
                        production, operationKey, managerSourceCanonical))
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
        if (safetyGate) ValidateSafetyGate(attemptConnection, *safetyGate);
        if (const auto existing = lookupExisting(
                attemptConnection, attemptNumber,
                UncertainCommitRecoveryClassification::
                    completeAuthoritativeBinding))
            return *existing;
        try
        {
            pqxx::work transaction{attemptConnection};
            SetRole(transaction, production
                ? kProductionPhase5TransactionalRole
                : kCampaignOperationsPhase5TransactionalRole);
            std::optional<DispatchAttemptRecord> persistedAttempt;
            if (production)
            {
                const auto productionAttempt =
                    FindProductionDispatchAttemptV2(
                        transaction, requestId, operationKey);
                if (productionAttempt)
                {
                    if (managerSourceCanonical)
                        RequireExactManagerOperationSourceEvidence(transaction,
                            productionAttempt->attemptId, requestId,
                            operationKey,
                            productionAttempt->attempt.requestIdentityCanonical,
                            expectedRequestVersion,
                            *managerSourceCanonical);
                    persistedAttempt.emplace(DispatchAttemptRecord{
                        productionAttempt->attemptId, lease->acquisition});
                }
            }
            else
            {
                const auto isolatedAttempt = FindDispatchAttempt(
                    transaction, lease->attemptId);
                if (isolatedAttempt)
                    persistedAttempt.emplace(*isolatedAttempt);
            }
            if (!persistedAttempt ||
                persistedAttempt->acquisition != lease->acquisition)
                throw std::runtime_error(
                    "campaign_operations_dispatch_attempt_mismatch");
            DispatchLockedAuthority authority =
                LockAndRevalidateDispatchAuthority(transaction, requestId,
                    lease->acquisition.resultingRequestVersion,
                    leaseTokenDigest, false, production, operationKey,
                    executingBuild ? executingBuild->identity.canonicalText()
                                    : std::string{}
#if defined(CAMPAIGN_OPERATIONS_H2_TESTING)
                    , testHook
#endif
                    );
            if (const auto existing =
                    FindAndValidateCompleteDispatchBinding(
                        transaction, requestId, production))
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
                        "dispatch_control_owner_collision", production);
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
                        "dispatch_adoption_authorization_required", production);
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
                    "dispatch_downstream_reconciliation_required", production);
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
                    }, production, operationKey,
                    approvedBuildContractCanonical);
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
#ifdef CAMPAIGN_OPERATIONS_H2_TESTING
            std::cerr << "H2_DISPATCH_RETRY sqlstate=" << error.sqlstate()
                      << " diagnostic=" << error.what() << " query="
                      << error.query() << '\n';
#endif
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
        catch (const pqxx::in_doubt_error&)
        {
            // The handoff connection is never reused after an uncertain
            // commit.  A fresh read can acknowledge only a complete durable
            // binding; otherwise recovery remains bounded and indeterminate.
            pqxx::connection recoveryConnection{connectionString};
            if (safetyGate)
                ValidateSafetyGate(recoveryConnection, *safetyGate);
            if (const auto existing = lookupExisting(
                    recoveryConnection, attemptNumber,
                    UncertainCommitRecoveryClassification::
                        completeAuthoritativeBinding))
                return *existing;
            if (attemptNumber <
                kCampaignOperationsDispatchMaximumTransactionRetries)
            {
                RetryBackoff(attemptNumber);
                continue;
            }
            return {DispatchResultClassification::reconciliationRequired,
                DownstreamEvidenceClassification::causallyAmbiguous,
                ExactReplayDisposition::reconciliationRequired,
                UncertainCommitRecoveryClassification::ambiguousEvidence,
                requestId, {}, attemptNumber,
                DispatchServiceFailureClassification::commitOutcomeUnknown,
                "dispatch_commit_outcome_unknown"};
        }
        catch (const pqxx::broken_connection&)
        {
            pqxx::connection recoveryConnection{connectionString};
            if (safetyGate)
                ValidateSafetyGate(recoveryConnection, *safetyGate);
            if (const auto existing = lookupExisting(
                    recoveryConnection, attemptNumber,
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
        catch (const std::runtime_error& error)
        {
            // A same-key contender can own the exact recovered Attempt V2
            // while the winner is finishing the reservation projection.  The
            // winner's commit makes the transient held-reservation predicate
            // false; retry through the normal complete-binding replay lookup
            // rather than treating that serialization window as a conflict.
            if (production &&
                std::string(error.what()) ==
                    "campaign_operations_dispatch_reservation_unavailable" &&
                attemptNumber <
                    kCampaignOperationsDispatchMaximumTransactionRetries)
            {
                RetryBackoff(attemptNumber);
                continue;
            }
            throw;
        }
    }
    throw std::runtime_error(
        "campaign_operations_dispatch_retry_exhausted");
}

DispatchServiceResult DispatchOneRequestForIsolatedTest(
    const std::string& connectionString, OperationalRequestId requestId,
    int expectedRequestVersion, const ActorIdentity& dispatcher,
    const IsolatedDispatchSafetyGate& safetyGate, DispatchTestHook testHook)
{
    return RunDispatchAdapter(connectionString, requestId,
        expectedRequestVersion, dispatcher, safetyGate, false, {}, nullptr,
        std::move(testHook));
}

DispatchServiceResult DispatchOneRequestForProduction(
    const std::string& connectionString, const ProductionDispatchRequest& request,
    const std::string& executablePath)
{
    if (!request.acknowledged)
        throw std::invalid_argument(
            "production dispatch requires the literal --yes acknowledgement");
    if (!IsValidProductionOperationKey(request.operationKey))
        throw std::invalid_argument(
            "campaign_operations_production_operation_key_invalid");
    const bool hasCompatibilityInventory =
        HasH2ManagerKeyCompatibilityInventory(connectionString);
    const bool grandfatheredH2 = IsReservedManagerOperationKey(
        request.operationKey) && IsExactGrandfatheredH2ManagerOperation(
            connectionString, request.requestId, request.operationKey);
    if (IsReservedManagerOperationKey(request.operationKey) && !grandfatheredH2)
        throw std::invalid_argument(
            "campaign_operations_manager_operation_key_reserved");
    if (request.expectedRequestVersion <= 0)
        throw std::invalid_argument(
            "campaign_operations_production_expected_request_version_invalid");
    ValidateManagerBuildContract(request.executingBuild);
    const auto actualBuild = CaptureActualManagerBuildContract(executablePath);
    if (!actualBuild || request.executingBuild != *actualBuild)
        throw std::runtime_error("production_dispatch_build_mismatch");
    pqxx::connection readinessConnection{connectionString};
    const auto readiness = LoadProductionReadiness(readinessConnection,
        actualBuild);
    if (!readiness.ready)
        throw std::runtime_error(
            "campaign_operations_production_readiness_blocked:" +
            [&]
            {
                std::string joined;
                for (const auto& blocker : readiness.blockers)
                {
                    if (!joined.empty()) joined += ';';
                    joined += blocker;
                }
                return joined;
            }());
    return RunDispatchAdapter(connectionString, request.requestId,
        request.expectedRequestVersion, request.requestingActor, std::nullopt,
        true, request.operationKey, &request.executingBuild, {}, std::nullopt,
        grandfatheredH2 && hasCompatibilityInventory);
}

DispatchServiceResult DispatchOneRequestForProduction(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString,
    const ProductionDispatchRequest& request, const std::string& executablePath)
{
    if (!request.acknowledged)
        throw std::invalid_argument(
            "production dispatch requires the literal --yes acknowledgement");
    if (!IsValidProductionOperationKey(request.operationKey))
        throw std::invalid_argument(
            "campaign_operations_production_operation_key_invalid");
    const bool hasCompatibilityInventory =
        HasH2ManagerKeyCompatibilityInventory(managerConnectionString);
    const bool grandfatheredH2 = IsReservedManagerOperationKey(
        request.operationKey) && IsExactGrandfatheredH2ManagerOperation(
            managerConnectionString, request.requestId, request.operationKey);
    if (IsReservedManagerOperationKey(request.operationKey) && !grandfatheredH2)
        throw std::invalid_argument(
            "campaign_operations_manager_operation_key_reserved");
    if (request.expectedRequestVersion <= 0)
        throw std::invalid_argument(
            "campaign_operations_production_expected_request_version_invalid");
    ValidateManagerBuildContract(request.executingBuild);
    const auto actualBuild = CaptureActualManagerBuildContract(executablePath);
    if (!actualBuild || request.executingBuild != *actualBuild)
        throw std::runtime_error("production_dispatch_build_mismatch");
    pqxx::connection readinessConnection{managerConnectionString};
    const auto readiness = LoadProductionReadiness(readinessConnection,
        actualBuild);
    if (!readiness.ready)
        throw std::runtime_error(
            "campaign_operations_production_readiness_blocked:" +
            [&]
            {
                std::string joined;
                for (const auto& blocker : readiness.blockers)
                {
                    if (!joined.empty()) joined += ';';
                    joined += blocker;
                }
                return joined;
            }());
    return RunDispatchAdapter(dispatchServiceConnectionString, request.requestId,
        request.expectedRequestVersion, request.requestingActor, std::nullopt,
        true, request.operationKey, &request.executingBuild, {}, std::nullopt,
        grandfatheredH2 && hasCompatibilityInventory);
}

DispatchServiceResult DispatchOneRequestForProductionManager(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString,
    const ProductionDispatchRequest& request,
    const std::string& managerSourceCanonical, const std::string& executablePath)
{
    if (!request.acknowledged)
        throw std::invalid_argument(
            "production dispatch requires the literal --yes acknowledgement");
    if (!IsValidProductionOperationKey(request.operationKey))
        throw std::invalid_argument(
            "campaign_operations_production_operation_key_invalid");
    if (request.expectedRequestVersion <= 0)
        throw std::invalid_argument(
            "campaign_operations_production_expected_request_version_invalid");
    if (managerSourceCanonical.empty())
        throw std::invalid_argument(
            "campaign_operations_manager_source_canonical_required");
    ValidateManagerBuildContract(request.executingBuild);
    const auto actualBuild = CaptureActualManagerBuildContract(executablePath);
    if (!actualBuild || request.executingBuild != *actualBuild)
        throw std::runtime_error("production_dispatch_build_mismatch");
    pqxx::connection readinessConnection{managerConnectionString};
    const auto readiness = LoadProductionReadiness(readinessConnection,
        actualBuild);
    if (!readiness.ready)
        throw std::runtime_error(
            "campaign_operations_production_readiness_blocked:" +
            [&]
            {
                std::string joined;
                for (const auto& blocker : readiness.blockers)
                {
                    if (!joined.empty()) joined += ';';
                    joined += blocker;
                }
                return joined;
            }());
    return RunDispatchAdapter(dispatchServiceConnectionString, request.requestId,
        request.expectedRequestVersion, request.requestingActor, std::nullopt,
        true, request.operationKey, &request.executingBuild, {},
        managerSourceCanonical);
}

#if defined(CAMPAIGN_OPERATIONS_H3_TESTING)
DispatchServiceResult DispatchOneRequestForProductionManagerWithFixture(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString,
    const ProductionDispatchRequest& request,
    const std::string& managerSourceCanonical,
    const ManagerBuildContract& fixtureBuild, ManagerAdapterTestHook testHook)
{
    if (!request.acknowledged)
        throw std::invalid_argument(
            "production dispatch requires the literal --yes acknowledgement");
    if (!IsValidProductionOperationKey(request.operationKey))
        throw std::invalid_argument(
            "campaign_operations_production_operation_key_invalid");
    if (request.expectedRequestVersion <= 0 || managerSourceCanonical.empty())
        throw std::invalid_argument(
            "campaign_operations_manager_fixture_identity_invalid");
    ValidateManagerBuildContract(request.executingBuild);
    ValidateManagerBuildContract(fixtureBuild);
    if (request.executingBuild != fixtureBuild)
        throw std::runtime_error("campaign_operations_manager_fixture_build_mismatch");
    pqxx::connection readinessConnection{managerConnectionString};
    const auto readiness = LoadProductionReadiness(readinessConnection,
        fixtureBuild);
    std::string blockers;
    for (const auto& blocker : readiness.blockers)
    {
        // The repository-native H1 disposable fixture intentionally retains
        // historical Attempt V1 materializations.  They are immutable test
        // data and make only the aggregate version summary non-ready; all
        // runtime authority/readiness blockers remain enforced here.
        if (blocker == "canonical_contract_versions") continue;
        if (!blockers.empty()) blockers += ';';
        blockers += blocker;
    }
    if (!blockers.empty())
        throw std::runtime_error(
            "campaign_operations_production_readiness_blocked:" + blockers);
    return RunDispatchAdapter(dispatchServiceConnectionString, request.requestId,
        request.expectedRequestVersion, request.requestingActor, std::nullopt,
        true, request.operationKey, &fixtureBuild, std::move(testHook),
        managerSourceCanonical);
}
#endif

#if defined(CAMPAIGN_OPERATIONS_H2_TESTING)
DispatchServiceResult DispatchOneRequestForProductionForTest(
    const std::string& connectionString, const ProductionDispatchRequest& request,
    DispatchTestHook testHook)
{
    if (!request.acknowledged)
        throw std::invalid_argument(
            "production dispatch requires the literal --yes acknowledgement");
    if (!IsValidProductionOperationKey(request.operationKey))
        throw std::invalid_argument(
            "campaign_operations_production_operation_key_invalid");
    const bool hasCompatibilityInventory =
        HasH2ManagerKeyCompatibilityInventory(connectionString);
    const bool grandfatheredH2 = IsReservedManagerOperationKey(
        request.operationKey) && IsExactGrandfatheredH2ManagerOperation(
            connectionString, request.requestId, request.operationKey);
    if (IsReservedManagerOperationKey(request.operationKey) && !grandfatheredH2)
        throw std::invalid_argument(
            "campaign_operations_manager_operation_key_reserved");
    if (request.expectedRequestVersion <= 0)
        throw std::invalid_argument(
            "campaign_operations_production_expected_request_version_invalid");
    ValidateManagerBuildContract(request.executingBuild);
    return RunDispatchAdapter(connectionString, request.requestId,
        request.expectedRequestVersion, request.requestingActor, std::nullopt,
        true, request.operationKey, &request.executingBuild,
        std::move(testHook), std::nullopt,
        grandfatheredH2 && hasCompatibilityInventory);
}

DispatchServiceResult DispatchOneRequestForProductionForTest(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString,
    const ProductionDispatchRequest& request, DispatchTestHook testHook)
{
    if (!request.acknowledged)
        throw std::invalid_argument(
            "production dispatch requires the literal --yes acknowledgement");
    if (!IsValidProductionOperationKey(request.operationKey))
        throw std::invalid_argument(
            "campaign_operations_production_operation_key_invalid");
    const bool hasCompatibilityInventory =
        HasH2ManagerKeyCompatibilityInventory(managerConnectionString);
    const bool grandfatheredH2 = IsReservedManagerOperationKey(
        request.operationKey) && IsExactGrandfatheredH2ManagerOperation(
            managerConnectionString, request.requestId, request.operationKey);
    if (IsReservedManagerOperationKey(request.operationKey) && !grandfatheredH2)
        throw std::invalid_argument(
            "campaign_operations_manager_operation_key_reserved");
    if (request.expectedRequestVersion <= 0)
        throw std::invalid_argument(
            "campaign_operations_production_expected_request_version_invalid");
    ValidateManagerBuildContract(request.executingBuild);
    return RunDispatchAdapter(dispatchServiceConnectionString, request.requestId,
        request.expectedRequestVersion, request.requestingActor, std::nullopt,
        true, request.operationKey, &request.executingBuild,
        std::move(testHook), std::nullopt,
        grandfatheredH2 && hasCompatibilityInventory);
}
#endif

} // namespace EA::CampaignOperations
