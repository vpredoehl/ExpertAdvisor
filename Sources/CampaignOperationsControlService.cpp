#include "CampaignOperationsControlService.hpp"

#include "CampaignOperationsRepository.hpp"
#include "CampaignOperationsService.hpp"

#include <algorithm>
#include <chrono>
#include <locale>
#include <ostream>
#include <random>
#include <sstream>
#include <thread>

namespace EA::CampaignOperations
{
namespace
{

void SetRole(pqxx::transaction_base& transaction, const char* role)
{
    transaction.exec("SET LOCAL ROLE " + transaction.quote_name(role) + ";");
}

void RequireControlSchema(pqxx::transaction_base& transaction)
{
    if (!ControlSchemaExists(transaction))
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_control_schema_required");
}

bool Retryable(const pqxx::sql_error& error)
{
    return error.sqlstate() == "40001" || error.sqlstate() == "40P01" ||
        error.sqlstate() == "23505";
}

void Backoff(int attempt)
{
    std::random_device source;
    std::uniform_int_distribution<int> jitter(1, 5);
    std::this_thread::sleep_for(std::chrono::milliseconds(
        attempt * 2 + jitter(source)));
}

void ValidateCancellationReplay(
    const PersistedCampaignCancellationRequest& persisted,
    const CampaignCancellationCommandRequest& request)
{
    if (request.requestId.has_value() !=
            persisted.request.requestId.has_value() ||
        (request.requestId &&
            *request.requestId !=
                persisted.request.requestId->value()) ||
        (request.expectedRequestVersion &&
            *request.expectedRequestVersion !=
                *persisted.request.expectedRequestVersion) ||
        persisted.request.actor.value() != request.actorIdentity ||
        persisted.request.reason.value() != request.reason)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_cancellation_request_conflict");
}

int FailureCode(const std::exception& error)
{
    const auto* domain = dynamic_cast<const Error*>(&error);
    if (!domain) return 1;
    if (domain->code() == ErrorCode::persistenceConflict) return 2;
    return 1;
}

template <typename Function>
int RunCommand(Function&& function, std::ostream& errors,
    const char* prefix)
{
    try
    {
        return function();
    }
    catch (const pqxx::sql_error& error)
    {
        errors << prefix
               << ",status=failed,diagnostic_code=postgresql_"
               << CampaignOperationsMachineText(error.sqlstate())
               << ",diagnostic_detail="
               << CampaignOperationsMachineText(error.what()) << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        const auto* domain = dynamic_cast<const Error*>(&error);
        errors << prefix << ",status=failed,diagnostic_code="
               << CampaignOperationsMachineText(
                      domain ? error.what()
                             : "campaign_operations_unexpected_exception");
        if (!domain)
            errors << ",diagnostic_detail="
                   << CampaignOperationsMachineText(error.what());
        errors << '\n';
        return FailureCode(error);
    }
}

ReconciliationObservation ObservationFor(
    const ReconciliationObserveRequest& request,
    const ReconciliationCandidate& candidate)
{
    ReconciliationReason reason =
        ReconciliationReason::causalityAmbiguous;
    std::string service = "campaign_operations_operator";
    std::string action = "inspect_ambiguous_evidence";
    std::string diagnostic = "causality_ambiguous";
    if (candidate.cancellationRequestId)
    {
        reason = ReconciliationReason::cancellationSettlementPending;
        service = "campaign_operations_cancellation_coordinator";
        action = "replay_cancellation";
        diagnostic = "cancellation_settlement_pending";
    }
    else if (candidate.reservationState == ReservationState::held &&
             candidate.reservationExpired &&
             candidate.requestState == RequestState::ready &&
             candidate.bindingCount == 0 &&
             candidate.downstreamExecutionCount == 0 &&
             candidate.dispatchOutcomeCount == 0)
    {
        reason = ReconciliationReason::
            reservationExpiredNoDownstreamEvidence;
        service = "campaign_operations_reservation_service";
        action = "expire_held_reservation";
        diagnostic =
            "reservation_expired_no_downstream_evidence";
    }
    else if (candidate.requestState == RequestState::dispatching &&
             candidate.leaseExpired &&
             candidate.bindingCount == 0 &&
             candidate.downstreamExecutionCount == 0 &&
             candidate.dispatchOutcomeCount == 0)
    {
        reason = ReconciliationReason::
            dispatchLeaseExpiredNoDownstreamEvidence;
        service = "campaign_operations_dispatch_recovery";
        action = "clear_stale_dispatch_lease";
        diagnostic =
            "dispatch_lease_expired_no_downstream_evidence";
    }
    else if (candidate.bindingCount > 0)
    {
        reason = ReconciliationReason::bindingCardinalityMismatch;
        diagnostic = "binding_cardinality_mismatch";
    }
    else if (candidate.downstreamExecutionCount > 0)
    {
        reason = ReconciliationReason::progressedUnboundEvidence;
        diagnostic = "progressed_unbound_evidence";
    }
    else if (candidate.dispatchOutcomeCount > 0)
    {
        reason = ReconciliationReason::dispatchOutcomeUnknown;
        diagnostic = "dispatch_outcome_unknown";
    }
    return BuildReconciliationObservation(request.runKey,
        candidate.campaignId, candidate.requestId,
        candidate.requestCanonicalText, candidate.requestState,
        candidate.requestVersion, reason,
        BuildReconciliationEvidenceCanonical(candidate),
        std::move(service), std::move(action), std::move(diagnostic));
}

std::optional<PersistedReconciliationBatch>
LookupCommittedReconciliationBatch(
    const std::string& connectionString,
    const ReconciliationObserveRequest& request)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    SetRole(transaction, kCampaignOperationsReconcilerRole);
    RequireControlSchema(transaction);
    const auto cursor = FindReconciliationCursor(
        transaction, request.runKey, request.afterRequestId);
    if (!cursor) return std::nullopt;
    if (cursor->requestedLimit != request.limit)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_reconciliation_cursor_conflict");
    auto observations = LoadReconciliationObservationsForCursor(
        transaction, cursor->reconciliationCursorEventId);
    if (static_cast<int>(observations.size()) !=
            cursor->selectedCount)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_reconciliation_cursor_incomplete");
    return PersistedReconciliationBatch{
        *cursor, std::move(observations), true};
}

bool LookupCommittedReconciliationResolution(
    const std::string& connectionString,
    ReconciliationObservationId observationId)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    SetRole(transaction, kCampaignOperationsRecoveryRole);
    RequireControlSchema(transaction);
    return FindReconciliationResolution(
        transaction, observationId).has_value();
}

} // namespace

CampaignControlRequest ValidateCampaignControlRequest(
    const CampaignControlRequest& request)
{
    (void)OperationalCampaignId(request.campaignId);
    (void)ActorIdentity(request.actorIdentity);
    (void)Reason(request.reason);
    if (request.expectedControlVersion < 0)
        throw Error(ErrorCode::invalidControlEvent,
            "campaign_operations_control_expected_version_invalid");
    return request;
}

CampaignCancellationCommandRequest ValidateCampaignCancellationRequest(
    const CampaignCancellationCommandRequest& request)
{
    (void)OperationalCampaignId(request.campaignId);
    (void)ActorIdentity(request.actorIdentity);
    (void)Reason(request.reason);
    if (request.operationKey.empty() ||
        request.operationKey.size() > 128U ||
        !std::all_of(request.operationKey.begin(),
            request.operationKey.end(), [](unsigned char c)
            {
                return (c >= 'a' && c <= 'z') ||
                    (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '_' ||
                    c == '-' || c == '.' || c == ':' || c == '/';
            }))
        throw Error(ErrorCode::invalidCancellationRequest,
            "campaign_operations_cancellation_operation_key_invalid");
    if (request.requestId.has_value() !=
            request.expectedRequestVersion.has_value() ||
        (request.requestId && *request.requestId <= 0) ||
        (request.expectedRequestVersion &&
            *request.expectedRequestVersion <= 0))
        throw Error(ErrorCode::invalidCancellationRequest,
            "campaign_operations_cancellation_target_invalid");
    return request;
}

ReconciliationObserveRequest ValidateReconciliationObserveRequest(
    const ReconciliationObserveRequest& request)
{
    if (request.runKey.empty() || request.runKey.size() > 128U ||
        !std::all_of(request.runKey.begin(), request.runKey.end(),
            [](unsigned char c)
            {
                return (c >= 'a' && c <= 'z') ||
                    (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '_' ||
                    c == '-' || c == '.' || c == ':' || c == '/';
            }) ||
        request.afterRequestId < 0 || request.limit <= 0 ||
        request.limit > kCampaignOperationsMaximumReconciliationBatchSize)
        throw Error(ErrorCode::invalidReconciliationObservation,
            "campaign_operations_reconciliation_request_invalid");
    return request;
}

CampaignControlResult ControlCampaign(
    pqxx::connection& connection, const CampaignControlRequest& request)
{
    const auto validated = ValidateCampaignControlRequest(request);
    for (int attempt = 1; attempt <= 3; ++attempt)
    {
        try
        {
            pqxx::work transaction{connection};
            SetRole(transaction, kCampaignOperationsControllerRole);
            RequireControlSchema(transaction);
            const auto campaign = FindOperationalCampaign(
                transaction, OperationalCampaignId(validated.campaignId));
            if (!campaign)
                throw Error(ErrorCode::persistenceConflict,
                    "campaign_operations_campaign_not_found");
            const auto head = FindCampaignControlHead(
                transaction, campaign->campaignId);
            const int currentVersion =
                head ? head->event.controlVersion : 0;
            if (head &&
                currentVersion == validated.expectedControlVersion + 1 &&
                head->event.eventKind == validated.action &&
                head->event.actor.value() == validated.actorIdentity &&
                head->event.reason.value() == validated.reason)
            {
                transaction.commit();
                return {ControlReplayDisposition::existingIdentical, *head};
            }
            if (validated.expectedControlVersion != currentVersion)
                throw Error(ErrorCode::persistenceConflict,
                    "campaign_operations_control_version_conflict");
            if ((validated.action == ControlEventKind::pause &&
                    head && head->event.eventKind ==
                        ControlEventKind::pause) ||
                (validated.action == ControlEventKind::resume &&
                    (!head || head->event.eventKind !=
                        ControlEventKind::pause)))
                throw Error(ErrorCode::persistenceConflict,
                    "campaign_operations_control_transition_invalid");
            const auto event = BuildCampaignControlEvent(
                campaign->campaignId,
                campaign->campaign.identity.canonicalText(),
                head ? std::optional<ControlEventId>(
                           head->controlEventId) : std::nullopt,
                head ? std::optional<std::string>(
                           head->event.identity.canonicalText())
                     : std::nullopt,
                currentVersion + 1, validated.action,
                ActorIdentity(validated.actorIdentity),
                Reason(validated.reason));
            const auto persisted =
                PersistCampaignControlEvent(transaction, event);
            transaction.commit();
            return {ControlReplayDisposition::recorded, persisted};
        }
        catch (const pqxx::sql_error& error)
        {
            if (!Retryable(error) || attempt == 3) throw;
            Backoff(attempt);
        }
    }
    throw std::runtime_error(
        "campaign_operations_control_retry_exhausted");
}

CampaignCancellationResult CancelCampaign(
    const std::string& connectionString,
    const CampaignCancellationCommandRequest& request,
    CampaignOperationsControlTestHook testHook)
{
    const auto validated = ValidateCampaignCancellationRequest(request);
    struct CommittedAttempt final
    {
        PersistedCampaignCancellationRequest persisted;
        bool replay = false;
        std::optional<ReconciliationCandidate> candidate;
        std::optional<PersistedCampaignCancellationSettlement> settlement;
    };
    std::optional<CommittedAttempt> committed;
    for (int attempt = 1; attempt <= 3; ++attempt)
    {
        try
        {
            std::optional<PersistedCampaignCancellationRequest> persisted;
            bool replay = false;
            std::optional<ReconciliationCandidate> candidate;
            std::optional<PersistedCampaignCancellationSettlement> settlement;
            pqxx::connection connection{connectionString};
            pqxx::work transaction{connection};
            SetRole(transaction, kCampaignOperationsCancellationRole);
            RequireControlSchema(transaction);
            const auto campaign = FindOperationalCampaign(
                transaction, OperationalCampaignId(validated.campaignId));
            if (!campaign)
                throw Error(ErrorCode::persistenceConflict,
                    "campaign_operations_campaign_not_found");
            if (const auto existing =
                    FindCampaignCancellationRequestByOperation(
                        transaction, campaign->campaignId,
                        validated.operationKey))
            {
                persisted.emplace(*existing);
                replay = true;
                ValidateCancellationReplay(*persisted, validated);
                if (const auto existingSettlement =
                        FindCampaignCancellationSettlement(
                            transaction,
                            persisted->cancellationRequestId))
                    settlement.emplace(*existingSettlement);
            }
            else
            {
                std::optional<OperationalRequestId> targetId;
                if (validated.requestId)
                    targetId.emplace(*validated.requestId);
                else
                {
                    const auto target =
                        FindOperationalRequestByCampaignAction(
                            transaction, campaign->campaignId,
                            campaign->campaign.actionKind,
                            campaign->campaign.actionContractVersion);
                    if (target) targetId.emplace(target->requestId);
                }
                if (targetId)
                {
                    candidate.emplace(LoadReconciliationCandidate(
                        transaction, *targetId));
                    if (candidate->campaignId != campaign->campaignId)
                        throw Error(ErrorCode::persistenceConflict,
                            "campaign_operations_cancellation_target_mismatch");
                    if (validated.expectedRequestVersion &&
                        *validated.expectedRequestVersion !=
                            candidate->requestVersion)
                        throw Error(ErrorCode::persistenceConflict,
                            "campaign_operations_cancellation_version_conflict");
                    if (candidate->reservationState ==
                            ReservationState::held)
                        LockCancellationDomains(transaction, *candidate);
                    else
                    {
                        transaction.exec(
                            "SELECT lock_campaign_operations_campaign($1);",
                            pqxx::params{campaign->campaignId.value()});
                        transaction.exec(
                            "SELECT lock_campaign_operations_reservation($1);",
                            pqxx::params{candidate->reservationId.value()});
                        transaction.exec(
                            "SELECT lock_campaign_operations_request($1);",
                            pqxx::params{candidate->requestId.value()});
                    }
                    candidate.emplace(LoadReconciliationCandidate(
                        transaction, *targetId));
                    if (candidate->campaignId != campaign->campaignId)
                        throw Error(ErrorCode::persistenceConflict,
                            "campaign_operations_cancellation_target_mismatch");
                    if (const auto concurrent =
                            FindCampaignCancellationRequestByOperation(
                                transaction, campaign->campaignId,
                                validated.operationKey))
                    {
                        persisted.emplace(*concurrent);
                        replay = true;
                        ValidateCancellationReplay(*persisted, validated);
                        if (const auto existingSettlement =
                                FindCampaignCancellationSettlement(
                                    transaction,
                                    persisted->cancellationRequestId))
                            settlement.emplace(*existingSettlement);
                    }
                    else if (validated.expectedRequestVersion &&
                        *validated.expectedRequestVersion !=
                            candidate->requestVersion)
                        throw Error(ErrorCode::persistenceConflict,
                            "campaign_operations_cancellation_version_conflict");
                }
                else
                    transaction.exec(
                        "SELECT lock_campaign_operations_campaign($1);",
                        pqxx::params{campaign->campaignId.value()});
                if (!persisted)
                    if (const auto concurrent =
                            FindCampaignCancellationRequestByOperation(
                                transaction, campaign->campaignId,
                                validated.operationKey))
                    {
                        persisted.emplace(*concurrent);
                        replay = true;
                        ValidateCancellationReplay(*persisted, validated);
                        if (const auto existingSettlement =
                                FindCampaignCancellationSettlement(
                                    transaction,
                                    persisted->cancellationRequestId))
                            settlement.emplace(*existingSettlement);
                    }
                if (!persisted &&
                    FindCampaignCancellationRequestByTarget(
                        transaction, campaign->campaignId, targetId))
                    throw Error(ErrorCode::persistenceConflict,
                        "campaign_operations_cancellation_target_already_owned");
                if (!persisted)
                {
                    auto cancellation = BuildCampaignCancellationRequest(
                        campaign->campaignId,
                        campaign->campaign.identity.canonicalText(),
                        targetId,
                        candidate
                            ? std::optional<std::string>(
                                  candidate->requestCanonicalText)
                            : std::nullopt,
                        candidate
                            ? std::optional<RequestState>(
                                  candidate->requestState)
                            : std::nullopt,
                        candidate
                            ? std::optional<int>(candidate->requestVersion)
                            : std::nullopt,
                        validated.operationKey,
                        ActorIdentity(validated.actorIdentity),
                        Reason(validated.reason));
                    persisted.emplace(PersistCampaignCancellationRequest(
                        transaction, cancellation));
                    if (testHook)
                        testHook(CampaignOperationsControlTestInjectionPoint::
                                afterCancellationIntentInsertion,
                            transaction);
                }
            }
            if (!persisted)
                throw Error(ErrorCode::persistenceCorruption,
                    "campaign_operations_cancellation_request_missing");
            if (!settlement && persisted->request.requestId)
            {
                candidate.emplace(LoadReconciliationCandidate(
                    transaction, *persisted->request.requestId));
                const bool activeLease =
                    candidate->requestState == RequestState::dispatching &&
                    candidate->leaseExpiresAt &&
                    transaction.exec(
                        "SELECT $1::timestamptz > "
                        "transaction_timestamp();",
                        pqxx::params{*candidate->leaseExpiresAt})
                        .one_row()[0].as<bool>();
                if (!activeLease &&
                    (candidate->requestState == RequestState::ready ||
                     candidate->requestState ==
                         RequestState::dispatching) &&
                    candidate->bindingCount == 0 &&
                    candidate->downstreamExecutionCount == 0 &&
                    (candidate->requestState == RequestState::ready ||
                     candidate->dispatchOutcomeCount == 0) &&
                    candidate->reservationState ==
                        ReservationState::held)
                    settlement.emplace(
                        SettleUnboundCancellationInTransaction(
                            transaction, *persisted, *candidate,
                            std::nullopt));
            }
            transaction.commit();
            committed.emplace(CommittedAttempt{std::move(*persisted),
                replay, std::move(candidate), std::move(settlement)});
            break;
        }
        catch (const pqxx::sql_error& error)
        {
            if (!Retryable(error) || attempt == 3) throw;
            Backoff(attempt);
        }
    }

    if (!committed)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_cancellation_request_missing");
    auto& persisted = committed->persisted;
    const bool replay = committed->replay;
    auto& candidate = committed->candidate;
    auto& settlement = committed->settlement;
    if (settlement)
        return {replay
                    ? ControlReplayDisposition::existingIdentical
                    : ControlReplayDisposition::recorded,
            CancellationProgress::settled, persisted, settlement};
    if (!persisted.request.requestId)
        return {replay
                    ? ControlReplayDisposition::existingIdentical
                    : ControlReplayDisposition::recorded,
            CancellationProgress::settled, persisted, std::nullopt};

    if (!candidate)
    {
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        SetRole(transaction, kCampaignOperationsCancellationRole);
        candidate.emplace(LoadReconciliationCandidate(
            transaction, *persisted.request.requestId));
    }
    if (candidate->requestState == RequestState::dispatching)
        return {replay
                    ? ControlReplayDisposition::existingIdentical
                    : ControlReplayDisposition::recorded,
            candidate->leaseExpiresAt && !candidate->leaseExpired
                ? CancellationProgress::waitingForLeaseExpiry
                : CancellationProgress::reconciliationRequired,
            persisted, std::nullopt};
    if (candidate->requestState != RequestState::bound)
        return {replay
                    ? ControlReplayDisposition::existingIdentical
                    : ControlReplayDisposition::recorded,
            CancellationProgress::reconciliationRequired,
            persisted, std::nullopt};

    std::vector<std::pair<DownstreamControlOwnerId, long long>> owners;
    {
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        SetRole(transaction, kCampaignOperationsCancellationRole);
        owners = LoadDownstreamControlOwners(
            transaction, candidate->requestId);
    }
    for (const auto& [ownerId, experimentId] : owners)
    {
        for (int attempt = 1; attempt <= 3; ++attempt)
        {
            try
            {
                pqxx::connection connection{connectionString};
                pqxx::work transaction{connection};
                SetRole(transaction, kExperimentLifecycleCancellationRole);
                (void)ApplyLifecycleCancellationInTransaction(
                    transaction, persisted, ownerId, experimentId);
                transaction.commit();
                break;
            }
            catch (const pqxx::sql_error& error)
            {
                if (!Retryable(error) || attempt == 3) throw;
                Backoff(attempt);
            }
        }
    }
    for (int attempt = 1; attempt <= 3; ++attempt)
        try
        {
            pqxx::connection connection{connectionString};
            pqxx::work transaction{connection};
            SetRole(transaction, kCampaignOperationsCancellationRole);
            const auto evidence = LoadLifecycleCancellationEvidence(
                transaction, persisted.cancellationRequestId);
            if (evidence.size() != owners.size())
                throw Error(ErrorCode::persistenceConflict,
                    "campaign_operations_lifecycle_cancellation_incomplete");
            auto attemptSettlement = SettleBoundCancellationInTransaction(
                transaction, persisted, evidence);
            transaction.commit();
            settlement.emplace(std::move(attemptSettlement));
            break;
        }
        catch (const pqxx::sql_error& error)
        {
            if (!Retryable(error) || attempt == 3) throw;
            Backoff(attempt);
        }
    return {replay
                ? ControlReplayDisposition::existingIdentical
                : ControlReplayDisposition::recorded,
        CancellationProgress::settled, persisted, settlement};
}

ReconciliationBatchResult ObserveAndRecoverCampaignOperations(
    const std::string& connectionString,
    const ReconciliationObserveRequest& request,
    CampaignOperationsControlTestHook testHook)
{
    const auto validated = ValidateReconciliationObserveRequest(request);
    std::optional<PersistedReconciliationBatch> committedBatch;
    for (int attempt = 1;
         attempt <=
             kCampaignOperationsReconciliationMaximumTransactionRetries;
         ++attempt)
    {
        try
        {
            pqxx::connection connection{connectionString};
            pqxx::work transaction{connection};
            SetRole(transaction, kCampaignOperationsReconcilerRole);
            RequireControlSchema(transaction);
            PersistedReconciliationBatch attemptBatch;
            if (const auto existing = FindReconciliationCursor(
                    transaction, validated.runKey,
                    validated.afterRequestId))
            {
                if (existing->requestedLimit != validated.limit)
                    throw Error(ErrorCode::persistenceConflict,
                        "campaign_operations_reconciliation_cursor_conflict");
                auto attemptObservations =
                    LoadReconciliationObservationsForCursor(
                        transaction,
                        existing->reconciliationCursorEventId);
                if (static_cast<int>(attemptObservations.size()) !=
                        existing->selectedCount)
                    throw Error(ErrorCode::persistenceCorruption,
                        "campaign_operations_reconciliation_cursor_incomplete");
                attemptBatch = {*existing,
                    std::move(attemptObservations), true};
            }
            else
            {
                const auto candidates = SelectReconciliationCandidates(
                    transaction, validated.afterRequestId,
                    validated.limit);
                std::vector<ReconciliationObservation> selected;
                selected.reserve(candidates.size());
                for (const auto& candidate : candidates)
                    selected.push_back(
                        ObservationFor(validated, candidate));
                attemptBatch = PersistReconciliationBatch(transaction,
                    validated.runKey, validated.afterRequestId,
                    validated.limit, selected, testHook);
            }
            if (testHook)
                testHook(CampaignOperationsControlTestInjectionPoint::
                        beforeReconciliationBatchCommit,
                    transaction);
            transaction.commit();
            if (testHook)
                testHook(CampaignOperationsControlTestInjectionPoint::
                        afterReconciliationBatchCommitBeforeResponse,
                    transaction);
            committedBatch.emplace(std::move(attemptBatch));
            break;
        }
        catch (const pqxx::sql_error& error)
        {
            if (!Retryable(error) ||
                attempt ==
                    kCampaignOperationsReconciliationMaximumTransactionRetries)
                throw;
            Backoff(attempt);
        }
        catch (const pqxx::in_doubt_error&)
        {
            if (auto durable = LookupCommittedReconciliationBatch(
                    connectionString, validated))
            {
                committedBatch.emplace(std::move(*durable));
                break;
            }
            if (attempt ==
                kCampaignOperationsReconciliationMaximumTransactionRetries)
                throw;
            Backoff(attempt);
        }
        catch (const pqxx::broken_connection&)
        {
            if (auto durable = LookupCommittedReconciliationBatch(
                    connectionString, validated))
            {
                committedBatch.emplace(std::move(*durable));
                break;
            }
            if (attempt ==
                kCampaignOperationsReconciliationMaximumTransactionRetries)
                throw;
            Backoff(attempt);
        }
    }
    if (!committedBatch)
        throw std::runtime_error(
            "campaign_operations_reconciliation_retry_exhausted");
    auto& cursor = committedBatch->cursor;
    auto& observations = committedBatch->observations;
    ReconciliationBatchResult result;
    result.runKey = validated.runKey;
    result.priorTargetId = validated.afterRequestId;
    result.lastTargetId = cursor.lastTargetId;
    result.selectedCount = cursor.selectedCount;
    result.observationCount = static_cast<int>(observations.size());
    if (validated.resolveSafeTransitions)
        for (const auto& persisted : observations)
            if (persisted.observation.reason ==
                ReconciliationReason::
                    dispatchLeaseExpiredNoDownstreamEvidence)
            {
                bool resolved = false;
                for (int attempt = 1;
                     attempt <=
                         kCampaignOperationsReconciliationMaximumTransactionRetries;
                     ++attempt)
                {
                    try
                    {
                        pqxx::connection connection{connectionString};
                        pqxx::work transaction{connection};
                        SetRole(transaction,
                            kCampaignOperationsRecoveryRole);
                        (void)RecoverExpiredDispatchLeaseInTransaction(
                            transaction, persisted);
                        if (testHook)
                            testHook(
                                CampaignOperationsControlTestInjectionPoint::
                                    beforeReconciliationRecoveryCommit,
                                transaction);
                        transaction.commit();
                        if (testHook)
                            testHook(
                                CampaignOperationsControlTestInjectionPoint::
                                    afterReconciliationRecoveryCommitBeforeResponse,
                                transaction);
                        resolved = true;
                        break;
                    }
                    catch (const pqxx::sql_error& error)
                    {
                        if (!Retryable(error) ||
                            attempt ==
                                kCampaignOperationsReconciliationMaximumTransactionRetries)
                            throw;
                        Backoff(attempt);
                    }
                    catch (const pqxx::in_doubt_error&)
                    {
                        if (LookupCommittedReconciliationResolution(
                                connectionString,
                                persisted.observationId))
                        {
                            resolved = true;
                            break;
                        }
                        if (attempt ==
                            kCampaignOperationsReconciliationMaximumTransactionRetries)
                            throw;
                        Backoff(attempt);
                    }
                    catch (const pqxx::broken_connection&)
                    {
                        if (LookupCommittedReconciliationResolution(
                                connectionString,
                                persisted.observationId))
                        {
                            resolved = true;
                            break;
                        }
                        if (attempt ==
                            kCampaignOperationsReconciliationMaximumTransactionRetries)
                            throw;
                        Backoff(attempt);
                    }
                }
                if (!resolved)
                    throw std::runtime_error(
                        "campaign_operations_reconciliation_recovery_retry_exhausted");
                ++result.resolutionCount;
            }
    return result;
}

CampaignControlStatus LoadCampaignControlStatus(
    pqxx::connection& connection, OperationalCampaignId campaignId,
    CampaignOperationsControlTestHook testHook)
{
    pqxx::read_transaction transaction{connection};
    transaction.exec(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
    SetRole(transaction, kCampaignOperationsReaderRole);
    RequireControlSchema(transaction);
    return LoadCampaignControlStatusProjection(
        transaction, campaignId, std::move(testHook));
}

int RunCampaignControlCommand(const std::string& connectionString,
    const CampaignControlRequest& request, std::ostream& output,
    std::ostream& errors)
{
    return RunCommand([&]
    {
        pqxx::connection connection{connectionString};
        const auto result = ControlCampaign(connection, request);
        output << "CAMPAIGN_OPERATIONS_CONTROL"
               << ",operational_campaign_id="
               << result.persisted.event.campaignId.value()
               << ",action="
               << CampaignOperationsMachineText(
                      ToText(result.persisted.event.eventKind))
               << ",control_event_id="
               << result.persisted.controlEventId.value()
               << ",control_version="
               << result.persisted.event.controlVersion
               << ",replay="
               << CampaignOperationsMachineText(ToText(result.replay))
               << '\n';
        return 0;
    }, errors, "CAMPAIGN_OPERATIONS_CONTROL");
}

int RunCampaignCancellationCommand(const std::string& connectionString,
    const CampaignCancellationCommandRequest& request,
    std::ostream& output, std::ostream& errors)
{
    return RunCommand([&]
    {
        const auto result = CancelCampaign(connectionString, request);
        output << "CAMPAIGN_OPERATIONS_CANCELLATION"
               << ",operational_campaign_id="
               << result.request.request.campaignId.value()
               << ",cancellation_request_id="
               << result.request.cancellationRequestId.value()
               << ",progress="
               << CampaignOperationsMachineText(ToText(result.progress))
               << ",settlement_id="
               << (result.settlement
                       ? std::to_string(result.settlement->
                             cancellationSettlementId.value())
                       : "NULL")
               << ",disposition="
               << (result.settlement
                       ? CampaignOperationsMachineText(ToText(
                             result.settlement->settlement.disposition))
                       : "NULL")
               << ",replay="
               << CampaignOperationsMachineText(ToText(result.replay))
               << '\n';
        return 0;
    }, errors, "CAMPAIGN_OPERATIONS_CANCELLATION");
}

int RunCampaignReconciliationCommand(const std::string& connectionString,
    const ReconciliationObserveRequest& request, std::ostream& output,
    std::ostream& errors)
{
    return RunCommand([&]
    {
        const auto result = ObserveAndRecoverCampaignOperations(
            connectionString, request);
        output << "CAMPAIGN_OPERATIONS_RECONCILIATION"
               << ",run_key="
               << CampaignOperationsMachineText(result.runKey)
               << ",prior_target_id=" << result.priorTargetId
               << ",last_target_id=" << result.lastTargetId
               << ",selected_count=" << result.selectedCount
               << ",observation_count=" << result.observationCount
               << ",resolution_count=" << result.resolutionCount << '\n';
        return 0;
    }, errors, "CAMPAIGN_OPERATIONS_RECONCILIATION");
}

int RunCampaignControlStatusCommand(const std::string& connectionString,
    OperationalCampaignId campaignId, std::ostream& output,
    std::ostream& errors)
{
    return RunCommand([&]
    {
        pqxx::connection connection{connectionString};
        const auto status = LoadCampaignControlStatus(
            connection, campaignId);
        output << "CAMPAIGN_OPERATIONS_CONTROL_STATUS"
               << ",operational_campaign_id="
               << status.campaignId.value()
               << ",paused=" << (status.paused ? "true" : "false")
               << ",control_version=" << status.controlVersion
               << ",control_event_id="
               << (status.controlEventId
                       ? std::to_string(status.controlEventId->value())
                       : "NULL")
               << ",cancellation_requested="
               << (status.cancellationRequested ? "true" : "false")
               << ",cancellation_settled="
               << (status.cancellationSettled ? "true" : "false")
               << ",cancellation_request_id="
               << (status.cancellationRequestId
                       ? std::to_string(
                             status.cancellationRequestId->value())
                       : "NULL")
               << ",cancellation_disposition="
               << (status.cancellationDisposition
                       ? CampaignOperationsMachineText(
                             ToText(*status.cancellationDisposition))
                       : "NULL")
               << ",unresolved_observations="
               << status.unresolvedObservationCount << '\n';
        return 0;
    }, errors, "CAMPAIGN_OPERATIONS_CONTROL_STATUS");
}

} // namespace EA::CampaignOperations
