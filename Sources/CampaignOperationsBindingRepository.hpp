#pragma once

#include "CampaignOperationsDispatchRepository.hpp"
#include "ExperimentRecommendationCampaignLaunchRepository.hpp"

#include <optional>

namespace EA::CampaignOperations
{

struct PersistedDispatchBinding final
{
    RequestBindingSet bindingSet;
    DispatchAttemptOutcomeEvidence outcome;
};

std::optional<PersistedDispatchBinding> FindAndValidateCompleteDispatchBinding(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    bool production = false);
std::optional<DispatchAttemptOutcomeEvidence>
FindLatestDispatchOutcome(pqxx::transaction_base& transaction,
    OperationalRequestId requestId, bool production = false);

RequestBindingSet PersistCompleteDispatchBinding(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const ExperimentRecommendation::
        PersistedRecommendationCampaignMaterialization& materialization,
    const ExperimentRecommendation::
        RecommendationCampaignLaunchPersistResult& launchResult,
    BindingDisposition disposition, DispatchTestHook testHook = {});

void PersistControlOwners(pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const RequestBindingSet& bindingSet, DispatchTestHook testHook = {});

ReservationCommitment CommitReservationForBinding(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const RequestBindingSet& bindingSet, DispatchTestHook testHook = {});

DispatchAttemptOutcomeEvidence BindRequestAndPersistSuccessfulOutcome(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const DispatchAttemptRecord& attempt,
    const RequestBindingSet& bindingSet,
    DownstreamEvidenceClassification downstreamEvidence,
    BindingDisposition disposition, DispatchTestHook testHook = {},
    bool production = false, const std::string& productionOperationKey = {},
    const std::string& approvedBuildContractCanonical = {});

DispatchAttemptOutcomeEvidence PersistFailedDispatchOutcome(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const DispatchAttemptRecord& attempt,
    DownstreamEvidenceClassification downstreamEvidence,
    SemanticConflictClassification conflict,
    const std::string& diagnosticCode, bool production = false);

} // namespace EA::CampaignOperations
