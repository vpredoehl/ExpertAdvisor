#include "CampaignOperationsDispatch.hpp"

#include <algorithm>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace EA::CampaignOperations
{
namespace
{

std::string Framed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

void RequirePositive(long long value, const char* diagnostic)
{
    if (value <= 0)
        throw Error(ErrorCode::invalidIdentifier, diagnostic);
}

void RequireCanonical(const std::string& value, const char* diagnostic)
{
    if (value.empty() ||
        value.size() > kCampaignOperationsCanonicalMaximumBytes)
        throw Error(ErrorCode::invalidCanonicalText, diagnostic);
}

void RequireTaggedHash(const std::string& value, const char* diagnostic)
{
    if (value.size() != 24U || value.rfind("fnv1a64:", 0U) != 0U ||
        !std::all_of(value.begin() + 8, value.end(), [](unsigned char c)
        {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }))
        throw Error(ErrorCode::invalidCanonicalHash, diagnostic);
}

CanonicalIdentity Identity(const std::string& canonical)
{
    return CanonicalIdentity::Create(
        kCampaignOperationsDispatchContractVersion, canonical);
}

template <typename Enum>
Enum ParseText(const std::string& text,
    std::initializer_list<std::pair<const char*, Enum>> values,
    const char* diagnostic)
{
    for (const auto& [name, value] : values)
        if (text == name) return value;
    throw Error(ErrorCode::invalidEnumText, diagnostic);
}

} // namespace

DispatchTestSqlState::DispatchTestSqlState(std::string sqlState)
    : sqlState_(std::move(sqlState))
{
    if (sqlState_.size() != 5U ||
        !std::all_of(sqlState_.begin(), sqlState_.end(),
            [](unsigned char value)
            {
                return (value >= '0' && value <= '9') ||
                    (value >= 'A' && value <= 'Z');
            }))
        throw std::invalid_argument(
            "campaign_operations_test_sqlstate_invalid");
}

const char* DispatchTestSqlState::what() const noexcept
{
    return "campaign_operations_test_sqlstate";
}

const std::string& DispatchTestSqlState::sqlState() const noexcept
{
    return sqlState_;
}

LeaseTokenDigest::LeaseTokenDigest(std::string value)
    : value_(std::move(value))
{
}

LeaseTokenDigest LeaseTokenDigest::Derive(const std::string& opaqueToken)
{
    if (opaqueToken.size() < 32U || opaqueToken.size() > 512U)
        throw Error(ErrorCode::invalidOperationalRequest,
            "campaign_operations_lease_token_invalid");
    return LeaseTokenDigest(
        CanonicalIdentity::Create(1, opaqueToken).hash());
}

LeaseTokenDigest LeaseTokenDigest::Hydrate(std::string value)
{
    RequireTaggedHash(value, "campaign_operations_lease_digest_invalid");
    return LeaseTokenDigest(std::move(value));
}

const std::string& LeaseTokenDigest::value() const noexcept
{
    return value_;
}

std::string ToText(DownstreamEvidenceClassification value)
{
    switch (value)
    {
        case DownstreamEvidenceClassification::noPhase5Evidence:
            return "no_phase5_evidence";
        case DownstreamEvidenceClassification::exactCompletePendingTrain:
            return "exact_complete_pending_train";
        case DownstreamEvidenceClassification::partialPhase5Evidence:
            return "partial_phase5_evidence";
        case DownstreamEvidenceClassification::pausedOnlyEvidence:
            return "paused_only_evidence";
        case DownstreamEvidenceClassification::progressedUnboundEvidence:
            return "progressed_unbound_evidence";
        case DownstreamEvidenceClassification::causallyAmbiguous:
            return "causally_ambiguous";
        case DownstreamEvidenceClassification::
                completeCampaignOperationsBinding:
            return "complete_campaign_operations_binding";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_downstream_evidence_invalid");
}

std::string ToText(DispatchResultClassification value)
{
    switch (value)
    {
        case DispatchResultClassification::createdAndBound:
            return "created_and_bound";
        case DispatchResultClassification::adoptedExistingPendingAndBound:
            return "adopted_existing_pending_and_bound";
        case DispatchResultClassification::existingIdentical:
            return "existing_identical";
        case DispatchResultClassification::rejected: return "rejected";
        case DispatchResultClassification::semanticConflict:
            return "semantic_conflict";
        case DispatchResultClassification::reconciliationRequired:
            return "reconciliation_required";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_dispatch_result_invalid");
}

std::string ToText(ExactReplayDisposition value)
{
    switch (value)
    {
        case ExactReplayDisposition::newOperation: return "new_operation";
        case ExactReplayDisposition::authoritativeExisting:
            return "authoritative_existing";
        case ExactReplayDisposition::provenAbsent: return "proven_absent";
        case ExactReplayDisposition::changedPayloadConflict:
            return "changed_payload_conflict";
        case ExactReplayDisposition::reconciliationRequired:
            return "reconciliation_required";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_exact_replay_invalid");
}

std::string ToText(SemanticConflictClassification value)
{
    switch (value)
    {
        case SemanticConflictClassification::none: return "none";
        case SemanticConflictClassification::stateVersionMismatch:
            return "state_version_mismatch";
        case SemanticConflictClassification::leaseUnavailable:
            return "lease_unavailable";
        case SemanticConflictClassification::authorizationInactive:
            return "authorization_inactive";
        case SemanticConflictClassification::budgetInactive:
            return "budget_inactive";
        case SemanticConflictClassification::reservationMismatch:
            return "reservation_mismatch";
        case SemanticConflictClassification::requestMismatch:
            return "request_mismatch";
        case SemanticConflictClassification::controlOwnerCollision:
            return "control_owner_collision";
        case SemanticConflictClassification::partialDownstreamEvidence:
            return "partial_downstream_evidence";
        case SemanticConflictClassification::pausedOnlyEvidence:
            return "paused_only_evidence";
        case SemanticConflictClassification::progressedUnboundEvidence:
            return "progressed_unbound_evidence";
        case SemanticConflictClassification::causalityMismatch:
            return "causality_mismatch";
        case SemanticConflictClassification::bindingMismatch:
            return "binding_mismatch";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_semantic_conflict_invalid");
}

std::string ToText(UncertainCommitRecoveryClassification value)
{
    switch (value)
    {
        case UncertainCommitRecoveryClassification::
                completeAuthoritativeBinding:
            return "complete_authoritative_binding";
        case UncertainCommitRecoveryClassification::provenNoCommit:
            return "proven_no_commit";
        case UncertainCommitRecoveryClassification::ambiguousEvidence:
            return "ambiguous_evidence";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_uncertain_commit_recovery_invalid");
}

std::string ToText(Phase5ExecutionDisposition value)
{
    switch (value)
    {
        case Phase5ExecutionDisposition::created: return "created";
        case Phase5ExecutionDisposition::reused: return "reused";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_phase5_execution_disposition_invalid");
}

std::string ToText(Phase5ActivationDisposition value)
{
    switch (value)
    {
        case Phase5ActivationDisposition::created: return "created";
        case Phase5ActivationDisposition::reused: return "reused";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_phase5_activation_disposition_invalid");
}

DownstreamEvidenceClassification DownstreamEvidenceClassificationFromText(
    const std::string& text)
{
    return ParseText<DownstreamEvidenceClassification>(text, {
        {"no_phase5_evidence",
            DownstreamEvidenceClassification::noPhase5Evidence},
        {"exact_complete_pending_train",
            DownstreamEvidenceClassification::exactCompletePendingTrain},
        {"partial_phase5_evidence",
            DownstreamEvidenceClassification::partialPhase5Evidence},
        {"paused_only_evidence",
            DownstreamEvidenceClassification::pausedOnlyEvidence},
        {"progressed_unbound_evidence",
            DownstreamEvidenceClassification::progressedUnboundEvidence},
        {"causally_ambiguous",
            DownstreamEvidenceClassification::causallyAmbiguous},
        {"complete_campaign_operations_binding",
            DownstreamEvidenceClassification::
                completeCampaignOperationsBinding}},
        "campaign_operations_downstream_evidence_text_invalid");
}

DispatchResultClassification DispatchResultClassificationFromText(
    const std::string& text)
{
    return ParseText<DispatchResultClassification>(text, {
        {"created_and_bound",
            DispatchResultClassification::createdAndBound},
        {"adopted_existing_pending_and_bound",
            DispatchResultClassification::adoptedExistingPendingAndBound},
        {"existing_identical",
            DispatchResultClassification::existingIdentical},
        {"rejected", DispatchResultClassification::rejected},
        {"semantic_conflict",
            DispatchResultClassification::semanticConflict},
        {"reconciliation_required",
            DispatchResultClassification::reconciliationRequired}},
        "campaign_operations_dispatch_result_text_invalid");
}

ExactReplayDisposition ExactReplayDispositionFromText(
    const std::string& text)
{
    return ParseText<ExactReplayDisposition>(text, {
        {"new_operation", ExactReplayDisposition::newOperation},
        {"authoritative_existing",
            ExactReplayDisposition::authoritativeExisting},
        {"proven_absent", ExactReplayDisposition::provenAbsent},
        {"changed_payload_conflict",
            ExactReplayDisposition::changedPayloadConflict},
        {"reconciliation_required",
            ExactReplayDisposition::reconciliationRequired}},
        "campaign_operations_exact_replay_text_invalid");
}

SemanticConflictClassification SemanticConflictClassificationFromText(
    const std::string& text)
{
    return ParseText<SemanticConflictClassification>(text, {
        {"none", SemanticConflictClassification::none},
        {"state_version_mismatch",
            SemanticConflictClassification::stateVersionMismatch},
        {"lease_unavailable",
            SemanticConflictClassification::leaseUnavailable},
        {"authorization_inactive",
            SemanticConflictClassification::authorizationInactive},
        {"budget_inactive",
            SemanticConflictClassification::budgetInactive},
        {"reservation_mismatch",
            SemanticConflictClassification::reservationMismatch},
        {"request_mismatch",
            SemanticConflictClassification::requestMismatch},
        {"control_owner_collision",
            SemanticConflictClassification::controlOwnerCollision},
        {"partial_downstream_evidence",
            SemanticConflictClassification::partialDownstreamEvidence},
        {"paused_only_evidence",
            SemanticConflictClassification::pausedOnlyEvidence},
        {"progressed_unbound_evidence",
            SemanticConflictClassification::progressedUnboundEvidence},
        {"causality_mismatch",
            SemanticConflictClassification::causalityMismatch},
        {"binding_mismatch",
            SemanticConflictClassification::bindingMismatch}},
        "campaign_operations_semantic_conflict_text_invalid");
}

UncertainCommitRecoveryClassification
UncertainCommitRecoveryClassificationFromText(const std::string& text)
{
    return ParseText<UncertainCommitRecoveryClassification>(text, {
        {"complete_authoritative_binding",
            UncertainCommitRecoveryClassification::
                completeAuthoritativeBinding},
        {"proven_no_commit",
            UncertainCommitRecoveryClassification::provenNoCommit},
        {"ambiguous_evidence",
            UncertainCommitRecoveryClassification::ambiguousEvidence}},
        "campaign_operations_uncertain_commit_recovery_text_invalid");
}

Phase5ExecutionDisposition Phase5ExecutionDispositionFromText(
    const std::string& text)
{
    return ParseText<Phase5ExecutionDisposition>(text, {
        {"created", Phase5ExecutionDisposition::created},
        {"reused", Phase5ExecutionDisposition::reused}},
        "campaign_operations_phase5_execution_disposition_text_invalid");
}

Phase5ActivationDisposition Phase5ActivationDispositionFromText(
    const std::string& text)
{
    return ParseText<Phase5ActivationDisposition>(text, {
        {"created", Phase5ActivationDisposition::created},
        {"reused", Phase5ActivationDisposition::reused}},
        "campaign_operations_phase5_activation_disposition_text_invalid");
}

DispatchAttemptAcquisition::DispatchAttemptAcquisition(
    CanonicalIdentity identityValue, OperationalRequestId requestIdValue,
    std::string requestCanonicalTextValue, int attemptOrdinalValue,
    int expectedRequestVersionValue, int resultingRequestVersionValue,
    LeaseTokenDigest leaseTokenDigestValue,
    UtcTimestamp leaseExpiresAtValue, ActorIdentity dispatcherValue)
    : identity(std::move(identityValue)), requestId(requestIdValue),
      requestCanonicalText(std::move(requestCanonicalTextValue)),
      attemptOrdinal(attemptOrdinalValue),
      expectedRequestVersion(expectedRequestVersionValue),
      resultingRequestVersion(resultingRequestVersionValue),
      leaseTokenDigest(std::move(leaseTokenDigestValue)),
      leaseExpiresAt(std::move(leaseExpiresAtValue)),
      dispatcher(std::move(dispatcherValue))
{
}

DispatchAttemptAcquisition BuildDispatchAttemptAcquisition(
    OperationalRequestId requestId, std::string requestCanonicalText,
    int attemptOrdinal, int expectedRequestVersion,
    int resultingRequestVersion, LeaseTokenDigest leaseTokenDigest,
    UtcTimestamp leaseExpiresAt, ActorIdentity dispatcher)
{
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_dispatch_attempt_v1"
              << ";request_id=" << requestId.value()
              << ";request_identity_canonical="
              << Framed(requestCanonicalText)
              << ";attempt_ordinal=" << attemptOrdinal
              << ";expected_request_version=" << expectedRequestVersion
              << ";resulting_request_version=" << resultingRequestVersion
              << ";lease_token_digest=" << Framed(leaseTokenDigest.value())
              << ";lease_expires_at=" << Framed(leaseExpiresAt.value())
              << ";dispatcher_identity=" << Framed(dispatcher.value());
    DispatchAttemptAcquisition result(Identity(canonical.str()), requestId,
        std::move(requestCanonicalText), attemptOrdinal,
        expectedRequestVersion, resultingRequestVersion,
        std::move(leaseTokenDigest), std::move(leaseExpiresAt),
        std::move(dispatcher));
    ValidateDispatchAttemptAcquisition(result);
    return result;
}

void ValidateDispatchAttemptAcquisition(
    const DispatchAttemptAcquisition& acquisition)
{
    RequireCanonical(acquisition.requestCanonicalText,
        "campaign_operations_dispatch_attempt_request_invalid");
    if (acquisition.attemptOrdinal <= 0 ||
        acquisition.expectedRequestVersion <= 0 ||
        acquisition.resultingRequestVersion !=
            acquisition.expectedRequestVersion + 1)
        throw Error(ErrorCode::invalidOperationalRequest,
            "campaign_operations_dispatch_attempt_version_invalid");
}

RequestMemberBinding::RequestMemberBinding(
    CanonicalIdentity identityValue, OperationalRequestId requestIdValue,
    std::string requestCanonicalTextValue, long long materializationIdValue,
    std::string materializationCanonicalTextValue,
    long long materializationMemberIdValue, int memberOrdinalValue,
    std::string selectedMemberCanonicalTextValue,
    std::string selectedMemberIdentityHashValue, long long proposalIdValue,
    std::string proposalCanonicalTextValue,
    std::string proposalIdentityHashValue, long long reviewDecisionIdValue,
    long long executionIdValue, std::string executionCanonicalTextValue,
    std::string executionIdentityHashValue, long long activationIdValue,
    std::string activationCanonicalTextValue,
    std::string activationIdentityHashValue, long long experimentIdValue,
    BindingDisposition bindingDispositionValue,
    Phase5ExecutionDisposition executionDispositionValue,
    Phase5ActivationDisposition activationDispositionValue)
    : identity(std::move(identityValue)), requestId(requestIdValue),
      requestCanonicalText(std::move(requestCanonicalTextValue)),
      materializationId(materializationIdValue),
      materializationCanonicalText(
          std::move(materializationCanonicalTextValue)),
      materializationMemberId(materializationMemberIdValue),
      memberOrdinal(memberOrdinalValue),
      selectedMemberCanonicalText(
          std::move(selectedMemberCanonicalTextValue)),
      selectedMemberIdentityHash(std::move(selectedMemberIdentityHashValue)),
      proposalId(proposalIdValue),
      proposalCanonicalText(std::move(proposalCanonicalTextValue)),
      proposalIdentityHash(std::move(proposalIdentityHashValue)),
      reviewDecisionId(reviewDecisionIdValue), executionId(executionIdValue),
      executionCanonicalText(std::move(executionCanonicalTextValue)),
      executionIdentityHash(std::move(executionIdentityHashValue)),
      activationId(activationIdValue),
      activationCanonicalText(std::move(activationCanonicalTextValue)),
      activationIdentityHash(std::move(activationIdentityHashValue)),
      experimentId(experimentIdValue),
      bindingDisposition(bindingDispositionValue),
      executionDisposition(executionDispositionValue),
      activationDisposition(activationDispositionValue)
{
}

RequestMemberBinding BuildRequestMemberBinding(
    OperationalRequestId requestId, std::string requestCanonicalText,
    long long materializationId, std::string materializationCanonicalText,
    long long materializationMemberId, int memberOrdinal,
    std::string selectedMemberCanonicalText,
    std::string selectedMemberIdentityHash, long long proposalId,
    std::string proposalCanonicalText, std::string proposalIdentityHash,
    long long reviewDecisionId, long long executionId,
    std::string executionCanonicalText, std::string executionIdentityHash,
    long long activationId, std::string activationCanonicalText,
    std::string activationIdentityHash, long long experimentId,
    BindingDisposition bindingDisposition,
    Phase5ExecutionDisposition executionDisposition,
    Phase5ActivationDisposition activationDisposition)
{
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_dispatch_binding_v1"
              << ";request_id=" << requestId.value()
              << ";request_identity_canonical="
              << Framed(requestCanonicalText)
              << ";materialization_id=" << materializationId
              << ";materialization_identity_canonical="
              << Framed(materializationCanonicalText)
              << ";materialization_member_id=" << materializationMemberId
              << ";member_ordinal=" << memberOrdinal
              << ";selected_member_identity_canonical="
              << Framed(selectedMemberCanonicalText)
              << ";selected_member_identity_hash="
              << Framed(selectedMemberIdentityHash)
              << ";proposal_id=" << proposalId
              << ";proposal_identity_canonical="
              << Framed(proposalCanonicalText)
              << ";proposal_identity_hash=" << Framed(proposalIdentityHash)
              << ";review_decision_id=" << reviewDecisionId
              << ";execution_id=" << executionId
              << ";execution_identity_canonical="
              << Framed(executionCanonicalText)
              << ";execution_identity_hash=" << Framed(executionIdentityHash)
              << ";activation_id=" << activationId
              << ";activation_identity_canonical="
              << Framed(activationCanonicalText)
              << ";activation_identity_hash="
              << Framed(activationIdentityHash)
              << ";experiment_id=" << experimentId
              << ";binding_disposition=" << ToText(bindingDisposition)
              << ";execution_disposition=" << ToText(executionDisposition)
              << ";activation_disposition=" << ToText(activationDisposition);
    RequestMemberBinding result(Identity(canonical.str()), requestId,
        std::move(requestCanonicalText), materializationId,
        std::move(materializationCanonicalText), materializationMemberId,
        memberOrdinal, std::move(selectedMemberCanonicalText),
        std::move(selectedMemberIdentityHash), proposalId,
        std::move(proposalCanonicalText), std::move(proposalIdentityHash),
        reviewDecisionId, executionId, std::move(executionCanonicalText),
        std::move(executionIdentityHash), activationId,
        std::move(activationCanonicalText),
        std::move(activationIdentityHash), experimentId, bindingDisposition,
        executionDisposition, activationDisposition);
    ValidateRequestMemberBinding(result);
    return result;
}

void ValidateRequestMemberBinding(const RequestMemberBinding& binding)
{
    RequirePositive(binding.materializationId,
        "campaign_operations_binding_materialization_id_invalid");
    RequirePositive(binding.materializationMemberId,
        "campaign_operations_binding_member_id_invalid");
    RequirePositive(binding.proposalId,
        "campaign_operations_binding_proposal_id_invalid");
    RequirePositive(binding.reviewDecisionId,
        "campaign_operations_binding_review_id_invalid");
    RequirePositive(binding.executionId,
        "campaign_operations_binding_execution_id_invalid");
    RequirePositive(binding.activationId,
        "campaign_operations_binding_activation_id_invalid");
    RequirePositive(binding.experimentId,
        "campaign_operations_binding_experiment_id_invalid");
    if (binding.memberOrdinal <= 0)
        throw Error(ErrorCode::invalidIdentifier,
            "campaign_operations_binding_ordinal_invalid");
    RequireCanonical(binding.requestCanonicalText,
        "campaign_operations_binding_request_invalid");
    RequireCanonical(binding.materializationCanonicalText,
        "campaign_operations_binding_materialization_invalid");
    RequireCanonical(binding.selectedMemberCanonicalText,
        "campaign_operations_binding_selected_member_invalid");
    RequireCanonical(binding.proposalCanonicalText,
        "campaign_operations_binding_proposal_invalid");
    RequireCanonical(binding.executionCanonicalText,
        "campaign_operations_binding_execution_invalid");
    RequireCanonical(binding.activationCanonicalText,
        "campaign_operations_binding_activation_invalid");
    RequireTaggedHash(binding.selectedMemberIdentityHash,
        "campaign_operations_binding_selected_member_hash_invalid");
    RequireTaggedHash(binding.proposalIdentityHash,
        "campaign_operations_binding_proposal_hash_invalid");
    RequireTaggedHash(binding.executionIdentityHash,
        "campaign_operations_binding_execution_hash_invalid");
    RequireTaggedHash(binding.activationIdentityHash,
        "campaign_operations_binding_activation_hash_invalid");
    const bool created =
        binding.bindingDisposition == BindingDisposition::created;
    const bool exactCreated =
        binding.executionDisposition == Phase5ExecutionDisposition::created &&
        binding.activationDisposition == Phase5ActivationDisposition::created;
    const bool exactAdoption =
        binding.executionDisposition == Phase5ExecutionDisposition::reused &&
        binding.activationDisposition == Phase5ActivationDisposition::reused;
    if ((created && !exactCreated) || (!created && !exactAdoption))
        throw Error(ErrorCode::invalidOperationalRequest,
            "campaign_operations_binding_disposition_invalid");
}

RequestBindingSet::RequestBindingSet(CanonicalIdentity identityValue,
    OperationalRequestId requestIdValue,
    std::string requestCanonicalTextValue, int memberCountValue,
    std::vector<RequestMemberBinding> membersValue)
    : identity(std::move(identityValue)), requestId(requestIdValue),
      requestCanonicalText(std::move(requestCanonicalTextValue)),
      memberCount(memberCountValue), members(std::move(membersValue))
{
}

RequestBindingSet BuildRequestBindingSet(OperationalRequestId requestId,
    std::string requestCanonicalText,
    std::vector<RequestMemberBinding> members)
{
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_request_binding_v1"
              << ";request_id=" << requestId.value()
              << ";request_identity_canonical="
              << Framed(requestCanonicalText)
              << ";member_count=" << members.size();
    for (const auto& member : members)
        canonical << ";member_" << member.memberOrdinal
                  << "_binding_canonical="
                  << Framed(member.identity.canonicalText());
    RequestBindingSet result(Identity(canonical.str()), requestId,
        std::move(requestCanonicalText),
        static_cast<int>(members.size()), std::move(members));
    ValidateRequestBindingSet(result);
    return result;
}

void ValidateRequestBindingSet(const RequestBindingSet& bindingSet)
{
    RequireCanonical(bindingSet.requestCanonicalText,
        "campaign_operations_binding_set_request_invalid");
    if (bindingSet.memberCount <= 0 ||
        bindingSet.memberCount !=
            static_cast<int>(bindingSet.members.size()))
        throw Error(ErrorCode::invalidOperationalRequest,
            "campaign_operations_binding_set_count_invalid");
    for (std::size_t index = 0; index < bindingSet.members.size(); ++index)
    {
        const auto& member = bindingSet.members[index];
        ValidateRequestMemberBinding(member);
        if (member.requestId != bindingSet.requestId ||
            member.requestCanonicalText !=
                bindingSet.requestCanonicalText ||
            member.memberOrdinal != static_cast<int>(index) + 1)
            throw Error(ErrorCode::invalidOperationalRequest,
                "campaign_operations_binding_set_member_invalid");
        if (member.bindingDisposition !=
                bindingSet.members.front().bindingDisposition ||
            member.executionDisposition !=
                bindingSet.members.front().executionDisposition ||
            member.activationDisposition !=
                bindingSet.members.front().activationDisposition)
            throw Error(ErrorCode::invalidOperationalRequest,
                "campaign_operations_binding_set_disposition_invalid");
    }
}

DownstreamControlOwner::DownstreamControlOwner(
    CanonicalIdentity identityValue, OperationalRequestId requestIdValue,
    std::string requestCanonicalTextValue,
    std::string bindingCanonicalTextValue, long long experimentIdValue,
    DownstreamControlMode modeValue,
    std::optional<AuthorizationEventId> adoptionAuthorizationEventIdValue,
    std::optional<std::string> adoptionAuthorizationCanonicalTextValue,
    std::optional<std::string> adoptionAuthorizationIdentityHashValue)
    : identity(std::move(identityValue)), requestId(requestIdValue),
      requestCanonicalText(std::move(requestCanonicalTextValue)),
      bindingCanonicalText(std::move(bindingCanonicalTextValue)),
      experimentId(experimentIdValue), mode(modeValue),
      adoptionAuthorizationEventId(
          std::move(adoptionAuthorizationEventIdValue)),
      adoptionAuthorizationCanonicalText(
          std::move(adoptionAuthorizationCanonicalTextValue)),
      adoptionAuthorizationIdentityHash(
          std::move(adoptionAuthorizationIdentityHashValue))
{
}

DownstreamControlOwner BuildDownstreamControlOwner(
    OperationalRequestId requestId, std::string requestCanonicalText,
    std::string bindingCanonicalText, long long experimentId,
    DownstreamControlMode mode,
    std::optional<AuthorizationEventId> adoptionAuthorizationEventId,
    std::optional<std::string> adoptionAuthorizationCanonicalText,
    std::optional<std::string> adoptionAuthorizationIdentityHash)
{
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_downstream_control_owner_v1"
              << ";request_id=" << requestId.value()
              << ";request_identity_canonical="
              << Framed(requestCanonicalText)
              << ";binding_identity_canonical="
              << Framed(bindingCanonicalText)
              << ";experiment_id=" << experimentId
              << ";control_mode=" << ToText(mode)
              << ";adoption_authorization_event_id=";
    if (adoptionAuthorizationEventId)
        canonical << adoptionAuthorizationEventId->value();
    else
        canonical << "none";
    canonical << ";adoption_authorization_identity_canonical="
              << (adoptionAuthorizationCanonicalText
                      ? Framed(*adoptionAuthorizationCanonicalText)
                      : "none")
              << ";adoption_authorization_identity_hash="
              << (adoptionAuthorizationIdentityHash
                      ? Framed(*adoptionAuthorizationIdentityHash)
                      : "none");
    DownstreamControlOwner result(Identity(canonical.str()), requestId,
        std::move(requestCanonicalText), std::move(bindingCanonicalText),
        experimentId, mode, std::move(adoptionAuthorizationEventId),
        std::move(adoptionAuthorizationCanonicalText),
        std::move(adoptionAuthorizationIdentityHash));
    ValidateDownstreamControlOwner(result);
    return result;
}

void ValidateDownstreamControlOwner(const DownstreamControlOwner& owner)
{
    RequirePositive(owner.experimentId,
        "campaign_operations_control_owner_experiment_invalid");
    RequireCanonical(owner.requestCanonicalText,
        "campaign_operations_control_owner_request_invalid");
    RequireCanonical(owner.bindingCanonicalText,
        "campaign_operations_control_owner_binding_invalid");
    const bool adoption =
        owner.mode == DownstreamControlMode::authorizedAdoptionControl;
    if (adoption != owner.adoptionAuthorizationEventId.has_value() ||
        adoption != owner.adoptionAuthorizationCanonicalText.has_value() ||
        adoption != owner.adoptionAuthorizationIdentityHash.has_value())
        throw Error(ErrorCode::invalidOperationalRequest,
            "campaign_operations_control_owner_adoption_shape_invalid");
    if (adoption)
    {
        RequireCanonical(*owner.adoptionAuthorizationCanonicalText,
            "campaign_operations_control_owner_authorization_invalid");
        RequireTaggedHash(*owner.adoptionAuthorizationIdentityHash,
            "campaign_operations_control_owner_authorization_hash_invalid");
    }
}

ReservationCommitment::ReservationCommitment(
    CanonicalIdentity identityValue, ReservationId reservationIdValue,
    std::string reservationCanonicalTextValue,
    OperationalRequestId requestIdValue,
    std::string requestCanonicalTextValue,
    std::string bindingSetCanonicalTextValue,
    std::string bindingSetIdentityHashValue,
    int expectedReservationVersionValue,
    int resultingReservationVersionValue, long long amountValue)
    : identity(std::move(identityValue)), reservationId(reservationIdValue),
      reservationCanonicalText(std::move(reservationCanonicalTextValue)),
      requestId(requestIdValue),
      requestCanonicalText(std::move(requestCanonicalTextValue)),
      bindingSetCanonicalText(std::move(bindingSetCanonicalTextValue)),
      bindingSetIdentityHash(std::move(bindingSetIdentityHashValue)),
      expectedReservationVersion(expectedReservationVersionValue),
      resultingReservationVersion(resultingReservationVersionValue),
      amount(amountValue)
{
}

ReservationCommitment BuildReservationCommitment(
    ReservationId reservationId, std::string reservationCanonicalText,
    OperationalRequestId requestId, std::string requestCanonicalText,
    std::string bindingSetCanonicalText, std::string bindingSetIdentityHash,
    int expectedReservationVersion, int resultingReservationVersion,
    long long amount)
{
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_reservation_event_v1"
              << ";reservation_id=" << reservationId.value()
              << ";reservation_identity_canonical="
              << Framed(reservationCanonicalText)
              << ";transition_kind=committed"
              << ";expected_state=held"
              << ";resulting_state=committed"
              << ";expected_version=" << expectedReservationVersion
              << ";resulting_version=" << resultingReservationVersion
              << ";request_id=" << requestId.value()
              << ";request_identity_canonical="
              << Framed(requestCanonicalText)
              << ";binding_set_identity_canonical="
              << Framed(bindingSetCanonicalText)
              << ";binding_set_identity_hash="
              << Framed(bindingSetIdentityHash)
              << ";amount=" << amount;
    ReservationCommitment result(Identity(canonical.str()), reservationId,
        std::move(reservationCanonicalText), requestId,
        std::move(requestCanonicalText), std::move(bindingSetCanonicalText),
        std::move(bindingSetIdentityHash), expectedReservationVersion,
        resultingReservationVersion, amount);
    ValidateReservationCommitment(result);
    return result;
}

void ValidateReservationCommitment(const ReservationCommitment& commitment)
{
    RequireCanonical(commitment.reservationCanonicalText,
        "campaign_operations_commitment_reservation_invalid");
    RequireCanonical(commitment.requestCanonicalText,
        "campaign_operations_commitment_request_invalid");
    RequireCanonical(commitment.bindingSetCanonicalText,
        "campaign_operations_commitment_binding_set_invalid");
    RequireTaggedHash(commitment.bindingSetIdentityHash,
        "campaign_operations_commitment_binding_set_hash_invalid");
    if (commitment.expectedReservationVersion <= 0 ||
        commitment.resultingReservationVersion !=
            commitment.expectedReservationVersion + 1 ||
        commitment.amount <= 0)
        throw Error(ErrorCode::invalidReservationEvent,
            "campaign_operations_commitment_version_invalid");
}

DispatchAttemptOutcomeEvidence::DispatchAttemptOutcomeEvidence(
    CanonicalIdentity identityValue, DispatchAttemptId attemptIdValue,
    std::string attemptCanonicalTextValue,
    DispatchResultClassification resultValue,
    DownstreamEvidenceClassification downstreamEvidenceValue,
    SemanticConflictClassification conflictValue,
    UncertainCommitRecoveryClassification recoveryValue,
    std::string diagnosticCodeValue, int expectedRequestVersionValue,
    int resultingRequestVersionValue, int expectedReservationVersionValue,
    int resultingReservationVersionValue,
    std::optional<std::string> bindingSetCanonicalTextValue,
    std::optional<std::string> bindingSetIdentityHashValue)
    : identity(std::move(identityValue)), attemptId(attemptIdValue),
      attemptCanonicalText(std::move(attemptCanonicalTextValue)),
      result(resultValue), downstreamEvidence(downstreamEvidenceValue),
      conflict(conflictValue), recovery(recoveryValue),
      diagnosticCode(std::move(diagnosticCodeValue)),
      expectedRequestVersion(expectedRequestVersionValue),
      resultingRequestVersion(resultingRequestVersionValue),
      expectedReservationVersion(expectedReservationVersionValue),
      resultingReservationVersion(resultingReservationVersionValue),
      bindingSetCanonicalText(std::move(bindingSetCanonicalTextValue)),
      bindingSetIdentityHash(std::move(bindingSetIdentityHashValue))
{
}

DispatchAttemptOutcomeEvidence BuildDispatchAttemptOutcomeEvidence(
    DispatchAttemptId attemptId, std::string attemptCanonicalText,
    DispatchResultClassification result,
    DownstreamEvidenceClassification downstreamEvidence,
    SemanticConflictClassification conflict,
    UncertainCommitRecoveryClassification recovery,
    std::string diagnosticCode, int expectedRequestVersion,
    int resultingRequestVersion, int expectedReservationVersion,
    int resultingReservationVersion,
    std::optional<std::string> bindingSetCanonicalText,
    std::optional<std::string> bindingSetIdentityHash)
{
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_dispatch_attempt_outcome_v1"
              << ";attempt_id=" << attemptId.value()
              << ";attempt_identity_canonical="
              << Framed(attemptCanonicalText)
              << ";result=" << ToText(result)
              << ";downstream_evidence=" << ToText(downstreamEvidence)
              << ";semantic_conflict=" << ToText(conflict)
              << ";uncertain_commit_recovery=" << ToText(recovery)
              << ";diagnostic_code=" << Framed(diagnosticCode)
              << ";expected_request_version=" << expectedRequestVersion
              << ";resulting_request_version=" << resultingRequestVersion
              << ";expected_reservation_version="
              << expectedReservationVersion
              << ";resulting_reservation_version="
              << resultingReservationVersion
              << ";binding_set_identity_canonical="
              << (bindingSetCanonicalText
                      ? Framed(*bindingSetCanonicalText)
                      : "none")
              << ";binding_set_identity_hash="
              << (bindingSetIdentityHash
                      ? Framed(*bindingSetIdentityHash)
                      : "none");
    DispatchAttemptOutcomeEvidence outcome(Identity(canonical.str()),
        attemptId, std::move(attemptCanonicalText), result,
        downstreamEvidence, conflict, recovery, std::move(diagnosticCode),
        expectedRequestVersion, resultingRequestVersion,
        expectedReservationVersion, resultingReservationVersion,
        std::move(bindingSetCanonicalText),
        std::move(bindingSetIdentityHash));
    ValidateDispatchAttemptOutcomeEvidence(outcome);
    return outcome;
}

void ValidateDispatchAttemptOutcomeEvidence(
    const DispatchAttemptOutcomeEvidence& outcome)
{
    RequireCanonical(outcome.attemptCanonicalText,
        "campaign_operations_outcome_attempt_invalid");
    if (outcome.diagnosticCode.empty() ||
        outcome.diagnosticCode.size() > 256U)
        throw Error(ErrorCode::invalidOperationalRequest,
            "campaign_operations_outcome_diagnostic_invalid");
    if (outcome.expectedRequestVersion <= 0 ||
        outcome.resultingRequestVersion < outcome.expectedRequestVersion ||
        outcome.expectedReservationVersion <= 0 ||
        outcome.resultingReservationVersion <
            outcome.expectedReservationVersion)
        throw Error(ErrorCode::invalidOperationalRequest,
            "campaign_operations_outcome_version_invalid");
    if (outcome.bindingSetCanonicalText.has_value() !=
            outcome.bindingSetIdentityHash.has_value())
        throw Error(ErrorCode::invalidOperationalRequest,
            "campaign_operations_outcome_binding_shape_invalid");
    const bool successful =
        outcome.result == DispatchResultClassification::createdAndBound ||
        outcome.result ==
            DispatchResultClassification::adoptedExistingPendingAndBound ||
        outcome.result == DispatchResultClassification::existingIdentical;
    if (successful != outcome.bindingSetCanonicalText.has_value())
        throw Error(ErrorCode::invalidOperationalRequest,
            "campaign_operations_outcome_result_shape_invalid");
    if (outcome.bindingSetCanonicalText)
    {
        RequireCanonical(*outcome.bindingSetCanonicalText,
            "campaign_operations_outcome_binding_invalid");
        RequireTaggedHash(*outcome.bindingSetIdentityHash,
            "campaign_operations_outcome_binding_hash_invalid");
    }
}

} // namespace EA::CampaignOperations
