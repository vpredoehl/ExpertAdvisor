#include "CampaignOperationsBindingRepository.hpp"
#include "CampaignOperationsProductionAdmission.hpp"
#include "CampaignOperationsProductionAdmissionRepository.hpp"

#include <stdexcept>

namespace EA::CampaignOperations
{
namespace
{

RequestMemberBinding MapBinding(const pqxx::row& row)
{
    const auto disposition = [&]
    {
        const auto text = row[19].as<std::string>();
        if (text == "created") return BindingDisposition::created;
        if (text == "adopted_existing_pending")
            return BindingDisposition::adoptedExistingPending;
        throw std::runtime_error(
            "campaign_operations_binding_disposition_corrupt");
    }();
    return BuildRequestMemberBinding(
        OperationalRequestId(row[0].as<long long>()),
        row[1].as<std::string>(), row[2].as<long long>(),
        row[3].as<std::string>(), row[4].as<long long>(),
        row[5].as<int>(), row[6].as<std::string>(),
        row[7].as<std::string>(), row[8].as<long long>(),
        row[9].as<std::string>(), row[10].as<std::string>(),
        row[11].as<long long>(), row[12].as<long long>(),
        row[13].as<std::string>(), row[14].as<std::string>(),
        row[15].as<long long>(), row[16].as<std::string>(),
        row[17].as<std::string>(), row[18].as<long long>(),
        disposition,
        Phase5ExecutionDispositionFromText(row[20].as<std::string>()),
        Phase5ActivationDispositionFromText(row[21].as<std::string>()));
}

pqxx::result LoadBindingRows(pqxx::transaction_base& transaction,
    OperationalRequestId requestId)
{
    return transaction.exec(
        "SELECT operational_request_id,request_identity_canonical,"
        "recommendation_campaign_materialization_id,"
        "materialization_identity_canonical,"
        "recommendation_campaign_materialization_member_id,member_ordinal,"
        "selected_member_identity_canonical,selected_member_identity_hash,"
        "recommendation_conversion_proposal_id,"
        "proposal_identity_canonical,proposal_identity_hash,"
        "recommendation_conversion_review_decision_id,"
        "recommendation_conversion_execution_id,"
        "execution_identity_canonical,execution_identity_hash,"
        "recommendation_conversion_activation_id,"
        "activation_identity_canonical,activation_identity_hash,"
        "experiment_id,binding_disposition,execution_disposition,"
        "activation_disposition,binding_identity_canonical,"
        "binding_identity_hash "
        "FROM campaign_operations_request_binding "
        "WHERE operational_request_id=$1 ORDER BY member_ordinal;",
        pqxx::params{requestId.value()});
}

long long InsertOutcome(pqxx::transaction_base& transaction,
    const DispatchAttemptOutcomeEvidence& outcome)
{
    return transaction.exec(
        "INSERT INTO campaign_operations_dispatch_attempt_outcome("
        "dispatch_attempt_id,attempt_identity_canonical,"
        "result_classification,downstream_evidence_classification,"
        "semantic_conflict_classification,"
        "uncertain_commit_recovery_classification,diagnostic_code,"
        "request_binding_set_identity_canonical,"
        "request_binding_set_identity_hash,expected_request_version,"
        "resulting_request_version,expected_reservation_version,"
        "resulting_reservation_version,outcome_contract_version,"
        "outcome_identity_canonical,outcome_identity_hash) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,1,$14,$15) "
        "RETURNING dispatch_attempt_outcome_id;",
        pqxx::params{outcome.attemptId.value(),
            outcome.attemptCanonicalText, ToText(outcome.result),
            ToText(outcome.downstreamEvidence), ToText(outcome.conflict),
            ToText(outcome.recovery), outcome.diagnosticCode,
            outcome.bindingSetCanonicalText,
            outcome.bindingSetIdentityHash, outcome.expectedRequestVersion,
            outcome.resultingRequestVersion,
            outcome.expectedReservationVersion,
            outcome.resultingReservationVersion,
            outcome.identity.canonicalText(), outcome.identity.hash()})
        .one_row()[0].as<long long>();
}

void InsertOutcomeAudit(pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const DispatchAttemptRecord& attempt, long long outcomeId,
    const DispatchAttemptOutcomeEvidence& outcome, bool production)
{
    (void)production;
    const bool successful =
        outcome.result == DispatchResultClassification::createdAndBound ||
        outcome.result ==
            DispatchResultClassification::adoptedExistingPendingAndBound ||
        outcome.result == DispatchResultClassification::existingIdentical;
    transaction.exec(
        "INSERT INTO campaign_operations_dispatch_audit_reference_event("
        "operational_campaign_id,operational_request_id,"
        "dispatch_attempt_id,dispatch_attempt_outcome_id,cause_kind,"
        "actor_identity,capability,prior_version,resulting_version,"
        "outcome,replay_disposition,diagnostic_code) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12);",
        pqxx::params{authority.campaignId.value(),
            authority.requestId.value(), attempt.attemptId.value(),
            outcomeId, successful ? "dispatch_handoff_completed"
                                  : "dispatch_handoff_failed",
            attempt.acquisition.dispatcher.value(),
            // H2 changes runtime reachability, not migration 048's persisted
            // Phase E capability value.
            "campaign_operations_phase5_transactional",
            outcome.expectedRequestVersion,
            outcome.resultingRequestVersion,
            successful ? "recorded" :
                (outcome.result ==
                    DispatchResultClassification::reconciliationRequired
                    ? "reconciliation_required" : "rejected"),
            successful ? "new_operation" : "reconciliation_required",
            outcome.diagnosticCode});
}

bool IsSuccessfulOutcome(const DispatchAttemptOutcomeEvidence& outcome)
{
    return outcome.result == DispatchResultClassification::createdAndBound ||
        outcome.result ==
            DispatchResultClassification::adoptedExistingPendingAndBound ||
        outcome.result == DispatchResultClassification::existingIdentical;
}

std::string OutcomeAttemptCanonical(pqxx::transaction_base& transaction,
    const DispatchAttemptRecord& attempt, bool production)
{
    if (!production) return attempt.acquisition.identity.canonicalText();
    return transaction.exec(
        "SELECT attempt_identity_canonical "
        "FROM campaign_operations_dispatch_attempt "
        "WHERE dispatch_attempt_id=$1 AND attempt_contract_version=2;",
        pqxx::params{attempt.attemptId.value()}).one_row()[0].as<std::string>();
}

DispatchAttemptRecord ValidateOutcomeAttempt(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    const DispatchAttemptOutcomeEvidence& outcome, bool production)
{
    std::optional<DispatchAttemptRecord> attempt;
    std::string expectedAttemptCanonical;
    if (production)
    {
        const auto persisted = FindProductionDispatchAttemptV2(
            transaction, outcome.attemptId);
        if (persisted)
        {
            expectedAttemptCanonical = persisted->attempt.identity.canonicalText();
            attempt.emplace(DispatchAttemptRecord{
                persisted->attemptId,
                BuildDispatchAttemptAcquisition(
                    persisted->attempt.requestId,
                    persisted->attempt.requestIdentityCanonical,
                    persisted->attempt.attemptOrdinal,
                    persisted->attempt.expectedRequestVersion,
                    persisted->attempt.resultingRequestVersion,
                    persisted->attempt.leaseTokenDigest,
                    persisted->attempt.leaseExpiresAt,
                    persisted->attempt.requestingActor)});
        }
    }
    else
    {
        const auto isolated = FindDispatchAttempt(transaction, outcome.attemptId);
        if (isolated)
        {
            expectedAttemptCanonical =
                isolated->acquisition.identity.canonicalText();
            attempt.emplace(*isolated);
        }
    }
    const auto storedOutcomeAttemptCanonical = transaction.exec(
        "SELECT attempt_identity_canonical "
        "FROM campaign_operations_dispatch_attempt_outcome "
        "WHERE dispatch_attempt_id=$1;",
        pqxx::params{outcome.attemptId.value()});
    if (!attempt || storedOutcomeAttemptCanonical.size() != 1 ||
        attempt->acquisition.requestId != requestId ||
        expectedAttemptCanonical !=
            outcome.attemptCanonicalText ||
        storedOutcomeAttemptCanonical.one_row()[0].as<std::string>() !=
            outcome.attemptCanonicalText)
        throw std::runtime_error(
            "campaign_operations_dispatch_outcome_attempt_corrupt");
    return *attempt;
}

void ValidateOutcomeAudit(pqxx::transaction_base& transaction,
    OperationalRequestId requestId,
    const DispatchAttemptOutcomeEvidence& outcome,
    const DispatchAttemptRecord& attempt, bool production)
{
    (void)production;
    const pqxx::result rows = transaction.exec(
        "SELECT audit.operational_request_id,audit.dispatch_attempt_id,"
        "audit.dispatch_attempt_outcome_id,audit.cause_kind,"
        "audit.actor_identity,audit.capability,audit.prior_version,"
        "audit.resulting_version,audit.outcome,audit.replay_disposition,"
        "audit.diagnostic_code "
        "FROM campaign_operations_dispatch_audit_reference_event audit "
        "JOIN campaign_operations_dispatch_attempt_outcome persisted "
        "ON persisted.dispatch_attempt_outcome_id="
        "audit.dispatch_attempt_outcome_id "
        "WHERE persisted.dispatch_attempt_id=$1;",
        pqxx::params{outcome.attemptId.value()});
    if (rows.size() != 1)
        throw std::runtime_error(
            "campaign_operations_dispatch_outcome_audit_corrupt");
    const auto& audit = rows.one_row();
    const bool successful = IsSuccessfulOutcome(outcome);
    const bool recoveryOutcome = outcome.diagnosticCode ==
        "dispatch_lease_expired_no_downstream_evidence";
    const std::string expectedAuditOutcome = successful || recoveryOutcome
        ? "recorded"
        : outcome.result ==
            DispatchResultClassification::reconciliationRequired
            ? "reconciliation_required"
            : "rejected";
    if (audit[0].as<long long>() != requestId.value() ||
        audit[1].as<long long>() != outcome.attemptId.value() ||
        audit[2].is_null() ||
        audit[3].as<std::string>() !=
            (successful ? "dispatch_handoff_completed"
             : recoveryOutcome ? "dispatch_lease_recovered"
                               : "dispatch_handoff_failed") ||
        audit[4].as<std::string>() !=
            (recoveryOutcome ? "campaign_operations_recovery"
                             : attempt.acquisition.dispatcher.value()) ||
        audit[5].as<std::string>() !=
            (recoveryOutcome ? "campaign_operations_recovery"
                             : kCampaignOperationsPhase5TransactionalRole) ||
        audit[6].as<int>() != outcome.expectedRequestVersion ||
        audit[7].as<int>() != outcome.resultingRequestVersion ||
        audit[8].as<std::string>() != expectedAuditOutcome ||
        audit[9].as<std::string>() !=
            (successful ? "new_operation"
             : recoveryOutcome ? "proven_absent"
                               : "reconciliation_required") ||
        audit[10].as<std::string>() != outcome.diagnosticCode)
        throw std::runtime_error(
            "campaign_operations_dispatch_outcome_audit_corrupt");
}

} // namespace

std::optional<PersistedDispatchBinding> FindAndValidateCompleteDispatchBinding(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    bool production)
{
    const pqxx::result rows = LoadBindingRows(transaction, requestId);
    if (rows.empty()) return std::nullopt;
    std::vector<RequestMemberBinding> members;
    members.reserve(rows.size());
    for (const auto& row : rows)
    {
        RequestMemberBinding binding = MapBinding(row);
        if (binding.identity.canonicalText() != row[22].as<std::string>() ||
            binding.identity.hash() != row[23].as<std::string>())
            throw std::runtime_error(
                "campaign_operations_binding_canonical_corrupt");
        members.push_back(std::move(binding));
    }
    RequestBindingSet bindingSet = BuildRequestBindingSet(
        requestId, members.front().requestCanonicalText, std::move(members));
    const int authoritativeBindingCount = transaction.exec(
        "SELECT count(*) "
        "FROM campaign_operations_request_binding binding "
        "JOIN campaign_operations_operational_request request "
        "ON request.operational_request_id=binding.operational_request_id "
        "JOIN experiment_recommendation_campaign_materialization materialization "
        "ON materialization.recommendation_campaign_materialization_id="
        "binding.recommendation_campaign_materialization_id "
        "JOIN experiment_recommendation_campaign_materialization_member member "
        "ON member.recommendation_campaign_materialization_member_id="
        "binding.recommendation_campaign_materialization_member_id "
        "JOIN experiment_recommendation_conversion_execution execution "
        "ON execution.recommendation_conversion_execution_id="
        "binding.recommendation_conversion_execution_id "
        "JOIN experiment_recommendation_conversion_activation activation "
        "ON activation.recommendation_conversion_activation_id="
        "binding.recommendation_conversion_activation_id "
        "WHERE binding.operational_request_id=$1 "
        "AND request.request_identity_canonical="
        "binding.request_identity_canonical "
        "AND request.recommendation_campaign_materialization_id="
        "binding.recommendation_campaign_materialization_id "
        "AND materialization.materialization_identity_canonical="
        "binding.materialization_identity_canonical "
        "AND member.recommendation_campaign_materialization_id="
        "binding.recommendation_campaign_materialization_id "
        "AND member.member_ordinal=binding.member_ordinal "
        "AND member.selected_member_identity_canonical="
        "binding.selected_member_identity_canonical "
        "AND member.selected_member_identity_hash="
        "binding.selected_member_identity_hash "
        "AND member.recommendation_conversion_proposal_id="
        "binding.recommendation_conversion_proposal_id "
        "AND member.proposal_identity_canonical="
        "binding.proposal_identity_canonical "
        "AND member.proposal_identity_hash=binding.proposal_identity_hash "
        "AND execution.recommendation_conversion_proposal_id="
        "binding.recommendation_conversion_proposal_id "
        "AND execution.recommendation_conversion_review_decision_id="
        "binding.recommendation_conversion_review_decision_id "
        "AND execution.execution_identity_canonical="
        "binding.execution_identity_canonical "
        "AND execution.execution_identity_hash="
        "binding.execution_identity_hash "
        "AND activation.recommendation_conversion_execution_id="
        "binding.recommendation_conversion_execution_id "
        "AND activation.recommendation_conversion_proposal_id="
        "binding.recommendation_conversion_proposal_id "
        "AND activation.recommendation_conversion_review_decision_id="
        "binding.recommendation_conversion_review_decision_id "
        "AND activation.experiment_id=binding.experiment_id "
        "AND activation.activation_identity_canonical="
        "binding.activation_identity_canonical "
        "AND activation.activation_identity_hash="
        "binding.activation_identity_hash "
        "AND execution.experiment_id=binding.experiment_id;",
        pqxx::params{requestId.value()}).one_row()[0].as<int>();
    if (authoritativeBindingCount != bindingSet.memberCount)
        throw std::runtime_error(
            "campaign_operations_binding_provenance_corrupt");
    const pqxx::result ownerRows = transaction.exec(
        "SELECT binding.member_ordinal,binding.binding_identity_canonical,"
        "owner.operational_request_id,owner.experiment_id,"
        "owner.control_mode,owner.adoption_authorization_event_id,"
        "owner.adoption_authorization_identity_canonical,"
        "owner.adoption_authorization_identity_hash,"
        "owner.owner_identity_canonical,owner.owner_identity_hash "
        "FROM campaign_operations_request_binding binding "
        "LEFT JOIN campaign_operations_downstream_control_owner owner "
        "ON owner.request_binding_id=binding.request_binding_id "
        "WHERE binding.operational_request_id=$1 "
        "ORDER BY binding.member_ordinal;",
        pqxx::params{requestId.value()});
    if (static_cast<std::size_t>(ownerRows.size()) !=
        bindingSet.members.size())
        throw std::runtime_error(
            "campaign_operations_control_owner_cardinality_corrupt");
    for (pqxx::result::size_type index = 0;
         index < ownerRows.size(); ++index)
    {
        const auto& row = ownerRows[index];
        const auto memberIndex = static_cast<std::size_t>(index);
        if (row[2].is_null() || row[3].is_null() || row[4].is_null() ||
            row[8].is_null() || row[9].is_null())
            throw std::runtime_error(
                "campaign_operations_control_owner_missing");
        const auto modeText = row[4].as<std::string>();
        const auto mode =
            modeText == "created_control"
            ? DownstreamControlMode::createdControl
            : modeText == "authorized_adoption_control"
                ? DownstreamControlMode::authorizedAdoptionControl
                : throw std::runtime_error(
                    "campaign_operations_control_owner_mode_corrupt");
        const std::optional<AuthorizationEventId> adoptionId =
            row[5].is_null()
            ? std::nullopt
            : std::optional<AuthorizationEventId>(
                AuthorizationEventId(row[5].as<long long>()));
        const std::optional<std::string> adoptionCanonical =
            row[6].is_null()
            ? std::nullopt
            : std::optional<std::string>(row[6].as<std::string>());
        const std::optional<std::string> adoptionHash =
            row[7].is_null()
            ? std::nullopt
            : std::optional<std::string>(row[7].as<std::string>());
        const auto owner = BuildDownstreamControlOwner(
            OperationalRequestId(row[2].as<long long>()),
            bindingSet.requestCanonicalText, row[1].as<std::string>(),
            row[3].as<long long>(), mode, adoptionId,
            adoptionCanonical, adoptionHash);
        if (row[0].as<int>() !=
                bindingSet.members[memberIndex].memberOrdinal ||
            owner.identity.canonicalText() != row[8].as<std::string>() ||
            owner.identity.hash() != row[9].as<std::string>() ||
            owner.requestId != requestId ||
            owner.experimentId !=
                bindingSet.members[memberIndex].experimentId)
            throw std::runtime_error(
                "campaign_operations_control_owner_corrupt");
    }
    const pqxx::row evidence = transaction.exec(
        "SELECT request.request_state,request.materialization_member_count,"
        "reservation.reservation_state,"
        "(SELECT count(*) FROM "
        "campaign_operations_downstream_control_owner owner "
        "WHERE owner.operational_request_id=request.operational_request_id),"
        "(SELECT count(*) FROM "
        "campaign_operations_reservation_commitment commitment "
        "WHERE commitment.operational_request_id="
        "request.operational_request_id),"
        "(SELECT count(*) FROM "
        "campaign_operations_dispatch_attempt_outcome outcome "
        "JOIN campaign_operations_dispatch_attempt attempt "
        "ON attempt.dispatch_attempt_id=outcome.dispatch_attempt_id "
        "WHERE attempt.operational_request_id=request.operational_request_id "
        "AND outcome.request_binding_set_identity_canonical=$2 "
        "AND outcome.request_binding_set_identity_hash=$3),"
        "request.request_identity_canonical,"
        "request.lease_token_hash IS NULL AND "
        "request.lease_expires_at IS NULL AND "
        "request.dispatcher_identity IS NULL,"
        "request.state_version,reservation.state_version "
        "FROM campaign_operations_operational_request request "
        "JOIN campaign_operations_reservation reservation "
        "ON reservation.reservation_id=request.reservation_id "
        "WHERE request.operational_request_id=$1;",
        pqxx::params{requestId.value(), bindingSet.identity.canonicalText(),
            bindingSet.identity.hash()}).one_row();
    if (evidence[0].as<std::string>() != "bound" ||
        evidence[1].as<int>() != bindingSet.memberCount ||
        evidence[2].as<std::string>() != "committed" ||
        evidence[3].as<int>() != bindingSet.memberCount ||
        evidence[4].as<int>() != 1 || evidence[5].as<int>() != 1 ||
        evidence[6].as<std::string>() !=
            bindingSet.requestCanonicalText ||
        !evidence[7].as<bool>())
        throw std::runtime_error(
            "campaign_operations_complete_binding_corrupt");
    const pqxx::row commitmentRow = transaction.exec(
        "SELECT reservation_id,reservation_identity_canonical,"
        "operational_request_id,request_identity_canonical,"
        "expected_reservation_version,resulting_reservation_version,amount,"
        "request_binding_set_identity_canonical,"
        "request_binding_set_identity_hash,commitment_identity_canonical,"
        "commitment_identity_hash "
        "FROM campaign_operations_reservation_commitment "
        "WHERE operational_request_id=$1;",
        pqxx::params{requestId.value()}).one_row();
    const auto commitment = BuildReservationCommitment(
        ReservationId(commitmentRow[0].as<long long>()),
        commitmentRow[1].as<std::string>(),
        OperationalRequestId(commitmentRow[2].as<long long>()),
        commitmentRow[3].as<std::string>(),
        commitmentRow[7].as<std::string>(),
        commitmentRow[8].as<std::string>(),
        commitmentRow[4].as<int>(), commitmentRow[5].as<int>(),
        commitmentRow[6].as<long long>());
    if (commitment.requestId != requestId ||
        commitment.requestCanonicalText !=
            bindingSet.requestCanonicalText ||
        commitment.bindingSetCanonicalText !=
            bindingSet.identity.canonicalText() ||
        commitment.bindingSetIdentityHash != bindingSet.identity.hash() ||
        commitment.identity.canonicalText() !=
            commitmentRow[9].as<std::string>() ||
        commitment.identity.hash() !=
            commitmentRow[10].as<std::string>())
        throw std::runtime_error(
            "campaign_operations_reservation_commitment_corrupt");
    const pqxx::result outcomeRows = transaction.exec(
        "SELECT outcome.dispatch_attempt_id,"
        "attempt.attempt_identity_canonical,outcome.result_classification,"
        "outcome.downstream_evidence_classification,"
        "outcome.semantic_conflict_classification,"
        "outcome.uncertain_commit_recovery_classification,"
        "outcome.diagnostic_code,outcome.expected_request_version,"
        "outcome.resulting_request_version,"
        "outcome.expected_reservation_version,"
        "outcome.resulting_reservation_version,"
        "outcome.outcome_identity_canonical,outcome.outcome_identity_hash "
        "FROM campaign_operations_dispatch_attempt_outcome outcome "
        "JOIN campaign_operations_dispatch_attempt attempt "
        "ON attempt.dispatch_attempt_id=outcome.dispatch_attempt_id "
        "JOIN campaign_operations_reservation_commitment commitment "
        "ON commitment.operational_request_id=attempt.operational_request_id "
        "WHERE attempt.operational_request_id=$1 "
        "AND outcome.request_binding_set_identity_canonical=$2 "
        "AND outcome.request_binding_set_identity_hash=$3;",
        pqxx::params{requestId.value(), bindingSet.identity.canonicalText(),
            bindingSet.identity.hash()});
    if (outcomeRows.size() != 1)
        throw std::runtime_error(
            "campaign_operations_dispatch_outcome_cardinality_corrupt");
    const pqxx::row outcomeRow = outcomeRows.one_row();
    DispatchAttemptOutcomeEvidence outcome =
        BuildDispatchAttemptOutcomeEvidence(
            DispatchAttemptId(outcomeRow[0].as<long long>()),
            outcomeRow[1].as<std::string>(),
            DispatchResultClassificationFromText(
                outcomeRow[2].as<std::string>()),
            DownstreamEvidenceClassificationFromText(
                outcomeRow[3].as<std::string>()),
            SemanticConflictClassificationFromText(
                outcomeRow[4].as<std::string>()),
            UncertainCommitRecoveryClassificationFromText(
                outcomeRow[5].as<std::string>()),
            outcomeRow[6].as<std::string>(), outcomeRow[7].as<int>(),
            outcomeRow[8].as<int>(), outcomeRow[9].as<int>(),
            outcomeRow[10].as<int>(),
            bindingSet.identity.canonicalText(), bindingSet.identity.hash());
    if (outcome.identity.canonicalText() !=
            outcomeRow[11].as<std::string>() ||
        outcome.identity.hash() != outcomeRow[12].as<std::string>())
        throw std::runtime_error(
            "campaign_operations_dispatch_outcome_identity_corrupt");
    if (outcome.result != DispatchResultClassification::createdAndBound &&
        outcome.result != DispatchResultClassification::
            adoptedExistingPendingAndBound)
        throw std::runtime_error(
            "campaign_operations_dispatch_outcome_result_corrupt");
    if (outcome.resultingRequestVersion != evidence[8].as<int>() ||
        outcome.resultingReservationVersion != evidence[9].as<int>())
        throw std::runtime_error(
            "campaign_operations_dispatch_outcome_version_corrupt:" +
            std::to_string(outcome.resultingRequestVersion) + ":" +
            std::to_string(evidence[8].as<int>()) + ":" +
            std::to_string(outcome.resultingReservationVersion) + ":" +
            std::to_string(evidence[9].as<int>()));
    const auto persistedAttempt =
        ValidateOutcomeAttempt(transaction, requestId, outcome, production);
    ValidateOutcomeAudit(
        transaction, requestId, outcome, persistedAttempt, production);
    return PersistedDispatchBinding{
        std::move(bindingSet), std::move(outcome)};
}

std::optional<DispatchAttemptOutcomeEvidence>
FindLatestDispatchOutcome(pqxx::transaction_base& transaction,
    OperationalRequestId requestId, bool production)
{
    const pqxx::result rows = transaction.exec(
        "SELECT outcome.dispatch_attempt_id,"
        "attempt.attempt_identity_canonical,"
        "outcome.result_classification,"
        "outcome.downstream_evidence_classification,"
        "outcome.semantic_conflict_classification,"
        "outcome.uncertain_commit_recovery_classification,"
        "outcome.diagnostic_code,outcome.expected_request_version,"
        "outcome.resulting_request_version,"
        "outcome.expected_reservation_version,"
        "outcome.resulting_reservation_version,"
        "outcome.request_binding_set_identity_canonical,"
        "outcome.request_binding_set_identity_hash,"
        "outcome.outcome_identity_canonical,outcome.outcome_identity_hash "
        "FROM campaign_operations_dispatch_attempt_outcome outcome "
        "JOIN campaign_operations_dispatch_attempt attempt "
        "ON attempt.dispatch_attempt_id=outcome.dispatch_attempt_id "
        "WHERE attempt.operational_request_id=$1 "
        "AND attempt.attempt_ordinal=(SELECT max(latest.attempt_ordinal) "
        "FROM campaign_operations_dispatch_attempt latest "
        "WHERE latest.operational_request_id=$1) "
        "ORDER BY attempt.attempt_ordinal DESC LIMIT 1;",
        pqxx::params{requestId.value()});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
    const auto outcome = BuildDispatchAttemptOutcomeEvidence(
        DispatchAttemptId(row[0].as<long long>()),
        row[1].as<std::string>(),
        DispatchResultClassificationFromText(row[2].as<std::string>()),
        DownstreamEvidenceClassificationFromText(row[3].as<std::string>()),
        SemanticConflictClassificationFromText(row[4].as<std::string>()),
        UncertainCommitRecoveryClassificationFromText(
            row[5].as<std::string>()),
        row[6].as<std::string>(), row[7].as<int>(), row[8].as<int>(),
        row[9].as<int>(), row[10].as<int>(),
        row[11].is_null()
            ? std::nullopt
            : std::optional<std::string>(row[11].as<std::string>()),
        row[12].is_null()
            ? std::nullopt
            : std::optional<std::string>(row[12].as<std::string>()));
    if (outcome.identity.canonicalText() != row[13].as<std::string>() ||
        outcome.identity.hash() != row[14].as<std::string>())
        throw std::runtime_error(
            "campaign_operations_dispatch_outcome_corrupt");
    const auto persistedAttempt =
        ValidateOutcomeAttempt(transaction, requestId, outcome, production);
    ValidateOutcomeAudit(
        transaction, requestId, outcome, persistedAttempt, production);
    return outcome;
}

RequestBindingSet PersistCompleteDispatchBinding(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const ExperimentRecommendation::
        PersistedRecommendationCampaignMaterialization& materialization,
    const ExperimentRecommendation::
        RecommendationCampaignLaunchPersistResult& launchResult,
    BindingDisposition disposition, DispatchTestHook testHook)
{
    (void)launchResult;
    const pqxx::result evidence = transaction.exec(
        "SELECT member."
        "recommendation_campaign_materialization_member_id,"
        "member.member_ordinal,member.selected_member_identity_canonical,"
        "member.selected_member_identity_hash,"
        "member.recommendation_conversion_proposal_id,"
        "member.proposal_identity_canonical,member.proposal_identity_hash,"
        "execution.recommendation_conversion_review_decision_id,"
        "execution.recommendation_conversion_execution_id,"
        "execution.execution_identity_canonical,"
        "execution.execution_identity_hash,"
        "activation.recommendation_conversion_activation_id,"
        "activation.activation_identity_canonical,"
        "activation.activation_identity_hash,execution.experiment_id "
        "FROM experiment_recommendation_campaign_materialization_member member "
        "JOIN experiment_recommendation_conversion_execution execution "
        "ON execution.recommendation_conversion_proposal_id="
        "member.recommendation_conversion_proposal_id "
        "JOIN experiment_recommendation_conversion_activation activation "
        "ON activation.recommendation_conversion_execution_id="
        "execution.recommendation_conversion_execution_id "
        "JOIN experiment downstream "
        "ON downstream.experiment_id=execution.experiment_id "
        "WHERE member.recommendation_campaign_materialization_id=$1 "
        "AND downstream.status='pending' AND downstream.phase='train' "
        "ORDER BY member.member_ordinal;",
        pqxx::params{authority.materializationId});
    if (static_cast<int>(evidence.size()) != authority.memberCount ||
        materialization.selectedMemberCount != authority.memberCount)
        throw std::runtime_error(
            "campaign_operations_phase5_complete_evidence_required");
    std::vector<RequestMemberBinding> members;
    members.reserve(evidence.size());
    const bool adopted =
        disposition == BindingDisposition::adoptedExistingPending;
    for (const auto& row : evidence)
    {
        members.push_back(BuildRequestMemberBinding(
            authority.requestId, authority.requestCanonicalText,
            authority.materializationId, materialization.identityCanonical,
            row[0].as<long long>(), row[1].as<int>(),
            row[2].as<std::string>(), row[3].as<std::string>(),
            row[4].as<long long>(), row[5].as<std::string>(),
            row[6].as<std::string>(), row[7].as<long long>(),
            row[8].as<long long>(), row[9].as<std::string>(),
            row[10].as<std::string>(), row[11].as<long long>(),
            row[12].as<std::string>(), row[13].as<std::string>(),
            row[14].as<long long>(), disposition,
            adopted ? Phase5ExecutionDisposition::reused
                    : Phase5ExecutionDisposition::created,
            adopted ? Phase5ActivationDisposition::reused
                    : Phase5ActivationDisposition::created));
    }
    RequestBindingSet set = BuildRequestBindingSet(authority.requestId,
        authority.requestCanonicalText, std::move(members));
    std::size_t inserted = 0;
    for (const auto& member : set.members)
    {
        if (testHook)
            testHook(DispatchTestInjectionPoint::duringBindingInsertion);
        transaction.exec(
            "INSERT INTO campaign_operations_request_binding("
            "operational_request_id,request_identity_canonical,"
            "recommendation_campaign_materialization_id,"
            "materialization_identity_canonical,"
            "recommendation_campaign_materialization_member_id,"
            "member_ordinal,selected_member_identity_canonical,"
            "selected_member_identity_hash,"
            "recommendation_conversion_proposal_id,"
            "proposal_identity_canonical,proposal_identity_hash,"
            "recommendation_conversion_review_decision_id,"
            "recommendation_conversion_execution_id,"
            "execution_identity_canonical,execution_identity_hash,"
            "recommendation_conversion_activation_id,"
            "activation_identity_canonical,activation_identity_hash,"
            "experiment_id,binding_disposition,execution_disposition,"
            "activation_disposition,binding_contract_version,"
            "binding_identity_canonical,binding_identity_hash) "
            "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,"
            "$15,$16,$17,$18,$19,$20,$21,$22,1,$23,$24);",
            pqxx::params{member.requestId.value(),
                member.requestCanonicalText, member.materializationId,
                member.materializationCanonicalText,
                member.materializationMemberId, member.memberOrdinal,
                member.selectedMemberCanonicalText,
                member.selectedMemberIdentityHash, member.proposalId,
                member.proposalCanonicalText, member.proposalIdentityHash,
                member.reviewDecisionId, member.executionId,
                member.executionCanonicalText,
                member.executionIdentityHash, member.activationId,
                member.activationCanonicalText,
                member.activationIdentityHash, member.experimentId,
                ToText(member.bindingDisposition),
                ToText(member.executionDisposition),
                ToText(member.activationDisposition),
                member.identity.canonicalText(), member.identity.hash()});
        ++inserted;
        if (testHook && inserted < set.members.size())
            testHook(DispatchTestInjectionPoint::afterBindingSubset);
    }
    return set;
}

void PersistControlOwners(pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const RequestBindingSet& bindingSet, DispatchTestHook testHook)
{
    const bool adopted = bindingSet.members.front().bindingDisposition ==
        BindingDisposition::adoptedExistingPending;
    std::optional<std::string> adoptionCanonical;
    std::optional<std::string> adoptionHash;
    if (adopted)
    {
        if (!authority.adoptionAuthorizationId)
            throw std::runtime_error(
                "campaign_operations_adoption_authorization_required");
        const auto row = transaction.exec(
            "SELECT authorization_identity_canonical,"
            "authorization_identity_hash FROM "
            "campaign_operations_authorization_event "
            "WHERE authorization_event_id=$1 AND event_kind='granted' "
            "AND action_kind='adopt_existing_pending_and_control' "
            "AND not_before<=transaction_timestamp() "
            "AND (expires_at IS NULL OR "
            "expires_at>transaction_timestamp());",
            pqxx::params{authority.adoptionAuthorizationId->value()});
        if (row.empty())
            throw std::runtime_error(
                "campaign_operations_adoption_authorization_inactive");
        adoptionCanonical = row.one_row()[0].as<std::string>();
        adoptionHash = row.one_row()[1].as<std::string>();
    }
    std::size_t inserted = 0;
    for (const auto& binding : bindingSet.members)
    {
        const DownstreamControlOwner owner = BuildDownstreamControlOwner(
            authority.requestId, authority.requestCanonicalText,
            binding.identity.canonicalText(), binding.experimentId,
            adopted ? DownstreamControlMode::authorizedAdoptionControl
                    : DownstreamControlMode::createdControl,
            adopted ? authority.adoptionAuthorizationId
                    : std::optional<AuthorizationEventId>{},
            adopted ? adoptionCanonical : std::optional<std::string>{},
            adopted ? adoptionHash : std::optional<std::string>{});
        const long long bindingId = transaction.exec(
            "SELECT request_binding_id FROM "
            "campaign_operations_request_binding "
            "WHERE operational_request_id=$1 AND member_ordinal=$2;",
            pqxx::params{authority.requestId.value(),
                binding.memberOrdinal}).one_row()[0].as<long long>();
        if (testHook)
            testHook(
                DispatchTestInjectionPoint::duringControlOwnerInsertion);
        transaction.exec(
            "INSERT INTO campaign_operations_downstream_control_owner("
            "request_binding_id,operational_request_id,"
            "binding_identity_canonical,experiment_id,control_mode,"
            "adoption_authorization_event_id,"
            "adoption_authorization_identity_canonical,"
            "adoption_authorization_identity_hash,owner_contract_version,"
            "owner_identity_canonical,owner_identity_hash) "
            "VALUES($1,$2,$3,$4,$5,$6,$7,$8,1,$9,$10);",
            pqxx::params{bindingId, authority.requestId.value(),
                binding.identity.canonicalText(), binding.experimentId,
                ToText(owner.mode),
                owner.adoptionAuthorizationEventId
                    ? std::optional<long long>(
                          owner.adoptionAuthorizationEventId->value())
                    : std::nullopt,
                owner.adoptionAuthorizationCanonicalText,
                owner.adoptionAuthorizationIdentityHash,
                owner.identity.canonicalText(), owner.identity.hash()});
        ++inserted;
        if (testHook && inserted < bindingSet.members.size())
            testHook(DispatchTestInjectionPoint::afterControlOwnerSubset);
    }
}

ReservationCommitment CommitReservationForBinding(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const RequestBindingSet& bindingSet, DispatchTestHook testHook)
{
    const ReservationCommitment commitment = BuildReservationCommitment(
        authority.reservationId, authority.reservationCanonicalText,
        authority.requestId, authority.requestCanonicalText,
        bindingSet.identity.canonicalText(), bindingSet.identity.hash(),
        authority.reservationVersion, authority.reservationVersion + 1,
        authority.amount);
    if (testHook)
        testHook(DispatchTestInjectionPoint::
            beforeReservationCommitmentEventInsertion);
    transaction.exec(
        "INSERT INTO campaign_operations_reservation_commitment("
        "reservation_id,reservation_identity_canonical,"
        "operational_request_id,request_identity_canonical,"
        "expected_reservation_version,resulting_reservation_version,amount,"
        "request_binding_set_identity_canonical,"
        "request_binding_set_identity_hash,commitment_contract_version,"
        "commitment_identity_canonical,commitment_identity_hash) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,1,$10,$11);",
        pqxx::params{commitment.reservationId.value(),
            commitment.reservationCanonicalText,
            commitment.requestId.value(), commitment.requestCanonicalText,
            commitment.expectedReservationVersion,
            commitment.resultingReservationVersion, commitment.amount,
            commitment.bindingSetCanonicalText,
            commitment.bindingSetIdentityHash,
            commitment.identity.canonicalText(),
            commitment.identity.hash()});
    if (testHook)
        testHook(DispatchTestInjectionPoint::
            afterReservationCommitmentEventBeforeProjection);
    const pqxx::row changed = transaction.exec(
        "SELECT state_version FROM "
        "transition_campaign_operations_reservation_committed($1,$2);",
        pqxx::params{authority.reservationId.value(),
            authority.reservationVersion}).one_row();
    if (changed[0].as<int>() != commitment.resultingReservationVersion)
        throw std::runtime_error(
            "campaign_operations_reservation_commitment_version_mismatch");
    if (testHook)
        testHook(DispatchTestInjectionPoint::
            afterReservationProjectionBeforeRequestTransition);
    return commitment;
}

DispatchAttemptOutcomeEvidence BindRequestAndPersistSuccessfulOutcome(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const DispatchAttemptRecord& attempt,
    const RequestBindingSet& bindingSet,
    DownstreamEvidenceClassification downstreamEvidence,
    BindingDisposition disposition, DispatchTestHook testHook, bool production,
    const std::string& productionOperationKey,
    const std::string& approvedBuildContractCanonical)
{
    const pqxx::row changed = production
        ? transaction.exec(
              "SELECT state_version FROM "
              "transition_campaign_operations_request_bound_production_v2("
              "$1,$2,$3,$4,$5,$6);",
              pqxx::params{authority.requestId.value(),
                  authority.requestVersion, authority.leaseTokenDigest,
                  attempt.attemptId.value(), productionOperationKey,
                  approvedBuildContractCanonical}).one_row()
        : transaction.exec(
              "SELECT state_version FROM "
              "transition_campaign_operations_request_bound($1,$2,$3);",
              pqxx::params{authority.requestId.value(),
                  authority.requestVersion, authority.leaseTokenDigest})
              .one_row();
    if (testHook)
        testHook(DispatchTestInjectionPoint::
            afterRequestTransitionBeforeAttemptOutcome);
    const auto outcome = BuildDispatchAttemptOutcomeEvidence(
        attempt.attemptId, OutcomeAttemptCanonical(transaction, attempt,
            production),
        disposition == BindingDisposition::created
            ? DispatchResultClassification::createdAndBound
            : DispatchResultClassification::
                adoptedExistingPendingAndBound,
        downstreamEvidence, SemanticConflictClassification::none,
        UncertainCommitRecoveryClassification::provenNoCommit,
        disposition == BindingDisposition::created
            ? "dispatch_created_and_bound"
            : "dispatch_adopted_and_bound",
        authority.requestVersion, changed[0].as<int>(),
        authority.reservationVersion, authority.reservationVersion + 1,
        bindingSet.identity.canonicalText(), bindingSet.identity.hash());
    const long long outcomeId = InsertOutcome(transaction, outcome);
    if (testHook)
        testHook(
            DispatchTestInjectionPoint::afterAttemptOutcomeBeforeAudit);
    InsertOutcomeAudit(
        transaction, authority, attempt, outcomeId, outcome, production);
    return outcome;
}

DispatchAttemptOutcomeEvidence PersistFailedDispatchOutcome(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority,
    const DispatchAttemptRecord& attempt,
    DownstreamEvidenceClassification downstreamEvidence,
    SemanticConflictClassification conflict,
    const std::string& diagnosticCode, bool production)
{
    const auto outcome = BuildDispatchAttemptOutcomeEvidence(
        attempt.attemptId, OutcomeAttemptCanonical(transaction, attempt,
            production),
        DispatchResultClassification::reconciliationRequired,
        downstreamEvidence, conflict,
        UncertainCommitRecoveryClassification::ambiguousEvidence,
        diagnosticCode, authority.requestVersion, authority.requestVersion,
        authority.reservationVersion, authority.reservationVersion,
        std::nullopt, std::nullopt);
    const long long outcomeId = InsertOutcome(transaction, outcome);
    InsertOutcomeAudit(
        transaction, authority, attempt, outcomeId, outcome, production);
    return outcome;
}

} // namespace EA::CampaignOperations
