#include "CampaignOperationsProductionAdmissionRepository.hpp"

#include <charconv>
#include <set>
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

[[noreturn]] void ReadinessEvidenceCorrupt(const char* diagnostic)
{
    throw Error(ErrorCode::persistenceCorruption, diagnostic);
}

void RequireFields(const pqxx::row& row, const char* diagnostic)
{
    for (const auto& field : row)
        if (field.is_null()) ReadinessEvidenceCorrupt(diagnostic);
}

int CanonicalVersionSuffix(const std::string& canonical,
    const char* suffix, const char* diagnostic)
{
    const std::string marker = std::string{";"} + suffix + "=";
    const auto markerPosition = canonical.rfind(marker);
    if (markerPosition == std::string::npos ||
        markerPosition + marker.size() >= canonical.size())
        ReadinessEvidenceCorrupt(diagnostic);
    const std::string value = canonical.substr(markerPosition + marker.size());
    int result = 0;
    const auto conversion = std::from_chars(value.data(),
        value.data() + value.size(), result);
    if (conversion.ec != std::errc{} || conversion.ptr !=
            value.data() + value.size() || result <= 0)
        ReadinessEvidenceCorrupt(diagnostic);
    return result;
}

CanonicalIdentity HydrateStructuralIdentity(const std::string& storedCanonical,
    const std::string& storedHash, const std::string& expectedCanonical,
    const char* diagnostic)
{
    if (storedCanonical != expectedCanonical)
        ReadinessEvidenceCorrupt(diagnostic);
    try
    {
        return CanonicalIdentity::Hydrate(1, storedCanonical, storedHash);
    }
    catch (const Error&)
    {
        ReadinessEvidenceCorrupt(diagnostic);
    }
}

std::string JoinVersions(const std::set<int>& values)
{
    if (values.empty()) return {};
    std::ostringstream output;
    for (const int value : values)
    {
        if (output.tellp() != std::streampos{0}) output << '|';
        output << value;
    }
    return output.str();
}

std::optional<ProductionEnablementEventId> OptionalEventId(
    const pqxx::field& field)
{
    if (field.is_null()) return std::nullopt;
    return ProductionEnablementEventId(field.as<long long>());
}

ManagerBuildContract BuildContractFromRow(const pqxx::row& row,
    int service, int commit, int compiler,
    int executable, int canonical, int hash)
{
    auto result = BuildManagerBuildContract(row[service].as<std::string>(),
        row[commit].as<std::string>(), row[compiler].as<std::string>(),
        row[executable].as<std::string>());
    if (result.identity.canonicalText() != row[canonical].as<std::string>() ||
        result.identity.hash() != row[hash].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_manager_build_corrupt");
    return result;
}

SchedulerProtocolEvidence SchedulerFromRow(const pqxx::row& row,
    int generation, int state, int completedAt,
    int completedBy, int path, int process,
    int canonical, int hash)
{
    auto result = BuildSchedulerProtocolEvidence(row[generation].as<int>(),
        row[state].as<std::string>(),
        UtcTimestamp(row[completedAt].as<std::string>()),
        row[completedBy].as<std::string>(), row[path].as<std::string>(),
        row[process].as<std::string>());
    if (result.identity.canonicalText() != row[canonical].as<std::string>() ||
        result.identity.hash() != row[hash].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_scheduler_evidence_corrupt");
    return result;
}

ProductionEnablementAuditEvidence LoadEnablementAudit(
    pqxx::transaction_base& transaction, const pqxx::row& eventRow)
{
    const auto rows = transaction.exec(
        "SELECT production_enablement_audit_reference_event_id,"
        "production_enablement_event_id,operation_key,event_kind,"
        "actor_identity,capability,reason,outcome,replay_disposition,"
        "diagnostic_code FROM "
        "campaign_operations_production_enablement_audit_reference_event "
        "WHERE production_enablement_event_id=$1;",
        pqxx::params{eventRow[0].as<long long>()});
    if (rows.size() != 1U)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_enablement_audit_missing");
    const auto& row = rows.one_row();
    const auto kind = ProductionEnablementEventKindFromText(
        row[3].as<std::string>());
    if (row[1].as<long long>() != eventRow[0].as<long long>() ||
        row[2].as<std::string>() != eventRow[2].as<std::string>() ||
        kind != ProductionEnablementEventKindFromText(
            eventRow[1].as<std::string>()) ||
        row[4].as<std::string>() != eventRow[15].as<std::string>() ||
        row[5].as<std::string>() != eventRow[26].as<std::string>() ||
        row[6].as<std::string>() != eventRow[25].as<std::string>() ||
        row[7].as<std::string>() != "recorded" ||
        row[8].as<std::string>() != "new_operation" ||
        row[9].as<std::string>() != "immutable_enablement_event_recorded")
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_enablement_audit_corrupt");
    return {row[0].as<long long>(),
        ProductionEnablementEventId(row[1].as<long long>()),
        row[2].as<std::string>(), kind,
        ActorIdentity(row[4].as<std::string>()), row[5].as<std::string>(),
        Reason(row[6].as<std::string>()), row[7].as<std::string>(),
        row[8].as<std::string>(), row[9].as<std::string>()};
}

ProductionDispatchAuditEvidence LoadAcquisitionAudit(
    pqxx::transaction_base& transaction, DispatchAttemptId attemptId,
    OperationalRequestId requestId, RequestProductionAdmissionId admissionId,
    ProductionEnablementEventId enablementEventId, const ActorIdentity& actor,
    int priorVersion, int resultingVersion)
{
    const auto rows = transaction.exec(
        "SELECT audit.dispatch_audit_reference_event_id,"
        "audit.operational_campaign_id,audit.operational_request_id,"
        "audit.dispatch_attempt_id,audit.cause_kind,audit.actor_identity,"
        "audit.capability,audit.prior_version,audit.resulting_version,"
        "audit.outcome,audit.replay_disposition,audit.diagnostic_code,"
        "audit.request_production_admission_id,"
        "audit.production_enablement_event_id,"
        "audit.dispatch_attempt_outcome_id,request.operational_campaign_id "
        "FROM campaign_operations_dispatch_audit_reference_event audit JOIN "
        "campaign_operations_operational_request request ON "
        "request.operational_request_id=audit.operational_request_id "
        "WHERE audit.dispatch_attempt_id=$1 AND "
        "audit.cause_kind='dispatch_lease_acquired';",
        pqxx::params{attemptId.value()});
    if (rows.size() != 1U)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_acquisition_audit_missing");
    const auto& row = rows.one_row();
    for (pqxx::row::size_type index = 0; index < row.size(); ++index)
        if (index != 14U && row[index].is_null())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_acquisition_audit_corrupt");
    if (row[1].as<long long>() != row[15].as<long long>() ||
        row[2].as<long long>() != requestId.value() ||
        row[3].as<long long>() != attemptId.value() ||
        row[4].as<std::string>() != "dispatch_lease_acquired" ||
        row[5].as<std::string>() != actor.value() ||
        row[6].as<std::string>() != kProductionDispatcherRole ||
        row[7].as<int>() != priorVersion ||
        row[8].as<int>() != resultingVersion ||
        row[9].as<std::string>() != "recorded" ||
        row[10].as<std::string>() != "new_operation" ||
        row[11].as<std::string>() != "dispatch_lease_acquired" ||
        row[12].is_null() || row[12].as<long long>() != admissionId.value() ||
        row[13].is_null() ||
        row[13].as<long long>() != enablementEventId.value() ||
        !row[14].is_null())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_acquisition_audit_corrupt");
    return {row[0].as<long long>(),
        OperationalCampaignId(row[1].as<long long>()), requestId, attemptId,
        row[4].as<std::string>(), ActorIdentity(row[5].as<std::string>()),
        row[6].as<std::string>(), row[7].as<int>(), row[8].as<int>(),
        row[9].as<std::string>(), row[10].as<std::string>(),
        row[11].as<std::string>(), admissionId, enablementEventId};
}

PersistedProductionEnablementHead MapHead(const pqxx::row& row,
    ProductionEnablementAuditEvidence auditEvidence)
{
    const auto eventId = ProductionEnablementEventId(row[0].as<long long>());
    const auto kind = ProductionEnablementEventKindFromText(
        row[1].as<std::string>());
    const auto predecessorId = OptionalEventId(row[3]);
    const std::string storedCanonical = row[23].as<std::string>();
    const std::string storedHash = row[24].as<std::string>();
    const std::string storedCapability = row[26].as<std::string>();
    const int storedContractVersion = row[27].as<int>();
    const char* const expectedCapability =
        kind == ProductionEnablementEventKind::enable
        ? kProductionEnablerRole : kProductionDisablerRole;
    if (storedCapability != expectedCapability || storedContractVersion !=
            kProductionEnablementContractVersion)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_enablement_typed_mirror_corrupt");
    if (kind == ProductionEnablementEventKind::enable)
    {
        auto scheduler = SchedulerFromRow(
            row, 8, 9, 10, 11, 12, 13, 6, 7);
        auto build = BuildContractFromRow(row, 17, 20, 21, 22, 18, 19);
        auto event = BuildProductionEnableEvent(row[2].as<std::string>(),
            predecessorId, row[4].as<std::string>(), row[5].as<int>(),
            row[16].as<int>(), scheduler, row[14].as<std::string>(),
            ActorIdentity(row[15].as<std::string>()),
            row[17].as<std::string>(), build,
            Reason(row[25].as<std::string>()));
        if (event.identity.canonicalText() != storedCanonical ||
            event.identity.hash() != storedHash)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_enablement_head_corrupt");
        return {eventId, kind, event.identity, event.resultingVersion,
            std::move(scheduler), std::move(build),
            std::move(auditEvidence)};
    }
    if (!predecessorId)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_disable_predecessor_missing");
    auto event = BuildProductionDisableEvent(row[2].as<std::string>(),
        *predecessorId, row[4].as<std::string>(), row[5].as<int>(),
        row[16].as<int>(), ActorIdentity(row[15].as<std::string>()),
        Reason(row[25].as<std::string>()));
    if (event.identity.canonicalText() != storedCanonical ||
        event.identity.hash() != storedHash)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_enablement_head_corrupt");
    return {eventId, kind, event.identity, event.resultingVersion,
        std::nullopt, std::nullopt, std::move(auditEvidence)};
}

std::string HeadColumns()
{
    return "production_enablement_event_id,event_kind,operation_key,"
        "predecessor_event_id,predecessor_event_canonical,"
        "expected_prior_version,scheduler_protocol_evidence_canonical,"
        "scheduler_protocol_evidence_hash,scheduler_required_generation,"
        "scheduler_cutover_state,to_char(scheduler_cutover_completed_at AT "
        "TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"'),"
        "scheduler_cutover_completed_by,scheduler_cutover_executable_path,"
        "scheduler_cutover_process_evidence,"
        "independent_verification_reference,actor_identity,"
        "resulting_version,manager_service_contract,"
        "approved_build_contract_canonical,approved_build_contract_hash,"
        "approved_build_source_commit,approved_build_compiler_contract,"
        "approved_build_executable_sha256,enablement_identity_canonical,"
        "enablement_identity_hash,reason,capability,"
        "enablement_contract_version";
}

ObservedProductionEnablementEvidence LoadObservedEnablementEventById(
    pqxx::transaction_base& transaction, ProductionEnablementEventId eventId,
    int remainingDepth = 1024)
{
    if (remainingDepth <= 0)
        ReadinessEvidenceCorrupt("campaign_operations_enablement_chain_cycle");
    const auto rows = transaction.exec(
        "SELECT " + HeadColumns() + " FROM "
        "campaign_operations_production_enablement_event "
        "WHERE production_enablement_event_id=$1;",
        pqxx::params{eventId.value()});
    if (rows.size() != 1U)
        ReadinessEvidenceCorrupt("campaign_operations_enablement_missing");
    const auto& row = rows.one_row();
    const auto kind = ProductionEnablementEventKindFromText(
        row[1].as<std::string>());
    const auto predecessorId = OptionalEventId(row[3]);
    const int enablementVersion = row[27].as<int>();
    const std::string expectedCapability = kind ==
        ProductionEnablementEventKind::enable ? kProductionEnablerRole :
        kProductionDisablerRole;
    if (row[26].as<std::string>() != expectedCapability)
        ReadinessEvidenceCorrupt(
            "campaign_operations_enablement_typed_mirror_corrupt");
    const auto audit = LoadEnablementAudit(transaction, row);

    if (kind == ProductionEnablementEventKind::enable)
    {
        for (pqxx::row::size_type index = 0; index < row.size(); ++index)
            if (index != 3U && row[index].is_null())
                ReadinessEvidenceCorrupt(
                    "campaign_operations_enablement_typed_mirror_corrupt");
        const int managerBuildVersion = CanonicalVersionSuffix(
            row[18].as<std::string>(), "build_contract_version",
            "campaign_operations_manager_build_corrupt");
        std::ostringstream buildCanonical;
        buildCanonical << "campaign_operations_manager_build_v1"
            << ";manager_service_contract=" << Framed(row[17].as<std::string>())
            << ";source_commit=" << row[20].as<std::string>()
            << ";source_tree_state=clean;build_configuration=Release"
            << ";compiler_contract=" << Framed(row[21].as<std::string>())
            << ";executable_sha256=" << row[22].as<std::string>()
            << ";build_contract_version=" << managerBuildVersion;
        const auto build = HydrateStructuralIdentity(row[18].as<std::string>(),
            row[19].as<std::string>(), buildCanonical.str(),
            "campaign_operations_manager_build_corrupt");
        auto scheduler = SchedulerFromRow(row, 8, 9, 10, 11, 12, 13, 6, 7);
        std::ostringstream canonical;
        canonical << "campaign_operations_production_enable_event_v1"
            << ";operation_key=" << Framed(row[2].as<std::string>())
            << ";predecessor_event_id="
            << (predecessorId ? std::to_string(predecessorId->value()) : "none")
            << ";predecessor_event_canonical=" << Framed(row[4].as<std::string>())
            << ";expected_prior_version=" << row[5].as<int>()
            << ";resulting_version=" << row[16].as<int>()
            << ";scheduler_protocol_evidence="
            << Framed(scheduler.identity.canonicalText())
            << ";independent_verification_reference="
            << Framed(row[14].as<std::string>())
            << ";authorizing_actor=" << Framed(row[15].as<std::string>())
            << ";capability=" << kProductionEnablerRole
            << ";manager_service_contract=" << Framed(row[17].as<std::string>())
            << ";approved_build_contract=" << Framed(build.canonicalText())
            << ";reason=" << Framed(row[25].as<std::string>())
            << ";enablement_contract_version=" << enablementVersion;
        const auto identity = HydrateStructuralIdentity(row[23].as<std::string>(),
            row[24].as<std::string>(), canonical.str(),
            "campaign_operations_enablement_head_corrupt");
        if (predecessorId)
        {
            const auto predecessor = LoadObservedEnablementEventById(
                transaction, *predecessorId, remainingDepth - 1);
            if (predecessor.identity.canonicalText() != row[4].as<std::string>() ||
                predecessor.resultingVersion != row[5].as<int>() ||
                predecessor.kind == kind)
                ReadinessEvidenceCorrupt(
                    "campaign_operations_enablement_predecessor_corrupt");
        }
        else if (!row[4].as<std::string>().empty() || row[5].as<int>() != 0)
            ReadinessEvidenceCorrupt("campaign_operations_enablement_genesis_corrupt");
        (void)audit;
        return {eventId, kind, identity, row[16].as<int>(), enablementVersion,
            row[17].as<std::string>(), build, managerBuildVersion};
    }

    for (const pqxx::row::size_type index : {0U, 1U, 2U, 4U, 5U, 15U,
             16U, 23U, 24U, 25U, 26U, 27U})
        if (row[index].is_null())
            ReadinessEvidenceCorrupt(
                "campaign_operations_enablement_typed_mirror_corrupt");
    for (const pqxx::row::size_type index : {6U, 7U, 8U, 9U, 10U, 11U,
             12U, 13U, 14U, 17U, 18U, 19U, 20U, 21U, 22U})
        if (!row[index].is_null())
            ReadinessEvidenceCorrupt(
                "campaign_operations_enablement_typed_mirror_corrupt");
    if (!predecessorId)
        ReadinessEvidenceCorrupt("campaign_operations_disable_predecessor_missing");
    std::ostringstream canonical;
    canonical << "campaign_operations_production_disable_event_v1"
        << ";operation_key=" << Framed(row[2].as<std::string>())
        << ";predecessor_event_id=" << predecessorId->value()
        << ";predecessor_event_canonical=" << Framed(row[4].as<std::string>())
        << ";expected_prior_version=" << row[5].as<int>()
        << ";resulting_version=" << row[16].as<int>()
        << ";disabling_actor=" << Framed(row[15].as<std::string>())
        << ";capability=" << kProductionDisablerRole
        << ";reason=" << Framed(row[25].as<std::string>())
        << ";enablement_contract_version=" << enablementVersion;
    const auto identity = HydrateStructuralIdentity(row[23].as<std::string>(),
        row[24].as<std::string>(), canonical.str(),
        "campaign_operations_enablement_head_corrupt");
    const auto predecessor = LoadObservedEnablementEventById(transaction,
        *predecessorId, remainingDepth - 1);
    if (predecessor.identity.canonicalText() != row[4].as<std::string>() ||
        predecessor.resultingVersion != row[5].as<int>() ||
        predecessor.kind == kind)
        ReadinessEvidenceCorrupt(
            "campaign_operations_enablement_predecessor_corrupt");
    (void)audit;
    return {eventId, kind, identity, row[16].as<int>(), enablementVersion,
        std::nullopt, std::nullopt, std::nullopt};
}

struct ObservedAdmissionEvidence final
{
    RequestProductionAdmissionId admissionId;
    OperationalRequestId requestId;
    CanonicalIdentity identity;
    int contractVersion;
    ProductionEnablementEventId enablementEventId;
    CanonicalIdentity enablementIdentity;
    CanonicalIdentity approvedBuildIdentity;
    std::string requestIdentityCanonical;
    std::string requestingActor;
    int expectedRequestVersion;
};

ObservedAdmissionEvidence LoadObservedAdmissionById(
    pqxx::transaction_base& transaction, RequestProductionAdmissionId admissionId)
{
    const auto rows = transaction.exec(
        "SELECT admission.request_production_admission_id,"
        "admission.operational_request_id,admission.request_identity_canonical,"
        "admission.expected_request_version,admission.dispatch_operation_key,"
        "admission.production_enablement_event_id,admission.enable_event_canonical,"
        "admission.requesting_actor,admission.original_executing_service_principal,"
        "admission.approved_build_contract_canonical,"
        "admission.approved_build_contract_hash,admission.capability,"
        "admission.admission_contract_version,admission.admission_identity_canonical,"
        "admission.admission_identity_hash,request.request_identity_canonical "
        "FROM campaign_operations_request_production_admission admission "
        "JOIN campaign_operations_operational_request request ON "
        "request.operational_request_id=admission.operational_request_id "
        "WHERE admission.request_production_admission_id=$1;",
        pqxx::params{admissionId.value()});
    if (rows.size() != 1U)
        ReadinessEvidenceCorrupt("campaign_operations_admission_missing");
    const auto& row = rows.one_row();
    RequireFields(row, "campaign_operations_admission_typed_mirror_corrupt");
    const auto enablement = LoadObservedEnablementEventById(transaction,
        ProductionEnablementEventId(row[5].as<long long>()));
    if (enablement.kind != ProductionEnablementEventKind::enable ||
        !enablement.approvedBuildIdentity ||
        enablement.identity.canonicalText() != row[6].as<std::string>() ||
        enablement.approvedBuildIdentity->canonicalText() !=
            row[9].as<std::string>() ||
        enablement.approvedBuildIdentity->hash() != row[10].as<std::string>() ||
        row[11].as<std::string>() != kProductionDispatcherRole ||
        row[1].as<long long>() <= 0 || row[3].as<int>() <= 0 ||
        row[2].as<std::string>() != row[15].as<std::string>())
        ReadinessEvidenceCorrupt("campaign_operations_admission_typed_mirror_corrupt");
    std::ostringstream canonical;
    canonical << "campaign_operations_request_production_admission_v1"
        << ";operational_request_id=" << row[1].as<long long>()
        << ";request_identity_canonical=" << Framed(row[2].as<std::string>())
        << ";expected_request_version=" << row[3].as<int>()
        << ";dispatch_operation_key=" << Framed(row[4].as<std::string>())
        << ";enable_event_id=" << row[5].as<long long>()
        << ";enable_event_canonical=" << Framed(row[6].as<std::string>())
        << ";requesting_actor=" << Framed(row[7].as<std::string>())
        << ";original_executing_service_principal=" << Framed(row[8].as<std::string>())
        << ";approved_build_contract=" << Framed(row[9].as<std::string>())
        << ";capability=" << kProductionDispatcherRole
        << ";admission_contract_version=" << row[12].as<int>();
    const auto identity = HydrateStructuralIdentity(row[13].as<std::string>(),
        row[14].as<std::string>(), canonical.str(),
        "campaign_operations_admission_corrupt");
    return {admissionId, OperationalRequestId(row[1].as<long long>()), identity,
        row[12].as<int>(), ProductionEnablementEventId(row[5].as<long long>()),
        enablement.identity, *enablement.approvedBuildIdentity,
        row[2].as<std::string>(), row[7].as<std::string>(), row[3].as<int>()};
}

struct ObservedAttemptEvidence final
{
    int contractVersion;
};

ObservedAttemptEvidence LoadObservedAttemptById(
    pqxx::transaction_base& transaction, DispatchAttemptId attemptId)
{
    const auto rows = transaction.exec(
        "SELECT attempt.operational_request_id,attempt.request_identity_canonical,"
        "attempt.request_production_admission_id,"
        "attempt.request_production_admission_canonical,"
        "attempt.request_production_admission_hash,"
        "attempt.production_enablement_event_id,"
        "attempt.production_enablement_event_canonical,"
        "attempt.production_enablement_event_hash,attempt.operation_key,"
        "attempt.attempt_ordinal,attempt.expected_request_version,"
        "attempt.resulting_request_version,attempt.lease_token_digest,"
        "to_char(attempt.lease_expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"'),attempt.dispatcher_identity,"
        "attempt.requesting_actor,attempt.original_executing_service_principal,"
        "attempt.approved_build_contract_canonical,"
        "attempt.approved_build_contract_hash,attempt.production_capability,"
        "attempt.attempt_contract_version,attempt.attempt_identity_canonical,"
        "attempt.attempt_identity_hash,request.request_identity_canonical "
        "FROM campaign_operations_dispatch_attempt attempt JOIN "
        "campaign_operations_operational_request request ON "
        "request.operational_request_id=attempt.operational_request_id "
        "WHERE attempt.dispatch_attempt_id=$1;", pqxx::params{attemptId.value()});
    if (rows.size() != 1U)
        ReadinessEvidenceCorrupt("campaign_operations_attempt_v2_missing");
    const auto& row = rows.one_row();
    RequireFields(row, "campaign_operations_attempt_v2_typed_mirror_corrupt");
    if (row[20].as<int>() <= 0)
        ReadinessEvidenceCorrupt("campaign_operations_attempt_v2_scope_corrupt");
    const auto admission = LoadObservedAdmissionById(transaction,
        RequestProductionAdmissionId(row[2].as<long long>()));
    const auto enablement = LoadObservedEnablementEventById(transaction,
        ProductionEnablementEventId(row[5].as<long long>()));
    if (row[0].as<long long>() != admission.requestId.value() ||
        row[1].as<std::string>() != row[23].as<std::string>() ||
        row[1].as<std::string>() != admission.requestIdentityCanonical ||
        row[3].as<std::string>() != admission.identity.canonicalText() ||
        row[4].as<std::string>() != admission.identity.hash() ||
        enablement.kind != ProductionEnablementEventKind::enable ||
        !enablement.approvedBuildIdentity ||
        row[5].as<long long>() != admission.enablementEventId.value() ||
        row[6].as<std::string>() != enablement.identity.canonicalText() ||
        row[7].as<std::string>() != enablement.identity.hash() ||
        row[17].as<std::string>() != admission.approvedBuildIdentity.canonicalText() ||
        row[18].as<std::string>() != admission.approvedBuildIdentity.hash() ||
        row[14].as<std::string>() != row[15].as<std::string>() ||
        row[19].as<std::string>() != kProductionDispatcherRole ||
        row[9].as<int>() <= 0 ||
        row[11].as<int>() != row[10].as<int>() + 1)
        ReadinessEvidenceCorrupt("campaign_operations_attempt_v2_typed_mirror_corrupt");
    std::ostringstream canonical;
    canonical << "campaign_operations_dispatch_attempt_v2"
        << ";operational_request_id=" << row[0].as<long long>()
        << ";request_identity_canonical=" << Framed(row[1].as<std::string>())
        << ";request_production_admission_canonical=" << Framed(row[3].as<std::string>())
        << ";enable_event_id=" << row[5].as<long long>()
        << ";enable_event_canonical=" << Framed(row[6].as<std::string>())
        << ";operation_key=" << Framed(row[8].as<std::string>())
        << ";attempt_ordinal=" << row[9].as<int>()
        << ";expected_request_version=" << row[10].as<int>()
        << ";resulting_request_version=" << row[11].as<int>()
        << ";lease_token_digest=" << Framed(row[12].as<std::string>())
        << ";lease_expires_at=" << Framed(row[13].as<std::string>())
        << ";requesting_actor=" << Framed(row[15].as<std::string>())
        << ";original_executing_service_principal=" << Framed(row[16].as<std::string>())
        << ";approved_build_contract=" << Framed(row[17].as<std::string>())
        << ";capability=" << kProductionDispatcherRole
        << ";attempt_contract_version=" << row[20].as<int>();
    (void)HydrateStructuralIdentity(row[21].as<std::string>(),
        row[22].as<std::string>(), canonical.str(),
        "campaign_operations_attempt_v2_corrupt");
    (void)LoadAcquisitionAudit(transaction, attemptId, admission.requestId,
        admission.admissionId, admission.enablementEventId,
        ActorIdentity(row[15].as<std::string>()), row[10].as<int>(),
        row[11].as<int>());
    return {row[20].as<int>()};
}

std::optional<PersistedProductionEnablementHead> LoadEnablementEventById(
    pqxx::transaction_base& transaction, ProductionEnablementEventId eventId,
    int remainingDepth = 1024)
{
    if (remainingDepth <= 0)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_enablement_chain_cycle");
    const auto rows = transaction.exec(
        "SELECT " + HeadColumns() + " FROM "
        "campaign_operations_production_enablement_event "
        "WHERE production_enablement_event_id=$1;",
        pqxx::params{eventId.value()});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
    auto result = MapHead(row, LoadEnablementAudit(transaction, row));
    const auto predecessorId = OptionalEventId(row[3]);
    if (predecessorId)
    {
        auto predecessor = LoadEnablementEventById(
            transaction, *predecessorId, remainingDepth - 1);
        if (!predecessor ||
            predecessor->identity.canonicalText() != row[4].as<std::string>() ||
            predecessor->resultingVersion != row[5].as<int>() ||
            predecessor->kind == result.kind)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_enablement_predecessor_corrupt");
    }
    else if (!row[4].as<std::string>().empty() || row[5].as<int>() != 0 ||
             result.kind != ProductionEnablementEventKind::enable)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_enablement_genesis_corrupt");
    return result;
}

PersistedProductionDispatchAttemptV2Evidence LoadAttemptV2Evidence(
    pqxx::transaction_base& transaction, DispatchAttemptId attemptId,
    RequestProductionAdmissionId admissionId,
    const RequestProductionAdmission& admission)
{
    const auto rows = transaction.exec(
        "SELECT attempt.operational_request_id,"
        "attempt.request_identity_canonical,"
        "attempt.request_production_admission_id,"
        "attempt.request_production_admission_canonical,"
        "attempt.request_production_admission_hash,"
        "attempt.production_enablement_event_id,"
        "attempt.production_enablement_event_canonical,"
        "attempt.production_enablement_event_hash,attempt.operation_key,"
        "attempt.attempt_ordinal,attempt.expected_request_version,"
        "attempt.resulting_request_version,attempt.lease_token_digest,"
        "to_char(attempt.lease_expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"'),"
        "attempt.dispatcher_identity,attempt.requesting_actor,"
        "attempt.original_executing_service_principal,"
        "attempt.approved_build_contract_canonical,"
        "attempt.approved_build_contract_hash,attempt.production_capability,"
        "attempt.attempt_contract_version,attempt.attempt_identity_canonical,"
        "attempt.attempt_identity_hash,request.request_identity_canonical "
        "FROM campaign_operations_dispatch_attempt attempt JOIN "
        "campaign_operations_operational_request request ON "
        "request.operational_request_id=attempt.operational_request_id "
        "WHERE attempt.dispatch_attempt_id=$1;",
        pqxx::params{attemptId.value()});
    if (rows.size() != 1U)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_attempt_v2_missing");
    const auto& row = rows.one_row();
    for (const auto& field : row)
        if (field.is_null())
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_attempt_v2_typed_mirror_corrupt");
    const OperationalRequestId requestId(row[0].as<long long>());
    auto enablement = LoadEnablementEventById(transaction,
        ProductionEnablementEventId(row[5].as<long long>()));
    if (requestId != admission.requestId ||
        row[1].as<std::string>() != row[23].as<std::string>() ||
        row[1].as<std::string>() != admission.requestIdentityCanonical ||
        row[2].as<long long>() != admissionId.value() ||
        row[3].as<std::string>() != admission.identity.canonicalText() ||
        row[4].as<std::string>() != admission.identity.hash() ||
        !enablement || enablement->kind !=
            ProductionEnablementEventKind::enable ||
        !enablement->approvedBuildContract ||
        row[6].as<std::string>() != enablement->identity.canonicalText() ||
        row[7].as<std::string>() != enablement->identity.hash() ||
        row[17].as<std::string>() != enablement->approvedBuildContract->
            identity.canonicalText() ||
        row[18].as<std::string>() != enablement->approvedBuildContract->
            identity.hash() ||
        row[14].as<std::string>() != row[15].as<std::string>() ||
        row[19].as<std::string>() != kProductionDispatcherRole ||
        row[20].as<int>() != kProductionAttemptContractVersion)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_attempt_v2_typed_mirror_corrupt");
    auto attempt = BuildProductionDispatchAttemptV2(requestId,
        row[1].as<std::string>(), admission,
        ProductionEnablementEventId(row[5].as<long long>()),
        row[6].as<std::string>(), row[8].as<std::string>(), row[9].as<int>(),
        row[10].as<int>(), row[11].as<int>(),
        LeaseTokenDigest::Hydrate(row[12].as<std::string>()),
        UtcTimestamp(row[13].as<std::string>()),
        ActorIdentity(row[15].as<std::string>()), row[16].as<std::string>(),
        *enablement->approvedBuildContract);
    if (attempt.identity.canonicalText() != row[21].as<std::string>() ||
        attempt.identity.hash() != row[22].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_attempt_v2_corrupt");
    auto acquisitionAudit = LoadAcquisitionAudit(transaction, attemptId,
        requestId, admissionId, attempt.enableEventId,
        attempt.requestingActor, attempt.expectedRequestVersion,
        attempt.resultingRequestVersion);
    return {attemptId, std::move(attempt), std::move(*enablement),
        std::move(acquisitionAudit)};
}

} // namespace

bool ProductionAdmissionSchemaExists(pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT to_regclass("
        "'campaign_operations_production_enablement_event') IS NOT NULL "
        "AND to_regclass("
        "'campaign_operations_request_production_admission') IS NOT NULL "
        "AND to_regclass("
        "'campaign_operations_production_readiness_v1') IS NOT NULL;")
        .one_row()[0].as<bool>();
}

std::optional<SchedulerProtocolEvidence>
LoadSchedulerProtocolEvidenceSnapshot(pqxx::transaction_base& transaction)
{
    const auto row = transaction.exec(
        "SELECT required_generation,cutover_state,cutover_completed_at,"
        "cutover_completed_by,cutover_executable_path,"
        "cutover_process_evidence,evidence_complete,evidence_canonical,"
        "evidence_hash FROM "
        "campaign_operations_scheduler_protocol_evidence_snapshot_v1();")
        .one_row();
    if (!row[6].as<bool>()) return std::nullopt;
    return SchedulerFromRow(row, 0, 1, 2, 3, 4, 5, 7, 8);
}

SchedulerProtocolEvidence LockSchedulerProtocolEvidence(
    pqxx::transaction_base& transaction)
{
    const auto row = transaction.exec(
        "SELECT required_generation,cutover_state,cutover_completed_at,"
        "cutover_completed_by,cutover_executable_path,"
        "cutover_process_evidence,evidence_complete,evidence_canonical,"
        "evidence_hash FROM "
        "campaign_operations_scheduler_protocol_evidence_lock_v1();")
        .one_row();
    if (!row[6].as<bool>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_scheduler_evidence_incomplete");
    return SchedulerFromRow(row, 0, 1, 2, 3, 4, 5, 7, 8);
}

std::optional<PersistedProductionEnablementHead>
FindCurrentProductionEnablementHead(pqxx::transaction_base& transaction)
{
    const auto rows = transaction.exec(
        "SELECT production_enablement_event_id FROM "
        "campaign_operations_production_enablement_event "
        "ORDER BY resulting_version DESC LIMIT 1;");
    if (rows.empty()) return std::nullopt;
    return LoadEnablementEventById(transaction,
        ProductionEnablementEventId(rows.one_row()[0].as<long long>()));
}

std::optional<PersistedProductionEnablementHead>
FindProductionEnablementEventByOperationKey(
    pqxx::transaction_base& transaction, const std::string& operationKey)
{
    const auto rows = transaction.exec(
        "SELECT production_enablement_event_id FROM "
        "campaign_operations_production_enablement_event "
        "WHERE operation_key=$1;", pqxx::params{operationKey});
    if (rows.empty()) return std::nullopt;
    return LoadEnablementEventById(transaction,
        ProductionEnablementEventId(rows.one_row()[0].as<long long>()));
}

std::optional<PersistedRequestProductionAdmission>
FindRequestProductionAdmission(pqxx::transaction_base& transaction,
    OperationalRequestId requestId)
{
    const auto rows = transaction.exec(
        "SELECT admission.request_production_admission_id,"
        "admission.operational_request_id,"
        "admission.request_identity_canonical,"
        "admission.expected_request_version,"
        "admission.dispatch_operation_key,"
        "admission.production_enablement_event_id,"
        "admission.enable_event_canonical,admission.requesting_actor,"
        "admission.original_executing_service_principal,"
        "admission.approved_build_contract_canonical,"
        "admission.approved_build_contract_hash,admission.capability,"
        "admission.admission_contract_version,"
        "admission.admission_identity_canonical,"
        "admission.admission_identity_hash,"
        "request.request_identity_canonical "
        "FROM campaign_operations_request_production_admission admission "
        "JOIN campaign_operations_operational_request request ON "
        "request.operational_request_id=admission.operational_request_id "
        "WHERE admission.operational_request_id=$1;",
        pqxx::params{requestId.value()});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
    const auto storedRequestId = OperationalRequestId(row[1].as<long long>());
    if (storedRequestId != requestId ||
        row[2].as<std::string>() != row[15].as<std::string>() ||
        row[11].as<std::string>() != kProductionDispatcherRole ||
        row[12].as<int>() != kProductionAdmissionContractVersion)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_admission_typed_mirror_corrupt");
    auto enablement = LoadEnablementEventById(transaction,
        ProductionEnablementEventId(row[5].as<long long>()));
    if (!enablement || enablement->kind !=
            ProductionEnablementEventKind::enable ||
        !enablement->approvedBuildContract ||
        enablement->identity.canonicalText() != row[6].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_admission_enablement_corrupt");
    auto build = *enablement->approvedBuildContract;
    if (build.identity.canonicalText() != row[9].as<std::string>() ||
        build.identity.hash() != row[10].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_admission_build_corrupt");
    auto admission = BuildRequestProductionAdmission(storedRequestId,
        row[2].as<std::string>(), row[3].as<int>(),
        row[4].as<std::string>(),
        ProductionEnablementEventId(row[5].as<long long>()),
        row[6].as<std::string>(), ActorIdentity(row[7].as<std::string>()),
        row[8].as<std::string>(), std::move(build));
    if (admission.identity.canonicalText() != row[13].as<std::string>() ||
        admission.identity.hash() != row[14].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_admission_corrupt");
    const auto admissionId = RequestProductionAdmissionId(
        row[0].as<long long>());
    const auto firstAttemptRows = transaction.exec(
        "SELECT dispatch_attempt_id FROM "
        "campaign_operations_dispatch_attempt WHERE "
        "request_production_admission_id=$1 AND attempt_ordinal=1;",
        pqxx::params{admissionId.value()});
    if (firstAttemptRows.size() != 1U)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_admission_first_attempt_missing");
    const DispatchAttemptId firstAttemptId(
        firstAttemptRows.one_row()[0].as<long long>());
    auto firstAttempt = LoadAttemptV2Evidence(transaction, firstAttemptId,
        admissionId, admission);
    if (firstAttempt.attempt.requestId != admission.requestId ||
        firstAttempt.attempt.requestIdentityCanonical !=
            admission.requestIdentityCanonical ||
        firstAttempt.attempt.admission.identity != admission.identity ||
        firstAttempt.attempt.enableEventId != admission.enableEventId ||
        firstAttempt.attempt.enableEventCanonical !=
            admission.enableEventCanonical ||
        firstAttempt.authorizingEnablement.eventId != admission.enableEventId ||
        firstAttempt.authorizingEnablement.identity !=
            enablement->identity ||
        firstAttempt.attempt.operationKey != admission.dispatchOperationKey ||
        firstAttempt.attempt.attemptOrdinal != 1 ||
        firstAttempt.attempt.expectedRequestVersion !=
            admission.expectedRequestVersion ||
        firstAttempt.attempt.resultingRequestVersion !=
            admission.expectedRequestVersion + 1 ||
        firstAttempt.attempt.requestingActor != admission.requestingActor ||
        firstAttempt.attempt.originalExecutingServicePrincipal !=
            admission.originalExecutingServicePrincipal ||
        firstAttempt.attempt.approvedBuildContract.identity !=
            admission.approvedBuildContract.identity)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_admission_first_attempt_corrupt");
    return PersistedRequestProductionAdmission{
        admissionId, std::move(admission), std::move(*enablement),
        std::move(firstAttempt)};
}

std::optional<PersistedProductionDispatchAttemptV2>
FindProductionDispatchAttemptV2(pqxx::transaction_base& transaction,
    DispatchAttemptId attemptId)
{
    const auto rows = transaction.exec(
        "SELECT operational_request_id,request_production_admission_id "
        "FROM campaign_operations_dispatch_attempt WHERE "
        "dispatch_attempt_id=$1 AND attempt_contract_version=2;",
        pqxx::params{attemptId.value()});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
    const OperationalRequestId requestId(row[0].as<long long>());
    auto persistedAdmission = FindRequestProductionAdmission(
        transaction, requestId);
    if (!persistedAdmission || persistedAdmission->admissionId.value() !=
            row[1].as<long long>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_attempt_v2_admission_missing");
    auto evidence = LoadAttemptV2Evidence(transaction, attemptId,
        persistedAdmission->admissionId, persistedAdmission->admission);
    return PersistedProductionDispatchAttemptV2{attemptId,
        std::move(evidence.attempt), std::move(*persistedAdmission),
        std::move(evidence.authorizingEnablement),
        std::move(evidence.acquisitionAudit)};
}

std::optional<PersistedProductionDispatchAttemptV2>
FindProductionDispatchAttemptV2(pqxx::transaction_base& transaction,
    OperationalRequestId requestId, const std::string& operationKey)
{
    const auto rows = transaction.exec(
        "SELECT dispatch_attempt_id "
        "FROM campaign_operations_dispatch_attempt "
        "WHERE operational_request_id=$1 AND operation_key=$2 "
        "AND attempt_contract_version=2;",
        pqxx::params{requestId.value(), operationKey});
    if (rows.empty()) return std::nullopt;
    return FindProductionDispatchAttemptV2(transaction,
        DispatchAttemptId(rows.one_row()[0].as<long long>()));
}

ProductionReadinessSnapshot LoadProductionReadinessSnapshot(
    pqxx::transaction_base& transaction, bool hydrateEvidence)
{
    const auto row = transaction.exec(
        "SELECT migration_version,migration_filename,migration_checksum,"
        "scheduler_evidence_contract_version,manager_build_contract_version,"
        "enablement_contract_version,admission_contract_version,"
        "production_attempt_contract_version,"
        "scheduler_required_generation,scheduler_cutover_state,"
        "scheduler_evidence_complete,independent_verification_reference,"
        "session_principal,current_principal,"
        "reader_member,dispatcher_member,phase5_transactional_member,"
        "scheduler_evidence_reader_member,enabler_member,disabler_member,"
        "prohibited_test_dispatcher_member,prohibited_test_phase5_member,"
        "coalesce(enablement_effective,false),ready_admitted_request_count,"
        "ready_unadmitted_request_count,active_current_event_lease_count,"
        "old_event_blocked_lease_count,reconciliation_required_count,"
        "completion_nested_v2_proof_version,"
        "completion_nested_v2_proof_valid "
#ifdef CAMPAIGN_OPERATIONS_H1_READINESS_VIEW
        "FROM campaign_operations_production_readiness_v1;").one_row();
#else
        "FROM campaign_operations_production_readiness_snapshot_v1();").one_row();
#endif
    ProductionReadinessSnapshot result;
    result.migrationVersion = row[0].as<std::string>();
    if (!row[1].is_null())
        result.migrationFilename = row[1].as<std::string>();
    if (!row[2].is_null())
        result.migrationChecksum = row[2].as<std::string>();
    for (const auto [index, destination] :
         {std::pair{3, &result.schedulerEvidenceContractVersion},
          std::pair{4, &result.managerBuildContractVersion},
          std::pair{5, &result.enablementContractVersion},
          std::pair{6, &result.admissionContractVersion},
          std::pair{7, &result.productionAttemptContractVersion}})
        if (!row[index].is_null()) *destination = row[index].as<std::string>();
    if (!row[8].is_null()) result.schedulerGeneration = row[8].as<int>();
    if (!row[9].is_null())
        result.schedulerCutoverState = row[9].as<std::string>();
    result.schedulerEvidenceComplete = row[10].as<bool>();
    if (!row[11].is_null())
        result.independentVerificationReference = row[11].as<std::string>();
    if (auto scheduler = LoadSchedulerProtocolEvidenceSnapshot(transaction))
        result.schedulerEvidence.emplace(std::move(*scheduler));
    if (!hydrateEvidence) return result;

    // Presence is deliberately collected independently of the nullable
    // aggregate fields.  In particular, a row whose version cannot be
    // hydrated must not look like an empty genesis family.
    const auto admissionRows = transaction.exec(
        "SELECT admission.request_production_admission_id FROM "
        "campaign_operations_request_production_admission admission "
        "ORDER BY admission.request_production_admission_id;");
    result.admissionEvidenceCount =
        static_cast<long long>(admissionRows.size());
    const auto attemptRows = transaction.exec(
        "SELECT dispatch_attempt_id FROM campaign_operations_dispatch_attempt "
        "WHERE attempt_contract_version=2 OR "
        "request_production_admission_id IS NOT NULL OR "
        "request_production_admission_canonical IS NOT NULL OR "
        "request_production_admission_hash IS NOT NULL OR "
        "production_enablement_event_id IS NOT NULL OR "
        "production_enablement_event_canonical IS NOT NULL OR "
        "production_enablement_event_hash IS NOT NULL OR operation_key IS NOT NULL OR "
        "requesting_actor IS NOT NULL OR "
        "original_executing_service_principal IS NOT NULL OR "
        "approved_build_contract_canonical IS NOT NULL OR "
        "approved_build_contract_hash IS NOT NULL OR production_capability IS NOT NULL "
        "ORDER BY dispatch_attempt_id;");
    result.productionAttemptEvidenceCount =
        static_cast<long long>(attemptRows.size());
    // Structural hydration intentionally precedes every normative comparison.
    // In particular, an unsupported but internally consistent version remains
    // visible, while a stale hash, bad audit, or broken relationship fails at
    // this boundary instead of being reduced to a version blocker.
    std::set<int> enablementVersions;
    std::set<int> managerBuildVersions;
    std::optional<ObservedProductionEnablementEvidence> observedHead;
    const auto enablementRows = transaction.exec(
        "SELECT production_enablement_event_id FROM "
        "campaign_operations_production_enablement_event "
        "ORDER BY resulting_version DESC;");
    for (const auto& eventRow : enablementRows)
    {
        auto observed = LoadObservedEnablementEventById(transaction,
            ProductionEnablementEventId(eventRow[0].as<long long>()));
        enablementVersions.insert(observed.enablementContractVersion);
        if (observed.managerBuildContractVersion)
            managerBuildVersions.insert(*observed.managerBuildContractVersion);
        if (!observedHead) observedHead.emplace(std::move(observed));
    }
    if (observedHead)
        result.observedEnablementHead.emplace(std::move(*observedHead));
    if (!enablementVersions.empty())
        result.enablementContractVersion = JoinVersions(enablementVersions);
    else result.enablementContractVersion.reset();
    if (!managerBuildVersions.empty())
        result.managerBuildContractVersion = JoinVersions(managerBuildVersions);
    else result.managerBuildContractVersion.reset();

    std::set<int> admissionVersions;
    for (const auto& admissionRow : admissionRows)
    {
        const RequestProductionAdmissionId admissionId(
            admissionRow[0].as<long long>());
        const auto admission = LoadObservedAdmissionById(transaction, admissionId);
        admissionVersions.insert(admission.contractVersion);
    }
    if (!admissionVersions.empty())
        result.admissionContractVersion = JoinVersions(admissionVersions);
    else result.admissionContractVersion.reset();

    std::set<int> attemptVersions;
    for (const auto& attemptRow : attemptRows)
    {
        auto attempt = LoadObservedAttemptById(transaction,
            DispatchAttemptId(attemptRow[0].as<long long>()));
        attemptVersions.insert(attempt.contractVersion);
    }
    if (!attemptVersions.empty())
        result.productionAttemptContractVersion = JoinVersions(attemptVersions);
    else result.productionAttemptContractVersion.reset();
    const auto isExpected = [](const std::optional<std::string>& observed,
                                int expected)
    {
        return observed && *observed == std::to_string(expected);
    };
    // This second hydration produces the existing normative domain object for
    // callers that need it.  It is not an integrity gate: the unconditional
    // observed hydrator above has already validated the entire graph.
    if (isExpected(result.enablementContractVersion,
            kProductionEnablementContractVersion) &&
        isExpected(result.managerBuildContractVersion,
            kManagerBuildContractVersion))
        if (auto head = FindCurrentProductionEnablementHead(transaction))
            result.enablementHead.emplace(std::move(*head));
    result.sessionPrincipal = row[12].as<std::string>();
    result.currentPrincipal = row[13].as<std::string>();
    result.readerMember = row[14].as<bool>();
    result.dispatcherMember = row[15].as<bool>();
    result.phase5TransactionalMember = row[16].as<bool>();
    result.schedulerEvidenceReaderMember = row[17].as<bool>();
    result.enablerMember = row[18].as<bool>();
    result.disablerMember = row[19].as<bool>();
    result.prohibitedTestDispatcherMember = row[20].as<bool>();
    result.prohibitedTestPhase5Member = row[21].as<bool>();
    result.enablementEffective = row[22].as<bool>();
    result.readyAdmittedRequestCount = row[23].as<long long>();
    result.readyUnadmittedRequestCount = row[24].as<long long>();
    result.activeCurrentEventLeaseCount = row[25].as<long long>();
    result.oldEventBlockedLeaseCount = row[26].as<long long>();
    result.reconciliationRequiredCount = row[27].as<long long>();
    if (!row[28].is_null())
        result.completionNestedV2ProofVersion = row[28].as<std::string>();
    result.completionNestedV2ProofValid = row[29].as<bool>();
    return result;
}

std::vector<ProductionRequestStatus> LoadProductionStatusSnapshot(
    pqxx::transaction_base& transaction)
{
    const auto rows = transaction.exec(
        "SELECT operational_request_id,operational_campaign_id,request_state,"
        "state_version,production_dispatch_enabled,"
        "request_production_admission_id,dispatch_attempt_id,operation_key,"
        "production_enablement_event_id,"
        "CASE WHEN lease_expires_at IS NULL THEN NULL ELSE to_char("
        "lease_expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') END,"
        "coalesce(lease_expired,false),coalesce(phase_f_recovery_eligible,false),"
        "reconciliation_required FROM "
        "campaign_operations_production_status_v1 "
        "ORDER BY operational_request_id;");
    std::vector<ProductionRequestStatus> result;
    result.reserve(rows.size());
    for (const auto& row : rows)
    {
        ProductionRequestStatus status{
            OperationalRequestId(row[0].as<long long>()),
            OperationalCampaignId(row[1].as<long long>()),
            row[2].as<std::string>(), row[3].as<int>(), row[4].as<bool>(),
            std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, false, false, false};
        if (!row[5].is_null()) status.admissionId.emplace(
            row[5].as<long long>());
        if (!row[6].is_null()) status.attemptId.emplace(
            row[6].as<long long>());
        if (!row[7].is_null()) status.operationKey = row[7].as<std::string>();
        if (!row[8].is_null()) status.enablementEventId.emplace(
            row[8].as<long long>());
        if (!row[9].is_null()) status.leaseExpiresAt = row[9].as<std::string>();
        status.leaseExpired = row[10].as<bool>();
        status.phaseFRecoveryEligible = row[11].as<bool>();
        status.reconciliationRequired = row[12].as<bool>();
        result.push_back(std::move(status));
    }
    return result;
}

} // namespace EA::CampaignOperations
