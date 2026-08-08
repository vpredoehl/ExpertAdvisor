#include "CampaignOperationsProductionAdmissionService.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <ostream>
#include <sstream>
#include <utility>

namespace EA::CampaignOperations
{
namespace
{

std::string MachineText(const std::string& value)
{
    std::string result;
    result.reserve(value.size());
    for (const unsigned char character : value)
    {
        if (std::isalnum(character) || character == '_' || character == '-' ||
            character == '.' || character == ':' || character == '/' ||
            character == '@' || character == '|')
            result.push_back(static_cast<char>(character));
        else result.push_back('_');
    }
    return result;
}

template <typename Value>
std::string MachineOptional(const std::optional<Value>& value,
    const char* absent = "missing")
{
    if (!value) return absent;
    if constexpr (std::is_same_v<Value, std::string>) return MachineText(*value);
    else return std::to_string(*value);
}

bool ObservedVersionMatches(const std::optional<std::string>& observed,
    int expected)
{
    return observed && *observed == std::to_string(expected);
}

bool HasObservedEnablement(const ProductionReadinessSnapshot& snapshot)
{
    return snapshot.observedEnablementHead.has_value() ||
        snapshot.enablementHead.has_value();
}

ProductionEnablementEventKind ObservedEnablementKind(
    const ProductionReadinessSnapshot& snapshot)
{
    if (snapshot.observedEnablementHead)
        return snapshot.observedEnablementHead->kind;
    return snapshot.enablementHead->kind;
}

std::optional<std::string> ObservedManagerServiceContract(
    const ProductionReadinessSnapshot& snapshot)
{
    if (snapshot.observedEnablementHead)
        return snapshot.observedEnablementHead->managerServiceContract;
    if (snapshot.enablementHead && snapshot.enablementHead->approvedBuildContract)
        return snapshot.enablementHead->approvedBuildContract->managerServiceContract;
    return std::nullopt;
}

std::optional<CanonicalIdentity> ObservedEnablementIdentity(
    const ProductionReadinessSnapshot& snapshot)
{
    if (snapshot.observedEnablementHead)
        return snapshot.observedEnablementHead->identity;
    if (snapshot.enablementHead) return snapshot.enablementHead->identity;
    return std::nullopt;
}

std::optional<CanonicalIdentity> ObservedApprovedBuildIdentity(
    const ProductionReadinessSnapshot& snapshot)
{
    if (snapshot.observedEnablementHead)
        return snapshot.observedEnablementHead->approvedBuildIdentity;
    if (snapshot.enablementHead && snapshot.enablementHead->approvedBuildContract)
        return snapshot.enablementHead->approvedBuildContract->identity;
    return std::nullopt;
}

template <typename Function>
int RunReadCommand(Function&& function, std::ostream& errors,
    const char* marker)
{
    try { return function(); }
    catch (const pqxx::sql_error& error)
    {
        errors << marker << ",status=failed,diagnostic_code=postgresql_"
               << MachineText(error.sqlstate())
               << ",diagnostic_detail="
               << MachineText(error.what()) << '\n';
    }
    catch (const std::exception& error)
    {
        errors << marker << ",status=failed,diagnostic_code="
               << MachineText(error.what()) << '\n';
    }
    return 1;
}

std::string Join(const std::vector<std::string>& values)
{
    if (values.empty()) return "none";
    std::string result;
    for (const auto& value : values)
    {
        if (!result.empty()) result += ';';
        result += value;
    }
    return result;
}

std::string ShellQuoted(const std::string& value)
{
    std::string result{"'"};
    for (const char character : value)
        result += character == '\'' ? "'\\''" : std::string(1, character);
    return result + "'";
}

std::optional<std::string> ExecutableSha256(const std::string& path)
{
    const std::string command =
        "/usr/bin/shasum -a 256 -- " + ShellQuoted(path) + " 2>/dev/null";
    FILE* pipe = ::popen(command.c_str(), "r");
    if (!pipe) return std::nullopt;
    std::array<char, 256> buffer{};
    std::ostringstream output;
    while (::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe))
        output << buffer.data();
    if (::pclose(pipe) != 0) return std::nullopt;
    const std::string text = output.str();
    const auto separator = text.find_first_of(" \t\r\n");
    const std::string digest = text.substr(0, separator);
    if (digest.size() != 64U ||
        !std::all_of(digest.begin(), digest.end(), [](unsigned char character)
        {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f');
        }))
        return std::nullopt;
    return "sha256:" + digest;
}

std::optional<std::string> CommandOutput(const std::string& command)
{
    FILE* pipe = ::popen(command.c_str(), "r");
    if (!pipe) return std::nullopt;
    std::array<char, 256> buffer{};
    std::ostringstream output;
    while (::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe))
        output << buffer.data();
    if (::pclose(pipe) != 0) return std::nullopt;
    std::string result = output.str();
    while (!result.empty() && std::isspace(
        static_cast<unsigned char>(result.back())))
        result.pop_back();
    return result;
}

} // namespace

std::optional<ManagerBuildContract> CaptureActualManagerBuildContract(
    const std::string& executablePath)
{
    const auto sourceCommit = CommandOutput(
        "git rev-parse HEAD 2>/dev/null");
    const auto sourceStatus = CommandOutput(
        "git status --porcelain 2>/dev/null");
    const auto executableSha256 = ExecutableSha256(executablePath);
#if defined(__clang_version__)
    const std::string compilerContract = __clang_version__;
#elif defined(__VERSION__)
    const std::string compilerContract = __VERSION__;
#else
    const std::string compilerContract;
#endif
#if !defined(NDEBUG)
    return std::nullopt;
#endif
    if (!sourceCommit || sourceCommit->size() != 40U || !sourceStatus ||
        !sourceStatus->empty() || compilerContract.empty() ||
        !executableSha256)
        return std::nullopt;
    try
    {
        return BuildManagerBuildContract(kManagerServiceContract,
            *sourceCommit, compilerContract,
            *executableSha256);
    }
    catch (const Error&)
    {
        return std::nullopt;
    }
}

ProductionReadinessEvaluation EvaluateProductionReadiness(
    ProductionReadinessSnapshot snapshot,
    std::optional<ManagerBuildContract> actualBuildContract)
{
    ProductionReadinessEvaluation result{
        std::move(snapshot), std::nullopt, false, {}};
    if (actualBuildContract)
        result.actualBuildContract.emplace(std::move(*actualBuildContract));
    if (result.snapshot.migrationVersion != "055" ||
        result.snapshot.migrationFilename !=
          kProductionAdmissionMigrationFilename ||
        !result.snapshot.migrationChecksum ||
        *result.snapshot.migrationChecksum !=
          kProductionAdmissionMigrationChecksum)
        result.blockers.push_back("migration_055_identity_or_checksum");
    if (!ObservedVersionMatches(result.snapshot.schedulerEvidenceContractVersion,
            kSchedulerProtocolEvidenceContractVersion) ||
        !ObservedVersionMatches(result.snapshot.managerBuildContractVersion,
            kManagerBuildContractVersion) ||
        !ObservedVersionMatches(result.snapshot.enablementContractVersion,
            kProductionEnablementContractVersion) ||
        !ObservedVersionMatches(result.snapshot.admissionContractVersion,
            kProductionAdmissionContractVersion) ||
        !ObservedVersionMatches(result.snapshot.productionAttemptContractVersion,
            kProductionAttemptContractVersion))
        result.blockers.push_back("canonical_contract_versions");
    if (!result.snapshot.schedulerEvidenceComplete ||
        !result.snapshot.schedulerEvidence ||
        !result.snapshot.schedulerGeneration ||
        *result.snapshot.schedulerGeneration !=
            kRequiredSchedulerProtocolGeneration ||
        !result.snapshot.schedulerCutoverState ||
        *result.snapshot.schedulerCutoverState != "complete")
        result.blockers.push_back("scheduler_protocol_evidence");
    if (!HasObservedEnablement(result.snapshot))
        result.blockers.push_back("enablement_absent");
    else if (ObservedEnablementKind(result.snapshot) !=
                 ProductionEnablementEventKind::enable ||
             !result.snapshot.enablementEffective)
        result.blockers.push_back("enablement_ineffective");
    if (!result.snapshot.independentVerificationReference)
        result.blockers.push_back("independent_verification_reference");
    if (!result.actualBuildContract)
        result.blockers.push_back("actual_manager_build_contract");
    else if (!ObservedApprovedBuildIdentity(result.snapshot) ||
             result.actualBuildContract->identity.canonicalText() !=
                 ObservedApprovedBuildIdentity(result.snapshot)->canonicalText())
        result.blockers.push_back("manager_build_contract_mismatch");
    if (const auto service = ObservedManagerServiceContract(result.snapshot);
        service && *service != kManagerServiceContract)
        result.blockers.push_back("manager_service_contract");
    if (result.snapshot.sessionPrincipal.empty() ||
        result.snapshot.currentPrincipal.empty())
        result.blockers.push_back("session_principal_identity");
    if (!result.snapshot.readerMember || !result.snapshot.dispatcherMember ||
        !result.snapshot.phase5TransactionalMember ||
        !result.snapshot.schedulerEvidenceReaderMember ||
        result.snapshot.enablerMember || result.snapshot.disablerMember ||
        result.snapshot.prohibitedTestDispatcherMember ||
        result.snapshot.prohibitedTestPhase5Member)
        result.blockers.push_back("principal_role_membership");
    if (!result.snapshot.completionNestedV2ProofValid ||
        !ObservedVersionMatches(result.snapshot.completionNestedV2ProofVersion,
            1))
        result.blockers.push_back("completion_nested_v2_proof");
    if (result.snapshot.reconciliationRequiredCount != 0)
        result.blockers.push_back("reconciliation_required");
    result.ready = result.blockers.empty();
    return result;
}

ProductionReadinessEvaluation LoadProductionReadiness(
    pqxx::connection& connection,
    std::optional<ManagerBuildContract> actualBuildContract)
{
    pqxx::read_transaction transaction{connection};
    transaction.exec(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
    if (!ProductionAdmissionSchemaExists(transaction))
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_production_schema_required");
    auto snapshot = LoadProductionReadinessSnapshot(transaction);
    transaction.commit();
    return EvaluateProductionReadiness(
        std::move(snapshot), std::move(actualBuildContract));
}

std::string RenderProductionReadiness(
    const ProductionReadinessEvaluation& value)
{
    const auto& snapshot = value.snapshot;
    const auto observedEnablement = ObservedEnablementIdentity(snapshot);
    const auto observedBuild = ObservedApprovedBuildIdentity(snapshot);
    const auto observedManagerService = ObservedManagerServiceContract(snapshot);
    std::ostringstream output;
    output << "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS"
           << ",ready=" << (value.ready ? "true" : "false")
           << ",migration_version=" << snapshot.migrationVersion
           << ",migration_filename="
           << MachineText(snapshot.migrationFilename.value_or("missing"))
           << ",migration_checksum="
           << MachineText(snapshot.migrationChecksum.value_or("missing"))
           << ",scheduler_contract_version="
           << MachineOptional(snapshot.schedulerEvidenceContractVersion)
           << ",expected_scheduler_contract_version="
           << kSchedulerProtocolEvidenceContractVersion
           << ",scheduler_generation="
           << MachineOptional(snapshot.schedulerGeneration)
           << ",scheduler_cutover_state="
           << MachineOptional(snapshot.schedulerCutoverState)
           << ",scheduler_evidence_complete="
           << (snapshot.schedulerEvidenceComplete ? "true" : "false")
           << ",scheduler_evidence_canonical="
           << MachineText(snapshot.schedulerEvidence
                ? snapshot.schedulerEvidence->identity.canonicalText()
                : "none")
           << ",scheduler_evidence_hash="
           << MachineText(snapshot.schedulerEvidence
                ? snapshot.schedulerEvidence->identity.hash() : "none")
           << ",enablement_contract_version="
           << MachineOptional(snapshot.enablementContractVersion)
           << ",expected_enablement_contract_version="
           << kProductionEnablementContractVersion
           << ",manager_build_contract_version="
           << MachineOptional(snapshot.managerBuildContractVersion)
           << ",expected_manager_build_contract_version="
           << kManagerBuildContractVersion
           << ",manager_service_contract="
           << MachineText(observedManagerService.value_or("none"))
           << ",expected_manager_service_contract="
           << kManagerServiceContract
           << ",admission_contract_version="
           << MachineOptional(snapshot.admissionContractVersion)
           << ",expected_admission_contract_version="
           << kProductionAdmissionContractVersion
           << ",production_attempt_contract_version="
           << MachineOptional(snapshot.productionAttemptContractVersion)
           << ",expected_production_attempt_contract_version="
           << kProductionAttemptContractVersion
           << ",independent_verification_reference="
           << MachineText(snapshot.independentVerificationReference.
                value_or("none"))
           << ",enablement_kind="
           << (HasObservedEnablement(snapshot)
                 ? ToText(ObservedEnablementKind(snapshot)) : "none")
           << ",enablement_version="
           << (snapshot.observedEnablementHead
                 ? std::to_string(snapshot.observedEnablementHead->resultingVersion)
                 : snapshot.enablementHead
                 ? std::to_string(snapshot.enablementHead->resultingVersion)
                 : "missing")
           << ",enablement_canonical="
           << MachineText(observedEnablement
                ? observedEnablement->canonicalText() : "none")
           << ",enablement_hash="
           << MachineText(observedEnablement
                ? observedEnablement->hash() : "none")
           << ",approved_build_canonical="
           << MachineText(observedBuild
               ? observedBuild->canonicalText() : "none")
           << ",approved_build_hash="
           << MachineText(observedBuild ? observedBuild->hash() : "none")
           << ",actual_running_build_canonical="
           << MachineText(value.actualBuildContract
               ? value.actualBuildContract->identity.canonicalText()
               : "unavailable")
           << ",actual_running_build_hash="
           << MachineText(value.actualBuildContract
               ? value.actualBuildContract->identity.hash() : "unavailable")
           << ",build_comparison="
           << ((!value.actualBuildContract || !observedBuild)
                   ? "unavailable"
                   : value.actualBuildContract->identity.canonicalText() ==
                       observedBuild->canonicalText() ? "match" : "mismatch")
           << ",enablement_effective="
           << (snapshot.enablementEffective ? "true" : "false")
           << ",session_principal=" << MachineText(snapshot.sessionPrincipal)
           << ",current_principal=" << MachineText(snapshot.currentPrincipal)
           << ",reader_member="
           << (snapshot.readerMember ? "true" : "false")
           << ",dispatcher_member="
           << (snapshot.dispatcherMember ? "true" : "false")
           << ",phase5_transactional_member="
           << (snapshot.phase5TransactionalMember ? "true" : "false")
           << ",scheduler_evidence_reader_member="
           << (snapshot.schedulerEvidenceReaderMember ? "true" : "false")
           << ",enabler_member="
           << (snapshot.enablerMember ? "true" : "false")
           << ",disabler_member="
           << (snapshot.disablerMember ? "true" : "false")
           << ",prohibited_test_dispatcher_member="
           << (snapshot.prohibitedTestDispatcherMember ? "true" : "false")
           << ",prohibited_test_phase5_member="
           << (snapshot.prohibitedTestPhase5Member ? "true" : "false")
           << ",ready_admitted=" << snapshot.readyAdmittedRequestCount
           << ",ready_unadmitted=" << snapshot.readyUnadmittedRequestCount
           << ",current_event_leases=" << snapshot.activeCurrentEventLeaseCount
           << ",old_event_blocked_leases=" << snapshot.oldEventBlockedLeaseCount
           << ",reconciliation_required="
           << snapshot.reconciliationRequiredCount
           << ",completion_nested_v2_proof_version="
           << MachineOptional(snapshot.completionNestedV2ProofVersion)
           << ",expected_completion_nested_v2_proof_version=1"
           << ",completion_nested_v2_proof="
           << (snapshot.completionNestedV2ProofValid ? "valid" : "invalid")
           << ",blockers=" << MachineText(Join(value.blockers)) << '\n';
    return output.str();
}

std::vector<ProductionRequestStatus> LoadProductionStatus(
    pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    transaction.exec(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
    if (!ProductionAdmissionSchemaExists(transaction))
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_production_schema_required");
    auto result = LoadProductionStatusSnapshot(transaction);
    transaction.commit();
    return result;
}

int RunProductionReadinessCommand(const std::string& connectionString,
    std::ostream& output, std::ostream& errors,
    std::optional<ManagerBuildContract> actualBuildContract)
{
    return RunReadCommand([&]
    {
        pqxx::connection connection{connectionString};
        const auto value = LoadProductionReadiness(
            connection, std::move(actualBuildContract));
        output << RenderProductionReadiness(value);
        return value.ready ? 0 : 2;
    }, errors, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS");
}

int RunProductionStatusCommand(const std::string& connectionString,
    std::ostream& output, std::ostream& errors)
{
    return RunReadCommand([&]
    {
        pqxx::connection connection{connectionString};
        const auto rows = LoadProductionStatus(connection);
        for (const auto& row : rows)
        {
            output << "CAMPAIGN_OPERATIONS_PRODUCTION_REQUEST_STATUS"
                   << ",request_id=" << row.requestId.value()
                   << ",campaign_id=" << row.campaignId.value()
                   << ",request_state="
                   << MachineText(row.requestState)
                   << ",state_version=" << row.stateVersion
                   << ",admitted="
                   << (row.productionDispatchEnabled ? "true" : "false")
                   << ",admission_id="
                   << (row.admissionId
                        ? std::to_string(row.admissionId->value()) : "none")
                   << ",attempt_id="
                   << (row.attemptId
                        ? std::to_string(row.attemptId->value()) : "none")
                   << ",operation_key="
                   << MachineText(
                        row.operationKey.value_or("none"))
                   << ",enablement_event_id="
                   << (row.enablementEventId
                        ? std::to_string(row.enablementEventId->value())
                        : "none")
                   << ",lease_expires_at="
                   << row.leaseExpiresAt.value_or("none")
                   << ",lease_expired="
                   << (row.leaseExpired ? "true" : "false")
                   << ",phase_f_recovery_eligible="
                   << (row.phaseFRecoveryEligible ? "true" : "false")
                   << ",reconciliation_required="
                   << (row.reconciliationRequired ? "true" : "false")
                   << '\n';
        }
        output << "CAMPAIGN_OPERATIONS_PRODUCTION_STATUS_COMPLETE,count="
               << rows.size() << '\n';
        return 0;
    }, errors, "CAMPAIGN_OPERATIONS_PRODUCTION_STATUS");
}

} // namespace EA::CampaignOperations
