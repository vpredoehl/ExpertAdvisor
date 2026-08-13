#include "CampaignOperationsProductionAdmission.hpp"

#include <algorithm>
#include <locale>
#include <sstream>
#include <utility>

namespace EA::CampaignOperations
{
namespace
{

std::string Framed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

void RequireText(const std::string& value, const char* diagnostic,
    bool allowEmpty = false)
{
    if ((!allowEmpty && value.empty()) ||
        value.size() > kCampaignOperationsCanonicalMaximumBytes ||
        value.find('\0') != std::string::npos)
        throw Error(ErrorCode::invalidCanonicalText, diagnostic);
    if (!value.empty()) (void)CanonicalIdentity::Create(1, value);
}

bool ValidOperationKey(const std::string& value)
{
    if (value.empty() || value.size() > 128U) return false;
    const auto alphanumeric = [](unsigned char c)
    {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9');
    };
    if (!alphanumeric(static_cast<unsigned char>(value.front()))) return false;
    return std::all_of(value.begin() + 1, value.end(), [&](unsigned char c)
    {
        return alphanumeric(c) || c == '.' || c == '_' || c == ':' ||
            c == '/' || c == '-';
    });
}

void RequireOperationKey(const std::string& value)
{
    if (!ValidOperationKey(value))
        throw Error(ErrorCode::invalidCanonicalText,
            "campaign_operations_production_operation_key_invalid");
}

void RequirePrincipal(const std::string& value)
{
    if (value.empty() || value.size() > 128U)
        throw Error(ErrorCode::invalidActorIdentity,
            "campaign_operations_service_principal_invalid");
    (void)ActorIdentity(value);
}

void RequireSha256(const std::string& value)
{
    if (value.size() != 71U || value.rfind("sha256:", 0U) != 0U ||
        !std::all_of(value.begin() + 7, value.end(), [](unsigned char c)
        {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }))
        throw Error(ErrorCode::invalidCanonicalHash,
            "campaign_operations_manager_executable_sha256_invalid");
}

std::string OptionalId(
    const std::optional<ProductionEnablementEventId>& value)
{
    return value ? std::to_string(value->value()) : "none";
}

CanonicalIdentity Identity(const std::string& value)
{
    return CanonicalIdentity::Create(1, value);
}

} // namespace

bool IsValidProductionOperationKey(const std::string& value)
{
    return ValidOperationKey(value);
}

SchedulerProtocolEvidence::SchedulerProtocolEvidence(
    CanonicalIdentity identityValue, int requiredGenerationValue,
    std::string cutoverStateValue, UtcTimestamp cutoverCompletedAtValue,
    std::string cutoverCompletedByValue,
    std::string cutoverExecutablePathValue,
    std::string cutoverProcessEvidenceValue)
    : identity(std::move(identityValue)),
      requiredGeneration(requiredGenerationValue),
      cutoverState(std::move(cutoverStateValue)),
      cutoverCompletedAt(std::move(cutoverCompletedAtValue)),
      cutoverCompletedBy(std::move(cutoverCompletedByValue)),
      cutoverExecutablePath(std::move(cutoverExecutablePathValue)),
      cutoverProcessEvidence(std::move(cutoverProcessEvidenceValue))
{
}

SchedulerProtocolEvidence BuildSchedulerProtocolEvidence(
    int requiredGeneration, std::string cutoverState,
    UtcTimestamp cutoverCompletedAt, std::string cutoverCompletedBy,
    std::string cutoverExecutablePath,
    std::string cutoverProcessEvidence)
{
    RequireText(cutoverCompletedBy,
        "campaign_operations_scheduler_cutover_actor_invalid");
    RequireText(cutoverExecutablePath,
        "campaign_operations_scheduler_cutover_path_invalid");
    RequireText(cutoverProcessEvidence,
        "campaign_operations_scheduler_cutover_process_invalid");
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_scheduler_protocol_evidence_v1"
              << ";required_generation=" << requiredGeneration
              << ";cutover_state=" << cutoverState
              << ";migration_contract=61:migration-052-scheduler-generation-52-exact-attempt-authority"
              << ";protocol_contract=50:scheduler-generation-52-exact-attempt-authority-v1"
              << ";cutover_completed_at="
              << Framed(cutoverCompletedAt.value())
              << ";cutover_completed_by=" << Framed(cutoverCompletedBy)
              << ";cutover_executable_path="
              << Framed(cutoverExecutablePath)
              << ";cutover_process_evidence="
              << Framed(cutoverProcessEvidence);
    SchedulerProtocolEvidence result(Identity(canonical.str()),
        requiredGeneration, std::move(cutoverState),
        std::move(cutoverCompletedAt), std::move(cutoverCompletedBy),
        std::move(cutoverExecutablePath),
        std::move(cutoverProcessEvidence));
    ValidateSchedulerProtocolEvidence(result);
    return result;
}

void ValidateSchedulerProtocolEvidence(const SchedulerProtocolEvidence& value)
{
    if (value.requiredGeneration != kRequiredSchedulerProtocolGeneration ||
        value.cutoverState != "complete")
        throw Error(ErrorCode::unsupportedContractVersion,
            "campaign_operations_scheduler_protocol_generation_invalid");
}

ManagerBuildContract::ManagerBuildContract(CanonicalIdentity identityValue,
    std::string managerServiceContractValue, std::string sourceCommitValue,
    std::string compilerContractValue, std::string executableSha256Value)
    : identity(std::move(identityValue)),
      managerServiceContract(std::move(managerServiceContractValue)),
      sourceCommit(std::move(sourceCommitValue)),
      compilerContract(std::move(compilerContractValue)),
      executableSha256(std::move(executableSha256Value))
{
}

ManagerBuildContract BuildManagerBuildContract(
    std::string managerServiceContract, std::string sourceCommit,
    std::string compilerContract, std::string executableSha256)
{
    RequireText(managerServiceContract,
        "campaign_operations_manager_service_contract_invalid");
    RequireText(compilerContract,
        "campaign_operations_manager_compiler_contract_invalid");
    if (sourceCommit.size() != 40U ||
        !std::all_of(sourceCommit.begin(), sourceCommit.end(),
            [](unsigned char c)
            {
                return (c >= '0' && c <= '9') ||
                    (c >= 'a' && c <= 'f');
            }))
        throw Error(ErrorCode::invalidCanonicalHash,
            "campaign_operations_manager_source_commit_invalid");
    RequireSha256(executableSha256);
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_manager_build_v1"
              << ";manager_service_contract="
              << Framed(managerServiceContract)
              << ";source_commit=" << sourceCommit
              << ";source_tree_state=clean"
              << ";build_configuration=Release"
              << ";compiler_contract=" << Framed(compilerContract)
              << ";executable_sha256=" << executableSha256
              << ";build_contract_version=1";
    ManagerBuildContract result(Identity(canonical.str()),
        std::move(managerServiceContract), std::move(sourceCommit),
        std::move(compilerContract), std::move(executableSha256));
    ValidateManagerBuildContract(result);
    return result;
}

void ValidateManagerBuildContract(const ManagerBuildContract& value)
{
    if (value.managerServiceContract != kManagerServiceContract)
        throw Error(ErrorCode::unsupportedContractVersion,
            "campaign_operations_manager_service_contract_unsupported");
}

std::string ToText(ProductionEnablementEventKind value)
{
    switch (value)
    {
        case ProductionEnablementEventKind::enable: return "enable";
        case ProductionEnablementEventKind::disable: return "disable";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_enablement_kind_invalid");
}

ProductionEnablementEventKind ProductionEnablementEventKindFromText(
    const std::string& value)
{
    if (value == "enable") return ProductionEnablementEventKind::enable;
    if (value == "disable") return ProductionEnablementEventKind::disable;
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_enablement_kind_invalid");
}

ProductionEnableEvent::ProductionEnableEvent(CanonicalIdentity identityValue,
    std::string operationKeyValue,
    std::optional<ProductionEnablementEventId> predecessorEventIdValue,
    std::string predecessorEventCanonicalValue, int expectedPriorVersionValue,
    int resultingVersionValue, SchedulerProtocolEvidence schedulerValue,
    std::string verificationValue, ActorIdentity actorValue,
    std::string serviceContractValue, ManagerBuildContract buildValue,
    Reason reasonValue)
    : identity(std::move(identityValue)),
      operationKey(std::move(operationKeyValue)),
      predecessorEventId(std::move(predecessorEventIdValue)),
      predecessorEventCanonical(std::move(predecessorEventCanonicalValue)),
      expectedPriorVersion(expectedPriorVersionValue),
      resultingVersion(resultingVersionValue),
      schedulerProtocolEvidence(std::move(schedulerValue)),
      independentVerificationReference(std::move(verificationValue)),
      authorizingActor(std::move(actorValue)),
      managerServiceContract(std::move(serviceContractValue)),
      approvedBuildContract(std::move(buildValue)),
      reason(std::move(reasonValue))
{
}

ProductionEnableEvent BuildProductionEnableEvent(std::string operationKey,
    std::optional<ProductionEnablementEventId> predecessorEventId,
    std::string predecessorEventCanonical, int expectedPriorVersion,
    int resultingVersion, SchedulerProtocolEvidence schedulerEvidence,
    std::string independentVerificationReference,
    ActorIdentity authorizingActor, std::string managerServiceContract,
    ManagerBuildContract approvedBuildContract, Reason reason)
{
    RequireOperationKey(operationKey);
    RequireText(predecessorEventCanonical,
        "campaign_operations_enable_predecessor_invalid", true);
    RequireText(independentVerificationReference,
        "campaign_operations_verification_reference_invalid");
    RequireText(managerServiceContract,
        "campaign_operations_manager_service_contract_invalid");
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_production_enable_event_v1"
              << ";operation_key=" << Framed(operationKey)
              << ";predecessor_event_id=" << OptionalId(predecessorEventId)
              << ";predecessor_event_canonical="
              << Framed(predecessorEventCanonical)
              << ";expected_prior_version=" << expectedPriorVersion
              << ";resulting_version=" << resultingVersion
              << ";scheduler_protocol_evidence="
              << Framed(schedulerEvidence.identity.canonicalText())
              << ";independent_verification_reference="
              << Framed(independentVerificationReference)
              << ";authorizing_actor=" << Framed(authorizingActor.value())
              << ";capability=" << kProductionEnablerRole
              << ";manager_service_contract="
              << Framed(managerServiceContract)
              << ";approved_build_contract="
              << Framed(approvedBuildContract.identity.canonicalText())
              << ";reason=" << Framed(reason.value())
              << ";enablement_contract_version=1";
    ProductionEnableEvent result(Identity(canonical.str()),
        std::move(operationKey), std::move(predecessorEventId),
        std::move(predecessorEventCanonical), expectedPriorVersion,
        resultingVersion, std::move(schedulerEvidence),
        std::move(independentVerificationReference),
        std::move(authorizingActor), std::move(managerServiceContract),
        std::move(approvedBuildContract), std::move(reason));
    ValidateProductionEnableEvent(result);
    return result;
}

void ValidateProductionEnableEvent(const ProductionEnableEvent& value)
{
    const bool genesis = !value.predecessorEventId;
    if (value.resultingVersion != value.expectedPriorVersion + 1 ||
        (genesis && (value.expectedPriorVersion != 0 ||
            !value.predecessorEventCanonical.empty())) ||
        (!genesis && (value.expectedPriorVersion <= 0 ||
            value.predecessorEventCanonical.empty())) ||
        value.managerServiceContract != kManagerServiceContract ||
        value.approvedBuildContract.managerServiceContract !=
            value.managerServiceContract)
        throw Error(ErrorCode::invalidCanonicalText,
            "campaign_operations_enable_event_invalid");
}

ProductionDisableEvent::ProductionDisableEvent(
    CanonicalIdentity identityValue, std::string operationKeyValue,
    ProductionEnablementEventId predecessorEventIdValue,
    std::string predecessorEventCanonicalValue, int expectedPriorVersionValue,
    int resultingVersionValue, ActorIdentity actorValue, Reason reasonValue)
    : identity(std::move(identityValue)),
      operationKey(std::move(operationKeyValue)),
      predecessorEventId(predecessorEventIdValue),
      predecessorEventCanonical(std::move(predecessorEventCanonicalValue)),
      expectedPriorVersion(expectedPriorVersionValue),
      resultingVersion(resultingVersionValue),
      disablingActor(std::move(actorValue)), reason(std::move(reasonValue))
{
}

ProductionDisableEvent BuildProductionDisableEvent(std::string operationKey,
    ProductionEnablementEventId predecessorEventId,
    std::string predecessorEventCanonical, int expectedPriorVersion,
    int resultingVersion, ActorIdentity disablingActor, Reason reason)
{
    RequireOperationKey(operationKey);
    RequireText(predecessorEventCanonical,
        "campaign_operations_disable_predecessor_invalid");
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_production_disable_event_v1"
              << ";operation_key=" << Framed(operationKey)
              << ";predecessor_event_id=" << predecessorEventId.value()
              << ";predecessor_event_canonical="
              << Framed(predecessorEventCanonical)
              << ";expected_prior_version=" << expectedPriorVersion
              << ";resulting_version=" << resultingVersion
              << ";disabling_actor=" << Framed(disablingActor.value())
              << ";capability=" << kProductionDisablerRole
              << ";reason=" << Framed(reason.value())
              << ";enablement_contract_version=1";
    ProductionDisableEvent result(Identity(canonical.str()),
        std::move(operationKey), predecessorEventId,
        std::move(predecessorEventCanonical), expectedPriorVersion,
        resultingVersion, std::move(disablingActor), std::move(reason));
    ValidateProductionDisableEvent(result);
    return result;
}

void ValidateProductionDisableEvent(const ProductionDisableEvent& value)
{
    if (value.expectedPriorVersion <= 0 ||
        value.resultingVersion != value.expectedPriorVersion + 1)
        throw Error(ErrorCode::invalidCanonicalText,
            "campaign_operations_disable_event_invalid");
}

RequestProductionAdmission::RequestProductionAdmission(
    CanonicalIdentity identityValue, OperationalRequestId requestIdValue,
    std::string requestCanonicalValue, int expectedVersionValue,
    std::string operationKeyValue,
    ProductionEnablementEventId enableEventIdValue,
    std::string enableCanonicalValue, ActorIdentity actorValue,
    std::string principalValue, ManagerBuildContract buildValue)
    : identity(std::move(identityValue)), requestId(requestIdValue),
      requestIdentityCanonical(std::move(requestCanonicalValue)),
      expectedRequestVersion(expectedVersionValue),
      dispatchOperationKey(std::move(operationKeyValue)),
      enableEventId(enableEventIdValue),
      enableEventCanonical(std::move(enableCanonicalValue)),
      requestingActor(std::move(actorValue)),
      originalExecutingServicePrincipal(std::move(principalValue)),
      approvedBuildContract(std::move(buildValue))
{
}

RequestProductionAdmission BuildRequestProductionAdmission(
    OperationalRequestId requestId, std::string requestIdentityCanonical,
    int expectedRequestVersion, std::string dispatchOperationKey,
    ProductionEnablementEventId enableEventId,
    std::string enableEventCanonical, ActorIdentity requestingActor,
    std::string originalExecutingServicePrincipal,
    ManagerBuildContract approvedBuildContract)
{
    RequireText(requestIdentityCanonical,
        "campaign_operations_admission_request_canonical_invalid");
    RequireText(enableEventCanonical,
        "campaign_operations_admission_enable_canonical_invalid");
    RequireOperationKey(dispatchOperationKey);
    RequirePrincipal(originalExecutingServicePrincipal);
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_request_production_admission_v1"
              << ";operational_request_id=" << requestId.value()
              << ";request_identity_canonical="
              << Framed(requestIdentityCanonical)
              << ";expected_request_version=" << expectedRequestVersion
              << ";dispatch_operation_key=" << Framed(dispatchOperationKey)
              << ";enable_event_id=" << enableEventId.value()
              << ";enable_event_canonical=" << Framed(enableEventCanonical)
              << ";requesting_actor=" << Framed(requestingActor.value())
              << ";original_executing_service_principal="
              << Framed(originalExecutingServicePrincipal)
              << ";approved_build_contract="
              << Framed(approvedBuildContract.identity.canonicalText())
              << ";capability=" << kProductionDispatcherRole
              << ";admission_contract_version=1";
    RequestProductionAdmission result(Identity(canonical.str()), requestId,
        std::move(requestIdentityCanonical), expectedRequestVersion,
        std::move(dispatchOperationKey), enableEventId,
        std::move(enableEventCanonical), std::move(requestingActor),
        std::move(originalExecutingServicePrincipal),
        std::move(approvedBuildContract));
    ValidateRequestProductionAdmission(result);
    return result;
}

void ValidateRequestProductionAdmission(
    const RequestProductionAdmission& value)
{
    if (value.expectedRequestVersion <= 0)
        throw Error(ErrorCode::invalidCanonicalText,
            "campaign_operations_request_admission_invalid");
}

ProductionDispatchAttemptV2::ProductionDispatchAttemptV2(
    CanonicalIdentity identityValue, OperationalRequestId requestIdValue,
    std::string requestCanonicalValue,
    RequestProductionAdmission admissionValue,
    ProductionEnablementEventId enableEventIdValue,
    std::string enableCanonicalValue, std::string operationKeyValue,
    int ordinalValue, int expectedVersionValue, int resultingVersionValue,
    LeaseTokenDigest digestValue, UtcTimestamp expiresValue,
    ActorIdentity actorValue, std::string principalValue,
    ManagerBuildContract buildValue)
    : identity(std::move(identityValue)), requestId(requestIdValue),
      requestIdentityCanonical(std::move(requestCanonicalValue)),
      admission(std::move(admissionValue)), enableEventId(enableEventIdValue),
      enableEventCanonical(std::move(enableCanonicalValue)),
      operationKey(std::move(operationKeyValue)), attemptOrdinal(ordinalValue),
      expectedRequestVersion(expectedVersionValue),
      resultingRequestVersion(resultingVersionValue),
      leaseTokenDigest(std::move(digestValue)),
      leaseExpiresAt(std::move(expiresValue)),
      requestingActor(std::move(actorValue)),
      originalExecutingServicePrincipal(std::move(principalValue)),
      approvedBuildContract(std::move(buildValue))
{
}

ProductionDispatchAttemptV2 BuildProductionDispatchAttemptV2(
    OperationalRequestId requestId, std::string requestIdentityCanonical,
    RequestProductionAdmission admission,
    ProductionEnablementEventId enableEventId,
    std::string enableEventCanonical, std::string operationKey,
    int attemptOrdinal, int expectedRequestVersion,
    int resultingRequestVersion, LeaseTokenDigest leaseTokenDigest,
    UtcTimestamp leaseExpiresAt, ActorIdentity requestingActor,
    std::string originalExecutingServicePrincipal,
    ManagerBuildContract approvedBuildContract)
{
    RequireText(requestIdentityCanonical,
        "campaign_operations_attempt_v2_request_invalid");
    RequireText(enableEventCanonical,
        "campaign_operations_attempt_v2_enable_invalid");
    RequireOperationKey(operationKey);
    RequirePrincipal(originalExecutingServicePrincipal);
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_dispatch_attempt_v2"
              << ";operational_request_id=" << requestId.value()
              << ";request_identity_canonical="
              << Framed(requestIdentityCanonical)
              << ";request_production_admission_canonical="
              << Framed(admission.identity.canonicalText())
              << ";enable_event_id=" << enableEventId.value()
              << ";enable_event_canonical=" << Framed(enableEventCanonical)
              << ";operation_key=" << Framed(operationKey)
              << ";attempt_ordinal=" << attemptOrdinal
              << ";expected_request_version=" << expectedRequestVersion
              << ";resulting_request_version=" << resultingRequestVersion
              << ";lease_token_digest=" << Framed(leaseTokenDigest.value())
              << ";lease_expires_at=" << Framed(leaseExpiresAt.value())
              << ";requesting_actor=" << Framed(requestingActor.value())
              << ";original_executing_service_principal="
              << Framed(originalExecutingServicePrincipal)
              << ";approved_build_contract="
              << Framed(approvedBuildContract.identity.canonicalText())
              << ";capability=" << kProductionDispatcherRole
              << ";attempt_contract_version=2";
    ProductionDispatchAttemptV2 result(Identity(canonical.str()), requestId,
        std::move(requestIdentityCanonical), std::move(admission),
        enableEventId, std::move(enableEventCanonical),
        std::move(operationKey), attemptOrdinal, expectedRequestVersion,
        resultingRequestVersion, std::move(leaseTokenDigest),
        std::move(leaseExpiresAt), std::move(requestingActor),
        std::move(originalExecutingServicePrincipal),
        std::move(approvedBuildContract));
    ValidateProductionDispatchAttemptV2(result);
    return result;
}

void ValidateProductionDispatchAttemptV2(
    const ProductionDispatchAttemptV2& value)
{
    if (value.attemptOrdinal <= 0 || value.expectedRequestVersion <= 0 ||
        value.resultingRequestVersion != value.expectedRequestVersion + 1 ||
        value.requestId != value.admission.requestId ||
        value.requestIdentityCanonical !=
            value.admission.requestIdentityCanonical ||
        value.admission.expectedRequestVersion <= 0)
        throw Error(ErrorCode::invalidCanonicalText,
            "campaign_operations_dispatch_attempt_v2_invalid");
}

} // namespace EA::CampaignOperations
