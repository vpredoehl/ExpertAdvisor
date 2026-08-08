#pragma once

#include "CampaignOperationsProductionAdmission.hpp"

#include <pqxx/pqxx>

#include <optional>
#include <string>
#include <vector>

namespace EA::CampaignOperations
{

struct ProductionEnablementAuditEvidence final
{
    long long auditId;
    ProductionEnablementEventId eventId;
    std::string operationKey;
    ProductionEnablementEventKind kind;
    ActorIdentity actor;
    std::string capability;
    Reason reason;
    std::string outcome;
    std::string replayDisposition;
    std::string diagnosticCode;
};

struct ProductionDispatchAuditEvidence final
{
    long long auditId;
    OperationalCampaignId campaignId;
    OperationalRequestId requestId;
    DispatchAttemptId attemptId;
    std::string causeKind;
    ActorIdentity actor;
    std::string capability;
    int priorVersion;
    int resultingVersion;
    std::string outcome;
    std::string replayDisposition;
    std::string diagnosticCode;
    RequestProductionAdmissionId admissionId;
    ProductionEnablementEventId enablementEventId;
};

struct PersistedProductionEnablementHead final
{
    ProductionEnablementEventId eventId;
    ProductionEnablementEventKind kind;
    CanonicalIdentity identity;
    int resultingVersion;
    std::optional<SchedulerProtocolEvidence> schedulerEvidence;
    std::optional<ManagerBuildContract> approvedBuildContract;
    ProductionEnablementAuditEvidence auditEvidence;
};

struct PersistedProductionDispatchAttemptV2Evidence final
{
    DispatchAttemptId attemptId;
    ProductionDispatchAttemptV2 attempt;
    PersistedProductionEnablementHead authorizingEnablement;
    ProductionDispatchAuditEvidence acquisitionAudit;
};

struct PersistedRequestProductionAdmission final
{
    RequestProductionAdmissionId admissionId;
    RequestProductionAdmission admission;
    PersistedProductionEnablementHead authorizingEnablement;
    PersistedProductionDispatchAttemptV2Evidence firstAttempt;
};

struct PersistedProductionDispatchAttemptV2 final
{
    DispatchAttemptId attemptId;
    ProductionDispatchAttemptV2 attempt;
    PersistedRequestProductionAdmission admissionEvidence;
    PersistedProductionEnablementHead authorizingEnablement;
    ProductionDispatchAuditEvidence acquisitionAudit;
};

// This is deliberately a structural, observed representation.  It is used
// by readiness before the observed contract versions are compared with the
// deployment's normative versions, so a readable unsupported version never
// erases trustworthy persisted evidence from the diagnostic output.
struct ObservedProductionEnablementEvidence final
{
    ProductionEnablementEventId eventId;
    ProductionEnablementEventKind kind;
    CanonicalIdentity identity;
    int resultingVersion;
    int enablementContractVersion;
    std::optional<std::string> managerServiceContract;
    std::optional<CanonicalIdentity> approvedBuildIdentity;
    std::optional<int> managerBuildContractVersion;
};

struct ProductionReadinessSnapshot final
{
    std::string migrationVersion;
    std::optional<std::string> migrationFilename;
    std::optional<std::string> migrationChecksum;
    // These are observed evidence values, never deployment constants.  A
    // multi-value string (for example "1|2") is readable contradictory
    // evidence; an absent value is distinct from zero and from "invalid".
    std::optional<std::string> schedulerEvidenceContractVersion;
    std::optional<std::string> managerBuildContractVersion;
    std::optional<std::string> enablementContractVersion;
    std::optional<std::string> admissionContractVersion;
    std::optional<std::string> productionAttemptContractVersion;
    std::optional<int> schedulerGeneration;
    std::optional<std::string> schedulerCutoverState;
    bool schedulerEvidenceComplete = false;
    std::optional<SchedulerProtocolEvidence> schedulerEvidence;
    std::optional<PersistedProductionEnablementHead> enablementHead;
    std::optional<ObservedProductionEnablementEvidence>
        observedEnablementHead;
    std::optional<std::string> independentVerificationReference;
    std::string sessionPrincipal;
    std::string currentPrincipal;
    bool readerMember = false;
    bool dispatcherMember = false;
    bool phase5TransactionalMember = false;
    bool schedulerEvidenceReaderMember = false;
    bool enablerMember = false;
    bool disablerMember = false;
    bool prohibitedTestDispatcherMember = false;
    bool prohibitedTestPhase5Member = false;
    bool enablementEffective = false;
    long long readyAdmittedRequestCount = 0;
    long long readyUnadmittedRequestCount = 0;
    long long activeCurrentEventLeaseCount = 0;
    long long oldEventBlockedLeaseCount = 0;
    long long reconciliationRequiredCount = 0;
    std::optional<std::string> completionNestedV2ProofVersion;
    bool completionNestedV2ProofValid = false;
};

struct ProductionRequestStatus final
{
    OperationalRequestId requestId;
    OperationalCampaignId campaignId;
    std::string requestState;
    int stateVersion = 0;
    bool productionDispatchEnabled = false;
    std::optional<RequestProductionAdmissionId> admissionId;
    std::optional<DispatchAttemptId> attemptId;
    std::optional<std::string> operationKey;
    std::optional<ProductionEnablementEventId> enablementEventId;
    std::optional<std::string> leaseExpiresAt;
    bool leaseExpired = false;
    bool phaseFRecoveryEligible = false;
    bool reconciliationRequired = false;
};

bool ProductionAdmissionSchemaExists(pqxx::transaction_base&);
std::optional<SchedulerProtocolEvidence>
LoadSchedulerProtocolEvidenceSnapshot(pqxx::transaction_base&);
SchedulerProtocolEvidence LockSchedulerProtocolEvidence(
    pqxx::transaction_base&);
std::optional<PersistedProductionEnablementHead>
FindCurrentProductionEnablementHead(pqxx::transaction_base&);
std::optional<PersistedProductionEnablementHead>
FindProductionEnablementEventByOperationKey(
    pqxx::transaction_base&, const std::string& operationKey);
std::optional<PersistedRequestProductionAdmission>
FindRequestProductionAdmission(pqxx::transaction_base&, OperationalRequestId);
std::optional<PersistedProductionDispatchAttemptV2>
FindProductionDispatchAttemptV2(pqxx::transaction_base&, DispatchAttemptId);
std::optional<PersistedProductionDispatchAttemptV2>
FindProductionDispatchAttemptV2(pqxx::transaction_base&,
    OperationalRequestId, const std::string& operationKey);
ProductionReadinessSnapshot LoadProductionReadinessSnapshot(
    pqxx::transaction_base&);
std::vector<ProductionRequestStatus> LoadProductionStatusSnapshot(
    pqxx::transaction_base&);

} // namespace EA::CampaignOperations
