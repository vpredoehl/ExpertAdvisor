#pragma once

#include "CampaignOperations.hpp"
#include "CampaignOperationsDispatch.hpp"

#include <optional>
#include <string>

namespace EA::CampaignOperations
{

inline constexpr int kProductionEnablementContractVersion = 1;
inline constexpr int kProductionAdmissionContractVersion = 1;
inline constexpr int kProductionAttemptContractVersion = 2;
inline constexpr int kSchedulerProtocolEvidenceContractVersion = 1;
inline constexpr int kManagerBuildContractVersion = 1;
inline constexpr int kRequiredSchedulerProtocolGeneration = 52;
inline constexpr char kProductionAdmissionMigrationFilename[] =
    "055_campaign_operations_production_admission_foundation.sql";
inline constexpr char kProductionAdmissionMigrationChecksum[] =
    "86a35844edd3cc233e8f72ff985c339474dc09d3cd79d354fcb3adeb902aa66f";
inline constexpr char kProductionEnablerRole[] =
    "campaign_operations_production_enabler";
inline constexpr char kProductionDisablerRole[] =
    "campaign_operations_production_disabler";
inline constexpr char kProductionDispatcherRole[] =
    "campaign_operations_production_dispatcher";
inline constexpr char kProductionPhase5TransactionalRole[] =
    "campaign_operations_production_phase5_transactional";
inline constexpr char kProductionReaderRole[] =
    "campaign_operations_production_reader";
inline constexpr char kSchedulerProtocolEvidenceReaderRole[] =
    "campaign_operations_scheduler_protocol_evidence_reader";
inline constexpr char kManagerServiceContract[] =
    "campaign-operations-production-dispatch-and-manager-run-once-v1";

struct SchedulerProtocolEvidence final
{
    const CanonicalIdentity identity;
    const int requiredGeneration;
    const std::string cutoverState;
    const UtcTimestamp cutoverCompletedAt;
    const std::string cutoverCompletedBy;
    const std::string cutoverExecutablePath;
    const std::string cutoverProcessEvidence;

    SchedulerProtocolEvidence(const SchedulerProtocolEvidence&) = default;
    SchedulerProtocolEvidence(SchedulerProtocolEvidence&&) = default;
    SchedulerProtocolEvidence& operator=(
        const SchedulerProtocolEvidence&) = delete;
    bool operator==(const SchedulerProtocolEvidence&) const = default;

private:
    SchedulerProtocolEvidence(CanonicalIdentity, int, std::string,
        UtcTimestamp, std::string, std::string, std::string);
    friend SchedulerProtocolEvidence BuildSchedulerProtocolEvidence(
        int, std::string, UtcTimestamp, std::string, std::string,
        std::string);
};

SchedulerProtocolEvidence BuildSchedulerProtocolEvidence(
    int requiredGeneration, std::string cutoverState,
    UtcTimestamp cutoverCompletedAt, std::string cutoverCompletedBy,
    std::string cutoverExecutablePath,
    std::string cutoverProcessEvidence);
void ValidateSchedulerProtocolEvidence(const SchedulerProtocolEvidence&);

struct ManagerBuildContract final
{
    const CanonicalIdentity identity;
    const std::string managerServiceContract;
    const std::string sourceCommit;
    const std::string compilerContract;
    const std::string executableSha256;

    ManagerBuildContract(const ManagerBuildContract&) = default;
    ManagerBuildContract(ManagerBuildContract&&) = default;
    ManagerBuildContract& operator=(const ManagerBuildContract&) = delete;
    bool operator==(const ManagerBuildContract&) const = default;

private:
    ManagerBuildContract(CanonicalIdentity, std::string, std::string,
        std::string, std::string);
    friend ManagerBuildContract BuildManagerBuildContract(std::string,
        std::string, std::string, std::string);
};

ManagerBuildContract BuildManagerBuildContract(
    std::string managerServiceContract, std::string sourceCommit,
    std::string compilerContract, std::string executableSha256);
void ValidateManagerBuildContract(const ManagerBuildContract&);

enum class ProductionEnablementEventKind { enable, disable };
std::string ToText(ProductionEnablementEventKind);
ProductionEnablementEventKind ProductionEnablementEventKindFromText(
    const std::string&);

struct ProductionEnableEvent final
{
    const CanonicalIdentity identity;
    const std::string operationKey;
    const std::optional<ProductionEnablementEventId> predecessorEventId;
    const std::string predecessorEventCanonical;
    const int expectedPriorVersion;
    const int resultingVersion;
    const SchedulerProtocolEvidence schedulerProtocolEvidence;
    const std::string independentVerificationReference;
    const ActorIdentity authorizingActor;
    const std::string managerServiceContract;
    const ManagerBuildContract approvedBuildContract;
    const Reason reason;

    ProductionEnableEvent(const ProductionEnableEvent&) = default;
    ProductionEnableEvent(ProductionEnableEvent&&) = default;
    ProductionEnableEvent& operator=(const ProductionEnableEvent&) = delete;
    bool operator==(const ProductionEnableEvent&) const = default;

private:
    ProductionEnableEvent(CanonicalIdentity, std::string,
        std::optional<ProductionEnablementEventId>, std::string, int, int,
        SchedulerProtocolEvidence, std::string, ActorIdentity, std::string,
        ManagerBuildContract, Reason);
    friend ProductionEnableEvent BuildProductionEnableEvent(std::string,
        std::optional<ProductionEnablementEventId>, std::string, int, int,
        SchedulerProtocolEvidence, std::string, ActorIdentity, std::string,
        ManagerBuildContract, Reason);
};

ProductionEnableEvent BuildProductionEnableEvent(std::string operationKey,
    std::optional<ProductionEnablementEventId> predecessorEventId,
    std::string predecessorEventCanonical, int expectedPriorVersion,
    int resultingVersion, SchedulerProtocolEvidence schedulerEvidence,
    std::string independentVerificationReference,
    ActorIdentity authorizingActor, std::string managerServiceContract,
    ManagerBuildContract approvedBuildContract, Reason reason);
void ValidateProductionEnableEvent(const ProductionEnableEvent&);

struct ProductionDisableEvent final
{
    const CanonicalIdentity identity;
    const std::string operationKey;
    const ProductionEnablementEventId predecessorEventId;
    const std::string predecessorEventCanonical;
    const int expectedPriorVersion;
    const int resultingVersion;
    const ActorIdentity disablingActor;
    const Reason reason;

    ProductionDisableEvent(const ProductionDisableEvent&) = default;
    ProductionDisableEvent(ProductionDisableEvent&&) = default;
    ProductionDisableEvent& operator=(const ProductionDisableEvent&) = delete;
    bool operator==(const ProductionDisableEvent&) const = default;

private:
    ProductionDisableEvent(CanonicalIdentity, std::string,
        ProductionEnablementEventId, std::string, int, int, ActorIdentity,
        Reason);
    friend ProductionDisableEvent BuildProductionDisableEvent(std::string,
        ProductionEnablementEventId, std::string, int, int, ActorIdentity,
        Reason);
};

ProductionDisableEvent BuildProductionDisableEvent(std::string operationKey,
    ProductionEnablementEventId predecessorEventId,
    std::string predecessorEventCanonical, int expectedPriorVersion,
    int resultingVersion, ActorIdentity disablingActor, Reason reason);
void ValidateProductionDisableEvent(const ProductionDisableEvent&);

struct RequestProductionAdmission final
{
    const CanonicalIdentity identity;
    const OperationalRequestId requestId;
    const std::string requestIdentityCanonical;
    const int expectedRequestVersion;
    const std::string dispatchOperationKey;
    const ProductionEnablementEventId enableEventId;
    const std::string enableEventCanonical;
    const ActorIdentity requestingActor;
    const std::string originalExecutingServicePrincipal;
    const ManagerBuildContract approvedBuildContract;

    RequestProductionAdmission(const RequestProductionAdmission&) = default;
    RequestProductionAdmission(RequestProductionAdmission&&) = default;
    RequestProductionAdmission& operator=(
        const RequestProductionAdmission&) = delete;
    bool operator==(const RequestProductionAdmission&) const = default;

private:
    RequestProductionAdmission(CanonicalIdentity, OperationalRequestId,
        std::string, int, std::string, ProductionEnablementEventId,
        std::string, ActorIdentity, std::string, ManagerBuildContract);
    friend RequestProductionAdmission BuildRequestProductionAdmission(
        OperationalRequestId, std::string, int, std::string,
        ProductionEnablementEventId, std::string, ActorIdentity, std::string,
        ManagerBuildContract);
};

RequestProductionAdmission BuildRequestProductionAdmission(
    OperationalRequestId requestId, std::string requestIdentityCanonical,
    int expectedRequestVersion, std::string dispatchOperationKey,
    ProductionEnablementEventId enableEventId,
    std::string enableEventCanonical, ActorIdentity requestingActor,
    std::string originalExecutingServicePrincipal,
    ManagerBuildContract approvedBuildContract);
void ValidateRequestProductionAdmission(const RequestProductionAdmission&);

struct ProductionDispatchAttemptV2 final
{
    const CanonicalIdentity identity;
    const OperationalRequestId requestId;
    const std::string requestIdentityCanonical;
    const RequestProductionAdmission admission;
    const ProductionEnablementEventId enableEventId;
    const std::string enableEventCanonical;
    const std::string operationKey;
    const int attemptOrdinal;
    const int expectedRequestVersion;
    const int resultingRequestVersion;
    const LeaseTokenDigest leaseTokenDigest;
    const UtcTimestamp leaseExpiresAt;
    const ActorIdentity requestingActor;
    const std::string originalExecutingServicePrincipal;
    const ManagerBuildContract approvedBuildContract;

    ProductionDispatchAttemptV2(const ProductionDispatchAttemptV2&) = default;
    ProductionDispatchAttemptV2(ProductionDispatchAttemptV2&&) = default;
    ProductionDispatchAttemptV2& operator=(
        const ProductionDispatchAttemptV2&) = delete;
    bool operator==(const ProductionDispatchAttemptV2&) const = default;

private:
    ProductionDispatchAttemptV2(CanonicalIdentity, OperationalRequestId,
        std::string, RequestProductionAdmission, ProductionEnablementEventId,
        std::string, std::string, int, int, int, LeaseTokenDigest,
        UtcTimestamp, ActorIdentity, std::string, ManagerBuildContract);
    friend ProductionDispatchAttemptV2 BuildProductionDispatchAttemptV2(
        OperationalRequestId, std::string, RequestProductionAdmission,
        ProductionEnablementEventId, std::string, std::string, int, int, int,
        LeaseTokenDigest, UtcTimestamp, ActorIdentity, std::string,
        ManagerBuildContract);
};

ProductionDispatchAttemptV2 BuildProductionDispatchAttemptV2(
    OperationalRequestId requestId, std::string requestIdentityCanonical,
    RequestProductionAdmission admission,
    ProductionEnablementEventId enableEventId,
    std::string enableEventCanonical, std::string operationKey,
    int attemptOrdinal, int expectedRequestVersion,
    int resultingRequestVersion, LeaseTokenDigest leaseTokenDigest,
    UtcTimestamp leaseExpiresAt, ActorIdentity requestingActor,
    std::string originalExecutingServicePrincipal,
    ManagerBuildContract approvedBuildContract);
void ValidateProductionDispatchAttemptV2(
    const ProductionDispatchAttemptV2&);

} // namespace EA::CampaignOperations
