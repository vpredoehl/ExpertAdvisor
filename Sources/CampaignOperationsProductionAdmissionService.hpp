#pragma once

#include "CampaignOperationsProductionAdmissionRepository.hpp"

#include <iosfwd>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace EA::CampaignOperations
{

struct ProductionReadinessEvaluation final
{
    ProductionReadinessSnapshot snapshot;
    std::optional<ManagerBuildContract> actualBuildContract;
    bool ready = false;
    std::vector<std::string> blockers;
};

ProductionReadinessEvaluation EvaluateProductionReadiness(
    ProductionReadinessSnapshot,
    std::optional<ManagerBuildContract> actualBuildContract = std::nullopt);
ProductionReadinessEvaluation LoadProductionReadiness(
    pqxx::connection&,
    std::optional<ManagerBuildContract> actualBuildContract = std::nullopt);
std::string RenderProductionReadiness(
    const ProductionReadinessEvaluation&);
std::optional<ManagerBuildContract> CaptureActualManagerBuildContract(
    const std::string& executablePath);
std::vector<ProductionRequestStatus> LoadProductionStatus(pqxx::connection&);

int RunProductionReadinessCommand(const std::string& connectionString,
    std::ostream& output, std::ostream& errors,
    std::optional<ManagerBuildContract> actualBuildContract = std::nullopt);
int RunProductionStatusCommand(const std::string& connectionString,
    std::ostream& output, std::ostream& errors);

enum class ProductionMutationDisposition
{
    newOperation,
    exactReplay,
    conflictingReplay,
    ambiguous,
    rejected
};

enum class ProductionMutationInjectionPoint
{
    beforeCommit,
#if defined(CAMPAIGN_OPERATIONS_H2_TESTING)
    afterTransitionBeforeCommit,
#endif
    afterCommitBeforeResponse
};

using ProductionMutationConnectionFactory =
    std::function<std::unique_ptr<pqxx::connection>()>;
using ProductionMutationTestHook =
    std::function<void(ProductionMutationInjectionPoint)>;

struct ProductionMutationResult final
{
    ProductionMutationDisposition disposition =
        ProductionMutationDisposition::rejected;
    std::optional<ProductionEnablementEventId> eventId;
    int resultingVersion = 0;
    UncertainCommitRecoveryClassification recovery =
        UncertainCommitRecoveryClassification::provenNoCommit;
    std::string diagnosticCode;
};

struct ProductionEnableRequest final
{
    std::string operationKey;
    int expectedPriorVersion = 0;
    std::string independentVerificationReference;
    ActorIdentity authorizingActor;
    Reason reason;
    ManagerBuildContract approvedBuildContract;
    bool acknowledged = false;
};

struct ProductionDisableRequest final
{
    std::string operationKey;
    int expectedPriorVersion = 0;
    ActorIdentity disablingActor;
    Reason reason;
    bool acknowledged = false;
};

ProductionMutationResult EnableProduction(
    const ProductionMutationConnectionFactory&, const ProductionEnableRequest&,
    ProductionMutationTestHook testHook = {});
ProductionMutationResult DisableProduction(
    const ProductionMutationConnectionFactory&, const ProductionDisableRequest&,
    ProductionMutationTestHook testHook = {});

int RunProductionEnableCommand(const std::string& connectionString,
    const ProductionEnableRequest&, std::ostream& output,
    std::ostream& errors);
int RunProductionDisableCommand(const std::string& connectionString,
    const ProductionDisableRequest&, std::ostream& output,
    std::ostream& errors);

} // namespace EA::CampaignOperations
