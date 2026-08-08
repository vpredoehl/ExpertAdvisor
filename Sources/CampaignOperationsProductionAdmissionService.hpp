#pragma once

#include "CampaignOperationsProductionAdmissionRepository.hpp"

#include <iosfwd>
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

} // namespace EA::CampaignOperations
