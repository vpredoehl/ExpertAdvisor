#include "ExperimentReplicationPlanningService.hpp"

#include <ostream>
#include <stdexcept>

namespace EA::ExperimentReplicationPlanning
{
namespace
{

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-' || character == '.';
        if (!safe) character = '_';
    }
    return value.empty() ? "EMPTY" : value;
}

void Validate(const PlanningCommand& command)
{
    if (command.sourceExperimentIds.first <= 0 ||
        command.sourceExperimentIds.second <= 0)
        throw std::invalid_argument(
            "replication planning source IDs must be positive integers");
    if (command.sourceExperimentIds.first ==
        command.sourceExperimentIds.second)
        throw std::invalid_argument(
            "replication planning requires distinct source IDs");
    if (command.requestedSeeds.empty())
        throw std::invalid_argument(
            "replication planning requires at least one seed");
}

} // namespace

int RunPlanningCommand(const PlanningCommand& command,
                       const ExperimentPairComparison::EvidenceSource& evidence,
                       const EquivalentExperimentSource& equivalents,
                       std::ostream& output,
                       std::ostream& errors)
{
    try
    {
        Validate(command);
        const auto armAEvidence = evidence.Load(
            command.sourceExperimentIds.first);
        const auto armBEvidence = evidence.Load(
            command.sourceExperimentIds.second);
        const auto request = ExperimentPairComparison::MakeComparisonRequest(
            armAEvidence, armBEvidence);
        const Plan plan = MakePlan(
            ExperimentPairComparison::MakeArmResultSet(armAEvidence),
            ExperimentPairComparison::MakeArmResultSet(armBEvidence),
            request, command.requestedSeeds, &equivalents);
        output << Render(plan);
        return 0;
    }
    catch (const ExperimentPairComparison::EvidenceUnavailableError& error)
    {
        errors << "CONTROLLED_REPLICATION_WAVE_PLAN_LOAD_FAILED"
               << ",source_experiment_a_id="
               << command.sourceExperimentIds.first
               << ",source_experiment_b_id="
               << command.sourceExperimentIds.second
               << ",reason=" << MachineText(error.reason())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "CONTROLLED_REPLICATION_WAVE_PLAN_LOAD_FAILED"
               << ",source_experiment_a_id="
               << command.sourceExperimentIds.first
               << ",source_experiment_b_id="
               << command.sourceExperimentIds.second
               << ",reason=" << MachineText(error.what())
               << ",exit_code=3,read_only=true\n";
        return 3;
    }
}

} // namespace EA::ExperimentReplicationPlanning
