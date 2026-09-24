#pragma once

#include "ExperimentPairComparisonService.hpp"
#include "ExperimentReplicationPlanning.hpp"

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace EA::ExperimentReplicationPlanning
{

struct PlanningCommand
{
    std::pair<long long, long long> sourceExperimentIds;
    std::vector<unsigned int> requestedSeeds;
};

int RunPlanningCommand(const PlanningCommand& command,
                       const ExperimentPairComparison::EvidenceSource& evidence,
                       const EquivalentExperimentSource& equivalents,
                       std::ostream& output,
                       std::ostream& errors);

// Source loading and every equivalence lookup share one repeatable-read,
// read-only PostgreSQL snapshot. Database failures propagate to the existing
// scheduler CLI exit-2 handler.
int RunPlanningCommand(const std::string& connectionString,
                       const PlanningCommand& command,
                       std::ostream& output,
                       std::ostream& errors);

} // namespace EA::ExperimentReplicationPlanning
