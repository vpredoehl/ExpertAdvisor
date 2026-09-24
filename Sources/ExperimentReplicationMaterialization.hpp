#pragma once

#include "ExperimentReplicationPlanningService.hpp"

#include <iosfwd>
#include <string>

namespace EA::ExperimentReplicationMaterialization
{

namespace Planning = ExperimentReplicationPlanning;

inline constexpr int kMaterializerVersion = 1;

using MaterializationCommand = Planning::PlanningCommand;

class ExperimentInserter
{
public:
    virtual ~ExperimentInserter() = default;
    virtual long long InsertFreshPausedExperiment(
        const Planning::ProposedExperimentSpecification& specification) = 0;
};

// Executes against an already established transaction/serialization boundary.
// A nonzero return performs no inserts for validation/equivalence failures.
// Insert exceptions propagate so the owning transaction necessarily rolls back.
int RunMaterializationInTransaction(
    const MaterializationCommand& command,
    const ExperimentPairComparison::EvidenceSource& evidence,
    const Planning::EquivalentExperimentSource& equivalents,
    ExperimentInserter& inserter,
    std::ostream& output,
    std::ostream& errors);

// Owns one PostgreSQL write transaction.  It takes a table lock before source
// reload, preflight, equivalence checks, or inserts and commits only a complete
// successful wave. Database failures retain the scheduler CLI exit-2 contract.
int RunMaterializationCommand(const std::string& connectionString,
                              const MaterializationCommand& command,
                              std::ostream& output,
                              std::ostream& errors);

} // namespace EA::ExperimentReplicationMaterialization
