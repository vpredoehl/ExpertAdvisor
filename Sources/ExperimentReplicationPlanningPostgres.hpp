#pragma once

#include "ExperimentReplicationPlanning.hpp"
#include "ExperimentPairComparisonService.hpp"

#include <pqxx/pqxx>

namespace EA::ExperimentReplicationPlanning
{

// Shared PostgreSQL evidence/equivalence adapter.  Read-only planning and
// transactional materialization use this exact implementation; the latter
// invokes it only while holding its experiment-table serialization lock.
class PostgresPlanningSource final :
    public ExperimentPairComparison::EvidenceSource,
    public EquivalentExperimentSource
{
public:
    explicit PostgresPlanningSource(pqxx::transaction_base& transaction);

    FeatureAblationPairEvaluation::ArmEvidence Load(
        long long experimentId) const override;

    EquivalentExperimentResult FindEquivalent(
        const ExperimentPairComparison::ArmResultSet& proposedArm) const override;

private:
    pqxx::transaction_base& transaction_;
};

} // namespace EA::ExperimentReplicationPlanning
