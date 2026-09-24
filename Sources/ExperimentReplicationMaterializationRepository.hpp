#pragma once

#include "ExperimentReplicationPlanning.hpp"

#include <pqxx/pqxx>

namespace EA::ExperimentReplicationMaterialization
{

long long InsertFreshPausedReplicationExperiment(
    pqxx::transaction_base& transaction,
    const ExperimentReplicationPlanning::ProposedExperimentSpecification&
        specification);

} // namespace EA::ExperimentReplicationMaterialization
