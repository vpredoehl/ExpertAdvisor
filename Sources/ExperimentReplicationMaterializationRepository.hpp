#pragma once

#include "ExperimentReplicationPlanning.hpp"

#include <pqxx/pqxx>

namespace EA::ExperimentReplicationMaterialization
{

long long InsertFreshPausedReplicationExperiment(
    pqxx::transaction_base& transaction,
    const ExperimentReplicationPlanning::ProposedExperimentSpecification&
        specification);

// Clones the authoritative configured source row while changing only its
// canonical symbol.  Generated database identity and operational columns are
// deliberately fresh; the result is paused/train and is never dispatched.
long long InsertFreshPausedCrossSymbolExperiment(
    pqxx::transaction_base& transaction, long long sourceExperimentId,
    const std::string& targetSymbol);

long long InsertFreshPausedLayout11ConfluenceExperiment(
    pqxx::transaction_base& transaction, long long templateExperimentId,
    unsigned int freshInitializationSeed, const std::string& featureAblationMask);

} // namespace EA::ExperimentReplicationMaterialization
