#pragma once

#include "CausalSurpriseObservability.hpp"

#include <pqxx/pqxx>

namespace EA::CausalSurpriseObservability
{

struct BarPopulation
{
    std::vector<PriceTP> sourceBarStarts;
    std::size_t warmupRowCount = 0;
};

// The caller owns the read-only transaction. This repository exposes no
// persistence operation and does not require checkpoint/final-model evidence.
ExperimentContext LoadExperimentContext(
    pqxx::transaction_base& transaction,
    long long experimentId);

BarPopulation LoadBarPopulation(
    pqxx::transaction_base& transaction,
    const std::string& symbol,
    const std::string& sourceStart,
    const std::string& outputStart,
    const std::string& outputEnd);

} // namespace EA::CausalSurpriseObservability
