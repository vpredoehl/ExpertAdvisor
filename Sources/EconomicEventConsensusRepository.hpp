#pragma once

#include "EconomicEventConsensusImport.hpp"

#include <vector>

#include <pqxx/pqxx>

namespace EA::EconomicCalendar
{

EconomicEventConsensusImportReport CompareEconomicEventConsensusBatch(
    pqxx::transaction_base& transaction,
    const std::vector<EconomicEventConsensusCandidate>& candidates);

EconomicEventConsensusImportReport ApplyEconomicEventConsensusBatch(
    pqxx::connection& connection,
    const std::vector<EconomicEventConsensusCandidate>& candidates);

} // namespace EA::EconomicCalendar
