#pragma once

#include "AuthoritativeEconomicEventCandidate.hpp"

#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::EconomicCalendar
{

enum class EconomicEventImportDisposition
{
    inserted,
    unchanged,
    updated,
    rejected
};

struct EconomicEventImportItemResult
{
    std::string sourceEventId;
    EconomicEventImportDisposition disposition =
        EconomicEventImportDisposition::rejected;
    std::string diagnostic;
};

struct EconomicEventImportReport
{
    std::size_t inserted = 0;
    std::size_t unchanged = 0;
    std::size_t updated = 0;
    std::size_t rejected = 0;
    std::vector<EconomicEventImportItemResult> items;
};

const char* EconomicEventImportDispositionName(
    EconomicEventImportDisposition disposition);

// Read-only comparison used by dry-run.  "inserted" means would insert.
EconomicEventImportReport CompareEconomicEventImportBatch(
    pqxx::transaction_base& transaction,
    const std::vector<AuthoritativeEconomicEventCandidate>& candidates);

// Sole SQL mutation boundary.  Opens and commits one transaction for the
// complete batch.  Any conflict rejects and rolls back the complete batch.
EconomicEventImportReport ApplyEconomicEventImportBatch(
    pqxx::connection& connection,
    const std::vector<AuthoritativeEconomicEventCandidate>& candidates);

} // namespace EA::EconomicCalendar
