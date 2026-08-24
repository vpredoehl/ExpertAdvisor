#pragma once

#include "AuthoritativeEconomicEventCandidate.hpp"
#include "EconomicEventImportRepository.hpp"

#include <vector>

#include <pqxx/pqxx>

namespace EA::EconomicCalendar
{

enum class EconomicEventImportMode
{
    dryRun,
    apply
};

EconomicEventImportReport RunEconomicEventImport(
    pqxx::connection& connection,
    std::vector<AuthoritativeEconomicEventCandidate> candidates,
    EconomicEventImportMode mode);

bool IsEconomicEventImportCommand(
    int argc,
    const char* const argv[]);

// Complete isolated CLI boundary.  This is dispatched before scheduler/model
// argument parsing and never opens the forex database.
int RunEconomicEventImportCli(
    int argc,
    const char* const argv[]);

} // namespace EA::EconomicCalendar
