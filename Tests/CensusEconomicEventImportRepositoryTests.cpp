#include <cassert>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <pqxx/pqxx>

#include "../Headers/HistoricalFxTimestamp.hpp"
#include "../Sources/EconomicEventImportService.hpp"

using namespace EA::EconomicCalendar;

namespace
{

std::string RequiredEnvironment(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || !*value)
        throw std::runtime_error(std::string{"missing environment: "} + name);
    return value;
}

AuthoritativeEconomicEventCandidate Candidate(
    std::string agency,
    std::string id,
    std::string family,
    std::string date,
    std::string time,
    std::string referencePeriod,
    std::string url)
{
    PriceTP instant;
    assert(EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        date + " " + time, instant));

    AuthoritativeEconomicEventCandidate candidate;
    candidate.currency = "USD";
    candidate.eventFamily = std::move(family);
    candidate.eventTimestampUnixMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            instant.time_since_epoch()).count();
    candidate.sourceAgency = std::move(agency);
    candidate.sourceEventId = std::move(id);
    candidate.sourceUrl = std::move(url);
    candidate.referencePeriod = std::move(referencePeriod);
    candidate.eventImportance = 3;
    candidate.historicalTimeConfidence = "exact";
    candidate.sourceReleaseDate = std::move(date);
    candidate.sourceReleaseTime = std::move(time);
    candidate.sourceTimezone = "America/New_York";
    return candidate;
}

long long RowCount(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return transaction.query_value<long long>(
        "SELECT count(*) FROM economic_event;");
}

long long AgencyCount(
    pqxx::connection& connection,
    const std::string& agency)
{
    pqxx::read_transaction transaction{connection};
    return transaction.query_value<long long>(
        "SELECT count(*) FROM economic_event WHERE source_agency = $1;",
        pqxx::params{agency});
}

} // namespace


int main()
{
    const std::string database = RequiredEnvironment("LSTM_DB_NAME");
    if (database.rfind("ea_economic_calendar_phase4_census_", 0) != 0)
        throw std::runtime_error("refusing_non_disposable_database");

    const std::string connectionString =
        "host=" + RequiredEnvironment("LSTM_DB_HOST") +
        " user=" + RequiredEnvironment("LSTM_DB_USER") +
        " dbname=" + database;
    pqxx::connection connection{connectionString};

    const auto dolSeed = Candidate(
        "DOL_ETA", "dol_eta:phase4-isolation-seed", "WEEKLY_CLAIMS",
        "2012-01-26", "08:30:00", "week ending 2012-01-21",
        "https://oui.doleta.gov/press/2012/012612.asp");
    const auto beaSeed = Candidate(
        "BEA", "bea:phase4-isolation-seed", "GDP_ADVANCE",
        "2012-01-27", "08:30:00", "Q4 2011",
        "https://www.bea.gov/news/2012/test-phase4-seed");
    assert(RunEconomicEventImport(
        connection, {dolSeed}, EconomicEventImportMode::apply).inserted == 1);
    assert(RunEconomicEventImport(
        connection, {beaSeed}, EconomicEventImportMode::apply).inserted == 1);
    assert(AgencyCount(connection, "DOL_ETA") == 1);
    assert(AgencyCount(connection, "BEA") == 1);

    const auto original = Candidate(
        "CENSUS", "census:test-retail", "RETAIL_SALES_ADVANCE",
        "2012-02-14", "08:30:00", "2012-01",
        "https://www2.census.gov/retail/releases/historical/marts/adv1201.pdf");

    const auto dryRun = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::dryRun);
    assert(dryRun.inserted == 1);
    assert(dryRun.unchanged == 0);
    assert(dryRun.rejected == 0);
    assert(RowCount(connection) == 2);
    assert(AgencyCount(connection, "CENSUS") == 0);

    const auto firstApply = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::apply);
    assert(firstApply.inserted == 1);
    assert(firstApply.rejected == 0);
    assert(RowCount(connection) == 3);

    const auto secondApply = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::apply);
    assert(secondApply.inserted == 0);
    assert(secondApply.unchanged == 1);
    assert(secondApply.updated == 0);
    assert(secondApply.rejected == 0);
    assert(RowCount(connection) == 3);

    auto timestampConflict = original;
    timestampConflict.sourceReleaseTime = "08:31:00";
    timestampConflict.eventTimestampUnixMicros += 60 * 1000000;
    const auto timestampResult = RunEconomicEventImport(
        connection, {timestampConflict}, EconomicEventImportMode::apply);
    assert(timestampResult.rejected == 1);
    assert(timestampResult.items.front().diagnostic ==
           "existing_exact_timestamp_conflict");
    assert(RowCount(connection) == 3);

    auto familyConflict = original;
    familyConflict.eventFamily = "CONSTRUCTION_SPENDING";
    const auto familyResult = RunEconomicEventImport(
        connection, {familyConflict}, EconomicEventImportMode::apply);
    assert(familyResult.rejected == 1);
    assert(RowCount(connection) == 3);

    auto sameTimestampDifferentFamily = original;
    sameTimestampDifferentFamily.sourceEventId = "census:test-construction";
    sameTimestampDifferentFamily.eventFamily = "CONSTRUCTION_SPENDING";
    sameTimestampDifferentFamily.sourceUrl =
        "https://www.census.gov/construction/c30/pdf/pr201201.pdf";
    const auto differentFamilyResult = RunEconomicEventImport(
        connection, {sameTimestampDifferentFamily},
        EconomicEventImportMode::apply);
    assert(differentFamilyResult.inserted == 1);
    assert(differentFamilyResult.rejected == 0);
    assert(RowCount(connection) == 4);

    const auto wouldInsert = Candidate(
        "CENSUS", "census:test-new-home-sales", "NEW_RESIDENTIAL_SALES",
        "2012-02-24", "10:00:00", "2012-01",
        "https://www.census.gov/construction/nrs/pdf/newressales_201201.pdf");
    const auto rollback = RunEconomicEventImport(
        connection, {wouldInsert, familyConflict},
        EconomicEventImportMode::apply);
    assert(rollback.inserted == 0);
    assert(rollback.rejected == 2);
    assert(RowCount(connection) == 4);

    const auto censusCrossAgency = Candidate(
        "CENSUS", "census:cross-agency", "MANUFACTURERS_ORDERS",
        "2012-03-29", "10:00:00", "2012-02",
        "https://www.census.gov/manufacturing/m3/historical_data/pressreleases/prel/2012/feb12prel.pdf");
    const auto beaCrossAgency = Candidate(
        "BEA", "bea:phase4-cross-agency", "MANUFACTURERS_ORDERS",
        "2012-03-29", "10:00:00", "2012-02",
        "https://www.bea.gov/news/2012/test-phase4-cross-agency");
    assert(RunEconomicEventImport(
        connection, {censusCrossAgency}, EconomicEventImportMode::apply).inserted == 1);
    assert(RunEconomicEventImport(
        connection, {beaCrossAgency}, EconomicEventImportMode::apply).inserted == 1);

    assert(AgencyCount(connection, "DOL_ETA") == 1);
    assert(AgencyCount(connection, "BEA") == 2);
    assert(AgencyCount(connection, "CENSUS") == 3);
    assert(RowCount(connection) == 6);
    return 0;
}
