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
    candidate.eventImportance = candidate.eventFamily == "BEIGE_BOOK" ? 2 : 3;
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
    if (database.rfind("ea_economic_calendar_phase5_fed_", 0) != 0)
        throw std::runtime_error("refusing_non_disposable_database");

    const std::string connectionString =
        "host=" + RequiredEnvironment("LSTM_DB_HOST") +
        " user=" + RequiredEnvironment("LSTM_DB_USER") +
        " dbname=" + database;
    pqxx::connection connection{connectionString};

    const auto dolSeed = Candidate(
        "DOL_ETA", "dol_eta:phase5-isolation-seed", "WEEKLY_CLAIMS",
        "2012-01-26", "08:30:00", "week ending 2012-01-21",
        "https://oui.doleta.gov/press/2012/012612.asp");
    const auto beaSeed = Candidate(
        "BEA", "bea:phase5-isolation-seed", "GDP_ADVANCE",
        "2012-01-27", "08:30:00", "Q4 2011",
        "https://www.bea.gov/news/2012/test-phase5-seed");
    const auto censusSeed = Candidate(
        "CENSUS", "census:phase5-isolation-seed", "RETAIL_SALES_ADVANCE",
        "2012-02-14", "08:30:00", "2012-01",
        "https://www2.census.gov/retail/releases/historical/marts/adv1201.pdf");
    assert(RunEconomicEventImport(
        connection, {dolSeed, beaSeed, censusSeed},
        EconomicEventImportMode::apply).inserted == 3);

    const auto original = Candidate(
        "FEDERAL_RESERVE", "federal_reserve:test-statement",
        "FOMC_STATEMENT", "2012-03-13", "14:15:00",
        "meeting ending 2012-03-13",
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120313a.htm");

    const auto dryRun = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::dryRun);
    assert(dryRun.inserted == 1);
    assert(dryRun.rejected == 0);
    assert(RowCount(connection) == 3);
    assert(AgencyCount(connection, "FEDERAL_RESERVE") == 0);

    const auto firstApply = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::apply);
    assert(firstApply.inserted == 1);
    assert(firstApply.rejected == 0);
    assert(RowCount(connection) == 4);

    const auto secondApply = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::apply);
    assert(secondApply.inserted == 0);
    assert(secondApply.unchanged == 1);
    assert(secondApply.updated == 0);
    assert(secondApply.rejected == 0);
    assert(RowCount(connection) == 4);

    auto timestampConflict = original;
    timestampConflict.sourceReleaseTime = "14:16:00";
    timestampConflict.eventTimestampUnixMicros += 60 * 1000000;
    const auto timestampResult = RunEconomicEventImport(
        connection, {timestampConflict}, EconomicEventImportMode::apply);
    assert(timestampResult.rejected == 1);
    assert(timestampResult.items.front().diagnostic ==
           "existing_exact_timestamp_conflict");
    assert(RowCount(connection) == 4);

    auto identityConflict = original;
    identityConflict.eventFamily = "FOMC_MINUTES";
    identityConflict.referencePeriod = "meeting 2012-02-01";
    const auto identityResult = RunEconomicEventImport(
        connection, {identityConflict}, EconomicEventImportMode::apply);
    assert(identityResult.rejected == 1);
    assert(RowCount(connection) == 4);

    auto sameTimestampMinutes = original;
    sameTimestampMinutes.sourceEventId =
        "federal_reserve:test-minutes-same-time";
    sameTimestampMinutes.eventFamily = "FOMC_MINUTES";
    sameTimestampMinutes.referencePeriod = "meeting 2012-02-01";
    sameTimestampMinutes.sourceUrl =
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120313b.htm";
    const auto differentFamilyResult = RunEconomicEventImport(
        connection, {sameTimestampMinutes}, EconomicEventImportMode::apply);
    assert(differentFamilyResult.inserted == 1);
    assert(differentFamilyResult.rejected == 0);
    assert(RowCount(connection) == 5);

    const auto wouldInsert = Candidate(
        "FEDERAL_RESERVE", "federal_reserve:test-beige-book",
        "BEIGE_BOOK", "2012-04-11", "14:00:00",
        "publication 2012-04-11",
        "https://www.federalreserve.gov/monetarypolicy/beigebook/files/BeigeBook_20120411.pdf");
    const auto rollback = RunEconomicEventImport(
        connection, {wouldInsert, identityConflict},
        EconomicEventImportMode::apply);
    assert(rollback.inserted == 0);
    assert(rollback.rejected == 2);
    assert(RowCount(connection) == 5);

    const auto fedCrossAgency = Candidate(
        "FEDERAL_RESERVE", "federal_reserve:test-cross-agency",
        "FOMC_STATEMENT", "2012-06-20", "12:30:00",
        "meeting ending 2012-06-20",
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20120620a.htm");
    const auto censusCrossAgency = Candidate(
        "CENSUS", "census:phase5-cross-agency",
        "FOMC_STATEMENT", "2012-06-20", "12:30:00", "2012-05",
        "https://www.census.gov/construction/c30/pdf/pr201205.pdf");
    assert(RunEconomicEventImport(
        connection, {fedCrossAgency}, EconomicEventImportMode::apply).inserted == 1);
    assert(RunEconomicEventImport(
        connection, {censusCrossAgency}, EconomicEventImportMode::apply).inserted == 1);

    auto fomcA = Candidate(
        "FEDERAL_RESERVE", "federal_reserve:monetary20140917a",
        "FOMC_STATEMENT", "2014-09-18", "00:00:00",
        "meeting ending 2014-09-17",
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140917a.htm");
    fomcA.historicalTimeConfidence = "date_only";
    fomcA.sourceReleaseDate = "2014-09-17";
    fomcA.sourceReleaseTime.reset();
    auto fomcC = fomcA;
    fomcC.sourceEventId = "federal_reserve:monetary20140917c";
    fomcC.sourceUrl =
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20140917c.htm";
    const auto sameBoundary = RunEconomicEventImport(
        connection, {fomcA, fomcC}, EconomicEventImportMode::apply);
    assert(sameBoundary.inserted == 2);
    assert(sameBoundary.rejected == 0);
    const auto sameBoundaryRepeat = RunEconomicEventImport(
        connection, {fomcA, fomcC}, EconomicEventImportMode::apply);
    assert(sameBoundaryRepeat.unchanged == 2);
    assert(sameBoundaryRepeat.rejected == 0);

    assert(AgencyCount(connection, "DOL_ETA") == 1);
    assert(AgencyCount(connection, "BEA") == 1);
    assert(AgencyCount(connection, "CENSUS") == 2);
    assert(AgencyCount(connection, "FEDERAL_RESERVE") == 5);
    assert(RowCount(connection) == 9);
    return 0;
}
