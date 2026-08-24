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
    std::string referencePeriod)
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
    candidate.sourceUrl = candidate.sourceAgency == "BEA"
        ? "https://www.bea.gov/news/2012/test-authoritative-occurrence"
        : "https://oui.doleta.gov/press/2012/012712.asp";
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
    if (database.rfind("ea_economic_calendar_phase3_bea_", 0) != 0)
        throw std::runtime_error("refusing_non_disposable_database");

    const std::string connectionString =
        "host=" + RequiredEnvironment("LSTM_DB_HOST") +
        " user=" + RequiredEnvironment("LSTM_DB_USER") +
        " dbname=" + database;
    pqxx::connection connection{connectionString};

    const auto dolSeed = Candidate(
        "DOL_ETA", "dol_eta:phase3-isolation-seed", "WEEKLY_CLAIMS",
        "2012-01-26", "08:30:00", "week ending 2012-01-21");
    const auto dolApply = RunEconomicEventImport(
        connection, {dolSeed}, EconomicEventImportMode::apply);
    assert(dolApply.inserted == 1);
    assert(AgencyCount(connection, "DOL_ETA") == 1);

    const auto original = Candidate(
        "BEA", "bea:test-gdp-advance", "GDP_ADVANCE",
        "2012-01-27", "08:30:00", "Q4 2011");

    const auto dryRun = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::dryRun);
    assert(dryRun.inserted == 1);
    assert(dryRun.unchanged == 0);
    assert(dryRun.rejected == 0);
    assert(RowCount(connection) == 1);
    assert(AgencyCount(connection, "DOL_ETA") == 1);

    const auto firstApply = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::apply);
    assert(firstApply.inserted == 1);
    assert(firstApply.rejected == 0);
    assert(RowCount(connection) == 2);

    const auto secondApply = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::apply);
    assert(secondApply.inserted == 0);
    assert(secondApply.unchanged == 1);
    assert(secondApply.updated == 0);
    assert(secondApply.rejected == 0);
    assert(RowCount(connection) == 2);

    auto sourceIdConflict = original;
    sourceIdConflict.sourceUrl =
        "https://www.bea.gov/news/2012/conflicting-authoritative-occurrence";
    const auto identityResult = RunEconomicEventImport(
        connection, {sourceIdConflict}, EconomicEventImportMode::apply);
    assert(identityResult.rejected == 1);
    assert(RowCount(connection) == 2);

    auto timestampConflict = original;
    timestampConflict.sourceReleaseTime = "08:31:00";
    timestampConflict.eventTimestampUnixMicros += 60 * 1000000;
    const auto timestampResult = RunEconomicEventImport(
        connection, {timestampConflict}, EconomicEventImportMode::apply);
    assert(timestampResult.rejected == 1);
    assert(timestampResult.items.front().diagnostic ==
           "existing_exact_timestamp_conflict");
    assert(RowCount(connection) == 2);

    auto sameTimestampDifferentFamily = original;
    sameTimestampDifferentFamily.sourceEventId = "bea:test-gdp-second";
    sameTimestampDifferentFamily.eventFamily = "GDP_SECOND";
    sameTimestampDifferentFamily.sourceUrl =
        "https://www.bea.gov/news/2012/test-second-occurrence";
    const auto differentFamilyResult = RunEconomicEventImport(
        connection, {sameTimestampDifferentFamily},
        EconomicEventImportMode::apply);
    assert(differentFamilyResult.inserted == 1);
    assert(differentFamilyResult.rejected == 0);
    assert(RowCount(connection) == 3);

    const auto wouldInsert = Candidate(
        "BEA", "bea:test-personal-income-outlays", "PERSONAL_INCOME_OUTLAYS",
        "2012-02-01", "08:30:00", "2011-12");
    const auto rollback = RunEconomicEventImport(
        connection, {wouldInsert, sourceIdConflict},
        EconomicEventImportMode::apply);
    assert(rollback.inserted == 0);
    assert(rollback.rejected == 2);
    assert(RowCount(connection) == 3);

    const auto dolCrossAgency = Candidate(
        "DOL_ETA", "dol_eta:cross-agency", "GDP_THIRD",
        "2012-03-29", "08:30:00", "Q4 2011");
    const auto beaCrossAgency = Candidate(
        "BEA", "bea:cross-agency", "GDP_THIRD",
        "2012-03-29", "08:30:00", "Q4 2011");
    assert(RunEconomicEventImport(
        connection, {dolCrossAgency}, EconomicEventImportMode::apply).inserted == 1);
    assert(RunEconomicEventImport(
        connection, {beaCrossAgency}, EconomicEventImportMode::apply).inserted == 1);
    assert(AgencyCount(connection, "DOL_ETA") == 2);
    assert(AgencyCount(connection, "BEA") == 3);
    assert(RowCount(connection) == 5);
    return 0;
}
