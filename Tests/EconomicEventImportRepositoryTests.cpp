#include <cassert>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

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
    std::string id,
    std::string family,
    std::string date,
    std::string time)
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
    candidate.sourceAgency = "DOL_ETA";
    candidate.sourceEventId = std::move(id);
    candidate.sourceUrl = "https://oui.doleta.gov/press/2010/072210.asp";
    candidate.referencePeriod = "week ending 2010-07-17";
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

} // namespace


int main()
{
    const std::string database = RequiredEnvironment("LSTM_DB_NAME");
    if (database.rfind("ea_economic_calendar_phase2_dol_", 0) != 0)
        throw std::runtime_error("refusing_non_disposable_database");

    const std::string connectionString =
        "host=" + RequiredEnvironment("LSTM_DB_HOST") +
        " user=" + RequiredEnvironment("LSTM_DB_USER") +
        " dbname=" + database;
    pqxx::connection connection{connectionString};

    const auto original = Candidate(
        "dol_eta:usdl-10-990-nat",
        "WEEKLY_CLAIMS",
        "2010-07-22",
        "08:30:00");

    const auto dryRun = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::dryRun);
    assert(dryRun.inserted == 1);
    assert(dryRun.unchanged == 0);
    assert(dryRun.rejected == 0);
    assert(RowCount(connection) == 0);

    const auto firstApply = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::apply);
    assert(firstApply.inserted == 1);
    assert(firstApply.unchanged == 0);
    assert(firstApply.updated == 0);
    assert(firstApply.rejected == 0);
    assert(RowCount(connection) == 1);

    const auto secondApply = RunEconomicEventImport(
        connection, {original}, EconomicEventImportMode::apply);
    assert(secondApply.inserted == 0);
    assert(secondApply.unchanged == 1);
    assert(secondApply.updated == 0);
    assert(secondApply.rejected == 0);
    assert(RowCount(connection) == 1);

    auto metadataDifference = original;
    metadataDifference.sourceUrl =
        "https://oui.doleta.gov/press/2010/corrected.asp";
    const auto metadataConflict = RunEconomicEventImport(
        connection, {metadataDifference}, EconomicEventImportMode::apply);
    assert(metadataConflict.inserted == 0);
    assert(metadataConflict.updated == 0);
    assert(metadataConflict.rejected == 1);
    assert(RowCount(connection) == 1);

    const auto exactTimestampCorrection = Candidate(
        original.sourceEventId,
        original.eventFamily,
        "2010-07-22",
        "08:31:00");
    const auto timestampConflict = RunEconomicEventImport(
        connection, {exactTimestampCorrection}, EconomicEventImportMode::apply);
    assert(timestampConflict.rejected == 1);
    assert(timestampConflict.items.front().diagnostic ==
           "existing_exact_timestamp_conflict");
    assert(RowCount(connection) == 1);

    auto identityCollision = Candidate(
        original.sourceEventId,
        "OTHER_CLAIMS_ARTIFACT",
        "2010-07-29",
        "08:30:00");
    identityCollision.sourceUrl =
        "https://oui.doleta.gov/press/2010/072910.asp";
    identityCollision.referencePeriod = "week ending 2010-07-24";
    const auto collision = RunEconomicEventImport(
        connection, {identityCollision}, EconomicEventImportMode::apply);
    assert(collision.rejected == 1);
    assert(RowCount(connection) == 1);

    auto newCandidate = Candidate(
        "dol_eta:usdl-10-1001-nat",
        "WEEKLY_CLAIMS",
        "2010-08-05",
        "08:30:00");
    newCandidate.sourceUrl =
        "https://oui.doleta.gov/press/2010/080510.asp";
    newCandidate.referencePeriod = "week ending 2010-07-31";
    const auto rollback = RunEconomicEventImport(
        connection,
        {newCandidate, metadataDifference},
        EconomicEventImportMode::apply);
    assert(rollback.inserted == 0);
    assert(rollback.rejected == 2);
    assert(RowCount(connection) == 1);

    auto sameTimestampDifferentFamily = original;
    sameTimestampDifferentFamily.sourceEventId =
        "dol_eta:distinct-publication-same-time";
    sameTimestampDifferentFamily.eventFamily = "DISTINCT_RAW_FAMILY";
    sameTimestampDifferentFamily.sourceUrl =
        "https://oui.doleta.gov/press/2010/distinct.asp";
    const auto differentFamilyApply = RunEconomicEventImport(
        connection,
        {sameTimestampDifferentFamily},
        EconomicEventImportMode::apply);
    assert(differentFamilyApply.inserted == 1);
    assert(differentFamilyApply.rejected == 0);
    assert(RowCount(connection) == 2);
    return 0;
}
