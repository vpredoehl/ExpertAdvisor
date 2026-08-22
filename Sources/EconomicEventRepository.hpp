#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::EconomicCalendar
{

struct EconomicEvent
{
    long long economicEventId = 0;

    std::string currency;
    std::string eventFamily;

    // Canonical instant, independent of PostgreSQL session timezone.
    std::int64_t eventTimestampUnixMicros = 0;

    // Canonical UTC representation intended for diagnostics/tests.
    std::string eventTimestampUtc;

    std::string sourceAgency;
    std::optional<std::string> sourceEventId;
    std::string sourceUrl;

    std::optional<std::string> referencePeriod;

    int eventImportance = 0;
    std::string historicalTimeConfidence;

    std::optional<std::string> sourceReleaseDate;
    std::optional<std::string> sourceReleaseTime;
    std::optional<std::string> sourceTimezone;
};

bool EconomicEventSchemaExists(
    pqxx::transaction_base& transaction);

std::vector<EconomicEvent> LoadEconomicEvents(
    pqxx::transaction_base& transaction,
    const std::string& currency,
    const std::string& startUtc,
    const std::string& endUtc);

} // namespace EA::EconomicCalendar
