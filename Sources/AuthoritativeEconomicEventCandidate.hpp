#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace EA::EconomicCalendar
{

// One normalized raw-ledger occurrence.  This type deliberately contains no
// model features or published statistical values.
struct AuthoritativeEconomicEventCandidate
{
    std::string currency;
    std::string eventFamily;

    // Canonical UTC instant as Unix microseconds.
    std::int64_t eventTimestampUnixMicros = 0;

    std::string sourceAgency;
    std::string sourceEventId;
    std::string sourceUrl;
    std::optional<std::string> referencePeriod;

    int eventImportance = 0;
    std::string historicalTimeConfidence;

    std::optional<std::string> sourceReleaseDate;
    std::optional<std::string> sourceReleaseTime;
    std::optional<std::string> sourceTimezone;
};

} // namespace EA::EconomicCalendar
