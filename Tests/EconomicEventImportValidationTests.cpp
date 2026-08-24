#include <cassert>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Headers/HistoricalFxTimestamp.hpp"
#include "../Sources/EconomicEventImportValidation.hpp"

using namespace EA::EconomicCalendar;

namespace
{

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

void ExpectFailure(
    const std::function<void()>& operation,
    const std::string& expected)
{
    try
    {
        operation();
        assert(false && "expected failure");
    }
    catch (const std::exception& error)
    {
        assert(std::string{error.what()}.find(expected) != std::string::npos);
    }
}

} // namespace


int main()
{
    const auto later = Candidate(
        "dol_eta:usdl-10-991-nat", "WEEKLY_CLAIMS",
        "2010-07-29", "08:30:00");
    const auto earlier = Candidate(
        "dol_eta:usdl-10-990-nat", "WEEKLY_CLAIMS",
        "2010-07-22", "08:30:00");

    const auto ordered = ValidateAndOrderEconomicEventCandidates({later, earlier});
    assert(ordered[0].sourceEventId == "dol_eta:usdl-10-990-nat");
    assert(ordered[1].sourceEventId == "dol_eta:usdl-10-991-nat");

    ExpectFailure(
        [&] { (void)ValidateAndOrderEconomicEventCandidates({earlier, earlier}); },
        "duplicate_source_event_id");

    auto duplicateTimestamp = earlier;
    duplicateTimestamp.sourceEventId = "dol_eta:other-id";
    duplicateTimestamp.sourceUrl = "https://oui.doleta.gov/press/2010/other.asp";
    ExpectFailure(
        [&] {
            (void)ValidateAndOrderEconomicEventCandidates(
                {earlier, duplicateTimestamp});
        },
        "duplicate_agency_family_timestamp");

    auto incompatibleDuplicate = earlier;
    incompatibleDuplicate.sourceUrl =
        "https://oui.doleta.gov/press/2010/different.asp";
    ExpectFailure(
        [&] {
            (void)ValidateAndOrderEconomicEventCandidates(
                {earlier, incompatibleDuplicate});
        },
        "duplicate_source_event_id");

    auto invalidCurrency = earlier;
    invalidCurrency.currency = "usd";
    ExpectFailure(
        [&] { (void)ValidateAndOrderEconomicEventCandidates({invalidCurrency}); },
        "currency_invalid");

    auto invalidConfidence = earlier;
    invalidConfidence.historicalTimeConfidence = "estimated";
    ExpectFailure(
        [&] { (void)ValidateAndOrderEconomicEventCandidates({invalidConfidence}); },
        "confidence_invalid");

    auto mismatch = earlier;
    mismatch.eventTimestampUnixMicros += 1000000;
    ExpectFailure(
        [&] { (void)ValidateAndOrderEconomicEventCandidates({mismatch}); },
        "local_utc_mismatch");

    auto missingProvenance = earlier;
    missingProvenance.sourceUrl.clear();
    ExpectFailure(
        [&] { (void)ValidateAndOrderEconomicEventCandidates({missingProvenance}); },
        "source_url_invalid");

    auto differentFamily = earlier;
    differentFamily.eventFamily = "ANOTHER_RAW_FAMILY";
    differentFamily.sourceEventId = "dol_eta:another-artifact";
    const auto sameTimeDifferentFamilies =
        ValidateAndOrderEconomicEventCandidates({earlier, differentFamily});
    assert(sameTimeDifferentFamilies.size() == 2);

    auto dateOnly = earlier;
    dateOnly.sourceEventId = "dol_eta:date-only-artifact";
    dateOnly.historicalTimeConfidence = "date_only";
    dateOnly.sourceReleaseTime.reset();
    PriceTP nextMidnight;
    assert(EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
        "2010-07-23 00:00:00", nextMidnight));
    dateOnly.eventTimestampUnixMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            nextMidnight.time_since_epoch()).count();
    assert(ValidateAndOrderEconomicEventCandidates({dateOnly}).size() == 1);
    return 0;
}
