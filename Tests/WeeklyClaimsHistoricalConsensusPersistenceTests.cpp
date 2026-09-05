#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <string>

#include <pqxx/pqxx>

#include "EconomicEventFeatures.hpp"
#include "EconomicEventRepository.hpp"

using namespace EA::EconomicCalendar;

namespace
{

std::string RequiredEnvironment(const char* name)
{
    const char* value = std::getenv(name);
    assert(value != nullptr && *value != '\0');
    return value;
}

PriceTP At(std::int64_t seconds)
{
    return PriceTP{std::chrono::seconds{seconds}};
}

const EconomicEvent& RequireEvent(
    const std::vector<EconomicEvent>& events,
    long long economicEventId)
{
    const auto found = std::find_if(
        events.begin(),
        events.end(),
        [economicEventId](const EconomicEvent& event)
        {
            return event.economicEventId == economicEventId;
        });

    assert(found != events.end());
    return *found;
}

} // namespace

int main()
{
    pqxx::connection connection{
        "host=" + RequiredEnvironment("LSTM_DB_HOST") +
        " user=" + RequiredEnvironment("LSTM_DB_USER") +
        " dbname=" + RequiredEnvironment("LSTM_TEST_DB_NAME")};
    pqxx::read_transaction read{connection};

    const auto before = LoadEconomicEventsForFeatureRange(
        read, "USD", "2025-06-05 12:20:00+00",
        "2025-06-05 12:34:59+00");
    const auto& beforeWeeklyClaims = RequireEvent(before, 3467);
    assert(beforeWeeklyClaims.selectedConsensus);
    assert(beforeWeeklyClaims.selectedConsensus->forecast.canonicalValueLow ==
           235000.0);
    assert(beforeWeeklyClaims.firstReleaseActualState ==
           EconomicEventFirstReleaseActualState::notYetAvailable);
    EconomicEventFeatureEngine beforeEngine{before};
    const auto beforeValues = beforeEngine.AdvanceCompletedBar(At(1749125999));
    assert(beforeValues.causalFirstReleaseSurpriseAvailable == 0.0F);
    assert(beforeValues.causalFirstReleaseSurprise == 0.0F);

    const auto atPublication = LoadEconomicEventsForFeatureRange(
        read, "USD", "2025-06-05 12:20:00+00",
        "2025-06-05 12:35:00+00");
    const auto& publishedWeeklyClaims =
        RequireEvent(atPublication, 3467);
    assert(publishedWeeklyClaims.firstReleaseActualState ==
           EconomicEventFirstReleaseActualState::provenFirstRelease);
    EconomicEventFeatureEngine atPublicationEngine{atPublication};
    const auto zero =
        atPublicationEngine.AdvanceCompletedBar(At(1749126000));
    assert(zero.causalFirstReleaseSurpriseAvailable == 1.0F);
    assert(zero.causalFirstReleaseSurprise == 0.0F);

    const auto missing = LoadEconomicEventsForFeatureRange(
        read, "USD", "2025-06-12 12:20:00+00",
        "2025-06-12 12:35:00+00");
    const auto& missingWeeklyClaims = RequireEvent(missing, 3468);
    assert(!missingWeeklyClaims.selectedConsensus);
    EconomicEventFeatureEngine missingEngine{missing};
    const auto missingValues =
        missingEngine.AdvanceCompletedBar(At(1749730800));
    assert(missingValues.causalFirstReleaseSurpriseAvailable == 0.0F);
    assert(missingValues.causalFirstReleaseSurprise == 0.0F);
    assert(missingEngine.LastCausalSurpriseObservation().disposition ==
           CausalSurpriseDisposition::missingForecast);

    return 0;
}
