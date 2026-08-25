#include "EconomicEventFeatures.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace EA::EconomicCalendar
{
namespace
{

std::size_t FamilyIndex(
    EconomicEventModelFamily family)
{
    return static_cast<std::size_t>(family);
}


PriceTP ValidatedEventTime(
    const EconomicEvent& event)
{
    if (event.eventTimestampUnixMicros % 1000000LL != 0)
    {
        throw std::invalid_argument(
            "economic_event_feature_timestamp_not_whole_second");
    }

    return PriceTP{
        std::chrono::seconds{
            event.eventTimestampUnixMicros / 1000000LL}};
}


void SetIndicator(
    EconomicEventFeatureValues& values,
    EconomicEventModelFamily family)
{
    switch (family)
    {
        case EconomicEventModelFamily::inflation:
            values.inflationEvent = 1.0F;
            return;

        case EconomicEventModelFamily::employment:
            values.employmentEvent = 1.0F;
            return;

        case EconomicEventModelFamily::growth:
            values.growthEvent = 1.0F;
            return;

        case EconomicEventModelFamily::fedPolicy:
            values.fedPolicyEvent = 1.0F;
            return;

        case EconomicEventModelFamily::consumerDemand:
            values.consumerDemandEvent = 1.0F;
            return;

        case EconomicEventModelFamily::count:
            break;
    }

    throw std::logic_error(
        "economic_event_feature_invalid_model_family");
}


void SetDecay(
    EconomicEventFeatureValues& values,
    EconomicEventModelFamily family,
    float decay)
{
    switch (family)
    {
        case EconomicEventModelFamily::inflation:
            values.inflationRecencyDecay = decay;
            return;

        case EconomicEventModelFamily::employment:
            values.employmentRecencyDecay = decay;
            return;

        case EconomicEventModelFamily::growth:
            values.growthRecencyDecay = decay;
            return;

        case EconomicEventModelFamily::fedPolicy:
            values.fedPolicyRecencyDecay = decay;
            return;

        case EconomicEventModelFamily::consumerDemand:
            values.consumerDemandRecencyDecay = decay;
            return;

        case EconomicEventModelFamily::count:
            break;
    }

    throw std::logic_error(
        "economic_event_feature_invalid_model_family");
}

} // namespace


std::array<float, kEconomicEventFeatureWidth>
EconomicEventFeatureValues::Ordered() const noexcept
{
    return {
        inflationEvent,
        employmentEvent,
        growthEvent,
        fedPolicyEvent,
        consumerDemandEvent,
        inflationRecencyDecay,
        employmentRecencyDecay,
        growthRecencyDecay,
        fedPolicyRecencyDecay,
        consumerDemandRecencyDecay,
    };
}


EconomicEventModelFamily MapEconomicEventModelFamily(
    std::string_view sourceAgency,
    std::string_view canonicalEventFamily)
{
    if (sourceAgency == "BLS")
    {
        if (
            canonicalEventFamily == "CPI" ||
            canonicalEventFamily == "PPI")
        {
            return EconomicEventModelFamily::inflation;
        }

        if (
            canonicalEventFamily == "EMPLOYMENT" ||
            canonicalEventFamily == "EMPLOYMENT_ANNUAL" ||
            canonicalEventFamily == "JOLTS")
        {
            return EconomicEventModelFamily::employment;
        }
    }
    else if (sourceAgency == "BEA")
    {
        if (canonicalEventFamily == "PCE")
            return EconomicEventModelFamily::inflation;

        if (canonicalEventFamily == "GDP")
            return EconomicEventModelFamily::growth;
    }
    else if (sourceAgency == "FEDERAL_RESERVE")
    {
        if (canonicalEventFamily == "FOMC")
            return EconomicEventModelFamily::fedPolicy;
    }
    else if (sourceAgency == "CENSUS")
    {
        if (canonicalEventFamily == "DURABLE_GOODS")
            return EconomicEventModelFamily::growth;

        if (canonicalEventFamily == "RETAIL_SALES")
            return EconomicEventModelFamily::consumerDemand;
    }

    throw std::invalid_argument(
        "unsupported_authoritative_economic_event_family:" +
        std::string{sourceAgency} +
        ":" +
        std::string{canonicalEventFamily});
}


EconomicEventFeatureEngine::EconomicEventFeatureEngine(
    std::vector<EconomicEvent> chronologicalEvents)
{
    events_.reserve(
        chronologicalEvents.size());

    std::optional<PriceTP> previousEventTime;

    for (const EconomicEvent& event : chronologicalEvents)
    {
        const PriceTP eventTime =
            ValidatedEventTime(event);

        if (
            previousEventTime &&
            eventTime < *previousEventTime)
        {
            throw std::invalid_argument(
                "economic_event_features_must_be_chronological");
        }

        events_.push_back(
            MappedEvent{
                eventTime,
                MapEconomicEventModelFamily(
                    event.sourceAgency,
                    event.eventFamily)});

        previousEventTime = eventTime;
    }
}


EconomicEventFeatureValues
EconomicEventFeatureEngine::AdvanceCompletedBar(
    PriceTP barStart)
{
    if (
        previousBarStart_ &&
        !(*previousBarStart_ < barStart))
    {
        throw std::invalid_argument(
            "economic_event_feature_bars_must_be_strictly_increasing");
    }

    const PriceTP informationCutoff =
        barStart + kEconomicEventBarDuration;

    EconomicEventFeatureValues values;

    while (
        nextEventIndex_ < events_.size() &&
        events_[nextEventIndex_].timestamp < informationCutoff)
    {
        const MappedEvent& event =
            events_[nextEventIndex_];

        mostRecentEventTimes_[FamilyIndex(event.family)] =
            event.timestamp;

        // A current-bar indicator describes an event contained in this actual
        // observed bar: [barStart, barStart + 15m). An event learned across a
        // market-data gap updates recency on the first later bar but does not
        // claim to have occurred inside that bar.
        if (!(event.timestamp < barStart))
            SetIndicator(values, event.family);

        ++nextEventIndex_;
    }

    for (
        std::size_t index = 0;
        index < mostRecentEventTimes_.size();
        ++index)
    {
        if (!mostRecentEventTimes_[index])
            continue;

        const auto elapsedSeconds =
            std::chrono::duration_cast<std::chrono::seconds>(
                informationCutoff -
                *mostRecentEventTimes_[index])
                .count();

        if (elapsedSeconds < 0)
        {
            throw std::logic_error(
                "economic_event_feature_negative_elapsed_time");
        }

        const double decay =
            std::exp(
                -static_cast<double>(elapsedSeconds) /
                static_cast<double>(
                    kEconomicEventRecencyTimeConstant.count()));

        SetDecay(
            values,
            static_cast<EconomicEventModelFamily>(index),
            static_cast<float>(decay));
    }

    previousBarStart_ = barStart;
    return values;
}


std::size_t
EconomicEventFeatureEngine::ConsumedEventCount() const noexcept
{
    return nextEventIndex_;
}

} // namespace EA::EconomicCalendar
