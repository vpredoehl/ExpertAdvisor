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


bool ValidValueShape(
    const EconomicEventConsensusValue& value)
{
    if (
        !std::isfinite(value.canonicalValueLow) ||
        !std::isfinite(value.scale) ||
        value.scale <= 0.0)
    {
        return false;
    }

    if (value.valueKind == "scalar")
        return !value.canonicalValueHigh;

    return
        value.valueKind == "range" &&
        value.canonicalValueHigh &&
        std::isfinite(*value.canonicalValueHigh) &&
        value.canonicalValueLow <= *value.canonicalValueHigh;
}


template <typename MappedEvent>
double Normalize(
    const MappedEvent& event,
    double canonicalValue)
{
    return canonicalValue /
        EconomicEventNormalizationScale(
            event.eventFamily,
            event.selectedConsensus->forecast.unit);
}


template <typename MappedEvent>
void SetConsensus(
    EconomicEventFeatureValues& values,
    const MappedEvent& event)
{
    if (!event.selectedConsensus)
        return;

    const EconomicEventConsensusValue& forecast =
        event.selectedConsensus->forecast;

    if (!ValidValueShape(forecast))
    {
        throw std::invalid_argument(
            "economic_event_consensus_invalid_forecast_shape");
    }

    values.relevantEventHasConsensus = 1.0F;
    values.relevantEventConsensusLow =
        static_cast<float>(Normalize(event, forecast.canonicalValueLow));
    values.relevantEventConsensusHigh =
        static_cast<float>(Normalize(
            event,
            forecast.canonicalValueHigh.value_or(
                forecast.canonicalValueLow)));
    values.relevantEventConsensusIsRange =
        forecast.valueKind == "range" ? 1.0F : 0.0F;
}


template <typename MappedEvent>
bool MoreRelevantAtSameTimestamp(
    const MappedEvent& candidate,
    const MappedEvent& selected)
{
    if (candidate.eventImportance != selected.eventImportance)
        return candidate.eventImportance > selected.eventImportance;

    return
        candidate.economicEventId > 0 &&
        selected.economicEventId > 0 &&
        candidate.economicEventId < selected.economicEventId;
}


template <typename Events>
std::size_t MostRelevantAtTimestamp(
    const Events& events,
    std::size_t first)
{
    std::size_t selected = first;
    for (
        std::size_t index = first + 1;
        index < events.size() &&
            events[index].timestamp == events[first].timestamp;
        ++index)
    {
        if (MoreRelevantAtSameTimestamp(events[index], events[selected]))
            selected = index;
    }
    return selected;
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
        relevantEventHasConsensus,
        relevantEventConsensusLow,
        relevantEventConsensusHigh,
        relevantEventConsensusIsRange,
        releasedEventHasSurprise,
        releasedEventSurprise,
        releasedEventSurpriseAbs,
        releasedEventSurpriseDirection,
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
    else if (
        sourceAgency == "DOL_ETA" &&
        canonicalEventFamily == "WEEKLY_CLAIMS")
    {
        return EconomicEventModelFamily::employment;
    }

    throw std::invalid_argument(
        "unsupported_authoritative_economic_event_family:" +
        std::string{sourceAgency} +
        ":" +
        std::string{canonicalEventFamily});
}


double EconomicEventNormalizationScale(
    std::string_view canonicalEventFamily,
    std::string_view canonicalUnit)
{
    if (
        canonicalEventFamily == "EMPLOYMENT" ||
        canonicalEventFamily == "EMPLOYMENT_ANNUAL")
    {
        if (canonicalUnit != "count")
            throw std::invalid_argument(
                "economic_event_employment_consensus_unit_mismatch");
        return kEconomicEmploymentNormalizationScale;
    }

    if (canonicalEventFamily == "JOLTS")
    {
        if (canonicalUnit != "count")
            throw std::invalid_argument(
                "economic_event_jolts_consensus_unit_mismatch");
        return kEconomicJoltsNormalizationScale;
    }

    if (
        canonicalEventFamily == "CPI" ||
        canonicalEventFamily == "PPI" ||
        canonicalEventFamily == "PCE" ||
        canonicalEventFamily == "GDP" ||
        canonicalEventFamily == "DURABLE_GOODS" ||
        canonicalEventFamily == "RETAIL_SALES" ||
        canonicalEventFamily == "FOMC")
    {
        if (canonicalUnit != "percent")
            throw std::invalid_argument(
                "economic_event_percent_consensus_unit_mismatch");
        return kEconomicPercentNormalizationScale;
    }

    throw std::invalid_argument(
        "unsupported_economic_event_consensus_family:" +
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
                event.economicEventId,
                eventTime,
                MapEconomicEventModelFamily(
                    event.sourceAgency,
                    event.eventFamily),
                event.eventFamily,
                event.eventImportance,
                event.selectedConsensus});

        if (event.selectedConsensus)
        {
            const EconomicEventConsensusValue& forecast =
                event.selectedConsensus->forecast;
            if (!ValidValueShape(forecast))
                throw std::invalid_argument(
                    "economic_event_consensus_invalid_forecast_shape");

            (void)EconomicEventNormalizationScale(
                event.eventFamily,
                forecast.unit);
        }

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

        if (!mostRecentReleasedEventIndex_)
        {
            mostRecentReleasedEventIndex_ = nextEventIndex_;
        }
        else
        {
            const MappedEvent& selected =
                events_[*mostRecentReleasedEventIndex_];
            if (
                selected.timestamp < event.timestamp ||
                (selected.timestamp == event.timestamp &&
                 MoreRelevantAtSameTimestamp(event, selected)))
            {
                mostRecentReleasedEventIndex_ = nextEventIndex_;
            }
        }

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

    // Final pre-release consensus is exposed only when the completed
    // information cutoff is exactly the authoritative release timestamp. On
    // all other bars, consensus describes the most-recent released event.
    // This makes a provider's final historical forecast
    // impossible to leak into arbitrarily early bars when no historical
    // provider-observation timestamp exists.
    const bool exactBoundaryEvent =
        nextEventIndex_ < events_.size() &&
        events_[nextEventIndex_].timestamp == informationCutoff;

    if (exactBoundaryEvent)
    {
        const std::size_t relevant =
            MostRelevantAtTimestamp(events_, nextEventIndex_);
        SetConsensus(values, events_[relevant]);
    }
    else if (mostRecentReleasedEventIndex_)
    {
        const MappedEvent& relevant =
            events_[*mostRecentReleasedEventIndex_];
        SetConsensus(values, relevant);
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
