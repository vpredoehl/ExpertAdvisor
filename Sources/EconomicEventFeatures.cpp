#include "EconomicEventFeatures.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
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


bool IsLowerHexSha256(std::string_view value)
{
    return value.size() == 64 &&
        std::all_of(
            value.begin(), value.end(),
            [](unsigned char character)
            {
                return std::isdigit(character) != 0 ||
                    (character >= 'a' && character <= 'f');
            });
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
float NormalizedFloat(
    const MappedEvent& event,
    double canonicalValue)
{
    const double normalized = Normalize(event, canonicalValue);
    if (!std::isfinite(normalized) ||
        std::abs(normalized) >
            static_cast<double>(std::numeric_limits<float>::max()))
    {
        throw std::invalid_argument(
            "economic_event_consensus_normalized_value_not_finite_float");
    }
    const float result = static_cast<float>(normalized);
    if (!std::isfinite(result))
        throw std::invalid_argument(
            "economic_event_consensus_normalized_value_not_finite_float");
    return result;
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
        NormalizedFloat(event, forecast.canonicalValueLow);
    values.relevantEventConsensusHigh =
        NormalizedFloat(
            event,
            forecast.canonicalValueHigh.value_or(
                forecast.canonicalValueLow));
    values.relevantEventConsensusIsRange =
        forecast.valueKind == "range" ? 1.0F : 0.0F;
}


bool CompatibleScalarSurprise(
    const EconomicEventSelectedConsensus& selected,
    const EconomicEventReleaseActual& releaseActual)
{
    const EconomicEventConsensusValue& forecast = selected.forecast;
    const EconomicEventConsensusValue& actual = releaseActual.actual;

    return
        ValidValueShape(actual) &&
        forecast.valueKind == "scalar" &&
        actual.valueKind == "scalar" &&
        forecast.unit == actual.unit &&
        forecast.qualifier == actual.qualifier;
}


enum class SurpriseDisposition
{
    missingActual,
    notYetAvailable,
    missingForecast,
    incompatible,
    available,
};


template <typename MappedEvent>
SurpriseDisposition SetSurprise(
    EconomicEventFeatureValues& values,
    const MappedEvent& event,
    std::int64_t informationCutoffUnixMicros)
{
    if (!event.releaseActual)
        return SurpriseDisposition::missingActual;

    if (!(event.releaseActual->availableAtUnixMicros <
          informationCutoffUnixMicros))
        return SurpriseDisposition::notYetAvailable;

    if (!event.selectedConsensus)
        return SurpriseDisposition::missingForecast;

    if (!CompatibleScalarSurprise(
            *event.selectedConsensus,
            event.releaseActual->provenance))
        return SurpriseDisposition::incompatible;

    const double difference =
        event.releaseActual->provenance.actual.canonicalValueLow -
        event.selectedConsensus->forecast.canonicalValueLow;
    const float normalized = NormalizedFloat(event, difference);

    values.authoritativeInitialHasSurprise = 1.0F;
    values.authoritativeInitialSurprise = normalized;
    values.authoritativeInitialSurpriseAbs = std::abs(normalized);
    values.authoritativeInitialSurpriseDirection =
        normalized > 0.0F ? 1.0F : normalized < 0.0F ? -1.0F : 0.0F;
    return SurpriseDisposition::available;
}


template <typename MappedEvent>
CausalSurpriseObservation SetCausalFirstReleaseSurprise(
    EconomicEventFeatureValues& values,
    const MappedEvent& event,
    std::int64_t informationCutoffUnixMicros)
{
    CausalSurpriseObservation observation;
    observation.economicEventId = event.economicEventId;
    observation.eventTimestampUnixMicros = event.eventTimestampUnixMicros;
    observation.eventFamily = event.eventFamily;
    observation.sourceAgency = event.sourceAgency;
    observation.firstReleaseSelectionReason =
        event.firstReleaseActualSelectionReason;
    if (event.selectedConsensus)
        observation.consensusSource = event.selectedConsensus->provider;

    switch (event.firstReleaseActualState)
    {
        case EconomicEventFirstReleaseActualState::provenanceUnavailable:
            observation.disposition =
                CausalSurpriseDisposition::provenanceUnavailable;
            return observation;
        case EconomicEventFirstReleaseActualState::ambiguous:
            observation.disposition = CausalSurpriseDisposition::ambiguous;
            return observation;
        case EconomicEventFirstReleaseActualState::notYetAvailable:
            observation.disposition =
                CausalSurpriseDisposition::notYetAvailable;
            return observation;
        case EconomicEventFirstReleaseActualState::provenFirstRelease:
            break;
    }

    if (!event.firstReleaseActual)
        throw std::logic_error(
            "economic_event_causal_surprise_proven_value_missing");
    if (event.firstReleaseActual->provenAvailableAtUnixMicros >
        informationCutoffUnixMicros)
    {
        observation.disposition =
            CausalSurpriseDisposition::notYetAvailable;
        return observation;
    }
    if (!event.selectedConsensus)
    {
        observation.disposition =
            CausalSurpriseDisposition::missingForecast;
        return observation;
    }
    observation.incompatibilityReason =
        AssessCausalScalarSurpriseCompatibility(
            *event.selectedConsensus, *event.firstReleaseActual,
            event.eventFamily);
    if (observation.incompatibilityReason !=
        CausalSurpriseIncompatibilityReason::none)
    {
        observation.disposition = CausalSurpriseDisposition::incompatible;
        return observation;
    }

    const long double difference =
        static_cast<long double>(
            event.firstReleaseActual->actual.canonicalValueLow) -
        static_cast<long double>(
            event.selectedConsensus->forecast.canonicalValueLow);
    const long double normalizationScale =
        static_cast<long double>(EconomicEventNormalizationScale(
            event.eventFamily,
            event.selectedConsensus->forecast.unit));
    const long double normalized = difference / normalizationScale;
    if (!std::isfinite(normalized))
    {
        observation.disposition = CausalSurpriseDisposition::incompatible;
        observation.incompatibilityReason =
            CausalSurpriseIncompatibilityReason::normalizedValueNotFinite;
        return observation;
    }

    observation.lowerClamped =
        normalized < -static_cast<long double>(
                         kEconomicEventCausalSurpriseClamp);
    observation.upperClamped =
        normalized > static_cast<long double>(
                         kEconomicEventCausalSurpriseClamp);

    const float bounded = static_cast<float>(std::clamp(
        normalized,
        -static_cast<long double>(kEconomicEventCausalSurpriseClamp),
        static_cast<long double>(kEconomicEventCausalSurpriseClamp)));
    if (!std::isfinite(bounded))
    {
        observation.disposition = CausalSurpriseDisposition::incompatible;
        observation.incompatibilityReason =
            CausalSurpriseIncompatibilityReason::boundedValueNotFinite;
        observation.lowerClamped = false;
        observation.upperClamped = false;
        return observation;
    }

    values.causalFirstReleaseSurpriseAvailable = 1.0F;
    values.causalFirstReleaseSurprise = bounded;
    observation.disposition = CausalSurpriseDisposition::available;
    observation.surprise = bounded;
    observation.firstReleaseSource =
        event.firstReleaseActual->sourceName;
    return observation;
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
        authoritativeInitialHasSurprise,
        authoritativeInitialSurprise,
        authoritativeInitialSurpriseAbs,
        authoritativeInitialSurpriseDirection,
        causalFirstReleaseSurpriseAvailable,
        causalFirstReleaseSurprise,
    };
}


const char* CausalSurpriseIncompatibilityReasonText(
    CausalSurpriseIncompatibilityReason reason) noexcept
{
    switch (reason)
    {
        case CausalSurpriseIncompatibilityReason::none:
            return "none";
        case CausalSurpriseIncompatibilityReason::invalidForecastValueShape:
            return "invalid_forecast_value_shape";
        case CausalSurpriseIncompatibilityReason::invalidActualValueShape:
            return "invalid_actual_value_shape";
        case CausalSurpriseIncompatibilityReason::forecastNotScalar:
            return "forecast_not_scalar";
        case CausalSurpriseIncompatibilityReason::actualNotScalar:
            return "actual_not_scalar";
        case CausalSurpriseIncompatibilityReason::unitMismatch:
            return "unit_mismatch";
        case CausalSurpriseIncompatibilityReason::scaleMismatch:
            return "scale_mismatch";
        case CausalSurpriseIncompatibilityReason::qualifierMismatch:
            return "qualifier_mismatch";
        case CausalSurpriseIncompatibilityReason::
                unsupportedNormalizationFamilyOrUnit:
            return "unsupported_normalization_family_or_unit";
        case CausalSurpriseIncompatibilityReason::normalizedValueNotFinite:
            return "normalized_value_not_finite";
        case CausalSurpriseIncompatibilityReason::boundedValueNotFinite:
            return "bounded_value_not_finite";
    }
    return "unknown";
}


CausalSurpriseIncompatibilityReason
AssessCausalScalarSurpriseCompatibility(
    const EconomicEventSelectedConsensus& selected,
    const EconomicEventFirstReleaseActual& firstReleaseActual,
    std::string_view eventFamily) noexcept
{
    const EconomicEventConsensusValue& forecast = selected.forecast;
    const EconomicEventConsensusValue& actual = firstReleaseActual.actual;
    if (!ValidValueShape(forecast))
        return CausalSurpriseIncompatibilityReason::
            invalidForecastValueShape;
    if (!ValidValueShape(actual))
        return CausalSurpriseIncompatibilityReason::invalidActualValueShape;
    if (forecast.valueKind != "scalar")
        return CausalSurpriseIncompatibilityReason::forecastNotScalar;
    if (actual.valueKind != "scalar")
        return CausalSurpriseIncompatibilityReason::actualNotScalar;
    if (forecast.unit != actual.unit)
        return CausalSurpriseIncompatibilityReason::unitMismatch;
    if (forecast.scale != actual.scale)
        return CausalSurpriseIncompatibilityReason::scaleMismatch;
    if (forecast.qualifier != actual.qualifier)
        return CausalSurpriseIncompatibilityReason::qualifierMismatch;
    try
    {
        (void)EconomicEventNormalizationScale(eventFamily, forecast.unit);
    }
    catch (const std::invalid_argument&)
    {
        return CausalSurpriseIncompatibilityReason::
            unsupportedNormalizationFamilyOrUnit;
    }
    return CausalSurpriseIncompatibilityReason::none;
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
        canonicalEventFamily == "EMPLOYMENT_ANNUAL" ||
        canonicalEventFamily == "WEEKLY_CLAIMS")
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

        std::optional<MappedReleaseActual> releaseActual;
        if (event.releaseActual)
        {
            if (
                event.releaseActual->availableAtUnixMicros <
                    event.eventTimestampUnixMicros ||
                event.releaseActual->sourceAgency != event.sourceAgency ||
                event.releaseActual->sourceObservationId.empty() ||
                event.releaseActual->sourceArtifactPath.empty() ||
                !IsLowerHexSha256(
                    event.releaseActual->sourceArtifactSha256) ||
                event.releaseActual->semanticContract.empty() ||
                event.releaseActual->actual.unit.empty() ||
                !ValidValueShape(event.releaseActual->actual))
            {
                throw std::invalid_argument(
                    "economic_event_release_actual_provenance_invalid");
            }

            releaseActual = MappedReleaseActual{
                event.releaseActual->availableAtUnixMicros,
                *event.releaseActual};
        }

        const bool hasPitActual = event.firstReleaseActual.has_value();
        const bool stateIsProven = event.firstReleaseActualState ==
            EconomicEventFirstReleaseActualState::provenFirstRelease;
        if (hasPitActual != stateIsProven)
        {
            throw std::invalid_argument(
                "economic_event_first_release_actual_pit_state_value_mismatch");
        }
        if (event.firstReleaseActual)
        {
            const auto& actual = *event.firstReleaseActual;
            if (actual.provenAvailableAtUnixMicros <
                    event.eventTimestampUnixMicros ||
                actual.sourceName != event.sourceAgency ||
                actual.sourceObservationId.empty() ||
                actual.evidenceKey.empty() ||
                actual.actual.unit.empty() ||
                !ValidValueShape(actual.actual))
            {
                throw std::invalid_argument(
                    "economic_event_first_release_actual_pit_invalid");
            }
        }

        events_.push_back(
            MappedEvent{
                event.economicEventId,
                eventTime,
                event.eventTimestampUnixMicros,
                MapEconomicEventModelFamily(
                    event.sourceAgency,
                    event.eventFamily),
                event.eventFamily,
                event.sourceAgency,
                event.eventImportance,
                event.selectedConsensus,
                std::move(releaseActual),
                event.firstReleaseActualState,
                event.firstReleaseActualSelectionReason,
                event.firstReleaseActual});

        if (event.selectedConsensus)
        {
            if (event.selectedConsensus->provider.empty())
                throw std::invalid_argument(
                    "economic_event_consensus_provider_missing");
            const EconomicEventConsensusValue& forecast =
                event.selectedConsensus->forecast;
            if (!ValidValueShape(forecast))
                throw std::invalid_argument(
                    "economic_event_consensus_invalid_forecast_shape");

            (void)EconomicEventNormalizationScale(
                event.eventFamily,
                forecast.unit);
            (void)NormalizedFloat(events_.back(),
                                  forecast.canonicalValueLow);
            (void)NormalizedFloat(
                events_.back(),
                forecast.canonicalValueHigh.value_or(
                    forecast.canonicalValueLow));
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
    const std::int64_t informationCutoffUnixMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            informationCutoff.time_since_epoch()).count();

    EconomicEventFeatureValues values;
    lastCausalSurpriseObservation_ = CausalSurpriseObservation{};

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

    const MappedEvent* relevantEvent = nullptr;
    if (exactBoundaryEvent)
    {
        const std::size_t relevant =
            MostRelevantAtTimestamp(events_, nextEventIndex_);
        relevantEvent = &events_[relevant];
    }
    else if (mostRecentReleasedEventIndex_)
    {
        relevantEvent = &events_[*mostRecentReleasedEventIndex_];
    }

    ++diagnostics_.completedBarCount;
    if (relevantEvent)
    {
        ++diagnostics_.relevantEventRowCount;
        SetConsensus(values, *relevantEvent);
        if (!relevantEvent->selectedConsensus)
        {
            ++diagnostics_.missingConsensusRowCount;
        }
        else
        {
            ++diagnostics_.selectedConsensusRowCount;
            ++diagnostics_.selectedConsensusProviderRowCounts[
                relevantEvent->selectedConsensus->provider];
            if (relevantEvent->selectedConsensus->forecast.valueKind ==
                "range")
                ++diagnostics_.rangeConsensusRowCount;
            else
                ++diagnostics_.scalarConsensusRowCount;
        }

        if (relevantEvent->releaseActual)
        {
            ++diagnostics_.selectedInitialActualRowCount;
            ++diagnostics_.initialActualSourceRowCounts[
                relevantEvent->releaseActual->provenance.sourceAgency];
        }

        if (!exactBoundaryEvent)
        {
            switch (SetSurprise(
                values, *relevantEvent, informationCutoffUnixMicros))
            {
                case SurpriseDisposition::missingActual:
                    break;
                case SurpriseDisposition::notYetAvailable:
                    ++diagnostics_.notYetAvailableInitialActualRowCount;
                    break;
                case SurpriseDisposition::missingForecast:
                    // missingConsensusRowCount already describes this row;
                    // absence is not a semantic incompatibility.
                    break;
                case SurpriseDisposition::incompatible:
                    ++diagnostics_.incompatibleInitialActualRowCount;
                    break;
                case SurpriseDisposition::available:
                    ++diagnostics_.availableSurpriseRowCount;
                    break;
            }
        }
        else if (relevantEvent->releaseActual)
        {
            ++diagnostics_.notYetAvailableInitialActualRowCount;
        }

        lastCausalSurpriseObservation_ = SetCausalFirstReleaseSurprise(
            values, *relevantEvent, informationCutoffUnixMicros);
        switch (lastCausalSurpriseObservation_.disposition)
        {
            case CausalSurpriseDisposition::noRelevantEvent:
                throw std::logic_error(
                    "economic_event_causal_surprise_relevant_event_missing");
            case CausalSurpriseDisposition::provenanceUnavailable:
                ++diagnostics_.causalSurpriseProvenanceUnavailableRowCount;
                break;
            case CausalSurpriseDisposition::ambiguous:
                ++diagnostics_.causalSurpriseAmbiguousRowCount;
                break;
            case CausalSurpriseDisposition::notYetAvailable:
                ++diagnostics_.causalSurpriseNotYetAvailableRowCount;
                break;
            case CausalSurpriseDisposition::missingForecast:
                ++diagnostics_.causalSurpriseMissingConsensusRowCount;
                break;
            case CausalSurpriseDisposition::incompatible:
                ++diagnostics_.causalSurpriseIncompatibleRowCount;
                break;
            case CausalSurpriseDisposition::available:
                ++diagnostics_.causalSurpriseAvailableRowCount;
                ++diagnostics_.causalFirstReleaseSourceRowCounts[
                    relevantEvent->firstReleaseActual->sourceName];
                break;
        }
    }

    previousBarStart_ = barStart;
    return values;
}


std::size_t
EconomicEventFeatureEngine::ConsumedEventCount() const noexcept
{
    return nextEventIndex_;
}


const EconomicEventFeatureAvailabilityDiagnostics&
EconomicEventFeatureEngine::Diagnostics() const noexcept
{
    return diagnostics_;
}


const CausalSurpriseObservation&
EconomicEventFeatureEngine::LastCausalSurpriseObservation() const noexcept
{
    return lastCausalSurpriseObservation_;
}

} // namespace EA::EconomicCalendar
