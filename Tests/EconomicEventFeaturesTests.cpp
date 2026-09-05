#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../Sources/EconomicEventFeatures.hpp"

using namespace EA::EconomicCalendar;

namespace
{

constexpr std::int64_t kBase = 1'700'000'000;

PriceTP At(std::int64_t seconds)
{
    return PriceTP{
        std::chrono::seconds{seconds}};
}


EconomicEvent EventAt(
    std::int64_t seconds,
    std::string agency,
    std::string family)
{
    EconomicEvent event;
    event.currency = "USD";
    event.sourceAgency = std::move(agency);
    event.eventFamily = std::move(family);
    event.eventTimestampUnixMicros =
        seconds * 1000000LL;
    return event;
}


EconomicEvent ScalarConsensusEventAt(
    std::int64_t seconds,
    std::string agency,
    std::string family,
    std::string provider,
    double forecast,
    std::optional<double> actual,
    std::string unit,
    double sourceScale,
    std::optional<std::string> qualifier = std::nullopt)
{
    EconomicEvent event = EventAt(
        seconds,
        std::move(agency),
        std::move(family));
    EconomicEventSelectedConsensus selected;
    selected.provider = std::move(provider);
    selected.forecast = EconomicEventConsensusValue{
        "scalar", forecast, std::nullopt, unit, sourceScale, qualifier};
    if (actual)
    {
        selected.unprovenProviderActual = EconomicEventConsensusValue{
            "scalar", *actual, std::nullopt, std::move(unit), sourceScale,
            std::move(qualifier)};
    }
    event.selectedConsensus = std::move(selected);
    return event;
}


EconomicEvent RangeConsensusEventAt(
    std::int64_t seconds,
    double forecastLow,
    double forecastHigh,
    double actualLow,
    double actualHigh)
{
    EconomicEvent event = EventAt(
        seconds, "FEDERAL_RESERVE", "FOMC");
    EconomicEventSelectedConsensus selected;
    selected.provider = "OANDA";
    selected.forecast = EconomicEventConsensusValue{
        "range", forecastLow, forecastHigh, "percent", 1.0, std::nullopt};
    selected.unprovenProviderActual = EconomicEventConsensusValue{
        "range", actualLow, actualHigh, "percent", 1.0, std::nullopt};
    event.selectedConsensus = std::move(selected);
    return event;
}


void AttachInitialActual(
    EconomicEvent& event,
    std::int64_t availableAtSeconds,
    double actual,
    std::string unit,
    double sourceScale,
    std::optional<std::string> qualifier = std::nullopt)
{
    EconomicEventReleaseActual releaseActual;
    releaseActual.actual = EconomicEventConsensusValue{
        "scalar", actual, std::nullopt, std::move(unit), sourceScale,
        std::move(qualifier)};
    releaseActual.availableAtUnixMicros =
        availableAtSeconds * 1000000LL;
    releaseActual.sourceAgency = event.sourceAgency;
    releaseActual.sourceObservationId =
        "fixture:initial:" + std::to_string(availableAtSeconds);
    releaseActual.sourceArtifactPath = "fixture/authoritative-release.html";
    releaseActual.sourceArtifactSha256 = std::string(64, 'a');
    releaseActual.semanticContract =
        "authoritative_initial_release_actual_v1";
    event.releaseActual = std::move(releaseActual);
}


void AttachPitFirstReleaseActual(
    EconomicEvent& event,
    std::int64_t provenAvailableAtSeconds,
    double actual,
    std::string unit,
    double sourceScale,
    std::optional<std::string> qualifier = std::nullopt)
{
    EconomicEventFirstReleaseActual firstRelease;
    firstRelease.actual = EconomicEventConsensusValue{
        "scalar", actual, std::nullopt, std::move(unit), sourceScale,
        std::move(qualifier)};
    firstRelease.provenAvailableAtUnixMicros =
        provenAvailableAtSeconds * 1000000LL;
    firstRelease.sourceName = event.sourceAgency;
    firstRelease.sourceObservationId =
        "fixture:pit-first-release:" +
        std::to_string(provenAvailableAtSeconds);
    firstRelease.evidenceKey =
        "fixture:pit-evidence:" +
        std::to_string(provenAvailableAtSeconds);
    event.firstReleaseActualState =
        EconomicEventFirstReleaseActualState::provenFirstRelease;
    event.firstReleaseActual = std::move(firstRelease);
}


bool Near(
    float actual,
    double expected,
    double tolerance = 1.0e-6)
{
    return
        std::abs(
            static_cast<double>(actual) -
            expected) <= tolerance;
}


void AssertAllZero(
    const EconomicEventFeatureValues& values)
{
    for (float value : values.Ordered())
        assert(value == 0.0F);
}


void AssertReservedSurpriseZero(
    const EconomicEventFeatureValues& values)
{
    assert(values.releasedEventHasSurprise == 0.0F);
    assert(values.releasedEventSurprise == 0.0F);
    assert(values.releasedEventSurpriseAbs == 0.0F);
    assert(values.releasedEventSurpriseDirection == 0.0F);
}


void AssertAuthoritativeSurpriseZero(
    const EconomicEventFeatureValues& values)
{
    assert(values.authoritativeInitialHasSurprise == 0.0F);
    assert(values.authoritativeInitialSurprise == 0.0F);
    assert(values.authoritativeInitialSurpriseAbs == 0.0F);
    assert(values.authoritativeInitialSurpriseDirection == 0.0F);
}


void AssertCausalFirstReleaseSurpriseUnavailable(
    const EconomicEventFeatureValues& values)
{
    assert(values.causalFirstReleaseSurpriseAvailable == 0.0F);
    assert(values.causalFirstReleaseSurprise == 0.0F);
}


template <typename Function>
void AssertInvalidArgument(Function&& function)
{
    bool threw = false;

    try
    {
        function();
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }

    assert(threw);
}

} // namespace


int main()
{
    assert(
        EconomicEventNormalizationScale("WEEKLY_CLAIMS", "count") ==
        kEconomicEmploymentNormalizationScale);

    static_assert(kEconomicEventModelFamilyCount == 5);
    static_assert(kPreConsensusEconomicEventFeatureWidth == 10);
    static_assert(kEconomicEventConsensusFeatureWidth == 8);
    static_assert(kEconomicEventReleaseActualFeatureWidth == 4);
    static_assert(kCausalEconomicEventSurpriseFeatureWidth == 2);
    static_assert(kEconomicEventFeatureWidth == 24);

    // All eleven authoritative canonical families map explicitly.
    const std::array mappings{
        std::pair{"BLS:CPI", EconomicEventModelFamily::inflation},
        std::pair{"BLS:PPI", EconomicEventModelFamily::inflation},
        std::pair{"BEA:PCE", EconomicEventModelFamily::inflation},
        std::pair{"BLS:EMPLOYMENT", EconomicEventModelFamily::employment},
        std::pair{"BLS:EMPLOYMENT_ANNUAL", EconomicEventModelFamily::employment},
        std::pair{"BLS:JOLTS", EconomicEventModelFamily::employment},
        std::pair{"DOL_ETA:WEEKLY_CLAIMS", EconomicEventModelFamily::employment},
        std::pair{"BEA:GDP", EconomicEventModelFamily::growth},
        std::pair{"CENSUS:DURABLE_GOODS", EconomicEventModelFamily::growth},
        std::pair{"FEDERAL_RESERVE:FOMC", EconomicEventModelFamily::fedPolicy},
        std::pair{"CENSUS:RETAIL_SALES", EconomicEventModelFamily::consumerDemand},
    };

    for (const auto& [canonical, expected] : mappings)
    {
        const std::string value{canonical};
        const std::size_t separator = value.find(':');
        assert(separator != std::string::npos);
        assert(
            MapEconomicEventModelFamily(
                value.substr(0, separator),
                value.substr(separator + 1)) ==
            expected);
    }

    // Weekly Claims is an explicit employment-family release. It activates
    // only employment occurrence/recency, has the missing-consensus encoding,
    // and cannot activate the reserved surprise channels.
    {
        EconomicEvent claims =
            EventAt(kBase + 100, "DOL_ETA", "WEEKLY_CLAIMS");
        claims.historicalTimeConfidence = "reconstructed";
        EconomicEventFeatureEngine engine{{claims}};
        const auto values = engine.AdvanceCompletedBar(At(kBase));

        assert(values.employmentEvent == 1.0F);
        assert(Near(
            values.employmentRecencyDecay,
            std::exp(-800.0 / 86400.0)));
        assert(values.inflationEvent == 0.0F);
        assert(values.growthEvent == 0.0F);
        assert(values.fedPolicyEvent == 0.0F);
        assert(values.consumerDemandEvent == 0.0F);
        assert(values.inflationRecencyDecay == 0.0F);
        assert(values.growthRecencyDecay == 0.0F);
        assert(values.fedPolicyRecencyDecay == 0.0F);
        assert(values.consumerDemandRecencyDecay == 0.0F);
        assert(values.relevantEventHasConsensus == 0.0F);
        assert(values.relevantEventConsensusLow == 0.0F);
        assert(values.relevantEventConsensusHigh == 0.0F);
        assert(values.relevantEventConsensusIsRange == 0.0F);
        AssertReservedSurpriseZero(values);
    }

    // A claims release exactly at a completed-bar cutoff is excluded from the
    // preceding bar and first activates the bar beginning at that boundary.
    {
        EconomicEventFeatureEngine engine{
            {EventAt(kBase + 900, "DOL_ETA", "WEEKLY_CLAIMS")}};

        AssertAllZero(engine.AdvanceCompletedBar(At(kBase)));
        const auto boundary = engine.AdvanceCompletedBar(At(kBase + 900));
        assert(boundary.employmentEvent == 1.0F);
        assert(Near(
            boundary.employmentRecencyDecay,
            std::exp(-900.0 / 86400.0)));
    }

    // A claims release learned across a market-data gap advances employment
    // recency without being relabeled as a later-bar occurrence.
    {
        EconomicEventFeatureEngine engine{
            {EventAt(
                kBase + 24 * 60 * 60,
                "DOL_ETA",
                "WEEKLY_CLAIMS")}};

        AssertAllZero(engine.AdvanceCompletedBar(At(kBase)));
        const auto postGap =
            engine.AdvanceCompletedBar(At(kBase + 48 * 60 * 60));
        assert(postGap.employmentEvent == 0.0F);
        assert(postGap.employmentRecencyDecay > 0.0F);
        assert(postGap.employmentRecencyDecay < 1.0F);
    }

    // No history and an exact future boundary are both all-zero.
    {
        EconomicEventFeatureEngine empty{{}};
        AssertAllZero(
            empty.AdvanceCompletedBar(
                At(kBase)));

        EconomicEventFeatureEngine future{
            {EventAt(kBase + 900, "BLS", "CPI")}};

        AssertAllZero(
            future.AdvanceCompletedBar(
                At(kBase)));

        assert(future.ConsumedEventCount() == 0);
    }

    // Exact-boundary event: excluded from the prior bar, included in the bar
    // beginning on that boundary, with elapsed time measured at bar close.
    {
        EconomicEventFeatureEngine engine{
            {EventAt(kBase + 900, "BLS", "CPI")}};

        AssertAllZero(
            engine.AdvanceCompletedBar(
                At(kBase)));

        const auto onBoundaryBar =
            engine.AdvanceCompletedBar(
                At(kBase + 900));

        assert(onBoundaryBar.inflationEvent == 1.0F);
        assert(
            Near(
                onBoundaryBar.inflationRecencyDecay,
                std::exp(-900.0 / 86400.0)));
        assert(engine.ConsumedEventCount() == 1);
    }

    // An event inside a bar activates that completed bar and approaches one
    // as the event approaches the close cutoff.
    {
        EconomicEventFeatureEngine engine{
            {EventAt(kBase + 899, "BEA", "GDP")}};

        const auto values =
            engine.AdvanceCompletedBar(
                At(kBase));

        assert(values.growthEvent == 1.0F);
        assert(
            Near(
                values.growthRecencyDecay,
                std::exp(-1.0 / 86400.0)));
    }

    // Exact 24-hour wall-clock decay and monotonicity.
    {
        EconomicEventFeatureEngine engine{
            {EventAt(kBase, "FEDERAL_RESERVE", "FOMC")}};

        const auto initial =
            engine.AdvanceCompletedBar(
                At(kBase));

        const auto oneHour =
            engine.AdvanceCompletedBar(
                At(kBase + 60 * 60 - 900));

        const auto twentyFourHours =
            engine.AdvanceCompletedBar(
                At(kBase + 24 * 60 * 60 - 900));

        const auto thirtyDays =
            engine.AdvanceCompletedBar(
                At(kBase + 30 * 24 * 60 * 60 - 900));

        assert(initial.fedPolicyRecencyDecay > oneHour.fedPolicyRecencyDecay);
        assert(oneHour.fedPolicyRecencyDecay > twentyFourHours.fedPolicyRecencyDecay);
        assert(twentyFourHours.fedPolicyRecencyDecay > thirtyDays.fedPolicyRecencyDecay);
        assert(
            Near(
                twentyFourHours.fedPolicyRecencyDecay,
                std::exp(-1.0)));
        assert(
            Near(
                thirtyDays.fedPolicyRecencyDecay,
                std::exp(-30.0),
                1.0e-18));
    }

    // A later source family mapped to the same model family replaces the
    // prior recency anchor.
    {
        EconomicEventFeatureEngine engine{
            {
                EventAt(kBase, "BLS", "CPI"),
                EventAt(kBase + 1800, "BEA", "PCE"),
            }};

        const auto first =
            engine.AdvanceCompletedBar(
                At(kBase));

        const auto between =
            engine.AdvanceCompletedBar(
                At(kBase + 900));

        const auto replacement =
            engine.AdvanceCompletedBar(
                At(kBase + 1800));

        assert(first.inflationEvent == 1.0F);
        assert(between.inflationEvent == 0.0F);
        assert(replacement.inflationEvent == 1.0F);
        assert(
            replacement.inflationRecencyDecay >
            between.inflationRecencyDecay);
        assert(
            Near(
                replacement.inflationRecencyDecay,
                std::exp(-900.0 / 86400.0)));
    }

    // Simultaneous events exercise every model-facing family. Multiple
    // canonical families mapping to one model family remain binary.
    {
        EconomicEventFeatureEngine engine{
            {
                EventAt(kBase, "BLS", "CPI"),
                EventAt(kBase, "BEA", "PCE"),
                EventAt(kBase, "BLS", "EMPLOYMENT_ANNUAL"),
                EventAt(kBase, "CENSUS", "DURABLE_GOODS"),
                EventAt(kBase, "FEDERAL_RESERVE", "FOMC"),
                EventAt(kBase, "CENSUS", "RETAIL_SALES"),
            }};

        const auto values =
            engine.AdvanceCompletedBar(
                At(kBase));

        assert(values.inflationEvent == 1.0F);
        assert(values.employmentEvent == 1.0F);
        assert(values.growthEvent == 1.0F);
        assert(values.fedPolicyEvent == 1.0F);
        assert(values.consumerDemandEvent == 1.0F);

        const float expected =
            static_cast<float>(
                std::exp(-900.0 / 86400.0));

        for (std::size_t index = 5;
             index < kPreConsensusEconomicEventFeatureWidth;
             ++index)
            assert(Near(values.Ordered()[index], expected));
    }

    // Incremental advancement consumes only newly causal events and cannot
    // leak a future event backward.
    {
        EconomicEventFeatureEngine engine{
            {
                EventAt(kBase + 100, "BLS", "JOLTS"),
                EventAt(kBase + 1800, "CENSUS", "RETAIL_SALES"),
                EventAt(kBase + 2700, "BEA", "GDP"),
            }};

        const auto first =
            engine.AdvanceCompletedBar(
                At(kBase));

        assert(first.employmentEvent == 1.0F);
        assert(first.consumerDemandEvent == 0.0F);
        assert(first.consumerDemandRecencyDecay == 0.0F);
        assert(engine.ConsumedEventCount() == 1);

        const auto second =
            engine.AdvanceCompletedBar(
                At(kBase + 900));

        assert(second.consumerDemandEvent == 0.0F);
        assert(engine.ConsumedEventCount() == 1);

        const auto third =
            engine.AdvanceCompletedBar(
                At(kBase + 1800));

        assert(third.consumerDemandEvent == 1.0F);
        assert(third.growthEvent == 0.0F);
        assert(engine.ConsumedEventCount() == 2);
    }

    // Events that occur during an observed-data gap update recency without
    // falsely claiming occurrence inside the first post-gap bar.
    {
        EconomicEventFeatureEngine engine{
            {EventAt(kBase + 24 * 60 * 60, "BLS", "PPI")}};

        AssertAllZero(
            engine.AdvanceCompletedBar(
                At(kBase)));

        const auto postGap =
            engine.AdvanceCompletedBar(
                At(kBase + 48 * 60 * 60));

        assert(postGap.inflationEvent == 0.0F);
        assert(postGap.inflationRecencyDecay > 0.0F);
        assert(postGap.inflationRecencyDecay < 1.0F);
    }

    // Stable order is explicit and cannot silently follow struct layout.
    {
        EconomicEventFeatureValues values;
        values.inflationEvent = 1.0F;
        values.employmentEvent = 2.0F;
        values.growthEvent = 3.0F;
        values.fedPolicyEvent = 4.0F;
        values.consumerDemandEvent = 5.0F;
        values.inflationRecencyDecay = 6.0F;
        values.employmentRecencyDecay = 7.0F;
        values.growthRecencyDecay = 8.0F;
        values.fedPolicyRecencyDecay = 9.0F;
        values.consumerDemandRecencyDecay = 10.0F;
        values.relevantEventHasConsensus = 11.0F;
        values.relevantEventConsensusLow = 12.0F;
        values.relevantEventConsensusHigh = 13.0F;
        values.relevantEventConsensusIsRange = 14.0F;
        values.releasedEventHasSurprise = 15.0F;
        values.releasedEventSurprise = 16.0F;
        values.releasedEventSurpriseAbs = 17.0F;
        values.releasedEventSurpriseDirection = 18.0F;
        values.authoritativeInitialHasSurprise = 19.0F;
        values.authoritativeInitialSurprise = 20.0F;
        values.authoritativeInitialSurpriseAbs = 21.0F;
        values.authoritativeInitialSurpriseDirection = 22.0F;
        values.causalFirstReleaseSurpriseAvailable = 23.0F;
        values.causalFirstReleaseSurprise = 24.0F;

        const std::array<float, kEconomicEventFeatureWidth> expected{
            1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
            6.0F, 7.0F, 8.0F, 9.0F, 10.0F,
            11.0F, 12.0F, 13.0F, 14.0F,
            15.0F, 16.0F, 17.0F, 18.0F,
            19.0F, 20.0F, 21.0F, 22.0F,
            23.0F, 24.0F};

        assert(values.Ordered() == expected);
    }


    // Provider identity is diagnostic-only: equal selected forecast semantics
    // from OANDA and Myfxbook produce bit-identical model features. The exact
    // boundary row knows the final consensus. Reserved surprise stays zero.
    {
        const EconomicEvent oanda = ScalarConsensusEventAt(
            kBase + 900, "BLS", "CPI", "OANDA", 0.3, std::nullopt,
            "percent", 1.0, "m/m");
        const EconomicEvent myfxbook = ScalarConsensusEventAt(
            kBase + 900, "BLS", "CPI", "MYFXBOOK", 0.3, std::nullopt,
            "percent", 1.0, "m/m");

        EconomicEventFeatureEngine oandaEngine{{oanda}};
        EconomicEventFeatureEngine myfxbookEngine{{myfxbook}};
        const auto oandaValues = oandaEngine.AdvanceCompletedBar(At(kBase));
        const auto myfxbookValues =
            myfxbookEngine.AdvanceCompletedBar(At(kBase));

        assert(oandaValues.Ordered() == myfxbookValues.Ordered());
        assert(oandaValues.relevantEventHasConsensus == 1.0F);
        assert(Near(oandaValues.relevantEventConsensusLow, 0.03));
        assert(Near(oandaValues.relevantEventConsensusHigh, 0.03));
        assert(oandaValues.relevantEventConsensusIsRange == 0.0F);
        AssertReservedSurpriseZero(oandaValues);
    }

    // A final forecast is not projected into arbitrarily early bars. It first
    // appears at the exact release cutoff. A populated historical OANDA actual
    // cannot activate surprise after release without first-release provenance.
    {
        const EconomicEvent event = ScalarConsensusEventAt(
            kBase + 1800, "BLS", "PPI", "OANDA", 0.2, 0.5,
            "percent", 1.0, "m/m");
        EconomicEventFeatureEngine engine{{event}};

        const auto tooEarly = engine.AdvanceCompletedBar(At(kBase));
        AssertAllZero(tooEarly);

        const auto exactRelease =
            engine.AdvanceCompletedBar(At(kBase + 900));
        assert(exactRelease.relevantEventHasConsensus == 1.0F);
        assert(Near(exactRelease.relevantEventConsensusLow, 0.02));
        AssertReservedSurpriseZero(exactRelease);

        const auto released =
            engine.AdvanceCompletedBar(At(kBase + 1800));
        assert(released.relevantEventHasConsensus == 1.0F);
        AssertReservedSurpriseZero(released);
    }

    // A separately persisted authoritative initial actual is still invisible
    // immediately before release and at the exact completed-bar cutoff. It
    // first activates surprise after its strict known-at boundary.
    {
        EconomicEvent event = ScalarConsensusEventAt(
            kBase + 1800, "BLS", "PPI", "OANDA", 0.2, 9.9,
            "percent", 1.0, "m/m");
        AttachInitialActual(
            event, kBase + 1800, 0.5, "percent", 1.0, "m/m");
        EconomicEventFeatureEngine engine{{event}};

        AssertAllZero(engine.AdvanceCompletedBar(At(kBase)));

        const auto exactRelease =
            engine.AdvanceCompletedBar(At(kBase + 900));
        assert(exactRelease.relevantEventHasConsensus == 1.0F);
        AssertReservedSurpriseZero(exactRelease);
        AssertAuthoritativeSurpriseZero(exactRelease);

        const auto postRelease =
            engine.AdvanceCompletedBar(At(kBase + 1800));
        AssertReservedSurpriseZero(postRelease);
        assert(postRelease.authoritativeInitialHasSurprise == 1.0F);
        assert(Near(postRelease.authoritativeInitialSurprise, 0.03));
        assert(Near(postRelease.authoritativeInitialSurpriseAbs, 0.03));
        assert(postRelease.authoritativeInitialSurpriseDirection == 1.0F);
    }

    // An initial value published after the scheduled event remains unavailable
    // through an exact availability cutoff. Ingestion/retrieval time is not
    // substituted for this explicit source-backed instant.
    {
        EconomicEvent event = ScalarConsensusEventAt(
            kBase, "CENSUS", "RETAIL_SALES", "MYFXBOOK", 0.4,
            std::nullopt, "percent", 1.0, "m/m");
        AttachInitialActual(
            event, kBase + 1800, -0.1, "percent", 1.0, "m/m");
        EconomicEventFeatureEngine engine{{event}};

        const auto before = engine.AdvanceCompletedBar(At(kBase));
        AssertReservedSurpriseZero(before);
        AssertAuthoritativeSurpriseZero(before);
        const auto exactAvailability =
            engine.AdvanceCompletedBar(At(kBase + 900));
        AssertReservedSurpriseZero(exactAvailability);
        AssertAuthoritativeSurpriseZero(exactAvailability);
        const auto after = engine.AdvanceCompletedBar(At(kBase + 1800));
        AssertReservedSurpriseZero(after);
        assert(after.authoritativeInitialHasSurprise == 1.0F);
        assert(Near(after.authoritativeInitialSurprise, -0.05));
        assert(Near(after.authoritativeInitialSurpriseAbs, 0.05));
        assert(after.authoritativeInitialSurpriseDirection == -1.0F);
        const auto& diagnostics = engine.Diagnostics();
        assert(diagnostics.notYetAvailableInitialActualRowCount == 2);
        assert(diagnostics.availableSurpriseRowCount == 1);
        assert(diagnostics.initialActualSourceRowCounts.at("CENSUS") == 3);
    }

    // Presence distinguishes a legitimate zero surprise from unavailable
    // surprise. Provider actuals still cannot compete with the authoritative
    // initial observation.
    {
        EconomicEvent event = ScalarConsensusEventAt(
            kBase, "BEA", "GDP", "OANDA", 2.0, 7.0,
            "percent", 1.0);
        AttachInitialActual(
            event, kBase, 2.0, "percent", 1.0);
        EconomicEventFeatureEngine engine{{event}};
        const auto values = engine.AdvanceCompletedBar(At(kBase));
        AssertReservedSurpriseZero(values);
        assert(values.authoritativeInitialHasSurprise == 1.0F);
        assert(values.authoritativeInitialSurprise == 0.0F);
        assert(values.authoritativeInitialSurpriseAbs == 0.0F);
        assert(values.authoritativeInitialSurpriseDirection == 0.0F);
    }

    // A provenance-certified actual with incompatible qualifier semantics
    // fails closed instead of manufacturing a surprise.
    {
        EconomicEvent event = ScalarConsensusEventAt(
            kBase, "BLS", "CPI", "OANDA", 0.3, std::nullopt,
            "percent", 1.0, "m/m");
        AttachInitialActual(
            event, kBase, 3.1, "percent", 1.0, "y/y");
        EconomicEventFeatureEngine engine{{event}};
        const auto values = engine.AdvanceCompletedBar(At(kBase));
        AssertReservedSurpriseZero(values);
        AssertAuthoritativeSurpriseZero(values);
        assert(engine.Diagnostics().incompatibleInitialActualRowCount == 1);
    }

    // Microsecond availability is compared without truncation. An actual one
    // microsecond after a completed-bar cutoff is not visible at that cutoff.
    {
        EconomicEvent event = ScalarConsensusEventAt(
            kBase, "BLS", "CPI", "OANDA", 0.2, std::nullopt,
            "percent", 1.0, "m/m");
        AttachInitialActual(
            event, kBase + 1800, 0.4, "percent", 1.0, "m/m");
        ++event.releaseActual->availableAtUnixMicros;
        EconomicEventFeatureEngine engine{{event}};

        AssertAuthoritativeSurpriseZero(
            engine.AdvanceCompletedBar(At(kBase)));
        AssertAuthoritativeSurpriseZero(
            engine.AdvanceCompletedBar(At(kBase + 900)));
        const auto after =
            engine.AdvanceCompletedBar(At(kBase + 1800));
        assert(after.authoritativeInitialHasSurprise == 1.0F);
        assert(Near(after.authoritativeInitialSurprise, 0.02));
    }

    // Unit, value-shape, and qualifier contracts must agree. Each mismatch is
    // unavailable rather than a numeric surprise or an exception.
    {
        std::vector<EconomicEvent> incompatible;

        EconomicEvent unit = ScalarConsensusEventAt(
            kBase, "BLS", "CPI", "OANDA", 0.3, std::nullopt,
            "percent", 1.0, "m/m");
        AttachInitialActual(unit, kBase, 400.0, "count", 1.0, "m/m");
        incompatible.push_back(std::move(unit));

        EconomicEvent range = ScalarConsensusEventAt(
            kBase, "BLS", "CPI", "OANDA", 0.3, std::nullopt,
            "percent", 1.0, "m/m");
        AttachInitialActual(range, kBase, 0.4, "percent", 1.0, "m/m");
        range.releaseActual->actual.valueKind = "range";
        range.releaseActual->actual.canonicalValueHigh = 0.5;
        incompatible.push_back(std::move(range));

        for (const EconomicEvent& event : incompatible)
        {
            EconomicEventFeatureEngine engine{{event}};
            const auto values = engine.AdvanceCompletedBar(At(kBase));
            AssertAuthoritativeSurpriseZero(values);
            assert(
                engine.Diagnostics().incompatibleInitialActualRowCount == 1);
        }
    }

    // Scale is the audited raw-to-canonical multiplier, not a second unit.
    // Different positive source encodings remain compatible after both have
    // been converted to the same canonical unit.
    {
        EconomicEvent event = ScalarConsensusEventAt(
            kBase, "BLS", "JOLTS", "MYFXBOOK", 4'530'000.0,
            std::nullopt, "count", 1.0);
        AttachInitialActual(
            event, kBase, 5'000'000.0, "count", 1000.0);
        EconomicEventFeatureEngine engine{{event}};
        const auto values = engine.AdvanceCompletedBar(At(kBase));
        assert(values.authoritativeInitialHasSurprise == 1.0F);
        assert(Near(values.authoritativeInitialSurprise, 0.047));
    }

    // A certified actual cannot substitute for a missing selected forecast,
    // and the absence is not mislabeled as semantic incompatibility.
    {
        EconomicEvent event = EventAt(kBase, "BLS", "CPI");
        AttachInitialActual(
            event, kBase, 0.4, "percent", 1.0, "m/m");
        EconomicEventFeatureEngine engine{{event}};
        const auto values = engine.AdvanceCompletedBar(At(kBase));
        AssertAuthoritativeSurpriseZero(values);
        assert(engine.Diagnostics().missingConsensusRowCount == 1);
        assert(engine.Diagnostics().incompatibleInitialActualRowCount == 0);
    }

    // Numeric zero remains present for either side of the subtraction.
    {
        EconomicEvent zeroForecast = ScalarConsensusEventAt(
            kBase, "BLS", "PPI", "OANDA", 0.0, std::nullopt,
            "percent", 1.0, "m/m");
        AttachInitialActual(
            zeroForecast, kBase, 0.5, "percent", 1.0, "m/m");
        EconomicEventFeatureEngine forecastEngine{{zeroForecast}};
        const auto forecastValues =
            forecastEngine.AdvanceCompletedBar(At(kBase));
        assert(forecastValues.authoritativeInitialHasSurprise == 1.0F);
        assert(Near(forecastValues.authoritativeInitialSurprise, 0.05));

        EconomicEvent zeroActual = ScalarConsensusEventAt(
            kBase, "BLS", "PPI", "OANDA", 0.5, std::nullopt,
            "percent", 1.0, "m/m");
        AttachInitialActual(
            zeroActual, kBase, 0.0, "percent", 1.0, "m/m");
        EconomicEventFeatureEngine actualEngine{{zeroActual}};
        const auto actualValues = actualEngine.AdvanceCompletedBar(At(kBase));
        assert(actualValues.authoritativeInitialHasSurprise == 1.0F);
        assert(Near(actualValues.authoritativeInitialSurprise, -0.05));
    }

    // Neither OANDA nor Myfxbook provider actual presence proves first-release
    // provenance. Positive, negative, and equal actual values are all ignored.
    {
        EconomicEventFeatureEngine negative{{ScalarConsensusEventAt(
            kBase, "CENSUS", "RETAIL_SALES", "OANDA", 0.4, -0.1,
            "percent", 1.0, "m/m")}};
        const auto negativeValues =
            negative.AdvanceCompletedBar(At(kBase));
        AssertReservedSurpriseZero(negativeValues);

        EconomicEventFeatureEngine zero{{ScalarConsensusEventAt(
            kBase, "BEA", "GDP", "OANDA", 2.0, 2.0,
            "percent", 1.0)}};
        const auto zeroValues = zero.AdvanceCompletedBar(At(kBase));
        AssertReservedSurpriseZero(zeroValues);

        EconomicEventFeatureEngine myfxbook{{ScalarConsensusEventAt(
            kBase, "BLS", "PPI", "MYFXBOOK", 0.2, 0.8,
            "percent", 1.0, "m/m")}};
        const auto myfxbookValues =
            myfxbook.AdvanceCompletedBar(At(kBase));
        assert(myfxbookValues.relevantEventHasConsensus == 1.0F);
        AssertReservedSurpriseZero(myfxbookValues);
    }

    // Fixed count-family scales are family-specific and dataset-independent.
    {
        EconomicEventFeatureEngine employment{{ScalarConsensusEventAt(
            kBase, "BLS", "EMPLOYMENT", "OANDA", 200'000.0, 250'000.0,
            "count", 1000.0)}};
        const auto employmentValues =
            employment.AdvanceCompletedBar(At(kBase));
        assert(Near(employmentValues.relevantEventConsensusLow, 0.2));
        AssertReservedSurpriseZero(employmentValues);

        EconomicEventFeatureEngine jolts{{ScalarConsensusEventAt(
            kBase, "BLS", "JOLTS", "MYFXBOOK", 9'250'000.0,
            std::nullopt, "count", 1.0)}};
        const auto joltsValues = jolts.AdvanceCompletedBar(At(kBase));
        assert(Near(joltsValues.relevantEventConsensusLow, 0.925));
        AssertReservedSurpriseZero(joltsValues);
    }

    // FOMC endpoints are preserved independently. Range surprise is explicitly
    // unavailable; no midpoint or interval subtraction is invented.
    {
        EconomicEventFeatureEngine fomc{{RangeConsensusEventAt(
            kBase, 5.25, 5.5, 5.0, 5.25)}};
        const auto values = fomc.AdvanceCompletedBar(At(kBase));
        assert(values.relevantEventHasConsensus == 1.0F);
        assert(values.relevantEventConsensusIsRange == 1.0F);
        assert(Near(values.relevantEventConsensusLow, 0.525));
        assert(Near(values.relevantEventConsensusHigh, 0.55));
        AssertReservedSurpriseZero(values);
    }

    // Provider actual semantics remain irrelevant even when populated. The
    // repository contract has no first-release provenance predicate to prove.
    {
        EconomicEvent incompatible = ScalarConsensusEventAt(
            kBase, "BLS", "CPI", "OANDA", 0.3, 0.4,
            "percent", 1.0, "m/m");
        incompatible.selectedConsensus->unprovenProviderActual->qualifier =
            "y/y";
        EconomicEventFeatureEngine engine{{incompatible}};
        const auto values = engine.AdvanceCompletedBar(At(kBase));
        assert(values.relevantEventHasConsensus == 1.0F);
        AssertReservedSurpriseZero(values);
    }

    // A genuine zero consensus is distinct from missing through the presence
    // bit; scalar low/high remain equal and no midpoint is invented.
    {
        EconomicEventFeatureEngine zeroConsensus{{ScalarConsensusEventAt(
            kBase, "BLS", "CPI", "OANDA", 0.0, std::nullopt,
            "percent", 1.0, "m/m")}};
        const auto values = zeroConsensus.AdvanceCompletedBar(At(kBase));
        assert(values.relevantEventHasConsensus == 1.0F);
        assert(values.relevantEventConsensusLow == 0.0F);
        assert(values.relevantEventConsensusHigh == 0.0F);
        assert(values.relevantEventConsensusIsRange == 0.0F);
        AssertReservedSurpriseZero(values);
    }

    // Missing selected consensus, including the two emergency FOMC release
    // dates, remains the all-zero missing encoding.
    for (const std::string& releaseDate : {"2020-03-03", "2020-03-15"})
    {
        EconomicEvent emergency =
            EventAt(kBase, "FEDERAL_RESERVE", "FOMC");
        emergency.sourceReleaseDate = releaseDate;
        EconomicEventFeatureEngine engine{{emergency}};
        const auto values = engine.AdvanceCompletedBar(At(kBase));
        assert(values.relevantEventHasConsensus == 0.0F);
        assert(values.relevantEventConsensusLow == 0.0F);
        assert(values.relevantEventConsensusHigh == 0.0F);
        AssertReservedSurpriseZero(values);
    }

    // Phase-1 PIT semantics are inclusive. A proved first release remains
    // unavailable immediately before publication, appears exactly at its
    // proven boundary, and remains the same afterward.
    {
        EconomicEvent event = ScalarConsensusEventAt(
            kBase, "BLS", "CPI", "OANDA", 0.2, std::nullopt,
            "percent", 1.0, "m/m");
        AttachPitFirstReleaseActual(
            event, kBase + 1800, 0.5, "percent", 1.0, "m/m");
        EconomicEventFeatureEngine engine{{event}};

        const auto before = engine.AdvanceCompletedBar(At(kBase));
        AssertCausalFirstReleaseSurpriseUnavailable(before);
        const auto atBoundary =
            engine.AdvanceCompletedBar(At(kBase + 900));
        assert(atBoundary.causalFirstReleaseSurpriseAvailable == 1.0F);
        assert(Near(atBoundary.causalFirstReleaseSurprise, 0.03));
        const auto after =
            engine.AdvanceCompletedBar(At(kBase + 1800));
        assert(after.causalFirstReleaseSurpriseAvailable == 1.0F);
        assert(after.causalFirstReleaseSurprise ==
               atBoundary.causalFirstReleaseSurprise);
        assert(engine.Diagnostics().causalSurpriseNotYetAvailableRowCount == 1);
        assert(engine.Diagnostics().causalSurpriseAvailableRowCount == 2);
    }

    // Missing/provenance-unavailable, not-yet-available, and ambiguous states
    // all fail closed, but remain independently visible in diagnostics.
    {
        EconomicEvent unavailable = ScalarConsensusEventAt(
            kBase, "BLS", "CPI", "OANDA", 0.2, std::nullopt,
            "percent", 1.0, "m/m");
        EconomicEventFeatureEngine unavailableEngine{{unavailable}};
        AssertCausalFirstReleaseSurpriseUnavailable(
            unavailableEngine.AdvanceCompletedBar(At(kBase)));
        assert(unavailableEngine.Diagnostics()
                   .causalSurpriseProvenanceUnavailableRowCount == 1);

        EconomicEvent notYet = unavailable;
        notYet.firstReleaseActualState =
            EconomicEventFirstReleaseActualState::notYetAvailable;
        EconomicEventFeatureEngine notYetEngine{{notYet}};
        AssertCausalFirstReleaseSurpriseUnavailable(
            notYetEngine.AdvanceCompletedBar(At(kBase)));
        assert(notYetEngine.Diagnostics()
                   .causalSurpriseNotYetAvailableRowCount == 1);

        EconomicEvent ambiguous = unavailable;
        ambiguous.firstReleaseActualState =
            EconomicEventFirstReleaseActualState::ambiguous;
        EconomicEventFeatureEngine ambiguousEngine{{ambiguous}};
        AssertCausalFirstReleaseSurpriseUnavailable(
            ambiguousEngine.AdvanceCompletedBar(At(kBase)));
        assert(ambiguousEngine.Diagnostics()
                   .causalSurpriseAmbiguousRowCount == 1);
    }

    // A proved first release still has no usable surprise when the selected
    // consensus association is absent.
    {
        EconomicEvent missingConsensus = EventAt(kBase, "BLS", "CPI");
        AttachPitFirstReleaseActual(
            missingConsensus, kBase, 0.4, "percent", 1.0, "m/m");
        EconomicEventFeatureEngine engine{{missingConsensus}};
        AssertCausalFirstReleaseSurpriseUnavailable(
            engine.AdvanceCompletedBar(At(kBase)));
        assert(engine.Diagnostics()
                   .causalSurpriseMissingConsensusRowCount == 1);
    }

    // Genuine zero surprise differs from missing actual through the explicit
    // availability channel.
    {
        EconomicEvent zero = ScalarConsensusEventAt(
            kBase, "BEA", "GDP", "OANDA", 2.0, std::nullopt,
            "percent", 1.0);
        AttachPitFirstReleaseActual(
            zero, kBase, 2.0, "percent", 1.0);
        EconomicEventFeatureEngine zeroEngine{{zero}};
        const auto values = zeroEngine.AdvanceCompletedBar(At(kBase));
        assert(values.causalFirstReleaseSurpriseAvailable == 1.0F);
        assert(values.causalFirstReleaseSurprise == 0.0F);

        EconomicEvent missing = ScalarConsensusEventAt(
            kBase, "BEA", "GDP", "OANDA", 2.0, std::nullopt,
            "percent", 1.0);
        EconomicEventFeatureEngine missingEngine{{missing}};
        const auto missingValues =
            missingEngine.AdvanceCompletedBar(At(kBase));
        AssertCausalFirstReleaseSurpriseUnavailable(missingValues);
        assert(values.Ordered() != missingValues.Ordered());
    }

    // Unit, scale, qualifier, and scalar/range shape must all agree. Phase 2
    // intentionally defines no range midpoint or interval subtraction.
    {
        std::vector<EconomicEvent> incompatible;
        for (const int kind : {0, 1, 2, 3, 4})
        {
            EconomicEvent event = ScalarConsensusEventAt(
                kBase, "BLS", "CPI", "OANDA", 0.2, std::nullopt,
                "percent", 1.0, "m/m");
            AttachPitFirstReleaseActual(
                event, kBase, 0.4, "percent", 1.0, "m/m");
            if (kind == 0) event.firstReleaseActual->actual.unit = "count";
            if (kind == 1) event.firstReleaseActual->actual.scale = 100.0;
            if (kind == 2) event.firstReleaseActual->actual.qualifier = "y/y";
            if (kind == 3)
            {
                event.firstReleaseActual->actual.valueKind = "range";
                event.firstReleaseActual->actual.canonicalValueHigh = 0.5;
            }
            if (kind == 4)
            {
                event.selectedConsensus->forecast.valueKind = "range";
                event.selectedConsensus->forecast.canonicalValueHigh = 0.3;
            }
            incompatible.push_back(std::move(event));
        }
        for (const auto& event : incompatible)
        {
            EconomicEventFeatureEngine engine{{event}};
            AssertCausalFirstReleaseSurpriseUnavailable(
                engine.AdvanceCompletedBar(At(kBase)));
            assert(engine.Diagnostics().causalSurpriseIncompatibleRowCount == 1);
        }
    }

    // Fixed family scaling is independent of future observations. Extreme
    // finite values use the project's standard normalized-feature ±10 bound.
    {
        EconomicEvent historical = ScalarConsensusEventAt(
            kBase, "BLS", "JOLTS", "MYFXBOOK", 4'500'000.0,
            std::nullopt, "count", 1.0);
        AttachPitFirstReleaseActual(
            historical, kBase, 5'000'000.0, "count", 1.0);
        EconomicEvent future = ScalarConsensusEventAt(
            kBase + 86400, "BLS", "JOLTS", "MYFXBOOK", 1'000'000.0,
            std::nullopt, "count", 1.0);
        AttachPitFirstReleaseActual(
            future, kBase + 86400, 9'000'000.0, "count", 1.0);

        EconomicEventFeatureEngine historicalOnly{{historical}};
        EconomicEventFeatureEngine withFuture{{historical, future}};
        const auto baseline = historicalOnly.AdvanceCompletedBar(At(kBase));
        const auto futureCorpus = withFuture.AdvanceCompletedBar(At(kBase));
        assert(baseline.causalFirstReleaseSurpriseAvailable == 1.0F);
        assert(Near(baseline.causalFirstReleaseSurprise, 0.05));
        assert(baseline.Ordered() == futureCorpus.Ordered());

        EconomicEvent extreme = ScalarConsensusEventAt(
            kBase, "BLS", "CPI", "OANDA", 0.0, std::nullopt,
            "percent", 1.0, "m/m");
        AttachPitFirstReleaseActual(
            extreme, kBase, 1.0e100, "percent", 1.0, "m/m");
        EconomicEventFeatureEngine extremeEngine{{extreme}};
        const auto extremeValues =
            extremeEngine.AdvanceCompletedBar(At(kBase));
        assert(extremeValues.causalFirstReleaseSurpriseAvailable == 1.0F);
        assert(extremeValues.causalFirstReleaseSurprise == 10.0F);
    }

    // Separate instances produce byte-identical consensus and reserved-zero
    // surprise output for train/infer parity.
    {
        const std::vector<EconomicEvent> events{
            ScalarConsensusEventAt(
                kBase, "BLS", "EMPLOYMENT", "OANDA", 200'000.0,
                250'000.0, "count", 1000.0),
            ScalarConsensusEventAt(
                kBase + 900, "CENSUS", "DURABLE_GOODS", "MYFXBOOK",
                0.4, std::nullopt, "percent", 1.0, "m/m"),
        };

        EconomicEventFeatureEngine training{events};
        EconomicEventFeatureEngine inference{events};

        for (std::int64_t offset : {0LL, 900LL, 1800LL, 86400LL})
        {
            assert(
                training.AdvanceCompletedBar(At(kBase + offset)).Ordered() ==
                inference.AdvanceCompletedBar(At(kBase + offset)).Ordered());
        }
        assert(training.Diagnostics() == inference.Diagnostics());
        const auto& diagnostics = training.Diagnostics();
        assert(diagnostics.completedBarCount == 4);
        assert(diagnostics.relevantEventRowCount == 4);
        assert(diagnostics.selectedConsensusRowCount == 4);
        assert(diagnostics.scalarConsensusRowCount == 4);
        assert(diagnostics.rangeConsensusRowCount == 0);
        assert(diagnostics.missingConsensusRowCount == 0);
        assert(diagnostics.selectedConsensusProviderRowCounts.count("OANDA") == 0);
        assert(diagnostics.selectedConsensusProviderRowCounts.at("MYFXBOOK") == 4);
    }

    // Selected consensus must preserve source identity, and normalized model
    // channels must be representable as finite floats.
    AssertInvalidArgument(
        []
        {
            EconomicEvent missingProvider = ScalarConsensusEventAt(
                kBase, "BLS", "CPI", "", 0.3, std::nullopt,
                "percent", 1.0, "m/m");
            EconomicEventFeatureEngine engine{{missingProvider}};
            (void)engine;
        });

    AssertInvalidArgument(
        []
        {
            EconomicEvent event = ScalarConsensusEventAt(
                kBase, "BLS", "CPI", "OANDA", 0.3, std::nullopt,
                "percent", 1.0, "m/m");
            AttachInitialActual(
                event, kBase - 1, 0.4, "percent", 1.0, "m/m");
            EconomicEventFeatureEngine engine{{event}};
            (void)engine;
        });

    AssertInvalidArgument(
        []
        {
            EconomicEvent event = ScalarConsensusEventAt(
                kBase, "BLS", "CPI", "OANDA", 0.3, std::nullopt,
                "percent", 1.0, "m/m");
            AttachInitialActual(
                event, kBase, 0.4, "percent", 1.0, "m/m");
            event.releaseActual->sourceAgency = "BEA";
            EconomicEventFeatureEngine engine{{event}};
            (void)engine;
        });

    AssertInvalidArgument(
        []
        {
            EconomicEvent overflow = ScalarConsensusEventAt(
                kBase, "BLS", "CPI", "OANDA",
                std::numeric_limits<double>::max(), std::nullopt,
                "percent", 1.0, "m/m");
            EconomicEventFeatureEngine engine{{overflow}};
            (void)engine;
        });

    // Impossible canonical inputs and non-chronological use fail fast.
    AssertInvalidArgument(
        []
        {
            (void)MapEconomicEventModelFamily("BLS", "GDP");
        });

    AssertInvalidArgument(
        []
        {
            (void)MapEconomicEventModelFamily("DOL_ETA", "UNKNOWN");
        });

    AssertInvalidArgument(
        []
        {
            EconomicEventFeatureEngine engine{
                {
                    EventAt(kBase + 1, "BLS", "CPI"),
                    EventAt(kBase, "BLS", "PPI"),
                }};
            (void)engine;
        });

    AssertInvalidArgument(
        []
        {
            EconomicEvent malformed =
                EventAt(kBase, "BLS", "CPI");
            ++malformed.eventTimestampUnixMicros;
            EconomicEventFeatureEngine engine{{malformed}};
            (void)engine;
        });

    AssertInvalidArgument(
        []
        {
            EconomicEventFeatureEngine engine{{}};
            (void)engine.AdvanceCompletedBar(At(kBase));
            (void)engine.AdvanceCompletedBar(At(kBase));
        });

    std::cout
        << "ECONOMIC_EVENT_FEATURES_TEST_PASS"
        << ",canonical_mappings="
        << mappings.size()
        << ",feature_width="
        << kEconomicEventFeatureWidth
        << '\n';

    return 0;
}
