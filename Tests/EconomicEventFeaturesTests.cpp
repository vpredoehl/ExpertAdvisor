#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
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
    static_assert(kEconomicEventModelFamilyCount == 5);
    static_assert(kEconomicEventFeatureWidth == 10);

    // All ten authoritative canonical families map explicitly.
    const std::array mappings{
        std::pair{"BLS:CPI", EconomicEventModelFamily::inflation},
        std::pair{"BLS:PPI", EconomicEventModelFamily::inflation},
        std::pair{"BEA:PCE", EconomicEventModelFamily::inflation},
        std::pair{"BLS:EMPLOYMENT", EconomicEventModelFamily::employment},
        std::pair{"BLS:EMPLOYMENT_ANNUAL", EconomicEventModelFamily::employment},
        std::pair{"BLS:JOLTS", EconomicEventModelFamily::employment},
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

        for (std::size_t index = 5; index < values.Ordered().size(); ++index)
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

        const std::array<float, kEconomicEventFeatureWidth> expected{
            1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
            6.0F, 7.0F, 8.0F, 9.0F, 10.0F};

        assert(values.Ordered() == expected);
    }

    // Separate instances produce bit-identical output for train/infer parity.
    {
        const std::vector<EconomicEvent> events{
            EventAt(kBase, "BLS", "EMPLOYMENT"),
            EventAt(kBase + 900, "CENSUS", "DURABLE_GOODS"),
        };

        EconomicEventFeatureEngine training{events};
        EconomicEventFeatureEngine inference{events};

        for (std::int64_t offset : {0LL, 900LL, 1800LL, 86400LL})
        {
            assert(
                training.AdvanceCompletedBar(At(kBase + offset)).Ordered() ==
                inference.AdvanceCompletedBar(At(kBase + offset)).Ordered());
        }
    }

    // Impossible canonical inputs and non-chronological use fail fast.
    AssertInvalidArgument(
        []
        {
            (void)MapEconomicEventModelFamily("BLS", "GDP");
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
