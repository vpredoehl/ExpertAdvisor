#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "ModelInputContract.hpp"
#include "PricePoint.hpp"
#include "Tensor.hpp"

using namespace EA::EconomicCalendar;

namespace
{

constexpr std::int64_t kBase = 1'700'000'000;

PriceTP At(std::int64_t seconds)
{
    return PriceTP{std::chrono::seconds{seconds}};
}

EconomicEvent EventAt(std::int64_t seconds,
                      std::string agency,
                      std::string family)
{
    EconomicEvent event;
    event.currency = "USD";
    event.sourceAgency = std::move(agency);
    event.eventFamily = std::move(family);
    event.eventTimestampUnixMicros = seconds * 1'000'000LL;
    return event;
}

Feature BarAt(std::int64_t seconds, std::size_t index)
{
    const float base = 1.0F + static_cast<float>(index) * 0.001F;
    return Feature{base, base + 0.0002F, base + 0.0005F,
                   base - 0.0004F, At(seconds),
                   100.0F + static_cast<float>(index)};
}

std::array<float, feature_size> Row(const Tensor& tensor, std::size_t index)
{
    const auto access = MetaNN::LowerAccess(
        *(tensor.begin() + static_cast<std::ptrdiff_t>(index)));
    std::array<float, feature_size> values{};
    std::memcpy(values.data(), access.RawMemory(), values.size() * sizeof(float));
    return values;
}

bool Near(float actual, double expected, double tolerance = 1.0e-6)
{
    return std::abs(static_cast<double>(actual) - expected) <= tolerance;
}

} // namespace

int main()
{
    static_assert(return_autocorrelation_feature_size == 49);
    static_assert(economicEventFeatureStartCol == 49);
    static_assert(feature_size == 59);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kCurrentModelInputWidth == 63);
    static_assert(EA::kCurrentModelInputWidth ==
                  EA::kPreEconomicEventModelInputWidth +
                  kEconomicEventFeatureWidth);

    // Pre-window state is reconstructed from authoritative prior rows. The
    // first requested bar has no occurrence indicator but has exact nonzero
    // recency for two independent model families.
    const std::vector<EconomicEvent> preWindowEvents{
        EventAt(kBase - 3600, "BLS", "CPI"),
        EventAt(kBase - 1800, "BLS", "EMPLOYMENT"),
    };
    Tensor preWindow{"pre-window", kDefaultDonchian20Mode,
                     kDefaultDonchianLookback, preWindowEvents};
    preWindow.Add(BarAt(kBase, 0));
    const auto first = Row(preWindow, 0);
    assert(first[inflationEventCol] == 0.0F);
    assert(first[employmentEventCol] == 0.0F);
    assert(Near(first[inflationRecencyDecayCol],
                std::exp(-4500.0 / 86400.0)));
    assert(Near(first[employmentRecencyDecayCol],
                std::exp(-2700.0 / 86400.0)));

    // Compare an event-enabled Tensor to the unchanged prefix produced by the
    // pre-Phase-2 pipeline. Only the final ten columns may differ.
    const std::vector<EconomicEvent> causalEvents{
        EventAt(kBase + 100, "FEDERAL_RESERVE", "FOMC"),
        EventAt(kBase + 900, "BEA", "GDP"),
        EventAt(kBase + 2700, "CENSUS", "RETAIL_SALES"),
    };
    Tensor withEvents{"with-events", kDefaultDonchian20Mode,
                      kDefaultDonchianLookback, causalEvents};
    Tensor withoutEvents{"without-events"};
    Tensor inferenceParity{"inference-parity", kDefaultDonchian20Mode,
                           kDefaultDonchianLookback, causalEvents};
    for (std::size_t index = 0; index < 4; ++index)
    {
        const Feature bar = BarAt(kBase + static_cast<std::int64_t>(index) * 900,
                                  index);
        withEvents.Add(bar);
        withoutEvents.Add(bar);
        inferenceParity.Add(bar);

        const auto trainingRow = Row(withEvents, index);
        const auto inferenceRow = Row(inferenceParity, index);
        assert(std::memcmp(trainingRow.data(), inferenceRow.data(),
                           feature_size * sizeof(float)) == 0);

        const auto noEventRow = Row(withoutEvents, index);
        assert(std::memcmp(trainingRow.data(), noEventRow.data(),
                           economicEventFeatureStartCol * sizeof(float)) == 0);
    }

    const auto bar0 = Row(withEvents, 0);
    const auto bar1 = Row(withEvents, 1);
    const auto bar2 = Row(withEvents, 2);
    const auto bar3 = Row(withEvents, 3);
    assert(bar0[fedPolicyEventCol] == 1.0F);       // inside the completed bar
    assert(bar0[growthEventCol] == 0.0F);          // exact cutoff is excluded
    assert(bar1[growthEventCol] == 1.0F);          // bar-start event is included
    assert(bar2[consumerDemandEventCol] == 0.0F);  // future event cannot leak
    assert(bar3[consumerDemandEventCol] == 1.0F);

    // A market-gap event advances recency without becoming a false current-bar
    // occurrence on the next observed bar.
    Tensor gap{"gap", kDefaultDonchian20Mode, kDefaultDonchianLookback,
               {EventAt(kBase + 86400, "BLS", "PPI")}};
    gap.Add(BarAt(kBase, 0));
    gap.Add(BarAt(kBase + 2 * 86400, 1));
    const auto postGap = Row(gap, 1);
    assert(postGap[inflationEventCol] == 0.0F);
    assert(postGap[inflationRecencyDecayCol] > 0.0F);
    assert(postGap[inflationRecencyDecayCol] < 1.0F);

    // Persisted width 53 sees the exact historical Tensor prefix. New width 63
    // sees the same prefix followed by the documented ten-feature order.
    const auto oldContract = EA::ResolveModelInputContract(
        EA::kPreEconomicEventModelInputWidth, feature_size);
    const auto newContract = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, feature_size);
    assert(oldContract.tensorFeatureCount == economicEventFeatureStartCol);
    assert(newContract.tensorFeatureCount == feature_size);
    std::array<float, EA::kCurrentModelInputWidth> modelInput{};
    EA::CopyTensorFeaturesForModelInput(modelInput.data(), bar1.data(),
                                        newContract);
    assert(modelInput[inflationEventCol] == bar1[inflationEventCol]);
    assert(modelInput[employmentEventCol] == bar1[employmentEventCol]);
    assert(modelInput[growthEventCol] == bar1[growthEventCol]);
    assert(modelInput[fedPolicyEventCol] == bar1[fedPolicyEventCol]);
    assert(modelInput[consumerDemandEventCol] == bar1[consumerDemandEventCol]);
    assert(modelInput[inflationRecencyDecayCol] ==
           bar1[inflationRecencyDecayCol]);
    assert(modelInput[consumerDemandRecencyDecayCol] ==
           bar1[consumerDemandRecencyDecayCol]);

    return 0;
}
