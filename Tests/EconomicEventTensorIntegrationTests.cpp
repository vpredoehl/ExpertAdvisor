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

EconomicEvent ConsensusEventAt(std::int64_t seconds,
                               std::string agency,
                               std::string family,
                               double forecast,
                               double actual)
{
    EconomicEvent event = EventAt(
        seconds, std::move(agency), std::move(family));
    EconomicEventSelectedConsensus selected;
    selected.provider = "OANDA";
    selected.forecast = EconomicEventConsensusValue{
        "scalar", forecast, std::nullopt, "percent", 1.0, std::nullopt};
    selected.unprovenProviderActual = EconomicEventConsensusValue{
        "scalar", actual, std::nullopt, "percent", 1.0, std::nullopt};
    event.selectedConsensus = std::move(selected);
    return event;
}

EconomicEvent ProvenConsensusEventAt(std::int64_t seconds,
                                     std::string agency,
                                     std::string family,
                                     double forecast,
                                     double actual)
{
    EconomicEvent event = ConsensusEventAt(
        seconds, std::move(agency), std::move(family), forecast, actual);
    EconomicEventReleaseActual releaseActual;
    releaseActual.actual = EconomicEventConsensusValue{
        "scalar", actual, std::nullopt, "percent", 1.0, std::nullopt};
    releaseActual.availableAtUnixMicros = seconds * 1'000'000LL;
    releaseActual.sourceAgency = event.sourceAgency;
    releaseActual.sourceObservationId = "fixture:initial";
    releaseActual.sourceArtifactPath = "fixture/release.html";
    releaseActual.sourceArtifactSha256 = std::string(64, 'a');
    releaseActual.semanticContract = "fixture_initial_actual_v1";
    event.releaseActual = std::move(releaseActual);
    return event;
}

EconomicEvent RangeConsensusEventAt(std::int64_t seconds,
                                    double forecastLow,
                                    double forecastHigh)
{
    EconomicEvent event = EventAt(
        seconds, "FEDERAL_RESERVE", "FOMC");
    EconomicEventSelectedConsensus selected;
    selected.provider = "OANDA";
    selected.forecast = EconomicEventConsensusValue{
        "range", forecastLow, forecastHigh, "percent", 1.0, std::nullopt};
    event.selectedConsensus = std::move(selected);
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

std::array<float, EA::kCurrentModelInputWidth> Project(
    const std::array<float, feature_size>& row,
    const EA::FeatureAblationMask& mask = {})
{
    const auto contract = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, feature_size);
    std::array<float, EA::kCurrentModelInputWidth> result{};
    EA::CopyTensorFeaturesForModelInput(
        result.data(), row.data(), contract, mask);
    return result;
}

void AssertConsensusControlParity(
    const std::array<float, EA::kCurrentModelInputWidth>& treatment,
    const std::array<float, EA::kCurrentModelInputWidth>& control)
{
    for (std::size_t col = 0; col < relevantEventHasConsensusCol; ++col)
        assert(control[col] == treatment[col]);
    for (std::size_t col = relevantEventHasConsensusCol;
         col <= relevantEventConsensusIsRangeCol; ++col)
        assert(control[col] == 0.0F);
    for (std::size_t col = releasedEventHasSurpriseCol;
         col < EA::kCurrentModelInputWidth; ++col)
        assert(control[col] == treatment[col]);
}

} // namespace

int main()
{
    static_assert(return_autocorrelation_feature_size == 49);
    static_assert(economicEventFeatureStartCol == 49);
    static_assert(pre_consensus_economic_event_feature_size == 59);
    static_assert(consensus_economic_event_feature_size == 67);
    static_assert(feature_size == 71);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kEconomicEventModelInputWidth == 63);
    static_assert(EA::kEconomicEventConsensusModelInputWidth == 71);
    static_assert(EA::kCurrentModelInputWidth == 75);
    static_assert(EA::kCurrentModelInputWidth ==
                  EA::kEconomicEventModelInputWidth +
                  kEconomicEventConsensusFeatureWidth +
                  kEconomicEventReleaseActualFeatureWidth);

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
    // pre-economic-event pipeline. Only the final eighteen columns may differ.
    const std::vector<EconomicEvent> causalEvents{
        EventAt(kBase + 100, "FEDERAL_RESERVE", "FOMC"),
        ProvenConsensusEventAt(kBase + 900, "BEA", "GDP", 2.0, 2.5),
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
    assert(bar0[relevantEventHasConsensusCol] == 1.0F);
    assert(Near(bar0[relevantEventConsensusLowCol], 0.2));
    assert(bar0[releasedEventHasSurpriseCol] == 0.0F);
    assert(bar1[growthEventCol] == 1.0F);          // bar-start event is included
    assert(bar1[releasedEventHasSurpriseCol] == 0.0F);
    assert(bar1[releasedEventSurpriseCol] == 0.0F);
    assert(bar1[releasedEventSurpriseAbsCol] == 0.0F);
    assert(bar1[releasedEventSurpriseDirectionCol] == 0.0F);
    assert(bar1[authoritativeInitialHasSurpriseCol] == 1.0F);
    assert(Near(bar1[authoritativeInitialSurpriseCol], 0.05));
    assert(Near(bar1[authoritativeInitialSurpriseAbsCol], 0.05));
    assert(bar1[authoritativeInitialSurpriseDirectionCol] == 1.0F);
    assert(bar2[consumerDemandEventCol] == 0.0F);  // future event cannot leak
    assert(bar3[consumerDemandEventCol] == 1.0F);

    // A Weekly Claims market-gap event advances employment recency without
    // becoming a false current-bar occurrence on the next observed bar.
    Tensor gap{"gap", kDefaultDonchian20Mode, kDefaultDonchianLookback,
               {EventAt(kBase + 86400, "DOL_ETA", "WEEKLY_CLAIMS")}};
    gap.Add(BarAt(kBase, 0));
    gap.Add(BarAt(kBase + 2 * 86400, 1));
    const auto postGap = Row(gap, 1);
    assert(postGap[employmentEventCol] == 0.0F);
    assert(postGap[employmentRecencyDecayCol] > 0.0F);
    assert(postGap[employmentRecencyDecayCol] < 1.0F);

    // Persisted widths 53 and 63 retain their exact historical prefixes. New
    // Width 71 appends the documented eight consensus components and width 75
    // appends authoritative-initial surprise without changing either prefix.
    const auto preEventContract = EA::ResolveModelInputContract(
        EA::kPreEconomicEventModelInputWidth, feature_size);
    const auto oldContract = EA::ResolveModelInputContract(
        EA::kEconomicEventModelInputWidth, feature_size);
    const auto newContract = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, feature_size);
    assert(preEventContract.tensorFeatureCount == economicEventFeatureStartCol);
    assert(oldContract.tensorFeatureCount ==
           pre_consensus_economic_event_feature_size);
    const auto consensusContract = EA::ResolveModelInputContract(
        EA::kEconomicEventConsensusModelInputWidth, feature_size);
    assert(consensusContract.tensorFeatureCount ==
           consensus_economic_event_feature_size);
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
    assert(modelInput[relevantEventHasConsensusCol] ==
           bar1[relevantEventHasConsensusCol]);
    assert(modelInput[relevantEventConsensusLowCol] ==
           bar1[relevantEventConsensusLowCol]);
    assert(modelInput[releasedEventSurpriseCol] ==
           bar1[releasedEventSurpriseCol]);
    assert(modelInput[authoritativeInitialSurpriseCol] ==
           bar1[authoritativeInitialSurpriseCol]);

    const auto releaseActualMask = EA::FeatureAblationMask::Parse(
        std::string{EA::kEconomicEventReleaseActualAblationMaskText});
    assert(releaseActualMask.CanonicalText() ==
           EA::kEconomicEventReleaseActualAblationMaskText);
    const auto releaseTreatment = Project(bar1);
    const auto releaseControl = Project(bar1, releaseActualMask);
    for (std::size_t col = 0; col < authoritativeInitialHasSurpriseCol; ++col)
        assert(releaseControl[col] == releaseTreatment[col]);
    for (std::size_t col = authoritativeInitialHasSurpriseCol;
         col <= authoritativeInitialSurpriseDirectionCol; ++col)
        assert(releaseControl[col] == 0.0F);

    const auto consensusControlMask = EA::FeatureAblationMask::Parse(
        std::string{EA::kEconomicEventConsensusAblationMaskText});
    assert(consensusControlMask.CanonicalText() ==
           EA::kEconomicEventConsensusAblationMaskText);
    const auto scalarTreatment = Project(bar0);
    const auto scalarControl = Project(bar0, consensusControlMask);
    AssertConsensusControlParity(scalarTreatment, scalarControl);
    assert(scalarTreatment[relevantEventHasConsensusCol] == 1.0F);
    assert(Near(scalarTreatment[relevantEventConsensusLowCol], 0.2));
    assert(Near(scalarTreatment[relevantEventConsensusHighCol], 0.2));
    assert(scalarTreatment[relevantEventConsensusIsRangeCol] == 0.0F);
    for (std::size_t col = releasedEventHasSurpriseCol;
         col <= releasedEventSurpriseDirectionCol; ++col)
    {
        assert(scalarTreatment[col] == 0.0F);
        assert(scalarControl[col] == 0.0F);
    }

    // Missing and genuine-zero consensus remain distinguishable in treatment;
    // the control zeros only the four active channels in both cases.
    Tensor missingConsensus{
        "missing-consensus", kDefaultDonchian20Mode,
        kDefaultDonchianLookback,
        {EventAt(kBase + 100, "DOL_ETA", "WEEKLY_CLAIMS")}};
    missingConsensus.Add(BarAt(kBase, 0));
    const auto missingTreatment = Project(Row(missingConsensus, 0));
    const auto missingControl = Project(
        Row(missingConsensus, 0), consensusControlMask);
    AssertConsensusControlParity(missingTreatment, missingControl);
    assert(missingTreatment[relevantEventHasConsensusCol] == 0.0F);
    assert(missingTreatment[relevantEventConsensusLowCol] == 0.0F);
    assert(missingTreatment[relevantEventConsensusHighCol] == 0.0F);
    assert(missingTreatment[employmentEventCol] == 1.0F);
    assert(missingControl[employmentEventCol] ==
           missingTreatment[employmentEventCol]);
    assert(missingControl[employmentRecencyDecayCol] ==
           missingTreatment[employmentRecencyDecayCol]);

    Tensor zeroConsensus{
        "zero-consensus", kDefaultDonchian20Mode,
        kDefaultDonchianLookback,
        {ConsensusEventAt(kBase + 100, "BLS", "CPI", 0.0, 0.0)}};
    zeroConsensus.Add(BarAt(kBase, 0));
    const auto zeroTreatment = Project(Row(zeroConsensus, 0));
    const auto zeroControl = Project(Row(zeroConsensus, 0),
                                     consensusControlMask);
    AssertConsensusControlParity(zeroTreatment, zeroControl);
    assert(zeroTreatment[relevantEventHasConsensusCol] == 1.0F);
    assert(zeroTreatment[relevantEventConsensusLowCol] == 0.0F);
    assert(zeroTreatment[relevantEventConsensusHighCol] == 0.0F);

    // FOMC range endpoints and the range flag survive in treatment without a
    // midpoint; the same-width control zeros all four consensus channels.
    Tensor fomcRange{
        "fomc-range", kDefaultDonchian20Mode, kDefaultDonchianLookback,
        {RangeConsensusEventAt(kBase + 100, 5.25, 5.5)}};
    fomcRange.Add(BarAt(kBase, 0));
    const auto rangeTreatment = Project(Row(fomcRange, 0));
    const auto rangeControl = Project(Row(fomcRange, 0),
                                      consensusControlMask);
    AssertConsensusControlParity(rangeTreatment, rangeControl);
    assert(rangeTreatment[relevantEventHasConsensusCol] == 1.0F);
    assert(Near(rangeTreatment[relevantEventConsensusLowCol], 0.525));
    assert(Near(rangeTreatment[relevantEventConsensusHighCol], 0.55));
    assert(rangeTreatment[relevantEventConsensusIsRangeCol] == 1.0F);

    return 0;
}
