#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "CausalFibonacciConfluenceIntegration.hpp"

namespace
{
using EA::TG1A::Candle;
using EA::TG1A::ConfirmedFractal;
using EA::TG1A::FractalKind;
using EA::TG1A::TrendLineDirection;
using EA::TG1B::SteepnessClassification;
using EA::TG2::BreakObservation;
using EA::TG2::CensorReason;
using EA::TG2::ResolutionState;
using EA::TG3::ABDirection;
using EA::TG3::ABIdentity;
using EA::TG3::ABStructure;
using EA::TG3::Configuration;
using EA::TG3::ConfluenceObservation;
using EA::TG3::ConfluenceState;
using EA::TG3::DirectionalStudyPolicy;
using EA::TG3::FibonacciConfluenceTracker;
using EA::TG3::IneligibleReason;

constexpr std::int64_t kStart = 1'700'000'000;
constexpr std::int64_t kHour = 3'600;

std::int64_t Timestamp(std::size_t bar)
{
    return kStart + static_cast<std::int64_t>(bar) * kHour;
}

Configuration Config(double tolerance = 1.0)
{
    Configuration result;
    result.retracementRatios = {0.5, 0.25, 0.5};
    result.absolutePriceTolerance = tolerance;
    result.maxConfirmedFractalsPerKind = 32;
    result.maxABAgeBars = 100;
    result.maxActiveABStructures = 32;
    result.maxActiveConfluenceObservations = 64;
    result.maxRetainedConfluenceObservations = 64;
    return result;
}

ConfirmedFractal Fractal(FractalKind kind, std::size_t anchor, double price)
{
    return {kind, anchor, Timestamp(anchor), price,
            anchor + 2, Timestamp(anchor + 2)};
}

void AdvanceTo(FibonacciConfluenceTracker& tracker,
               std::size_t& currentBar,
               std::size_t target)
{
    while (currentBar <= target)
    {
        tracker.Advance(currentBar, Timestamp(currentBar));
        ++currentBar;
    }
}

void AddUpAB(FibonacciConfluenceTracker& tracker,
             std::size_t& nextBar,
             std::size_t aBar = 2,
             double aPrice = 100.0,
             std::size_t bBar = 5,
             double bPrice = 120.0)
{
    AdvanceTo(tracker, nextBar, aBar + 2);
    assert(tracker.ObserveConfirmedFractals(
        {Fractal(FractalKind::Low, aBar, aPrice)}).empty());
    AdvanceTo(tracker, nextBar, bBar + 2);
    const auto created = tracker.ObserveConfirmedFractals(
        {Fractal(FractalKind::High, bBar, bPrice)});
    assert(created.size() == 1);
    assert(created[0].direction == ABDirection::UpAB);
}

BreakObservation InnerBreak(std::uint64_t sequence,
                            std::size_t bar,
                            TrendLineDirection direction,
                            std::optional<double> outerProjection)
{
    BreakObservation result;
    result.breakEvent.eventSequence = sequence;
    result.breakEvent.candidate.direction = direction;
    result.breakEvent.candidate.anchor1Bar = 1;
    result.breakEvent.candidate.anchor1Timestamp = Timestamp(1);
    result.breakEvent.candidate.anchor2Bar = 4;
    result.breakEvent.candidate.anchor2Timestamp = Timestamp(4);
    result.breakEvent.candidate.creationBar = 6;
    result.breakEvent.candidate.creationTimestamp = Timestamp(6);
    result.breakEvent.frozenClassification = SteepnessClassification::Inner;
    result.breakEvent.bar = bar;
    result.breakEvent.timestamp = Timestamp(bar);
    result.retest.state = ResolutionState::Pending;
    result.outerTargetAfterRetest.state =
        ResolutionState::StructurallyIneligible;
    if (outerProjection.has_value())
    {
        EA::TG2::PairedOuterLine outer;
        outer.candidate.direction = direction;
        outer.candidate.anchor1Bar = 2;
        outer.candidate.anchor1Timestamp = Timestamp(2);
        outer.candidate.anchor2Bar = 6;
        outer.candidate.anchor2Timestamp = Timestamp(6);
        outer.candidate.creationBar = 8;
        outer.candidate.creationTimestamp = Timestamp(8);
        outer.anchor1Price = *outerProjection;
        outer.projectedPriceAtBreak = *outerProjection;
        result.pairedOuter = outer;
        result.outerTarget.state = ResolutionState::Pending;
    }
    else
    {
        result.outerTarget.state = ResolutionState::StructurallyIneligible;
    }
    return result;
}

void Resolve(EA::TG2::OutcomeResolution& outcome,
             ResolutionState state,
             std::size_t bar)
{
    outcome.state = state;
    outcome.resolutionBar = bar;
    outcome.resolutionTimestamp = Timestamp(bar);
    outcome.latencyBars = 1;
    if (state == ResolutionState::Censored)
        outcome.censorReason = CensorReason::EndOfInput;
}

auto ImmutableSnapshot(const ConfluenceObservation& observation)
{
    std::vector<std::tuple<double, double, bool>> levels;
    for (const auto& level : observation.levels)
        levels.emplace_back(level.ratio, level.levelPrice, level.matched);
    std::optional<EA::TG2::CandidateIdentity> outerIdentity;
    std::optional<double> outerProjection;
    if (observation.pairedOuter.has_value())
    {
        outerIdentity = observation.pairedOuter->candidate;
        outerProjection = observation.pairedOuter->projectedPriceAtBreak;
    }
    std::optional<ABIdentity> abIdentity;
    std::optional<double> aPrice;
    std::optional<double> bPrice;
    if (observation.selectedAB.has_value())
    {
        abIdentity = observation.selectedAB->identity;
        aPrice = observation.selectedAB->aPrice;
        bPrice = observation.selectedAB->bPrice;
    }
    return std::tuple{
        observation.breakEventSequence,
        observation.innerCandidate,
        observation.frozenClassification,
        observation.innerBreakBar,
        observation.innerBreakTimestamp,
        outerIdentity,
        outerProjection,
        abIdentity,
        aPrice,
        bPrice,
        observation.observationBar,
        observation.observationTimestamp,
        observation.outerProjectedPriceAtObservation,
        levels,
        observation.matchedRatios,
        observation.confluenceState,
        observation.ineligibleReason,
        observation.minimumRawPriceDistance,
        observation.minimumAtrNormalizedDistance};
}

void TestCausalUpAndDownABCreationTiming()
{
    FibonacciConfluenceTracker tracker(Config());
    std::size_t nextBar = 0;
    AdvanceTo(tracker, nextBar, 3);

    bool earlyARejected = false;
    try
    {
        tracker.ObserveConfirmedFractals(
            {Fractal(FractalKind::Low, 2, 100.0)});
    }
    catch (const std::invalid_argument&) { earlyARejected = true; }
    assert(earlyARejected);
    assert(tracker.ABStructures().empty());

    AdvanceTo(tracker, nextBar, 4);
    assert(tracker.ObserveConfirmedFractals(
        {Fractal(FractalKind::Low, 2, 100.0)}).empty());
    AdvanceTo(tracker, nextBar, 6);
    bool earlyBRejected = false;
    try
    {
        tracker.ObserveConfirmedFractals(
            {Fractal(FractalKind::High, 5, 120.0)});
    }
    catch (const std::invalid_argument&) { earlyBRejected = true; }
    assert(earlyBRejected);
    assert(tracker.ABStructures().empty());

    AdvanceTo(tracker, nextBar, 7);
    const auto up = tracker.ObserveConfirmedFractals(
        {Fractal(FractalKind::High, 5, 120.0)});
    assert(up.size() == 1);
    assert(up[0].availabilityBar == 7);
    assert(tracker.ABStructures()[0].aConfirmationBar == 4);
    assert(tracker.ABStructures()[0].bConfirmationBar == 7);

    AdvanceTo(tracker, nextBar, 10);
    tracker.ObserveConfirmedFractals(
        {Fractal(FractalKind::High, 8, 130.0)});
    AdvanceTo(tracker, nextBar, 13);
    const auto down = tracker.ObserveConfirmedFractals(
        {Fractal(FractalKind::Low, 11, 105.0)});
    assert(down.size() == 1);
    assert(down[0].direction == ABDirection::DownAB);
    assert(down[0].availabilityBar == 13);
}

void TestExactLevelEquationsOrderingDedupAndValidation()
{
    ABStructure up;
    up.identity.direction = ABDirection::UpAB;
    up.aPrice = 100.0;
    up.bPrice = 120.0;
    up.priceRange = 20.0;
    const auto upLevels =
        FibonacciConfluenceTracker::CalculateRetracementLevels(
            up, {1.0, 0.0, 0.25, 0.5});
    assert(upLevels.size() == 4);
    assert(upLevels[0].price == 100.0);
    assert(upLevels[1].price == 110.0);
    assert(upLevels[2].price == 115.0);
    assert(upLevels[3].price == 120.0);

    ABStructure down = up;
    down.identity.direction = ABDirection::DownAB;
    down.aPrice = 120.0;
    down.bPrice = 100.0;
    const auto downLevels =
        FibonacciConfluenceTracker::CalculateRetracementLevels(
            down, {0.0, 0.25, 0.5, 1.0});
    assert(downLevels[0].price == 100.0);
    assert(downLevels[1].price == 105.0);
    assert(downLevels[2].price == 110.0);
    assert(downLevels[3].price == 120.0);

    FibonacciConfluenceTracker normalized(Config());
    assert(normalized.GetConfiguration().retracementRatios ==
           std::vector<double>({0.25, 0.5}));

    bool emptyRejected = false;
    bool nonfiniteRejected = false;
    bool rangeRejected = false;
    bool toleranceRejected = false;
    bool boundsRejected = false;
    try
    {
        Configuration invalid = Config();
        invalid.retracementRatios.clear();
        (void)FibonacciConfluenceTracker(invalid);
    }
    catch (const std::invalid_argument&) { emptyRejected = true; }
    try
    {
        Configuration invalid = Config();
        invalid.retracementRatios = {
            std::numeric_limits<double>::quiet_NaN()};
        (void)FibonacciConfluenceTracker(invalid);
    }
    catch (const std::invalid_argument&) { nonfiniteRejected = true; }
    try
    {
        Configuration invalid = Config();
        invalid.retracementRatios = {1.1};
        (void)FibonacciConfluenceTracker(invalid);
    }
    catch (const std::invalid_argument&) { rangeRejected = true; }
    try
    {
        Configuration invalid = Config();
        invalid.absolutePriceTolerance = -1.0;
        (void)FibonacciConfluenceTracker(invalid);
    }
    catch (const std::invalid_argument&) { toleranceRejected = true; }
    try
    {
        Configuration invalid = Config();
        invalid.maxActiveABStructures = 0;
        (void)FibonacciConfluenceTracker(invalid);
    }
    catch (const std::invalid_argument&) { boundsRejected = true; }
    assert(emptyRejected && nonfiniteRejected && rangeRejected &&
           toleranceRejected && boundsRejected);
}

void TestStableIdentityAndDeterministicMultipleABSelection()
{
    FibonacciConfluenceTracker tracker(Config());
    std::size_t nextBar = 0;
    AddUpAB(tracker, nextBar);
    const ABStructure fixed = tracker.ABStructures().front();
    AdvanceTo(tracker, nextBar, 8);
    tracker.ObserveInnerBreak(
        InnerBreak(1, 8, TrendLineDirection::UTL, 110.0));
    const auto fixedObservation = ImmutableSnapshot(
        tracker.Observations().front());

    AdvanceTo(tracker, nextBar, 10);
    tracker.ObserveConfirmedFractals(
        {Fractal(FractalKind::Low, 8, 104.0)});
    AdvanceTo(tracker, nextBar, 13);
    tracker.ObserveConfirmedFractals(
        {Fractal(FractalKind::High, 11, 124.0)});
    assert(tracker.ABStructures().size() >= 2);
    assert(tracker.ABStructures().front().identity == fixed.identity);
    assert(tracker.ABStructures().front().aPrice == fixed.aPrice);
    assert(tracker.ABStructures().front().bPrice == fixed.bPrice);
    assert(ImmutableSnapshot(tracker.Observations().front()) ==
           fixedObservation);

    AdvanceTo(tracker, nextBar, 14);
    const BreakObservation event = InnerBreak(
        2, 14, TrendLineDirection::UTL, 114.0);
    tracker.ObserveInnerBreak(event, 2.0);
    const ConfluenceObservation& observation = tracker.Observations().back();
    assert(observation.selectedAB.has_value());
    assert(observation.selectedAB->identity.aBar == 8);
    assert(observation.selectedAB->identity.bBar == 11);
    assert(observation.minimumAtrNormalizedDistance == 0.0);
}

void TestTG2PairIdentityAndExactAdjacentBoundaries()
{
    FibonacciConfluenceTracker tracker(Config(1.0));
    std::size_t nextBar = 0;
    AddUpAB(tracker, nextBar);
    AdvanceTo(tracker, nextBar, 8);

    bool futureObservationRejected = false;
    try
    {
        tracker.ObserveInnerBreak(
            InnerBreak(99, 9, TrendLineDirection::UTL, 110.0));
    }
    catch (const std::invalid_argument&)
    {
        futureObservationRejected = true;
    }
    assert(futureObservationRejected);

    const BreakObservation boundary = InnerBreak(
        1, 8, TrendLineDirection::UTL, 111.0);
    const auto pairedIdentity = boundary.pairedOuter->candidate;
    tracker.ObserveInnerBreak(boundary, 2.0);
    const ConfluenceObservation& exact = tracker.Observations().back();
    assert(exact.confluenceState == ConfluenceState::Confluence);
    assert(exact.pairedOuter->candidate == pairedIdentity);
    assert(exact.minimumRawPriceDistance == 1.0);

    BreakObservation inside = InnerBreak(
        2, 8, TrendLineDirection::UTL,
        std::nextafter(111.0, 110.0));
    tracker.ObserveInnerBreak(inside);
    assert(tracker.Observations().back().confluenceState ==
           ConfluenceState::Confluence);

    BreakObservation outside = InnerBreak(
        3, 8, TrendLineDirection::UTL,
        std::nextafter(111.0, 112.0));
    tracker.ObserveInnerBreak(outside);
    assert(tracker.Observations().back().confluenceState ==
           ConfluenceState::NoConfluence);
}

void TestStructuralIneligibilityAndDirectionalScope()
{
    FibonacciConfluenceTracker noAB(Config());
    std::size_t nextBar = 0;
    AdvanceTo(noAB, nextBar, 3);
    noAB.ObserveInnerBreak(
        InnerBreak(1, 3, TrendLineDirection::UTL, 110.0));
    assert(noAB.Observations()[0].confluenceState ==
           ConfluenceState::StructurallyIneligible);
    assert(noAB.Observations()[0].ineligibleReason ==
           IneligibleReason::NoCausallyEligibleAB);

    FibonacciConfluenceTracker unpaired(Config());
    nextBar = 0;
    AddUpAB(unpaired, nextBar);
    AdvanceTo(unpaired, nextBar, 8);
    unpaired.ObserveInnerBreak(
        InnerBreak(1, 8, TrendLineDirection::UTL, std::nullopt));
    assert(unpaired.Observations()[0].ineligibleReason ==
           IneligibleReason::NoTG2PairedOuter);

    FibonacciConfluenceTracker sourceOnly(Config());
    nextBar = 0;
    AdvanceTo(sourceOnly, nextBar, 4);
    sourceOnly.ObserveConfirmedFractals(
        {Fractal(FractalKind::High, 2, 120.0)});
    AdvanceTo(sourceOnly, nextBar, 7);
    sourceOnly.ObserveConfirmedFractals(
        {Fractal(FractalKind::Low, 5, 100.0)});
    AdvanceTo(sourceOnly, nextBar, 8);
    sourceOnly.ObserveInnerBreak(
        InnerBreak(1, 8, TrendLineDirection::DTL, 110.0));
    assert(sourceOnly.Observations()[0].ineligibleReason ==
           IneligibleReason::DirectionUnsupportedByStudy);

    Configuration symmetricConfiguration = Config();
    symmetricConfiguration.directionalStudyPolicy =
        DirectionalStudyPolicy::SymmetricDirectionalDiagnostic;
    FibonacciConfluenceTracker symmetric(symmetricConfiguration);
    nextBar = 0;
    AdvanceTo(symmetric, nextBar, 4);
    symmetric.ObserveConfirmedFractals(
        {Fractal(FractalKind::High, 2, 120.0)});
    AdvanceTo(symmetric, nextBar, 7);
    symmetric.ObserveConfirmedFractals(
        {Fractal(FractalKind::Low, 5, 100.0)});
    AdvanceTo(symmetric, nextBar, 8);
    symmetric.ObserveInnerBreak(
        InnerBreak(1, 8, TrendLineDirection::DTL, 110.0));
    assert(symmetric.Observations()[0].confluenceState ==
           ConfluenceState::Confluence);
}

void TestEmpiricalGroupingOutcomePropagationAndRates()
{
    FibonacciConfluenceTracker tracker(Config(0.0));
    std::size_t nextBar = 0;
    AddUpAB(tracker, nextBar);
    AdvanceTo(tracker, nextBar, 8);

    BreakObservation confluenceSuccess = InnerBreak(
        1, 8, TrendLineDirection::UTL, 110.0);
    Resolve(confluenceSuccess.retest, ResolutionState::Succeeded, 9);
    Resolve(confluenceSuccess.outerTarget, ResolutionState::Succeeded, 9);
    Resolve(confluenceSuccess.outerTargetAfterRetest,
            ResolutionState::Succeeded, 10);
    tracker.ObserveInnerBreak(confluenceSuccess);

    BreakObservation confluenceFailure = InnerBreak(
        2, 8, TrendLineDirection::UTL, 110.0);
    Resolve(confluenceFailure.retest, ResolutionState::Succeeded, 9);
    Resolve(confluenceFailure.outerTarget, ResolutionState::Failed, 10);
    Resolve(confluenceFailure.outerTargetAfterRetest,
            ResolutionState::Failed, 10);
    tracker.ObserveInnerBreak(confluenceFailure);

    BreakObservation noConfluenceCensored = InnerBreak(
        3, 8, TrendLineDirection::UTL, 117.0);
    Resolve(noConfluenceCensored.retest, ResolutionState::Succeeded, 9);
    Resolve(noConfluenceCensored.outerTarget,
            ResolutionState::Censored, 9);
    Resolve(noConfluenceCensored.outerTargetAfterRetest,
            ResolutionState::Censored, 9);
    tracker.ObserveInnerBreak(noConfluenceCensored);

    BreakObservation confluenceCensored = InnerBreak(
        4, 8, TrendLineDirection::UTL, 110.0);
    Resolve(confluenceCensored.retest, ResolutionState::Succeeded, 9);
    Resolve(confluenceCensored.outerTarget,
            ResolutionState::Censored, 9);
    Resolve(confluenceCensored.outerTargetAfterRetest,
            ResolutionState::Censored, 9);
    tracker.ObserveInnerBreak(confluenceCensored);

    BreakObservation noConfluenceSuccess = InnerBreak(
        5, 8, TrendLineDirection::UTL, 117.0);
    Resolve(noConfluenceSuccess.retest, ResolutionState::Succeeded, 9);
    Resolve(noConfluenceSuccess.outerTarget,
            ResolutionState::Succeeded, 9);
    Resolve(noConfluenceSuccess.outerTargetAfterRetest,
            ResolutionState::Succeeded, 9);
    tracker.ObserveInnerBreak(noConfluenceSuccess);

    BreakObservation noConfluenceFailure = InnerBreak(
        6, 8, TrendLineDirection::UTL, 117.0);
    Resolve(noConfluenceFailure.retest, ResolutionState::Succeeded, 9);
    Resolve(noConfluenceFailure.outerTarget,
            ResolutionState::Failed, 9);
    Resolve(noConfluenceFailure.outerTargetAfterRetest,
            ResolutionState::Failed, 9);
    tracker.ObserveInnerBreak(noConfluenceFailure);

    tracker.ObserveInnerBreak(
        InnerBreak(7, 8, TrendLineDirection::UTL, std::nullopt));

    const auto summary = tracker.AggregateSummary();
    assert(summary.innerBreakObservations == 7);
    assert(summary.eligibleWithConfluence == 3);
    assert(summary.eligibleWithoutConfluence == 3);
    assert(summary.structurallyIneligible == 1);
    assert(summary.noPairedOuter == 1);
    assert(summary.confluence.outerTarget.successes == 1);
    assert(summary.confluence.outerTarget.failures == 1);
    assert(summary.confluence.outerTarget.censored == 1);
    assert(summary.confluence.outerTarget.resolved == 2);
    assert(summary.confluence.outerTarget.empiricalRate == 0.5);
    assert(summary.confluence.retestThenOuter.successes == 1);
    assert(summary.confluence.retestThenOuter.failures == 1);
    assert(summary.confluence.retestThenOuter.empiricalRate == 0.5);
    assert(summary.noConfluence.outerTarget.successes == 1);
    assert(summary.noConfluence.outerTarget.failures == 1);
    assert(summary.noConfluence.outerTarget.censored == 1);
    assert(summary.noConfluence.outerTarget.empiricalRate == 0.5);
    assert(summary.noConfluence.retestThenOuter.successes == 1);
    assert(summary.noConfluence.retestThenOuter.failures == 1);
    assert(summary.noConfluence.retestThenOuter.censored == 1);
    assert(summary.noConfluence.retestThenOuter.empiricalRate == 0.5);
    assert(summary.ineligible.outerTarget.structurallyIneligible == 1);
}

void TestFutureOutcomeCannotRewriteConfluence()
{
    FibonacciConfluenceTracker tracker(Config(0.0));
    std::size_t nextBar = 0;
    AddUpAB(tracker, nextBar);
    AdvanceTo(tracker, nextBar, 8);
    BreakObservation event = InnerBreak(
        1, 8, TrendLineDirection::UTL, 110.0);
    tracker.ObserveInnerBreak(event, 2.0);
    const auto fixed = ImmutableSnapshot(tracker.Observations()[0]);

    AdvanceTo(tracker, nextBar, 9);
    Resolve(event.retest, ResolutionState::Succeeded, 9);
    Resolve(event.outerTarget, ResolutionState::Failed, 9);
    Resolve(event.outerTargetAfterRetest, ResolutionState::Failed, 9);
    tracker.SynchronizeOutcomes({event}, 9, Timestamp(9));
    assert(ImmutableSnapshot(tracker.Observations()[0]) == fixed);
    assert(tracker.Observations()[0].outerTarget.state ==
           ResolutionState::Failed);
}

Candle Bar(std::size_t index, double high, double low, double close)
{
    return {Timestamp(index), close, high, low, close};
}

std::vector<Candle> IntegratedCandles()
{
    const std::vector<double> lows =
        {10.0, 9.0, 5.0, 8.0, 10.0, 9.0,
         10.0, 7.0, 10.0, 10.0, 7.0, 7.1};
    const std::vector<double> highs =
        {20.0, 20.5, 21.0, 23.0, 25.0, 22.0,
         21.0, 20.5, 21.0, 20.0, 8.1, 8.0};
    std::vector<Candle> candles;
    for (std::size_t index = 0; index < lows.size(); ++index)
    {
        const double close = 0.5 * (highs[index] + lows[index]);
        candles.push_back(Bar(index, highs[index], lows[index], close));
    }
    return candles;
}

auto ABSnapshot(const std::vector<ABStructure>& structures)
{
    std::vector<std::tuple<ABIdentity, double, double>> result;
    for (const ABStructure& structure : structures)
        result.emplace_back(structure.identity,
                            structure.aPrice, structure.bPrice);
    return result;
}

auto ConfluenceSnapshot(
    const std::vector<ConfluenceObservation>& observations)
{
    std::vector<decltype(ImmutableSnapshot(observations.front()))> result;
    for (const ConfluenceObservation& observation : observations)
        result.push_back(ImmutableSnapshot(observation));
    return result;
}

void TestIntegratedHistoricalStreamingPrefixParity()
{
    const auto candles = IntegratedCandles();
    EA::TG1A::Configuration geometry;
    geometry.atrPeriod = 3;
    EA::TG2::Configuration behavior;
    behavior.retestHorizonBars = 2;
    behavior.outerTargetHorizonBars = 3;

    EA::TG3::CausalFibonacciConfluenceIntegration streaming(
        EA::TG1B::CalibrationConfiguration(50.0), Config(0.0),
        geometry, behavior, {"EURUSD", "1h"});
    for (std::size_t index = 0; index < candles.size(); ++index)
    {
        streaming.AddCompletedBar(candles[index]);
        std::vector<Candle> prefix(candles.begin(),
                                   candles.begin() + index + 1);
        const auto historical =
            EA::TG3::CausalFibonacciConfluenceIntegration::FromHistorical(
                prefix, EA::TG1B::CalibrationConfiguration(50.0),
                Config(0.0), geometry, behavior, {"EURUSD", "1h"});
        assert(ABSnapshot(streaming.ABStructures()) ==
               ABSnapshot(historical.ABStructures()));
        assert(ConfluenceSnapshot(streaming.Observations()) ==
               ConfluenceSnapshot(historical.Observations()));
    }
    assert(!streaming.ABStructures().empty());
    assert(!streaming.Observations().empty());

    bool duplicateRejected = false;
    try
    {
        auto duplicate = candles;
        duplicate.push_back(candles.back());
        (void)EA::TG3::CausalFibonacciConfluenceIntegration::FromHistorical(
            duplicate, EA::TG1B::CalibrationConfiguration(50.0),
            Config(), geometry, behavior);
    }
    catch (const std::invalid_argument&) { duplicateRejected = true; }
    assert(duplicateRejected);
}

void TestDeterministicBoundsAndLongStreamPerformance()
{
    Configuration configuration = Config(0.5);
    configuration.maxConfirmedFractalsPerKind = 4;
    configuration.maxABAgeBars = 20;
    configuration.maxActiveABStructures = 8;
    configuration.maxActiveConfluenceObservations = 8;
    configuration.maxRetainedConfluenceObservations = 16;
    FibonacciConfluenceTracker first(configuration);
    FibonacciConfluenceTracker second(configuration);

    constexpr std::size_t barCount = 50'000;
    std::uint64_t sequence = 1;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t bar = 0; bar < barCount; ++bar)
    {
        first.Advance(bar, Timestamp(bar));
        second.Advance(bar, Timestamp(bar));
        if (bar >= 4 && bar % 4 == 0)
        {
            const auto low = Fractal(FractalKind::Low, bar - 2, 100.0);
            first.ObserveConfirmedFractals({low});
            second.ObserveConfirmedFractals({low});
        }
        if (bar >= 6 && bar % 4 == 2)
        {
            const auto high = Fractal(FractalKind::High, bar - 2, 120.0);
            first.ObserveConfirmedFractals({high});
            second.ObserveConfirmedFractals({high});
        }
        if (bar >= 7)
        {
            const BreakObservation event = InnerBreak(
                sequence++, bar, TrendLineDirection::UTL, 110.0);
            first.ObserveInnerBreak(event);
            second.ObserveInnerBreak(event);
        }
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    assert(first.ABStructures().size() <=
           configuration.maxActiveABStructures);
    assert(first.Observations().size() <=
           configuration.maxRetainedConfluenceObservations);
    assert(first.ActiveObservationCount() <=
           configuration.maxActiveConfluenceObservations);
    assert(ABSnapshot(first.ABStructures()) ==
           ABSnapshot(second.ABStructures()));
    const auto firstSummary = first.AggregateSummary();
    const auto secondSummary = second.AggregateSummary();
    assert(firstSummary.innerBreakObservations == barCount - 7);
    assert(firstSummary.innerBreakObservations ==
           secondSummary.innerBreakObservations);
    assert(firstSummary.capacityEvictedObservations > 0);
    assert(firstSummary.confluence.outerTarget.censored > 0);
    assert(firstSummary.confluence.outerTarget.failures == 0);
    assert(first.ABCapacityEvictions() > 0);
    assert(first.TotalABStructuresCreated() ==
           second.TotalABStructuresCreated());
    assert(elapsed.count() < 10'000);
    std::cout << "TG3_BOUNDED_STREAM bars=" << barCount
              << ",milliseconds=" << elapsed.count()
              << ",retained_ab=" << first.ABStructures().size()
              << ",retained_observations=" << first.Observations().size()
              << ",observations=" << firstSummary.innerBreakObservations
              << '\n';
}

void TestDiagnostics()
{
    FibonacciConfluenceTracker tracker(Config(0.5), {"EURUSD", "1h"});
    std::size_t nextBar = 0;
    AddUpAB(tracker, nextBar);
    AdvanceTo(tracker, nextBar, 8);
    tracker.ObserveInnerBreak(
        InnerBreak(42, 8, TrendLineDirection::UTL, 110.0), 2.0);
    const std::string diagnostic =
        tracker.FormatDiagnostic(tracker.Observations()[0]);
    assert(diagnostic.find("symbol=EURUSD") != std::string::npos);
    assert(diagnostic.find("tg2_break_event_sequence=42") !=
           std::string::npos);
    assert(diagnostic.find("observation_timing=completed_inner_break_bar") !=
           std::string::npos);
    assert(diagnostic.find("ab_direction=UpAB") != std::string::npos);
    assert(diagnostic.find("ratio_provenance=explicit_caller_configuration") !=
           std::string::npos);
    assert(diagnostic.find("implementation_convention") !=
           std::string::npos);
    assert(diagnostic.find("confluence=yes") != std::string::npos);
    assert(diagnostic.find("observed_frequency_not_probability") !=
           std::string::npos);
    const std::string summary = tracker.FormatSummaryDiagnostic();
    assert(summary.find("confluence_outer_denominator=") !=
           std::string::npos);
    assert(summary.find("empirical_rate_formula=successes/(successes+failures)") !=
           std::string::npos);
}
} // namespace

int main()
{
    TestCausalUpAndDownABCreationTiming();
    TestExactLevelEquationsOrderingDedupAndValidation();
    TestStableIdentityAndDeterministicMultipleABSelection();
    TestTG2PairIdentityAndExactAdjacentBoundaries();
    TestStructuralIneligibilityAndDirectionalScope();
    TestEmpiricalGroupingOutcomePropagationAndRates();
    TestFutureOutcomeCannotRewriteConfluence();
    TestIntegratedHistoricalStreamingPrefixParity();
    TestDeterministicBoundsAndLongStreamPerformance();
    TestDiagnostics();
    std::cout << "TG3FibonacciConfluenceIntegrationTests passed\n";
    return 0;
}
