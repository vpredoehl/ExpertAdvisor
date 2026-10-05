#include "CausalFibonacciRetracementLifecycle.hpp"
#include "CausalFibonacciRetracementLifecycleHistoricalEvaluation.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace Fib = EA::FibonacciResearch::RetracementLifecycle;
namespace TG3 = EA::TG3;
namespace TG1A = EA::TG1A;

static TG3::ABStructure UpAB()
{
    TG3::ABStructure ab;
    ab.identity.direction = TG3::ABDirection::UpAB;
    ab.identity.aBar = 0;
    ab.identity.aTimestamp = 100;
    ab.identity.bBar = 4;
    ab.identity.bTimestamp = 500;
    ab.identity.availabilityBar = 6;
    ab.identity.availabilityTimestamp = 700;
    ab.aPrice = 1.0000;
    ab.bPrice = 1.1000;
    ab.priceRange = 0.1000;
    return ab;
}

static TG3::ABStructure DownAB()
{
    TG3::ABStructure ab = UpAB();
    ab.identity.direction = TG3::ABDirection::DownAB;
    ab.aPrice = 1.1000;
    ab.bPrice = 1.0000;
    return ab;
}

static TG1A::Candle C(std::int64_t timestamp,
                      double open,
                      double high,
                      double low,
                      double close)
{
    return {timestamp, open, high, low, close};
}

static TG1A::ConfirmedFractal F(TG1A::FractalKind kind,
                                std::size_t bar, double price)
{
    return {kind, bar - 2, static_cast<std::int64_t>(bar - 2), price,
            bar, static_cast<std::int64_t>(bar)};
}

static void TestRetracementPrices()
{
    Fib::Tracker tracker(UpAB(), 1.1618);
    const auto& r = tracker.GetRecord();

    assert(std::abs(r.retracement0382.price - 1.0618) < 1e-12);
    assert(std::abs(r.retracement0500.price - 1.0500) < 1e-12);
    assert(std::abs(r.retracement0618.price - 1.0382) < 1e-12);
}

static void TestProgressiveRetracementAndRally()
{
    Fib::Tracker tracker(UpAB(), 1.1618);

    tracker.AddCompletedBar(6, C(700, 1.0900, 1.0950, 1.0700, 1.0750));
    assert(!tracker.GetRecord().retracement0382.reached.has_value());

    tracker.AddCompletedBar(7, C(800, 1.0700, 1.0750, 1.0600, 1.0650));
    assert(tracker.GetRecord().retracement0382.reached.has_value());
    assert(!tracker.GetRecord().retracement0382
                .firstDirectionalClose.has_value());

    tracker.AddCompletedBar(8, C(900, 1.0650, 1.0680, 1.0480, 1.0520));
    assert(tracker.GetRecord().retracement0500.reached.has_value());

    tracker.AddCompletedBar(9, C(1000, 1.0520, 1.0580, 1.0500, 1.0570));

    assert(tracker.GetRecord().retracement0382
               .firstDirectionalClose.has_value());
    assert(tracker.GetRecord().retracement0500
               .firstDirectionalClose.has_value());
    assert(tracker.GetRecord().retracement0382.barsBToReached ==
           std::optional<std::size_t>{3});
}

static void TestAllRetracementsAndNoSameBarConfirmation()
{
    Fib::Tracker tracker(UpAB(), 1.1618);
    // This reaches every retracement but is bearish, so it cannot serve as a
    // same-bar directional continuation confirmation.
    tracker.AddCompletedBar(6, C(700, 1.0900, 1.0950, 1.0350, 1.0400));
    const auto& first = tracker.GetRecord();
    assert(first.retracement0382.reached && first.retracement0500.reached &&
           first.retracement0618.reached);
    assert(!first.retracement0382.firstDirectionalClose);
    assert(!first.retracement0382.firstDirectionalBreak);
    assert(!first.retracement0382.firstCloseBackThroughLevel);
    tracker.AddCompletedBar(7, C(800, 1.0400, 1.0700, 1.0380, 1.0650));
    assert(tracker.GetRecord().retracement0618.firstDirectionalClose);
    assert(tracker.GetRecord().retracement0618.firstCloseBackThroughLevel);
}

static void TestPenetrationDoesNotImplyCloseInvalidation()
{
    Fib::Tracker tracker(UpAB(), 1.1618);

    tracker.AddCompletedBar(6, C(700, 1.0300, 1.0400, 0.9950, 1.0100));

    assert(tracker.GetRecord().firstAPenetration.has_value());
    assert(!tracker.GetRecord().firstCloseBeyondA.has_value());
}

static void TestCloseBeyondARecordedSeparately()
{
    Fib::Tracker tracker(UpAB(), 1.1618);

    tracker.AddCompletedBar(6, C(700, 1.0100, 1.0200, 0.9900, 0.9950));

    assert(tracker.GetRecord().firstAPenetration.has_value());
    assert(tracker.GetRecord().firstCloseBeyondA.has_value());
}

static void TestNoArbitraryDTimeout()
{
    Fib::Tracker tracker(UpAB(), 1.1618);

    std::int64_t timestamp = 700;

    for (std::size_t bar = 6; bar < 306; ++bar)
    {
        tracker.AddCompletedBar(
            bar, C(timestamp, 1.0500, 1.0600, 1.0400, 1.0500));
        timestamp += 100;
    }

    assert(!tracker.GetRecord().firstDReached.has_value());

    tracker.AddCompletedBar(
        306, C(timestamp, 1.1500, 1.1620, 1.1450, 1.1600));

    assert(tracker.GetRecord().firstDReached.has_value());
    assert(tracker.GetRecord().firstDReached->bar == 306);
    assert(tracker.GetRecord().barsBToD ==
           std::optional<std::size_t>{302});
}

static void TestDownDirectionSymmetry()
{
    Fib::Tracker tracker(DownAB(), 0.9382);

    const auto& initial = tracker.GetRecord();
    assert(std::abs(initial.retracement0382.price - 1.0382) < 1e-12);
    assert(std::abs(initial.retracement0500.price - 1.0500) < 1e-12);
    assert(std::abs(initial.retracement0618.price - 1.0618) < 1e-12);

    tracker.AddCompletedBar(
        6, C(700, 1.0400, 1.0620, 1.0350, 1.0550));

    assert(tracker.GetRecord().retracement0382.reached.has_value());
    assert(tracker.GetRecord().retracement0500.reached.has_value());
    assert(tracker.GetRecord().retracement0618.reached.has_value());

    tracker.AddCompletedBar(
        7, C(800, 1.0550, 1.0570, 1.0300, 1.0350));

    assert(tracker.GetRecord().retracement0618
               .firstDirectionalClose.has_value());

    tracker.AddCompletedBar(
        8, C(900, 1.0000, 1.0100, 0.9370, 0.9450));

    assert(tracker.GetRecord().firstDReached.has_value());
}

static void TestRightCensoring()
{
    Fib::Tracker tracker(UpAB(), 1.1618);

    tracker.AddCompletedBar(
        6, C(700, 1.0700, 1.0800, 1.0600, 1.0700));

    tracker.Finalize();

    assert(tracker.IsFinalized());
    assert(tracker.GetRecord().rightCensored);
}

static void TestIndependentDHypothesesAndEntryExcursion()
{
    Fib::Tracker d1272(UpAB(), 1.1272);
    Fib::Tracker d1618(UpAB(), 1.1618);

    const auto pullback = C(700, 1.0700, 1.0750, 1.0600, 1.0650);
    const auto rally = C(800, 1.0650, 1.1300, 1.0640, 1.1280);

    d1272.AddCompletedBar(6, pullback);
    d1618.AddCompletedBar(6, pullback);
    d1272.AddCompletedBar(7, rally);
    d1618.AddCompletedBar(7, rally);

    assert(d1272.GetRecord().firstDReached);
    assert(!d1618.GetRecord().firstDReached);

    const auto& entry = d1272.GetRecord().directionalClose0382;
    assert(entry.entry && entry.entry->bar == 7);
    assert(entry.barsBToEntry == std::optional<std::size_t>{3});
    assert(entry.barsRetracementToEntry == std::optional<std::size_t>{1});

    // D and the closing confirmation occurred on the same completed candle.
    // Preserve the structural zero-bar coincidence, but do not use that
    // candle's full high/low as post-entry MFE/MAE because OHLC cannot tell us
    // whether those extremes occurred before or after the confirming close.
    assert(entry.barsEntryToD == std::optional<std::size_t>{0});
    assert(!entry.maximumFavorableExcursion);
    assert(!entry.maximumAdverseExcursion);
    assert(!entry.maximumFavorableExcursionABRanges);
    assert(!entry.maximumAdverseExcursionABRanges);

    const auto postEntry = C(900, 1.1280, 1.1400, 1.1200, 1.1350);
    d1272.AddCompletedBar(8, postEntry);
    d1618.AddCompletedBar(8, postEntry);

    const auto& measured = d1272.GetRecord().directionalClose0382;
    assert(measured.highestHigh &&
           std::abs(*measured.highestHigh - 1.1400) < 1e-12);
    assert(measured.lowestLow &&
           std::abs(*measured.lowestLow - 1.1200) < 1e-12);
    assert(measured.maximumFavorableExcursion &&
           std::abs(*measured.maximumFavorableExcursion - 0.0120) < 1e-12);
    assert(measured.maximumAdverseExcursion &&
           std::abs(*measured.maximumAdverseExcursion - 0.0080) < 1e-12);
    assert(measured.maximumFavorableExcursionABRanges &&
           std::abs(*measured.maximumFavorableExcursionABRanges - 0.12) <
               1e-12);
    assert(measured.maximumAdverseExcursionABRanges &&
           std::abs(*measured.maximumAdverseExcursionABRanges - 0.08) <
               1e-12);
}

static void TestMultipleIndependentCandidatesSurviveLongPaths()
{
    Fib::Tracker older(UpAB(), 1.1618);
    TG3::ABStructure newer = UpAB();
    newer.identity.aBar = 1;
    newer.identity.bBar = 5;
    newer.identity.availabilityBar = 7;
    newer.identity.aTimestamp = 200;
    newer.identity.bTimestamp = 600;
    newer.identity.availabilityTimestamp = 800;
    Fib::Tracker retained(newer, 1.1618);
    older.AddCompletedBar(6, C(700, 1.0700, 1.0750, 1.0600, 1.0650));
    older.AddCompletedBar(7, C(800, 1.0650, 1.0700, 1.0400, 1.0500));
    retained.AddCompletedBar(7, C(800, 1.0650, 1.0700, 1.0400, 1.0500));
    assert(older.GetRecord().retracement0382.reached);
    assert(retained.GetRecord().retracement0382.reached);
    // A retained copy continues independently; TG3 active-collection pruning
    // cannot erase its already-created lifecycle state.
    older.AddCompletedBar(8, C(900, 1.0500, 1.1650, 1.0490, 1.1600));
    retained.AddCompletedBar(8, C(900, 1.0500, 1.1650, 1.0490, 1.1600));
    assert(older.GetRecord().firstDReached);
    assert(retained.GetRecord().firstDReached);
}

static void TestRetainedStateSurvivesActualTG3Pruning()
{
    TG3::Configuration config;
    config.retracementRatios = {.618};
    config.absolutePriceTolerance = .1;
    config.maxABAgeBars = 100;
    config.maxActiveABStructures = 1;
    TG3::FibonacciConfluenceTracker tg3(config);
    tg3.Advance(2, 2);
    tg3.ObserveConfirmedFractals({F(TG1A::FractalKind::Low, 2, 10)});
    tg3.Advance(3, 3);
    const auto available = tg3.ObserveConfirmedFractals(
        {F(TG1A::FractalKind::High, 3, 20)});
    assert(available.size() == 1 && tg3.ABStructures().size() == 1);
    Fib::Tracker retained(tg3.ABStructures().front(), 26.18);
    retained.AddCompletedBar(3, C(3, 18, 19, 17, 18));
    tg3.Advance(4, 4);
    tg3.ObserveConfirmedFractals({F(TG1A::FractalKind::Low, 4, 11)});
    assert(tg3.ABStructures().size() == 1 && tg3.ABCapacityEvictions() == 1);
    retained.AddCompletedBar(4, C(4, 18, 19, 16, 17));
    assert(retained.GetRecord().retracement0382.reached);
}

int main()
{
    TestRetracementPrices();
    TestProgressiveRetracementAndRally();
    TestAllRetracementsAndNoSameBarConfirmation();
    TestPenetrationDoesNotImplyCloseInvalidation();
    TestCloseBeyondARecordedSeparately();
    TestNoArbitraryDTimeout();
    TestDownDirectionSymmetry();
    TestRightCensoring();
    TestIndependentDHypothesesAndEntryExcursion();
    TestMultipleIndependentCandidatesSurviveLongPaths();
    TestRetainedStateSurvivesActualTG3Pruning();

    std::cout
        << "Causal Fibonacci retracement lifecycle tests PASS\n";
}
