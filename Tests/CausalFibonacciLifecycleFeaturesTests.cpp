#include "CausalFibonacciLifecycleFeatures.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace LF = EA::CausalFibonacciLifecycleFeatures;
namespace Fib = EA::FibonacciResearch::RetracementLifecycle;
namespace TG3 = EA::TG3;

static TG3::ABStructure AB(TG3::ABDirection direction, std::size_t availabilityBar = 6)
{
    TG3::ABStructure ab;
    ab.identity.direction = direction;
    ab.identity.aBar = 0;
    ab.identity.aTimestamp = 100;
    ab.identity.bBar = 4;
    ab.identity.bTimestamp = 500;
    ab.identity.availabilityBar = availabilityBar;
    ab.identity.availabilityTimestamp = static_cast<std::int64_t>(100 * (availabilityBar + 1));
    if (direction == TG3::ABDirection::UpAB) {
        ab.aPrice = 1.0000;
        ab.bPrice = 1.1000;
    } else {
        ab.aPrice = 1.1000;
        ab.bPrice = 1.0000;
    }
    ab.priceRange = 0.1000;
    return ab;
}

static Fib::Occurrence O(std::size_t bar)
{
    return {bar, static_cast<std::int64_t>(100 * (bar + 1))};
}

static bool Near(float actual, double expected, double eps = 1e-6)
{
    return std::abs(static_cast<double>(actual) - expected) <= eps;
}

static void TestEmptyAggregate()
{
    const auto x = LF::Aggregate({}, 20);
    for (float v : x) assert(v == 0.0f);
}

static void TestExactTransformsAndDirectionIsolation()
{
    Fib::Record up1;
    up1.sourceAB = AB(TG3::ABDirection::UpAB);
    up1.retracement0382.reached = O(10);
    up1.retracement0382.firstDirectionalClose = O(12);
    up1.retracement0382.firstDirectionalBreak = O(13);
    up1.retracement0382.firstCloseBackThroughLevel = O(14);
    up1.retracement0500.reached = O(15);
    up1.retracement0618.reached = O(16);
    up1.firstAPenetration = O(17);
    up1.firstCloseBeyondA = O(18);

    Fib::Record up2;
    up2.sourceAB = AB(TG3::ABDirection::UpAB, 7);
    up2.retracement0382.reached = O(19);
    up2.retracement0382.firstDirectionalClose = O(18);

    Fib::Record down;
    down.sourceAB = AB(TG3::ABDirection::DownAB);
    down.retracement0618.reached = O(11);
    down.retracement0618.firstDirectionalClose = O(12);
    down.firstAPenetration = O(13);

    const auto x = LF::Aggregate({up1, up2, down}, 20);

    assert(Near(x[LF::Up0382ReachedCountLog], std::log1p(2.0)));
    assert(Near(x[LF::Up0382DirectionalCloseCountLog], std::log1p(2.0)));
    assert(Near(x[LF::Up0382DirectionalBreakCountLog], std::log1p(1.0)));
    assert(Near(x[LF::Up0382CloseBackThroughCountLog], std::log1p(1.0)));
    assert(Near(x[LF::Up0382ReachedYoungestAgeLog], std::log1p(1.0)));
    assert(Near(x[LF::Up0382DirectionalCloseYoungestAgeLog], std::log1p(2.0)));
    assert(Near(x[LF::Up0500ReachedCountLog], std::log1p(1.0)));
    assert(Near(x[LF::Up0500ReachedYoungestAgeLog], std::log1p(5.0)));
    assert(Near(x[LF::Up0618ReachedCountLog], std::log1p(1.0)));
    assert(Near(x[LF::Up0618ReachedYoungestAgeLog], std::log1p(4.0)));
    assert(Near(x[LF::UpAPenetrationCountLog], std::log1p(1.0)));
    assert(Near(x[LF::UpACloseBeyondCountLog], std::log1p(1.0)));
    assert(Near(x[LF::UpAPenetrationYoungestAgeLog], std::log1p(3.0)));
    assert(Near(x[LF::UpACloseBeyondYoungestAgeLog], std::log1p(2.0)));

    assert(x[LF::Down0382ReachedCountLog] == 0.0f);
    assert(Near(x[LF::Down0618ReachedCountLog], std::log1p(1.0)));
    assert(Near(x[LF::Down0618DirectionalCloseCountLog], std::log1p(1.0)));
    assert(Near(x[LF::Down0618ReachedYoungestAgeLog], std::log1p(9.0)));
    assert(Near(x[LF::Down0618DirectionalCloseYoungestAgeLog], std::log1p(8.0)));
    assert(Near(x[LF::DownAPenetrationCountLog], std::log1p(1.0)));
    assert(Near(x[LF::DownAPenetrationYoungestAgeLog], std::log1p(7.0)));
    assert(x[LF::DownACloseBeyondCountLog] == 0.0f);
    assert(x[LF::DownACloseBeyondYoungestAgeLog] == 0.0f);
}

static void TestAgeZeroIsDisambiguatedByCount()
{
    Fib::Record r;
    r.sourceAB = AB(TG3::ABDirection::UpAB);
    r.retracement0500.reached = O(20);
    const auto x = LF::Aggregate({r}, 20);
    assert(Near(x[LF::Up0500ReachedCountLog], std::log1p(1.0)));
    assert(x[LF::Up0500ReachedYoungestAgeLog] == 0.0f);
    assert(x[LF::Up0500DirectionalCloseCountLog] == 0.0f);
    assert(x[LF::Up0500DirectionalCloseYoungestAgeLog] == 0.0f);
}

static void TestFutureOccurrenceRejected()
{
    Fib::Record r;
    r.sourceAB = AB(TG3::ABDirection::UpAB);
    r.retracement0382.reached = O(21);
    bool threw = false;
    try { (void)LF::Aggregate({r}, 20); }
    catch (const std::logic_error&) { threw = true; }
    assert(threw);
}

static void TestTrackerDBarCanContributeAndNextBarFreezes()
{
    Fib::Tracker tracker(AB(TG3::ABDirection::UpAB), 1.1272);
    tracker.AddCompletedBar(6, {700, 1.0900, 1.0950, 1.0600, 1.0650});
    tracker.AddCompletedBar(7, {800, 1.0650, 1.1300, 1.0640, 1.1280});

    const auto& atD = tracker.GetRecord();
    assert(atD.firstDReached && atD.firstDReached->bar == 7);
    assert(atD.retracement0382.reached);
    assert(atD.retracement0382.firstDirectionalClose);

    const auto dRow = LF::Aggregate({atD}, 7);
    assert(Near(dRow[LF::Up0382ReachedCountLog], std::log1p(1.0)));
    assert(Near(dRow[LF::Up0382DirectionalCloseCountLog], std::log1p(1.0)));
    assert(dRow[LF::Up0382DirectionalCloseYoungestAgeLog] == 0.0f);

    tracker.AddCompletedBar(8, {900, 1.1280, 1.1400, 0.9900, 0.9950});
    const auto& afterD = tracker.GetRecord();
    assert(afterD.firstDReached && afterD.firstDReached->bar == 7);
    assert(!afterD.firstAPenetration);
    assert(!afterD.firstCloseBeyondA);

    const auto nextRow = LF::Aggregate({}, 8);
    for (float v : nextRow) assert(v == 0.0f);
}

int main()
{
    static_assert(LF::kFeatureCount == 44);
    TestEmptyAggregate();
    TestExactTransformsAndDirectionIsolation();
    TestAgeZeroIsDisambiguatedByCount();
    TestFutureOccurrenceRejected();
    TestTrackerDBarCanContributeAndNextBarFreezes();
    std::cout << "Causal Fibonacci lifecycle feature tests PASS\n";
    return 0;
}
