#include "CausalPocketDetector.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using EA::Pocket::CausalPocketDetector;
using EA::Pocket::CompletedBar;
using EA::Pocket::PocketDirection;
using EA::Pocket::PocketObservation;

CompletedBar Bar(std::int64_t timestamp, double open, double high, double low,
    double close)
{
    return {timestamp, open, high, low, close};
}

void AddReferenceBars(CausalPocketDetector& detector, std::int64_t& timestamp,
    std::size_t count = CausalPocketDetector::kSourceDefaultLookbackBars)
{
    for (std::size_t index = 0; index < count; ++index)
    {
        const auto observation = detector.AddCompletedBar(
            Bar(timestamp++, 5.0, 10.0, 0.0, 5.0));
        assert(!observation.has_value());
    }
}

void AssertObservation(const PocketObservation& observation,
    PocketDirection direction, double lower, double upper, std::size_t eventBar,
    std::int64_t eventTime, std::size_t confirmationBar,
    std::int64_t confirmationTime)
{
    assert(observation.direction == direction);
    assert(observation.range.lower == lower);
    assert(observation.range.upper == upper);
    assert(observation.eventBar == eventBar);
    assert(observation.eventTimestamp == eventTime);
    assert(observation.confirmationBar == confirmationBar);
    assert(observation.confirmationTimestamp == confirmationTime);
    assert(observation.informationCutoffBar == confirmationBar);
    assert(observation.informationCutoffTimestamp == confirmationTime);
    assert(observation.eventBar < observation.confirmationBar);
    assert(observation.sourceTimeframe == "1h");
}

void TestBullishCreationAndAntiLookAhead()
{
    CausalPocketDetector detector("1h");
    std::int64_t timestamp = 100;
    AddReferenceBars(detector, timestamp);

    const auto beforeConfirmation = detector.AddCompletedBar(
        Bar(timestamp++, 10.0, 12.0, 9.0, 11.0));
    assert(!beforeConfirmation.has_value());

    const auto observation = detector.AddCompletedBar(
        Bar(timestamp++, 12.0, 13.0, 11.5, 12.0));
    assert(observation.has_value());
    AssertObservation(*observation, PocketDirection::Bullish, 10.0, 11.5, 15,
        115, 16, 116);
    assert(observation->ClosePrice() == 10.0);
    assert(observation->TouchPrice() == 11.5);
}

void TestBearishCreationAndEndpoints()
{
    CausalPocketDetector detector("1h");
    std::int64_t timestamp = 200;
    AddReferenceBars(detector, timestamp);
    assert(!detector.AddCompletedBar(Bar(timestamp++, 0.0, 1.0, -2.0, -1.0)));
    const auto observation = detector.AddCompletedBar(
        Bar(timestamp++, -2.0, -1.5, -3.0, -2.0));
    assert(observation.has_value());
    AssertObservation(*observation, PocketDirection::Bearish, -1.5, 0.0, 15,
        215, 16, 216);
    assert(observation->TouchPrice() == -1.5);
    assert(observation->ClosePrice() == 0.0);
}

void TestInsufficientHistoryAndExactBoundary()
{
    CausalPocketDetector insufficient("1h");
    std::int64_t timestamp = 300;
    AddReferenceBars(insufficient, timestamp,
        CausalPocketDetector::kSourceDefaultLookbackBars - 1);
    assert(!insufficient.AddCompletedBar(Bar(timestamp++, 10.0, 12.0, 9.0, 11.0)));
    assert(!insufficient.AddCompletedBar(Bar(timestamp++, 12.0, 13.0, 11.5, 12.0)));

    CausalPocketDetector exact("1h");
    timestamp = 400;
    AddReferenceBars(exact, timestamp);
    assert(!exact.AddCompletedBar(Bar(timestamp++, 10.0, 12.0, 9.0, 11.0)));
    assert(exact.AddCompletedBar(Bar(timestamp++, 12.0, 13.0, 11.5, 12.0)));
}

void TestTieAndStrictBoundaryPolicy()
{
    CausalPocketDetector tiedReference("1h");
    std::int64_t timestamp = 500;
    AddReferenceBars(tiedReference, timestamp);
    assert(!tiedReference.AddCompletedBar(Bar(timestamp++, 10.0, 12.0, 9.0, 11.0)));
    assert(tiedReference.AddCompletedBar(Bar(timestamp++, 12.0, 13.0, 11.5, 12.0)));

    CausalPocketDetector equalBoundary("1h");
    timestamp = 600;
    AddReferenceBars(equalBoundary, timestamp);
    assert(!equalBoundary.AddCompletedBar(Bar(timestamp++, 10.0, 12.0, 9.0, 11.0)));
    assert(!equalBoundary.AddCompletedBar(Bar(timestamp++, 11.0, 12.0, 10.0, 11.0)));
}

std::vector<PocketObservation> Replay(const std::vector<CompletedBar>& bars)
{
    CausalPocketDetector detector("1h");
    std::vector<PocketObservation> observations;
    for (const CompletedBar& bar : bars)
    {
        const auto observation = detector.AddCompletedBar(bar);
        if (observation.has_value()) observations.push_back(*observation);
    }
    return observations;
}

void TestReplayConsecutiveAndAlternatingDirections()
{
    std::vector<CompletedBar> bars;
    std::int64_t timestamp = 700;
    for (std::size_t index = 0;
         index < CausalPocketDetector::kSourceDefaultLookbackBars; ++index)
        bars.push_back(Bar(timestamp++, 5.0, 10.0, 0.0, 5.0));
    bars.push_back(Bar(timestamp++, 10.0, 12.0, 9.0, 11.0));
    bars.push_back(Bar(timestamp++, 12.0, 13.0, 11.5, 12.0));
    bars.push_back(Bar(timestamp++, 13.0, 15.0, 12.5, 14.0));
    bars.push_back(Bar(timestamp++, 15.0, 16.0, 14.5, 15.0));
    bars.push_back(Bar(timestamp++, 0.0, 1.0, -2.0, -1.0));
    bars.push_back(Bar(timestamp++, -2.0, -1.5, -3.0, -2.0));

    const std::vector<PocketObservation> first = Replay(bars);
    const std::vector<PocketObservation> second = Replay(bars);
    assert(first.size() == 3);
    assert(second.size() == first.size());
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        assert(first[index].direction == second[index].direction);
        assert(first[index].range.lower == second[index].range.lower);
        assert(first[index].range.upper == second[index].range.upper);
        assert(first[index].eventBar == second[index].eventBar);
        assert(first[index].confirmationBar == second[index].confirmationBar);
    }
    assert(first[0].direction == PocketDirection::Bullish);
    assert(first[1].direction == PocketDirection::Bullish);
    assert(first[2].direction == PocketDirection::Bearish);
}

void TestInvalidInputAndTimestamps()
{
    bool emptyTimeframeRejected = false;
    try
    {
        CausalPocketDetector detector("");
        (void)detector;
    }
    catch (const std::invalid_argument&)
    {
        emptyTimeframeRejected = true;
    }
    assert(emptyTimeframeRejected);

    CausalPocketDetector detector("1h");
    bool malformedRejected = false;
    try
    {
        (void)detector.AddCompletedBar(Bar(1,
            std::numeric_limits<double>::quiet_NaN(), 10.0, 0.0, 5.0));
    }
    catch (const std::invalid_argument&)
    {
        malformedRejected = true;
    }
    assert(malformedRejected);

    bool inconsistentRejected = false;
    try
    {
        (void)detector.AddCompletedBar(Bar(1, 5.0, 0.0, 10.0, 5.0));
    }
    catch (const std::invalid_argument&)
    {
        inconsistentRejected = true;
    }
    assert(inconsistentRejected);

    (void)detector.AddCompletedBar(Bar(2, 5.0, 10.0, 0.0, 5.0));
    bool timestampRejected = false;
    try
    {
        (void)detector.AddCompletedBar(Bar(2, 5.0, 10.0, 0.0, 5.0));
    }
    catch (const std::invalid_argument&)
    {
        timestampRejected = true;
    }
    assert(timestampRejected);
}

} // namespace

int main()
{
    TestBullishCreationAndAntiLookAhead();
    TestBearishCreationAndEndpoints();
    TestInsufficientHistoryAndExactBoundary();
    TestTieAndStrictBoundaryPolicy();
    TestReplayConsecutiveAndAlternatingDirections();
    TestInvalidInputAndTimestamps();
    std::cout << "CausalPocketDetectorTests passed\n";
}
