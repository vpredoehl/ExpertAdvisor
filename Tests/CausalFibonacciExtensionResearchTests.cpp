#include "CausalFibonacciExtensionResearch.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

namespace
{

namespace Fib = EA::FibonacciResearch;
using EA::TG1A::Candle;
using EA::TG3::ABDirection;
using EA::TG3::ABStructure;

constexpr std::int64_t kStart = 1'700'000'000;
constexpr std::int64_t kMinute = 60;

std::int64_t Timestamp(std::size_t bar)
{
    return kStart + static_cast<std::int64_t>(bar) * kMinute;
}

ABStructure UpAB()
{
    ABStructure result;
    result.identity = {ABDirection::UpAB, 1, Timestamp(1), 3, Timestamp(3),
                       5, Timestamp(5)};
    result.aPrice = 100.0;
    result.aConfirmationBar = 3;
    result.aConfirmationTimestamp = Timestamp(3);
    result.bPrice = 120.0;
    result.bConfirmationBar = 5;
    result.bConfirmationTimestamp = Timestamp(5);
    result.priceRange = 20.0;
    return result;
}

ABStructure DownAB()
{
    ABStructure result;
    result.identity = {ABDirection::DownAB, 1, Timestamp(1), 3, Timestamp(3),
                       5, Timestamp(5)};
    result.aPrice = 120.0;
    result.aConfirmationBar = 3;
    result.aConfirmationTimestamp = Timestamp(3);
    result.bPrice = 100.0;
    result.bConfirmationBar = 5;
    result.bConfirmationTimestamp = Timestamp(5);
    result.priceRange = 20.0;
    return result;
}

Fib::GroupingMetadata Grouping()
{
    return {"eurusdrmp", "15m", "validation_2023_2024", std::nullopt,
            std::nullopt};
}

Fib::Configuration FrozenConfiguration()
{
    return Fib::FrozenProspectiveEvaluationConfiguration(Grouping().symbol);
}

Candle Bar(std::size_t bar, double high, double low, double close)
{
    return {Timestamp(bar), close, high, low, close};
}

void AssertNear(double actual, double expected)
{
    assert(std::fabs(actual - expected) < 1e-12);
}

template <typename Function>
void AssertInvalid(Function function)
{
    bool rejected = false;
    try
    {
        function();
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

void TestBullishAndBearishGeometryAndExactFormula()
{
    const auto up1272 = Fib::CalculateExtensionLevel(UpAB(), 1.272, 0.1);
    const auto up1618 = Fib::CalculateExtensionLevel(UpAB(), 1.618, 0.1);
    const auto down1272 = Fib::CalculateExtensionLevel(DownAB(), 1.272, 0.1);
    const auto down1618 = Fib::CalculateExtensionLevel(DownAB(), 1.618, 0.1);

    // Up: A + r(B-A). Down: A - r(A-B).
    AssertNear(up1272.price, 100.0 + 1.272 * 20.0);
    AssertNear(up1618.price, 100.0 + 1.618 * 20.0);
    AssertNear(down1272.price, 120.0 - 1.272 * 20.0);
    AssertNear(down1618.price, 120.0 - 1.618 * 20.0);
    AssertNear(up1272.zoneLowerPrice, up1272.price - 0.1);
    AssertNear(up1272.zoneUpperPrice, up1272.price + 0.1);
    assert(up1272.availabilityBar == 5);
    assert(up1272.availabilityTimestamp == Timestamp(5));

    // Reflection about 110 proves directional symmetry.
    AssertNear(up1272.price + down1272.price, 220.0);
    AssertNear(up1618.price + down1618.price, 220.0);

    // Pullbacks delegate to the existing TG3 retracement formula.
    AssertNear(Fib::CalculatePullbackLevel(UpAB(), 0.382, 0.0).price,
               120.0 - 0.382 * 20.0);
    AssertNear(Fib::CalculatePullbackLevel(DownAB(), 0.382, 0.0).price,
               100.0 + 0.382 * 20.0);
}

void TestInvalidGeometryAndConfiguration()
{
    ABStructure degenerate = UpAB();
    degenerate.bPrice = degenerate.aPrice;
    degenerate.priceRange = 0.0;
    AssertInvalid([&]
    {
        (void)Fib::CalculateExtensionLevel(degenerate, 1.272, 0.0);
    });

    ABStructure nonfinite = UpAB();
    nonfinite.aPrice = std::numeric_limits<double>::quiet_NaN();
    AssertInvalid([&]
    {
        (void)Fib::CalculateExtensionLevel(nonfinite, 1.272, 0.0);
    });
    AssertInvalid([&]
    {
        (void)Fib::CalculateExtensionLevel(
            UpAB(), std::numeric_limits<double>::infinity(), 0.0);
    });
    AssertInvalid([&]
    {
        Fib::Configuration configuration;
        configuration.absolutePriceTolerance = -0.1;
        Fib::CausalExtensionTracker tracker(UpAB(), Grouping(), configuration);
        (void)tracker;
    });

    auto futureGrouping = Grouping();
    futureGrouping.preexistingVolatilityRegime =
        Fib::CausalLabel{"future", 6, Timestamp(6)};
    AssertInvalid([&]
    {
        Fib::CausalExtensionTracker tracker(
            UpAB(), futureGrouping, Fib::Configuration{});
        (void)tracker;
    });
}

void TestFrozenProspectivePolicyConfiguration()
{
    const Fib::Configuration configuration = FrozenConfiguration();
    assert(Fib::IsFrozenProspectiveEvaluationConfiguration(
        configuration, Grouping().symbol));
    assert(configuration.absolutePriceTolerance == 0.0001);
    assert(configuration.h1.maxBarsAfterEligibility == 20);
    assert(configuration.h2.maxBarsAfterEligibility == 20);
    assert(!configuration.h1.invalidation.has_value());
    assert(!configuration.h2.invalidation.has_value());
    assert(Fib::FrozenProspectiveEvaluationConfiguration("usdjpyrmp")
               .absolutePriceTolerance == 0.01);

    Fib::Configuration noncanonical = configuration;
    noncanonical.h1.maxBarsAfterEligibility = 19;
    assert(!Fib::IsFrozenProspectiveEvaluationConfiguration(
        noncanonical, Grouping().symbol));
}

void TestCausalAvailabilityTouchBeyondAndReversal()
{
    Fib::Configuration configuration;
    configuration.absolutePriceTolerance = 0.1;
    Fib::CausalExtensionTracker tracker(UpAB(), Grouping(), configuration);

    tracker.AddCompletedBar(3, Bar(3, 140.0, 99.0, 130.0));
    tracker.AddCompletedBar(4, Bar(4, 140.0, 99.0, 130.0));
    assert(!tracker.Record().extension1272Touched.has_value());
    assert(!tracker.Record().extension1272Beyond.has_value());

    tracker.AddCompletedBar(5, Bar(5, 125.0, 119.0, 124.0));
    assert(!tracker.Record().extension1272Touched.has_value());

    // Inclusive near-edge reach is a touch, but a close inside the tolerance
    // zone is not the strict far-edge close-beyond event.
    const double nearEdge = tracker.Record().extension1272.zoneLowerPrice;
    const double farEdge = tracker.Record().extension1272.zoneUpperPrice;
    tracker.AddCompletedBar(6, Bar(6, nearEdge, 124.0, nearEdge));
    assert(tracker.Record().extension1272Touched.has_value());
    assert(!tracker.Record().extension1272Beyond.has_value());

    tracker.AddCompletedBar(7, Bar(7, 126.0, 125.0, farEdge));
    assert(!tracker.Record().extension1272Beyond.has_value());
    tracker.AddCompletedBar(8, Bar(8, 126.0, 125.0,
                                  std::nextafter(farEdge, 126.0)));
    assert(tracker.Record().extension1272Beyond.has_value());
    assert(tracker.Record().h1.state == Fib::OutcomeState::Pending);

    // Rejection is not retroactively assigned to the touch bar; it needs a
    // later completed close through the near edge.
    assert(!tracker.Record().rejectionConfirmed.has_value());
    tracker.AddCompletedBar(9, Bar(9, 125.4, 124.0, 125.33));
    assert(tracker.Record().rejectionConfirmed.has_value());
    assert(tracker.Record().rejectionConfirmed->bar == 9);
    assert(tracker.Record().h2.state == Fib::OutcomeState::Pending);
}

void TestContinuationAndPullbacksRequireEligiblePriorState()
{
    Fib::Configuration configuration;
    configuration.absolutePriceTolerance = 0.0;
    Fib::CausalExtensionTracker tracker(UpAB(), Grouping(), configuration);

    // A 1.618 high before close-beyond is not H1 success.
    tracker.AddCompletedBar(5, Bar(5, 133.0, 119.0, 125.0));
    assert(tracker.Record().extension1272Touched.has_value());
    assert(!tracker.Record().extension1618ReachedAfterBeyond.has_value());
    assert(tracker.Record().h1.state == Fib::OutcomeState::NotEligible);

    // A broad pullback before causal rejection is not an H2 endpoint.
    tracker.AddCompletedBar(6, Bar(6, 133.0, 109.0, 126.0));
    assert(tracker.Record().extension1272Beyond.has_value());
    assert(!tracker.Record().extension1618ReachedAfterBeyond.has_value());
    assert(!tracker.Record().pullback0382ReachedAfterRejection.has_value());

    tracker.AddCompletedBar(7, Bar(7, 126.0, 124.0, 125.0));
    assert(tracker.Record().rejectionConfirmed.has_value());
    tracker.AddCompletedBar(8, Bar(8, 133.0, 112.0, 113.0));

    assert(tracker.Record().extension1618ReachedAfterBeyond.has_value());
    assert(tracker.Record().extension1618ReachedAfterBeyond->bar == 8);
    assert(tracker.Record().h1.state == Fib::OutcomeState::Success);
    assert(tracker.Record().h1.barsToTarget == 2);
    assert(tracker.Record().pullback0382ReachedAfterRejection.has_value());
    assert(tracker.Record().h2.state == Fib::OutcomeState::Success);
    assert(tracker.Record().h2.barsToTarget == 1);

    // 112 does not reach the deeper 0.500 or secondary 0.618 levels.
    assert(!tracker.Record().pullback0500ReachedAfterRejection.has_value());
    assert(!tracker.Record().pullback0618ReachedAfterRejection.has_value());
    tracker.AddCompletedBar(9, Bar(9, 113.0, 109.9, 111.0));
    assert(tracker.Record().pullback0500ReachedAfterRejection.has_value());
    assert(tracker.Record().h2Pullback0500Descriptive.state ==
           Fib::OutcomeState::Success);
    assert(!tracker.Record().pullback0618ReachedAfterRejection.has_value());
    tracker.AddCompletedBar(10, Bar(10, 111.0, 107.5, 108.0));
    assert(tracker.Record().pullback0618ReachedAfterRejection.has_value());
    assert(tracker.Record().h2Pullback0618Descriptive.state ==
           Fib::OutcomeState::Success);
}

void TestBearishEventSymmetry()
{
    Fib::CausalExtensionTracker tracker(
        DownAB(), Grouping(), FrozenConfiguration());
    tracker.AddCompletedBar(5, Bar(5, 101.0, 95.0, 96.0));
    assert(!tracker.Record().extension1272Touched.has_value());
    tracker.AddCompletedBar(6, Bar(6, 96.0, 94.5, 94.5));
    assert(tracker.Record().extension1272Touched.has_value());
    assert(tracker.Record().extension1272Beyond.has_value());
    tracker.AddCompletedBar(7, Bar(7, 95.0, 87.5, 88.0));
    assert(tracker.Record().h1.state == Fib::OutcomeState::Success);
    tracker.AddCompletedBar(8, Bar(8, 95.0, 93.0, 95.0));
    assert(tracker.Record().rejectionConfirmed.has_value());
    tracker.AddCompletedBar(9, Bar(9, 108.0, 95.0, 107.0));
    assert(tracker.Record().h2.state == Fib::OutcomeState::Success);
}

void TestFailureCensoringAndAmbiguity()
{
    Fib::Configuration horizon;
    horizon.h1.maxBarsAfterEligibility = 1;
    Fib::CausalExtensionTracker failed(UpAB(), Grouping(), horizon);
    failed.AddCompletedBar(5, Bar(5, 126.0, 120.0, 126.0));
    failed.AddCompletedBar(6, Bar(6, 127.0, 124.0, 126.0));
    assert(failed.Record().h1.state == Fib::OutcomeState::Failure);

    Fib::CausalExtensionTracker censored(
        UpAB(), Grouping(), Fib::Configuration{});
    censored.AddCompletedBar(5, Bar(5, 126.0, 120.0, 126.0));
    censored.AddCompletedBar(6, Bar(6, 127.0, 124.0, 126.0));
    censored.Finalize();
    assert(censored.Record().h1.state == Fib::OutcomeState::RightCensored);
    assert(censored.Record().h1.censorReason ==
           Fib::CensorReason::EndOfDataset);

    Fib::Configuration ambiguousConfiguration;
    ambiguousConfiguration.h1.invalidation =
        Fib::InvalidationBoundary{Fib::BoundarySide::AtOrBelow, 90.0};
    Fib::CausalExtensionTracker ambiguous(
        UpAB(), Grouping(), ambiguousConfiguration);
    ambiguous.AddCompletedBar(5, Bar(5, 126.0, 120.0, 126.0));
    ambiguous.AddCompletedBar(6, Bar(6, 140.0, 80.0, 100.0));
    assert(ambiguous.Record().h1.state ==
           Fib::OutcomeState::AmbiguousIneligible);
}

void TestH1FrozenHorizonBoundariesAndExactTolerance()
{
    Fib::CausalExtensionTracker boundary(
        UpAB(), Grouping(), FrozenConfiguration());
    boundary.AddCompletedBar(5, Bar(5, 126.0, 120.0, 126.0));
    for (std::size_t bar = 6; bar < 25; ++bar)
        boundary.AddCompletedBar(bar, Bar(bar, 127.0, 120.0, 126.0));
    assert(boundary.Record().h1.state == Fib::OutcomeState::Pending);
    const double exactTarget = boundary.Record().extension1618.zoneLowerPrice;
    boundary.AddCompletedBar(25, Bar(25, exactTarget, 120.0, 126.0));
    assert(boundary.Record().h1.state == Fib::OutcomeState::Success);
    assert(boundary.Record().h1.barsToTarget == 20);

    Fib::CausalExtensionTracker tooLate(
        UpAB(), Grouping(), FrozenConfiguration());
    tooLate.AddCompletedBar(5, Bar(5, 126.0, 120.0, 126.0));
    for (std::size_t bar = 6; bar <= 25; ++bar)
        tooLate.AddCompletedBar(bar, Bar(bar, 127.0, 120.0, 126.0));
    assert(tooLate.Record().h1.state == Fib::OutcomeState::Failure);
    tooLate.AddCompletedBar(26, Bar(26, 133.0, 120.0, 126.0));
    assert(!tooLate.Record().extension1618ReachedAfterBeyond.has_value());
    assert(tooLate.Record().h1.state == Fib::OutcomeState::Failure);
    assert(!tooLate.Record().h1.barsToTarget.has_value());

    Fib::CausalExtensionTracker studyEnd(
        UpAB(), Grouping(), FrozenConfiguration());
    studyEnd.AddCompletedBar(5, Bar(5, 126.0, 120.0, 126.0));
    studyEnd.AddCompletedBar(6, Bar(6, 127.0, 120.0, 126.0));
    studyEnd.Finalize(Fib::CensorReason::StudyWindowBoundary);
    assert(studyEnd.Record().h1.state == Fib::OutcomeState::RightCensored);
    assert(studyEnd.Record().h1.censorReason ==
           Fib::CensorReason::StudyWindowBoundary);
}

void TestH2CausalStartHorizonCensoringAndAmbiguity()
{
    Fib::CausalExtensionTracker rejectionBar(
        UpAB(), Grouping(), FrozenConfiguration());
    const double touch = rejectionBar.Record().extension1272.zoneLowerPrice;
    rejectionBar.AddCompletedBar(5, Bar(5, touch, 120.0, 125.0));
    rejectionBar.AddCompletedBar(6, Bar(6, 126.0, 100.0, 125.0));
    assert(rejectionBar.Record().rejectionConfirmed->bar == 6);
    assert(!rejectionBar.Record().pullback0382ReachedAfterRejection.has_value());
    assert(rejectionBar.Record().h2.state == Fib::OutcomeState::Pending);
    rejectionBar.AddCompletedBar(7, Bar(7, 126.0, 100.0, 110.0));
    assert(rejectionBar.Record().h2.state == Fib::OutcomeState::Success);
    assert(rejectionBar.Record().h2.barsToTarget == 1);

    Fib::CausalExtensionTracker boundary(
        UpAB(), Grouping(), FrozenConfiguration());
    boundary.AddCompletedBar(5, Bar(5, touch, 120.0, 125.0));
    boundary.AddCompletedBar(6, Bar(6, 126.0, 120.0, 125.0));
    for (std::size_t bar = 7; bar < 26; ++bar)
        boundary.AddCompletedBar(bar, Bar(bar, 126.0, 120.0, 125.0));
    const double exactPullback = boundary.Record().pullback0382.zoneUpperPrice;
    boundary.AddCompletedBar(26, Bar(26, 126.0, exactPullback, 120.0));
    assert(boundary.Record().h2.state == Fib::OutcomeState::Success);
    assert(boundary.Record().h2.barsToTarget == 20);

    Fib::CausalExtensionTracker tooLate(
        UpAB(), Grouping(), FrozenConfiguration());
    tooLate.AddCompletedBar(5, Bar(5, touch, 120.0, 125.0));
    tooLate.AddCompletedBar(6, Bar(6, 126.0, 120.0, 125.0));
    for (std::size_t bar = 7; bar <= 26; ++bar)
        tooLate.AddCompletedBar(bar, Bar(bar, 126.0, 120.0, 125.0));
    assert(tooLate.Record().h2.state == Fib::OutcomeState::Failure);
    assert(tooLate.Record().h2Pullback0500Descriptive.state ==
           Fib::OutcomeState::Failure);
    assert(tooLate.Record().h2Pullback0618Descriptive.state ==
           Fib::OutcomeState::Failure);
    tooLate.AddCompletedBar(27, Bar(27, 126.0, 100.0, 110.0));
    assert(!tooLate.Record().pullback0382ReachedAfterRejection.has_value());
    assert(tooLate.Record().h2.state == Fib::OutcomeState::Failure);

    Fib::CausalExtensionTracker datasetEnd(
        UpAB(), Grouping(), FrozenConfiguration());
    datasetEnd.AddCompletedBar(5, Bar(5, touch, 120.0, 125.0));
    datasetEnd.AddCompletedBar(6, Bar(6, 126.0, 120.0, 125.0));
    datasetEnd.Finalize();
    assert(datasetEnd.Record().h2.state == Fib::OutcomeState::RightCensored);
    assert(datasetEnd.Record().h2.censorReason ==
           Fib::CensorReason::EndOfDataset);
    assert(datasetEnd.Record().h2Pullback0500Descriptive.state ==
           Fib::OutcomeState::RightCensored);

    Fib::CausalExtensionTracker studyEnd(
        UpAB(), Grouping(), FrozenConfiguration());
    studyEnd.AddCompletedBar(5, Bar(5, touch, 120.0, 125.0));
    studyEnd.AddCompletedBar(6, Bar(6, 126.0, 120.0, 125.0));
    studyEnd.Finalize(Fib::CensorReason::StudyWindowBoundary);
    assert(studyEnd.Record().h2.state == Fib::OutcomeState::RightCensored);
    assert(studyEnd.Record().h2.censorReason ==
           Fib::CensorReason::StudyWindowBoundary);

    Fib::Configuration ambiguity = FrozenConfiguration();
    ambiguity.h2.invalidation =
        Fib::InvalidationBoundary{Fib::BoundarySide::AtOrAbove, 140.0};
    Fib::CausalExtensionTracker ambiguous(UpAB(), Grouping(), ambiguity);
    ambiguous.AddCompletedBar(5, Bar(5, touch, 120.0, 125.0));
    ambiguous.AddCompletedBar(6, Bar(6, 126.0, 120.0, 125.0));
    ambiguous.AddCompletedBar(7, Bar(7, 140.0, 100.0, 120.0));
    assert(ambiguous.Record().h2.state ==
           Fib::OutcomeState::AmbiguousIneligible);
    assert(ambiguous.Record().h2Pullback0500Descriptive.state ==
           Fib::OutcomeState::AmbiguousIneligible);
    assert(ambiguous.Record().h2Pullback0618Descriptive.state ==
           Fib::OutcomeState::AmbiguousIneligible);
}

void TestH3FailsClosedWithoutAuthoritativeDContract()
{
    Fib::CausalExtensionTracker tracker(
        UpAB(), Grouping(), Fib::Configuration{});
    assert(!tracker.Record().dPoint.has_value());
    assert(tracker.Record().dContractStatus ==
           Fib::DContractStatus::MissingAuthoritativeDefinition);
    assert(tracker.Record().h3Pullback0382.state ==
           Fib::OutcomeState::AmbiguousIneligible);
    assert(tracker.Record().h3Pullback0500.state ==
           Fib::OutcomeState::AmbiguousIneligible);
    assert(tracker.Record().h3Pullback0382Or0500.detail.find(
        "missing_authoritative_d_extension_contract") != std::string::npos);
}

Fib::ObservationRecord SuccessfulRecord()
{
    Fib::CausalExtensionTracker tracker(
        UpAB(), Grouping(), FrozenConfiguration());
    tracker.AddCompletedBar(5, Bar(5, 126.0, 120.0, 126.0));
    tracker.AddCompletedBar(6, Bar(6, 133.0, 124.0, 132.0));
    tracker.AddCompletedBar(7, Bar(7, 126.0, 124.0, 125.0));
    tracker.AddCompletedBar(8, Bar(8, 126.0, 112.0, 113.0));
    tracker.Finalize();
    return tracker.Record();
}

Fib::ObservationRecord FailedH1Record()
{
    auto grouping = Grouping();
    grouping.timeframe = "15m_h1_failure_fixture";
    Fib::CausalExtensionTracker tracker(
        UpAB(), grouping,
        Fib::FrozenProspectiveEvaluationConfiguration(grouping.symbol));
    tracker.AddCompletedBar(5, Bar(5, 126.0, 120.0, 126.0));
    for (std::size_t bar = 6; bar <= 25; ++bar)
        tracker.AddCompletedBar(bar, Bar(bar, 127.0, 120.0, 126.0));
    assert(tracker.Record().h1.state == Fib::OutcomeState::Failure);
    return tracker.Record();
}

Fib::ObservationRecord FailedH2Record()
{
    auto grouping = Grouping();
    grouping.timeframe = "15m_h2_failure_fixture";
    Fib::CausalExtensionTracker tracker(
        UpAB(), grouping,
        Fib::FrozenProspectiveEvaluationConfiguration(grouping.symbol));
    const double touch = tracker.Record().extension1272.zoneLowerPrice;
    tracker.AddCompletedBar(5, Bar(5, touch, 120.0, 125.0));
    tracker.AddCompletedBar(6, Bar(6, 126.0, 120.0, 125.0));
    for (std::size_t bar = 7; bar <= 26; ++bar)
        tracker.AddCompletedBar(bar, Bar(bar, 126.0, 120.0, 125.0));
    assert(tracker.Record().h2.state == Fib::OutcomeState::Failure);
    return tracker.Record();
}

void TestStableIdentityDeduplicationStatisticsAndSchema()
{
    const Fib::ObservationRecord record = SuccessfulRecord();
    assert(record.h1.state == Fib::OutcomeState::Success);
    assert(record.h2.state == Fib::OutcomeState::Success);
    assert(record.eventIdentity ==
           Fib::StructuralEventIdentity(record.grouping, record.sourceAB));

    Fib::AggregateAccumulator accumulator;
    accumulator.Add(record);
    accumulator.Add(record); // repeated materialization of one structure
    const auto observation = accumulator.Summarize(
        Fib::Endpoint::H2Pullback0382, Fib::CountingUnit::Observation);
    const auto unique = accumulator.Summarize(
        Fib::Endpoint::H2Pullback0382,
        Fib::CountingUnit::UniqueStructuralEvent);
    assert(observation.observations == 2);
    assert(observation.uniqueStructuralEvents == 1);
    assert(observation.successes == 2);
    assert(unique.successes == 1);
    assert(unique.measured.rate == 1.0);
    assert(unique.expertBenchmark == 0.8);
    AssertNear(*unique.measuredMinusBenchmark, 0.2);
    assert(unique.timeToTarget.count == 1);
    const auto cohorts = accumulator.SummarizeCohorts(
        Fib::Endpoint::H2Pullback0382,
        Fib::CountingUnit::UniqueStructuralEvent);
    assert(cohorts.size() == 1);
    assert(cohorts.begin()->first.symbol == "eurusdrmp");
    assert(cohorts.begin()->first.direction == "up_ab");
    assert(cohorts.begin()->first.calendarPeriod == "validation_2023_2024");
    assert(cohorts.begin()->second.successes == 1);

    Fib::ObservationRecord mislabeledPeriod = record;
    mislabeledPeriod.grouping.calendarPeriod = "incorrect_source_period";
    assert(Fib::CohortFor(mislabeledPeriod, Fib::Endpoint::H1Extension1618)
               .calendarPeriod == "validation_2023_2024");

    const auto h1 = accumulator.Summarize(
        Fib::Endpoint::H1Extension1618,
        Fib::CountingUnit::UniqueStructuralEvent);
    assert(!h1.expertBenchmark.has_value());

    Fib::AggregateAccumulator h1Mixed;
    h1Mixed.Add(record);
    h1Mixed.Add(FailedH1Record());
    const auto h1Binary = h1Mixed.Summarize(
        Fib::Endpoint::H1Extension1618,
        Fib::CountingUnit::UniqueStructuralEvent);
    assert(h1Binary.eligible == 2);
    assert(h1Binary.successes == 1);
    assert(h1Binary.failures == 1);
    assert(h1Binary.measured.denominator == 2);
    assert(h1Binary.measured.rate == 0.5);
    const auto expectedWilson = EA::TG4::Wilson95(1, 1);
    assert(h1Binary.measured.lower95 == expectedWilson.lower95);
    assert(h1Binary.measured.upper95 == expectedWilson.upper95);

    Fib::AggregateAccumulator h2Mixed;
    h2Mixed.Add(record);
    h2Mixed.Add(FailedH2Record());
    const auto h2Binary = h2Mixed.Summarize(
        Fib::Endpoint::H2Pullback0382,
        Fib::CountingUnit::UniqueStructuralEvent);
    assert(h2Binary.eligible == 2);
    assert(h2Binary.successes == 1);
    assert(h2Binary.failures == 1);
    assert(h2Binary.measured.denominator == 2);
    const auto h3 = accumulator.Summarize(
        Fib::Endpoint::H3Pullback0382Or0500,
        Fib::CountingUnit::UniqueStructuralEvent);
    assert(h3.eligible == 0);
    assert(h3.ambiguousOrIneligible == 1);
    assert(h3.expertBenchmark == 0.8);
    assert(!h3.measured.rate.has_value());

    const std::string header = Fib::ObservationCsvHeader();
    const std::string first = Fib::ObservationCsvRow(record);
    const std::string second = Fib::ObservationCsvRow(record);
    assert(first == second);
    assert(first.find("causal-fibonacci-extension-observation-v2") == 0);
    assert(std::count(header.begin(), header.end(), ',') ==
           std::count(first.begin(), first.end(), ','));
    assert(header.find("d_confirmation_timestamp") != std::string::npos);
    assert(header.find("evaluation_policy_version") != std::string::npos);
    assert(header.find("h2_0_500_state") != std::string::npos);
    assert(header.find("h2_0_618_state") != std::string::npos);
    assert(header.find("h3_union_state") != std::string::npos);
    const std::string summaryHeader = Fib::EndpointSummaryCsvHeader();
    const std::string summaryRow = Fib::EndpointSummaryCsvRow(
        Fib::Endpoint::H2Pullback0382,
        Fib::CountingUnit::UniqueStructuralEvent, unique);
    assert(summaryRow == Fib::EndpointSummaryCsvRow(
        Fib::Endpoint::H2Pullback0382,
        Fib::CountingUnit::UniqueStructuralEvent, unique));
    assert(std::count(summaryHeader.begin(), summaryHeader.end(), ',') ==
           std::count(summaryRow.begin(), summaryRow.end(), ','));

    Fib::ObservationRecord inconsistent = record;
    inconsistent.h2.detail += ":different";
    AssertInvalid([&] { accumulator.Add(inconsistent); });

    Fib::ObservationRecord noncanonical = record;
    noncanonical.configuration.h1.maxBarsAfterEligibility = 19;
    Fib::AggregateAccumulator rejected;
    AssertInvalid([&] { rejected.Add(noncanonical); });
}

void TestFutureBarsCannotRewriteFrozenGeometryOrEarlierEvents()
{
    Fib::CausalExtensionTracker tracker(
        UpAB(), Grouping(), Fib::Configuration{});
    tracker.AddCompletedBar(5, Bar(5, 125.0, 120.0, 124.0));
    const auto identity = tracker.Record().eventIdentity;
    const auto level = tracker.Record().extension1272;
    tracker.AddCompletedBar(6, Bar(6, 126.0, 120.0, 126.0));
    const auto beyond = tracker.Record().extension1272Beyond;
    tracker.AddCompletedBar(7, Bar(7, 140.0, 80.0, 100.0));
    assert(tracker.Record().eventIdentity == identity);
    assert(tracker.Record().extension1272.price == level.price);
    assert(tracker.Record().extension1272.availabilityBar ==
           level.availabilityBar);
    assert(tracker.Record().extension1272Beyond == beyond);
}

} // namespace

int main()
{
    TestBullishAndBearishGeometryAndExactFormula();
    TestInvalidGeometryAndConfiguration();
    TestFrozenProspectivePolicyConfiguration();
    TestCausalAvailabilityTouchBeyondAndReversal();
    TestContinuationAndPullbacksRequireEligiblePriorState();
    TestBearishEventSymmetry();
    TestFailureCensoringAndAmbiguity();
    TestH1FrozenHorizonBoundariesAndExactTolerance();
    TestH2CausalStartHorizonCensoringAndAmbiguity();
    TestH3FailsClosedWithoutAuthoritativeDContract();
    TestStableIdentityDeduplicationStatisticsAndSchema();
    TestFutureBarsCannotRewriteFrozenGeometryOrEarlierEvents();
    std::cout << "Causal Fibonacci extension research tests passed\n";
    return 0;
}
