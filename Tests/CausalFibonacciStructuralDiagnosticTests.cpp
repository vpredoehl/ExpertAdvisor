#include "CausalFibonacciConfluenceIntegration.hpp"
#include "CausalFibonacciExtensionResearch.hpp"
#include "CausalFibonacciStructuralDiagnostic.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace {
using EA::TG1A::Candle;
using EA::TG1A::ConfirmedFractal;
using EA::TG1A::FractalKind;

ConfirmedFractal Fractal(FractalKind kind, std::size_t bar, double price) {
  return {kind,  bar - 2, static_cast<std::int64_t>(bar - 2),
          price, bar,     static_cast<std::int64_t>(bar)};
}

EA::TG3::Configuration Config() {
  EA::TG3::Configuration result;
  result.retracementRatios = {.618};
  result.absolutePriceTolerance = .1;
  result.directionalStudyPolicy =
      EA::TG3::DirectionalStudyPolicy::SymmetricDirectionalDiagnostic;
  result.maxABAgeBars = 20;
  result.maxActiveABStructures = 10;
  return result;
}

void TestActualTrackerCausalityAndBounds() {
  const auto frozen = EA::TG4::LoadConfigurationFile(
      "Scripts/tg4_analysis_config.frozen_v1.conf");
  assert(frozen.name == "tg4a-first-study-preconfirmation-frozen-v1");
  assert(frozen.fibonacci.maxABAgeBars == 2048);
  assert(frozen.fibonacci.maxActiveABStructures == 512);
  assert(frozen.fibonacci.directionalStudyPolicy ==
         EA::TG3::DirectionalStudyPolicy::SourceUTLUpABOnly);
  auto config = Config();
  config.maxABAgeBars = 1;
  config.maxActiveABStructures = 1;
  EA::TG3::FibonacciConfluenceTracker tracker(config);
  tracker.Advance(2, 2);
  assert(tracker.ObserveConfirmedFractals({Fractal(FractalKind::Low, 2, 10)})
             .empty());
  tracker.Advance(3, 3);
  // B confirmation is the first possible A/B availability; no pivot is used
  // early.
  const auto first =
      tracker.ObserveConfirmedFractals({Fractal(FractalKind::High, 3, 20)});
  assert(first.size() == 1 && tracker.ABStructures().size() == 1);
  tracker.Advance(4, 4);
  (void)tracker.ObserveConfirmedFractals({Fractal(FractalKind::Low, 4, 11)});
  assert(tracker.ABCapacityEvictions() == 1);
  tracker.Advance(5, 5);
  (void)tracker.ObserveConfirmedFractals({});
  tracker.Advance(6, 6);
  assert(tracker.ABAgeExpirations() >= 1);

  auto simultaneousConfig = Config();
  simultaneousConfig.maxABAgeBars = 100;
  simultaneousConfig.maxActiveABStructures = 10;
  EA::TG3::FibonacciConfluenceTracker simultaneous(simultaneousConfig);
  simultaneous.Advance(2, 2);
  simultaneous.ObserveConfirmedFractals({Fractal(FractalKind::Low, 2, 10)});
  simultaneous.Advance(3, 3);
  simultaneous.ObserveConfirmedFractals({Fractal(FractalKind::High, 3, 20)});
  simultaneous.Advance(4, 4);
  simultaneous.ObserveConfirmedFractals({Fractal(FractalKind::Low, 4, 11)});
  simultaneous.Advance(5, 5);
  simultaneous.ObserveConfirmedFractals({Fractal(FractalKind::High, 5, 21)});
  std::size_t upCount = 0, downCount = 0;
  for (const auto &ab : simultaneous.ABStructures()) {
    if (ab.identity.direction == EA::TG3::ABDirection::UpAB)
      ++upCount;
    else
      ++downCount;
  }
  assert(upCount >= 2);   // simultaneous same-direction A/B structures
  assert(downCount >= 1); // simultaneous opposing-direction structure
}

void TestFrozenBoundariesAndMirrorNormalization() {
  EA::TG3::ABStructure up{
      {EA::TG3::ABDirection::UpAB, 1, 1, 3, 3, 5, 5}, 100, 3, 3, 120, 5, 5, 20};
  EA::TG3::ABStructure down = up;
  down.identity.direction = EA::TG3::ABDirection::DownAB;
  down.aPrice = 120;
  down.bPrice = 100;
  const auto upLevel =
      EA::FibonacciResearch::CalculateExtensionLevel(up, 1.272, .1);
  const auto downLevel =
      EA::FibonacciResearch::CalculateExtensionLevel(down, 1.272, .1);
  assert(upLevel.zoneLowerPrice < upLevel.price &&
         upLevel.zoneUpperPrice > upLevel.price);
  assert(std::fabs(upLevel.price + downLevel.price - 220.0) < 1e-12);
  // Strict far edge is deliberately not a beyond close; a later near-edge
  // close is the only valid rejection ordering after a prior touch.
  assert(!(upLevel.zoneUpperPrice > upLevel.zoneUpperPrice));
  const auto normalizedUp = EA::FibonacciDiagnostic::NormalizeDistance(
      true, upLevel.price, 110.0, 2.0, .1);
  const auto normalizedDown = EA::FibonacciDiagnostic::NormalizeDistance(
      false, downLevel.price, 110.0, 2.0, .1);
  assert(normalizedUp && normalizedDown);
  assert(std::fabs(normalizedUp->value - normalizedDown->value) < 1e-12);
  assert(!std::isfinite(std::numeric_limits<double>::quiet_NaN()));
}

void AssertRepresentativeBound(double input, double representative) {
  if (input == 0.0) {
    assert(representative == 0.0);
    return;
  }
  const double relative = std::fabs(representative - input) / std::fabs(input);
  assert(relative <=
         EA::FibonacciDiagnostic::DistanceDistribution::MaxRelativeError() +
             1e-12);
}

void TestDistanceDistributionAccuracyOrderingAndMerge() {
  using EA::FibonacciDiagnostic::DistanceDistribution;
  const std::array<double, 11> values{
      -std::numeric_limits<double>::max(),
      -1000.0,
      -std::numeric_limits<double>::denorm_min(),
      -0.0000000001,
      -0.0,
      0.0,
      std::numeric_limits<double>::denorm_min(),
      0.0000000001,
      1.0,
      1000.0,
      std::numeric_limits<double>::max()};
  DistanceDistribution whole, left, right;
  for (std::size_t i = 0; i < values.size(); ++i) {
    whole.Add(values[i]);
    (i % 2 ? right : left).Add(values[i]);
  }
  left.Merge(right);
  assert(whole.Count() == values.size());
  assert(left.Count() == whole.Count());
  for (double q : {.01, .5, .99, 1.0}) {
    const auto expected = whole.Quantile(q);
    const auto merged = left.Quantile(q);
    assert(expected && merged && *expected == *merged);
  }
  const auto minimum = whole.Quantile(.01);
  const auto maximum = whole.Quantile(1.0);
  const auto absoluteMaximum = whole.Quantile(1.0, true);
  assert(minimum && maximum && absoluteMaximum && *minimum < 0.0 &&
         *maximum > 0.0);
  assert(std::isfinite(*absoluteMaximum));
  // Each singleton bin's returned representative must meet the published
  // relative-value bound, including subnormal-adjacent and extreme values.
  for (double input : values) {
    if (input == 0.0)
      continue;
    DistanceDistribution one;
    one.Add(input);
    const auto representative = one.Quantile(1.0);
    assert(representative);
    AssertRepresentativeBound(input, *representative);
  }

  EA::FibonacciDiagnostic::ExactDistribution exact;
  for (double value : {1.0, 2.0, 3.0, 4.0})
    exact.Add(value);
  assert(exact.Quantile(.25) == std::optional<double>(1.0));
  assert(exact.Quantile(.5) == std::optional<double>(2.0));
  assert(exact.Quantile(.75) == std::optional<double>(3.0));
  assert(exact.Quantile(1.0) == std::optional<double>(4.0));
  assert(!exact.Quantile(0.0) && !exact.Quantile(1.01));
}

void TestNormalizationAndEventHorizon() {
  using EA::FibonacciDiagnostic::AdvanceH1H2EventState;
  using EA::FibonacciDiagnostic::ClassifyEventRelevance;
  using EA::FibonacciDiagnostic::EventAgeAtBar;
  using EA::FibonacciDiagnostic::EventRelevance;
  using EA::FibonacciDiagnostic::H1H2EventState;
  using EA::FibonacciDiagnostic::IsEventRelevantAtBar;
  using EA::FibonacciDiagnostic::NormalizeDistance;
  const auto missing = NormalizeDistance(true, 11.0, 10.0, std::nullopt, .1);
  const auto nan = NormalizeDistance(
      true, 11.0, 10.0, std::numeric_limits<double>::quiet_NaN(), .1);
  const auto small = NormalizeDistance(true, 11.0, 10.0, .05, .1);
  const auto atr = NormalizeDistance(false, 11.0, 10.0, .5, .1);
  assert(missing && missing->usedPipFallback &&
         std::fabs(missing->value - 10.0) < 1e-12);
  assert(nan && nan->usedPipFallback && std::fabs(nan->value - 10.0) < 1e-12);
  assert(small && small->usedPipFallback);
  assert(atr && !atr->usedPipFallback && std::fabs(atr->value + 2.0) < 1e-12);
  assert(!NormalizeDistance(true, 11.0, 10.0, .5, 0.0));
  assert(!NormalizeDistance(true, std::numeric_limits<double>::quiet_NaN(),
                            10.0, .5, .1));
  assert(!NormalizeDistance(true, 11.0, std::numeric_limits<double>::infinity(),
                            .5, .1));
  assert(IsEventRelevantAtBar(10, 10));  // age 0
  assert(IsEventRelevantAtBar(10, 30));  // age 20 included
  assert(!IsEventRelevantAtBar(10, 31)); // age 21 excluded
  assert(EventAgeAtBar(10, 10) == 0);
  assert(EventAgeAtBar(10, 11) == 1);
  assert(EventAgeAtBar(10, 12) == 2);
  H1H2EventState events;
  AdvanceH1H2EventState(events, 5, true, true, true);
  assert(events.touchBar && *events.touchBar == 5);
  assert(events.h1BeyondBar && *events.h1BeyondBar == 5);
  assert(!events.h2RejectionBar); // same-bar touch cannot reject
  AdvanceH1H2EventState(events, 6, false, false, true);
  assert(events.h2RejectionBar && *events.h2RejectionBar == 6);
  AdvanceH1H2EventState(events, 7, true, true, true);
  assert(*events.h1BeyondBar == 5 &&
         *events.h2RejectionBar == 6); // one-count-only events
  assert(ClassifyEventRelevance(10U, std::nullopt, 10) ==
         EventRelevance::H1Only);
  assert(ClassifyEventRelevance(std::nullopt, 10U, 10) ==
         EventRelevance::H2Only);
  assert(ClassifyEventRelevance(10U, 11U, 11) ==
         EventRelevance::Both); // union is one observation
  assert(ClassifyEventRelevance(10U, 10U, 31) == EventRelevance::None);

  EA::FibonacciDiagnostic::Summary accounting;
  EA::FibonacciDiagnostic::RecordNormalizedDistance(
      accounting, 0, 0, NormalizeDistance(true, 11.0, 10.0, std::nullopt, .1));
  EA::FibonacciDiagnostic::RecordNormalizedDistance(
      accounting, 0, 0,
      NormalizeDistance(true, std::numeric_limits<double>::quiet_NaN(), 10.0,
                        .5, .1));
  EA::FibonacciDiagnostic::RecordNormalizedDistance(
      accounting, 0, 0,
      NormalizeDistance(true, 11.0, std::numeric_limits<double>::infinity(), .5,
                        .1));
  EA::FibonacciDiagnostic::RecordNormalizedDistance(
      accounting, 0, 0, NormalizeDistance(true, 11.0, 10.0, .5, 0.0));
  assert(accounting.finiteDistances[0][0] == 1 &&
         accounting.pipFallbacks[0][0] == 1 &&
         accounting.invalidDistances[0][0] == 3 &&
         accounting.distances[0][0].Count() == 1);
}

void TestOutcomeBlindDependencyBoundary() {
  std::ifstream source("Sources/CausalFibonacciStructuralDiagnostic.cpp");
  std::ifstream header("Headers/CausalFibonacciStructuralDiagnostic.hpp");
  assert(source && header);
  const std::string contents{std::istreambuf_iterator<char>(source), {}};
  const std::string declarations{std::istreambuf_iterator<char>(header), {}};
  for (const std::string &forbidden : {"TargetTracker", "TargetResolution",
                                       "OutcomeTracker", "OutcomeResolution"}) {
    assert(contents.find(forbidden) == std::string::npos);
    assert(declarations.find(forbidden) == std::string::npos);
  }
}

void TestSummaryMergeAndExplicitSchema() {
  using EA::FibonacciDiagnostic::Summary;
  Summary a, b;
  a.symbol = b.symbol = "test";
  a.bars = 2;
  b.bars = 3;
  a.firstMeasurementTimestamp = 20;
  b.firstMeasurementTimestamp = 10;
  a.lastMeasurementTimestamp = 30;
  b.lastMeasurementTimestamp = 40;
  a.zeroBars[0] = 1;
  b.oneBars[0] = 2;
  a.geometric[0].Add(1);
  b.geometric[0].Add(7);
  a.h1Bars[1].Add(2);
  b.h2Bars[2].Add(3);
  a.relevantBars[0].Add(4);
  b.abAges.Add(5);
  a.h1Ages.Add(6);
  b.h2Ages.Add(7);
  a.distances[0][0].Add(-2.0);
  b.distances[1][4].Add(3.0);
  a.finiteDistances[0][0] = 1;
  b.invalidDistances[1][4] = 2;
  b.pipFallbacks[1][4] = 1;
  a.capacityEvictions = 4;
  b.capacityEvictions = 6;
  a.ageExpirations = 7;
  b.ageExpirations = 8;
  a.noTouch = 9;
  b.touchObserved = 10;
  a.beyondObserved = 11;
  b.rejectionConfirmed = 12;
  a.h1Age0To20 = 13;
  b.h1AgeOver20 = 14;
  a.h2Age0To20 = 15;
  b.h2AgeOver20 = 16;
  a.h1SameDirectionBars = 17;
  b.h1OpposingDirectionBars = 18;
  a.h2SameDirectionBars = 19;
  b.h2OpposingDirectionBars = 20;
  a.relevantBoth = 1;
  b.relevantH1Only = 2;
  a.Merge(b);
  assert(a.bars == 5 && a.firstMeasurementTimestamp == 10 &&
         a.lastMeasurementTimestamp == 40);
  assert(a.geometric[0].Count() == 2 && a.h1Bars[1].Count() == 1 &&
         a.h2Bars[2].Count() == 1);
  assert(a.abAges.Count() == 1 && a.h1Ages.Count() == 1 &&
         a.h2Ages.Count() == 1);
  assert(a.distances[0][0].Count() == 1 && a.distances[1][4].Count() == 1);
  assert(a.finiteDistances[0][0] == 1 && a.invalidDistances[1][4] == 2 &&
         a.pipFallbacks[1][4] == 1);
  assert(a.capacityEvictions == 10 && a.ageExpirations == 15 &&
         a.noTouch == 9 && a.touchObserved == 10 && a.beyondObserved == 11 &&
         a.rejectionConfirmed == 12 && a.h1Age0To20 == 13 &&
         a.h1AgeOver20 == 14 && a.h2Age0To20 == 15 && a.h2AgeOver20 == 16 &&
         a.h1SameDirectionBars == 17 && a.h1OpposingDirectionBars == 18 &&
         a.h2SameDirectionBars == 19 && a.h2OpposingDirectionBars == 20);
  const std::string header = Summary::CsvHeader(), row = a.ToCsv();
  assert(header.find("h1_up_p50") != std::string::npos);
  assert(header.find("relevant_both_direction_pct") != std::string::npos);
  assert(std::count(header.begin(), header.end(), ',') ==
         std::count(row.begin(), row.end(), ','));
  assert(row.find(",1,20.000000,2,40.000000,") != std::string::npos);
  assert(Summary::DistanceCsvHeader().find("pip_fallback_count") !=
         std::string::npos);
  assert(a.DistanceCsvRows().find("log_bins_gamma_2log1p001") !=
         std::string::npos);
}
} // namespace

int main() {
  TestActualTrackerCausalityAndBounds();
  TestFrozenBoundariesAndMirrorNormalization();
  TestDistanceDistributionAccuracyOrderingAndMerge();
  TestNormalizationAndEventHorizon();
  TestOutcomeBlindDependencyBoundary();
  TestSummaryMergeAndExplicitSchema();
  std::cout << "CausalFibonacciStructuralDiagnosticTests passed\n";
}
