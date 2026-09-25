#pragma once

#include "CausalFibonacciExtensionResearch.hpp"
#include "TG4HistoricalEmpiricalEvaluation.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace EA::FibonacciDiagnostic {
// Exact, mergeable nearest-rank distribution. Quantile(q) is the smallest
// value whose cumulative count is >= ceil(q*N); q==1 is the maximum.
class ExactDistribution {
public:
  void Add(double value);
  void Merge(const ExactDistribution &other);
  std::uint64_t Count() const;
  std::optional<double> Quantile(double q, bool absolute = false) const;

private:
  std::uint64_t count_ = 0;
  std::map<double, std::uint64_t> values_;
};

// Bounded-memory deterministic logarithmic histogram.  Its bins are spaced by
// 2*log(1.001) and represented at their geometric centre, so every nonzero
// finite input has a representative relative error of at most 0.1%; zero is
// represented exactly.  This is a representative-value guarantee, not an
// estimate of the rank error of a quantile.
class DistanceDistribution {
public:
  void Add(double value);
  void Merge(const DistanceDistribution &other);
  std::uint64_t Count() const;
  std::optional<double> Quantile(double q, bool absolute = false) const;
  static constexpr double MaxRelativeError() { return 0.001; }

private:
  std::uint64_t count_ = 0, zero_ = 0;
  std::map<std::int32_t, std::uint64_t> positive_, negative_;
};

struct NormalizedDistance {
  double value = 0.0;
  bool usedPipFallback = false;
};

// Returns nullopt only when the price inputs or canonical pip size are not
// finite/positive.  Missing, nonfinite, nonpositive, or no-larger-than-pip ATR
// deliberately selects the pip denominator.
std::optional<NormalizedDistance>
NormalizeDistance(bool upAB, double level, double close,
                  const std::optional<double> &atr, double canonicalPipSize);

// Event ages are inclusive at the twenty-bar horizon: event bar is age zero.
bool IsEventRelevantAtBar(std::size_t eventBar, std::size_t currentBar,
                          std::size_t horizonBars = 20);
std::size_t EventAgeAtBar(std::size_t eventBar, std::size_t currentBar);

enum class EventRelevance { None, H1Only, H2Only, Both };
EventRelevance ClassifyEventRelevance(const std::optional<std::size_t> &h1Bar,
                                      const std::optional<std::size_t> &h2Bar,
                                      std::size_t currentBar,
                                      std::size_t horizonBars = 20);

struct H1H2EventState {
  std::optional<std::size_t> touchBar;
  std::optional<std::size_t> h1BeyondBar;
  std::optional<std::size_t> h2RejectionBar;
};

// Applies the documented causal order on one completed bar: touch, H1 beyond,
// then H2 rejection (which always requires a strictly earlier touch).
void AdvanceH1H2EventState(H1H2EventState &state, std::size_t bar, bool touched,
                           bool beyond, bool rejected);

struct Summary {
  std::string symbol;
  std::uint64_t bars = 0, capacityEvictions = 0, ageExpirations = 0;
  std::int64_t firstMeasurementTimestamp = 0, lastMeasurementTimestamp = 0;
  std::array<ExactDistribution, 3> geometric, h1Bars, h2Bars, relevantBars;
  ExactDistribution abAges, h1Ages, h2Ages;
  std::array<std::uint64_t, 4> zeroBars{}, oneBars{}, multipleBars{},
      bothDirectionBars{};
  std::uint64_t noTouch = 0, touchObserved = 0, beyondObserved = 0,
                rejectionConfirmed = 0;
  std::uint64_t h1Age0To20 = 0, h1AgeOver20 = 0, h2Age0To20 = 0,
                h2AgeOver20 = 0;
  std::uint64_t h1SameDirectionBars = 0, h1OpposingDirectionBars = 0;
  std::uint64_t h2SameDirectionBars = 0, h2OpposingDirectionBars = 0;
  std::uint64_t relevantH1Only = 0, relevantH2Only = 0, relevantBoth = 0;
  std::array<std::array<DistanceDistribution, 5>, 2> distances;
  std::array<std::array<std::uint64_t, 5>, 2> finiteDistances{},
      invalidDistances{}, pipFallbacks{};
  void Merge(const Summary &other);
  std::string ToCsv() const;
  static std::string CsvHeader();
  static std::string DistanceCsvHeader();
  std::string DistanceCsvRows() const;
};

// Updates accounting only after normalization has made an explicit decision.
void RecordNormalizedDistance(Summary &summary, std::size_t population,
                              std::size_t level,
                              const std::optional<NormalizedDistance> &value);

// Structural-only observer of the actual bounded TG3 A/B live set.
class StructuralDiagnostic {
public:
  StructuralDiagnostic(std::string symbol,
                       TG4::EvaluationConfiguration configuration,
                       std::int64_t measurementStart);
  ~StructuralDiagnostic();
  StructuralDiagnostic(StructuralDiagnostic &&) noexcept;
  StructuralDiagnostic &operator=(StructuralDiagnostic &&) noexcept;
  StructuralDiagnostic(const StructuralDiagnostic &) = delete;
  StructuralDiagnostic &operator=(const StructuralDiagnostic &) = delete;
  void AddCompletedBar(const TG1A::Candle &candle);
  Summary Finalize();

private:
  class Implementation;
  std::unique_ptr<Implementation> implementation_;
};
} // namespace EA::FibonacciDiagnostic
