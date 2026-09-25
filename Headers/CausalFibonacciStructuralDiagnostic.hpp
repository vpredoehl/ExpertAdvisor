#pragma once

#include "CausalFibonacciExtensionResearch.hpp"
#include "CausalFibonacciStructuralPrimitives.hpp"
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

// The diagnostic and runtime producer share these outcome-blind causal
// primitives; this namespace retains aliases for its stable test API.
using CausalFibonacciStructural::AdvanceH1H2EventState;
using CausalFibonacciStructural::ClassifyEventRelevance;
using CausalFibonacciStructural::EventAgeAtBar;
using CausalFibonacciStructural::EventRelevance;
using CausalFibonacciStructural::H1H2EventState;
using CausalFibonacciStructural::IsEventRelevantAtBar;
using CausalFibonacciStructural::NormalizedDistance;
using CausalFibonacciStructural::NormalizeDistance;

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
