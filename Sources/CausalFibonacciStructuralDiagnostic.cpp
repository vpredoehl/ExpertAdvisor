#include "CausalFibonacciStructuralDiagnostic.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace EA::FibonacciDiagnostic {
namespace {
constexpr std::size_t kHorizonBars = 20;
// gamma=2*log(1.001): geometric-centre representative error is <=0.1%.
const double kLogGamma = 2.0 * std::log(1.001);
const std::array<double, 5> kLevels{1.272, 1.618, .382, .500, .618};
std::string Key(const TG3::ABIdentity &x) {
  return std::to_string(x.availabilityBar) + ":" + std::to_string(x.bBar) +
         ":" + std::to_string(static_cast<int>(x.direction)) + ":" +
         std::to_string(x.aTimestamp) + ":" + std::to_string(x.bTimestamp);
}
std::string Number(std::optional<double> x) {
  if (!x)
    return "";
  std::ostringstream o;
  o << std::setprecision(12) << *x;
  return o.str();
}
std::string Percent(std::uint64_t n, std::uint64_t d) {
  std::ostringstream o;
  o << std::fixed << std::setprecision(6)
    << (d ? 100.0 * static_cast<double>(n) / static_cast<double>(d) : 0.0);
  return o.str();
}
void AddCounts(std::array<ExactDistribution, 3> &d,
               std::array<std::uint64_t, 4> &z, std::array<std::uint64_t, 4> &o,
               std::array<std::uint64_t, 4> &m, std::array<std::uint64_t, 4> &b,
               std::size_t p, std::uint64_t total, std::uint64_t up,
               std::uint64_t down) {
  d[0].Add(total);
  d[1].Add(up);
  d[2].Add(down);
  if (!total)
    ++z[p];
  if (total == 1)
    ++o[p];
  if (total > 1)
    ++m[p];
  if (up && down)
    ++b[p];
}
void AppendQ(std::ostringstream &o, const ExactDistribution &d) {
  for (double q : {.5, .9, .95, .99, 1.0})
    o << ',' << Number(d.Quantile(q));
}
void AppendPH(std::ostringstream &o, std::string_view n, bool dirs) {
  for (const char *q : {"p50", "p90", "p95", "p99", "max"})
    o << ',' << n << "_total_" << q;
  if (dirs)
    for (const char *dir : {"up", "down"})
      for (const char *q : {"p50", "p90", "p95", "p99", "max"})
        o << ',' << n << '_' << dir << '_' << q;
}
void AppendP(std::ostringstream &o, const std::array<ExactDistribution, 3> &d,
             bool dirs) {
  AppendQ(o, d[0]);
  if (dirs) {
    AppendQ(o, d[1]);
    AppendQ(o, d[2]);
  }
}
} // namespace

void ExactDistribution::Add(double v) {
  if (std::isfinite(v)) {
    ++count_;
    ++values_[v];
  }
}
void ExactDistribution::Merge(const ExactDistribution &x) {
  count_ += x.count_;
  for (const auto &[v, n] : x.values_)
    values_[v] += n;
}
std::uint64_t ExactDistribution::Count() const { return count_; }
std::optional<double> ExactDistribution::Quantile(double q,
                                                  bool absolute) const {
  if (!(q > 0 && q <= 1) || !count_)
    return {};
  std::map<double, std::uint64_t> ordered;
  for (const auto &[v, n] : values_)
    ordered[absolute ? std::fabs(v) : v] += n;
  auto target = static_cast<std::uint64_t>(std::ceil(q * count_));
  std::uint64_t seen = 0;
  for (const auto &[v, n] : ordered)
    if ((seen += n) >= target)
      return v;
  return {};
}

void DistanceDistribution::Add(double v) {
  if (!std::isfinite(v))
    return;
  ++count_;
  if (v == 0) {
    ++zero_;
    return;
  }
  const double index = std::floor(std::log(std::fabs(v)) / kLogGamma);
  if (!std::isfinite(index) ||
      index < std::numeric_limits<std::int32_t>::min() ||
      index > std::numeric_limits<std::int32_t>::max())
    throw std::overflow_error("distance logarithmic bin overflow");
  auto &bins = v > 0 ? positive_ : negative_;
  ++bins[static_cast<std::int32_t>(index)];
}
void DistanceDistribution::Merge(const DistanceDistribution &x) {
  count_ += x.count_;
  zero_ += x.zero_;
  for (const auto &[k, n] : x.positive_)
    positive_[k] += n;
  for (const auto &[k, n] : x.negative_)
    negative_[k] += n;
}
std::uint64_t DistanceDistribution::Count() const { return count_; }
std::optional<double> DistanceDistribution::Quantile(double q,
                                                     bool absolute) const {
  if (!(q > 0 && q <= 1) || !count_)
    return {};
  std::map<double, std::uint64_t> ordered;
  auto add = [&](const auto &bins, double sign) {
    for (const auto &[bin, n] : bins) {
      const double exponent = (static_cast<double>(bin) + .5) * kLogGamma;
      const double maxLog = std::log(std::numeric_limits<double>::max());
      const double r = exponent >= maxLog ? std::numeric_limits<double>::max()
                                          : std::exp(exponent);
      if (!std::isfinite(r) || r <= 0)
        throw std::overflow_error("distance bin representative is non-finite");
      ordered[absolute ? r : sign * r] += n;
    }
  };
  add(negative_, -1);
  if (zero_)
    ordered[0] += zero_;
  add(positive_, 1);
  const auto target = static_cast<std::uint64_t>(std::ceil(q * count_));
  std::uint64_t seen = 0;
  for (const auto &[v, n] : ordered)
    if ((seen += n) >= target)
      return v;
  return {};
}

void Summary::Merge(const Summary &x) {
  bars += x.bars;
  capacityEvictions += x.capacityEvictions;
  ageExpirations += x.ageExpirations;
  if (!firstMeasurementTimestamp ||
      (x.firstMeasurementTimestamp &&
       x.firstMeasurementTimestamp < firstMeasurementTimestamp))
    firstMeasurementTimestamp = x.firstMeasurementTimestamp;
  lastMeasurementTimestamp =
      std::max(lastMeasurementTimestamp, x.lastMeasurementTimestamp);
  for (int i = 0; i < 3; ++i) {
    geometric[i].Merge(x.geometric[i]);
    h1Bars[i].Merge(x.h1Bars[i]);
    h2Bars[i].Merge(x.h2Bars[i]);
    relevantBars[i].Merge(x.relevantBars[i]);
  }
  abAges.Merge(x.abAges);
  h1Ages.Merge(x.h1Ages);
  h2Ages.Merge(x.h2Ages);
  for (int i = 0; i < 4; ++i) {
    zeroBars[i] += x.zeroBars[i];
    oneBars[i] += x.oneBars[i];
    multipleBars[i] += x.multipleBars[i];
    bothDirectionBars[i] += x.bothDirectionBars[i];
  }
  noTouch += x.noTouch;
  touchObserved += x.touchObserved;
  beyondObserved += x.beyondObserved;
  rejectionConfirmed += x.rejectionConfirmed;
  h1Age0To20 += x.h1Age0To20;
  h1AgeOver20 += x.h1AgeOver20;
  h2Age0To20 += x.h2Age0To20;
  h2AgeOver20 += x.h2AgeOver20;
  h1SameDirectionBars += x.h1SameDirectionBars;
  h1OpposingDirectionBars += x.h1OpposingDirectionBars;
  h2SameDirectionBars += x.h2SameDirectionBars;
  h2OpposingDirectionBars += x.h2OpposingDirectionBars;
  relevantH1Only += x.relevantH1Only;
  relevantH2Only += x.relevantH2Only;
  relevantBoth += x.relevantBoth;
  for (int p = 0; p < 2; ++p)
    for (int l = 0; l < 5; ++l) {
      distances[p][l].Merge(x.distances[p][l]);
      finiteDistances[p][l] += x.finiteDistances[p][l];
      invalidDistances[p][l] += x.invalidDistances[p][l];
      pipFallbacks[p][l] += x.pipFallbacks[p][l];
    }
}

void RecordNormalizedDistance(Summary &summary, std::size_t population,
                              std::size_t level,
                              const std::optional<NormalizedDistance> &value) {
  if (population >= summary.distances.size() ||
      level >= summary.distances[population].size())
    throw std::out_of_range("invalid diagnostic distance population or level");
  if (!value) {
    ++summary.invalidDistances[population][level];
    return;
  }
  ++summary.finiteDistances[population][level];
  if (value->usedPipFallback)
    ++summary.pipFallbacks[population][level];
  summary.distances[population][level].Add(value->value);
}

std::string Summary::CsvHeader() {
  std::ostringstream o;
  o << "symbol,bars,first_measurement_epoch,last_measurement_epoch,capacity_"
       "evictions,age_expirations";
  AppendPH(o, "geometric", true);
  for (const char *n : {"geometric", "h1", "h2", "relevant"})
    o << ',' << n << "_zero_count," << n << "_zero_pct," << n << "_one_count,"
      << n << "_one_pct," << n << "_multiple_count," << n << "_multiple_pct,"
      << n << "_both_direction_count," << n << "_both_direction_pct";
  o << ",ab_age_p50,ab_age_p90,ab_age_p95,ab_age_p99,ab_age_max,no_touch_"
       "observations,touch_observed_observations,beyond_observed_observations,"
       "rejection_confirmed_observations";
  AppendPH(o, "h1", true);
  o << ",h1_age_p50,h1_age_p90,h1_age_p95,h1_age_p99,h1_age_max,h1_age_0_20_"
       "count,h1_age_0_20_pct,h1_age_over_20_count,h1_age_over_20_pct,h1_"
       "multiple_same_direction_bars,h1_opposing_direction_bars";
  AppendPH(o, "h2", true);
  o << ",h2_age_p50,h2_age_p90,h2_age_p95,h2_age_p99,h2_age_max,h2_age_0_20_"
       "count,h2_age_0_20_pct,h2_age_over_20_count,h2_age_over_20_pct,h2_"
       "multiple_same_direction_bars,h2_opposing_direction_bars";
  AppendPH(o, "relevant", true);
  o << ",h1_only_observations,h2_only_observations,both_h1_h2_observations";
  return o.str();
}
std::string Summary::ToCsv() const {
  std::ostringstream o;
  o << symbol << ',' << bars << ',' << firstMeasurementTimestamp << ','
    << lastMeasurementTimestamp << ',' << capacityEvictions << ','
    << ageExpirations;
  AppendP(o, geometric, true);
  for (int p = 0; p < 4; ++p)
    o << ',' << zeroBars[p] << ',' << Percent(zeroBars[p], bars) << ','
      << oneBars[p] << ',' << Percent(oneBars[p], bars) << ','
      << multipleBars[p] << ',' << Percent(multipleBars[p], bars) << ','
      << bothDirectionBars[p] << ',' << Percent(bothDirectionBars[p], bars);
  AppendQ(o, abAges);
  o << ',' << noTouch << ',' << touchObserved << ',' << beyondObserved << ','
    << rejectionConfirmed;
  AppendP(o, h1Bars, true);
  AppendQ(o, h1Ages);
  o << ',' << h1Age0To20 << ',' << Percent(h1Age0To20, h1Age0To20 + h1AgeOver20)
    << ',' << h1AgeOver20 << ','
    << Percent(h1AgeOver20, h1Age0To20 + h1AgeOver20) << ','
    << h1SameDirectionBars << ',' << h1OpposingDirectionBars;
  AppendP(o, h2Bars, true);
  AppendQ(o, h2Ages);
  o << ',' << h2Age0To20 << ',' << Percent(h2Age0To20, h2Age0To20 + h2AgeOver20)
    << ',' << h2AgeOver20 << ','
    << Percent(h2AgeOver20, h2Age0To20 + h2AgeOver20) << ','
    << h2SameDirectionBars << ',' << h2OpposingDirectionBars;
  AppendP(o, relevantBars, true);
  o << ',' << relevantH1Only << ',' << relevantH2Only << ',' << relevantBoth;
  return o.str();
}
std::string Summary::DistanceCsvHeader() {
  return "symbol,population,level,finite_count,invalid_count,pip_fallback_"
         "count,signed_p01,signed_p05,signed_p50,signed_p95,signed_p99,"
         "absolute_p50,absolute_p90,absolute_p95,absolute_p99,absolute_max,"
         "approximation";
}
std::string Summary::DistanceCsvRows() const {
  std::ostringstream o;
  for (int p = 0; p < 2; ++p)
    for (int l = 0; l < 5; ++l) {
      o << symbol << ',' << (p ? "event_relevant_20" : "geometric") << ','
        << kLevels[l] << ',' << finiteDistances[p][l] << ','
        << invalidDistances[p][l] << ',' << pipFallbacks[p][l];
      for (double q : {.01, .05, .5, .95, .99})
        o << ',' << Number(distances[p][l].Quantile(q));
      for (double q : {.5, .9, .95, .99, 1.0})
        o << ',' << Number(distances[p][l].Quantile(q, true));
      o << ",log_bins_gamma_2log1p001_zero_exact_representative_relative_error_"
           "le_0.1pct\n";
    }
  return o.str();
}

struct State {
  TG3::ABStructure ab;
  std::array<FibonacciResearch::FibonacciLevel, 5> levels;
  std::optional<std::size_t> touch, beyond, rejection;
};
class StructuralDiagnostic::Implementation {
public:
  Implementation(std::string s, TG4::EvaluationConfiguration c,
                 std::int64_t start)
      : symbol(std::move(s)), config(std::move(c)), measurementStart(start),
        geometry(config.geometry, {symbol, config.timeframe}),
        tracker(TrackerConfig(), {symbol, config.timeframe}) {
    summary.symbol = symbol;
  }
  TG3::Configuration TrackerConfig() const {
    auto x = config.fibonacci;
    x.absolutePriceTolerance =
        TG4::EffectiveFibonacciAbsolutePriceTolerance(config, symbol);
    x.directionalStudyPolicy =
        TG3::DirectionalStudyPolicy::SymmetricDirectionalDiagnostic;
    return x;
  }
  bool Up(const State &s) const {
    return s.ab.identity.direction == TG3::ABDirection::UpAB;
  }
  State Make(const TG3::ABStructure &a) const {
    const auto t =
        TG4::EffectiveFibonacciAbsolutePriceTolerance(config, symbol);
    return {a,
            {FibonacciResearch::CalculateExtensionLevel(a, 1.272, t),
             FibonacciResearch::CalculateExtensionLevel(a, 1.618, t),
             FibonacciResearch::CalculatePullbackLevel(a, .382, t),
             FibonacciResearch::CalculatePullbackLevel(a, .5, t),
             FibonacciResearch::CalculatePullbackLevel(a, .618, t)},
            std::nullopt,
            std::nullopt,
            std::nullopt};
  }
  void AddCompleted(const TG1A::Candle &c) {
    auto u = geometry.AddCompletedBar(c);
    tracker.Advance(u.bar, c.timestamp);
    tracker.ObserveConfirmedFractals(u.newlyConfirmedFractals);
    std::map<std::string, bool> live;
    for (const auto &a : tracker.ABStructures()) {
      auto k = Key(a.identity);
      live[k] = true;
      if (!states.contains(k))
        states.emplace(k, Make(a));
    }
    for (auto i = states.begin(); i != states.end();)
      if (!live.contains(i->first))
        i = states.erase(i);
      else
        ++i;
    for (auto &[k, s] : states) {
      H1H2EventState events{s.touch, s.beyond, s.rejection};
      AdvanceH1H2EventState(events, u.bar,
                            Up(s) ? c.high >= s.levels[0].zoneLowerPrice
                                  : c.low <= s.levels[0].zoneUpperPrice,
                            Up(s) ? c.close > s.levels[0].zoneUpperPrice
                                  : c.close < s.levels[0].zoneLowerPrice,
                            Up(s) ? c.close < s.levels[0].zoneLowerPrice
                                  : c.close > s.levels[0].zoneUpperPrice);
      s.touch = events.touchBar;
      s.beyond = events.h1BeyondBar;
      s.rejection = events.h2RejectionBar;
    }
    if (c.timestamp >= measurementStart)
      Measure(u.bar, c);
    summary.capacityEvictions = tracker.ABCapacityEvictions();
    summary.ageExpirations = tracker.ABAgeExpirations();
  }
  void Measure(std::size_t bar, const TG1A::Candle &c) {
    ++summary.bars;
    if (!summary.firstMeasurementTimestamp)
      summary.firstMeasurementTimestamp = c.timestamp;
    summary.lastMeasurementTimestamp = c.timestamp;
    std::uint64_t gt = 0, gu = 0, gd = 0, h1t = 0, h1u = 0, h1d = 0, h2t = 0,
                  h2u = 0, h2d = 0, rt = 0, ru = 0, rd = 0;
    for (auto &[k, s] : states) {
      bool up = Up(s);
      ++gt;
      up ? ++gu : ++gd;
      summary.abAges.Add(bar - s.ab.identity.availabilityBar);
      s.touch ? ++summary.touchObserved : ++summary.noTouch;
      if (s.beyond) {
        ++summary.beyondObserved;
        ++h1t;
        up ? ++h1u : ++h1d;
        auto age = EventAgeAtBar(*s.beyond, bar);
        summary.h1Ages.Add(age);
        age <= kHorizonBars ? ++summary.h1Age0To20 : ++summary.h1AgeOver20;
      }
      if (s.rejection) {
        ++summary.rejectionConfirmed;
        ++h2t;
        up ? ++h2u : ++h2d;
        auto age = EventAgeAtBar(*s.rejection, bar);
        summary.h2Ages.Add(age);
        age <= kHorizonBars ? ++summary.h2Age0To20 : ++summary.h2AgeOver20;
      }
      const EventRelevance relevance =
          ClassifyEventRelevance(s.beyond, s.rejection, bar, kHorizonBars);
      const bool relevant = relevance != EventRelevance::None;
      if (relevant) {
        ++rt;
        up ? ++ru : ++rd;
        if (relevance == EventRelevance::Both)
          ++summary.relevantBoth;
        else if (relevance == EventRelevance::H1Only)
          ++summary.relevantH1Only;
        else
          ++summary.relevantH2Only;
      }
      for (int p = 0; p < 2; ++p)
        if (!p || relevant)
          for (int l = 0; l < 5; ++l) {
            auto normal = NormalizeDistance(up, s.levels[l].price, c.close,
                                            geometry.CurrentAtr(),
                                            TG4::CanonicalFxPipSize(symbol));
            RecordNormalizedDistance(summary, p, l, normal);
          }
    }
    AddCounts(summary.geometric, summary.zeroBars, summary.oneBars,
              summary.multipleBars, summary.bothDirectionBars, 0, gt, gu, gd);
    AddCounts(summary.h1Bars, summary.zeroBars, summary.oneBars,
              summary.multipleBars, summary.bothDirectionBars, 1, h1t, h1u,
              h1d);
    AddCounts(summary.h2Bars, summary.zeroBars, summary.oneBars,
              summary.multipleBars, summary.bothDirectionBars, 2, h2t, h2u,
              h2d);
    AddCounts(summary.relevantBars, summary.zeroBars, summary.oneBars,
              summary.multipleBars, summary.bothDirectionBars, 3, rt, ru, rd);
    if (h1u > 1 || h1d > 1)
      ++summary.h1SameDirectionBars;
    if (h1u && h1d)
      ++summary.h1OpposingDirectionBars;
    if (h2u > 1 || h2d > 1)
      ++summary.h2SameDirectionBars;
    if (h2u && h2d)
      ++summary.h2OpposingDirectionBars;
  }
  Summary Finish() { return summary; }
  std::string symbol;
  TG4::EvaluationConfiguration config;
  std::int64_t measurementStart;
  TG1A::CausalFractalTrendLineGeometry geometry;
  TG3::FibonacciConfluenceTracker tracker;
  std::map<std::string, State> states;
  Summary summary;
};
StructuralDiagnostic::StructuralDiagnostic(std::string s,
                                           TG4::EvaluationConfiguration c,
                                           std::int64_t t)
    : implementation_(
          std::make_unique<Implementation>(std::move(s), std::move(c), t)) {}
StructuralDiagnostic::~StructuralDiagnostic() = default;
StructuralDiagnostic::StructuralDiagnostic(StructuralDiagnostic &&) noexcept =
    default;
StructuralDiagnostic &
StructuralDiagnostic::operator=(StructuralDiagnostic &&) noexcept = default;
void StructuralDiagnostic::AddCompletedBar(const TG1A::Candle &c) {
  implementation_->AddCompleted(c);
}
Summary StructuralDiagnostic::Finalize() { return implementation_->Finish(); }
} // namespace EA::FibonacciDiagnostic
