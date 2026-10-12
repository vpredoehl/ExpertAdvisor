#include "CausalFractalHistoryShadowAudit.hpp"
#include "CausalFibonacciTrendlineHistoricalArtifact.hpp"
#include <deque>
#include <fstream>
#include <regex>
#include <set>
#include <zlib.h>

namespace EA::FibonacciResearch::HistoryShadow
{
namespace S = TrendlineHistorical;
namespace
{
using F = TG1A::ConfirmedFractal;
using AB = TG3::ABStructure;
using L = TG1A::TrendLineCandidate;
std::string N(std::size_t n)
{
    return std::to_string(n);
}
std::string T(std::int64_t n)
{
    return std::to_string(n);
}
std::string FractalIdentity(const F &f)
{
    return N(static_cast<std::size_t>(f.kind)) + ":" + N(f.anchorBar) + ":" + T(f.anchorTimestamp) + ":" +
           N(f.confirmationBar) + ":" + T(f.confirmationTimestamp);
}
bool FractalLess(const F &a, const F &b)
{
    return std::tuple(a.confirmationBar, a.anchorBar, static_cast<int>(a.kind), a.anchorTimestamp) <
           std::tuple(b.confirmationBar, b.anchorBar, static_cast<int>(b.kind), b.anchorTimestamp);
}
std::string Natural(const TG3::ABIdentity &x)
{
    return N(static_cast<std::size_t>(x.direction)) + ":" + T(x.aTimestamp) + ":" + T(x.bTimestamp) + ":" +
           T(x.availabilityTimestamp);
}
std::string Natural(const AB &x)
{
    return Natural(x.identity);
}
std::string Natural(const L &x)
{
    return N(static_cast<std::size_t>(x.direction)) + ":" + T(x.anchor1Timestamp) + ":" +
           T(x.anchor2Timestamp) + ":" + T(x.creationTimestamp);
}
std::string Native(const TG3::ABIdentity &x)
{
    return Natural(x) + ":" + N(x.aBar) + ":" + N(x.bBar) + ":" + N(x.availabilityBar);
}
auto Snapshot(const AB &x)
{
    return std::tuple(x.identity, x.aPrice, x.bPrice, x.aConfirmationBar, x.aConfirmationTimestamp,
                      x.bConfirmationBar, x.bConfirmationTimestamp, x.priceRange, x.anchorSelectionPolicy);
}
auto Snapshot(const L &x)
{
    return std::tuple(x.direction, x.anchor1Bar, x.anchor1Timestamp, x.anchor1Price, x.anchor1ConfirmationBar,
                      x.anchor1ConfirmationTimestamp, x.anchor2Bar, x.anchor2Timestamp, x.anchor2Price,
                      x.anchor2ConfirmationBar, x.anchor2ConfirmationTimestamp, x.anchorSeparationBars,
                      x.rawPriceSlopePerBar, x.creationBar, x.creationTimestamp, x.candidateAgeBars,
                      x.currentBar, x.currentTimestamp, x.projectedLinePriceAtCurrentBar,
                      x.rawPriceToLineDistance, x.currentAtr, x.atrNormalizedSlope,
                      x.atrNormalizedPriceToLineDistance, x.candleTouchCount, x.fractalTouchCount);
}
template <class A, class B, class Fn> bool Equal(const A &a, const B &b, Fn fn)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (fn(a[i]) != fn(b[i]))
            return false;
    return true;
}
auto ABOrder(const AB &a)
{
    return std::tuple(a.identity.availabilityBar, a.identity.bBar, static_cast<int>(a.identity.direction),
                      a.identity.aTimestamp, a.identity.bTimestamp);
}
auto LineEviction(const L &l)
{
    return std::tuple(l.creationBar, static_cast<int>(l.direction), l.anchor1Bar, l.anchor2Bar);
}
auto LineOrder(const L &l)
{
    return std::tuple(static_cast<int>(l.direction), l.anchor1Bar, l.anchor2Bar, l.creationBar);
}
AB MakeAB(const F &a, const F &b)
{
    AB x;
    x.identity = {b.kind == TG1A::FractalKind::High ? TG3::ABDirection::UpAB : TG3::ABDirection::DownAB,
                  a.anchorBar,
                  a.anchorTimestamp,
                  b.anchorBar,
                  b.anchorTimestamp,
                  b.confirmationBar,
                  b.confirmationTimestamp};
    x.aPrice = a.price;
    x.bPrice = b.price;
    x.priceRange = std::fabs(b.price - a.price);
    x.aConfirmationBar = a.confirmationBar;
    x.aConfirmationTimestamp = a.confirmationTimestamp;
    x.bConfirmationBar = b.confirmationBar;
    x.bConfirmationTimestamp = b.confirmationTimestamp;
    return x;
}
L MakeLine(const F &a, const F &b)
{
    L l;
    l.direction =
        b.kind == TG1A::FractalKind::Low ? TG1A::TrendLineDirection::UTL : TG1A::TrendLineDirection::DTL;
    l.anchor1Bar = a.anchorBar;
    l.anchor1Timestamp = a.anchorTimestamp;
    l.anchor1Price = a.price;
    l.anchor1ConfirmationBar = a.confirmationBar;
    l.anchor1ConfirmationTimestamp = a.confirmationTimestamp;
    l.anchor2Bar = b.anchorBar;
    l.anchor2Timestamp = b.anchorTimestamp;
    l.anchor2Price = b.price;
    l.anchor2ConfirmationBar = b.confirmationBar;
    l.anchor2ConfirmationTimestamp = b.confirmationTimestamp;
    l.anchorSeparationBars = b.anchorBar - a.anchorBar;
    l.rawPriceSlopePerBar = (b.price - a.price) / static_cast<double>(l.anchorSeparationBars);
    l.creationBar = b.confirmationBar;
    l.creationTimestamp = b.confirmationTimestamp;
    return l;
}
std::string Json(std::string_view s)
{
    std::string r = "\"";
    for (unsigned char c : s)
    {
        if (c == '"' || c == '\\')
        {
            r += '\\';
            r += static_cast<char>(c);
        }
        else if (c == '\n')
            r += "\\n";
        else if (c < 32)
            throw std::invalid_argument("audit metadata control character");
        else
            r += static_cast<char>(c);
    }
    return r + '"';
}
void Write(const std::filesystem::path &p, std::string_view s)
{
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.exceptions(std::ios::badbit | std::ios::failbit);
    f.write(s.data(), static_cast<std::streamsize>(s.size()));
    f.close();
}
} // namespace

struct TG3BoundedHistoryParityMirror
{
    TG3::FibonacciConfluenceTracker detector;
    std::array<std::vector<F>, 2> history;
    std::vector<AB> reconstructedCandidates;
    explicit TG3BoundedHistoryParityMirror(TG3::Configuration c) : detector(std::move(c), {"AUDCAD", "15m"})
    {
    }
};
struct TG1ABoundedHistoryParityMirror
{
    TG1A::CausalFractalTrendLineGeometry detector;
    std::array<std::vector<F>, 2> history;
    explicit TG1ABoundedHistoryParityMirror(TG1A::Configuration c) : detector(c, {"AUDCAD", "15m"}) {}
};
struct RestoredAnchor
{
    F fractal;
    bool removed = false;
};
struct TG3HistoryRestorationCounterfactual
{
    std::array<std::vector<RestoredAnchor>, 2> history;
    std::vector<AB> candidates;
};
struct TG1AHistoryRestorationCounterfactual
{
    std::array<std::vector<RestoredAnchor>, 2> history;
    std::vector<L> candidates;
};
std::string CandidateIdentityReconciliation::Natural(const TG3::ABIdentity &x)
{
    return HistoryShadow::Natural(x);
}
std::string CandidateIdentityReconciliation::Natural(const L &x)
{
    return HistoryShadow::Natural(x);
}
std::string CandidateIdentityReconciliation::Native(const TG3::ABIdentity &x)
{
    return HistoryShadow::Native(x);
}

struct Audit::State
{
    using Anchor = RestoredAnchor;
    struct Witness
    {
        std::string detector, identity, attribution;
        AB ab;
        L line;
        bool survived = false, decision = false, contributed = false;
        std::string contributingPair;
        std::optional<std::int64_t> contributionTimestamp;
        double contributionClose = 0, contributionAtr = 0, contributionRatio = 0, contributionFib = 0,
               contributionLine = 0;
    };
    struct Indexed
    {
        std::size_t bar;
        TG1A::Candle candle;
    };
    TG3::Configuration fc;
    TG1A::Configuration gc;
    TG3BoundedHistoryParityMirror tg3;
    TG1ABoundedHistoryParityMirror tg1;
    TG3HistoryRestorationCounterfactual restored3;
    TG1AHistoryRestorationCounterfactual restored1;
    TG3::FibonacciConfluenceTracker &fibonacciMirror = tg3.detector;
    TG1A::CausalFractalTrendLineGeometry &geometryMirror = tg1.detector;
    Budgets budgets;
    PublicationHook hook;
    Summary summary;
    std::filesystem::path dir;
    S::CompressedCsv removals, witnesses;
    std::array<std::vector<F>, 2> &boundedTG3 = tg3.history;
    std::array<std::vector<F>, 2> &boundedTG1 = tg1.history;
    std::array<std::vector<Anchor>, 2> &restoredTG3 = restored3.history;
    std::array<std::vector<Anchor>, 2> &restoredTG1 = restored1.history;
    std::vector<AB> &boundedAB = tg3.reconstructedCandidates;
    std::vector<AB> &shadowAB = restored3.candidates;
    std::vector<L> &shadowLines = restored1.candidates;
    std::map<std::string, Witness> liveWitnesses;
    std::map<std::string, std::pair<std::size_t, std::size_t>> capacityEvicted;
    std::deque<Indexed> bars;
    std::optional<std::int64_t> lastTimestamp;
    bool finished = false;
    S::Sha256 candleHash;
    std::size_t decodedBytes = 0;
    State(TG3::Configuration f, TG1A::Configuration g, std::filesystem::path d, Budgets b, PublicationHook h)
        : fc(std::move(f)), gc(g), tg3(fc), tg1(gc), budgets(b), hook(std::move(h)), dir(std::move(d)),
          removals(dir / "history_removals.csv.gz", [this](S::CompressionOperation op) { Fault(op); }),
          witnesses(dir / "candidate_witnesses.csv.gz", [this](S::CompressionOperation op) { Fault(op); })
    {
        summary.failure.reserve(256); // Failure reporting must not allocate in Observe.
        if (fc.maxConfirmedFractalsPerKind != 128 || fc.maxActiveABStructures != 512 ||
            fc.maxABAgeBars != 2048 || gc.maxConfirmedFractalsPerKind != 64 ||
            gc.maxFractalAnchorLookbackBars != 512 || gc.maxCandidates != 4096 ||
            gc.maxCandidateAgeBars != 512)
            throw std::invalid_argument("shadow audit requires frozen history/candidate bounds");
        if (b.publicationBufferBytes < 2 * S::CompressedCsv::kBufferBytes || !b.shadowFractals ||
            !b.witnessRecords || !b.removalRecords || !b.diagnosticBytes)
            throw std::invalid_argument("invalid audit budgets");
        Append(removals, "detector,kind,phase,bar,bar_start_utc_epoch,anchor_bar,anchor_utc_epoch,"
                         "confirmation_bar,confirmation_utc_epoch,reason,occupancy_before,logical_demand,"
                         "remaining_lookback,temporally_relevant\n");
        Append(
            witnesses,
            "detector,natural_identity,native_identity,stage,bar,bar_start_utc_epoch,creation_bar,creation_"
            "utc_epoch,attribution,confidence,survived,decision_eligible,cohort_contribution,detail,anchor_a_"
            "price,anchor_b_price,anchor_a_confirmation_utc_epoch,anchor_b_confirmation_utc_epoch,"
            "contribution_utc_epoch,contribution_close,contribution_atr,contributing_native_pair,ratio,fib_"
            "price,line_price\n");
    }
    // Budgets are conservative accounted bounds, not an OS RSS measurement.
    // Native containers, vector spare capacity, map/set nodes and transient
    // publication/codec work are included; no evidence is evicted to meet them.
    static void Increment(std::size_t &value)
    {
        if (value == std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("audit_counter_overflow");
        ++value;
    }
    void Append(S::CompressedCsv &stream, std::string_view data)
    {
        if (data.size() >
            budgets.decodedPublicationBytes - std::min(decodedBytes, budgets.decodedPublicationBytes))
        {
            Fail("decoded_publication_byte_budget", false, true);
            throw std::runtime_error(summary.failure);
        }
        decodedBytes += data.size();
        stream.Append(data);
    }
    void Fault(S::CompressionOperation op)
    {
        if (hook)
            hook(op == S::CompressionOperation::Compress ? PublicationStep::Compress
                 : op == S::CompressionOperation::Write  ? PublicationStep::Write
                                                         : PublicationStep::Finalize);
    }
    void Publish(std::string_view name, std::string_view data)
    {
        if (hook)
            hook(PublicationStep::Write);
        Write(dir / std::string(name), data);
    }
    void Fail(std::string_view reason, bool parity = false, bool exhausted = false) noexcept
    {
        summary.complete = false;
        summary.parity = summary.parity && !parity;
        summary.exhausted = summary.exhausted || exhausted;
        if (summary.failure.empty())
        {
            try
            {
                summary.failure.assign(reason.substr(0, 256));
            }
            catch (...)
            { /* Flags remain authoritative even on allocator failure. */
            }
        }
    }
    void Check(bool value, const char *reason)
    {
        if (!value)
        {
            Fail(reason, true);
            throw std::runtime_error(reason);
        }
    }
    template <class Fn> void Count(std::size_t detector, std::size_t kind, bool scoring, Fn fn)
    {
        fn(summary.history[detector][kind][0]);
        fn(summary.history[detector][kind][scoring ? 2 : 1]);
    }
    void ResourceCheck()
    {
        std::size_t anchors = 0;
        for (const auto &h : restoredTG3)
            anchors += h.size();
        for (const auto &h : restoredTG1)
            anchors += h.size();
        // Conservative accounted bound includes vector geometric over-allocation,
        // strings/map nodes, native mirrors and gzip working space (2 MiB each).
        const std::size_t bytes =
            anchors * 256 +
            (boundedAB.size() + shadowAB.size() + fibonacciMirror.ABStructures().size()) * 512 +
            (shadowLines.size() + geometryMirror.Candidates().size()) * 1024 + liveWitnesses.size() * 2048 +
            capacityEvicted.size() * 512 + bars.size() * 128 + 8 * 1024 * 1024;
        summary.peakAccountedBytes = std::max(summary.peakAccountedBytes, bytes);
        if (anchors > budgets.shadowFractals)
            Fail("shadow_history_budget", false, true);
        else if (shadowAB.size() + shadowLines.size() > budgets.liveCandidates)
            Fail("live_candidate_budget", false, true);
        else if (bytes > budgets.diagnosticBytes)
            Fail("diagnostic_memory_budget", false, true);
        if (!summary.complete)
            throw std::runtime_error(summary.failure);
    }
    void Remove(std::size_t detector, std::size_t kind, F f, const CompletedBarObservation &o, bool capacity,
                std::size_t occupancy, std::size_t demand)
    {
        bool relevant = detector == 0 || CausalRules::TG1ATemporallyEligible(o.bar, f.anchorBar);
        Count(detector, kind, o.scoring,
              [&](Counts &c)
              {
                  if (capacity)
                  {
                      Increment(c.capacityRemovals);
                      if (relevant)
                          Increment(c.relevantRemovals);
                      if (!c.firstRemovalBar)
                      {
                          c.firstRemovalBar = o.bar;
                          c.firstRemovalTimestamp = o.candle.timestamp;
                      }
                  }
                  else
                      Increment(c.expirations);
              });
        if (summary.removalRecords >= budgets.removalRecords)
        {
            Fail("removal_record_budget", false, true);
            throw std::runtime_error(summary.failure);
        }
        std::string remaining = detector == 0
                                    ? "NOT_APPLICABLE"
                                    : N(gc.maxFractalAnchorLookbackBars -
                                        std::min(gc.maxFractalAnchorLookbackBars, o.bar - f.anchorBar));
        Append(removals, std::string(detector == 0 ? "TG3" : "TG1A") + "," +
                             (kind == static_cast<std::size_t>(TG1A::FractalKind::High) ? "High" : "Low") +
                             "," + (o.scoring ? "scoring" : "prefix") + "," + N(o.bar) + "," +
                             T(o.candle.timestamp) + "," + N(f.anchorBar) + "," + T(f.anchorTimestamp) + "," +
                             N(f.confirmationBar) + "," + T(f.confirmationTimestamp) + "," +
                             (capacity ? "HISTORY_CAPACITY" : "LOOKBACK_EXPIRATION") + "," + N(occupancy) +
                             "," + N(demand) + "," + remaining + "," + (relevant ? "1" : "0") + "\n");
        Increment(summary.removalRecords);
        if (capacity)
        {
            auto &h = detector == 0 ? restoredTG3[kind] : restoredTG1[kind];
            auto i = std::find_if(h.begin(), h.end(), [&](const Anchor &a)
                                  { return FractalIdentity(a.fractal) == FractalIdentity(f); });
            Check(i != h.end(), "removed_anchor_absent_in_shadow");
            i->removed = true;
        }
    }
    void PruneHistory(std::size_t kind, const CompletedBarObservation &o)
    {
        auto &h = boundedTG1[kind];
        while (!h.empty() && o.bar - h.front().anchorBar > gc.maxFractalAnchorLookbackBars)
        {
            Remove(1, kind, h.front(), o, false, h.size(), h.size());
            h.erase(h.begin());
        }
        auto &s = restoredTG1[kind];
        s.erase(std::remove_if(s.begin(), s.end(), [&](const Anchor &a)
                               { return o.bar - a.fractal.anchorBar > gc.maxFractalAnchorLookbackBars; }),
                s.end());
    }
    template <class Hist> std::optional<F> Select(const Hist &h, const F &b)
    {
        return CausalRules::SelectTG3Anchor(h, b);
    }
    void Emit(const Witness &w, std::string_view stage, const CompletedBarObservation &o,
              std::string_view detail)
    {
        if (summary.witnessRecords >= budgets.witnessRecords)
        {
            Fail("witness_record_budget", false, true);
            throw std::runtime_error(summary.failure);
        }
        bool ab = w.detector == "TG3";
        auto birth = ab ? w.ab.identity.availabilityBar : w.line.creationBar;
        auto ts = ab ? w.ab.identity.availabilityTimestamp : w.line.creationTimestamp;
        std::string native = ab ? Native(w.ab.identity)
                                : Natural(w.line) + ":" + N(w.line.anchor1Bar) + ":" + N(w.line.anchor2Bar) +
                                      ":" + N(w.line.creationBar);
        Append(witnesses,
               w.detector + "," + w.identity + "," + native + "," + std::string(stage) + "," + N(o.bar) +
                   "," + T(o.candle.timestamp) + "," + N(birth) + "," + T(ts) + "," + w.attribution + "," +
                   (w.attribution == "STATE_DIVERGENCE_UNATTRIBUTED" ? "UNATTRIBUTED"
                                                                     : "EXACT_HISTORY_ONLY_INTERVENTION") +
                   "," + (w.survived ? "1" : "0") + "," + (w.decision ? "1" : "0") + "," +
                   (w.contributed ? "YES" : "UNKNOWN") + "," + std::string(detail) + "," +
                   S::Number(ab ? w.ab.aPrice : w.line.anchor1Price) + "," +
                   S::Number(ab ? w.ab.bPrice : w.line.anchor2Price) + "," +
                   T(ab ? w.ab.aConfirmationTimestamp : w.line.anchor1ConfirmationTimestamp) + "," +
                   T(ab ? w.ab.bConfirmationTimestamp : w.line.anchor2ConfirmationTimestamp) + "," +
                   (w.contributionTimestamp ? T(*w.contributionTimestamp) : "") + "," +
                   (w.contributionTimestamp ? S::Number(w.contributionClose) : "") + "," +
                   (w.contributionTimestamp ? S::Number(w.contributionAtr) : "") + "," + w.contributingPair +
                   "," + (w.contributionTimestamp ? S::Number(w.contributionRatio) : "") + "," +
                   (w.contributionTimestamp ? S::Number(w.contributionFib) : "") + "," +
                   (w.contributionTimestamp ? S::Number(w.contributionLine) : "") + "\n");
        Increment(summary.witnessRecords);
    }
    void Lost(const AB &ab, const CompletedBarObservation &o)
    {
        Witness w;
        w.detector = "TG3";
        w.identity = Natural(ab);
        w.attribution = "DIRECT_HISTORY_SUPPRESSION";
        w.ab = ab;
        Check(liveWitnesses.emplace("TG3:" + w.identity, w).second, "duplicate_TG3_natural_birth");
        Increment(summary.creationLossesTG3);
        Emit(w, "RAW_BIRTH", o, "policy_selected_removed_anchor");
    }
    void Lost(const L &line, const CompletedBarObservation &o)
    {
        Witness w;
        w.detector = "TG1A";
        w.identity = Natural(line);
        w.attribution = "DIRECT_HISTORY_SUPPRESSION";
        w.line = line;
        Check(liveWitnesses.emplace("TG1A:" + w.identity, w).second, "duplicate_TG1A_natural_birth");
        Increment(summary.creationLossesTG1A);
        Emit(w, "RAW_BIRTH", o, "removed_anchor_structural_predicates_pass");
    }
    bool Valid(const L &l) const
    {
        for (const auto &b : bars)
        {
            if (b.bar <= l.anchor1Bar || b.bar >= l.anchor2Bar)
                continue;
            auto p = l.ProjectedPrice(b.bar);
            if (l.direction == TG1A::TrendLineDirection::UTL
                    ? b.candle.low + gc.interveningPriceTolerance < p
                    : b.candle.high - gc.interveningPriceTolerance > p)
                return false;
        }
        return true;
    }
    template <class V, class Birth> void Expire(V &v, std::size_t bar, std::size_t age, Birth birth)
    {
        v.erase(std::remove_if(v.begin(), v.end(),
                               [&](const auto &c) { return bar > birth(c) && bar - birth(c) > age; }),
                v.end());
    }
    void AddAB(std::vector<AB> &pool, const AB &ab, bool shadow = false)
    {
        pool.push_back(ab);
        std::sort(pool.begin(), pool.end(), [](const AB &a, const AB &b) { return ABOrder(a) < ABOrder(b); });
        while (pool.size() > fc.maxActiveABStructures)
        {
            if (shadow)
                capacityEvicted["TG3:" + Natural(pool.front())] = {pool.front().identity.availabilityBar,
                                                                   fc.maxABAgeBars};
            pool.erase(pool.begin());
        }
    }
    void TG3Step(const CompletedBarObservation &o)
    {
        std::vector<F> confirmed(o.confirmed.begin(), o.confirmed.end());
        std::sort(confirmed.begin(), confirmed.end(), FractalLess);
        std::vector<TG3::ABIdentity> births;
        for (const F &b : confirmed)
        {
            auto kind = static_cast<std::size_t>(b.kind), other = 1 - kind;
            auto &bounded = boundedTG3[kind];
            auto &restored = restoredTG3[kind];
            auto existing = std::find_if(bounded.begin(), bounded.end(), [&](const F &a)
                                         { return FractalIdentity(a) == FractalIdentity(b); });
            if (existing != bounded.end())
            {
                Count(0, kind, o.scoring, [](Counts &c) { Increment(c.duplicates); });
                continue;
            }
            auto a = Select(boundedTG3[other], b), shadow = Select(restoredTG3[other], b);
            if (a)
            {
                auto ab = MakeAB(*a, b);
                if (ab.priceRange > 0 && std::isfinite(ab.priceRange))
                {
                    births.push_back(ab.identity);
                    AddAB(boundedAB, ab);
                }
            }
            if (shadow)
            {
                auto ab = MakeAB(*shadow, b);
                if (ab.priceRange > 0 && std::isfinite(ab.priceRange))
                {
                    bool present = std::any_of(o.abBirths.begin(), o.abBirths.end(),
                                               [&](const auto &id) { return Natural(id) == Natural(ab); });
                    if (!present)
                    {
                        auto removed = std::find_if(
                            restoredTG3[other].begin(), restoredTG3[other].end(), [&](const Anchor &x)
                            { return FractalIdentity(x.fractal) == FractalIdentity(*shadow) && x.removed; });
                        Check(removed != restoredTG3[other].end(), "TG3_loss_without_removed_anchor");
                        Lost(ab, o);
                    }
                    AddAB(shadowAB, ab, true);
                }
            }
            bounded.push_back(b);
            std::sort(bounded.begin(), bounded.end(), FractalLess);
            restored.push_back({b, false});
            std::sort(restored.begin(), restored.end(),
                      [](const Anchor &a, const Anchor &z) { return FractalLess(a.fractal, z.fractal); });
            Count(0, kind, o.scoring,
                  [&](Counts &c)
                  {
                      Increment(c.insertions);
                      c.peakDemand = std::max(c.peakDemand, bounded.size());
                  });
            while (bounded.size() > fc.maxConfirmedFractalsPerKind)
            {
                Remove(0, kind, bounded.front(), o, true, bounded.size(), bounded.size());
                bounded.erase(bounded.begin());
            }
            Count(0, kind, o.scoring,
                  [&](Counts &c) { c.peakOccupancy = std::max(c.peakOccupancy, bounded.size()); });
            ResourceCheck();
        }
        Check(Equal(births, o.abBirths, [](const auto &x) { return Native(x); }),
              "TG3_reconstructed_birth_mismatch");
        Check(Equal(boundedAB, fibonacciMirror.ABStructures(), [](const auto &x) { return Snapshot(x); }),
              "TG3_reconstructed_retained_mismatch");
    }
    void TG1Step(const CompletedBarObservation &o)
    {
        // The native mirror establishes strict-five confirmation and native
        // High-before-Low sequencing independently from the observation.
        for (const F &b : o.confirmed)
        {
            auto kind = static_cast<std::size_t>(b.kind);
            PruneHistory(kind, o);
            auto &bounded = boundedTG1[kind];
            auto &restored = restoredTG1[kind];
            Count(1, kind, o.scoring,
                  [&](Counts &c)
                  {
                      Increment(c.insertions);
                      c.peakDemand = std::max(c.peakDemand, bounded.size() + 1);
                  });
            while (bounded.size() >= gc.maxConfirmedFractalsPerKind)
            {
                Remove(1, kind, bounded.front(), o, true, bounded.size(), bounded.size() + 1);
                bounded.erase(bounded.begin());
            }
            for (const Anchor &a : restored)
            {
                if (!(b.kind == TG1A::FractalKind::Low ? b.price > a.fractal.price
                                                       : b.price < a.fractal.price))
                    continue;
                auto line = MakeLine(a.fractal, b);
                if (!Valid(line))
                {
                    if (a.removed)
                    {
                        Increment(summary.structuralRejections);
                        Witness rejected;
                        rejected.detector = "TG1A";
                        rejected.identity = Natural(line);
                        rejected.line = line;
                        rejected.attribution = "UNRELATED_RULE_REJECTION";
                        Emit(rejected, "RULE_REJECTED", o, "intervening_structural_price_violation");
                    }
                    continue;
                }
                bool present = std::any_of(o.lineBirths.begin(), o.lineBirths.end(),
                                           [&](const L &l) { return Natural(l) == Natural(line); });
                if (!present)
                {
                    Check(a.removed, "TG1A_loss_without_removed_anchor");
                    Lost(line, o);
                }
                shadowLines.push_back(line);
            }
            bounded.push_back(b);
            restored.push_back({b, false});
            Count(1, kind, o.scoring,
                  [&](Counts &c) { c.peakOccupancy = std::max(c.peakOccupancy, bounded.size()); });
            ResourceCheck();
        }
        while (shadowLines.size() > gc.maxCandidates)
        {
            auto i = std::min_element(shadowLines.begin(), shadowLines.end(), [](const L &a, const L &b)
                                      { return LineEviction(a) < LineEviction(b); });
            capacityEvicted["TG1A:" + Natural(*i)] = {i->creationBar, gc.maxCandidateAgeBars};
            shadowLines.erase(i);
        }
        std::sort(shadowLines.begin(), shadowLines.end(),
                  [](const L &a, const L &b) { return LineOrder(a) < LineOrder(b); });
        PruneHistory(0, o);
        PruneHistory(1, o);
        std::vector<F> combined = boundedTG1[0];
        combined.insert(combined.end(), boundedTG1[1].begin(), boundedTG1[1].end());
        std::sort(combined.begin(), combined.end(),
                  [](const F &a, const F &b)
                  {
                      return std::tuple(a.anchorBar, static_cast<int>(a.kind), a.confirmationBar) <
                             std::tuple(b.anchorBar, static_cast<int>(b.kind), b.confirmationBar);
                  });
        Check(Equal(combined, geometryMirror.ConfirmedFractals(),
                    [](const F &x) { return FractalIdentity(x) + ":" + S::Number(x.price); }),
              "TG1A_reconstructed_history_mismatch");
    }
    void Current(L &l, const CompletedBarObservation &o) const
    {
        l.currentBar = o.bar;
        l.currentTimestamp = o.candle.timestamp;
        l.candidateAgeBars = o.bar - l.creationBar;
        l.projectedLinePriceAtCurrentBar = l.ProjectedPrice(o.bar);
        l.rawPriceToLineDistance = l.direction == TG1A::TrendLineDirection::UTL
                                       ? o.candle.low - l.projectedLinePriceAtCurrentBar
                                       : l.projectedLinePriceAtCurrentBar - o.candle.high;
        l.currentAtr = o.atr;
        if (o.atr)
        {
            l.atrNormalizedSlope = l.rawPriceSlopePerBar / *o.atr;
            l.atrNormalizedPriceToLineDistance = l.rawPriceToLineDistance / *o.atr;
        }
        else
        {
            l.atrNormalizedSlope.reset();
            l.atrNormalizedPriceToLineDistance.reset();
        }
    }
    bool Contributes(Witness &w, const CompletedBarObservation &o) const
    {
        if (!o.atr || !std::isfinite(*o.atr) || *o.atr <= 0)
            return false;
        // A YES is a witnessed same-pair frozen-ratio C conjunction in the
        // history-restored universe. It is NOT a claimed primary cohort change.
        auto pair = [&](const AB &a, const L &l)
        {
            if (a.identity.direction != TG3::ABDirection::UpAB ||
                l.direction != TG1A::TrendLineDirection::UTL)
                return false;
            if (!S::Diagnostic::Detail::Eligible(a, o.bar) || !S::Diagnostic::Detail::Eligible(l, o.bar))
                return false;
            for (double ratio : S::kRatios)
            {
                auto f = S::Diagnostic::Detail::FiniteDouble(std::lerp(a.bPrice, a.aPrice, ratio));
                if (!f)
                    continue;
                double p = l.ProjectedPrice(o.bar);
                auto dF = S::Diagnostic::Detail::NormalizedDistance(o.candle.close, *f, *o.atr);
                auto dL = S::Diagnostic::Detail::NormalizedDistance(o.candle.close, p, *o.atr);
                auto dS = S::Diagnostic::Detail::NormalizedDistance(*f, p, *o.atr);
                if (dF && dL && dS && *dF <= .25 && *dL <= .25 && *dS <= .25)
                {
                    w.contributingPair = Native(a.identity) + "/" + Natural(l) + ":" + N(l.anchor1Bar) + ":" +
                                         N(l.anchor2Bar) + ":" + N(l.creationBar);
                    w.contributionTimestamp = o.candle.timestamp;
                    w.contributionClose = o.candle.close;
                    w.contributionAtr = *o.atr;
                    w.contributionRatio = ratio;
                    w.contributionFib = *f;
                    w.contributionLine = p;
                    return true;
                }
            }
            return false;
        };
        if (w.detector == "TG3")
        {
            for (const auto &l : o.retainedLines)
                if (pair(w.ab, l))
                    return true;
            for (const auto &l : shadowLines)
                if (pair(w.ab, l))
                    return true;
        }
        else
        {
            for (const auto &a : o.retainedAB)
                if (pair(a, w.line))
                    return true;
            for (const auto &a : shadowAB)
                if (pair(a, w.line))
                    return true;
        }
        return false;
    }
    void Survival(const CompletedBarObservation &o)
    {
        std::set<std::string> shadowABIdentities, shadowLineIdentities;
        for (const auto &a : shadowAB)
            shadowABIdentities.insert(Natural(a));
        for (const auto &l : shadowLines)
            shadowLineIdentities.insert(Natural(l));
        // Attribute absence to capacity only with an explicit eviction witness.
        for (const auto &a : o.retainedAB)
            if (!shadowABIdentities.contains(Natural(a)))
            {
                auto key = "TG3:displaced:" + Natural(a);
                if (!liveWitnesses.contains(key))
                {
                    Witness w;
                    w.detector = "TG3";
                    w.identity = Natural(a);
                    w.ab = a;
                    w.attribution = capacityEvicted.contains("TG3:" + w.identity)
                                        ? "CAPACITY_MEDIATED_DISPLACEMENT"
                                        : "STATE_DIVERGENCE_UNATTRIBUTED";
                    liveWitnesses.emplace(key, w);
                    if (w.attribution == "CAPACITY_MEDIATED_DISPLACEMENT")
                        Increment(summary.capacityDisplacements);
                    Emit(w, "DISPLACED", o,
                         w.attribution == "CAPACITY_MEDIATED_DISPLACEMENT"
                             ? "explicit_counterfactual_capacity_eviction"
                             : "absence_without_capacity_witness");
                }
            }
        for (const auto &l : o.retainedLines)
            if (!shadowLineIdentities.contains(Natural(l)))
            {
                auto key = "TG1A:displaced:" + Natural(l);
                if (!liveWitnesses.contains(key))
                {
                    Witness w;
                    w.detector = "TG1A";
                    w.identity = Natural(l);
                    w.line = l;
                    w.attribution = capacityEvicted.contains("TG1A:" + w.identity)
                                        ? "CAPACITY_MEDIATED_DISPLACEMENT"
                                        : "STATE_DIVERGENCE_UNATTRIBUTED";
                    liveWitnesses.emplace(key, w);
                    if (w.attribution == "CAPACITY_MEDIATED_DISPLACEMENT")
                        Increment(summary.capacityDisplacements);
                    Emit(w, "DISPLACED", o,
                         w.attribution == "CAPACITY_MEDIATED_DISPLACEMENT"
                             ? "explicit_counterfactual_capacity_eviction"
                             : "absence_without_capacity_witness");
                }
            }
        for (auto i = liveWitnesses.begin(); i != liveWitnesses.end();)
        {
            auto &w = i->second;
            if (w.detector == "TG1A")
                Current(w.line, o);
            bool ab = w.detector == "TG3", displaced = w.attribution != "DIRECT_HISTORY_SUPPRESSION";
            bool present =
                ab ? shadowABIdentities.contains(w.identity) : shadowLineIdentities.contains(w.identity);
            auto birth = ab ? w.ab.identity.availabilityBar : w.line.creationBar;
            auto age = ab ? fc.maxABAgeBars : gc.maxCandidateAgeBars;
            if (displaced)
            {
                if (o.bar - birth > age)
                {
                    Emit(w, "RETIRED", o, "age_expiration");
                    i = liveWitnesses.erase(i);
                }
                else
                    ++i;
                continue;
            }
            if (!present)
            {
                Emit(w, "RETIRED", o,
                     o.bar - birth > age ? "age_expiration" : "counterfactual_candidate_capacity");
                i = liveWitnesses.erase(i);
                continue;
            }
            if (!w.survived)
            {
                w.survived = true;
                Increment(summary.survivingLosses);
                Emit(w, "SURVIVED_COMPLETED_BAR", o, "original_candidate_limits");
            }
            bool directional = ab ? w.ab.identity.direction == TG3::ABDirection::UpAB
                                  : w.line.direction == TG1A::TrendLineDirection::UTL;
            if (o.scheduled && o.decisionWindowEligible && directional)
            {
                if (!w.decision)
                {
                    w.decision = true;
                    Increment(summary.decisionEligibleLosses);
                    Emit(w, "FIRST_ELIGIBLE_DECISION", o, "scheduled_post_warmup_candidate_eligibility_only");
                }
                if (!w.contributed && Contributes(w, o))
                {
                    w.contributed = true;
                    Increment(summary.contributingLosses);
                    Emit(w, "FIRST_GEOMETRY_CONTRIBUTION", o,
                         "same_pair_C_conjunction_not_primary_cohort_change");
                }
            }
            ++i;
        }
    }
    void Observe(const CompletedBarObservation &o)
    {
        if (finished)
            throw std::logic_error("audit already finalized");
        Check(o.bar == summary.bars && (!lastTimestamp || o.candle.timestamp > *lastTimestamp),
              "observation_order_mismatch");
        candleHash.Add(T(o.candle.timestamp) + "," + S::Number(o.candle.open) + "," +
                       S::Number(o.candle.high) + "," + S::Number(o.candle.low) + "," +
                       S::Number(o.candle.close) + "\n");
        auto gu = geometryMirror.AddCompletedBar(o.candle);
        Check(Equal(gu.newlyConfirmedFractals, o.confirmed,
                    [](const F &f) { return FractalIdentity(f) + ":" + S::Number(f.price); }),
              "TG1A_confirmation_parity");
        Check(Equal(gu.newlyCreatedCandidates, o.lineBirths, [](const L &l) { return Snapshot(l); }),
              "TG1A_raw_birth_parity");
        Check(Equal(geometryMirror.Candidates(), o.retainedLines, [](const L &l) { return Snapshot(l); }),
              "TG1A_retained_parity");
        Check(geometryMirror.CurrentAtr() == o.atr, "TG1A_ATR_parity");
        fibonacciMirror.Advance(o.bar, o.candle.timestamp);
        auto ab =
            fibonacciMirror.ObserveConfirmedFractals(std::vector<F>(o.confirmed.begin(), o.confirmed.end()));
        Check(Equal(ab, o.abBirths, [](const auto &a) { return Native(a); }), "TG3_raw_birth_parity");
        Check(Equal(fibonacciMirror.ABStructures(), o.retainedAB, [](const AB &a) { return Snapshot(a); }),
              "TG3_retained_parity");
        Check(fibonacciMirror.TotalABStructuresCreated() == o.totalABCreated &&
                  fibonacciMirror.ABCapacityEvictions() == o.abCapacityEvictions &&
                  fibonacciMirror.ABAgeExpirations() == o.abAgeExpirations,
              "TG3_lifecycle_counter_parity");
        Increment(summary.parityChecks);
        bars.push_back({o.bar, o.candle});
        Expire(boundedAB, o.bar, fc.maxABAgeBars, [](const AB &a) { return a.identity.availabilityBar; });
        Expire(shadowAB, o.bar, fc.maxABAgeBars, [](const AB &a) { return a.identity.availabilityBar; });
        Expire(shadowLines, o.bar, gc.maxCandidateAgeBars, [](const L &l) { return l.creationBar; });
        for (auto i = capacityEvicted.begin(); i != capacityEvicted.end();)
        {
            if (o.bar - i->second.first > i->second.second)
                i = capacityEvicted.erase(i);
            else
                ++i;
        }
        TG3Step(o);
        TG1Step(o);
        for (auto &l : shadowLines)
            Current(l, o);
        Survival(o);
        while (!bars.empty() && o.bar - bars.front().bar > gc.maxFractalAnchorLookbackBars + 2)
            bars.pop_front();
        for (std::size_t detector = 0; detector < 2; ++detector)
            for (std::size_t kind = 0; kind < 2; ++kind)
            {
                const auto size = (detector == 0 ? boundedTG3[kind] : boundedTG1[kind]).size();
                Count(detector, kind, o.scoring,
                      [&](Counts &c)
                      {
                          c.peakOccupancy = std::max(c.peakOccupancy, size);
                          c.peakDemand = std::max(c.peakDemand, size);
                      });
                const auto &full = summary.history[detector][kind][0];
                Check(full.insertions >= full.capacityRemovals + full.expirations &&
                          full.insertions - full.capacityRemovals - full.expirations == size,
                      "bounded_history_counter_reconciliation");
            }
        ResourceCheck();
        lastTimestamp = o.candle.timestamp;
        Increment(summary.bars);
    }
};

Audit::Audit(TG3::Configuration f, TG1A::Configuration g, std::filesystem::path d, Budgets b,
             PublicationHook hook)
{
    // Never overwrite evidence or create this sidecar inside an existing primary
    // publication. Caller additionally enforces repository/snapshot boundaries.
    if (std::filesystem::exists(d) || !std::filesystem::create_directories(d))
        throw std::invalid_argument("audit output must be a new directory");
    state_ = std::make_unique<State>(std::move(f), g, std::move(d), b, std::move(hook));
}
Audit::~Audit() = default;
void ValidateFrozenSourceSnapshot(const std::filesystem::path &snapshot)
{
    if (S::FileHash(snapshot / "Headers/CausalFibonacciConfluenceIntegration.hpp") != kTG3SourceSHA256 ||
        S::FileHash(snapshot / "Headers/CausalFractalTrendLineGeometry.hpp") != kTG1ASourceSHA256)
        throw std::runtime_error("audit frozen detector source pin mismatch before execution");
}
void Audit::Observe(const CompletedBarObservation &o) noexcept
{
    if (!state_->summary.complete)
        return;
    try
    {
        state_->Observe(o);
    }
    catch (const std::bad_alloc &)
    {
        state_->Fail("diagnostic_allocation_exhaustion", false, true);
    }
    catch (const std::overflow_error &)
    {
        state_->Fail("audit_counter_overflow", false, true);
    }
    catch (const std::exception &e)
    {
        state_->Fail(e.what());
    }
    catch (...)
    {
        state_->Fail("unknown_diagnostic_failure");
    }
}
const Summary &Audit::GetSummary() const
{
    return state_->summary;
}

bool Audit::Complete(const Provenance &p)
{
    auto &s = *state_;
    if (s.finished)
        throw std::logic_error("audit already finalized");
    s.finished = true;
    auto removed = s.removals.Finish(), witness = s.witnesses.Finish();
    auto verifiedRemoved = S::InspectGzip(s.dir / "history_removals.csv.gz"),
         verifiedWitness = S::InspectGzip(s.dir / "candidate_witnesses.csv.gz");
    if (removed.sha256 != verifiedRemoved.sha256 || witness.sha256 != verifiedWitness.sha256 ||
        removed.rows != s.summary.removalRecords || witness.rows != s.summary.witnessRecords)
        throw std::runtime_error("audit compressed-content reconciliation failure");
    S::ValidateArtifactPublication(p.parentPrimary);
    if (s.summary.complete && s.candleHash.Digest() != p.candleDigest)
        throw std::runtime_error("audit observed candle digest differs from parent");
    if (!S::IsHash(p.semanticFingerprint) || !S::IsHash(p.sourceSnapshotDigest) ||
        !S::IsHash(p.candleDigest) || p.gitCommit.size() != 40)
        throw std::invalid_argument("audit provenance missing");
    // Parent identity is validated as literal JSON hash fields; the sidecar never
    // parses outcome values (the parent validator hashes their opaque bytes).
    auto read = [](const std::filesystem::path &path)
    {
        std::ifstream f(path, std::ios::binary);
        f.exceptions(std::ios::badbit);
        return std::string(std::istreambuf_iterator<char>(f), {});
    };
    auto primaryProtocol = read(p.parentPrimary / "protocol.json"),
         primaryProvenance = read(p.parentPrimary / "provenance_manifest.json");
    if (primaryProtocol.find("\"artifact_format\":\"fib-trendline-artifacts-v3\"") == std::string::npos ||
        primaryProtocol.find("\"schema\":\"fib-trendline-decision-study-v2\"") == std::string::npos)
        throw std::runtime_error("audit parent must retain frozen semantic-v2/artifact-v3 contract");
    auto field = [](const std::string &document, const char *key)
    {
        std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*\"([0-9a-f]+)\"");
        std::smatch match;
        if (!std::regex_search(document, match, pattern))
            throw std::runtime_error("audit parent metadata field absent");
        return match[1].str();
    };
    if (field(primaryProtocol, "protocol_fingerprint") != p.semanticFingerprint ||
        field(primaryProvenance, "canonical_candle_content_sha256") != p.candleDigest ||
        field(primaryProvenance, "source_manifest_sha256") != p.sourceSnapshotDigest ||
        field(primaryProvenance, "git_commit") != p.gitCommit)
        throw std::runtime_error("audit parent identity mismatch");
    const auto manifest = read(p.parentPrimary / "source_manifest.sha256");
    for (const auto &required : {"5f2d52a810dd175321ece512d6f4693c92913ec98b1234daed40d3095aeec2fa  "
                                 "Headers/CausalFibonacciConfluenceIntegration.hpp",
                                 "c6eaa92bfec6c8dde0e67c54920e34d0fcf133c98c8fc5a500c2227214cdf507  "
                                 "Headers/CausalFractalTrendLineGeometry.hpp"})
        if (manifest.find(required) == std::string::npos)
            throw std::runtime_error("audit frozen detector source pin mismatch");
    std::ostringstream csv;
    csv.imbue(std::locale::classic());
    csv << "detector,kind,phase,insertions,duplicate_suppressions,capacity_removals,lookback_expirations,"
           "relevant_removals,peak_occupancy,peak_logical_demand,first_capacity_removal_bar,first_capacity_"
           "removal_utc_epoch\n";
    for (std::size_t d = 0; d < 2; ++d)
        for (std::size_t k = 0; k < 2; ++k)
            for (std::size_t phase = 0; phase < 3; ++phase)
            {
                const auto &c = s.summary.history[d][k][phase];
                csv << (d == 0 ? "TG3" : "TG1A") << ','
                    << (k == static_cast<std::size_t>(TG1A::FractalKind::High) ? "High" : "Low") << ','
                    << (phase == 0   ? "full_replay"
                        : phase == 1 ? "prefix"
                                     : "scoring")
                    << ',' << c.insertions << ',' << c.duplicates << ',' << c.capacityRemovals << ','
                    << c.expirations << ',' << c.relevantRemovals << ',' << c.peakOccupancy << ','
                    << c.peakDemand << ',' << (c.firstRemovalBar ? N(*c.firstRemovalBar) : "") << ','
                    << (c.firstRemovalTimestamp ? T(*c.firstRemovalTimestamp) : "") << '\n';
            }
    s.Publish("history_summary.csv", csv.str());
    std::ostringstream protocol;
    protocol
        << "{\"schema\":" << Json(kSchema)
        << ",\"semantic_protocol\":\"fib-trendline-decision-study-v2\",\"primary_artifact_format\":\"fib-"
           "trendline-artifacts-v3\",\"history_intervention_only\":true,\"decision_eligibility\":\"scheduled "
           "temporal warmup/ATR/embargo/horizon eligibility; not primary "
           "acceptance\",\"timestamp_convention\":\"UTC bar-start; confirmation available after completed "
           "bar\",\"TG3_history_per_kind\":128,\"TG1A_history_per_kind\":64,\"TG3_candidate_capacity\":512,"
           "\"TG1A_candidate_capacity\":4096,\"TG3_anchor_age_limit\":\"NOT_APPLICABLE\",\"TG1A_lookback_"
           "inclusive\":512,\"candidate_age_limits\":[2048,512],\"contribution\":\"same_pair_C_conjunction_"
           "in_restored_universe; no means UNKNOWN; not primary cohort "
           "change\",\"unrelated_rule_rejection\":\"structural rejection counter; no lost-candidate "
           "claim\",\"state_divergence\":\"parity mismatch stops audit; "
           "STATE_DIVERGENCE_UNATTRIBUTED\",\"TG3_source_sha256\":"
        << Json(kTG3SourceSHA256) << ",\"TG1A_source_sha256\":" << Json(kTG1ASourceSHA256)
        << ",\"geometry_intervening_tolerance\":" << S::Number(s.gc.interveningPriceTolerance)
        << ",\"geometry_touch_tolerance\":" << S::Number(s.gc.touchPriceTolerance)
        << ",\"atr_period\":" << s.gc.atrPeriod << ",\"shadow_fractal_budget\":" << s.budgets.shadowFractals
        << ",\"live_candidate_budget\":" << s.budgets.liveCandidates
        << ",\"witness_record_budget\":" << s.budgets.witnessRecords
        << ",\"removal_record_budget\":" << s.budgets.removalRecords
        << ",\"diagnostic_byte_budget\":" << s.budgets.diagnosticBytes
        << ",\"decoded_publication_byte_budget\":" << s.budgets.decodedPublicationBytes
        << ",\"publication_buffer_bytes\":" << s.budgets.publicationBufferBytes
        << ",\"compression\":{\"codec\":\"gzip\",\"level\":6,\"window_bits\":31,\"memory_level\":8,"
           "\"strategy\":\"default\",\"mtime\":0,\"os\":255,\"zlib_version\":"
        << Json(zlibVersion()) << ",\"header_version\":" << Json(ZLIB_VERSION) << "}}\n";
    s.Publish("audit_protocol.json", protocol.str());
    std::ostringstream prov;
    prov << "{\"schema\":" << Json(kSchema) << ",\"semantic_fingerprint\":" << Json(p.semanticFingerprint)
         << ",\"git_commit\":" << Json(p.gitCommit)
         << ",\"source_snapshot_digest\":" << Json(p.sourceSnapshotDigest)
         << ",\"canonical_candle_sha256\":" << Json(p.candleDigest)
         << ",\"parent_primary_checksums_sha256\":" << Json(S::FileHash(p.parentPrimary / "checksums.sha256"))
         << ",\"parent_complete_sha256\":" << Json(S::FileHash(p.parentPrimary / "COMPLETE"))
         << ",\"mirror_parity\":" << (s.summary.parity ? "true" : "false")
         << ",\"diagnostic_complete\":" << (s.summary.complete ? "true" : "false")
         << ",\"capacity_exhausted\":" << (s.summary.exhausted ? "true" : "false")
         << ",\"failure\":" << Json(s.summary.failure) << ",\"lost_candidate_totals\":"
         << (s.summary.complete ? Json("EXACT_WITHIN_OBSERVED_REPLAY") : Json("PARTIAL_NOT_EXACT"))
         << ",\"observed_bars\":" << s.summary.bars
         << ",\"audited_prefix_candle_sha256\":" << Json(s.candleHash.Digest())
         << ",\"parity_checks\":" << s.summary.parityChecks
         << ",\"creation_losses_TG3\":" << s.summary.creationLossesTG3
         << ",\"creation_losses_TG1A\":" << s.summary.creationLossesTG1A
         << ",\"capacity_mediated_displacements\":" << s.summary.capacityDisplacements
         << ",\"unrelated_structural_rule_rejections\":" << s.summary.structuralRejections
         << ",\"surviving_losses\":" << s.summary.survivingLosses
         << ",\"decision_eligible_losses\":" << s.summary.decisionEligibleLosses
         << ",\"contributing_losses\":" << s.summary.contributingLosses
         << ",\"peak_accounted_bytes\":" << s.summary.peakAccountedBytes
         << ",\"compressed_contents\":{\"history_removals.csv.gz\":{\"decoded_sha256\":"
         << Json(removed.sha256) << ",\"decoded_rows\":" << removed.rows
         << "},\"candidate_witnesses.csv.gz\":{\"decoded_sha256\":" << Json(witness.sha256)
         << ",\"decoded_rows\":" << witness.rows << "}}}\n";
    s.Publish("audit_provenance.json", prov.str());
    std::string ledger;
    for (const char *name : {"audit_protocol.json", "audit_provenance.json", "candidate_witnesses.csv.gz",
                             "history_removals.csv.gz", "history_summary.csv"})
        ledger += S::FileHash(s.dir / name) + "  " + name + "\n";
    s.Publish("audit_checksums.sha256", ledger);
    if (s.hook)
        s.hook(PublicationStep::Checksum);
    // Re-read every published byte before the atomic completion boundary.
    std::istringstream rows(ledger);
    std::string row;
    while (std::getline(rows, row))
        if (S::FileHash(s.dir / row.substr(66)) != row.substr(0, 64))
            throw std::runtime_error("audit checksum mismatch");
    if (!s.summary.complete || !s.summary.parity)
        return false;
    if (s.hook)
        s.hook(PublicationStep::Marker);
    Write(s.dir / "AUDIT_COMPLETE.pending",
          S::FileHash(s.dir / "audit_checksums.sha256") + "  audit_checksums.sha256\n");
    if (std::filesystem::exists(s.dir / "AUDIT_COMPLETE"))
        throw std::runtime_error("audit completion target already exists");
    std::filesystem::rename(s.dir / "AUDIT_COMPLETE.pending", s.dir / "AUDIT_COMPLETE");
    return true;
}
void ValidateSidecar(const std::filesystem::path &d)
{
    std::ifstream f(d / "audit_checksums.sha256");
    if (!f)
        throw std::runtime_error("audit checksums absent");
    std::set<std::string> expected{"audit_protocol.json", "audit_provenance.json",
                                   "candidate_witnesses.csv.gz", "history_removals.csv.gz",
                                   "history_summary.csv"};
    std::string row;
    while (std::getline(f, row))
    {
        if (row.size() < 67 || row.substr(64, 2) != "  " || !expected.erase(row.substr(66)) ||
            S::FileHash(d / row.substr(66)) != row.substr(0, 64))
            throw std::runtime_error("audit checksum invalid");
    }
    if (!expected.empty())
        throw std::runtime_error("audit checksum coverage incomplete");
    std::ifstream provenance(d / "audit_provenance.json");
    std::string json(std::istreambuf_iterator<char>(provenance), {});
    if (json.find("\"mirror_parity\":true") == std::string::npos ||
        json.find("\"diagnostic_complete\":true") == std::string::npos ||
        json.find("\"capacity_exhausted\":false") == std::string::npos)
        throw std::runtime_error("audit incomplete metadata");
    for (const char *name : {"candidate_witnesses.csv.gz", "history_removals.csv.gz"})
    {
        auto decoded = S::InspectGzip(d / name);
        std::string claim = Json(name) + ":{\"decoded_sha256\":" + Json(decoded.sha256) +
                            ",\"decoded_rows\":" + N(decoded.rows) + "}";
        if (json.find(claim) == std::string::npos)
            throw std::runtime_error("audit decoded-content claim mismatch");
    }
    std::ifstream marker(d / "AUDIT_COMPLETE");
    std::getline(marker, row);
    if (row != S::FileHash(d / "audit_checksums.sha256") + "  audit_checksums.sha256")
        throw std::runtime_error("audit completion invalid");
}
} // namespace EA::FibonacciResearch::HistoryShadow
