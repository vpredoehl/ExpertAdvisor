#include "CausalFibonacciTrendlineHistoricalArtifact.hpp"
#include "CausalFibonacciResearchCandleSnapshot.hpp"
#include <cassert>
#include <iostream>
#include <random>

namespace
{
namespace S = EA::FibonacciResearch::TrendlineHistorical;
using Candle = EA::TG1A::Candle;
constexpr std::size_t kFirst = 2051; // UTC :45, after 2048-bar warmup.
std::int64_t Time(std::size_t bar) { return S::kOrigin + static_cast<std::int64_t>(bar) * 900; }
Candle Bar(std::size_t bar, double close = 100) { return {Time(bar), close, close + 1, close - 1, close}; }
EA::TG3::ABStructure AB(double low = 100, double high = 100.1, int offset = 0)
{
    EA::TG3::ABStructure ab;
    ab.identity = {EA::TG3::ABDirection::UpAB, 2042, Time(2042) + offset * 900, 2045, Time(2045), 2047, Time(2047)};
    ab.aPrice = low; ab.bPrice = high;
    ab.aConfirmationBar = 2044; ab.aConfirmationTimestamp = Time(2044);
    ab.bConfirmationBar = 2047; ab.bConfirmationTimestamp = Time(2047);
    return ab;
}
EA::TG1A::TrendLineCandidate Line(std::size_t bar, double price = 100, int offset = 0)
{
    EA::TG1A::TrendLineCandidate l;
    l.anchor1Bar = 2041; l.anchor1Timestamp = Time(2041) + offset * 900; l.anchor1Price = price;
    l.anchor1ConfirmationBar = 2043; l.anchor1ConfirmationTimestamp = Time(2043);
    l.anchor2Bar = 2044; l.anchor2Timestamp = Time(2044); l.anchor2Price = price;
    l.anchor2ConfirmationBar = 2046; l.anchor2ConfirmationTimestamp = Time(2046);
    l.creationBar = 2046; l.creationTimestamp = Time(2046); l.anchorSeparationBars = 3;
    l.currentBar = bar; l.currentTimestamp = Time(bar);
    return l;
}
template<class F> void Reject(F f)
{
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
struct Harness
{
    std::vector<S::Decision> decisions;
    std::vector<S::OutcomeRecord> outcomes;
    std::vector<S::GeometryMeasurement> geometry;
    std::vector<S::Episode> episodes;
    std::vector<S::Gap> gaps;
    S::EventEngine engine;
    std::size_t bar = 0;
    explicit Harness(S::Protocol p = {}, S::ScoreRange range = {Time(2048), Time(2076)}, S::SessionCalendar calendar = {})
        : engine(p, "fixture", {[this](const auto& v) { decisions.push_back(v); }, [this](const auto& v) { outcomes.push_back(v); },
            [this](const auto& v) { geometry.push_back(v); }, [this](const auto& v) { episodes.push_back(v); },
            [this](const auto& v) { gaps.push_back(v); }}, range, std::move(calendar)) {}
    void Empty(std::int64_t time = -1, double close = 100)
    {
        auto c = Bar(bar, close); if (time != -1) c.timestamp = time;
        engine.Add(bar++, c, 2, {}, {});
    }
    void Warm() { while (bar < kFirst) Empty(); }
    void Geometry(double close = 100, double low = 100, double high = 100.1, double linePrice = 100, double atr = 2)
    {
        const auto ab = AB(low, high); const auto line = Line(bar, linePrice);
        engine.Add(bar, Bar(bar, close), atr, std::span{&ab, 1}, std::span{&line, 1}); ++bar;
    }
};
std::string Decisions(const std::vector<S::Decision>& rows)
{
    std::ostringstream out;
    for (const auto& d : rows)
    {
        out << d.identity << '|' << d.timestamp << '|' << d.status << '|'; if (d.cohort) out << S::Name(*d.cohort);
        out << '|' << d.aggregate.measurements << '|' << d.aggregate.intersections << '|' << d.aggregate.complete
            << '|' << d.aggregate.evaluated << '|' << d.aggregate.fibNear << '|' << d.aggregate.lineNear << '|' << d.aggregate.intersection << '|';
        if (d.aggregate.minDF) out << S::Number(*d.aggregate.minDF);
        out << '|'; if (d.aggregate.minDL) out << S::Number(*d.aggregate.minDL);
        out << '|'; if (d.aggregate.minDS) out << S::Number(*d.aggregate.minDS);
        for (const auto& v : {d.close, d.atr, d.covariates.return4, d.covariates.return16, d.covariates.return64,
                              d.covariates.logAtrClose, d.covariates.logAtrTR64})
        { out << '|'; if (v) out << S::Number(*v); }
        for (double v : {d.covariates.meanABAge, d.covariates.meanLineAge, d.covariates.meanABRangeAtr,
                         d.covariates.meanLineSlopeAtr, d.covariates.newABFraction, d.covariates.newLineFraction})
            out << '|' << S::Number(v);
        out << '|' << d.covariates.utcHour << '|' << d.covariates.utcWeekday << '|' << d.covariates.barsSinceGap << '\n';
    }
    return out.str();
}
void Cohorts()
{
    for (const auto& [close, low, high, line, atr, expected] : std::array{
        std::tuple{100., 100., 100.1, 102., 2., S::Cohort::A},
        std::tuple{100., 110., 110.1, 100., 2., S::Cohort::B},
        std::tuple{100., 100., 100.1, 100., 2., S::Cohort::C},
        std::tuple{100., 110., 110.1, 115., 2., S::Cohort::D},
        std::tuple{100.2, 100., 100.01, 100.4, 1., S::Cohort::FStratum}})
    {
        Harness h; h.Warm(); h.Geometry(close, low, high, line, atr);
        assert(h.decisions.back().status == "classified" && h.decisions.back().cohort == expected);
        assert(h.decisions.back().aggregate.measurements == 3);
    }
    Harness missing; missing.Warm(); missing.Empty();
    assert(missing.decisions.back().status == "no_geometry" && !missing.decisions.back().cohort);
    Harness missingFib; missingFib.Warm(); const auto line = Line(kFirst);
    missingFib.engine.Add(kFirst, Bar(kFirst), 2, {}, std::span{&line, 1});
    assert(missingFib.decisions.back().status == "no_geometry");
    Harness missingLine; missingLine.Warm(); const auto ab = AB();
    missingLine.engine.Add(kFirst, Bar(kFirst), 2, std::span{&ab, 1}, {});
    assert(missingLine.decisions.back().status == "no_geometry");
    // A distant pair can have dS near zero without supporting intersection at
    // the market. Independent minima cannot establish same-pair confluence.
    Harness same; same.Warm();
    const std::array abs{AB(99.79, 99.8), AB(109.99, 110, -1)};
    const std::array lines{Line(kFirst, 100.2), Line(kFirst, 110, -1)};
    same.engine.Add(kFirst, Bar(kFirst), 1, abs, lines);
    assert(same.decisions.back().cohort == S::Cohort::FStratum);
    assert(same.decisions.back().aggregate.minDS && *same.decisions.back().aggregate.minDS < .25);
    assert(!same.decisions.back().aggregate.intersection);
    // D geometry ordered first cannot suppress a simultaneous C geometry.
    Harness all; all.Warm(); const std::array a2{AB(110, 110.1, -1), AB()};
    all.engine.Add(kFirst, Bar(kFirst), 2, a2, std::span{&line, 1});
    assert(all.decisions.back().cohort == S::Cohort::C && all.decisions.back().aggregate.measurements == 6);
}
void ScheduleCoverageOutcomes()
{
    Harness h; h.Warm(); h.Geometry();
    for (std::size_t i = 1; i <= 4; ++i)
    {
        const double close = 100 + static_cast<double>(i);
        Candle c{Time(kFirst + i), close, 102 + static_cast<double>(i), 100 - static_cast<double>(i), close};
        h.engine.Add(h.bar++, c, 999, {}, {});
    }
    assert(h.outcomes.size() == 1 && h.outcomes[0].outcome);
    const auto& o = *h.outcomes[0].outcome;
    assert(o.signedAtrChange == 2 && o.favorableAtr == 3 && o.adverseAtr == 2);
    assert(std::abs(o.signedLogReturn - std::log(1.04)) < 1e-14 && o.directionalHit && !o.flat);
    assert(h.outcomes[0].resolutionTimestamp == Time(kFirst + 4));
    h.engine.Finalize();
    std::set<std::int64_t> slots;
    for (const auto& d : h.decisions) { assert(d.timestamp % 3600 == 2700); assert(slots.insert(d.timestamp).second); }
    assert(slots.size() == 7 && h.engine.GetAudit().classified == 1 && h.engine.GetAudit().resolved == 1);
    assert(h.decisions.back().status == "missing_slot_unverified");
    assert(h.decisions.back().reason == "source_coverage_shortfall");
    std::size_t sum = 0; for (const auto& [key, value] : h.engine.GetAudit().decisions) { (void)key; sum += value; }
    assert(sum == h.engine.GetAudit().scheduled);
    for (std::size_t position = 1; position <= 4; ++position)
    {
        Harness gap; gap.Warm(); gap.Geometry();
        for (std::size_t i = 1; i < position; ++i) gap.Empty();
        gap.Empty(Time(kFirst + position + 1));
        assert(gap.outcomes.size() == 1 && gap.outcomes[0].reason == "unexpected_timestamp_gap" && !gap.outcomes[0].outcome);
        gap.engine.Finalize();
        for (const auto& d : gap.decisions) assert(d.timestamp % 3600 == 2700);
    }
    for (auto delta : {172800LL, 1800LL})
    {
        Harness gap({}, {Time(2048), Time(kFirst) + delta + 7200}); gap.Warm(); gap.Geometry(); gap.Empty(Time(kFirst) + delta);
        gap.engine.Finalize(); assert(gap.outcomes[0].status == "censored");
        assert(gap.engine.GetAudit().decisions.at("missing_slot_unverified") > 0);
    }
    S::SessionCalendar calendar{"synthetic-calendar", S::Hash("fixture calendar"), [](std::int64_t t)
        { return t == Time(kFirst) ? S::Session::Closed : S::Session::Open; }};
    Harness calendarH({}, {Time(kFirst), Time(kFirst + 8)}, calendar);
    while (calendarH.bar < kFirst) calendarH.Empty();
    calendarH.Empty(Time(kFirst + 5)); calendarH.engine.Finalize();
    assert(calendarH.decisions[0].status == "closed_market");
    assert(calendarH.decisions[1].status == "missing_open_session");
    Harness conflict({}, {Time(kFirst), Time(kFirst + 8)}, calendar); conflict.Warm(); conflict.Geometry();
    assert(!conflict.engine.GetAudit().calendarConsistent && conflict.decisions.back().status == "calendar_conflict");
    S::SessionCalendar forwardClosed{"synthetic-forward-calendar", S::Hash("fixture forward calendar"), [](std::int64_t t)
        { return t == Time(kFirst + 1) ? S::Session::Closed : S::Session::Open; }};
    Harness expected({}, {Time(kFirst), Time(kFirst + 8)}, forwardClosed); expected.Warm(); expected.Geometry();
    assert(expected.decisions[0].status == "closed_horizon" && expected.outcomes.empty());
    Harness warm({}, {Time(0), Time(16)});
    for (int i = 0; i < 16; ++i) warm.Empty(); warm.engine.Finalize();
    assert(warm.engine.GetAudit().decisions.at("warmup") == 4);
    Harness purge({}, {Time(kFirst), Time(kFirst + 4)}); purge.Warm(); purge.Geometry(); purge.engine.Finalize();
    assert(purge.decisions[0].status == "partition_purged" && purge.outcomes.empty());
    Harness eof; eof.Warm(); eof.Geometry(); eof.Empty(); eof.engine.Finalize();
    assert(eof.outcomes[0].reason == "source_coverage_shortfall");
    // Fixed :45 opportunities continue while a geometric episode remains near.
    Harness repeated; repeated.Warm(); for (int i = 0; i < 13; ++i) repeated.Geometry(); repeated.engine.Finalize();
    assert(repeated.engine.GetAudit().classified == 4 && repeated.engine.GetAudit().resolved == 3);
    assert(repeated.episodes.size() == 3); // One initial near episode per ratio.
}
void InvarianceAndCapacity()
{
    const auto run = [](std::size_t batch, bool reverse, bool aliases)
    {
        S::Protocol p; p.lineBatchSize = batch; Harness h(p); h.Warm();
        std::vector<EA::TG3::ABStructure> abs{AB(), AB(110, 110.1, -1)};
        std::vector<EA::TG1A::TrendLineCandidate> lines{Line(kFirst), Line(kFirst, 115, -1)};
        if (aliases)
        {
            auto a = abs[0]; --a.identity.aBar; abs.push_back(a);
            auto l = lines[0]; --l.anchor1Bar; lines.push_back(l);
            // No economic geometry changed; arbitrary alias/index fields cannot
            // change multiplicity, output order, cohort or decision identity.
        }
        if (reverse) { std::reverse(abs.begin(), abs.end()); std::reverse(lines.begin(), lines.end()); }
        h.engine.Add(kFirst, Bar(kFirst), 2, abs, lines); h.engine.Finalize();
        assert(h.decisions[0].aggregate.abs == 2 && h.decisions[0].aggregate.lines == 2);
        assert(h.geometry.size() == 12);
        std::string out = Decisions(h.decisions);
        for (const auto& m : h.geometry) out += m.ab + "|" + m.line + "|" + S::Number(m.ratio) + "|" + S::Number(m.dF) + "\n";
        return out;
    };
    assert(run(1, false, false) == run(2, true, true));
    assert(run(1, false, false) == run(128, true, false));
    Harness conflict; conflict.Warm(); auto ab = AB(); auto other = ab; other.bPrice += 1;
    std::array abs{ab, other}; auto line = Line(kFirst);
    Reject([&] { conflict.engine.Add(kFirst, Bar(kFirst), 2, abs, std::span{&line, 1}); });
    Harness confirmationConflict; confirmationConflict.Warm();
    ab = AB(); other = ab; other.aConfirmationTimestamp -= 900;
    const std::array confirmations{ab, other};
    Reject([&] { confirmationConflict.engine.Add(kFirst, Bar(kFirst), 2, confirmations, std::span{&line, 1}); });
    Harness lineConfirmationConflict; lineConfirmationConflict.Warm();
    auto lineAlias = line; lineAlias.anchor1ConfirmationTimestamp -= 900;
    const std::array lineConfirmations{line, lineAlias};
    Reject([&] { lineConfirmationConflict.engine.Add(kFirst, Bar(kFirst), 2, std::span{&ab, 1}, lineConfirmations); });
    Harness across; across.Warm(); across.Geometry(); ab.bPrice += 1; line = Line(kFirst + 1);
    Reject([&] { across.engine.Add(kFirst + 1, Bar(kFirst + 1), 2, std::span{&ab, 1}, std::span{&line, 1}); });
    for (bool fib : {false, true})
    {
        Harness future; future.Warm(); ab = AB(); line = Line(kFirst);
        if (fib) ab.identity.availabilityTimestamp = Time(kFirst + 1); else line.creationTimestamp = Time(kFirst + 1);
        future.engine.Add(kFirst, Bar(kFirst), 2, std::span{&ab, 1}, std::span{&line, 1});
        assert(future.decisions[0].status == "no_geometry");
    }
    for (double invalid : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    { Harness atr; atr.Warm(); atr.Geometry(100, 100, 100.1, 100, invalid); assert(atr.decisions[0].status == "invalid_atr"); }
    S::Protocol cap; cap.maxPairsPerCandle = 1; Harness pairs(cap); pairs.Warm();
    const std::array two{AB(), AB(110, 110.1, -1)}; line = Line(kFirst);
    pairs.engine.Add(kFirst, Bar(kFirst), 2, two, std::span{&line, 1}); pairs.engine.Finalize();
    assert(pairs.decisions[0].status == "capacity_invalid" && !pairs.engine.GetAudit().geometryComplete && pairs.geometry.empty());
    assert(pairs.engine.GetAudit().eligiblePairs.scoring == 2 && pairs.engine.GetAudit().evaluatedPairs.scoring == 0);
    cap = {}; cap.maxDetectorIdentities = 1; Harness identities(cap); identities.Warm(); identities.Geometry();
    assert(identities.decisions[0].status == "capacity_invalid" && identities.engine.GetAudit().peakIdentities == 1);
    cap = {}; cap.maxEpisodeStates = 1; Harness diagnostic(cap); diagnostic.Warm(); diagnostic.Geometry();
    assert(diagnostic.decisions[0].cohort == S::Cohort::C && diagnostic.engine.GetAudit().geometryComplete);
    assert(!diagnostic.engine.GetAudit().diagnosticsComplete);
    // Episode accounting observes initialization separately from subsequent
    // complement/union entries, and partitions observations by decision range.
    Harness transitions({}, {Time(kFirst), Time(kFirst + 8)});
    while (transitions.bar < kFirst - 3) transitions.Empty();
    transitions.Geometry(); transitions.Geometry(110); transitions.Geometry();
    transitions.Geometry(110); transitions.Geometry();
    const auto& episodes = transitions.engine.GetAudit().episodeRecords;
    assert((episodes[0] == std::array<std::size_t, 3>{3, 6, 6}));
    assert((episodes[1] == std::array<std::size_t, 3>{3, 3, 3}));
    assert((episodes[2] == std::array<std::size_t, 3>{0, 3, 3}));
    cap = {}; cap.maxABInputs = 1; Harness input(cap); input.Warm();
    input.engine.Add(kFirst, Bar(kFirst), 2, two, std::span{&line, 1});
    assert(input.decisions[0].status == "capacity_invalid");
}
EA::TG4::EvaluationConfiguration SmallConfig(EA::TG4::EvaluationConfiguration c)
{
    c.geometry.maxConfirmedFractalsPerKind = 8;
    c.geometry.maxFractalAnchorLookbackBars = 128; c.geometry.maxCandidateAgeBars = 64;
    c.fibonacci.maxABAgeBars = 128; c.fibonacci.maxConfirmedFractalsPerKind = 16;
    c.fibonacci = S::HistoricalEvaluator::ResearchFibonacci(c.fibonacci);
    return c;
}
std::vector<Candle> RandomCandles(std::size_t count)
{
    std::mt19937_64 random(0x54473450554c5345ULL); std::uniform_int_distribution<int> delta(-6, 6), wick(1, 8);
    std::vector<Candle> candles; double price = 1.2;
    for (std::size_t i = 0; i < count; ++i)
    {
        double open = price; price += delta(random) * .0001;
        double high = std::max(open, price) + wick(random) * .0001;
        double low = std::min(open, price) - wick(random) * .0001;
        candles.push_back({Time(i), open, high, low, price});
    }
    return candles;
}
void ReplayPartitionsPrefix(const EA::TG4::EvaluationConfiguration& original)
{
    const auto c = SmallConfig(original); auto candles = RandomCandles(2112);
    std::vector<S::Decision> first, second, prefix;
    const auto make = [&](auto& rows)
    {
        S::Protocol p; p.maxABAgeBars = c.fibonacci.maxABAgeBars; p.maxLineAgeBars = c.geometry.maxCandidateAgeBars;
        return S::HistoricalEvaluator(p, EA::TG1B::CalibrationConfiguration(c.referenceBarScale), c.fibonacci, c.geometry,
            c.behavior, EA::TG4::ConfigurationFingerprint(c), {[&rows](const auto& d) { rows.push_back(d); }, [](const auto&) {}, {}, {}, {}},
            S::ScoreRange{Time(2048), Time(2112)});
    };
    auto a = make(first); for (const auto& bar : candles) a.AddCompletedBar(bar); a.Finalize();
    auto b = make(second); for (const auto& bar : candles) b.AddCompletedBar(bar); b.Finalize();
    assert(Decisions(first) == Decisions(second) && a.Engine().CandleDigest() == b.Engine().CandleDigest());
    auto shortReplay = make(prefix); for (std::size_t i = 0; i < 2080; ++i) shortReplay.AddCompletedBar(candles[i]); shortReplay.Finalize();
    std::vector<S::Decision> completedPrefix;
    for (const auto& d : first) if (d.timestamp < Time(2080)) completedPrefix.push_back(d);
    std::vector<S::Decision> beforeFinal;
    for (const auto& d : prefix) if (d.timestamp < Time(2080)) beforeFinal.push_back(d);
    assert(Decisions(completedPrefix) == Decisions(beforeFinal));
    auto changed = candles; changed.back().close += .0001; changed.back().high += .0001;
    std::vector<S::Decision> modified; auto future = make(modified);
    for (const auto& bar : changed) future.AddCompletedBar(bar); future.Finalize();
    std::vector<S::Decision> priorOriginal, priorModified;
    for (const auto& d : first) if (d.timestamp < changed.back().timestamp) priorOriginal.push_back(d);
    for (const auto& d : modified) if (d.timestamp < changed.back().timestamp) priorModified.push_back(d);
    assert(Decisions(priorOriginal) == Decisions(priorModified)); // Future revision cannot affect prior decisions.
    assert(a.Engine().CandleDigest() != future.Engine().CandleDigest());
    assert(a.Engine().GetAudit().geometryComplete);
    auto limited = c; limited.geometry.maxCandidates = 1; limited.fibonacci.maxActiveABStructures = 1;
    S::Protocol lp; lp.maxABAgeBars = limited.fibonacci.maxABAgeBars; lp.maxLineAgeBars = limited.geometry.maxCandidateAgeBars;
    S::HistoricalEvaluator sourceLimited(lp, EA::TG1B::CalibrationConfiguration(limited.referenceBarScale), limited.fibonacci,
        limited.geometry, limited.behavior, "limited-source", {[](const auto&) {}, [](const auto&) {}, {}, {}, {}},
        S::ScoreRange{Time(2048), Time(2112)});
    for (const auto& bar : candles) sourceLimited.AddCompletedBar(bar);
    sourceLimited.Finalize();
    assert(!sourceLimited.Engine().GetAudit().geometryComplete);
    assert(sourceLimited.Engine().GetAudit().diagnostics.at("source_ab_capacity_evictions") > 0);
    assert(sourceLimited.Engine().GetAudit().diagnostics.at("source_line_capacity_evictions") > 0);
    S::Protocol cp; cp.partition = S::Partition::Calibration;
    cp.maxABAgeBars = c.fibonacci.maxABAgeBars; cp.maxLineAgeBars = c.geometry.maxCandidateAgeBars;
    std::vector<S::Decision> calibration;
    S::HistoricalEvaluator continuous(cp, EA::TG1B::CalibrationConfiguration(c.referenceBarScale), c.fibonacci, c.geometry, c.behavior,
        EA::TG4::ConfigurationFingerprint(c), {[&](const auto& d) { calibration.push_back(d); }, [](const auto&) {}, {}, {}, {}},
        S::ScoreRange{S::kCalibrationStart, S::kCalibrationStart + 7200});
    std::vector<Candle> data(candles.begin(), candles.begin() + 2056);
    for (auto& bar : data) continuous.AddCompletedBar(bar);
    for (std::size_t i = 0; i < 8; ++i)
    {
        Candle bar{S::kCalibrationStart + static_cast<std::int64_t>(i) * 900, 1.2, 1.201, 1.199, 1.2};
        continuous.AddCompletedBar(bar); data.push_back(bar);
    }
    assert(calibration.size() == 2 && calibration[0].status == "embargo");
    assert(calibration[1].status != "embargo" && calibration[1].status != "warmup");
    assert(calibration[1].status == "partition_purged" || calibration[1].status == "no_geometry");
    assert(continuous.Engine().GetAudit().prefixCandles == 2056);
    const auto batch = EA::TG3::CausalFibonacciConfluenceIntegration::FromHistorical(data,
        EA::TG1B::CalibrationConfiguration(c.referenceBarScale), c.fibonacci, c.geometry, c.behavior, {"AUDCAD", "15m"});
    assert(continuous.Integration().FormatSummaryDiagnostic() == batch.FormatSummaryDiagnostic());
    assert(continuous.Integration().Behavior().Classification().Geometry().CurrentAtr() == batch.Behavior().Classification().Geometry().CurrentAtr());
    const auto count = continuous.Integration().Behavior().Classification().Geometry().CompletedBarCount();
    Candle heldout{S::kFinalStart, 1.2, 1.201, 1.199, 1.2}; Reject([&] { continuous.AddCompletedBar(heldout); });
    assert(continuous.Integration().Behavior().Classification().Geometry().CompletedBarCount() == count);
    Reject([] { S::Protocol p; p.partition = S::Partition::Final2025; Harness bad(p); });
    std::cout << "V2_REPLAY_EQUIVALENCE bars=" << candles.size() << " scheduled=" << first.size()
              << " classified=" << a.Engine().GetAudit().classified << " peak_identities=" << a.Engine().GetAudit().peakIdentities << '\n';
}
std::string Text(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary); assert(file);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
std::string LegacyMeasurement(const S::GeometryMeasurement& m)
{
    // Frozen pre-revision CSV serialization, independent of the gzip writer.
    return std::to_string(m.timestamp) + "," + m.ab + "," + m.line + "," + S::Number(m.ratio) + "," + S::Number(m.fibPrice)
        + "," + S::Number(m.linePrice) + "," + S::Number(m.dF) + "," + S::Number(m.dL) + "," + S::Number(m.dS) + "," + S::Name(m.cohort);
}
void Artifacts(const EA::TG4::EvaluationConfiguration& config, const std::filesystem::path& temp)
{
    // All fixture files are inside this script's own mktemp directory.
    // No C++ cleanup/deletion or historical input is used.
    const auto write = [&](const std::filesystem::path& p, std::string_view content)
        { std::ofstream out(p, std::ios::binary); out.exceptions(std::ios::failbit | std::ios::badbit); out << content; out.close(); };
    write(temp / "fixture-source", "synthetic sources\n"); write(temp / "fixture-executable", "synthetic executable\n");
    write(temp / "fixture-config", "synthetic configuration\n"); write(temp / "fixture-archive", "synthetic archive fixture\n");
    S::Provenance provenance;
    provenance.gitCommit = std::string(40, 'a'); provenance.compiler = "synthetic compiler fixture"; provenance.libraryVersion = "synthetic library fixture";
    provenance.sourceManifest = temp / "fixture-source"; provenance.executable = temp / "fixture-executable";
    provenance.configurationFile = temp / "fixture-config"; provenance.sourceArchive = temp / "fixture-archive";
    provenance.sourceSHA256 = S::FileHash(provenance.sourceManifest); provenance.executableSHA256 = S::FileHash(provenance.executable);
    provenance.configurationFileSHA256 = S::FileHash(provenance.configurationFile); provenance.preflightRows = kFirst + 5;
    auto researchConfig = config; researchConfig.fibonacci = S::HistoricalEvaluator::ResearchFibonacci(config.fibonacci);
    const std::string header = "bar_start_utc_epoch,ab_natural_key,line_natural_key,ratio,fib_price,line_price,dF,dL,dS,cohort";
    std::string legacyGeometry, legacyEpisodes;
    const auto produce = [&](const std::filesystem::path& path, int failure, S::EpisodePublication tier = S::EpisodePublication::Summary,
                             bool saturated = false, std::size_t batch = 128, bool reverse = false, bool alias = false)
    {
        bool armed = false; unsigned finalizations = 0;
        const auto hook = [&](S::CompressionOperation operation)
        {
            if (!armed) return;
            if ((failure == 6 && operation == S::CompressionOperation::Compress) ||
                (failure == 7 && operation == S::CompressionOperation::Write) ||
                (failure == 8 && operation == S::CompressionOperation::Finalize))
                throw std::runtime_error("synthetic codec/write/finalization failure");
            if ((failure == 9 || failure == 10) && operation == S::CompressionOperation::Finalize && ++finalizations == 2)
            {
                // Geometry was already closed. Corrupt its CRC before publication
                // verification; the episode stream is still being finalized.
                if (failure == 9)
                {
                    std::fstream file(path / "geometry_measurements.csv.gz", std::ios::binary | std::ios::in | std::ios::out);
                    file.seekg(-8, std::ios::end); char byte = 0; file.get(byte); byte ^= 1;
                    file.seekp(-8, std::ios::end); file.put(byte); file.close();
                }
                else
                {
                    // A valid gzip trailer is insufficient: decoded hash/row
                    // mismatches must also prevent COMPLETE.
                    S::CompressedCsv replacement(path / "geometry_measurements.csv.gz");
                    replacement.Append("header\nwrong synthetic geometry\n"); replacement.Finish();
                }
            }
        };
        const auto operation = [&]
        {
            S::ArtifactWriter writer(path, tier, hook); S::Protocol p; p.lineBatchSize = batch;
            if (failure == 3) p.maxDetectorIdentities = 1;
            if (saturated) p.maxEpisodeStates = 1;
            auto sinks = writer.GetSinks();
            std::string expectedGeometry = header + "\n", expectedEpisodes = header + ",episode_sequence,role\n";
            const auto geometrySink = sinks.geometry; const auto episodeSink = sinks.episode;
            sinks.geometry = [&](const S::GeometryMeasurement& m) { expectedGeometry += LegacyMeasurement(m) + "\n"; geometrySink(m); };
            sinks.episode = [&](const S::Episode& e)
                { expectedEpisodes += LegacyMeasurement(e.measurement) + "," + std::to_string(e.sequence) + ",DIAGNOSTIC_ONLY\n"; episodeSink(e); };
            S::EventEngine engine(p, "fixture", sinks, S::ScoreRange{Time(kFirst), Time(kFirst + 8)});
            for (std::size_t i = 0; i < kFirst; ++i)
            {
                // Saturate diagnostically BEFORE scoring, without restricting
                // primary geometry. Retained input maxima occur in the prefix.
                if (saturated && i == kFirst - 3)
                {
                    const std::array abs{AB(), AB(110, 110.1, -1), AB(120, 120.1, -2)};
                    const std::array lines{Line(i), Line(i, 115, -1), Line(i, 120, -2)};
                    engine.Add(i, Bar(i), 2, abs, lines);
                }
                else engine.Add(i, Bar(i), 2, {}, {});
            }
            std::vector abs{AB(), AB(110, 110.1, -1)};
            std::vector lines{Line(kFirst), Line(kFirst, 115, -1)};
            if (!saturated) { abs.resize(1); lines.resize(1); }
            if (alias) { auto a = abs[0]; --a.identity.aBar; abs.push_back(a); auto l = lines[0]; --l.anchor1Bar; lines.push_back(l); }
            if (reverse) { std::reverse(abs.begin(), abs.end()); std::reverse(lines.begin(), lines.end()); }
            armed = true;
            engine.Add(kFirst, Bar(kFirst), 2, abs, lines);
            for (std::size_t i = kFirst + 1; i <= kFirst + 4; ++i) engine.Add(i, Bar(i), 2, {}, {});
            engine.Finalize(); auto meta = provenance;
            if (saturated)
            {
                const auto& a = engine.GetAudit();
                assert(!a.diagnosticsComplete && a.geometryComplete && a.firstEpisodeSaturationBar == kFirst - 3);
                assert(a.firstEpisodeSaturationTimestamp == Time(kFirst - 3));
                assert(a.skippedEpisodeOpportunities[0] == 38 && a.skippedEpisodeOpportunities[1] == 26 && a.skippedEpisodeOpportunities[2] == 12);
                assert(a.episodeRecords[0][0] == 1 && a.episodeRecords[1][0] == 1 && a.episodeRecords[2][0] == 0);
                assert(a.retainedAB.prefix == 3 && a.retainedLines.prefix == 3);
                assert(a.eligiblePairs.prefix == 9 && a.evaluatedPairs.prefix == 9);
                assert(a.identityLedger.prefix == 6 && a.episodeLedger.prefix == 1 && a.episodeLedger.scoring == 1);
                assert(a.identityLedger.full == 6 && a.eligiblePairs.full == 9 && a.retainedAB.full == 3);
            }
            if (failure == 1) meta.sourceSHA256 = std::string(64, 'f');
            if (failure == 2) std::filesystem::create_directory(path / "provenance_manifest.json");
            if (failure == 4) std::filesystem::create_directory(path / "checksums.sha256");
            auto effective = researchConfig;
            if (failure == 5) effective.fibonacci.retracementRatios = {0.5};
            writer.Complete(engine, effective, meta);
            S::ValidateArtifactPublication(path);
            std::string decodedGeometry;
            const auto geometryInfo = S::InspectGzip(path / "geometry_measurements.csv.gz", [&](auto bytes) { decodedGeometry += bytes; });
            assert(decodedGeometry == expectedGeometry && geometryInfo.sha256 == S::Hash(expectedGeometry));
            assert(!std::filesystem::exists(path / "geometry_measurements.csv"));
            if (tier == S::EpisodePublication::Full)
            {
                std::string decodedEpisodes;
                S::InspectGzip(path / "interaction_episodes.csv.gz", [&](auto bytes) { decodedEpisodes += bytes; });
                assert(decodedEpisodes == expectedEpisodes);
            }
            else assert(!std::filesystem::exists(path / "interaction_episodes.csv.gz"));
            if (!saturated) { legacyGeometry = expectedGeometry; legacyEpisodes = expectedEpisodes; }
        };
        if (failure) Reject(operation); else operation();
        assert(std::filesystem::exists(path / "COMPLETE") == (failure == 0));
    };
    produce(temp / "first", 0); produce(temp / "second", 0); produce(temp / "full", 0, S::EpisodePublication::Full);
    for (const auto& file : std::filesystem::directory_iterator(temp / "first"))
        assert(S::FileHash(file.path()) == S::FileHash(temp / "second" / file.path().filename()));
    for (const char* file : {"decisions.csv", "outcomes.csv", "geometry_measurements.csv.gz", "effective_configuration.txt"})
        assert(Text(temp / "first" / file) == Text(temp / "full" / file));
    for (int i = 1; i <= 10; ++i)
        produce(temp / ("failure" + std::to_string(i)), i, i >= 9 ? S::EpisodePublication::Full : S::EpisodePublication::Summary);
    assert(Text(temp / "failure3" / "decisions.csv").find("capacity_invalid") != std::string::npos);
    assert(std::filesystem::exists(temp / "failure3" / "INVALID"));
    produce(temp / "saturated-summary", 0, S::EpisodePublication::Summary, true);
    produce(temp / "saturated-full", 0, S::EpisodePublication::Full, true);
    produce(temp / "saturated-permuted", 0, S::EpisodePublication::Full, true, 1, true, true);
    for (const char* file : {"decisions.csv", "outcomes.csv", "geometry_measurements.csv.gz"})
    {
        assert(Text(temp / "saturated-summary" / file) == Text(temp / "saturated-full" / file));
        assert(Text(temp / "saturated-full" / file) == Text(temp / "saturated-permuted" / file));
    }
    assert(Text(temp / "saturated-summary" / "episode_summary.csv").find("tracking_complete,0,OBSERVED") != std::string::npos);
    assert(Text(temp / "saturated-summary" / "episode_summary.csv").find("publication_detail_omitted,1,INTENTIONAL") != std::string::npos);
    assert(Text(temp / "saturated-full" / "episode_summary.csv").find("publication_detail_omitted,0,INTENTIONAL") != std::string::npos);
    // Synthesize an original v2 publication, not a rewritten historical artifact.
    const auto legacy = temp / "legacy-v2"; std::filesystem::create_directory(legacy);
    for (const char* file : {"decisions.csv", "outcomes.csv", "coverage.csv", "exclusions.csv", "gaps.csv", "detector_configuration.json",
                           "effective_configuration.txt", "source_manifest.sha256", "source_snapshot.tar.gz", "detector_source_configuration.conf"})
        std::filesystem::copy_file(temp / "first" / file, legacy / file);
    write(legacy / "geometry_measurements.csv", legacyGeometry); write(legacy / "interaction_episodes.csv", legacyEpisodes);
    std::istringstream revisedProtocol(Text(temp / "first" / "protocol.json")); std::string protocol, row;
    while (std::getline(revisedProtocol, row))
        if (row.find("\"artifact_format\"") == std::string::npos && row.find("\"episode_publication\"") == std::string::npos &&
            row.find("\"episode_detail_omitted\"") == std::string::npos) protocol += row + "\n";
    write(legacy / "protocol.json", protocol);
    const auto manifest = Text(temp / "first" / "provenance_manifest.json");
    write(legacy / "provenance_manifest.json", "{\n" + manifest.substr(manifest.find("\"schema\":\"fib-trendline-provenance-v2\"")));
    std::map<std::string, std::string> hashes;
    for (const auto& file : std::filesystem::directory_iterator(legacy)) hashes[file.path().filename().string()] = S::FileHash(file.path());
    std::string checksums; for (const auto& [file, hash] : hashes) checksums += hash + "  " + file + "\n";
    write(legacy / "checksums.sha256", checksums);
    write(legacy / "COMPLETE", "processing=COMPLETE\nchecksums_sha256=" + S::Hash(checksums) + "\n");
    S::ValidateArtifactPublication(legacy);
    // Corruption after publication is detected by both the byte-hash chain and
    // strict gzip validation. Existing artifacts are never involved.
    const auto corrupt = temp / "corrupt"; std::filesystem::copy(temp / "first", corrupt, std::filesystem::copy_options::recursive);
    std::string compressed = Text(corrupt / "geometry_measurements.csv.gz");
    compressed[compressed.size() - 8] ^= 1; write(corrupt / "geometry_measurements.csv.gz", compressed);
    Reject([&] { S::InspectGzip(corrupt / "geometry_measurements.csv.gz"); });
    Reject([&] { S::ValidateArtifactPublication(corrupt); });
    const auto truncated = temp / "truncated.gz"; write(truncated, compressed.substr(0, 12)); Reject([&] { S::InspectGzip(truncated); });
    write(temp / "trailing.gz", Text(temp / "first" / "geometry_measurements.csv.gz") + "garbage");
    Reject([&] { S::InspectGzip(temp / "trailing.gz"); });
    Reject([&] { S::ArtifactWriter existing(temp / "first"); });
    assert(S::Hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    std::cout << "SCALABLE_ARTIFACTS primary_and_synthetic_outcome_equivalence=true legacy_decoded_bytes=true deterministic_bytes=true publication_failures=10 legacy_v2_validation=true\n";
}
void StreamingMemory(const std::filesystem::path& temp)
{
    // Larger input than the codec buffers; retain only one fixed row and hashes.
    // Different producer chunking cannot introduce compression flush boundaries.
    const std::string row = std::string(1023, 'x') + "\n";
    S::Sha256 expected; expected.Add("header\n");
    S::CompressedCsv first(temp / "stream-one.gz"), second(temp / "stream-chunks.gz");
    first.Append("header\n"); second.Append("header\n");
    constexpr std::size_t rows = 32768;
    for (std::size_t i = 0; i < rows; ++i)
    {
        first.Append(row); second.Append(std::string_view(row).substr(0, 17)); second.Append(std::string_view(row).substr(17));
        expected.Add(row);
    }
    const auto a = first.Finish(), b = second.Finish();
    assert(a.sha256 == expected.Digest() && a.sha256 == b.sha256 && a.bytes == rows * row.size() + 7 && a.rows == rows);
    assert(S::FileHash(temp / "stream-one.gz") == S::FileHash(temp / "stream-chunks.gz"));
    assert(S::InspectGzip(temp / "stream-one.gz").sha256 == a.sha256);
    assert(S::CompressedCsv::kBufferBytes == 65536);
    Reject([&] { first.Append("after-finalization"); }); Reject([&] { first.Finish(); });
    // Exercise output-buffer draining with low-compressibility input too.
    S::CompressedCsv noisy(temp / "stream-noisy.gz"); S::Sha256 noiseHash;
    std::mt19937 generator(12345); std::string block(65536, ' '); std::size_t noiseBytes = 0;
    for (int i = 0; i < 64; ++i)
    {
        for (auto& byte : block) byte = static_cast<char>(32 + generator() % 95);
        for (std::size_t j = 1023; j < block.size(); j += 1024) block[j] = '\n';
        noisy.Append(block); noiseHash.Add(block); noiseBytes += block.size();
    }
    const auto noise = noisy.Finish();
    assert(noise.bytes == noiseBytes && noise.sha256 == noiseHash.Digest());
    assert(S::InspectGzip(temp / "stream-noisy.gz").sha256 == noise.sha256);
    std::cout << "STREAMING_COMPRESSION synthetic_bytes=" << a.bytes << " decoded_rows=" << a.rows
              << " output_buffer_bytes=" << S::CompressedCsv::kBufferBytes << " producer_chunk_invariance=true\n";
}

void SnapshotLoading(const EA::TG4::EvaluationConfiguration& original, const std::filesystem::path& temp)
{
    auto data = RandomCandles(2112);
    for (std::size_t i = 1056; i < data.size(); ++i) data[i].timestamp += 900; // Preserve a real gap.
    S::ResearchCandleSnapshot snapshot(temp, S::kOrigin, Time(2120), data.size(), {});
    S::ResearchCandleSnapshot repeat(temp, S::kOrigin, Time(2120), data.size(), {});
    for (const auto& c : data) { snapshot.Append(c); repeat.Append(c); }
    snapshot.Seal(); repeat.Seal();
    assert(S::FileHash(snapshot.Path()) == S::FileHash(repeat.Path()));
    std::size_t at = 0;
    snapshot.Replay([&](const Candle& c)
    {
        const auto& expected = data.at(at++);
        assert(c.timestamp == expected.timestamp);
        for (const auto [a,b] : {std::pair{c.open,expected.open}, {c.high,expected.high},
             {c.low,expected.low}, {c.close,expected.close}})
            assert(std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b));
    });
    assert(at == data.size());
    Reject([&] { snapshot.Append(data.back()); }); Reject([&] { snapshot.Seal(); });
    {
        S::ResearchCandleSnapshot partial(temp, S::kOrigin, Time(2120), 2, {});
        partial.Append(data.front()); Reject([&] { partial.Seal(); });
        Reject([&] { partial.Append(data.front()); });
        Reject([&] { partial.Append(Bar(2120)); });
        Reject([&] { partial.Replay([](const auto&) {}); });
    }
    auto limited = S::ResearchCandleSnapshot::Budgets{}; limited.maxRows = 1;
    Reject([&] { S::ResearchCandleSnapshot bad(temp, S::kOrigin, Time(2120), 2, limited); });
    limited = {}; limited.maxBytes = 100;
    Reject([&] { S::ResearchCandleSnapshot bad(temp, S::kOrigin, Time(2120), 2, limited); });
    limited = {}; limited.reserveDiskBytes = std::numeric_limits<std::uint64_t>::max();
    Reject([&] { S::ResearchCandleSnapshot bad(temp, S::kOrigin, Time(2120), 2, limited); });
    const auto config = SmallConfig(original);
    S::Provenance provenance;
    provenance.gitCommit = std::string(40, 'a'); provenance.compiler = "synthetic compiler fixture"; provenance.libraryVersion = "synthetic library fixture";
    provenance.sourceManifest = temp / "fixture-source"; provenance.executable = temp / "fixture-executable";
    provenance.configurationFile = temp / "fixture-config"; provenance.sourceArchive = temp / "fixture-archive";
    provenance.sourceSHA256 = S::FileHash(provenance.sourceManifest); provenance.executableSHA256 = S::FileHash(provenance.executable);
    provenance.configurationFileSHA256 = S::FileHash(provenance.configurationFile); provenance.preflightRows = data.size();
    for (bool staged : {false, true})
    {
        S::ArtifactWriter writer(temp / (staged ? "snapshot-staged" : "snapshot-direct"));
        S::Protocol p; p.maxABAgeBars = config.fibonacci.maxABAgeBars; p.maxLineAgeBars = config.geometry.maxCandidateAgeBars;
        S::HistoricalEvaluator evaluator(p, EA::TG1B::CalibrationConfiguration(config.referenceBarScale), config.fibonacci,
            config.geometry, config.behavior, EA::TG4::ConfigurationFingerprint(config), writer.GetSinks(), S::ScoreRange{Time(2048),Time(2120)});
        if (staged) snapshot.Replay([&](const auto& c) { evaluator.AddCompletedBar(c); });
        else for (const auto& c : data) evaluator.AddCompletedBar(c);
        evaluator.Finalize(); assert(evaluator.Engine().CandleDigest() == snapshot.CanonicalDigest());
        if (staged)
        {
            provenance.candleSnapshotSHA256 = S::FileHash(snapshot.Path());
            provenance.candleSnapshotEncoding = S::ResearchCandleSnapshot::kEncoding;
            provenance.candleSnapshotBytes = snapshot.Bytes();
        }
        writer.Complete(evaluator.Engine(),config,provenance);
        S::ValidateArtifactPublication(temp / (staged ? "snapshot-staged" : "snapshot-direct"));
    }
    for (const char* name : {"decisions.csv","outcomes.csv","geometry_measurements.csv.gz","exclusions.csv","gaps.csv","protocol.json"})
        assert(S::FileHash(temp / "snapshot-staged" / name) == S::FileHash(temp / "snapshot-direct" / name));
    // Corruption must be detected BEFORE a consumer is invoked, including a
    // bit-exact payload revision that leaves ordering/OHLC apparently plausible.
    assert(chmod(repeat.Path().c_str(), S_IRUSR | S_IWUSR) == 0);
    {
        std::fstream f(repeat.Path(),std::ios::in|std::ios::out|std::ios::binary);
        f.seekg(47); char byte=0; f.get(byte); byte ^= 1; f.seekp(47); f.put(byte);
    }
    bool consumed=false; Reject([&] { repeat.Replay([&](const auto&) { consumed=true; }); }); assert(!consumed);
    std::filesystem::resize_file(repeat.Path(),repeat.Bytes()-1); Reject([&] { repeat.Validate(); });
    {
        S::ResearchCandleSnapshot large(temp,S::kOrigin,Time(50001),50000,{});
        for (std::size_t i=0;i<50000;++i) large.Append(Bar(i));
        large.Seal(); assert(large.Bytes()==50000*40+96); static_assert(sizeof(large)<4096);
    }
    std::cout << "RESEARCH_SNAPSHOT bit_exact=true deterministic=true primary_bytes_equal=true gap_preserved=true corruption_before_replay=true truncation=true budgets=true streamed_rows=50000\n";
}
}
int main(int argc, char** argv)
{
    assert(argc == 3); const auto config = EA::TG4::LoadConfigurationFile(argv[1]);
    Cohorts(); ScheduleCoverageOutcomes(); InvarianceAndCapacity(); ReplayPartitionsPrefix(config); Artifacts(config, argv[2]); StreamingMemory(argv[2]); SnapshotLoading(config, argv[2]);
    std::cout << "CausalFibonacciTrendlineHistoricalEvaluationTests v2 passed\n";
}
