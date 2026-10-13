#include "CausalABCapacityCounterfactual.hpp"
#include "CausalFibonacciTrendlineHistoricalArtifact.hpp"
#include <fstream>
#include <iostream>
namespace A = EA::FibonacciResearch::ABCapacity;
namespace H = EA::FibonacciResearch::HistoryShadow;
namespace S = EA::FibonacciResearch::TrendlineHistorical;
namespace F = EA::TG3;
namespace G = EA::TG1A;
void Check(bool x, const char *msg)
{
    if (!x)
        throw std::runtime_error(msg);
}
std::int64_t Time(std::size_t i) { return S::kOrigin + static_cast<std::int64_t>(i) * 900; }
void Write(const std::filesystem::path &p, std::string_view s)
{
    std::ofstream f(p);
    f << s;
    f.close();
    Check(bool(f), "fixture write");
}
std::string Read(const std::filesystem::path &p)
{
    std::ifstream f(p);
    return {std::istreambuf_iterator<char>(f), {}};
}
template <class Fn> void Reject(Fn fn)
{
    bool bad = false;
    try
    {
        fn();
    }
    catch (...)
    {
        bad = true;
    }
    Check(bad, "expected rejection");
}
S::Provenance Provenance(const std::filesystem::path &root)
{
    Write(root / "source", "synthetic source manifest\n");
    Write(root / "archive", "synthetic source archive\n");
    Write(root / "executable", "synthetic executable\n");
    Write(root / "config", "synthetic configuration\n");
    S::Provenance p;
    p.gitCommit = std::string(40, 'a');
    p.sourceManifest = root / "source";
    p.sourceArchive = root / "archive";
    p.executable = root / "executable";
    p.configurationFile = root / "config";
    p.sourceSHA256 = S::FileHash(p.sourceManifest);
    p.executableSHA256 = S::FileHash(p.executable);
    p.configurationFileSHA256 = S::FileHash(p.configurationFile);
    p.compiler = "synthetic standalone fixture";
    p.libraryVersion = "synthetic";
    return p;
}
G::TrendLineCandidate Line(double price, std::size_t bar)
{
    G::TrendLineCandidate l;
    l.direction = G::TrendLineDirection::UTL;
    l.anchor1Bar = bar - 10;
    l.anchor2Bar = bar - 7;
    l.anchor1Timestamp = Time(bar - 10);
    l.anchor2Timestamp = Time(bar - 7);
    l.anchor1ConfirmationBar = bar - 8;
    l.anchor2ConfirmationBar = bar - 5;
    l.anchor1ConfirmationTimestamp = Time(bar - 8);
    l.anchor2ConfirmationTimestamp = Time(bar - 5);
    l.creationBar = bar - 5;
    l.creationTimestamp = Time(bar - 5);
    l.anchorSeparationBars = 3;
    l.rawPriceSlopePerBar = .01;
    l.anchor1Price = price - .1;
    l.anchor2Price = l.anchor1Price + .03;
    l.currentBar = bar;
    l.currentTimestamp = Time(bar);
    return l;
}
void GeometryCases(const EA::TG4::EvaluationConfiguration &config)
{
    // Native TG3 capacity=512 and history=128. These are causal confirmed-fractal
    // interface fixtures, not a claim of TG1A candle-detection parity.
    auto fixture = [&](double low, double firstHigh, bool redundant)
    {
        F::FibonacciConfluenceTracker f(config.fibonacci);
        F::ABStructure victim;
        for (std::size_t i = 0; i <= 3582; ++i)
        {
            f.Advance(i, Time(i));
            if (i == 2042)
                f.ObserveConfirmedFractals({{G::FractalKind::Low, 2040, Time(2040), low, i, Time(i)}});
            if (i >= 2046 && (i - 2046) % 3 == 0)
            {
                double high = i == 2046 ? firstHigh : (redundant && i == 2049 ? 101 : 107);
                f.ObserveConfirmedFractals({{G::FractalKind::High, i - 2, Time(i - 2), high, i, Time(i)}});
                if (i == 2046)
                    victim = f.ABStructures().at(0);
            }
        }
        Check(f.ABCapacityEvictions() == 1 && f.TotalABStructuresCreated() == 513 &&
                  f.ABStructures().size() == 512,
              "native 512 eviction");
        return std::pair{f.ABStructures(), victim};
    };
    auto evaluate = [&](const auto &abs, double linePrice, std::size_t bar = 3583)
    {
        auto l = Line(linePrice, bar);
        return H::EvaluateDecisionGeometry(bar, {Time(bar), 100, 101, 99, 100}, 1, abs, std::span{&l, 1},
                                           {8192, 4096, 16 * 1024 * 1024});
    };
    auto [far, victimFar] = fixture(90, 92, false);
    auto restoredFar = far;
    restoredFar.push_back(victimFar);
    Check(evaluate(far, 102).cohort == "D" && evaluate(restoredFar, 102).cohort == "D",
          "B: surviving victim no qualifying geometry");
    auto [retained, victim] = fixture(99, 101, false);
    auto restored = retained;
    restored.push_back(victim);
    auto before = evaluate(retained, 102), after = evaluate(restored, 102);
    Check(before.cohort == "D" && after.cohort == "A" && before.fibNear == false && after.fibNear == true,
          "C/E: Ffib and cohort effect");
    before = evaluate(retained, 100.1);
    after = evaluate(restored, 100.1);
    Check(before.cohort == "B" && after.cohort == "C" && before.intersection == false &&
              after.intersection == true,
          "D/E: conjunction/cohort effect");
    auto [near, victimNear] = fixture(99, 101, true);
    auto restoredNear = near;
    restoredNear.push_back(victimNear);
    before = evaluate(near, 100.1);
    after = evaluate(restoredNear, 100.1);
    Check(before.cohort == "C" && after.cohort == "C" && after.abGeometry.at(S::ABKey(victimNear)).near,
          "F: redundant qualifying victim");
    std::array onlyVictim{victim};
    Check(evaluate(onlyVictim, 100.1, 4094).eligibleAB == 1, "G: age 2048 inclusive");
    Check(evaluate(onlyVictim, 100.1, 4095).eligibleAB == 0, "H: age 2049 expired");
    Check(evaluate(restored, 100.1).eligibleAB == 513, "restored inputs above frozen source capacity");
    // Changing only AB retention cannot alter births, including when history
    // pressure suppresses an otherwise usable anchor. Compare real trackers.
    auto c = config.fibonacci;
    c.maxActiveABStructures = 8192;
    F::FibonacciConfluenceTracker native(config.fibonacci), retentionOnly(c);
    std::size_t births = 0;
    for (std::size_t i = 0; i < 2600; ++i)
    {
        native.Advance(i, Time(i));
        retentionOnly.Advance(i, Time(i));
        std::vector<G::ConfirmedFractal> fs;
        if (i >= 2 && i % 3 == 2)
        {
            double base = 100 + static_cast<double>(i % 17) * .01;
            fs = {{G::FractalKind::High, i - 2, Time(i - 2), base + 7, i, Time(i)},
                  {G::FractalKind::Low, i - 2, Time(i - 2), base, i, Time(i)}};
        }
        auto x = native.ObserveConfirmedFractals(fs), y = retentionOnly.ObserveConfirmedFractals(fs);
        Check(x.size() == y.size(), "birth cardinality independent of retention");
        for (std::size_t j = 0; j < x.size(); ++j)
            Check(H::CandidateIdentityReconciliation::Native(x[j]) ==
                      H::CandidateIdentityReconciliation::Native(y[j]),
                  "exact birth order independent of retention");
        births += x.size();
    }
    Check(native.ABCapacityEvictions() > 0 && births > 512 &&
              native.TotalABStructuresCreated() == retentionOnly.TotalABStructuresCreated(),
          "both directions/history pressure independence");
    F::FibonacciConfluenceTracker suppressed(config.fibonacci);
    std::vector<G::ConfirmedFractal> completeHistory;
    for (std::size_t bar = 0; bar <= 392; ++bar)
    {
        suppressed.Advance(bar, Time(bar));
        if (bar >= 2 && (bar - 2) % 3 == 0 && bar <= 386)
        {
            G::ConfirmedFractal f{G::FractalKind::Low,   bar - 2, Time(bar - 2),
                                  bar == 2 ? 90. : 200., bar,     Time(bar)};
            completeHistory.push_back(f);
            suppressed.ObserveConfirmedFractals({f});
        }
        if (bar == 392)
        {
            G::ConfirmedFractal high{G::FractalKind::High, 390, Time(390), 100, bar, Time(bar)};
            Check(H::CausalRules::SelectTG3Anchor(completeHistory, high).has_value(),
                  "historical suppressed anchor exists");
            Check(suppressed.ObserveConfirmedFractals({high}).empty(),
                  "K: frozen history suppression unchanged");
        }
    }
    std::cout << "AB_GEOMETRY_CASES D_to_A=true B_to_C=true redundant_C=true age_2048_2049=true native_512=true "
                 "birth_order_independence=true history_128_preserved=true\n";
}
G::Candle Candle(std::size_t i, bool dense, bool expired)
{
    if (!dense || (expired && i >= 1000) || (!expired && i >= 2200))
        return {Time(i), 101.4, 107, 100, 101.4};
    // Both fractal kinds pressure the real 512 AB pool. Declining lows and
    // equal highs avoid a large irrelevant trendline universe in the prefix.
    // One late rising low introduces real UTL geometry after the primary stop.
    const double base = !expired && i == 2193 ? 100. : 100 - static_cast<double>(i) * .0001;
    const bool center = i % 3 == 0;
    return {Time(i), 101.4, center ? 107. : 106., base + (center ? 0 : .5), 101.4};
}
G::Candle PositiveCandle(std::size_t i)
{
    if (i < 310)
    {
        return {Time(i), 100, i == 301 ? 101. : 100.5, i == 298 ? 99. : 99.5, 100};
    }
    if (i < 1950)
    {
        const bool center = i % 3 == 0;
        return {Time(i), 104, center ? 107. : 106., center ? 101. : 101.5, 104};
    }
    double low = i == 2001 ? 99.99999 : (i == 2004 ? 99.9999901 : 100.);
    return {Time(i), 100, 100.0001, low, 100};
}
struct Result
{
    A::Summary summary;
    bool parentInvalid = false, complete = false;
};
Result Replay(const std::filesystem::path &root, const std::string &name,
              const EA::TG4::EvaluationConfiguration &config, const S::Provenance &provenance, bool dense,
              bool expired, int mode = 1, A::Budgets budgets = {}, A::Hook hook = {})
{
    const std::size_t count = mode == 5 ? 2356 : (expired ? 3240 : 2220);
    const std::size_t start = mode == 5 ? 2351 : (expired ? 3200 : 2200);
    const auto parent = root / (name + "-primary"), sidecar = root / (name + "-audit");
    S::Protocol p;
    S::ScoreRange range{Time(start), Time(count)};
    const auto fingerprint =
        S::Hash(S::CanonicalProtocol(p, range, EA::TG4::ConfigurationFingerprint(config), {}));
    S::ArtifactWriter writer(parent);
    auto sinks = writer.GetSinks();
    std::unique_ptr<A::Audit> audit;
    if (mode)
        audit = std::make_unique<A::Audit>(
            config.fibonacci, config.geometry,
            H::DecisionAuditConfiguration{fingerprint, range.start, range.end, false}, sidecar, budgets,
            hook);
    auto write = sinks.decision;
    if (audit)
        sinks.decision = [&, write](const auto &d)
        {
            write(d);
            audit->ObservePrimaryDecision(
                {d.identity, d.status, d.cohort ? S::Name(*d.cohort) : "", d.timestamp, d.close.has_value(),
                 d.aggregate.complete, d.aggregate.evaluated, d.aggregate.abs, d.aggregate.lines,
                 d.aggregate.fibNear, d.aggregate.lineNear, d.aggregate.intersection});
        };
    H::Observer observer;
    if (audit)
        observer = [&](const auto &o)
        {
            if (mode == 2 && o.bar == 2203)
            {
                auto copy = o;
                std::vector<F::ABStructure> reordered(o.retainedAB.begin(), o.retainedAB.end());
                Check(reordered.size() > 1, "real retained universe for ordering fault");
                std::swap(reordered[0], reordered[1]);
                copy.retainedAB = reordered;
                audit->Observe(copy);
            }
            else
                audit->Observe(o);
        };
    S::HistoricalEvaluator evaluator(p, EA::TG1B::CalibrationConfiguration(config.referenceBarScale),
                                     config.fibonacci, config.geometry, config.behavior,
                                     EA::TG4::ConfigurationFingerprint(config), sinks, range, {}, observer);
    for (std::size_t i = 0; i < count; ++i)
    {
        if (mode == 6 && i == 2207)
            continue;
        auto candle = Candle(i, dense, expired);
        if (mode == 3)
        {
            const double base = 100 + std::sqrt(static_cast<double>(i) + 1) * .01;
            const bool center = i % 100 == 50;
            candle = {Time(i), base + 1.4, base + (center ? 7 : 6), base + (center ? 0 : 1), base + 1.4};
        }
        if (mode == 4 || mode == 5)
            candle = PositiveCandle(i);
        evaluator.AddCompletedBar(candle);
    }
    evaluator.Finalize();
    Result result;
    auto parentProvenance = provenance;
    parentProvenance.preflightRows = count - (mode == 6 ? 1 : 0);
    try
    {
        writer.Complete(evaluator.Engine(), config, parentProvenance);
    }
    catch (const std::exception &error)
    {
        if (!std::filesystem::exists(parent / "INVALID"))
            throw std::runtime_error(std::string("unexpected primary publication failure: ") + error.what());
        result.parentInvalid = true;
    }
    if (audit)
    {
        result.complete = audit->Complete({fingerprint, provenance.gitCommit, provenance.sourceSHA256,
                                           evaluator.Engine().CandleDigest(), parent});
        result.summary = audit->GetSummary();
    }
    if (result.complete)
        A::ValidateSidecar(sidecar, parent);
    return result;
}
void SameFiles(const std::filesystem::path &first, const std::filesystem::path &second)
{
    for (const auto &entry : std::filesystem::directory_iterator(first))
        Check(S::FileHash(entry.path()) == S::FileHash(second / entry.path().filename()),
              "byte-identical artifacts");
}
int main(int argc, char **argv)
{
    try
    {
        Check(argc == 3, "arguments");
        auto config = EA::TG4::LoadConfigurationFile(argv[1]);
        config.fibonacci = S::HistoricalEvaluator::ResearchFibonacci(config.fibonacci);
        const std::filesystem::path root = std::filesystem::path(argv[2]) / "AB-capacity";
        Check(std::filesystem::create_directory(root), "new fixture output");
        const auto provenance = Provenance(root);
        GeometryCases(config);
        auto first = Replay(root, "enabled", config, provenance, true, false);
        Check(first.complete && first.parentInvalid && first.summary.evictions[0] > 0 &&
                  first.summary.parityChecks == 2220,
              "native candle mirrors complete with INVALID parent");
        Check(first.summary.firstEvictionBar && *first.summary.firstEvictionBar < 2200 &&
                  first.summary.evictions[1] > 0 &&
                  first.summary.evictions[0] == first.summary.evictions[1] + first.summary.evictions[2],
              "first prefix eviction");
        Check(first.summary.reconstructed == 5 && first.summary.unknown == 0,
              "capacity-invalid primary independently reconstructed");
        Replay(root, "disabled", config, provenance, true, false, 0);
        SameFiles(root / "enabled-primary", root / "disabled-primary");
        auto positive = Replay(root, "positive", config, provenance, true, false, 4);
        Check(positive.complete && positive.parentInvalid && positive.summary.changedF == 5 &&
                  positive.summary.changedI == 5 && positive.summary.changedCohort == 5 &&
                  positive.summary.changedL == 0,
              "native candle observer: restored victim changes Ffib/I/cohort, identical lines");
        auto boundary = Replay(root, "observer-boundary", config, provenance, true, false, 5);
        Check(boundary.complete && boundary.summary.reconstructed == 2 && boundary.summary.changedF == 1 &&
                  boundary.summary.changedI == 1 && boundary.summary.changedCohort == 1,
              "G/H: native victim contributes at age 2048 and expires immediately after");
        auto missing = Replay(root, "missing-slot", config, provenance, true, false, 6);
        Check(missing.complete && missing.summary.scheduled == 5 && missing.summary.missing == 1 &&
                  missing.summary.unknown == 0 && missing.summary.reconstructed == 4,
              "missing market slot remains unobservable, never false geometry");
        auto repeat = Replay(root, "repeat", config, provenance, true, false);
        Check(repeat.complete, "repeat complete");
        SameFiles(root / "enabled-audit", root / "repeat-audit");
        auto expired = Replay(root, "expired", config, provenance, true, true);
        Check(expired.complete && expired.parentInvalid && expired.summary.candidateRows == 0 &&
                  expired.summary.victimAgeExpirations == expired.summary.evictions[0],
              "A: victims expired before decisions, persistent INVALID");
        auto clean = Replay(root, "clean", config, provenance, false, false, 3);
        Check(clean.complete && !clean.parentInvalid && clean.summary.primaryParityChecks == 5,
              "complete primary geometry parity");
        Check(Read(root / "clean-primary" / "decisions.csv").find(",classified,") != std::string::npos,
              "nonempty classified primary cohort parity");
        A::Budgets small;
        small.pairsPerDecision = 1;
        auto pair = Replay(root, "pair-failure", config, provenance, true, false, 1, small);
        Check(!pair.complete && pair.summary.failure == "actual_decision_pair_work_budget" &&
                  pair.summary.unknown == 5,
              "L: pair budget fail closed");
        small = {};
        small.pairsPerDecision = 512;
        auto cfPair = Replay(root, "CF-pair-failure", config, provenance, true, false, 1, small);
        Check(!cfPair.complete && cfPair.summary.failure == "counterfactual_decision_pair_work_budget",
              "CF pair exhaustion retains independent ACTUAL reconstruction");
        small = {};
        small.totalPairs = 1;
        auto totalPair = Replay(root, "total-pair-failure", config, provenance, true, false, 1, small);
        Check(!totalPair.complete && totalPair.summary.failure == "actual_total_pair_work_budget",
              "exact total pair budget attribution");
        small = {};
        small.memoryBytes = 1;
        auto memory = Replay(root, "memory-failure", config, provenance, true, false, 1, small);
        Check(!memory.complete && memory.summary.failure == "observer_memory_budget", "M: memory budget");
        small = {};
        small.decodedOutputBytes = 32768;
        auto disk = Replay(root, "output-failure", config, provenance, true, false, 1, small);
        Check(!disk.complete && disk.summary.failure == "diagnostic_output_budget" &&
                  disk.summary.unpublishedDecisions > 0,
              "N: output budget");
        small = {};
        small.liveABRecords = 1;
        auto live = Replay(root, "live-failure", config, provenance, true, false, 1, small);
        Check(!live.complete && live.summary.failure == "live_AB_record_budget", "live record budget");
        bool parityFault = false;
        auto parity = Replay(root, "parity-failure", config, provenance, true, false, 1, {},
                             [&](A::Step s)
                             {
                                 if (s == A::Step::ActualParity && !parityFault)
                                 {
                                     parityFault = true;
                                     throw std::runtime_error("actual_parity_fault");
                                 }
                             });
        Check(!parity.complete && !parity.summary.parity, "O: actual parity fault");
        auto orderFailure = Replay(root, "order-failure", config, provenance, true, false, 2);
        Check(!orderFailure.complete && !orderFailure.summary.parity &&
                  orderFailure.summary.failure == "TG3_retained_order_parity",
              "O: real detector snapshot ordering mismatch");
        auto cf = Replay(root, "CF-failure", config, provenance, true, false, 1, {},
                         [](A::Step s)
                         {
                             if (s == A::Step::Counterfactual)
                                 throw std::runtime_error("counterfactual_reconstruction_failure");
                         });
        Check(!cf.complete && cf.summary.failure == "counterfactual_reconstruction_failure" &&
                  cf.summary.unknown == 5,
              "P: counterfactual unknown");
        for (const auto *name :
             {"pair-failure", "CF-pair-failure", "total-pair-failure", "memory-failure", "output-failure",
              "live-failure", "parity-failure", "order-failure", "CF-failure"})
        {
            SameFiles(root / "enabled-primary", root / (std::string(name) + "-primary"));
            Check(!std::filesystem::exists(root / (std::string(name) + "-audit") / "AUDIT_COMPLETE"),
                  "no failed completion marker");
            Check(Read(root / (std::string(name) + "-audit") / "summary.csv")
                          .find("changed_cohort,UNKNOWN,INCOMPLETE") != std::string::npos,
                  "UNKNOWN total effects");
        }
        for (auto step : {A::Step::Finalize, A::Step::Checksum, A::Step::Marker})
        {
            auto name = "publication-" + std::to_string(static_cast<int>(step));
            auto failure = Replay(root, name, config, provenance, false, false, 3, {},
                                  [&](A::Step s)
                                  {
                                      if (s == step)
                                          throw std::runtime_error("publication_fault");
                                  });
            Check(!failure.complete && !std::filesystem::exists(root / (name + "-audit") / "AUDIT_COMPLETE"),
                  "T: publication failure no marker");
            SameFiles(root / "clean-primary", root / (name + "-primary"));
        }
        Write(root / "enabled-audit" / "decisions.csv", "corruption\n");
        Reject([&] { A::ValidateSidecar(root / "enabled-audit", root / "enabled-primary"); });
        std::cout << "AB_CAPACITY_OBSERVER native_candle_parity=true "
                     "both_directions=true deterministic_evictions=true "
                     "expired_before_decision=true first_prefix_event=true "
                     "primary_INVALID_preserved=true all_primary_bytes_equal=true "
                     "sidecars_deterministic=true budgets_fail_closed=true "
                     "parity_and_CF_fail_closed=true publication_faults=true "
                     "corruption_detected=true\n";
        std::cout << "CausalABCapacityCounterfactualTests passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "AB capacity test failure: " << e.what() << '\n';
        return 1;
    }
}
