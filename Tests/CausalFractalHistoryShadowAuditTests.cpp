#include "CausalFibonacciTrendlineHistoricalArtifact.hpp"
#include "CausalFractalHistoryShadowAudit.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

namespace H = EA::FibonacciResearch::HistoryShadow;
namespace S = EA::FibonacciResearch::TrendlineHistorical;
namespace G = EA::TG1A;
namespace F = EA::TG3;
void Require(bool p, const char *message)
{
    if (!p)
        throw std::runtime_error(message);
}
std::string Read(const std::filesystem::path &p)
{
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), {});
}
G::Candle Candle(std::size_t i)
{
    double low = 100 + std::sqrt(static_cast<double>(i) + 1) * .1;
    // Strict low every three bars; concave increasing envelope validates chords.
    if (i % 3 != 0)
        low += 1;
    double high = low + 3 + (i % 7 == 0 ? 2 : 0);
    return {S::kOrigin + static_cast<std::int64_t>(i) * 900, low + 1, high, low, low + 1};
}
struct Run
{
    F::Configuration f;
    G::Configuration g;
    G::CausalFractalTrendLineGeometry nativeG;
    F::FibonacciConfluenceTracker nativeF;
    H::Audit audit;
    std::size_t lineEvictions = 0;
    explicit Run(const std::filesystem::path &d, H::Budgets b = {}, H::PublicationHook hook = {})
        : f(
              []
              {
                  F::Configuration c;
                  c.retracementRatios = {.382, .5, .618};
                  return c;
              }()),
          g(), nativeG(g), nativeF(f), audit(f, g, d, b, std::move(hook))
    {
    }
    void Add(std::size_t bar, G::Candle c, bool scoring = false)
    {
        auto survivors = static_cast<std::size_t>(std::count_if(nativeG.Candidates().begin(),
                                                                nativeG.Candidates().end(), [&](const auto &l)
                                                                { return bar - l.creationBar <= 512; }));
        auto u = nativeG.AddCompletedBar(c);
        lineEvictions += survivors + u.newlyCreatedCandidates.size() - nativeG.Candidates().size();
        nativeF.Advance(bar, c.timestamp);
        auto ab = nativeF.ObserveConfirmedFractals(u.newlyConfirmedFractals);
        audit.Observe({bar, c, nativeG.CurrentAtr(), scoring, scoring && c.timestamp % 3600 == 2700,
                       bar >= 2048, u.newlyConfirmedFractals, ab, u.newlyCreatedCandidates,
                       nativeF.ABStructures(), nativeG.Candidates(), nativeF.TotalABStructuresCreated(),
                       nativeF.ABCapacityEvictions(), nativeF.ABAgeExpirations()});
    }
};

template <class T>
concept HasOutcome = requires(T t) { t.outcome; };
static_assert(!HasOutcome<H::CompletedBarObservation>);
void Write(const std::filesystem::path &p, std::string_view s)
{
    std::ofstream f(p, std::ios::binary);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    f << s;
    f.close();
}
template <class Fn> void Reject(Fn fn)
{
    bool rejected = false;
    try
    {
        fn();
    }
    catch (const std::exception &)
    {
        rejected = true;
    }
    Require(rejected, "expected rejection");
}
std::int64_t Time(std::size_t bar)
{
    return S::kOrigin + static_cast<std::int64_t>(bar) * 900;
}
EA::TG4::EvaluationConfiguration Frozen(EA::TG4::EvaluationConfiguration c)
{
    c.fibonacci = S::HistoricalEvaluator::ResearchFibonacci(c.fibonacci);
    return c;
}
S::Provenance FixtureProvenance(const std::filesystem::path &root, std::size_t count)
{
    Write(root / "shadow-fixture-source", "5f2d52a810dd175321ece512d6f4693c92913ec98b1234daed40d3095aeec2fa  "
                                          "Headers/CausalFibonacciConfluenceIntegration.hpp\n"
                                          "c6eaa92bfec6c8dde0e67c54920e34d0fcf133c98c8fc5a500c2227214cdf507  "
                                          "Headers/CausalFractalTrendLineGeometry.hpp\n");
    Write(root / "shadow-fixture-executable", "synthetic executable fixture\n");
    Write(root / "shadow-fixture-config", "synthetic configuration fixture\n");
    Write(root / "shadow-fixture-archive", "synthetic archive fixture\n");
    S::Provenance p;
    p.gitCommit = std::string(40, 'a');
    p.compiler = "synthetic compiler";
    p.libraryVersion = "synthetic libraries";
    p.sourceManifest = root / "shadow-fixture-source";
    p.executable = root / "shadow-fixture-executable";
    p.configurationFile = root / "shadow-fixture-config";
    p.sourceArchive = root / "shadow-fixture-archive";
    p.sourceSHA256 = S::FileHash(p.sourceManifest);
    p.executableSHA256 = S::FileHash(p.executable);
    p.configurationFileSHA256 = S::FileHash(p.configurationFile);
    p.preflightRows = count;
    return p;
}
// A small empty-geometry parent fixture for sidecar serialization. Primary
// detector/output equivalence is tested separately with HistoricalEvaluator.
H::Provenance Parent(const std::filesystem::path &root, const std::filesystem::path &path,
                     const EA::TG4::EvaluationConfiguration &config, const std::vector<G::Candle> &candles)
{
    auto p = FixtureProvenance(root, candles.size());
    S::ArtifactWriter writer(path);
    S::Protocol protocol;
    S::EventEngine engine(protocol, EA::TG4::ConfigurationFingerprint(config), writer.GetSinks(),
                          S::ScoreRange{Time(0), Time(candles.size())});
    for (std::size_t i = 0; i < candles.size(); ++i)
        engine.Add(i, candles[i], 1, {}, {});
    engine.Finalize();
    writer.Complete(engine, config, p);
    return {engine.Fingerprint(), p.gitCommit, p.sourceSHA256, engine.CandleDigest(), path};
}
std::vector<std::vector<std::string>> Rows(const std::filesystem::path &path)
{
    std::string csv;
    S::InspectGzip(path, [&](std::string_view chunk) { csv += chunk; });
    std::istringstream stream(csv);
    std::string line;
    std::vector<std::vector<std::string>> result;
    while (std::getline(stream, line))
    {
        std::istringstream fields(line);
        std::string field;
        std::vector<std::string> row;
        while (std::getline(fields, field, ','))
            row.push_back(field);
        result.push_back(std::move(row));
    }
    return result;
}
void Rules()
{
    std::vector<G::ConfirmedFractal> lows{{G::FractalKind::Low, 1, Time(1), 1, 3, Time(3)},
                                          {G::FractalKind::Low, 2, Time(2), 2, 4, Time(4)},
                                          {G::FractalKind::Low, 3, Time(3), 10, 5, Time(5)}};
    G::ConfirmedFractal b{G::FractalKind::High, 3000, Time(3000), 5, 3002, Time(3002)};
    auto a = H::CausalRules::SelectTG3Anchor(lows, b);
    Require(a && a->anchorBar == 2, "most recent qualifying old anchor; no age cutoff");
    lows.push_back(
        {G::FractalKind::Low, b.anchorBar, b.anchorTimestamp, 0, b.confirmationBar, b.confirmationTimestamp});
    Require(H::CausalRules::SelectTG3Anchor(lows, b)->anchorBar == 2, "same-anchor-bar excluded");
    lows.push_back({G::FractalKind::Low, 2999, Time(2999), 0, 3003, Time(3003)});
    Require(H::CausalRules::SelectTG3Anchor(lows, b)->anchorBar == 2, "future confirmation excluded");
    Require(H::CausalRules::TG1ATemporallyEligible(514, 2) && !H::CausalRules::TG1ATemporallyEligible(515, 2),
            "TG1A age 512 inclusive");
    F::ABIdentity x{F::ABDirection::UpAB, 1, Time(1), 5, Time(5), 7, Time(7)}, alias = x;
    alias.aBar += 100;
    alias.bBar += 100;
    alias.availabilityBar += 100;
    Require(H::CandidateIdentityReconciliation::Natural(x) ==
                    H::CandidateIdentityReconciliation::Natural(alias) &&
                H::CandidateIdentityReconciliation::Native(x) !=
                    H::CandidateIdentityReconciliation::Native(alias),
            "natural/native alias reconciliation");
    F::Configuration config;
    config.retracementRatios = {.382, .5, .618};
    F::FibonacciConfluenceTracker mirror(config);
    G::ConfirmedFractal low{G::FractalKind::Low, 0, Time(0), 1, 2, Time(2)};
    mirror.Advance(2, Time(2));
    Require(mirror.ObserveConfirmedFractals({low, low}).empty(), "duplicate low creates no AB");
    G::ConfirmedFractal high{G::FractalKind::High, 3, Time(3), 5, 5, Time(5)};
    mirror.Advance(3, Time(3));
    mirror.Advance(4, Time(4));
    mirror.Advance(5, Time(5));
    Require(mirror.ObserveConfirmedFractals({high, high}).size() == 1, "duplicate confirmed B suppressed");
}
G::Candle OldAnchorCandle(std::size_t i)
{
    double low = 2, high = 3;
    if (i == 2)
        low = 1;
    if (i >= 2110 && i < 2503)
    {
        low = i >= 2112 && (i - 2112) % 3 == 0 ? 10 : 11;
        high = 12;
    }
    if (i >= 2503)
    {
        low = 3;
        high = i == 2505 ? 5 : 4;
    }
    return {Time(i), (low + high) / 2, high, low, (low + high) / 2};
}
void OldAnchorAndBoundary(const std::filesystem::path &root)
{
    Run old(root / "shadow-old-anchor");
    for (std::size_t i = 0; i < 2510; ++i)
        old.Add(i, OldAnchorCandle(i), i >= 2450);
    Require(old.audit.GetSummary().complete, "very old anchor parity");
    Require(old.audit.GetSummary().creationLossesTG3 == 1,
            "exactly single policy-selected restored old AB birth");
    Require(old.audit.GetSummary().decisionEligibleLosses >= 1,
            "restored old AB survives to scheduled decision");
    Run boundary(root / "shadow-age-boundary");
    for (std::size_t i = 0; i < 1029; ++i)
    {
        double low = i == 2 ? 1 : i == 512 ? 3 : 4;
        G::Candle c{Time(i), 4, 5, low, 4};
        boundary.Add(i, c);
        if (i == 514)
            Require(std::any_of(boundary.nativeG.Candidates().begin(), boundary.nativeG.Candidates().end(),
                                [](const auto &l) { return l.anchor1Bar == 2 && l.anchor2Bar == 512; }),
                    "age-512 anchor can generate line");
        if (i == 1026)
            Require(!boundary.nativeG.Candidates().empty(), "line age512 remains");
        if (i == 1027)
            Require(boundary.nativeG.Candidates().empty(), "line age513 expires");
    }
    Require(boundary.audit.GetSummary().complete, "age boundary native parity");
    const auto &low =
        boundary.audit.GetSummary().history[1][static_cast<std::size_t>(G::FractalKind::Low)][0];
    Require(low.capacityRemovals == 0 && low.expirations == 2,
            "lookback expiration distinct from capacity removal");
    Run dense(root / "shadow-candidate-capacity");
    for (std::size_t i = 0; i < 850; ++i)
    {
        double base = 100 + std::sqrt(static_cast<double>(i) + 1) * .1;
        bool center = i % 3 == 0;
        double low = base + (center ? 0 : 1), high = base + (center ? 5 : 4);
        dense.Add(i, {Time(i), base + 2, high, low, base + 2}, i >= 600);
    }
    Require(dense.audit.GetSummary().complete, "same-bar high/low and capacity parity");
    Require(dense.nativeF.ABCapacityEvictions() > 0 && dense.lineEvictions > 0,
            "independent AB and line capacity pressure exercised");
    Require(dense.audit.GetSummary().capacityDisplacements > 0,
            "explicit restored-pool capacity displacement exercised");
    Run contribution(root / "shadow-decision-contribution");
    for (std::size_t i = 0; i < 2120; ++i)
    {
        double base = 100 + std::sqrt(static_cast<double>(i) + 1) * .1;
        bool center = i % 3 == 0;
        double low = base + (center ? 0 : .5), high = base + (center ? 7 : 6);
        // A completed decision-bar volatility spike raises causal ATR while
        // previously confirmed AB prices remain unchanged. It has no future
        // candle dependency and lets all three frozen distances pass.
        if (i >= 2048 && i % 4 == 3)
            high = base + 100;
        contribution.Add(i, {Time(i), base + 1.4, high, low, base + 1.4}, i >= 2048);
    }
    Require(contribution.audit.GetSummary().complete &&
                contribution.audit.GetSummary().decisionEligibleLosses > 0 &&
                contribution.audit.GetSummary().contributingLosses > 0,
            (std::string("causal post-warmup scheduled cohort-geometry witness: complete=") +
             std::to_string(contribution.audit.GetSummary().complete) +
             " eligible=" + std::to_string(contribution.audit.GetSummary().decisionEligibleLosses) +
             " contribution=" + std::to_string(contribution.audit.GetSummary().contributingLosses) +
             " failure=" + contribution.audit.GetSummary().failure)
                .c_str());
    Run rejected(root / "shadow-structural-rejection");
    for (std::size_t i = 0; i < 450; ++i)
    {
        auto c = Candle(i);
        if (i == 210)
            c.low = 50;
        rejected.Add(i, c);
    }
    Require(rejected.audit.GetSummary().complete && rejected.audit.GetSummary().structuralRejections > 0,
            "structural rejection is not a demonstrated lost birth");
}
void PrimaryIsolation(const std::filesystem::path &root, const EA::TG4::EvaluationConfiguration &config)
{
    constexpr std::size_t count = 2112;
    auto provenance = FixtureProvenance(root, count);
    std::array<std::filesystem::path, 4> directories{root / "isolation-disabled", root / "isolation-enabled",
                                                     root / "isolation-exhausted",
                                                     root / "isolation-allocation-failed"};
    for (std::size_t mode = 0; mode < 4; ++mode)
    {
        S::ArtifactWriter writer(directories[mode]);
        std::unique_ptr<H::Audit> audit;
        H::Observer observer;
        bool auditArmed = false;
        if (mode)
        {
            H::Budgets b;
            if (mode == 2)
                b.shadowFractals = 1;
            audit = std::make_unique<H::Audit>(config.fibonacci, config.geometry,
                                               root / (mode == 1   ? "isolation-audit"
                                                       : mode == 2 ? "isolation-failed-audit"
                                                                   : "isolation-allocation-audit"),
                                               b,
                                               mode == 3 ? H::PublicationHook(
                                                               [&](H::PublicationStep op)
                                                               {
                                                                   if (auditArmed &&
                                                                       op == H::PublicationStep::Compress)
                                                                       throw std::bad_alloc();
                                                               })
                                                         : H::PublicationHook{});
            observer = [&](const auto &o) { audit->Observe(o); };
            auditArmed = true;
        }
        S::Protocol p;
        S::HistoricalEvaluator evaluator(p, EA::TG1B::CalibrationConfiguration(config.referenceBarScale),
                                         config.fibonacci, config.geometry, config.behavior,
                                         EA::TG4::ConfigurationFingerprint(config), writer.GetSinks(),
                                         S::ScoreRange{Time(2048), Time(count)}, {}, std::move(observer));
        for (std::size_t i = 0; i < count; ++i)
        {
            double base = 100 + std::sqrt(static_cast<double>(i) + 1) * .01;
            bool center = i % 100 == 50;
            double low = base + (center ? 0 : 1), high = base + (center ? 5 : 4);
            evaluator.AddCompletedBar({Time(i), base + 2, high, low, base + 2});
        }
        evaluator.Finalize();
        writer.Complete(evaluator.Engine(), config, provenance);
        if (audit)
        {
            auto finish = [&]
            {
                return audit->Complete({evaluator.Engine().Fingerprint(), provenance.gitCommit,
                                        provenance.sourceSHA256, evaluator.Engine().CandleDigest(),
                                        directories[mode]});
            };
            if (mode == 3)
            {
                Require(audit->GetSummary().exhausted &&
                            audit->GetSummary().failure == "diagnostic_allocation_exhaustion",
                        "allocation failure captured without primary feedback");
                Reject(finish);
            }
            else
                Require(finish() == (mode == 1), "audit failure is isolated from primary publication");
        }
    }
    for (const auto &entry : std::filesystem::directory_iterator(directories[0]))
        if (entry.is_regular_file())
            for (std::size_t mode = 1; mode < 4; ++mode)
                Require(S::FileHash(entry.path()) == S::FileHash(directories[mode] / entry.path().filename()),
                        "all primary artifact bytes identical enabled/disabled/exhausted");
    H::ValidateSidecar(root / "isolation-audit");
    Reject([&] { H::ValidateSidecar(root / "isolation-failed-audit"); });
    std::cout << "SHADOW_PRIMARY_ISOLATION all_primary_artifact_bytes=true synthetic_outcomes_opaque=true "
                 "failure_isolated=true bars=2112\n";
}
void Publication(const std::filesystem::path &root, const EA::TG4::EvaluationConfiguration &config,
                 Run &pressure)
{
    std::vector<G::Candle> candles;
    for (std::size_t i = 0; i < 900; ++i)
        candles.push_back(Candle(i));
    auto parent = Parent(root, root / "shadow-pressure-parent", config, candles);
    Require(pressure.audit.Complete(parent), "pressure sidecar publication");
    H::ValidateSidecar(root / "shadow-pressure");
    std::set<std::string> rawIdentities;
    std::size_t t1 = 0, t3 = 0;
    for (const auto &row : Rows(root / "shadow-pressure/candidate_witnesses.csv.gz"))
        if (row[3] == "RAW_BIRTH")
        {
            Require(rawIdentities.insert(row[0] + ":" + row[1]).second,
                    "raw natural birth identity counted exactly once");
            if (row[0] == "TG1A")
                ++t1;
            else
                ++t3;
        }
    Require(t1 == pressure.audit.GetSummary().creationLossesTG1A &&
                t3 == pressure.audit.GetSummary().creationLossesTG3,
            "raw rows reconcile exactly with unique creation losses");
    Require(t1 == 19474, "preliminary 19474 is unique raw lost births for this frozen synthetic fixture");
    Run repeat(root / "shadow-pressure-repeat");
    for (std::size_t i = 0; i < 900; ++i)
        repeat.Add(i, Candle(i), i >= 500);
    Require(repeat.audit.Complete(parent), "repeat audit complete");
    for (const auto &entry : std::filesystem::directory_iterator(root / "shadow-pressure"))
        Require(S::FileHash(entry.path()) ==
                    S::FileHash(root / "shadow-pressure-repeat" / entry.path().filename()),
                "deterministic complete sidecar bytes");
    std::vector<G::Candle> small;
    for (std::size_t i = 0; i < 8; ++i)
        small.push_back(Candle(i));
    auto shortParent = Parent(root, root / "shadow-small-parent", config, small);
    for (int failure = 0; failure < 7; ++failure)
    {
        auto path = root / ("shadow-publication-failure-" + std::to_string(failure));
        bool armed = false;
        H::PublicationHook hook = [&](H::PublicationStep op)
        {
            if (!armed)
                return;
            if (failure == 5 && op == H::PublicationStep::Checksum)
            {
                Write(path / "history_summary.csv", "corrupt fixture\n");
                return;
            }
            if (static_cast<int>(op) == failure)
                throw std::runtime_error("injected publication failure");
        };
        Run run(path, {}, hook);
        if (failure == 0)
            armed = true;
        for (std::size_t i = 0; i < (failure == 0 ? 900 : 8); ++i)
            run.Add(i, Candle(i));
        armed = true;
        if (failure == 6)
        {
            auto wrong = shortParent;
            wrong.candleDigest = std::string(64, 'b');
            Reject([&] { run.audit.Complete(wrong); });
        }
        else
            Reject([&] { run.audit.Complete(shortParent); });
        Require(!std::filesystem::exists(path / "AUDIT_COMPLETE"), "failed publication never gets marker");
    }
    for (std::size_t kind = 0; kind < 6; ++kind)
    {
        H::Budgets b;
        if (kind == 0)
            b.shadowFractals = 1;
        if (kind == 1)
            b.liveCandidates = 1;
        if (kind == 2)
            b.witnessRecords = 1;
        if (kind == 3)
            b.removalRecords = 1;
        if (kind == 4)
            b.diagnosticBytes = 1;
        if (kind == 5)
            b.decodedPublicationBytes = 2048;
        Run run(root / ("shadow-budget-" + std::to_string(kind)), b);
        for (std::size_t i = 0; i < 900; ++i)
            run.Add(i, Candle(i));
        Require(!run.audit.GetSummary().complete && run.audit.GetSummary().exhausted,
                "every configured resource budget fails closed");
        Require(!run.audit.Complete(parent), "exhausted diagnostic publication explicitly incomplete");
        Require(!std::filesystem::exists(root / ("shadow-budget-" + std::to_string(kind)) / "AUDIT_COMPLETE"),
                "exhausted marker absent");
    }
    // Public observation divergence must not be silently repaired.
    Run divergent(root / "shadow-divergent");
    auto c = Candle(0);
    auto u = divergent.nativeG.AddCompletedBar(c);
    divergent.nativeF.Advance(0, c.timestamp);
    auto ab = divergent.nativeF.ObserveConfirmedFractals(u.newlyConfirmedFractals);
    std::vector<G::ConfirmedFractal> impossible{
        {G::FractalKind::High, 0, c.timestamp, c.high, 0, c.timestamp}};
    divergent.audit.Observe({0, c, divergent.nativeG.CurrentAtr(), false, false, false, impossible, ab,
                             u.newlyCreatedCandidates, divergent.nativeF.ABStructures(),
                             divergent.nativeG.Candidates(), 0, 0, 0});
    Require(!divergent.audit.GetSummary().parity && !divergent.audit.GetSummary().complete,
            "independent detector mirror divergence explicit");
    Require(!divergent.audit.Complete(shortParent), "parity failure prevents successful marker");
    // Existing audit publications may not be overwritten.
    Reject([&] { Run overwrite(root / "shadow-pressure"); });
    std::cout << "SHADOW_SIDECAR unique_TG1A_raw_births=" << t1
              << " deterministic_bytes=true failure_cases=7 budgets=6 parity_divergence=true\n";
}
int main(int argc, char **argv)
{
    try
    {
        Require(argc == 3, "config and test directory arguments");
        auto config = Frozen(EA::TG4::LoadConfigurationFile(argv[1]));
        auto root = std::filesystem::path(argv[2]);
        Rules();
        Run plain(root / "shadow-plain");
        for (std::size_t i = 0; i < 80; ++i)
            plain.Add(i, Candle(i));
        Require(plain.audit.GetSummary().complete && plain.audit.GetSummary().parityChecks == 80,
                "unsaturated native mirror parity");
        Run pressure(root / "shadow-pressure");
        for (std::size_t i = 0; i < 900; ++i)
            pressure.Add(i, Candle(i), i >= 500);
        auto q = pressure.audit.GetSummary();
        Require(q.complete, "saturated parity and attribution");
        std::size_t t3 = 0, t1 = 0;
        for (auto &kind : q.history[0])
            t3 += kind[0].capacityRemovals;
        for (auto &kind : q.history[1])
            t1 += kind[0].capacityRemovals;
        Require(t3 > 0 && t1 > 0, "both histories saturated");
        Require(q.history[0][static_cast<std::size_t>(G::FractalKind::Low)][0].peakOccupancy == 128 &&
                    q.history[0][static_cast<std::size_t>(G::FractalKind::Low)][0].peakDemand == 129,
                "TG3 post-insertion logical demand");
        Require(q.history[1][static_cast<std::size_t>(G::FractalKind::Low)][0].peakOccupancy == 64 &&
                    q.history[1][static_cast<std::size_t>(G::FractalKind::Low)][0].peakDemand == 65,
                "TG1A pre-insertion logical demand");
        Require(q.creationLossesTG1A > 0, "TG1A demonstrated raw creation losses");
        for (auto &detector : q.history)
            for (auto &kind : detector)
            {
                Require(kind[0].insertions == kind[1].insertions + kind[2].insertions,
                        "phase insertion reconciliation");
                Require(kind[0].capacityRemovals == kind[1].capacityRemovals + kind[2].capacityRemovals,
                        "phase removal reconciliation");
            }
        H::Budgets b;
        b.shadowFractals = 10;
        Run exhausted(root / "shadow-exhausted", b);
        for (std::size_t i = 0; i < 80; ++i)
            exhausted.Add(i, Candle(i));
        Require(!exhausted.audit.GetSummary().complete && exhausted.audit.GetSummary().exhausted,
                "history budget fails closed");
        OldAnchorAndBoundary(root);
        PrimaryIsolation(root, config);
        Publication(root, config, pressure);
        std::cout << "SHADOW_PARITY unsaturated_bars=80 saturated_bars=900 TG3_removals=" << t3
                  << " TG1A_removals=" << t1 << " TG1A_creation_losses=" << q.creationLossesTG1A
                  << " resource_exhaustion=true\n";
        std::cout << "CausalFractalHistoryShadowAuditTests passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "SHADOW_TEST_FAILURE " << e.what() << '\n';
        return 1;
    }
}
