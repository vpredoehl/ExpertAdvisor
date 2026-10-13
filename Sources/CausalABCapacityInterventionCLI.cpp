#include "CausalABCapacityCounterfactual.hpp"
#include "CausalFibonacciResearchCandleSnapshot.hpp"
#include "CausalFibonacciTrendlineHistoricalArtifact.hpp"
#include "RuntimeDatabaseConnection.hpp"
#include "TG4HistoricalMarketDataRepository.hpp"
#include <fstream>
#include <iostream>
#include <sys/resource.h>
namespace A = EA::FibonacciResearch::ABCapacity;
namespace H = EA::FibonacciResearch::HistoryShadow;
namespace S = EA::FibonacciResearch::TrendlineHistorical;
void Require(bool x, const char *m)
{
    if (!x)
        throw std::runtime_error(m);
}
std::vector<std::string> Split(std::string_view line, char sep = '\t')
{
    std::vector<std::string> out;
    std::istringstream in{std::string(line)};
    std::string cell;
    while (std::getline(in, cell, sep))
        out.push_back(cell);
    return out;
}
std::string IdentityHash(const H::DecisionGeometry &g)
{
    S::Sha256 h;
    for (const auto &[key, a] : g.abs)
        h.Add(key + ":" + H::CandidateIdentityReconciliation::Native(a.identity) + ":" + S::Snapshot(a) +
              "\n");
    return h.Digest();
}
std::string Bits(const H::DecisionGeometry &g)
{
    Require(g.fibNear.has_value() && g.lineNear.has_value() && g.intersection.has_value(),
            "ineligible_intervention_geometry");
    return std::to_string(*g.fibNear) + "," + std::to_string(*g.lineNear) + "," +
           std::to_string(*g.intersection) + "," + g.cohort;
}
struct Target
{
    std::vector<std::string> cells;
    std::vector<std::string> qualifiers;
};
int main(int argc, char **argv)
{
    try
    {
        Require(argc == 4, "usage: intervention CONFIG PLAN NEW_STAGE_DIRECTORY");
        const std::filesystem::path stage = argv[3];
        Require(!std::filesystem::exists(stage) && std::filesystem::create_directory(stage),
                "new_stage_required");
        Require(std::filesystem::space(stage).available >= 40ULL * 1024 * 1024 * 1024, "40_GiB_disk_reserve");
        auto config = EA::TG4::LoadConfigurationFile(argv[1]);
        config.fibonacci = S::HistoricalEvaluator::ResearchFibonacci(config.fibonacci);
        config.fibonacci.absolutePriceTolerance =
            EA::TG4::EffectiveFibonacciAbsolutePriceTolerance(config, "audcadrmp");
        std::map<std::int64_t, Target> targets;
        std::map<std::int64_t, std::pair<std::string, std::size_t>> checkpoints;
        std::ifstream plan(argv[2]);
        std::string line;
        while (std::getline(plan, line))
        {
            auto cells = Split(line);
            if (cells[0] == "D")
            {
                Require(cells.size() == 22, "plan_decision_schema");
                auto ts = std::stoll(cells[1]);
                Target t{cells, Split(cells[21], ';')};
                Require(targets.emplace(ts, std::move(t)).second, "duplicate_target_timestamp");
            }
            else if (cells[0] == "C")
            {
                Require(cells.size() == 4, "checkpoint_schema");
                checkpoints.emplace(std::stoll(cells[1]), std::pair{cells[2], std::stoull(cells[3])});
            }
            else
                throw std::runtime_error("plan_schema");
        }
        Require(targets.size() == 39 && !checkpoints.empty(), "authorized_39_targets");
        const auto end = checkpoints.rbegin()->first;
        Require(end < S::kCalibrationStart, "pre_2020_intervention_batch_only");
        const EA::TG4::TemporalRange range{S::kOrigin, S::kOrigin, end, end};
        std::unique_ptr<S::ResearchCandleSnapshot> candles;
        auto begin = std::chrono::steady_clock::now();
        {
            pqxx::connection connection{EA::RuntimeDatabaseConnection::ForexConnectionString() +
                                        " application_name=ab_selective_intervention_read_only"};
            pqxx::read_transaction tx(connection);
            tx.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
            Require(tx.exec("SELECT current_setting('default_transaction_read_only')")
                            .one_row()[0]
                            .as<std::string>() == "on",
                    "read_only_required");
            const auto pre =
                EA::TG4::HistoricalMarketDataRepository::Preflight(tx, "audcadrmp", config, range);
            Require(pre.rowCount > 0 && pre.duplicateTimestampCount == 0, "canonical_preflight");
            S::ResearchCandleSnapshot::Budgets b;
            b.reserveDiskBytes = 40ULL * 1024 * 1024 * 1024;
            candles =
                std::make_unique<S::ResearchCandleSnapshot>("/private/tmp", S::kOrigin, end, pre.rowCount, b);
            EA::TG4::HistoricalMarketDataRepository::StreamCanonicalCandles(
                tx, "audcadrmp", config, range, [&](const auto &c) { candles->Append(c); });
            candles->Seal();
            tx.commit();
        }
        auto loadEnd = std::chrono::steady_clock::now();
        auto run = [&](const std::filesystem::path &directory)
        {
            std::filesystem::create_directory(directory);
            std::ofstream decisions(directory / "decisions.csv"), subsets(directory / "subsets.csv"),
                attributes(directory / "attribution.csv"), states(directory / "state_validation.csv");
            for (auto *f : {&decisions, &subsets, &attributes, &states})
                f->exceptions(std::ios::badbit | std::ios::failbit);
            decisions
                << "decision_id,bar_start_utc_epoch,actual_Ffib,actual_Lline,actual_I,actual_cohort,complete_"
                   "Ffib,complete_Lline,complete_I,complete_cohort,actual_AB_identity_sha256,complete_AB_"
                   "identity_sha256,actual_ABs,complete_ABs,lines,restored_records,qualifying_ABs,"
                   "individually_sufficient_ABs,minimal_sufficient_subsets,requires_multiple_ABs,redundant_"
                   "sufficient_ABs,subset_evaluations,pairs,baseline_parity,complete_parity\n";
            subsets << "decision_id,mask,AB_identities,Ffib,Lline,I,cohort,AB_identity_sha256,ABs,sufficient,"
                       "minimal_sufficient\n";
            attributes << "decision_id,AB_identity,Ffib,Lline,I,cohort,individually_sufficient,necessary_in_"
                          "every_sufficient_subset,minimal_subset_memberships,redundant_sufficient\n";
            states << "decision_id,AB_identity,availability_bar,age,upward,eligible_pairs,fib_near,"
                      "conjunction,eviction_bar,eviction_utc_epoch,a_price,b_price,a_confirmation_utc_epoch,"
                      "b_confirmation_utc_epoch\n";
            A::InterventionReplay replay(config.fibonacci, config.geometry);
            S::Sha256 hash;
            std::size_t bars = 0, processed = 0, pairs = 0, peak = 0;
            auto cp = checkpoints.begin();
            auto checkpoint = [&]
            {
                Require(cp->second.first == hash.Digest() && cp->second.second == bars,
                        "canonical_prefix_digest_parity");
                ++cp;
            };
            candles->Replay(
                [&](const auto &c)
                {
                    while (cp != checkpoints.end() && c.timestamp >= cp->first)
                        checkpoint();
                    hash.Add(std::to_string(c.timestamp) + "," + S::Number(c.open) + "," + S::Number(c.high) +
                             "," + S::Number(c.low) + "," + S::Number(c.close) + "\n");
                    replay.AddCompletedBar(c);
                    const auto bar = bars++;
                    if (bars % 4096 == 0)
                        Require(std::filesystem::space(stage).available >= 40ULL * 1024 * 1024 * 1024,
                                "40_GiB_disk_reserve");
                    peak = std::max(peak, replay.Victims().size() + replay.Actual().ABStructures().size());
                    auto found = targets.find(c.timestamp);
                    if (found == targets.end())
                        return;
                    const auto &t = found->second;
                    const auto &e = t.cells;
                    const auto id = e[2];
                    Require(std::stoull(e[3]) == bar && c.close == std::stod(e[4]) &&
                                replay.Geometry().CurrentAtr() == std::stod(e[5]),
                            "decision_market_bar_parity");
                    auto result = A::EvaluateInterventions(
                        bar, c, replay.Geometry().CurrentAtr(), replay.Actual().ABStructures(),
                        replay.Geometry().Candidates(), replay.Victims(), t.qualifiers);
                    Require(Bits(result.actual) == e[6] + "," + e[7] + "," + e[8] + "," + e[9] &&
                                result.actual.eligibleAB == std::stoull(e[10]) &&
                                result.actual.eligibleLines == std::stoull(e[11]),
                            "historical_baseline_parity");
                    Require(Bits(result.complete) == e[12] + "," + e[13] + "," + e[14] + "," + e[15] &&
                                result.complete.eligibleAB == std::stoull(e[16]) &&
                                replay.Victims().size() == std::stoull(e[17]),
                            "historical_complete_parity");
                    for (const auto &v : replay.Victims())
                    {
                        const auto key = S::ABKey(v.ab);
                        bool eligible = v.ab.identity.direction == EA::TG3::ABDirection::UpAB &&
                                        result.complete.abs.contains(key);
                        states << id << ',' << H::CandidateIdentityReconciliation::Native(v.ab.identity)
                               << ',' << v.ab.identity.availabilityBar << ','
                               << bar - v.ab.identity.availabilityBar << ','
                               << (v.ab.identity.direction == EA::TG3::ABDirection::UpAB) << ','
                               << (eligible ? result.complete.eligibleLines : 0) << ',';
                        if (eligible)
                            states << result.complete.abGeometry.at(key).near << ','
                                   << result.complete.abGeometry.at(key).conjunction;
                        else
                            states << "INELIGIBLE,INELIGIBLE";
                        states << ',' << v.evictionBar << ',' << v.evictionTimestamp << ','
                               << S::Number(v.ab.aPrice) << ',' << S::Number(v.ab.bPrice) << ','
                               << v.ab.aConfirmationTimestamp << ',' << v.ab.bConfirmationTimestamp << '\n';
                    }
                    std::size_t singles = 0, minimal = 0, minSize = 6;
                    for (const auto &s : result.subsets)
                    {
                        std::string keys;
                        for (std::size_t i = 0; i < result.qualifyingIdentities.size(); ++i)
                            if (s.mask & (std::size_t{1} << i))
                            {
                                if (!keys.empty())
                                    keys += ';';
                                keys += result.qualifyingIdentities[i];
                            }
                        auto size = std::popcount(s.mask);
                        if (s.sufficient && size == 1)
                            ++singles;
                        if (s.minimal)
                        {
                            ++minimal;
                            minSize = std::min(minSize, static_cast<std::size_t>(size));
                        }
                        subsets << id << ',' << s.mask << ',' << keys << ',' << Bits(s.geometry) << ','
                                << IdentityHash(s.geometry) << ',' << s.geometry.eligibleAB << ','
                                << s.sufficient << ',' << s.minimal << '\n';
                    }
                    Require(minimal > 0, "unresolved_minimal_subsets");
                    for (std::size_t i = 0; i < result.qualifyingIdentities.size(); ++i)
                    {
                        auto mask = std::size_t{1} << i;
                        const auto &single = result.subsets.at(mask);
                        bool necessary = true;
                        std::size_t memberships = 0;
                        for (const auto &s : result.subsets)
                        {
                            if (s.sufficient && !(s.mask & mask))
                                necessary = false;
                            if (s.minimal && (s.mask & mask))
                                ++memberships;
                        }
                        attributes << id << ',' << result.qualifyingIdentities[i] << ','
                                   << Bits(single.geometry) << ',' << single.sufficient << ',' << necessary
                                   << ',' << memberships << ',' << (single.sufficient && singles > 1) << '\n';
                    }
                    decisions << id << ',' << c.timestamp << ',' << Bits(result.actual) << ','
                              << Bits(result.complete) << ',' << IdentityHash(result.actual) << ','
                              << IdentityHash(result.complete) << ',' << result.actual.eligibleAB << ','
                              << result.complete.eligibleAB << ',' << result.complete.eligibleLines << ','
                              << replay.Victims().size() << ',' << result.qualifyingIdentities.size() << ','
                              << singles << ',' << minimal << ',' << (minSize > 1) << ','
                              << (singles > 1 ? singles : 0) << ',' << result.subsets.size() << ','
                              << result.pairs << ",PASS,PASS\n";
                    pairs = A::AccumulateInterventionPairs(1024ULL * 1024 * 1024, pairs, result.pairs, 1);
                    ++processed;
                });
            while (cp != checkpoints.end())
                checkpoint();
            Require(processed == 39, "all_39_accounted");
            for (auto *f : {&decisions, &subsets, &attributes, &states})
                f->close();
            std::ofstream stats(directory / "resource.csv");
            stats << "metric,value\nsource_rows," << bars << "\nparity_checks," << replay.ParityChecks()
                  << "\npeak_live_ABs," << peak << "\npairs," << pairs << "\ncanonical_candle_sha256,"
                  << hash.Digest() << '\n';
        };
        run(stage / "first");
        run(stage / "repeat");
        for (const auto &entry : std::filesystem::directory_iterator(stage / "first"))
            Require(S::FileHash(entry.path()) == S::FileHash(stage / "repeat" / entry.path().filename()),
                    "repeated_execution_byte_parity");
        auto finish = std::chrono::steady_clock::now();
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        std::ofstream timing(stage / "timing.csv");
        timing << "metric,value\nload_seconds," << std::chrono::duration<double>(loadEnd - begin).count()
               << "\nreplay_and_intervention_seconds,"
               << std::chrono::duration<double>(finish - loadEnd).count() << "\ntotal_seconds,"
               << std::chrono::duration<double>(finish - begin).count() << "\nmax_rss_bytes,"
               << usage.ru_maxrss << '\n';
        std::cout << "all_39_complete repeat_bytes_equal=true database_closed_before_replay=true\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "intervention_failure: " << e.what() << '\n';
        return 1;
    }
}
